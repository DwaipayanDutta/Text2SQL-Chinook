import os
import json
import random
import numpy as np
import torch
import re
import streamlit as st
import streamlit.components.v1 as components

from pathlib import Path
from urllib.parse import urlparse
import sqlglot
from sqlglot import exp
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from sqlalchemy import inspect, text
from langchain_core.prompts import PromptTemplate


# =====================================================
# GLOBAL STABILITY
# =====================================================


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(999)

torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =====================================================
# PATHS
# =====================================================

LOG_PATH = Path("chinook_sql_log.json")
FAIL_PATH = Path("chinook_failures.json")

for path in [LOG_PATH, FAIL_PATH]:
    if not path.exists():
        with open(path, "w") as f:
            json.dump([], f)

MODEL_PATH = "cycloneboy/SLM-SQL-0.6B"
SQL_DIALECT = "sqlite"
BASE_DIR = Path.cwd().resolve()

st.set_page_config(page_title="Chinook SQL Generator + Evaluator", layout="wide")


# =====================================================
# MERMAID ER DIAGRAM
# =====================================================


def render_mermaid(code: str):
    html_code = f"""
    <pre class="mermaid">
        {code}
    </pre>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true }});
    </script>
    """
    components.html(html_code, height=600, scrolling=True)


def generate_mermaid_er(db_engine):
    inspector = inspect(db_engine)
    tables = inspector.get_table_names()
    mermaid_code = "erDiagram\n"

    for table in tables:
        mermaid_code += f"{table} {{\n"

        for col in inspector.get_columns(table):
            col_type = str(col["type"]).split("(")[0]
            mermaid_code += f"{col_type} {col['name']}\n"

        mermaid_code += "}\n"

        for fk in inspector.get_foreign_keys(table):
            target = fk["referred_table"]
            mermaid_code += f'{table} ||--o{{ {target} : ""\n'

    return mermaid_code


# =====================================================
# MODEL
# =====================================================


@st.cache_resource
def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=600,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(pipeline=pipe)


llm = load_llm()


# =====================================================
# HELPERS
# =====================================================


def extract_sql(text: str):

    if isinstance(text, dict):
        text = text.get("result", "")

    text = re.sub(r"```sql|```", "", text, flags=re.IGNORECASE)

    if "SQLQuery:" in text:
        text = text.split("SQLQuery:")[-1]

    match = re.search(
        r"((WITH\b.*?SELECT\b.*)|SELECT\b.*)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if match:
        candidate = match.group(1).strip()

        if ";" in candidate:
            candidate = candidate[: candidate.rfind(";") + 1]

        return candidate.strip()

    return ""


def is_safe(sql: str):

    try:
        statements = sqlglot.parse(sql, read=SQL_DIALECT)
    except Exception:
        return False

    if len(statements) != 1:
        return False

    stmt = statements[0]
    if not stmt.is_select:
        return False

    forbidden_nodes = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Alter,
        exp.Create,
        exp.Replace,
        exp.Attach,
        exp.Command,
        exp.Transaction,
        exp.Pragma,
    )

    for node in stmt.walk():
        if isinstance(node, forbidden_nodes):
            return False

    return True


def normalize_sqlite_uri(uri: str):
    try:
        parsed = urlparse(uri)
    except Exception:
        return None, "Invalid database URI."

    if parsed.scheme != "sqlite":
        return None, "Only sqlite:// URIs are allowed."

    if not parsed.path:
        return None, "SQLite path is empty."

    db_path = Path(parsed.path)
    if not db_path.is_absolute():
        db_path = (BASE_DIR / db_path).resolve()

    try:
        if not db_path.is_relative_to(BASE_DIR):
            return None, "Database path must be within the project directory."
    except AttributeError:
        if BASE_DIR not in db_path.parents and db_path != BASE_DIR:
            return None, "Database path must be within the project directory."

    if not db_path.exists():
        return None, f"Database file not found: {db_path}"

    return f"sqlite:///{db_path}", None


def execute_sql(db, sql):
    try:
        engine = getattr(db, "_engine", None)
        if engine is None:
            return False, 0, "Database engine is unavailable."
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = len(result.fetchall()) if result.returns_rows else result.rowcount
        return True, rows, None
    except Exception as e:
        return False, 0, str(e)


def _append_json_entry(path: Path, entry: dict):
    try:
        import fcntl
    except Exception:
        fcntl = None

    with open(path, "r+") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)

        data = json.load(f)
        data.append(entry)
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2)

        if fcntl:
            fcntl.flock(f, fcntl.LOCK_UN)


def log_query(question, sql, success, rows, error):

    entry = {
        "question": question,
        "generated_sql": sql,
        "execution_success": success,
        "row_count": rows,
        "error": error,
    }

    _append_json_entry(LOG_PATH, entry)

    if not success:
        _append_json_entry(FAIL_PATH, entry)


# =====================================================
# SIDEBAR — DB
# =====================================================

with st.sidebar:

    st.header("Database")

    db_uri = st.text_input("Chinook DB URI", value="sqlite:///Chinook_Sqlite.sqlite")

    db = None

    if db_uri:
        try:
            normalized_uri, uri_error = normalize_sqlite_uri(db_uri)
            if uri_error:
                st.error(uri_error)
            else:
                db = SQLDatabase.from_uri(normalized_uri)
                st.success("Connected")

                if st.button("Show ER Diagram"):
                    render_mermaid(generate_mermaid_er(db._engine))

        except Exception as e:
            st.error(str(e))


st.sidebar.divider()
st.sidebar.header("Evaluation")

with open(LOG_PATH) as f:
    logs = json.load(f)

total = len(logs)
success = sum(l["execution_success"] for l in logs) if logs else 0
accuracy = (success / total) * 100 if total else 0

st.sidebar.metric("Execution Accuracy", f"{accuracy:.1f}%")
st.sidebar.write(f"Queries Logged: {total}")

if st.sidebar.button("Replay Evaluation") and db:

    results = []

    for item in logs:
        sql = item.get("generated_sql", "")
        ok = is_safe(sql)
        if ok:
            ok, _, _ = execute_sql(db, sql)
        results.append(ok)

    replay_acc = (sum(results) / len(results)) * 100 if results else 0

    st.sidebar.metric("Replay Accuracy", f"{replay_acc:.1f}%")


# =====================================================
# MAIN
# =====================================================

st.title("Chinook SQL Generator + Evaluator")

if prompt := st.chat_input("Ask a question about the Chinook database..."):

    if not db:
        st.error("Connect to the database first.")
        st.stop()

    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):

        custom_prompt = PromptTemplate.from_template(
            """You are a SQLite expert. Use the provided Table Info as your ER Diagram to create a join-heavy SQLite query.

You MUST only use the tables and columns provided below.

=====================
DATABASE SCHEMA:
{table_info}
=====================

JOIN RULES:
1. Many relationships require bridge tables.
2. Carefully trace Foreign Keys to connect tables.
3. Never guess column names.

RULES:
- Return ONLY SQL.
- No explanations.
- No markdown.
- Default LIMIT is {top_k} unless specified.

QUESTION:
{input}

SQLQuery:
"""
        )

        chain = create_sql_query_chain(llm, db, prompt=custom_prompt)

        raw_response = chain.invoke({"input": prompt})

        clean_sql = extract_sql(raw_response)

        if not clean_sql:
            st.error("Model failed to generate SQL.")
            st.stop()

        if not is_safe(clean_sql):
            st.error("Unsafe SQL blocked.")
            st.stop()

        st.code(clean_sql, language="sql")

        success, rows, error = execute_sql(db, clean_sql)

        if success:
            st.success(f"Query executed | Rows returned: {rows}")
        else:
            st.error(f"Execution failed: {error}")

        log_query(prompt, clean_sql, success, rows, error)
