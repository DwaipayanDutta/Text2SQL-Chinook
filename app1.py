import os
import random
import numpy as np
import torch
import re
import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sqlalchemy import create_engine, inspect, text
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =====================================================
# REPRODUCIBILITY
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(999)
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "cycloneboy/SLM-SQL-0.6B"

st.set_page_config(page_title="Text2SQL", layout="wide")


# =====================================================
# LOAD LLM (cached safely)
# =====================================================


@st.cache_resource
def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=700,
        temperature=0.0,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(pipeline=pipe)


llm = load_llm()


# =====================================================
# DATABASE ENGINE (cached)
# =====================================================


@st.cache_resource
def get_engine(db_uri):

    return create_engine(
        db_uri,
        connect_args={
            "check_same_thread": False,
            "timeout": 30,
        },
        pool_pre_ping=True,
    )


# =====================================================
def select_tables(engine, question, k=5):

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    q = question.lower()

    scored = []

    for table in tables:
        score = 0

        if table.lower() in q:
            score += 5

        for col in inspector.get_columns(table):
            if col["name"].lower() in q:
                score += 2

        if score > 0:
            scored.append((table, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored:
        return tables[:k]

    return [t[0] for t in scored[:k]]


# =====================================================
def build_schema(engine, tables):

    inspector = inspect(engine)
    parts = []

    for table in tables:

        cols = inspector.get_columns(table)

        column_lines = "\n".join(f"- {c['name']} ({c['type']})" for c in cols)

        fks = inspector.get_foreign_keys(table)

        fk_text = ""
        if fks:
            relations = [
                f"{table}.{fk['constrained_columns'][0]} → "
                f"{fk['referred_table']}.{fk['referred_columns'][0]}"
                for fk in fks
            ]
            fk_text = "\nFOREIGN KEYS:\n" + "\n".join(f"- {r}" for r in relations)

        parts.append(
            f"""
TABLE: {table}

COLUMNS:
{column_lines}
{fk_text}
"""
        )

    return "\n".join(parts)


# =====================================================
# PROMPT
# =====================================================

SQL_PROMPT = PromptTemplate.from_template(
    """
You are an elite SQLite engineer.

Generate a syntactically correct SQLite query.

STRICT RULES:
- Use ONLY tables shown below.
- Never invent tables.
- Never invent columns.
- Use column names EXACTLY as written.
- Pay attention to prefixes like Billing*, Ship*, etc.
- Prefer explicit JOINs.
- Avoid SELECT *
- Do NOT write anything before or after the query.

DATABASE SCHEMA:
{schema}

USER QUESTION:
{question}

FINAL SQL ONLY:
"""
)

parser = StrOutputParser()
chain = SQL_PROMPT | llm | parser

# =====================================================
# SQL EXTRACTION (production-grade)
# =====================================================


def extract_sql(text):
    if not text:
        return ""

    # Remove markdown blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove XML-style tags like <answer>
    text = re.sub(r"<.*?>", "", text, flags=re.DOTALL)

    # Grab the LAST real SQL statement
    matches = re.findall(
        r"(WITH\s+.*?SELECT\s+.*?FROM\s+.*?(?:;|$)|SELECT\s+.*?FROM\s+.*?(?:;|$))",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if matches:
        sql = matches[-1].strip()

        if not sql.endswith(";"):
            sql += ";"

        return sql

    return ""


# =====================================================
# SQL SAFETY
# =====================================================


def is_safe(sql):

    forbidden = [
        "DROP",
        "DELETE",
        "UPDATE",
        "ALTER",
        "INSERT",
        "TRUNCATE",
        "CREATE",
        "REPLACE",
        "ATTACH",
        "PRAGMA",
    ]

    return not any(re.search(rf"\b{f}\b", sql, re.IGNORECASE) for f in forbidden)


# =====================================================
# EXECUTION
# =====================================================


def run_sql(engine, sql):

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))

            if result.returns_rows:
                rows = result.fetchall()
                cols = result.keys()
                return True, rows, cols, None

            return True, [], [], None

    except Exception as e:
        return False, None, None, str(e)


# =====================================================
# ⭐ SELF-HEALING ENGINE
# =====================================================


def repair_sql(question, schema, bad_sql, error):

    repair_prompt = f"""
The following SQL failed.

QUESTION:
{question}

SCHEMA:
{schema}

BAD SQL:
{bad_sql}

ERROR:
{error}

Fix the SQL.

Return ONLY the corrected SQL.
"""

    repaired = llm.invoke(repair_prompt)
    return extract_sql(repaired)


def generate_with_retry(engine, question, retries=2):

    tables = select_tables(engine, question)
    schema = build_schema(engine, tables)

    raw = chain.invoke({"schema": schema, "question": question})

    sql = extract_sql(raw)

    if not sql:
        return None, None, "Model failed to generate SQL."

    if not is_safe(sql):
        return None, None, "Unsafe SQL blocked."

    ok, rows, cols, err = run_sql(engine, sql)

    attempt = 0

    while not ok and attempt < retries:

        sql = repair_sql(question, schema, sql, err)

        if not sql:
            break

        ok, rows, cols, err = run_sql(engine, sql)
        attempt += 1

    if not ok:
        return sql, None, err

    return sql, (rows, cols), None


# =====================================================
# UI
# =====================================================

st.title(" Text2SQL App")

db_uri = st.text_input("SQLite DB URI", value="sqlite:///Chinook_Sqlite.sqlite")

if db_uri:
    try:
        engine = get_engine(db_uri)
        st.success("Database connected")
    except Exception as e:
        st.error(str(e))
        st.stop()


if prompt := st.chat_input("Ask a database question..."):

    st.chat_message("user").write(prompt)

    with st.spinner("Generating SQL..."):

        sql, result, error = generate_with_retry(engine, prompt)

    if not sql:
        st.error(error)
        st.stop()

    st.code(sql, language="sql")

    if error:
        st.error(error)
    elif result:
        rows, cols = result
        st.success(f"{len(rows)} rows returned")
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("Query executed successfully.")
