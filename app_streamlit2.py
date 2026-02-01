import os
import random
import numpy as np
import torch
import re
import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from sqlalchemy import inspect
from langchain_core.prompts import PromptTemplate

MODEL_PATH = r"cycloneboy\SLM-SQL-0.6B"


# --- STEP 1: GLOBAL DETERMINISM ---
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(999)

st.set_page_config(page_title="SQL Generator", page_icon="🔍", layout="wide")


# --- 2. Reliable Mermaid Renderer ---
def render_mermaid(code: str):
    """Renders Mermaid.js code using a Streamlit component."""
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
        mermaid_code += f"    {table} {{\n"
        for col in inspector.get_columns(table):
            col_name = col["name"]
            col_type = str(col["type"]).split("(")[0]
            mermaid_code += f"        {col_type} {col_name}\n"
        mermaid_code += "    }\n"
        for fk in inspector.get_foreign_keys(table):
            target_table = fk["referred_table"]
            mermaid_code += f'    {table} ||--o{{ {target_table} : ""\n'
    return mermaid_code


# --- 3. Sidebar & DB Connection ---
with st.sidebar:
    st.header("Database Config")
    db_uri = st.text_input("Database URI", value="sqlite:///chinook_sqlite.sqlite")

    db = None
    if db_uri:
        try:
            db = SQLDatabase.from_uri(db_uri)
            st.success("Connected")
            if st.button("View/Refresh ER Diagram"):
                with st.expander("Database ER Diagram", expanded=True):
                    er_code = generate_mermaid_er(db._engine)
                    render_mermaid(er_code)
        except Exception as e:
            st.error(f"Error: {e}")


# --- 4. Model Loading ---
@st.cache_resource
def load_llm():
    # model_id = "cycloneboy/SLM-SQL-0.6B"
    model_id = MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


llm = load_llm()


# --- 5. Pure SQL Extraction ---
def extract_sql(text: str) -> str:
    # Remove markdown code blocks and specific model tags
    text = re.sub(r"```sql|```", "", text)
    if "SQLQuery:" in text:
        text = text.split("SQLQuery:")[-1]
        text = text.split("SQL query:")[-1]
    # Stop at the first semicolon to prevent trailing chat/explanation
    text = text.split(";")[0].strip()
    return text + ";" if text else ""


def is_safe(sql: str) -> bool:
    forbidden = ["DROP", "DELETE", "UPDATE", "ALTER", "INSERT", "TRUNCATE"]
    return not any(re.search(rf"\b{word}\b", sql, re.IGNORECASE) for word in forbidden)


# --- 6. Main UI ---
st.title("SQL Query Generator")

if prompt := st.chat_input("Enter your request..."):
    if not db:
        st.error("Please connect a database in the sidebar.")
    else:
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            custom_prompt = PromptTemplate(
                input_variables=["input", "table_info", "top_k"],
                template="""You are a SQLite expert. Use the provided Table Info as your ER Diagram to create a join-heavy SQLite query.

            ### JOIN RULES:
            1. Many relationships require bridge tables. For example, to get Artist details for a Track, you MUST join Track -> Album -> Artist.
            2. Carefully trace the Foreign Keys (FK) in the Table Info below to connect tables.
            3. If a column isn't in Table A, check Table B's schema for a relationship.

            ### TASK:
            - Return ONLY the SQL query. No explanation.
            - Use a LIMIT of {top_k} unless otherwise requested.
            - Use correct join paths based on the ER schema. Do not miss any joins

            ### TABLE INFO (ER SCHEMA):
            {table_info}

            ### QUESTION: 
            {input}

            SQLQuery:""",
            )
            chain = create_sql_query_chain(llm, db, prompt=custom_prompt)
            raw_response = chain.invoke({"question": prompt})

            clean_sql = extract_sql(raw_response)

            if clean_sql and is_safe(clean_sql):
                # Strictly output only the code block
                st.code(clean_sql, language="sql")
            elif not clean_sql:
                st.error("Model failed to generate a valid query.")
            else:
                st.error("Blocked: Unauthorized DDL/DML operation detected.")
