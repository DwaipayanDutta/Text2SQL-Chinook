
# Loading the packages 
import json
import os
import sqlite3
import tempfile
from dotenv import load_dotenv
import streamlit as st
from transformers import AutoTokenizer

# Art framework imports
import art
from art.serverless.backend import ServerlessBackend
from art.local.backend import LocalBackend

# Disable Weights & Biases logging
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_API_KEY"] = "dummy"


# ENV
load_dotenv()
OPENPIPE_API_KEY = os.getenv("OPENPIPE_API_KEY")


# CONSTANTS
BASE_MODEL = "cycloneboy/SLM-SQL-0.6B"
PROJECT_NAME = "chinook_text_to_sql_lora"
MODEL_NAME = "slm_sql_chinook_lora"
DB_PATH = "chinook.db"

CHINOOK_SCHEMA = """
CREATE TABLE Artist (
    ArtistId INTEGER PRIMARY KEY,
    Name TEXT
);
CREATE TABLE Album (
    AlbumId INTEGER PRIMARY KEY,
    Title TEXT,
    ArtistId INTEGER
);
CREATE TABLE Track (
    TrackId INTEGER PRIMARY KEY,
    Name TEXT,
    AlbumId INTEGER,
    GenreId INTEGER,
    Milliseconds INTEGER,
    UnitPrice REAL
);
CREATE TABLE Genre (
    GenreId INTEGER PRIMARY KEY,
    Name TEXT
);
CREATE TABLE Customer (
    CustomerId INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Country TEXT
);
CREATE TABLE Invoice (
    InvoiceId INTEGER PRIMARY KEY,
    CustomerId INTEGER,
    InvoiceDate TEXT,
    BillingCountry TEXT,
    Total REAL
);
CREATE TABLE InvoiceLine (
    InvoiceLineId INTEGER PRIMARY KEY,
    InvoiceId INTEGER,
    TrackId INTEGER,
    Quantity INTEGER,
    UnitPrice REAL
);
"""


# SQLITE HELPERS
def execute_sql(sql: str):
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed")

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    conn.close()
    return cols, sorted(rows)

# =================================================
# STREAMLIT UI
# =================================================
st.set_page_config("Chinook Text-to-SQL (ART + LoRA)", layout="wide")
st.title("Chinook Text-to-SQL Trainer")
st.caption("SLM-SQL-0.6B • LoRA • ART (Legacy API) • SQLite")

# =================================================
# SIDEBAR
# =================================================
st.sidebar.header("Training Settings")

backend_choice = st.sidebar.selectbox(
    "ART Backend",
    ["Local GPU", "Serverless (Cloud GPU)"]
)

epochs = st.sidebar.slider("Epochs", 1, 10, 3)
learning_rate = st.sidebar.selectbox(
    "Learning Rate", [5e-4, 2e-4, 1e-4], index=1
)

lora_r = st.sidebar.selectbox("LoRA Rank (r)", [8, 16, 32], index=1)
lora_alpha = st.sidebar.selectbox("LoRA Alpha", [16, 32, 64], index=1)

# =================================================
# DATASET UPLOAD
# =================================================
st.header("Upload Chinook Training Dataset (.jsonl)")

uploaded = st.file_uploader("Upload dataset", type="jsonl")
gold_sql = {}

if uploaded:
    rows = [json.loads(l) for l in uploaded]
    gold_sql = {r["question"]: r["sql"] for r in rows}
    st.success(f"Loaded {len(rows)} samples")

    def build_prompt(r):
        return f"""
You are a Text-to-SQL model.
Database schema:
{CHINOOK_SCHEMA}

Question:
{r['question']}

SQL:
{r['sql']}
""".strip()

    tmp_dir = tempfile.mkdtemp()
    dataset_path = os.path.join(tmp_dir, "train.jsonl")

    with open(dataset_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": build_prompt(r)}) + "\n")

    if st.button(" Train with LoRA"):
        backend = (
            ServerlessBackend(api_key=OPENPIPE_API_KEY)
            if backend_choice.startswith("Serverless")
            else LocalBackend()
        )

        model = art.TrainableModel(
            project=PROJECT_NAME,
            name=MODEL_NAME,
            base_model=BASE_MODEL,
            peft_config={
                "type": "lora",
                "r": lora_r,
                "lora_alpha": lora_alpha,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
            },
        )

        model.register(backend)

        # LEGACY ART API — ALL POSITIONAL ARGUMENTS
        model.train(
            dataset_path,
            epochs,
            learning_rate,
        )

        st.success("Training started successfully")


# INFERENCE
st.divider()
st.header("Ask Questions About Chinook")

st.code(CHINOOK_SCHEMA, language="sql")

question = st.text_input(
    "Question",
    placeholder="Which artists have the most albums?"
)

if st.button("Generate & Validate"):
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    prompt = f"""
You are a Text-to-SQL model.
Database schema:
{CHINOOK_SCHEMA}

Question:
{question}

SQL:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt")

    # Legacy ART inference — NO DeployedModel
    model = art.TrainableModel(
        project=PROJECT_NAME,
        name=MODEL_NAME,
        base_model=BASE_MODEL,
    )

    output = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=200,
        temperature=0.2,
    )

    gen_sql = tokenizer.decode(output[0], skip_special_tokens=True)

    st.subheader("Generated SQL")
    st.code(gen_sql, language="sql")

    if question in gold_sql:
        try:
            gen_cols, gen_rows = execute_sql(gen_sql)
            gold_cols, gold_rows = execute_sql(gold_sql[question])

            if (gen_cols, gen_rows) == (gold_cols, gold_rows):
                st.success("Execution Match")
            else:
                st.error("Execution Mismatch")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Generated Result**")
                    st.dataframe(gen_rows)
                with c2:
                    st.markdown("**Gold Result**")
                    st.dataframe(gold_rows)

        except Exception as e:
            st.error(f"SQL Execution Error: {e}")
