import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =====================================================
# CONFIG
# =====================================================

BASE_MODEL = "cycloneboy/SLM-SQL-0.6B"
DATA_PATH = "/mnt/data/chinook_train_large.json"
OUTPUT_DIR = "./chinook_sql_adapter"


# =====================================================
# LOAD MODEL (NO QUANTIZATION)
# =====================================================

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,  # use float32 if GPU not available
    trust_remote_code=True,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# =====================================================
# DATASET
# =====================================================

dataset = load_dataset("json", data_files=DATA_PATH)["train"]

SCHEMA = """
Artist(ArtistId, Name)
Album(AlbumId, Title, ArtistId)
Track(TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice)
Invoice(InvoiceId, CustomerId, BillingCountry, Total)
Customer(CustomerId, FirstName, LastName, Country, SupportRepId)
"""


def format_example(example):
    return {
        "text": f"""### Instruction:
Generate a SQLite query using this schema.

### Schema:
{SCHEMA}

### Question:
{example['input']}

### SQL:
{example['output']}"""
    }


dataset = dataset.map(format_example, remove_columns=dataset.column_names)


# =====================================================
# LoRA CONFIG
# (slightly larger since we are NOT quantized)
# =====================================================

peft_config = LoraConfig(
    r=64,  # safe now — no quantization instability
    lora_alpha=128,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


# =====================================================
# TRAINING CONFIG
# =====================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    push_to_hub=False,
    per_device_train_batch_size=2,  # fp16 uses more VRAM
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=5,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,  # set False if CPU training
    bf16=False,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_seq_length=768,
    packing=True,
    report_to="none",
)


# =====================================================
# TRAIN
# =====================================================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Adapter saved to: {OUTPUT_DIR}\n")
