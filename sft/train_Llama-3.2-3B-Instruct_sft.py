import os
import gc
import torch
import pandas as pd

from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig

# ============================================================
# STABILITY FIXES
# ============================================================

os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# CONFIG
# ============================================================

max_seq_length = 2048

model_path = "meta-llama/Llama-3.2-3B-Instruct"  # CHANGE HERE

output_dir = "outputs_llama_3.2_3B_instruct"

lora_output_dir = "llama_3.2_3B_instruct_lora"

merged_output_dir = "llama_3.2_3B_instruct_merged"

# ============================================================
# LOAD MODEL
# ============================================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    attn_implementation="eager",
)

# ============================================================
# CHAT TEMPLATE
# ============================================================

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
)

# ============================================================
# LORA
# ============================================================

model = FastLanguageModel.get_peft_model(
    model,

    r=64,
    lora_alpha=128,
    lora_dropout=0.0,

    bias="none",

    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],
)

model.config.use_cache = False

# ============================================================
# LOAD DATA (DATASET)
# ============================================================

df = pd.read_json("Dataset_3000.json")  # your dataset

df = df.sample(frac=1, random_state=3407).reset_index(drop=True)

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.05, seed=3407)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ============================================================
# SYSTEM PROMPT (STRICT PERSONA LOCK)
# ============================================================

SYSTEM_PROMPT = (
    "calm, authoritative, disciplined, and direct. "
    "You speak with precision and authority. "
    "You do NOT break character. "
    "You never mention being an AI or language model."
)

# ============================================================
# FORMAT DATA
# ============================================================

def formatting_prompts_func(examples):

    texts = []

    for instruction, output in zip(examples["instruction"], examples["response"]):

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        texts.append(text)

    return {"text": texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True, remove_columns=eval_dataset.column_names)

# ============================================================
# TRAINER
# ============================================================

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,

    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,

        num_train_epochs=15,
        learning_rate=1e-4,

        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,
        optim="adamw_8bit",

        packing=False,
        dataset_num_proc=2,

        assistant_only_loss=False,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=1,
        output_dir=output_dir,
        report_to="none",

        seed=3407,
    ),
)

# ============================================================
# TRAIN
# ============================================================

print("Starting training...")

trainer.train()

print("Training complete!")

# ============================================================
# SAVE
# ============================================================

model.save_pretrained(lora_output_dir)
tokenizer.save_pretrained(lora_output_dir)

gc.collect()
torch.cuda.empty_cache()

model.save_pretrained_merged(
    merged_output_dir,
    tokenizer,
    save_method="merged_16bit",
)

print("DONE:", merged_output_dir)