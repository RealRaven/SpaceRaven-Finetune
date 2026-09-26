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

model_path = "Qwen/Qwen3-4B-Instruct-2507"

output_dir = "outputs"

lora_output_dir = "qwen3_4B_instruct_lora"

merged_output_dir = "qwen3_4B_instruct_merged"

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
    chat_template="qwen3-instruct",
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
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model.config.use_cache = False

# ============================================================
# LOAD + SPLIT DATA (OPTION 2 FIX)
# ============================================================

df = pd.read_json("Dataset_3000.json")

df = df.sample(frac=1, random_state=3407).reset_index(drop=True)

dataset = Dataset.from_pandas(df)

# 95% train / 5% eval
dataset = dataset.train_test_split(test_size=0.05, seed=3407)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ============================================================
# FORMAT DATA
# ============================================================

SYSTEM_PROMPT = (
    "You are sarcastic, rebellious, witty, charming, "
    "street-smart, emotionally guarded, confident, and cynical. "
    "You speak casually and naturally. "
    "You do not speak like an AI assistant."
)

def formatting_prompts_func(examples):

    instructions = examples["instruction"]
    outputs = examples["response"]

    texts = []

    for instruction, output in zip(instructions, outputs):

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

        text += tokenizer.eos_token
        texts.append(text)

    return {"text": texts}

train_dataset = train_dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=train_dataset.column_names,
)

eval_dataset = eval_dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

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

        # ====================================================
        # BATCHING
        # ====================================================

        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,

        # ====================================================
        # TRAINING
        # ====================================================

        num_train_epochs=15,
        learning_rate=1e-4,

        warmup_ratio=0.05,
        lr_scheduler_type="cosine",

        weight_decay=0.01,

        # ====================================================
        # PRECISION
        # ====================================================

        bf16=True,
        fp16=False,

        # ====================================================
        # MEMORY
        # ====================================================

        gradient_checkpointing=True,
        optim="adamw_8bit",

        # ====================================================
        # DATASET
        # ====================================================

        packing=False,
        dataset_num_proc=2,
        assistant_only_loss=False,

        # ====================================================
        # EVAL + BEST MODEL FIX (OPTION 2)
        # ====================================================

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # ====================================================
        # LOGGING
        # ====================================================

        logging_steps=1,
        output_dir=output_dir,
        report_to="none",

        seed=3407,
    ),
)

# ============================================================
# TRAIN
# ============================================================

print("=" * 60)
print("Starting training...")
print("=" * 60)

trainer.train()

print("=" * 60)
print("Training complete!")
print("=" * 60)

# ============================================================
# SAVE LORA
# ============================================================

print("Saving LoRA adapter...")

model.save_pretrained(lora_output_dir)
tokenizer.save_pretrained(lora_output_dir)

print("LoRA saved!")

# ============================================================
# CLEANUP + MERGE
# ============================================================

gc.collect()
torch.cuda.empty_cache()

print("Merging model...")

model.save_pretrained_merged(
    merged_output_dir,
    tokenizer,
    save_method="merged_16bit",
)

print("=" * 60)
print("DONE")
print(f"LoRA: {lora_output_dir}")
print(f"Merged: {merged_output_dir}")
print("=" * 60)