import os
import gc
import torch

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
model_path = "Qwen/Qwen3-4B-Thinking-2507"
output_dir = "qwen3_4B_thinking_2507_sft_outputs"
lora_output_dir = "qwen3_4B_thinking_2507_sft_lora"
merged_output_dir = "qwen3_4B_thinking_2507_sft_merged"

# ============================================================
# LOAD MODEL
# ============================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

# ============================================================
# CHAT TEMPLATE
# ============================================================
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3",
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
# LOAD + SPLIT DATA (Updated for JSONL)
# ============================================================
# Load JSONL directly into a Hugging Face Dataset object
dataset = Dataset.from_json("Dataset_sft.jsonl")
dataset = dataset.shuffle(seed=3407)

# 95% train / 5% eval
dataset = dataset.train_test_split(test_size=0.05, seed=3407)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ============================================================
# FORMAT DATA FOR A THINKING MODEL (Updated for JSONL layout)
# ============================================================
def formatting_prompts_func(examples):
    # Since our JSONL already formatted the keys into standard 'messages',
    # we iterate through the list of chats and apply the template.
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not text.endswith(tokenizer.eos_token):
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
        
        # BATCHING
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        
        # TRAINING
        num_train_epochs=5,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        
        # PRECISION
        bf16=True,
        fp16=False,
        
        # MEMORY
        gradient_checkpointing=True,     # Swapped to True for stability
        optim="adamw_8bit",
        
        # DATASET
        packing=False,                   # Set to False to allow completion masking to work
        dataset_num_proc=8,
        
        # EVAL + BEST MODEL
        eval_strategy="steps",           
        eval_steps=50,                   
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # LOGGING
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