import os
import gc
import torch

from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import ORPOTrainer, ORPOConfig

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
max_prompt_length = 256

model_path = "qwen3_4B_thinking_2507_sft_merged"
output_dir = "qwen3_4B_thinking_2507_orpo_outputs"
lora_output_dir = "qwen3_4B_thinking_2507_orpo_qwen3_lora"
merged_output_dir = "qwen3_4B_thinking_2507_orpo_qwen3_merged"

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
# LOAD + SPLIT DATA
# ============================================================
dataset = Dataset.from_json("Dataset_orpo.jsonl")

dataset = dataset.shuffle(seed=3407)
dataset = dataset.train_test_split(test_size=0.05, seed=3407)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ============================================================
# ORPO TRAINER
# ============================================================
trainer = ORPOTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    args=ORPOConfig(
        max_length=max_seq_length,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        
        # ORPO SPECIFIC HYPERPARAMETERS
        beta=0.1,                        
        
        # BATCHING
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,  
        
        # TRAINING 
        num_train_epochs=4,              
        learning_rate=8e-6,              
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        
        # PRECISION
        bf16=True,
        fp16=False,
        
        # MEMORY
        gradient_checkpointing=True,     
        optim="adamw_8bit",              

        # DATASET MULTI-THREADING 
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
        logging_steps=5,                 
        output_dir=output_dir,
        report_to="none",
        seed=3407,                       
    ),
)

# ============================================================
# TRAIN
# ============================================================
print("=" * 60)
print("Starting ORPO Preference Alignment Mode training...")
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