"""
ORPO training script for Qwen3.5-4B with Unsloth + TRL.
Fully stabilized for pure-text post-training runs over Qwen Vision backbones.
"""

import os
import gc
from pathlib import Path

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import ORPOConfig, ORPOTrainer

# ============================================================
# STABILITY / ENV
# ============================================================
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

# ============================================================
# CONFIG
# ============================================================
# Wrapped in abspath to prevent the Hugging Face Hub validation crash
MODEL_PATH = os.path.abspath("qwen3.5_4B_sft_merged")  
DATASET_PATH = "Dataset_orpo_perfect.jsonl"  
OUTPUT_DIR = "outputs_orpo"
LORA_OUTPUT_DIR = "qwen3.5_4B_orpo_lora"
MERGED_OUTPUT_DIR = "qwen3.5_4B_orpo_merged"
MAX_SEQ_LENGTH = 2048
MAX_PROMPT_LENGTH = 128
SEED = 42

USE_EVAL_SPLIT = True
EVAL_SPLIT_RATIO = 0.05

# ============================================================
# LOAD MODEL & TEXT-ONLY EXTRACTED TOKENIZER
# ============================================================
model, processor = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

# Passing this into ORPOTrainer bypasses the visual media checking loops completely,
# resolving the "Incorrect image source" tokenization error.
text_tokenizer = processor.tokenizer

# Maintain native right-padding alignment rules required for training
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

# Force efficient matrix allocation shapes
text_tokenizer.padding_side = "right" 

# Set LoRA targets for the Qwen3.5 architecture
model = FastLanguageModel.get_peft_model(
    model,
    r=64,                
    lora_alpha=128,      
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = FastLanguageModel.for_training(model)

# ============================================================
# DATA LOADING 
# ============================================================
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

train_dataset = dataset
eval_dataset = None

if USE_EVAL_SPLIT:
    split = dataset.train_test_split(test_size=EVAL_SPLIT_RATIO, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]

# ============================================================
# ORPO CONFIGURATION
# ============================================================
training_args = ORPOConfig(
    output_dir=OUTPUT_DIR,

    # LEARNING RATE: PARAMETER GRADIENT STEP SIZE
    #
    # Controls the size of the weight updates during backpropagation. Because 
    # ORPO compares relative token log-probabilities across dual paths, it requires
    # significantly smaller steps than standard SFT (2e-4) to prevent model collapse.
    #
    # SCALING MATRIX FOR AN EFFECTIVE 1x8 BATCH SIZE (With LoRA Rank 64 / Alpha 128):
    # - 4B / 7B / 9B Models:  Use 1e-5 to 3e-5 (Safe capacity for rapid convergence)
    # - 27B Models:           Use 5e-6 to 8e-6 (Requires smaller steps for dense weights)
    # - 32B / 35B Models:     Use 3e-6 to 5e-6 (Highly sensitive; higher LRs break reasoning)
    #
    # If your loss explosions or output degrades into repetitive text loops,
    # scale this down toward the lower bound of your model's target range.
    learning_rate=1e-5,          
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1, 
    gradient_accumulation_steps=8, # Your exact 2x4 layout preference
    num_train_epochs=3,
    bf16=True,
    fp16=False,
    logging_steps=1,
    save_strategy="steps",
    save_steps=200,                 # Your preferred saving frequency
    eval_strategy="steps" if eval_dataset is not None else "no",
    eval_steps=200 if eval_dataset is not None else None,
    report_to="none",

    # BETA: PREFERENCE LOGIT WEIGHT (Fixed at 0.05)
    #
    # Beta acts as the scaling multiplier for the odds-ratio contrastive loss.
    # It controls how harshly the model is penalized for choosing 'rejected' tokens
    # versus 'chosen' tokens. 
    #
    # - A value of 0.05 is highly calibrated for reasoning models. It lowers the 
    #   penalty slightly compared to the standard 0.1 baseline, preventing the 
    #   odds-ratio equation from overcorrecting on long <think> reasoning sequences.
    # - By keeping beta at 0.05, the trainer isolates the preference loss 
    #   primarily onto the character dialogue choices without causing the model's 
    #   broader foundational knowledge or syntax to break down.
    beta=0.05,                   
    max_length=MAX_SEQ_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
    disable_dropout=True,
    remove_unused_columns=False,
    dataset_num_proc=4,

    max_grad_norm=1.0,
    
    # MANDATORY: Added back to stop your RTX 4090 from OOM crashing on long thinking tokens
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_8bit",          # Compresses optimizer states to keep background memory lean
    
    seed=SEED,
)

# ============================================================
# TRAINER RUNNER
# ============================================================
trainer = ORPOTrainer(
    model=model,
    processing_class=text_tokenizer,  
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

gc.collect()
torch.cuda.empty_cache()

# ============================================================
# EXECUTE TRAINING
# ============================================================
print("=" * 60)
print("Starting Multimodal ORPO preference alignment phase...")
print(f"Train rows: {len(train_dataset)}")
if eval_dataset is not None:
    print(f"Eval rows:  {len(eval_dataset)}")
print("=" * 60)

trainer.train()

print("=" * 60)
print("ORPO training complete!")
print("=" * 60)

# ============================================================
# SAVE LORA ADAPTER
# ============================================================
print("Saving ORPO LoRA adapter...")
model.save_pretrained(LORA_OUTPUT_DIR)

# Reload a completely fresh, clean processor instance directly from the base path
from transformers import AutoProcessor
fresh_processor = AutoProcessor.from_pretrained(MODEL_PATH)
fresh_processor.save_pretrained(LORA_OUTPUT_DIR)  
print("ORPO LoRA saved successfully!")

# ============================================================
# CLEANUP + MERGE
# ============================================================
del trainer
gc.collect()
torch.cuda.empty_cache()

print("Merging ORPO layers into 16-bit precision SafeTensors...")
model.save_pretrained_merged(
    MERGED_OUTPUT_DIR,
    fresh_processor,  # Use the fresh processor to avoid reference errors
    save_method="merged_16bit",
)

print("=" * 60)
print("ALL PIPELINES SUCCESSFUL")
print(f"ORPO LoRA:   {LORA_OUTPUT_DIR}")
print(f"ORPO Merged: {MERGED_OUTPUT_DIR}")
print("=" * 60)
