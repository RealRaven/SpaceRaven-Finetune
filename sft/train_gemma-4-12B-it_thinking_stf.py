import os
import gc
import json
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
max_seq_length = 8192
model_path = "TrueRealRaven/gemma-4-12B-it-heretic-v1"

output_dir       = "gemma4_12B_sft_outputs"
lora_output_dir  = "gemma4_12B_sft_lora"
merged_output_dir = "gemma4_12B_sft_merged"

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
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4-thinking")
print("Using gemma-4-thinking chat template (native thinking format)")

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
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
)
model.config.use_cache = False

# ============================================================
# LOAD DATA
# ============================================================
dataset = Dataset.from_json("Dataset_stf.jsonl")
dataset = dataset.shuffle(seed=3407)

train_dataset = dataset
eval_dataset  = dataset

# ============================================================
# 🛠️ FIXED: CHAT TEMPLATE COMPILER (RETURNS LIST OF STRINGS)
# ============================================================
def formatting_prompts_func(examples):
    # Ensure we isolate the raw message arrays regardless of batch structure
    messages_data = examples["messages"]
    texts = []

    # Scenario 1: Input is a batch / list of conversations
    if isinstance(messages_data, list) and len(messages_data) > 0 and isinstance(messages_data[0], list):
        for convo in messages_data:
            text = tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            )
            texts.append(text)
            
    # Scenario 2: Input is a single dictionary conversation (during internal testing checks)
    elif isinstance(messages_data, list) and len(messages_data) > 0 and isinstance(messages_data[0], dict):
        text = tokenizer.apply_chat_template(
            messages_data,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        texts.append(text)
        
    # Scenario 3: Fallback if individual elements are unpacked at the root level
    elif isinstance(examples, dict) and "role" in examples:
        text = tokenizer.apply_chat_template(
            [examples],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        texts.append(text)

    # 🛠️ CRITICAL FIX: Returns a flat Python list of raw strings directly 
    # instead of a wrapped dictionary structure, satisfying SFTTrainer.
    return texts

# ============================================================
# TRAINER 
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,
    args=SFTConfig(
        dataset_text_field="text", 
        max_seq_length=max_seq_length,

        # BATCHING
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,

        # TRAINING
        num_train_epochs=1,
        learning_rate=1e-4,
        warmup_steps=10, 
        lr_scheduler_type="cosine",
        weight_decay=0.01,

        # PRECISION
        bf16=True,
        fp16=False,

        # MEMORY
        gradient_checkpointing=False, 
        optim="adamw_8bit",

        # DATASET
        packing=False,
        dataset_num_proc=8,

        # EVAL + BEST MODEL 
        eval_strategy="no",
        eval_accumulation_steps=None,   
        per_device_eval_batch_size=1,
        save_strategy="epoch",        
        save_total_limit=2,
        load_best_model_at_end=False,
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
print("Starting Gemma 4 training...")
print("=" * 60)
trainer.train()

print("=" * 60)
print("Training complete!")
print("=" * 60)

# ============================================================
# SAVE LORA
# ============================================================
print("Saving LoRA adapter...")
trainer.model.save_pretrained(lora_output_dir)  
tokenizer.save_pretrained(lora_output_dir)
print("LoRA saved!")

# ============================================================
# CLEANUP + MERGE
# ============================================================
gc.collect()
torch.cuda.empty_cache()

print("Merging model...")
trainer.model.save_pretrained_merged(           
    merged_output_dir,
    tokenizer,
    save_method="merged_16bit",
)

# ============================================================
# POST-MERGE EOS PATCH
# ============================================================
cfg_path = os.path.join(merged_output_dir, "tokenizer_config.json")
try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    original = cfg.get("eos_token")
    if original != "<turn|>":
        cfg["eos_token"] = "<turn|>"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"🔧 Patched eos_token in {cfg_path}: {original!r} -> '<turn|>'")
    else:
        print(f"✅ eos_token in {cfg_path} is already '<turn|>'")
except FileNotFoundError:
    print(f"⚠️  {cfg_path} not found — skipping eos_token patch")
except Exception as e:
    print(f"⚠️  Failed to patch eos_token: {e}")

print("=" * 60)
print("DONE")
print(f"LoRA:   {lora_output_dir}")
print(f"Merged: {merged_output_dir}")
print("=" * 60)
