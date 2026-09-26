import os
import gc
import torch
import math

from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import DataCollatorForSeq2Seq
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
max_seq_length = 4048
model_path = "Qwen/Qwen3.5-4B"
output_dir = "outputs"
lora_output_dir = "qwen3.5_4B_lora"
merged_output_dir = "qwen3.5_4B_merged"

# ============================================================
# LOAD MODEL & NATIVE MULTIMODAL PROCESSOR
# ============================================================
model, processor = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

# ============================================================
# LORA SETUP
# ============================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=128,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)
model.config.use_cache = False

# ============================================================
# LOAD DATASET + NATIVE UNSLOTH COMPATIBILITY STEP
# ============================================================
dataset = load_dataset("json", data_files="Dataset_stf.jsonl", split="train")

# FIX 1: Pass the dataset items through the processor's template renderer
# to convert the messages matrix into a tokenized training mapping that Unsloth understands.
def format_multimodal_dataset(examples):
    # This invokes the core Jinja layout compilation over your dataset schema
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in examples["messages"]
    ]
    return {"text": texts}

# Map it to populate a clean "text" column that SFTTrainer expects for validation
dataset = dataset.map(format_multimodal_dataset, batched=True)

# Shuffle and Split (95% Train / 5% Eval)
dataset = dataset.shuffle(seed=3407)
dataset = dataset.train_test_split(test_size=0.05, seed=3407)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Explicitly calculate training steps to switch from deprecated warmup_ratio to warmup_steps
# 3093 samples * 0.95 = ~2938 train samples. Batch size 2 * 16 grad accumulation = 32 samples per step.
# 2938 / 32 = ~91 steps per epoch. 4 epochs = ~364 total steps. 10% warmup = 36 steps.
num_train_samples = len(train_dataset)
steps_per_epoch = math.ceil(num_train_samples / (2 * 16))
total_training_steps = steps_per_epoch * 4
calculated_warmup_steps = math.ceil(total_training_steps * 0.1)

# Sequence-to-Sequence data collator manages the user-loss token masking masks
data_collator = DataCollatorForSeq2Seq(
    processor, 
    model=model, 
    label_pad_token_id=-100, 
    pad_to_multiple_of=8
)

# ============================================================
# TRAINER CONFIGURATION
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=processor,  
    data_collator=data_collator, 
    args=SFTConfig(
        # FIX 2: Point to the compiled template text field to satisfy Unsloth's runtime checker
        dataset_text_field="text", 
        max_seq_length=max_seq_length,
        
        # BATCHING (Optimized 4x8 Configuration)
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        
        # LEARNING HYPERPARAMETERS
        num_train_epochs=4,
        learning_rate=2e-4,
        warmup_steps=calculated_warmup_steps, # FIX 3: Replaced deprecated warmup_ratio with steps
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        
        # PRECISION
        bf16=True,
        fp16=False,
        
        # ENGINE STABILITY
        gradient_checkpointing=True,
        optim="adamw_8bit",
        packing=False,  
        dataset_num_proc=2,
        
        # EVAL + CHECKPOINTS
        eval_strategy="steps",           
        eval_steps=20, # Reduced step interval since total steps are ~364                  
        save_strategy="steps",
        save_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # CONTROL LOGS
        logging_steps=1,
        output_dir=output_dir,
        report_to="none",
        seed=3407,
    ),
)

# ============================================================
# EXECUTE TRAINING
# ============================================================
print("=" * 60)
print(f"Launching Qwen3.5-VL SFT Run...")
print(f"Total Train Samples: {num_train_samples}")
print(f"Total Estimated Steps: {total_training_steps} | Warmup Steps: {calculated_warmup_steps}")
print("=" * 60)
trainer.train()

# ============================================================
# EXPORT AND EXTRAPOLATE LOOPS
# ============================================================
print("Saving LoRA adapters...")
model.save_pretrained(lora_output_dir)
processor.save_pretrained(lora_output_dir)

del trainer
gc.collect()
torch.cuda.empty_cache()

print("Consolidating layers into unified 16-bit precision SafeTensors...")
model.save_pretrained_merged(
    merged_output_dir,
    processor,
    save_method="merged_16bit",
)

print("=" * 60)
print("TRAINING PROCESS SUCCESSFUL")
print(f"LoRA Target:  {lora_output_dir}")
print(f"Merged Model: {merged_output_dir}")
print("=" * 60)
