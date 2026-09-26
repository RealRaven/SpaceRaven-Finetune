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
#os.environ["UNSLOTH_DISABLE_TRITON"] = "1"
#os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# CONFIG
# ============================================================
max_seq_length = 256
model_path = "microsoft/phi-2"  # Explicitly configured for your Phi-2 base model
output_dir = "outputs"
lora_output_dir = "phi2_lora"
merged_output_dir = "phi2_merged"

# ============================================================
# LOAD MODEL
# ============================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16
)

# ============================================================
# CHAT TEMPLATE (Fixed for Phi-2 Base Model)
# ============================================================
# Phi-2 base doesn't have a native template. We use 'chatml' 
# to structurally separate User vs Assistant blocks during training.
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml", 
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# ============================================================
# LORA
# ============================================================
for name, module in model.named_modules():
    if any(x in name.lower() for x in ["q", "k", "v", "proj", "fc"]):
        print(name)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=128,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    target_modules=[
        "Wqkv",
        "out_proj",
        "fc1",
        "fc2",
    ]
)
model.print_trainable_parameters()
model.config.use_cache = False

# ============================================================
# LOAD + SPLIT DATA (Dataset_3000.json)
# ============================================================
df = pd.read_json("Dataset_3000.json")
df = df.sample(frac=1, random_state=3407).reset_index(drop=True)
dataset = Dataset.from_pandas(df)

# 95% train / 5% eval
dataset = dataset.train_test_split(test_size=0.05, seed=3407)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ============================================================
# FORMAT DATA FOR A STANDARD BASE MODEL
# ============================================================
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    responses = examples["response"]  
    texts = []

    for instruction, response in zip(instructions, responses):
        # Clean conversational mapping without artificial <think> tokens 
        # that would cause vocabulary dilution in a non-thinking model.
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        text = text.strip() + tokenizer.eos_token
            
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
    tokenizer=tokenizer,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        
        # BATCHING
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        
        # TRAINING
        num_train_epochs=5,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # PRECISION
        bf16=True,
        fp16=False,
        
        # MEMORY 
        optim="adamw_8bit",
        
        # DATASET
        packing=True,                  
        dataset_num_proc=2,
        assistant_only_loss=False,
        
        # EVAL + BEST MODEL
        eval_strategy="steps",           
        eval_steps=100,                  
        save_strategy="steps",
        save_steps=100,
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
print("Starting Phi-2 Conversational Mode training...")
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