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
max_seq_length = 512
model_path = "Qwen/Qwen3-0.6B"
output_dir = "outputs"
lora_output_dir = "qwen3_0.6B_lora"
merged_output_dir = "qwen3_0.6B_merged"

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
# We use the default Qwen template. Because we removed the system 
# prompt, we rely entirely on the user/assistant turns.
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
# Update the JSON filename to match your new generated dataset
df = pd.read_json("Dataset_shuffled_final.json")
df = df.sample(frac=1, random_state=3407).reset_index(drop=True)
dataset = Dataset.from_pandas(df)

# 95% train / 5% eval
dataset = dataset.train_test_split(test_size=0.05, seed=3407)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ============================================================
# FORMAT DATA FOR A THINKING MODEL
# ============================================================
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    thoughts = examples["thought"]
    outputs = examples["output"]
    texts = []

    for instruction, thought, output in zip(instructions, thoughts, outputs):
        # We concatenate the thought and the output as the assistant's full response.
        # This trains the model to always output the <think> block before the final answer.
        full_assistant_response = f"{thought}\n{output}"

        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": full_assistant_response},
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
        
        # BATCHING
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        
        # TRAINING
        num_train_epochs=10,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        
        # PRECISION
        bf16=True,
        fp16=False,
        
        # MEMORY
        gradient_checkpointing=False,
        optim="adamw_8bit",
        
        # DATASET
        packing=True,
        dataset_num_proc=2,
        assistant_only_loss=False,
        
        # EVAL + BEST MODEL
        eval_strategy="steps",           # With packing, "epoch" can take a long time to hit
        eval_steps=50,                   # Check progress every 50 steps
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
print("Starting (Thinking Mode) training...")
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