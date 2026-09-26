import os
import gc
import torch
import math
import warnings

from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import DataCollatorForSeq2Seq, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# Silence warnings and clean terminal tracking
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

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
model_id = "mistralai/Ministral-3-3B-Reasoning-2512"
output_dir = "outputs_ministral_3_3B_reasoning_2512"
lora_output_dir = "ministral_3_3B_reasoning_2512_lora"
merged_output_dir = "ministral_3_3B_reasoning_2512_merged"

# ============================================================
# LOAD MODEL & TEXT TOKENIZER INTERFACE
# ============================================================
model, processor = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    trust_remote_code=True
)

# Isolate the underlying text tokenizer from the processor wrapper object
tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

# Ensure the specialized <|thought|> reasoning token exists in the model embeddings
if "<|thought|>" not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|thought|>"]})
    model.resize_token_embeddings(len(tokenizer))

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
# LOAD DATASET & DIRECT REASONING STRUCTURAL ALIGNMENT
# ============================================================
# Natively unrolls JSON Lines format line by line
dataset = load_dataset("json", data_files="Dataset_3000.json", split="train")

# Set up an inference fallback template for later token deployment
official_mistral3_template = (
    "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
            "{{ '<s>[INST] ' + message['content']['text'] + ' [/INST]<|thought|>\n' }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ message['content']['text'] + '</s>' }}"
        "{% endif %}"
    "{% endfor %}"
)
tokenizer.chat_template = official_mistral3_template

def format_reasoning_dataset(examples):
    texts = []
    # Extract structural rows directly out of the JSONL features matrix
    prompts = examples["prompt"]
    thoughts = examples["thought"]
    responses = examples["response"]
    
    for i in range(len(prompts)):
        # Hard-weld the tokens so loss calculation locks cleanly onto structural boundaries
        rendered_text = (
            f"<s>[INST] {str(prompts[i]).strip()} [/INST]<|thought|>\n"
            f"{str(thoughts[i]).strip()}</s>"
            f"{str(responses[i]).strip()}</s>"
        )
        texts.append(rendered_text)
    return {"text": texts}

# Map processing cleanly across explicit JSONL row blocks
dataset = dataset.map(format_reasoning_dataset, batched=True)

# Shuffle and Split (95% Train / 5% Eval)
dataset = dataset.shuffle(seed=3407)
dataset = dataset.train_test_split(test_size=0.05, seed=3407)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Explicitly calculate training steps to avoid deprecated parameters
num_train_samples = len(train_dataset)
steps_per_epoch = math.ceil(num_train_samples / (2 * 16))
total_training_steps = steps_per_epoch * 4
calculated_warmup_steps = math.ceil(total_training_steps * 0.1)

# Sequence-to-Sequence data collator manages label sequence token masking
data_collator = DataCollatorForSeq2Seq(
    tokenizer, 
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
    processing_class=tokenizer,  
    data_collator=data_collator, 
    args=SFTConfig(
        dataset_text_field="text", 
        max_seq_length=max_seq_length,
        
        # BATCHING (Optimized 4x8 Configuration)
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        
        # LEARNING HYPERPARAMETERS
        num_train_epochs=4,
        learning_rate=2e-4,
        warmup_steps=calculated_warmup_steps, 
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
        eval_steps=20,                    
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
print(f"Launching Ministral-3 Reasoning SFT Run...")
print(f"Total Train Samples: {num_train_samples}")
print(f"Total Estimated Steps: {total_training_steps} | Warmup Steps: {calculated_warmup_steps}")
print("=" * 60)
trainer.train()

# ============================================================
# EXPORT LOOPS
# ============================================================
print("Saving LoRA adapters...")
model.save_pretrained(lora_output_dir)
tokenizer.save_pretrained(lora_output_dir)

# Safely purge training VRAM allocations to avoid out-of-memory errors during consolidation
del trainer, model
gc.collect()
torch.cuda.empty_cache()

# FIX: Explicit manual PEFT layer merge loop for reasoning architectures
print("Consolidating layers via manual PEFT merge into 16-bit precision SafeTensors...")
from peft import PeftModel

# Load the un-quantized base model explicitly in 16-bit to preserve data distributions
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu", # Keeps GPU available for operations
    trust_remote_code=True
)
base_model.resize_token_embeddings(len(tokenizer))

# Load the saved LoRA weights onto the base model skeleton and smash them together
peft_model = PeftModel.from_pretrained(base_model, lora_output_dir)
merged_model = peft_model.merge_and_unload()

# Export a clean, unified deployment model
merged_model.save_pretrained(merged_output_dir, max_shard_size="5GB")
tokenizer.save_pretrained(merged_output_dir)

print("=" * 60)
print("TRAINING PROCESS SUCCESSFUL")
print(f"LoRA Target:  {lora_output_dir}")
print(f"Merged Model: {merged_output_dir}")
print("=" * 60)