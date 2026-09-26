"""
ORPO preference alignment training script for Ministral-3-3B-Reasoning.
Fully stabilized for long <|thought|> tokens and multi-pair preference files.
"""

import os
import gc
import torch
import warnings
from pathlib import Path

from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM
from trl import ORPOConfig, ORPOTrainer

# Silence warnings and clean terminal tracking
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ============================================================
# STABILITY / ENV
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

# ============================================================
# CONFIG
# ============================================================
BASE_SFT_MODEL_PATH = os.path.abspath("ministral_3_3B_reasoning_2512_merged")  
DATASET_PATH = "ministral_3_3B_reasoning_2512_orpo_perfect.jsonl"  
OUTPUT_DIR = "outputs_orpo"
LORA_OUTPUT_DIR = "ministral_3_3B_reasoning_2512_orpo_lora"
MERGED_OUTPUT_DIR = "ministral_3_3B_reasoning_2512_orpo_merged"

MAX_SEQ_LENGTH = 3200
MAX_PROMPT_LENGTH = 64
SEED = 3407

USE_EVAL_SPLIT = True
EVAL_SPLIT_RATIO = 0.05

# ============================================================
# DATA PRE-FILTERING UTILITIES
# ============================================================
def extract_text_from_messages(messages):
    """
    Extract concatenated text from conversational message format.
    Handles Mistral-3's content-as-list structure.
    """
    texts = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        texts.append(item)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts)


def extract_assistant_text_only(messages):
    """
    Extract ONLY the assistant's response text from a conversation array.
    """
    texts = []
    if messages and isinstance(messages, list):
        assistant_msg = messages[-1]  # Last message is the assistant response
        if isinstance(assistant_msg, dict) and assistant_msg.get("role") == "assistant":
            content = assistant_msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        texts.append(item)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts)


def filter_and_format_dataset(example, tokenizer, max_length=3200, max_prompt_length=64):
    """
    Checks token boundaries and reformats data to pure text blocks.
    """
    prompt_text = extract_text_from_messages(example.get("prompt", []))
    chosen_text = extract_assistant_text_only(example.get("chosen", []))
    rejected_text = extract_assistant_text_only(example.get("rejected", []))

    # Strip manual <|thought|> tokens from dataset strings if your template handles it
    if chosen_text.startswith("<|thought|>\n"):
        chosen_text = chosen_text[len("<|thought|>\n"):]
    if rejected_text.startswith("<|thought|>\n"):
        rejected_text = rejected_text[len("<|thought|>\n"):]

    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    chosen_tokens = tokenizer.encode(chosen_text, add_special_tokens=False)
    rejected_tokens = tokenizer.encode(rejected_text, add_special_tokens=False)

    prompt_len = len(prompt_tokens)
    max_total = max(prompt_len + len(chosen_tokens), prompt_len + len(rejected_tokens))

    if (max_total <= max_length) and (prompt_len <= max_prompt_length):
        return {
            "keep_sample": True,
            "prompt_clean": prompt_text,
            "chosen_clean": chosen_text,
            "rejected_clean": rejected_text
        }
    return {"keep_sample": False, "prompt_clean": "", "chosen_clean": "", "rejected_clean": ""}


# ============================================================
# LOAD MODEL & TEXT-ONLY EXTRACTED TOKENIZER
# ============================================================
model, processor = FastLanguageModel.from_pretrained(
    model_name=BASE_SFT_MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    trust_remote_code=True
)

text_tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

if "<|thought|>" not in text_tokenizer.get_vocab():
    text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|thought|>"]})
    model.resize_token_embeddings(len(text_tokenizer))

# STABILIZED JINJA TEMPLATE: Places the <|thought|> block under the assistant scope
# so that its generation is penalized and updated via the ORPO odds ratio.
official_mistral3_template = (
    "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
            "{{ '<s>[INST] ' + message['content'][0]['text'] + ' [/INST]' }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '<|thought|>\n' + message['content'][0]['text'] + '</s>' }}"
        "{% endif %}"
    "{% endfor %}"
)
text_tokenizer.chat_template = official_mistral3_template

if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token
text_tokenizer.padding_side = "right" 

model = FastLanguageModel.get_peft_model(
    model,
    r=64,                
    lora_alpha=128,      
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = FastLanguageModel.for_training(model)

# ============================================================
# PIPELINED DATA RE-MAPPING
# ============================================================
raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

print("=" * 60)
print("PROCESSING & PRE-FILTERING ORPO DATASET")
print("=" * 60)

# Step 1: Extract text and stage clean variations
mapped_dataset = raw_dataset.map(
    lambda x: filter_and_format_dataset(x, text_tokenizer, MAX_SEQ_LENGTH, MAX_PROMPT_LENGTH),
    num_proc=2
)

# Step 2: Filter out elements that don't fit
kept_dataset = mapped_dataset.filter(lambda x: x["keep_sample"])

# Step 3: Explicitly drop original column formats to avoid downstream processing exceptions
dataset = kept_dataset.map(
    lambda x: {
        "prompt": x["prompt_clean"],
        "chosen": x["chosen_clean"],
        "rejected": x["rejected_clean"]
    },
    remove_columns=kept_dataset.column_names,
    num_proc=2
)

dropped_count = len(raw_dataset) - len(dataset)
print(f"Raw samples loaded:    {len(raw_dataset)}")
print(f"Samples kept:          {len(dataset)}")
print(f"Samples dropped:       {dropped_count} ({dropped_count / len(raw_dataset) * 100:.1f}%)")
print("=" * 60)

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
    learning_rate=1e-5,          
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1, 
    gradient_accumulation_steps=8,  
    num_train_epochs=3,
    bf16=True,
    fp16=False,
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,                 
    eval_strategy="steps" if eval_dataset is not None else "no",
    eval_steps=100 if eval_dataset is not None else None,
    report_to="none",

    beta=0.05,                   
    max_length=MAX_SEQ_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
    disable_dropout=True,
    remove_unused_columns=True,  
    dataset_num_proc=2,

    max_grad_norm=1.0,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_8bit",          
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
print("Starting Ministral-3 ORPO preference alignment run...")
trainer.train()

# ============================================================
# SAVE LORA ADAPTER
# ============================================================
print("Saving ORPO LoRA adapters...")
model.save_pretrained(LORA_OUTPUT_DIR)
text_tokenizer.save_pretrained(LORA_OUTPUT_DIR)
print("ORPO LoRA saved successfully!")

del trainer, model
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# EXPLICIT MANUAL PEFT LAYER CONSOLIDATION
# ============================================================
print("Consolidating layers via manual PEFT merge into 16-bit precision SafeTensors...")
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_SFT_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",  
    trust_remote_code=True
)
base_model.resize_token_embeddings(len(text_tokenizer))

peft_model = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
merged_model = peft_model.merge_and_unload()

merged_model.save_pretrained(MERGED_OUTPUT_DIR, max_shard_size="5GB")
text_tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

print("=" * 60)
print("ALL PIPELINES SUCCESSFUL")
print(f"ORPO LoRA Target:  {LORA_OUTPUT_DIR}")
print(f"ORPO Merged Model: {MERGED_OUTPUT_DIR}")
print("=" * 60)