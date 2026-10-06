# SpaceRaven-Finetune 🐦‍⬛🧑‍🚀

**Production-grade SFT + ORPO fine-tuning scripts** for small reasoning & instruction models  
Built with **Unsloth + TRL**, fully stabilized for single-GPU (RTX 4090 class) training with long `<|thought|>` / thinking tokens.

> **This repository is actively maintained and will be updated over time.**  
> More models and training scripts will be added continuously.

---

## Overview

This repository contains battle-tested training pipelines for:

- **Supervised Fine-Tuning (SFT)**
- **Odds Ratio Preference Optimization (ORPO)**

All scripts are optimized for:
- Long reasoning / thinking sequences
- Memory stability (no OOM on 24 GB)
- Clean LoRA → merged 16-bit export
- Models in the 0.6B – 4B range

---

## Repository Structure

```
SpaceRaven-Finetune/
├── sft/                          # Supervised Fine-Tuning scripts
│   ├── train_Llama-3.2-3B-Instruct_sft.py
│   ├── train_Ministral-3-3B-Reasoning-2512_sft.py
│   ├── train_Qwen3-0.6B_sft.py
│   ├── train_Qwen3-4B-Instruct-2507_sft.py
│   ├── train_Qwen3-4B-Thinking-2507_sft.py
│   ├── train_Qwen3.5-4B_sft.py
│   └── train_phi-2_sft.py
│
├── orpo/                         # ORPO Preference Alignment scripts
│   ├── train_Ministral-3-3B-Reasoning-2512_orpo.py
│   ├── train_Qwen3-4B-Thinking-2507_orpo.py
│   └── train_Qwen3.5-4B_orpo.py
│
├── LICENSE
└── README.md
```

---

## Supported Models

| Model | Stage | Script |
|-------|-------|--------|
| Ministral-3-3B-Reasoning-2512 | SFT + ORPO | `train_Ministral-3-3B-Reasoning-2512_*.py` |
| Qwen3-4B-Thinking-2507 | SFT + ORPO | `train_Qwen3-4B-Thinking-2507_*.py` |
| Qwen3.5-4B | SFT + ORPO | `train_Qwen3.5-4B_*.py` |
| Qwen3-4B-Instruct-2507 | SFT | `train_Qwen3-4B-Instruct-2507_sft.py` |
| Qwen3-0.6B | SFT | `train_Qwen3-0.6B_sft.py` |
| Llama-3.2-3B-Instruct | SFT | `train_Llama-3.2-3B-Instruct_sft.py` |
| Phi-2 | SFT | `train_phi-2_sft.py` |

*More models will be added over time.*

---

## Key Features

- **Fully stabilized** for long `<|thought|>` / reasoning tokens
- 4-bit loading + LoRA (rank 64 / alpha 128)
- Gradient checkpointing + 8-bit AdamW
- Automatic dataset length filtering (ORPO)
- Clean chat template handling (including custom Ministral reasoning templates)
- Automatic LoRA save + 16-bit merge
- Eval split + best-model checkpointing on most scripts
- Memory cleanup between stages to prevent OOM

---

## Requirements

```bash
# Recommended environment
Python ≥ 3.10
CUDA 12.x / 13.x
torch ≥ 2.4
unsloth
trl
transformers
datasets
peft
bitsandbytes
```

Install Unsloth following the official instructions:  
https://github.com/unslothai/unsloth

---

## Quick Start

### 1. Prepare your data

**SFT** expects a JSONL with a `"messages"` field (or custom fields depending on the script).  
**ORPO** expects preference pairs:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

or the conversational message format used by the Ministral script.

### 2. Run SFT first

```bash
cd sft
python train_Ministral-3-3B-Reasoning-2512_sft.py
# or any other SFT script
```

This produces:
- `*_lora/` – LoRA adapters
- `*_merged/` – 16-bit merged model

### 3. Run ORPO on the merged SFT model

```bash
cd ../orpo
python train_Ministral-3-3B-Reasoning-2512_orpo.py
```

---

## ORPO Scripts Highlights

### Ministral-3-3B-Reasoning
- Custom Jinja template that places `<|thought|>` under the assistant scope
- Aggressive length filtering for long reasoning chains
- Manual PEFT merge for maximum reliability
- `beta=0.05` (gentler on long thoughts)

### Qwen3-4B-Thinking
- Native Qwen3 chat template via Unsloth
- Higher effective batch size
- `beta=0.1`

### Qwen3.5-4B
- Text-only tokenizer extraction (avoids multimodal processor issues)
- Extensive hyperparameter comments for different model sizes
- Fresh processor reload on save

---

## Typical Hyperparameters

| Setting | SFT | ORPO |
|---------|-----|------|
| Learning Rate | 2e-4 | 8e-6 → 1e-5 |
| LoRA Rank / Alpha | 64 / 128 | 64 / 128 |
| Effective Batch Size | 32 | 8–32 |
| Epochs | 3–4 | 3–4 |
| Max Seq Length | 2048–4048 | 2048–3200 |
| Beta (ORPO) | — | 0.05–0.1 |
| Optimizer | adamw_8bit | adamw_8bit |

---

## Hardware Notes

All scripts are designed and tested for a single **RTX 4090 24 GB**.

Stability environment variables used across the board:

```python
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

---

## License

MIT License – Copyright (c) 2026 RealRaven – see the [LICENSE](LICENSE) file for details.

---

## Author

**RealRaven**  
Full-Stack AI & Robotics Engineer  
[Hugging Face](https://huggingface.co/TrueRealRaven) · [GitHub](https://github.com/RealRaven)

---

> Built for people who actually train models on one GPU and want things to just work.  
> This collection will keep growing — more models coming.
