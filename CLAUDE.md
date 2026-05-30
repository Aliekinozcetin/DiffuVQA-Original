# DiffuVQA — Codebase Guide

## Project Summary

DiffuVQA is a **diffusion-based generative Medical VQA** system. Instead of classifying answers, it generates free-form answers via a conditional diffusion process over a learned embedding space. Published in *Biomedical Signal Processing and Control* (Jan 2026).

Supported datasets: **SLAKE**, **Kvasir-VQA**, **Med-VQA-2019**

---

## Directory Structure

```
DiffuVQA-Original/
├── train.py                     # Training entry point
├── train_util.py                # TrainLoop class (forward/backward/save/EMA)
├── sample_vqa_GPU.py            # Inference: load checkpoint → generate answers
├── eval_DiffuVQA.py             # Evaluation: BLEU, ROUGE, METEOR, CIDEr, BERTScore, F1
├── basic_utils.py               # Config loading, tokenizer, model factory
├── requirements.txt             # Full deps (local/server)
├── requirements_colab.txt       # Colab deps (torch/torchvision excluded)
├── notebooks/
│   └── run_diffuvqa_colab.ipynb # End-to-end Colab runner (A100-ready)
├── diffuvqa/
│   ├── config.json              # Default hyperparameters
│   ├── gaussian_diffusion.py    # Diffusion process (SpacedDiffusion)
│   ├── rounding.py              # Denoised-fn rounding (embedding → token)
│   ├── step_sample.py           # Timestep samplers (Uniform, LossAware)
│   ├── vqa_datasets.py          # Data loading (SLAKE, Kvasir, Med-VQA)
│   ├── vqa_model.py             # TransformerNetModel (main architecture)
│   ├── attention/
│   │   └── attention_model.py   # Cross-attention (vision × language)
│   ├── config/                  # BERT tokenizer files + training_args.json
│   ├── language_encoders/
│   │   └── bert_model.py        # BERT language encoder
│   ├── utils/
│   │   ├── dist_util.py         # Device/distributed helpers
│   │   ├── fp16_util.py         # Mixed-precision helpers
│   │   ├── logger.py            # CSV/log file logger (OpenAI Baselines style)
│   │   ├── losses.py            # Loss functions
│   │   └── nn.py                # EMA update, weight init helpers
│   └── vision_encoders/
│       ├── clip_model.py        # CLIP vision encoder (ViT-B/32 default)
│       ├── swin_transformer.py  # Swin Transformer backbone
│       └── swin_helpers.py
├── scripts/
│   ├── train.sh                 # Shell training launcher
│   └── run_decode.sh            # Shell decoding launcher
└── config/                      # Pre-trained EMA checkpoint samples
```

---

## Common Commands

### Training
```bash
python train.py \
  --checkpoint_path ./checkpoints/run1 \
  --dataset Kvasir_VQA \
  --batch_size 4 \
  --lr 1e-5 \
  --learning_steps 500000 \
  --save_interval 25000
```

### Sampling (inference)
```bash
python sample_vqa_GPU.py \
  --model_path ./checkpoints/run1/ema_0.9999_025000.pt \
  --out_dir ./outputs \
  --split test \
  --batch_size 8 \
  --step 2500
```

### Evaluation
```bash
python eval_DiffuVQA.py \
  --gen_path ./outputs/run1.ema_0.9999_025000.pt.samples/seed105_step0.jsonl
```

### Colab (A100)
Open `notebooks/run_diffuvqa_colab.ipynb` in Google Colab. Edit the **Config** cell in each section and run cells top to bottom.

---

## Architecture Summary

```
Image  ──► CLIP encoder (ViT-B/32) ──┐
                                      ├─► Cross-Attention (attention_model.py)
Question ──► BERT encoder ───────────┘
                    │
                    ▼
         Fused embedding (hidden_dim=768)
                    │
                    ▼
         Transformer UNet (vqa_model.py)
          [TransformerNetModel]
                    │
          Diffusion process (gaussian_diffusion.py)
          [SpacedDiffusion, noise_schedule=sqrt]
                    │
          Denoising + rounding (rounding.py)
                    │
                    ▼
         Generated answer tokens (seq_len=32)
```

**Loss:** Training uses the diffusion ELBO (MSE on predicted noise) weighted by the schedule sampler.

---

## Vocabulary / Tokenizer Options

Set via `args.vocab` (in config.json or CLI):

| Value | Tokenizer |
|-------|-----------|
| `bert` | `bert-base-uncased` (HuggingFace, 30522 tokens) |
| custom path | Load vocab from file (`diffuvqa/config/vocab.txt`) |

`args.vocab_size` is derived automatically from the loaded tokenizer.

---

## Key Hyperparameters (diffuvqa/config.json)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `hidden_dim` | 768 | Must match BERT hidden size |
| `seq_len` | 32 | Max answer token length |
| `diffusion_steps` | 2500 | Full DDPM steps |
| `noise_schedule` | sqrt | Sqrt schedule for text diffusion |
| `lr` | 1e-5 | AdamW learning rate |
| `batch_size` | 20 | Reduce to 4–8 on A100 for safety |
| `ema_rate` | 0.9999 | EMA decay for checkpoint |
| `gradient_clipping` | 0.5 | Gradient norm clip |
| `use_fp16` | false | Mixed precision (experimental) |

---

## Architecture files

Tüm dosyalar gerektiğinde değiştirilebilir. Mimari değişiklikler DECISIONS.md'ye kaydedilmeli.
