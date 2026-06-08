# DiffuVQA — Pipeline Versiyon Geçmişi

Her versiyon; eğitim pipeline'ını, inference pipeline'ını veya her ikisini etkileyen değişiklikleri kapsar.
Bugfix'ler (crash fix, KeyError vb.) versiyon numarasını artırmaz — sadece DECISIONS.md'ye eklenir.

Metrikler Kvasir-VQA test seti üzerinden hesaplanmıştır (5886 satır, 20 soru şablonu).

---

## v0.1 — Baseline
**Label:** `v0.1-baseline`
**Klasör:** `samples/old/`

### Değişiklikler
- Orijinal DiffuVQA pipeline, masking yok
- Inference: ham logit argmax, vocab kısıtlaması yok

### Metrikler
| Checkpoint | Exact | Y/N | OE | Empty | Avg Len |
|---|---|---|---|---|---|
| 40k | 0.24% | 0.72% | 0.09% | 23.1% | 2.34 |
| 80k | 0.15% | 0.33% | 0.12% | 13.0% | 3.80 |
| 100k | 0.12% | 0.26% | 0.09% | 14.0% | 3.50 |
| 120k | 0.14% | 0.26% | 0.12% | 10.9% | 4.11 |

---

## v0.2 — Question-Type Masking (Fix1/Fix2)
**Label:** `v0.2-yn-mask`
**Klasör:** `samples/old2/`

### Değişiklikler
- `sample_vqa_GPU.py`: CLOSED_STARTERS heuristic ile Y/N soruları tespit edildi
- `sample_vqa_GPU.py`: Y/N sorular için `yn_mask_bool` eklendi (yes/no/not/relevant/0/1 token'ları)
- `sample_vqa_GPU.py`: Empty/punct fallback eklendi
- `basic_utils.py`: WordPiece artifact cleanup regex eklendi
- **Bug:** `special_ids` (SEP/PAD/CLS) yn_mask'a dahil edildi → model position 0'da SEP seçti → empty% arttı

### Metrikler
| Checkpoint | Exact | Y/N | OE | Empty | Avg Len |
|---|---|---|---|---|---|
| 40k | 1.68% | 5.86% | 0.30% | 32.5% | 1.40 |
| 80k | 1.33% | 4.61% | 0.24% | 20.3% | 2.50 |
| 100k | 1.27% | 4.41% | 0.24% | 21.2% | 2.34 |
| 120k | 1.19% | 4.08% | 0.24% | 17.7% | 2.79 |

---

## v0.3 — Fix3 + Training İyileştirmeleri
**Label:** `v0.3-fix3-training`
**Klasör:** `samples/latest/`

### Değişiklikler
**Inference:**
- `sample_vqa_GPU.py` Fix3: İki ayrı YN mask
  - `yn_mask_bool` (SEP yok) → position 0: model gerçek Y/N token seçmek zorunda
  - `yn_mask_with_sep` (SEP dahil) → position 1+: decode_token SEP'de kesebiliyor
- `sample_vqa_GPU.py`: Classifier head (OR mantığı: classifier OR first-word heuristic)

**Training (yeni eğitilen checkpoint'lerde aktif):**
- `gaussian_diffusion.py`: Closed-ended örnekler için 2x loss weight
- `gaussian_diffusion.py` + `vqa_model.py`: Classifier head (BCE loss ile eğitim)
- `train_util.py`: Linear decay → Cosine decay + linear warmup (warmup=2000, lr_min=5e-6)
- `vqa_datasets.py`: `is_closed` label eklendi
- Batch size: 4 (henüz değiştirilmedi)

### Metrikler
| Checkpoint | Exact | Y/N | OE | Empty | Avg Len |
|---|---|---|---|---|---|
| 80k | **1.87%** | 5.01% | **1.01%** | **0%** | 3.10 |
| 100k | 1.48% | 4.08% | 0.74% | **0%** | 3.48 |
| 120k | 1.36% | 3.56% | 0.77% | **0%** | 3.71 |

### v0.2 → v0.3 Kazanımları
- Empty: 32.5% → **0%** (Fix3 tamamen çözdü)
- OE accuracy: 0.30% → **1.01%** (3.4x, 2x loss weight + cosine LR etkisi)
- Exact 80k: 1.33% → **1.87%**

---

## v0.4 — Kaynak Optimizasyonu
**Label:** `v0.4-resource-opt`
**Klasör:** `samples/v0.4-resource-opt/`
**Eğitim:** `TRAIN_BATCH_SIZE=96`, 180k adım (toplam ~17.3M sample-exposure)

### Değişiklikler (v0.3 üzerine)
- `notebooks/run_diffuvqa_colab.ipynb`: `TRAIN_BATCH_SIZE` 4 → 96, `LR` otomatik linear scaling (`1e-5 × batch/32` → `3e-5`)
- `notebooks/run_diffuvqa_colab.ipynb`: `WARMUP_STEPS=5000` değişkeni eklendi, `--warmup_steps` CLI'a iletiliyor
- `notebooks/run_diffuvqa_colab.ipynb`: `USE_BF16=True`, `USE_TORCH_COMPILE=True` (Triton kernel fusion, ~%25 hız)
- `diffuvqa/config.json`: `warmup_steps` 2000 → 5000
- `diffuvqa/vqa_datasets.py`: `num_workers` 4 → 8, `pin_memory=True`, `persistent_workers=True`
- `train_util.py`: BF16 AMP desteği (`th.autocast` + `th.amp.GradScaler`), deprecated `th.cuda.amp.GradScaler` düzeltildi
- `train.py`: `torch.compile(mode="reduce-overhead")` desteği eklendi

### Metrikler
| Checkpoint | Exact | Y/N | OE | Empty | Avg Len | BLEU-1 | F1 | METEOR |
|---|---|---|---|---|---|---|---|---|
| 10k | 5.33% | 9.11% | 1.75% | **0%** | 2.94 | 10.33% | 8.64% | 4.59% |
| 20k | **6.05%** | **10.65%** | 1.69% | **0%** | 3.23 | **10.46%** | **9.42%** | **5.27%** |

### v0.3 → v0.4 Kazanımları (80k karşılaştırması)
| Metrik | v0.3 80k | v0.4 20k | Fark |
|---|---|---|---|
| Exact | 1.87% | 6.05% | **+4.18 pp (3.2x)** |
| Y/N | 5.01% | 10.65% | **+5.64 pp (2.1x)** |
| OE | 1.01% | 1.69% | +0.68 pp |
| BLEU-1 | 7.37% | 10.46% | **+3.09 pp** |
| F1 | 10.02% | 9.42% | -0.60 pp |
| METEOR | 8.62% | 5.27% | -3.35 pp |

> Not: v0.4 20k adımda v0.3'ün 80k peak'ini birçok metrikte geçiyor — batch=96 ile her adımda 24x fazla data görülüyor (96/4=24x).

### Bugfix'ler (versiyon artırmadı)
| Tarih | Dosya | Açıklama |
|---|---|---|
| 2026-06-07 | `train.py` | `import torch._dynamo` → `import torch._dynamo as _dynamo` — `use_torch_compile=False` olduğunda `UnboundLocalError` çöktürüyordu |

---

## Versiyon Kuralları

- **Yeni versiyon:** Eğitim pipeline'ı veya inference logic'ini etkileyen her değişiklik
- **Bugfix (versiyon artmaz):** Crash fix, KeyError, shape hatası vb. düzeltmeler
- **Sample klasörü:** Yeni versiyon için `samples/vX.Y/` formatında klasör açılacak
- **Metrik güncelleme:** Sample alındıktan sonra bu dosyaya eklenir
