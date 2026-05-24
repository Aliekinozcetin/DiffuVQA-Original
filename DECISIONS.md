# DECISIONS — DiffuVQA Colab/A100 Compatibility

Decisions are listed newest-first.

---

## 2026-05-25 — Notebook Bert branch yapısına uyarlandı + Drive path düzeltildi

**What:** `notebooks/run_diffuvqa_colab.ipynb` Bert branch'teki (`Aliekinozcetin/DiffuVQA`) notebook yapısına birebir benzetildi.
- Drive path `DiffuVQA` → `DiffuVQA-Original` olarak değiştirildi (iki repo aynı Drive klasörüne yazmasın)
- Repo URL `DiffuVQA-Original` reposuna çevrildi, branch `main`
- Dil Türkçe yapıldı (markdown açıklamalar)
- Büyük tek Config hücresi (tüm ayarlar tek yerden) + bölüm bazlı markdown sub-header yapısı
- `resolve_dataset_image_source` ile çoklu Drive yolu deneme mantığı eklendi
- Dataset görüntü kopyalama + doğrulama + önizleme hücreleri eklendi
- `evaluate_and_export_csv` inline metrik fonksiyonu (BLEU, ROUGE-L, METEOR, CIDEr, BERTScore, F1)
- Bölüm 5: CSV Drive'a kaydet + indir + matplotlib/seaborn görselleştirme
- `HF_ENDPOINT` geçici kaldırma bloğu BERT indirme hücresine taşındı

**Why:** Kullanıcı mevcut DiffuVQA reposuyla (Drive'da `DiffuVQA` klasöründe) çalışıyor; yeni Original repo ayrı `DiffuVQA-Original` klasöründe tutulmalı. Notebook yapısının tutarlı olması farklı repolar arası geçişi kolaylaştırır.

---

## 2026-05-25 — Initial Colab/A100 compatibility pass

### 1. `requirements_colab.txt` created
**What:** New file listing only packages not pre-installed in Colab. Excludes `torch`, `torchvision`, `wandb`. Adds `timm`, `open_clip_torch`, `openpyxl`, `pycocoevalcap` which were missing from `requirements.txt`.
**Why:** `pip install -r requirements.txt` in Colab re-installs torch (slow, can break CUDA version). Colab already has torch; we only need the gap packages.

### 2. `wandb` removed from `train.py`
**What:** Removed `import wandb`, `wandb.init()`, `wandb.config.update()`, and `os.environ["WANDB_MODE"] = "offline"` block entirely.
**Why:** wandb requires authentication and network access; it adds unnecessary friction on Colab and is not needed for research training runs. All run metadata is already saved to `training_args.json`.

### 3. `import wandb` removed from `diffuvqa/utils/logger.py`
**What:** Removed the top-level `import wandb` line. The wandb integration in `dumpkvs()` was already commented out.
**Why:** Importing wandb at module level causes `ModuleNotFoundError` if wandb is not installed (e.g., fresh Colab with only `requirements_colab.txt`). The import served no active purpose.

### 4. `logger.configure()` updated in `train.py`
**What:** Changed `logger.configure()` → `logger.configure(dir=args.checkpoint_path, format_strs=["log", "csv"])`. Removed `"stdout"` from format_strs.
**Why:** `stdout` format prints a full key-value table every `log_interval` steps, flooding the Colab cell output. `tqdm` step print already shows live progress. `log` and `csv` write `log.txt` and `progress.csv` to the checkpoint directory, which persists on Drive across session restarts.

### 5. `parse_known_args` in `train.py`
**What:** `parser.parse_args()` → `parser.parse_known_args()[0]`, unknown args logged as warning.
**Why:** Colab / Jupyter kernels inject extra flags (e.g., `--f=/root/.local/...`) into `sys.argv`. `parse_args()` crashes on unknown flags; `parse_known_args()` silently skips them.

### 6. `dist.is_initialized()` guard in `train_util.py`
**What:** Added `dist.is_initialized()` check before `dist.get_world_size()` in the `TrainLoop.__init__` else-branch.
**Why:** On single-GPU Colab (no `dist.init_process_group` called), `dist.get_world_size()` raises `RuntimeError: Default process group has not been initialized`. The guard makes single-GPU runs safe without requiring `dist.init_process_group`.

### 7. `HF_ENDPOINT` already present in both scripts
**What:** No change needed — `os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'` already exists at the top of both `train.py` and `sample_vqa_GPU.py`.
**Why:** Verified during scan. If hf-mirror fails for BERT weights, users can comment out this line in the Colab notebook config cell.

### 8. `notebooks/run_diffuvqa_colab.ipynb` created
**What:** 4-section Colab notebook: Setup → Training → Sampling → Evaluation. Each section has a Config cell (all settings in one place) + an execution cell. Auto-detects latest checkpoint and output file.
**Why:** Makes the project runnable end-to-end on A100 Colab without any shell access. Crash-resistant: each section independently re-mounts Drive and sets env vars.

### 9. `CLAUDE.md` created
**What:** Codebase guide covering project summary, directory structure, common commands, architecture diagram, hyperparameter table, and protected files list.
**Why:** Required by project rules; also useful for future contributors and AI assistants understanding the repo.
