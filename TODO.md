# TODO — DiffuVQA Colab/A100 Compatibility

## Completed

- [x] Create `requirements_colab.txt` (torch/torchvision excluded, wandb excluded, timm/open_clip/openpyxl/pycocoevalcap added)
- [x] Remove `import wandb`, `wandb.init()`, `wandb.config.update()`, `WANDB_MODE` from `train.py`
- [x] Remove `import wandb` from `diffuvqa/utils/logger.py`
- [x] Fix `logger.configure()` in `train.py` → `dir=args.checkpoint_path, format_strs=["log", "csv"]`
- [x] Change `parser.parse_args()` → `parser.parse_known_args()[0]` in `train.py`
- [x] Add `dist.is_initialized()` guard before `dist.get_world_size()` in `train_util.py`
- [x] Verify `HF_ENDPOINT` already set in both `train.py` and `sample_vqa_GPU.py`
- [x] Create `notebooks/run_diffuvqa_colab.ipynb` (5 sections: Setup / Train / Sample / Eval / Download)
- [x] Create `CLAUDE.md`
- [x] Create `DECISIONS.md`
- [x] Create `TODO.md`
- [x] Drive path'i `DiffuVQA-Original` olarak düzelt (DiffuVQA ile çakışmasın)
- [x] Notebook yapısını Bert branch formatına uyarla (Config hücresi + Türkçe sub-header'lar + dataset copy + CSV export + görselleştirme)
- [x] Kvasir-VQA dataset indirme hücresi ekle (Drive'da varsa atla, yoksa HuggingFace'den indir)
- [x] Zenodo URL hatası düzelt → HuggingFace `SimulaMet-HOST/Kvasir-VQA` kullan (`raw` split)
- [x] HF parquet cache'ini Drive'a yönlendir (`hf_cache/`) — bir daha indirilmesin
- [x] Dataset split düzelt: `raw` split'ten 80/10/10 train/valid/test JSONL oluştur, görüntüleri `img_id.jpg` ile kaydet
- [x] `IMAGEFOLDER_NAME` düzelt: `images` → `imgs` (config.json ile eşitle)
- [x] `dataset-copy` hücresine boş klasör kontrolü + Drive fallback ekle
- [x] Dataset doğrulama hücresini düzelt: JSONL yolu `datasets/Kvasir_VQA/` olarak güncelle
- [x] Eğitim hücresine `--data_dir` ve `--image_dir` argümanları ekle (Drive mutlak yolları)
- [x] Clone hücresine `git log -3 --oneline` ekle (son 3 commit görünsün)
- [x] `transformers==4.22.2` → `>=4.36.0` yap (Python 3.12'de tokenizers build hatası)
- [x] `bert_model.py` import uyumluluğu: `file_utils` → `utils`, `apply_chunking_to_forward` + `find_pruneable_heads_and_indices` + `prune_linear_layer` → `pytorch_utils` (try/except ile geriye dönük uyumlu)

## Open (nice-to-have, not blocking)

- [ ] Eğitimi A100 Colab'da uçtan uca test et, `progress.csv` yazıldığını doğrula
- [ ] `train_util.py` `run_loop`'una tqdm ekle (epoch/step progress bar)
- [ ] `pycocoevalcap` CIDEr metriğinin güncel NLTK versiyonuyla çalıştığını doğrula
- [ ] SLAKE dataset için de indirme hücresi ekle
