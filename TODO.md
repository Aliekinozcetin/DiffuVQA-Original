# TODO — DiffuVQA Colab/A100 Compatibility

## Completed

- [x] `is_resume` "none" string bug düzeltildi — `"none"` string'i artık resume sayılmıyor, fresh start'ta progress.csv doğru şekilde `w+t` modunda açılıyor
- [x] `log_interval` 1000 adıma çıkarıldı (config.json: 200→1000, notebook: 100→1000)

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
- [x] `bert_model.py` import uyumluluğu: `file_utils` → `utils`, `apply_chunking_to_forward` + `prune_linear_layer` → `pytorch_utils` (try/except), `find_pruneable_heads_and_indices` ayrı try/except + inline fallback (transformers>=4.40 uyumlu)

- [x] `diffuvqa/utils/answer_pre.py` eksikti → DiffuVQA reposundan alınarak eklendi (`find_most_similar_answers`, difflib tabanlı)
- [x] `basic_utils.py` HF_ENDPOINT bypass: `myTokenizer.__init__` ve `create_model_and_diffusion`'da BERT yüklemesi sırasında `HF_ENDPOINT` geçici kaldırılıyor (hf-mirror BERT servis etmiyor, `OSError` veriyordu)
- [x] `train.py` + `sample_vqa_GPU.py` başındaki `HF_ENDPOINT=hf-mirror` set'i kaldırıldı — kök neden bu satırdı, BERT from_pretrained tüm process boyunca hf-mirror'a gidiyordu
- [x] `vqa_model.py` `get_extended_attention_mask` `device=` kwarg kaldırıldı (transformers>=4.36 imza değişikliği, `TypeError` veriyordu)
- [x] `train_util.py` `_SingleGPUDDP` wrapper eklendi: tek GPU'da `model.model.module.*` erişimi (`gaussian_diffusion.py`) `AttributeError` veriyordu, `.module` attribute'unu expose eden minimal wrapper ile düzeltildi
- [x] `vqa_datasets.py` `load_image_path` çift `imgs/` prefix hatası düzeltildi: `f'{image_root}/{image_name}'` → `os.path.normpath(os.path.join(...))` (JSONL'deki `imgs/` prefix ile `image_root` çakışıyordu → `FileNotFoundError`)
- [x] `gaussian_diffusion.py` `p_sample_loop` + `ddim_sample_loop` içindeki `print(i)` kaldırıldı (her timestep basılıyordu, tqdm varken çıktıyı dolduruyordu)
- [x] `sample_vqa_GPU.py` çıktı yolu sadeleştirildi: `out_dir/lr1e-05/ema_0.9999_200000.jsonl` (eski: `*.pt.samples/seed105_step0.jsonl`)
- [x] `sample_vqa_GPU.py` her JSONL satırına `confidence` ve `avg_nn_l2` eklendi
- [x] `logger.py` `CSVOutputFormat` + `configure()`: sıfırdan eğitimde `progress.csv` temizlenir, resume'da append edilir (`append_csv` parametresi)
- [x] Notebook `evaluate_and_export_csv` dataset yolu düzeltildi: `"datasets/test.jsonl"` hardcoded → `DRIVE_PROJECT_PATH/datasets/DATASET/test.jsonl`
- [x] Notebook `BATCH_SIZE` → `TRAIN_BATCH_SIZE=4` + `SAMPLE_BATCH_SIZE=64` olarak ayrıldı (batch_size=64 eğitimi ~7x yavaşlatıyordu)
- [x] `logger.py` `dumpkvs()` writekvs döngüsü uncomment edildi — wandb kaldırılırken yanlışlıkla comment'e alınmıştı, progress.csv hiç yazılmıyordu

## PubMedBERT Branch — Tamamlanan

- [x] `diffuvqa/config.json` → `config_name` + `language_encoder_name` PubMedBERT'e güncellendi (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`)
- [x] `diffuvqa/config/training_args.json` → `config_name` + `language_encoder_name` PubMedBERT'e güncellendi
- [x] `basic_utils.py` → `myTokenizer.__init__` `AutoTokenizer` + `create_model_and_diffusion` `TransformerNetModel` çağrısı PubMedBERT'e güncellendi
- [x] `diffuvqa/vqa_model.py` → `TransformerNetModel` default `config_name` PubMedBERT'e güncellendi
- [x] `diffuvqa/vqa_datasets.py` → `__main__` argparse `--config_name` default PubMedBERT'e güncellendi
- [x] Notebook config hücresi: `BRANCH=pubmedbert`, `MODEL_NAME=pubmedbert`, `MODEL_LABEL=DiffuVQA-PubMedBERT`, `DRIVE_PROJECT_PATH=DiffuVQA-PubMedBERT`, `RESUME_CHECKPOINT=""` (sıfırdan eğitim)
- [x] Notebook BERT cache hücresi: `BertTokenizer`/`BertModel` → `AutoTokenizer`/`AutoModel` + PubMedBERT model adı, mesaj güncellendi
- [x] Notebook config header: `DiffuVQA PubMedBERT` olarak güncellendi

## Open (nice-to-have, not blocking)

- [ ] Eğitimi A100 Colab'da uçtan uca test et, `progress.csv` yazıldığını doğrula
- [x] `train_util.py` `run_loop`'una tqdm progress bar eklendi — `loss=X.XXXX` postfix, satır satır print kaldırıldı
- [ ] `pycocoevalcap` CIDEr metriğinin güncel NLTK versiyonuyla çalıştığını doğrula
- [ ] SLAKE dataset için de indirme hücresi ekle
