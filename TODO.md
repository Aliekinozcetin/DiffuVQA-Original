# TODO — DiffuVQA Colab/A100 Compatibility

## Completed

- [x] sample_shape x_start.shape[1] kullanacak şekilde düzeltildi (seq_len yerine)
- [x] vqa_model.py bert_config dead code silindi
- [x] NLL loss padding token'ları hariç tutacak şekilde düzeltildi (answer_mask eklendi)
- [x] Training loop learning_steps'te durmuyor bugı düzeltildi — `for epoch in range(learning_steps)` → `while global_step < learning_steps`; log/eval global_step bazlı yapıldı
- [x] Resume'da progress.csv kırpma eklendi — resume_step üstündeki stale log satırları silinerek duplicate önlendi
- [x] `decode_token` [SEP]'de truncate eklendi — model artık 32 token yerine [SEP]'e kadar olan kısımı döndürüyor; `skip_special_tokens=True` ile [CLS]/[PAD] temizlendi
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
- [x] `train_util.py` optimizer state kaydetme/yükleme eklendi — `opt_{step:06d}.pt` her checkpoint'te yazılıyor; resume'da yükleniyor (yoksa fresh start)
- [x] `eval_DiffuVQA.py` indentation düzeltildi — `with open(path)` bloğu `for path in files:` döngüsünün dışına kaçmıştı, tüm dosyalar yerine sadece son dosya değerlendiriliyordu
- [x] `eval_DiffuVQA.py` `create_argparser().parse_args()` → `parse_known_args()[0]` — `--folder` argümanı bilinmeyen arg hatası veriyordu
- [x] `train.py` progress.csv corruption düzeltildi — `logger.configure()` trim'den önce çağrılıyordu; CSVOutputFormat eski header'ı cache'liyordu, trim sonrası state uyumsuzluğu satırları bozuyordu. Düzeltme: trim → sonra `logger.configure()`
- [x] Notebook `RESUME_CHECKPOINT = ""` yapıldı — eski değer `ema_0.9999_375000.pt` hardcoded kalmıştı
- [x] Notebook train hücresine `--microbatch {MICROBATCH}` eklendi
- [x] Loss ağırlıkları yeniden ayarlandı — reg `time_weight` sabit 1.0, `lambda_reg` 0.1→0.5, NLL 2x ağırlık
- [x] `train_util.py` `forward_backward` microbatch bug düzeltildi — `del cond['image_name']` loop içindeydi (KeyError), `loss.backward()` loop dışındaydı (sadece son microbatch gradyanı geri yayılıyordu)
- [x] `train_util.py` `forward_only` aynı `del` bug'ı düzeltildi
- [x] `gaussian_diffusion.py` `_token_discrete_loss` NaN guard — `mask.sum().clamp(min=1)` eklendi
- [x] `eval_DiffuVQA.py` hardcoded çıktı yolu düzeltildi — her dosya için ayrı `{basename}_eval.json`; boş JSONL için ZeroDivisionError guard eklendi
- [x] `basic_utils.py` `decode_token` `##` wordpiece fix — `convert_ids_to_tokens` + `convert_tokens_to_string` kullanıldı
- [x] `sample_vqa_GPU.py` `clamp_step` default 0→200 — rounding sadece son 200 adımda aktif
- [x] `rounding.py` + `sample_vqa_GPU.py` answer vocabulary kısıtlama — KNN tüm 30522 token yerine dataset'teki answer token subspace'i üzerinde yapılıyor
- [x] `sample_vqa_GPU.py` confidence L2 tabanlı hesaplamaya geçildi — lm_head softmax yerine nearest answer vocab token'a L2 mesafesi

## Open (nice-to-have, not blocking)

- [ ] Eğitimi A100 Colab'da uçtan uca test et, `progress.csv` yazıldığını doğrula
- [x] `train_util.py` `run_loop`'una tqdm progress bar eklendi — `loss=X.XXXX` postfix, satır satır print kaldırıldı
- [ ] `pycocoevalcap` CIDEr metriğinin güncel NLTK versiyonuyla çalıştığını doğrula
- [ ] SLAKE dataset için de indirme hücresi ekle
