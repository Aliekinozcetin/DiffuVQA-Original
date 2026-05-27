# TODO.md — DiffuVQA-Original (Branch: biobert)

Tamamlanan maddeler [x], bekleyenler [ ] olarak işaretlenir. Her değişiklikte bu dosya güncellenir.

---

## SETUP_COMMON.md — Ortak Fix'ler

- [x] **1. requirements_colab.txt** — `pillow`, `pandas`, `scikit-learn` eklendi
- [x] **2. train.py** — `wandb` kaldırıldı, `HF_ENDPOINT` satırı kaldırıldı, `parse_known_args` yapıldı, `logger.configure` + `append_csv` eklendi
- [x] **3. logger.py** — `wandb` import silindi, `CSVOutputFormat` `append` parametresi eklendi, `make_output_format` ve `configure` fonksiyonlarına `append_csv` iletildi
- [x] **4a. train_util.py** — `dist.is_initialized()` guard eklendi
- [x] **4b. train_util.py** — `_SingleGPUDDP` wrapper eklendi
- [x] **4c. train_util.py** — `_load_and_sync_parameters` düzeltildi (orijinal `pass` ile boştu)
- [x] **4d. train_util.py** — `_load_ema_parameters` dist guard eklendi
- [x] **4e. train_util.py** — tqdm progress bar eklendi
- [x] **5. gaussian_diffusion.py** — `print(i)` satırları silindi
- [x] **6. vqa_model.py** — `get_extended_attention_mask` `device=` kwarg kaldırıldı
- [x] **7. bert_model.py** — transformers>=4.36 import try/except blokları eklendi
- [x] **8. diffuvqa/utils/answer_pre.py** — oluşturuldu (`find_most_similar_answers`)
- [x] **9. basic_utils.py** — `HF_ENDPOINT` bypass eklendi (`os.environ.pop`)
- [x] **10. vqa_datasets.py** — `load_image_path` normpath fix uygulandı
- [x] **11. sample_vqa_GPU.py** — `HF_ENDPOINT` satırı kaldırıldı, `use_noising_f` getattr fix, CLI arg koruma `_keep`, çıktı yolu sadeleştirildi, `confidence` + `avg_nn_l2` eklendi, tqdm eklendi
- [x] **12. Notebook** — `TRAIN_BATCH_SIZE=4`, `SAMPLE_BATCH_SIZE=64`, `RESUME_CHECKPOINT` desteği, `evaluate_and_export_csv` dataset_file fix

---

## SETUP_BIOBERT.md — BioBERT Encoder Fix'leri

- [x] **1. diffuvqa/config.json** — `config_name`, `language_encoder_name` → `dmis-lab/biobert-base-cased-v1.2`, `vocab_size` → 28996
- [x] **2. diffuvqa/config/training_args.json** — aynı değişiklikler
- [x] **3. basic_utils.py** — `myTokenizer` → `AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")`, `create_model_and_diffusion` default `config_name` + `vocab_size`
- [x] **4. diffuvqa/vqa_model.py** — `TransformerNetModel` default `config_name` → BioBERT
- [x] **5. diffuvqa/vqa_datasets.py** — `--config_name` default arg → BioBERT
- [x] **6. Notebook config** — `MODEL_NAME="biobert"`, `MODEL_LABEL="DiffuVQA-BioBERT"`, `BRANCH="biobert"`, `DRIVE_PROJECT_PATH=".../DiffuVQA-BioBERT"`
- [x] **6. Notebook BERT cache hücresi** — `AutoTokenizer` + `AutoModel` ile `dmis-lab/biobert-base-cased-v1.2` yükleniyor
- [x] **8. requirements_colab.txt** — ek bağımlılık gerekmedi (`transformers>=4.36.0` BioBERT destekliyor)

---

## Akademik Kontrol Listesi (Kontrollü Deney)

- [ ] BioBERT eğitimi aynı `LEARNING_STEPS=500000` ile başlatıldı
- [ ] Aynı `LR=1e-5`, `BATCH_SIZE=4`, `DIFFUSION_STEPS=2500`, `SEQ_LEN=32` kullanıldı
- [ ] Aynı dataset split (`random.seed(42)`) ile train/valid/test ayrıldı
- [ ] Aynı `SAMPLE_STEP=200`, `SAMPLE_SEED2=105` ile sampling yapıldı
- [ ] BLEU/ROUGE/METEOR/CIDEr/BERTScore/F1 metrikleri main branch sonuçlarıyla karşılaştırıldı

---

## Açık Görevler

- [ ] BioBERT checkpoint 25k adımda ilk kontrol
- [ ] 200k adım sonunda eval ve main branch ile karşılaştırma
- [ ] SLAKE ve Med-VQA-2019 dataset'lerinde de BioBERT deneyi
