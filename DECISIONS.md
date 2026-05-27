# DECISIONS.md — DiffuVQA-Original (Branch: biobert)

En son karar en üste eklenir. Her değişiklik `TODO.md`'de [x] olarak işaretlenir.

---

## 2026-05-27

### BioBERT Encoder Seçimi
**Karar:** Dil encoderı `bert-base-uncased` → `dmis-lab/biobert-base-cased-v1.2` olarak değiştirildi.
**Gerekçe:** BioBERT, PubMed makalelerinden pre-train edilmiş bir BERT variant'ı. Biyomedikal terminoloji (hastalık adları, anatomik terimler, endoskopik bulgular) için bert-base-uncased'e göre daha güçlü representation üretir. Kontrollü ablation çalışması için aynı `LEARNING_STEPS`, `LR`, `BATCH_SIZE`, `DIFFUSION_STEPS`, `SEQ_LEN` ve dataset split (aynı `random.seed(42)`) kullanılıyor.
**Etkilenen dosyalar:**
- `diffuvqa/config.json` → `config_name`, `language_encoder_name`, `vocab_size: 28996`
- `diffuvqa/config/training_args.json` → aynı
- `basic_utils.py` → `AutoTokenizer.from_pretrained`, `create_model_and_diffusion` default config_name + vocab_size
- `diffuvqa/vqa_model.py` → `TransformerNetModel` default `config_name`
- `diffuvqa/vqa_datasets.py` → `--config_name` argümanı default
- `notebooks/run_diffuvqa_colab.ipynb` → `MODEL_NAME`, `MODEL_LABEL`, `DRIVE_PROJECT_PATH`, `BRANCH`, BERT cache hücresi

### vocab_size Değişikliği
**Karar:** `vocab_size: 30522` → `28996`
**Gerekçe:** BioBERT cased tokenizer kullanır; vocabulary boyutu bert-base-uncased'den farklı (28996). Kelime gömme boyutunun eşleşmesi için güncellendi. `hidden_dim=768` değişmedi.

### requirements_colab.txt Eksik Paketler
**Karar:** `pillow`, `pandas`, `scikit-learn` eklendi.
**Gerekçe:** Bu paketler SETUP_COMMON.md'de listelenmişti ancak mevcut `requirements_colab.txt`'te eksikti. `pillow` görüntü yükleme, `pandas` CSV export, `scikit-learn` CIDEr hesaplama (TF-IDF + cosine similarity) için gerekli.

### Notebook Branch ve Drive Path Güncelleme
**Karar:** Notebook config hücresinde `BRANCH="biobert"`, `DRIVE_PROJECT_PATH="/content/drive/MyDrive/DiffuVQA-BioBERT"`, `MODEL_NAME="biobert"`, `MODEL_LABEL="DiffuVQA-BioBERT"`.
**Gerekçe:** Her branch için ayrı Drive klasörü kullanarak checkpoint ve sample çıktılarını karıştırmamak. main branch verileri DiffuVQA-Original, biobert branch verileri DiffuVQA-BioBERT altında saklanır.

---

## Önceki Kararlar (main branch'ten taşındı)

### SETUP_COMMON Fix'leri (uygulanan, zaten mevcut)
- **wandb kaldırıldı:** `train.py` ve `logger.py`'den wandb bağımlılığı silindi — Colab'da gereksiz ve bağlantı hatası veriyordu.
- **HF_ENDPOINT kaldırıldı:** `train.py`, `basic_utils.py`, `sample_vqa_GPU.py`'den `HF_ENDPOINT=hf-mirror.com` satırı silindi. hf-mirror BERT ağırlıklarını servis etmediği için BERT yüklemelerini kırıyordu; bunun yerine `os.environ.pop` ile geçici bypass uygulandı.
- **`parse_known_args`:** `train.py`'de `parser.parse_args()` → `parser.parse_known_args()[0]` yapıldı. Colab kernel ekstra `argv` enjekte ettiği için crash önlendi.
- **`logger.configure` + `append_csv`:** Resume eğitiminde `progress.csv` üzerine yazmak yerine append yapılıyor. `CSVOutputFormat`, `make_output_format`, `configure` fonksiyonlarına `append_csv=False` parametresi eklendi.
- **`_SingleGPUDDP` wrapper:** `train_util.py`'ye eklendi. `gaussian_diffusion.py` DDP convention varsayıyor (`model.model.module.*` erişimi); tek GPU'da process group olmadan crash önlendi.
- **`_load_and_sync_parameters` düzeltildi:** Orijinal `pass` ile boştu; checkpoint yüklemesi artık çalışıyor.
- **`dist.is_initialized()` guard:** `train_util.py`'de tüm `dist.*` çağrılarına guard eklendi; tek GPU'da crash önlendi.
- **tqdm progress bar:** `run_loop` ve `sample_vqa_GPU.py`'de `print` döngüleri kaldırılıp tqdm ile değiştirildi.
- **`gaussian_diffusion.py` print(i) silindi:** Her timestep'te terminale basılan `print(i)` satırları kaldırıldı.
- **`vqa_model.py` device= kwarg kaldırıldı:** `get_extended_attention_mask` çağrısından `device=` kwarg silindi — transformers>=4.36 bu parametreyi kaldırdı.
- **`bert_model.py` import try/except:** `transformers>=4.36` uyumluluğu için üç farklı try/except bloğu eklendi (`file_utils→utils`, `pytorch_utils`, `find_pruneable_heads_and_indices`).
- **`diffuvqa/utils/answer_pre.py` oluşturuldu:** Orijinal repoda eksikti; `find_most_similar_answers` fonksiyonu eklendi.
- **`vqa_datasets.py` image path normpath:** Çift prefix hatasını önlemek için `os.path.normpath(os.path.join(...))` kullanıldı.
- **`sample_vqa_GPU.py` CLI arg koruma:** `training_args.json` update'i CLI argümanlarının üzerine yazmasın diye `_keep` dict koruması eklendi.
- **`sample_vqa_GPU.py` çıktı yolu sadeleştirildi:** `lr_dir/ckpt_name.jsonl` formatına getirildi.
- **`sample_vqa_GPU.py` confidence + avg_nn_l2:** Her JSONL satırına eklendi.
- **Notebook `evaluate_and_export_csv` dataset_file fix:** `os.path.join(DRIVE_PROJECT_PATH, "datasets", DATASET, "test.jsonl")` ile doğru path oluşturuldu.
- **Notebook `TRAIN_BATCH_SIZE=4`, `SAMPLE_BATCH_SIZE=64`:** A100 için güvenli batch boyutları belirlendi.
