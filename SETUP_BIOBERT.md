# SETUP_BIOBERT.md — BioBERT Branch Kurulum Kılavuzu

Önce `SETUP_COMMON.md`'deki tüm adımları uygula, ardından bu dosyadaki encoder-specific değişiklikleri yap.

**Kullanılacak model:** `dmis-lab/biobert-base-cased-v1.2`
HuggingFace: https://huggingface.co/dmis-lab/biobert-base-cased-v1.2

**Neden BioBERT?**
PubMed makalelerinden pre-train edilmiş BERT variant. Biyomedikal terminoloji (hastalık adları, anatomik terimler, endoskopik bulgular) için bert-base-uncased'e göre daha güçlü representation üretir.

---

## 1. `diffuvqa/config.json`

```json
"config_name": "dmis-lab/biobert-base-cased-v1.2",
"language_encoder_name": "dmis-lab/biobert-base-cased-v1.2",
"vocab_size": 28996,
"hidden_dim": 768,
"hidden_size": 768
```

> **Not:** BioBERT cased tokenizer kullanır → vocab_size 28996 (bert-base-uncased 30522'dir).
> `hidden_dim=768` aynı kalır, mimari değişmez.

---

## 2. `diffuvqa/config/training_args.json`

```json
"config_name": "dmis-lab/biobert-base-cased-v1.2",
"language_encoder_name": "dmis-lab/biobert-base-cased-v1.2",
"vocab_size": 28996
```

---

## 3. `basic_utils.py`

`myTokenizer.__init__` içinde:
```python
# ÖNCE:
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# SONRA:
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
```

`create_model_and_diffusion` içinde default `config_name`:
```python
# ÖNCE:
config_name="bert-base-uncased",
# SONRA:
config_name="dmis-lab/biobert-base-cased-v1.2",
```

---

## 4. `diffuvqa/vqa_model.py`

```python
# ÖNCE:
config_name='bert-base-uncased',
# SONRA:
config_name='dmis-lab/biobert-base-cased-v1.2',
```

`BertModel.from_pretrained` ve `AutoConfig.from_pretrained` çağrıları `config_name` değişkenini kullandığı için otomatik güncellenir.

---

## 5. `diffuvqa/vqa_datasets.py`

```python
# ÖNCE:
parser.add_argument('--config_name', type=str, default='bert-base-uncased')
# SONRA:
parser.add_argument('--config_name', type=str, default='dmis-lab/biobert-base-cased-v1.2')
```

---

## 6. Notebook (`notebooks/run_diffuvqa_colab.ipynb`)

### Config hücresi
```python
MODEL_NAME  = "biobert"
MODEL_LABEL = "DiffuVQA-BioBERT"
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/DiffuVQA-BioBERT"
```

### BERT cache hücresi
```python
from transformers import AutoModel, AutoTokenizer

_endpoint = os.environ.pop("HF_ENDPOINT", None)
AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
if _endpoint:
    os.environ["HF_ENDPOINT"] = _endpoint
print("BioBERT cache'e alındı.")
```

### Header güncelle
```
## ⚙️ Konfigürasyon — DiffuVQA BioBERT
```

---

## 7. `requirements_colab.txt`

Ek bağımlılık gerekmez — `transformers>=4.36.0` BioBERT'i destekler.

---

## 8. Kontrol Listesi

- [ ] `config.json` → `biobert-base-cased-v1.2`, `vocab_size=28996`
- [ ] `training_args.json` → aynı
- [ ] `basic_utils.py` → tokenizer + config_name güncellendi
- [ ] `vqa_model.py` → config_name güncellendi
- [ ] `vqa_datasets.py` → default arg güncellendi
- [ ] Notebook config → `MODEL_NAME="biobert"`, `DRIVE_PROJECT_PATH=.../DiffuVQA-BioBERT`
- [ ] Notebook BERT cache hücresi → BioBERT model adı
- [ ] SETUP_COMMON.md adımları tamamlandı

---

## 9. Akademik Not

BioBERT karşılaştırmasında şunları aynı tut (kontrollü deney):
- `LEARNING_STEPS`, `LR`, `BATCH_SIZE`, `DIFFUSION_STEPS`, `SEQ_LEN`
- Dataset split (aynı `random.seed(42)` ile üretilmiş train/valid/test)
- `SAMPLE_STEP`, `SAMPLE_SEED2`