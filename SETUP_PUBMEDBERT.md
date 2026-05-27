# SETUP_PUBMEDBERT.md — PubMedBERT Branch Kurulum Kılavuzu

Önce `SETUP_COMMON.md`'deki tüm adımları uygula, ardından bu dosyadaki encoder-specific değişiklikleri yap.

**Kullanılacak model:** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
HuggingFace: https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext

**Neden PubMedBERT?**
Sıfırdan yalnızca PubMed full-text üzerinde pre-train edilmiş (domain-specific pretraining). Genel corpus'tan fine-tune edilen modellere (BioBERT dahil) göre biyomedikal NLP benchmark'larında daha yüksek performans gösterir. Uncased olduğu için bert-base-uncased ile vocab boyutu aynı (30522) — mimari değişiklik minimumdur.

---

## 1. `diffuvqa/config.json`

```json
"config_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
"language_encoder_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
"vocab_size": 30522,
"hidden_dim": 768,
"hidden_size": 768
```

> **Not:** PubMedBERT uncased, vocab_size=30522 — bert-base-uncased ile aynı.
> `hidden_dim=768` aynı kalır, mimari değişmez.

---

## 2. `diffuvqa/config/training_args.json`

```json
"config_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
"language_encoder_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
"vocab_size": 30522
```

---

## 3. `basic_utils.py`

`myTokenizer.__init__` içinde:
```python
# ÖNCE:
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# SONRA:
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
```

`create_model_and_diffusion` içinde default `config_name`:
```python
# ÖNCE:
config_name="bert-base-uncased",
# SONRA:
config_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
```

---

## 4. `diffuvqa/vqa_model.py`

```python
# ÖNCE:
config_name='bert-base-uncased',
# SONRA:
config_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
```

`BertModel.from_pretrained` ve `AutoConfig.from_pretrained` çağrıları `config_name` değişkenini kullandığı için otomatik güncellenir.

---

## 5. `diffuvqa/vqa_datasets.py`

```python
# ÖNCE:
parser.add_argument('--config_name', type=str, default='bert-base-uncased')
# SONRA:
parser.add_argument('--config_name', type=str, default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
```

---

## 6. Notebook (`notebooks/run_diffuvqa_colab.ipynb`)

### Config hücresi
```python
MODEL_NAME  = "pubmedbert"
MODEL_LABEL = "DiffuVQA-PubMedBERT"
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/DiffuVQA-PubMedBERT"
```

### BERT cache hücresi
```python
from transformers import AutoModel, AutoTokenizer

_endpoint = os.environ.pop("HF_ENDPOINT", None)
AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
if _endpoint:
    os.environ["HF_ENDPOINT"] = _endpoint
print("PubMedBERT cache'e alındı.")
```

### Header güncelle
```
## ⚙️ Konfigürasyon — DiffuVQA PubMedBERT
```

---

## 7. `requirements_colab.txt`

Ek bağımlılık gerekmez — `transformers>=4.36.0` PubMedBERT'i destekler.

---

## 8. Kontrol Listesi

- [ ] `config.json` → `BiomedBERT-base-uncased-abstract-fulltext`, `vocab_size=30522`
- [ ] `training_args.json` → aynı
- [ ] `basic_utils.py` → tokenizer + config_name güncellendi
- [ ] `vqa_model.py` → config_name güncellendi
- [ ] `vqa_datasets.py` → default arg güncellendi
- [ ] Notebook config → `MODEL_NAME="pubmedbert"`, `DRIVE_PROJECT_PATH=.../DiffuVQA-PubMedBERT`
- [ ] Notebook BERT cache hücresi → PubMedBERT model adı
- [ ] SETUP_COMMON.md adımları tamamlandı

---

## 9. Akademik Not

PubMedBERT karşılaştırmasında şunları aynı tut (kontrollü deney):
- `LEARNING_STEPS`, `LR`, `BATCH_SIZE`, `DIFFUSION_STEPS`, `SEQ_LEN`
- Dataset split (aynı `random.seed(42)` ile üretilmiş train/valid/test)
- `SAMPLE_STEP`, `SAMPLE_SEED2`

BioBERT (cased) ile karşılaştırırken tokenizasyon farkına dikkat:
PubMedBERT uncased → girdi lowercase'e normalize edilir.
BioBERT cased → büyük/küçük harf korunur.
Bu fark özellikle anatomik terim ve kısaltmalarda (CT, MRI, IBD) sonuçları etkileyebilir.