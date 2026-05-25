# DECISIONS — DiffuVQA Colab/A100 Compatibility

Decisions are listed newest-first.

---

## 2026-05-25 — `bert_model.py` `find_pruneable_heads_and_indices` inline fallback (3. tur)

**What:** `find_pruneable_heads_and_indices` ayrı try/except zincirine alındı: önce `pytorch_utils`, sonra `modeling_utils`, ikisi de başarısız olursa inline implementasyon tanımlanıyor. `apply_chunking_to_forward` ve `prune_linear_layer` ise ayrı bir try/except ile yalnızca `pytorch_utils` → `modeling_utils` zincirini izliyor.
**Why:** `find_pruneable_heads_and_indices` transformers>=4.40'ta `pytorch_utils`'den de kaldırıldı. Önceki tek try/except bloğunda `pytorch_utils` import'u patlayınca (`find_pruneable` yoktu), fallback `modeling_utils` da `apply_chunking_to_forward`'ı oradan kaldırdığı için ikinci kez patlıyordu. Fonksiyon küçük (10 satır) ve stabil, inline tanımlamak güvenli.

---

## 2026-05-25 — `bert_model.py` transformers import uyumluluğu (2. tur)

**What:** `find_pruneable_heads_and_indices` ve `prune_linear_layer` da `modeling_utils`'den kaldırılmış. `apply_chunking_to_forward` ile birlikte üçü tek bir `try/except` bloğuna alındı: önce `transformers.pytorch_utils` denenir, yoksa `transformers.modeling_utils`.
**Why:** `transformers>=4.36` bu üç yardımcıyı `pytorch_utils`'e taşıdı. Python 3.12 + yeni transformers kombinasyonunda `ImportError` veriyor. Mimari hiç değiştirilmedi.

---

## 2026-05-25 — `bert_model.py` + `requirements_colab.txt` + notebook path/arg düzeltmeleri

**What:**
- `bert_model.py`: `transformers.file_utils` → `transformers.utils` (try/except); `apply_chunking_to_forward` → `pytorch_utils` (try/except)
- `requirements_colab.txt`: `transformers==4.22.2` → `>=4.36.0`
- Notebook dataset-verify hücresi: JSONL yolu `datasets/*.jsonl` → `datasets/Kvasir_VQA/*.jsonl`
- Notebook train-cell: `--data_dir` ve `--image_dir` argümanları eklendi (Drive mutlak yolları)

**Why:** Colab Python 3.12 + transformers≥4.36 kombinasyonunda `transformers.file_utils` ve `modeling_utils`'deki bazı semboller kaldırıldı. `transformers==4.22.2` Python 3.12'de `tokenizers` wheel'ini derleyemiyor. JSONL'ler `datasets/Kvasir_VQA/` altında, doğrulama hücresi yanlış yeri arıyordu. `--data_dir`/`--image_dir` verilmeyince `vqa_datasets.py` config.json'daki relative path'i çalışma dizinine göre arıyordu.

---

## 2026-05-25 — `IMAGEFOLDER_NAME` düzeltme + `dataset-copy` iyileştirme

**What:** `IMAGEFOLDER_NAME = "Kvasir_VQA/images"` → `"Kvasir_VQA/imgs"`. `resolve_dataset_image_source` boş klasörü geçerli kaynak saymayacak şekilde güncellendi. Drive'a kopyalama başarısız olursa `ACTIVE_IMAGE_DIR` direkt `DRIVE_IMAGE_DIR`'a fallback yapıyor. Sonunda görüntü sayısı yazdırılıyor.
**Why:** Drive'daki gerçek klasör adı `imgs`; notebook `images` arıyordu, hiç bulamıyordu. Boş klasör kontrolü olmadan çalışma dizini boş `imgs/` klasörünü kaynak sanıyordu.

---

## 2026-05-25 — Dataset split + görüntü isimlendirme düzeltmesi

**What:** HuggingFace Kvasir-VQA'nın tek split'i `raw` (58 849 soru, 6 500 benzersiz görüntü). `split_map`'teki `train/validation/test` isimleri yoktu, hiç görüntü kaydedilmedi. Düzeltme: `ds["raw"]` üzerinden okur, görüntüleri `img_id.jpg` ile kaydeder (tekrarsız), soru-cevap satırlarını `random.seed(42)` ile 80/10/10 böler.
**Why:** Önceki kod `train/validation/test` split adlarını arıyordu; bunlar yoktu, 0 görüntü + boş JSONL oluştu.

---

## 2026-05-25 — HF parquet cache Drive'a yönlendirildi

**What:** `HF_DATASETS_CACHE` ve `HF_HOME` → `Drive/DiffuVQA-Original/hf_cache/`. `load_dataset(..., cache_dir=DRIVE_HF_CACHE)` ile parquet dosyaları Drive'da kalıcı.
**Why:** Session restart'ta parquet'lar `/root/.cache/`'den silinir, her seferinde 1.5 GB yeniden indirilir. Drive cache ile sadece görüntü → JPEG dönüşümü yapılır.

---

## 2026-05-25 — Kvasir-VQA indirme: Zenodo → HuggingFace

**What:** Zenodo URL (`/records/10048817`) Kvasir-VQA değil, başka bir kayıt; 404 HTML dönüyor, `unzip` patlıyordu. HuggingFace `SimulaMet-HOST/Kvasir-VQA` (`datasets` kütüphanesi) ile değiştirildi.
**Why:** Kvasir-VQA zip olarak Zenodo'da yok; resmi kayıt HuggingFace'de parquet formatında.

---

## 2026-05-25 — Kvasir-VQA dataset indirme hücresi eklendi

**What:** Drive'da `imgs/` klasörü + `train.jsonl` varsa atlar; yoksa HuggingFace'den indirir, `Drive/DiffuVQA-Original/datasets/Kvasir_VQA/` altına kaydeder.
**Why:** Kullanıcının her session'da dataset'i manuel yüklemesi gerekmemeli. Bir kere indirilip Drive'da kalmalı.

---

## 2026-05-25 — Clone hücresine `git log -3` eklendi

**What:** Repo clone + Drive sync hücresine `git -C {LOCAL_CLONE_PATH} log -3 --oneline` satırı eklendi.
**Why:** Clone sonrası hangi commit'te olduğu görülsün, yanlış branch/commit ile çalışılmasın.

---

## 2026-05-25 — Notebook Bert branch yapısına uyarlandı + Drive path düzeltildi

**What:** `notebooks/run_diffuvqa_colab.ipynb` Bert branch'teki (`Aliekinozcetin/DiffuVQA`) notebook yapısına birebir benzetildi.
- Drive path `DiffuVQA` → `DiffuVQA-Original` (iki repo aynı klasöre yazmasın)
- Repo URL `DiffuVQA-Original` reposuna çevrildi, branch `main`
- Dil Türkçe yapıldı
- Büyük tek Config hücresi + bölüm bazlı markdown sub-header yapısı
- `resolve_dataset_image_source` ile çoklu Drive yolu deneme mantığı
- Dataset görüntü kopyalama + doğrulama + önizleme hücreleri
- `evaluate_and_export_csv` inline metrik fonksiyonu
- Bölüm 5: CSV Drive'a kaydet + indir + görselleştirme
- `HF_ENDPOINT` geçici kaldırma bloğu BERT indirme hücresine taşındı

**Why:** Drive'daki `DiffuVQA` klasörüyle çakışmayı önlemek ve notebook yapısını diğer repoyla tutarlı tutmak.

---

## 2026-05-25 — Initial Colab/A100 compatibility pass

### 1. `requirements_colab.txt` oluşturuldu
**What:** Colab'da önceden yüklü olmayan paketler. `torch`, `torchvision`, `wandb` hariç. `timm`, `open_clip_torch`, `openpyxl`, `pycocoevalcap` eklendi.
**Why:** `pip install -r requirements.txt` Colab'da torch'u yeniden yükler, CUDA versiyonunu bozabilir.

### 2. `wandb` `train.py`'den kaldırıldı
**What:** `import wandb`, `wandb.init()`, `wandb.config.update()`, `WANDB_MODE` bloğu tamamen silindi.
**Why:** wandb kimlik doğrulama gerektiriyor, araştırma eğitimi için gereksiz. Tüm metadata zaten `training_args.json`'a kaydediliyor.

### 3. `import wandb` `logger.py`'den kaldırıldı
**What:** Top-level `import wandb` satırı silindi. `dumpkvs()` içindeki entegrasyon zaten comment'teydi.
**Why:** Modül yüklenirken `ModuleNotFoundError` veriyordu.

### 4. `logger.configure()` güncellendi
**What:** `logger.configure()` → `logger.configure(dir=args.checkpoint_path, format_strs=["log", "csv"])`. `stdout` formatı kaldırıldı.
**Why:** `stdout` her `log_interval`'da büyük tablo basıyor, Colab çıktısını dolduruyor. `log.txt` + `progress.csv` checkpoint klasöründe Drive'da kalıcı.

### 5. `parse_known_args` eklendi
**What:** `parser.parse_args()` → `parser.parse_known_args()[0]`, bilinmeyenler loglanıyor.
**Why:** Colab/Jupyter kernel `sys.argv`'ye ekstra flag enjekte ediyor, `parse_args()` crash veriyor.

### 6. `dist.is_initialized()` guard eklendi
**What:** `dist.get_world_size()` öncesine `dist.is_initialized()` kontrolü eklendi.
**Why:** Tek GPU Colab'da `dist.init_process_group` çağrılmadan `get_world_size()` `RuntimeError` veriyor.

### 7. `HF_ENDPOINT` zaten mevcut
**What:** Her iki scriptte de `os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'` zaten vardı, değişiklik gerekmedi.

### 8. `notebooks/run_diffuvqa_colab.ipynb` oluşturuldu
**What:** 4 bölümlü Colab notebook: Setup → Training → Sampling → Evaluation.
**Why:** A100 Colab'da shell erişimi olmadan uçtan uca çalıştırılabilir olsun.

### 9. `CLAUDE.md` oluşturuldu
**What:** Proje özeti, dizin yapısı, komutlar, mimari diyagram, hiperparametre tablosu.
**Why:** Gelecekteki katkıcılar ve AI asistanlar için kılavuz.
