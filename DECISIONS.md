# DECISIONS — DiffuVQA Colab/A100 Compatibility

Decisions are listed newest-first.

---

## 2026-05-28 — Optimizer state checkpoint kaydetme/yükleme

**What:** `train_util.py`'de her `save()` çağrısında artık `opt_{step:06d}.pt` dosyası da yazılıyor. Resume'da `_load_optimizer_state()` bu dosyayı `checkpoint_path`'ten arayıp yüklüyor; dosya yoksa (eski checkpoint) "starting fresh" log'u yazılıyor ve devam ediyor. `__init__`'teki `# self._load_optimizer_state()` satırı aktif hale getirildi.
**Why:** Her resume'da Adam'ın momentum ve variance state'i sıfırlanıyordu. Bu, özellikle learning rate annealing aktifken yakınsama kalitesini bozuyor; model yavaş bir LR'den başlıyor ama optimizer "unutmuş" gibi davranıyor. Düzeltme: ~150MB ekstra checkpoint dosyası, sıfırdan optimizer yerine iyi momentum state'iyle devam.

---

## 2026-05-28 — Kod denetimi: 3 bug düzeltildi

**What:**
1. `sample_vqa_GPU.py`: `sample_shape` artık `(B, args.seq_len, D)` değil `(B, x_start.shape[1], D)` — x_start (B,64,768) olduğundan sample_shape da 64 olmalıydı. Önceki haliyle p_sample_loop içinde noise verilince sample_shape yok sayılıyordu, ama misleading kod kaldı.
2. `vqa_model.py`: `bert_config` dead code silindi — `BertConfig(max_position_embeddings=seq_len=32)` oluşturulup `from_pretrained`'e geçilmiyordu; `config=AutoConfig` (512 position) kullanılıyordu.
3. `gaussian_diffusion.py`: NLL loss artık padding token'ları hariç tutuyor — `answer_mask = (input_ids_a != 0).float()` ile `_token_discrete_loss`'a mask geçiliyor.
**Why:** sample_shape tutarsızlığı sampling güvenilirliğini etkiliyor; dead code kafa karıştırıcı; padding token'larının loss'a dahil edilmesi answer token sinyalini zayıflatıyor.

---

## 2026-05-28 — `decode_token` [SEP]'de truncate + skip_special_tokens

**What:** `basic_utils.py`'deki `decode_token` her zaman 32 token'lık tam diziyi decode ediyordu. Artık ilk `[SEP]` token id'sinde keserek durduruyor; HuggingFace decode'a `skip_special_tokens=True` eklendi.
**Why:** Model her zaman tam `seq_len=32` token üretmek zorundaydı. Kvasir-VQA referans cevapları 1-3 kelime iken model 18 kelimelik output yazıyordu. Bu BLEU/METEOR/ROUGE skorlarını doğrudan bastırıyordu.
**How to apply:** Hem `generate_answer` hem `reference_answer` aynı `decode_token`'ı kullandığından ikisi de tutarlı şekilde kesilecek. Eğitim kodu bu fonksiyonu çağırmıyor — sadece sampling etkilendi.

---

## 2026-05-28 — Training loop `learning_steps`'te durmuyor bugı düzeltildi

**What:** `train_util.py`'deki `for epoch in range(self.learning_steps)` döngüsü `learning_steps` kez epoch dönüyordu — her epoch tüm dataset'i işlediğinden çok fazla adım atılıyordu. `while global_step < learning_steps` döngüsüne çevrildi. Log/eval koşulları da `global_step` bazlı yapıldı.
**Why:** Orijinal kod epoch sayısını step sayısı gibi kullanmış. Resume'da `step=0`'dan başlandığında da `0 % log_interval == 0` tetiklenip gereksiz log/eval yapılıyordu.
**How to apply:** `self.step + self.resume_step >= self.learning_steps` koşulu her batch başında kontrol edilir, hedef adıma ulaşınca döngü kırılır.

---

## 2026-05-28 — Resume'da progress.csv resume_step üstündeki satırları temizle

**What:** `train.py`'de resume başlarken `progress.csv`'deki `step > resume_step` satırları siliniyor. Checkpoint dosya adından step numarası parse ediliyor (örn. `ema_0.9999_378000.pt` → 378000), o adımın üstündeki satırlar kırpılıyor.
**Why:** 378k'da durdurup tekrar 375k'dan resume edilirse 375k-378k arası loglar CSV'de kalır. Yeni eğitim aynı step numaralarını tekrar yazınca duplicate satırlar oluşur. Kırpma ile resume_step'in üstündeki stale loglar temizlenir, grafik temiz devam eder.
**How to apply:** Sadece `is_resume=True` ve `resume_step > 0` durumunda çalışır. Fresh start'ta dokunmaz.

---

## 2026-05-28 — `is_resume` "none" string bug düzeltildi; `log_interval` 1000'e alındı

**What:** `train.py`'deki `is_resume` kontrolü `"none"` string'ini `True` sayıyordu → her fresh start'ta `append_csv=True` oluyordu. Düzeltildi: `"none"` / `"false"` / boş string → `is_resume=False`. Aynı zamanda `log_interval` config.json'da 200→1000, notebook'ta 100→1000 yapıldı.
**Why:** `args.resume_checkpoint` default değeri config.json'da `"none"` string'i. `bool("none")=True` olduğundan fresh start'ta bile CSV append modunda açılıyordu. Eğer Drive'da eski bir `progress.csv` varsa, append+header restore mantığı çakışarak CSV'nin bozulmasına yol açabiliyordu. Log interval ise gereksiz yere çok sık yazıyordu (100 adımda bir).
**How to apply:** `is_resume = bool(_rc) and _rc.lower() not in ('', 'none', 'false')` — sadece gerçek dosya yolu geldiğinde append modu açılır.

---

## 2026-05-28 — `logger.py` `dumpkvs()` writekvs döngüsü uncomment edildi

**What:** `Logger.dumpkvs()` içindeki `for fmt in self.output_formats: fmt.writekvs(d)` döngüsü yanlışlıkla comment'e alınmıştı. Uncomment edildi.
**Why:** wandb entegrasyonu kaldırılırken wandb.log çağrısıyla birlikte writekvs döngüsü de comment'e alınmış. Sonuç: `logkv()` ile kaydedilen tüm metrikler (step, loss, grad_norm vb.) hiçbir output format'a yazılmıyordu. `progress.csv` açılıp oluşturuluyordu ama içi daima boştu. Görselleştirme hücresindeki loss curve ve tüm eğitim grafikleri bu dosyaya bağımlı olduğundan çalışmıyordu.

---

## 2026-05-27 — TRAIN_BATCH_SIZE=4, SAMPLE_BATCH_SIZE=64 olarak ayrıldı

**What:** Notebook config hücresinde `BATCH_SIZE` tek değişken yerine `TRAIN_BATCH_SIZE=4` ve `SAMPLE_BATCH_SIZE=64` olarak ikiye ayrıldı. `train-cell` → `TRAIN_BATCH_SIZE`, `sample-cell` → `SAMPLE_BATCH_SIZE` kullanıyor.
**Why:** `BATCH_SIZE=64` hem eğitim hem sampling'e uygulanınca eğitim ~7x yavaşladı (10h → 70h). Büyük batch eğitimde gradient gürültüsünü azaltır, genellemeyi düşürür; sampling'de ise bellek baskısı olmadığından 64 güvenli.

---

## 2026-05-27 — Notebook `evaluate_and_export_csv` dataset yolu düzeltildi

**What:** `dataset_file` parametresi `"datasets/test.jsonl"` hardcoded'dan `None` default'a çevrildi. Runtime'da `DRIVE_PROJECT_PATH/datasets/DATASET/test.jsonl` olarak çözümleniyor.
**Why:** Relative path cwd'ye bağlıydı; `DRIVE_PROJECT_PATH` altında çalışırken `datasets/Kvasir_VQA/test.jsonl` bulunamıyordu → `[Errno 2] No such file or directory`.

---

## 2026-05-27 — `logger.py` progress.csv: sıfırdan eğitimde temizle, resume'da devam et

**What:** `CSVOutputFormat.__init__` `append` parametresi aldı. `logger.configure()` `append_csv=True/False` parametresi aldı. `train.py`'de `resume_checkpoint` set ise `append_csv=True`, boşsa `False` geçiliyor.
**Why:** `open(filename, "w+t")` her `logger.configure()` çağrısında dosyayı siliyordu. Resume'da önceki eğitim kayıtları kayboluyordu; Drive'dan indirilen `progress.csv` boş geliyordu.

---

## 2026-05-27 — `sample_vqa_GPU.py` confidence + avg_nn_l2 eklendi

**What:** Her JSONL satırına `confidence` (ortalama top-1 softmax olasılığı) ve `avg_nn_l2` (denoised embedding ile en yakın vocab embedding arası ortalama L2 mesafesi) alanları eklendi.
**Why:** Sampling kalitesini per-sample düzeyde ölçmek için. Düşük confidence veya yüksek avg_nn_l2 rounding kalitesinin bozuk olduğunu gösterir.

---

## 2026-05-27 — `sample_vqa_GPU.py` çıktı yolu sadeleştirildi

**What:** Eski: `out_dir/lr1e-05/ema_0.9999_200000.pt.samples/seed105_step0.jsonl`. Yeni: `out_dir/lr1e-05/ema_0.9999_200000.jsonl`.
**Why:** `.samples` alt klasörü gereksiz iç içe yapı yaratıyordu. Checkpoint adı doğrudan dosya adı olunca hangi step'ten üretildiği okunabilir.

---

## 2026-05-27 — `gaussian_diffusion.py` `print(i)` kaldırıldı

**What:** `p_sample_loop` ve `ddim_sample_loop` içindeki `print(i)` satırları kaldırıldı.
**Why:** Her diffusion timestep'i (2499→0) ayrı satıra basılıyordu, tqdm bar varken de Colab çıktısını sayı sütunlarıyla dolduruyordu.

---

## 2026-05-26 — `sample_vqa_GPU.py` tqdm progress bar + tensor print'ler kaldırıldı

**What:** `print(fuse_feats.shape)`, `print(sample.shape)`, `print(cands.indices)`, `print(args.batch_size)` satırları kaldırıldı. Ana sampling döngüsü `tqdm` ile sarıldı: `Sampling: 45%|████| 663/1472 [batch/s]`.
**Why:** Her batch'te tensor boyutları ve raw indeks değerleri basılıyordu, çıktı okunamaz hale geliyordu.

---

## 2026-05-26 — `sample_vqa_GPU.py` `use_noising_f` AttributeError düzeltildi

**What:** `if args.use_noising_f:` → `if getattr(args, 'use_noising_f', False):` olarak değiştirildi.
**Why:** `use_noising_f` ne `config.json`'da ne argparse defaults'ta tanımlı; `training_args.json` update'inden sonra `args` namespace'inde hiç oluşmuyordu → `AttributeError`.

---

## 2026-05-26 — `sample_vqa_GPU.py` CLI `--data_dir` / `--image_dir` korunuyor

**What:** `training_args.json` update sonrası `model_path`, `out_dir`, `data_dir`, `image_dir` gibi sampling-specific CLI argümanları `_keep` dict ile kaydedilip geri yükleniyor.
**Why:** `args.__dict__.update(training_args)` tüm CLI argümanlarının üzerine yazıyordu. `model_path` training_args.json'daki eski relative path'e (`diffuvqa/config/ema_...`) dönüyordu → `FileNotFoundError`. Yeni session'da `image_dir` de silinmiş runtime path'e işaret ediyordu.

---

## 2026-05-26 — `train_util.py` `dist.get_rank()` + `sync_params` guard eklendi

**What:** `_load_ema_parameters` içindeki `dist.get_rank()` → `dist.get_rank() if dist.is_initialized() else 0`. `dist_util.sync_params(ema_params)` da `if dist.is_initialized():` koşuluna alındı.
**Why:** Resume checkpoint yüklemesi aktif olunca `_load_ema_parameters` ilk kez gerçekten çalıştı; `dist.get_rank()` process group olmadan `ValueError` fırlatıyordu. İlk eğitimde `resume_checkpoint=""` olduğu için bu kod hiç execute edilmemişti.

---

## 2026-05-26 — `train_util.py` `_load_and_sync_parameters` düzeltildi + resume_step parse

**What:** `_load_and_sync_parameters` sadece `pass` içeriyordu. Artık `resume_checkpoint` dosyasını yükleyip `parse_resume_step_from_filename` ile adım numarasını parse ediyor. tqdm `initial=self.step + self.resume_step` olarak güncellendi.
**Why:** Checkpoint yüklenmiyordu, `self.resume_step=0` kalıyordu → tqdm bar 0'dan başlıyordu. Model ağırlıkları da yüklenmiyordu, eğitim sıfırdan devam ediyordu.

---


## 2026-05-25 — `train_util.py` tqdm progress bar eklendi

**What:** `run_loop`'taki `print(f'Epoch: {epoch}, Step: {self.step}')` ve `print("loss", ...)` satırları kaldırıldı. Yerine `tqdm` progress bar eklendi: toplam `learning_steps` adım, her step'te `loss=X.XXXX` postfix olarak gösteriliyor. `_last_loss` attribute'u `forward_backward`'da set ediliyor.
**Why:** Her step ayrı satıra basılıyordu, Colab çıktısı doluyordu. Progress bar çok daha okunabilir.

---

## 2026-05-25 — `vqa_model.py` `get_extended_attention_mask` `device` kwarg kaldırıldı

**What:** `vqa_model.py` satır 124'teki `self.bert.get_extended_attention_mask(q_mask, q_input_shape, device=q_ids.device)` çağrısından `device=` kwarg'ı kaldırıldı.
**Why:** `transformers>=4.36`'da `ModuleUtilsMixin.get_extended_attention_mask()` imzasından `device` parametresi kaldırıldı; device artık attention_mask tensor'ından otomatik alınıyor. `TypeError: unexpected keyword argument 'device'` veriyordu. Model davranışı değişmiyor.

---

## 2026-05-25 — `train_util.py` tek GPU'da `.module` AttributeError düzeltildi

**What:** `_SingleGPUDDP` wrapper sınıfı eklendi (`th.nn.Module` subclass, `.module = model`). Tek GPU / CPU'da `DDP` yerine bu wrapper kullanılıyor. Çok GPU durumunda hâlâ gerçek `DDP` kullanılıyor.
**Why:** `gaussian_diffusion.py` `model.model.module.get_ddpm_input(...)` ile erişiyor — DDP convention'ını varsayıyor. Tek GPU Colab'da `dist.init_process_group` çağrılmadan gerçek `DDP` instantiate edilemez, model wrap edilmiyordu, `.module` attribute'u yoktu → `AttributeError`.

---

## 2026-05-25 — `vqa_datasets.py` çift `imgs/` prefix hatası düzeltildi

**What:** `ImageDataset.load_image_path` içindeki `f'{self.image_root}/{image_name}'` → `os.path.normpath(os.path.join(self.image_root, image_name))` olarak değiştirildi.
**Why:** JSONL'deki `image_name` değerleri `"imgs/xxx.jpg"` formatında; `image_root` zaten `…/Kvasir_VQA/imgs` olunca sonuç `imgs/imgs/xxx.jpg` → `FileNotFoundError`. `os.path.join` + `normpath` ile path birleşimi OS-native ve prefix tekrarına karşı dayanıklı.

---

## 2026-05-25 — `train.py` + `sample_vqa_GPU.py` HF_ENDPOINT kaldırıldı

**What:** `train.py` ve `sample_vqa_GPU.py`'nin başındaki `os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'` satırları kaldırıldı. Yerine açıklayıcı bir yorum bırakıldı.
**Why:** `HF_ENDPOINT=hf-mirror` set iken `AutoTokenizer.from_pretrained("bert-base-uncased")` ve `BertModel.from_pretrained(...)` hf-mirror'a gidip `FileMetadataError` → `OSError` ile patlıyor. hf-mirror BERT dosyalarını servis etmiyor. Dataset indirme (Kvasir-VQA) notebook hücresinde ayrıca yönetildiğinden `HF_ENDPOINT`'in script içinde set edilmesine gerek yok. `basic_utils.py`'deki pop/restore korundu (defense in depth).

---

## 2026-05-25 — `basic_utils.py` HF_ENDPOINT bypass: tokenizer + model yüklemesi

**What:** `myTokenizer.__init__` ve `create_model_and_diffusion` içinde `HF_ENDPOINT` geçici olarak `os.environ.pop` ile kaldırılıyor; BERT ağırlıkları/config'i doğrudan `huggingface.co`'dan indiriliyor. Yükleme bittikten sonra `HF_ENDPOINT` geri ekleniyor.
**Why:** `HF_ENDPOINT=https://hf-mirror.com` set iken `AutoTokenizer.from_pretrained("bert-base-uncased")` ve `BertModel.from_pretrained("bert-base-uncased")` çağrıları `hf-mirror`'a gidip `FileMetadataError` → `LocalEntryNotFoundError` → `OSError` zinciriyle çöküyor. hf-mirror BERT dosyalarını servis etmiyor. Notebook'taki `bert-cache` hücresi aynı bypass'ı yapıyor ama `train.py` subprocess olarak çalıştığında environment inherit ediyor, bypass devre dışı kalıyordu.

---

## 2026-05-25 — `diffuvqa/utils/answer_pre.py` eklendi

**What:** `vqa_model.py`'nin import ettiği `diffuvqa.utils.answer_pre` modülü repoda eksikti. `DiffuVQA` (diğer repo) `utils/` klasöründen alındı: `difflib.SequenceMatcher` tabanlı `find_most_similar_answers` fonksiyonu (dependency-free, lightweight).
**Why:** `train.py` → `basic_utils.py` → `vqa_model.py` import zincirinde `ModuleNotFoundError` ile çöküyordu. Dosya orijinal repoda mevcut ama bu repoya eklenmemiş.

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
