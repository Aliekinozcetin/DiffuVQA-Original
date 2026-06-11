# DECISIONS — DiffuVQA Colab/A100 Compatibility

Decisions are listed newest-first.

---

## 2026-06-09 — Bugfix: Progress CSV positional writer — resume'da bütünlük garantisi

**What:**
1. `diffuvqa/utils/logger.py`: `PositionalCSVOutputFormat` sınıfı eklendi. Her satır `row_idx = step // eval_interval` ile konumlandırılır; aynı adım için ikinci bir write gelirse (örn. save_interval == eval_interval durumu) overwrite eder, duplicate satır oluşturmaz. Resume'da mevcut CSV okunup `rows` dict'ine yüklenir, yeni satırlar doğru konuma yazılır.
2. `diffuvqa/utils/logger.py`: `make_output_format` ve `configure` fonksiyonlarına `eval_interval` parametresi eklendi. `eval_interval > 0` ise `PositionalCSVOutputFormat` devreye girer, aksi hâlde eski `CSVOutputFormat` kullanılır.
3. `train_util.py`: `log_interval` ve `eval_interval` için ayrı ayrı çağrılan iki `logger.dumpkvs()` → `eval_interval`'da tek çağrıya indirildi. Eval varsa önce `forward_only` çalışır, sonra tek `dumpkvs()` ile train + eval metrikleri aynı satıra yazılır.
4. `train.py`: Trim mantığı `0 < step <= resume_step` olarak düzeltildi — `step` alanı olmayan orphan eval satırları (`_step_val` → 0 döndürüyordu) artık trim'den kaçamaz. `eval_interval` `logger.configure()` çağrısına iletiliyor.

**Why:** İki bağımsız `dumpkvs()` çağrısı train ve eval metriklerini ayrı satırlara yazıyordu. Eval satırlarında `step` alanı yoktu; trim `0 <= resume_step` koşuluyla bu satırları "tutuyordu". Resume'da orphan eval satırları CSV'de birikiyordu. `save_interval == eval_interval` olan adımlarda checkpoint kaydedilip training çöküp resume edildiğinde aynı adım için duplicate satır oluşabiliyordu. `row = step // eval_interval` konumlandırması tüm bu senaryoları kapatır: her adım tek bir satıra sahip olur ve idempotent write garantisi verir.

**How to apply:** Yalnızca `eval_interval > 0` olduğunda `PositionalCSVOutputFormat` aktif olur. Mevcut checkpoint ve dataset uyumluluğu etkilenmiyor.

---

## 2026-06-09 — Bugfix: CUDAGraph buffer overwrite — `torch.compile` ile eğitim patlaması

**What:**
1. `diffuvqa/vqa_model.py` satır 305: `timestep_embedding(timesteps, self.hidden_t_dim)` → `.clone()` eklendi. CUDAGraph buffer'ından bağımsız kopya alınır.
2. `train_util.py` satır 304: `torch.compiler.cudagraph_mark_step_begin()` her microbatch döngüsünde model çağrısından önce eklendi.
3. `train.py` satır 119: `torch.compile(model, mode="reduce-overhead")` → `mode="default"`. `default` mod Triton kernel fusion'ı korur ama CUDAGraph kullanmaz.

**Why:** `torch.compile(mode="reduce-overhead")` CUDAGraph kullanır. CUDAGraph her adımda buffer'ları in-place overwrite eder; `timestep_embedding` çıktısı sonraki graph replay'de eziliyordu. Backward pass bu tensöre erişince `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run` fırlatılıyordu. Önceki eğitim run'larında görülmemişti çünkü yeni PyTorch sürümü (Colab ortamı güncellemesi) bu kontrolü sıkılaştırdı. Hız kaybını minimumda tutmak için `default` mod seçildi (~%5-10 yavaş, `reduce-overhead`'e kıyasla).

**How to apply:** `USE_TORCH_COMPILE=True` ile eğitim başlatılabilir. Checkpoint uyumluluğu etkilenmiyor.

---

## 2026-06-09 — Bugfix: `decode_token` tek-token fallback — `TypeError: argument of type 'int' is not iterable`

**What:** `basic_utils.py` `decode_token` içinde `seq.squeeze(-1).tolist()` çağrısı sonrası `isinstance(seq, int)` guard eklendi: skaler sonuç `[seq]` listesine sarılıyor.

**Why:** Sampling fallback path'inde (`sample_vqa_GPU.py:354`) `masked_logits[_wi, 0].argmax().unsqueeze(0)` shape `[1]` döndürüyor. `decode_token` içinde `squeeze(-1)` bunu skaler `[]`'e daraltıyor, `.tolist()` → `int`. Sonrasında `self.sep_token_id in seq` → `TypeError: argument of type 'int' is not iterable`. Hata yalnızca winner boş/noktalama olduğunda ve BioBERT tokenizer (PreTrainedTokenizerFast) kullanıldığında tetikleniyordu.

**How to apply:** Eğitim kodu etkilenmiyor; yalnızca sampling.

---

## 2026-06-07 — v0.4 güncelleme: batch=160, LR linear scaling, warmup_steps=5000

**What:**
1. `notebooks/run_diffuvqa_colab.ipynb`: `TRAIN_BATCH_SIZE` 128 → 160. `LR = 1e-5 * (TRAIN_BATCH_SIZE / 32)` zaten dinamikti — batch=160 ile LR otomatik `5e-5` oluyor.
2. `notebooks/run_diffuvqa_colab.ipynb`: `WARMUP_STEPS = 5000` değişkeni eklendi, `--warmup_steps {WARMUP_STEPS}` CLI argümanı training komutuna eklendi.
3. `diffuvqa/config.json`: `warmup_steps` 2000 → 5000 (CLI override olmayan durumlar için).

**Why:** batch=32 ile eğitimde GPU VRAM yalnızca %22 kullanılıyordu. batch=160 ile aynı 180k adımda 5x fazla sample görülüyor. Linear scaling rule gereği LR de 5x artırılıyor (1e-5 → 5e-5). Büyük LR ile ilk adımlarda gradientler kararsız olabileceğinden warmup 2000 → 5000'e çıkarıldı.

**How to apply:** Mevcut eğitimi durdur, `TRAIN_BATCH_SIZE=160` ve `WARMUP_STEPS=5000` ile sıfırdan başlat. Checkpoint uyumluluğu etkilenmiyor.

---

## 2026-06-07 — Bugfix: `train.py` UnboundLocalError — `import torch._dynamo` yerel scope çakışması

**What:** `train.py` `main()` içindeki `import torch._dynamo` satırı `import torch._dynamo as _dynamo` olarak değiştirildi. İçeride `torch._dynamo.config.suppress_errors = True` → `_dynamo.config.suppress_errors = True` olarak güncellendi.

**Why:** Python, bir fonksiyon içinde herhangi bir yerde `import X` (veya `X = ...`) görürse `X`'i o fonksiyonun *yerel* değişkeni olarak işaretler. `use_torch_compile=False` olduğunda if bloğu hiç çalışmıyor, yani `torch` hiç bind edilmiyor — ama Python yine de onu local sayıyor. Fonksiyonun üstündeki `if torch.cuda.device_count() > 1:` satırına gelindiğinde `UnboundLocalError: cannot access local variable 'torch'` fırlatılıyor. `as _dynamo` ile farklı bir isim kullanınca `torch` adı local scope'a bağlanmıyor, modül-seviyesi `import torch` geri devreye giriyor.

**How to apply:** Eğitim kodu değişikliği; checkpoint veya veri seti etkilenmiyor. Mevcut tüm checkpoint'lerle uyumlu.

---

## 2026-06-07 — v0.4: Kaynak optimizasyonu — batch=32, BF16, DataLoader, torch.compile

**What:**
1. `notebooks/run_diffuvqa_colab.ipynb`: `TRAIN_BATCH_SIZE` 4 → 32. A100 GPU VRAM %14 kullanımdan ~%80'e çıkıyor.
2. `diffuvqa/vqa_datasets.py`: DataLoader `num_workers` 4 → 8, `pin_memory=True`, `persistent_workers=True`. RAM %9 kullanımda iken data loading bottleneck'ti; pin_memory CPU→GPU transferini pinned memory üzerinden yapıyor.
3. `train_util.py` + `train.py`: BF16 (bfloat16) AMP desteği eklendi. A100 native Tensor Core'ları kullanıyor, FP32'den 1.5-2x hızlı. BF16'nın dynamic range'i FP16'dan geniş, loss scaling gerekmez. `use_bf16=True` ile etkinleştiriliyor.
4. `train.py`: `torch.compile(model, mode="reduce-overhead")` desteği eklendi. Triton kernel fusion ile ~%20-30 hız kazanımı. Varsayılan kapalı (`USE_TORCH_COMPILE=False`), ilk çalıştırmada ~5 dakika warmup süresi var.
5. `diffuvqa/config.json`: `use_bf16: false`, `use_torch_compile: false` flag'leri eklendi.
6. `notebooks/run_diffuvqa_colab.ipynb`: `USE_BF16=True`, `USE_TORCH_COMPILE=False` config değişkenleri ve `--use_bf16`, `--use_torch_compile` argümanları training komutuna eklendi.

**Why:** v0.3 sampling analizi gösterdi ki GPU VRAM 5.8/40 GB (%14), RAM 7.5/83.5 GB (%9) kullanılıyor. Batch=4 ile her gradient güncellemesi çok gürültülü — cosine LR ve 2x loss weight uygulanmış olmasına rağmen v0.3'te overfit 80k'da başlıyor. Batch=32 ile effective training 3M → 24M sample-exposure/750k step, gradient kalitesi ve genelleme artmalı.

**How to apply:** Notebook Config hücresinde `TRAIN_BATCH_SIZE=32`, `USE_BF16=True` ayarla. Sıfırdan veya mevcut checkpoint'ten devam edilebilir. `strict=False` load zaten mevcut.

**Beklenen etki:** Overfit başlangıcının 80k'dan çok daha geç gelmesi; v0.3'te 80k peak olan exact match'in daha yüksek bir checkpoint'te peak yapması.

---

## 2026-06-07 — Training + inference iyileştirmeleri: Fix3, loss weighting, cosine LR, seq_len, classifier head

**What:**
1. `sample_vqa_GPU.py` Fix3: İki ayrı YN mask — `yn_mask_bool` (SEP yok) position 0 için, `yn_mask_with_sep` (SEP dahil) position 1+ için. Model position 0'da gerçek cevap token'ı (yes/no) seçmek zorunda, position 1+'da SEP seçip decode_token kesebiliyor.
2. `diffuvqa/gaussian_diffusion.py`: Closed-ended (yes/no) örnekler için 2x loss weight — `is_closed` flag dataset'ten geliyor, training loss'u bu örneklerde iki katına çıkarıyor.
3. `diffuvqa/gaussian_diffusion.py` + `vqa_model.py`: Classifier head (`question_type_head = Linear(768, 1)`) eklendi — fused feature mean-pool → binary logit. Training sırasında BCE loss ile öğreniyor. Inference'ta first-word heuristic yerine OR mantığıyla kullanılıyor (classifier OR heuristic = closed).
4. `train_util.py`: Linear decay → Cosine decay + linear warmup. Parametreler: `warmup_steps=2000`, `lr_min=5e-6`. 40k'dan sonra LR yavaşça 1e-5 → 5e-6'ya iner, erken adımlarda 0'dan linear warmup.
5. `diffuvqa/config.json`: `seq_len` 32'de bırakıldı. Kvasir-VQA'da 30 token uzunluğuna ulaşan cevaplar var — 16'ya düşürmek bu örnekleri kesecekti.
6. `diffuvqa/vqa_datasets.py`: `is_closed` label eklendi (answer ∈ {yes, no, none, not, not relevant, not applicable} → 1, diğerleri 0). Dataset cache devre dışı bırakıldı.

**Why:**
- Fix3: Fix2'de yn_mask tüm pozisyonlara uygulandığı için SEP hiç üretilmiyordu → decode_token 32 pozisyonu decode → avg_len 17, Y/N %0.0.
- 2x loss weight: Y/N soruları veri setinin %48'i ama model öğrenemiyor — embedding space open-ended yanıtlara baskın. Ek ağırlık gradyan dengesini düzeltiyor.
- Classifier head: First-word heuristic edge case'leri kaçırıyor ("what colour" sorusu ama cevap yes/no). Learned head tüm modality sinyalini (image + question) kullanıyor.
- Cosine LR: 40k exact match zirvesine bakılırsa model 40k'dan sonra overfit. Cosine decay 40k–120k arasında LR'yi yavaşça düşürüyor.
- seq_len=32 korundu: Kvasir-VQA'da 30-token uzunluğunda cevaplar mevcut; kesmek zararlı olurdu.

**How to apply:**
- Fix3 + classifier inference değişiklikleri: Mevcut checkpoint'lerle çalışır (`strict=False` load). Classifier henüz eğitilmemişse rastgele init → heuristic fallback devrede.
- Loss weighting + cosine LR + classifier BCE: Sıfırdan veya 40k checkpoint'ten retraining gerektirir.

---

## 2026-06-07 — Inference-side iyileştirmeler: question-type masking, empty fallback, artifact cleanup

**What:**
1. `sample_vqa_GPU.py`: Question-type aware answer vocab kısıtlaması — `is/are/does/do/have/has/was/were/can/could/would/will/did/should` ile başlayan sorularda rounding ve final decode yalnızca `{yes, no, not, relevant, not applicable, 0, 1}` token'larına kısıtlandı. Her sample için ayrı per-sample masking uygulandı.
2. `sample_vqa_GPU.py`: `[SEP]/[PAD]/[CLS]` token'ları YN vocab mask'tan çıkarıldı — bu token'lar seçilince `decode_token` boş string üretiyordu, closed-ended soru başına empty artışına neden oluyordu.
3. `sample_vqa_GPU.py`: Empty / punctuation-only fallback eklendi — winner `''`, `-`, `;` vb. ise tüm seq pozisyonları sırayla denenerek ilk geçerli non-empty token seçiliyor.
4. `basic_utils.py` `decode_token`: Baş ve son noktalama artifact'larını silen regex eklendi (`"- center"` → `"center"`, `"; yes"` → `"yes"`). Tek-token fallback path için `isinstance(seq, int)` guard eklendi.

**Why:** 40k–120k checkpoint sample analizi şunu gösterdi: Y/N accuracy %0.3 (1518 sorudan 4-11 doğru), boş cevap %11-23, top üretilen token'lar spatial kelimeler (`center`, `left`, `-`, `;`). Model yes/no öğrenmemiş değil — doğru vocab'a yönlendirilmediği için spatial token'lara kayıyor. Inference-side kısıtlama eğitim gerektirmeden uygulandı.

**Sonuç (yeni sample ölçümü):**
- Exact match: 40k %0.42 → %1.90 (4.5x), 80k %0.25 → %1.43 (5.7x)
- Y/N accuracy: 40k %0.7 → %5.9 (8x), 80k %0.3 → %4.6 (15x)
- Empty artışı ([SEP] bug): `special_ids -= yn_mask` fix ile giderildi

**How to apply:** Eğitim gerekmez. Mevcut checkpoint'leri `sample_vqa_GPU.py` ile yeniden sample et.

---

## 2026-06-06 — main'den sync: CIGN fix, deprecated API, N_SAMPLES, eval fix, rounding fix

*What:*
1. gaussian_diffusion.py: CIGN f vektörü düzeltildi — [ddpm_input_pre, ddpm_input_pre] → [ddpm_input_pre, ans_emb_pre]
2. gaussian_diffusion.py: p_sample_loop final=None → final=[] list accumulation
3. vqa_model.py: get_extended_attention_mask deprecated API → direkt additive mask hesabı
4. rounding.py: gereksiz .to(model.weight.device) cast'ları kaldırıldı
5. notebook: N_SAMPLES 5→1
6. eval_DiffuVQA.py: --file argümanı + BERTScore OverflowError fix
*Why:* main branch'teki düzeltmeler BioBERT branch'ine sync edildi.

---

## 2026-06-02 — SEP anchor kaldırıldı + loss/grad/LR düzeltmeleri (main'den sync)

**What:**
1. `vqa_datasets.py` `merge_and_mask`: SEP anchor kaldırıldı, `mask_a = [1] * len(a_ids)` — tüm answer token'ları noised.
2. `sample_vqa_GPU.py`: SEP embedding injection bloğu kaldırıldı.
3. `gaussian_diffusion.py` `_token_discrete_loss`: `sep_weight` parametresi eklendi; `decoder_nll` 1x, `terms["nll"]` 5x.
4. `gaussian_diffusion.py` loss: `2.0 * nll + 2.0 * decoder_nll` → `1.0 * nll + 1.0 * decoder_nll`.
5. `config.json`: `learning_steps` 500k→750k, `gradient_clipping` 0.5→0.75.
6. `train_util.py`: `pre_answer_loss` gate lineer → cosine; `import math` eklendi.
7. `vqa_model.py` `logits_mode=2` reshape bug düzeltildi.
**Why:** main branch ile senkronize edildi. Biobert-spesifik değerler (`vocab_size=28996`, `config_name`, `language_encoder_name`, `eval_interval=500`) korundu.

---

## 2026-06-02 — `rounding.py` CUDA/CPU device mismatch düzeltildi

**What:** `denoised_fn_round` içindeki üç `model(tokens)` çağrısına `.to(model.weight.device)` eklendi (satır 113, 129, 147).
**Why:** `get_logits` path'inde `q_tokens`/`a_tokens`/`rounded_tokens`, `old_device` (CUDA) üzerinde oluşuyordu; embedding `model` ise CPU'daydı. `RuntimeError: Expected all tensors to be on the same device` ile sampling tamamen patlıyordu. `get_efficient_knn` path'i `.to(model_emb.device)` ile zaten doğruydu; `get_logits` path'i bu adımı atlıyordu.
**How to apply:** Yalnızca `sample_vqa_GPU.py` (DDIM inference) etkili. Training kodu `denoised_fn_round`'u çağırmıyor — checkpoint yeniden eğitim gerektirmiyor.

---

## 2026-06-01 — 350k sampling analizi: 5 bug düzeltildi

**What:**
1. `gaussian_diffusion.py` [C-3]: `ddim_sample_loop_progressive`'e `top_p`, `clamp_step`, `clamp_first` artık iletiliyor. Önceden `ddim_sample_loop` bu parametreleri kabul edip `ddim_sample_loop_progressive`'e geçirmiyordu → DDIM modunda (inference default) rounding her adımda zorla uygulanıyordu, clamp_step/clamp_first hiç çalışmıyordu.
2. `rounding.py` [M-1]: `denoised_fn_round` answer_vocab kısıtlamasını artık sadece cevap yarısına (`[:, seq_len:, :]`) uyguluyor. Önceden `(B, 2*seq_len, hidden)` tensörü düzleştirilip tüm konumlara answer_vocab projeksiyonu yapılıyordu; soru yarısı (conditioning) bozuluyordu.
3. `eval_DiffuVQA.py` [M-2]: `calculate_f1` karakter-level `edit_distance` threshold'u yerine standart token-level F1 (TP/FP/FN token overlap) kullanıyor. "yes" in "paris yes" senaryosunda önceki kod FP sayıyordu.
4. `diffuvqa/vqa_model.py` [M-3]: `feature_fusion`'da `question_emb` (pre-encoder, embedding layer çıktısı) yerine `question_feats` (post-encoder, transformer çıktısı) kullanılıyor. Hem CVAE branch'inde hem alpha/beta/theta fusion'ında düzeltildi.
5. `gaussian_diffusion.py` [M-4]: `_WrappedModel.__call__` artık `**kwargs` iletiyor.
6. `sample_vqa_GPU.py` [m-4]: `answer_vocab` artık test split yerine train split'ten oluşturuluyor; test set answer istatistikleri rounding adımına sızmıyor.
**Why:** 350k checkpoint JSONL analizi şunu gösterdi: YN accuracy %3 (random'dan kötü), over-generation %38, suffix artifact %47. C-3 ve M-1 DDIM inference yolunu doğrudan kırıyordu; M-2 raporlanan F1'i yanıltıcı yapıyordu; M-3 feature fusion'ı semantic seviye uyumsuzluğuyla zayıflatıyordu.
**How to apply:** 350k checkpoint'ini mevcut kodla yeniden sample etmek yeterli (C-1/C-2 zaten düzeltilmişti). YN accuracy'nin ~3%'ten 40–60%'e çıkması bekleniyor.

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
