# DECISIONS — DiffuVQA Colab/A100 Compatibility

Decisions are listed newest-first.

---

## 2026-06-08 — Content-weighted MSE, tT_loss mask, reg_loss_type=len

**What:**
1. `gaussian_diffusion.py`: MSE loss'a content-weight eklendi — answer content 5x, question+padding 1x. t0_loss'a aynı mask uygulandı. tT_loss'a answer_mask uygulandı (padding bias kaldırıldı).
2. Notebook train komutu: `--reg_loss_type len` eklendi — uzunluk regularizasyonu aktif edildi (lambda_reg=0.03).

**Why:**
- MSE'nin %87.5'i padding'den geliyordu (ort cevap 3 token, seq_len=32 → 29 pozisyon boş). Model padding'i optimize edince loss düşüyor ama content token'ları yanlış kalıyor → loss-EM kopukluğu. 5x content weight ile gradient enerjisin büyük kısmı cevap token'larına gider.
- t0_loss ve tT_loss padding dahil hesaplanıyordu → aynı bias üçüncü ve dördüncü kaynak. Her ikisine de mask uygulandı.
- reg_loss_type='len' kodda mevcuttu ama hiç aktif edilmemişti. Model tahmin edilen cevap uzunluğunu referansa karşı penalize eder → over-generation (6.3 kelime, ref 2.9) azalır. t < T/2 adımlarında aktif.

---

## 2026-06-08 — sep_weight=2.0, seq_len=16, subword filtering — sıfırdan training

**What:**
1. `gaussian_diffusion.py`: `sep_weight=1.0` → `sep_weight=2.0` (`terms["nll"]` çağrısında)
2. `config.json`: `seq_len: 32` → `seq_len: 16`; notebook `SEQ_LEN=16`
3. `sample_vqa_GPU.py`: `answer_vocab_ids`'ten `##` subword token'ları filtrelendi

**Why:**
- **sep_weight=2**: sw=1'de SEP payı ~%25 → over-generation (100k'da uzunluk 6.9, ref 2.9). sw=2'de SEP payı ~%40 → içerik ile denge. Matematiksel: 3 content + 1 SEP → 3×1+1×2=5 → SEP 2/5=%40.
- **seq_len=16**: Referans cevapların %61.8'i 1 token, %76.3'ü ≤3 token. 32 pozisyonda model SEP'i doğru yere koyamıyor. seq_len=16'da arama uzayı yarıya iner, model SEP konumunu çok daha hızlı öğrenir. Cevapların %95+'ı 16 token'a sığıyor.
- **subword filtering**: Garbled çıktının kaynağı `##` tokenlar — colonoscopy→co+##os+##co+##py, gastroscopy→gas+##tro+##sco+##py, ulcerative→ul+##cera+##tive. Bu parçalar vocab'dayken model bunları ayrı üretiyor. Decode'da filtreleme nearest whole-word embedding'e snap eder.

---

## 2026-06-08 — RESUME_CHECKPOINT 40k sw=1.0 checkpoint'e set edildi

**What:** `notebooks/run_diffuvqa_colab.ipynb` `RESUME_CHECKPOINT = f"{CHECKPOINT_PATH}/ema_0.9999_040000.pt"` yapıldı.
**Why:** 40k sw=1.0 analizi: boş cevap %25.7 (sw=5'te %34'tü), F1 %3.88, temiz cevap F1 %6.40. sep_weight=1.0 doğru yönde. Garbled %63.1 beklenen — model 40k'da henüz vocabulary'yi öğreniyor. 100-120k'da garbled < %30, F1 > %5 hedefi.

---

## 2026-06-07 — sep_weight 5.0 → 1.0, notebook RESUME_CHECKPOINT sıfırlandı, sıfırdan training

**What:** `gaussian_diffusion.py` `_token_discrete_loss` çağrısında `sep_weight=5.0` → `sep_weight=1.0`. `notebooks/run_diffuvqa_colab.ipynb` `RESUME_CHECKPOINT = ""` yapıldı (sıfırdan eğitim için).
**Why:** 180k sonrası analiz: EM %0.22, garbled %32, boş %41. Root cause — sep_weight=5 NLL'nin %83.3'ünü [SEP]'e harcıyordu, content token'larına yalnızca %17 gradient kalıyordu. Model anlamlı kelime üretmek yerine [SEP] pozisyonunu optimize etti. sep_weight=1 ile SEP ve content token'lar eşit ağırlıkta; tüm answer token'lar tam gradient alıyor. Sıfırdan training gerekli (mevcut checkpoint sep_weight=5 eğitimi).

---

## 2026-06-07 — answer_vocab'a [SEP] geri eklendi, [CLS]/[PAD] dışarıda kaldı

**What:** `sample_vqa_GPU.py`: `answer_vocab_set.discard(sep_token_id)` satırı kaldırıldı. [CLS] ve [PAD] hâlâ dışarıda.
**Why:** Önceki fix ([SEP] dahil tüm special token'ları çıkarma) garbled output'u %99.8'e çıkardı — model sequence'ı nerede keseceğini bilemedi, 32 pozisyonu rastgele doldurdu. [SEP] training'de sep_weight=5x NLL ile güçlü sinyal aldı; vocab'da olunca model onu doğru pozisyona koyar, decode_token orada keser → kısa temiz cevap. Training gerektirmez.

---

## 2026-06-07 — sample_vqa_GPU: output dosyası append → write moduna alındı

**What:** `sample_vqa_GPU.py`: `fout = open(out_path, 'a')` döngü içinden kaldırıldı. Döngü öncesi `open(out_path, 'w')` ile bir kez açılıyor, her batch'te `flush()` yapılıyor, döngü sonrası kapatılıyor.
**Why:** Append modunda aynı checkpoint iki kez çalıştırılınca eski+yeni satırlar aynı dosyada birikiyor, analiz bozuluyor. Write modu her sampling'de dosyayı sıfırlar; Drive'da manuel silme gerekmez.

---

## 2026-06-07 — answer_vocab'dan special token'lar çıkarıldı (boş cevap fix)

**What:** `sample_vqa_GPU.py`: `answer_vocab_set`'e `[CLS]`/`[SEP]`/`[PAD]` eklenmesi kaldırıldı; bunun yerine `discard()` ile aktif olarak dışlanıyor.
**Why:** 180k analizinde boş cevap oranı %34→%40.5'e çıktı. Sebep: special token'lar answer vocabulary'deyken model confidence arttıkça `[SEP]`/`[PAD]` embedding'ine collapse ediyor, `decode_token` [SEP]'de kesiyor → boş string. Training gerektirmez, sadece yeniden sampling yeterli.

---

## 2026-06-06 — RESUME_CHECKPOINT 40k'ya set edildi

**What:** `notebooks/run_diffuvqa_colab.ipynb` config hücresinde `RESUME_CHECKPOINT = ""` → `RESUME_CHECKPOINT = f"{CHECKPOINT_PATH}/ema_0.9999_040000.pt"`.
**Why:** 40k checkpoint'ten training'e devam edilecek. 40k analizi: rounding agreement 0.11→0.788 (CIGN fix çalışıyor), garbled output %72→%31. EM/BLEU henüz düşük ama 40k'da normal — 100k'da tam değerlendirme yapılacak.

---

## 2026-06-06 — N_SAMPLES 5→1: majority voting devre dışı

**What:** `notebooks/run_diffuvqa_colab.ipynb` inference config: `N_SAMPLES = 5` → `N_SAMPLES = 1`.
**Why:** Model henüz erken aşamada tutarsız cevaplar üretiyor; 5 sampling de farklı saçma sonuç verince majority vote rastgele seçiyor, fayda sağlamıyor. Sampling ~5x hızlanır. Model olgunlaştıkça (yüksek adımlarda) tekrar artırılabilir.

---

## 2026-06-05 — `get_extended_attention_mask` deprecated API kaldırıldı

**What:** `vqa_model.py` satır 124: `self.bert.get_extended_attention_mask(q_mask, q_input_shape)` → `(1.0 - q_mask[:, None, None, :].float()) * -10000.0`. `q_input_shape` dead variable da silindi.
**Why:** `transformers v5.9.0`'da deprecation uyarısı veriyor; v5.12.0'da kaldırılacak → training crash eder. Matematiksel çıktı aynı: [0,1] mask'i BERT encoder'ın beklediği `[batch,1,1,seq_len]` additive mask'e çeviriyor. Training'i etkilemez, resume'da da sorunsuz çalışır.

---

## 2026-06-05 — Thesis Reports + notebook outputs main'e taşındı; eval/dataset fix'leri

**What:**
1. `Thesis Reports/` klasörü (tex, pdf, docx, png) pubmedbert branch'inden main'e taşındı.
2. `notebooks/`: 100k/300k/500k sampling JSONL'leri, `progress.csv`, eval CSV main'e eklendi.
3. `eval_DiffuVQA.py`: `--file` argümanı eklendi (tek JSONL dosyasını evaluate etmek için `--folder` scan'ını bypass eder).
4. `eval_DiffuVQA.py`: BERTScore OverflowError fix — `sent_encode` monkey-patch + 128 kelime truncate (deberta-xlarge-mnli uzun metinlerde token limit aşımı veriyordu).
5. Branch politikası belirlendi: bundan sonra tüm değişiklikler **sadece `main`'de** yapılır; pubmedbert/biobert'e taşıma gerekirse ilgili kişiye prompt verilir.
**Why:** Dosyalar yanlışlıkla pubmedbert branch'inde birikmiş, main'de yoktu. BERTScore fix pubmedbert'te vardı ama main'e alınmamıştı.

---

## 2026-06-05 — clamp geri alındı: mevcut checkpoint ile uyumsuz

**What:** `sample_vqa_GPU.py`: `clamp_step` 50→0, `clamp_first` False→True. Notebook sample-cell'de `--clamp_step 0`.
**Why:** N_SAMPLES=5 + yeni clamp değerleriyle sampling yapıldı, metrikler geriledi (EM %2.55→%1.26, BLEU %6.3→%4.5). Model 500k adım boyunca `clamp_first=True, clamp_step=0` ile eğitilmişti; inference'ta farklı rounding davranışı uyumsuzluk yarattı. `clamp_first=False + clamp_step=50` sıfırdan training için doğru — mevcut checkpoint için değil.
**Durum:** `##` wordpiece filtresi ve `-inf` fallback korundu; sadece clamp eski haline döndü.

---

## 2026-06-05 — 500k analizi sonrası kritik fix'ler: sıfırdan training

**What:**
1. `gaussian_diffusion.py` satır 686: CIGN `f` vektörü düzeltildi. `[ddpm_input_pre, ddpm_input_pre]` (sadece görüntü, iki kez) → `[ddpm_input_pre, ans_emb_pre]` (görüntü + CVAE ön-cevap tahmini). Soru bilgisi artık ileri difüzyon sürecine giriyor.
2. `sample_vqa_GPU.py`: `clamp_first=False` + `clamp_step=50` — sıfırdan training için ayarlandı, mevcut checkpoint'te geri alındı (yukarıdaki karar).
3. `sample_vqa_GPU.py`: `##` wordpiece filtresi kaldırıldı — `colonoscopy` gibi kelimeler artık subword token'larıyla üretilebilir.
4. `sample_vqa_GPU.py`: masked_logits tamamen `-inf` olduğunda unmasked logits fallback eklendi — boş cevap oranını düşürmek için.
5. Notebook `evaluate_and_export_csv`: `yes_no_accuracy` fix — Kvasir-VQA'da `answer_type` alanı olmadığından ref string `yes`/`no` ise `CLOSED`, değilse `OPEN` olarak çıkarım yapılıyor.
**Why:** 500k sampling analizi: EM %2.55, BLEU %6.3, boş cevap %20.9. CIGN'in yanlış `f` vektörüyle çalışması (soru bilgisi yok) ana performans düşüklüğünün kaynağı. Sıfırdan training gerekiyor.
**Sıfırdan training konfigürasyonu:** CIGN fix + clamp_first=False + clamp_step=50 + SEP anchor kaldırılmış + M-3 fix — tüm mimari değişiklikler aktif.

---

## 2026-06-04 — 4 bug fix: image path cache, SEP_ID, sampler state, vocab mask

**What:**
1. `vqa_datasets.py` `ImageDataset`: `load_image_path()` artık `__init__`'de bir kez çağrılıyor, `_image_paths` önbelleği oluşturuluyor. Her `__getitem__`'da O(N) path listesi yeniden oluşturuluyordu.
2. `vqa_datasets.py` `merge_and_mask`: `SEP_ID = 102` hardcoded → `vocab_dict.tokenizer.sep_token_id` — biobert/pubmedbert gibi farklı tokenizer'larda SEP ID değişebilir.
3. `train_util.py`: `LossAwareSampler` state (`_loss_history`, `_loss_counts`) artık `sampler_XXXXXX.pt` olarak checkpoint'e kaydediliyor ve resume'da geri yükleniyor. Her resume'da sampler ~25k adım warm-up kaybı önlendi.
4. `sample_vqa_GPU.py`: `answer_mask_bool` `tokenizer.vocab_size` → `args.vocab_size` — model lm_head çıktı boyutuyla tutarlı.
**Why:** Code review bulguları. Tüm değişiklikler training restart gerektirmiyor.

---

## 2026-06-04 — config: batch_size 20→4, eval_interval 1000→5000

**What:** `diffuvqa/config.json`: `batch_size` 20→4, `eval_interval` 1000→5000.
**Why:** `batch_size=20` notebook'ta `--batch_size 4` ile override ediliyordu ama config.json'da 20 kalıyordu — tutarsızlık ve eval dataloader da 20 kullanıyordu. `eval_interval=1000` her 1000 adımda tam CLIP+BERT forward pass yapıyordu, gereksiz yavaşlama.

---

## 2026-06-03 — train_util: bf.exists fallback, corrupt opt guard, _is_real guard

**What:**
1. `_load_and_sync_parameters`: `bf.exists()` → `bf.exists() or os.path.exists()` fallback. Colab Drive FUSE'da `bf.exists()` bazen `False` döner → model sıfırdan başlıyordu, sessizce. Checkpoint varsa-yoksa `FileNotFoundError` fırlatıyor. `### Resuming from step X` logu eklendi.
2. `_load_optimizer_state`: Aynı `bf.exists` fallback + `try/except` guard — kesinti sırasında yarım yazılmış `opt_*.pt` dosyası artık training'i çöktürmüyor, "starting fresh" logu yazılıyor.
3. `_is_real` guard: `resume_checkpoint` değeri `"none"`, `"false"` veya boş string ise checkpoint yok sayılıyor — Biobert branch'teki fix ile senkronize edildi.
**Why:** Biobert'te training 0'dan başladığı halde tqdm 0'dan gösteriyordu, loss 6.9'dan başlıyordu. `bf.exists()` Drive FUSE path'lerinde güvenilmez. Artık `FileNotFoundError` ile anında hata veriliyor.

---

## 2026-06-02 — SEP anchor kaldırıldı: sıfırdan training

**What:**
1. `diffuvqa/vqa_datasets.py` `merge_and_mask`: `mask_a` içinde SEP (tok==102) için `mask=0` (anchored) → tüm answer token'ları `mask=1` (noised). Model artık SEP'i noise'dan denoise etmeyi öğreniyor.
2. `sample_vqa_GPU.py`: SEP embedding injection bloğu kaldırıldı (x_start'taki SEP pozisyonuna `model_emb.weight[102]` yerleştirme). Tüm answer pozisyonları saf noise'dan başlıyor — training ile tutarlı.
**Why:** D1 (en güçlü mimari bulgu): SEP anchor mekanizması training–inference tutarsızlığı yaratıyordu. Training'de SEP mask=0 ile clean x_start'tan görülüyor, hiç noise altında denoise öğrenilmiyor. Inference'ta SEP noise'dan üretilmek zorunda — model bu problemi hiç görmemişti → 0% SEP üretimi. Sıfırdan training ile bu tutarsızlık ortadan kalktı.
**Not:** Bu değişiklikle birlikte mevcut checkpoint'ler geçersiz — SEP anchor varsayımına göre eğitilmişler. Sıfırdan eğitim gerekiyor.

---

## 2026-06-02 — Loss/grad/LR düzeltmeleri: 350k checkpoint'ten devam

**What:**
1. `gaussian_diffusion.py` NLL ağırlığı `2.0 * nll + 2.0 * decoder_nll` → `1.0 * nll + 1.0 * decoder_nll`. Grad norm clip'ten çıkarmak için.
2. `gaussian_diffusion.py` `_token_discrete_loss` `sep_weight` parametresi eklendi. `decoder_nll` çağrısında `sep_weight=1.0` (x_start clean, trivial signal), `terms["nll"]` çağrısında `sep_weight=5.0` (model tahmininden gerçek gradient). Önceki 3x ağırlık her iki çağrıda da eşit uygulanıyordu — decoder_nll'deki SEP near-zero olduğundan 3x sıfırın 3 katıydı.
3. `diffuvqa/config.json` `learning_steps` 500k → 750k. LR annealing'in 500k'da 0'a inmesi önleniyor; 350k'da LR ≈ 5.3e-6 yerine ≈ 7.0e-6 oluyor, kalan adımlarda gradient'ler daha anlamlı.
4. `diffuvqa/config.json` `gradient_clipping` 0.5 → 0.75. NLL ağırlığı düşünce toplam loss magnitude azalıyor; 0.5 artık çok agresif.
5. `train_util.py` `pre_answer_loss` gate lineer → cosine: `max(0, 1 - step/150k)` → `0.5*(1 + cos(π*min(step,150k)/150k))`. Sıfır eğim her iki uçta; lineer formülde ~130k'daki abrupt cliff yok.
6. `diffuvqa/vqa_model.py` `get_logits` `logits_mode=2` reshape bug düzeltildi: `view(vocab, bsz, seqlen)` yanlış shape'e yol açıyordu → `(bsz, seqlen, vocab)` şeklinde doğru yeniden şekillendirildi. `logits_mode=1` (aktif) etkilenmedi.
**Why:** Kod inceleme analizi: NLL terimleri toplam loss'un ~%90'ını oluşturuyor, grad norm sürekli 0.5 clip'te → SEP pathway ve vision coupling güncellenemiyor. 350k'dan devam eden mevcut training'e uygulanabilir, sıfırdan başlamayı gerektirmiyor.

---

## 2026-06-01 — M-3 mimari fix: sıfırdan training başlatılırken uygulanacak

**What:** `vqa_model.py` `feature_fusion.forward()`:
- `pre_simu_answer_feats = self.cvae(question_emb + image_feats)` → `self.cvae(question_feats + image_feats)`
- `f = alpha*f4 + beta*image_feats + theta*(question_feats + question_emb)` → `theta*question_feats`

**Why uygulanmadı:** Gerekmiyor — main branch'te zaten uygulanmış (`5dfd4be`, 2026-06-02). Satır 148 `question_feats`, satır 156 `theta * question_feats`. DECISIONS'daki "bekliyor" notu hataydı.
**Durum:** ✅ Tamamlandı.
**Dosya:** `diffuvqa/vqa_model.py` satır 148 ve 156.

---

## 2026-06-01 — SEP anchor: sıfırdan training kararı

**What:**
1. `vqa_datasets.py` `merge_and_mask`: SEP token (id=102) answer pozisyonlarında artık `mask=0` (anchored, noise eklenmez). Önceki haliyle tüm answer pozisyonları `mask=1` (noised) idi — SEP dahil. Model SEP'in nerede olduğunu hiç bilmiyordu.
2. `gaussian_diffusion.py` `_token_discrete_loss`: SEP loss ayrı hesaplanıyor, sequence normalizasyonundan çıkarıldı. `total = content_nll + 3.0 * sep_loss`. Önceki haliyle SEP loss sequence ortalamasına gömülüyordu.
3. `sample_vqa_GPU.py`: Sampling başlangıcında SEP embedding, x_start'taki SEP pozisyonuna yerleştiriliyor. Training mask convention ile tutarlı.
**Why:** 400k adım sonunda [SEP] üretim oranı %0. Root cause: tüm answer pozisyonları (SEP dahil) saf noise'dan başlıyordu. Model SEP'in hangi pozisyonda olduğunu öğrenemiyor, loss sinyali sequence ortalamasında kayboluyor. SEP anchor ile model her zaman SEP'in doğru pozisyonunu görüyor. Sıfırdan training başlatılıyor.

---

## 2026-06-01 — Derin kod incelemesi: 6 bug düzeltildi

**What:**
1. `weight_decay` 0.0→0.01: `config.json` ve `AdamW` çağrıları. Biobert 500k confidence collapse (0.50→0.14) root cause: tied word_embedding 6 gradient kaynağı altında L2 kısıtı olmadan drift ediyordu.
2. `_anneal_lr` LR floor: `max(lr, 0.0)` — negatif LR koruması. `log_step`'e `lr` loglama eklendi.
3. Test dataloader: `shuffle=False, drop_last=False` — metrikler artık reproducible; önceki tüm exact match rakamları farklı sample population üzerindeydi.
4. `input_a_id` model_kwargs'tan `.pop()` ile temizlendi — `_WrappedModel` maskeliyordu ama label leakage riski vardı.
5. `pre_answer_loss` gate: 150k adım sonra 0'a iniyor. Late-training'de near-zero katkı ama embedding drift'inde gradient spike riski var.
**Why:** Biobert 500k analizi root cause araştırması. weight_decay=0.0 + 6 loss terimi + tied weights = embedding drift garantisi. Bu değişikliklerle embedding collapse riskini minimize ettik.

---

## 2026-05-31 — Decode fix: lm_head masked logits, l2_argmin kaldırıldı

**What:** `sample_vqa_GPU.py` decode adımında `l2_argmin` → `masked_logits.argmax()`. `lm_head` logitleri `answer_vocab_ids` dışındaki token'lara `-inf` uygulanarak maskeleniyor, sonra argmax alınıyor. `l2_argmin` tamamen kaldırıldı. Confidence `masked_logits` üzerinden softmax max olarak güncellendi.
**Why:** `l2_argmin` answer vocab subspace'i üzerinden hesaplanıyordu ama bu subspace training data'daki tüm answer token'larını içeriyordu — "clip", "barrett", "snare", "injection" gibi gürültülü token'lar da vardı. 200k sampling'de exact match %0.74 → %0.00'a düştü, gen_length 7 → 20 kelimeye çıktı. `masked_logits.argmax()` ile lm_head'in öğrendiği dağılım korunurken answer vocab kısıtlaması da sağlanıyor.

---

## 2026-05-31 — Biobert branch analizi: 4 bug düzeltildi

**What:**
1. `sample_vqa_GPU.py`: final decode `cands.indices` (full 30K lm_head argmax) → `l2_argmin` (answer_vocab_ids üzerinden). Denoising adımlarında answer vocab kısıtlaması uygulanıyordu ama son decode adımı bunu görmezden geliyordu — "anaphylaxis", "zika", "belgium" gibi tokenlar çıkmasının sebebi buydu.
2. `basic_utils.py`: `convert_tokens_to_string` sonrası başta kalan `##` strip edildi; ∑, π, ∂ gibi non-ASCII biomedical semboller regex ile temizlendi.
3. `eval_DiffuVQA.py`: `question="none"` veya `reference_answer="nan/none/""` olan corrupt Kvasir-VQA satırları `continue` ile atlandı.
**Why:** Error #1 near-zero BLEU/F1'ın birincil sebebiydi — denoising ile decode mekanizması tamamen ayrıştıydı. Error #2-3 metrik hesaplamalarını kirleten post-processing sorunları. Error #4 corrupt veriyle corpus ortalaması bozuluyordu.

---

## 2026-05-30 — [SEP] token 2x ağırlık: sequence boundary sinyali

**What:** `_token_discrete_loss`'ta CrossEntropyLoss çıktısına `sep_weight = where(input_ids==102, 2.0, 1.0)` çarpılıyor. Hem `decoder_nll` hem `terms["nll"]` bu ağırlığı alıyor (her ikisi de `_token_discrete_loss` üzerinden geçiyor). Mask uygulamasından önce yapılıyor.
**Why:** 400k adım boyunca [SEP] üretim oranı %0 kaldı. Model cevabın nerede biteceğini hiç öğrenemedi — gen_length kontrolsüz büyüdü, BLEU/ROUGE düştü. Bunun sebebi loss'ta [SEP] pozisyonuna özel sinyal olmamasıydı; diğer token'larla eşit ağırlıkla işleniyordu. 2x ağırlık ile model bu pozisyona daha fazla odaklanacak. Checkpoint'ten resume ile uygulandı — mimari değişiklik yok, sadece loss sinyali eklendi.

---

## 2026-05-30 — lm_head argmax rounding (L2 KNN yerine)

**What:** `denoised_fn_round` artık `get_logits` parametresi alıyor. Verildiğinde token seçimi L2 KNN yerine `lm_head(embedding).argmax()` ile yapılıyor. `answer_vocab_ids` kısıtlaması varsa argmax önce o subspace'e uygulanıyor. `get_logits=None` durumunda L2 KNN fallback çalışıyor (geriye dönük uyumlu). `sample_vqa_GPU.py`'de `model.get_logits` geçiriliyor.
**Why:** L2 KNN ve lm_head iki farklı metrik uzayında çalışıyordu — model NLL'i lm_head üzerinden öğreniyor ama sampling L2 mesafesiyle token seçiyordu. Bu tutarsızlık rounding_agreement'ın 0.12'de kalmasına yol açıyordu. lm_head argmax ile training ve sampling aynı mekanizmayı kullanıyor.

---

## 2026-05-30 — Sampling kalitesi: 4 inference-time düzeltme

**What:**
1. `basic_utils.py` `decode_token`: `tokenizer.decode()` → `convert_ids_to_tokens` + `convert_tokens_to_string`. `##` wordpiece token'ları artık birleşiyor (col+##on+##oscopy → colonoscopy).
2. `sample_vqa_GPU.py` `clamp_step` default: 0 → 200. Rounding tüm 2500 adım yerine sadece son 200 adımda aktif — erken denoising'de serbest continuous keşif.
3. `rounding.py` + `sample_vqa_GPU.py` answer vocab kısıtlama: KNN 30522 token yerine dataset'teki answer position token'larıyla kısıtlı (~birkaç yüz token). Pattern A (%72 garbled output) için ana fix.
4. `sample_vqa_GPU.py` confidence: lm_head softmax → `1/(1+mean_L2_dist)`. lm_head ile L2 rounding %53-72 pozisyonda anlaşmazlık içindeydi, confidence skoru anlamsızdı.
**Why:** 400k sampling analizinde rounding_agreement 0.11'e düşmüş, [SEP] hiç üretilmemiş, %72 garbled output gözlemlendi. Tüm sorunlar inference-time — eğitimi etkilemiyor, mevcut ve gelecek checkpoint'lerde hemen etkili.

---

## 2026-05-30 — Code review: 3 kritik bug düzeltildi

**What:**
1. `train_util.py` `forward_backward` ve `forward_only`: `del cond['image_name']` microbatch for-loop içindeydi → ikinci iterasyonda `KeyError`. `loss.backward()` loop dışındaydı → sadece son microbatch'in gradyanı geri yayılıyordu, diğer microbatch forward pass'ları boşa gidiyordu. Düzeltme: `image_name` loop öncesinde `pop`, `backward()` loop içine alındı, `micro_image`/`micro_cond` değişkenleri ayrıldı.
2. `gaussian_diffusion.py` `_token_discrete_loss`: `mask.sum(dim=-1)` sıfır olduğunda `0/0 = NaN` → tüm batch loss'unu zehirliyordu. Düzeltme: `.clamp(min=1)` guard eklendi.
3. `eval_DiffuVQA.py`: sonuçlar hardcoded `'ema_0.9999_300000.pt.samples.jsonl'`'e yazılıyordu → çoklu dosya değerlendirmesinde her iterasyon öncekini eziyordu. Boş JSONL'de `acc/cnt` ZeroDivisionError. Düzeltme: çıktı `{basename}_eval.json`, boş dosya için `continue` guard.
**Why:** Microbatch bug'ı `microbatch=0` (default) durumunda tetiklenmediğinden gizli kalmıştı — loop tek iterasyonda tamamlanıyor. `--microbatch` CLI ile küçük değer verilseydi training sessizce yanlış gradyanlarla devam edecekti. NaN bug'ı NLL 2x ağırlık sonrası daha kritik hale geldi.

---

## 2026-05-30 — Loss ağırlıkları yeniden ayarlandı (embedding collapse düzeltmesi)

**What:**
1. `reg` `time_weight`: `_extract_into_tensor(1 - alphabar(t))` → `th.ones(...)` (sabit 1.0)
2. `lambda_reg` (`sim` tipi): `0.1` → `0.5`
3. Loss formülü: `terms["nll"]` ve `decoder_nll` katsayısı `1.0` → `2.0`

**Why:** 400k adım sonunda rounding_agreement 0.22→0.11'e düştü, [SEP] hiç üretilmedi, exact match %0.26'ya geriledi. Üç kök neden:
- `time_weight = 1 - alphabar(t)` → t küçüldükçe reg sıfırlanıyor; vocabulary snap tam t→0'da gerekli ama orada reg kapalı.
- `lambda_reg=0.1` → MSE+NLL ~0.04'e inince reg ~0.00026 kalıyor, tamamen etkisiz.
- NLL 1x → `lm_head` çıktısı ile L2 rounding arasındaki gap kapanmıyor; 2x ile vocabulary alignment sinyali güçlendiriliyor.

**Etki:** 50-100k adımda `reg` loss'un sabit kalıp kalmaması ve `rounding_agreement`'ın artmaya başlayıp başlamaması izlenecek. İlk anlamlı kontrol: 100k sampling.

---

## 2026-05-29 — `train.py` progress.csv corruption on resume düzeltildi

**What:** `logger.configure()` çağrısı CSV trim bloğundan **önce** yapılıyordu. `CSVOutputFormat` dosyayı `a+t` modunda açıp eski header'ı `self.keys`'e yüklüyordu. Trim sonra dosyayı `w` modunda yeniden yazıyordu ama logger'ın bellekteki state stale kalıyordu. İlk `dumpkvs()`'da `writekvs` `extra_keys` görüp header'ı tekrar yazıyor, eski satırları bozuyordu. Düzeltme: trim → sonra `logger.configure()` sırasına alındı.
**Why:** Resume'da progress.csv sıfırdan bozuk yazılıyordu — eski adımların üzerine yanlış veriler geliyordu.

---

## 2026-05-29 — `eval_DiffuVQA.py` iki bug düzeltildi

**What:**
1. `with open(path)` bloğu `for path in files:` döngüsünün dışına kaçmıştı (indentation hatası) — birden fazla `.jsonl` varsa sadece son dosya değerlendiriliyordu, ama DeBERTa inference her dosya için çağrılıyordu.
2. `create_argparser().parse_args()` → `parse_known_args()[0]` — `--folder` argümanı `create_argparser`'a bilinmeyen arg olarak geldiğinde `unrecognized arguments` hatası veriyordu.
**Why:** Evaluation hücresi `--folder` argümanıyla çağrılınca crash, birden fazla dosya varsa yanlış sonuç.

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
