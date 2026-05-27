# SETUP_COMMON.md — DiffuVQA Branch Kurulum Kılavuzu

Bu dosya, `https://github.com/cloneiq/DiffuVQA.git` reposundan yeni bir branch açıldığında
uygulanması gereken **tüm ortak fix'leri** listeler. Encoder-specific değişiklikler için
`SETUP_BIOBERT.md` veya `SETUP_PUBMEDBERT.md` dosyasına bak.

---

## 0. Başlamadan Önce

```bash
git clone https://github.com/cloneiq/DiffuVQA.git
cd DiffuVQA
git checkout -b biobert   # veya pubmedbert
```

Aşağıdaki değişiklikleri sırayla uygula. Her madde hangi dosyada ne değişeceğini açıklar.

---

## 1. Bağımlılıklar

**`requirements_colab.txt` oluştur** (torch/torchvision/wandb hariç):
```
timm
open_clip_torch
openpyxl
pycocoevalcap
transformers>=4.36.0
datasets
pillow
pandas
bert-score
scikit-learn
```

---

## 2. `train.py`

- `import wandb`, `wandb.init()`, `wandb.config.update()`, `WANDB_MODE` bloğunu sil
- `os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'` satırını sil (BERT yüklemesini kırıyor)
- `parser.parse_args()` → `parser.parse_known_args()[0]` (Colab kernel ekstra argv enjekte ediyor)
- `logger.configure()` → `logger.configure(dir=args.checkpoint_path, format_strs=["log", "csv"], append_csv=is_resume)` şeklinde güncelle:
  ```python
  is_resume = bool(getattr(args, 'resume_checkpoint', ''))
  logger.configure(dir=args.checkpoint_path, format_strs=["log", "csv"], append_csv=is_resume)
  ```

---

## 3. `diffuvqa/utils/logger.py`

- `import wandb` satırını sil
- `CSVOutputFormat.__init__` → `append` parametresi ekle, sıfırdan açışta `"w+t"`, resume'da `"a+t"`:
  ```python
  class CSVOutputFormat(KVWriter):
      def __init__(self, filename, append=False):
          file_exists = os.path.isfile(filename) and append
          self.file = open(filename, "a+t" if append else "w+t")
          self.keys = []
          self.sep = ","
          if file_exists:
              self.file.seek(0)
              first_line = self.file.readline()
              if first_line:
                  self.keys = first_line.strip().split(self.sep)
              self.file.seek(0, 2)
  ```
- `make_output_format(format, ev_dir, log_suffix="")` → `append_csv=False` parametresi ekle, `CSVOutputFormat(..., append=append_csv)` geç
- `configure(...)` → `append_csv=False` parametresi ekle, `make_output_format`'a ilet

---

## 4. `train_util.py`

### 4a. `dist.is_initialized()` guard
`dist.get_world_size()` çağrısından önce:
```python
if dist.is_initialized():
    world_size = dist.get_world_size()
else:
    world_size = 1
```

### 4b. `_SingleGPUDDP` wrapper
Tek GPU'da `model.model.module.*` erişimi için (gaussian_diffusion.py DDP convention varsayıyor):
```python
class _SingleGPUDDP(th.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
```
`DDP` yerine tek GPU'da `_SingleGPUDDP(model)` kullan.

### 4c. `_load_and_sync_parameters` düzelt
Orijinal `pass` ile boş — checkpoint yüklemesi çalışmıyor:
```python
def _load_and_sync_parameters(self):
    resume_checkpoint = self.resume_checkpoint
    if resume_checkpoint:
        self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
        state_dict = dist_util.load_state_dict(resume_checkpoint, map_location="cpu")
        self.model.load_state_dict(state_dict)
```

### 4d. `_load_ema_parameters` dist guard
```python
# dist.get_rank() → crash on single GPU
rank = dist.get_rank() if dist.is_initialized() else 0
# dist_util.sync_params(ema_params) → guard:
if dist.is_initialized():
    dist_util.sync_params(ema_params)
```

### 4e. tqdm progress bar
`run_loop`'taki `print(f'Epoch/Step/loss...')` satırlarını kaldır, yerine:
```python
from tqdm import tqdm
pbar = tqdm(total=self.lr_anneal_steps, initial=self.step + self.resume_step, desc="Training")
# her step sonunda:
pbar.set_postfix(loss=f"{self._last_loss:.4f}")
pbar.update(1)
```
`forward_backward`'da `self._last_loss = loss.item()` set et.

---

## 5. `diffuvqa/gaussian_diffusion.py`

`p_sample_loop` ve `ddim_sample_loop` içindeki `print(i)` satırlarını sil (her timestep basılıyor):
```python
# SİL:
for i in indices:
    print(i)   # ← bu satırı sil
    t = th.tensor(...)
```

---

## 6. `diffuvqa/vqa_model.py`

`get_extended_attention_mask` çağrısından `device=` kwarg'ını kaldır:
```python
# ÖNCE:
self.bert.get_extended_attention_mask(q_mask, q_input_shape, device=q_ids.device)
# SONRA:
self.bert.get_extended_attention_mask(q_mask, q_input_shape)
```

---

## 7. `diffuvqa/language_encoders/bert_model.py`

transformers>=4.36 import uyumluluğu — üç farklı try/except bloğu:

```python
# 1. file_utils → utils
try:
    from transformers.utils import (ModelOutput, add_start_docstrings, ...)
except ImportError:
    from transformers.file_utils import (ModelOutput, add_start_docstrings, ...)

# 2. apply_chunking_to_forward + prune_linear_layer
try:
    from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer
except ImportError:
    from transformers.modeling_utils import apply_chunking_to_forward, prune_linear_layer

# 3. find_pruneable_heads_and_indices (transformers>=4.40'ta pytorch_utils'den kaldırıldı)
try:
    from transformers.pytorch_utils import find_pruneable_heads_and_indices
except ImportError:
    try:
        from transformers.modeling_utils import find_pruneable_heads_and_indices
    except ImportError:
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head -= sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index
```

---

## 8. `diffuvqa/utils/answer_pre.py` oluştur

Orijinal repoda eksik — şu içerikle ekle:
```python
import difflib

def find_most_similar_answers(generated, candidates):
    best, best_score = candidates[0], 0
    for c in candidates:
        s = difflib.SequenceMatcher(None, generated, c).ratio()
        if s > best_score:
            best, best_score = c, s
    return best
```

---

## 9. `basic_utils.py`

- `os.environ['HF_ENDPOINT']` set eden satırı sil
- `myTokenizer.__init__` ve `create_model_and_diffusion` içinde BERT yüklemesi sırasında HF_ENDPOINT bypass:
  ```python
  _hf = os.environ.pop("HF_ENDPOINT", None)
  # ... from_pretrained çağrısı ...
  if _hf: os.environ["HF_ENDPOINT"] = _hf
  ```

---

## 10. `diffuvqa/vqa_datasets.py`

`load_image_path` içinde çift prefix hatası:
```python
# ÖNCE:
f'{self.image_root}/{image_name}'
# SONRA:
os.path.normpath(os.path.join(self.image_root, image_name))
```

---

## 11. `sample_vqa_GPU.py`

- `os.environ['HF_ENDPOINT']` set eden satırı sil
- `use_noising_f` AttributeError:
  ```python
  # ÖNCE:
  if args.use_noising_f:
  # SONRA:
  if getattr(args, 'use_noising_f', False):
  ```
- CLI argümanlarını `training_args.json` update'inden koru:
  ```python
  _keep = {k: args.__dict__[k] for k in ('model_path', 'out_dir', 'step', 'split',
                                           'seed2', 'clamp_step', 'top_p',
                                           'data_dir', 'image_dir')
           if k in args.__dict__}
  args.__dict__.update(training_args)
  args.__dict__.update(_keep)
  ```
- Çıktı yolunu sadeleştir (`lr_dir/ema_step.jsonl`):
  ```python
  lr_dir = os.path.basename(os.path.split(args.model_path)[0])
  ckpt_name = os.path.splitext(os.path.basename(args.model_path))[0]
  out_dir = os.path.join(args.out_dir, lr_dir)
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, f"{ckpt_name}.jsonl")
  ```
- Her JSONL satırına `confidence` ve `avg_nn_l2` ekle (ayrıntı için `sample_vqa_GPU.py`'ye bak)
- tqdm progress bar ekle (sampling döngüsünü tqdm ile sar)

---

## 12. Notebook (`notebooks/run_diffuvqa_colab.ipynb`)

Config hücresinde:
- `TRAIN_BATCH_SIZE = 4` (eğitim)
- `SAMPLE_BATCH_SIZE = 64` (sampling)
- `RESUME_CHECKPOINT = ""` (sıfırdan) veya tam checkpoint yolu
- `DRIVE_PROJECT_PATH` branch adına göre ayarla (ör. `DiffuVQA-BioBERT`)

BERT cache hücresi encoder model adına göre güncelle (bkz. encoder-specific MD).

`evaluate_and_export_csv` fonksiyonunda `dataset_file`:
```python
_dataset_file = dataset_file or os.path.join(DRIVE_PROJECT_PATH, "datasets", DATASET, "test.jsonl")
```

---

## 13. Doğrulama

Tüm değişiklikler uygulandıktan sonra:
```bash
python train.py --checkpoint_path ./test_ckpt --dataset Kvasir_VQA \
  --data_dir datasets/Kvasir_VQA --image_dir datasets/Kvasir_VQA/imgs \
  --batch_size 4 --lr 1e-5 --learning_steps 10 --save_interval 10
```
Hata yoksa kurulum tamamdır.