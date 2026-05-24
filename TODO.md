# TODO — DiffuVQA Colab/A100 Compatibility

## Completed

- [x] Create `requirements_colab.txt` (torch/torchvision excluded, wandb excluded, timm/open_clip/openpyxl/pycocoevalcap added)
- [x] Remove `import wandb`, `wandb.init()`, `wandb.config.update()`, `WANDB_MODE` from `train.py`
- [x] Remove `import wandb` from `diffuvqa/utils/logger.py`
- [x] Fix `logger.configure()` in `train.py` → `dir=args.checkpoint_path, format_strs=["log", "csv"]`
- [x] Change `parser.parse_args()` → `parser.parse_known_args()[0]` in `train.py`
- [x] Add `dist.is_initialized()` guard before `dist.get_world_size()` in `train_util.py`
- [x] Verify `HF_ENDPOINT` already set in both `train.py` and `sample_vqa_GPU.py`
- [x] Create `notebooks/run_diffuvqa_colab.ipynb` (4 sections: Setup / Train / Sample / Eval)
- [x] Create `CLAUDE.md`
- [x] Create `DECISIONS.md`
- [x] Create `TODO.md`

- [x] Drive path'i `DiffuVQA-Original` olarak düzelt (DiffuVQA ile çakışmasın)
- [x] Notebook yapısını Bert branch formatına uyarla (Config hücresi + Türkçe sub-header'lar + dataset copy + CSV export + görselleştirme)

## Open (nice-to-have, not blocking)

- [ ] Test full training run on A100 Colab and confirm `progress.csv` writes correctly
- [ ] Add tqdm wrapper inside `run_loop` in `train_util.py` for epoch/step progress bar in Colab
- [ ] Confirm `pycocoevalcap` CIDEr metric works with current NLTK version in Colab
- [ ] Consider pinning `transformers` version in `requirements_colab.txt` to match `requirements.txt` (4.22.2)
- [ ] Add dataset download/symlink helpers to Section 1 of the notebook for SLAKE and Kvasir-VQA
