"""
Train a diffusion model on images.
"""
import os
# HF_ENDPOINT is NOT set here: hf-mirror doesn't serve bert-base-uncased,
# so setting it would break AutoTokenizer/BertModel.from_pretrained calls.
# Dataset downloads that need the mirror are handled in the notebook before train.py runs.

import argparse
import torch.nn as nn
import json, torch, os
import numpy as np
from diffuvqa.utils import dist_util, logger
from diffuvqa.vqa_datasets import load_data_vqa
from diffuvqa.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer
)
from train_util import TrainLoop
from transformers import set_seed

import sys
import os
from torchvision import transforms

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(current_dir)


def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser

def main():
    parser = create_argparser()
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.log(f"### Warning: ignoring unknown args: {unknown}")
    set_seed(args.seed)
    # dist_util.setup_dist()
    _rc = getattr(args, 'resume_checkpoint', '') or ''
    is_resume = bool(_rc) and _rc.lower() not in ('', 'none', 'false')

    _trim_msg = None
    if is_resume:
        import re, csv
        resume_step = 0
        m = re.search(r'(\d{6})', os.path.basename(_rc))
        if m:
            resume_step = int(m.group(1))
        csv_path = os.path.join(args.checkpoint_path, "progress.csv")
        if os.path.exists(csv_path) and resume_step > 0:
            with open(csv_path, 'r') as f:
                rows = list(csv.DictReader(f))
            def _step_val(r):
                try:
                    return int(r.get('step', 0))
                except (ValueError, TypeError):
                    return 0
            kept = [r for r in rows if _step_val(r) <= resume_step]
            if len(kept) < len(rows):
                fieldnames = [k for k in (rows[0].keys() if rows else []) if k is not None]
                cleaned = [{k: v for k, v in r.items() if k in fieldnames} for r in kept]
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(cleaned)
                _trim_msg = f"### Trimmed progress.csv to step {resume_step} ({len(rows) - len(kept)} rows removed)"

    logger.configure(dir=args.checkpoint_path, format_strs=["log", "csv"], append_csv=is_resume)
    if _trim_msg:
        logger.log(_trim_msg)

    logger.log("### Creating data loader...")

    tokenizer = load_tokenizer(args)
    model_weight, tokenizer = load_model_emb(args, tokenizer)
    transform = transforms.Compose([
        transforms.Resize((args.image_resolution, args.image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = load_data_vqa(batch_size=args.batch_size, seq_len=args.seq_len, args=args, model_emb=model_weight,
                         transform=transform, split="train", loaded_vocab=tokenizer)
    if args.valid:
        data_valid = load_data_vqa(batch_size=args.batch_size, seq_len=args.seq_len, args=args, model_emb=model_weight,
                               transform=transform, split='valid', loaded_vocab=tokenizer)
    else:
        data_valid = None


    print('#'*30, 'size of vocab', args.vocab_size)

    logger.log("### Creating model and diffusion...")
    print("use{}".format(args.model))
    model, diffusion = create_model_and_diffusion(args=args)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(dist_util.dev())

    # torch.compile: Triton kernel fusion on A100, ~20-30% speedup.
    # Disabled by default; set USE_TORCH_COMPILE=1 in notebook config to enable.
    if getattr(args, 'use_torch_compile', False):
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        model = torch.compile(model, mode="reduce-overhead")
        print("### torch.compile enabled (reduce-overhead mode)")

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logger.log("### Training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=0,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval,
        warmup_steps=getattr(args, 'warmup_steps', 2000),
        lr_min=getattr(args, 'lr_min', 5e-6),
        use_bf16=getattr(args, 'use_bf16', False),
    ).run_loop(args)

if __name__ == "__main__":
    main()
