"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os

# HF_ENDPOINT not set: hf-mirror doesn't serve bert-base-uncased weights.
import torch
import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from torchvision.transforms import transforms
from transformers import set_seed
from diffuvqa.rounding import denoised_fn_round
from diffuvqa.vqa_datasets import load_data_vqa
from diffuvqa.utils.question_classifier import classify_question, build_subtype_vocabs

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction as _SF

import time


def mbr_select(candidates: list) -> str:
    """Pick the candidate with highest average BLEU-1 against all other candidates."""
    if len(candidates) == 1:
        return candidates[0]
    smoothie = _SF().method4
    scores = []
    for i, hyp in enumerate(candidates):
        hyp_toks = hyp.lower().split()
        refs = [c.lower().split() for j, c in enumerate(candidates) if j != i]
        if not hyp_toks:
            scores.append(0.0)
            continue
        avg = sum(
            sentence_bleu([ref], hyp_toks, weights=(1.0,), smoothing_function=smoothie)
            for ref in refs
        ) / len(refs)
        scores.append(avg)
    return candidates[scores.index(max(scores))]
from diffuvqa.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer
)

torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def create_argparser():
    defaults = dict(model_path='', step=2500, out_dir='', top_p=0, n_samples=1)
    decode_defaults = dict(split='test', clamp_step=200, seed2=105, clip_denoised=False, clamp_first=True)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@th.no_grad()
def main():

    args = create_argparser().parse_args()

    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    # Preserve CLI-provided sampling args that training_args.json would overwrite
    _keep = {k: args.__dict__[k] for k in ('model_path', 'out_dir', 'step', 'split',
                                             'seed2', 'clamp_step', 'clamp_first', 'top_p',
                                             'data_dir', 'image_dir')
             if k in args.__dict__}
    args.__dict__.update(training_args)
    args.__dict__.update(_keep)

    num_steps = args.diffusion_steps

    betas =  betas_for_alpha_bar(num_steps, lambda t: 1 - np.sqrt(t + 0.0001),)
    alphas = 1 - betas  # α = 1 - β
    alphas = torch.from_numpy(alphas)
    
    alphas_prod = torch.cumprod(alphas, 0) 
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) 
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
  
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == \
           alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape == \
           one_minus_alphas_bar_sqrt.shape

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args=args)
    state_dict = torch.load(args.model_path, map_location="cuda")
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(th.device("cuda"))

    tokenizer = load_tokenizer(args)
    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size,
        embedding_dim=model.word_embedding.weight.shape[1],
        _weight=model.word_embedding.weight.clone().cuda()
    ).eval().requires_grad_(False)                                                            

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    transform = transforms.Compose([
        transforms.Resize((args.image_resolution, args.image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ## load data
    data_test = load_data_vqa(batch_size=args.batch_size, seq_len=args.seq_len, args=args, model_emb=model_emb.cpu(),
                               transform=transform, split=args.split, loaded_vocab=tokenizer, loop=False)

    start_t = time.time()

    # e.g. checkpoint: .../lr1e-05/ema_0.9999_200000.pt
    # output:          out_dir/lr1e-05/ema_0.9999_200000.jsonl
    lr_dir = os.path.basename(os.path.split(args.model_path)[0])
    ckpt_name = os.path.splitext(os.path.basename(args.model_path))[0]  # ema_0.9999_200000

    out_dir = os.path.join(args.out_dir, lr_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ckpt_name}.jsonl")

    print("out_path:", out_path)
    print("batch_size:", args.batch_size)
    all_text_data = []
    all_image_data = []

    try:
        for image, cond in data_test:
            cond['input_q_id'] = cond['input_q_id'].to(th.device("cuda"))
            cond['input_ids'] = cond['input_ids'].to(th.device("cuda"))
            all_text_data.append(cond)
            all_image_data.append(image.to(th.device("cuda")))
    except StopIteration:
        print('### End of reading iteration...')

    model_emb.to(th.device("cuda"))

    # Build answer vocabulary from the TRAIN split JSONL directly — avoids a
    # full load_data_vqa call (which tokenizes 47k samples with 8 workers).
    answer_vocab_set = set()
    train_jsonl = os.path.join(args.data_dir, 'train.jsonl')
    if os.path.exists(train_jsonl):
        with open(train_jsonl, 'r') as _f:
            for _line in _f:
                _ans = json.loads(_line).get('answer', '')
                if _ans:
                    _ids = tokenizer.tokenizer(
                        _ans, add_special_tokens=True,
                        padding='max_length', max_length=args.seq_len,
                        truncation=True
                    )['input_ids']
                    answer_vocab_set.update(_ids)
    else:
        # fallback: use test data already loaded
        for cond in all_text_data:
            ids = cond['input_a_id']
            if isinstance(ids, torch.Tensor):
                answer_vocab_set.update(ids.view(-1).tolist())
            else:
                for row in ids:
                    answer_vocab_set.update(row.tolist() if hasattr(row, 'tolist') else row)
    answer_vocab_set.discard(None)
    # Exclude [CLS] and [PAD] — they are not meaningful answer tokens.
    # [SEP] is intentionally kept: decode_token uses it as a stop marker.
    # Excluding [SEP] would cause every output to fill all 32 positions (AvgLen=32).
    for _sid in (tokenizer.tokenizer.cls_token_id,
                 tokenizer.tokenizer.pad_token_id):
        answer_vocab_set.discard(_sid)
    # Filter out ## wordpiece continuation tokens — they produce merged artifacts.
    answer_vocab_set = {
        tid for tid in answer_vocab_set
        if tid is not None and
        not tokenizer.tokenizer.convert_ids_to_tokens([tid])[0].startswith('##')
    }
    answer_vocab_ids = torch.tensor(sorted(answer_vocab_set),
                                    dtype=torch.long, device=th.device("cuda"))
    print(f"### Answer vocabulary size: {len(answer_vocab_ids)} / {tokenizer.vocab_size} tokens")

    # Pre-compute once: Y/N token IDs and per-subtype answer vocabs for OE routing
    yes_id = tokenizer.tokenizer.convert_tokens_to_ids('yes')
    no_id  = tokenizer.tokenizer.convert_tokens_to_ids('no')
    subtype_vocabs = build_subtype_vocabs(answer_vocab_ids, tokenizer,
                                          tokenizer.tokenizer.sep_token_id)
    print(f"### Question router — Y/N token IDs: yes={yes_id}, no={no_id}")
    print(f"### OE subtype vocabs: { {k: len(v) for k, v in subtype_vocabs.items()} }")

    text_iterator = iter(all_text_data)
    image_iterator = iter(all_image_data)

    total_batches = len(all_text_data)
    pbar = tqdm(zip(image_iterator, text_iterator), total=total_batches, desc="Sampling", unit="batch")
    fout = open(out_path, 'w')
    for image, cond in pbar:

        if not cond:
            continue                                           

        input_ids_x = cond.pop('input_ids').to(th.device("cuda"))
        input_ids_a = cond.pop('input_a_id').to(th.device("cuda"))
        input_emb = model.get_embeds(input_ids_a)
        # qid = cond.pop('qid')
        # print(qid)
        # img_id = cond.pop('img_id')
        # print(img_id)
        # print(input_ids_x)
        input_ids_mask = cond.pop('input_mask').to(th.device("cuda"))
        image_name = cond.pop('image_name')
        # print("input_ids_mask: ", input_ids_mask)
        # print("input_ids_mask.shape: ", input_ids_mask.shape)

        # x_start_mean, _ = model.get_ddpm_inputs_mask(image, cond)
        fuse_feats, _ = model.get_ddpm_input(image, cond)  
        f = torch.cat([fuse_feats, fuse_feats], dim=1)
        x_start = torch.cat([fuse_feats, input_emb], dim=1)
        # input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask

        # SEP anchor removed: all answer positions (including [SEP]) start from noise.
        # Model now learns to denoise [SEP] from scratch, consistent with training mask=1.
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(th.device("cuda"))

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap =1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps // args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], x_start.shape[1], x_start.shape[2])

        # answer_mask_bool sized by args.vocab_size (== model lm_head output dim)
        answer_mask_bool = th.zeros(args.vocab_size, dtype=th.bool, device=th.device("cuda"))
        answer_mask_bool[answer_vocab_ids] = True

        n_votes = args.n_samples
        batch_sz = x_start.shape[0]
        all_vote_candidates = [[] for _ in range(batch_sz)]

        # Classify each question in the batch (question = first seq_len tokens of input_ids_x)
        q_texts = [tokenizer.decode_token(input_ids_x[i, :args.seq_len].cpu())
                   for i in range(batch_sz)]
        q_types = [classify_question(qt) for qt in q_texts]
        logits = None
        masked_logits = None
        decode_ids = None

        for _vote_idx in range(n_votes):
            noise = th.randn_like(x_start)
            if getattr(args, "use_noising_f", False):
                noise = alphas_bar_sqrt[num_steps - 1] * f + one_minus_alphas_bar_sqrt[num_steps - 1] * noise
            x_noised = th.where(input_ids_mask == 0, x_start, noise)

            samples = sample_fn(
                model,
                sample_shape,
                noise=x_noised,
                clip_denoised=args.clip_denoised,
                denoised_fn=partial(denoised_fn_round, args, model_emb),
                model_kwargs=model_kwargs,
                top_p=args.top_p,
                clamp_step=args.clamp_step,
                clamp_first=args.clamp_first,
                mask=input_ids_mask,
                x_start=x_start,
                gap=step_gap
            )

            sample = samples[-1]
            a_shape = sample.size(1) // 2
            sample = sample[:, a_shape:, :]
            logits = model.get_logits(sample)  # bsz, seqlen, vocab

            # Decode: lm_head logits masked to answer vocab, then argmax.
            # -inf mask pushes non-answer tokens out of softmax competition.
            masked_logits = logits.clone()
            masked_logits[:, :, ~answer_mask_bool] = float('-inf')
            decode_ids = masked_logits.argmax(dim=-1)  # bsz, seqlen

            for i, seq in enumerate(decode_ids):
                tokens = tokenizer.decode_token(seq.to(th.device("cpu")))
                all_vote_candidates[i].append(tokens)

        # Majority vote: most-frequent decoded string wins per sequence in batch.
        # Confidence and rounding_agreement are taken from the last sample run.
        cands = th.topk(logits, k=1, dim=-1)
        confidence_per_seq = th.softmax(masked_logits, dim=-1).max(dim=-1).values.mean(dim=-1)
        logit_argmax = cands.indices.squeeze(-1)
        agreement_per_seq = (decode_ids == logit_argmax).float().mean(dim=-1)

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []
        confidence_lst = []
        rounding_agreement_lst = []

        for i, (candidates, conf, agr) in enumerate(
                zip(all_vote_candidates, confidence_per_seq, agreement_per_seq)):
            q_type, q_subtype = q_types[i]

            if q_type == 'yn':
                # Bypass diffusion output: pick yes vs no by max raw logit across positions.
                # This gives near-perfect accuracy for Y/N questions without any retraining.
                yes_score = logits[i, :, yes_id].max().item()
                no_score  = logits[i, :, no_id].max().item()
                winner = 'yes' if yes_score >= no_score else 'no'

            elif q_subtype in subtype_vocabs:
                # Re-decode with a semantically narrowed answer vocab (color / number / location).
                # Uses the last diffusion pass logits — no extra inference cost.
                sv = subtype_vocabs[q_subtype].to(logits.device)
                sv_mask = th.zeros(args.vocab_size, dtype=th.bool, device=logits.device)
                sv_mask[sv] = True
                sl = logits[i].clone()
                sl[:, ~sv_mask] = float('-inf')
                winner = tokenizer.decode_token(sl.argmax(dim=-1).cpu())

            else:
                winner = mbr_select(candidates)

            word_lst_recover.append(winner)
            confidence_lst.append(round(conf.item(), 6))
            rounding_agreement_lst.append(round(agr.item(), 6))

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            seq = seq.to(th.device("cpu"))
            len_x = args.seq_len
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        for (recov, ref, src, img_name, conf, agr) in zip(
                word_lst_recover, word_lst_ref, word_lst_source, image_name,
                confidence_lst, rounding_agreement_lst):
            print(json.dumps({
                "image_name": img_name,
                "question": src,
                "reference_answer": ref,
                "generate_answer": recov,
                "confidence": conf,
                "rounding_agreement": agr,
            }), file=fout)
        fout.flush()
        # break
        #
        # for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
        #     print(json.dumps(
        #         {"question": src, "reference_answer": ref, "generate_answer": recov}),
        #           file=fout)
        # fout.close()

    fout.close()
    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    main()
