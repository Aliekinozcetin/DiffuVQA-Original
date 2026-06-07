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

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from collections import Counter
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
    decode_defaults = dict(split='test', clamp_step=50, seed2=105, clip_denoised=False)
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
                                             'seed2', 'clamp_step', 'top_p',
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
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"### New params (random init): {missing}")
    if unexpected:
        print(f"### Unexpected keys ignored: {unexpected}")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(th.device("cuda"))

    tokenizer = load_tokenizer(args)
    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size,
        embedding_dim=args.hidden_dim,
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
    # always keep special tokens so [CLS]/[SEP]/[PAD] boundaries work
    special_ids = {tokenizer.tokenizer.cls_token_id,
                   tokenizer.tokenizer.sep_token_id,
                   tokenizer.tokenizer.pad_token_id}
    answer_vocab_set.update(special_ids)
    answer_vocab_set.discard(None)
    # Keep all tokens including ## wordpiece continuations — decode_token uses
    # convert_tokens_to_string which correctly merges them (col + ##on + ##oscopy
    # → colonoscopy). Filtering ## tokens would break multi-subword answers.
    answer_vocab_ids = torch.tensor(sorted(tid for tid in answer_vocab_set if tid is not None),
                                    dtype=torch.long, device=th.device("cuda"))
    print(f"### Answer vocabulary size: {len(answer_vocab_ids)} / {tokenizer.vocab_size} tokens")

    # --- Question-type aware vocab kısıtlaması ---
    # Closed-ended sorular (is/are/does/...) → sadece yes/no/not relevant token'larına izin ver.
    # İstisna: "are there any X? check all that are present" → OE cevap (polyp, none, z-line vb.)
    CLOSED_STARTERS = frozenset({
        'is', 'are', 'does', 'do', 'have', 'has', 'was', 'were',
        'can', 'could', 'would', 'will', 'did', 'should'
    })
    OE_OVERRIDE_PHRASES = frozenset({'check all', 'list all', 'select all'})
    _yn_surface = ['yes', 'no', 'not', 'relevant', 'not relevant', 'not applicable', '0', '1']
    _yn_token_ids = set()
    for _ans in _yn_surface:
        _ids = tokenizer.tokenizer(_ans, add_special_tokens=False)['input_ids']
        _yn_token_ids.update(_ids)
    _yn_token_ids.update(special_ids)
    _yn_token_ids = {t for t in _yn_token_ids if t in answer_vocab_set and t is not None}
    # Fix3: iki ayrı mask.
    # yn_mask_bool        → position 0 için, SEP YOK  → model gerçek cevap token'ı seçmek zorunda
    # yn_mask_with_sep    → position 1+ için, SEP VAR → decode_token SEP'de keser → kısa temiz cevap
    _yn_content_ids = _yn_token_ids - special_ids
    _yn_content_ids.discard(None)
    yn_mask_bool = th.zeros(tokenizer.vocab_size, dtype=th.bool, device=th.device("cuda"))
    yn_mask_bool[torch.tensor(sorted(_yn_content_ids), dtype=torch.long, device=th.device("cuda"))] = True

    # Fix4-B: yn_mask_with_sep for positions 1+ contains ONLY [SEP] and [PAD].
    # Previous version also included yes/no/not/relevant at pos 1+, causing "yes no" / "no 0" artifacts.
    # With only SEP+PAD allowed at pos 1+, decode_token truncates at SEP → clean single-word answer.
    yn_mask_with_sep = th.zeros(tokenizer.vocab_size, dtype=th.bool, device=th.device("cuda"))
    _sep_id = tokenizer.tokenizer.sep_token_id
    _pad_id = tokenizer.tokenizer.pad_token_id
    if _sep_id is not None:
        yn_mask_with_sep[_sep_id] = True
    if _pad_id is not None:
        yn_mask_with_sep[_pad_id] = True

    print(f"### YN vocab size: {int(yn_mask_bool.sum())} content tokens (pos 1+ = SEP+PAD only)")

    text_iterator = iter(all_text_data)
    image_iterator = iter(all_image_data)

    total_batches = len(all_text_data)
    pbar = tqdm(zip(image_iterator, text_iterator), total=total_batches, desc="Sampling", unit="batch")
    for image, cond in pbar:

        if not cond:
            continue                                           

        input_ids_x = cond.pop('input_ids').to(th.device("cuda"))
        input_ids_a = cond.pop('input_a_id').to(th.device("cuda"))

        # Soru tipini tespit et: ilk kelimeye göre fallback heuristic (classifier için de sakla).
        # OE override: "check all that are present" soruları CLOSED_STARTERS ile başlasa da OE cevap istiyor.
        q_first_words = []
        q_texts_lower = []
        for q_seq in input_ids_x[:, :args.seq_len]:
            q_text = tokenizer.decode_token(q_seq.cpu())
            fw = q_text.strip().lower().split()[0] if q_text.strip() else ''
            q_first_words.append(fw)
            q_texts_lower.append(q_text.strip().lower())

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

        # Classifier head: learned q-type routing (falls back to first-word heuristic
        # if question_type_head was not trained yet — randomly initialized weights
        # will give ~50% closed, so first-word heuristic acts as a safety override).
        _cls_logits = model.classify_question(fuse_feats)  # B
        _cls_closed = (torch.sigmoid(_cls_logits) > 0.5).tolist()  # list[bool]
        # Blend: if classifier says closed OR first-word heuristic says closed → closed.
        # During early training when classifier is random, heuristic dominates; later
        # the classifier takes over and catches harder cases (e.g. "what colour" with yes/no ref).
        # OE override: "check all / list all / select all" soruları OE cevap istiyor (polyp, none, z-line).
        is_closed_pred = [
            (_cls_closed[_i] or (q_first_words[_i] in CLOSED_STARTERS))
            and not any(ph in q_texts_lower[_i] for ph in OE_OVERRIDE_PHRASES)
            for _i in range(len(q_first_words))
        ]

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
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps // args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], x_start.shape[1], args.hidden_dim)

        # answer_mask_bool is the same for every vote; build it once per batch
        answer_mask_bool = th.zeros(tokenizer.vocab_size, dtype=th.bool, device=th.device("cuda"))
        answer_mask_bool[answer_vocab_ids] = True

        n_votes = args.n_samples
        batch_sz = x_start.shape[0]
        all_vote_candidates = [[] for _ in range(batch_sz)]
        logits = None
        masked_logits = None
        decode_ids = None

        if getattr(args, "use_noising_f", False):
            print("noising f")

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
                denoised_fn=partial(denoised_fn_round, args, model_emb,
                                    answer_vocab_ids=answer_vocab_ids,
                                    get_logits=model.get_logits),
                model_kwargs=model_kwargs,
                top_p=args.top_p,
                clamp_step=args.clamp_step,
                clamp_first=False,
                mask=input_ids_mask,
                x_start=x_start,
                gap=step_gap
            )

            sample = samples[-1]
            a_shape = sample.size(1) // 2
            sample = sample[:, a_shape:, :]
            logits = model.get_logits(sample)  # bsz, seqlen, vocab

            # Per-sample vocab masking: closed-ended sorular → yn_mask, open → answer_mask.
            # Fix3: position 0 → yn_mask_bool (SEP YOK, gerçek cevap zorla)
            #        position 1+ → yn_mask_with_sep (SEP VAR, decode_token kesebilir)
            masked_logits = logits.clone()
            for _i, _is_closed in enumerate(is_closed_pred):
                if _is_closed:
                    masked_logits[_i, 0, ~yn_mask_bool] = float('-inf')
                    if masked_logits.size(1) > 1:
                        masked_logits[_i, 1:, ~yn_mask_with_sep] = float('-inf')
                else:
                    masked_logits[_i, :, ~answer_mask_bool] = float('-inf')
            # Fallback: tüm pozisyonlar -inf ise (vocab mask boş) ham logits kullan.
            all_inf = (masked_logits == float('-inf')).all(dim=-1, keepdim=True)
            masked_logits = th.where(all_inf.expand_as(masked_logits), logits, masked_logits)
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

        _PUNCT_ONLY = frozenset({'-', ';', ',', '.', ':', '–', '—', '', ' '})
        for _wi, (candidates, conf, agr) in enumerate(
                zip(all_vote_candidates, confidence_per_seq, agreement_per_seq)):
            winner = Counter(candidates).most_common(1)[0][0]
            # Empty / punctuation-only fallback: tüm pozisyonları dene, ilk non-empty decode'u al.
            if winner.strip() in _PUNCT_ONLY:
                winner = ''
                for _pos in range(masked_logits.size(1)):
                    _best_id = masked_logits[_wi, _pos].argmax().unsqueeze(0)
                    _candidate = tokenizer.decode_token(_best_id.cpu())
                    if _candidate.strip() and _candidate.strip() not in _PUNCT_ONLY:
                        winner = _candidate
                        break
            word_lst_recover.append(winner)
            confidence_lst.append(round(conf.item(), 6))
            rounding_agreement_lst.append(round(agr.item(), 6))

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            seq = seq.to(th.device("cpu"))
            len_x = args.seq_len
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        fout = open(out_path, 'a')
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
        fout.close()
        # break
        #
        # for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
        #     print(json.dumps(
        #         {"question": src, "reference_answer": ref, "generate_answer": recov}),
        #           file=fout)
        # fout.close()

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    main()
