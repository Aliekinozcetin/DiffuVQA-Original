import torch
# bert results
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator, GPT2TokenizerFast
import sys, yaml, os
import json

import numpy as np

def get_knn(model_emb, text_emb, dist='cos'):
    if dist == 'cos':
        adjacency = model_emb @ text_emb.transpose(1, 0).to(model_emb.device)
    elif dist == 'l2':
        adjacency = model_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
            model_emb.size(0), -1, -1)
        adjacency = -torch.norm(adjacency, dim=-1)
    topk_out = torch.topk(adjacency, k=6, dim=0)
    return topk_out.values, topk_out.indices

def get_efficient_knn(model_emb, text_emb):
    emb_norm = (model_emb**2).sum(-1).view(-1, 1)
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)
    arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t)
    dist = torch.clamp(dist, 0.0, np.inf)
    topk_out = torch.topk(-dist, k=1, dim=0)
    return topk_out.values, topk_out.indices

def rounding_func(text_emb_lst, model, tokenizer, emb_scale_factor=1.0):
    decoded_out_lst = []
    
    model_emb = model.weight  # input_embs
    down_proj_emb2 = None

    dist = 'l2'
    
    for text_emb in text_emb_lst:
        import torch
        if not isinstance(text_emb, torch.Tensor):
            text_emb = torch.tensor(text_emb)
        # print(text_emb.shape)
        if len(text_emb.shape) > 2:
            text_emb = text_emb.view(-1, text_emb.size(-1))
        else:
            text_emb = text_emb
        val, indices = get_knn((down_proj_emb2 if dist == 'cos' else model_emb),
                                text_emb.to(model_emb.device), dist=dist)
    
        # decoded_out_lst.append(tokenizer.decode(indices[0]))
        decoded_out_lst.append(tokenizer.decode_token(indices[0])) 
        
    return decoded_out_lst

def compute_logp(args, model, x, input_ids):
    word_emb = model.weight
    sigma = 0.1
    if args.model_arch == '1d-unet':
        x = x.permute(0, 2, 1)

    bsz, seqlen, dim = x.shape

    x_flat = x.reshape(-1, x.size(-1)).unsqueeze(0)  # 1, bsz*sample*seqlen, dim
    word_emb_flat = word_emb.unsqueeze(1)  # vocab, 1,  dim
    diff = (x_flat - word_emb_flat) ** 2  # vocab, seqlen, dim

    logp_expanded = -diff.sum(dim=-1) / (2 * sigma ** 2)  # vocab, seqlen
    logp_expanded = logp_expanded.permute((1, 0))

    ce = torch.nn.CrossEntropyLoss(reduction='none')
    loss = ce(logp_expanded, input_ids.view(-1)).view(bsz, seqlen)

    return loss

def get_weights(model, args):
    if hasattr(model, 'transformer'):
        input_embs = model.transformer.wte
        down_proj = model.down_proj
        model_emb = down_proj(input_embs.weight)
        print(model_emb.shape)
        model = torch.nn.Embedding(model_emb.size(0), model_emb.size(1))
        print(args.emb_scale_factor)
        model.weight.data = model_emb * args.emb_scale_factor

    elif hasattr(model, 'weight'):
        pass
    else:
        assert NotImplementedError

    model.weight.requires_grad = False
    return model

def denoised_fn_round(args, model, text_emb, t, answer_vocab_ids=None, get_logits=None):
    old_shape = text_emb.shape
    old_device = text_emb.device

    # text_emb shape: (B, 2*seq_len, hidden) — first half is question, second is answer.
    # answer_vocab restriction must only apply to the answer half; applying it to the
    # question half distorts the conditioning signal fed into the next denoising step.
    if answer_vocab_ids is not None and len(old_shape) == 3:
        B, seq2, hidden = old_shape
        seq_len = seq2 // 2
        q_emb = text_emb[:, :seq_len, :]   # (B, seq_len, hidden)
        a_emb = text_emb[:, seq_len:, :]   # (B, seq_len, hidden)

        q_rounded = _round_half(args, model, q_emb, old_device,
                                answer_vocab_ids=None, get_logits=get_logits)
        a_rounded = _round_half(args, model, a_emb, old_device,
                                answer_vocab_ids=answer_vocab_ids, get_logits=get_logits)
        return torch.cat([q_rounded, a_rounded], dim=1)

    # fallback: no answer_vocab or flat input — round everything uniformly
    flat = text_emb.reshape(-1, text_emb.size(-1)) if len(old_shape) > 2 else text_emb
    rounded = _round_half(args, model, flat.unsqueeze(0), old_device,
                          answer_vocab_ids=None, get_logits=get_logits)
    return rounded.view(old_shape).to(old_device)


def _round_half(args, model, emb, device, answer_vocab_ids=None, get_logits=None):
    """Round a (B, L, hidden) embedding to nearest token embeddings."""
    old_shape = emb.shape
    flat = emb.reshape(-1, emb.size(-1))  # (B*L, hidden)

    if get_logits is not None:
        logits = get_logits(flat.to(device))  # (B*L, vocab)
        if answer_vocab_ids is not None:
            logits = logits[:, answer_vocab_ids]
            rounded_tokens = answer_vocab_ids[logits.argmax(dim=-1)]
        else:
            rounded_tokens = logits.argmax(dim=-1)
    else:
        model_emb = model.weight
        knn_emb = model_emb[answer_vocab_ids] if answer_vocab_ids is not None else model_emb
        val, indices = get_efficient_knn(knn_emb, flat.to(knn_emb.device))
        rounded_tokens = answer_vocab_ids[indices[0]] if answer_vocab_ids is not None else indices[0]

    return model(rounded_tokens).view(old_shape).to(device)

