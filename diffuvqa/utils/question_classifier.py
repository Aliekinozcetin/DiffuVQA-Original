"""
Question type classifier for inference-time answer routing.

classify_question(text) → (type, subtype)
  type    : 'yn'  — yes/no question; answer forced to argmax of {yes, no} logits
            'oe'  — open-ended; answer from diffusion (with optional subtype narrowing)
  subtype : 'yn' | 'color' | 'number' | 'location' | 'other'

build_subtype_vocabs(answer_vocab_ids, tokenizer, sep_token_id)
  → dict[subtype → LongTensor of token IDs]
  Used to re-decode OE answers with a narrowed answer vocab.
"""

import torch

# Auxiliary verbs that start yes/no questions in English
_YN_STARTERS = frozenset({
    'is', 'are', 'was', 'were', 'does', 'do', 'did',
    'has', 'have', 'had', 'can', 'could', 'should',
    'would', 'will', 'may', 'might', 'shall',
})

# OE subtype keyword triggers (matched against full question token set)
_SUBTYPE_TRIGGERS = {
    'color': frozenset({
        'color', 'colour', 'hue', 'colored', 'coloured', 'coloration', 'shade', 'pigment',
    }),
    'number': frozenset({
        'many', 'much', 'number', 'count', 'few', 'several', 'total', 'quantity',
    }),
    'location': frozenset({
        'where', 'location', 'located', 'region', 'area', 'site', 'position', 'side',
        'place', 'found', 'seen',
    }),
}

# Token-level keyword sets used to build subtype vocabs from answer_vocab_ids
_SUBTYPE_ANSWER_TOKENS = {
    'color': frozenset({
        'red', 'blue', 'green', 'yellow', 'white', 'black', 'purple',
        'pink', 'orange', 'brown', 'gray', 'grey', 'transparent', 'pale',
        'dark', 'light', 'bright', 'reddish', 'yellowish', 'pinkish', 'whitish',
        'clear', 'bloody', 'erythematous',
    }),
    'number': frozenset({
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'none', 'zero', 'single', 'multiple', 'several', 'few', 'many', 'no', 'yes',
    }),
    'location': frozenset({
        'left', 'right', 'upper', 'lower', 'middle', 'center', 'central',
        'distal', 'proximal', 'anterior', 'posterior', 'lateral', 'medial',
        'colon', 'stomach', 'esophagus', 'rectum', 'duodenum', 'cecum', 'sigmoid',
        'transverse', 'ascending', 'descending', 'fundus', 'antrum', 'pylorus',
        'cardia', 'body', 'ileocecal', 'jejunum', 'ileum', 'appendix',
    }),
}


def classify_question(text: str):
    """
    Returns (type, subtype).
    Fast heuristic — no model required, runs in microseconds per sample.
    """
    words = text.lower().strip().split()
    if not words:
        return 'oe', 'other'

    if words[0] in _YN_STARTERS:
        return 'yn', 'yn'

    word_set = set(words)
    for subtype, triggers in _SUBTYPE_TRIGGERS.items():
        if word_set & triggers:
            return 'oe', subtype

    return 'oe', 'other'


def build_subtype_vocabs(answer_vocab_ids: torch.Tensor, tokenizer, sep_token_id: int):
    """
    Scans answer_vocab_ids and returns a dict mapping subtype name → LongTensor of
    matching token IDs (always includes sep_token_id as a stop marker).

    Only subtypes with at least 2 matched tokens (beyond [SEP]) are included,
    so if the training set never uses color words the key is simply absent.
    """
    id_list = answer_vocab_ids.tolist()
    result = {}

    for subtype, kw_set in _SUBTYPE_ANSWER_TOKENS.items():
        matched = []
        for tid in id_list:
            tok = tokenizer.tokenizer.convert_ids_to_tokens([tid])[0].lower().lstrip('#')
            if tok in kw_set:
                matched.append(tid)
        if sep_token_id not in matched:
            matched.append(sep_token_id)
        if len(matched) > 2:  # at least 2 real tokens + [SEP]
            result[subtype] = torch.tensor(matched, dtype=torch.long)

    return result
