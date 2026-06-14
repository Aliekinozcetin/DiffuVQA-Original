import json, numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from torchmetrics.text.rouge import ROUGEScore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("omw-1.4", quiet=True)

path = r"C:\Users\mertg\Downloads\ema_0.9999_150000.jsonl"

rougeScore = ROUGEScore()

def get_bleu1(hyp, ref):
    return sentence_bleu([ref.split()], hyp.split(), weights=(1.0,), smoothing_function=SmoothingFunction().method4)

def token_f1(refs, preds):
    total_tp = total_fp = total_fn = 0
    for ref, pred in zip(refs, preds):
        rc = {}
        for t in ref.lower().split():
            rc[t] = rc.get(t, 0) + 1
        tp = 0
        for t in pred.lower().split():
            if rc.get(t, 0) > 0:
                tp += 1; rc[t] -= 1
        total_tp += tp
        total_fp += len(pred.split()) - tp
        total_fn += len(ref.split()) - tp
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f = 2*p*r/(p+r) if (p+r) > 0 else 0
    return p, r, f

def cider_score(cands, refs):
    vec = TfidfVectorizer()
    mat = vec.fit_transform(cands + refs).toarray()
    scores = []
    n = len(cands)
    for i in range(n):
        sim = cosine_similarity(mat[i].reshape(1,-1), mat[n+i].reshape(1,-1))
        scores.append(sim[0][0])
    return np.mean(scores)

sources, references, recovers = [], [], []
acc = acc_yn = acc_oe = 0
c_yn = c_oe = 1e-9
bleus, rougels, meteors, avg_lens = [], [], [], []

# Count total lines first for progress
with open(path, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)
print(f"Total lines: {total_lines}", flush=True)

with open(path, "r", encoding="utf-8") as f:
    for idx, row in enumerate(f):
        d = json.loads(row)
        src = d["question"].strip()
        ref = d["reference_answer"].strip()
        rec = d["generate_answer"].strip()
        if src.lower() == "none" or ref.lower() in ("nan", "none", ""):
            continue
        ref = ref.replace("[SEP]", "").replace("[CLS]", "").strip()
        rec = rec.replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
        sources.append(src); references.append(ref); recovers.append(rec)
        avg_lens.append(len(rec.split()))
        bleus.append(get_bleu1(rec, ref))
        rougels.append(rougeScore(rec, ref)["rougeL_fmeasure"].item())
        ref_tok = word_tokenize(ref)
        rec_tok = word_tokenize(rec)
        meteors.append(meteor_score([ref_tok], rec_tok))
        if rec == ref: acc += 1
        if ref in ("yes", "no"):
            c_yn += 1
            if rec == ref: acc_yn += 1
        else:
            c_oe += 1
            if rec == ref: acc_oe += 1
        if (idx + 1) % 500 == 0:
            pct = (idx + 1) / total_lines * 100
            print(f"  [{idx+1}/{total_lines}] {pct:.1f}% done...", flush=True)

cnt = len(references)
p, r, f1 = token_f1(references, recovers)
cider = cider_score(recovers, references)

print(f"N samples        : {cnt}")
print(f"BLEU-1           : {np.mean(bleus):.4f}")
print(f"ROUGE-L          : {np.mean(rougels):.4f}")
print(f"METEOR           : {np.mean(meteors):.4f}")
print(f"CIDEr (TF-IDF)   : {cider:.4f}")
print(f"Token F1         : {f1:.4f}  (P={p:.4f} R={r:.4f})")
print(f"Accuracy (all)   : {acc/cnt:.4f}  ({int(acc)}/{cnt})")
print(f"Accuracy YN      : {acc_yn/c_yn:.4f}")
print(f"Accuracy OE      : {acc_oe/c_oe:.4f}")
print(f"Avg answer len   : {np.mean(avg_lens):.2f} tokens")
