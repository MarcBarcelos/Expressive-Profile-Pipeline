import math
from collections import Counter
import numpy as np
import pandas as pd
import csv

def score_lexicon(tokens, lexicon, prefix, agg=("mean", "std"), coverage=True):
    vals = [lexicon[t] for t in tokens if t in lexicon]   # per-token scores for tokens covered by the lexicon
    out = {}
    if not vals:                                          # no covered tokens -> defined all-NaN output (matches semantic scorer contract)
        for stat in agg:
            out[f"{prefix}_{stat}"] = float("nan")
        if coverage:
            out[f"{prefix}_coverage"] = 0.0 if tokens else float("nan")
        return out
    arr = np.asarray(vals, dtype=float)
    if "mean" in agg:
        out[f"{prefix}_mean"] = float(arr.mean())
    if "std" in agg:
        out[f"{prefix}_std"] = float(arr.std())
    if "min" in agg:
        out[f"{prefix}_min"] = float(arr.min())
    if "max" in agg:
        out[f"{prefix}_max"] = float(arr.max())
    if coverage:
        out[f"{prefix}_coverage"] = float(len(vals) / len(tokens)) if tokens else float("nan")
    return out

# Dimensional affect: NRC-VAD (v2)
def vad_metrics(tokens, vad_lexicon):
    out = {}
    for dim in ("valence", "arousal", "dominance"):
        single = {t: scores[dim] for t, scores in vad_lexicon.items()}        # collapse the multi-dim lexicon to one dimension at a time
        out.update(score_lexicon(tokens, single, prefix=f"vad_{dim}",
                                 agg=("mean", "std"), coverage=False))
    # one shared coverage figure for the whole VAD lexicon (cheaper than 3 identical ones)
    covered = sum(1 for t in tokens if t in vad_lexicon)
    out["vad_coverage"] = float(covered / len(tokens)) if tokens else float("nan")
    return out

# Categorical affect: NRC Emotion / Affect Intensity Lexicon
EMOTIONS = ("anger", "anticipation", "disgust", "fear",
            "joy", "sadness", "surprise", "trust")

def emotion_metrics(tokens, emo_lexicon, emotions=EMOTIONS):
    out = {}
    if not tokens:
        for e in emotions:
            out[f"emo_{e}"] = float("nan")
        out["emo_diversity"] = float("nan")
        return out
    n = len(tokens)
    means = {}
    for cat in emotions:
        total = sum(emo_lexicon.get(t, {}).get(cat, 0.0) for t in tokens)      # sum association scores for this category across tokens
        means[cat] = total / n                                                 # mean association per token (token-normalized so it is length-robust)
        out[f"emo_{cat}"] = float(means[cat])
    # emotion diversity: entropy over the 8 basic-emotion means (ignores sentiment)
    emo_vec = np.array([means[e] for e in emotions], dtype=float)
    s = emo_vec.sum()
    if s <= 0:
        out["emo_diversity"] = 0.0                                             # no emotion-bearing tokens -> zero diversity
    else:
        p = emo_vec / s
        p = p[p > 0]
        out["emo_diversity"] = float(-np.sum(p * np.log(p)))                   # Shannon entropy in nats
    return out

# Lexicon loaders (format-tolerant; adjust paths/columns to your files)
def load_vad_lexicon(path):
    lex = {}
    with open(path, newline="", encoding="utf-8") as f:
        sniffed = "\t" if "\t" in f.readline() else ","
        f.seek(0)
        reader = csv.reader(f, delimiter=sniffed)
        for row in reader:
            if len(row) < 4:
                continue
            term = row[0].strip().lower()
            try:
                v, a, d = float(row[1]), float(row[2]), float(row[3])
            except ValueError:                                                 # header or malformed line
                continue
            lex[term] = {"valence": v, "arousal": a, "dominance": d}
    return lex


def load_emotion_lexicon(path):
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            term, cat, score = parts[0].strip().lower(), parts[1].strip().lower(), parts[2]
            try:
                score = float(score)
            except ValueError:
                continue
            lex.setdefault(term, {})[cat] = score
    return lex

def load_norm_by_name(path, term_field, score_field, sep=None):
    p = str(path)
    if p.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p, sep=sep, engine="python")
    if term_field not in df.columns or score_field not in df.columns:
        raise KeyError(
            f"{path}: expected columns {term_field!r} and {score_field!r}; "
            f"found {list(df.columns)}")
    lex = {}
    for term, score in zip(df[term_field], df[score_field]):
        try:
            lex[str(term).strip().lower()] = float(score)
        except (ValueError, TypeError):
            continue
    return lex
 
 
def load_norm_lexicon(path, term_col=0, score_col=1, sep=None, skip_header=True):
    lex = {}
    with open(path, newline="", encoding="utf-8") as f:
        if sep is None:
            first = f.readline()
            sep = "\t" if "\t" in first else ","
            f.seek(0)
        reader = csv.reader(f, delimiter=sep)
        if skip_header:
            next(reader, None)
        for row in reader:
            if len(row) <= max(term_col, score_col):
                continue
            try:
                lex[row[term_col].strip().lower()] = float(row[score_col])
            except ValueError:
                continue
    return lex
