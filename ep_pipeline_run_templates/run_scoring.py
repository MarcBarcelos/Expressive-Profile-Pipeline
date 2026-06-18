from pathlib import Path
import pandas as pd

from ep_pipeline.config import MetricsConfig
from ep_pipeline.io import load_jsonl, load_csv, read_checkpoint, write_table
from ep_pipeline.models import pick_device, load_embedder, load_spacy_model
from ep_pipeline.scoring.get_td_linguistic import tokenize, get_td_metrics
from ep_pipeline.scoring.get_other_linguistic import (mattr, windowed_unigram_entropy, trigram_entropy)
from ep_pipeline.scoring.get_semantic import (
    _empty_metrics, chunk_by_words, l2_normalize, adj_cos, shannon_entropy,
    compute_semantic_metrics)
from ep_pipeline.assemble_corpus import extract_passage, build_text_table
from ep_pipeline.scoring.runner import map_with_checkpoints, _key

# PATHS FOR THIS PROJECT
PROJECT_ROOT  = Path("/path/to/your/project")
FULL_FP       = PROJECT_ROOT / "data" / "corpus.jsonl"
PROMPTS_FP    = PROJECT_ROOT / "outputs" / "ai_continuation" / "prompts.jsonl"
OUTPUT_FP     = PROJECT_ROOT / "outputs" / "ai_continuation" / "outputs.jsonl"
EXCERPTS_FP   = PROJECT_ROOT / "outputs" / "ai_continuation" / "excerpts.csv"
E5_MODEL_PATH = PROJECT_ROOT / "models" / "e5-small"

OUT_DIR = PROJECT_ROOT / "outputs" / "AI_Comparison" / "results"
VIS_DIR = PROJECT_ROOT / "outputs" / "AI_Comparison" / "visualizations"
TD_FP, SEM_FP, LING_FP = OUT_DIR / "td_checkpoint.csv", OUT_DIR / "sem_checkpoint.csv", OUT_DIR / "ling_checkpoint.csv"
OUT_FP = OUT_DIR / "data" / "metrics_full.csv"

cfg = MetricsConfig() # defaults set in config.py

# LOADING EXPENSIVE OBJECTS ONCE
embedder = load_embedder(E5_MODEL_PATH)
nlp_tok =load_spacy_model(cfg.spacy_model, for_tokenizing=True)

# IO
full_texts = load_jsonl(FULL_FP)
prompts = load_jsonl(PROMPTS_FP)
outputs = load_jsonl(OUTPUT_FP)
excerpts = load_jsonl(EXCERPTS_FP)
chunks_df = build_text_table(full_texts)  # Full text -> build_text_table(docs)
records = chunks_df.to_dict("records")                              # Prompt + AI -> build_text_table(prompts, prompt_key="prompt", outputs=outputs)


#SCORING
def score_linguistic(r):
    toks = tokenize(r["text"], nlp_tok)
    H_mean, H_std, PPL = windowed_unigram_entropy(toks, entropy_window=cfg.entropy_window)
    H3, PPL3 = trigram_entropy(toks, cfg.trigram_test_frac, cfg.trigram_alpha, cfg.seed)
    mattr_score = mattr(toks, cfg.mattr_window)
    return {**r, "id": r["id"], "source": r["source"], "mattr": mattr_score, "H_unigram_win_mean_nats": H_mean, "H_unigram_win_std_nats": H_std, "PPL_unigram_win_mean": PPL, "H_3gram_self_nats": H3, "PPL_3gram_self_nats": PPL3}

def score_td(r): 
    td_metrics = get_td_metrics(r["text"], spacy_model=cfg.spacy_model, td_metrics=cfg.td_metrics)
    return {**r, "id": r["id"], "source": r["source"], **td_metrics}

def score_semantic(r):
    semantic_metrics = compute_semantic_metrics(r["text"], embedder, chunk_size=cfg.embed_chunk_size, overlap=cfg.embed_overlap, batch_size=cfg.batch_size, prefix=cfg.e5_prefix, seed=cfg.seed)
    return {**r, "id": r["id"], "source": r["source"], **semantic_metrics}

ling = map_with_checkpoints(records, score_linguistic, LING_FP, ["id", "source"], checkpoint_every=cfg.checkpoint_every)
td = map_with_checkpoints(records, score_td, TD_FP, ["id", "source"], checkpoint_every=cfg.checkpoint_every)
sem = map_with_checkpoints(records, score_semantic, SEM_FP, ["id", "source"], checkpoint_every=cfg.checkpoint_every)

# MERGE AND WRITE FINAL SCORES
for df in (ling, td, sem):
    df["id"], df["source"] = df["id"].astype(str), df["source"].astype(str)
all_metrics = ling
for df in (td, sem):
    all_metrics = all_metrics.merge(df, on=["id", "source"], how="left")
meta_cols = ["id", "source"] + [c for c in chunks_df.columns
                                if c != "text" and c not in all_metrics.columns]
all_metrics = chunks_df[meta_cols].merge(all_metrics, on=["id", "source"], how="left") 
write_table(all_metrics, OUT_FP)
print(f"Saved: {all_metrics.shape} to {OUT_FP}")