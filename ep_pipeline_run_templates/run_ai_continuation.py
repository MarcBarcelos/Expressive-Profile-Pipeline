import json
from pathlib import Path

from ep_pipeline.config import PromptConfig
from ep_pipeline.io import load_jsonl
from ep_pipeline.ai_imitation.make_prompts import build_excerpts_and_prompts
from ep_pipeline.ai_imitation.run_batch_mlx import run_batch

# PATHS FOR THIS PROJECT 
PROJECT_ROOT = Path("/path/to/your/project")

CORPUS_FP    = PROJECT_ROOT / "data" / "corpus.jsonl"   # id + text (+ any metadata)
MODEL_ID     = PROJECT_ROOT / "models" / "your-model"   # path or HuggingFace model ID

OUT_DIR      = PROJECT_ROOT / "outputs" / "ai_continuation"
EXCERPTS_FP  = OUT_DIR / "excerpts.csv"
PROMPTS_FP   = OUT_DIR / "prompts.jsonl"
OUTPUTS_FP   = OUT_DIR / "outputs.jsonl"

# CONFIG
cfg = PromptConfig(
    n_chunks  = 3,    # chunks sampled per document
    chunk_min = 500,  # min words per chunk
    chunk_max = 600,  # max words per chunk
    seed      = 19,
)

# BUILD EXCERPTS + PROMPTS
# Load corpus — expects at minimum {"id": ..., "text": ...} per record.
# Any extra fields (author, date, label, etc.) pass through to excerpts.csv.
# If your corpus uses different column names, pass id_key= and text_key=.

corpus = load_jsonl(CORPUS_FP)          # or load_csv(CORPUS_FP).to_dict("records")

excerpts_df, prompts = build_excerpts_and_prompts(
    corpus,
    id_key   = "id",
    text_key = "text",
    cfg      = cfg,
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
excerpts_df.to_csv(EXCERPTS_FP, index=False)
print(f"Saved {len(excerpts_df)} excerpts -> {EXCERPTS_FP}")

with PROMPTS_FP.open("w", encoding="utf-8") as f:
    for p in prompts:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"Saved {len(prompts)} prompts -> {PROMPTS_FP}")

# RUN MODEL


run_batch(                                              # Reads prompts.jsonl, writes outputs.jsonl.
    PROMPTS_FP, OUTPUTS_FP,                             # Resumes automatically if interrupted — already-completed IDs are skipped.
    model_id       = str(MODEL_ID),
    max_new_tokens = cfg.max_new_tokens,
    temp           = cfg.temp,
    top_p          = cfg.top_p,
)

# ── WHAT YOU HAVE AFTERWARDS ──────────────────────────────────────────────────
# excerpts.csv  — human text chunks + metadata, keyed by id
# outputs.jsonl — AI completions, keyed by the same id