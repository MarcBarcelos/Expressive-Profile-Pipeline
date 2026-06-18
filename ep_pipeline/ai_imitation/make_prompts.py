import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ep_pipeline.config import PromptConfig
from ep_pipeline.io import load_jsonl, load_csv

_WORD_RE = re.compile(r"\S+")
_SENT_END_RE = re.compile(r'[.!?]["\')\]]*')


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)

def detokenize_words(words: list[str]) -> str:
    return " ".join(words)

def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))

def _snap_to_sentence_bounds(words: list[str], min_len: int, max_len: int) -> list[str]:
    words = words[: max_len + 200]
    text = detokenize_words(words)

    ends = [m.end() for m in _SENT_END_RE.finditer(text)]
    if not ends:
        out = tokenize_words(text)[:max_len]
        return out if len(out) >= min_len else []

    end_cut = None
    for epos in reversed(ends):
        if len(tokenize_words(text[:epos].strip())) <= max_len:
            end_cut = epos
            break
    if end_cut is None:
        end_cut = ends[0]

    text2 = text[:end_cut].strip()
    w2 = tokenize_words(text2)
    if len(w2) > max_len:
        w2 = w2[:max_len]
        text2 = detokenize_words(w2)

    for epos in [m.end() for m in _SENT_END_RE.finditer(text2)]:
        cand = tokenize_words(text2[epos:].strip())
        if min_len <= len(cand) <= max_len:
            w2 = cand
            break

    return w2 if len(w2) >= min_len else []

def sample_chunk(
    words: list[str],
    min_len: int,
    max_len: int,
    extra_min: int = 50,
    extra_max: int = 120,
    max_tries: int = 60,
) -> list[str]:
    n = len(words)
    if n < min_len:
        return []
    for _ in range(max_tries):
        raw_len = min(n, max_len + random.randint(extra_min, extra_max))
        start = random.randint(0, max(0, n - raw_len - 1))
        out = _snap_to_sentence_bounds(words[start : start + raw_len], min_len, max_len)
        if len(out) >= min_len:
            return out
    out = _snap_to_sentence_bounds(words[: min(n, max_len + extra_max)], min_len, max_len)
    return out[:max_len] if out else []

def build_prompt(excerpt: str, target_words: int) -> str:
    return (
        "You are a prose continuation engine. "
        "Output ONLY the continuation text itself. "
        "Do not include any preamble, commentary, labels, or explanation. "
        "Do not repeat any part of the passage. "
        "Do not reproduce any text from the original work. "
        "Match the style, tone, tense, pacing, and narrative voice exactly. "
        f"Write approximately {target_words} words.\n\n"
        f"PASSAGE:{excerpt}"
    )

def build_excerpts_and_prompts(
    records: list[dict],
    *,
    id_key: str = "id",
    text_key: str = "text",
    cfg: PromptConfig | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    if cfg is None:
        cfg = PromptConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    excerpt_rows: list[dict] = []
    skipped: list[str] = []
    for rec in records:
        uid = rec[id_key]
        raw = re.sub(r"\s+", " ", rec[text_key]).strip()
        words = tokenize_words(raw)
        extra = {k: v for k, v in rec.items() if k not in {id_key, text_key}}
        if len(words) < cfg.chunk_min:
            skipped.append(f"{uid}: too short ({len(words)} words)")
            continue
        segment_size = len(words) // cfg.n_chunks
        for j in range(cfg.n_chunks):
            seg_start = j * segment_size
            seg_end = (j + 1) * segment_size if j < cfg.n_chunks - 1 else len(words)
            target = random.randint(cfg.chunk_min, cfg.chunk_max)
            chunk = sample_chunk(
                words[seg_start:seg_end],
                min_len=cfg.chunk_min,
                max_len=cfg.chunk_max,
                extra_min=cfg.extra_min,
                extra_max=cfg.extra_max,
                max_tries=cfg.max_tries,
            )
            if not chunk:
                skipped.append(f"{uid}: no sentence-complete chunk in segment {j + 1}/{cfg.n_chunks}")
                continue
            excerpt = detokenize_words(chunk)
            chunk_id = f"{uid}__chunk{j:02d}"
            excerpt_rows.append({
                "id":            chunk_id,
                "source_id":     uid,
                "chunk_idx":     j,
                "target_words":  target,
                "excerpt_words": count_words(excerpt),
                "excerpt":       excerpt,
                **extra,
            })
    if skipped:
        print(f"Skipped {len(skipped)} chunk(s) — first few:")
        for msg in skipped[:5]:
            print(f"  {msg}")
    excerpts_df = pd.DataFrame(excerpt_rows)
    prompts = [
        {"id": r["id"], "prompt": build_prompt(r["excerpt"], r["target_words"])}
        for r in excerpt_rows
    ]
    return excerpts_df, prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build excerpts and prompts for the MLX batch runner. "
                    "Input must be a JSONL or CSV with at least 'id' and 'text' columns."
    )
    parser.add_argument("--input",        required=True, type=Path,
                        help="JSONL or CSV of documents (needs 'id' and 'text' columns)")
    parser.add_argument("--prompts-out",  required=True, type=Path,
                        help="Output path for prompts .jsonl (input to run_batch_mlx)")
    parser.add_argument("--excerpts-out", required=True, type=Path,
                        help="Output path for excerpts .csv")
    parser.add_argument("--id-col",   default="id",   help="Column name for document ID")
    parser.add_argument("--text-col", default="text", help="Column name for document text")
    parser.add_argument("--n-chunks",  type=int, default=PromptConfig.n_chunks)
    parser.add_argument("--chunk-min", type=int, default=PromptConfig.chunk_min)
    parser.add_argument("--chunk-max", type=int, default=PromptConfig.chunk_max)
    parser.add_argument("--seed",      type=int, default=PromptConfig.seed)
    args = parser.parse_args()

    input_path: Path = args.input
    if input_path.suffix == ".jsonl":
        records = load_jsonl(input_path)
    else:
        records = load_csv(input_path).to_dict("records")

    cfg = PromptConfig(
        n_chunks=args.n_chunks,
        chunk_min=args.chunk_min,
        chunk_max=args.chunk_max,
        seed=args.seed,
    )

    excerpts_df, prompts = build_excerpts_and_prompts(
        records, id_key=args.id_col, text_key=args.text_col, cfg=cfg,
    )

    if excerpts_df.empty:
        raise RuntimeError("No excerpts generated — check input file and chunk size settings.")

    args.excerpts_out.parent.mkdir(parents=True, exist_ok=True)
    excerpts_df.to_csv(args.excerpts_out, index=False)
    print(f"Saved {len(excerpts_df)} excerpts -> {args.excerpts_out}")

    args.prompts_out.parent.mkdir(parents=True, exist_ok=True)
    with args.prompts_out.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(prompts)} prompts -> {args.prompts_out}")


if __name__ == "__main__":
    main()
