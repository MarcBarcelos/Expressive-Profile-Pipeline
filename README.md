# Expressive Profile Pipeline

**Expressive Profile Pipeline** — a Python toolkit for extracting stylometric features from text, reducing them to interpretable expressive dimensions via Exploratory Factor Analysis (EFA), and generating AI continuations of text passages for human–AI comparison.

The pipeline scores each text on dozens of linguistic and semantic metrics, learns a factor structure across them so any document can be described by a handful of latent "expressive profile" factors, and can generate matched AI continuations using a local MLX language model. It was built for comparing the stylistic properties of human and AI-generated text, but works on any corpus of English documents.

## What it does

The workflow has four stages:

1. **AI Continuation** — sample sentence-complete excerpts from a corpus, build continuation prompts, and run a local MLX model to generate matched AI text.
2. **Scoring** — compute per-document metrics across three families of features (for both human and AI texts).
3. **EFA** — check factorability, choose the number of factors via parallel analysis, fit an oblique factor model, and produce factor scores + loadings.
4. **Apply** — project new, unseen documents onto the already-fitted factor model.

### Feature families

**Linguistic (TextDescriptives + custom)**
Lexical richness (MATTR, type-token ratios, OOV/alpha ratios), word-form complexity (length and syllable statistics), document scale, repetition (duplicate/top n-gram fractions), POS distribution, readability indices (Flesch, SMOG, LIX, etc.), dependency distance, and coherence. Custom additions include moving-average type-token ratio (MATTR), windowed unigram entropy/perplexity, and held-out trigram entropy.

**Semantic (sentence-embedding based)**
Computed on overlapping word-chunks embedded with an [E5](https://huggingface.co/intfloat/e5-small) sentence-transformer model. Metrics include first- and second-order coherence, semantic drift, trajectory length, dispersion, PCA-based topic strength, semantic volume, and KMeans topic-cluster entropy. Inspired by Elkins et al. (2023), *"Thematic and semantic coherence in discourse."*

**Structural / meta**
Document and chunk counts, embedding dimension, and other shape features used to control for size during factor analysis.

Metrics are organized into a taxonomy (`metric_taxonomy.py`) that groups each feature into a conceptual category and tracks which features to drop before EFA (e.g. collinear measures, perplexities, reference POS categories).

## Installation

Requires **Python ≥ 3.12**.

```bash
git clone https://github.com/MarcBarcelos/Expressive-Profile-Pipeline.git
cd Expressive-Profile-Pipeline
pip install -e .
```

This installs the core dependencies: PyTorch, spaCy, sentence-transformers, TextDescriptives, factor-analyzer, scikit-learn, pandas, numpy, joblib, and matplotlib.

You will also need:

- A spaCy English model (default `en_core_web_sm`):
  ```bash
  python -m spacy download en_core_web_sm
  ```
- A sentence-transformer embedding model (the templates assume a local E5-small checkpoint).
- For AI continuation: an MLX-compatible model (Apple Silicon only). Install `mlx-lm` separately:
  ```bash
  pip install mlx-lm
  ```

## Project structure

```
ep_pipeline/
├── config.py                   # MetricsConfig + PromptConfig — all parameters in one place
├── io.py                       # JSONL/CSV loading, parquet/csv checkpoints
├── models.py                   # device selection, spaCy + embedder loading
├── assemble_corpus.py          # build a unified (id, source, text) table from raw inputs
├── ai_imitation/
│   ├── completion_mlx.py       # CompletionMLX — load and generate with an MLX model
│   ├── make_prompts.py         # sample sentence-complete excerpts + build prompts
│   └── run_batch_mlx.py        # run a prompt JSONL through the model, write completions
├── scoring/
│   ├── get_td_linguistic.py    # TextDescriptives metrics
│   ├── get_other_linguistic.py # MATTR, entropy, perplexity
│   ├── get_semantic.py         # embedding-based semantic metrics
│   └── runner.py               # checkpointed map over records (resumable)
└── efa/
    ├── efa.py                  # factorability checks, parallel analysis, fit/apply
    └── metric_taxonomy.py      # feature categories + drop list

ep_pipeline_run_templates/
├── run_ai_continuation.py      # stage 0: generate AI continuations from a corpus
├── run_scoring.py              # stage 1: score a corpus
├── run_efa.py                  # stage 2: fit the factor model
└── run_apply_efa.py            # stage 3: apply the model to new data

tests/
```

## Usage

Copy the relevant template from `ep_pipeline_run_templates/`, set `PROJECT_ROOT` at the top, and run it. The full pipeline runs in four steps:

### Step 0 — Generate AI continuations (`run_ai_continuation.py`)

**Input:** a JSONL or CSV file where each row has at least `id` and `text`.

1. `build_excerpts_and_prompts` samples sentence-complete chunks from each text and builds a continuation prompt for each.
2. `run_batch` loads the MLX model and generates a completion for every prompt, skipping any already done (safe to interrupt and resume).

**Outputs:**
- `excerpts.csv` — human text chunks with metadata, one row per chunk
- `prompts.jsonl` — the prompts sent to the model
- `outputs.jsonl` — AI completions, keyed by the same `id` as the excerpts

### Step 1 — Score human + AI text (`run_scoring.py`)

**Input:** `prompts.jsonl` + `outputs.jsonl` from step 0 (or any JSONL corpus).

`build_text_table` merges them into a unified table with one `"human"` row and one `"ai"` row per chunk `id`. Each scoring function then runs through `map_with_checkpoints`, which saves intermediate results and resumes after interruption.

**Output:** `metrics_full.csv` — one row per (chunk, source) with all linguistic, semantic, and structural metrics merged.

### Step 2 — Fit the factor model (`run_efa.py`)

**Input:** `metrics_full.csv` from step 1.

Scales features, checks factorability (KMO + Bartlett), runs parallel analysis to pick the number of factors, and fits an oblimin-rotated factor model.

**Outputs:**
- `scaler.joblib`, `fa.joblib`, `feature_cols.joblib` — fitted objects for step 3
- `efa_loadings.csv`, `efa_variance.csv`, `top_N_efa_loadings.csv`
- `efa_factor_scores.csv`, `metrics_with_factor_scores.csv`
- `scree.png`

### Step 3 — Apply to new data (`run_apply_efa.py`)

**Input:** a new `metrics_full.csv` + the fitted objects from step 2.

Projects new documents onto the existing factor dimensions without refitting, so scores are comparable across datasets.

## Configuration

Two config dataclasses in `config.py`:

**`MetricsConfig`** — scoring parameters: spaCy model, MATTR/entropy window sizes, trigram entropy settings, embedding chunk size/overlap/batch size, E5 prefix, parallel-analysis permutations, number of top loadings, parallel `n_jobs`, checkpoint frequency, and `seed`.

**`PromptConfig`** — AI continuation parameters: number of chunks per text (`n_chunks`), chunk word range (`chunk_min`/`chunk_max`), sentence-snapping settings (`extra_min`/`extra_max`, `max_tries`), `seed`, and generation parameters (`max_new_tokens`, `temp`, `top_p`).

## Notes

- AI continuation requires Apple Silicon (MLX). All other pipeline stages work on any platform.
- Checkpointing always writes parquet for efficiency but can read either parquet or csv.
- The semantic scorer defines an explicit all-NaN output for texts too short to chunk, so missingness is part of the metric contract rather than silent failure.
- `run_batch_mlx` resumes automatically if interrupted — already-completed IDs are skipped.
- Set `PROJECT_ROOT` in each run template before use.

## License

No license file is currently included. Contact the repository owner before reuse.
