# Expressive Profile Pipeline

**Expressive Profile Pipeline** — a Python toolkit for extracting a wide range of stylometric features from text and reducing them to a small set of interpretable expressive dimensions via Exploratory Factor Analysis (EFA).

The pipeline scores each text on dozens of linguistic and semantic metrics, then learns a factor structure across them so that any document (human- or AI-written) can be described by a handful of latent "expressive profile" factors. It was built for comparing the stylistic properties of human and AI-generated text, but works on any corpus of English documents.

## What it does

The workflow has three stages:

1. **Scoring** — compute per-document metrics across three families of features.
2. **EFA** — check factorability, choose the number of factors via parallel analysis, fit an oblique factor model, and produce factor scores + loadings.
3. **Apply** — project new, unseen documents onto the already-fitted factor model.

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

## Project structure

```
ep_pipeline/
├── config.py              # MetricsConfig — all tunable parameters in one dataclass
├── io.py                  # JSONL/CSV loading, parquet/csv checkpoints
├── models.py              # device selection, spaCy + embedder loading
├── assemble_corpus.py     # build a unified (id, source, text) table from raw inputs
├── scoring/
│   ├── get_td_linguistic.py    # TextDescriptives metrics
│   ├── get_other_linguistic.py # MATTR, entropy, perplexity
│   ├── get_semantic.py         # embedding-based semantic metrics
│   └── runner.py               # checkpointed map over records (resumable)
└── efa/
    ├── efa.py             # factorability checks, parallel analysis, fit/apply
    └── metric_taxonomy.py # feature categories + drop list

ep_pipeline_run_templates/
├── run_scoring.py         # stage 1: score a corpus
├── run_efa.py             # stage 2: fit the factor model
└── run_apply_efa.py       # stage 3: apply the model to new data

tests/
```

## Usage

The `ep_pipeline_run_templates/` scripts are working examples. Copy one, set the paths at the top to point at your own data, and run it.

### 1. Score a corpus

Input is JSONL with records carrying an `id` and either full `text` or a `prompt` (+ optional model `completion`). `assemble_corpus.build_text_table` normalizes these into a `(id, source, text)` table, where `source` distinguishes e.g. `human` vs `ai`. Each scoring function runs through `map_with_checkpoints`, which writes intermediate results and resumes after interruption.

```python
from ep_pipeline.config import MetricsConfig
from ep_pipeline.models import load_embedder, load_spacy_model
from ep_pipeline.scoring.runner import map_with_checkpoints
# ... see run_scoring.py for the full example
```

Output: `metrics_full.csv` — one row per (document, source) with all metrics merged.

### 2. Fit the factor model

```bash
python ep_pipeline_run_templates/run_efa.py
```

This scales the features, reports KMO and Bartlett's test (factorability), runs **parallel analysis** (500 permutations by default) to select the number of factors, fits an oblimin-rotated factor model, and saves:

- `scaler.joblib`, `fa.joblib`, `feature_cols.joblib` — the fitted objects
- `efa_loadings.csv`, `efa_variance.csv`, `top_N_efa_loadings.csv`
- `efa_factor_scores.csv` and `metrics_with_factor_scores.csv`
- `scree.png` — scree plot with the parallel-analysis threshold

### 3. Apply to new data

```bash
python ep_pipeline_run_templates/run_apply_efa.py
```

Loads the saved scaler + factor model and projects a new metrics table onto the existing factors, so new documents are scored on the *same* expressive dimensions.

## Configuration

All parameters live in `MetricsConfig` (`config.py`) — spaCy model, MATTR/entropy window sizes, trigram entropy settings, embedding chunk size/overlap/batch size, E5 prefix, parallel-analysis permutations, number of top loadings to report, parallel `n_jobs`, checkpoint frequency, and the random `seed` for reproducibility.

## Notes

- Checkpointing always writes parquet for efficiency but can read either parquet or csv.
- The semantic scorer defines an explicit all-NaN output for texts too short to chunk, so missingness is part of the metric contract rather than silent failure.
- The example scripts contain absolute paths from the original author's machine — replace `PROJECT_ROOT` with your own before running.

## License

No license file is currently included. Contact the repository owner before reuse.
