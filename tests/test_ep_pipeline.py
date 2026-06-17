"""
Tests for ep_pipeline.
Run with: pytest tests/test_ep_pipeline.py -v

Model-dependent tests (spaCy, sentence-transformers) are skipped automatically
if the models are not present.
"""
import math

import numpy as np
import pandas as pd
import pytest

import os
os.environ.setdefault("EP_E5_MODEL_PATH", 
    "/Users/au728638/Library/CloudStorage/OneDrive-Aarhusuniversitet/Desktop/3. PhD Project/3. Code/models/e5-small")

# ── Helpers ──────────────────────────────────────────────────────────────────

TOKENS = "the quick brown fox jumps over the lazy dog the fox".split()
TEXT   = " ".join(TOKENS * 20)   # ~220 words, enough for most functions


# ── Config ───────────────────────────────────────────────────────────────────

def test_metrics_config_defaults():
    from ep_pipeline.config import MetricsConfig
    cfg = MetricsConfig()
    assert cfg.seed == 2001
    assert cfg.mattr_window == 100
    assert cfg.trigram_test_frac == 0.2
    assert cfg.embed_chunk_size == 100
    assert cfg.n_permutations == 500


# ── IO ───────────────────────────────────────────────────────────────────────

def test_write_and_read_checkpoint_csv(tmp_path):
    from ep_pipeline.io import write_table, read_checkpoint
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    p = tmp_path / "out.csv"
    write_table(df, p)
    loaded = read_checkpoint(p)
    assert list(loaded.columns) == ["a", "b"]
    assert len(loaded) == 2

def _pyarrow_available():
    try:
        import pyarrow
        return True
    except ImportError:
        return False

@pytest.mark.skipif(not _pyarrow_available(), reason="pyarrow not installed")
def test_write_and_read_checkpoint_parquet(tmp_path):
    from ep_pipeline.io import write_table, read_checkpoint
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    p = tmp_path / "out.parquet"
    write_table(df, p)
    loaded = read_checkpoint(p)
    assert len(loaded) == 2

def test_read_checkpoint_missing_returns_none(tmp_path):
    from ep_pipeline.io import read_checkpoint
    assert read_checkpoint(tmp_path / "nonexistent.csv") is None

def test_write_table_creates_parent_dirs(tmp_path):
    from ep_pipeline.io import write_table
    df = pd.DataFrame({"x": [1]})
    p = tmp_path / "a" / "b" / "c.csv"
    write_table(df, p)
    assert p.exists()

def test_load_jsonl(tmp_path):
    from ep_pipeline.io import load_jsonl
    import json
    p = tmp_path / "data.jsonl"
    records = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
    p.write_text("\n".join(json.dumps(r) for r in records))
    loaded = load_jsonl(p)
    assert len(loaded) == 2
    assert loaded[0]["text"] == "hello"


# ── Linguistic metrics ────────────────────────────────────────────────────────

def test_mattr_normal():
    from ep_pipeline.scoring.get_other_linguistic import mattr
    score = mattr(TOKENS, mattr_window=5)
    assert 0.0 < score <= 1.0

def test_mattr_empty():
    from ep_pipeline.scoring.get_other_linguistic import mattr
    assert math.isnan(mattr([], mattr_window=5))

def test_mattr_shorter_than_window():
    from ep_pipeline.scoring.get_other_linguistic import mattr
    score = mattr(["a", "b", "c"], mattr_window=10)
    assert 0.0 < score <= 1.0

def test_windowed_unigram_entropy_normal():
    from ep_pipeline.scoring.get_other_linguistic import windowed_unigram_entropy
    mean, std, ppl = windowed_unigram_entropy(TOKENS, entropy_window=5)
    assert mean > 0
    assert std >= 0
    assert ppl > 1

def test_windowed_unigram_entropy_empty():
    from ep_pipeline.scoring.get_other_linguistic import windowed_unigram_entropy
    mean, std, ppl = windowed_unigram_entropy([], entropy_window=5)
    assert all(math.isnan(v) for v in (mean, std, ppl))

def test_trigram_entropy_normal():
    from ep_pipeline.scoring.get_other_linguistic import trigram_entropy
    tokens = (TOKENS * 10)
    H, ppl = trigram_entropy(tokens, trigram_test_frac=0.2, trigram_alpha=0.1, seed=42)
    assert H > 0
    assert ppl > 1

def test_trigram_entropy_too_short():
    from ep_pipeline.scoring.get_other_linguistic import trigram_entropy
    H, ppl = trigram_entropy(["a", "b"], trigram_test_frac=0.2, trigram_alpha=0.1, seed=42)
    assert math.isnan(H) and math.isnan(ppl)


# ── Semantic helpers (no model needed) ───────────────────────────────────────

def test_chunk_by_words():
    from ep_pipeline.scoring.get_semantic import chunk_by_words
    chunks = chunk_by_words(TEXT, chunk_size=10, overlap=2)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)

def test_l2_normalize():
    from ep_pipeline.scoring.get_semantic import l2_normalize
    v = np.array([3.0, 4.0])
    n = l2_normalize(v)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-6

def test_shannon_entropy_uniform():
    from ep_pipeline.scoring.get_semantic import shannon_entropy
    counts = np.array([10.0, 10.0, 10.0, 10.0])
    H = shannon_entropy(counts)
    assert H > 0

def test_shannon_entropy_single_cluster():
    from ep_pipeline.scoring.get_semantic import shannon_entropy
    counts = np.array([100.0, 0.0, 0.0])
    H = shannon_entropy(counts)
    assert H < 0.01

def test_semantic_keys_match_output():
    """SEMANTIC_KEYS must match the keys returned by compute_semantic_metrics."""
    from ep_pipeline.scoring.get_semantic import SEMANTIC_KEYS, _empty_metrics
    empty = _empty_metrics()
    assert set(SEMANTIC_KEYS) == set(empty.keys())


# ── EFA pipeline (synthetic data, no models) ─────────────────────────────────

@pytest.fixture
def synthetic_all_metrics():
    """50 documents × 10 fake metrics + id/source columns."""
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame(rng.standard_normal((n, 10)),
                      columns=[f"metric_{i}" for i in range(10)])
    df.insert(0, "source", "test")
    df.insert(0, "id", [str(i) for i in range(n)])
    return df

@pytest.fixture
def tiny_taxonomy():
    category_map = {f"metric_{i}": "Category A" for i in range(8)}
    category_map.update({f"metric_{i}": "Category B" for i in range(8, 10)})
    drop_these = {"metric_9"}
    return category_map, drop_these

def test_prepare_features(synthetic_all_metrics, tiny_taxonomy):
    from ep_pipeline.efa.efa import prepare_features
    category_map, drop_these = tiny_taxonomy
    feat_df, feature_cols, final_map, meta_df = prepare_features(
        synthetic_all_metrics, category_map, drop_these
    )
    assert "metric_9" not in feature_cols
    assert set(meta_df.columns) == {"id", "source"}
    assert len(feat_df) == len(synthetic_all_metrics)
    assert not feat_df.isnull().any().any()

def test_scale_metrics(synthetic_all_metrics, tiny_taxonomy):
    from ep_pipeline.efa.efa import prepare_features, scale_metrics
    category_map, drop_these = tiny_taxonomy
    feat_df, _, _, _ = prepare_features(synthetic_all_metrics, category_map, drop_these)
    scaled, scaler = scale_metrics(feat_df)
    assert scaled.shape == feat_df.shape
    assert abs(scaled.mean()) < 0.1

def test_check_efa_assumptions(synthetic_all_metrics, tiny_taxonomy):
    from ep_pipeline.efa.efa import prepare_features, scale_metrics, check_efa_assumptions
    category_map, drop_these = tiny_taxonomy
    feat_df, _, _, _ = prepare_features(synthetic_all_metrics, category_map, drop_these)
    scaled, _ = scale_metrics(feat_df)
    kmo, p = check_efa_assumptions(scaled)
    assert 0 < kmo <= 1

def test_parallel_analysis(synthetic_all_metrics, tiny_taxonomy):
    from ep_pipeline.efa.efa import prepare_features, scale_metrics, parallel_analysis
    category_map, drop_these = tiny_taxonomy
    feat_df, _, _, _ = prepare_features(synthetic_all_metrics, category_map, drop_these)
    scaled, _ = scale_metrics(feat_df)
    eigenvalues, pa_95th, n_factors = parallel_analysis(scaled, n_perm=50, seed=42)
    assert len(eigenvalues) == scaled.shape[1]
    assert n_factors >= 0

def test_fit_efa(synthetic_all_metrics, tiny_taxonomy):
    from ep_pipeline.efa.efa import prepare_features, scale_metrics, fit_efa
    category_map, drop_these = tiny_taxonomy
    feat_df, feature_cols, final_map, _ = prepare_features(
        synthetic_all_metrics, category_map, drop_these
    )
    scaled, _ = scale_metrics(feat_df)
    fa, loadings_df, var_df = fit_efa(scaled, feature_cols, final_map, n_factors=2)
    assert loadings_df.shape == (len(feature_cols), 3)  # F1, F2, category
    assert list(var_df.columns) == ["factor", "SS_loadings", "pct_var", "cumulative_pct"]

def test_add_factor_scores(synthetic_all_metrics, tiny_taxonomy):
    from ep_pipeline.efa.efa import prepare_features, scale_metrics, fit_efa, add_factor_scores
    category_map, drop_these = tiny_taxonomy
    feat_df, feature_cols, final_map, meta_df = prepare_features(
        synthetic_all_metrics, category_map, drop_these
    )
    scaled, _ = scale_metrics(feat_df)
    fa, _, _ = fit_efa(scaled, feature_cols, final_map, n_factors=2)
    all_with_scores, scores_df = add_factor_scores(
        synthetic_all_metrics, fa, scaled, meta_df, n_factors=2
    )
    assert "F1" in all_with_scores.columns
    assert "F2" in all_with_scores.columns
    assert len(all_with_scores) == len(synthetic_all_metrics)

def test_apply_efa(synthetic_all_metrics, tiny_taxonomy):
    from ep_pipeline.efa.efa import prepare_features, scale_metrics, fit_efa, apply_efa
    category_map, drop_these = tiny_taxonomy
    feat_df, feature_cols, final_map, meta_df = prepare_features(
        synthetic_all_metrics, category_map, drop_these
    )
    scaled, scaler = scale_metrics(feat_df)
    fa, _, _ = fit_efa(scaled, feature_cols, final_map, n_factors=2)

    # Apply to "new" data (same data for simplicity)
    result = apply_efa(synthetic_all_metrics, feature_cols, scaler, fa)
    assert "F1" in result.columns
    assert "id" in result.columns
    assert len(result) == len(synthetic_all_metrics)

def test_apply_efa_uses_transform_not_fit(synthetic_all_metrics, tiny_taxonomy):
    """Scores from apply_efa should match scores from add_factor_scores."""
    from ep_pipeline.efa.efa import prepare_features, scale_metrics, fit_efa, add_factor_scores, apply_efa
    category_map, drop_these = tiny_taxonomy
    feat_df, feature_cols, final_map, meta_df = prepare_features(
        synthetic_all_metrics, category_map, drop_these
    )
    scaled, scaler = scale_metrics(feat_df)
    fa, _, _ = fit_efa(scaled, feature_cols, final_map, n_factors=2)

    _, scores_df = add_factor_scores(synthetic_all_metrics, fa, scaled, meta_df, n_factors=2)
    applied = apply_efa(synthetic_all_metrics, feature_cols, scaler, fa)

    np.testing.assert_allclose(
        scores_df[["F1", "F2"]].values,
        applied[["F1", "F2"]].values,
        rtol=1e-5,
    )

def test_top_n_factor_loadings(synthetic_all_metrics, tiny_taxonomy):
    from ep_pipeline.efa.efa import prepare_features, scale_metrics, fit_efa, top_n_factor_loadings
    category_map, drop_these = tiny_taxonomy
    feat_df, feature_cols, final_map, _ = prepare_features(
        synthetic_all_metrics, category_map, drop_these
    )
    scaled, _ = scale_metrics(feat_df)
    _, loadings_df, var_df = fit_efa(scaled, feature_cols, final_map, n_factors=2)
    result = top_n_factor_loadings(loadings_df, var_df, n_factors=2, top_n=3)
    assert len(result) == 6  # 2 factors × 3 features
    assert "loading" in result.columns


# ── Taxonomy ──────────────────────────────────────────────────────────────────

def test_select_features_drops_correctly():
    from ep_pipeline.efa.metric_taxonomy import CATEGORY_OF, DROP_THESE, select_features
    fake_cols = list(CATEGORY_OF.keys())[:10]
    feature_cols, category_map = select_features(fake_cols, drop=set())
    assert all(c in CATEGORY_OF for c in feature_cols)

def test_drop_these_all_in_taxonomy():
    """Every name in DROP_THESE must exist in the taxonomy to avoid silent misses."""
    from ep_pipeline.efa.metric_taxonomy import CATEGORY_OF, DROP_THESE
    unknown = DROP_THESE - set(CATEGORY_OF.keys())
    assert unknown == set(), f"DROP_THESE contains names not in taxonomy: {unknown}"


# ── Runner ────────────────────────────────────────────────────────────────────

def test_map_with_checkpoints(tmp_path):
    from ep_pipeline.scoring.runner import map_with_checkpoints
    records = [{"id": str(i), "source": "test", "val": i} for i in range(5)]
    def score_fn(r):
        return {**r, "doubled": r["val"] * 2}
    result = map_with_checkpoints(
        records, score_fn,
        checkpoint_path=tmp_path / "ckpt.csv",
        key_cols=["id", "source"],
        checkpoint_every=2,
    )
    assert len(result) == 5
    assert list(result["doubled"]) == [0, 2, 4, 6, 8]

def test_map_with_checkpoints_resumes(tmp_path):
    from ep_pipeline.scoring.runner import map_with_checkpoints
    from ep_pipeline.io import write_table
    records = [{"id": str(i), "source": "test", "val": i} for i in range(5)]
    # Pre-populate checkpoint with first 3 records already done
    done = pd.DataFrame([{**r, "doubled": r["val"] * 2} for r in records[:3]])
    ckpt = tmp_path / "ckpt.csv"
    write_table(done, ckpt)

    calls = []
    def score_fn(r):
        calls.append(r["id"])
        return {**r, "doubled": r["val"] * 2}

    result = map_with_checkpoints(records, score_fn, ckpt, ["id", "source"])
    assert len(calls) == 2       # only records 3 and 4 should be processed
    assert len(result) == 5


# ── Model-dependent tests (skipped if models missing) ────────────────────────
# Set EP_E5_MODEL_PATH env var to the path of your E5 model to enable embedder tests.
# e.g. export EP_E5_MODEL_PATH="/path/to/models/e5-small"

import os

def _spacy_model_available(model="en_core_web_sm"):
    try:
        import spacy
        spacy.load(model)
        return True
    except Exception:
        return False

def _e5_model_path():
    return os.environ.get("EP_E5_MODEL_PATH", "")

def _e5_available():
    path = _e5_model_path()
    return bool(path) and os.path.exists(path)

@pytest.fixture(scope="module")
def nlp_tok():
    if not _spacy_model_available():
        pytest.skip("spaCy model not available")
    from ep_pipeline.models import load_spacy_model
    return load_spacy_model("en_core_web_sm", for_tokenizing=True)

@pytest.fixture(scope="module")
def nlp_full():
    if not _spacy_model_available():
        pytest.skip("spaCy model not available")
    from ep_pipeline.models import load_spacy_model
    return load_spacy_model("en_core_web_sm", for_tokenizing=False)

@pytest.fixture(scope="module")
def embedder():
    if not _e5_available():
        pytest.skip("E5 model not available — set EP_E5_MODEL_PATH env var")
    from ep_pipeline.models import load_embedder
    return load_embedder(_e5_model_path())

# spaCy tests

def test_tokenize(nlp_tok):
    from ep_pipeline.scoring.get_td_linguistic import tokenize
    tokens = tokenize("The quick brown fox.", nlp_tok)
    assert isinstance(tokens, list)
    assert "the" in tokens
    assert all(t == t.lower() for t in tokens)

def test_tokenize_long_text(nlp_tok):
    from ep_pipeline.scoring.get_td_linguistic import tokenize
    long_text = "word " * 10000
    tokens = tokenize(long_text, nlp_tok)
    assert len(tokens) > 0

def test_get_td_metrics_returns_dict(nlp_full):
    from ep_pipeline.scoring.get_td_linguistic import get_td_metrics
    result = get_td_metrics("The quick brown fox jumped.", "en_core_web_sm", ["descriptive_stats"])
    assert isinstance(result, dict)
    assert "text" not in result

def test_score_linguistic_full(nlp_tok):
    from ep_pipeline.scoring.get_td_linguistic import tokenize
    from ep_pipeline.scoring.get_other_linguistic import mattr, windowed_unigram_entropy, trigram_entropy
    from ep_pipeline.config import MetricsConfig
    cfg = MetricsConfig()
    r = {"id": "1", "source": "test", "text": TEXT}
    toks = tokenize(r["text"], nlp_tok)
    H_mean, H_std, PPL = windowed_unigram_entropy(toks, entropy_window=cfg.entropy_window)
    H3, PPL3 = trigram_entropy(toks, cfg.trigram_test_frac, cfg.trigram_alpha, cfg.seed)
    mattr_score = mattr(toks, cfg.mattr_window)
    out = {**r, "mattr": mattr_score, "H_unigram_win_mean_nats": H_mean,
           "H_unigram_win_std_nats": H_std, "PPL_unigram_win_mean": PPL,
           "H_3gram_self_nats": H3, "PPL_3gram_self_nats": PPL3}
    assert out["mattr"] > 0
    assert out["H_unigram_win_mean_nats"] > 0

# embedder tests

def test_compute_semantic_metrics(embedder):
    from ep_pipeline.scoring.get_semantic import compute_semantic_metrics, SEMANTIC_KEYS
    from ep_pipeline.config import MetricsConfig
    cfg = MetricsConfig()
    result = compute_semantic_metrics(
        TEXT, embedder,
        chunk_size=cfg.embed_chunk_size,
        overlap=cfg.embed_overlap,
        batch_size=cfg.batch_size,
        prefix=cfg.e5_prefix,
        seed=cfg.seed,
    )
    assert set(result.keys()) == set(SEMANTIC_KEYS)
    assert not all(np.isnan(v) for v in result.values())

def test_compute_semantic_metrics_short_text(embedder):
    from ep_pipeline.scoring.get_semantic import compute_semantic_metrics, _empty_metrics
    from ep_pipeline.config import MetricsConfig
    cfg = MetricsConfig()
    result = compute_semantic_metrics(
        "", embedder,
        chunk_size=cfg.embed_chunk_size,
        overlap=cfg.embed_overlap,
        batch_size=cfg.batch_size,
        prefix=cfg.e5_prefix,
        seed=cfg.seed,
    )
    assert result == _empty_metrics()
