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
os.environ.setdefault("EP_MLX_MODEL_PATH",
    "/Users/au728638/Library/CloudStorage/OneDrive-Aarhusuniversitet/Desktop/3. PhD Project/3. Code/models/Qwen3.5-9B-OptiQ-4bit")
os.environ.setdefault("EP_LEX_DIR",
    "/Users/au728638/Library/CloudStorage/OneDrive-Aarhusuniversitet/Desktop/3. PhD Project/3. Code/models/lexicons")

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


# ── Lexicon-based affect metrics ─────────────────────────────────────────────

# --- score_lexicon ---

def test_score_lexicon_normal():
    from ep_pipeline.scoring.get_lexicon import score_lexicon
    lex = {"quick": 0.8, "brown": 0.4, "fox": 0.6}
    out = score_lexicon(["the", "quick", "brown", "fox"], lex, prefix="t")
    assert abs(out["t_mean"] - (0.8 + 0.4 + 0.6) / 3) < 1e-6
    assert "t_std" in out
    assert abs(out["t_coverage"] - 3 / 4) < 1e-6

def test_score_lexicon_empty_tokens():
    from ep_pipeline.scoring.get_lexicon import score_lexicon
    out = score_lexicon([], {}, prefix="t")
    assert math.isnan(out["t_mean"])
    assert math.isnan(out["t_coverage"])

def test_score_lexicon_no_overlap():
    from ep_pipeline.scoring.get_lexicon import score_lexicon
    out = score_lexicon(["a", "b"], {"x": 1.0}, prefix="t")
    assert math.isnan(out["t_mean"])
    assert out["t_coverage"] == 0.0

def test_score_lexicon_coverage_disabled():
    from ep_pipeline.scoring.get_lexicon import score_lexicon
    out = score_lexicon(["a"], {"a": 1.0}, prefix="t", coverage=False)
    assert "t_coverage" not in out

def test_score_lexicon_custom_agg():
    from ep_pipeline.scoring.get_lexicon import score_lexicon
    out = score_lexicon(["a", "b"], {"a": 1.0, "b": 3.0}, prefix="p",
                        agg=("min", "max"), coverage=False)
    assert out["p_min"] == 1.0
    assert out["p_max"] == 3.0
    assert "p_mean" not in out


# --- vad_metrics ---

_VAD_LEX = {
    "happy": {"valence": 0.9, "arousal": 0.7, "dominance": 0.6},
    "sad":   {"valence": 0.2, "arousal": 0.3, "dominance": 0.3},
    "angry": {"valence": 0.1, "arousal": 0.8, "dominance": 0.7},
}

def test_vad_metrics_normal():
    from ep_pipeline.scoring.get_lexicon import vad_metrics
    out = vad_metrics(["happy", "sad", "unknown"], _VAD_LEX)
    assert abs(out["vad_valence_mean"] - (0.9 + 0.2) / 2) < 1e-6
    assert abs(out["vad_coverage"] - 2 / 3) < 1e-6
    assert "vad_arousal_mean" in out
    assert "vad_dominance_mean" in out

def test_vad_metrics_empty_tokens():
    from ep_pipeline.scoring.get_lexicon import vad_metrics
    out = vad_metrics([], _VAD_LEX)
    assert math.isnan(out["vad_valence_mean"])
    assert math.isnan(out["vad_coverage"])

def test_vad_metrics_no_overlap():
    from ep_pipeline.scoring.get_lexicon import vad_metrics
    out = vad_metrics(["unknown"], _VAD_LEX)
    assert math.isnan(out["vad_valence_mean"])
    assert out["vad_coverage"] == 0.0


# --- emotion_metrics ---

_EMO_LEX = {
    "joyful":  {"joy": 0.9, "trust": 0.4, "anticipation": 0.3},
    "furious": {"anger": 0.95, "disgust": 0.6},
    "scared":  {"fear": 0.8, "surprise": 0.5},
}

def test_emotion_metrics_normal():
    from ep_pipeline.scoring.get_lexicon import emotion_metrics, EMOTIONS
    out = emotion_metrics(["joyful", "furious", "scared"], _EMO_LEX)
    expected_keys = {f"emo_{e}" for e in EMOTIONS} | {"emo_diversity"}
    assert set(out.keys()) == expected_keys
    assert out["emo_joy"] > 0
    assert out["emo_anger"] > 0
    assert out["emo_diversity"] > 0

def test_emotion_metrics_empty_tokens():
    from ep_pipeline.scoring.get_lexicon import emotion_metrics, EMOTIONS
    out = emotion_metrics([], _EMO_LEX)
    assert all(math.isnan(out[f"emo_{e}"]) for e in EMOTIONS)
    assert math.isnan(out["emo_diversity"])

def test_emotion_metrics_no_overlap_gives_zero_diversity():
    from ep_pipeline.scoring.get_lexicon import emotion_metrics
    out = emotion_metrics(["unknown"], _EMO_LEX)
    assert out["emo_diversity"] == 0.0
    assert out["emo_joy"] == 0.0

def test_emotion_metrics_uniform_gives_max_diversity():
    """Uniform distribution across all 8 emotions → entropy = log(8)."""
    from ep_pipeline.scoring.get_lexicon import emotion_metrics, EMOTIONS
    lex = {e: {e: 1.0} for e in EMOTIONS}
    out = emotion_metrics(list(EMOTIONS), lex)
    assert abs(out["emo_diversity"] - math.log(8)) < 1e-5

def test_emotion_metrics_no_sentiment_keys():
    """Removed sentiment columns must not appear in output."""
    from ep_pipeline.scoring.get_lexicon import emotion_metrics
    out = emotion_metrics(["joyful"], _EMO_LEX)
    assert "emo_positive" not in out
    assert "emo_negative" not in out


# --- lexicon loaders ---

def test_load_vad_lexicon(tmp_path):
    from ep_pipeline.scoring.get_lexicon import load_vad_lexicon
    p = tmp_path / "vad.txt"
    p.write_text("word\tvalence\tarousal\tdominance\nhappy\t0.9\t0.7\t0.6\nsad\t0.2\t0.3\t0.3\n")
    lex = load_vad_lexicon(p)
    assert "happy" in lex
    assert abs(lex["happy"]["valence"] - 0.9) < 1e-6
    assert "sad" in lex

def test_load_emotion_lexicon(tmp_path):
    from ep_pipeline.scoring.get_lexicon import load_emotion_lexicon
    p = tmp_path / "emo.txt"
    p.write_text("happy\tjoy\t0.8\nhappy\ttrust\t0.5\nsad\tsadness\t0.9\n")
    lex = load_emotion_lexicon(p)
    assert abs(lex["happy"]["joy"] - 0.8) < 1e-6
    assert abs(lex["sad"]["sadness"] - 0.9) < 1e-6

def test_load_norm_lexicon(tmp_path):
    from ep_pipeline.scoring.get_lexicon import load_norm_lexicon
    p = tmp_path / "norms.csv"
    p.write_text("word,score\nhappy,4.5\nsad,2.1\n")
    lex = load_norm_lexicon(p)
    assert abs(lex["happy"] - 4.5) < 1e-6
    assert abs(lex["sad"] - 2.1) < 1e-6

def test_load_norm_by_name_csv(tmp_path):
    from ep_pipeline.scoring.get_lexicon import load_norm_by_name
    p = tmp_path / "norms.csv"
    p.write_text("Word,Rating\nhappy,4.5\nsad,2.1\n")
    lex = load_norm_by_name(p, term_field="Word", score_field="Rating")
    assert abs(lex["happy"] - 4.5) < 1e-6
    assert abs(lex["sad"] - 2.1) < 1e-6

def test_load_norm_by_name_wrong_columns(tmp_path):
    from ep_pipeline.scoring.get_lexicon import load_norm_by_name
    p = tmp_path / "bad.csv"
    p.write_text("A,B\n1,2\n")
    with pytest.raises(KeyError):
        load_norm_by_name(p, term_field="Word", score_field="Score")


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


# ── Real lexicon integration tests (skip if EP_LEX_DIR not available) ────────
# Set EP_LEX_DIR env var to override the default lexicon directory.

def _lex_dir():
    return os.environ.get("EP_LEX_DIR", "")

def _lex_available():
    return bool(_lex_dir()) and os.path.isdir(_lex_dir())

_LEX_SKIP = pytest.mark.skipif(not _lex_available(),
                                reason="lexicons not available — set EP_LEX_DIR")

@pytest.fixture(scope="module")
def real_vad_lex():
    from ep_pipeline.scoring.get_lexicon import load_vad_lexicon
    from pathlib import Path
    return load_vad_lexicon(
        Path(_lex_dir()) / "NRC-VAD-Lexicon-v2.1" / "NRC-VAD-Lexicon-v2.1.txt"
    )

@pytest.fixture(scope="module")
def real_emo_lex():
    from ep_pipeline.scoring.get_lexicon import load_emotion_lexicon
    from pathlib import Path
    return load_emotion_lexicon(
        Path(_lex_dir()) / "NRC-Emotion-Intensity-Lexicon" / "NRC-Emotion-Intensity-Lexicon-v1.txt"
    )

@_LEX_SKIP
def test_real_vad_lexicon_size(real_vad_lex):
    assert len(real_vad_lex) > 50_000

@_LEX_SKIP
def test_real_vad_lexicon_known_values(real_vad_lex):
    assert abs(real_vad_lex["happy"]["valence"] - 0.985) < 0.01
    assert abs(real_vad_lex["angry"]["valence"] - (-0.756)) < 0.01

@_LEX_SKIP
def test_real_vad_metrics_positive_tokens(real_vad_lex):
    from ep_pipeline.scoring.get_lexicon import vad_metrics
    out = vad_metrics(["happy", "joyful", "pleasant"], real_vad_lex)
    assert out["vad_valence_mean"] > 0.5
    assert out["vad_coverage"] == 1.0

@_LEX_SKIP
def test_real_vad_metrics_negative_tokens(real_vad_lex):
    from ep_pipeline.scoring.get_lexicon import vad_metrics
    out = vad_metrics(["angry", "furious", "terrible"], real_vad_lex)
    assert out["vad_valence_mean"] < 0.0

@_LEX_SKIP
def test_real_emotion_lexicon_size(real_emo_lex):
    assert len(real_emo_lex) > 5_000

@_LEX_SKIP
def test_real_emotion_lexicon_known_values(real_emo_lex):
    assert abs(real_emo_lex["happy"]["joy"] - 0.788) < 0.01
    assert abs(real_emo_lex["furious"]["anger"] - 0.929) < 0.01

@_LEX_SKIP
def test_real_emotion_metrics_joy_tokens(real_emo_lex):
    from ep_pipeline.scoring.get_lexicon import emotion_metrics
    out = emotion_metrics(["happy", "joyful", "delightful"], real_emo_lex)
    assert out["emo_joy"] > out["emo_anger"]
    assert out["emo_diversity"] > 0

@_LEX_SKIP
def test_real_emotion_metrics_anger_tokens(real_emo_lex):
    from ep_pipeline.scoring.get_lexicon import emotion_metrics
    out = emotion_metrics(["angry", "furious", "enraged"], real_emo_lex)
    assert out["emo_anger"] > out["emo_joy"]


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


# ── PromptConfig ──────────────────────────────────────────────────────────────

def test_prompt_config_defaults():
    from ep_pipeline.config import PromptConfig
    cfg = PromptConfig()
    assert cfg.n_chunks == 3
    assert cfg.chunk_min == 500
    assert cfg.chunk_max == 600
    assert cfg.extra_min == 50
    assert cfg.extra_max == 120
    assert cfg.max_tries == 60
    assert cfg.seed == 19
    assert cfg.max_new_tokens == 650
    assert cfg.temp == 0.7
    assert cfg.top_p == 0.9


# ── make_prompts: word helpers ────────────────────────────────────────────────

def test_tokenize_words():
    from ep_pipeline.ai_imitation.make_prompts import tokenize_words
    assert tokenize_words("hello world") == ["hello", "world"]
    assert tokenize_words("") == []

def test_detokenize_words():
    from ep_pipeline.ai_imitation.make_prompts import detokenize_words
    assert detokenize_words(["hello", "world"]) == "hello world"
    assert detokenize_words([]) == ""

def test_count_words():
    from ep_pipeline.ai_imitation.make_prompts import count_words
    assert count_words("hello world") == 2
    assert count_words("  spaced  out  ") == 2
    assert count_words("") == 0


# ── make_prompts: sentence snapping ──────────────────────────────────────────

def test_snap_ends_at_sentence_boundary():
    from ep_pipeline.ai_imitation.make_prompts import _snap_to_sentence_bounds, tokenize_words
    words = tokenize_words(
        "She walked in. He stayed out. They talked later. The end is near now."
    )
    out = _snap_to_sentence_bounds(words, min_len=3, max_len=20)
    assert len(out) >= 3
    assert any(" ".join(out).rstrip().endswith(p) for p in [".", "!", "?", '."', '!"'])

def test_snap_no_punctuation_fallback():
    from ep_pipeline.ai_imitation.make_prompts import _snap_to_sentence_bounds
    words = ["word"] * 15
    out = _snap_to_sentence_bounds(words, min_len=3, max_len=10)
    assert len(out) <= 10

def test_snap_too_short_returns_empty():
    from ep_pipeline.ai_imitation.make_prompts import _snap_to_sentence_bounds, tokenize_words
    out = _snap_to_sentence_bounds(tokenize_words("Hi."), min_len=10, max_len=20)
    assert out == []


# ── make_prompts: sample_chunk ────────────────────────────────────────────────

_LONG_TEXT = (
    "She walked into the room and looked around carefully. "
    "Everything was exactly as she had left it behind. "
    "The books were stacked neatly on the old wooden table. "
    "She sat down in the chair by the window slowly. "
    "Outside, the street was quiet and completely still. "
) * 40  # ~800 words

def test_sample_chunk_length_in_range():
    from ep_pipeline.ai_imitation.make_prompts import sample_chunk, tokenize_words
    words = tokenize_words(_LONG_TEXT)
    chunk = sample_chunk(words, min_len=50, max_len=100)
    assert 50 <= len(chunk) <= 100

def test_sample_chunk_too_short_returns_empty():
    from ep_pipeline.ai_imitation.make_prompts import sample_chunk
    assert sample_chunk(["a", "b"], min_len=50, max_len=100) == []

def test_sample_chunk_never_exceeds_max():
    from ep_pipeline.ai_imitation.make_prompts import sample_chunk, tokenize_words
    words = tokenize_words(_LONG_TEXT)
    for _ in range(10):
        chunk = sample_chunk(words, min_len=20, max_len=50)
        assert len(chunk) <= 50


# ── make_prompts: build_prompt ────────────────────────────────────────────────

def test_build_prompt_contains_passage_marker():
    from ep_pipeline.ai_imitation.make_prompts import build_prompt
    p = build_prompt("Some excerpt text.", 100)
    assert "PASSAGE:" in p
    assert "Some excerpt text." in p

def test_build_prompt_mentions_target_words():
    from ep_pipeline.ai_imitation.make_prompts import build_prompt
    assert "250" in build_prompt("text", 250)


# ── make_prompts: build_excerpts_and_prompts ─────────────────────────────────

@pytest.fixture
def corpus_records():
    para = (
        "The morning light filtered through the curtains slowly. "
        "She rose carefully, her feet finding the cold hard floor. "
        "Outside, the birds had already begun their morning chorus. "
        "She filled the kettle and waited quietly by the window. "
        "The street below was empty and still, as always at this hour. "
    )
    text = (para * 20).strip()  # ~800 words
    return [
        {"id": f"doc_{i}", "text": text, "author": f"author_{i}"}
        for i in range(5)
    ]

def test_build_excerpts_returns_dataframe(corpus_records):
    from ep_pipeline.ai_imitation.make_prompts import build_excerpts_and_prompts
    from ep_pipeline.config import PromptConfig
    excerpts_df, prompts = build_excerpts_and_prompts(
        corpus_records, cfg=PromptConfig(n_chunks=2, chunk_min=50, chunk_max=100)
    )
    assert isinstance(excerpts_df, pd.DataFrame)
    assert isinstance(prompts, list)

def test_build_excerpts_chunk_count(corpus_records):
    from ep_pipeline.ai_imitation.make_prompts import build_excerpts_and_prompts
    from ep_pipeline.config import PromptConfig
    cfg = PromptConfig(n_chunks=2, chunk_min=50, chunk_max=100)
    excerpts_df, prompts = build_excerpts_and_prompts(corpus_records, cfg=cfg)
    assert len(excerpts_df) == len(corpus_records) * 2
    assert len(prompts) == len(excerpts_df)

def test_build_excerpts_extra_fields_passthrough(corpus_records):
    from ep_pipeline.ai_imitation.make_prompts import build_excerpts_and_prompts
    from ep_pipeline.config import PromptConfig
    excerpts_df, _ = build_excerpts_and_prompts(
        corpus_records, cfg=PromptConfig(n_chunks=1, chunk_min=50, chunk_max=100)
    )
    assert "author" in excerpts_df.columns

def test_build_excerpts_ids_match_prompts(corpus_records):
    from ep_pipeline.ai_imitation.make_prompts import build_excerpts_and_prompts
    from ep_pipeline.config import PromptConfig
    excerpts_df, prompts = build_excerpts_and_prompts(
        corpus_records, cfg=PromptConfig(n_chunks=1, chunk_min=50, chunk_max=100)
    )
    assert set(excerpts_df["id"]) == {p["id"] for p in prompts}

def test_build_excerpts_skips_short_text():
    from ep_pipeline.ai_imitation.make_prompts import build_excerpts_and_prompts
    from ep_pipeline.config import PromptConfig
    records = [{"id": "short", "text": "Too short."}]
    excerpts_df, prompts = build_excerpts_and_prompts(
        records, cfg=PromptConfig(chunk_min=500, chunk_max=600)
    )
    assert len(excerpts_df) == 0 and len(prompts) == 0

def test_build_excerpts_custom_keys(corpus_records):
    from ep_pipeline.ai_imitation.make_prompts import build_excerpts_and_prompts
    from ep_pipeline.config import PromptConfig
    renamed = [{"doc_id": r["id"], "content": r["text"]} for r in corpus_records]
    excerpts_df, _ = build_excerpts_and_prompts(
        renamed, id_key="doc_id", text_key="content",
        cfg=PromptConfig(n_chunks=1, chunk_min=50, chunk_max=100),
    )
    assert len(excerpts_df) == len(corpus_records)

def test_build_excerpts_prompt_has_passage_marker(corpus_records):
    from ep_pipeline.ai_imitation.make_prompts import build_excerpts_and_prompts
    from ep_pipeline.config import PromptConfig
    _, prompts = build_excerpts_and_prompts(
        corpus_records, cfg=PromptConfig(n_chunks=1, chunk_min=50, chunk_max=100)
    )
    assert all("PASSAGE:" in p["prompt"] for p in prompts)

def test_build_excerpts_reproducible(corpus_records):
    from ep_pipeline.ai_imitation.make_prompts import build_excerpts_and_prompts
    from ep_pipeline.config import PromptConfig
    cfg = PromptConfig(n_chunks=2, chunk_min=50, chunk_max=100, seed=42)
    df1, _ = build_excerpts_and_prompts(corpus_records, cfg=cfg)
    df2, _ = build_excerpts_and_prompts(corpus_records, cfg=cfg)
    assert list(df1["excerpt"]) == list(df2["excerpt"])

def test_build_excerpts_source_id_is_original_id(corpus_records):
    from ep_pipeline.ai_imitation.make_prompts import build_excerpts_and_prompts
    from ep_pipeline.config import PromptConfig
    excerpts_df, _ = build_excerpts_and_prompts(
        corpus_records, cfg=PromptConfig(n_chunks=1, chunk_min=50, chunk_max=100)
    )
    original_ids = {r["id"] for r in corpus_records}
    assert set(excerpts_df["source_id"]).issubset(original_ids)


# ── run_batch_mlx (CompletionMLX mocked) ─────────────────────────────────────

def _mlx_lm_available():
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False

@pytest.mark.skipif(not _mlx_lm_available(), reason="mlx_lm not installed")
def test_run_batch_writes_output(tmp_path):
    import json
    from unittest.mock import MagicMock, patch
    from ep_pipeline.ai_imitation.run_batch_mlx import run_batch

    prompts_file = tmp_path / "prompts.jsonl"
    output_file  = tmp_path / "outputs.jsonl"
    prompts_file.write_text(
        json.dumps({"id": "doc_1__chunk00", "prompt": "PASSAGE:some text"}) + "\n" +
        json.dumps({"id": "doc_2__chunk00", "prompt": "PASSAGE:other text"}) + "\n"
    )

    mock_model = MagicMock()
    mock_model.tokenizer.apply_chat_template.return_value = "formatted_prompt"
    mock_model.generate.return_value = "AI continuation text."

    with patch("ep_pipeline.ai_imitation.run_batch_mlx.CompletionMLX", return_value=mock_model):
        run_batch(prompts_file, output_file, model_id="fake-model")

    lines = [json.loads(l) for l in output_file.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["id"] == "doc_1__chunk00"
    assert lines[0]["completion"] == "AI continuation text."
    assert "model" in lines[0]
    assert "timestamp_utc" in lines[0]
    assert "max_new_tokens" in lines[0]

@pytest.mark.skipif(not _mlx_lm_available(), reason="mlx_lm not installed")
def test_run_batch_skips_completed(tmp_path):
    import json
    from unittest.mock import MagicMock, patch
    from ep_pipeline.ai_imitation.run_batch_mlx import run_batch

    prompts_file = tmp_path / "prompts.jsonl"
    output_file  = tmp_path / "outputs.jsonl"
    prompts_file.write_text(
        json.dumps({"id": "a", "prompt": "PASSAGE:text1"}) + "\n" +
        json.dumps({"id": "b", "prompt": "PASSAGE:text2"}) + "\n"
    )
    output_file.write_text(json.dumps({"id": "a", "completion": "already done"}) + "\n")

    mock_model = MagicMock()
    mock_model.tokenizer.apply_chat_template.return_value = "prompt"
    mock_model.generate.return_value = "new completion"

    with patch("ep_pipeline.ai_imitation.run_batch_mlx.CompletionMLX", return_value=mock_model):
        run_batch(prompts_file, output_file, model_id="fake-model")

    assert mock_model.generate.call_count == 1  # only "b" was processed
    lines = [json.loads(l) for l in output_file.read_text().splitlines()]
    assert {l["id"] for l in lines} == {"a", "b"}

@pytest.mark.skipif(not _mlx_lm_available(), reason="mlx_lm not installed")
def test_run_batch_respects_generation_params(tmp_path):
    import json
    from unittest.mock import MagicMock, patch
    from ep_pipeline.ai_imitation.run_batch_mlx import run_batch

    prompts_file = tmp_path / "prompts.jsonl"
    output_file  = tmp_path / "outputs.jsonl"
    prompts_file.write_text(json.dumps({"id": "x", "prompt": "PASSAGE:text"}) + "\n")

    mock_model = MagicMock()
    mock_model.tokenizer.apply_chat_template.return_value = "prompt"
    mock_model.generate.return_value = "result"

    with patch("ep_pipeline.ai_imitation.run_batch_mlx.CompletionMLX", return_value=mock_model):
        run_batch(prompts_file, output_file, model_id="fake", max_new_tokens=300, temp=0.5)

    mock_model.generate.assert_called_once_with("prompt", max_new_tokens=300)


# ── CompletionMLX (model-dependent) ──────────────────────────────────────────

def _mlx_model_path():
    return os.environ.get("EP_MLX_MODEL_PATH", "")

def _mlx_model_available():
    path = _mlx_model_path()
    return _mlx_lm_available() and bool(path) and os.path.exists(path)

@pytest.mark.skipif(not _mlx_lm_available(), reason="mlx_lm not installed")
def test_completion_mlx_init_does_not_load():
    from ep_pipeline.ai_imitation.completion_mlx import CompletionMLX
    c = CompletionMLX(model_id="fake-path")
    assert c.model is None
    assert c.tokenizer is None

@pytest.mark.skipif(not _mlx_lm_available(), reason="mlx_lm not installed")
def test_completion_mlx_sampling_params_set():
    from ep_pipeline.ai_imitation.completion_mlx import CompletionMLX
    c = CompletionMLX(model_id="fake", sampling_params={"temp": 0.5, "top_p": 0.8})
    assert c.sampler is not None

@pytest.mark.skipif(not _mlx_lm_available(), reason="mlx_lm not installed")
def test_completion_mlx_no_sampling_params():
    from ep_pipeline.ai_imitation.completion_mlx import CompletionMLX
    c = CompletionMLX(model_id="fake")
    assert c.sampler is None
    assert c.logits_processors is None

@pytest.fixture(scope="module")
def mlx_model():
    if not _mlx_model_available():
        pytest.skip("MLX model not available — set EP_MLX_MODEL_PATH")
    from ep_pipeline.ai_imitation.completion_mlx import CompletionMLX
    c = CompletionMLX(
        model_id=_mlx_model_path(),
        sampling_params={"temp": 0.7, "top_p": 0.9},
    )
    c.load()
    return c

@pytest.mark.skipif(not _mlx_model_available(), reason="MLX model not available — set EP_MLX_MODEL_PATH")
def test_completion_mlx_load(mlx_model):
    assert mlx_model.model is not None
    assert mlx_model.tokenizer is not None

@pytest.mark.skipif(not _mlx_model_available(), reason="MLX model not available — set EP_MLX_MODEL_PATH")
def test_completion_mlx_generate_returns_string(mlx_model):
    result = mlx_model.generate("Hello, please continue:", max_new_tokens=20)
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.skipif(not _mlx_model_available(), reason="MLX model not available — set EP_MLX_MODEL_PATH")
def test_completion_mlx_lazy_load_on_generate():
    from ep_pipeline.ai_imitation.completion_mlx import CompletionMLX
    c = CompletionMLX(model_id=_mlx_model_path())
    assert c.model is None
    c.generate("test", max_new_tokens=5)
    assert c.model is not None
