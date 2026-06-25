from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import factor_analyzer.factor_analyzer as _fa_mod
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import sklearn.utils.validation as _skval
import matplotlib.pyplot as plt

def prepare_features(all_metrics, category_map, drop_these, key_cols=["id", "source"], missing_thresh=0.3):
    feature_cols = [c for c in category_map if c in all_metrics.columns and c not in drop_these]
    feat_df = all_metrics[feature_cols].copy()
    too_sparse = feat_df.isnull().mean()
    feature_cols = [c for c in feature_cols if too_sparse[c] < missing_thresh]
    feat_df = feat_df[feature_cols].fillna(feat_df[feature_cols].median())
    final_category_map = {k: v for k, v in category_map.items() if k in feature_cols}
    meta_df = all_metrics[key_cols].reset_index(drop=True)
    return feat_df, feature_cols, final_category_map, meta_df

def scale_metrics(feat_df):
    scaler = StandardScaler()
    scaled_metrics = scaler.fit_transform(feat_df)
    return scaled_metrics, scaler

def check_efa_assumptions(scaled_metrics):
    kmo_all, kmo_model = calculate_kmo(scaled_metrics)              # (≥0.60 adequate, ≥0.80 good)
    chi2, p_val = calculate_bartlett_sphericity(scaled_metrics)     # (p<0.05 required)
    return kmo_model, p_val

def parallel_analysis(scaled_metrics, seed, n_perm=500, percentile=95):
    corr_matrix = np.corrcoef(scaled_metrics, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(corr_matrix)[::-1]
    n_obs, n_vars = scaled_metrics.shape
    rng = np.random.default_rng(seed)
    rand_eigs = np.zeros((n_perm, n_vars))
    for i in range(n_perm):
        rand_data = rng.standard_normal((n_obs, n_vars))
        rand_eigs[i] = np.linalg.eigvalsh(np.corrcoef(rand_data, rowvar=False))[::-1]
    pa_threshold = np.percentile(rand_eigs, percentile, axis=0)
    n_factors = int(np.sum(eigenvalues > pa_threshold))
    return eigenvalues, pa_threshold, n_factors

def plot_scree(eigenvalues, pa_95th, n_factors, viz_path=None):
    n_show = min(20, len(eigenvalues))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, n_show + 1), eigenvalues[:n_show], "o-", label="Actual eigenvalues", color="steelblue")
    ax.plot(range(1, n_show + 1), pa_95th[:n_show], "--", label="Random 95th pct (parallel analysis)", color="tomato")
    ax.axhline(1, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(n_factors, color="tomato", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree plot with parallel analysis")
    ax.legend()
    plt.tight_layout()
    if viz_path:
        plt.savefig(viz_path, dpi=300, bbox_inches="tight")

_orig_check_array = _skval.check_array
def _patched_check_array(X, *args, force_all_finite=None, **kwargs):
    if force_all_finite is not None:
        kwargs["ensure_all_finite"] = force_all_finite
    return _orig_check_array(X, *args, **kwargs)
_fa_mod.check_array = _patched_check_array


def fit_efa(scaled_metrics, feature_cols, category_map, n_factors, rotation="oblimin", method="minres"):
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method)
    fa.fit(scaled_metrics)
    loadings_df = pd.DataFrame(
        fa.loadings_,
        index=feature_cols,
        columns=[f"F{i+1}" for i in range(n_factors)],
    )
    loadings_df["category"] = [category_map.get(m, "?") for m in feature_cols]
    loadings_df = loadings_df.sort_values("category")
    var = fa.get_factor_variance()
    var_df = pd.DataFrame({
        "factor":         [f"F{i+1}" for i in range(n_factors)],
        "SS_loadings":    var[0],
        "pct_var":        var[1] * 100,
        "cumulative_pct": var[2] * 100,
    })
    print(f"EFA fitted with {n_factors} factors ({rotation} rotation, {method} estimation)")
    return fa, loadings_df, var_df

def add_factor_scores(all_metrics, fa, scaled_metrics, meta_df, n_factors, id_cols=["id", "source"]):
    scores = pd.DataFrame(
        fa.transform(scaled_metrics),
        columns=[f"F{i+1}" for i in range(n_factors)],
    )
    scores = pd.concat([meta_df[id_cols].reset_index(drop=True), scores], axis=1)
    return all_metrics.merge(scores, on=id_cols, how="left"), scores

def top_n_factor_loadings(loadings_df, var_df, n_factors, top_n=6):
    factor_cols = [f"F{i+1}" for i in range(n_factors)]
    rows = []
    for f in factor_cols:
        top = loadings_df[f].abs().nlargest(top_n)
        var_row = var_df[var_df["factor"] == f].iloc[0]
        for feat in top.index:
            rows.append({
                "factor":         f,
                "SS_loadings":    var_row["SS_loadings"],
                "pct_var":        var_row["pct_var"],
                "cumulative_pct": var_row["cumulative_pct"],
                "feature":        feat,
                "loading":        loadings_df.loc[feat, f],
                "category":       loadings_df.loc[feat, "category"],
            })
    result = pd.DataFrame(rows)
    return result

def apply_efa(new_metrics, feature_cols, scaler, fa, id_cols=["id", "source"]):
    meta_df = new_metrics[id_cols].reset_index(drop=True)
    feat_df = new_metrics[feature_cols].fillna(new_metrics[feature_cols].median())
    scaled = scaler.transform(feat_df)
    scores = pd.DataFrame(
        fa.transform(scaled),
        columns=[f"F{i+1}" for i in range(fa.n_factors)],
    )
    return pd.concat([meta_df, scores], axis=1)
