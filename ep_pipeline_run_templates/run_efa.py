from pathlib import Path
import joblib

from ep_pipeline.config import MetricsConfig
from ep_pipeline.io import load_csv, write_table
from ep_pipeline.efa.efa import (
    prepare_features, scale_metrics, check_efa_assumptions,
    parallel_analysis, plot_scree, fit_efa, add_factor_scores,
    top_n_factor_loadings)
from ep_pipeline.efa.metric_taxonomy import CATEGORY_OF, DROP_THESE

# PATHS FOR THIS PROJECT
PROJECT_ROOT = Path("/Users/au728638/Library/CloudStorage/OneDrive-Aarhusuniversitet/Desktop/3. PhD Project/3. Code/Specific_Project")
IN_FP    = PROJECT_ROOT / "outputs" / "AI_Comparison" / "results" / "data" / "metrics_full.csv"
OUT_DIR  = PROJECT_ROOT / "outputs" / "AI_Comparison" / "results" / "efa"
VIS_DIR  = PROJECT_ROOT / "outputs" / "AI_Comparison" / "visualizations"
OUT_FP   = OUT_DIR / "metrics_with_factor_scores.csv"

cfg = MetricsConfig()

# LOAD
all_metrics = load_csv(IN_FP)

# PREPARE
feat_df, feature_cols, final_category_map, meta_df = prepare_features(
    all_metrics, CATEGORY_OF, DROP_THESE)

# SCALE
scaled_metrics, scaler = scale_metrics(feat_df)

# FACTORABILITY CHECKS
kmo, p = check_efa_assumptions(scaled_metrics)
print(f"KMO: {kmo:.3f}  (>=0.60 adequate, >=0.80 good)")
print(f"Bartlett's test: p={p:.2e}  (p<0.05 required)")

# PARALLEL ANALYSIS + SCREE
eigenvalues, pa_95th, n_factors = parallel_analysis(scaled_metrics, n_perm=cfg.n_permutations, 
                                                    seed=cfg.seed)
print(f"Parallel analysis suggests: {n_factors} factors")
plot_scree(eigenvalues, pa_95th, n_factors, viz_path=VIS_DIR / "scree.png")

# FIT EFA
fa, loadings_df, var_df = fit_efa(scaled_metrics, feature_cols, final_category_map, n_factors)
top_n_loadings_df = top_n_factor_loadings(loadings_df, var_df, n_factors)

# SAVE FITTED OBJECTS (for run_apply_efa.py)
OUT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler,       OUT_DIR / "scaler.joblib")
joblib.dump(fa,           OUT_DIR / "fa.joblib")
joblib.dump(feature_cols, OUT_DIR / "feature_cols.joblib")

# SAVE OUTPUTS
write_table(top_n_loadings_df, OUT_DIR / f"top_{n_factors}_efa_loadings.csv")
write_table(loadings_df, OUT_DIR / "efa_loadings.csv")
write_table(var_df,      OUT_DIR / "efa_variance.csv")

all_metrics_with_scores, scores_df = add_factor_scores(all_metrics, fa, scaled_metrics, meta_df, n_factors)

write_table(scores_df, OUT_DIR / "efa_factor_scores.csv")
write_table(all_metrics_with_scores, OUT_FP)
print(f"Saved: {all_metrics_with_scores.shape} -> {OUT_FP}")
