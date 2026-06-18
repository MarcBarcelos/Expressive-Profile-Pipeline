from pathlib import Path
import joblib

from ep_pipeline.config import MetricsConfig
from ep_pipeline.io import load_csv, write_table
from ep_pipeline.efa.efa import apply_efa

# PATHS FOR THIS PROJECT
PROJECT_ROOT = Path("/path/to/your/project")
IN_FP    = PROJECT_ROOT / "outputs" / "AI_Comparison" / "results" / "data" / "new_metrics_full.csv"
EFA_DIR  = PROJECT_ROOT / "outputs" / "AI_Comparison" / "results" / "efa"
OUT_DIR  = PROJECT_ROOT / "outputs" / "AI_Comparison" / "results" / "efa_applied"
OUT_FP   = OUT_DIR / "metrics_with_factor_scores.csv"

cfg = MetricsConfig()

# LOAD FITTED OBJECTS FROM run_efa.py
scaler       = joblib.load(EFA_DIR / "scaler.joblib")
fa           = joblib.load(EFA_DIR / "fa.joblib")
feature_cols = joblib.load(EFA_DIR / "feature_cols.joblib")

# LOAD NEW DATA
all_metrics = load_csv(IN_FP)

# APPLY
factor_scores = apply_efa(all_metrics, feature_cols, scaler, fa)

# MERGE SCORES BACK ONTO all_metrics
all_metrics_with_scores = all_metrics.merge(factor_scores, on=["id", "source"], how="left")

# SAVE OUTPUTS
write_table(factor_scores, OUT_DIR / "efa_factor_scores.csv")
write_table(all_metrics_with_scores, OUT_FP)
print(f"Saved: {all_metrics_with_scores.shape} -> {OUT_FP}")
