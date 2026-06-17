import json
from pathlib import Path
import pandas as pd

# Loads and saves data, including checkpointing for the runner
def load_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]

def load_csv(path, **kwargs):
    return pd.read_csv(Path(path), **kwargs)

def read_checkpoint(path):
    "Returns the checkpoint as a dateframe if a checkpoint exists, otherwise returns None."
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path) # supports both csv and parquet, but always writes parquet (see write_table)

# Writes a dataframe to the given path, creating parent directories if needed. Supports both csv and parquet, but always writes parquet (for efficiency), even if the path ends with .csv (for compatibility with existing pipelines).
def write_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
        