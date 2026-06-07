"""
model/persistence.py
====================
Save and reload the trained SAR model + item metadata.
"""
from __future__ import annotations
import logging
from pathlib import Path
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

_ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
_MODEL_FILE    = "sar_model.joblib"
_META_FILE     = "item_metadata.csv"
_METRICS_FILE  = "metrics.json"


def save(model, item_meta: pd.DataFrame, metrics: dict | None = None,
         artifacts_dir: Path = _ARTIFACTS_DIR) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / _MODEL_FILE, compress=3)
    item_meta.to_csv(artifacts_dir / _META_FILE, index=False)
    if metrics:
        import json
        (artifacts_dir / _METRICS_FILE).write_text(json.dumps(metrics, indent=2))
    logger.info("Saved model to %s", artifacts_dir)


def load(artifacts_dir: Path = _ARTIFACTS_DIR):
    """Returns (model, item_meta_df)."""
    model_path = artifacts_dir / _MODEL_FILE
    meta_path  = artifacts_dir / _META_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run train.py first.")
    model     = joblib.load(model_path)
    item_meta = pd.read_csv(meta_path) if meta_path.exists() \
                else pd.DataFrame(columns=["item_id", "title", "genres"])
    logger.info("Loaded model from %s", artifacts_dir)
    return model, item_meta


def exists(artifacts_dir: Path = _ARTIFACTS_DIR) -> bool:
    return (artifacts_dir / _MODEL_FILE).exists()


def load_metrics(artifacts_dir: Path = _ARTIFACTS_DIR) -> dict:
    import json
    p = artifacts_dir / _METRICS_FILE
    return json.loads(p.read_text()) if p.exists() else {}
