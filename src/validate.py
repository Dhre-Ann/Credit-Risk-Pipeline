from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from src.ingest import load_raw_data
from src.transform import clean_and_engineer_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "credit_risk_model.joblib"


def validate_trained_model() -> Dict[str, Any]:
    """
    Load the persisted model and run a quick evaluation on the full dataset.
    This is intended as a lightweight health check rather than a full study.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Train the model first by running `python -m src.train`."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    df = load_raw_data()
    X, y, _ = clean_and_engineer_features(df)

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y, proba)
    report = classification_report(y, preds, output_dict=True)

    return {
        "auc": float(auc),
        "report": report,
    }


if __name__ == "__main__":
    metrics = validate_trained_model()
    print("AUC:", metrics["auc"])
    print("Classification report:")
    for label, stats in metrics["report"].items():
        print(label, stats)

