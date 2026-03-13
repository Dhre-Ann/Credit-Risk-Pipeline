from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ingest import load_raw_data
from src.transform import clean_and_engineer_features, FeatureSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "credit_risk_model.joblib"


def train_model(random_state: int = 42) -> Dict[str, Any]:
    """
    Train a simple, interpretable credit risk model and persist it.

    Returns
    -------
    info : dict
        Dictionary containing basic training metrics and paths.
    """
    df = load_raw_data()
    X, y, spec = clean_and_engineer_features(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Simple validation metric for sanity check
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": pipeline,
        "feature_spec": spec,
        "training_auc": float(auc),
    }
    joblib.dump(bundle, MODEL_PATH)

    return {
        "model_path": str(MODEL_PATH),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": spec.feature_names,
        "validation_auc": float(auc),
    }


if __name__ == "__main__":
    info = train_model()
    print("Model trained and saved.")
    for k, v in info.items():
        print(f"{k}: {v}")

