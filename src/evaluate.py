from __future__ import annotations

"""
Small helper module that can be extended to perform
additional offline analyses on the credit risk model.
For now, this simply re-exports validate_trained_model
to provide a clear entry point for evaluation scripts
or notebooks.
"""

from validate import validate_trained_model


__all__ = ["validate_trained_model"]


if __name__ == "__main__":
    metrics = validate_trained_model()
    print("AUC:", metrics["auc"])

