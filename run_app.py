"""
Start the Credit Risk web app so you can view it in your browser.

Do NOT use VS Code Live Server for this project. This app is a Python
backend (FastAPI) that serves both the API and the HTML page. You must
run this script (or uvicorn) to see it.

Usage (from project root):
    python run_app.py

Then open:  http://127.0.0.1:8000/
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "credit_risk_model.joblib"


def main():
    if not MODEL_PATH.exists():
        print(
            "No trained model found. Train it first:\n"
            "  python -m src.train\n"
            "Then run this script again."
        )
        sys.exit(1)

    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
