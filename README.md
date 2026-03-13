# Credit-Risk-Pipeline

End-to-end credit risk prediction system: data pipeline → ML model → prediction API → web UI.

## How to view the web app

This project uses **FastAPI** (a Python web framework). The app is not a static site, so **VS Code Live Server will not work** — you need to run the Python server.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (once)

```bash
python -m src.train
```

### 3. Start the web app

From the project root:

```bash
python run_app.py
```

Or with uvicorn directly:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open in your browser

Go to: **http://127.0.0.1:8000/**

You should see the Credit Risk Helper form. Submitting it will call the API and show the risk result.

---

## Optional: Run from VS Code

Use **Run and Debug** (or F5) with the "FastAPI" launch config so the app starts in the integrated terminal and you can open the URL in your browser (no Live Server needed).
