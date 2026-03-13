from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.transform import features_from_user_input


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "credit_risk_model.joblib"

app = FastAPI(
    title="Credit Risk Prediction API",
    description=(
        "Predicts the probability that a loan applicant will default, "
        "based on a small set of financial inputs."
    ),
    version="0.1.0",
)


class BorrowerInput(BaseModel):
    annual_income: float = Field(
        ...,
        description="Your yearly income before tax, in your local currency.",
        example=55000,
    )
    loan_amount: float = Field(
        ...,
        description="The total amount you want to borrow.",
        example=10000,
    )
    credit_history_years: float = Field(
        ...,
        description="How long you have had any credit (credit cards, loans, etc.), in years.",
        example=5.0,
    )
    employment_years: float = Field(
        ...,
        description="How many years you have been in your current or most recent job.",
        example=3.0,
    )
    debt_to_income: float = Field(
        ...,
        description=(
            "Your total monthly debt payments divided by your gross monthly income, "
            "as a percentage. For example, 20 means 20%."
        ),
        example=20.0,
    )
    interest_rate: float = Field(
        ...,
        description=(
            "The interest rate on the loan, as a percentage. "
            "For example, 15.5 means 15.5%."
        ),
        example=15.5,
    )


class PredictionOutput(BaseModel):
    default_probability: float = Field(
        ...,
        description="Predicted probability that this loan will default (0 to 1).",
        example=0.27,
    )
    risk_level: Literal["Low", "Medium", "High"] = Field(
        ...,
        description="Simple risk bucket based on the predicted probability.",
        example="Medium",
    )
    recommendation: str = Field(
        ...,
        description="Plain-language recommendation for how to treat this application.",
        example="Review application",
    )
    explanation: str = Field(
        ...,
        description="Friendly explanation of what this result means for a non-expert.",
    )


def _load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Train the model first by running `python -m src.train` "
            "from the project root."
        )
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    return model


def _risk_bucket(p_default: float) -> tuple[str, str]:
    """
    Map a raw default probability to a human label + recommendation.
    """
    if p_default < 0.2:
        return "Low", "Likely safe to approve, but still follow your normal checks."
    if p_default < 0.5:
        return "Medium", "Consider approving after a careful manual review."
    return "High", "High risk of default – consider declining or asking for stronger terms."


@app.on_event("startup")
def startup_event():
    # Load the model once at startup so that predictions are fast.
    global MODEL
    try:
        MODEL = _load_model()
    except Exception as exc:
        # Keep startup alive but surface a clear error later when /predict is called.
        MODEL = None
        print(f"Warning: could not load model at startup: {exc}")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """
    Very simple guided UI for non-expert users.
    """
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>Credit Risk Helper</title>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        body {
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #0f172a;
          color: #e5e7eb;
          margin: 0;
          padding: 0;
        }
        .page {
          max-width: 900px;
          margin: 0 auto;
          padding: 2rem 1.5rem 3rem;
        }
        .card {
          background: #020617;
          border-radius: 1rem;
          padding: 2rem;
          box-shadow: 0 24px 60px rgba(15,23,42,0.8);
          border: 1px solid rgba(148,163,184,0.25);
        }
        h1 {
          font-size: 1.9rem;
          margin-bottom: 0.25rem;
        }
        .subtitle {
          color: #9ca3af;
          margin-bottom: 1.5rem;
        }
        .step-label {
          font-size: 0.85rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: #60a5fa;
          margin-bottom: 0.75rem;
        }
        .grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
          gap: 1rem 1.5rem;
        }
        .field {
          display: flex;
          flex-direction: column;
          gap: 0.35rem;
        }
        label {
          font-size: 0.9rem;
          font-weight: 500;
        }
        .hint {
          font-size: 0.78rem;
          color: #9ca3af;
        }
        input[type="number"] {
          background: #020617;
          border-radius: 0.6rem;
          border: 1px solid #1f2937;
          padding: 0.55rem 0.7rem;
          color: #e5e7eb;
          font-size: 0.9rem;
          outline: none;
          transition: border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
        }
        input[type="number"]:focus {
          border-color: #60a5fa;
          box-shadow: 0 0 0 1px rgba(59,130,246,0.45);
          background: #020617;
        }
        .units {
          font-size: 0.78rem;
          color: #9ca3af;
        }
        .actions {
          margin-top: 1.5rem;
          display: flex;
          gap: 1rem;
          align-items: center;
          flex-wrap: wrap;
        }
        button {
          background: linear-gradient(135deg, #22c55e, #16a34a);
          color: white;
          border-radius: 999px;
          border: none;
          padding: 0.65rem 1.6rem;
          font-size: 0.95rem;
          font-weight: 600;
          cursor: pointer;
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
          box-shadow: 0 18px 35px rgba(22,163,74,0.45);
        }
        button:disabled {
          opacity: 0.6;
          cursor: default;
          box-shadow: none;
        }
        button span.chevron {
          font-size: 1.15rem;
        }
        .status-text {
          font-size: 0.85rem;
          color: #9ca3af;
        }
        .results {
          margin-top: 2rem;
          padding-top: 1.5rem;
          border-top: 1px solid #111827;
          display: none;
        }
        .pill {
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          padding: 0.3rem 0.75rem;
          border-radius: 999px;
          font-size: 0.78rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }
        .pill-low {
          background: rgba(22,163,74,0.12);
          color: #4ade80;
        }
        .pill-medium {
          background: rgba(234,179,8,0.12);
          color: #fde047;
        }
        .pill-high {
          background: rgba(239,68,68,0.12);
          color: #fca5a5;
        }
        .prob {
          font-size: 2.1rem;
          font-weight: 700;
          margin: 0.5rem 0;
        }
        .rec {
          font-size: 0.98rem;
        }
        .explanation {
          margin-top: 0.75rem;
          font-size: 0.9rem;
          color: #9ca3af;
        }
        .footnote {
          font-size: 0.78rem;
          color: #6b7280;
          margin-top: 1.25rem;
        }
        @media (max-width: 640px) {
          .card {
            padding: 1.4rem 1.3rem 1.6rem;
            border-radius: 0;
            box-shadow: none;
            border-left: none;
            border-right: none;
          }
        }
      </style>
    </head>
    <body>
      <div class="page">
        <div class="card">
          <h1>Credit Risk Helper</h1>
          <div class="subtitle">
            Answer a few questions about the borrower. We’ll estimate the chance
            that the loan will default and translate that into a simple risk level.
          </div>

          <div class="step-label">Step 1 · Tell us about the borrower</div>
          <form id="risk-form">
            <div class="grid">
              <div class="field">
                <label for="annual_income">Annual income</label>
                <span class="hint">Before tax, per year.</span>
                <input id="annual_income" name="annual_income" type="number" min="0" step="100" required />
                <span class="units">Example: 55000</span>
              </div>

              <div class="field">
                <label for="loan_amount">Loan amount</label>
                <span class="hint">How much they want to borrow.</span>
                <input id="loan_amount" name="loan_amount" type="number" min="0" step="100" required />
                <span class="units">Example: 10000</span>
              </div>

              <div class="field">
                <label for="credit_history_years">Credit history length</label>
                <span class="hint">How long they have had any credit at all.</span>
                <input id="credit_history_years" name="credit_history_years" type="number" min="0" step="0.5" required />
                <span class="units">Years · Example: 5</span>
              </div>

              <div class="field">
                <label for="employment_years">Employment length</label>
                <span class="hint">Time in their current or most recent job.</span>
                <input id="employment_years" name="employment_years" type="number" min="0" step="0.5" required />
                <span class="units">Years · Example: 3</span>
              </div>

              <div class="field">
                <label for="debt_to_income">Debt-to-income ratio</label>
                <span class="hint">Total monthly debt payments ÷ monthly income.</span>
                <input id="debt_to_income" name="debt_to_income" type="number" min="0" max="100" step="0.1" required />
                <span class="units">Percent · Example: 20 means 20%</span>
              </div>

              <div class="field">
                <label for="interest_rate">Interest rate</label>
                <span class="hint">Rate on this specific loan.</span>
                <input id="interest_rate" name="interest_rate" type="number" min="0" max="100" step="0.1" required />
                <span class="units">Percent · Example: 15.5</span>
              </div>
            </div>

            <div class="actions">
              <button type="submit" id="submit-btn">
                Get risk estimate
                <span class="chevron">›</span>
              </button>
              <div class="status-text" id="status-text">
                We’ll never store these details. This is a teaching tool, not financial advice.
              </div>
            </div>
          </form>

          <div class="results" id="results">
            <div class="step-label">Step 2 · Understand the result</div>
            <div id="risk-pill"></div>
            <div class="prob" id="prob-display"></div>
            <div class="rec" id="rec-display"></div>
            <div class="explanation" id="explanation-display"></div>
            <div class="footnote">
              This is a simplified model trained on historical lending data.
              Real-world credit decisions should combine data, policy, and human judgement.
            </div>
          </div>
        </div>
      </div>

      <script>
        const form = document.getElementById("risk-form");
        const statusText = document.getElementById("status-text");
        const submitBtn = document.getElementById("submit-btn");
        const resultsEl = document.getElementById("results");
        const riskPillEl = document.getElementById("risk-pill");
        const probDisplay = document.getElementById("prob-display");
        const recDisplay = document.getElementById("rec-display");
        const explanationDisplay = document.getElementById("explanation-display");

        form.addEventListener("submit", async (event) => {
          event.preventDefault();

          const payload = {
            annual_income: Number(form.annual_income.value),
            loan_amount: Number(form.loan_amount.value),
            credit_history_years: Number(form.credit_history_years.value),
            employment_years: Number(form.employment_years.value),
            debt_to_income: Number(form.debt_to_income.value),
            interest_rate: Number(form.interest_rate.value),
          };

          submitBtn.disabled = true;
          statusText.textContent = "Calculating risk using the trained model…";

          try {
            const res = await fetch("/predict", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload),
            });
            if (!res.ok) {
              const errorText = await res.text();
              throw new Error(errorText || "Request failed");
            }
            const data = await res.json();

            const pct = Math.round(data.default_probability * 100);
            probDisplay.textContent = pct + "% chance of default";
            recDisplay.textContent = "Recommendation: " + data.recommendation;
            explanationDisplay.textContent = data.explanation;

            let pillClass = "pill-medium";
            if (data.risk_level === "Low") pillClass = "pill-low";
            if (data.risk_level === "High") pillClass = "pill-high";
            riskPillEl.innerHTML = '<span class="pill ' + pillClass + '">Risk level: ' + data.risk_level + "</span>";

            resultsEl.style.display = "block";
            statusText.textContent = "Scroll down to read what this result means.";
          } catch (err) {
            console.error(err);
            statusText.textContent = "Something went wrong. Please check your inputs and try again.";
            resultsEl.style.display = "none";
          } finally {
            submitBtn.disabled = false;
          }
        });
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: BorrowerInput) -> PredictionOutput:
    """
    Predict default probability for a single borrower.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model is not loaded. Train it first with `python -m src.train` "
                "and restart the API."
            ),
        )

    X, spec = features_from_user_input(
        annual_income=input_data.annual_income,
        loan_amount=input_data.loan_amount,
        credit_history_years=input_data.credit_history_years,
        employment_years=input_data.employment_years,
        debt_to_income=input_data.debt_to_income,
        interest_rate_percent=input_data.interest_rate,
    )

    proba = MODEL.predict_proba(X)[0, 1]
    risk_level, base_recommendation = _risk_bucket(float(proba))

    explanation = (
        f"The model estimates about {proba * 100:.0f}% chance that this loan "
        "would default, based only on the information you entered. "
        "This does not guarantee what will happen, but it helps you compare "
        "this borrower to thousands of similar cases in the past."
    )

    return PredictionOutput(
        default_probability=float(proba),
        risk_level=risk_level,
        recommendation=base_recommendation,
        explanation=explanation,
    )

