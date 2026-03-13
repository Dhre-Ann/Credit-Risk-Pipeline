from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd


DATE_FORMAT = "%b-%Y"  # e.g. 'Jan-2015'


@dataclass
class FeatureSpec:
    feature_names: List[str]


def _parse_emp_length(emp_length: str) -> float:
    """
    Convert LendingClub-style employment length text into approx. years.
    Examples: '10+ years', '2 years', '< 1 year', 'n/a'
    """
    if pd.isna(emp_length):
        return np.nan

    text = str(emp_length).strip().lower()
    if text in {"n/a", "na", "none"}:
        return np.nan
    if text.startswith("<"):
        return 0.5
    if text.startswith("10+"):
        return 10.0
    if "year" in text:
        try:
            return float(text.split()[0])
        except (ValueError, IndexError):
            return np.nan
    return np.nan


def _parse_rate(rate: str) -> float:
    """
    Convert interest rate text like '13.56%' into numeric percent (13.56).
    """
    if pd.isna(rate):
        return np.nan
    text = str(rate).strip().replace("%", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def _parse_month(date_str: str) -> datetime:
    """
    Parse LendingClub-style month strings like 'Jan-2015'.
    """
    return datetime.strptime(date_str, DATE_FORMAT)


def clean_and_engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, FeatureSpec]:
    """
    Clean the raw LendingClub data and engineer the model-ready features.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with numeric columns.
    y : pd.Series
        Binary target: 1 = default/charged-off, 0 = non-default.
    spec : FeatureSpec
        Metadata describing the feature ordering used for the model.
    """
    work = df.copy()

    # --- Target: define what "default" means ---
    default_statuses = {
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
    }
    work["is_default"] = work["loan_status"].isin(default_statuses).astype(int)

    # --- Core numeric fields ---
    work["loan_amnt"] = pd.to_numeric(work["loan_amnt"], errors="coerce")
    work["annual_inc"] = pd.to_numeric(work["annual_inc"], errors="coerce")
    work["dti"] = pd.to_numeric(work["dti"], errors="coerce")

    # Employment length (years, approximate)
    work["emp_length_years"] = work["emp_length"].apply(_parse_emp_length)

    # Interest rate as numeric percent
    work["int_rate_pct"] = work["int_rate"].apply(_parse_rate)

    # Credit history length in years, derived from earliest_cr_line and issue_d
    def credit_history_years(row) -> float:
        try:
            earliest = _parse_month(str(row["earliest_cr_line"]))
            issue = _parse_month(str(row["issue_d"]))
            months = (issue.year - earliest.year) * 12 + (issue.month - earliest.month)
            return max(months, 0) / 12.0
        except Exception:
            return np.nan

    work["credit_history_years"] = work.apply(credit_history_years, axis=1)

    # Drop rows with missing key fields
    key_cols = [
        "loan_amnt",
        "annual_inc",
        "dti",
        "emp_length_years",
        "int_rate_pct",
        "credit_history_years",
        "is_default",
    ]
    work = work[key_cols].dropna()

    feature_cols = [
        "loan_amnt",
        "annual_inc",
        "credit_history_years",
        "emp_length_years",
        "dti",
        "int_rate_pct",
    ]
    X = work[feature_cols].astype(float)
    y = work["is_default"].astype(int)

    spec = FeatureSpec(feature_names=feature_cols)
    return X, y, spec


def features_from_user_input(
    annual_income: float,
    loan_amount: float,
    credit_history_years: float,
    employment_years: float,
    debt_to_income: float,
    interest_rate_percent: float,
) -> Tuple[np.ndarray, FeatureSpec]:
    """
    Build a feature vector in the same ordering as the training data
    from user-entered values.
    """
    feature_names = [
        "loan_amnt",
        "annual_inc",
        "credit_history_years",
        "emp_length_years",
        "dti",
        "int_rate_pct",
    ]
    values = np.array(
        [
            loan_amount,
            annual_income,
            credit_history_years,
            employment_years,
            debt_to_income,
            interest_rate_percent,
        ],
        dtype=float,
    )
    # Shape (1, n_features) for sklearn
    return values.reshape(1, -1), FeatureSpec(feature_names=feature_names)


if __name__ == "__main__":
    # Small manual test if run directly.
    from ingest import load_raw_data

    df_raw = load_raw_data()
    X, y, spec = clean_and_engineer_features(df_raw)
    print("Features shape:", X.shape)
    print("Positive default rate:", y.mean().round(3))
    print("Feature names:", spec.feature_names)

