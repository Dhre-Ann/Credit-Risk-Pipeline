# Collect and import data from source

import os
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "lending_club.csv"


def load_raw_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the historical lending data from CSV.

    Parameters
    ----------
    csv_path:
        Optional path override. If not provided, defaults to the repository
        CSV at data/raw/lending_club.csv.
    """
    path = Path(csv_path) if csv_path is not None else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find lending data at {path}. "
            "Make sure the CSV is present or pass csv_path explicitly."
        )

    # Using low_memory=False so that pandas infers dtypes more consistently.
    df = pd.read_csv(path, low_memory=False)
    return df


if __name__ == "__main__":
    # Simple manual check helper.
    df_head = load_raw_data().head()
    print(df_head[["loan_amnt", "int_rate", "annual_inc", "dti", "emp_length", "earliest_cr_line", "issue_d", "loan_status"]])