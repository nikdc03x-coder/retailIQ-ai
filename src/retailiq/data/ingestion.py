"""
Phase 01 — Load Walmart sales data.

Input: CSV files inside data/raw/
Output: Pandas DataFrames
Next: 02_inspect_raw_data.py
"""

import pandas as pd

# Locate the dataset folder.
from retailiq.config import ROOT

RAW_DIR = ROOT / "data" / "raw"


def load_sales_data():
    """Load the original Walmart CSV files."""

    # Map dataset names to their source files.
    files = {
        "train": "train.csv",
        "features": "features.csv",
        "stores": "stores.csv",
    }

    datasets = {}

    # Read each CSV and check that it exists.
    for name, filename in files.items():
        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        datasets[name] = pd.read_csv(path)

    return datasets


if __name__ == "__main__":
    # Display the size of each loaded dataset.
    datasets = load_sales_data()

    for name, df in datasets.items():
        print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")
