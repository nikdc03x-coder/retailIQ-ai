"""
Phase 01 — Inspect Walmart dataset quality.

Input: Original Walmart CSV files
Output: JSON data inspection report
"""

import json

import pandas as pd

# Set project paths.
from retailiq.config import ROOT

RAW_DIR = ROOT / "data" / "raw"
REPORT = ROOT / "reports" / "data_quality" / "01_raw_data_report.json"


def inspect_dataset(df):
    """Collect basic information about a dataset."""

    # Count rows, columns and missing values.
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(count) for col, count in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def main():
    """Inspect the three raw CSV files."""

    # Keep each source dataset separate.
    files = {
        "train": "train.csv",
        "features": "features.csv",
        "stores": "stores.csv",
    }

    report = {}

    # Inspect the data without changing its values.
    for name, filename in files.items():
        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        report[name] = inspect_dataset(df)

    # Save the report for later dashboard use.
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # Print a quick summary in the terminal.
    for name, info in report.items():
        print(
            f"{name}: {info['rows']} rows, "
            f"{info['columns']} columns, "
            f"{info['duplicate_rows']} duplicates"
        )

    print(f"Report saved to: {REPORT}")


if __name__ == "__main__":
    main()
