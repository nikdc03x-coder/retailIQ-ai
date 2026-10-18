"""Inspect sales values and feature coverage for training dates."""

import json
from pathlib import Path

import pandas as pd


# Locate the original data and report folders.
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORT = (
    ROOT / "13_results" / "data_quality"
    / "04_training_data_inspection.json"
)

MARKDOWNS = [f"MarkDown{i}" for i in range(1, 6)]
FEATURES = MARKDOWNS + ["CPI", "Unemployment"]


def inspect_training_data(sales, features):
    """Check sales values and missing features on matched dates."""

    # Use copies to preserve the original CSV data.
    sales = sales.copy()
    features = features.copy()

    sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
    features["Date"] = pd.to_datetime(features["Date"], errors="coerce")

    # Check the recorded sales values.
    sales_quality = {
        "rows": len(sales),
        "negative_sales": int((sales["Weekly_Sales"] < 0).sum()),
        "zero_sales": int((sales["Weekly_Sales"] == 0).sum()),
        "missing_sales": int(sales["Weekly_Sales"].isna().sum()),
        "invalid_dates": int(sales["Date"].isna().sum()),
    }

    # Join only the columns needed for this investigation.
    matched = sales.merge(
        features[["Store", "Date"] + FEATURES],
        on=["Store", "Date"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    # Count missing features in the actual training records.
    missing_features = {
        column: {
            "count": int(matched[column].isna().sum()),
            "percentage": round(
                float(matched[column].isna().mean() * 100), 2
            ),
        }
        for column in FEATURES
    }

    return {
        "sales_quality": sales_quality,
        "joined_rows": len(matched),
        "unmatched_sales_rows": int(
            (matched["_merge"] == "left_only").sum()
        ),
        "training_missing_features": missing_features,
        "sales_total_preserved": bool(
            matched["Weekly_Sales"].sum()
            == sales["Weekly_Sales"].sum()
        ),
    }


def main():
    """Generate the training-data quality report."""

    # Load the original sales and feature records.
    sales = pd.read_csv(RAW / "train.csv")
    features = pd.read_csv(RAW / "features.csv")

    results = inspect_training_data(sales, features)

    # Save the findings without modifying source files.
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"Report saved: {REPORT}")


if __name__ == "__main__":
    main()