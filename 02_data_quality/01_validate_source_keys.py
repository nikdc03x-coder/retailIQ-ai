"""Check source-table keys before joining Walmart datasets."""

import json
from pathlib import Path

import pandas as pd


# Locate the original data and report folders.
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORT_PATH = (
    ROOT / "13_results" / "data_quality"
    / "02_source_key_validation.json"
)


def validate_keys(df, columns):
    """Count duplicate business keys in a dataset."""

    # A duplicate key can cause incorrect data joins.
    return {
        "rows": len(df),
        "key_columns": columns,
        "missing_key_rows": int(df[columns].isna().any(axis=1).sum()),
        "duplicate_key_rows": int(
            df.duplicated(subset=columns).sum()
        ),
    }


def main():
    """Validate the expected keys in each source table."""

    # Read the three original datasets.
    sales = pd.read_csv(RAW_DIR / "train.csv")
    features = pd.read_csv(RAW_DIR / "features.csv")
    stores = pd.read_csv(RAW_DIR / "stores.csv")

    # Check each dataset at its intended level.
    report = {
        "sales": validate_keys(
            sales, ["Store", "Dept", "Date"]
        ),
        "features": validate_keys(
            features, ["Store", "Date"]
        ),
        "stores": validate_keys(
            stores, ["Store"]
        ),
    }

    # Verify that sales keys can match the other datasets.
    sales_store_dates = sales[
        ["Store", "Date"]
    ].drop_duplicates()

    feature_keys = features[
        ["Store", "Date"]
    ].drop_duplicates()

    unmatched_features = sales_store_dates.merge(
        feature_keys,
        on=["Store", "Date"],
        how="left",
        indicator=True,
    )

    report["relationships"] = {
        "sales_store_dates_without_features": int(
            (unmatched_features["_merge"] == "left_only").sum()
        ),
        "sales_stores_without_metadata": int(
            (~sales["Store"].isin(stores["Store"])).sum()
        ),
    }

    # Save results for reproducible research.
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()