"""Investigate missing feature values without changing source data."""

import json

import pandas as pd

# Locate the original dataset and report directory.
from retailiq.config import ROOT

SOURCE = ROOT / "data" / "raw" / "features.csv"
REPORT = ROOT / "reports" / "data_quality" / "03_missing_value_analysis.json"

FEATURE_COLUMNS = [
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
]


def analyze_missing_values(df):
    """Summarize missing values by feature, store and year."""

    # Work with a copy to keep the original data unchanged.
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Year"] = data["Date"].dt.year

    # Calculate missing counts and percentages for each feature.
    by_feature = {
        column: {
            "missing_count": int(data[column].isna().sum()),
            "missing_pct": round(float(data[column].isna().mean() * 100), 2),
        }
        for column in FEATURE_COLUMNS
    }

    # Count rows with at least one missing feature.
    missing_any = data[FEATURE_COLUMNS].isna().any(axis=1)

    # Compare missing-value percentages between stores.
    by_store = data.groupby("Store")[FEATURE_COLUMNS].apply(
        lambda group: group.isna().mean().mul(100).round(2)
    )

    # Compare missing-value percentages between years.
    by_year = data.groupby("Year")[FEATURE_COLUMNS].apply(
        lambda group: group.isna().mean().mul(100).round(2)
    )

    return {
        "total_rows": len(data),
        "invalid_dates": int(data["Date"].isna().sum()),
        "rows_with_missing_features": int(missing_any.sum()),
        "rows_with_missing_features_pct": round(float(missing_any.mean() * 100), 2),
        "by_feature": by_feature,
        "by_store_missing_pct": {
            str(store): values for store, values in by_store.to_dict(orient="index").items()
        },
        "by_year_missing_pct": {
            str(int(year)): values
            for year, values in by_year.dropna().to_dict(orient="index").items()
        },
    }


def main():
    """Generate the reproducible missing-value report."""

    # Read the original feature records.
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing source: {SOURCE}")

    features = pd.read_csv(SOURCE)
    results = analyze_missing_values(features)

    # Save the detailed report.
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    # Print a concise summary for verification.
    print("Rows:", results["total_rows"])
    print("Invalid dates:", results["invalid_dates"])
    print(
        "Rows with missing features:",
        results["rows_with_missing_features"],
    )

    print("\nMissing values by feature:")
    for name, info in results["by_feature"].items():
        print(f"{name}: {info['missing_count']} ({info['missing_pct']}%)")

    print("\nMissing percentages by year:")
    for year, values in results["by_year_missing_pct"].items():
        print(year, values)

    print(f"\nReport saved: {REPORT}")


if __name__ == "__main__":
    main()
