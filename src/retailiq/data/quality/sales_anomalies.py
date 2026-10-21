"""Investigate negative and zero weekly sales records."""

import json

import pandas as pd

from retailiq.config import ROOT

SOURCE = ROOT / "data" / "raw" / "train.csv"
REPORT = ROOT / "reports" / "data_quality" / "05_sales_anomaly_analysis.json"


def analyze_sales(df):
    """Describe unusual sales values without removing records."""

    # Copy the source data to avoid modifying it.
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    negative = data[data["Weekly_Sales"] < 0]
    zero = data[data["Weekly_Sales"] == 0]

    # Summarize where negative sales appear most frequently.
    top_stores = negative.groupby("Store").size().sort_values(ascending=False).head(10)

    top_departments = negative.groupby("Dept").size().sort_values(ascending=False).head(10)

    negative_by_year = negative.groupby(negative["Date"].dt.year).size()

    return {
        "total_records": int(len(data)),
        "negative_records": int(len(negative)),
        "zero_records": int(len(zero)),
        "negative_percentage": round(len(negative) / len(data) * 100, 3),
        "minimum_sales": float(data["Weekly_Sales"].min()),
        "maximum_sales": float(data["Weekly_Sales"].max()),
        "negative_sales_total": float(negative["Weekly_Sales"].sum()),
        "negative_by_year": {str(year): int(count) for year, count in negative_by_year.items()},
        "top_10_stores": {str(store): int(count) for store, count in top_stores.items()},
        "top_10_departments": {str(dept): int(count) for dept, count in top_departments.items()},
    }


def main():
    """Generate a reproducible sales anomaly report."""

    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing dataset: {SOURCE}")

    sales = pd.read_csv(SOURCE)
    results = analyze_sales(sales)

    # Store research findings separately from the raw dataset.
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"Report saved: {REPORT}")


if __name__ == "__main__":
    main()
