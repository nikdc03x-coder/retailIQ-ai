"""Audit store-level weekly sales before forecasting."""

import pandas as pd

from retailiq.config import ROOT

SOURCE = ROOT / "data/processed/retail_analytical.parquet"
OUTPUT = ROOT / "reports/sales_forecasting"


def build_store_week_series(data):
    """Combine department sales into store-level weekly totals."""

    store_week = (
        data.groupby(["Store", "Date"], as_index=False)
        .agg(
            Weekly_Store_Sales=("Weekly_Sales", "sum"),
            Departments_Reported=("Dept", "nunique"),
        )
        .sort_values(["Store", "Date"])
        .reset_index(drop=True)
    )

    store_week["Date"] = pd.to_datetime(store_week["Date"])

    return store_week


def audit_store_coverage(store_week):
    """Measure historical coverage for each store."""

    all_weeks = pd.date_range(
        start=store_week["Date"].min(),
        end=store_week["Date"].max(),
        freq="W-FRI",
    )

    coverage = store_week.groupby("Store", as_index=False).agg(
        Observed_Weeks=("Date", "nunique"),
        First_Week=("Date", "min"),
        Last_Week=("Date", "max"),
        Min_Sales=("Weekly_Store_Sales", "min"),
        Max_Sales=("Weekly_Store_Sales", "max"),
    )

    coverage["Expected_Weeks"] = len(all_weeks)
    coverage["Missing_Weeks"] = coverage["Expected_Weeks"] - coverage["Observed_Weeks"]

    return coverage


def main():
    """Build the forecasting series and audit its coverage."""

    if not SOURCE.is_file():
        raise FileNotFoundError("Analytical dataset missing. Run Phase 02 first.")

    data = pd.read_parquet(SOURCE)

    store_week = build_store_week_series(data)
    coverage = audit_store_coverage(store_week)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    store_week.to_csv(
        OUTPUT / "01_store_week_series.csv",
        index=False,
    )

    coverage.to_csv(
        OUTPUT / "02_store_coverage.csv",
        index=False,
    )

    print("Forecasting data audit completed")
    print("Stores:", store_week["Store"].nunique())
    print("Observed store-weeks:", len(store_week))
    print("Unique calendar weeks:", store_week["Date"].nunique())
    print("Stores with missing weeks:", (coverage["Missing_Weeks"] > 0).sum())

    print("\nCoverage summary:")
    print(coverage[["Observed_Weeks", "Expected_Weeks", "Missing_Weeks"]].describe().to_string())

    print("\nFirst five store-week observations:")
    print(store_week.head().to_string(index=False))

    print("\nOutput:", OUTPUT)


if __name__ == "__main__":
    main()
