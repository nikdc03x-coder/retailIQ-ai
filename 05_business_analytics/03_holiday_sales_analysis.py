"""Compare historical sales in holiday and non-holiday weeks."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/retail_analytical.parquet"
OUTPUT = ROOT / "13_results/business_analytics"


def calculate_store_week_sales(data):
    """Combine department records into one total per store and week."""

    # Each store-week must have one consistent holiday flag.
    holiday_counts = (
        data.groupby(["Store", "Date"])["IsHoliday"].nunique()
    )

    if (holiday_counts != 1).any():
        raise ValueError("Conflicting holiday flags within a store-week")

    return (
        data.groupby(["Store", "Date"], as_index=False)
        .agg(
            Weekly_Store_Sales=("Weekly_Sales", "sum"),
            IsHoliday=("IsHoliday", "first"),
            Department_Records=("Dept", "size"),
        )
        .sort_values(["Date", "Store"])
        .reset_index(drop=True)
    )


def compare_holiday_sales(store_weeks):
    """Compare average store-week sales across holiday categories."""

    summary = (
        store_weeks.groupby("IsHoliday", as_index=False)
        .agg(
            Store_Weeks=("Weekly_Store_Sales", "size"),
            Total_Sales=("Weekly_Store_Sales", "sum"),
            Average_Store_Week_Sales=(
                "Weekly_Store_Sales", "mean"
            ),
            Median_Store_Week_Sales=(
                "Weekly_Store_Sales", "median"
            ),
        )
    )

    return summary.sort_values("IsHoliday").reset_index(drop=True)


def calculate_holiday_dates(store_weeks):
    """Summarize recorded sales on individual holiday dates."""

    holiday = store_weeks[store_weeks["IsHoliday"]].copy()

    return (
        holiday.groupby("Date", as_index=False)
        .agg(
            Total_Sales=("Weekly_Store_Sales", "sum"),
            Stores=("Store", "nunique"),
        )
        .sort_values("Total_Sales", ascending=False)
        .reset_index(drop=True)
    )


def main():
    """Generate holiday comparison reports."""

    if not SOURCE.is_file():
        raise FileNotFoundError("Run Phase 02 first.")

    data = pd.read_parquet(SOURCE)

    store_weeks = calculate_store_week_sales(data)
    summary = compare_holiday_sales(store_weeks)
    holiday_dates = calculate_holiday_dates(store_weeks)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    store_weeks.to_csv(
        OUTPUT / "04_store_week_sales.csv", index=False
    )
    summary.to_csv(
        OUTPUT / "05_holiday_comparison.csv", index=False
    )
    holiday_dates.to_csv(
        OUTPUT / "06_holiday_dates.csv", index=False
    )

    print("Holiday analysis completed")
    print("Store-week observations:", len(store_weeks))

    print("\nHoliday comparison:")
    print(summary.to_string(index=False))

    print("\nTop 5 holiday dates by total sales:")
    print(holiday_dates.head().to_string(index=False))

    print("\nReports saved in:", OUTPUT)


if __name__ == "__main__":
    main()