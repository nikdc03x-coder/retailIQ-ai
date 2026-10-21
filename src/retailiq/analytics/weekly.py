"""Analyze historical weekly retail sales using Pandas."""

import pandas as pd

# Locate the validated analytical dataset.
from retailiq.config import ROOT

SOURCE = ROOT / "data" / "processed" / "retail_analytical.parquet"

OUTPUT = ROOT / "reports" / "business_analytics" / "01_weekly_sales.csv"


def calculate_weekly_sales(df):
    """Aggregate department sales into weekly business totals."""

    # Group all store-department records by date.
    weekly = (
        df.groupby("Date", as_index=False)
        .agg(
            Total_Sales=("Weekly_Sales", "sum"),
            Average_Department_Sales=("Weekly_Sales", "mean"),
            Sales_Records=("Weekly_Sales", "size"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return weekly


def main():
    """Generate the weekly sales analysis dataset."""

    if not SOURCE.exists():
        raise FileNotFoundError("Analytical dataset missing. Run Phase 02 first.")

    data = pd.read_parquet(SOURCE)
    weekly = calculate_weekly_sales(data)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(OUTPUT, index=False)

    print("Weekly sales analysis completed")
    print("Weeks:", len(weekly))
    print("Total sales:", round(weekly["Total_Sales"].sum(), 2))
    print("Average weekly sales:", round(weekly["Total_Sales"].mean(), 2))
    print("Output:", OUTPUT)

    print("\nFirst five weeks:")
    print(weekly.head().to_string(index=False))


if __name__ == "__main__":
    main()
