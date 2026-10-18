"""Analyze historical sales performance by store and department."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/retail_analytical.parquet"
OUTPUT = ROOT / "13_results/business_analytics"


def analyze_stores(data):
    """Summarize sales for each store."""

    stores = (
        data.groupby(["Store", "Type", "Size"], as_index=False)
        .agg(
            Total_Sales=("Weekly_Sales", "sum"),
            Sales_Records=("Weekly_Sales", "size"),
            Active_Weeks=("Date", "nunique"),
            Departments=("Dept", "nunique"),
        )
    )

    # Divide by observed weeks for a fairer comparison.
    stores["Average_Weekly_Sales"] = (
        stores["Total_Sales"] / stores["Active_Weeks"]
    )

    return stores.sort_values(
        "Total_Sales", ascending=False
    ).reset_index(drop=True)


def analyze_departments(data):
    """Summarize sales for each department across stores."""

    departments = (
        data.groupby("Dept", as_index=False)
        .agg(
            Total_Sales=("Weekly_Sales", "sum"),
            Sales_Records=("Weekly_Sales", "size"),
            Stores=("Store", "nunique"),
        )
    )

    departments["Average_Record_Sales"] = (
        departments["Total_Sales"]
        / departments["Sales_Records"]
    )

    return departments.sort_values(
        "Total_Sales", ascending=False
    ).reset_index(drop=True)


def main():
    """Generate and save the performance reports."""

    if not SOURCE.is_file():
        raise FileNotFoundError(
            "Run the Phase 02 transformation first."
        )

    data = pd.read_parquet(SOURCE)

    stores = analyze_stores(data)
    departments = analyze_departments(data)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    stores.to_csv(
        OUTPUT / "02_store_performance.csv",
        index=False,
    )

    departments.to_csv(
        OUTPUT / "03_department_performance.csv",
        index=False,
    )

    print("Store and department analysis completed")
    print("Stores analyzed:", len(stores))
    print("Departments analyzed:", len(departments))

    print("\nTop 5 stores by total sales:")
    print(
        stores[
            ["Store", "Type", "Total_Sales",
             "Average_Weekly_Sales"]
        ].head().to_string(index=False)
    )

    print("\nTop 5 departments by total sales:")
    print(
        departments[
            ["Dept", "Total_Sales", "Stores"]
        ].head().to_string(index=False)
    )

    print("\nReports saved in:", OUTPUT)


if __name__ == "__main__":
    main()