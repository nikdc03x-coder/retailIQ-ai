"""Build leakage-safe features for one-week-ahead sales forecasting."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/retail_analytical.parquet"
OUTPUT = ROOT / "13_results/sales_forecasting/05_model_features.parquet"


def build_features(data):
    """Create store-level sales history and calendar features."""

    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    # Confirm that holiday flags are consistent within each store-week.
    holiday_counts = (
        data.groupby(["Store", "Date"])["IsHoliday"].nunique()
    )

    if (holiday_counts != 1).any():
        raise ValueError("Conflicting holiday flags within a store-week")

    # Aggregate department sales to one row per store-week.
    weekly = (
        data.groupby(["Store", "Date"], as_index=False)
        .agg(
            Weekly_Store_Sales=("Weekly_Sales", "sum"),
            IsHoliday=("IsHoliday", "first"),
        )
        .sort_values(["Store", "Date"])
        .reset_index(drop=True)
    )

    # Calendar information is known before the target week.
    weekly["Month"] = weekly["Date"].dt.month
    weekly["WeekOfYear"] = weekly["Date"].dt.isocalendar().week.astype(int)
    weekly["Year"] = weekly["Date"].dt.year

    # Lag features contain only sales from previous weeks.
    grouped = weekly.groupby("Store")["Weekly_Store_Sales"]

    for lag in [1, 2, 4]:
        weekly[f"Lag_{lag}"] = grouped.shift(lag)

    # The rolling window excludes the current target week.
    weekly["Rolling_Mean_4"] = (
        weekly.groupby("Store")["Weekly_Store_Sales"]
        .transform(
            lambda sales: sales.shift(1).rolling(4).mean()
        )
    )

    weekly["Rolling_Std_4"] = (
        weekly.groupby("Store")["Weekly_Store_Sales"]
        .transform(
            lambda sales: sales.shift(1).rolling(4).std()
        )
    )

    # Four prior weeks are required for all selected features.
    required = [
        "Lag_1",
        "Lag_2",
        "Lag_4",
        "Rolling_Mean_4",
        "Rolling_Std_4",
    ]

    return weekly.dropna(subset=required).reset_index(drop=True)


def main():
    """Generate and save the forecasting feature dataset."""

    if not SOURCE.is_file():
        raise FileNotFoundError("Run Phase 02 first.")

    data = pd.read_parquet(SOURCE)
    features = build_features(data)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT, index=False)

    print("Feature engineering completed")
    print("Rows:", len(features))
    print("Stores:", features["Store"].nunique())
    print("Weeks:", features["Date"].nunique())

    print("\nFeature columns:")
    print(features.columns.tolist())

    print("\nFirst five rows:")
    print(features.head().to_string(index=False))

    print("\nOutput:", OUTPUT)


if __name__ == "__main__":
    main()