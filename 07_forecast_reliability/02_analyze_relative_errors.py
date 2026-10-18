"""Compare forecast reliability across stores of different sizes."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "13_results/forecast_reliability/"
    "01_test_prediction_errors.csv"
)

OUTPUT = (
    ROOT / "13_results/forecast_reliability/"
    "05_store_relative_errors.csv"
)


def main():
    """Calculate relative forecasting errors for every store."""

    data = pd.read_csv(SOURCE)

    required = {
        "Store",
        "Date",
        "Weekly_Store_Sales",
        "Prediction",
        "Error",
        "Absolute_Error",
        "Split",
    }

    if not required.issubset(data.columns):
        raise ValueError("Required error columns are missing.")

    if not (data["Split"] == "Test").all():
        raise ValueError("Only test predictions should be analyzed.")

    if data.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate store-week predictions.")

    summary = (
        data.groupby("Store")
        .agg(
            Observations=("Date", "size"),
            MAE=("Absolute_Error", "mean"),
            Total_Absolute_Error=("Absolute_Error", "sum"),
            Total_Signed_Error=("Error", "sum"),
            Total_Actual_Sales=("Weekly_Store_Sales", "sum"),
            Mean_Actual_Sales=("Weekly_Store_Sales", "mean"),
        )
        .reset_index()
    )

    # Absolute sales in the denominator make WAPE robust to
    # any negative actual values that might exist.
    absolute_sales = (
        data.assign(
            Absolute_Actual_Sales=data["Weekly_Store_Sales"].abs()
        )
        .groupby("Store")["Absolute_Actual_Sales"]
        .sum()
        .rename("Total_Absolute_Actual_Sales")
        .reset_index()
    )

    summary = summary.merge(
        absolute_sales,
        on="Store",
        validate="one_to_one",
    )

    denominator = summary["Total_Absolute_Actual_Sales"]

    # Avoid undefined percentages if a store has zero sales.
    summary["WAPE_Pct"] = np.where(
        denominator > 0,
        100 * summary["Total_Absolute_Error"] / denominator,
        np.nan,
    )

    summary["Bias_Pct"] = np.where(
        denominator > 0,
        100 * summary["Total_Signed_Error"] / denominator,
        np.nan,
    )

    summary = summary.sort_values(
        "WAPE_Pct",
        ascending=False,
        na_position="last",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)

    print("Store-relative error analysis completed")

    print("\nFive stores with highest WAPE:")
    print(
        summary[
            ["Store", "Observations", "MAE", "WAPE_Pct", "Bias_Pct"]
        ].head(5).to_string(index=False)
    )

    print("\nFive stores with lowest WAPE:")
    print(
        summary[
            ["Store", "Observations", "MAE", "WAPE_Pct", "Bias_Pct"]
        ].tail(5).to_string(index=False)
    )

    print("\nOverall WAPE (%):")
    total_absolute_error = data["Absolute_Error"].sum()
    total_absolute_sales = data["Weekly_Store_Sales"].abs().sum()

    if total_absolute_sales == 0:
        raise ValueError("Cannot calculate WAPE with zero total sales.")

    print(round(100 * total_absolute_error / total_absolute_sales, 2))

    print("\nSaved report:", OUTPUT)


if __name__ == "__main__":
    main()