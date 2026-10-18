"""Evaluate a previous-week sales forecasting baseline."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "13_results/sales_forecasting/01_store_week_series.csv"
)
OUTPUT = ROOT / "13_results/sales_forecasting"


def prepare_baseline(data):
    """Predict each store's sales using its previous week's sales."""

    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values(["Store", "Date"])

    # Shift uses only past sales from the same store.
    data["Prediction"] = (
        data.groupby("Store")["Weekly_Store_Sales"].shift(1)
    )

    return data.dropna(subset=["Prediction"]).reset_index(drop=True)


def split_by_date(data):
    """Split the dataset chronologically using common dates."""

    dates = sorted(data["Date"].unique())
    n = len(dates)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train_dates = dates[:train_end]
    validation_dates = dates[train_end:validation_end]
    test_dates = dates[validation_end:]

    train = data[data["Date"].isin(train_dates)].copy()
    validation = data[data["Date"].isin(validation_dates)].copy()
    test = data[data["Date"].isin(test_dates)].copy()

    return train, validation, test


def evaluate(data, split_name):
    """Calculate standard regression error metrics."""

    actual = data["Weekly_Store_Sales"]
    predicted = data["Prediction"]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    return {
        "Split": split_name,
        "Observations": len(data),
        "MAE": mae,
        "RMSE": rmse,
    }


def main():
    """Evaluate the baseline and save the results."""

    if not SOURCE.is_file():
        raise FileNotFoundError(
            "Store-week series missing. Run the data audit first."
        )

    data = pd.read_csv(SOURCE)
    prepared = prepare_baseline(data)

    train, validation, test = split_by_date(prepared)

    results = pd.DataFrame([
        evaluate(train, "Train"),
        evaluate(validation, "Validation"),
        evaluate(test, "Test"),
    ])

    OUTPUT.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT / "03_baseline_metrics.csv", index=False
    )

    predictions = pd.concat([train, validation, test])
    predictions["Split"] = (
        ["Train"] * len(train)
        + ["Validation"] * len(validation)
        + ["Test"] * len(test)
    )

    predictions.to_csv(
        OUTPUT / "04_baseline_predictions.csv",
        index=False,
    )

    print("Baseline forecasting completed")

    print("\nEvaluation metrics:")
    print(results.to_string(index=False))

    print("\nChronological split:")
    for name, subset in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        print(
            name,
            "| weeks:", subset["Date"].nunique(),
            "| from:", subset["Date"].min().date(),
            "| to:", subset["Date"].max().date(),
        )

    print("\nReports saved in:", OUTPUT)


if __name__ == "__main__":
    main()