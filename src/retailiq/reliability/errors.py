"""Analyze XGBoost forecast errors across stores and historical weeks."""

import numpy as np
import pandas as pd

from retailiq.config import ROOT

PREDICTIONS = ROOT / "reports/sales_forecasting/10_xgboost_predictions.csv"

ANALYTICAL_DATA = ROOT / "data/processed/retail_analytical.parquet"

OUTPUT = ROOT / "reports/forecast_reliability"


def load_data():
    """Load saved predictions and known holiday calendar flags."""

    predictions = pd.read_csv(PREDICTIONS)
    predictions["Date"] = pd.to_datetime(predictions["Date"])

    analytical = pd.read_parquet(ANALYTICAL_DATA)
    analytical["Date"] = pd.to_datetime(analytical["Date"])

    # A store-week can contain multiple departments.
    # Keep one holiday flag for each store and week.
    holiday_counts = analytical.groupby(["Store", "Date"])["IsHoliday"].nunique(dropna=False)

    if (holiday_counts != 1).any():
        raise ValueError("Conflicting holiday flags detected.")

    holidays = analytical[["Store", "Date", "IsHoliday"]].drop_duplicates()

    data = predictions.merge(
        holidays,
        on=["Store", "Date"],
        how="left",
        validate="one_to_one",
    )

    if data["IsHoliday"].isna().any():
        raise ValueError("Missing holiday information.")

    if data.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate store-week predictions.")

    return data


def calculate_errors(data):
    """Calculate signed and absolute forecast errors."""

    result = data.copy()

    # Positive error means the prediction was too high.
    result["Error"] = result["Prediction"] - result["Weekly_Store_Sales"]

    result["Absolute_Error"] = result["Error"].abs()
    result["Squared_Error"] = result["Error"] ** 2

    return result


def summarize_stores(data):
    """Measure how reliably the model forecasts each store."""

    summary = (
        data.groupby("Store")
        .agg(
            Observations=("Absolute_Error", "size"),
            MAE=("Absolute_Error", "mean"),
            Mean_Error=("Error", "mean"),
            Mean_Actual_Sales=("Weekly_Store_Sales", "mean"),
            RMSE=("Squared_Error", lambda x: np.sqrt(x.mean())),
        )
        .reset_index()
        .sort_values("MAE", ascending=False)
    )

    return summary


def summarize_weeks(data):
    """Find calendar weeks with unusually large forecasting errors."""

    return (
        data.groupby("Date")
        .agg(
            Stores=("Store", "nunique"),
            MAE=("Absolute_Error", "mean"),
            Mean_Error=("Error", "mean"),
            RMSE=("Squared_Error", lambda x: np.sqrt(x.mean())),
        )
        .reset_index()
        .sort_values("MAE", ascending=False)
    )


def summarize_holidays(data):
    """Compare holiday and non-holiday forecasting errors."""

    return (
        data.groupby("IsHoliday")
        .agg(
            Observations=("Absolute_Error", "size"),
            MAE=("Absolute_Error", "mean"),
            Mean_Error=("Error", "mean"),
            RMSE=("Squared_Error", lambda x: np.sqrt(x.mean())),
        )
        .reset_index()
    )


def main():
    """Analyze held-out test predictions and save research evidence."""

    data = load_data()

    # Analyze the test period rather than mixing training errors.
    test = data[data["Split"] == "Test"].copy()

    if test.empty:
        raise ValueError("No test predictions available.")

    test = calculate_errors(test)

    stores = summarize_stores(test)
    weeks = summarize_weeks(test)
    holidays = summarize_holidays(test)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    test.to_csv(
        OUTPUT / "01_test_prediction_errors.csv",
        index=False,
    )

    stores.to_csv(
        OUTPUT / "02_store_error_summary.csv",
        index=False,
    )

    weeks.to_csv(
        OUTPUT / "03_week_error_summary.csv",
        index=False,
    )

    holidays.to_csv(
        OUTPUT / "04_holiday_error_summary.csv",
        index=False,
    )

    print("Forecast error analysis completed")

    print("\nTest observations:", len(test))
    print("Stores:", test["Store"].nunique())
    print("Weeks:", test["Date"].nunique())

    print("\nOverall test MAE:")
    print(round(test["Absolute_Error"].mean(), 2))

    print("\nFive stores with highest MAE:")
    print(stores.head(5).to_string(index=False))

    print("\nFive weeks with highest MAE:")
    print(weeks.head(5).to_string(index=False))

    print("\nHoliday comparison:")
    print(holidays.to_string(index=False))

    print("\nSaved reports to:", OUTPUT)


if __name__ == "__main__":
    main()
