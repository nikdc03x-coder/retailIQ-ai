"""Train XGBoost using the same chronological forecasting split."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "13_results/sales_forecasting/05_model_features.parquet"
)

BASELINE = (
    ROOT / "13_results/sales_forecasting/04_baseline_predictions.csv"
)

OUTPUT = ROOT / "13_results/sales_forecasting"
MODEL_PATH = ROOT / "models/xgboost_store_sales.joblib"

FEATURE_COLUMNS = [
    "Store",
    "IsHoliday",
    "Month",
    "WeekOfYear",
    "Year",
    "Lag_1",
    "Lag_2",
    "Lag_4",
    "Rolling_Mean_4",
    "Rolling_Std_4",
]


def load_data():
    """Reuse the exact date boundaries from the baseline experiment."""

    data = pd.read_parquet(SOURCE)
    baseline = pd.read_csv(BASELINE)

    data["Date"] = pd.to_datetime(data["Date"])
    baseline["Date"] = pd.to_datetime(baseline["Date"])

    date_splits = baseline[["Date", "Split"]].drop_duplicates()

    if date_splits.duplicated("Date").any():
        raise ValueError("Overlapping chronological split dates.")

    data = data.merge(
        date_splits,
        on="Date",
        how="left",
        validate="many_to_one",
    )

    if data["Split"].isna().any():
        raise ValueError("Missing split labels.")

    return data


def evaluate(actual, predicted):
    """Calculate standard forecasting errors."""

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
    }


def main():
    """Train XGBoost and save its predictions and metrics."""

    data = load_data()

    train = data[data["Split"] == "Train"].copy()
    validation = data[data["Split"] == "Validation"].copy()
    test = data[data["Split"] == "Test"].copy()

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[FEATURE_COLUMNS],
        train["Weekly_Store_Sales"],
    )

    results = []
    predictions = []

    for name, subset in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        predicted = model.predict(subset[FEATURE_COLUMNS])

        results.append({
            "Model": "XGBoost",
            "Split": name,
            "Observations": len(subset),
            **evaluate(subset["Weekly_Store_Sales"], predicted),
        })

        output = subset[
            ["Store", "Date", "Weekly_Store_Sales"]
        ].copy()

        output["Prediction"] = predicted
        output["Split"] = name

        predictions.append(output)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(
        OUTPUT / "09_xgboost_metrics.csv",
        index=False,
    )

    pd.concat(predictions).to_csv(
        OUTPUT / "10_xgboost_predictions.csv",
        index=False,
    )

    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    importance.to_csv(
        OUTPUT / "11_xgboost_importance.csv",
        index=False,
    )

    joblib.dump(model, MODEL_PATH)

    print("XGBoost training completed")

    print("\nEvaluation:")
    print(pd.DataFrame(results).to_string(index=False))

    print("\nFeature importance:")
    print(importance.to_string(index=False))

    print("\nSaved model:", MODEL_PATH)


if __name__ == "__main__":
    main()