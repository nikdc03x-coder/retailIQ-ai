"""Evaluate XGBoost with expanding-window historical backtesting."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "13_results/sales_forecasting/05_model_features.parquet"
OUTPUT = ROOT / "13_results/forecast_reliability"

FEATURES = [
    "Store", "IsHoliday", "Month", "WeekOfYear", "Year",
    "Lag_1", "Lag_2", "Lag_4",
    "Rolling_Mean_4", "Rolling_Std_4",
]

# The original held-out test begins on June 1, 2012.
TEST_START = pd.Timestamp("2012-06-01")
VALIDATION_WEEKS = 8
FOLDS = 3


def main():
    """Train and evaluate XGBoost across three historical folds."""

    data = pd.read_parquet(SOURCE)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Date", "Store"])

    # Never use the original test period for backtesting.
    history = data[data["Date"] < TEST_START].copy()
    dates = sorted(history["Date"].unique())

    required_weeks = FOLDS * VALIDATION_WEEKS

    if len(dates) <= required_weeks:
        raise ValueError("Not enough history for backtesting.")

    results = []

    for fold in range(FOLDS):
        # Each fold advances by eight weeks.
        validation_start = (
            len(dates) - required_weeks
            + fold * VALIDATION_WEEKS
        )
        validation_end = validation_start + VALIDATION_WEEKS

        train_dates = dates[:validation_start]
        validation_dates = dates[validation_start:validation_end]

        train = history[history["Date"].isin(train_dates)]
        validation = history[
            history["Date"].isin(validation_dates)
        ]

        if train["Date"].max() >= validation["Date"].min():
            raise ValueError("Training and validation overlap.")

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
            train[FEATURES],
            train["Weekly_Store_Sales"],
        )

        actual = validation["Weekly_Store_Sales"]
        xgb_predictions = model.predict(validation[FEATURES])

        # The naive baseline uses the previous observed week.
        baseline_predictions = validation["Lag_1"]

        xgb_mae = mean_absolute_error(actual, xgb_predictions)
        baseline_mae = mean_absolute_error(
            actual, baseline_predictions
        )

        improvement = (
            100 * (baseline_mae - xgb_mae) / baseline_mae
            if baseline_mae > 0
            else np.nan
        )

        results.append({
            "Fold": fold + 1,
            "Train_Weeks": len(train_dates),
            "Validation_Weeks": len(validation_dates),
            "Train_End": train["Date"].max().date(),
            "Validation_Start": validation["Date"].min().date(),
            "Validation_End": validation["Date"].max().date(),
            "Observations": len(validation),
            "Baseline_MAE": baseline_mae,
            "XGBoost_MAE": xgb_mae,
            "Improvement_Pct": improvement,
        })

        print(
            f"Fold {fold + 1}: "
            f"Baseline MAE={baseline_mae:.2f}, "
            f"XGBoost MAE={xgb_mae:.2f}, "
            f"Improvement={improvement:.2f}%"
        )

    report = pd.DataFrame(results)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    report.to_csv(
        OUTPUT / "06_rolling_backtest_results.csv",
        index=False,
    )

    print("\nBacktesting completed")
    print("\n", report.to_string(index=False))
    print("\nMean improvement (%):",
          round(report["Improvement_Pct"].mean(), 2))


if __name__ == "__main__":
    main()