"""Detect unusual store-week sales using forecast residuals."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "13_results/sales_forecasting/"
    "10_xgboost_predictions.csv"
)

OUTPUT = ROOT / "13_results/anomaly_detection"

Z_THRESHOLD = 3.0


def load_predictions():
    """Read saved XGBoost predictions and validate their structure."""

    data = pd.read_csv(SOURCE)
    data["Date"] = pd.to_datetime(data["Date"])

    required = {
        "Store",
        "Date",
        "Split",
        "Weekly_Store_Sales",
        "Prediction",
    }

    if not required.issubset(data.columns):
        raise ValueError("Prediction file is missing required columns.")

    if data.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate store-week predictions detected.")

    if data[
        ["Weekly_Store_Sales", "Prediction"]
    ].isna().any().any():
        raise ValueError("Missing sales or predictions detected.")

    return data


def fit_threshold(validation):
    """Estimate normal residual variation from validation data."""

    residuals = (
        validation["Weekly_Store_Sales"]
        - validation["Prediction"]
    )

    mean = residuals.mean()
    std = residuals.std(ddof=0)

    if not np.isfinite(std) or std <= 0:
        raise ValueError("Residual standard deviation must be positive.")

    return mean, std


def detect_anomalies(test, mean, std):
    """Flag unusually large standardized test residuals."""

    result = test.copy()

    result["Residual"] = (
        result["Weekly_Store_Sales"] - result["Prediction"]
    )

    result["Z_Score"] = (
        result["Residual"] - mean
    ) / std

    result["Absolute_Z_Score"] = result["Z_Score"].abs()

    result["Is_Anomaly"] = (
        result["Absolute_Z_Score"] >= Z_THRESHOLD
    )

    result["Anomaly_Direction"] = np.select(
        [
            result["Is_Anomaly"] & (result["Residual"] > 0),
            result["Is_Anomaly"] & (result["Residual"] < 0),
        ],
        [
            "Higher than expected",
            "Lower than expected",
        ],
        default="Normal",
    )

    return result


def main():
    """Fit on validation residuals and analyze test observations."""

    data = load_predictions()

    validation = data[data["Split"] == "Validation"].copy()
    test = data[data["Split"] == "Test"].copy()

    if validation.empty or test.empty:
        raise ValueError("Validation or test predictions are missing.")

    if validation["Date"].max() >= test["Date"].min():
        raise ValueError("Validation must precede the test period.")

    mean, std = fit_threshold(validation)
    results = detect_anomalies(test, mean, std)

    anomalies = (
        results[results["Is_Anomaly"]]
        .sort_values("Absolute_Z_Score", ascending=False)
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT / "01_statistical_test_scores.csv",
        index=False,
    )

    anomalies.to_csv(
        OUTPUT / "02_statistical_anomalies.csv",
        index=False,
    )

    summary = pd.DataFrame([{
        "Method": "Global residual Z-score",
        "Threshold": Z_THRESHOLD,
        "Calibration_Split": "Validation",
        "Validation_Observations": len(validation),
        "Test_Observations": len(test),
        "Residual_Mean": mean,
        "Residual_Std": std,
        "Anomalies": len(anomalies),
        "Anomaly_Rate_Pct": 100 * len(anomalies) / len(test),
    }])

    summary.to_csv(
        OUTPUT / "03_statistical_summary.csv",
        index=False,
    )

    print("Statistical anomaly detection completed")

    print("\nValidation observations:", len(validation))
    print("Test observations:", len(test))
    print("Validation residual mean:", round(mean, 2))
    print("Validation residual std:", round(std, 2))

    print("\nAnomalies detected:", len(anomalies))
    print(
        "Anomaly rate (%):",
        round(100 * len(anomalies) / len(test), 2),
    )

    print("\nMost unusual store-weeks:")

    columns = [
        "Store",
        "Date",
        "Weekly_Store_Sales",
        "Prediction",
        "Residual",
        "Z_Score",
        "Anomaly_Direction",
    ]

    print(anomalies[columns].head(10).to_string(index=False))

    print("\nSaved reports to:", OUTPUT)


if __name__ == "__main__":
    main()