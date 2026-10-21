"""Detect unusual retail sales patterns with Isolation Forest."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from retailiq.config import ROOT

PREDICTIONS = ROOT / "reports/sales_forecasting/10_xgboost_predictions.csv"

FEATURE_DATA = ROOT / "reports/sales_forecasting/05_model_features.parquet"

OUTPUT = ROOT / "reports/anomaly_detection"

FEATURES = [
    "Absolute_Residual",
    "Relative_Error",
    "Weekly_Sales_Change",
    "Log_Sales_Volume",
]


def load_data():
    """Combine saved forecasts with historical sales features."""

    predictions = pd.read_csv(PREDICTIONS)
    features = pd.read_parquet(FEATURE_DATA)

    predictions["Date"] = pd.to_datetime(predictions["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    required = {"Store", "Date", "Lag_1"}

    if not required.issubset(features.columns):
        raise ValueError("Historical features are incomplete.")

    history = features[["Store", "Date", "Lag_1"]].copy()

    data = predictions.merge(
        history,
        on=["Store", "Date"],
        how="inner",
        validate="one_to_one",
    )

    if len(data) != len(predictions):
        raise ValueError("Some forecast observations lack history.")

    return data


def build_features(data):
    """Create numerical signals describing sales deviations."""

    result = data.copy()

    result["Residual"] = result["Weekly_Store_Sales"] - result["Prediction"]

    result["Absolute_Residual"] = result["Residual"].abs()

    # Prevent division by zero for small or zero sales values.
    denominator = result["Prediction"].abs().clip(lower=1.0)

    result["Relative_Error"] = result["Absolute_Residual"] / denominator

    result["Weekly_Sales_Change"] = (result["Weekly_Store_Sales"] - result["Lag_1"]) / result[
        "Lag_1"
    ].abs().clip(lower=1.0)

    result["Log_Sales_Volume"] = np.log1p(result["Weekly_Store_Sales"].clip(lower=0))

    if not np.isfinite(result[FEATURES].to_numpy()).all():
        raise ValueError("Invalid anomaly detection features.")

    return result


def main():
    """Fit on validation data and score held-out test weeks."""

    data = build_features(load_data())

    validation = data[data["Split"] == "Validation"].copy()

    test = data[data["Split"] == "Test"].copy()

    if validation.empty or test.empty:
        raise ValueError("Validation or test data missing.")

    if validation["Date"].max() >= test["Date"].min():
        raise ValueError("Chronological split is invalid.")

    # Fit scaling only on validation observations.
    scaler = StandardScaler()

    x_validation = scaler.fit_transform(validation[FEATURES])

    x_test = scaler.transform(test[FEATURES])

    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_validation)

    # sklearn: -1 means anomaly, +1 means normal.
    test["Isolation_Label"] = model.predict(x_test)

    test["Is_Anomaly"] = test["Isolation_Label"] == -1

    # Lower decision scores indicate more unusual observations.
    test["Anomaly_Score"] = model.decision_function(x_test)

    anomalies = test[test["Is_Anomaly"]].sort_values("Anomaly_Score")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    test.to_csv(
        OUTPUT / "04_isolation_forest_test_scores.csv",
        index=False,
    )

    anomalies.to_csv(
        OUTPUT / "05_isolation_forest_anomalies.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "Method": "Isolation Forest",
                "Calibration_Split": "Validation",
                "Contamination": 0.02,
                "Validation_Observations": len(validation),
                "Test_Observations": len(test),
                "Anomalies": len(anomalies),
                "Anomaly_Rate_Pct": 100 * len(anomalies) / len(test),
            }
        ]
    )

    summary.to_csv(
        OUTPUT / "06_isolation_forest_summary.csv",
        index=False,
    )

    print("Isolation Forest anomaly detection completed")

    print("\nValidation observations:", len(validation))
    print("Test observations:", len(test))

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
        "Relative_Error",
        "Anomaly_Score",
    ]

    print(anomalies[columns].head(10).to_string(index=False))

    print("\nSaved reports to:", OUTPUT)


if __name__ == "__main__":
    main()
