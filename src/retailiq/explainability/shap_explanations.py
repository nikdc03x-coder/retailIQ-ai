"""Explain saved XGBoost forecasts using SHAP feature contributions."""

import joblib
import numpy as np
import pandas as pd
import shap

from retailiq.config import ROOT

MODEL_PATH = ROOT / "models/xgboost_store_sales.joblib"

FEATURE_PATH = ROOT / "reports/sales_forecasting/05_model_features.parquet"

PREDICTION_PATH = ROOT / "reports/sales_forecasting/10_xgboost_predictions.csv"

OUTPUT = ROOT / "reports/explainable_ai"


def load_test_data():
    """Match saved test predictions with their original features."""

    model = joblib.load(MODEL_PATH)

    features = pd.read_parquet(FEATURE_PATH)
    predictions = pd.read_csv(PREDICTION_PATH)

    features["Date"] = pd.to_datetime(features["Date"])
    predictions["Date"] = pd.to_datetime(predictions["Date"])

    # Only explain the observations used in final test evaluation.
    test = predictions[predictions["Split"] == "Test"].copy()

    if test.empty:
        raise ValueError("No test predictions were found.")

    if test.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate store-week predictions found.")

    if features.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate store-week features found.")

    # Merge the original model inputs with saved predictions.
    matched = test.merge(
        features,
        on=["Store", "Date"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_Feature"),
    )

    columns = list(model.feature_names_in_)

    if matched[columns].isna().any().any():
        raise ValueError("Some test observations have missing features.")

    X = matched[columns].copy()

    # Verify that we are explaining the original trained model.
    calculated = model.predict(X)

    if not np.allclose(
        calculated,
        matched["Prediction"].to_numpy(),
        rtol=1e-5,
        atol=0.1,
    ):
        raise ValueError("Model predictions do not match saved results.")

    return model, X, matched, calculated


def calculate_shap(model, X, predictions):
    """Calculate contributions and verify they reproduce predictions."""

    # TreeExplainer is designed for tree-based models such as XGBoost.
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    values = np.asarray(explanation.values)
    baseline = np.broadcast_to(
        np.asarray(explanation.base_values),
        (len(X),),
    )

    if values.shape != (len(X), len(X.columns)):
        raise ValueError("Unexpected SHAP contribution dimensions.")

    # Baseline + all feature contributions should equal the prediction.
    reconstructed = baseline + values.sum(axis=1)

    differences = np.abs(reconstructed - predictions)

    print("Maximum SHAP reconstruction difference:", differences.max())
    print("Mean SHAP reconstruction difference:", differences.mean())

    if not np.allclose(
        reconstructed,
        predictions,
        rtol=1e-5,
        atol=0.1,
    ):
        raise ValueError("SHAP contributions do not reconstruct model predictions.")

    return values, baseline


def save_reports(X, matched, values, baseline):
    """Save global feature importance and local prediction explanations."""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Average absolute contribution measures overall model influence.
    global_report = pd.DataFrame(
        {
            "Feature": X.columns,
            "Mean_Absolute_SHAP": np.abs(values).mean(axis=0),
            "Mean_SHAP": values.mean(axis=0),
        }
    ).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    )

    global_report.to_csv(
        OUTPUT / "01_global_shap_importance.csv",
        index=False,
    )

    # Keep each observation linked to its actual and predicted sales.
    local_report = matched[["Store", "Date", "Weekly_Store_Sales", "Prediction"]].copy()

    local_report["SHAP_Base_Value"] = baseline

    for index, feature in enumerate(X.columns):
        local_report[f"SHAP_{feature}"] = values[:, index]

    local_report["SHAP_Reconstructed_Prediction"] = baseline + values.sum(axis=1)

    local_report.to_csv(
        OUTPUT / "02_local_shap_contributions.csv",
        index=False,
    )

    return global_report, local_report


def main():
    """Run the SHAP analysis without changing the original model."""

    model, X, matched, predictions = load_test_data()

    print("Loaded existing model:", type(model).__name__)
    print("Test observations:", len(X))
    print("Input features:", len(X.columns))

    values, baseline = calculate_shap(model, X, predictions)

    global_report, local_report = save_reports(X, matched, values, baseline)

    print("\nGlobal SHAP importance:")
    print(global_report.to_string(index=False))

    print("\nSaved local explanations:", len(local_report))
    print("Output folder:", OUTPUT)
    print("\nPASS: SHAP analysis completed.")


if __name__ == "__main__":
    main()
