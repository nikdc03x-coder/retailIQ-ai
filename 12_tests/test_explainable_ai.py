
"""Tests for the saved SHAP explanation reports."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "13_results/explainable_ai"

GLOBAL_PATH = REPORTS / "01_global_shap_importance.csv"
LOCAL_PATH = REPORTS / "02_local_shap_contributions.csv"

FEATURES = [
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


def test_shap_reports_exist():
    """Both explanation reports should be available."""

    assert GLOBAL_PATH.is_file()
    assert LOCAL_PATH.is_file()


def test_global_shap_report():
    """Every forecasting feature should have a valid SHAP score."""

    data = pd.read_csv(GLOBAL_PATH)

    assert len(data) == 10
    assert set(data["Feature"]) == set(FEATURES)
    assert data["Feature"].is_unique

    assert np.isfinite(
        data[["Mean_Absolute_SHAP", "Mean_SHAP"]].to_numpy()
    ).all()

    assert (data["Mean_Absolute_SHAP"] >= 0).all()


def test_local_shap_report():
    """Each test store-week should have one complete explanation."""

    data = pd.read_csv(LOCAL_PATH)

    assert len(data) == 990
    assert data["Store"].nunique() == 45
    assert not data.duplicated(["Store", "Date"]).any()

    shap_columns = [f"SHAP_{name}" for name in FEATURES]

    required = [
        "Weekly_Store_Sales",
        "Prediction",
        "SHAP_Base_Value",
        "SHAP_Reconstructed_Prediction",
        *shap_columns,
    ]

    assert set(required).issubset(data.columns)
    assert data[required].notna().all().all()
    assert np.isfinite(data[required].to_numpy()).all()


def test_shap_reconstructs_predictions():
    """SHAP contributions must reproduce the saved predictions."""

    data = pd.read_csv(LOCAL_PATH)

    shap_columns = [f"SHAP_{name}" for name in FEATURES]

    reconstructed = (
        data["SHAP_Base_Value"].to_numpy()
        + data[shap_columns].sum(axis=1).to_numpy()
    )

    predictions = data["Prediction"].to_numpy()

    assert np.allclose(
        reconstructed,
        predictions,
        rtol=1e-5,
        atol=0.1,
    )

    assert np.allclose(
        reconstructed,
        data["SHAP_Reconstructed_Prediction"].to_numpy(),
        rtol=1e-5,
        atol=0.1,
    )


def test_global_importance_matches_local_values():
    """Global SHAP importance must come from local contributions."""

    global_data = pd.read_csv(GLOBAL_PATH).set_index("Feature")
    local_data = pd.read_csv(LOCAL_PATH)

    for feature in FEATURES:
        values = local_data[f"SHAP_{feature}"]

        calculated = values.abs().mean()
        reported = global_data.loc[feature, "Mean_Absolute_SHAP"]

        assert np.isclose(
            calculated,
            reported,
            rtol=1e-5,
            atol=0.1,
        )
