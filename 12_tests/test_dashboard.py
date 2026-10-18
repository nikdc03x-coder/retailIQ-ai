
"""Automated tests for the RetailIQ AI dashboard."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "08_dashboard"

if str(DASHBOARD) not in sys.path:
    sys.path.insert(0, str(DASHBOARD))

from data_loader import load_csv, load_weekly_sales
from pages.forecast_reliability import calculate_store_errors


def test_weekly_sales_report():
    """The overview should load valid historical sales data."""

    data = load_weekly_sales()

    assert not data.empty
    assert data["Date"].notna().all()
    assert data["Date"].is_unique
    assert "Total_Sales" in data.columns


def test_store_performance_report():
    """Store analytics should use the saved business report."""

    data = load_csv(
        "business_analytics/02_store_performance.csv"
    )

    assert not data.empty
    assert {"Store", "Total_Sales"}.issubset(data.columns)
    assert data["Store"].is_unique


def test_forecast_prediction_report():
    """Saved predictions should contain the required fields."""

    data = load_csv(
        "sales_forecasting/10_xgboost_predictions.csv"
    )

    required = {
        "Store",
        "Date",
        "Split",
        "Weekly_Store_Sales",
        "Prediction",
    }

    assert required.issubset(data.columns)
    assert not data.empty
    assert data["Prediction"].notna().all()


def test_store_error_calculation():
    """MAE and WAPE should be calculated per store."""

    sample = pd.DataFrame({
        "Store": [1, 1, 2, 2],
        "Date": [
            "2020-01-01",
            "2020-01-08",
            "2020-01-01",
            "2020-01-08",
        ],
        "Weekly_Store_Sales": [100, 200, 100, 100],
        "Prediction": [90, 180, 100, 80],
    })

    result = calculate_store_errors(sample)

    store_1 = result.loc[result["Store"] == 1].iloc[0]
    store_2 = result.loc[result["Store"] == 2].iloc[0]

    assert np.isclose(store_1["MAE"], 15)
    assert np.isclose(store_1["WAPE"], 10)

    assert np.isclose(store_2["MAE"], 10)
    assert np.isclose(store_2["WAPE"], 10)


def test_anomaly_reports_exist():
    """Anomaly investigation should have its saved research reports."""

    paths = [
        "anomaly_detection/07_detector_comparison.csv",
        "anomaly_detection/08_detector_agreement_summary.csv",
        "anomaly_detection/09_anomaly_investigation_queue.csv",
    ]

    for path in paths:
        data = load_csv(path)
        assert not data.empty


def test_missing_report_raises_error():
    """Missing files should produce an error rather than fake data."""

    with pytest.raises(FileNotFoundError):
        load_csv(
            "business_analytics/this_file_does_not_exist.csv"
        )
