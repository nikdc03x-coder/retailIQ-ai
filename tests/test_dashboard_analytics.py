"""Validate the saved reports used by the advanced dashboard charts."""

from pathlib import Path

import pandas as pd
import pytest

from retailiq.dashboard.data_loader import load_csv
from retailiq.dashboard.pages.anomaly_investigation import prepare_anomalies

ROOT = Path(__file__).resolve().parents[1]


def test_model_comparison():
    """Each model must have comparable evaluation results."""

    data = load_csv("sales_forecasting/12_aligned_model_comparison.csv")

    assert len(data) == 9
    assert data["Model"].nunique() == 3

    assert set(data["Split"]) == {"Train", "Validation", "Test"}

    assert not data.duplicated(["Model", "Split"]).any()
    assert (data[["MAE", "RMSE"]] >= 0).all().all()


def test_xgboost_feature_importance():
    """Feature importance should be available for charting."""

    data = load_csv("sales_forecasting/11_xgboost_importance.csv")

    assert len(data) == 10
    assert data["Feature"].is_unique
    assert data["Importance"].notna().all()
    assert (data["Importance"] >= 0).all()


def test_rolling_backtest():
    """Backtesting should include three valid comparison folds."""

    data = load_csv("forecast_reliability/06_rolling_backtest_results.csv")

    assert len(data) == 3
    assert data["Fold"].is_unique
    assert (data["Observations"] > 0).all()

    assert (data["XGBoost_MAE"] < data["Baseline_MAE"]).all()

    assert (data["Improvement_Pct"] > 0).all()


def test_holiday_reliability():
    """Holiday and non-holiday reports must contain observations."""

    data = load_csv("forecast_reliability/04_holiday_error_summary.csv")

    assert len(data) == 2
    assert data["Observations"].sum() == 990
    assert (data["Observations"] > 0).all()
    assert (data[["MAE", "RMSE"]] >= 0).all().all()


def test_anomaly_agreement_counts():
    """Detector counts must reconcile with the test population."""

    data = load_csv("anomaly_detection/08_detector_agreement_summary.csv")

    assert len(data) == 1

    result = data.iloc[0]

    assert result["Both"] + result["Statistical_Only"] + result["Isolation_Only"] == result["Union"]

    assert result["Union"] + result["Neither"] == result["Test_Observations"]

    assert result["Both"] + result["Statistical_Only"] == result["Statistical_Anomalies"]

    assert result["Both"] + result["Isolation_Only"] == result["Isolation_Anomalies"]


def test_anomaly_queue_matches_comparison():
    """The investigation queue must contain exactly the flagged rows."""

    comparison = prepare_anomalies(load_csv("anomaly_detection/07_detector_comparison.csv"))

    queue = prepare_anomalies(load_csv("anomaly_detection/09_anomaly_investigation_queue.csv"))

    flagged = comparison[comparison["Any_Anomaly"]].copy()

    keys = ["Store", "Date"]

    assert not flagged.duplicated(keys).any()
    assert not queue.duplicated(keys).any()

    assert set(map(tuple, flagged[keys].to_numpy())) == set(map(tuple, queue[keys].to_numpy()))


def test_detector_filter_counts():
    """Detector filters must return the correct flagged observations."""

    queue = prepare_anomalies(load_csv("anomaly_detection/09_anomaly_investigation_queue.csv"))

    statistical = queue["Statistical_Anomaly"]
    isolation = queue["Isolation_Anomaly"]

    both = queue[statistical & isolation]
    statistical_only = queue[statistical & ~isolation]
    isolation_only = queue[isolation & ~statistical]

    assert len(queue) == 15
    assert len(both) == 4
    assert len(statistical_only) == 6
    assert len(isolation_only) == 5


def test_invalid_anomaly_schema():
    """Missing required columns should raise a clear error."""

    invalid = pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2012-06-01"],
        }
    )

    with pytest.raises(ValueError, match="Missing anomaly columns"):
        prepare_anomalies(invalid)
