"""Tests for RetailIQ AI forecast reliability experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "13_results/forecast_reliability"


@pytest.fixture(scope="module")
def reports():
    """Load reports produced by the Phase 05 experiments."""

    return {
        "errors": pd.read_csv(
            RESULTS / "01_test_prediction_errors.csv"
        ),
        "stores": pd.read_csv(
            RESULTS / "02_store_error_summary.csv"
        ),
        "weeks": pd.read_csv(
            RESULTS / "03_week_error_summary.csv"
        ),
        "holidays": pd.read_csv(
            RESULTS / "04_holiday_error_summary.csv"
        ),
        "relative": pd.read_csv(
            RESULTS / "05_store_relative_errors.csv"
        ),
        "backtest": pd.read_csv(
            RESULTS / "06_rolling_backtest_results.csv"
        ),
    }


def test_expected_test_observations(reports):
    assert len(reports["errors"]) == 990


def test_test_store_week_keys_unique(reports):
    assert not reports["errors"].duplicated(
        ["Store", "Date"]
    ).any()


def test_absolute_errors_correct(reports):
    data = reports["errors"]

    expected = abs(
        data["Prediction"] - data["Weekly_Store_Sales"]
    )

    assert np.allclose(data["Absolute_Error"], expected)


def test_store_counts_reconcile(reports):
    assert reports["stores"]["Observations"].sum() == 990


def test_week_counts_reconcile(reports):
    data = reports["weeks"]

    assert len(data) == 22
    assert (data["Stores"] == 45).all()


def test_holiday_counts_reconcile(reports):
    assert reports["holidays"]["Observations"].sum() == 990


def test_relative_errors_cover_all_stores(reports):
    data = reports["relative"]

    assert len(data) == 45
    assert data["Store"].is_unique
    assert (data["Observations"] == 22).all()


def test_relative_error_values_valid(reports):
    data = reports["relative"]

    assert np.isfinite(
        data[["WAPE_Pct", "Bias_Pct"]].to_numpy()
    ).all()

    assert (data["WAPE_Pct"] >= 0).all()


def test_backtest_has_three_folds(reports):
    data = reports["backtest"]

    assert len(data) == 3
    assert (data["Validation_Weeks"] == 8).all()
    assert (data["Observations"] == 360).all()


def test_backtest_is_chronological(reports):
    data = reports["backtest"]

    train_end = pd.to_datetime(data["Train_End"])
    validation_start = pd.to_datetime(
        data["Validation_Start"]
    )
    validation_end = pd.to_datetime(
        data["Validation_End"]
    )

    assert (train_end < validation_start).all()
    assert (
        validation_end < pd.Timestamp("2012-06-01")
    ).all()

    assert data["Train_Weeks"].is_monotonic_increasing


def test_backtest_metrics_valid(reports):
    data = reports["backtest"]

    assert np.isfinite(
        data[["Baseline_MAE", "XGBoost_MAE"]].to_numpy()
    ).all()

    assert (
        data[["Baseline_MAE", "XGBoost_MAE"]] >= 0
    ).all().all()


def test_xgboost_wins_recorded_backtests(reports):
    """Verify the results of this specific historical experiment."""

    data = reports["backtest"]

    assert (
        data["XGBoost_MAE"] < data["Baseline_MAE"]
    ).all()
