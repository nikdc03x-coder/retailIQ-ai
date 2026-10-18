"""Tests for the RetailIQ AI sales forecasting pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "13_results/sales_forecasting"


@pytest.fixture(scope="module")
def forecasting_data():
    """Load the saved outputs from the verified forecasting pipeline."""

    series = pd.read_csv(RESULTS / "01_store_week_series.csv")
    coverage = pd.read_csv(RESULTS / "02_store_coverage.csv")
    features = pd.read_parquet(RESULTS / "05_model_features.parquet")
    comparison = pd.read_csv(
        RESULTS / "12_aligned_model_comparison.csv"
    )

    return series, coverage, features, comparison


def test_store_week_keys_unique(forecasting_data):
    series, _, _, _ = forecasting_data

    assert not series.duplicated(["Store", "Date"]).any()


def test_complete_store_coverage(forecasting_data):
    _, coverage, _, _ = forecasting_data

    assert coverage["Store"].nunique() == 45
    assert (coverage["Missing_Weeks"] == 0).all()


def test_expected_feature_rows(forecasting_data):
    _, _, features, _ = forecasting_data

    assert len(features) == 6255
    assert features["Store"].nunique() == 45


def test_feature_keys_unique(forecasting_data):
    _, _, features, _ = forecasting_data

    assert not features.duplicated(["Store", "Date"]).any()


def test_no_missing_model_features(forecasting_data):
    _, _, features, _ = forecasting_data

    columns = [
        "Lag_1", "Lag_2", "Lag_4",
        "Rolling_Mean_4", "Rolling_Std_4",
    ]

    assert not features[columns].isna().any().any()


def test_lag_one_uses_previous_week(forecasting_data):
    series, _, features, _ = forecasting_data

    series["Date"] = pd.to_datetime(series["Date"])

    expected = series.sort_values(["Store", "Date"]).copy()
    expected["Expected"] = (
        expected.groupby("Store")["Weekly_Store_Sales"].shift(1)
    )

    merged = features.merge(
        expected[["Store", "Date", "Expected"]],
        on=["Store", "Date"],
        validate="one_to_one",
    )

    assert np.allclose(merged["Lag_1"], merged["Expected"])


def test_rolling_mean_excludes_current_week(forecasting_data):
    series, _, features, _ = forecasting_data

    series["Date"] = pd.to_datetime(series["Date"])

    expected = series.sort_values(["Store", "Date"]).copy()

    expected["Expected"] = (
        expected.groupby("Store")["Weekly_Store_Sales"]
        .transform(
            lambda sales: sales.shift(1).rolling(4).mean()
        )
    )

    merged = features.merge(
        expected[["Store", "Date", "Expected"]],
        on=["Store", "Date"],
        validate="one_to_one",
    )

    assert np.allclose(
        merged["Rolling_Mean_4"],
        merged["Expected"],
    )


def test_comparison_contains_all_models(forecasting_data):
    _, _, _, comparison = forecasting_data

    assert set(comparison["Model"]) == {
        "Baseline", "Random Forest", "XGBoost"
    }


def test_comparison_uses_equal_observation_counts(forecasting_data):
    _, _, _, comparison = forecasting_data

    counts = comparison.groupby("Split")["Observations"].nunique()

    assert (counts == 1).all()


def test_metrics_are_valid(forecasting_data):
    _, _, _, comparison = forecasting_data

    assert np.isfinite(comparison[["MAE", "RMSE"]]).all().all()
    assert (comparison[["MAE", "RMSE"]] >= 0).all().all()


def test_validation_selects_xgboost(forecasting_data):
    _, _, _, comparison = forecasting_data

    validation = comparison[
        comparison["Split"] == "Validation"
    ].sort_values("MAE")

    assert validation.iloc[0]["Model"] == "XGBoost"
