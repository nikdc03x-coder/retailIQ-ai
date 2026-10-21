"""Test RetailIQ business analytics calculations."""

from pathlib import Path

import pandas as pd
import pytest

from retailiq.analytics import holidays as holiday_module
from retailiq.analytics import performance as performance_module
from retailiq.analytics import weekly as weekly_module

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/retail_analytical.parquet"
SOURCE = ROOT / "data/processed/retail_analytical.parquet"


@pytest.fixture(scope="module")
def data():
    """Load the validated analytical dataset once."""
    return pd.read_parquet(SOURCE)


def test_weekly_sales_preserved(data):
    """Weekly totals must equal the original sales total."""
    weekly = weekly_module.calculate_weekly_sales(data)

    assert len(weekly) == 143
    assert weekly["Sales_Records"].sum() == len(data)
    assert weekly["Total_Sales"].sum() == pytest.approx(data["Weekly_Sales"].sum())


def test_weekly_dates_unique(data):
    """Each week must appear once in the summary."""
    weekly = weekly_module.calculate_weekly_sales(data)

    assert not weekly["Date"].duplicated().any()


def test_store_performance(data):
    """Store aggregation must preserve sales and records."""
    stores = performance_module.analyze_stores(data)

    assert len(stores) == 45
    assert stores["Sales_Records"].sum() == len(data)
    assert stores["Total_Sales"].sum() == pytest.approx(data["Weekly_Sales"].sum())


def test_department_performance(data):
    """Department aggregation must preserve sales and records."""
    departments = performance_module.analyze_departments(data)

    assert len(departments) == 81
    assert departments["Sales_Records"].sum() == len(data)
    assert departments["Total_Sales"].sum() == pytest.approx(data["Weekly_Sales"].sum())


def test_store_week_sales(data):
    """Holiday analysis must use unique store-week observations."""
    store_weeks = holiday_module.calculate_store_week_sales(data)

    assert len(store_weeks) == 6435
    assert not store_weeks.duplicated(["Store", "Date"]).any()
    assert store_weeks["Weekly_Store_Sales"].sum() == pytest.approx(data["Weekly_Sales"].sum())


def test_holiday_comparison(data):
    """Both categories must account for every store-week."""
    store_weeks = holiday_module.calculate_store_week_sales(data)
    summary = holiday_module.compare_holiday_sales(store_weeks)

    assert len(summary) == 2
    assert summary["Store_Weeks"].sum() == len(store_weeks)
    assert set(summary["IsHoliday"]) == {False, True}


def test_holiday_date_totals(data):
    """Holiday date totals must preserve holiday sales."""
    store_weeks = holiday_module.calculate_store_week_sales(data)
    holiday_dates = holiday_module.calculate_holiday_dates(store_weeks)

    holiday_total = store_weeks.loc[store_weeks["IsHoliday"], "Weekly_Store_Sales"].sum()

    assert holiday_dates["Total_Sales"].sum() == pytest.approx(holiday_total)


def test_holiday_flags_conflict():
    """Conflicting flags for the same store-week must be rejected."""
    sample = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": pd.to_datetime(["2012-01-06", "2012-01-06"]),
            "Dept": [1, 2],
            "Weekly_Sales": [100.0, 200.0],
            "IsHoliday": [True, False],
        }
    )

    with pytest.raises(ValueError, match="Conflicting holiday"):
        holiday_module.calculate_store_week_sales(sample)
