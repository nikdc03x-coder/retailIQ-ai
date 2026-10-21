"""Regression tests for RetailIQ AI natural-language analytics."""

from pathlib import Path

import pandas as pd
import pytest

import retailiq.assistant.router as router
from retailiq.assistant.planner import (
    requested_direction,
    validate_plan,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_reports(tmp_path, monkeypatch):
    """Create controlled reports without touching the real dataset."""

    stores = pd.DataFrame(
        {
            "Store": [10, 20, 30, 40],
            "Total_Sales": [1000, 3000, 2000, 500],
        }
    )

    departments = pd.DataFrame(
        {
            "Dept": list(range(1, 82)),
            "Total_Sales": [float(value * 100) for value in range(1, 82)],
        }
    )

    store_path = tmp_path / "stores.csv"
    department_path = tmp_path / "departments.csv"

    stores.to_csv(store_path, index=False)
    departments.to_csv(department_path, index=False)

    monkeypatch.setattr(
        router,
        "STORE_REPORT",
        store_path,
    )

    monkeypatch.setattr(
        router,
        "DEPARTMENT_REPORT",
        department_path,
    )


def test_top_60_departments(sample_reports):
    """Department rankings must support more than 45 entries."""

    result = router.retrieve_evidence("Show top 60 departments.")

    assert result["supported"] is True

    records = result["evidence"][0]["records"]

    assert len(records) == 60
    assert records[0]["Dept"] == 81
    assert records[-1]["Dept"] == 22


def test_top_81_departments(sample_reports):
    """All 81 departments should be available for ranking."""

    result = router.retrieve_evidence("Show top 81 departments.")

    assert result["supported"] is True
    assert len(result["evidence"][0]["records"]) == 81


def test_store_count_above_limit_is_rejected(sample_reports):
    """A ranking cannot request more than 45 stores."""

    result = router.retrieve_evidence("Show top 46 stores.")

    assert result["supported"] is False
    assert result["evidence"] == []


def test_department_count_above_limit_is_rejected(sample_reports):
    """A ranking cannot request more than 81 departments."""

    result = router.retrieve_evidence("Show top 82 departments.")

    assert result["supported"] is False
    assert result["evidence"] == []


@pytest.mark.parametrize(
    "question",
    [
        "Show top 5 stores by sales in 2025.",
        "Compare Store 10 and Store 20 in September.",
        "Rank the 5 most profitable stores.",
        "Which five stores had the lowest average sales?",
    ],
)
def test_unsupported_scope_is_rejected(
    sample_reports,
    question,
):
    """Never replace unsupported filters with all-time totals."""

    result = router.retrieve_evidence(question)

    assert result["supported"] is False
    assert result["evidence"] == []


def test_lowest_direction():
    """Identify a request for lowest historical sales."""

    assert requested_direction("Rank the five lowest-selling stores.") is False


def test_highest_direction():
    """Identify a request for highest historical sales."""

    assert requested_direction("Rank the five best-performing stores.") is True


def test_mixed_direction_is_ambiguous():
    """Never guess when both highest and lowest are requested."""

    assert requested_direction("Show the five highest and lowest stores.") is None


def test_gemini_cannot_reverse_lowest_direction():
    """Reject a plan that reverses the user's requested direction."""

    question = "Rank the five lowest-selling stores."

    incorrect_plan = {
        "operation": "rank_stores",
        "count": 5,
        "highest": True,
    }

    assert (
        validate_plan(
            incorrect_plan,
            question,
        )
        is None
    )


def test_gemini_cannot_reverse_highest_direction():
    """Reject a plan that reverses a top-store ranking."""

    question = "Rank the five best-performing stores."

    incorrect_plan = {
        "operation": "rank_stores",
        "count": 5,
        "highest": False,
    }

    assert (
        validate_plan(
            incorrect_plan,
            question,
        )
        is None
    )


def test_valid_department_plan():
    """A ranking of 60 departments must remain supported."""

    question = "Show top 60 departments."

    valid_plan = {
        "operation": "rank_departments",
        "count": 60,
        "highest": True,
    }

    assert (
        validate_plan(
            valid_plan,
            question,
        )
        == valid_plan
    )


def test_existing_forecast_question_is_preserved(monkeypatch):
    """Forecasting questions still use the previous context engine."""

    expected = {
        "supported": True,
        "evidence": [],
        "message": "Existing engine response.",
    }

    monkeypatch.setattr(
        router,
        "retrieve_existing_evidence",
        lambda question: expected,
    )

    result = router.retrieve_evidence("Which forecasting model has the best MAE?")

    assert result is expected


def test_flexible_ranking_still_uses_pandas(
    sample_reports,
    monkeypatch,
):
    """Approved Gemini plans must reach real Python calculations."""

    monkeypatch.setattr(
        router,
        "plan_question",
        lambda question: {
            "operation": "rank_stores",
            "count": 3,
            "highest": True,
        },
    )

    result = router.retrieve_evidence(
        "Could you rank the three best-performing stores based on total sales?"
    )

    assert result["supported"] is True

    records = result["evidence"][0]["records"]

    assert [record["Store"] for record in records] == [20, 30, 10]
