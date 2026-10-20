
"""Offline tests for the Gemini planner and custom-question router."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_dashboard"))

import custom_question_router as router

from question_planner import (
    is_candidate_question,
    requested_count,
    validate_plan,
    plan_question,
)


QUESTION = (
    "Could you rank the three best-performing stores "
    "based on total sales?"
)


def test_eligible_question():
    """Recognize a valid custom sales question."""

    assert is_candidate_question(QUESTION)


def test_unsupported_year():
    """Reject an unsupported year filter."""

    assert not is_candidate_question(
        "Show top 5 stores by sales in 2025."
    )


def test_written_count():
    """Recognize ranking counts written as words."""

    assert requested_count(QUESTION) == 3


def test_valid_plan():
    """Accept a valid store ranking plan."""

    plan = {
        "operation": "rank_stores",
        "count": 3,
        "highest": True,
    }

    assert validate_plan(plan, QUESTION) == plan


def test_wrong_count():
    """Gemini must not change the requested count."""

    plan = {
        "operation": "rank_stores",
        "count": 10,
        "highest": True,
    }

    assert validate_plan(plan, QUESTION) is None


def test_wrong_direction():
    """Reject a plan that reverses the ranking."""

    plan = {
        "operation": "rank_stores",
        "count": 3,
        "highest": False,
    }

    assert validate_plan(plan, QUESTION) is None


def test_fabricated_store_id():
    """Store identifiers must match the original question."""

    question = "Compare Store 10 and Store 20."

    plan = {
        "operation": "compare_stores",
        "first_store": 10,
        "second_store": 30,
    }

    assert validate_plan(plan, question) is None


def test_unknown_operation():
    """Reject arbitrary commands or executable code."""

    plan = {
        "operation": "execute_python",
        "code": "print(1)",
    }

    assert validate_plan(plan, QUESTION) is None


def test_extra_parameters():
    """Reject fields outside the approved planner contract."""

    plan = {
        "operation": "rank_stores",
        "count": 3,
        "highest": True,
        "sql": "SELECT * FROM stores",
    }

    assert validate_plan(plan, QUESTION) is None


def test_fake_gemini_json_response():
    """Test Gemini response parsing without contacting the API."""

    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(
                text=(
                    '{"operation":"rank_stores",'
                    '"count":3,"highest":true}'
                )
            )

    class FakeClient:
        models = FakeModels()

    result = plan_question(
        QUESTION,
        client=FakeClient(),
    )

    assert result == {
        "operation": "rank_stores",
        "count": 3,
        "highest": True,
    }


def test_planner_connects_to_pandas(tmp_path, monkeypatch):
    """A validated plan should reach the real calculation function."""

    stores = pd.DataFrame({
        "Store": [10, 20, 30, 40],
        "Total_Sales": [1000, 3000, 2000, 500],
    })

    store_path = tmp_path / "stores.csv"
    stores.to_csv(store_path, index=False)

    monkeypatch.setattr(
        router,
        "STORE_REPORT",
        store_path,
    )

    monkeypatch.setattr(
        router,
        "plan_question",
        lambda question: {
            "operation": "rank_stores",
            "count": 3,
            "highest": True,
        },
    )

    # An overly broad original match must not replace this ranking.
    monkeypatch.setattr(
        router,
        "retrieve_existing_evidence",
        lambda question: {
            "supported": True,
            "message": "Generic result",
            "evidence": [{"records": []}],
        },
    )

    result = router.retrieve_evidence(QUESTION)

    assert result["supported"] is True
    assert result["topic"] == "custom_store_ranking"

    records = result["evidence"][0]["records"]

    assert [row["Store"] for row in records] == [
        20, 30, 10
    ]


def test_unavailable_planner_is_safe(monkeypatch):
    """An unavailable planner must not invent results."""

    monkeypatch.setattr(
        router,
        "plan_question",
        lambda question: None,
    )

    result = router.retrieve_evidence(QUESTION)

    assert result["supported"] is False
    assert result["evidence"] == []


def test_original_question_not_overridden(monkeypatch):
    """Keep existing forecasting questions working."""

    expected = {
        "supported": True,
        "message": "Original result",
        "evidence": [1],
    }

    monkeypatch.setattr(
        router,
        "retrieve_existing_evidence",
        lambda question: expected,
    )

    result = router.retrieve_evidence(
        "Which forecasting model has the best MAE?"
    )

    assert result is expected


def test_year_request_cannot_be_silently_ignored(monkeypatch):
    """Never ignore unsupported historical filters."""

    monkeypatch.setattr(
        router,
        "retrieve_existing_evidence",
        lambda question: {
            "supported": True,
            "message": "Generic result",
            "evidence": [1],
        },
    )

    result = router.retrieve_evidence(
        "Show top 5 stores by sales in 2025."
    )

    assert result["supported"] is False

