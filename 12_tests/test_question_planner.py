
"""Test safe Gemini question planning without real API requests."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_dashboard"))

import custom_question_router as router

from question_planner import (
    is_candidate_question,
    plan_question,
    requested_count,
    validate_plan,
)


QUESTION = (
    "Could you rank the three best-performing stores "
    "based on total sales?"
)


def test_custom_question_is_eligible():
    """Recognize a flexible store-ranking question."""

    assert is_candidate_question(QUESTION)


def test_unsupported_period_is_rejected():
    """Never ignore a requested date filter."""

    question = "Rank the top 5 stores during 2025."

    assert not is_candidate_question(question)


def test_written_ranking_count():
    """Recognize numbers written as words."""

    assert requested_count(QUESTION) == 3


def test_valid_ranking_plan():
    """Accept a safe ranking plan."""

    plan = {
        "operation": "rank_stores",
        "count": 3,
        "highest": True,
    }

    assert validate_plan(plan, QUESTION) == plan


def test_wrong_ranking_count():
    """Gemini must not change the requested count."""

    plan = {
        "operation": "rank_stores",
        "count": 10,
        "highest": True,
    }

    assert validate_plan(plan, QUESTION) is None


def test_fabricated_store_id():
    """Gemini must use only explicitly requested store IDs."""

    question = "Compare Store 10 and Store 20."

    plan = {
        "operation": "compare_stores",
        "first_store": 10,
        "second_store": 30,
    }

    assert validate_plan(plan, question) is None


def test_unknown_operation_is_rejected():
    """Never accept arbitrary operations or generated commands."""

    plan = {
        "operation": "execute_python",
        "code": "print('not allowed')",
    }

    assert validate_plan(plan, QUESTION) is None


def test_unexpected_parameters_are_rejected():
    """Reject extra parameters not in the approved contract."""

    plan = {
        "operation": "rank_stores",
        "count": 3,
        "highest": True,
        "sql": "SELECT * FROM stores",
    }

    assert validate_plan(plan, QUESTION) is None


def test_fake_gemini_json_response():
    """Test model-response parsing without contacting Gemini."""

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
    """A validated plan should use the approved Pandas calculation."""

    import pandas as pd

    stores = pd.DataFrame({
        "Store": [10, 20, 30, 40],
        "Total_Sales": [1000, 3000, 2000, 500],
    })

    store_path = tmp_path / "stores.csv"
    stores.to_csv(store_path, index=False)

    monkeypatch.setattr(router, "STORE_REPORT", store_path)

    monkeypatch.setattr(
        router,
        "plan_question",
        lambda question: {
            "operation": "rank_stores",
            "count": 3,
            "highest": True,
        },
    )

    result = router.retrieve_evidence(QUESTION)

    assert result["supported"] is True
    assert result["topic"] == "custom_store_ranking"

    records = result["evidence"][0]["records"]

    assert [row["Store"] for row in records] == [20, 30, 10]


def test_unavailable_planner_is_safe(monkeypatch):
    """If Gemini planning fails, never invent a ranking."""

    monkeypatch.setattr(
        router,
        "plan_question",
        lambda question: None,
    )

    result = router.retrieve_evidence(QUESTION)

    assert result["supported"] is False
    assert result["evidence"] == []