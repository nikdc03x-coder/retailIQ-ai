"""Test verified local answers and existing Gemini explanations."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import retailiq.assistant.service as service

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ranked_result():
    """Create a verified ranking with known sales values."""

    return {
        "supported": True,
        "topic": "custom_store_ranking",
        "message": "",
        "evidence": [
            {
                "source": "business_analytics/02_store_performance.csv",
                "description": ("The 2 stores with the highest historical total sales."),
                "records": [
                    {
                        "Rank": 1,
                        "Store": 20,
                        "Total_Sales": 3000,
                    },
                    {
                        "Rank": 2,
                        "Store": 30,
                        "Total_Sales": 2000,
                    },
                ],
            }
        ],
    }


@pytest.fixture
def comparison_result():
    """Create a verified comparison between two stores."""

    return {
        "supported": True,
        "topic": "custom_store_comparison",
        "message": "",
        "evidence": [
            {
                "source": "business_analytics/02_store_performance.csv",
                "description": ("Historical total sales comparison."),
                "records": [
                    {
                        "Store": 10,
                        "Total_Sales": 1000,
                        "Absolute_Sales_Difference": 2000,
                        "Higher_Sales_Store": 20,
                    },
                    {
                        "Store": 20,
                        "Total_Sales": 3000,
                        "Absolute_Sales_Difference": 2000,
                        "Higher_Sales_Store": 20,
                    },
                ],
            }
        ],
    }


def test_ranking_uses_local_explanation(
    ranked_result,
    monkeypatch,
):
    """Ranking answers should not need a Gemini explanation call."""

    monkeypatch.setattr(
        service,
        "retrieve_evidence",
        lambda question: ranked_result,
    )

    result = service.answer_question("Show top 2 stores by sales.")

    assert result["supported"] is True
    assert result["generated"] is False
    assert result["validation_status"] == "verified_local"

    assert "Store 20" in result["answer"]
    assert "3,000" in result["answer"]
    assert len(result["sources"]) == 1


def test_store_comparison_uses_local_explanation(
    comparison_result,
    monkeypatch,
):
    """Store comparisons should show verified values and differences."""

    monkeypatch.setattr(
        service,
        "retrieve_evidence",
        lambda question: comparison_result,
    )

    result = service.answer_question("Compare Store 10 and Store 20.")

    assert result["supported"] is True
    assert result["generated"] is False
    assert result["validation_status"] == "verified_local"

    assert "Store 10" in result["answer"]
    assert "Store 20" in result["answer"]
    assert "2,000" in result["answer"]


def test_unsupported_question_skips_explanation(monkeypatch):
    """Never generate an answer without supported evidence."""

    monkeypatch.setattr(
        service,
        "retrieve_evidence",
        lambda question: {
            "supported": False,
            "message": "This analysis is not supported.",
            "evidence": [],
        },
    )

    result = service.answer_question("Show historical profits for 2025.")

    assert result["supported"] is False
    assert result["generated"] is False
    assert result["validation_status"] == "unsupported"
    assert result["evidence"] == []


def test_existing_gemini_path_with_supplied_client(
    ranked_result,
    monkeypatch,
):
    """An injected Gemini client should still use numeric validation."""

    monkeypatch.setattr(
        service,
        "retrieve_evidence",
        lambda question: ranked_result,
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(
                text=("Store 20 recorded sales of 3,000, the highest among these stores.")
            )

    class FakeClient:
        models = FakeModels()

    result = service.answer_question(
        "Show top 2 stores by sales.",
        client=FakeClient(),
    )

    assert result["supported"] is True
    assert result["generated"] is True
    assert result["validation_status"] == "numeric_check_passed"


def test_gemini_rejects_unsupported_number(
    ranked_result,
    monkeypatch,
):
    """Generated values not found in evidence must be withheld."""

    monkeypatch.setattr(
        service,
        "retrieve_evidence",
        lambda question: ranked_result,
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="Store 20 recorded sales of 999,999.")

    class FakeClient:
        models = FakeModels()

    result = service.answer_question(
        "Show top 2 stores by sales.",
        client=FakeClient(),
    )

    assert result["supported"] is True
    assert result["generated"] is False
    assert result["validation_status"] == "validation_failed"
