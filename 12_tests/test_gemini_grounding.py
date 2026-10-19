
"""Test numerical grounding of Gemini-generated explanations."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_dashboard"))

from gemini_service import (
    answer_question,
    validate_generated_answer,
)


class FakeModels:
    """Return a chosen answer without contacting Gemini."""

    def __init__(self, text):
        self.text = text

    def generate_content(self, **kwargs):
        return SimpleNamespace(text=self.text)


class FakeClient:
    """Simulate the Gemini client for offline tests."""

    def __init__(self, text):
        self.models = FakeModels(text)


def test_matching_numbers_are_accepted():
    """Rounded report values should be accepted."""

    evidence = [{
        "source": "model_results.csv",
        "records": [{"model": "XGBoost", "mae": 38304.75899}],
    }]

    assert validate_generated_answer(
        "XGBoost achieved an MAE of 38,304.76.",
        evidence,
    )


def test_fabricated_numbers_are_rejected():
    """Reject a numerical claim missing from the reports."""

    evidence = [{
        "source": "model_results.csv",
        "records": [{"model": "XGBoost", "mae": 38304.75899}],
    }]

    assert not validate_generated_answer(
        "The MAE was 12,345.67.",
        evidence,
    )


def test_question_identifier_is_accepted():
    """An identifier from the supported question is allowed."""

    evidence = [{
        "source": "store_results.csv",
        "records": [{"store": 14, "mae": 87635.1477}],
    }]

    assert validate_generated_answer(
        "Store 14 had an MAE of 87,635.15.",
        evidence,
        question="What is the MAE for Store 14?",
    )


def test_actual_service_accepts_verified_number():
    """A fake Gemini response can use the retrieved MAE."""

    result = answer_question(
        "Which forecasting model has the best MAE?",
        client=FakeClient(
            "XGBoost had a test MAE of 38,304.76."
        ),
    )

    assert result["supported"] is True
    assert result["generated"] is True
    assert result["validation_status"] == "numeric_check_passed"


def test_actual_service_rejects_fabricated_number():
    """Unsupported numbers must not reach the chat answer."""

    result = answer_question(
        "Which forecasting model has the best MAE?",
        client=FakeClient(
            "XGBoost achieved a test MAE of 12,345.67."
        ),
    )

    assert result["supported"] is True
    assert result["generated"] is False
    assert result["validation_status"] == "validation_failed"
    assert "12,345.67" not in result["answer"]
    assert result["evidence"]


def test_empty_model_response_uses_fallback():
    """An empty Gemini response must not break the dashboard."""

    result = answer_question(
        "Which forecasting model has the best MAE?",
        client=FakeClient(""),
    )

    assert result["supported"] is True
    assert result["generated"] is False
    assert result["evidence"]
