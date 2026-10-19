
"""Test the Gemini service without making external API calls."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_dashboard"))

from gemini_service import answer_question


class FakeModels:
    """Return a fixed explanation instead of contacting Gemini."""

    def generate_content(self, **kwargs):
        return SimpleNamespace(
            text="XGBoost had the lowest test MAE."
        )


class FakeClient:
    """Provide the API interface needed by the service."""

    models = FakeModels()


def test_supported_question():
    """Supported project questions should use verified evidence."""

    result = answer_question(
        "Which forecasting model has the best MAE?",
        client=FakeClient(),
    )

    assert result["supported"] is True
    assert result["generated"] is True
    assert result["sources"]
    assert "XGBoost" in result["answer"]


def test_unrelated_question():
    """Unrelated requests must be rejected before calling Gemini."""

    result = answer_question(
        "Who won yesterday's football match?",
        client=FakeClient(),
    )

    assert result["supported"] is False
    assert result["generated"] is False
    assert result["sources"] == []


def test_missing_api_key(monkeypatch):
    """Missing credentials should not break the application."""

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = answer_question(
        "Which store has the highest sales?"
    )

    assert result["supported"] is True
    assert result["generated"] is False
    assert result["evidence"]


def test_unsupported_year():
    """The assistant must not invent results for unavailable years."""

    result = answer_question(
        "Which store had the highest sales in 2025?",
        client=FakeClient(),
    )

    assert result["supported"] is False
    assert result["generated"] is False
