
"""Test retrieval of verified RetailIQ AI business evidence."""

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "08_dashboard"

if str(DASHBOARD) not in sys.path:
    sys.path.insert(0, str(DASHBOARD))

from business_context import retrieve_evidence, identify_topic


@pytest.mark.parametrize(
    "question, expected_topic",
    [
        ("Which store has the highest sales?", "store_sales"),
        ("Which department has the highest sales?", "department_sales"),
        ("Which forecasting model has the best MAE?", "model_comparison"),
        ("What is the MAE for Store 14?", "store_reliability"),
        ("How many anomalies were flagged?", "anomaly_summary"),
        ("What are the top SHAP features?", "shap_importance"),
    ],
)
def test_supported_questions(question, expected_topic):
    """Known project questions should retrieve the correct evidence."""

    result = retrieve_evidence(question)

    assert result["supported"] is True
    assert result["topic"] == expected_topic
    assert result["evidence"]
    assert result["evidence"][0]["records"]
    assert result["evidence"][0]["source"]


@pytest.mark.parametrize(
    "question",
    [
        "",
        "Who won yesterday's football match?",
        "Tell me a joke",
        "Write Python code for my homework",
    ],
)
def test_unrelated_questions(question):
    """Unrelated questions should not retrieve project evidence."""

    assert identify_topic(question) is None

    result = retrieve_evidence(question)

    assert result["supported"] is False
    assert result["evidence"] == []


def test_missing_store():
    """A store without saved results must not produce invented data."""

    result = retrieve_evidence("What is the MAE for Store 999?")

    assert result["supported"] is False
    assert result["evidence"] == []
