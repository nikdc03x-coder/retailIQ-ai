
"""Test custom question routing using controlled sample reports."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_dashboard"))

import custom_question_router as router
from gemini_service import answer_question


@pytest.fixture
def sample_reports(tmp_path, monkeypatch):
    """Create small test reports without changing actual project data."""

    stores = pd.DataFrame({
        "Store": [10, 20, 30, 40],
        "Total_Sales": [1000, 3000, 2000, 500],
    })

    departments = pd.DataFrame({
        "Dept": [1, 92, 7],
        "Total_Sales": [1200, 5000, 2500],
    })

    store_path = tmp_path / "stores.csv"
    department_path = tmp_path / "departments.csv"

    stores.to_csv(store_path, index=False)
    departments.to_csv(department_path, index=False)

    monkeypatch.setattr(router, "STORE_REPORT", store_path)
    monkeypatch.setattr(
        router,
        "DEPARTMENT_REPORT",
        department_path,
    )


def test_top_store_ranking(sample_reports):
    """Return the highest-selling stores in descending order."""

    result = router.retrieve_evidence(
        "Which are the top 2 stores by sales?"
    )

    assert result["supported"] is True
    assert result["topic"] == "custom_store_ranking"

    records = result["evidence"][0]["records"]

    assert [row["Store"] for row in records] == [20, 30]
    assert [row["Total_Sales"] for row in records] == [
        3000,
        2000,
    ]


def test_bottom_store_ranking(sample_reports):
    """Return the lowest-selling stores first."""

    result = router.retrieve_evidence(
        "Which are the bottom 2 stores?"
    )

    records = result["evidence"][0]["records"]

    assert [row["Store"] for row in records] == [40, 10]


def test_department_ranking(sample_reports):
    """Use the actual department identifier column."""

    result = router.retrieve_evidence(
        "Show the top 2 departments."
    )

    records = result["evidence"][0]["records"]

    assert [row["Dept"] for row in records] == [92, 7]


def test_store_comparison(sample_reports):
    """Calculate the difference between two stores."""

    result = router.retrieve_evidence(
        "Compare Store 10 and Store 20."
    )

    assert result["supported"] is True
    assert result["topic"] == "custom_store_comparison"

    records = result["evidence"][0]["records"]

    assert len(records) == 2
    assert records[0]["Total_Sales"] == 1000
    assert records[1]["Total_Sales"] == 3000
    assert records[0]["Absolute_Sales_Difference"] == 2000
    assert records[0]["Higher_Sales_Store"] == 20


def test_missing_store(sample_reports):
    """Never invent results for a store that is absent."""

    result = router.retrieve_evidence(
        "Compare Store 10 and Store 999."
    )

    assert result["supported"] is False
    assert result["evidence"] == []


def test_invalid_ranking_count(sample_reports):
    """Reject invalid ranking requests."""

    result = router.retrieve_evidence(
        "Show the top 0 stores."
    )

    assert result["supported"] is False


def test_unsupported_date_is_not_ignored(monkeypatch):
    """Do not silently discard a requested year filter."""

    marker = {
        "supported": False,
        "message": "Historical year filter is unsupported.",
        "evidence": [],
    }

    monkeypatch.setattr(
        router,
        "retrieve_existing_evidence",
        lambda question: marker,
    )

    result = router.retrieve_evidence(
        "Show top 2 stores by sales in 2025."
    )

    assert result is marker


def test_custom_question_reaches_gemini_service(sample_reports):
    """Check routing, evidence retrieval and numerical validation."""

    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(
                text="Store 20 recorded total sales of 3,000."
            )

    class FakeClient:
        models = FakeModels()

    result = answer_question(
        "Show top 2 stores by sales.",
        client=FakeClient(),
    )

    assert result["supported"] is True
    assert result["generated"] is True
    assert result["validation_status"] == "numeric_check_passed"
    assert result["evidence"]

# Verify that different natural-language phrasings
# produce the same approved store-comparison calculation.
@pytest.mark.parametrize(
    "question",
    [
        "Compare Store 10 and Store 20.",
        "How much more did Store 20 sell than Store 10?",
        "What's the sales difference between Store 10 and Store 20?",
        "How do Store 10 and Store 20 compare in total sales?",
    ],
)
def test_flexible_store_comparison(sample_reports, question):
    """Different wording should produce the same verified comparison."""

    result = router.retrieve_evidence(question)

    assert result["supported"] is True
    assert result["topic"] == "custom_store_comparison"

    records = result["evidence"][0]["records"]

    assert len(records) == 2

    sales_by_store = {
        row["Store"]: row["Total_Sales"]
        for row in records
    }

    assert sales_by_store == {
        10: 1000,
        20: 3000,
    }

    assert records[0]["Absolute_Sales_Difference"] == 2000
    assert records[0]["Higher_Sales_Store"] == 20


# Verify ranking questions with written numbers
# and alternative ways of asking for highest or lowest sales.
@pytest.mark.parametrize(
    "question, expected_entity, expected_ids",
    [
        (
            "Which five stores generated the most sales?",
            "Store",
            [20, 30, 10, 40],
        ),
        (
            "Which 2 stores had the lowest sales?",
            "Store",
            [40, 10],
        ),
        (
            "List the 2 lowest-selling departments.",
            "Dept",
            [1, 7],
        ),
        (
            "Show the top two departments.",
            "Dept",
            [92, 7],
        ),
    ],
)
def test_flexible_sales_rankings(
    sample_reports,
    question,
    expected_entity,
    expected_ids,
):
    """Ranking requests should produce the correct data order."""

    result = router.retrieve_evidence(question)

    assert result["supported"] is True

    records = result["evidence"][0]["records"]

    actual_ids = [
        row[expected_entity]
        for row in records
    ]

    assert actual_ids == expected_ids


def test_date_filter_cannot_be_silently_ignored():
    """A ranking with an unsupported year must not be simplified."""

    request = router.parse_custom_question(
        "Show top 5 stores by sales in 2025."
    )

    assert request is None


def test_unrelated_question_uses_existing_engine(monkeypatch):
    """Preserve forecasting, anomaly and SHAP question routing."""

    expected = {
        "supported": False,
        "message": "Handled by the existing engine.",
        "evidence": [],
    }

    monkeypatch.setattr(
        router,
        "retrieve_existing_evidence",
        lambda question: expected,
    )

    result = router.retrieve_evidence(
        "What are the top SHAP features?"
    )

    assert result is expected
