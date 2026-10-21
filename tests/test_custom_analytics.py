"""Test RetailIQ AI custom sales calculations."""

from pathlib import Path

import pandas as pd
import pytest

from retailiq.assistant.analytics import compare_stores, rank_sales

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_sales():
    """Small test dataset with known answers."""

    return pd.DataFrame(
        {
            "Store": [10, 20, 30, 40],
            "Total_Sales": [1000, 3000, 2000, 500],
        }
    )


def test_top_stores(sample_sales):
    """Highest sales should appear first."""

    result = rank_sales(
        sample_sales,
        "Store",
        "Total_Sales",
        count=2,
    )

    assert result["Store"].tolist() == [20, 30]


def test_bottom_stores(sample_sales):
    """Lowest sales should appear first."""

    result = rank_sales(
        sample_sales,
        "Store",
        "Total_Sales",
        count=2,
        highest=False,
    )

    assert result["Store"].tolist() == [40, 10]


def test_compare_stores(sample_sales):
    """Compare two stores without generating values."""

    result = compare_stores(
        sample_sales,
        10,
        20,
        "Store",
        "Total_Sales",
    )

    assert result["difference"] == -2000
    assert result["higher_sales_store"] == 20


def test_unknown_store(sample_sales):
    """A missing store must produce a clear error."""

    with pytest.raises(ValueError, match="not found"):
        compare_stores(
            sample_sales,
            10,
            999,
            "Store",
            "Total_Sales",
        )


def test_missing_column(sample_sales):
    """Invalid report schemas must be rejected."""

    with pytest.raises(ValueError, match="Missing"):
        rank_sales(
            sample_sales,
            "Store",
            "Unknown_Sales",
        )


def test_duplicate_store(sample_sales):
    """Do not accidentally compare duplicated aggregated records."""

    duplicated = pd.concat(
        [sample_sales, sample_sales.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="one row"):
        rank_sales(
            duplicated,
            "Store",
            "Total_Sales",
        )


def test_invalid_count(sample_sales):
    """Reject zero or negative ranking sizes."""

    # Test both invalid values against the updated error message.
    for invalid_count in [0, -1]:
        with pytest.raises(ValueError, match="at least 1"):
            rank_sales(
                sample_sales,
                "Store",
                "Total_Sales",
                count=invalid_count,
            )
