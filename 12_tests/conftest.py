
"""Shared sample reports for RetailIQ AI analytics tests."""

import sys
from pathlib import Path

import pandas as pd
import pytest


# Make dashboard modules available to the test suite.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_dashboard"))


@pytest.fixture
def sample_reports(tmp_path, monkeypatch):
    """Create known sales data without modifying project reports."""

    import custom_question_router as router

    # Small store report with predictable sales totals.
    stores = pd.DataFrame({
        "Store": [10, 20, 30, 40],
        "Total_Sales": [1000, 3000, 2000, 500],
    })

    # Small department report for ranking tests.
    departments = pd.DataFrame({
        "Dept": [1, 92, 7],
        "Total_Sales": [1200, 5000, 2500],
    })

    # Write sample reports into Pytest's temporary folder.
    store_path = tmp_path / "stores.csv"
    department_path = tmp_path / "departments.csv"

    stores.to_csv(store_path, index=False)
    departments.to_csv(department_path, index=False)

    # Redirect analytics to the sample reports for this test.
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
