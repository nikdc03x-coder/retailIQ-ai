"""Shared sample reports for RetailIQ AI analytics tests."""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_reports(tmp_path, monkeypatch):
    """Create known sales data without modifying project reports."""

    import retailiq.assistant.router as router

    # Small store report with predictable sales totals.
    stores = pd.DataFrame(
        {
            "Store": [10, 20, 30, 40],
            "Total_Sales": [1000, 3000, 2000, 500],
        }
    )

    # Small department report for ranking tests.
    departments = pd.DataFrame(
        {
            "Dept": [1, 92, 7],
            "Total_Sales": [1200, 5000, 2500],
        }
    )

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


@pytest.fixture(autouse=True)
def offline_ai(monkeypatch):
    """Exercise fake-client and fallback paths without using developer credentials."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_TIMEOUT_MS", "15000")
    monkeypatch.setenv("LOG_LEVEL", "INFO")


INTEGRATION_MODULES = {
    "test_anomaly_detection.py",
    "test_business_analytics.py",
    "test_business_context.py",
    "test_dashboard.py",
    "test_dashboard_analytics.py",
    "test_data_ingestion.py",
    "test_data_transformation.py",
    "test_explainable_ai.py",
    "test_forecast_reliability.py",
    "test_gemini_service.py",
    "test_gemini_grounding.py",
    "test_sales_forecasting.py",
    "test_business_assistant_scope.py",
}


def pytest_collection_modifyitems(items):
    """Keep the full suite default; allow fresh checkouts to select isolated tests."""
    for item in items:
        if item.path.name in INTEGRATION_MODULES:
            item.add_marker(pytest.mark.integration)
