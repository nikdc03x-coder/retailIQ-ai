"""Shared sample reports for custom analytics tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_reports(tmp_path, monkeypatch):
    """Provide known sales totals without changing real project files."""

    import custom_question_router as router

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