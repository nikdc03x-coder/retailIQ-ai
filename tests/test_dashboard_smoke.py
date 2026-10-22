"""Render every supported dashboard destination with Streamlit's test runtime."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from retailiq.dashboard.app import NAVIGATION

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("page", NAVIGATION)
def test_dashboard_page_renders(page):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    app.session_state["retailiq_active_page"] = page
    app.run()
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]


@pytest.mark.parametrize("page", ["Project Overview", "Dataset Overview", "Sales Explorer"])
def test_consolidated_pages_open_from_shared_sidebar(page):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.button(key=f"nav_{NAVIGATION[page]['key']}").click().run()
    assert app.session_state["retailiq_active_page"] == page
    assert len(app.title) == 1
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]


def test_all_page_files_are_registered_in_one_dashboard():
    page_modules = {
        f"retailiq.dashboard.pages.{path.stem}"
        for path in (ROOT / "src/retailiq/dashboard/pages").glob("*.py")
        if path.stem != "__init__"
    }
    assert {config["render"].__module__ for config in NAVIGATION.values()} == page_modules


def test_unknown_saved_page_recovers_to_overview():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    app.session_state["retailiq_active_page"] = "Old Dashboard Page"
    app.run()
    assert app.session_state["retailiq_active_page"] == "Overview"
    assert not app.exception, [item.message for item in app.exception]
    assert app.metric[2].value == "143"
