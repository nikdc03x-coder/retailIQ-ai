"""
Phase 00 foundation checks.

These tests verify that the essential project files
exist before we begin working with real sales data.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_exists():
    """The dashboard must have an entry file."""
    assert (ROOT / "app.py").is_file()


def test_documentation_exists():
    """Our project must include its problem statement."""
    assert (ROOT / "docs/01_problem_statement.md").is_file()


def test_notebook_exists():
    """The first learning notebook must be present."""
    assert (ROOT / "notebooks/00_python_setup.ipynb").is_file()
