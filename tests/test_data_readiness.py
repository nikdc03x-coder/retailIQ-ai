"""Regression coverage for the missing-data startup failure."""

import pytest

from retailiq import cli
from retailiq.dashboard import readiness


def test_missing_data_reports_sources_and_dashboard_artifacts(tmp_path):
    issues = readiness.find_data_issues(tmp_path)
    assert "Missing: data/raw/train.csv" in issues
    assert "Missing: reports/business_analytics/01_weekly_sales.csv" in issues


@pytest.mark.parametrize("content", ["", "Store,Sales\n", "Store,Sales\n\n"])
def test_empty_or_header_only_report_is_not_ready(tmp_path, monkeypatch, content):
    monkeypatch.setattr(readiness, "REQUIRED_DATA", ("sales.csv",))
    (tmp_path / "sales.csv").write_text(content, encoding="utf-8")
    assert readiness.find_data_issues(tmp_path)


def test_nonempty_report_is_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, "REQUIRED_DATA", ("sales.csv",))
    (tmp_path / "sales.csv").write_text("Store,Sales\n1,200\n", encoding="utf-8")
    assert readiness.find_data_issues(tmp_path) == []


def test_check_data_fails_with_recovery_instructions(monkeypatch, capsys):
    monkeypatch.setattr(cli, "find_data_issues", lambda: ["Missing: reports/sales.csv"])
    assert cli.main(["check-data"]) == 1
    message = capsys.readouterr().out
    assert "reports/sales.csv" in message
    assert "python -m retailiq pipeline" in message
