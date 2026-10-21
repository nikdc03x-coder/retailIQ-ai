"""Isolated regression coverage for configuration, report access, and orchestration."""

import importlib
import logging
import pkgutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import retailiq
from retailiq import cli, pipeline
from retailiq.assistant import service
from retailiq.assistant.analytics import rank_sales
from retailiq.config import ConfigurationError, get_settings
from retailiq.dashboard import data_loader
from retailiq.forecasting.baseline import prepare_baseline
from retailiq.forecasting.features import build_features
from retailiq.logging_config import configure_logging
from retailiq.reports import load_csv, resolve_report_path


def test_offline_configuration_and_secret_repr():
    assert get_settings({}).gemini_api_key == ""
    assert "private-key" not in repr(get_settings({"GEMINI_API_KEY": "private-key"}))


@pytest.mark.parametrize(
    "values",
    [
        {"GEMINI_MODEL": " "},
        {"LOG_LEVEL": "verbose"},
        {"GEMINI_TIMEOUT_MS": "oops"},
        {"GEMINI_TIMEOUT_MS": "0"},
        {"GEMINI_TIMEOUT_MS": "120001"},
    ],
)
def test_invalid_configuration(values):
    with pytest.raises(ConfigurationError):
        get_settings(values)


@pytest.mark.parametrize("path", ["../secret.csv", "/outside.csv", "../../secret.csv", ".env"])
def test_reports_cannot_escape_directory(tmp_path, path):
    with pytest.raises(ValueError):
        resolve_report_path(path, tmp_path)


def test_reports_load_and_missing_file_is_actionable(tmp_path):
    (tmp_path / "sales.csv").write_text("Store,Sales\n1,20\n", encoding="utf-8")
    assert load_csv("sales.csv", tmp_path).iloc[0]["Sales"] == 20
    with pytest.raises(FileNotFoundError, match="pipeline"):
        load_csv("missing.csv", tmp_path)


def test_dashboard_cache_refreshes_after_report_change(tmp_path, monkeypatch):
    report = tmp_path / "sales.csv"
    monkeypatch.setattr(data_loader, "resolve_report_path", lambda _: report)
    report.write_text("Sales\n20\n", encoding="utf-8")
    assert data_loader.load_csv("sales.csv")["Sales"].iloc[0] == 20
    report.write_text("Sales\n3000\n", encoding="utf-8")
    assert data_loader.load_csv("sales.csv")["Sales"].iloc[0] == 3000


@pytest.mark.parametrize("question", [None, 42, "", "x" * 501])
def test_invalid_assistant_input(question):
    assert service.answer_question(question)["validation_status"] == "unsupported"


@pytest.mark.parametrize("error", [FileNotFoundError, ValueError, KeyError])
def test_missing_or_corrupt_evidence_is_contained(monkeypatch, error):
    def fail(_):
        raise error("private diagnostic")

    monkeypatch.setattr(service, "retrieve_evidence", fail)
    result = service.answer_question("Which model has the best MAE?")
    assert result["validation_status"] == "reports_unavailable"
    assert "private diagnostic" not in result["answer"]


def test_ai_client_is_bounded_and_closed(monkeypatch):
    from google import genai

    calls = {}

    def create(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **_: SimpleNamespace(text="Verified result.")
            ),
            close=lambda: calls.update(closed=True),
        )

    monkeypatch.setattr(genai, "Client", create)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(
        service,
        "retrieve_evidence",
        lambda _: {
            "supported": True,
            "evidence": [{"source": "test.csv", "records": [{"MAE": 20}]}],
        },
    )
    monkeypatch.setattr(service, "format_verified_answer", lambda _: None)
    assert service.answer_question("Which model is best?")["generated"] is True
    assert calls["http_options"] == {"timeout": 15000, "retry_options": {"attempts": 1}}
    assert calls["closed"] is True


def test_pipeline_stops_on_failure(monkeypatch):
    called = []

    def import_step(name):
        called.append(name)

        def fail():
            raise ValueError("invalid data")

        return SimpleNamespace(main=fail)

    monkeypatch.setattr(pipeline.importlib, "import_module", import_step)
    with pytest.raises(ValueError):
        pipeline.run_pipeline("analytics")
    assert called == ["retailiq.analytics.weekly"]


def test_cli_dry_run_does_not_execute(monkeypatch, capsys):
    def fail(_):
        pytest.fail("A dry run must not execute a pipeline stage")

    monkeypatch.setattr(cli, "run_pipeline", fail)
    assert cli.main(["pipeline", "--stage", "analytics", "--list"]) == 0
    assert "retailiq.analytics.weekly" in capsys.readouterr().out


def test_cli_configuration_error_is_nonzero(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "invalid")
    with pytest.raises(SystemExit) as error:
        cli.main(["check-config"])
    assert error.value.code == 2


def test_logging_is_idempotent():
    configure_logging()
    logger = logging.getLogger("retailiq")
    handlers = list(logger.handlers)
    configure_logging()
    assert logger.handlers == handlers


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_rankings_reject_nonfinite_sales(value):
    with pytest.raises(ValueError):
        rank_sales(pd.DataFrame({"Store": [1], "Sales": [value]}), "Store", "Sales")


def test_lags_reject_missing_calendar_week():
    data = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": ["2012-01-06", "2012-01-20"],
            "Weekly_Sales": [100, 200],
            "IsHoliday": [False, False],
        }
    )
    with pytest.raises(ValueError, match="consecutive weekly"):
        build_features(data)
    with pytest.raises(ValueError, match="consecutive weekly"):
        prepare_baseline(data.rename(columns={"Weekly_Sales": "Weekly_Store_Sales"}))


def test_all_package_modules_import():
    for module in pkgutil.walk_packages(retailiq.__path__, prefix="retailiq."):
        if module.name != "retailiq.__main__":
            importlib.import_module(module.name)
