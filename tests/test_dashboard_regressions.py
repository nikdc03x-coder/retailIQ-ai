"""Exercise dashboard filtering, file refresh and invalid-data recovery."""

import os

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from retailiq.dashboard import readiness
from retailiq.dashboard.pages import business_analytics, dataset_overview, sales_explorer


def render_page(module):
    return AppTest.from_string(
        f"from retailiq.dashboard.pages import {module}\n{module}.render()",
        default_timeout=15,
    ).run()


@pytest.fixture
def analytics_reports(monkeypatch):
    stores = pd.DataFrame(
        {
            "Store": [1, 2],
            "Type": ["A", "B"],
            "Total_Sales": [100.0, 200.0],
            "Average_Weekly_Sales": [50.0, 100.0],
        }
    )
    departments = pd.DataFrame({"Dept": [1], "Total_Sales": [300.0], "Stores": [2]})
    weeks = pd.DataFrame(
        {
            "Store": [1, 2],
            "Date": ["2012-01-06", "2012-01-06"],
            "Weekly_Store_Sales": [100.0, 200.0],
            "IsHoliday": [False, False],
        }
    )
    holidays = pd.DataFrame({"IsHoliday": [False], "Average_Store_Week_Sales": [150.0]})
    monkeypatch.setattr(business_analytics, "load_store_performance", lambda: stores.copy())
    monkeypatch.setattr(
        business_analytics, "load_department_performance", lambda: departments.copy()
    )
    monkeypatch.setattr(
        business_analytics,
        "load_csv",
        lambda path: weeks.copy() if "04_store_week" in path else holidays.copy(),
    )


@pytest.mark.parametrize("selection", [[1], [], [1, 2]])
def test_store_filter_handles_one_none_and_many(analytics_reports, selection):
    app = render_page("business_analytics")
    assert not app.exception
    app.multiselect(key="analytics_selected_stores").set_value(selection).run()
    assert not app.exception
    assert not app.error
    if selection:
        assert app.metric[1].value == str(len(selection))
        assert len(app.get("plotly_chart")) == 4
    else:
        assert "Select at least one store" in app.info[0].value


def test_store_type_filter_can_reduce_to_one_store(analytics_reports):
    app = render_page("business_analytics")
    app.multiselect(key="analytics_store_types").set_value(["A"]).run()
    assert not app.exception
    assert not app.error
    assert app.metric[1].value == "1"


@pytest.fixture
def sources(tmp_path):
    sales = pd.DataFrame(
        {
            "Store": [1],
            "Dept": [1],
            "Date": ["2012-01-06"],
            "Weekly_Sales": [100.0],
            "IsHoliday": [False],
        }
    )
    sales.to_csv(tmp_path / "train.csv", index=False)
    sales[["Store", "Date", "IsHoliday"]].to_csv(tmp_path / "features.csv", index=False)
    pd.DataFrame({"Store": [1], "Type": ["A"], "Size": [100]}).to_csv(
        tmp_path / "stores.csv",
        index=False,
    )
    sales.assign(Type="A").to_parquet(tmp_path / "sales.parquet")
    return tmp_path


def bump_mtime(path, old_stat):
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000_000))


def test_parquet_replacement_refreshes_sales_explorer(sources, monkeypatch):
    path = sources / "sales.parquet"
    monkeypatch.setattr(sales_explorer, "SOURCE", path)
    assert sales_explorer.load_data()["Weekly_Sales"].iloc[0] == 100
    stat = path.stat()
    changed = pd.read_parquet(path)
    changed["Weekly_Sales"] = 200.0
    changed.to_parquet(path)
    bump_mtime(path, stat)
    assert sales_explorer.load_data()["Weekly_Sales"].iloc[0] == 200


def test_csv_replacement_refreshes_dataset_overview(sources, monkeypatch):
    monkeypatch.setattr(dataset_overview, "RAW_DIR", sources)
    assert dataset_overview.load_data()["Sales"]["Weekly_Sales"].iloc[0] == 100
    path = sources / "train.csv"
    stat = path.stat()
    changed = pd.read_csv(path)
    changed["Weekly_Sales"] = 200.0
    changed.to_csv(path, index=False)
    bump_mtime(path, stat)
    assert dataset_overview.load_data()["Sales"]["Weekly_Sales"].iloc[0] == 200


@pytest.mark.parametrize(
    "content",
    [
        "Store,Dept,Weekly_Sales,IsHoliday\n1,1,100,False\n",
        "",
        "Store,Dept,Date,Weekly_Sales,IsHoliday\n1,1,invalid,100,False\n",
        "Store,Dept,Date,Weekly_Sales,IsHoliday\n1,1,2012-01-06,inf,False\n",
    ],
)
def test_bad_raw_sources_show_recovery_message(sources, monkeypatch, content):
    monkeypatch.setattr(dataset_overview, "RAW_DIR", sources)
    (sources / "train.csv").write_text(content, encoding="utf-8")
    app = render_page("dataset_overview")
    assert not app.exception
    assert "Check the CSV files in data/raw" in app.error[0].value


@pytest.mark.parametrize("kind", ["missing", "corrupt", "schema", "empty", "holiday"])
def test_bad_parquet_shows_rebuild_message(sources, monkeypatch, kind):
    path = sources / f"{kind}.parquet"
    if kind == "corrupt":
        path.write_bytes(b"not a parquet file")
    elif kind != "missing":
        data = pd.read_parquet(sources / "sales.parquet")
        if kind == "schema":
            data = data.drop(columns="Date")
        elif kind == "empty":
            data = data.iloc[:0]
        else:
            data["IsHoliday"] = "unexpected"
        data.to_parquet(path)
    monkeypatch.setattr(sales_explorer, "SOURCE", path)
    app = render_page("sales_explorer")
    assert not app.exception
    assert "pipeline --stage data" in app.error[0].value


def test_string_dates_and_holiday_flags_render(sources, monkeypatch):
    path = sources / "sales.parquet"
    data = pd.read_parquet(path)
    data["IsHoliday"] = "False"
    data["Weekly_Sales"] = -10.0
    data.to_parquet(path)
    monkeypatch.setattr(sales_explorer, "SOURCE", path)
    app = render_page("sales_explorer")
    assert not app.exception
    assert not app.error
    assert app.metric[3].value == "1"


def test_preflight_covers_every_required_report_schema():
    reports = {
        p.removeprefix("reports/") for p in readiness.REQUIRED_DATA if p.startswith("reports/")
    }
    assert reports == set(readiness.REPORT_SCHEMAS)


def test_preflight_rejects_wrong_columns_and_corrupt_parquet(tmp_path, monkeypatch):
    paths = (
        "reports/business_analytics/01_weekly_sales.csv",
        "data/processed/retail_analytical.parquet",
    )
    monkeypatch.setattr(readiness, "REQUIRED_DATA", paths)
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("WrongColumn\nvalue\n", encoding="utf-8")
    issues = readiness.find_data_issues(tmp_path)
    assert len(issues) == 2
    assert "missing required columns" in issues[0]
    assert paths[1] in issues[1]
