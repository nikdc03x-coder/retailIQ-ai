"""Load saved RetailIQ AI research outputs for the dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st

from retailiq.reports import resolve_report_path


@st.cache_data(show_spinner=False)
def _read_csv(path: Path, modified_ns: int, size: int) -> pd.DataFrame:
    """Cache by file identity and metadata, not just the report name."""
    return pd.read_csv(path)


def load_csv(relative_path: str) -> pd.DataFrame:
    """Read a confined report and invalidate the cache when it is regenerated."""
    path = resolve_report_path(relative_path)
    stat = path.stat()
    return _read_csv(path, stat.st_mtime_ns, stat.st_size)


def load_weekly_sales() -> pd.DataFrame:
    """Return the verified 143-week retail sales report."""

    data = load_csv("business_analytics/01_weekly_sales.csv").copy()

    required = {
        "Date",
        "Total_Sales",
        "Average_Department_Sales",
        "Sales_Records",
    }

    if not required.issubset(data.columns):
        raise ValueError("Weekly sales report has missing columns.")

    data["Date"] = pd.to_datetime(data["Date"])

    if data["Date"].isna().any():
        raise ValueError("Weekly sales report contains invalid dates.")

    if data["Date"].duplicated().any():
        raise ValueError("Weekly sales report contains duplicate weeks.")

    return data.sort_values("Date").reset_index(drop=True)


def load_store_performance() -> pd.DataFrame:
    """Load the existing store-level business report."""

    return load_csv("business_analytics/02_store_performance.csv")


def load_department_performance() -> pd.DataFrame:
    """Load the existing department-level business report."""

    return load_csv("business_analytics/03_department_performance.csv")
