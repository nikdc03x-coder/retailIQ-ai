"""Load saved RetailIQ AI research outputs for the dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "13_results"


@st.cache_data(show_spinner=False)
def load_csv(relative_path: str) -> pd.DataFrame:
    """Read an existing report without rerunning any experiment."""

    path = RESULTS / relative_path

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing report: {path.relative_to(ROOT)}"
        )

    return pd.read_csv(path)


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