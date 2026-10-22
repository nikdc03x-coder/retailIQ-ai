"""
RetailIQ AI — Dataset Overview

Input: Original Walmart CSV files
Output: Interactive Streamlit dataset overview
"""

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from retailiq.config import ROOT
from retailiq.dashboard.data_loader import read_csv_file
from retailiq.dashboard.validation import RAW_SCHEMAS, validate_source

RAW_DIR = ROOT / "data" / "raw"

DATA_FILES = {
    "Sales": "train.csv",
    "Features": "features.csv",
    "Stores": "stores.csv",
}
logger = logging.getLogger(__name__)


def load_data():
    """Read the original datasets without modifying them."""

    datasets = {}

    for name, filename in DATA_FILES.items():
        path = RAW_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset: {path}")

        datasets[name] = validate_source(read_csv_file(path), RAW_SCHEMAS[filename], filename)

    return datasets


def render():
    """Render this page inside the shared dashboard navigation."""
    st.title("Dataset Overview")

    st.caption("Data understanding and quality assessment")

    st.markdown(
        "Explore the historical retail data used in RetailIQ AI. "
        "All metrics below are calculated from the source CSV files."
    )

    try:
        datasets = load_data()
    except (OSError, ValueError, TypeError) as error:
        logger.warning("Cannot load Dataset Overview sources: %s", error, exc_info=True)
        st.error(f"Cannot load source data. {error} Check the CSV files in data/raw and retry.")
        return

    sales = datasets["Sales"]

    sales_dates = pd.to_datetime(sales["Date"], errors="coerce")

    valid_dates = sales_dates.dropna()

    if valid_dates.empty:
        st.error("No valid sales dates were found.")
        return

    st.subheader("Dataset at a glance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Sales Records", f"{len(sales):,}")

    col2.metric("Stores", f"{sales['Store'].nunique():,}")

    col3.metric("Departments", f"{sales['Dept'].nunique():,}")

    col4.metric("Unique Weeks", f"{valid_dates.nunique():,}")

    st.info(
        f"Historical coverage: {valid_dates.min():%d %b %Y} "
        f"to {valid_dates.max():%d %b %Y}. "
        "Each sales record describes one store, department and week."
    )

    st.subheader("Data source")

    st.markdown(
        "This project uses a public Kaggle copy of the historical Walmart sales forecasting dataset."
    )

    st.link_button(
        "View dataset source",
        "https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast",
    )

    st.caption(
        "The data covers 2010–2012. It supports historical analysis "
        "and model evaluation, not claims about current Walmart performance. "
        "Check the source's copyright and reuse terms before redistribution."
    )

    st.subheader("Explore source files")

    selected = st.selectbox(
        "Select dataset",
        list(datasets.keys()),
    )

    current = datasets[selected]

    left, right = st.columns(2)

    left.metric("Rows", f"{len(current):,}")

    right.metric("Columns", len(current.columns))

    st.dataframe(current.head(20), use_container_width=True)

    with st.expander("Column names and data types"):
        schema = pd.DataFrame(
            {
                "Column": current.columns,
                "Data type": current.dtypes.astype(str).values,
            }
        )
        st.dataframe(schema, use_container_width=True, hide_index=True)

    st.subheader("Missing-value analysis")

    missing = current.isna().sum().reset_index()

    missing.columns = ["Column", "Missing values"]

    missing["Missing %"] = (missing["Missing values"] / len(current) * 100).round(2)

    missing = missing.sort_values("Missing values", ascending=False)

    st.dataframe(missing, use_container_width=True, hide_index=True)

    if missing["Missing values"].sum() > 0:
        fig_missing = px.bar(
            missing[missing["Missing values"] > 0],
            x="Column",
            y="Missing values",
            hover_data=["Missing %"],
            title=f"Missing values — {selected}",
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("No missing values found in this source file.")

    with st.expander("How to interpret missing values"):
        st.write(
            "Missing values mean information is unavailable. "
            "They do not automatically mean the data is incorrect. "
            "Investigate their meaning before deciding "
            "whether to fill, retain or exclude them."
        )

    st.subheader("Historical weekly sales")

    weekly_sales = (
        sales.assign(Date=sales_dates)
        .dropna(subset=["Date"])
        .groupby("Date", as_index=False)["Weekly_Sales"]
        .sum()
        .sort_values("Date")
    )

    fig_sales = px.line(
        weekly_sales,
        x="Date",
        y="Weekly_Sales",
        title="Total sales across stores and departments",
        labels={
            "Date": "Week",
            "Weekly_Sales": "Total weekly sales",
        },
    )

    st.plotly_chart(fig_sales, use_container_width=True)

    with st.expander("How this chart works"):
        st.markdown(
            "**Metric:** Sum of Weekly_Sales for each recorded date.\n\n"
            "**Why a line chart?** It shows changes over time.\n\n"
            "**Business use:** Identify seasonal patterns and unusual weeks "
            "for further investigation.\n\n"
            "**Limitation:** This is descriptive analysis, not a forecast. "
            "The chart does not establish what caused sales changes."
        )

    st.subheader("Research observations")

    st.markdown(
        "- Historical data is available at store–department–week level.\n"
        "- Additional feature columns contain missing information.\n"
        "- Sales trends can be examined across the recorded weeks.\n"
        "- Data cleaning and validation are required before modeling."
    )

    st.caption(
        "Quality checks validate join keys, investigate missing values "
        "and prepare a reproducible analytical dataset."
    )
