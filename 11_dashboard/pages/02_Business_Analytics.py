"""Interactive business analytics for historical Walmart sales."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/retail_analytical.parquet"

st.set_page_config(
    page_title="RetailIQ | Business Analytics",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data():
    """Read the verified dataset once and reuse it across reruns."""
    return pd.read_parquet(SOURCE)


def show_chart(data, x, y, title, chart_type, color=None):
    """Display a Plotly bar or line chart."""
    if chart_type == "Line":
        fig = px.line(
            data, x=x, y=y, color=color,
            title=title, markers=True,
        )
    else:
        fig = px.bar(
            data, x=x, y=y, color=color,
            title=title,
        )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)


st.title("RetailIQ — Business Analytics")
st.caption(
    "Historical Walmart sales | February 2010 – October 2012"
)

if not SOURCE.is_file():
    st.error(
        "Analytical dataset not found. "
        "Run Phase 02 transformation first."
    )
    st.stop()

data = load_data()

# Filters operate on the original store-department-week records.
st.sidebar.header("Analysis Filters")

store_types = sorted(data["Type"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect(
    "Store type",
    store_types,
    default=store_types,
)

type_filtered = data[data["Type"].isin(selected_types)]

stores = sorted(type_filtered["Store"].unique().tolist())
selected_stores = st.sidebar.multiselect(
    "Stores",
    stores,
    default=stores,
)

store_filtered = type_filtered[
    type_filtered["Store"].isin(selected_stores)
]

departments = sorted(store_filtered["Dept"].unique().tolist())
selected_departments = st.sidebar.multiselect(
    "Departments",
    departments,
    default=departments,
)

filtered = store_filtered[
    store_filtered["Dept"].isin(selected_departments)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# Display historical sales indicators.
weekly = (
    filtered.groupby("Date", as_index=False)
    .agg(Total_Sales=("Weekly_Sales", "sum"))
    .sort_values("Date")
)

total_sales = filtered["Weekly_Sales"].sum()
average_weekly = weekly["Total_Sales"].mean()

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total recorded sales", f"{total_sales / 1e9:,.2f}B")
k2.metric("Average weekly sales", f"{average_weekly / 1e6:,.2f}M")
k3.metric("Stores selected", filtered["Store"].nunique())
k4.metric("Weeks recorded", filtered["Date"].nunique())

st.divider()

# Weekly trend section.
st.subheader("Weekly Sales Trends")

trend_type = st.radio(
    "Trend chart",
    ["Line", "Bar"],
    horizontal=True,
    key="trend_chart_type",
)

show_chart(
    weekly,
    "Date",
    "Total_Sales",
    "Historical Weekly Sales",
    trend_type,
)

# Store and department rankings.
left, right = st.columns(2)

with left:
    st.subheader("Store Performance")

    store_sales = (
        filtered.groupby("Store", as_index=False)
        .agg(Total_Sales=("Weekly_Sales", "sum"))
        .sort_values("Total_Sales", ascending=False)
        .head(10)
    )

    store_sales["Store_Label"] = (
        "Store " + store_sales["Store"].astype(str)
    )

    store_chart = st.selectbox(
        "Store chart",
        ["Bar", "Line"],
        key="store_chart_type",
    )

    show_chart(
        store_sales,
        "Store_Label",
        "Total_Sales",
        "Top Stores by Recorded Sales",
        store_chart,
    )

with right:
    st.subheader("Department Performance")

    department_sales = (
        filtered.groupby("Dept", as_index=False)
        .agg(Total_Sales=("Weekly_Sales", "sum"))
        .sort_values("Total_Sales", ascending=False)
        .head(10)
    )

    department_sales["Department_Label"] = (
        "Dept " + department_sales["Dept"].astype(str)
    )

    department_chart = st.selectbox(
        "Department chart",
        ["Bar", "Line"],
        key="department_chart_type",
    )

    show_chart(
        department_sales,
        "Department_Label",
        "Total_Sales",
        "Top Departments by Recorded Sales",
        department_chart,
    )

# Compare holiday and non-holiday store-week observations.
st.divider()
st.subheader("Holiday Sales Comparison")

store_week = (
    filtered.groupby(["Store", "Date", "IsHoliday"], as_index=False)
    .agg(Weekly_Store_Sales=("Weekly_Sales", "sum"))
)

holiday_summary = (
    store_week.groupby("IsHoliday", as_index=False)
    .agg(
        Average_Sales=("Weekly_Store_Sales", "mean"),
        Store_Weeks=("Weekly_Store_Sales", "size"),
    )
)

holiday_summary["Period"] = holiday_summary[
    "IsHoliday"
].map({True: "Holiday", False: "Non-holiday"})

fig = px.bar(
    holiday_summary,
    x="Period",
    y="Average_Sales",
    color="Period",
    title="Average Sales per Store-Week",
    text_auto=".2s",
)

fig.update_layout(
    template="plotly_white",
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    holiday_summary[
        ["Period", "Store_Weeks", "Average_Sales"]
    ],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Comparisons are descriptive, not causal. "
    "Changing filters changes the population being analyzed. "
    "Sales figures are not profits."
)

with st.expander("About the dataset"):
    st.write(
        "Source: historical Walmart store-department-week "
        "sales data. The analytical dataset was validated "
        "during RetailIQ Phase 02."
    )
    st.write("Filtered records:", len(filtered))
    st.write(
        "Date range:",
        filtered["Date"].min().date(),
        "to",
        filtered["Date"].max().date(),
    )