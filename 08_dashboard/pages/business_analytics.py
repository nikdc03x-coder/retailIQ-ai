
"""Business analytics dashboard using saved RetailIQ AI reports."""

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import (
    load_csv,
    load_department_performance,
    load_store_performance,
)


def sales_label(value):
    """Format dataset sales values for readable dashboard metrics."""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    return f"{value:,.2f}"


def chart(fig, key):
    """Apply consistent spacing and display a Plotly chart."""
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def render():
    """Render store, department, trend and holiday analytics."""

    st.title("Business Analytics")
    st.caption(
        "Explore historical store performance, departments, "
        "sales trends and holiday patterns."
    )

    try:
        stores = load_store_performance().copy()
        departments = load_department_performance().copy()
        store_weeks = load_csv(
            "business_analytics/04_store_week_sales.csv"
        ).copy()
        holidays = load_csv(
            "business_analytics/05_holiday_comparison.csv"
        ).copy()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return

    # Verify the required columns before displaying results.
    requirements = [
        (
            stores,
            {"Store", "Type", "Total_Sales", "Average_Weekly_Sales"},
        ),
        (
            departments,
            {"Dept", "Total_Sales", "Stores"},
        ),
        (
            store_weeks,
            {"Store", "Date", "Weekly_Store_Sales", "IsHoliday"},
        ),
        (
            holidays,
            {"IsHoliday", "Average_Store_Week_Sales"},
        ),
    ]

    for data, required in requirements:
        if not required.issubset(data.columns):
            st.error(
                "An analytics report is missing required columns: "
                + ", ".join(sorted(required - set(data.columns)))
            )
            return

    if stores.empty or departments.empty or store_weeks.empty:
        st.warning("One or more analytics reports are empty.")
        return

    store_weeks["Date"] = pd.to_datetime(
        store_weeks["Date"], errors="coerce"
    )

    if store_weeks["Date"].isna().any():
        st.error("Store-week data contains invalid dates.")
        return

    # CSV files may represent booleans as text.
    holiday_values = store_weeks["IsHoliday"].astype(str).str.lower()
    if not holiday_values.isin(["true", "false"]).all():
        st.error("Unexpected holiday values in store-week data.")
        return

    store_weeks["IsHoliday"] = holiday_values.eq("true")

    # Sidebar controls apply to store-related analysis.
    with st.sidebar:
        st.divider()
        st.subheader("Business Filters")

        store_types = sorted(stores["Type"].dropna().unique())

        selected_types = st.multiselect(
            "Store type",
            options=store_types,
            default=store_types,
            key="analytics_store_types",
        )

        eligible_stores = stores[
            stores["Type"].isin(selected_types)
        ].copy()

        available_ids = sorted(
            eligible_stores["Store"].unique().tolist()
        )

        selected_stores = st.multiselect(
            "Select stores",
            options=available_ids,
            default=available_ids,
            key="analytics_selected_stores",
        )

    if not selected_stores:
        st.info("Select at least one store to view analytics.")
        return

    selected = stores[
        stores["Store"].isin(selected_stores)
    ].copy()

    weekly = store_weeks[
        store_weeks["Store"].isin(selected_stores)
    ].copy()

    if weekly.empty:
        st.warning("No observations for the selected stores.")
        return

    # All displayed store metrics use the selected population.
    total_sales = selected["Total_Sales"].sum()
    top_store = selected.loc[selected["Total_Sales"].idxmax()]
    average_store_sales = selected["Total_Sales"].mean()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Selected Store Sales", sales_label(total_sales))
    k2.metric("Stores Selected", len(selected))
    k3.metric("Top Store", int(top_store["Store"]))
    k4.metric(
        "Average Sales per Store",
        sales_label(average_store_sales),
    )

    st.divider()

    # STORE RANKINGS
    st.subheader("Store Performance")

    left, right = st.columns([1, 1])

    with left:
        ranking = st.radio(
            "Ranking",
            ["Top Stores", "Bottom Stores"],
            horizontal=True,
            key="analytics_ranking",
        )

    with right:
        metric = st.selectbox(
            "Compare by",
            ["Total_Sales", "Average_Weekly_Sales"],
            format_func=lambda value: {
                "Total_Sales": "Total sales",
                "Average_Weekly_Sales": "Average weekly sales",
            }[value],
            key="analytics_store_metric",
        )

    limit = st.slider(
        "Number of stores",
        min_value=1,
        max_value=len(selected),
        value=min(10, len(selected)),
        key="analytics_ranking_limit",
    )

    ascending = ranking == "Bottom Stores"

    ranked = selected.sort_values(
        metric, ascending=ascending
    ).head(limit)

    ranked = ranked.assign(
        Store_Label="Store " + ranked["Store"].astype(str)
    )

    fig = px.bar(
        ranked,
        x=metric,
        y="Store_Label",
        orientation="h",
        color="Type",
        hover_data=["Store", "Total_Sales"],
        title=ranking,
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Sales (dataset units)",
        yaxis_title="",
    )

    chart(fig, "analytics_store_ranking_chart")

    # STORE-WEEK TREND
    st.divider()
    st.subheader("Store Sales Trends")

    trend_stores = st.multiselect(
        "Compare individual stores",
        options=selected_stores,
        default=selected_stores[: min(5, len(selected_stores))],
        key="analytics_trend_stores",
    )

    trend_type = st.radio(
        "Trend visualization",
        ["Line", "Bar"],
        horizontal=True,
        key="analytics_trend_type",
    )

    if trend_stores:
        trend = weekly[
            weekly["Store"].isin(trend_stores)
        ].copy()

        trend["Store_Label"] = (
            "Store " + trend["Store"].astype(str)
        )

        if trend_type == "Bar":
            fig = px.bar(
                trend,
                x="Date",
                y="Weekly_Store_Sales",
                color="Store_Label",
                barmode="group",
                title="Weekly Store Sales Comparison",
            )
        else:
            fig = px.line(
                trend,
                x="Date",
                y="Weekly_Store_Sales",
                color="Store_Label",
                title="Weekly Store Sales Comparison",
            )

        fig.update_layout(
            xaxis_title="Week",
            yaxis_title="Sales (dataset units)",
        )
        chart(fig, "analytics_store_trend_chart")
    else:
        st.info("Select stores to compare their weekly sales.")

    # DEPARTMENT PERFORMANCE
    st.divider()
    st.subheader("Department Performance")

    st.caption(
        "Department rankings cover the complete historical dataset. "
        "The store filter does not apply to this section."
    )

    department_limit = st.slider(
        "Departments to display",
        min_value=5,
        max_value=len(departments),
        value=min(15, len(departments)),
        key="analytics_department_limit",
    )

    department_metric = st.selectbox(
        "Department metric",
        ["Total_Sales", "Stores"],
        format_func=lambda value: {
            "Total_Sales": "Total sales",
            "Stores": "Store coverage",
        }[value],
        key="analytics_department_metric",
    )

    top_departments = (
        departments.sort_values(
            department_metric, ascending=False
        )
        .head(department_limit)
        .copy()
    )

    top_departments["Department"] = (
        "Dept " + top_departments["Dept"].astype(str)
    )

    fig = px.bar(
        top_departments,
        x="Department",
        y=department_metric,
        title="Department Rankings",
    )

    fig.update_layout(
        xaxis_title="Department",
        yaxis_title=(
            "Sales (dataset units)"
            if department_metric == "Total_Sales"
            else "Number of Stores"
        ),
    )

    chart(fig, "analytics_department_chart")

    # HOLIDAY COMPARISON
    st.divider()
    st.subheader("Holiday vs Non-Holiday Performance")

    st.caption(
        "Comparison is recalculated from the selected stores' "
        "historical store-week observations."
    )

    holiday_summary = (
        weekly.groupby("IsHoliday", as_index=False)
        .agg(
            Average_Store_Week_Sales=(
                "Weekly_Store_Sales", "mean"
            ),
            Store_Weeks=("Weekly_Store_Sales", "size"),
        )
    )

    holiday_summary["Period"] = holiday_summary[
        "IsHoliday"
    ].map({
        True: "Holiday",
        False: "Non-Holiday",
    })

    fig = px.bar(
        holiday_summary,
        x="Period",
        y="Average_Store_Week_Sales",
        color="Period",
        hover_data=["Store_Weeks"],
        title="Average Sales per Store-Week",
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Average sales (dataset units)",
        showlegend=False,
    )

    chart(fig, "analytics_holiday_comparison_chart")

    st.caption(
        "These are descriptive comparisons, not estimates "
        "of the causal effect of holidays."
    )

    # EXPLORATION TABLES
    st.divider()

    with st.expander("Explore store performance data"):
        st.dataframe(
            selected,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Explore department performance data"):
        st.dataframe(
            departments,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Explore selected store-week data"):
        st.dataframe(
            weekly,
            use_container_width=True,
            hide_index=True,
        )
