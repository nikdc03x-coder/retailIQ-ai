"""Business overview page for RetailIQ AI."""

import plotly.express as px
import streamlit as st

from data_loader import load_weekly_sales


def render():
    """Show business KPIs and historical sales patterns."""

    st.title("Retail Business Overview")
    st.caption(
        "Historical retail performance across the available dataset"
    )

    try:
        sales = load_weekly_sales()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return

    if sales.empty:
        st.warning("No weekly sales observations are available.")
        return

    # Use a date filter without modifying the original report.
    minimum = sales["Date"].min().date()
    maximum = sales["Date"].max().date()

    selected = st.date_input(
        "Analysis period",
        value=(minimum, maximum),
        min_value=minimum,
        max_value=maximum,
    )

    if not isinstance(selected, (tuple, list)) or len(selected) != 2:
        st.info("Select a start and end date.")
        return

    start, end = selected

    filtered = sales[
        sales["Date"].dt.date.between(start, end)
    ].copy()

    if filtered.empty:
        st.warning("No weekly sales data in this date range.")
        return

    total_sales = filtered["Total_Sales"].sum()
    average_weekly_sales = filtered["Total_Sales"].mean()
    highest_week = filtered.loc[
        filtered["Total_Sales"].idxmax()
    ]

    first, second, third, fourth = st.columns(4)

    first.metric(
        "Total Sales",
        f"{total_sales / 1_000_000:,.2f}M",
    )

    second.metric(
        "Average Weekly Sales",
        f"{average_weekly_sales / 1_000_000:,.2f}M",
    )

    third.metric(
        "Weeks Analyzed",
        f"{len(filtered):,}",
    )

    fourth.metric(
        "Highest Sales Week",
        highest_week["Date"].strftime("%d %b %Y"),
    )

    st.divider()

    st.subheader("Weekly Sales Performance")

    chart_type = st.segmented_control(
        "Chart type",
        options=["Line", "Bar"],
        default="Line",
        key="overview_sales_chart_type",
    )

    if chart_type == "Bar":
        fig = px.bar(
            filtered,
            x="Date",
            y="Total_Sales",
            title="Historical Weekly Sales",
        )
    else:
        fig = px.line(
            filtered,
            x="Date",
            y="Total_Sales",
            title="Historical Weekly Sales",
        )

    fig.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Week",
        yaxis_title="Sales (dataset units)",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="overview_weekly_sales",
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Monthly Sales")

        monthly = (
            filtered.assign(
                Month=filtered["Date"].dt.to_period("M").dt.to_timestamp()
            )
            .groupby("Month", as_index=False)["Total_Sales"]
            .sum()
        )

        monthly_fig = px.bar(
            monthly,
            x="Month",
            y="Total_Sales",
        )

        monthly_fig.update_layout(
            height=350,
            xaxis_title="Month",
            yaxis_title="Sales (dataset units)",
        )

        st.plotly_chart(
            monthly_fig,
            width="stretch",
            key="overview_monthly_sales",
        )

    with right:
        st.subheader("Weekly Sales Distribution")

        distribution = px.histogram(
            filtered,
            x="Total_Sales",
            nbins=20,
        )

        distribution.update_layout(
            height=350,
            xaxis_title="Weekly Sales",
            yaxis_title="Number of Weeks",
        )

        st.plotly_chart(
            distribution,
            width="stretch",
            key="overview_sales_distribution",
        )

    with st.expander("Explore underlying weekly sales data"):
        st.dataframe(
            filtered,
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "Descriptive analysis only. These historical values "
        "are not future sales predictions."
    )