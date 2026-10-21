"""Explore the reliability of historical sales forecasts."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from retailiq.dashboard.data_loader import load_csv


def calculate_store_errors(predictions):
    """Calculate evaluation metrics separately for each store."""

    data = predictions.copy()

    data["Absolute_Error"] = (data["Weekly_Store_Sales"] - data["Prediction"]).abs()

    data["Absolute_Actual"] = data["Weekly_Store_Sales"].abs()

    summary = data.groupby("Store", as_index=False).agg(
        MAE=("Absolute_Error", "mean"),
        Total_Absolute_Error=("Absolute_Error", "sum"),
        Total_Absolute_Actual=("Absolute_Actual", "sum"),
        Weeks=("Date", "nunique"),
    )

    summary["WAPE"] = np.where(
        summary["Total_Absolute_Actual"] > 0,
        100 * summary["Total_Absolute_Error"] / summary["Total_Absolute_Actual"],
        np.nan,
    )

    return summary


def render_reliability_experiments():
    """Display saved rolling backtests and holiday error comparisons."""

    st.divider()
    st.subheader("Rolling Backtest Performance")

    try:
        backtests = load_csv("forecast_reliability/06_rolling_backtest_results.csv").copy()

        holidays = load_csv("forecast_reliability/04_holiday_error_summary.csv").copy()

    except (FileNotFoundError, pd.errors.EmptyDataError) as error:
        st.error(f"Unable to load reliability reports: {error}")
        return

    required_backtests = {"Fold", "Baseline_MAE", "XGBoost_MAE", "Improvement_Pct", "Observations"}
    required_holidays = {"IsHoliday", "Observations", "MAE", "RMSE"}

    if not required_backtests.issubset(backtests.columns):
        st.error("Rolling backtest report has missing columns.")
        return

    if not required_holidays.issubset(holidays.columns):
        st.error("Holiday reliability report has missing columns.")
        return

    if backtests.empty or holidays.empty:
        st.warning("One or more reliability reports are empty.")
        return

    # Transform the saved metrics into a chart-friendly structure.
    long_backtests = backtests.melt(
        id_vars=["Fold"],
        value_vars=["Baseline_MAE", "XGBoost_MAE"],
        var_name="Model",
        value_name="MAE",
    )

    long_backtests["Model"] = long_backtests["Model"].replace(
        {
            "Baseline_MAE": "Baseline",
            "XGBoost_MAE": "XGBoost",
        }
    )

    long_backtests["Fold"] = "Fold " + long_backtests["Fold"].astype(str)

    fig = px.bar(
        long_backtests,
        x="Fold",
        y="MAE",
        color="Model",
        barmode="group",
        title="Baseline vs XGBoost Across Backtest Folds",
    )

    fig.update_layout(
        height=430,
        yaxis_title="MAE (sales units)",
        xaxis_title="Backtest fold",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="reliability_backtest_comparison",
    )

    st.subheader("Improvement Across Backtests")

    improvement = backtests[["Fold", "Improvement_Pct"]].copy()

    improvement["Fold"] = "Fold " + improvement["Fold"].astype(str)

    fig = px.bar(
        improvement,
        x="Fold",
        y="Improvement_Pct",
        text="Improvement_Pct",
        title="XGBoost MAE Improvement Over Baseline",
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig.update_layout(
        height=350,
        xaxis_title="Backtest fold",
        yaxis_title="MAE improvement (%)",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="reliability_backtest_improvement",
    )

    st.caption(
        "These expanding-window experiments evaluate one-week-ahead "
        "predictions using available prior actual sales. "
        "They are not eight-week-ahead forecasts."
    )

    st.divider()
    st.subheader("Holiday Forecast Reliability")

    holidays["Period"] = (
        holidays["IsHoliday"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": "Holiday",
                "false": "Non-Holiday",
            }
        )
    )

    if holidays["Period"].isna().any():
        st.error("Unexpected holiday labels in reliability report.")
        return

    holiday_metric = st.radio(
        "Holiday comparison metric",
        ["MAE", "RMSE"],
        horizontal=True,
        key="reliability_holiday_metric",
    )

    fig = px.bar(
        holidays,
        x="Period",
        y=holiday_metric,
        color="Period",
        text=holiday_metric,
        title=f"Holiday vs Non-Holiday {holiday_metric}",
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
    )

    fig.update_layout(
        height=390,
        showlegend=False,
        xaxis_title="",
        yaxis_title=f"{holiday_metric} (sales units)",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="reliability_holiday_comparison",
    )

    st.caption(
        "Holiday and non-holiday groups contain different numbers "
        "of observations. This is a descriptive comparison."
    )

    with st.expander("View rolling backtest results"):
        st.dataframe(
            backtests,
            width="stretch",
            hide_index=True,
        )

    with st.expander("View holiday error results"):
        st.dataframe(
            holidays,
            width="stretch",
            hide_index=True,
        )


def render():
    """Display forecast errors and store-level reliability."""

    st.title("Forecast Reliability")

    st.caption(
        "Evaluate where historical XGBoost forecasts performed "
        "well and where prediction errors were higher."
    )

    try:
        predictions = load_csv("sales_forecasting/10_xgboost_predictions.csv").copy()
    except (FileNotFoundError, pd.errors.EmptyDataError) as error:
        st.error(str(error))
        return

    required = {
        "Store",
        "Date",
        "Split",
        "Weekly_Store_Sales",
        "Prediction",
    }

    if not required.issubset(predictions.columns):
        st.error("Missing required columns: " + str(sorted(required - set(predictions.columns))))
        return

    # Reliability findings are based on the held-out test split.
    test = predictions[predictions["Split"].astype(str).str.lower() == "test"].copy()

    if test.empty:
        st.warning("No test predictions are available.")
        return

    test["Date"] = pd.to_datetime(test["Date"], errors="coerce")

    if test["Date"].isna().any():
        st.error("Invalid dates in prediction results.")
        return

    test["Absolute_Error"] = (test["Weekly_Store_Sales"] - test["Prediction"]).abs()

    test["Absolute_Actual"] = test["Weekly_Store_Sales"].abs()

    total_abs_error = test["Absolute_Error"].sum()
    total_abs_actual = test["Absolute_Actual"].sum()

    overall_wape = 100 * total_abs_error / total_abs_actual if total_abs_actual > 0 else np.nan

    overall_mae = test["Absolute_Error"].mean()

    summary = calculate_store_errors(test)

    highest_mae = summary.loc[summary["MAE"].idxmax()]

    valid_wape = summary.dropna(subset=["WAPE"])

    highest_wape = valid_wape.loc[valid_wape["WAPE"].idxmax()] if not valid_wape.empty else None

    a, b, c, d = st.columns(4)

    a.metric("Test MAE", f"{overall_mae:,.2f}")
    b.metric(
        "Test WAPE",
        f"{overall_wape:.2f}%" if np.isfinite(overall_wape) else "N/A",
    )
    c.metric("Highest MAE Store", int(highest_mae["Store"]))
    d.metric(
        "Highest WAPE Store",
        int(highest_wape["Store"]) if highest_wape is not None else "N/A",
    )

    st.divider()

    st.subheader("Store Reliability Comparison")

    metric = st.radio(
        "Error measure",
        ["MAE", "WAPE"],
        horizontal=True,
        key="reliability_metric",
    )

    count = st.slider(
        "Number of stores",
        5,
        len(summary),
        min(15, len(summary)),
        key="reliability_count",
    )

    ranked = summary.dropna(subset=[metric]).sort_values(metric, ascending=False).head(count).copy()

    ranked["Store_Label"] = "Store " + ranked["Store"].astype(str)

    fig = px.bar(
        ranked,
        x="Store_Label",
        y=metric,
        title=f"Stores with Highest {metric}",
    )

    fig.update_layout(
        height=430,
        xaxis_title="Store",
        yaxis_title=metric,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="reliability_store_ranking",
    )

    st.divider()
    st.subheader("Weekly Forecast Errors")

    weekly = test.groupby("Date", as_index=False).agg(
        MAE=("Absolute_Error", "mean"),
        Total_Absolute_Error=("Absolute_Error", "sum"),
        Total_Absolute_Actual=("Absolute_Actual", "sum"),
    )

    weekly["WAPE"] = np.where(
        weekly["Total_Absolute_Actual"] > 0,
        100 * weekly["Total_Absolute_Error"] / weekly["Total_Absolute_Actual"],
        np.nan,
    )

    fig = px.line(
        weekly,
        x="Date",
        y=metric,
        markers=True,
        title=f"Weekly Test {metric}",
    )

    fig.update_layout(
        height=400,
        xaxis_title="Week",
        yaxis_title=metric,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="reliability_weekly_errors",
    )

    with st.expander("View store reliability metrics"):
        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "These results describe historical test-set errors. "
        "They do not guarantee future model performance."
    )
    # Show the saved reliability experiments.
    render_reliability_experiments()
