
"""Visualize saved historical sales forecasting results."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
from data_loader import load_csv

def render_model_comparison():
    """Compare saved models and explain XGBoost feature importance."""

    st.divider()
    st.subheader("Model Performance Comparison")

    try:
        comparison = load_csv(
            "sales_forecasting/12_aligned_model_comparison.csv"
        ).copy()

        importance = load_csv(
            "sales_forecasting/11_xgboost_importance.csv"
        ).copy()

    except (FileNotFoundError, pd.errors.EmptyDataError) as error:
        st.error(f"Unable to load model comparison: {error}")
        return

    required_comparison = {
        "Model", "Split", "Observations", "MAE", "RMSE"
    }
    required_importance = {"Feature", "Importance"}

    if not required_comparison.issubset(comparison.columns):
        st.error("Model comparison report has missing columns.")
        return

    if not required_importance.issubset(importance.columns):
        st.error("Feature importance report has missing columns.")
        return

    if comparison.empty or importance.empty:
        st.warning("One or more model reports are empty.")
        return

    # Compare models only within the same evaluation split.
    available_splits = [
        name for name in ["Train", "Validation", "Test"]
        if name in comparison["Split"].unique()
    ]

    if not available_splits:
        st.warning("No recognized model evaluation splits.")
        return

    selected_split = st.selectbox(
        "Model evaluation split",
        options=available_splits,
        index=(
            available_splits.index("Test")
            if "Test" in available_splits else 0
        ),
        key="model_comparison_split",
    )

    selected_metric = st.radio(
        "Performance metric",
        ["MAE", "RMSE"],
        horizontal=True,
        key="model_comparison_metric",
    )

    results = comparison[
        comparison["Split"] == selected_split
    ].copy()

    results = results.sort_values(selected_metric)

    if results.empty:
        st.info("No model results for this evaluation split.")
        return

    fig = px.bar(
        results,
        x="Model",
        y=selected_metric,
        color="Model",
        text=selected_metric,
        title=f"{selected_split} {selected_metric} Comparison",
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
    )

    fig.update_layout(
        height=430,
        showlegend=False,
        xaxis_title="Model",
        yaxis_title=f"{selected_metric} (sales units)",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="forecast_model_comparison_chart",
    )

    best = results.iloc[0]

    st.info(
        f"Lowest {selected_metric} on the {selected_split.lower()} "
        f"split: {best['Model']} "
        f"({best[selected_metric]:,.2f})."
    )

    st.caption(
        "Lower MAE and RMSE indicate smaller prediction errors. "
        "Models are compared on the same saved evaluation split."
    )

    st.divider()
    st.subheader("XGBoost Feature Importance")

    importance = importance.sort_values(
        "Importance", ascending=True
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Saved XGBoost Feature Importance",
    )

    fig.update_layout(
        height=450,
        xaxis_title="Reported importance",
        yaxis_title="Feature",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="forecast_feature_importance_chart",
    )

    st.caption(
        "Feature importance describes the trained model's "
        "reported reliance on features. It does not establish "
        "causal relationships."
    )

    with st.expander("View model comparison data"):
        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
        )

    with st.expander("View feature importance data"):
        st.dataframe(
            importance,
            width="stretch",
            hide_index=True,
        )
 

def render():
    """Display historical XGBoost predictions and evaluation metrics."""

    st.title("Sales Forecasting")
    st.caption(
        "Historical model evaluation using saved XGBoost predictions. "
        "These are not live or future forecasts."
    )

    try:
        data = load_csv(
            "sales_forecasting/10_xgboost_predictions.csv"
        ).copy()
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

    missing = required - set(data.columns)

    if missing:
        st.error(f"Missing prediction columns: {sorted(missing)}")
        return

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    if data["Date"].isna().any():
        st.error("Invalid prediction dates.")
        return

    if data.duplicated(["Store", "Date"]).any():
        st.error("Duplicate store-week predictions detected.")
        return

    # The saved evaluation results include validation and test data.
    available_splits = [
        value for value in ["Validation", "Test"]
        if value in data["Split"].unique()
    ]

    if not available_splits:
        st.warning("No recognized evaluation splits found.")
        return

    with st.sidebar:
        st.divider()
        st.subheader("Forecast Filters")

        split = st.selectbox(
            "Evaluation period",
            available_splits,
            index=(
                available_splits.index("Test")
                if "Test" in available_splits else 0
            ),
            key="forecast_split",
        )

        stores = sorted(data["Store"].unique().tolist())

        selected_store = st.selectbox(
            "Store",
            stores,
            key="forecast_store",
        )

    filtered = data[
        (data["Split"] == split)
        & (data["Store"] == selected_store)
    ].sort_values("Date").copy()

    if filtered.empty:
        st.warning("No predictions for the selected filters.")
        return

    actual = filtered["Weekly_Store_Sales"].to_numpy()
    predicted = filtered["Prediction"].to_numpy()

    errors = actual - predicted

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    denominator = float(np.abs(actual).sum())
    wape = (
        float(np.abs(errors).sum() / denominator * 100)
        if denominator > 0 else np.nan
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Selected Store", int(selected_store))
    k2.metric("MAE", f"{mae:,.2f}")
    k3.metric("RMSE", f"{rmse:,.2f}")
    k4.metric(
        "WAPE",
        f"{wape:.2f}%" if np.isfinite(wape) else "N/A",
    )

    st.divider()
    st.subheader("Actual vs Predicted Sales")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered["Date"],
            y=filtered["Weekly_Store_Sales"],
            mode="lines+markers",
            name="Actual",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=filtered["Date"],
            y=filtered["Prediction"],
            mode="lines+markers",
            name="XGBoost Prediction",
        )
    )

    fig.update_layout(
        height=450,
        xaxis_title="Week",
        yaxis_title="Weekly store sales (dataset units)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="forecast_actual_vs_predicted",
    )

    st.subheader("Weekly Forecast Errors")

    filtered["Residual"] = errors

    error_fig = go.Figure()

    error_fig.add_trace(
        go.Bar(
            x=filtered["Date"],
            y=filtered["Residual"],
            name="Actual minus predicted",
        )
    )

    error_fig.add_hline(y=0, line_dash="dash")

    error_fig.update_layout(
        height=350,
        xaxis_title="Week",
        yaxis_title="Forecast residual",
        margin=dict(l=10, r=10, t=30, b=10),
    )

    st.plotly_chart(
        error_fig,
        width="stretch",
        key="forecast_error_chart",
    )

    with st.expander("View saved prediction records"):
        st.dataframe(
            filtered,
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "Metrics shown above are recalculated for the selected "
        "store and evaluation split, not the entire model test set."
    )
    
    # Show model comparisons and feature importance.
    render_model_comparison()