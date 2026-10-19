
"""Explain existing XGBoost sales predictions using saved SHAP results."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_csv


GLOBAL_REPORT = "explainable_ai/01_global_shap_importance.csv"
LOCAL_REPORT = "explainable_ai/02_local_shap_contributions.csv"

FEATURES = [
    "Store",
    "IsHoliday",
    "Month",
    "WeekOfYear",
    "Year",
    "Lag_1",
    "Lag_2",
    "Lag_4",
    "Rolling_Mean_4",
    "Rolling_Std_4",
]

FEATURE_MEANINGS = {
    "Store": "Store identifier",
    "IsHoliday": "Holiday indicator for the selected week",
    "Month": "Month of the selected week",
    "WeekOfYear": "Week number in the year",
    "Year": "Year of the selected week",
    "Lag_1": "Sales from the previous week",
    "Lag_2": "Sales from two weeks earlier",
    "Lag_4": "Sales from four weeks earlier",
    "Rolling_Mean_4": "Average sales over the previous four weeks",
    "Rolling_Std_4": "Variation in sales over the previous four weeks",
}


def load_explanations():
    """Load saved SHAP reports and check their required columns."""

    global_data = load_csv(GLOBAL_REPORT).copy()
    local_data = load_csv(LOCAL_REPORT).copy()

    required_global = {"Feature", "Mean_Absolute_SHAP"}

    required_local = {
        "Store",
        "Date",
        "Weekly_Store_Sales",
        "Prediction",
        "SHAP_Base_Value",
        "SHAP_Reconstructed_Prediction",
        *[f"SHAP_{feature}" for feature in FEATURES],
    }

    if not required_global.issubset(global_data.columns):
        raise ValueError("Global SHAP report has missing columns.")

    if not required_local.issubset(local_data.columns):
        raise ValueError("Local SHAP report has missing columns.")

    if global_data.empty or local_data.empty:
        raise ValueError("One or more SHAP reports are empty.")

    if set(global_data["Feature"]) != set(FEATURES):
        raise ValueError("Global report has unexpected features.")

    local_data["Date"] = pd.to_datetime(
        local_data["Date"], errors="coerce"
    )

    if local_data["Date"].isna().any():
        raise ValueError("Local SHAP report contains invalid dates.")

    if local_data.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate store-week explanations found.")

    numeric_columns = [
        "Weekly_Store_Sales",
        "Prediction",
        "SHAP_Base_Value",
        "SHAP_Reconstructed_Prediction",
        *[f"SHAP_{feature}" for feature in FEATURES],
    ]

    if not np.isfinite(local_data[numeric_columns].to_numpy()).all():
        raise ValueError("Local SHAP report contains invalid numbers.")

    return global_data, local_data


def render_global_importance(global_data):
    """Show average SHAP influence across test observations."""

    st.subheader("Global Feature Importance")

    st.write(
        "This chart shows which features had the largest average "
        "influence on the model's test predictions."
    )

    data = global_data.sort_values("Mean_Absolute_SHAP")

    fig = px.bar(
        data,
        x="Mean_Absolute_SHAP",
        y="Feature",
        orientation="h",
        title="Average Absolute SHAP Contribution",
        labels={
            "Mean_Absolute_SHAP": "Average absolute contribution (sales units)",
            "Feature": "Model feature",
        },
    )

    fig.update_layout(height=480)

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="shap_global_importance",
    )

    top_feature = global_data.sort_values(
        "Mean_Absolute_SHAP", ascending=False
    ).iloc[0]

    st.info(
        f"The feature with the highest average SHAP contribution is "
        f"{top_feature['Feature']}."
    )

    st.caption(
        "A high SHAP contribution means the model relies strongly "
        "on that feature. It does not prove a cause-and-effect relationship."
    )


def render_local_explanation(local_data):
    """Explain one store-week prediction using its saved contributions."""

    st.divider()
    st.subheader("Explain an Individual Prediction")

    st.write(
        "Select a store and week to understand how the model "
        "calculated its sales prediction."
    )

    stores = sorted(local_data["Store"].unique().tolist())

    selected_store = st.selectbox(
        "Select store",
        stores,
        key="shap_selected_store",
    )

    store_data = local_data[
        local_data["Store"] == selected_store
    ].sort_values("Date")

    available_dates = store_data["Date"].dt.strftime(
        "%Y-%m-%d"
    ).tolist()

    selected_date = st.selectbox(
        "Select week",
        available_dates,
        index=len(available_dates) - 1,
        key="shap_selected_week",
    )

    selected = store_data[
        store_data["Date"].dt.strftime("%Y-%m-%d")
        == selected_date
    ].iloc[0]

    actual = float(selected["Weekly_Store_Sales"])
    prediction = float(selected["Prediction"])
    error = actual - prediction

    a, b, c = st.columns(3)

    a.metric("Actual Sales", f"{actual:,.2f}")
    b.metric("Predicted Sales", f"{prediction:,.2f}")
    c.metric("Forecast Error", f"{error:,.2f}")

    st.caption(
        "Forecast error = actual sales − predicted sales. "
        "A positive value means the model underestimated sales."
    )

    # Read the contributions for the selected prediction.
    contributions = [
        float(selected[f"SHAP_{feature}"])
        for feature in FEATURES
    ]

    baseline = float(selected["SHAP_Base_Value"])
    reconstructed = baseline + sum(contributions)

    # Explain the prediction using a waterfall chart.
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(FEATURES)
                    + ["total"],
            x=["Baseline"] + FEATURES + ["Prediction"],
            y=[baseline] + contributions + [0],
            textposition="outside",
            connector={"line": {"color": "gray"}},
        )
    )

    fig.update_layout(
        title=f"How XGBoost Predicted Store {selected_store} Sales",
        xaxis_title="Feature contributions",
        yaxis_title="Predicted sales (dataset units)",
        height=560,
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="shap_local_waterfall",
    )

    if not np.isclose(
        reconstructed,
        prediction,
        rtol=1e-5,
        atol=0.1,
    ):
        st.error("SHAP reconstruction does not match this prediction.")
        return

    st.success(
        "Verified: The SHAP contributions reconstruct "
        "the saved model prediction."
    )

    st.caption(
        "Positive contributions push the prediction above the "
        "SHAP baseline; negative contributions push it below. "
        "The baseline is the model's SHAP reference prediction, "
        "not the store's historical sales average."
    )

    contribution_table = pd.DataFrame({
        "Feature": FEATURES,
        "Contribution": contributions,
        "Meaning": [
            FEATURE_MEANINGS[feature] for feature in FEATURES
        ],
    })

    contribution_table["Absolute_Contribution"] = (
        contribution_table["Contribution"].abs()
    )

    contribution_table = contribution_table.sort_values(
        "Absolute_Contribution",
        ascending=False,
    )

    with st.expander("View feature contributions"):
        st.dataframe(
            contribution_table[
                ["Feature", "Contribution", "Meaning"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "SHAP explains how the trained model produced this prediction. "
        "It does not explain every reason for the difference between "
        "predicted and actual sales."
    )


def render():
    """Display the Explainable AI dashboard page."""

    st.title("Explainable AI")

    st.caption(
        "Understand the existing XGBoost forecasting model "
        "using saved SHAP explanations from the test dataset."
    )

    try:
        global_data, local_data = load_explanations()

    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as error:
        st.error(f"Unable to load SHAP reports: {error}")
        st.info(
            "Run 09_explainable_ai/01_generate_shap_explanations.py "
            "to generate the required reports."
        )
        return

    a, b, c = st.columns(3)

    a.metric("Explained Predictions", len(local_data))
    b.metric("Stores", local_data["Store"].nunique())
    c.metric("Model Features", len(FEATURES))

    render_global_importance(global_data)
    render_local_explanation(local_data)
