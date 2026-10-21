"""Introduce the implemented project inside the shared dashboard."""

import streamlit as st


def render():
    """Describe the problem, implemented workflow, features and limitations."""
    st.caption("RETAIL INTELLIGENCE / PROJECT OVERVIEW")
    st.title("Understand Retail Sales. Make Better Decisions.")
    st.write(
        "RetailIQ AI turns historical retail sales data into business insights, "
        "evaluated sales forecasts and evidence-backed answers."
    )
    st.divider()
    st.subheader("What problem are we solving?")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**The Problem**")
            st.write(
                "Raw sales records can be difficult to compare and interpret. "
                "Businesses need understandable trends, evaluated predictions "
                "and a way to investigate unusual periods."
            )
    with right:
        with st.container(border=True):
            st.markdown("**Our Solution**")
            st.write(
                "One dashboard for dataset exploration, business analytics, "
                "forecasting, reliability, anomalies, model explanations "
                "and grounded business assistance."
            )
    st.subheader("Which dataset are we using?")
    st.info(
        "Historical Walmart sales: 421,570 department-store-week records, "
        "45 stores and 143 weeks from February 2010 to October 2012. "
        "Open Dataset Overview to explore the original source files."
    )
    st.subheader("How does the project work?")
    st.code(
        "Raw CSV files -> Data quality checks -> Validated Parquet dataset\n"
        "                                    |\n"
        "                  Analytics + Forecasting + Anomalies + SHAP\n"
        "                                    |\n"
        "                        Saved reports and model artifacts\n"
        "                                    |\n"
        "                      Dashboard + Evidence-backed assistant",
        language="text",
    )
    st.subheader("Implemented features")
    features = [
        ("Data Engineering", "Validate joins and preserve original sales records."),
        ("Business Analytics", "Explore trends, stores, departments and holidays."),
        ("Sales Forecasting", "Compare baseline, Random Forest and XGBoost."),
        ("Forecast Reliability", "Inspect store errors, relative errors and backtests."),
        ("Anomaly Detection", "Investigate statistical and Isolation Forest flags."),
        ("Explainable AI", "Explore global and local SHAP contributions."),
        ("Business Assistant", "Use verified calculations and optional Gemini explanations."),
    ]
    for index in range(0, len(features), 2):
        for column, (title, description) in zip(st.columns(2), features[index : index + 2]):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(description)
    st.divider()
    st.caption(
        "This project analyzes historical data. Forecasts are one week ahead "
        "with previous actual sales available. Anomaly flags require human "
        "investigation, and SHAP describes model behavior rather than causation."
    )
