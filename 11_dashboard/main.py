"""
RetailIQ AI — Main Dashboard

Purpose:
    Introduce the business problem and explain our solution.

Input:
    No dataset is required during Phase 00.

Output:
    A working Streamlit application.

Next:
    Phase 01 will connect verified dataset information.

Limitations:
    This page does not display business results yet.
"""

import streamlit as st


st.set_page_config(
    page_title="RetailIQ AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Use one simple design style across the dashboard.
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
    }

    div[data-testid="stMetric"] {
        background: #f5f7fb;
        border: 1px solid #e6eaf1;
        border-radius: 12px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("RetailIQ AI")
    st.caption("Retail Decision Intelligence")
    st.divider()

    st.markdown("### Project Navigation")
    st.page_link(
        "main.py",
        label="Project Overview",
        icon="🏠",
    )

    st.divider()
    st.info("Phase 00 — Foundation")

st.caption("RETAIL INTELLIGENCE / PROJECT OVERVIEW")

st.title("Understand Retail Sales. Make Better Decisions.")

st.write(
    "RetailIQ AI is a student-built platform that turns "
    "historical retail sales data into understandable "
    "business insights and evaluated sales forecasts."
)

st.divider()

st.subheader("01 — What problem are we solving?")

st.write(
    "Retailers collect sales data, but raw records alone "
    "do not clearly show which stores perform well, "
    "how sales change over time, or what future "
    "sales might look like."
)

left, right = st.columns(2)

with left:
    st.error("The Problem")
    st.write(
        "Sales data can be difficult to clean, compare "
        "and interpret. Forecasting and identifying "
        "unusual changes also require additional analysis."
    )

with right:
    st.success("Our Solution")
    st.write(
        "A single dashboard for data quality, business "
        "analytics, forecasting, anomaly detection, "
        "model explanations and grounded AI assistance."
    )

st.divider()

st.subheader("02 — Which dataset are we using?")

st.info(
    "Dataset selection and validation will be completed "
    "in Phase 01. We will display the verified source, "
    "columns, row count, time coverage and limitations here."
)

st.subheader("03 — How will the project work?")

st.code(
    "Raw Sales Data → Data Quality → Clean Database\n"
    "                 ↓\n"
    "       Analytics + ML Models\n"
    "                 ↓\n"
    "       Interactive Dashboard",
    language="text",
)

st.subheader("04 — What will we build?")

features = [
    ("Data Engineering", "Prepare reliable sales records."),
    ("Business Analytics", "Understand historical performance."),
    ("Sales Forecasting", "Predict sales and measure errors."),
    ("Anomaly Detection", "Highlight unusual sales patterns."),
    ("Explainable AI", "Understand model predictions."),
    ("Gemini Assistant", "Answer questions using verified data."),
]

for index in range(0, len(features), 2):
    columns = st.columns(2)

    for column, feature in zip(
        columns, features[index:index + 2]
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"**{feature[0]}**")
                st.caption(feature[1])

st.divider()

st.subheader("05 — Current Project Status")

st.metric("Development Phase", "00 — Foundation")

st.caption(
    "No dataset analysis or model training has been "
    "performed yet. Results will appear only after "
    "we run and validate the relevant experiments."
)
