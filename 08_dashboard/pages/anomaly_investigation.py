
"""Investigate saved statistical and Isolation Forest anomaly results."""

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import load_csv


COMPARISON = "anomaly_detection/07_detector_comparison.csv"
AGREEMENT = "anomaly_detection/08_detector_agreement_summary.csv"
QUEUE = "anomaly_detection/09_anomaly_investigation_queue.csv"

REQUIRED_COLUMNS = {
    "Store",
    "Date",
    "Weekly_Store_Sales",
    "Prediction",
    "Residual",
    "Z_Score",
    "Statistical_Anomaly",
    "Isolation_Anomaly",
    "Agreement_Category",
    "Any_Anomaly",
    "Review_Priority",
}


def prepare_anomalies(data):
    """Validate and prepare the saved detector results."""

    missing = REQUIRED_COLUMNS - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing anomaly columns: {sorted(missing)}"
        )

    if data.empty:
        raise ValueError("The anomaly report is empty.")

    result = data.copy()

    result["Date"] = pd.to_datetime(
        result["Date"], errors="coerce"
    )

    if result["Date"].isna().any():
        raise ValueError("Invalid dates in anomaly report.")

    # Read CSV boolean values consistently.
    for column in [
        "Statistical_Anomaly",
        "Isolation_Anomaly",
        "Any_Anomaly",
    ]:
        values = result[column].astype(str).str.lower()

        if not values.isin(["true", "false"]).all():
            raise ValueError(
                f"Unexpected boolean values in {column}."
            )

        result[column] = values.eq("true")

    return result


def render():
    """Display anomaly counts, detector agreement and review queue."""

    st.title("Anomaly Investigation")

    st.caption(
        "Retrospective anomaly analysis using saved Phase 06 "
        "results. Detector flags are not confirmed incidents."
    )

    try:
        comparison = prepare_anomalies(
            load_csv(COMPARISON)
        )

        queue = prepare_anomalies(
            load_csv(QUEUE)
        )

        agreement = load_csv(AGREEMENT).copy()

    except (
        FileNotFoundError,
        pd.errors.EmptyDataError,
        ValueError,
    ) as error:
        st.error(f"Unable to load anomaly reports: {error}")
        return

    if agreement.empty:
        st.warning("Detector agreement summary is empty.")
        return

    required_agreement = {
        "Test_Observations",
        "Statistical_Anomalies",
        "Isolation_Anomalies",
        "Both",
        "Statistical_Only",
        "Isolation_Only",
        "Neither",
        "Union",
        "Jaccard_Similarity",
    }

    if not required_agreement.issubset(agreement.columns):
        st.error("Agreement report has missing columns.")
        return

    summary = agreement.iloc[0]

    # Display results from the saved detector agreement report.
    a, b, c, d = st.columns(4)

    a.metric(
        "Test Observations",
        int(summary["Test_Observations"]),
    )

    b.metric(
        "Statistical Flags",
        int(summary["Statistical_Anomalies"]),
    )

    c.metric(
        "Isolation Forest Flags",
        int(summary["Isolation_Anomalies"]),
    )

    d.metric(
        "Unique Flags",
        int(summary["Union"]),
    )

    st.divider()
    st.subheader("Detector Agreement")

    categories = pd.DataFrame({
        "Category": [
            "Both",
            "Statistical Only",
            "Isolation Only",
        ],
        "Observations": [
            int(summary["Both"]),
            int(summary["Statistical_Only"]),
            int(summary["Isolation_Only"]),
        ],
    })

    fig = px.bar(
        categories,
        x="Category",
        y="Observations",
        color="Category",
        text="Observations",
        title="Distribution of Flagged Observations",
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Detector agreement",
        yaxis_title="Observations",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="anomaly_detector_agreement_chart",
    )

    st.metric(
        "Jaccard Similarity",
        f"{summary['Jaccard_Similarity'] * 100:.2f}%",
    )

    st.caption(
        "Jaccard similarity measures overlap among flagged "
        "observations. High agreement on normal records does "
        "not necessarily mean the detectors identify the "
        "same anomalies."
    )

    st.divider()
    st.subheader("Investigation Filters")

    detector = st.selectbox(
        "Detection method",
        [
            "Any Detector",
            "Both Detectors",
            "Statistical Only",
            "Isolation Forest Only",
        ],
        key="anomaly_detector_filter",
    )

    filtered = queue.copy()

    if detector == "Both Detectors":
        filtered = filtered[
            filtered["Statistical_Anomaly"]
            & filtered["Isolation_Anomaly"]
        ]

    elif detector == "Statistical Only":
        filtered = filtered[
            filtered["Statistical_Anomaly"]
            & ~filtered["Isolation_Anomaly"]
        ]

    elif detector == "Isolation Forest Only":
        filtered = filtered[
            filtered["Isolation_Anomaly"]
            & ~filtered["Statistical_Anomaly"]
        ]

    available_priorities = sorted(
        filtered["Review_Priority"].dropna().unique()
    )

    selected_priorities = st.multiselect(
        "Investigation priority",
        options=available_priorities,
        default=available_priorities,
        key="anomaly_priority_filter",
    )

    filtered = filtered[
        filtered["Review_Priority"].isin(
            selected_priorities
        )
    ]

    available_stores = sorted(
        filtered["Store"].dropna().unique().tolist()
    )

    selected_stores = st.multiselect(
        "Stores",
        options=available_stores,
        default=available_stores,
        key="anomaly_store_filter",
    )

    filtered = filtered[
        filtered["Store"].isin(selected_stores)
    ].copy()

    st.metric(
        "Observations Matching Filters",
        len(filtered),
    )

    if filtered.empty:
        st.info("No observations match the selected filters.")
        return

    st.divider()
    st.subheader("Investigation Queue")

    columns = [
        "Store",
        "Date",
        "Weekly_Store_Sales",
        "Prediction",
        "Residual",
        "Z_Score",
        "Statistical_Anomaly",
        "Isolation_Anomaly",
        "Agreement_Category",
        "Review_Priority",
    ]

    st.dataframe(
        filtered[columns].sort_values(
            "Date", ascending=False
        ),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("Flagged Sales Observations")

    fig = px.scatter(
        filtered,
        x="Date",
        y="Weekly_Store_Sales",
        color="Agreement_Category",
        hover_data=[
            "Store",
            "Prediction",
            "Residual",
            "Z_Score",
            "Review_Priority",
        ],
        title="Historical Sales at Flagged Store-Weeks",
    )

    fig.update_layout(
        height=430,
        xaxis_title="Week",
        yaxis_title="Weekly store sales (dataset units)",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="anomaly_investigation_scatter",
    )

    st.divider()
    st.subheader("Research Interpretation")

    st.markdown(
        """
        **Statistical detector:** Flags unusually large
        prediction residuals using validation-calibrated
        Z-scores.

        **Isolation Forest:** Identifies unusual combinations
        of forecast errors and sales-related features.

        **Both detectors:** Observations flagged by both
        methods can receive higher investigation priority.

        **Important limitation:** These are retrospective
        detection results. Without verified anomaly labels,
        we cannot claim detection precision or recall.
        """
    )

    with st.expander("View complete detector comparison"):
        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
        )

    with st.expander("View detector agreement summary"):
        st.dataframe(
            agreement,
            width="stretch",
            hide_index=True,
        )
