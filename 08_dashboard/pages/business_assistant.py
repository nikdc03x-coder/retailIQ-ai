
"""Modern AI Business Assistant for RetailIQ AI."""

import logging
from numbers import Real

import pandas as pd
import streamlit as st

from gemini_service import answer_question


logger = logging.getLogger(__name__)


# These questions are supported by the existing context engine.
EXAMPLES = [
    (
        "Store Performance",
        "Find the strongest-performing store",
        "Which store has the highest sales?",
        ":material/storefront:",
    ),
    (
        "Forecast Intelligence",
        "Compare forecasting model accuracy",
        "Which forecasting model has the best MAE?",
        ":material/trending_up:",
    ),
    (
        "Anomaly Detection",
        "Explore unusual sales observations",
        "How many anomalies were flagged?",
        ":material/troubleshoot:",
    ),
    (
        "Explainable AI",
        "Understand SHAP feature importance",
        "What are the top SHAP features?",
        ":material/psychology:",
    ),
]


# Only use metrics that exist in the saved project reports.
METRIC_FIELDS = [
    "Union",
    "Statistical_Anomalies",
    "Isolation_Anomalies",
    "Both",
    "Test_Observations",
    "Total_Sales",
    "Test_MAE",
    "MAE",
    "WAPE",
]


def initialize_chat():
    """Keep conversation messages during the current session."""

    if "retailiq_chat_messages" not in st.session_state:
        st.session_state["retailiq_chat_messages"] = []


def apply_styles():
    """Use native Streamlit widgets with modern typography and spacing."""

    st.html(
        """
        <style>
        /* Modern typography and comfortable line spacing */
        [data-testid="stMain"] {
            font-family:
                "Segoe UI Variable",
                "Segoe UI",
                Inter,
                system-ui,
                sans-serif;
        }

        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3 {
            letter-spacing: -0.7px;
            font-weight: 700;
        }

        [data-testid="stMain"] h1 {
            font-size: clamp(30px, 3vw, 43px);
        }

        /* Modern bordered cards */
        [data-testid="stMain"]
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px;
        }

        /* Compact interactive buttons */
        [data-testid="stMain"] button {
            border-radius: 11px;
            min-height: 40px;
            transition: background 0.15s ease;
        }

        /* Custom question area */
        [data-testid="stMain"] [data-testid="stForm"] {
            border-radius: 18px;
            border: 1px solid rgba(139, 127, 245, 0.23);
            padding: 18px;
            background: rgba(115, 104, 225, 0.035);
        }

        /* Input field */
        [data-testid="stMain"] textarea {
            border-radius: 12px;
            line-height: 1.6;
        }

        /* Gemini response bubbles */
        [data-testid="stMain"] [data-testid="stChatMessage"] {
            border-radius: 16px;
        }

        /* KPI cards */
        [data-testid="stMain"] [data-testid="stMetric"] {
            background: rgba(113, 104, 210, 0.065);
            border: 1px solid rgba(139, 127, 245, 0.16);
            padding: 14px;
            border-radius: 13px;
        }

        /* Tabs */
        [data-testid="stMain"] [data-testid="stTabs"] button {
            font-weight: 600;
        }
        </style>
        """
    )


def show_header():
    """Display a clean product-style header."""

    left, right = st.columns(
        [5, 1],
        vertical_alignment="center",
    )

    with left:
        st.caption("RETAILIQ AI  /  INTELLIGENCE WORKSPACE")

        st.title("✦ Business Intelligence Assistant")

        st.write(
            "Explore sales performance, forecasting, anomalies "
            "and model explainability through verified project data."
        )

    with right:
        if st.button(
            "New chat",
            icon=":material/add_comment:",
            width="stretch",
            key="ri_new_chat",
        ):
            st.session_state["retailiq_chat_messages"] = []
            st.rerun()

    st.caption(
        "Gemini-powered explanations  ·  Historical Walmart data  ·  "
        "Verified project evidence"
    )

    st.divider()


def show_suggestions():
    """Show large clickable cards for common business questions."""

    st.subheader("Explore your data")

    st.caption(
        "Choose an analysis below, or enter your own question."
    )

    selected = None

    for start in range(0, len(EXAMPLES), 2):
        columns = st.columns(2, gap="medium")

        for offset, column in enumerate(columns):
            index = start + offset

            if index >= len(EXAMPLES):
                continue

            title, description, question, icon = EXAMPLES[index]

            with column:
                with st.container(border=True):
                    st.markdown(f"#### {title}")
                    st.caption(description)

                    if st.button(
                        "Explore insights",
                        icon=icon,
                        width="stretch",
                        key=f"ri_example_{index}",
                    ):
                        selected = question

    return selected


def show_question_input():
    """Provide an obvious input for custom user questions."""

    st.subheader("Ask your own question")

    with st.form(
        "ri_question_form",
        clear_on_submit=True,
    ):
        question = st.text_area(
            "Your business question",
            placeholder=(
                "For example: Which department has the highest sales?"
            ),
            height=95,
            max_chars=500,
        )

        left, right = st.columns(
            [4, 1],
            vertical_alignment="center",
        )

        with left:
            st.caption(
                "Answers are limited to analyses supported by "
                "the saved RetailIQ AI reports."
            )

        with right:
            submitted = st.form_submit_button(
                "Analyze",
                icon=":material/arrow_upward:",
                type="primary",
                width="stretch",
            )

    if submitted:
        if question.strip():
            return question.strip()

        st.warning("Enter a question before submitting.")

    return None


def format_metric(value):
    """Format supported numerical values for KPI cards."""

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, Real):
        number = float(value)

        if number.is_integer():
            return f"{int(number):,}"

        return f"{number:,.2f}"

    return str(value)


def show_metrics(evidence):
    """Display key metrics directly from the original evidence."""

    if not evidence:
        return

    records = evidence[0].get("records", [])

    if not records or not isinstance(records[0], dict):
        return

    record = records[0]
    metrics = []

    for field in METRIC_FIELDS:
        value = record.get(field)

        if isinstance(value, bool) or not isinstance(value, Real):
            continue

        metrics.append(
            (field.replace("_", " "), format_metric(value))
        )

        if len(metrics) == 4:
            break

    if not metrics:
        return

    st.markdown("#### Key metrics")

    columns = st.columns(len(metrics))

    for column, (label, value) in zip(columns, metrics):
        with column:
            st.metric(label, value)


def prepare_chart(evidence):
    """Find a supported comparison chart from the saved report."""

    for item in evidence:
        source = str(item.get("source", "")).lower()
        records = item.get("records", [])

        if not records:
            continue

        data = pd.DataFrame(records)

        # Anomaly comparison.
        if "detector_agreement_summary" in source:
            required = [
                "Statistical_Anomalies",
                "Isolation_Anomalies",
                "Both",
            ]

            if all(column in data.columns for column in required):
                record = data.iloc[0]

                chart = pd.DataFrame({
                    "Category": [
                        "Statistical detector",
                        "Isolation Forest",
                        "Both detectors",
                    ],
                    "Value": [
                        record["Statistical_Anomalies"],
                        record["Isolation_Anomalies"],
                        record["Both"],
                    ],
                })

                return (
                    "Detector comparison",
                    chart,
                    "Counts overlap. The 'Both detectors' group "
                    "is included in the individual detector totals.",
                )

        # Forecast comparison in long-format reports.
        if "aligned_model_comparison" in source:
            model_col = next(
                (
                    column
                    for column in data.columns
                    if column.lower() in {"model", "model_name"}
                ),
                None,
            )

            mae_col = next(
                (
                    column
                    for column in data.columns
                    if column.lower() in {"mae", "test_mae"}
                ),
                None,
            )

            if model_col and mae_col:
                chart = data[[model_col, mae_col]].copy()

                chart.columns = ["Category", "Value"]
                chart["Value"] = pd.to_numeric(
                    chart["Value"],
                    errors="coerce",
                )

                chart = chart.dropna()

                if not chart.empty:
                    return (
                        "Forecast model comparison",
                        chart,
                        "Lower MAE means better forecasting accuracy.",
                    )

    return None


def show_chart(evidence, message_id):
    """Let users switch between visual and tabular comparisons."""

    result = prepare_chart(evidence)

    if result is None:
        return

    title, data, explanation = result

    st.markdown(f"#### {title}")

    choice = st.segmented_control(
        "Display",
        ["Chart", "Data table"],
        default="Chart",
        key=f"ri_chart_display_{message_id}",
    )

    if choice == "Data table":
        st.dataframe(
            data,
            width="stretch",
            hide_index=True,
        )

    else:
        st.bar_chart(
            data.set_index("Category"),
            horizontal=True,
            color="#8981f5",
        )

    st.caption(explanation)


def show_sources(evidence):
    """Display verified source references and their raw records."""

    if not evidence:
        return

    st.markdown("#### Verified evidence")

    for index, item in enumerate(evidence, start=1):
        source = item.get("source", "Unknown report")

        st.caption(f"Source {index}: {source}")

    with st.expander("View original report data"):
        for item in evidence:
            source = item.get("source", "Unknown report")

            st.markdown(f"**{source}**")

            if item.get("description"):
                st.caption(item["description"])

            records = item.get("records", [])

            if records:
                st.dataframe(
                    pd.DataFrame(records),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("No report records available.")


def show_message(message, message_id):
    """Render conversation messages and supporting visual insights."""

    role = message["role"]

    avatar = (
        ":material/person:"
        if role == "user"
        else ":material/auto_awesome:"
    )

    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])

        if role == "user":
            return

        if not message.get("supported", False):
            st.caption("Question not supported by available reports.")
            return

        if message.get("generated", False):
            st.caption("✦ Gemini-generated explanation")
        else:
            st.caption(
                "AI explanation unavailable. Verified results "
                "are available below."
            )

        evidence = message.get("evidence", [])

        if not evidence:
            return

        st.divider()

        show_metrics(evidence)

        # Interactive insights and original report records.
        overview_tab, sources_tab = st.tabs([
            "Visual insights",
            "Source records",
        ])

        with overview_tab:
            show_chart(evidence, message_id)

            if prepare_chart(evidence) is None:
                st.info(
                    "The verified source records are available "
                    "in the Source records tab."
                )

        with sources_tab:
            show_sources(evidence)


def show_history():
    """Display all messages from the current Streamlit session."""

    messages = st.session_state["retailiq_chat_messages"]

    if not messages:
        return

    st.subheader("Conversation")

    for index, message in enumerate(messages):
        show_message(message, index)


def process_question(question):
    """Retrieve the project evidence and generate a Gemini response."""

    question = question.strip()

    if not question:
        return

    st.session_state["retailiq_chat_messages"].append({
        "role": "user",
        "content": question,
    })

    with st.spinner("Analyzing verified project results..."):
        try:
            result = answer_question(question)

        except Exception:
            logger.exception("RetailIQ assistant request failed")

            result = {
                "supported": False,
                "answer": (
                    "The assistant could not process the request. "
                    "Please try again."
                ),
                "evidence": [],
                "generated": False,
            }

    st.session_state["retailiq_chat_messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "supported": result["supported"],
        "generated": result["generated"],
        "evidence": result.get("evidence", []),
    })



def render():
    """Display the Business Assistant with custom question input."""

    initialize_chat()
    apply_styles()
    show_header()

    messages = st.session_state.retailiq_chat_messages

    selected_question = None

    # Display suggested questions only when chat is empty.
    if not messages:
        selected_question = show_suggestions()

    # Display all previous messages and their interactive insights.
    show_history()

    st.divider()

    # Let users enter their own business questions at any time.
    st.markdown("### Ask Your Own Question")

    with st.form("retailiq_custom_question_form", clear_on_submit=True):
        custom_question = st.text_area(
            "What would you like to know?",
            placeholder=(
                "Example: Which store has the highest total sales? "
                "Or: How many anomalies were detected?"
            ),
            height=100,
            max_chars=500,
        )

        submitted = st.form_submit_button(
            "Ask RetailIQ AI",
            type="primary",
            icon=":material/send:",
            width="stretch",
        )

    # Both suggested questions and custom questions use
    # the same verified project data and Gemini service.
    question = None

    if submitted:
        if custom_question.strip():
            question = custom_question.strip()
        else:
            st.warning("Please enter a question.")

    elif selected_question:
        question = selected_question

    if question:
        process_question(question)
        st.rerun()

    st.caption(
        "Ask about the historical RetailIQ AI dataset, sales, "
        "forecasting, reliability, anomalies or SHAP. "
        "Only questions supported by the saved reports can be answered."
    )
