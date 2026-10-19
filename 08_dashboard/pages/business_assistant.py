
"""Modern AI Business Assistant for RetailIQ AI."""

import logging
from numbers import Real

import pandas as pd
import streamlit as st

from gemini_service import answer_question


logger = logging.getLogger(__name__)


# Show common project questions as clickable cards.
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


# Suggestions appear as users type in the searchable question field.
# Users may also enter questions not included in this list.
QUESTION_SUGGESTIONS = [
    "Which store has the highest sales?",
    "Which department has the highest sales?",
    "Which forecasting model has the best MAE?",
    "What is the MAE for Store 14?",
    "How many anomalies were flagged?",
    "What are the top SHAP features?",
    "Show top 5 stores by sales.",
    "Show top 10 departments.",
    "Which are the bottom 5 stores?",
    "Compare Store 10 and Store 20.",
    "How much more did Store 20 sell than Store 10?",
    "What's the sales difference between Store 10 and Store 20?",
    "Which five stores generated the most sales?",
    "List the 10 lowest-selling departments.",
]

# Only display metrics available in saved project reports.
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
    """Keep messages during the current Streamlit session."""

    if "retailiq_chat_messages" not in st.session_state:
        st.session_state["retailiq_chat_messages"] = []


def apply_styles():
    """Apply modern styling using native Streamlit widgets."""

    st.html(
        """
        <style>
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

        [data-testid="stMain"]
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px;
        }

        [data-testid="stMain"] button {
            border-radius: 11px;
            min-height: 40px;
        }

        [data-testid="stMain"] [data-testid="stForm"] {
            border-radius: 18px;
            border: 1px solid rgba(139, 127, 245, 0.23);
            padding: 18px;
            background: rgba(115, 104, 225, 0.035);
        }

        [data-testid="stMain"]
        [data-testid="stChatMessage"] {
            border-radius: 16px;
        }

        [data-testid="stMain"] [data-testid="stMetric"] {
            background: rgba(113, 104, 210, 0.065);
            border: 1px solid rgba(139, 127, 245, 0.16);
            padding: 14px;
            border-radius: 13px;
        }

        [data-testid="stMain"] [data-testid="stTabs"] button {
            font-weight: 600;
        }
        </style>
        """
    )


def show_header():
    """Display the assistant title and new-chat button."""

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
    """Show clickable suggestions when starting a new conversation."""

    st.subheader("Explore your data")

    st.caption(
        "Choose an analysis below, or enter your own business question."
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
    """Provide autocomplete and allow completely custom questions."""

    st.subheader("Ask RetailIQ AI")

    st.caption(
        "Start typing to see suggestions, or enter your own question."
    )

    with st.form(
        "ri_question_form",
        clear_on_submit=True,
    ):
        question = st.selectbox(
            "Your business question",
            options=QUESTION_SUGGESTIONS,
            index=None,
            accept_new_options=True,
            placeholder="Type a question about your retail data...",
            key="retailiq_smart_question",
        )

        submitted = st.form_submit_button(
            "Analyze",
            icon=":material/arrow_upward:",
            type="primary",
            width="stretch",
        )

    if submitted:
        if question and question.strip():
            return question.strip()

        st.warning("Enter or select a question before submitting.")

    return None


def format_metric(value):
    """Format verified numbers for KPI cards."""

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, Real):
        number = float(value)

        if number.is_integer():
            return f"{int(number):,}"

        return f"{number:,.2f}"

    return str(value)


def is_custom_sales_result(evidence):
    """Identify results produced by approved custom sales operations."""

    if len(evidence) != 1:
        return False

    item = evidence[0]
    records = item.get("records", [])

    if not records:
        return False

    data = pd.DataFrame(records)

    has_sales = "Total_Sales" in data.columns
    has_entity = "Store" in data.columns or "Dept" in data.columns

    source = item.get("source", "")

    known_source = source in {
        "business_analytics/02_store_performance.csv",
        "business_analytics/03_department_performance.csv",
    }

    # Require a calculated ranking or comparison,
    # not just any report containing sales values.
    is_calculated = (
        "Rank" in data.columns
        or "Absolute_Sales_Difference" in data.columns
    )

    return (
        known_source
        and has_sales
        and has_entity
        and is_calculated
    )


def show_metrics(evidence):
    """Display ordinary KPI values from saved project evidence."""

    if not evidence or is_custom_sales_result(evidence):
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


def show_custom_comparison_metrics(evidence):
    """Show the sales totals and difference for two compared stores."""

    if not is_custom_sales_result(evidence):
        return False

    records = evidence[0].get("records", [])

    if len(records) != 2:
        return False

    first = records[0]
    second = records[1]

    required = {
        "Store",
        "Total_Sales",
        "Absolute_Sales_Difference",
        "Higher_Sales_Store",
    }

    if not (
        required.issubset(first)
        and required.issubset(second)
    ):
        return False

    first_store = first["Store"]
    second_store = second["Store"]

    first_sales = first["Total_Sales"]
    second_sales = second["Total_Sales"]

    difference = first["Absolute_Sales_Difference"]
    higher_store = first["Higher_Sales_Store"]

    st.markdown("#### Store comparison")

    columns = st.columns(3)

    with columns[0]:
        st.metric(
            f"Store {int(first_store)} sales",
            f"{float(first_sales):,.2f}",
        )

    with columns[1]:
        st.metric(
            f"Store {int(second_store)} sales",
            f"{float(second_sales):,.2f}",
        )

    with columns[2]:
        st.metric(
            "Absolute sales difference",
            f"{float(difference):,.2f}",
        )

    if higher_store is None:
        st.caption("Both stores recorded equal historical sales.")
    else:
        st.caption(
            f"Store {int(higher_store)} recorded higher "
            "historical total sales."
        )

    return True


def show_custom_ranking_metrics(evidence):
    """Summarize a calculated store or department ranking."""

    if not is_custom_sales_result(evidence):
        return False

    records = evidence[0].get("records", [])

    if not records or "Rank" not in records[0]:
        return False

    first = records[0]

    if "Store" in first:
        label = "Store"
        entity = first["Store"]

    elif "Dept" in first:
        label = "Department"
        entity = first["Dept"]

    else:
        return False

    st.markdown("#### Ranking summary")

    columns = st.columns(3)

    with columns[0]:
        st.metric("Entries ranked", len(records))

    with columns[1]:
        st.metric(
            "First ranked",
            f"{label} {int(entity)}",
        )

    with columns[2]:
        st.metric(
            "First ranked sales",
            f"{float(first['Total_Sales']):,.2f}",
        )

    st.caption(
        "The first entry is determined by the requested ranking "
        "direction: highest or lowest historical sales."
    )

    return True


def show_custom_sales_chart(evidence, message_id):
    """Visualize calculated store comparisons and rankings."""

    if not is_custom_sales_result(evidence):
        return False

    records = evidence[0].get("records", [])

    if not records:
        return False

    data = pd.DataFrame(records)

    # Identify whether the result concerns stores or departments.
    if "Store" in data.columns:
        entity_column = "Store"
        entity_label = "Store"

    elif "Dept" in data.columns:
        entity_column = "Dept"
        entity_label = "Department"

    else:
        return False

    # Use only the actual sales totals for the chart.
    chart = data[
        [entity_column, "Total_Sales"]
    ].copy()

    chart["Total_Sales"] = pd.to_numeric(
        chart["Total_Sales"],
        errors="coerce",
    )

    chart = chart.dropna(
        subset=[entity_column, "Total_Sales"]
    )

    if chart.empty:
        return False

    chart["Category"] = chart[entity_column].apply(
        lambda value: f"{entity_label} {int(value)}"
    )

    display = chart[
        ["Category", "Total_Sales"]
    ].copy()

    st.markdown("#### Historical sales comparison")

    # Each assistant response gets its own Streamlit widget key.
    choice = st.segmented_control(
        "Display",
        ["Chart", "Data table"],
        default="Chart",
        key=f"ri_custom_sales_view_{message_id}",
    )

    if choice == "Data table":
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
        )

    else:
        st.bar_chart(
            display.set_index("Category"),
            horizontal=True,
            color="#8981f5",
        )

    st.caption(
        "Sales totals are calculated or retrieved from saved "
        "historical RetailIQ AI reports. They are not live sales."
    )

    return True


def prepare_chart(evidence):
    """Prepare existing anomaly and forecasting comparisons."""

    for item in evidence:
        source = str(item.get("source", "")).lower()
        records = item.get("records", [])

        if not records:
            continue

        data = pd.DataFrame(records)

        # Anomaly detector comparison.
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
                    "Counts overlap. 'Both detectors' is included "
                    "in the individual detector totals.",
                )

        # Forecast model comparison, when saved in long format.
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
                chart = data[
                    [model_col, mae_col]
                ].copy()

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
                        "Lower MAE indicates better forecast accuracy.",
                    )

    return None


def show_chart(evidence, message_id):
    """Show existing forecasting or anomaly charts."""

    result = prepare_chart(evidence)

    if result is None:
        return False

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

    return True


def show_sources(evidence):
    """Display filenames and original verified report records."""

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
    """Render a conversation message and its verified visualizations."""

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
            st.caption(
                "Question not supported by the available reports."
            )
            return

        if message.get("generated", False):
            st.caption("✦ Gemini-generated explanation")
        else:
            st.caption(
                "AI explanation unavailable. "
                "Verified results are available below."
            )

        evidence = message.get("evidence", [])

        if not evidence:
            return

        st.divider()

        # Display the most relevant KPIs automatically.
        if is_custom_sales_result(evidence):
            comparison_shown = show_custom_comparison_metrics(
                evidence
            )

            if not comparison_shown:
                show_custom_ranking_metrics(evidence)

        else:
            show_metrics(evidence)

        overview_tab, sources_tab = st.tabs([
            "Visual insights",
            "Source records",
        ])

        with overview_tab:
            # New custom analytics charts take priority.
            chart_shown = show_custom_sales_chart(
                evidence,
                message_id,
            )

            # Otherwise preserve existing model and anomaly charts.
            if not chart_shown:
                chart_shown = show_chart(
                    evidence,
                    message_id,
                )

            if not chart_shown:
                st.info(
                    "No dedicated chart is available for this "
                    "report. Open Source records to explore "
                    "the verified values."
                )

        with sources_tab:
            show_sources(evidence)


def show_history():
    """Display messages from the current Streamlit session."""

    messages = st.session_state["retailiq_chat_messages"]

    if not messages:
        return

    st.subheader("Conversation")

    for index, message in enumerate(messages):
        show_message(message, index)


def process_question(question):
    """Send the question to the existing verified Gemini service."""

    question = question.strip()

    if not question:
        return

    # Preserve the submitted question in the conversation.
    st.session_state["retailiq_chat_messages"].append({
        "role": "user",
        "content": question,
    })

    # Generate only on submission, never during ordinary reruns.
    with st.spinner("Analyzing verified project results..."):
        try:
            result = answer_question(question)

        except Exception:
            logger.exception(
                "RetailIQ assistant request failed"
            )

            result = {
                "supported": False,
                "answer": (
                    "The assistant could not process the request. "
                    "Please try again."
                ),
                "evidence": [],
                "generated": False,
            }

    # Save the explanation and retrieved evidence together.
    st.session_state["retailiq_chat_messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "supported": result["supported"],
        "generated": result["generated"],
        "evidence": result.get("evidence", []),
    })


def render():
    """Display the complete interactive RetailIQ AI assistant."""

    initialize_chat()
    apply_styles()
    show_header()

    messages = st.session_state["retailiq_chat_messages"]

    # Suggested cards appear before the first conversation.
    selected_question = None

    if not messages:
        selected_question = show_suggestions()

    # Keep the custom question composer visible above chat history.
    custom_question = show_question_input()

    # Display previous questions and their visual insights.
    show_history()

    # Both suggestions and custom input use the same backend.
    question = custom_question or selected_question

    if question:
        process_question(question)
        st.rerun()

    st.caption(
        "RetailIQ AI analyzes historical project reports, "
        "not live sales data. Only supported analyses are answered. "
        "Gemini explanations should be checked against "
        "the verified source records."
    )
