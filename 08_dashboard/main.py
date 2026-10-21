"""RetailIQ AI — Interactive business intelligence platform."""

import sys
from pathlib import Path

import streamlit as st


# Make dashboard modules available for imports.
DASHBOARD = Path(__file__).resolve().parent

if str(DASHBOARD) not in sys.path:
    sys.path.insert(0, str(DASHBOARD))


# Import the seven existing dashboard pages.
from pages.overview import render as render_overview
from pages.business_analytics import render as render_business_analytics
from pages.sales_forecasting import render as render_sales_forecasting
from pages.forecast_reliability import render as render_forecast_reliability
from pages.anomaly_investigation import render as render_anomaly_investigation
from pages.explainable_ai import render as render_explainable_ai
from pages.business_assistant import render as render_business_assistant


# Configure the Streamlit application.
st.set_page_config(
    page_title="RetailIQ AI | Decision Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Define sidebar navigation in one place.
# Each page reuses its existing render function.
NAVIGATION = {
    "Overview": {
        "icon": ":material/dashboard:",
        "render": render_overview,
        "key": "overview",
    },
    "Business Analytics": {
        "icon": ":material/bar_chart:",
        "render": render_business_analytics,
        "key": "business_analytics",
    },
    "Sales Forecasting": {
        "icon": ":material/trending_up:",
        "render": render_sales_forecasting,
        "key": "sales_forecasting",
    },
    "Forecast Reliability": {
        "icon": ":material/verified:",
        "render": render_forecast_reliability,
        "key": "forecast_reliability",
    },
    "Anomaly Investigation": {
        "icon": ":material/search_insights:",
        "render": render_anomaly_investigation,
        "key": "anomaly_investigation",
    },
    "Explainable AI": {
        "icon": ":material/psychology:",
        "render": render_explainable_ai,
        "key": "explainable_ai",
    },
    "Business Assistant": {
        "icon": ":material/auto_awesome:",
        "render": render_business_assistant,
        "key": "business_assistant",
    },
}


def apply_sidebar_styles():
    """Give the sidebar a modern SaaS-style appearance."""

    st.markdown(
        """
        <style>
        /* Sidebar background and separator */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #171d30 0%,
                #111827 100%
            );
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }

        /* Sidebar content spacing */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.45rem;
        }

        /* Brand section */
        .retailiq-brand {
            display: flex;
            align-items: center;
            gap: 13px;
            padding: 12px 4px 17px;
        }

        .retailiq-brand-icon {
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 13px;
            background: linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6
            );
            color: white;
            font-size: 24px;
            font-weight: 700;
        }

        .retailiq-brand-name {
            color: #f8fafc;
            font-size: 21px;
            font-weight: 750;
            letter-spacing: -0.6px;
            line-height: 1.3;
        }

        .retailiq-brand-subtitle {
            color: #94a3b8;
            font-size: 11px;
            letter-spacing: 0.4px;
        }

        /* Navigation group headings */
        .retailiq-nav-heading {
            color: #8694ab;
            font-size: 10px;
            font-weight: 750;
            letter-spacing: 1.5px;
            padding: 15px 12px 8px;
        }

        /* All sidebar navigation buttons */
        [data-testid="stSidebar"] [data-testid="stButton"] button {
            width: 100%;
            min-height: 43px;
            border-radius: 10px;
            text-align: left;
            justify-content: flex-start;
            padding: 9px 13px;
            transition:
                background 0.15s ease,
                border-color 0.15s ease;
            font-size: 13px;
            font-weight: 500;
        }

        /* Inactive page */
        [data-testid="stSidebar"] button[kind="secondary"] {
            background: transparent;
            color: #b8c4d7;
            border: 1px solid transparent;
        }

        /* Inactive hover */
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            background: rgba(148, 163, 184, 0.11);
            color: #ffffff;
            border-color: rgba(148, 163, 184, 0.10);
        }

        /* Current active page */
        [data-testid="stSidebar"] button[kind="primary"] {
            background: rgba(99, 102, 241, 0.18);
            color: #e3e3ff;
            border: 1px solid rgba(129, 140, 248, 0.28);
            border-left: 3px solid #818cf8;
            font-weight: 650;
        }

        [data-testid="stSidebar"] button[kind="primary"]:hover {
            background: rgba(99, 102, 241, 0.25);
            color: #ffffff;
        }

        /* Lower sidebar information panel */
        .retailiq-sidebar-info {
            margin-top: 20px;
            padding: 15px;
            background: rgba(148, 163, 184, 0.07);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 12px;
        }

        .retailiq-info-title {
            color: #e2e8f0;
            font-size: 12px;
            font-weight: 650;
            margin-bottom: 6px;
        }

        .retailiq-info-detail {
            color: #94a3b8;
            font-size: 11px;
            line-height: 1.8;
        }

        .retailiq-footer {
            margin-top: 15px;
            padding: 10px 4px;
            color: #64748b;
            font-size: 10px;
            text-align: center;
            letter-spacing: 0.3px;
        }

        /* Adjust sidebar width for long navigation labels */
        [data-testid="stSidebar"] {
            min-width: 285px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_active_page(page_name):
    """Remember the selected page across Streamlit reruns."""

    st.session_state["retailiq_active_page"] = page_name


def navigation_button(page_name):
    """Show one clickable navigation item with active styling."""

    config = NAVIGATION[page_name]

    active_page = st.session_state["retailiq_active_page"]
    is_active = active_page == page_name

    st.button(
        page_name,
        key=f"nav_{config['key']}",
        icon=config["icon"],
        type="primary" if is_active else "secondary",
        width="stretch",
        on_click=set_active_page,
        args=(page_name,),
    )


def show_sidebar():
    """Display branded navigation grouped by dashboard purpose."""

    with st.sidebar:

        # Project branding.
        st.markdown(
            """
            <div class="retailiq-brand">
                <div class="retailiq-brand-icon">R</div>
                <div>
                    <div class="retailiq-brand-name">RetailIQ AI</div>
                    <div class="retailiq-brand-subtitle">
                        DECISION INTELLIGENCE
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # General business overview and analytics.
        st.markdown(
            '<div class="retailiq-nav-heading">WORKSPACE</div>',
            unsafe_allow_html=True,
        )

        navigation_button("Overview")
        navigation_button("Business Analytics")

        # Forecasting, reliability, anomalies and model explanation.
        st.markdown(
            '<div class="retailiq-nav-heading">'
            'PREDICTIVE INTELLIGENCE</div>',
            unsafe_allow_html=True,
        )

        navigation_button("Sales Forecasting")
        navigation_button("Forecast Reliability")
        navigation_button("Anomaly Investigation")
        navigation_button("Explainable AI")

        # Gemini-powered project assistant.
        st.markdown(
            '<div class="retailiq-nav-heading">AI WORKSPACE</div>',
            unsafe_allow_html=True,
        )

        navigation_button("Business Assistant")

        # Dataset information.
        st.markdown(
            """
            <div class="retailiq-sidebar-info">
                <div class="retailiq-info-title">
                    Historical Retail Dataset
                </div>
                <div class="retailiq-info-detail">
                    45 stores · 2010–2012<br>
                    Walmart weekly sales data<br>
                    Research & decision support
                </div>
            </div>
            <div class="retailiq-footer">
                RetailIQ AI · Data Intelligence Platform
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    """Render the navigation and currently selected dashboard page."""

    # Keep the selected page when Streamlit refreshes widgets.
    if "retailiq_active_page" not in st.session_state:
        st.session_state["retailiq_active_page"] = "Overview"

    apply_sidebar_styles()
    show_sidebar()

    # Call the original render function for the selected page.
    active_page = st.session_state["retailiq_active_page"]

    NAVIGATION[active_page]["render"]()


if __name__ == "__main__":
    main()