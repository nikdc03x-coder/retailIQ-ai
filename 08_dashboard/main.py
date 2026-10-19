
"""RetailIQ AI — Interactive business intelligence platform."""

import sys
from pathlib import Path

import streamlit as st


DASHBOARD = Path(__file__).resolve().parent

if str(DASHBOARD) not in sys.path:
    sys.path.insert(0, str(DASHBOARD))

from pages.overview import render as render_overview
from pages.business_analytics import render as render_business_analytics
from pages.sales_forecasting import render as render_sales_forecasting
from pages.forecast_reliability import render as render_forecast_reliability
from pages.anomaly_investigation import render as render_anomaly_investigation
from pages.explainable_ai import render as render_explainable_ai
st.set_page_config(
    page_title="RetailIQ AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Display navigation and the selected dashboard page."""

    with st.sidebar:
        st.title("RetailIQ AI")
        st.caption("Retail Intelligence Platform")
        st.divider()

        page = st.radio(
            "Explore",
           [
    "Overview",
    "Business Analytics",
    "Sales Forecasting",
    "Forecast Reliability",
    "Anomaly Investigation",
    "Explainable AI",
],
            key="main_navigation",
        )

        st.divider()
        st.caption("Research-based business decision support")
        st.caption("Historical Walmart retail dataset")

    if page == "Overview":
        render_overview()

    elif page == "Business Analytics":
        render_business_analytics()

    elif page == "Sales Forecasting":
        render_sales_forecasting()

    elif page == "Forecast Reliability":
        render_forecast_reliability()
   
    elif page == "Anomaly Investigation":
        render_anomaly_investigation()
    elif page == "Explainable AI":
        render_explainable_ai()
        
    else:
        st.title(page)
        st.info("This page is coming in a later checkpoint.")


if __name__ == "__main__":
    main()
