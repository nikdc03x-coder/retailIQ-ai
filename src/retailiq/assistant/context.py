"""Retrieve verified RetailIQ AI results for the business assistant."""

import re

from retailiq.reports import load_csv

# Only these question types are supported in the first batch.
SUPPORTED_TOPICS = {
    "store_sales",
    "model_comparison",
    "store_reliability",
}


def validate_question(question):
    """Reject unsupported requests before retrieving project data."""

    if not isinstance(question, str):
        return False, "Please enter a RetailIQ AI question."
    text = question.lower().strip()

    if not text:
        return False, "Please enter a RetailIQ AI question."

    # Reject known unrelated requests and prompt override attempts.
    blocked_patterns = [
        r"\bfootball\b",
        r"\bmatch yesterday\b",
        r"\bwrite (?:me )?a poem\b",
        r"\btell me a joke\b",
        r"\bignore (?:your|previous|all) instructions\b",
        r"\bhomework\b",
        r"\bconfirmed fraud\b",
        r"\bfraudulent\b",
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, text):
            return False, (
                "I can only answer supported RetailIQ AI questions using verified project results."
            )

    # Detect requested years, including years outside the dataset.
    years = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", text)]

    if any(year < 2010 or year > 2012 for year in years):
        return False, (
            "The RetailIQ AI dataset covers February 2010 "
            "to October 2012. The requested year is unavailable."
        )

    # Current ranking functions use the full historical period.
    # Do not return those rankings as if they were year-specific.
    if years:
        return False, (
            "Year-specific questions are not supported by this "
            "retrieval function yet. The available rankings "
            "cover the full historical dataset."
        )

    # Reject unsupported relative time requests.
    if re.search(
        r"\b(today|yesterday|tomorrow|last month|this month|"
        r"last year|this year|right now|currently)\b",
        text,
    ):
        return False, (
            "This question requires a time period that the "
            "current retrieval function does not support."
        )

    return True, ""


def identify_topic(question):
    """Recognize supported RetailIQ AI questions."""

    text = question.lower().strip()

    if not text:
        return None

    # Only accept questions that match a known analysis topic.
    if any(word in text for word in ["shap", "feature contribution", "feature importance"]) and any(
        word in text for word in ["top", "important", "influence", "feature", "contribution"]
    ):
        return "shap_importance"

    if any(
        word in text for word in ["anomaly", "anomalies", "detector", "isolation forest"]
    ) and any(word in text for word in ["how many", "count", "flag", "agreement"]):
        return "anomaly_summary"

    if (
        "department" in text
        and "sales" in text
        and any(word in text for word in ["highest", "lowest", "top", "most", "least"])
    ):
        return "department_sales"

    if any(word in text for word in ["model", "xgboost", "random forest", "baseline"]) and any(
        word in text for word in ["best", "compare", "performance", "mae", "rmse"]
    ):
        return "model_comparison"

    if re.search(r"\bstore\s+\d+\b", text) and any(
        word in text for word in ["error", "reliability", "mae", "wape"]
    ):
        return "store_reliability"

    if (
        "store" in text
        and "sales" in text
        and any(word in text for word in ["highest", "lowest", "top", "most", "least"])
    ):
        return "store_sales"

    return None


def retrieve_evidence(question):
    """Validate a question and retrieve verified project evidence."""

    # Reject unsupported requests before loading any reports.
    valid, reason = validate_question(question)

    if not valid:
        return {
            "supported": False,
            "message": reason,
            "evidence": [],
        }

    topic = identify_topic(question)

    if topic is None:
        return {
            "supported": False,
            "message": (
                "I can answer supported RetailIQ AI questions about "
                "sales, forecasting, reliability, anomalies and SHAP."
            ),
            "evidence": [],
        }

    if topic == "model_comparison":
        return get_model_comparison()

    if topic == "store_sales":
        return get_store_sales(question)

    if topic == "store_reliability":
        return get_store_reliability(question)

    if topic == "department_sales":
        return get_department_sales(question)

    if topic == "anomaly_summary":
        return get_anomaly_summary()

    if topic == "shap_importance":
        return get_shap_importance()

    return {
        "supported": False,
        "message": "This question is not supported.",
        "evidence": [],
    }


def get_model_comparison():
    """Retrieve the held-out test results of the forecasting models."""

    source = "sales_forecasting/12_aligned_model_comparison.csv"
    data = load_csv(source).copy()

    required = {"Model", "Split", "MAE", "RMSE", "Observations"}

    if not required.issubset(data.columns):
        raise ValueError("Model comparison report has missing columns.")

    test = data[data["Split"] == "Test"].copy()

    if test.empty:
        raise ValueError("No test model comparison results found.")

    test = test.sort_values("MAE")

    return {
        "supported": True,
        "topic": "model_comparison",
        "message": "Verified forecasting model comparison.",
        "evidence": [
            {
                "source": source,
                "description": "Held-out test model evaluation",
                "records": test.to_dict(orient="records"),
            }
        ],
    }


def get_store_sales(question):
    """Retrieve store rankings using historical total sales."""

    source = "business_analytics/02_store_performance.csv"
    data = load_csv(source).copy()

    required = {"Store", "Total_Sales", "Average_Weekly_Sales"}

    if not required.issubset(data.columns):
        raise ValueError("Store performance report has missing columns.")

    if data.empty:
        raise ValueError("Store performance report is empty.")

    # For this first batch, rankings use total historical sales.
    lowest = any(word in question.lower() for word in ["lowest", "least"])

    ranked = data.sort_values(
        "Total_Sales",
        ascending=lowest,
    ).head(5)

    return {
        "supported": True,
        "topic": "store_sales",
        "message": (
            "Stores ranked by lowest historical total sales."
            if lowest
            else "Stores ranked by highest historical total sales."
        ),
        "evidence": [
            {
                "source": source,
                "description": "Historical store sales ranking",
                "records": ranked[["Store", "Total_Sales", "Average_Weekly_Sales"]].to_dict(
                    orient="records"
                ),
            }
        ],
    }


def get_store_reliability(question):
    """Retrieve saved forecast errors for the requested store."""

    match = re.search(r"\bstore\s+(\d+)\b", question.lower())

    if not match:
        return {
            "supported": False,
            "message": "Please mention a store number, such as Store 14.",
            "evidence": [],
        }

    store_number = int(match.group(1))
    source = "forecast_reliability/02_store_error_summary.csv"

    data = load_csv(source).copy()

    required = {"Store", "MAE"}

    if not required.issubset(data.columns):
        raise ValueError("Store reliability report has missing columns.")

    selected = data[data["Store"] == store_number]

    if selected.empty:
        return {
            "supported": False,
            "message": f"No saved reliability results for Store {store_number}.",
            "evidence": [],
        }

    # Keep the original saved metric columns as evidence.
    return {
        "supported": True,
        "topic": "store_reliability",
        "message": f"Verified forecast reliability for Store {store_number}.",
        "evidence": [
            {
                "source": source,
                "description": f"Store {store_number} test error metrics",
                "records": selected.to_dict(orient="records"),
            }
        ],
    }


def get_department_sales(question):
    """Find departments with the highest or lowest historical sales."""

    source = "business_analytics/03_department_performance.csv"
    data = load_csv(source).copy()

    required = {"Dept", "Total_Sales", "Stores"}

    if not required.issubset(data.columns) or data.empty:
        raise ValueError("Department performance report is invalid.")

    lowest = any(word in question.lower() for word in ["lowest", "least"])

    results = data.sort_values(
        "Total_Sales",
        ascending=lowest,
    ).head(5)

    return {
        "supported": True,
        "topic": "department_sales",
        "message": "Historical department sales ranking.",
        "evidence": [
            {
                "source": source,
                "description": ("Five departments ranked by total historical sales"),
                "records": results[["Dept", "Total_Sales", "Stores"]].to_dict(orient="records"),
            }
        ],
    }


def get_anomaly_summary():
    """Retrieve verified detector agreement counts."""

    source = "anomaly_detection/08_detector_agreement_summary.csv"
    data = load_csv(source).copy()

    required = {
        "Test_Observations",
        "Statistical_Anomalies",
        "Isolation_Anomalies",
        "Both",
        "Statistical_Only",
        "Isolation_Only",
        "Union",
    }

    if not required.issubset(data.columns) or len(data) != 1:
        raise ValueError("Anomaly agreement report is invalid.")

    record = data.iloc[0]

    # Detector counts must describe the same test observations.
    if record["Both"] + record["Statistical_Only"] + record["Isolation_Only"] != record["Union"]:
        raise ValueError("Anomaly detector counts do not agree.")

    return {
        "supported": True,
        "topic": "anomaly_summary",
        "message": (
            "Saved anomaly detection results. These flags are not confirmed business incidents."
        ),
        "evidence": [
            {
                "source": source,
                "description": "Statistical and Isolation Forest agreement",
                "records": data.to_dict(orient="records"),
            }
        ],
    }


def get_shap_importance():
    """Retrieve the strongest model feature contributions."""

    source = "explainable_ai/01_global_shap_importance.csv"
    data = load_csv(source).copy()

    required = {"Feature", "Mean_Absolute_SHAP"}

    if not required.issubset(data.columns) or data.empty:
        raise ValueError("Global SHAP report is invalid.")

    if data["Mean_Absolute_SHAP"].isna().any():
        raise ValueError("Missing SHAP importance values.")

    results = data.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    ).head(5)

    return {
        "supported": True,
        "topic": "shap_importance",
        "message": (
            "Global SHAP contributions from the XGBoost test "
            "predictions. Contributions are not causal explanations."
        ),
        "evidence": [
            {
                "source": source,
                "description": "Top five features by mean absolute SHAP",
                "records": results[["Feature", "Mean_Absolute_SHAP"]].to_dict(orient="records"),
            }
        ],
    }
