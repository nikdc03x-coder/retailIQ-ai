
"""Understand approved natural-language RetailIQ AI questions."""

import re
from pathlib import Path

import pandas as pd

from business_context import (
    retrieve_evidence as retrieve_existing_evidence,
)
from custom_analytics import compare_stores, rank_sales


# Reuse the existing project reports and their verified columns.
ROOT = Path(__file__).resolve().parents[1]

STORE_SOURCE = "business_analytics/02_store_performance.csv"
DEPARTMENT_SOURCE = "business_analytics/03_department_performance.csv"

STORE_REPORT = ROOT / "13_results" / STORE_SOURCE
DEPARTMENT_REPORT = ROOT / "13_results" / DEPARTMENT_SOURCE


# Support numeric counts and common written numbers.
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

COUNT = (
    r"(?P<count>\d{1,3}|one|two|three|four|five|"
    r"six|seven|eight|nine|ten)"
)

ENTITY = r"(?P<entity>stores?|departments?|depts?)"

PREFIX = (
    r"(?:(?:which (?:are|were) the|"
    r"what (?:are|were) the|"
    r"show(?: me)?(?: the)?|"
    r"list(?: me)?(?: the)?)\s+)?"
)


# Recognize several natural-language ranking formats.
RANK_PATTERNS = [
    re.compile(
        rf"{PREFIX}"
        rf"(?P<direction>top|bottom|highest|lowest)\s+"
        rf"{COUNT}\s+{ENTITY}"
        rf"(?:\s+(?:by|in terms of)\s+(?:total\s+)?sales)?"
    ),

    re.compile(
        rf"(?:which|what)\s+{COUNT}\s+{ENTITY}\s+"
        rf"(?:had|have|generated|recorded|made)\s+"
        rf"(?:the\s+)?"
        rf"(?P<direction>highest|lowest|most|least)\s+"
        rf"(?:total\s+)?sales"
    ),

    re.compile(
        rf"(?:list|show)(?:\s+me)?(?:\s+the)?\s+"
        rf"{COUNT}\s+"
        rf"(?P<direction>highest|lowest|best|worst)"
        rf"[-\s]selling\s+{ENTITY}"
    ),
]


# Each comparison pattern extracts exactly two store IDs.
COMPARE_PATTERNS = [
    re.compile(
        r"compare\s+store\s+(\d{1,4})\s+"
        r"(?:and|with|vs|versus|to)\s+"
        r"store\s+(\d{1,4})"
        r"(?:\s+(?:by|on)\s+(?:total\s+)?sales)?"
    ),

    re.compile(
        r"compare\s+(?:the\s+)?(?:total\s+)?sales\s+"
        r"(?:of\s+|between\s+)?"
        r"store\s+(\d{1,4})\s+"
        r"(?:and|with|vs|versus|to)\s+"
        r"store\s+(\d{1,4})"
    ),

    re.compile(
        r"how\s+much\s+(?:more|less)\s+did\s+"
        r"store\s+(\d{1,4})\s+"
        r"(?:sell|generate|earn)\s+than\s+"
        r"store\s+(\d{1,4})"
    ),

    re.compile(
        r"(?:what is|what's)\s+the\s+"
        r"(?:(?:total\s+)?sales\s+)?difference\s+between\s+"
        r"store\s+(\d{1,4})\s+and\s+"
        r"store\s+(\d{1,4})"
    ),

    re.compile(
        r"how\s+do\s+store\s+(\d{1,4})\s+and\s+"
        r"store\s+(\d{1,4})\s+compare"
        r"(?:\s+(?:in|on|by)\s+(?:total\s+)?sales)?"
    ),
]


def normalize_question(question):
    """Remove unnecessary spaces and trailing punctuation."""

    normalized = re.sub(
        r"\s+",
        " ",
        question.strip().lower(),
    )

    normalized = normalized.rstrip("?.! ").strip()

    # Support common polite introductions.
    if normalized.startswith("please "):
        normalized = normalized[len("please "):]

    return normalized


def parse_count(value):
    """Convert a written count or digit into an integer."""

    if value in NUMBER_WORDS:
        return NUMBER_WORDS[value]

    return int(value)


def parse_custom_question(question):
    """Map recognized wording to an approved analytics operation.

    Return None when the question does not match a supported format.
    No arbitrary Python, SQL or Gemini-generated code is executed.
    """

    normalized = normalize_question(question)

    # First check for store-comparison questions.
    for pattern in COMPARE_PATTERNS:
        match = pattern.fullmatch(normalized)

        if match:
            return {
                "operation": "compare_stores",
                "first_store": int(match.group(1)),
                "second_store": int(match.group(2)),
            }

    # Then check ranking requests.
    for pattern in RANK_PATTERNS:
        match = pattern.fullmatch(normalized)

        if not match:
            continue

        values = match.groupdict()

        direction = values["direction"]
        count = parse_count(values["count"])
        entity = values["entity"]

        highest = direction in {
            "top",
            "highest",
            "most",
            "best",
        }

        operation = (
            "rank_stores"
            if entity.startswith("store")
            else "rank_departments"
        )

        return {
            "operation": operation,
            "count": count,
            "highest": highest,
        }

    return None


def unsupported(message):
    """Return a safe response for unavailable calculations."""

    return {
        "supported": False,
        "message": message,
        "evidence": [],
    }


def load_report(path):
    """Load a saved RetailIQ AI report without modifying it."""

    if not path.is_file():
        raise FileNotFoundError(
            "Required project report is missing."
        )

    return pd.read_csv(path)


def store_ranking(count, highest):
    """Calculate highest- or lowest-selling stores."""

    data = load_report(STORE_REPORT)

    ranked = rank_sales(
        data,
        id_column="Store",
        sales_column="Total_Sales",
        count=count,
        highest=highest,
    )

    records = []

    for index, row in ranked.iterrows():
        records.append({
            "Rank": index + 1,
            "Store": int(row["Store"]),
            "Total_Sales": round(
                float(row["Total_Sales"]),
                2,
            ),
        })

    direction = "highest" if highest else "lowest"

    return {
        "supported": True,
        "topic": "custom_store_ranking",
        "message": "",
        "evidence": [{
            "source": STORE_SOURCE,
            "description": (
                f"The {count} stores with the {direction} "
                "historical total sales, calculated from the "
                "saved aggregated store performance report."
            ),
            "records": records,
        }],
    }


def department_ranking(count, highest):
    """Calculate highest- or lowest-selling departments."""

    data = load_report(DEPARTMENT_REPORT)

    ranked = rank_sales(
        data,
        id_column="Dept",
        sales_column="Total_Sales",
        count=count,
        highest=highest,
    )

    records = []

    for index, row in ranked.iterrows():
        records.append({
            "Rank": index + 1,
            "Dept": int(row["Dept"]),
            "Total_Sales": round(
                float(row["Total_Sales"]),
                2,
            ),
        })

    direction = "highest" if highest else "lowest"

    return {
        "supported": True,
        "topic": "custom_department_ranking",
        "message": "",
        "evidence": [{
            "source": DEPARTMENT_SOURCE,
            "description": (
                f"The {count} departments with the {direction} "
                "historical total sales, calculated from the "
                "saved aggregated department performance report."
            ),
            "records": records,
        }],
    }


def store_comparison(first_store, second_store):
    """Compare two stores using historical total sales."""

    data = load_report(STORE_REPORT)

    result = compare_stores(
        data,
        first_store=first_store,
        second_store=second_store,
        id_column="Store",
        sales_column="Total_Sales",
    )

    first_sales = round(result["first_sales"], 2)
    second_sales = round(result["second_sales"], 2)

    difference = round(
        abs(result["difference"]),
        2,
    )

    # Keep the higher-selling store explicit in the evidence.
    higher_store = result["higher_sales_store"]

    records = [
        {
            "Store": first_store,
            "Total_Sales": first_sales,
            "Absolute_Sales_Difference": difference,
            "Higher_Sales_Store": higher_store,
        },
        {
            "Store": second_store,
            "Total_Sales": second_sales,
            "Absolute_Sales_Difference": difference,
            "Higher_Sales_Store": higher_store,
        },
    ]

    return {
        "supported": True,
        "topic": "custom_store_comparison",
        "message": "",
        "evidence": [{
            "source": STORE_SOURCE,
            "description": (
                f"Comparison between Store {first_store} and "
                f"Store {second_store} using historical total sales. "
                "The absolute difference and higher-selling store "
                "were calculated using Pandas."
            ),
            "records": records,
        }],
    }


def retrieve_evidence(question):
    """Run an approved calculation or use the existing context engine."""

    request = parse_custom_question(question)

    # Existing forecasting, SHAP, anomalies and original questions
    # continue through the established Business Context Engine.
    if request is None:
        return retrieve_existing_evidence(question)

    try:
        operation = request["operation"]

        if operation == "compare_stores":
            return store_comparison(
                request["first_store"],
                request["second_store"],
            )

        if operation == "rank_stores":
            return store_ranking(
                request["count"],
                request["highest"],
            )

        if operation == "rank_departments":
            return department_ranking(
                request["count"],
                request["highest"],
            )

        return unsupported(
            "This business analysis is not supported."
        )

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ):
        return unsupported(
            "I couldn't calculate this request from the saved "
            "RetailIQ AI reports. Check the selected store IDs, "
            "ranking size and available data."
        )
