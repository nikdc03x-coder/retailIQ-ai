
"""Verified sales calculations for custom RetailIQ AI questions."""

import pandas as pd


def prepare_sales(data, id_column, sales_column):
    """Validate the supplied report and prepare numeric sales values."""

    required = {id_column, sales_column}

    if not required.issubset(data.columns):
        missing = required - set(data.columns)
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = data[[id_column, sales_column]].copy()

    # Do not silently turn invalid values into zero sales.
    result[sales_column] = pd.to_numeric(
        result[sales_column],
        errors="raise",
    )

    if result[sales_column].isna().any():
        raise ValueError("Sales values contain missing data.")

    if result[id_column].isna().any():
        raise ValueError("Entity IDs contain missing data.")

    return result


def rank_sales(data, id_column, sales_column, count=5, highest=True):
    """Rank stores or departments by aggregated historical sales."""

    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("Count must be an integer.")

    if count < 1 or count > 45:
        raise ValueError("Count must be between 1 and 45.")

    prepared = prepare_sales(data, id_column, sales_column)

    # Each store or department must have one final aggregated value.
    if prepared[id_column].duplicated().any():
        raise ValueError("Expected one row per store or department.")

    ranked = prepared.sort_values(
        by=[sales_column, id_column],
        ascending=[not highest, True],
    ).head(count)

    return ranked.reset_index(drop=True)


def compare_stores(
    data,
    first_store,
    second_store,
    id_column,
    sales_column,
):
    """Compare historical aggregated sales for two selected stores."""

    prepared = prepare_sales(data, id_column, sales_column)

    if prepared[id_column].duplicated().any():
        raise ValueError("Expected one row per store.")

    first_store = int(first_store)
    second_store = int(second_store)

    if first_store == second_store:
        raise ValueError("Choose two different stores.")

    # Match store IDs numerically instead of comparing text labels.
    store_ids = pd.to_numeric(
        prepared[id_column],
        errors="raise",
    )

    first_rows = prepared.loc[store_ids == first_store]
    second_rows = prepared.loc[store_ids == second_store]

    if len(first_rows) != 1 or len(second_rows) != 1:
        raise ValueError("One or both store IDs were not found.")

    first_sales = float(first_rows.iloc[0][sales_column])
    second_sales = float(second_rows.iloc[0][sales_column])

    difference = first_sales - second_sales

    return {
        "first_store": first_store,
        "second_store": second_store,
        "first_sales": first_sales,
        "second_sales": second_sales,
        "difference": difference,
        "higher_sales_store": (
            first_store if difference > 0
            else second_store if difference < 0
            else None
        ),
    }
