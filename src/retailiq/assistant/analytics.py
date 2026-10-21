"""Verified sales calculations for RetailIQ AI custom questions."""

import numpy as np
import pandas as pd


def prepare_sales(data, id_column, sales_column):
    """Select and validate the columns needed for sales analysis."""

    required = {id_column, sales_column}

    if not required.issubset(data.columns):
        missing = required - set(data.columns)
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = data[[id_column, sales_column]].copy()

    # Sales must be numeric and complete.
    result[sales_column] = pd.to_numeric(
        result[sales_column],
        errors="raise",
    )

    if not np.isfinite(result[sales_column].to_numpy(dtype=float)).all():
        raise ValueError("Sales values must be finite and complete.")

    if result[id_column].isna().any():
        raise ValueError("Entity IDs contain missing data.")

    return result


def rank_sales(
    data,
    id_column,
    sales_column,
    count=5,
    highest=True,
):
    """Rank stores or departments using their verified sales totals."""

    if type(count) is not int:
        raise ValueError("Count must be an integer.")

    if count < 1:
        raise ValueError("Count must be at least 1.")

    if type(highest) is not bool:
        raise ValueError("Highest must be a boolean.")

    prepared = prepare_sales(
        data,
        id_column,
        sales_column,
    )

    if prepared.empty:
        raise ValueError("No sales records available.")

    if prepared[id_column].duplicated().any():
        raise ValueError("Expected one row per store or department.")

    # The report determines the available number of entities.
    # This works for both 45 stores and 81 departments.
    ranked = prepared.sort_values(
        by=[sales_column, id_column],
        ascending=[not highest, True],
    )

    # Return up to the requested number of available entities.
    return ranked.head(count).reset_index(drop=True)


def compare_stores(
    data,
    first_store,
    second_store,
    id_column,
    sales_column,
):
    """Compare the total sales of two distinct existing stores."""

    prepared = prepare_sales(
        data,
        id_column,
        sales_column,
    )

    if prepared[id_column].duplicated().any():
        raise ValueError("Expected one row per store.")

    first_store = int(first_store)
    second_store = int(second_store)

    if first_store == second_store:
        raise ValueError("Choose two different stores.")

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

    if difference > 0:
        higher_store = first_store
    elif difference < 0:
        higher_store = second_store
    else:
        higher_store = None

    return {
        "first_store": first_store,
        "second_store": second_store,
        "first_sales": first_sales,
        "second_sales": second_sales,
        "difference": difference,
        "higher_sales_store": higher_store,
    }
