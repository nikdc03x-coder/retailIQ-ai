"""Automated tests for the RetailIQ analytical dataset."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "03_data_transformation"
    / "01_build_analytical_dataset.py"
)


# Import the numbered script without renaming its file.
spec = importlib.util.spec_from_file_location(
    "analytical_builder", SCRIPT
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture(scope="module")
def datasets():
    """Load source files once for all transformation tests."""

    raw = ROOT / "data" / "raw"

    sales = pd.read_csv(raw / "train.csv")
    features = pd.read_csv(raw / "features.csv")
    stores = pd.read_csv(raw / "stores.csv")

    return sales, features, stores


@pytest.fixture(scope="module")
def transformed(datasets):
    """Build the analytical dataset once for the test suite."""

    return builder.build_analytical_dataset(*datasets)


def test_row_count_preserved(datasets, transformed):
    """Joining must not create or remove sales records."""

    sales = datasets[0]
    assert len(transformed) == len(sales)


def test_sales_values_preserved(datasets, transformed):
    """Original sales values must remain unchanged."""

    sales = datasets[0]

    pd.testing.assert_series_equal(
        transformed["Weekly_Sales"].reset_index(drop=True),
        sales["Weekly_Sales"].reset_index(drop=True),
    )


def test_business_keys_unique(transformed):
    """Every store-department-week must remain unique."""

    assert not transformed.duplicated(
        ["Store", "Dept", "Date"]
    ).any()


def test_negative_sales_preserved(transformed):
    """Negative sales must not disappear during cleaning."""

    assert int(transformed["Is_Negative_Sales"].sum()) == 1285


def test_store_metadata_complete(transformed):
    """Every sales record must have store information."""

    assert not transformed[["Type", "Size"]].isna().any().any()


def test_markdown_missing_flags(transformed):
    """Missingness indicators must match original missing values."""

    for number in range(1, 6):
        column = f"MarkDown{number}"

        pd.testing.assert_series_equal(
            transformed[f"{column}_Missing"],
            transformed[column].isna(),
            check_names=False,
        )


def test_duplicate_feature_keys_rejected(datasets):
    """The builder must reject ambiguous feature joins."""

    sales, features, stores = datasets

    duplicate = pd.concat(
        [features, features.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate business keys"):
        builder.build_analytical_dataset(
            sales, duplicate, stores
        )


def test_holiday_mismatch_rejected(datasets):
    """Conflicting holiday labels must stop transformation."""

    sales, features, stores = datasets
    modified = features.copy()

    modified.loc[0, "IsHoliday"] = (
        not bool(modified.loc[0, "IsHoliday"])
    )

    with pytest.raises(ValueError, match="Holiday flags"):
        builder.build_analytical_dataset(
            sales, modified, stores
        )