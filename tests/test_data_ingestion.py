"""Tests for the RetailIQ AI data-ingestion phase."""

import pandas as pd

from retailiq.data import ingestion as loader
from retailiq.data import inspection as inspector


def test_loader_reads_all_datasets():
    # Check that all expected source datasets load.
    datasets = loader.load_sales_data()

    assert set(datasets) == {"train", "features", "stores"}
    assert all(not df.empty for df in datasets.values())


def test_inspection_counts_missing_values():
    # Use a small sample to verify missing-value counting.
    sample = pd.DataFrame(
        {
            "Store": [1, 2],
            "Sales": [100.0, None],
        }
    )

    result = inspector.inspect_dataset(sample)

    assert result["rows"] == 2
    assert result["columns"] == 2
    assert result["missing_values"]["Sales"] == 1


def test_inspection_detects_duplicates():
    # Confirm exact duplicate records are detected.
    sample = pd.DataFrame(
        {
            "Store": [1, 1, 2],
            "Sales": [100, 100, 200],
        }
    )

    result = inspector.inspect_dataset(sample)

    assert result["duplicate_rows"] == 1
