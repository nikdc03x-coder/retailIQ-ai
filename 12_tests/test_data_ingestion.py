"""Tests for the RetailIQ AI data-ingestion phase."""

import importlib.util
from pathlib import Path

import pandas as pd


# Load numbered scripts using their file paths.
ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    """Import a Python file with a numbered filename."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loader = load_module(
    "sales_loader",
    ROOT / "01_data_ingestion" / "01_load_csv.py",
)

inspector = load_module(
    "sales_inspector",
    ROOT / "01_data_ingestion" / "02_inspect_raw_data.py",
)


def test_loader_reads_all_datasets():
    # Check that all expected source datasets load.
    datasets = loader.load_sales_data()

    assert set(datasets) == {"train", "features", "stores"}
    assert all(not df.empty for df in datasets.values())


def test_inspection_counts_missing_values():
    # Use a small sample to verify missing-value counting.
    sample = pd.DataFrame({
        "Store": [1, 2],
        "Sales": [100.0, None],
    })

    result = inspector.inspect_dataset(sample)

    assert result["rows"] == 2
    assert result["columns"] == 2
    assert result["missing_values"]["Sales"] == 1


def test_inspection_detects_duplicates():
    # Confirm exact duplicate records are detected.
    sample = pd.DataFrame({
        "Store": [1, 1, 2],
        "Sales": [100, 100, 200],
    })

    result = inspector.inspect_dataset(sample)

    assert result["duplicate_rows"] == 1