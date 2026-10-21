"""Verify saved Phase 06 anomaly-detection experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports/anomaly_detection"


@pytest.fixture(scope="module")
def reports():
    """Load the saved experiment reports once for all tests."""

    filenames = {
        "statistical": "01_statistical_test_scores.csv",
        "stat_anomalies": "02_statistical_anomalies.csv",
        "stat_summary": "03_statistical_summary.csv",
        "isolation": "04_isolation_forest_test_scores.csv",
        "iso_anomalies": "05_isolation_forest_anomalies.csv",
        "iso_summary": "06_isolation_forest_summary.csv",
        "comparison": "07_detector_comparison.csv",
        "agreement": "08_detector_agreement_summary.csv",
        "queue": "09_anomaly_investigation_queue.csv",
        "stores": "10_anomaly_store_summary.csv",
    }

    data = {}

    for name, filename in filenames.items():
        path = RESULTS / filename

        assert path.is_file(), f"Missing report: {filename}"

        data[name] = pd.read_csv(path)

    return data


def as_flags(data, column):
    """Convert saved CSV boolean flags to a boolean series."""

    values = data[column].astype(str).str.lower()

    assert values.isin(["true", "false"]).all()

    return values.eq("true")


def test_all_reports_exist(reports):
    """All ten expected CSV reports must be readable."""

    assert len(reports) == 10


def test_statistical_observation_coverage(reports):
    """Statistical scores must cover the complete held-out test set."""

    data = reports["statistical"]

    assert len(data) == 990
    assert data["Store"].nunique() == 45
    assert data["Date"].nunique() == 22
    assert not data.duplicated(["Store", "Date"]).any()


def test_statistical_residual_calculation(reports):
    """Residuals must equal actual sales minus predictions."""

    data = reports["statistical"]

    expected = data["Weekly_Store_Sales"] - data["Prediction"]

    assert np.allclose(data["Residual"], expected)


def test_statistical_threshold(reports):
    """Statistical anomaly flags must follow the saved threshold."""

    data = reports["statistical"]
    threshold = float(reports["stat_summary"]["Threshold"].iloc[0])

    expected = data["Absolute_Z_Score"] >= threshold
    actual = as_flags(data, "Is_Anomaly")

    assert threshold == 3.0
    assert actual.equals(expected)
    assert int(actual.sum()) == len(reports["stat_anomalies"])


def test_statistical_calibration(reports):
    """Statistical calibration must use validation observations."""

    summary = reports["stat_summary"].iloc[0]

    assert summary["Calibration_Split"] == "Validation"
    assert int(summary["Validation_Observations"]) == 945
    assert int(summary["Test_Observations"]) == 990
    assert float(summary["Residual_Std"]) > 0


def test_isolation_observation_coverage(reports):
    """Isolation Forest must evaluate the same test population."""

    data = reports["isolation"]

    assert len(data) == 990
    assert data["Store"].nunique() == 45
    assert data["Date"].nunique() == 22
    assert not data.duplicated(["Store", "Date"]).any()


def test_isolation_labels_and_flags(reports):
    """Isolation Forest uses -1 for anomalies and +1 for normal."""

    data = reports["isolation"]

    assert set(data["Isolation_Label"]).issubset({-1, 1})

    actual = as_flags(data, "Is_Anomaly")
    expected = data["Isolation_Label"].eq(-1)

    assert actual.equals(expected)
    assert int(actual.sum()) == len(reports["iso_anomalies"])


def test_isolation_scores_valid(reports):
    """All Isolation Forest scores must be finite."""

    data = reports["isolation"]

    assert np.isfinite(data["Anomaly_Score"]).all()

    # A negative decision score indicates an anomaly.
    assert (as_flags(data, "Is_Anomaly") == data["Anomaly_Score"].lt(0)).all()


def test_isolation_calibration(reports):
    """Isolation Forest must report validation-only calibration."""

    summary = reports["iso_summary"].iloc[0]

    assert summary["Calibration_Split"] == "Validation"
    assert int(summary["Validation_Observations"]) == 945
    assert int(summary["Test_Observations"]) == 990
    assert float(summary["Contamination"]) == pytest.approx(0.02)


def test_comparison_matches_original_detectors(reports):
    """Comparison flags must match both source experiments."""

    comparison = reports["comparison"]

    assert len(comparison) == 990
    assert not comparison.duplicated(["Store", "Date"]).any()

    for source, comparison_column in [
        ("statistical", "Statistical_Anomaly"),
        ("isolation", "Isolation_Anomaly"),
    ]:
        original = reports[source][["Store", "Date", "Is_Anomaly"]].copy()

        original["Source_Flag"] = as_flags(original, "Is_Anomaly")

        merged = comparison.merge(
            original[["Store", "Date", "Source_Flag"]],
            on=["Store", "Date"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        assert len(merged) == 990
        assert merged["_merge"].eq("both").all()

        assert (as_flags(merged, comparison_column) == merged["Source_Flag"]).all()


def test_agreement_categories(reports):
    """Every observation must belong to exactly one category."""

    data = reports["comparison"]

    statistical = as_flags(data, "Statistical_Anomaly")
    isolation = as_flags(data, "Isolation_Anomaly")

    expected = np.select(
        [
            statistical & isolation,
            statistical & ~isolation,
            ~statistical & isolation,
        ],
        [
            "Both",
            "Statistical only",
            "Isolation Forest only",
        ],
        default="Neither",
    )

    assert np.array_equal(
        data["Agreement_Category"].to_numpy(),
        expected,
    )


def test_agreement_summary_reconciles(reports):
    """Agreement counts and Jaccard similarity must be consistent."""

    data = reports["comparison"]
    summary = reports["agreement"].iloc[0]

    counts = data["Agreement_Category"].value_counts()

    both = int(counts.get("Both", 0))
    stat_only = int(counts.get("Statistical only", 0))
    iso_only = int(counts.get("Isolation Forest only", 0))
    neither = int(counts.get("Neither", 0))

    union = both + stat_only + iso_only

    assert both + stat_only + iso_only + neither == 990

    assert int(summary["Both"]) == both
    assert int(summary["Statistical_Only"]) == stat_only
    assert int(summary["Isolation_Only"]) == iso_only
    assert int(summary["Neither"]) == neither
    assert int(summary["Union"]) == union

    expected_jaccard = both / union if union else 1.0

    assert float(summary["Jaccard_Similarity"]) == pytest.approx(expected_jaccard)

    assert float(summary["Overall_Agreement_Pct"]) == pytest.approx(
        100 * (both + neither) / len(data)
    )


def test_investigation_queue(reports):
    """The queue must contain every uniquely flagged store-week."""

    comparison = reports["comparison"]
    queue = reports["queue"]

    flagged = comparison[as_flags(comparison, "Any_Anomaly")]

    assert len(queue) == len(flagged)
    assert not queue.duplicated(["Store", "Date"]).any()

    expected_keys = set(map(tuple, flagged[["Store", "Date"]].to_numpy()))

    actual_keys = set(map(tuple, queue[["Store", "Date"]].to_numpy()))

    assert actual_keys == expected_keys

    assert set(queue["Review_Priority"]).issubset({"High", "Review"})

    assert (
        queue.loc[
            queue["Agreement_Category"] == "Both",
            "Review_Priority",
        ]
        == "High"
    ).all()


def test_store_summary_reconciles(reports):
    """Store-level counts must equal the investigation queue."""

    queue = reports["queue"]
    stores = reports["stores"]

    assert stores["Store"].is_unique
    assert int(stores["Flagged_Weeks"].sum()) == len(queue)

    actual_counts = queue.groupby("Store").size()

    saved_counts = stores.set_index("Store")["Flagged_Weeks"]

    pd.testing.assert_series_equal(
        actual_counts.sort_index(),
        saved_counts.sort_index(),
        check_names=False,
    )

    assert (
        stores[["Both_Methods", "Statistical_Only", "Isolation_Only"]].sum(axis=1)
        == stores["Flagged_Weeks"]
    ).all()
