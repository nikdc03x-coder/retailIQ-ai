"""Compare statistical and Isolation Forest anomaly detections."""

import pandas as pd

from retailiq.config import ROOT

OUTPUT = ROOT / "reports/anomaly_detection"

STATISTICAL = OUTPUT / "01_statistical_test_scores.csv"
ISOLATION = OUTPUT / "04_isolation_forest_test_scores.csv"

KEYS = ["Store", "Date"]


def load_results():
    """Read both saved experiments and validate their store-week keys."""

    statistical = pd.read_csv(STATISTICAL)
    isolation = pd.read_csv(ISOLATION)

    for name, data in [
        ("Statistical", statistical),
        ("Isolation Forest", isolation),
    ]:
        if not set(KEYS + ["Is_Anomaly"]).issubset(data.columns):
            raise ValueError(f"{name} results are missing columns.")

        if data.duplicated(KEYS).any():
            raise ValueError(f"{name} contains duplicate store-weeks.")

    statistical = statistical.rename(
        columns={
            "Is_Anomaly": "Statistical_Anomaly",
        }
    )

    isolation = isolation.rename(
        columns={
            "Is_Anomaly": "Isolation_Anomaly",
        }
    )

    # Retain statistical sales information and Isolation Forest scores.
    isolation = isolation[KEYS + ["Isolation_Anomaly", "Anomaly_Score"]]

    comparison = statistical.merge(
        isolation,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not comparison["_merge"].eq("both").all():
        raise ValueError("Detector observation coverage differs.")

    comparison = comparison.drop(columns="_merge")

    for column in ["Statistical_Anomaly", "Isolation_Anomaly"]:
        comparison[column] = comparison[column].astype(bool)

    return comparison


def categorize_agreement(data):
    """Identify observations flagged by both, one, or neither detector."""

    result = data.copy()

    statistical = result["Statistical_Anomaly"]
    isolation = result["Isolation_Anomaly"]

    result["Agreement_Category"] = "Neither"

    result.loc[
        statistical & isolation,
        "Agreement_Category",
    ] = "Both"

    result.loc[
        statistical & ~isolation,
        "Agreement_Category",
    ] = "Statistical only"

    result.loc[
        ~statistical & isolation,
        "Agreement_Category",
    ] = "Isolation Forest only"

    result["Any_Anomaly"] = statistical | isolation

    # A review priority, not a validated probability of an anomaly.
    result["Review_Priority"] = result["Agreement_Category"].map(
        {
            "Both": "High",
            "Statistical only": "Review",
            "Isolation Forest only": "Review",
            "Neither": "Normal",
        }
    )

    return result


def create_summary(data):
    """Calculate detector overlap and agreement metrics."""

    both = int((data["Agreement_Category"] == "Both").sum())

    statistical_only = int((data["Agreement_Category"] == "Statistical only").sum())

    isolation_only = int((data["Agreement_Category"] == "Isolation Forest only").sum())

    neither = int((data["Agreement_Category"] == "Neither").sum())

    union = both + statistical_only + isolation_only

    # Jaccard measures overlap among observations flagged by either.
    jaccard = both / union if union else 1.0

    return pd.DataFrame(
        [
            {
                "Test_Observations": len(data),
                "Statistical_Anomalies": both + statistical_only,
                "Isolation_Anomalies": both + isolation_only,
                "Both": both,
                "Statistical_Only": statistical_only,
                "Isolation_Only": isolation_only,
                "Neither": neither,
                "Union": union,
                "Jaccard_Similarity": jaccard,
                "Overall_Agreement_Pct": (100 * (both + neither) / len(data)),
            }
        ]
    )


def create_store_summary(data):
    """Show stores with the most observations requiring investigation."""

    flagged = data[data["Any_Anomaly"]].copy()

    if flagged.empty:
        return pd.DataFrame(
            columns=[
                "Store",
                "Flagged_Weeks",
                "Both_Methods",
                "Statistical_Only",
                "Isolation_Only",
            ]
        )

    summary = (
        flagged.groupby("Store")
        .agg(
            Flagged_Weeks=("Date", "size"),
            Both_Methods=(
                "Agreement_Category",
                lambda values: (values == "Both").sum(),
            ),
            Statistical_Only=(
                "Agreement_Category",
                lambda values: (values == "Statistical only").sum(),
            ),
            Isolation_Only=(
                "Agreement_Category",
                lambda values: (values == "Isolation Forest only").sum(),
            ),
        )
        .reset_index()
        .sort_values(
            ["Flagged_Weeks", "Both_Methods", "Store"],
            ascending=[False, False, True],
        )
    )

    return summary


def main():
    """Compare saved results and export investigation reports."""

    data = categorize_agreement(load_results())

    summary = create_summary(data)
    stores = create_store_summary(data)

    flagged = data[data["Any_Anomaly"]].copy()

    priority_order = {
        "High": 0,
        "Review": 1,
    }

    flagged["Priority_Order"] = flagged["Review_Priority"].map(priority_order)

    flagged = flagged.sort_values(
        ["Priority_Order", "Absolute_Z_Score"],
        ascending=[True, False],
    ).drop(columns="Priority_Order")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    data.to_csv(
        OUTPUT / "07_detector_comparison.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT / "08_detector_agreement_summary.csv",
        index=False,
    )

    flagged.to_csv(
        OUTPUT / "09_anomaly_investigation_queue.csv",
        index=False,
    )

    stores.to_csv(
        OUTPUT / "10_anomaly_store_summary.csv",
        index=False,
    )

    print("Anomaly detector comparison completed")

    print("\nAgreement summary:")
    print(summary.to_string(index=False))

    print("\nStores with repeated flagged weeks:")
    print(stores.head(10).to_string(index=False))

    print("\nInvestigation queue:")
    columns = [
        "Store",
        "Date",
        "Residual",
        "Agreement_Category",
        "Review_Priority",
    ]
    print(flagged[columns].head(20).to_string(index=False))

    print("\nSaved comparison reports to:", OUTPUT)


if __name__ == "__main__":
    main()
