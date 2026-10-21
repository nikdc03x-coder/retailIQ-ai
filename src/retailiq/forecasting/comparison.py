"""Compare forecasting models on matching store-week observations."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from retailiq.config import ROOT

RESULTS = ROOT / "reports/sales_forecasting"

FILES = {
    "Baseline": "04_baseline_predictions.csv",
    "Random Forest": "07_random_forest_predictions.csv",
    "XGBoost": "10_xgboost_predictions.csv",
}

KEYS = ["Store", "Date", "Split"]


def load_predictions():
    """Load saved predictions without changing their values."""

    tables = {}

    for model, filename in FILES.items():
        path = RESULTS / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        table = pd.read_csv(path)
        table["Date"] = pd.to_datetime(table["Date"])

        if table.duplicated(KEYS).any():
            raise ValueError(f"Duplicate keys in {model}")

        tables[model] = table

    return tables


def align_predictions(tables):
    """Keep only observations available for every model."""

    common = None

    for table in tables.values():
        keys = table[KEYS]

        if common is None:
            common = keys.copy()
        else:
            common = common.merge(
                keys,
                on=KEYS,
                how="inner",
                validate="one_to_one",
            )

    aligned = {}

    for model, table in tables.items():
        aligned[model] = common.merge(
            table,
            on=KEYS,
            how="left",
            validate="one_to_one",
        )

        if aligned[model]["Prediction"].isna().any():
            raise ValueError(f"Missing predictions for {model}")

    # Verify that all models use identical actual sales.
    reference = aligned["Baseline"][KEYS + ["Weekly_Store_Sales"]]

    for model, table in aligned.items():
        comparison = reference.merge(
            table[KEYS + ["Weekly_Store_Sales"]],
            on=KEYS,
            suffixes=("_Reference", "_Model"),
            validate="one_to_one",
        )

        if not np.allclose(
            comparison["Weekly_Store_Sales_Reference"],
            comparison["Weekly_Store_Sales_Model"],
            rtol=0,
            atol=0.01,
        ):
            raise ValueError(f"Actual sales differ for {model}")

    return aligned


def evaluate_models(aligned):
    """Calculate metrics using exactly the same observations."""

    rows = []

    for model, table in aligned.items():
        for split in ["Train", "Validation", "Test"]:
            subset = table[table["Split"] == split]

            if subset.empty:
                raise ValueError(f"No {split} observations for {model}")

            actual = subset["Weekly_Store_Sales"]
            predicted = subset["Prediction"]

            rows.append(
                {
                    "Model": model,
                    "Split": split,
                    "Observations": len(subset),
                    "MAE": mean_absolute_error(actual, predicted),
                    "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
                }
            )

    return pd.DataFrame(rows)


def main():
    """Save an aligned comparison and select by validation MAE."""

    tables = load_predictions()
    aligned = align_predictions(tables)
    metrics = evaluate_models(aligned)

    validation = metrics[metrics["Split"] == "Validation"].sort_values("MAE")

    selected_model = validation.iloc[0]["Model"]

    metrics.to_csv(
        RESULTS / "12_aligned_model_comparison.csv",
        index=False,
    )

    print("Aligned model comparison completed")

    print("\nValidation ranking:")
    print(validation[["Model", "Observations", "MAE", "RMSE"]].to_string(index=False))

    print("\nSelected model:", selected_model)

    print("\nTest results (reporting only):")
    print(
        metrics[metrics["Split"] == "Test"][["Model", "Observations", "MAE", "RMSE"]]
        .sort_values("MAE")
        .to_string(index=False)
    )

    print("\nAligned observations per model:", len(aligned["XGBoost"]))


if __name__ == "__main__":
    main()
