"""Build a validated analytical dataset from Walmart sales files."""

from pathlib import Path

import pandas as pd


# Locate the source and output folders.
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "processed" / "retail_analytical.parquet"

MARKDOWNS = [f"MarkDown{i}" for i in range(1, 6)]


def build_analytical_dataset(sales, features, stores):
    """Join source tables while preserving original sales records."""

    # Work on copies so the input datasets remain unchanged.
    sales = sales.copy()
    features = features.copy()
    stores = stores.copy()

    # Convert dates into a consistent datetime format.
    sales["Date"] = pd.to_datetime(sales["Date"], errors="raise")
    features["Date"] = pd.to_datetime(
        features["Date"], errors="raise"
    )

    # Reject missing or duplicate business keys.
    source_keys = [
        (sales, ["Store", "Dept", "Date"], "sales"),
        (features, ["Store", "Date"], "features"),
        (stores, ["Store"], "stores"),
    ]

    for df, keys, name in source_keys:
        if df[keys].isna().any().any():
            raise ValueError(f"{name}: missing business keys")

        if df.duplicated(subset=keys).any():
            raise ValueError(f"{name}: duplicate business keys")

    # Keep one holiday flag and validate agreement separately.
    feature_columns = [
        "Store", "Date", "IsHoliday",
        "Temperature", "Fuel_Price",
        *MARKDOWNS, "CPI", "Unemployment",
    ]

    combined = sales.merge(
        features[feature_columns],
        on=["Store", "Date"],
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("_sales", "_features"),
    )

    if not combined["_merge"].eq("both").all():
        raise ValueError("Some sales records lack feature data")

    combined = combined.drop(columns="_merge")

    # Holiday information must agree across both sources.
    if not (
        combined["IsHoliday_sales"]
        == combined["IsHoliday_features"]
    ).all():
        raise ValueError("Holiday flags do not match")

    combined = combined.rename(
        columns={"IsHoliday_sales": "IsHoliday"}
    ).drop(columns="IsHoliday_features")

    # Add store metadata without changing the sales grain.
    combined = combined.merge(
        stores[["Store", "Type", "Size"]],
        on="Store",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not combined["_merge"].eq("both").all():
        raise ValueError("Some sales records lack store metadata")

    combined = combined.drop(columns="_merge")

    # Preserve missing promotional values and record their absence.
    for column in MARKDOWNS:
        combined[f"{column}_Missing"] = (
            combined[column].isna()
        )

    # This flag is for descriptive analytics, not forecasting inputs.
    combined["Is_Negative_Sales"] = (
        combined["Weekly_Sales"] < 0
    )

    # Add useful calendar information derived from the date.
    combined["Year"] = combined["Date"].dt.year
    combined["Month"] = combined["Date"].dt.month
    combined["WeekOfYear"] = (
        combined["Date"].dt.isocalendar().week.astype(int)
    )

    # Validate that transformation has not changed sales history.
    if len(combined) != len(sales):
        raise ValueError("Sales row count changed")

    if not combined["Weekly_Sales"].reset_index(drop=True).equals(
        sales["Weekly_Sales"].reset_index(drop=True)
    ):
        raise ValueError("Original sales values or order changed")

    return combined


def main():
    """Read raw files, build the dataset and save as Parquet."""

    sales = pd.read_csv(RAW / "train.csv")
    features = pd.read_csv(RAW / "features.csv")
    stores = pd.read_csv(RAW / "stores.csv")

    dataset = build_analytical_dataset(
        sales, features, stores
    )

    # Parquet stores column types and is efficient for analysis.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT, index=False)

    print("Transformation completed")
    print("Rows:", len(dataset))
    print("Columns:", len(dataset.columns))
    print("Negative sales:", int(dataset["Is_Negative_Sales"].sum()))
    print("Date range:", dataset["Date"].min(), "to", dataset["Date"].max())
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()