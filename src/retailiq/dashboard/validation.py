"""Data contracts shared by dashboard pages and startup checks."""

import numpy as np
import pandas as pd

RAW_SCHEMAS = {
    "train.csv": {"Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"},
    "features.csv": {"Store", "Date", "IsHoliday"},
    "stores.csv": {"Store", "Type", "Size"},
}
ANALYTICAL_SCHEMA = RAW_SCHEMAS["train.csv"] | {"Type"}


def require_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
    """Reject empty data or an incompatible schema before accessing columns."""
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns: {', '.join(sorted(missing))}.")
    if data.empty:
        raise ValueError(f"{name}: no observations found.")


def validate_source(data: pd.DataFrame, required: set[str], name: str) -> pd.DataFrame:
    """Normalize chart inputs while retaining legitimate negative sales and optional nulls."""
    require_columns(data, required, name)
    data = data.copy()
    for column in required & {"Store", "Dept", "Weekly_Sales", "Size"}:
        numeric = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: {column} contains missing or invalid numeric values.")
        data[column] = numeric
    if "Date" in required:
        dates = pd.to_datetime(data["Date"], errors="coerce")
        if dates.isna().any():
            raise ValueError(f"{name}: Date contains missing or invalid dates.")
        data["Date"] = dates
    if "IsHoliday" in required:
        flags = data["IsHoliday"].astype(str).str.strip().str.lower()
        if not flags.isin(["true", "false"]).all():
            raise ValueError(f"{name}: IsHoliday must contain True or False.")
        data["IsHoliday"] = flags.eq("true")
    if "Type" in required:
        if data["Type"].isna().any() or data["Type"].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name}: Type contains missing store types.")
    return data
