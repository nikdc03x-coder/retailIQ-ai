"""Temporal contracts shared by one-week-ahead forecasting experiments."""

import pandas as pd


def validate_weekly_history(weekly: pd.DataFrame) -> None:
    """A shifted row is a prior week only when each store has contiguous history."""
    if weekly.empty:
        raise ValueError("Weekly sales history is empty.")
    gaps = weekly.groupby("Store")["Date"].diff().dropna()
    if not gaps.dt.total_seconds().eq(7 * 24 * 60 * 60).all():
        raise ValueError(
            "Each store must have consecutive weekly observations before building lags."
        )
