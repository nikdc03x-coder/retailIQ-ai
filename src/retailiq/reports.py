"""Framework-independent, confined access to saved CSV reports."""

from pathlib import Path

import pandas as pd

from retailiq.config import REPORTS_DIR


def resolve_report_path(relative_path: str, reports_dir: Path | None = None) -> Path:
    """Reject absolute paths, traversal, and symlinks escaping the report directory."""
    base = (REPORTS_DIR if reports_dir is None else reports_dir).resolve()
    relative = Path(relative_path)
    path = (base / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(base) or path.suffix.lower() != ".csv":
        raise ValueError("Expected a CSV path inside the reports directory.")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing report: {relative_path}. Run python -m retailiq pipeline."
        )
    return path


def load_csv(relative_path: str, reports_dir: Path | None = None) -> pd.DataFrame:
    """Read a saved report without importing the dashboard or retraining models."""
    return pd.read_csv(resolve_report_path(relative_path, reports_dir))
