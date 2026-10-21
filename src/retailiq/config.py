"""Shared filesystem paths and validated application configuration."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# RETAILIQ_HOME is a process setting. A checkout defaults to its repository root;
# installed distributions can point at a directory containing data/reports/models.
ROOT = (
    Path(os.environ.get("RETAILIQ_HOME", Path(__file__).resolve().parents[2]))
    .expanduser()
    .resolve()
)
load_dotenv(ROOT / ".env", override=False)
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"


class ConfigurationError(ValueError):
    """An application setting has an unsupported value."""


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-2.5-flash"
    gemini_timeout_ms: int = 15000
    log_level: str = "INFO"


def get_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Validate settings without requiring a key for offline analytics."""
    env = os.environ if environ is None else environ
    model = env.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not model:
        raise ConfigurationError("GEMINI_MODEL must be nonempty.")
    level = env.get("LOG_LEVEL", "INFO").strip().upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigurationError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
    try:
        timeout = int(env.get("GEMINI_TIMEOUT_MS", "15000"))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            "GEMINI_TIMEOUT_MS must be an integer from 100 to 120000."
        ) from error
    if not 100 <= timeout <= 120000:
        raise ConfigurationError("GEMINI_TIMEOUT_MS must be an integer from 100 to 120000.")
    return Settings(env.get("GEMINI_API_KEY", "").strip(), model, timeout, level)
