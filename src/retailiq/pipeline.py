"""Ordered local research pipeline; imports heavy ML libraries only when needed."""

import importlib
import logging

logger = logging.getLogger(__name__)

STAGES = {
    "data": (
        "data.inspection",
        "data.quality.source_keys",
        "data.quality.missing_values",
        "data.quality.training_data",
        "data.quality.sales_anomalies",
        "data.transformation",
    ),
    "analytics": ("analytics.weekly", "analytics.performance", "analytics.holidays"),
    "forecasting": (
        "forecasting.series",
        "forecasting.baseline",
        "forecasting.features",
        "forecasting.random_forest",
        "forecasting.xgboost_model",
        "forecasting.comparison",
    ),
    "reliability": ("reliability.errors", "reliability.relative_errors", "reliability.backtest"),
    "anomalies": ("anomalies.statistical", "anomalies.isolation_forest", "anomalies.comparison"),
    "explainability": ("explainability.shap_explanations",),
}


def planned_steps(stage: str = "all") -> tuple[str, ...]:
    """Return deterministic stage order without importing or running any experiment."""
    groups = STAGES.values() if stage == "all" else (STAGES[stage],)
    return tuple(f"retailiq.{module}" for group in groups for module in group)


def run_pipeline(stage: str = "all") -> None:
    """Run selected steps in order, stopping immediately on the first failure."""
    for name in planned_steps(stage):
        logger.info("Starting %s", name)
        try:
            importlib.import_module(name).main()
        except Exception as error:
            logger.error("Stage %s failed (%s)", name, type(error).__name__)
            raise
        logger.info("Completed %s", name)
