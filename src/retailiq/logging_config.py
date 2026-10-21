"""Application-scoped logging that leaves host application handlers intact."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Install one handler even across Streamlit reruns; never log settings."""
    logger = logging.getLogger("retailiq")
    logger.setLevel(level)
    if not any(getattr(handler, "_retailiq", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler._retailiq = True
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
