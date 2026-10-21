"""Small command line interface for reproducible research runs."""

import argparse

from retailiq.config import ConfigurationError, get_settings
from retailiq.logging_config import configure_logging
from retailiq.pipeline import STAGES, planned_steps, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retailiq")
    commands = parser.add_subparsers(dest="command", required=True)
    pipeline = commands.add_parser("pipeline", help="Generate local research reports and models")
    pipeline.add_argument("--stage", choices=["all", *STAGES], default="all")
    pipeline.add_argument("--list", action="store_true", help="List steps without changing files")
    commands.add_parser("check-config", help="Validate configuration without printing secrets")
    args = parser.parse_args(argv)
    try:
        settings = get_settings()
        configure_logging(settings.log_level)
        if args.command == "check-config":
            print(
                "Configuration valid. Gemini: "
                + ("configured" if settings.gemini_api_key else "offline")
            )
        elif args.list:
            print("\n".join(planned_steps(args.stage)))
        else:
            run_pipeline(args.stage)
    except ConfigurationError as error:
        parser.exit(2, f"Configuration error: {error}\n")
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        parser.exit(
            1,
            f"Pipeline failed ({type(error).__name__}). Check stage prerequisites and setup in README.md.\n",
        )
    return 0
