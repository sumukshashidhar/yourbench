#!/usr/bin/env python3
"""YourBench CLI - Dynamic Evaluation Set Generation with Large Language Models."""

import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger


# Configure logging
logger.remove()
logger.add(sys.stderr, level=os.getenv("YOURBENCH_LOG_LEVEL", "INFO"))

load_dotenv()

app = typer.Typer(
    name="yourbench",
    help="YourBench - Dynamic Evaluation Set Generation with Large Language Models.",
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


@app.command()
def run(
    config_path: str = typer.Argument(..., help="Path to YAML config file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Run YourBench pipeline with a config file."""
    if debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Config file not found: {config_path}")
        raise typer.Exit(1)

    if config_file.suffix not in {".yaml", ".yml"}:
        logger.error(f"Config must be a YAML file (.yaml or .yml): {config_path}")
        raise typer.Exit(1)

    logger.info(f"Running with config: {config_file}")

    from yourbench.conf.loader import load_config
    from yourbench.pipeline.handler import run_pipeline_with_config
    from yourbench.utils.dataset_engine import upload_dataset_card

    try:
        config = load_config(config_file)
        if debug:
            config.debug = True
        run_pipeline_with_config(config, debug=debug)
        try:
            upload_dataset_card(config)
        except Exception as e:
            logger.warning(f"Failed to upload dataset card: {e}")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise typer.Exit(1)


@app.command("version")
def version_command() -> None:
    """Show YourBench version."""
    from importlib.metadata import version as get_version

    try:
        v = get_version("yourbench")
        print(f"YourBench version: {v}")
    except Exception:
        print("YourBench version: development")


def main() -> None:
    """Entry point for the CLI."""
    # Handle version flag
    if "--version" in sys.argv or "-v" in sys.argv:
        version_command()
        return

    # If no arguments, show help
    if len(sys.argv) == 1:
        app()
        return

    # If first arg looks like a path (not a command), assume it's 'run'
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if not first_arg.startswith("-") and first_arg not in ["run", "version"]:
            sys.argv = [sys.argv[0], "run"] + sys.argv[1:]

    app()


if __name__ == "__main__":
    main()
