#!/usr/bin/env python3
"""YourBench CLI - Dynamic Evaluation Set Generation with Large Language Models."""

import os
import sys
import atexit
from pathlib import Path
from datetime import datetime

import typer
from dotenv import load_dotenv
from loguru import logger


load_dotenv()


def configure_logging(debug: bool = False, log_dir: Path = None):
    """Configure structured logging with file output."""
    logger.remove()  # Remove default handler

    # Get log level from environment or debug flag
    log_level = "DEBUG" if debug else os.getenv("YOURBENCH_LOG_LEVEL", "INFO")

    # Console handler with structured format for logs with stage
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[stage]: <16}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        filter=lambda record: "stage" in record["extra"],  # Only if stage is set
        enqueue=True,  # Thread-safe, prevents I/O errors
    )

    # Fallback console handler for logs without stage
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
        filter=lambda record: "stage" not in record["extra"],
        enqueue=True,
    )

    # File handler - JSON structured logs
    if log_dir is None:
        log_dir = Path(os.getenv("YOURBENCH_LOG_DIR", "logs"))
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"yourbench_{timestamp}.jsonl"

    logger.add(
        str(log_file),
        format="{message}",
        level="DEBUG",  # Always capture everything in files
        serialize=True,  # JSON format
        enqueue=True,
        rotation="100 MB",  # Rotate large files
    )

    # Summary log file (INFO and above only)
    summary_file = log_dir / f"yourbench_{timestamp}_summary.log"
    logger.add(
        str(summary_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra} | {message}",
        level="INFO",
        filter=lambda record: record["level"].no >= 20,  # INFO and above
        enqueue=True,
    )

    logger.info(f"Logging configured. JSON logs: {log_file}, Summary: {summary_file}")
    return log_file, summary_file


def cleanup_logging():
    """Ensure all logs are flushed and closed."""
    logger.complete()  # Flush all pending logs


atexit.register(cleanup_logging)

# Initialize logging with default configuration
configure_logging()

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
        configure_logging(debug=True)

    # Log at the global level first
    logger.info(f"Starting YourBench with config: {config_path}")

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

    try:
        config = load_config(config_file)
        if debug:
            config.debug = True
        run_pipeline_with_config(config, debug=debug)
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
