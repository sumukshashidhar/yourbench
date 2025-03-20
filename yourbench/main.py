from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from yourbench.analysis import run_analysis
from yourbench.config_cache import get_last_config, save_last_config
from yourbench.pipeline.handler import run_pipeline


app = typer.Typer(
    name="yourbench",
    add_completion=True,
    help="YourBench - Dynamic Evaluation Set Generation with Large Language Models.",
    pretty_exceptions_show_locals=False,
)


@app.callback()
def main_callback() -> None:
    """
    Global callback for YourBench CLI.

    This function runs before any subcommand and can handle
    global flags or environment setup as needed.
    """
    pass


@app.command(help="Run the pipeline with the given configuration file.")
def run(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the configuration file (YAML, JSON). If not provided, attempts to use last used config.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging for additional details.",
    ),
    plot_stage_timing: bool = typer.Option(
        False,
        "--plot-stage-timing",
        help=(
            "If set, generates a bar chart illustrating how long each pipeline stage took. This requires matplotlib."
        ),
    ),
) -> None:
    """
    Run the YourBench pipeline using a specified config file.

    If no --config is provided, this command attempts to load the most recent
    cached config (if any). Use --plot-stage-timing to generate a stage-timing chart.
    """
    # Adjust logger according to debug level
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")

    # Use cached config if none specified
    if not config:
        cached_config: Optional[str] = get_last_config()
        if cached_config:
            config = Path(cached_config)
            logger.info(f"No config specified; using last used config: {config}")
        else:
            logger.error(
                "No config file specified and no cached config found.\n"
                "Please provide one with --config or create a config."
            )
            raise typer.Exit(code=1)

    # Ensure we have a valid config path
    if not config.exists():
        logger.error(f"Specified config file does not exist: {config}")
        raise typer.Exit(code=1)

    # Save to cache for later
    save_last_config(str(config))

    logger.info(f"Running pipeline with config: {config}")
    try:
        run_pipeline(
            config_file_path=str(config),
            debug=debug,
            plot_stage_timing=plot_stage_timing,
        )
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise typer.Exit(code=1)


@app.command(help="Run a specific analysis by name with optional arguments.")
def analyze(
    analysis_name: str = typer.Argument(..., help="Name of the analysis to run."),
    args: list[str] = typer.Argument(
        None,
        help="Additional arguments for the analysis (space-separated).",
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
) -> None:
    """
    Run a specific analysis by name, with optional space-separated arguments.

    Example:
        yourbench analyze summarization --debug arg1 arg2
    """
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")

    logger.info(f"Running analysis '{analysis_name}' with arguments: {args}")
    try:
        run_analysis(analysis_name, args, debug=debug)
    except Exception as e:
        logger.exception(f"Analysis '{analysis_name}' failed: {e}")
        raise typer.Exit(code=1)


@app.command(help="Launch the Gradio UI (if available).")
def gui() -> None:
    """
    Launch the Gradio UI for YourBench, if implemented.
    """
    logger.info("Launching the Gradio UI...")
    # TODO: Implement your Gradio UI logic here
    raise NotImplementedError("GUI support is not yet implemented.")


def main() -> None:
    """
    Main entry point for the CLI.

    If no arguments are provided, Typer shows the help message.
    """
    app()


if __name__ == "__main__":
    main()
