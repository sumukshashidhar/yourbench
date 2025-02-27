"""
Universal main.py script for UI and CLI

Usage:
    python main.py --gui                         # Launches the Gradio UI
    python main.py                               # Runs in CLI mode (default)
    python main.py --config config.yaml          # Runs the pipeline with the given config file
    python main.py --config config.yaml --debug  # Runs the pipeline with debug logging enabled

Options:
    --gui       Launch the Gradio UI.
    --config    Specify the path to the configuration file for pipeline execution.
    --debug     Enable debug logging (default is INFO level).
"""

import argparse
from loguru import logger
from yourbench.pipeline._handler import run_pipeline

def main() -> None:
    """
    Main entry point for the CLI application.
    Parses command line arguments and either launches the GUI or runs the pipeline in CLI mode.
    """
    parser = argparse.ArgumentParser(
        description="Dynamic Evaluation Set Generation with Large Language Models"
    )
    parser.add_argument("--gui", action="store_true", help="Launch the Gradio UI")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()

    # Configure logging level based on debug flag
    logger.remove()  # Remove default handlers
    logger.add(lambda msg: print(msg, end=""), level="DEBUG" if args.debug else "INFO")

    if args.gui:
        from yourbench.ui import launch_ui
        launch_ui()
    else:
        print("Running in CLI mode. No GUI is launched.")

    if args.config:
        logger.info(f"Running pipeline with config: {args.config} (Debug: {args.debug})")
        run_pipeline(args.config, debug=args.debug)

if __name__ == "__main__":
    main()
