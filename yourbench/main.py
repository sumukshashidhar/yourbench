"""
Universal main.py script for ui and cli

Usage:
    python main.py --gui # launches the gradio ui
    python main.py       # runs in cli mode
"""
import argparse
from yourbench.ui import launch_ui


def main():
    """
    Main entry point for the CLI application.
    Parses command line arguments and either launches the GUI or runs in CLI mode.
    """
    parser = argparse.ArgumentParser(
        description="Dynamic Evaluation Set Generation with Large Language Models"
    )
    parser.add_argument("--gui", action="store_true", help="Launch the gradio UI")
    args = parser.parse_args()

    if args.gui:
        launch_ui()
    else:
        # CLI mode logic can be added here
        print("Running in CLI mode. No GUI is launched.")


if __name__ == "__main__":
    main()