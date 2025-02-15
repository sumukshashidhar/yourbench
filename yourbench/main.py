"""
Universal main.py script for ui and cli

Usage:
    python main.py --gui # launches the gradio ui
    python main.py       # runs in cli mode
"""
import argparse

def main() -> None:
    """
    Main entry point for the CLI application.
    Parses command line arguments and either launches the GUI or runs in CLI mode.
    """
    parser = argparse.ArgumentParser(
        description="Dynamic Evaluation Set Generation with Large Language Models"
    )
    parser.add_argument("--gui", action="store_true", help="Launch the gradio UI")
    parser.add_argument("--config", type=str, help="Path to the config file")
    args = parser.parse_args()

    if args.gui:
        from yourbench.ui import launch_ui
        launch_ui()
    else:
        # CLI mode logic can be added here
        print("Running in CLI mode. No GUI is launched.")
    
    
    # if we get a config file, we need to run the pipeline
    if args.config:
        # get the handler
        from yourbench.pipeline._handler import run_pipeline
        run_pipeline(args.config)
    
    return




if __name__ == "__main__":
    main()