"""
Universal main.py script for UI and CLI

Usage:
    yourbench                    # Launches interactive CLI mode
    yourbench run --config ...   # Runs the pipeline with the given config file
    yourbench analyze ...        # Runs a specific analysis
    yourbench gui               # Launches the Gradio UI
"""

import os
import sys

import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from yourbench.analysis import run_analysis
from yourbench.config_cache import get_last_config, save_last_config
from yourbench.pipeline import run_pipeline


console = Console()


def get_config_files():
    """Get all config files from the configs directory."""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
    if not os.path.exists(config_dir):
        return []
    return [f for f in os.listdir(config_dir) if f.endswith(('.yaml', '.yml', '.json'))]


def display_welcome():
    """Display a welcome message with ASCII art and instructions."""
    welcome_text = """
    [bold blue]Welcome to YourBench![/bold blue]

    [italic]Dynamic Evaluation Set Generation with Large Language Models[/italic]

    [yellow]Available Commands:[/yellow]
    • [green]run[/green] - Run the pipeline with a config file
    • [green]analyze[/green] - Run a specific analysis
    • [green]gui[/green] - Launch the Gradio UI
    • [green]exit[/green] - Exit the program
    """
    console.print(Panel(welcome_text, title="[bold]YourBench CLI[/bold]", border_style="blue"))


def interactive_mode():
    """Run the interactive CLI mode."""
    display_welcome()

    while True:
        command = Prompt.ask("\n[bold blue]yourbench[/bold blue]")

        if command.lower() == "exit":
            console.print("[yellow]Goodbye![/yellow]")
            break

        elif command.lower() == "run":
            config_files = get_config_files()
            if not config_files:
                console.print("[red]No config files found in the configs directory![/red]")
                continue

            table = Table(title="Available Config Files")
            table.add_column("Index", style="cyan")
            table.add_column("Config File", style="green")

            for idx, config in enumerate(config_files, 1):
                table.add_row(str(idx), config)

            console.print(table)

            try:
                choice = int(Prompt.ask("\nSelect a config file by number", default="1"))
                if 1 <= choice <= len(config_files):
                    config_path = os.path.join("configs", config_files[choice - 1])
                    save_last_config(config_path)
                    run_pipeline(config_path)
                else:
                    console.print("[red]Invalid selection![/red]")
            except ValueError:
                console.print("[red]Please enter a valid number![/red]")

        elif command.lower() == "analyze":
            # TODO: Implement analysis selection
            console.print("[yellow]Analysis mode coming soon![/yellow]")

        else:
            console.print("[red]Unknown command![/red]")
            console.print("[yellow]Available commands: run, analyze, gui, exit[/yellow]")


@click.group()
def cli():
    """YourBench - Dynamic Evaluation Set Generation with Large Language Models"""
    pass


@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Path to the configuration file')
@click.option('--debug/--no-debug', default=False, help='Enable debug logging')
def run(config, debug):
    """Run the pipeline with a configuration file."""
    if not config:
        config = get_last_config()
        if config:
            logger.info(f"No config specified, using last used config: {config}")
        else:
            logger.error("No config file specified and no cached config found. Please specify a config file using --config")
            return

    logger.remove()  # Remove default handlers
    logger.add(lambda msg: print(msg, end=""), level="DEBUG" if debug else "INFO")

    save_last_config(config)
    run_pipeline(config, debug=debug)


@cli.command()
@click.argument('analysis_name')
@click.argument('args', nargs=-1)
@click.option('--debug/--no-debug', default=False, help='Enable debug logging')
def analyze(analysis_name, args, debug):
    """Run a specific analysis."""
    logger.remove()  # Remove default handlers
    logger.add(lambda msg: print(msg, end=""), level="DEBUG" if debug else "INFO")

    run_analysis(analysis_name, args, debug=debug)


def main():
    """Main entry point for the CLI application."""
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        cli()


if __name__ == "__main__":
    main()
