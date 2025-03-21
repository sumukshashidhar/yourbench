"""
Analysis module for running various analyses on the evaluation results.
"""

from typing import List, Optional

from loguru import logger


def run_analysis(
    analysis_name: str, args: Optional[List[str]] = None, debug: bool = False
) -> None:
    """
    Run a specific analysis by name.

    Args:
        analysis_name: Name of the analysis to run
        args: Additional arguments for the analysis
        debug: Whether to enable debug logging
    """
    try:
        # Import the analysis module dynamically
        module_name = f"yourbench.analysis.{analysis_name}"
        module = __import__(module_name, fromlist=["run"])

        # Run the analysis with the provided arguments
        if hasattr(module, "run"):
            module.run(*(args or []))
        else:
            logger.error(
                f"Analysis module {analysis_name} does not have a 'run' function"
            )
    except ImportError as e:
        logger.error(f"Could not find analysis module: {analysis_name}")
        if debug:
            logger.exception(e)
    except Exception as e:
        logger.error(f"Error running analysis {analysis_name}")
        if debug:
            logger.exception(e)
