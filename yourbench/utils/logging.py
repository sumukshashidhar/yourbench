import sys

from loguru import logger


def setup_logger(debug: bool = False):
    """Configure logger with appropriate format and level.
    Args:
        debug (bool): If True, show DEBUG level messages. Otherwise, show INFO and above.
    """
    logger.remove()  # Remove default handler

    # Define log format
    log_format = (
        "<dim>{time:HH:mm:ss}</dim> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )

    # Set appropriate level based on debug flag
    level = "DEBUG" if debug else "INFO"

    # Add handler with format
    logger.add(sys.stderr, format=log_format, level=level)
