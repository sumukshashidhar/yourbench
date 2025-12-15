"""Context managers and utilities for structured logging."""

import time
from typing import Any, Generator
from contextlib import contextmanager

from loguru import logger


@contextmanager
def log_stage(stage_name: str, **extra_context: Any) -> Generator[None, None, None]:
    """Context manager to track pipeline stages.

    Args:
        stage_name: Name of the stage (e.g., 'ingestion', 'summarization')
        **extra_context: Additional context to include in logs
    """
    start_time = time.time()

    # Bind stage context to all logs in this scope
    with logger.contextualize(stage=stage_name, **extra_context):
        logger.info(f"Starting {stage_name}")
        try:
            yield
            elapsed = time.time() - start_time
            logger.success(f"Completed {stage_name} in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed {stage_name} after {elapsed:.2f}s: {e}")
            raise


@contextmanager
def log_step(step_name: str, **extra_context: Any) -> Generator[None, None, None]:
    """Nested context for sub-steps within a stage.

    Args:
        step_name: Name of the step
        **extra_context: Additional context to include in logs
    """
    with logger.contextualize(step=step_name, **extra_context):
        logger.debug(f"Starting step: {step_name}")
        try:
            yield
            logger.debug(f"Completed step: {step_name}")
        except Exception as e:
            logger.warning(f"Step {step_name} failed: {e}")
            raise


def log_progress(current: int, total: int, item_name: str = "item") -> None:
    """Log progress for batch operations.

    Args:
        current: Current item number
        total: Total number of items
        item_name: Name of the item being processed
    """
    percentage = (current / total) * 100 if total > 0 else 0
    logger.debug(f"Processing {item_name} {current}/{total} ({percentage:.1f}%)")
