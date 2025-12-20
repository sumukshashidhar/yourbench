"""Single-shot question generation pipeline stage."""

from yourbench.pipeline.question_generation._core import run_single_shot


def run(config) -> None:
    """Run single-shot question generation."""
    run_single_shot(config)
