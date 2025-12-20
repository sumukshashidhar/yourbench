"""Cross-document question generation pipeline stage."""

from yourbench.pipeline.question_generation import run_cross_document


def run(config) -> None:
    """Run cross-document question generation."""
    run_cross_document(config)
