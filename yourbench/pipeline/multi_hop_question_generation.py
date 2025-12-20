"""Multi-hop question generation pipeline stage."""

from yourbench.pipeline.question_generation import run_multi_hop


def run(config) -> None:
    """Run multi-hop question generation."""
    run_multi_hop(config)
