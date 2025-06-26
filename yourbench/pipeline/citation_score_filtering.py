"""Compute overlap based citation scores for the lighteval dataset."""
from typing import Any, Sequence
from dataclasses import dataclass

from loguru import logger
from thefuzz import fuzz

from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset


@dataclass(slots=True)
class StageConfig:
    run: bool = False
    subset: str = "lighteval"
    alpha: float = 0.7
    beta: float = 0.3


def _get_stage_config(config: dict[str, Any]) -> StageConfig:
    return StageConfig(**config.get("pipeline", {}).get("citation_score_filtering", {}))


@dataclass(slots=True)
class CitationScoreCalculator:
    alpha: float
    beta: float

    def _ratio(self, a: str, b: str) -> int:
        return fuzz.partial_ratio(a, b)

    def compute(self, citations: Sequence[str], chunks: Sequence[str], answer: str) -> tuple[float, float, float]:
        if not citations or (not chunks and not answer):
            return 0.0, 0.0, 0.0

        citation_count = len(citations)
        chunk_scores = [max((self._ratio(c, ch) for ch in chunks), default=0) for c in citations]
        ans_scores = [self._ratio(c, answer) for c in citations]

        avg_chunk = sum(chunk_scores) / citation_count
        avg_ans = sum(ans_scores) / citation_count
        final = self.alpha * avg_chunk + self.beta * avg_ans

        return avg_ans, avg_chunk, final


def run(config: dict[str, Any]) -> None:
    """Entry point for the citation score filtering stage."""
    stage_cfg = _get_stage_config(config)
    if not stage_cfg.run:
        logger.info("citation_score_filtering stage is disabled. Skipping.")
        return

    logger.info(f"Loading '{stage_cfg.subset}' subset for citation score filtering...")
    try:
        dataset = custom_load_dataset(config=config, subset=stage_cfg.subset)
    except Exception as e:
        logger.exception(f"Could not load subset '{stage_cfg.subset}': {e}")
        return

    if len(dataset) == 0:
        logger.warning("Dataset is empty; nothing to process.")
        return

    logger.debug(f"Computing citation scores for {len(dataset)} rows")
    scorer = CitationScoreCalculator(stage_cfg.alpha, stage_cfg.beta)

    answer_scores: list[float] = []
    chunk_scores: list[float] = []
    final_scores: list[float] = []

    for row in dataset:
        ans, chunk, final = scorer.compute(
            citations=row.get("citations", []),
            chunks=row.get("chunks", []),
            answer=row.get("ground_truth_answer", ""),
        )
        answer_scores.append(ans)
        chunk_scores.append(chunk)
        final_scores.append(final)

    dataset = dataset.add_column("answer_citation_score", answer_scores)
    dataset = dataset.add_column("chunk_citation_score", chunk_scores)
    dataset = dataset.add_column("citation_score", final_scores)

    logger.info("Saving updated dataset with new citation score columns...")
    custom_save_dataset(dataset=dataset, config=config, subset=stage_cfg.subset)
    logger.success("citation_score_filtering stage completed successfully.")