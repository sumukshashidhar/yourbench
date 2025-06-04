# yourbench/pipeline/citation_score_filtering.py
# =============================================================================
#
# This module implements a "citation score filtering" stage in YourBench.
# It computes fuzzy-overlap scores of each citation (in the lighteval dataset)
# with both the chunk text(s) and the ground-truth answer. The chunk-overlap
# is weighted more heavily.
#
# The resulting columns:
#   - 'answer_citation_score': the average fuzzy match ratio between citations
#       and the answer text.
#   - 'chunk_citation_score': the average fuzzy match ratio between citations
#       and the chunk text(s).
#   - 'citation_score': a final combined score that heavily weights chunk overlap.
#
# Usage:
#   1) In your config, add:
#         pipeline:
#           citation_score_filtering:
#             run: true
#   2) Add "citation_score_filtering" to the pipeline order in handler.py
#      after 'lighteval'.
#   3) This stage loads the 'lighteval' subset, computes new citation
#      columns, and saves it back to the same subset or a new one.
# =============================================================================

from typing import Any, Dict, List

from loguru import logger
from thefuzz import fuzz  # pip install thefuzz

from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset, replace_dataset_columns


def run(config: Dict[str, Any]) -> None:
    """
    Main pipeline entry point for the citation_score_filtering stage.

    This stage computes overlap-based 'citation scores' for each row in the
    lighteval dataset, where we measure how similar each citation is
    to the chunk text as well as to the ground truth answer.

    New columns added:
        - answer_citation_score
        - chunk_citation_score
        - citation_score

    The final dataset is saved under the subset name 'lighteval' (unless
    you change it below).
    """
    stage_cfg = config.get("pipeline", {}).get("citation_score_filtering", {})
    if not stage_cfg.get("run", False):
        logger.info("citation_score_filtering stage is disabled. Skipping.")
        return

    # 1) Load the lighteval dataset
    logger.info("Loading lighteval subset for citation score filtering...")
    try:
        lighteval_ds = custom_load_dataset(config=config, subset="lighteval")
        logger.info(f"Loaded lighteval subset with {len(lighteval_ds)} rows.")
    except Exception as e:
        logger.error(f"Could not load lighteval subset: {e}")
        return

    if len(lighteval_ds) == 0:
        logger.warning("lighteval dataset is empty; nothing to process.")
        return

    # 2) Prepare lists for new columns
    all_answer_citation_scores = []
    all_chunk_citation_scores = []
    all_final_scores = []

    # Weighting factors, adjustable to your preference
    # Larger alpha => chunk overlap matters more
    alpha = 0.7  # chunk overlap weight
    beta = 0.3  # answer overlap weight

    # 3) Iterate through each row and compute the scores
    for idx, row in enumerate(lighteval_ds):
        citations: List[str] = row.get("citations", [])
        chunks: List[str] = row.get("chunks", [])
        answer: str = row.get("ground_truth_answer", "")

        if not citations or (not chunks and not answer):
            # If no citations or no text to compare, just store zeros
            all_answer_citation_scores.append(0.0)
            all_chunk_citation_scores.append(0.0)
            all_final_scores.append(0.0)
            continue

        citation_count = len(citations)

        # --- Compute chunk-citation overlap ---
        # For each citation c, find the best partial ratio among all chunk texts
        # Then average across all citations
        chunk_scores = []
        for citation in citations:
            # For each citation, compare with each chunk to get the best overlap
            best_for_this_citation = 0.0
            for ch_text in chunks:
                score = fuzz.partial_ratio(citation, ch_text)
                if score > best_for_this_citation:
                    best_for_this_citation = score
            chunk_scores.append(best_for_this_citation)

        if chunk_scores:
            avg_chunk_score = sum(chunk_scores) / citation_count
        else:
            avg_chunk_score = 0.0

        # --- Compute answer-citation overlap ---
        # For each citation c, partial ratio with the answer
        # Then average across all citations
        ans_scores = []
        for citation in citations:
            score = fuzz.partial_ratio(citation, answer)
            ans_scores.append(score)

        if ans_scores:
            avg_ans_score = sum(ans_scores) / citation_count
        else:
            avg_ans_score = 0.0

        # Weighted final
        final_score = alpha * avg_chunk_score + beta * avg_ans_score

        all_answer_citation_scores.append(avg_ans_score)
        all_chunk_citation_scores.append(avg_chunk_score)
        all_final_scores.append(final_score)

    # 4) Add these new columns to the dataset
    # Use helper function to replace columns cleanly
    # Note: This doesn't preserve original column metadata, but for computed float scores
    # this is acceptable as type inference will correctly identify them as numeric
    columns_data = {
        "answer_citation_score": all_answer_citation_scores,
        "chunk_citation_score": all_chunk_citation_scores,
        "citation_score": all_final_scores,
    }

    lighteval_ds = replace_dataset_columns(lighteval_ds, columns_data)

    # 5) Save the updated dataset
    #    We reuse the "lighteval" subset name, but you could save it elsewhere if you prefer.
    logger.info("Saving updated lighteval dataset with new citation score columns...")
    custom_save_dataset(dataset=lighteval_ds, config=config, subset="lighteval")
    logger.success("citation_score_filtering stage completed successfully.")
