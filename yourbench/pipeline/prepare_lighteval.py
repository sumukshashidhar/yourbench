"""
Lightweight Evaluation Dataset Assembly Stage

Overview:
---------
Combines single-shot and multi-hop question datasets into a unified "light evaluation"
dataset suitable for quick checking or downstream evaluations. This stage fetches
the necessary metadata (document text, chunk text, etc.) from the chunked dataset
to populate a final dataset with the following columns:

1) question                (str)  - The actual question text.
2) ground_truth_answer     (str)  - The supposed correct answer to the question.
3) question_category       (str)  - A label or taxonomy describing the question type.
4) kind                    (str)  - Either "single_shot" or "multi_hop".
5) estimated_difficulty    (int)  - Estimated difficulty (1-10).
6) citations               (List[str]) - List of source citations or references.
7) document_id             (str)  - The ID of the document from which the question is derived.
8) chunk_ids               (List[str]) - The chunk ID(s) used in forming the question.
9) question_generating_model (str) - The HF model ID that generated this question.
10) chunks                 (List[str]) - The actual chunk text(s) the question came from.
11) document               (str)  - The entire document text.

Configuration Example:
----------------------
pipeline:
  lighteval:
    run: true
    single_shot_subset: single_shot_questions_deduplicated
    multi_hop_subset: multi_hop_questions_deduplicated
    chunked_subset: chunked_documents
    output_subset: lighteval

Usage:
------
1. Load single-shot and multi-hop question subsets.
2. Merge them into a single dataset, marking 'kind' as "single_shot" or "multi_hop."
3. For each question row, look up the relevant chunks in the chunked dataset to
   populate 'chunks' and the full 'document' text.
4. Save final dataset to HF or local path as configured.
"""

from typing import Any, Dict, List

from loguru import logger

from datasets import Dataset
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
def run(config) -> None:
    """
    Main entry point for the lighteval pipeline stage.

    This stage merges single-shot and multi-hop question datasets with chunked
    document metadata into a unified "light evaluation" dataset containing the columns:

      1. question
      2. ground_truth_answer
      3. question_category
      4. kind
      5. estimated_difficulty
      6. citations
      7. document_id
      8. chunk_ids
      9. question_generating_model
      10. chunks
      11. document

    The result is saved under the subset name specified in config.pipeline.lighteval.

    Args:
        config (YourbenchConfig): The entire pipeline configuration.

    Returns:
        None. The merged dataset is saved to disk or HF Hub as configured.
    """
    stage_cfg = config.pipeline.prepare_lighteval

    logger.info("Saving lighteval compatible dataset")

    # Use configurable subset names with fallbacks
    single_shot_subset = getattr(stage_cfg, "single_shot_subset", "single_shot_questions")
    multi_hop_subset = getattr(stage_cfg, "multi_hop_subset", "multi_hop_questions")
    cross_doc_subset = getattr(stage_cfg, "cross_doc_subset", "cross_document_questions")
    chunked_subset = getattr(stage_cfg, "chunked_subset", "chunked")
    summarized_subset = getattr(stage_cfg, "summarized_subset", "summarized")
    output_subset = getattr(stage_cfg, "output_subset", "prepared_lighteval")

    # Load datasets
    try:
        single_shot_ds = custom_load_dataset(config=config, subset=single_shot_subset)
        logger.info(f"Loaded single-shot Q subset with {len(single_shot_ds)} rows.")
    except Exception as e:
        logger.warning(f"Could not load single-shot subset: {e}")
        single_shot_ds = Dataset.from_dict({})

    try:
        multi_hop_ds = custom_load_dataset(config=config, subset=multi_hop_subset)
        logger.info(f"Loaded multi-hop Q subset with {len(multi_hop_ds)} rows.")
    except Exception as e:
        logger.warning(f"Could not load multi-hop subset: {e}")
        multi_hop_ds = Dataset.from_dict({})

    try:
        cross_doc_ds = custom_load_dataset(config=config, subset=cross_doc_subset)
        logger.info(f"Loaded cross-document Q subset with {len(cross_doc_ds)} rows.")
    except Exception as e:
        logger.warning(f"Could not load cross-document subset: {e}")
        cross_doc_ds = Dataset.from_dict({})  # empty fallback

    try:
        chunked_ds = custom_load_dataset(config=config, subset=chunked_subset)
        logger.info(f"Loaded chunked subset with {len(chunked_ds)} rows.")
    except Exception as e:
        logger.error(f"Could not load chunked subset: {e}")
        logger.warning("Cannot proceed with chunk text or document text. They will be empty.")
        chunked_ds = Dataset.from_dict({})  # empty fallback

    try:
        summarized_ds = custom_load_dataset(config=config, subset=summarized_subset)
        logger.info(f"Loaded summarized subset with {len(summarized_ds)} rows.")
    except Exception as e:
        logger.error(f"Could not load summarized subset: {e}")
        summarized_ds = Dataset.from_dict({})

    if len(single_shot_ds) == 0 and len(multi_hop_ds) == 0 and len(cross_doc_ds) == 0:
        logger.warning(
            "No data in single-shot, multi-hop, or cross-document datasets. Creating empty prepared_lighteval subset."
        )
        # Create empty dataset with the expected schema
        empty_dataset = Dataset.from_dict({
            "task_id": [],
            "question": [],
            "answer": [],
            "choices": [],
            "gold": [],
            "question_type": [],
            "document_id": [],
            "document_text": [],
            "document_summary": [],
            "chunk_id": [],
            "chunk_text": [],
            "related_chunks": [],
            "type": [],
        })
        custom_save_dataset(
            empty_dataset, config=config, subset="prepared_lighteval", push_to_hub=config.hf_configuration.push_to_hub
        )
        return

    # Prepare lookups from chunked dataset
    doc_meta_map = {}
    for row in chunked_ds:
        doc_id = row.get("document_id", "")
        doc_text = row.get("document_text", "")
        # Build a map from chunk_id to chunk_text for single-hop lookups
        chunk_dict = {chunk.get("chunk_id", ""): chunk.get("chunk_text", "") for chunk in row.get("chunks", [])}
        doc_meta_map[doc_id] = {"document_text": doc_text, "chunks_map": chunk_dict}

    for row in summarized_ds:
        doc_id = row.get("document_id", "")
        if doc_id in doc_meta_map:
            doc_meta_map[doc_id].update({"document_summary": row.get("document_summary")})

    # Helper functions to transform a row
    def make_single_shot_record(row: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = row.get("document_id", "")
        chunk_id = row.get("chunk_id", "")

        # Grab doc meta
        doc_meta = doc_meta_map.get(doc_id, {})
        doc_text = doc_meta.get("document_text", "")
        doc_summary = doc_meta.get("document_summary", "")
        chunk_text = doc_meta.get("chunks_map", {}).get(chunk_id, "")

        # if multiple choice question convert to number
        gold = row.get("self_answer", "")
        if not gold:
            logger.warning("Row has empty answer line")

        stage_cfg_local = config.pipeline.single_shot_question_generation
        gold = (
            [ord(gold) - ord("A")]
            if getattr(stage_cfg_local, "question_mode", None) == "multi-choice" and gold
            else [0]
            if getattr(stage_cfg_local, "question_mode", None) == "multi-choice"
            else [gold]
        )

        return {
            "question": row.get("question", ""),
            "additional_instructions": row.get("additional_instructions", ""),
            "ground_truth_answer": row.get("self_answer", ""),
            "gold": gold,
            "choices": row.get("choices", []),
            "question_category": row.get("self_assessed_question_type", "unknown"),
            "kind": "single_shot",
            "estimated_difficulty": row.get("estimated_difficulty", 5),
            "citations": row.get("citations", []),
            "document_id": doc_id,
            "chunk_ids": [chunk_id] if chunk_id else [],
            "question_generating_model": row.get("generating_model", ""),
            "chunks": [chunk_text] if chunk_text else [],
            "document": doc_text,
            "document_summary": doc_summary,
        }

    def make_multi_hop_record(row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a multi-hop question row into a standardized dictionary
        for the final lighteval dataset.
        """
        doc_id: str = row.get("document_id", "")
        # e.g. row["source_chunk_ids"]: List[str]
        chunk_ids: List[str] = row.get("source_chunk_ids", [])
        doc_meta = doc_meta_map.get(doc_id, {})
        doc_text = doc_meta.get("document_text", "")
        doc_summary = doc_meta.get("document_summary", "")
        chunk_texts = [doc_meta.get("chunks_map", {}).get(cid, "") for cid in chunk_ids if cid]

        # if multiple choice question convert to number
        gold = row.get("self_answer", "")
        if not gold:
            logger.warning("Row has empty answer line")

        stage_cfg_local = config.pipeline.multi_hop_question_generation
        gold = (
            [ord(gold) - ord("A")]
            if getattr(stage_cfg_local, "question_mode", None) == "multi-choice" and gold
            else [0]
            if getattr(stage_cfg_local, "question_mode", None) == "multi-choice"
            else [gold]
        )

        return {
            "question": row.get("question", ""),
            "additional_instructions": row.get("additional_instructions", ""),
            "ground_truth_answer": row.get("self_answer", ""),
            "gold": gold,
            "choices": row.get("choices", []),
            "question_category": row.get("self_assessed_question_type", "unknown"),
            "kind": "multi_hop",
            "estimated_difficulty": row.get("estimated_difficulty", 5),
            "citations": row.get("citations", []),
            "document_id": doc_id,
            "chunk_ids": chunk_ids,
            "question_generating_model": row.get("generating_model", ""),
            "chunks": chunk_texts,
            "document": doc_text,
            "document_summary": doc_summary,
        }

    def make_cross_document_record(row: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = row.get("document_id", "")
        chunk_ids = row.get("source_chunk_ids", [])
        doc_meta = doc_meta_map.get(doc_id, {})
        doc_text = doc_meta.get("document_text", "")
        doc_summary = doc_meta.get("document_summary", "")
        chunk_texts = [doc_meta.get("chunks_map", {}).get(cid, "") for cid in chunk_ids if cid]

        gold = row.get("self_answer", "")
        if not gold:
            logger.warning("Row has empty answer line")

        stage_cfg_local = config.pipeline.cross_document_question_generation
        gold = (
            [ord(gold) - ord("A")]
            if getattr(stage_cfg_local, "question_mode", None) == "multi-choice" and gold
            else [0]
            if getattr(stage_cfg_local, "question_mode", None) == "multi-choice"
            else [gold]
        )

        return {
            "question": row.get("question", ""),
            "additional_instructions": row.get("additional_instructions", ""),
            "ground_truth_answer": row.get("self_answer", ""),
            "gold": gold,
            "choices": row.get("choices", []),
            "question_category": row.get("self_assessed_question_type", "unknown"),
            "kind": "cross_document",
            "estimated_difficulty": row.get("estimated_difficulty", 5),
            "citations": row.get("citations", []),
            "document_id": doc_id,
            "chunk_ids": chunk_ids,
            "question_generating_model": row.get("generating_model", ""),
            "chunks": chunk_texts,
            "document": doc_text,
            "document_summary": doc_summary,
        }

    # Final combination
    combined_records = (
        [make_single_shot_record(row) for row in single_shot_ds]
        + [make_multi_hop_record(row) for row in multi_hop_ds]
        + [make_cross_document_record(row) for row in cross_doc_ds]
    )

    if not combined_records:
        logger.warning("No final records to merge in lighteval. Exiting.")
        return

    # Create a Hugging Face Dataset
    logger.info(f"Assembling final dataset with {len(combined_records)} rows.")
    try:
        final_ds = Dataset.from_list(combined_records)
    except Exception as ds_error:
        logger.exception("Failed to create final dataset object")
        return

    # Save dataset
    custom_save_dataset(
        dataset=final_ds, config=config, subset=output_subset, push_to_hub=config.hf_configuration.push_to_hub
    )
    logger.success("Prepared Lighteval dataset saved successfully.")
