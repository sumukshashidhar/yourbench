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


def run(config: Dict[str, Any]) -> None:
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

    The result is saved under the subset name specified in config["pipeline"]["lighteval"]["output_subset"].

    Args:
        config (Dict[str, Any]):
            The entire pipeline configuration. Must have the following fields:
            - config["pipeline"]["lighteval"]["run"] (bool): Whether to run this stage.
            - config["pipeline"]["lighteval"]["single_shot_subset"] (str): Subset containing single-shot questions.
            - config["pipeline"]["lighteval"]["multi_hop_subset"]   (str): Subset containing multi-hop questions.
            - config["pipeline"]["lighteval"]["chunked_subset"]     (str): Subset containing chunked documents.
            - config["pipeline"]["lighteval"]["output_subset"]      (str): Subset name for saving final dataset.

    Returns:
        None. The merged dataset is saved to disk or HF Hub as configured.
    """
    stage_cfg = config.get("pipeline", {}).get("lighteval", {})
    if not stage_cfg.get("run", False):
        logger.info("lighteval stage is disabled. Skipping.")
        return

    logger.info("Saving lighteval compatible dataset")

    # ----------------------------------------
    # 2) Load datasets
    # ----------------------------------------
    try:
        single_shot_ds = custom_load_dataset(config=config, subset="single_shot_questions")
        logger.info(f"Loaded single-shot Q subset single_shot_questions with {len(single_shot_ds)} rows.")
    except Exception as e:
        logger.warning(f"Could not load single-shot subset single_shot_questions: {e}")
        single_shot_ds = Dataset.from_dict({})  # empty fallback

    try:
        multi_hop_ds = custom_load_dataset(config=config, subset="multi_hop_questions")
        logger.info(f"Loaded multi-hop Q subset multi_hop_subset with {len(multi_hop_ds)} rows.")
    except Exception as e:
        logger.warning(f"Could not load multi-hop subset multi_hop_subset: {e}")
        multi_hop_ds = Dataset.from_dict({})  # empty fallback

    try:
        chunked_ds = custom_load_dataset(config=config, subset="chunked")
        logger.info(f"Loaded chunked subset with {len(chunked_ds)} rows.")
    except Exception as e:
        logger.error(f"Could not load chunked subset: {e}")
        logger.warning("Cannot proceed with chunk text or document text. They will be empty.")
        chunked_ds = Dataset.from_dict({})  # empty fallback

    try:
        summarized_ds = custom_load_dataset(config=config, subset="summarized")
        logger.info(f"Loaded summarized subset with {len(summarized_ds)} rows.")
    except Exception as e:
        logger.error(f"Could not load summarized subset: {e}")
        summarized_ds = Dataset.from_dict({})  # empty fallback

    if len(single_shot_ds) == 0 and len(multi_hop_ds) == 0:
        logger.error("No data in single-shot or multi-hop datasets. Exiting.")
        return

    # ----------------------------------------
    # 3) Prepare lookups from chunked dataset
    # ----------------------------------------
    # We'll store: doc_id -> (document_text, chunk_id -> chunk_text).
    # chunked_ds typically has the following columns:
    #  - document_id (str)
    #  - document_text (str)
    #  - chunks (list of dicts with {chunk_id, chunk_text})
    # Possibly also "multihop_chunks" but we only need single-hop "chunks".
    doc_meta_map = {}
    for row in chunked_ds:
        doc_id = row.get("document_id", "")
        doc_text = row.get("document_text", "")
        # Build a map from chunk_id to chunk_text for single-hop lookups
        chunk_dict = {}
        for chunk_entry in row.get("chunks", []):
            cid = chunk_entry.get("chunk_id", "")
            ctext = chunk_entry.get("chunk_text", "")
            chunk_dict[cid] = ctext
        doc_meta_map[doc_id] = {"document_text": doc_text, "chunks_map": chunk_dict}

    for row in summarized_ds:
        doc_id = row.get("document_id", "")
        if doc_id in doc_meta_map:
            doc_meta_map[doc_id].update({"document_summary": row.get("document_summary")})

    # ----------------------------------------
    # 4) Helper functions to transform a row
    # ----------------------------------------
    def make_single_shot_record(row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single-shot question row into a standardized dictionary
        for the final lighteval dataset.
        """
        doc_id: str = row.get("document_id", "")
        chunk_id: str = row.get("chunk_id", "")
        # ground_truth is row["self_answer"]
        # question_category is row["self_assessed_question_type"]
        # question => row["question"], etc.

        # Grab doc meta
        doc_meta = doc_meta_map.get(doc_id, {})
        doc_text = doc_meta.get("document_text", "")
        doc_summary = doc_meta.get("document_summary", "")
        chunk_text_map = doc_meta.get("chunks_map", {})
        # chunk text is chunk_text_map[chunk_id] if it exists
        chunk_text = chunk_text_map.get(chunk_id, "")

        # if multiple choice question convert to number
        gold = row.get("self_answer", "")
        if row.get("choices"):
            gold = [ord(gold) - ord("A")]

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
        chunk_text_map = doc_meta.get("chunks_map", {})

        # Gather chunk_text for each chunk_id
        chunk_texts = []
        for cid in chunk_ids:
            ctxt = chunk_text_map.get(cid, "")
            if ctxt:
                chunk_texts.append(ctxt)

        # if multiple choice question convert to number
        gold = row.get("self_answer", "")
        if row.get("choices"):
            gold = [ord(gold) - ord("A")]

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

    # ----------------------------------------
    # 5) Convert each dataset to final records
    # ----------------------------------------
    combined_records = []

    for row in single_shot_ds:
        record = make_single_shot_record(row)
        combined_records.append(record)

    for row in multi_hop_ds:
        record = make_multi_hop_record(row)
        combined_records.append(record)

    if not combined_records:
        logger.warning("No final records to merge in lighteval. Exiting.")
        return

    # ----------------------------------------
    # 6) Create a Hugging Face Dataset
    # ----------------------------------------
    logger.info(f"Assembling final dataset with {len(combined_records)} rows.")
    try:
        # Convert to column-wise dict for HF Dataset
        col_names = list(combined_records[0].keys())
        final_dict = {c: [] for c in col_names}
        for rec in combined_records:
            for c in col_names:
                final_dict[c].append(rec[c])
        final_ds = Dataset.from_dict(final_dict)

    except Exception as ds_error:
        logger.error(f"Failed to create final dataset object: {ds_error}")
        return

    # ----------------------------------------
    # 7) Save dataset
    # ----------------------------------------
    custom_save_dataset(dataset=final_ds, config=config, subset="lighteval")
    logger.success("Lighteval dataset saved successfully.")
