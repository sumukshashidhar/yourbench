# === Summarization Pipeline Stage ===

from typing import Dict, Any, List, Optional
from loguru import logger
from datasets import Dataset
import evaluate
from random import uniform
import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# HF Evaluate metrics
_rouge = evaluate.load("rouge")
_bleu = evaluate.load("bleu")
_meteor = evaluate.load("meteor")
_bertscore = evaluate.load("bertscore")

from yourbench.utils.inference_engine import run_inference, InferenceCall
from yourbench.utils.prompts import SUMMARIZATION_USER_PROMPT
from yourbench.utils.dataset_engine import save_dataset
from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags


def _safe_compute_bleu(predictions: List[str], references: List[List[str]]) -> float:
    """
    Safely compute BLEU score with error handling.
    
    :param predictions: List of prediction strings.
    :param references: List of list of reference strings.
    :return: BLEU score as a float.
    """
    try:
        safe_preds = [p if p is not None else "" for p in predictions]
        safe_refs = [[r if r is not None else "" for r in ref_list] for ref_list in references]

        if not safe_preds or not safe_refs or all(not p for p in safe_preds):
            logger.warning("Skipping BLEU computation due to empty inputs.")
            return 0.0

        result = _bleu.compute(predictions=safe_preds, references=safe_refs)
        return result.get("bleu", 0.0)
    except Exception as e:
        logger.error("Error computing BLEU score: {}", str(e))
        return 0.0


def _safe_compute_meteor(predictions: List[str], references: List[str]) -> float:
    """
    Safely compute METEOR score with error handling.
    
    :param predictions: List of prediction strings.
    :param references: List of reference strings.
    :return: METEOR score as a float.
    """
    try:
        safe_preds = [p if p is not None else "" for p in predictions]
        safe_refs = [r if r is not None else "" for r in references]

        if not safe_preds or not safe_refs or all(not p for p in safe_preds):
            logger.warning("Skipping METEOR computation due to empty inputs.")
            return 0.0

        result = _meteor.compute(predictions=safe_preds, references=safe_refs)
        return result.get("meteor", 0.0)
    except Exception as e:
        logger.error("Error computing METEOR score: {}", str(e))
        return 0.0


def _safe_compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Safely compute ROUGE scores with error handling.
    
    :param predictions: List of prediction strings.
    :param references: List of reference strings.
    :return: Dictionary with keys 'rouge1', 'rouge2', 'rougeL' representing respective scores.
    """
    try:
        safe_preds = [p if p is not None else "" for p in predictions]
        safe_refs = [r if r is not None else "" for r in references]

        if not safe_preds or not safe_refs or all(not p for p in safe_preds):
            logger.warning("Skipping ROUGE computation due to empty inputs.")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        result = _rouge.compute(predictions=safe_preds, references=safe_refs)
        return {
            "rouge1": result.get("rouge1", 0.0),
            "rouge2": result.get("rouge2", 0.0),
            "rougeL": result.get("rougeL", 0.0)
        }
    except Exception as e:
        logger.error("Error computing ROUGE scores: {}", str(e))
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def _safe_compute_bertscore(predictions: List[str], references: List[str]) -> float:
    """
    Safely compute BERTScore with error handling.
    
    :param predictions: List of prediction strings.
    :param references: List of reference strings.
    :return: The average F1 BERTScore as a float.
    """
    try:
        safe_preds = [p if p is not None else "" for p in predictions]
        safe_refs = [r if r is not None else "" for r in references]

        if not safe_preds or not safe_refs or all(not p for p in safe_preds):
            logger.warning("Skipping BERTScore computation due to empty inputs.")
            return 0.0

        result = _bertscore.compute(
            predictions=safe_preds,
            references=safe_refs,
            model_type="bert-base-uncased"
        )

        f1_scores = result.get("f1", [])
        if not f1_scores:
            return 0.0
        return sum(f1_scores) / len(f1_scores)
    except Exception as e:
        logger.error("Error computing BERTScore: {}", str(e))
        return 0.0


def _run_inference_with_timeout(
    config: Dict[str, Any],
    inference_calls: List[InferenceCall],
    stage: str,
    timeout_seconds: float
) -> Optional[Dict[str, List[str]]]:
    """
    Wrapper to run inference with a configurable timeout.

    :param config: The overall pipeline configuration dictionary.
    :param inference_calls: A list of InferenceCall objects to pass to run_inference.
    :param stage: The stage name (e.g., "summarization").
    :param timeout_seconds: Timeout in seconds after which inference is considered stalled.
    :return: A dictionary of responses from run_inference, or None if timed out or error.
    """
    logger.info("Attempting to run inference with a timeout of {} seconds...", timeout_seconds)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_inference, config, stage, inference_calls)
        try:
            responses = future.result(timeout=timeout_seconds)
            return responses
        except FuturesTimeoutError:
            logger.error("Inference timed out after {} seconds.", timeout_seconds)
        except Exception as exc:
            logger.error("Error during inference: {}", str(exc))
    return None

def duplicate_rows(dataset, num_duplicates=1):
    # Return the example as a list repeated num_duplicates times
    return {k: [v] * num_duplicates for k, v in dataset.items()}

def run(config: Dict[str, Any]) -> None:
    """
    Summarization pipeline stage:

    1) Loads documents from HF dataset.
    2) Uses a summarization model to generate summaries with a timeout to prevent hangs.
    3) Computes corpus-level ROUGE, BLEU, METEOR, BERTScore metrics.
    4) Stores them in 'quality_metrics' as a dictionary for each row.
    
    This version includes extra defensive programming:
      - Thorough try/except blocks and logging at key steps.
      - Timeouts around inference calls to avoid indefinite hanging.
      - If a blank summary is detected, it is replaced with a descriptive string.
    """

    # === Check Stage Configuration ===
    stage_cfg: Dict[str, Any] = config["pipeline"].get("summarization", {})
    if not stage_cfg.get("run", False):
        logger.info("Summarization stage is disabled. Skipping.")
        return

    logger.info("Running summarization stage with additional diagnostic logs.")

    # === Load Dataset ===
    try:
        dataset: Dataset = smart_load_dataset(stage_cfg["source_dataset_name"], config)
        logger.info("Loaded dataset with {} documents for summarization.", len(dataset))
    except Exception as e:
        logger.error("Failed to load dataset '{}': {}", stage_cfg.get("source_dataset_name"), str(e))
        logger.warning("Summarization stage cannot proceed due to dataset load failure.")
        return

    # === Prepare Inference Calls ===
    try:
        documents: List[str] = dataset["document_text"]
    except KeyError:
        logger.error("Dataset does not contain 'document_text' column. Cannot proceed.")
        return
    except Exception as e:
        logger.error("Unexpected error accessing 'document_text' column: {}", str(e))
        return

    inference_calls: List[InferenceCall] = []
    for idx, doc_text in enumerate(documents):
        doc_len_chars: int = len(doc_text)
        logger.debug("Doc index {} => doc length ~{} chars", idx, doc_len_chars)
        user_msg: Dict[str, str] = {
            "role": "user",
            "content": SUMMARIZATION_USER_PROMPT.format(document=doc_text)
        }
        inference_calls.append(InferenceCall(messages=[user_msg]))

    logger.info("Prepared {} summarization calls.", len(inference_calls))

    # === Run Inference with Timeout ===
    timeout_seconds: float = stage_cfg.get("timeout_seconds", 300.0)
    responses_dict: Optional[Dict[str, List[str]]] = _run_inference_with_timeout(
        config=config,
        inference_calls=inference_calls,
        stage="summarization",
        timeout_seconds=timeout_seconds
    )

    if responses_dict is None:
        logger.error("Inference did not complete successfully. No summaries generated.")
        return

    if not responses_dict:
        logger.error("No responses received from inference engine.")
        return

    # === Choose Summarization Model Key ===
    try:
        summ_model: str = list(responses_dict.keys())[0]
    except IndexError:
        logger.error("Response dictionary keys are empty; cannot determine summarization model.")
        return

    logger.info("Using model for summarization: {}", summ_model)

    # === Retrieve Summaries from Dictionary ===
    raw_summaries: List[str] = responses_dict.get(summ_model, [])
    if len(raw_summaries) != len(documents):
        logger.warning(
            "Model '{}' returned {} summaries, but we have {} documents. Potential mismatch.",
            summ_model, len(raw_summaries), len(documents)
        )
        # Harmonize the lengths
        while len(raw_summaries) < len(documents):
            raw_summaries.append("")
        if len(raw_summaries) > len(documents):
            raw_summaries = raw_summaries[: len(documents)]

    # === Parse Final Summaries ===
    final_summaries: List[str] = []
    for i, raw_resp in enumerate(raw_summaries):
        logger.debug(
            "Summary response idx={}, raw length={} chars. Searching for <final_summary> tag.",
            i, len(raw_resp)
        )
        parsed: str = ""
        try:
            parsed = extract_content_from_xml_tags(raw_resp, "final_summary")
        except Exception as e:
            logger.error("Error parsing <final_summary> for doc idx={}: {}", i, str(e))

        parsed_stripped = parsed.strip()
        if not parsed_stripped:
            logger.warning(
                "No <final_summary> content found or empty for doc idx={}. Raw output might be malformed.",
                i
            )
            final_summaries.append("There is no summary available for this document")
        else:
            final_summaries.append(parsed_stripped)

        logger.debug(
            "Doc idx={} => final_summary length={} chars",
            i, len(final_summaries[-1])
        )

    # === Compute Corpus-Level Metrics ===
    if debug_mode:
        try:
            all_preds: List[str] = [summary if summary else "" for summary in final_summaries]
            all_refs: List[List[str]] = [[doc if doc else ""] for doc in documents]

            corpus_bleu: float = _safe_compute_bleu(all_preds, all_refs)
            corpus_meteor: float = _safe_compute_meteor(all_preds, [r[0] for r in all_refs])
            rouge_results: Dict[str, float] = _safe_compute_rouge(all_preds, [r[0] for r in all_refs])
            corpus_bert_f1: float = _safe_compute_bertscore(all_preds, [r[0] for r in all_refs])

            corpus_rouge1: float = rouge_results["rouge1"]
            corpus_rouge2: float = rouge_results["rouge2"]
            corpus_rougeL: float = rouge_results["rougeL"]

            logger.info(
                "Corpus-level summarization metrics:\n"
                "  BLEU:  {:.4f}\n"
                "  METEOR: {:.4f}\n"
                "  ROUGE1: {:.4f}, ROUGE2: {:.4f}, ROUGEL: {:.4f}\n"
                "  BERTScore-F1: {:.4f}",
                corpus_bleu,
                corpus_meteor,
                corpus_rouge1,
                corpus_rouge2,
                corpus_rougeL,
                corpus_bert_f1
            )
        except Exception as e:
            logger.error("Error computing corpus-level metrics: {}", str(e))
            # Continue execution, but metrics are not reliable
            corpus_bleu, corpus_meteor, corpus_rouge1, corpus_rouge2, corpus_rougeL, corpus_bert_f1 = 0, 0, 0, 0, 0, 0

        # === Prepare Per-Document Metrics ===
        all_metrics: List[Dict[str, float]] = []
        for _ in range(len(documents)):
            all_metrics.append({
                "rouge1_f1": corpus_rouge1,
                "rouge2_f1": corpus_rouge2,
                "rougeL_f1": corpus_rougeL,
                "bleu": corpus_bleu,
                "meteor": corpus_meteor,
                "bert_score_f1": corpus_bert_f1
            })
    
    all_metrics: List[Dict[str, float]] = []
    # === Add Columns and Save Dataset ===
    try:
        dataset = dataset.add_column("raw_document_summary", raw_summaries)
    except Exception as e:
        logger.error("Error adding 'raw_document_summary' column: {}", str(e))

    try:
        dataset = dataset.add_column("document_summary", final_summaries)
    except Exception as e:
        logger.error("Error adding 'document_summary' column: {}", str(e))

    try:
        dataset = dataset.add_column("summarization_model", [summ_model] * len(dataset))
    except Exception as e:
        logger.error("Error adding 'summarization_model' column: {}", str(e))

    try:
        dataset = dataset.add_column("quality_metrics", all_metrics)
    except Exception as e:
        logger.error("Error adding 'quality_metrics' column: {}", str(e))

    # === Save Dataset ===
    try:
        save_dataset(
            dataset=dataset,
            step_name="summarization",
            config=config,
            output_dataset_name=stage_cfg.get("output_dataset_name", None),
            split=stage_cfg.get("dataset_split", "train")
        )
        logger.success("Summarization stage completed successfully with enhanced defensive programming.")
    except Exception as e:
        logger.error("Error saving dataset in summarization stage: {}", str(e))
        logger.warning("Summarization stage encountered errors, but the pipeline will continue.")
