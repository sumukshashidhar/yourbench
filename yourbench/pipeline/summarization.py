# yourbench/pipeline/summarization.py

"""
Summarization Pipeline Stage
"""

from typing import Dict, Any, List, Optional
from loguru import logger
from datasets import Dataset
import evaluate
from random import uniform

# HF Evaluate metrics
_rouge = evaluate.load("rouge")
_bleu = evaluate.load("bleu")
_meteor = evaluate.load("meteor")
_bertscore = evaluate.load("bertscore")

from yourbench.utils.inference_engine import run_inference, InferenceCall
from yourbench.utils.prompts import SUMMARIZATION_USER_PROMPT
from yourbench.utils.saving_engine import save_dataset
from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from loguru import logger

def _safe_compute_bleu(predictions: List[str], references: List[List[str]]) -> float:
    """Safely compute BLEU score with error handling."""
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
    """Safely compute METEOR score with error handling."""
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
    """Safely compute ROUGE scores with error handling."""
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
    """Safely compute BERTScore with error handling."""
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


def run(config: Dict[str, Any]) -> None:
    """
    Summarization pipeline stage:
      1) Loads documents from HF dataset.
      2) Uses a summarization model to generate summaries.
      3) Computes corpus-level ROUGE, BLEU, METEOR, BERTScore.
      4) Stores them in 'quality_metrics' as a dictionary for each row.

    This version includes extra logging to investigate potential issues:
      - We check if <final_summary> was actually found in the raw output
      - We log raw output length, final summary length, and document length
      - We keep an eye out for empty or trivially short final summaries
      - We do not modify the actual summarization logic; just adding logs
    """

    stage_cfg = config["pipeline"]["summarization"]
    if not stage_cfg.get("run", False):
        logger.info("Summarization stage is disabled. Skipping.")
        return

    logger.info("Running summarization stage with additional diagnostic logs.")
    dataset = smart_load_dataset(stage_cfg["source_dataset_name"], config)
    logger.info("Loaded dataset with {} documents for summarization.", len(dataset))
    logger.info("Running summarization stage.")

    # Prepare inference calls
    documents: List[str] = dataset["document_text"]
    inference_calls = []
    for idx, doc_text in enumerate(documents):
        # Log the approximate length (to see if we might exceed model context)
        doc_len_chars = len(doc_text)
        logger.debug("Doc index {} => doc length ~{} chars", idx, doc_len_chars)

        user_msg = {"role": "user", "content": SUMMARIZATION_USER_PROMPT.format(document=doc_text)}
        inference_calls.append(InferenceCall(messages=[user_msg]))

    # Run inference in parallel
    logger.info("Sending {} summarization calls to inference engine.", len(inference_calls))
    responses_dict = run_inference(config, "summarization", inference_calls)

    # Get the model used for summarization
    # Use the first key from responses_dict as the model name
    # This is the safest approach since the inference engine returns results keyed by model name
    if not responses_dict:
        logger.error("No responses received from inference engine")
        return
        
    summ_model = list(responses_dict.keys())[0]
    logger.info("Using model for summarization: {}", summ_model)
    
    raw_summaries = responses_dict.get(summ_model, [])
    if len(raw_summaries) != len(documents):
        logger.warning(
            "Model '{}' returned {} summaries, but we have {} documents. "
            "Some mismatch or missing responses.",
            summ_model, len(raw_summaries), len(documents)
        )
        # Pad if fewer or trim if more
        while len(raw_summaries) < len(documents):
            raw_summaries.append("")
        if len(raw_summaries) > len(documents):
            raw_summaries = raw_summaries[: len(documents)]

    # Parse <final_summary> from each doc's output
    final_summaries = []
    for i in range(len(documents)):
        raw_resp = raw_summaries[i]
        # Additional logs to see raw length, check presence of <final_summary>
        logger.debug(
            "Summary response idx={}, raw length={} chars. Searching for <final_summary> tag.",
            i, len(raw_resp)
        )
        parsed = extract_content_from_xml_tags(raw_resp, "final_summary")

        if not parsed.strip():
            logger.warning(
                "No <final_summary> content found or empty for doc idx={}. Raw output might be malformed.",
                i
            )
        final_summaries.append(parsed.strip())

        # Log final summary length
        logger.debug(
            "Doc idx={} => final_summary length={} chars",
            i, len(final_summaries[-1])
        )

    # Compute corpus-level metrics
    all_preds = [summary if summary else "" for summary in final_summaries]
    # references for BLEU require list of lists
    all_refs = [[doc if doc else ""] for doc in documents]

    corpus_bleu = _safe_compute_bleu(all_preds, all_refs)
    corpus_meteor = _safe_compute_meteor(all_preds, [r[0] for r in all_refs])
    rouge_results = _safe_compute_rouge(all_preds, [r[0] for r in all_refs])
    corpus_bert_f1 = _safe_compute_bertscore(all_preds, [r[0] for r in all_refs])

    corpus_rouge1 = rouge_results["rouge1"]
    corpus_rouge2 = rouge_results["rouge2"]
    corpus_rougeL = rouge_results["rougeL"]

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

    all_metrics = []
    for _ in range(len(documents)):
        all_metrics.append({
            "rouge1_f1": corpus_rouge1,
            "rouge2_f1": corpus_rouge2,
            "rougeL_f1": corpus_rougeL,
            "bleu": corpus_bleu,
            "meteor": corpus_meteor,
            "bert_score_f1": corpus_bert_f1,
        })

    # Add new columns: raw_document_summary, document_summary, summarization_model, quality_metrics
    try:
        dataset = dataset.add_column("raw_document_summary", raw_summaries)
        dataset = dataset.add_column("document_summary", final_summaries)
        dataset = dataset.add_column("summarization_model", [summ_model] * len(dataset))
        dataset = dataset.add_column("quality_metrics", all_metrics)

        save_dataset(
            dataset=dataset,
            step_name="summarization",
            config=config,
            output_dataset_name=stage_cfg["output_dataset_name"]
        )
        logger.success("Summarization stage completed successfully with additional diagnostic logs.")
    except Exception as e:
        logger.error("Error adding columns or saving dataset in summarization: {}", str(e))
        logger.warning("Summarization stage encountered errors, but the pipeline will continue.")
