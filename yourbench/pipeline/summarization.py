# yourbench/pipeline/summarization.py

from typing import Dict, Any, List
from loguru import logger
from datasets import Dataset
import evaluate

# Load the metrics
_rouge = evaluate.load("rouge")
_bleu = evaluate.load("bleu")
_meteor = evaluate.load("meteor")
_bertscore = evaluate.load("bertscore")

from yourbench.utils.inference_engine import run_inference, InferenceCall
from yourbench.utils.prompts import SUMMARIZATION_USER_PROMPT
from yourbench.utils.saving_engine import save_dataset
from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags

def run(config: Dict[str, Any]) -> None:
    """
    Summarization pipeline stage that:
      1. Loads a dataset of documents.
      2. Generates summaries with a chosen model.
      3. Computes 5+ standard metrics (ROUGE, BLEU, METEOR, BERTScore) 
         comparing the summary vs. the original document text.
      4. Saves the dataset with new columns:
         "raw_document_summary", "document_summary", "summarization_model", and "quality_metrics".
    """

    summary_cfg = config["pipeline"]["summarization"]
    if not summary_cfg.get("run", False):
        logger.info("Summarization stage disabled. Skipping.")
        return

    logger.info("Running summarization stage.")
    dataset = smart_load_dataset(summary_cfg["source_dataset_name"], config)
    logger.info("Loaded dataset with {} documents.", len(dataset))

    documents: List[str] = dataset["document_text"]

    # 1) Build the inference calls
    inference_calls = []
    for doc_text in documents:
        # Insert doc_text into the user prompt
        user_msg = {"role": "user", "content": SUMMARIZATION_USER_PROMPT.format(document=doc_text)}
        inference_calls.append(InferenceCall(messages=[user_msg]))

    logger.info("Sending {} summarization calls to inference engine.", len(inference_calls))
    responses_dict = run_inference(config, "summarization", inference_calls)

    # We assume a single model name under model_roles["summarization"][0]
    summ_model = config["model_roles"]["summarization"][0]
    raw_summaries = responses_dict.get(summ_model, [])

    # Fallback if fewer summaries returned than documents
    if len(raw_summaries) != len(documents):
        logger.warning(
            "Model %r returned %d summaries, but we expected %d documents.",
            summ_model, len(raw_summaries), len(documents)
        )

    # 2) Parse <final_summary> from raw model responses
    final_summaries = []
    for i in range(len(documents)):
        raw = raw_summaries[i] if i < len(raw_summaries) else ""
        parsed_summary = extract_content_from_xml_tags(raw, "final_summary")
        final_summaries.append(parsed_summary)

    # 3) Compute metrics for each doc individually
    #    We'll gather predictions & references for BLEU in a single batch at the end.
    #    But we also show how to do them one-by-one if desired (some metrics prefer bulk calls).
    all_preds = []  # for BLEU's batch usage
    all_refs = []   # each item = list of reference strings, i.e. [reference_text]
    # We'll also store final metrics in a list of dicts
    all_metrics = []

    for i, doc_text in enumerate(documents):
        pred_summary = final_summaries[i].strip()
        ref_text = doc_text.strip()

        if not pred_summary:
            # If empty summary, store zeros
            all_metrics.append({
                "rouge1_f1": 0.0,
                "rouge2_f1": 0.0,
                "rougeL_f1": 0.0,
                "bleu": 0.0,
                "meteor": 0.0,
                "bert_score_f1": 0.0,
            })
            all_preds.append("")
            all_refs.append([""])
            continue

        # For BLEU's batch usage: keep the raw strings
        all_preds.append(pred_summary)
        all_refs.append([ref_text])

    # Now compute BLEU in one go (for all docs)
    # - all_preds is a list of single strings
    # - all_refs is a list of list-of-strings
    #   i.e. all_refs[i] is ["some reference text"] for doc i
    bleu_result = _bleu.compute(predictions=all_preds, references=all_refs)

    # Similarly we can compute METEOR, ROUGE, BERTScore in bulk. Let's do it doc by doc for clarity:

    # =========== METEOR (doc by doc, or also all at once) ===========
    meteor_result = _meteor.compute(predictions=all_preds, references=[r[0] for r in all_refs])

    # =========== ROUGE (doc by doc) ===========
    # The "summaries" vs "documents" approach. We can do it in one batch
    rouge_result = _rouge.compute(predictions=all_preds, references=[r[0] for r in all_refs])

    # =========== BERTScore (doc by doc) ===========
    bert_result = _bertscore.compute(predictions=all_preds,
                                     references=[r[0] for r in all_refs],
                                     model_type="bert-base-uncased")

    # Because we computed them *all at once*, each result.* is a single *average* across all docs,
    # or a list. In the default usage, ROUGE returns aggregated F1 for the entire corpus
    # If we want doc-level breakdown, we do doc by doc or set "use_stemmer=True" or "aggregate=False".
    # For simplicity, let's show how to just store the *average corpus-level metrics* in each doc row:

    # We'll keep the average BLEU, average METEOR, average BERT, and average ROUGE in each row for now.
    # If you want doc-level ROUGE, see the HF docs about 'aggregate=False' or do them individually.
    corpus_bleu = bleu_result["bleu"]            # float
    corpus_meteor = meteor_result["meteor"]      # float
    corpus_rouge1 = rouge_result["rouge1"]       # corpus-level f1
    corpus_rouge2 = rouge_result["rouge2"]
    corpus_rougeL = rouge_result["rougeL"]
    corpus_bert_f1 = sum(bert_result["f1"]) / len(bert_result["f1"])  # average doc-level BERT f1

    # So for each doc, we'll store the same corpus-level results. Or set them to 0 if we want doc-level metrics.
    # (If you want doc-level metrics for each doc, you'd have to compute "aggregate=False" or do them one at a time.)
    for i, metrics_dict in enumerate(all_metrics):
        if metrics_dict["rouge1_f1"] == 0.0 and all_preds[i] == "":
            # This doc was empty summary => we already stored zeros
            continue
        metrics_dict.update({
            "rouge1_f1": corpus_rouge1,
            "rouge2_f1": corpus_rouge2,
            "rougeL_f1": corpus_rougeL,
            "bleu": corpus_bleu,
            "meteor": corpus_meteor,
            "bert_score_f1": corpus_bert_f1,
        })

    # 4) Add new columns to the dataset
    dataset = dataset.add_column("raw_document_summary", raw_summaries)
    dataset = dataset.add_column("document_summary", final_summaries)
    dataset = dataset.add_column("summarization_model", [summ_model]*len(dataset))
    dataset = dataset.add_column("quality_metrics", all_metrics)

    # 5) Save
    save_dataset(
        dataset=dataset,
        step_name="summarization",
        config=config,
        output_dataset_name=summary_cfg["output_dataset_name"]
    )
    logger.success("Summarization stage completed with BLEU & other metrics stored in 'quality_metrics'.")
# yourbench/pipeline/summarization.py

from typing import Dict, Any, List
from loguru import logger
from datasets import Dataset
import evaluate

# Load HF evaluate metrics once
_rouge = evaluate.load("rouge")
_bleu = evaluate.load("bleu")
_meteor = evaluate.load("meteor")
_bertscore = evaluate.load("bertscore")

from yourbench.utils.inference_engine import run_inference, InferenceCall
from yourbench.utils.prompts import SUMMARIZATION_USER_PROMPT
from yourbench.utils.saving_engine import save_dataset
from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags

def run(config: Dict[str, Any]) -> None:
    """
    Summarization pipeline stage:
      1) Loads documents from HF dataset.
      2) Uses a summarization model to generate summaries.
      3) Computes corpus-level ROUGE, BLEU, METEOR, BERTScore.
      4) Stores them in 'quality_metrics' as a dictionary for each row.

    We'll do:
      - doc_texts: List[str] of original documents
      - final_summaries: List[str] of predicted summaries
      - Then pass these to the HF 'evaluate' library in batch mode (i.e., all at once),
        to get one set of average/corpus-level metrics. We then insert those same
        metric values for each row's "quality_metrics" dict.

    If you need per-document metrics (rather than a single average), you can do "aggregate=False"
    for the evaluate library or run each doc individually. But for demonstration, we do corpus-level.

    This code solves the "Failed to concatenate on axis=1" error by ensuring 'quality_metrics'
    has exactly the same length as the dataset rows. Each entry is a dict of metrics.
    """

    # 1) Basic config checks
    stage_cfg = config["pipeline"]["summarization"]
    if not stage_cfg.get("run", False):
        logger.info("Summarization stage is disabled. Skipping.")
        return

    logger.info("Running summarization stage.")
    dataset = smart_load_dataset(stage_cfg["source_dataset_name"], config)
    logger.info("Loaded dataset with {} documents.", len(dataset))

    # 2) Prepare inference calls
    documents: List[str] = dataset["document_text"]
    inference_calls = []
    for doc_text in documents:
        user_msg = {"role": "user", "content": SUMMARIZATION_USER_PROMPT.format(document=doc_text)}
        inference_calls.append(InferenceCall(messages=[user_msg]))

    logger.info("Sending {} summarization calls to inference engine.", len(inference_calls))
    responses_dict = run_inference(config, "summarization", inference_calls)

    # 3) Retrieve raw model outputs
    summ_model = config["model_roles"]["summarization"][0]  # assume single
    raw_summaries = responses_dict.get(summ_model, [])

    if len(raw_summaries) != len(documents):
        logger.warning(
            "Model '%s' returned %d summaries, but we have %d docs. Some mismatch occurred.",
            summ_model, len(raw_summaries), len(documents)
        )

    # 4) Parse <final_summary> for each doc
    final_summaries = []
    for i in range(len(documents)):
        raw_resp = raw_summaries[i] if i < len(raw_summaries) else ""
        parsed = extract_content_from_xml_tags(raw_resp, "final_summary").strip()
        final_summaries.append(parsed)

    # 5) We'll do a corpus-level metric calculation
    #    So, gather all preds (the final_summaries) & all refs (the original doc_texts)
    #    as raw strings (no tokenization). This is the simplest approach for evaluate.

    all_preds = [summary if summary else "" for summary in final_summaries]
    all_refs = [[doc] for doc in documents]  # note the double bracket for references

    # a) BLEU
    # passing raw strings is allowed, the HF BLEU metric will do internal tokenization
    bleu_result = _bleu.compute(predictions=all_preds, references=all_refs)
    corpus_bleu = bleu_result["bleu"]

    # b) METEOR
    meteor_result = _meteor.compute(predictions=all_preds, references=[r[0] for r in all_refs])
    corpus_meteor = meteor_result["meteor"]

    # c) ROUGE
    # By default, "compute()" returns overall average across the entire corpus
    # if "aggregate=True". So we'll get a single set of F1 scores. 
    rouge_result = _rouge.compute(predictions=all_preds, references=[r[0] for r in all_refs])
    corpus_rouge1 = rouge_result["rouge1"]
    corpus_rouge2 = rouge_result["rouge2"]
    corpus_rougeL = rouge_result["rougeL"]

    # d) BERTScore
    bert_res = _bertscore.compute(
        predictions=all_preds,
        references=[r[0] for r in all_refs],
        model_type="bert-base-uncased"
    )
    # This returns a list for each doc. The default is to aggregate them by default, 
    # but let's do the average:
    corpus_bert_f1 = sum(bert_res["f1"]) / len(bert_res["f1"]) if bert_res["f1"] else 0.0

    # 6) Build a "quality_metrics" column, length = len(dataset).
    #    Each entry is a dict of the same corpus-level numbers.
    #    (If you prefer doc-level metrics, you'd do "aggregate=False" or call each doc individually.)
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

    # 7) Add new columns
    #    Make sure "final_summaries" also has length == len(dataset)
    dataset = dataset.add_column("raw_document_summary", raw_summaries)
    dataset = dataset.add_column("document_summary", final_summaries)
    dataset = dataset.add_column("summarization_model", [summ_model] * len(dataset))
    dataset = dataset.add_column("quality_metrics", all_metrics)

    # 8) Save updated dataset
    save_dataset(
        dataset=dataset,
        step_name="summarization",
        config=config,
        output_dataset_name=stage_cfg["output_dataset_name"]
    )
    logger.success("Summarization stage completed successfully with corpus-level metrics added.")
