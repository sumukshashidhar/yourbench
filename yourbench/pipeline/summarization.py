from typing import Dict, Any
from yourbench.utils.inference_engine import run_inference, InferenceCall
from yourbench.utils.prompts import SUMMARIZATION_USER_PROMPT
from loguru import logger
from typing import List
from yourbench.utils.saving_engine import save_dataset
from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags

def run(config: Dict[str, Any]) -> None:
    """
    Run the summarization stage of the pipeline.
    """
    logger.info("Running summarization stage")
    # read the huggingface dataset of documents
    dataset = smart_load_dataset(config["pipeline"]["summarization"]["source_documents_dataset_name"], config)
    print(dataset)
    # for each document, make an infernece call
    documents: list[str] = dataset["document_text"]
    dataset["document_id"]
    # create the inference calls
    inference_calls = [
        InferenceCall(
            messages=[
                {"role": "user", "content": SUMMARIZATION_USER_PROMPT.format(document=document)}
            ]
        )
        for document in documents
    ]
    # run the inference calls
    logger.info("Running inference for summarization")
    responses: Dict[str, List[str]] = run_inference(config, "summarization", inference_calls)
    # parse the responses
    parsed_responses: list[str] = [extract_content_from_xml_tags(response, "final_summary") for response in responses[config["model_roles"]["summarization"][0]]]
    logger.debug("Summarization responses: {}", responses)
    # add a new column to the dataset with the summarization, with the summarization model
    dataset = dataset.add_column("raw_document_summary", responses[config["model_roles"]["summarization"][0]])
    dataset = dataset.add_column("document_summary", parsed_responses)
    # add the generating model to the dataset
    dataset = dataset.add_column("summarization_model", [config["model_roles"]["summarization"][0]] * len(dataset))
    # save the dataset
    save_dataset(dataset, "summarization", config, config["pipeline"]["summarization"]["output_documents_dataset_name"])
    return 