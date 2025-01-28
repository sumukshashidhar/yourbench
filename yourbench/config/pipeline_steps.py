from preprocessing.dataset_generation import generate_dataset
from preprocessing.generate_summaries import generate_summaries_for_documents
from preprocessing.create_chunks import create_chunks_for_documents
from preprocessing.create_multihop_chunks import create_multihop_chunks
from question_generation.generate_questions import (
    generate_single_shot_questions,
    generate_multihop_questions,
)
from postprocessing.reweight_and_deduplication import reweight_and_deduplicate_questions
from question_answering.answer_questions import answer_questions_with_llm


PIPELINE_STEPS = {
    "generate_dataset": {
        "func": generate_dataset, 
        "description": "Generate dataset"
    },
    "generate_summaries": {
        "func": generate_summaries_for_documents,
        "description": "Generate summaries for documents",
    },
    "create_chunks": {
        "func": create_chunks_for_documents,
        "description": "Create chunks for documents",
    },
    "make_chunk_pairings": {
        "func": create_multihop_chunks, 
        "description": "Create multi-hop chunks"
    },
    "create_single_hop_questions": {
        "func": generate_single_shot_questions,
        "description": "Generate single-shot questions",
    },
    "create_multi_hop_questions": {
        "func": generate_multihop_questions,
        "description": "Generate multi-hop questions",
    },
    "reweight_and_deduplicate_questions": {
        "func": reweight_and_deduplicate_questions,
        "description": "Reweight and deduplicate questions",
    },
    "answer_questions_with_llm": {
        "func": answer_questions_with_llm,
        "description": "Answer questions with LLM"
    },
    # # TODO: COMMENTED OUT FOR CURRENT IMPLEMENTATION
    # "reformat_for_judging": {
    #     "func": reformat_for_judging, 
    #     "description": "Reformat answers for judging"
    # },
    # "judge": {
    #     "func": judge_answers, 
    #     "description": "Judge answers"
    # },
    # "visualize_results": {
    #     "func": visualize_judge_results, 
    #     "description": "Visualize judge results"
    # },
}