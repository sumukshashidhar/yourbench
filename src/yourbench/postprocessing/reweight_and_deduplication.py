from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger

from yourbench.utils.load_task_config import _get_full_dataset_name_for_questions
from yourbench.postprocessing.clustering import cluster_and_dedupe

def handle_dataset_push(dataset: Dataset, dataset_name: str, config: dict) -> None:
    if config["configurations"]["push_to_huggingface"]:
        privacy = False if config["configurations"]["set_hf_repo_visibility"] != "private" else True
        logger.info(f"Pushing dataset '{dataset_name}' to Hugging Face Hub (privacy={privacy})")

        try:
            hub_path = f"{config['configurations']['hf_organization']}/{dataset_name}"
            # Try to load existing dataset
            try:
                existing_dataset = load_dataset(hub_path, split="train")
                logger.info(f"Found existing dataset at {hub_path}, concatenating...")
                dataset = concatenate_datasets([existing_dataset, dataset])
            except Exception as _:
                logger.info(f"No existing dataset found at {hub_path}, creating new...")

            # Push the dataset (either concatenated or new)
            dataset.push_to_hub(hub_path, private=privacy)
            logger.success(f"Successfully pushed dataset to Hugging Face Hub: {dataset_name}")
        except Exception as error:
            logger.error(f"Failed to push dataset to Hugging Face Hub: {str(error)}")
            raise
    else:
        logger.info(f"Saving dataset locally to: {dataset_name}")
        dataset.save_to_disk(dataset_name)
        logger.success(f"Successfully saved dataset to disk: {dataset_name}")


def _restructure_multihop(multihop_questions: Dataset) -> Dataset:
    """Restructure the multihop questions to match the single shot questions."""
    # make a new column, called "chunk", which is a string combination of the chunks list
    multihop_questions = multihop_questions.map(lambda x: {"chunk": "\n".join(x["chunks"])})

    # Create question_complexity column with chunk_ids length before removing the column
    multihop_questions = multihop_questions.map(
        lambda x: {"question_complexity": f"multi_hop_{len(x['chunk_ids'])}"}
    )

    # remove the chunks column
    multihop_questions = multihop_questions.remove_columns("chunks")
    # remove the chunk_ids column
    multihop_questions = multihop_questions.remove_columns("chunk_ids")

    return multihop_questions


def reweight_and_deduplicate_questions(config: dict) -> None:
    """Reweight and deduplicate questions based on the provided config."""
    # load in both datasets
    single_shot_questions_dataset = load_dataset(
        _get_full_dataset_name_for_questions(config, config["selected_choices"]["reweight_and_deduplicate_questions"]["source_single_shot_questions_dataset_name"]), split="train"
    )
    multihop_questions_dataset = load_dataset(
        _get_full_dataset_name_for_questions(config, config["selected_choices"]["reweight_and_deduplicate_questions"]["source_multihop_questions_dataset_name"]), split="train"
    )

    # first, strategically combine the two datasets. we need to reform the multihop questions to match the single shot questions
    multihop_questions_dataset = _restructure_multihop(multihop_questions_dataset)

    single_shot_questions_dataset = single_shot_questions_dataset.add_column("question_complexity", ["single_hop"] * len(single_shot_questions_dataset))

    # combine the two datasets
    large_q_dataset = concatenate_datasets([single_shot_questions_dataset, multihop_questions_dataset])

    # when we have a large dataset, we can push it to the hub
    # handle_dataset_push(large_q_dataset, config["selected_choices"]["reweight_and_deduplicate_questions"]["large_question_dataset_name"], config)

    # actual reweight and deduplication happens here:
    deduplicated_dataset = cluster_and_dedupe(large_q_dataset, config)

    # when we have a small dataset, we can push it to the hub
    handle_dataset_push(deduplicated_dataset, config["selected_choices"]["reweight_and_deduplicate_questions"]["target_dataset_name"], config)

    pass
