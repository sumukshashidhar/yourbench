from datasets import load_dataset

from yourbench.utils.load_task_config import _get_full_dataset_name_for_questions


def reweight_and_deduplicate_questions(config: dict) -> None:
    """Reweight and deduplicate questions based on the provided config."""
    # load in both datasets
    load_dataset(
        _get_full_dataset_name_for_questions(config, config["selected_choices"]["reweight_and_deduplicate_questions"]["source_single_shot_questions_dataset_name"])
    )
    load_dataset(
        _get_full_dataset_name_for_questions(config, config["selected_choices"]["reweight_and_deduplicate_questions"]["source_multihop_questions_dataset_name"])
    )
    pass
