from datasets import Dataset, concatenate_datasets, load_dataset

from yourbench.postprocessing.clustering import cluster_and_dedupe
from yourbench.utils.dataset_engine import handle_dataset_push, make_dataset_name


def _restructure_multihop(multihop_questions: Dataset) -> Dataset:
    """Restructure the multihop questions to match the single shot questions."""
    # make a new column, called "chunk", which is a string combination of the chunks list
    multihop_questions = multihop_questions.map(
        lambda x: {"chunk": "\n".join(x["chunks"])}
    )

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
    single_hop_questions_dataset = load_dataset(
        make_dataset_name(
            config,
            config["pipeline"]["reweight_and_deduplicate_questions"][
                "source_single_hop_questions_dataset_name"
            ],
        ),
        split="train",
    )
    multi_hop_questions_dataset = load_dataset(
        make_dataset_name(
            config,
            config["pipeline"]["reweight_and_deduplicate_questions"][
                "source_multi_hop_questions_dataset_name"
            ],
        ),
        split="train",
    )

    # first, strategically combine the two datasets. we need to reform the multihop questions to match the single shot questions
    multihop_questions_dataset = _restructure_multihop(multi_hop_questions_dataset)

    single_shot_questions_dataset = single_hop_questions_dataset.add_column(
        "question_complexity", ["single_hop"] * len(single_hop_questions_dataset)
    )

    # combine the two datasets
    large_q_dataset = concatenate_datasets(
        [single_shot_questions_dataset, multihop_questions_dataset]
    )

    # when we have a large dataset, we can push it to the hub
    handle_dataset_push(
        config,
        config["pipeline"]["reweight_and_deduplicate_questions"][
            "large_question_dataset_name"
        ],
        large_q_dataset,
    )

    # actual reweight and deduplication happens here:
    deduplicated_dataset = cluster_and_dedupe(large_q_dataset, config)

    # when we have a small dataset, we can push it to the hub
    handle_dataset_push(
        config,
        config["pipeline"]["reweight_and_deduplicate_questions"]["target_dataset_name"],
        deduplicated_dataset,
    )

    pass
