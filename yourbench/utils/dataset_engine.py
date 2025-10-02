import os
import json
import math
import random
import shutil
import tempfile

# TYPE_CHECKING import to avoid circular imports
from typing import TYPE_CHECKING, Any, Set, List, Union, TypeVar, Sequence
from pathlib import Path
from contextlib import suppress
from dataclasses import dataclass

from loguru import logger

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from huggingface_hub import HfApi, DatasetCard, DatasetCardData, whoami
from huggingface_hub.utils import HFValidationError


if TYPE_CHECKING:
    from yourbench.utils.configuration_engine import YourbenchConfig


__all__ = ["custom_load_dataset", "custom_save_dataset", "upload_dataset_card"]

T = TypeVar("T")


class ConfigurationError(Exception):
    """Configuration error."""


@dataclass(slots=True, frozen=True)
class HFSettings:
    """Normalized HuggingFace configuration."""

    dataset_name: str
    organization: str | None
    token: str | None
    local_dir: Path | None
    concat_if_exist: bool = False
    private: bool = True
    export_jsonl: bool = False
    jsonl_export_dir: Path | None = None

    @property
    def repo_id(self) -> str:
        """Full repository identifier."""
        if "/" in self.dataset_name:
            return self.dataset_name
        return f"{self.organization}/{self.dataset_name}" if self.organization else self.dataset_name


def _is_offline() -> bool:
    """Check if offline mode enabled."""
    return os.environ.get("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes")


def _expand_var(value: str, field: str) -> str:
    """Ensure value is not unexpanded $VAR placeholder."""
    if value.startswith("$"):
        var_name = value[1:].split("/")[0]
        msg = f"Environment variable '{var_name}' in '{field}' not set"
        logger.error(msg)
        raise ConfigurationError(msg)
    return value


def _extract_settings(config: Union[dict[str, Any], "YourbenchConfig"]) -> HFSettings:
    """Parse and validate configuration."""
    # Handle both dict and YourbenchConfig
    from yourbench.utils.configuration_engine import is_yourbench_config

    if is_yourbench_config(config):
        # YourbenchConfig dataclass
        hf = config.hf_configuration
        dataset_name = _expand_var(hf.hf_dataset_name, "hf_dataset_name")
        org_raw = hf.hf_organization
        token = hf.hf_token or os.getenv("HF_TOKEN")
        organization = _resolve_organization(org_raw, token)
        local_dir = hf.local_dataset_dir
        if local_dir and not isinstance(local_dir, Path):
            local_dir = Path(local_dir).expanduser().resolve()
        jsonl_export_dir = hf.jsonl_export_dir
        if jsonl_export_dir and not isinstance(jsonl_export_dir, Path):
            jsonl_export_dir = Path(jsonl_export_dir).expanduser().resolve()
        return HFSettings(
            dataset_name=dataset_name,
            organization=organization,
            token=token,
            local_dir=local_dir,
            concat_if_exist=hf.concat_if_exist,
            private=hf.private,
            export_jsonl=hf.export_jsonl,
            jsonl_export_dir=jsonl_export_dir,
        )
    else:
        # Legacy dict format
        if "hf_configuration" not in config:
            raise ConfigurationError("'hf_configuration' section missing")

        hf = config["hf_configuration"]
        if "hf_dataset_name" not in hf:
            raise ConfigurationError("'hf_dataset_name' required")

        dataset_name = _expand_var(hf["hf_dataset_name"], "hf_dataset_name")
        org_raw = hf.get("hf_organization")
        token = hf.get("token") or os.getenv("HF_TOKEN")

        organization = _resolve_organization(org_raw, token)

        local_raw = config.get("local_dataset_dir") or hf.get("local_dataset_dir")
        local_dir = Path(local_raw).expanduser().resolve() if local_raw else None

        jsonl_export_raw = hf.get("jsonl_export_dir")
        jsonl_export_dir = Path(jsonl_export_raw).expanduser().resolve() if jsonl_export_raw else None

        return HFSettings(
            dataset_name=dataset_name,
            organization=organization,
            token=token,
            local_dir=local_dir,
            concat_if_exist=hf.get("concat_if_exist", False),
            private=hf.get("private", True),
            export_jsonl=hf.get("export_jsonl", False),
            jsonl_export_dir=jsonl_export_dir,
        )


def _resolve_organization(org: str | None, token: str | None) -> str | None:
    """Resolve organization, fetching from HF if needed."""
    if _is_offline() or (org and not org.startswith("$")):
        return org

    if org and org.startswith("$"):
        var_name = org[1:].split("/")[0]
        logger.warning(f"Environment variable '{var_name}' in 'hf_organization' not set")

    if not token:
        return None

    try:
        if username := whoami(token=token).get("name"):
            logger.info(f"Using '{username}' as organization")
            return username
    except HFValidationError:
        logger.warning("Invalid HF token")
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Network error fetching organization: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching organization: {e}")

    return None


def _validate_repo(settings: HFSettings) -> None:
    """Validate repository ID format."""
    if _is_offline():
        return

    try:
        HfApi().repo_info(repo_id=settings.repo_id, repo_type="dataset", token=settings.token)
    except HFValidationError as e:
        raise ConfigurationError(f"Invalid repo ID '{settings.repo_id}': {e}") from e
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Network error validating repo: {e}")
    except Exception as e:
        if "404" not in str(e):
            logger.error(f"Unexpected error validating repo: {e}")
            raise


def _load_local(path: Path, subset: str | None) -> Dataset:
    """Load dataset from local path."""
    logger.info(f"Loading '{subset or 'default'}' from {path}")
    dataset = load_from_disk(str(path))

    if subset is None:
        return dataset

    if not isinstance(dataset, DatasetDict):
        # If subset is requested but dataset is not a DatasetDict,
        # return the dataset with a warning (assuming it's the one they want)
        logger.warning(f"Subset '{subset}' requested but dataset is not a DatasetDict. Returning the dataset anyway.")
        return dataset

    if subset in dataset:
        return dataset[subset]

    # Provide a helpful error message showing available subsets
    available_subsets = list(dataset.keys())
    raise ConfigurationError(f"Subset '{subset}' not found in local dataset. Available subsets: {available_subsets}")


def _load_hub(repo_id: str, subset: str | None, token: str | None) -> Dataset:
    """Load dataset from HuggingFace Hub."""
    logger.info(f"Loading '{subset or 'default'}' from Hub: {repo_id}")

    try:
        dataset = load_dataset(repo_id, name=subset, split="train", token=token)
        if len(dataset) == 0:
            raise ValueError(f"Dataset from Hub is empty (repo: {repo_id}, subset: {subset})")
        return dataset
    except ValueError as e:
        if "BuilderConfig" in str(e) and "not found" in str(e):
            raise ConfigurationError(f"Subset '{subset}' not found on Hub") from e
        if "split" in str(e):
            raise ConfigurationError("Split 'train' not found in dataset") from e
        raise


def _merge_datasets(
    existing: Dataset | DatasetDict, new: Dataset, subset: str | None, concat_if_exist: bool = False
) -> Dataset | DatasetDict:
    """Merge new dataset with existing. If subset exists and concat_if_exist is True, new data is concatenated."""
    if subset is None:
        if isinstance(existing, Dataset):
            if concat_if_exist:
                return concatenate_datasets([existing, new])
            else:
                return new
        return new

    if not isinstance(existing, DatasetDict):
        existing = DatasetDict({"default": existing})

    if subset in existing and concat_if_exist:
        try:
            # Concatenate new data with the existing subset
            new = concatenate_datasets([existing[subset], new])
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Could not concatenate for subset '{subset}' (e.g., schema mismatch). Overwriting. Error: {e}"
            )

    existing[subset] = new
    return existing


def _safe_save(dataset: Dataset | DatasetDict, path: Path) -> None:
    """Save dataset, handling overwrite issues."""
    try:
        dataset.save_to_disk(str(path))
        logger.success(f"Saved to {path}")
    except PermissionError as e:
        if "can't overwrite itself" not in str(e):
            raise

        with tempfile.TemporaryDirectory() as tmp:
            dataset.save_to_disk(tmp)
            shutil.rmtree(path, ignore_errors=True)
            shutil.copytree(tmp, path)
        logger.success(f"Saved to {path} (via temp)")


def _export_to_jsonl(dataset: Dataset | DatasetDict, export_dir: Path, subset: str | None = None) -> None:
    """Export dataset to JSONL format."""
    export_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset, Dataset):
        # Single dataset - export to single file
        file_name = f"{subset}.jsonl" if subset else "dataset.jsonl"
        file_path = export_dir / file_name

        logger.info(f"Exporting dataset to JSONL: {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.success(f"Exported {len(dataset)} rows to {file_path}")

        # Special handling for prepared_lighteval subset - create simplified questions_and_answers.jsonl
        if subset == "prepared_lighteval":
            # Also export simplified version to questions_and_answers.jsonl in current directory
            qa_file_path = Path.cwd() / "questions_and_answers.jsonl"

            logger.info(f"Creating simplified Q&A dataset at: {qa_file_path}")
            with open(qa_file_path, "w", encoding="utf-8") as f:
                for row in dataset:
                    # Create a filtered row without document/summary/chunks
                    filtered_row = {
                        "question": row.get("question", ""),
                        "ground_truth_answer": row.get("ground_truth_answer", ""),
                        "question_category": row.get("question_category", ""),
                        "kind": row.get("kind", ""),
                        "estimated_difficulty": row.get("estimated_difficulty", 5),
                        "citations": row.get("citations", []),
                        "document_id": row.get("document_id", ""),
                        "chunk_ids": row.get("chunk_ids", []),
                        "question_generating_model": row.get("question_generating_model", ""),
                        "choices": row.get("choices", []),
                        "gold": row.get("gold", []),
                    }
                    f.write(json.dumps(filtered_row, ensure_ascii=False) + "\n")
            logger.success(f"Created simplified questions_and_answers.jsonl with {len(dataset)} Q&A pairs")

    elif isinstance(dataset, DatasetDict):
        # Multiple subsets - export each to separate file
        logger.info(f"Exporting DatasetDict with {len(dataset)} subsets to JSONL")
        for subset_name, subset_data in dataset.items():
            # Use recursive call for all subsets to handle prepared_lighteval specially
            _export_to_jsonl(subset_data, export_dir, subset_name)

        # Create an index file listing all subsets
        index_path = export_dir / "index.json"
        index_data = {
            "subsets": list(dataset.keys()),
            "total_rows": sum(len(subset_data) for subset_data in dataset.values()),
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
        logger.info(f"Created index file: {index_path}")


def custom_load_dataset(config: Union[dict[str, Any], "YourbenchConfig"], subset: str | None = None) -> Dataset:
    """Load dataset subset from local path or Hub. Raises errors if data missing or invalid."""
    settings = _extract_settings(config)

    if settings.local_dir and settings.local_dir.exists():
        return _load_local(settings.local_dir, subset)

    if _is_offline():
        raise RuntimeError("Offline mode enabled but no local dataset found")

    _validate_repo(settings)
    return _load_hub(settings.repo_id, subset, settings.token)


def custom_save_dataset(
    dataset: Dataset,
    config: Union[dict[str, Any], "YourbenchConfig"],
    subset: str | None = None,
    *,
    save_local: bool = True,
    push_to_hub: bool = True,
) -> None:
    """Save dataset locally and/or push to Hub."""
    settings = _extract_settings(config)

    if _is_offline():
        save_local = True
        push_to_hub = False
        logger.info("Offline mode - only saving locally")

    if save_local and settings.local_dir:
        logger.info(f"Saving to {settings.local_dir}")

        existing = None
        if settings.local_dir.exists():
            try:
                existing = load_from_disk(str(settings.local_dir))
            except (FileNotFoundError, PermissionError, OSError) as e:
                logger.warning(f"Error loading existing dataset from disk: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading existing dataset: {e}")
                raise

        merged = (
            _merge_datasets(existing, dataset, subset, settings.concat_if_exist)
            if existing
            else (DatasetDict({subset: dataset}) if subset else dataset)
        )

        settings.local_dir.parent.mkdir(parents=True, exist_ok=True)
        _safe_save(merged, settings.local_dir)

        # Export to JSONL if enabled
        if settings.export_jsonl and settings.jsonl_export_dir:
            logger.info("JSONL export is enabled")
            _export_to_jsonl(merged, settings.jsonl_export_dir, subset)

    if push_to_hub and not _is_offline():
        if settings.concat_if_exist:
            with suppress(Exception):
                existing = _load_hub(settings.repo_id, subset, settings.token)
                dataset = concatenate_datasets([existing, dataset])
                logger.info("Concatenated with existing remote")

        _validate_repo(settings)
        logger.info(f"Pushing to Hub: {settings.repo_id}")
        dataset.push_to_hub(
            repo_id=settings.repo_id,
            private=settings.private,
            config_name=subset or "default",
            token=settings.token,
        )
        logger.success(f"Pushed to Hub: {settings.repo_id}")


def replace_dataset_columns(
    dataset: Dataset, columns_data: dict[str, list], preserve_metadata: bool = False
) -> Dataset:
    """Replace columns by removing existing and adding new ones."""
    to_remove = [col for col in columns_data if col in dataset.column_names]

    if to_remove:
        logger.info(f"Removing columns: {to_remove}")
        dataset = dataset.remove_columns(to_remove)

    for name, data in columns_data.items():
        dataset = dataset.add_column(name, data)

    return dataset


def _unrank_comb(n: int, k: int, rank: int) -> List[int]:
    """
    Return the k-combination of [0, n) corresponding to the given rank
    in colexicographic (colex) order.

    Colexicographic order sorts combinations by increasing values of the
    largest element, then second largest, and so on (i.e., right-to-left
    significance).

    Parameters
    ----------
    n : int
        Size of the universe (exclusive upper bound of elements).
    k : int
        Size of each combination.
    rank : int
        Integer in the range [0, C(n, k)) specifying the position of the combination
        in colexicographic order.

    Returns
    -------
    List[int]
        A strictly increasing list of k integers in the range [0, n),
        representing the rank-th combination in colex order.

    Raises
    ------
    ValueError
        If k is not in [0, n] or rank is not in [0, C(n, k)).
    """
    if not 0 <= k <= n:
        raise ValueError(f"require 0 ≤ k ≤ n, got k={k}, n={n}")
    max_rank = math.comb(n, k)
    if not 0 <= rank < max_rank:
        raise ValueError(f"rank must be in [0,{max_rank - 1}], got {rank}")

    combo: List[int] = []
    for i in range(k, 0, -1):
        # largest c such that C(c, i) ≤ rank (binary search)
        lo, hi = i - 1, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if math.comb(mid, i) <= rank:
                lo = mid
            else:
                hi = mid - 1
        combo.append(lo)
        rank -= math.comb(lo, i)
        n = lo  # next digit must be < current one
    combo.reverse()
    return combo


def _floyd_sample_indices(total: int, sample_size: int, *, rng: random.Random | None = None) -> Set[int]:
    """Select sample_size unique integers ∈ [0, total) uniformly at random"""
    if sample_size > total:
        raise ValueError("sample_size cannot exceed total")
    if rng is None:
        rng = random

    chosen: Set[int] = set()
    for j in range(total - sample_size, total):
        t = rng.randrange(0, j + 1)
        chosen.add(t if t not in chosen else j)
    return chosen


def _sample_exact_combinations(
    objects: Sequence[T], k: int, N: int, *, rng: random.Random | None = None
) -> List[List[T]]:
    """Draw N distinct k-combinations from objects exactly uniformly.

    The function first uses Bob Floyd to pick N distinct ranks in
    `[0, C(n,k))` (where `n = len(objects)`), then converts each rank to its
    combination via `_unrank_comb`, and finally maps the integer indices back
    to the actual objects.
    """
    n = len(objects)
    if not 0 <= k <= n:
        raise ValueError("require 0 ≤ k ≤ n")
    total = math.comb(n, k)
    if N > total:
        raise ValueError("cannot request more combinations than exist")
    if rng is None:
        rng = random

    ranks = _floyd_sample_indices(total, N, rng=rng)
    combos: List[List[T]] = []
    for r in ranks:
        idxs = _unrank_comb(n, k, r)
        combos.append([objects[i] for i in idxs])
    return combos


def create_cross_document_dataset(dataset: Dataset, stage_cfg: dict[str, Any]) -> Dataset:
    """Creates a cross-document Dataset by combining multi-hop chunks from different documents.

    Args:
        dataset: A HuggingFace Dataset where each row may contain a 'multihop_chunks' list and 'document_summary'.
        stage_cfg: Stage-specific config containing:
            - 'max_combinations' (int): The maximum number of cross-document combinations to generate.
            - 'chunks_per_document' (int): The number of chunks to sample from each document.
            - 'num_docs_per_combination' (List[int]): A list [min, max] specifying the range of documents to combine.
            - 'random_seed' (int): Seed for the random number generator.

    Returns:
        A new Dataset with cross-document combinations, preserving a similar schema but with an aggregated summary.
    """
    # Extract and validate configuration
    max_combinations = int(
        getattr(stage_cfg, "max_combinations", 100)
        if hasattr(stage_cfg, "max_combinations")
        else stage_cfg.get("max_combinations", 100)
        if isinstance(stage_cfg, dict)
        else 100
    )
    chunks_per_document = int(
        getattr(stage_cfg, "chunks_per_document", 1)
        if hasattr(stage_cfg, "chunks_per_document")
        else stage_cfg.get("chunks_per_document", 1)
        if isinstance(stage_cfg, dict)
        else 1
    )
    num_docs_range = (
        getattr(stage_cfg, "num_docs_per_combination", [2, 5])
        if hasattr(stage_cfg, "num_docs_per_combination")
        else stage_cfg.get("num_docs_per_combination", [2, 5])
        if isinstance(stage_cfg, dict)
        else [2, 5]
    )
    random_seed = int(
        getattr(stage_cfg, "random_seed", 42)
        if hasattr(stage_cfg, "random_seed")
        else stage_cfg.get("random_seed", 42)
        if isinstance(stage_cfg, dict)
        else 42
    )

    # Validate num_docs_range
    if not isinstance(num_docs_range, list) or len(num_docs_range) != 2:
        raise ValueError("num_docs_per_combination must be a list of exactly 2 integers")

    if not all(isinstance(x, int) for x in num_docs_range):
        raise ValueError("num_docs_per_combination must contain only integers")

    min_docs, max_docs = num_docs_range[0], num_docs_range[1]

    if min_docs < 2:
        raise ValueError("min_docs must be at least 2 for cross-document combinations")
    if max_docs < min_docs:
        raise ValueError("max_docs must be >= min_docs")

    if chunks_per_document < 1:
        raise ValueError("chunks_per_document must be at least 1")

    # Check for required column
    if "multihop_chunks" not in dataset.column_names:
        logger.warning("Dataset is missing 'multihop_chunks'. Cross-document generation aborted.")
        return Dataset.from_list([])

    # Extract documents with valid multihop_chunks
    docs = []
    for idx, row in enumerate(dataset):
        multihop_chunks = row.get("multihop_chunks", [])
        if isinstance(multihop_chunks, list) and multihop_chunks:
            valid_chunks = [
                chunk
                for chunk in multihop_chunks
                if isinstance(chunk, dict) and all(key in chunk for key in ("chunk_ids", "chunks_text"))
            ]
            if valid_chunks:
                # Create more readable and collision-resistant document IDs
                doc_id = row.get("document_id", f"doc_{idx}")
                # Clean doc_id for safe ID generation
                clean_doc_id = "".join(c for c in str(doc_id) if c.isalnum() or c in "_-")
                if not clean_doc_id:
                    clean_doc_id = f"doc_{idx}"

                docs.append({
                    "document_id": clean_doc_id,
                    "original_index": idx,
                    "document_summary": row.get("document_summary", ""),
                    "multihop_chunks": valid_chunks,
                })

    if len(docs) < min_docs:
        logger.warning(f"Found only {len(docs)} document(s) with valid 'multihop_chunks'. Need at least {min_docs}.")
        return Dataset.from_list([])

    logger.info(f"Found {len(docs)} documents with valid multihop_chunks")

    # Initialize random number generator
    rng = random.Random(random_seed)

    # Generate combinations efficiently using exact uniform sampling
    cross_rows = []

    # Strategy: distribute combinations across different group sizes
    # Calculate total possible combinations across all group sizes
    total_possible_combinations = sum(
        math.comb(len(docs), k) for k in range(min_docs, min(max_docs + 1, len(docs) + 1))
    )

    logger.info(f"Total possible combinations: {total_possible_combinations}")

    # Cap max_combinations to what's actually possible
    actual_max_combinations = min(max_combinations, total_possible_combinations)

    # For each possible number of documents to combine
    for num_docs_to_combine in range(min_docs, min(max_docs + 1, len(docs) + 1)):
        # Calculate how many combinations we can make with this number of docs
        combinations_for_this_size = math.comb(len(docs), num_docs_to_combine)

        if combinations_for_this_size == 0:
            continue

        # Determine how many combinations to generate for this group size
        remaining_combinations = actual_max_combinations - len(cross_rows)
        if remaining_combinations <= 0:
            break

        # Simple proportional allocation
        proportion = combinations_for_this_size / total_possible_combinations
        target_for_this_size = max(1, int(proportion * actual_max_combinations))
        actual_for_this_size = min(target_for_this_size, combinations_for_this_size, remaining_combinations)

        if actual_for_this_size <= 0:
            continue

        logger.info(f"Generating {actual_for_this_size} combinations with {num_docs_to_combine} documents")

        # Use exact uniform sampling to get distinct combinations
        try:
            doc_combinations = _sample_exact_combinations(docs, num_docs_to_combine, actual_for_this_size, rng=rng)
        except ValueError as e:
            logger.warning(f"Could not generate combinations for {num_docs_to_combine} docs: {e}")
            continue

        # Process each combination
        for doc_group in doc_combinations:
            sampled_chunks_from_group = []
            doc_ids_for_tracing = []

            # Sample chunks from each document in the group
            for doc in doc_group:
                doc_ids_for_tracing.append(doc["document_id"])

                if not doc["multihop_chunks"]:
                    continue

                # Sample the specified number of chunks from this document
                num_chunks_to_sample = min(chunks_per_document, len(doc["multihop_chunks"]))
                if num_chunks_to_sample == 1:
                    sampled_chunks = [rng.choice(doc["multihop_chunks"])]
                else:
                    sampled_chunks = rng.sample(doc["multihop_chunks"], num_chunks_to_sample)

                sampled_chunks_from_group.extend(sampled_chunks)

            # Validation: ensure we have chunks from the expected number of documents
            # (This addresses the original validation mismatch issue)
            expected_total_chunks = len(doc_group) * chunks_per_document
            if len(sampled_chunks_from_group) < len(doc_group):
                logger.warning(f"Insufficient chunks sampled from document group {doc_ids_for_tracing}")
                continue

            # Combine chunks from all documents in the group
            combined_ids = []
            combined_texts = []

            for chunk in sampled_chunks_from_group:
                chunk_ids = chunk.get("chunk_ids", [])
                chunk_texts = chunk.get("chunks_text", [])

                if isinstance(chunk_ids, list):
                    combined_ids.extend(chunk_ids)
                else:
                    combined_ids.append(chunk_ids)

                if isinstance(chunk_texts, list):
                    combined_texts.extend(chunk_texts)
                else:
                    combined_texts.append(chunk_texts)

            # Create combined multihop chunk
            combined_multihop_chunk = {
                "chunk_ids": combined_ids,
                "chunks_text": combined_texts,
            }

            # Combine document summaries
            doc_summaries = [
                doc["document_summary"]
                for doc in doc_group
                if doc.get("document_summary") and doc["document_summary"].strip()
            ]

            combined_summary = ""
            if doc_summaries:
                header = "Here are the summaries from the various documents involved in the chunking:"
                summary_bullets = "\n".join(f"- {s}" for s in doc_summaries)
                combined_summary = f"{header}\n\n{summary_bullets}"

            # Create readable and collision-resistant ID
            doc_ids_sorted = sorted(doc_ids_for_tracing)
            doc_ids_str = "_".join(doc_ids_sorted)

            # Create a human-readable, deterministic ID using number of documents, sorted document IDs, and chunks per document
            cross_doc_id = f"cross_{len(doc_group)}docs_{doc_ids_str}_chunks{chunks_per_document}"

            # Add comprehensive metadata for traceability
            metadata = {
                "source_documents": doc_ids_sorted,
                "num_source_docs": len(doc_group),
                "chunks_per_doc": chunks_per_document,
                "total_chunks_sampled": len(sampled_chunks_from_group),
                "source_indices": sorted([doc["original_index"] for doc in doc_group]),
                "generation_method": "exact_uniform_sampling",
            }

            cross_rows.append({
                "document_id": cross_doc_id,
                "document_summary": combined_summary,
                "chunks": [],  # keep consistent with original schema
                "multihop_chunks": [combined_multihop_chunk],
                "cross_document_metadata": metadata,  # add traceability
            })

    if not cross_rows:
        logger.warning("No cross-document combinations were generated.")
        return Dataset.from_list([])

    if len(cross_rows) < max_combinations:
        logger.info(f"Generated {len(cross_rows)} out of {max_combinations} requested combinations.")
    else:
        logger.info(f"Successfully generated {len(cross_rows)} cross-document combinations.")

    return Dataset.from_list(cross_rows)


# Dataset card generation functions


def extract_readme_metadata(repo_id: str, token: str | None = None) -> str:
    """Extracts the metadata from the README.md file of the dataset repository.
    We have to download the previous README.md file in the repo, extract the metadata from it.
    Args:
        repo_id: The ID of the repository to push to, from the `push_to_hub` method.
        token: The token to authenticate with the Hugging Face Hub, from the `push_to_hub` method.
    Returns:
        The metadata extracted from the README.md file of the dataset repository as a str.
    """
    try:
        import re
        from pathlib import Path

        from huggingface_hub.file_download import hf_hub_download

        readme_path = Path(hf_hub_download(repo_id, "README.md", repo_type="dataset", token=token))
        # Extract the content between the '---' markers
        metadata_match = re.findall(r"---\n(.*?)\n---", readme_path.read_text(), re.DOTALL)

        if not metadata_match:
            logger.debug("No YAML metadata found in the README.md")
            return ""

        return metadata_match[0]

    except Exception as e:
        logger.debug(f"Failed to extract metadata from README.md: {e}")
        return ""


def extract_dataset_info(repo_id: str, token: str | None = None) -> str:
    """
    Extract dataset_info section from README metadata.

    Args:
        repo_id: The dataset repository ID
        token: Optional HuggingFace token for authentication

    Returns:
        The dataset_info section as a string, or empty string if not found
    """
    readme_metadata = extract_readme_metadata(repo_id=repo_id, token=token)
    if not readme_metadata:
        return ""

    section_prefix = "dataset_info:"
    if section_prefix not in readme_metadata:
        return ""

    try:
        # Extract the part after `dataset_info:` prefix
        config_data = section_prefix + readme_metadata.split(section_prefix)[1]
        return config_data
    except IndexError:
        logger.debug("Failed to extract dataset_info section from metadata")
        return ""


def _serialize_config_for_card(config: Union[dict[str, Any], "YourbenchConfig"]) -> str:
    """
    Sanitize and serialize pipeline config to YAML for inclusion in dataset card.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config serialization")
    from copy import deepcopy

    # Load default prompts to compare against
    from yourbench.utils.configuration_engine import _load_prompt_from_package

    # Map of prompt fields to their default package paths
    default_prompt_paths = {
        "pdf_llm_prompt": "ingestion/pdf_llm_prompt.md",
        "summarization_user_prompt": "summarization/summarization_user_prompt.md",
        "combine_summaries_user_prompt": "summarization/combine_summaries_user_prompt.md",
        "single_shot_system_prompt": "question_generation/single_shot_system_prompt.md",
        "single_shot_system_prompt_multi": "question_generation/single_shot_system_prompt_multi.md",
        "single_shot_user_prompt": "question_generation/single_shot_user_prompt.md",
        "multi_hop_system_prompt": "question_generation/multi_hop_system_prompt.md",
        "multi_hop_user_prompt": "question_generation/multi_hop_user_prompt.md",
        "question_rewriting_system_prompt": "question_rewriting/question_rewriting_system_prompt.md",
        "question_rewriting_user_prompt": "question_rewriting/question_rewriting_user_prompt.md",
    }

    # Load default prompts for comparison
    default_prompts = {}
    for field, path in default_prompt_paths.items():
        content = _load_prompt_from_package(path)
        if content:
            default_prompts[field] = content

    def _is_default_prompt(value: str, field_name: str) -> bool:
        """Check if a prompt value matches the default."""
        if field_name in default_prompts:
            return value.strip() == default_prompts[field_name].strip()
        return False

    def _make_relative_path(path_str: str) -> str:
        """Convert absolute path to relative if possible."""
        try:
            path = Path(path_str)
            # If it's already relative, return as is
            if not path.is_absolute():
                return path_str

            # For absolute paths, try to make relative to cwd
            cwd = Path.cwd()

            # Handle paths that might not exist yet
            if path.exists():
                abs_path = path.resolve()
                try:
                    rel_path = abs_path.relative_to(cwd)
                    return str(rel_path)
                except ValueError:
                    pass
            else:
                # For non-existent paths, do string-based relative conversion
                cwd_str = str(cwd)
                if path_str.startswith(cwd_str):
                    return path_str[len(cwd_str) :].lstrip("/\\")

            # If we can't make it relative, return just the last parts
            # This helps avoid exposing full system paths
            parts = path.parts
            if len(parts) > 3:
                # Keep last 3 parts for context
                return str(Path(*parts[-3:]))

            return path_str
        except Exception:
            # If all else fails, return as is
            return path_str

    def _sanitize(obj, key=None, parent_key=None):
        if isinstance(obj, dict):
            sanitized_dict = {}
            for k, v in obj.items():
                sanitized_value = _sanitize(v, k, key)
                # Skip fields with None values or empty strings
                if sanitized_value is not None and sanitized_value != "":
                    sanitized_dict[k] = sanitized_value
            return sanitized_dict if sanitized_dict else None

        if isinstance(obj, list):
            return [_sanitize(v, key, parent_key) for v in obj]

        if isinstance(obj, Path):
            # Convert Path objects to relative strings
            return _make_relative_path(str(obj))

        if isinstance(obj, str):
            # Keep placeholders
            if obj.startswith("$"):
                return obj

            # Handle paths - make them relative
            if key and any(path_key in key.lower() for path_key in ["path", "dir", "directory"]):
                if "/" in obj or "\\" in obj:
                    return _make_relative_path(obj)

            # Handle prompt fields
            if key and "prompt" in key.lower():
                # Check if it's a default prompt
                if _is_default_prompt(obj, key):
                    # Return None to filter out default prompts entirely
                    return None
                
                # All non-default prompts are custom
                return f"custom_{key}.md"

            # Mask api_key arguments
            if key and "api_key" in key.lower():
                return "$API_KEY"
            # Mask OpenAI API keys
            if obj.startswith("sk-"):
                return "$OPENAI_API_KEY"
            # Mask HuggingFace tokens
            if obj.startswith("hf_"):
                return "$HF_TOKEN"
            # Mask HF organization/username in hf_organization field
            if key == "hf_organization" and not obj.startswith("$"):
                return "$HF_ORGANISATION"
            return obj

        # Explicitly return boolean, integer, float values unchanged
        if isinstance(obj, (bool, int, float)):
            return obj

        # Return None for None values (will be filtered out)
        if obj is None:
            return None

        return obj

    # Convert YourbenchConfig to dict if needed
    from yourbench.utils.configuration_engine import is_yourbench_config

    if is_yourbench_config(config):
        # Convert YourbenchConfig Pydantic model to dict format for serialization
        config_dict = config.model_dump()
    else:
        config_dict = config

    # First pass sanitization
    sanitized = _sanitize(deepcopy(config_dict))

    # Remove empty dictionaries and None values recursively
    def _remove_empty(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                cleaned_value = _remove_empty(v)
                if cleaned_value is not None and cleaned_value != {} and cleaned_value != []:
                    cleaned[k] = cleaned_value
            return cleaned if cleaned else None
        elif isinstance(obj, list):
            cleaned = [_remove_empty(item) for item in obj]
            return [item for item in cleaned if item is not None]
        else:
            return obj

    sanitized = _remove_empty(sanitized)

    # Filter out default values from hf_configuration
    if "hf_configuration" in sanitized:
        hf_config = sanitized["hf_configuration"]
        # Remove default values
        defaults_to_remove = {
            "private": False,
            "concat_if_exist": False,
            "local_saving": True,
            "upload_card": True,
            "export_jsonl": False,
            "local_dataset_dir": "data/saved_dataset",
            "jsonl_export_dir": "data/jsonl_export",
        }
        for key, default_value in defaults_to_remove.items():
            if key in hf_config and hf_config[key] == default_value:
                del hf_config[key]
    
    # Filter out default values from model_list
    if "model_list" in sanitized:
        model_list = sanitized["model_list"]
        if isinstance(model_list, list):
            for model in model_list:
                if isinstance(model, dict):
                    # Remove model-level defaults
                    model_defaults = {
                        "max_concurrent_requests": 32,
                        "encoding_name": "cl100k_base",
                    }
                    for key, default_value in model_defaults.items():
                        if key in model and model[key] == default_value:
                            del model[key]

    # Filter out default values from pipeline stages
    # Handle both 'pipeline' and 'pipeline_config' keys for backward compatibility
    pipeline_key = "pipeline_config" if "pipeline_config" in sanitized else "pipeline"
    if pipeline_key in sanitized:
        pipeline = sanitized[pipeline_key]
        # Remove stages that are not enabled
        stages_to_remove = []
        for stage, stage_config in pipeline.items():
            if isinstance(stage_config, dict):
                # Remove run: false stages entirely
                if stage_config.get("run") is False:
                    stages_to_remove.append(stage)
                # Remove run: true as it's redundant when stage is present
                elif stage_config.get("run") is True:
                    del stage_config["run"]

                # Remove other stage-specific defaults
                stage_defaults = {
                    # Ingestion defaults
                    "upload_to_hub": True,
                    "llm_ingestion": False,
                    "pdf_dpi": 300,
                    # Summarization defaults
                    "max_tokens": 32768,
                    "token_overlap": 512,
                    "encoding_name": "cl100k_base",
                    # Chunking defaults
                    "l_max_tokens": 8192,
                    "h_min": 2,
                    "h_max": 5,
                    "num_multihops_factor": 1,
                    # Cross-document defaults
                    "max_combinations": 100,
                    "chunks_per_document": 1,
                    "num_docs_per_combination": [2, 5],
                    "random_seed": 42,
                    # Citation filtering defaults
                    "subset": "prepared_lighteval",
                    "alpha": 0.7,
                    "beta": 0.3,
                    # Question generation defaults
                    "question_mode": "open-ended",
                    # Default file extensions
                    "supported_file_extensions": [
                        ".md",
                        ".txt",
                        ".html",
                        ".htm",
                        ".pdf",
                        ".docx",
                        ".doc",
                        ".pptx",
                        ".ppt",
                        ".xlsx",
                        ".xls",
                        ".rtf",
                        ".odt",
                    ],
                }

                for key, default_value in stage_defaults.items():
                    if key in stage_config and stage_config[key] == default_value:
                        del stage_config[key]

        for stage in stages_to_remove:
            del pipeline[stage]

    # Handle model_roles - if all roles use the same single model, remove it
    if "model_roles" in sanitized:
        model_roles = sanitized["model_roles"]
        # Get all unique models across all roles
        all_models = set()
        for role_models in model_roles.values():
            if isinstance(role_models, list):
                all_models.update(role_models)

        # If there's only one model used everywhere, remove model_roles entirely
        if len(all_models) <= 1:
            del sanitized["model_roles"]

    # Remove debug: false as it's the default
    if sanitized.get("debug") is False:
        del sanitized["debug"]

    # Rename pipeline_config to pipeline for YAML compatibility
    if "pipeline_config" in sanitized:
        sanitized["pipeline"] = sanitized.pop("pipeline_config")

    # Reorder sections: hf_configuration, model_list, model_roles, pipeline, then everything else
    ordered_config = {}
    if "hf_configuration" in sanitized:
        ordered_config["hf_configuration"] = sanitized.pop("hf_configuration")
    if "model_list" in sanitized:
        ordered_config["model_list"] = sanitized.pop("model_list")
    if "model_roles" in sanitized:
        ordered_config["model_roles"] = sanitized.pop("model_roles")
    if "pipeline" in sanitized:
        ordered_config["pipeline"] = sanitized.pop("pipeline")
    # Add remaining sections
    ordered_config.update(sanitized)

    return yaml.safe_dump(ordered_config, sort_keys=False, default_flow_style=False)


def _get_pipeline_subset_info(config: Union[dict[str, Any], "YourbenchConfig"]) -> str:
    """
    Generate a formatted markdown list of enabled pipeline stages with descriptions.
    The resulting markdown is used in the dataset card to document
    which processing steps were included in the pipeline.

    Args:
        config: The complete pipeline configuration dictionary containing
               the 'pipeline' section with enabled stages

    Returns:
        str: A markdown-formatted string with bullet points for each enabled pipeline stage,
             or an empty string if no stages are enabled
    """

    mapping = {
        "ingestion": "Read raw source documents, convert them to normalized markdown and save for downstream steps",
        "upload_ingest_to_hub": "Package and push ingested markdown dataset to the Hugging Face Hub or save locally with standardized fields",
        "summarization": "Perform hierarchical summarization: chunk-level LLM summaries followed by combine-stage reduction",
        "chunking": "Split texts into token-based single-hop and multi-hop chunks",
        "single_shot_question_generation": "Generate standalone question-answer pairs per chunk using LLM",
        "multi_hop_question_generation": "Generate multi-hop QA pairs requiring reasoning across multiple chunks",
        "lighteval": "Merge QA pairs and chunk metadata into a lighteval compatible dataset for quick model-based scoring",
        "citation_score_filtering": "Compute overlap-based citation scores and filter QA pairs accordingly",
    }
    # Handle both dict and YourbenchConfig
    from yourbench.utils.configuration_engine import is_yourbench_config

    if is_yourbench_config(config):
        # YourbenchConfig dataclass
        pipeline_config = config.pipeline_config
        lines = []
        for stage_name in [
            "ingestion",
            "summarization",
            "chunking",
            "single_shot_question_generation",
            "multi_hop_question_generation",
            "question_rewriting",
            "lighteval",
            "citation_score_filtering",
        ]:
            stage_config = getattr(pipeline_config, stage_name, None)
            if stage_config and getattr(stage_config, "run", False):
                desc = mapping.get(stage_name, stage_name.replace("_", " ").title())
                lines.append(f"- **{stage_name}**: {desc}")
    else:
        # Legacy dict format
        pipeline = config.get("pipeline", {})
        lines = []
        for stage, cfg in pipeline.items():
            if isinstance(cfg, dict) and cfg.get("run"):
                desc = mapping.get(stage, stage.replace("_", " ").title())
                lines.append(f"- **{stage}**: {desc}")
    return "\n".join(lines)


def _generate_and_upload_dataset_card(
    config: Union[dict[str, Any], "YourbenchConfig"], template_path: str | None = None
) -> None:
    """
    Internal implementation that generates and uploads a dataset card to Hugging Face Hub.

    This is the core implementation function called by the public upload_dataset_card() function.
    It handles the actual card generation and uploading without performing configuration checks.

    The dataset card includes:
    1. Pipeline subset descriptions based on enabled stages
    2. Full sanitized configuration for reproducibility
    3. YourBench version and other metadata
    4. Preserved dataset_info from the existing card for proper configuration display

    Args:
        config: Configuration dictionary containing HF settings
        template_path: Optional custom template path
    """
    logger.info("Starting dataset card upload process")

    if _is_offline():
        logger.warning("Offline mode enabled. Skipping dataset card upload.")
        return

    try:
        # Get dataset repo name
        settings = _extract_settings(config)
        dataset_repo_name = settings.repo_id
        logger.info(f"Uploading card for dataset: {dataset_repo_name}")

        # Load template
        if not template_path:
            # Try to find template in utils directory
            current_dir = os.path.dirname(__file__)
            template_path = os.path.join(current_dir, "yourbench_card_template.md")

        logger.info(f"Loading template from: {template_path}")

        if not os.path.exists(template_path):
            logger.error(f"Template file not found: {template_path}")
            return

        with open(template_path, "r", encoding="utf-8") as f:
            template_str = f.read()

        logger.debug(f"Template loaded successfully, length: {len(template_str)} characters")

        # Get HF token
        token = settings.token

        # Extract dataset_info section from existing README if available
        config_data = extract_dataset_info(repo_id=dataset_repo_name, token=token)
        logger.info(f"Extracted dataset_info section, length: {len(config_data) if config_data else 0} characters")

        # Use explicitly configured pretty_name or generate one from the dataset name
        from yourbench.utils.configuration_engine import is_yourbench_config

        if is_yourbench_config(config):
            # YourbenchConfig dataclass
            hf_config = config.hf_configuration
            pretty_name = getattr(hf_config, "pretty_name", None)
            if not pretty_name:
                dataset_name = dataset_repo_name.split("/")[-1]
                pretty_name = dataset_name.replace("-", " ").replace("_", " ").title()
        else:
            # Legacy dict format
            hf_config = config.get("hf_configuration", {})
            if "pretty_name" in hf_config:
                pretty_name = hf_config["pretty_name"]
            else:
                dataset_name = dataset_repo_name.split("/")[-1]
                pretty_name = dataset_name.replace("-", " ").replace("_", " ").title()

        card_data_kwargs = {"pretty_name": pretty_name}

        # Create DatasetCardData with our metadata
        card_data = DatasetCardData(**card_data_kwargs)
        logger.info(f"Created card data with pretty_name: {card_data.pretty_name}")

        # Get YourBench version
        from importlib.metadata import PackageNotFoundError, version

        try:
            version_str = version("yourbench")
        except PackageNotFoundError:
            # Fallback for development installs
            version_str = "dev"

        # Prepare template variables
        template_vars = {
            "pretty_name": card_data.pretty_name,
            "yourbench_version": version_str,
            "config_yaml": _serialize_config_for_card(config),
            "pipeline_subsets": _get_pipeline_subset_info(config),
            "config_data": config_data,  # Use the extracted dataset_info section
            "footer": getattr(hf_config, "footer", "*(This dataset card was automatically generated by YourBench)*")
            if is_yourbench_config(config)
            else hf_config.get("footer", "*(This dataset card was automatically generated by YourBench)*"),
        }

        logger.info("Rendering dataset card from template")
        logger.debug(f"Template variables: {list(template_vars.keys())}")

        # Render card with our template and variables
        card = DatasetCard.from_template(card_data=card_data, template_str=template_str, **template_vars)

        logger.info("Template rendered successfully")
        logger.debug(f"Rendered card content length: {len(str(card))} characters")

        # Push to hub
        logger.info(f"Pushing dataset card to hub: {dataset_repo_name}")
        card.push_to_hub(dataset_repo_name, token=token)

        logger.success(f"Dataset card successfully uploaded to: https://huggingface.co/datasets/{dataset_repo_name}")

    except Exception as e:
        logger.error(f"Failed to upload dataset card: {e}")
        logger.exception("Full traceback:")


def upload_dataset_card(config: Union[dict[str, Any], "YourbenchConfig"]) -> None:
    """
    Public interface to generate and upload a dataset card to Hugging Face Hub.

    This function performs configuration checks (like upload_card setting and offline mode)
    and then delegates to the internal _generate_and_upload_dataset_card() implementation.
    It should be called at the end of the pipeline when all subsets are available.

    Args:
        config: Pipeline configuration dictionary containing 'hf_configuration'
               with settings like 'upload_card' flag
    """
    try:
        # Check if card upload is enabled in config
        from yourbench.utils.configuration_engine import is_yourbench_config

        if is_yourbench_config(config):
            # YourbenchConfig dataclass
            hf_config = config.hf_configuration
            upload_card = getattr(hf_config, "upload_card", True)
        else:
            # Legacy dict format
            hf_config = config.get("hf_configuration", {})
            upload_card = hf_config.get("upload_card", True)

        if not upload_card:
            logger.info("Dataset card upload disabled in configuration. Skipping card upload.")
            return

        if _is_offline():
            logger.info("Offline mode enabled. Skipping dataset card upload.")
            return

        logger.info("Uploading dataset card with complete pipeline information")
        _generate_and_upload_dataset_card(config)

    except Exception as e:
        logger.error(f"Error uploading dataset card: {e}")
