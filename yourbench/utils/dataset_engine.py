import os
import random
import shutil
import tempfile
from typing import Any
from pathlib import Path
from itertools import combinations
from contextlib import suppress
from dataclasses import dataclass

from loguru import logger

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from huggingface_hub import HfApi, whoami
from huggingface_hub.utils import HFValidationError


__all__ = ["custom_load_dataset", "custom_save_dataset"]


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


def _extract_settings(config: dict[str, Any]) -> HFSettings:
    """Parse and validate configuration."""
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

    return HFSettings(
        dataset_name=dataset_name,
        organization=organization,
        token=token,
        local_dir=local_dir,
        concat_if_exist=hf.get("concat_if_exist", False),
        private=hf.get("private", True),
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

    if subset is None or not isinstance(dataset, DatasetDict):
        return dataset

    if subset in dataset:
        return dataset[subset]

    raise ConfigurationError(f"Subset '{subset}' not found in local dataset")


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


def _merge_datasets(existing: Dataset | DatasetDict, new: Dataset, subset: str | None) -> Dataset | DatasetDict:
    """Merge new dataset with existing, creating fresh object."""
    if subset is None:
        return new

    if not isinstance(existing, DatasetDict):
        existing = DatasetDict({"default": existing})

    merged = DatasetDict({k: v for k, v in existing.items() if k != subset})
    merged[subset] = new
    return merged


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


def custom_load_dataset(config: dict[str, Any], subset: str | None = None) -> Dataset:
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
    config: dict[str, Any],
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
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Network error loading existing: {e}")
            except Exception as e:
                logger.error(f"Error loading existing dataset: {e}")
                raise

        merged = (
            _merge_datasets(existing, dataset, subset)
            if existing
            else (DatasetDict({subset: dataset}) if subset else dataset)
        )

        settings.local_dir.parent.mkdir(parents=True, exist_ok=True)
        _safe_save(merged, settings.local_dir)

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


def _create_cross_document_dataset(dataset: Dataset, stage_cfg: dict[str, object]) -> Dataset:
    """Creates a cross-document Dataset by combining multi-hop chunks from different documents.

    Args:
        dataset: A HuggingFace Dataset where each row may contain a 'multihop_chunks' list.
        stage_cfg: Stage-specific config containing 'max_combinations' and 'chunks_per_document'.

    Returns:
        A new Dataset with cross-document combinations, preserving the same schema.
    """
    max_combinations = int(stage_cfg.get("max_combinations", 100))
    chunks_per_document = int(stage_cfg.get("chunks_per_document", 1))

    if "multihop_chunks" not in dataset.column_names:
        logger.warning("Dataset is missing 'multihop_chunks'. Cross-document generation aborted.")
        return Dataset.from_list([])

    docs = []
    for idx, row in enumerate(dataset):
        multihop_chunks = row.get("multihop_chunks", [])
        if isinstance(multihop_chunks, list) and multihop_chunks:
            docs.append({
                "document_id": row.get("document_id", f"doc_{idx}"),
                "multihop_chunks": multihop_chunks,
            })

    if len(docs) < 2:
        logger.warning(f"Found only {len(docs)} document(s) with 'multihop_chunks'. Need at least 2.")
        return Dataset.from_list([])

    rng = random.Random(42)
    rng.shuffle(docs)

    cross_rows = []
    for doc1, doc2 in combinations(docs, 2):
        samp1 = rng.sample(doc1["multihop_chunks"], min(len(doc1["multihop_chunks"]), chunks_per_document))
        samp2 = rng.sample(doc2["multihop_chunks"], min(len(doc2["multihop_chunks"]), chunks_per_document))

        for chunk1 in samp1:
            for chunk2 in samp2:
                if not all(k in chunk1 for k in ("chunk_ids", "chunks_text")):
                    logger.warning(f"Skipping malformed chunk in doc {doc1['document_id']}: {chunk1}")
                    continue
                if not all(k in chunk2 for k in ("chunk_ids", "chunks_text")):
                    logger.warning(f"Skipping malformed chunk in doc {doc2['document_id']}: {chunk2}")
                    continue

                combined = {
                    "chunk_ids": chunk1["chunk_ids"] + chunk2["chunk_ids"],
                    "chunks_text": chunk1["chunks_text"] + chunk2["chunks_text"],
                }

                cross_rows.append({
                    "document_id": f"cross_{doc1['document_id']}_{doc2['document_id']}",
                    "chunks": [],
                    "multihop_chunks": [combined],
                })

                if len(cross_rows) >= max_combinations:
                    logger.debug(f"Reached max_combinations: {max_combinations}")
                    return Dataset.from_list(cross_rows)

    return Dataset.from_list(cross_rows)
