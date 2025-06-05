import os
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass

from loguru import logger

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from huggingface_hub import HfApi, whoami
from huggingface_hub.utils import HFValidationError


__all__ = [
    "custom_load_dataset",
    "custom_save_dataset",
]


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""


def _is_offline_mode() -> bool:
    """Check if offline mode is enabled via environment variable."""
    return os.environ.get("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes")


@dataclass(slots=True, frozen=True)
class _HFSettings:
    """Normalized Hugging Face related configuration extracted from *config*."""

    dataset_name: str
    organization: Optional[str]
    token: Optional[str]
    local_dir: Optional[Path]
    concat_if_exist: bool = False
    private: bool = True

    @property
    def repo_id(self) -> str:
        """Return the full ``organization/name`` repo identifier (or just name)."""
        # if the dataset is already a full repo id, return it
        if "/" in self.dataset_name:
            return self.dataset_name
        else:
            return f"{self.organization}/{self.dataset_name}" if self.organization else self.dataset_name


def _expanded_or_raise(value: str, *, field: str) -> str:
    """Ensure *value* is not an unexpanded ``$VARNAME`` placeholder."""

    if value.startswith("$"):
        var_name = value[1:].split("/")[0]
        msg = f"Environment variable '{var_name}' referenced in '{field}' is not set. Define it or expand the configuration."
        logger.error(msg)
        raise ConfigurationError(msg)
    return value


def _extract_hf_settings(config: Dict[str, Any]) -> _HFSettings:
    """Parse *config* and return a validated :class:`_HFSettings`."""

    if "hf_configuration" not in config:
        raise ConfigurationError("'hf_configuration' section missing from config")

    hf_cfg = config["hf_configuration"]

    dataset_name = _expanded_or_raise(hf_cfg["hf_dataset_name"], field="hf_dataset_name")

    # Optional fields
    organization_raw: Optional[str] = hf_cfg.get("hf_organization")
    token: Optional[str] = hf_cfg.get("token") or os.getenv("HF_TOKEN")

    organization = _infer_organization(organization_raw, token)

    local_dir_raw: Optional[str] = config.get("local_dataset_dir") or hf_cfg.get("local_dataset_dir")
    local_dir = Path(local_dir_raw).expanduser().resolve() if local_dir_raw else None

    return _HFSettings(
        dataset_name=dataset_name,
        organization=organization,
        token=token,
        local_dir=local_dir,
        concat_if_exist=bool(hf_cfg.get("concat_if_exist", False)),
        private=bool(hf_cfg.get("private", True)),
    )


def _infer_organization(org: Optional[str], token: Optional[str]) -> Optional[str]:
    """Return an organization/user namespace or *None*.

    * If ``org`` is a plain string → return it as‑is.
    * If it looks like an unexpanded ``$VAR`` → log a warning then try token.
    * If absent → fall back to ``whoami`` if a token is present.
    """

    if _is_offline_mode():
        return org  # don’t attempt network calls in offline mode

    if org and not org.startswith("$"):
        return org

    if org and org.startswith("$"):
        var_name = org[1:].split("/")[0]
        logger.warning(
            f"Environment variable '{var_name}' referenced in 'hf_organization' is not set; falling back to token."
        )

    if token:
        try:
            username = whoami(token=token).get("name")
            if username:
                logger.info(f"Using '{username}' as the organization namespace")
                return username
        except HFValidationError:
            logger.warning("Provided HF token is invalid; proceeding without namespace")
        except Exception as exc:
            logger.warning(f"Could not infer organization via whoami: {exc}")

    return None


def _validate_repo_id(settings: _HFSettings) -> None:
    """Ensure the constructed repo name is syntactically valid using the Hub API."""

    if _is_offline_mode():
        return

    try:
        HfApi().repo_info(repo_id=settings.repo_id, repo_type="dataset", token=settings.token)
    except HFValidationError as exc:
        raise ConfigurationError(f"Invalid repo id '{settings.repo_id}': {exc}") from exc
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Local filesystem helpers
# ---------------------------------------------------------------------------


def _load_from_local(path: Path, subset: Optional[str]) -> Dataset:
    """Return the dataset (or split) located at *path*."""

    ds = load_from_disk(str(path))

    if subset is None or not isinstance(ds, DatasetDict):
        return ds

    if subset in ds:
        return ds[subset]

    logger.warning(f"Subset '{subset}' not found in local dataset; returning empty Dataset")
    return Dataset.from_dict({})


def _merge_with_existing(
    existing: Dataset | DatasetDict, new: Dataset, subset: Optional[str]
) -> Dataset | DatasetDict:
    """Create a fresh Dataset/DatasetDict with *new* inserted (no in‑place mutation)."""

    if subset is None:
        return new

    if not isinstance(existing, DatasetDict):
        existing = DatasetDict({"default": existing})

    merged = DatasetDict({k: v for k, v in existing.items() if k != subset})
    merged[subset] = new
    return merged


def _safe_save_to_disk(dataset: Dataset | DatasetDict, path: Path) -> None:
    """Handle the dreaded “dataset can't overwrite itself” error transparently."""

    try:
        dataset.save_to_disk(str(path))
        return
    except PermissionError as exc:
        if "dataset can't overwrite itself" not in str(exc):
            raise

    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset.save_to_disk(tmp)
        shutil.rmtree(path, ignore_errors=True)
        shutil.copytree(tmp_path, path)

    logger.success("Dataset saved via temporary directory to avoid overwrite issue")


# ---------------------------------------------------------------------------
# Public API (stable signatures)
# ---------------------------------------------------------------------------


def custom_load_dataset(config: Dict[str, Any], subset: Optional[str] = None) -> Dataset:  # noqa: D401
    """Return *subset* of the dataset described by *config* – local first, then Hub."""

    settings = _extract_hf_settings(config)

    if settings.local_dir and settings.local_dir.exists():
        logger.info(f"Loading dataset from '{settings.local_dir}'")
        return _load_from_local(settings.local_dir, subset)

    if _is_offline_mode():
        logger.warning("Offline mode with no local data; returning empty Dataset")
        return Dataset.from_dict({})

    _validate_repo_id(settings)
    logger.info(f"Loading dataset from Hub: '{settings.repo_id}'")

    try:
        return load_dataset(settings.repo_id, name=subset, split="train")
    except ValueError as exc:
        if "BuilderConfig" in str(exc) and "not found" in str(exc):
            logger.warning(f"Subset '{subset}' not found on Hub; returning empty Dataset")
            return Dataset.from_dict({})
        raise


def custom_save_dataset(
    dataset: Dataset,
    config: Dict[str, Any],
    subset: Optional[str] = None,
    *,
    save_local: bool = True,
    push_to_hub: bool = True,
) -> None:  # noqa: D401
    """Save *dataset* according to *config*.

    * If ``save_local`` → writes to ``local_dataset_dir``.
    * If ``push_to_hub`` → uploads the (possibly concatenated) data to the Hub.
    """

    settings = _extract_hf_settings(config)

    offline = _is_offline_mode()
    if offline:
        save_local = True
        push_to_hub = False
        logger.info("Offline mode – will only save locally")

    if save_local and settings.local_dir:
        logger.info(f"Saving dataset to '{settings.local_dir}'")

        existing: Dataset | DatasetDict | None = None
        if settings.local_dir.exists():
            try:
                existing = load_from_disk(str(settings.local_dir))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not load existing dataset: {exc}")

        merged = (
            _merge_with_existing(existing, dataset, subset)
            if existing is not None
            else (DatasetDict({subset: dataset}) if subset else dataset)
        )

        settings.local_dir.parent.mkdir(parents=True, exist_ok=True)
        _safe_save_to_disk(merged, settings.local_dir)

    if settings.concat_if_exist and not offline:
        existing_remote = custom_load_dataset(config, subset)
        dataset = concatenate_datasets([existing_remote, dataset])
        logger.info("Concatenated new data with existing remote split")

    if push_to_hub and not offline:
        _validate_repo_id(settings)
        logger.info(f"Pushing dataset to Hub: '{settings.repo_id}'")
        dataset.push_to_hub(
            repo_id=settings.repo_id,
            private=settings.private,
            config_name=subset or "default",
        )
        logger.success(f"Dataset pushed to Hub: '{settings.repo_id}'")


def replace_dataset_columns(
    dataset: Dataset, columns_data: dict[str, list], preserve_metadata: bool = False
) -> Dataset:
    """
    Replace columns in a dataset by removing existing columns and adding new ones.

    This helper function handles the common pattern of:
    1. Removing existing columns (if they exist)
    2. Adding new columns with computed data

    Args:
        dataset: The input dataset to modify
        columns_data: Dictionary mapping column names to their data lists
        preserve_metadata: If True, attempts to preserve column metadata (not implemented)

    Returns:
        Updated dataset with replaced columns

    Note:
        Column metadata (types, features) is not preserved in the current implementation.
        New columns will have types inferred from the provided data.
    """
    # Remove existing columns to prevent duplication errors
    columns_to_replace = list(columns_data.keys())
    existing_columns_to_remove = [col for col in columns_to_replace if col in dataset.column_names]

    if existing_columns_to_remove:
        logger.info(f"Removing existing columns before adding new ones: {existing_columns_to_remove}")
        dataset = dataset.remove_columns(existing_columns_to_remove)

    # Add new columns
    for column_name, column_data in columns_data.items():
        dataset = dataset.add_column(column_name, column_data)

    return dataset
