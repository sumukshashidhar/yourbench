"""Prompt loading utilities for Hydra configs."""

from importlib.resources import files
from pathlib import Path

from loguru import logger


DEFAULT_PROMPTS = {
    "pdf_llm_prompt": "ingestion/pdf_llm_prompt.md",
    "summarization_user_prompt": "summarization/summarization_user_prompt.md",
    "combine_summaries_user_prompt": "summarization/combine_summaries_user_prompt.md",
    "single_shot_system_prompt": "question_generation/single_shot_system_prompt.md",
    "single_shot_system_prompt_multi": "question_generation/single_shot_system_prompt_multi.md",
    "single_shot_user_prompt": "question_generation/single_shot_user_prompt.md",
    "multi_hop_system_prompt": "question_generation/multi_hop_system_prompt.md",
    "multi_hop_system_prompt_multi": "question_generation/multi_hop_system_prompt_multi.md",
    "multi_hop_user_prompt": "question_generation/multi_hop_user_prompt.md",
    "question_rewriting_system_prompt": "question_rewriting/question_rewriting_system_prompt.md",
    "question_rewriting_user_prompt": "question_rewriting/question_rewriting_user_prompt.md",
}


def load_prompt_from_package(package_path: str) -> str | None:
    """Load prompt content from package resources."""
    try:
        prompts_files = files("yourbench.prompts")
        parts = package_path.split("/")
        current = prompts_files

        for part in parts[:-1]:
            current = current / part

        file_resource = current / parts[-1]
        if file_resource.is_file():
            return file_resource.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.debug(f"Failed to load prompt from package {package_path}: {e}")
    return None


def load_prompt(value: str, default_package_path: str = "") -> str:
    """Load prompt from value, file, or package default."""
    if not value:
        if default_package_path:
            return load_prompt_from_package(default_package_path) or ""
        return ""

    if "\n" in value or len(value) > 300:
        return value

    path = Path(value)
    if path.suffix.lower() in {".md", ".txt", ".prompt"}:
        if path.exists():
            try:
                return path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                logger.warning(f"Failed to read prompt file {path}: {exc}")
        if content := load_prompt_from_package(value):
            return content
        logger.warning(f"Prompt file not found: {path}")

    return value
