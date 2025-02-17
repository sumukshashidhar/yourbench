import json
from pathlib import Path
from typing import Dict, Union

import yaml
from loguru import logger


def load_all_prompts(prompts_dir: Union[str, Path] = "yourbench/prompts") -> Dict[str, str]:
    """
    Recursively load all prompt files from the prompts directory and its subdirectories.
    Returns a dictionary mapping prompt names to their contents.

    Prompt naming convention:
    - Root directory: prompt_name
    - Subdirectories: subdir.prompt_name or subdir.subsubdir.prompt_name

    Supports .txt, .md, .yaml, and .json files
    """
    prompts_dir = Path(prompts_dir)
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    prompts = {}

    for file_path in prompts_dir.rglob("*"):
        if not file_path.is_file() or file_path.suffix not in [
            ".txt",
            ".md",
            ".yaml",
            ".json",
        ]:
            continue

        # Create prompt name from relative path
        relative_path = file_path.relative_to(prompts_dir)
        parts = list(relative_path.parts[:-1])  # Get directory parts
        parts.append(file_path.stem)  # Add filename without extension
        prompt_name = ".".join(parts)

        # Load prompt content based on file type
        if file_path.suffix in [".txt", ".md"]:
            content = file_path.read_text(encoding="utf-8")
        elif file_path.suffix == ".yaml":
            content = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        elif file_path.suffix == ".json":
            content = json.load(file_path.open(encoding="utf-8"))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        prompts[prompt_name] = content

    return prompts


def load_prompt(prompt_name: str) -> str:
    """Load a single prompt by name"""
    prompts = load_all_prompts()
    logger.debug("Available prompts:", prompts.keys())
    if prompt_name not in prompts:
        raise ValueError(f"Prompt {prompt_name} not found")
    return prompts[prompt_name]


if __name__ == "__main__":
    print(load_all_prompts())
