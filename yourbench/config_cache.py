"""
Module for handling configuration caching.
"""

import json
from pathlib import Path


CACHE_FILE = Path.home() / ".yourbench" / "config_cache.json"


def get_last_config() -> str | None:
    """
    Retrieve the last used configuration file path.

    Returns:
        str | None: The path to the last used config file, or None if no cache exists
    """
    if not CACHE_FILE.exists():
        return None

    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            return cache.get("last_config")
    except (json.JSONDecodeError, IOError):
        return None


def save_last_config(config_path: str) -> None:
    """
    Save the last used configuration file path.

    Args:
        config_path (str): Path to the configuration file
    """
    # Ensure the cache directory exists
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        cache = {}
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)

        cache["last_config"] = config_path

        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except (json.JSONDecodeError, IOError):
        # If there's an error, try to create a new cache file
        with open(CACHE_FILE, "w") as f:
            json.dump({"last_config": config_path}, f)
