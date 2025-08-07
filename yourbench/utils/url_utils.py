"""Utility functions for URL and API key handling."""

import os
from urllib.parse import urlparse


def get_api_key_for_url(base_url: str | None) -> str:
    """
    Determine the appropriate API key environment variable based on the base URL.

    Args:
        base_url: The base URL for the API endpoint

    Returns:
        The environment variable name to use (e.g., "$OPENAI_API_KEY", "$HF_TOKEN")
    """
    if not base_url:
        return "$HF_TOKEN"

    hostname = get_hostname_from_url(base_url)

    if hostname.endswith(".openai.com"):
        return "$OPENAI_API_KEY"
    elif hostname.endswith(".anthropic.com"):
        return "$ANTHROPIC_API_KEY"
    else:
        return "$HF_TOKEN"


def get_hostname_from_url(url: str) -> str:
    """
    Extract and normalize the hostname from a URL.

    Args:
        url: The URL to parse

    Returns:
        The lowercase hostname, or empty string if parsing fails
    """
    parsed_url = urlparse(url)
    return (parsed_url.hostname or "").lower()


def is_openai_url(base_url: str | None) -> bool:
    """Check if a URL is an OpenAI API endpoint."""
    if not base_url:
        return False
    hostname = get_hostname_from_url(base_url)
    return hostname.endswith(".openai.com")


def is_anthropic_url(base_url: str | None) -> bool:
    """Check if a URL is an Anthropic API endpoint."""
    if not base_url:
        return False
    hostname = get_hostname_from_url(base_url)
    return hostname.endswith(".anthropic.com")


def validate_api_key_for_url(
    base_url: str | None, api_key: str | None, model_name: str | None = None
) -> tuple[bool, str | None]:
    """
    Validate that the appropriate API key is set for the given URL.

    Args:
        base_url: The base URL for the API endpoint
        api_key: The API key environment variable name (e.g., "$OPENAI_API_KEY")
        model_name: Optional model name for better error messages

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    if is_openai_url(base_url):
        if api_key == "$OPENAI_API_KEY" and not os.getenv("OPENAI_API_KEY"):
            return False, (
                "OPENAI_API_KEY environment variable is required for OpenAI base URL. "
                "Please set it with: export OPENAI_API_KEY=your_token"
            )
    elif is_anthropic_url(base_url):
        if api_key == "$ANTHROPIC_API_KEY" and not os.getenv("ANTHROPIC_API_KEY"):
            return False, (
                "ANTHROPIC_API_KEY environment variable is required for Anthropic base URL. "
                "Please set it with: export ANTHROPIC_API_KEY=your_token"
            )
    elif not base_url:
        # No base_url, default to HF_TOKEN validation
        if api_key == "$HF_TOKEN" and not os.getenv("HF_TOKEN"):
            model_info = f" for model '{model_name}'" if model_name else ""
            return False, (
                f"HF_TOKEN environment variable is required{model_info} since base_url is not set. "
                "Please set it with: export HF_TOKEN=your_token"
            )
    # For other base URLs with custom API keys, we don't strictly validate
    return True, None
