"""Utility functions for URL and API key handling."""

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
