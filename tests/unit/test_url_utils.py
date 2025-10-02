import os
from unittest.mock import patch

from yourbench.utils.url_utils import (
    get_api_key_for_url,
    validate_api_key_for_url,
    is_openai_url,
    is_anthropic_url,
    is_openrouter_url,
)


def test_get_api_key_for_url_defaults():
    assert get_api_key_for_url(None) == "$HF_TOKEN"
    assert get_api_key_for_url("") == "$HF_TOKEN"


def test_get_api_key_for_known_hosts():
    assert get_api_key_for_url("https://api.openai.com/v1") == "$OPENAI_API_KEY"
    assert get_api_key_for_url("https://api.anthropic.com/v1") == "$ANTHROPIC_API_KEY"
    assert get_api_key_for_url("https://openrouter.ai/api/v1") == "$OPENROUTER_API_KEY"


def test_url_helpers():
    assert is_openai_url("https://api.openai.com/v1") is True
    assert is_anthropic_url("https://api.anthropic.com/v1") is True
    assert is_openrouter_url("https://openrouter.ai/api/v1") is True


def test_validate_api_key_for_url_openrouter_missing_env():
    with patch.dict(os.environ, {}, clear=True):
        valid, msg = validate_api_key_for_url("https://openrouter.ai/api/v1", "$OPENROUTER_API_KEY", "test-model")
        assert not valid
        assert "OPENROUTER_API_KEY" in (msg or "")


def test_validate_api_key_for_url_openrouter_present_env():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or_test"}, clear=True):
        valid, msg = validate_api_key_for_url("https://openrouter.ai/api/v1", "$OPENROUTER_API_KEY", "test-model")
        assert valid
        assert msg is None
