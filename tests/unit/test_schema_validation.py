"""Unit tests for schema validation."""

import pytest
from pydantic import ValidationError

from yourbench.conf.schema import (
    ModelConfig,
    ChunkingConfig,
    CrossDocConfig,
    SummarizationConfig,
    CitationFilteringConfig,
)


class TestSummarizationConfig:
    def test_valid_config(self):
        cfg = SummarizationConfig(max_tokens=1000, token_overlap=100)
        assert cfg.max_tokens == 1000

    def test_invalid_max_tokens(self):
        with pytest.raises(ValidationError, match="max_tokens must be > 0"):
            SummarizationConfig(max_tokens=0)

    def test_invalid_token_overlap(self):
        with pytest.raises(ValidationError, match="token_overlap must be >= 0"):
            SummarizationConfig(token_overlap=-1)

    def test_overlap_exceeds_max_tokens(self):
        with pytest.raises(ValidationError, match="token_overlap.*must be < max_tokens"):
            SummarizationConfig(max_tokens=100, token_overlap=100)


class TestChunkingConfig:
    def test_valid_config(self):
        cfg = ChunkingConfig(h_min=2, h_max=5)
        assert cfg.h_min == 2

    def test_invalid_h_min(self):
        with pytest.raises(ValidationError, match="h_min must be >= 1"):
            ChunkingConfig(h_min=0)

    def test_h_max_less_than_h_min(self):
        with pytest.raises(ValidationError, match="h_max.*must be >= h_min"):
            ChunkingConfig(h_min=5, h_max=2)


class TestCrossDocConfig:
    def test_valid_config(self):
        cfg = CrossDocConfig(num_docs_per_combination=[2, 5])
        assert cfg.max_combinations == 100

    def test_invalid_num_docs_length(self):
        with pytest.raises(ValidationError, match="2 elements"):
            CrossDocConfig(num_docs_per_combination=[2])

    def test_invalid_min_docs(self):
        with pytest.raises(ValidationError, match="must be >= 2"):
            CrossDocConfig(num_docs_per_combination=[1, 5])

    def test_max_less_than_min(self):
        with pytest.raises(ValidationError, match="must be >="):
            CrossDocConfig(num_docs_per_combination=[5, 2])


class TestModelConfig:
    def test_valid_config(self):
        cfg = ModelConfig(model_name="test", max_concurrent_requests=16)
        assert cfg.max_concurrent_requests == 16

    def test_invalid_concurrency(self):
        with pytest.raises(ValidationError, match="max_concurrent_requests must be >= 1"):
            ModelConfig(max_concurrent_requests=0)


class TestCitationFilteringConfig:
    def test_valid_config(self):
        cfg = CitationFilteringConfig(alpha=0.5, beta=0.5)
        assert cfg.alpha == 0.5

    def test_invalid_alpha(self):
        with pytest.raises(ValidationError, match="alpha must be in"):
            CitationFilteringConfig(alpha=1.5)

    def test_invalid_beta(self):
        with pytest.raises(ValidationError, match="beta must be in"):
            CitationFilteringConfig(beta=-0.1)
