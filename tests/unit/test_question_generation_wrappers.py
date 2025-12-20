"""Tests for question generation wrapper modules."""

from unittest.mock import Mock, patch


class TestSingleShotWrapper:
    """Tests for single_shot module run() function."""

    def test_run_delegates_to_core(self):
        """Verify run() calls the core run_single_shot function."""
        mock_config = Mock()
        with patch("yourbench.pipeline.question_generation.single_shot.run_single_shot") as mock_core:
            from yourbench.pipeline.question_generation import single_shot

            single_shot.run(mock_config)
            mock_core.assert_called_once_with(mock_config)


class TestMultiHopWrapper:
    """Tests for multi_hop module run() function."""

    def test_run_delegates_to_core(self):
        """Verify run() calls the core run_multi_hop function."""
        mock_config = Mock()
        with patch("yourbench.pipeline.question_generation.multi_hop.run_multi_hop") as mock_core:
            from yourbench.pipeline.question_generation import multi_hop

            multi_hop.run(mock_config)
            mock_core.assert_called_once_with(mock_config)


class TestCrossDocumentWrapper:
    """Tests for cross_document module run() function."""

    def test_run_delegates_to_core(self):
        """Verify run() calls the core run_cross_document function."""
        mock_config = Mock()
        with patch("yourbench.pipeline.question_generation.cross_document.run_cross_document") as mock_core:
            from yourbench.pipeline.question_generation import cross_document

            cross_document.run(mock_config)
            mock_core.assert_called_once_with(mock_config)
