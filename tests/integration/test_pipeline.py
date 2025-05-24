import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from datasets import Dataset


# Fixture for temporary directory
@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


# Fixture for mock configuration
@pytest.fixture
def mock_config(temp_dir):
    return {
        "settings": {"debug": False},
        "hf_configuration": {
            "token": "fake_token",
            "hf_organization": "fake_org",
            "private": True,
            "hf_dataset_name": "fake_dataset",
            "concat_if_exist": False,
        },
        "local_dataset_dir": temp_dir,
        "model_list": [
            {
                "model_name": "fake_model",
                "provider": None,
                "api_key": "fake_key",
                "base_url": "http://localhost:8000/v1",
                "max_concurrent_requests": 1,
            }
        ],
        "model_roles": {
            "ingestion": ["fake_model"],
            "summarization": ["fake_model"],
            "chunking": ["fake_model"],
            "single_shot_question_generation": ["fake_model"],
            "multi_hop_question_generation": ["fake_model"],
        },
        "pipeline": {
            "ingestion": {
                "run": True,
                "source_documents_dir": os.path.join(temp_dir, "raw"),
                "output_dir": os.path.join(temp_dir, "processed"),
            },
            "upload_ingest_to_hub": {"run": False, "source_documents_dir": os.path.join(temp_dir, "processed")},
            "summarization": {"run": True},
            "chunking": {
                "run": True,
                "chunking_configuration": {
                    "l_max_tokens": 128,  # Only max_tokens is used now
                    "h_min": 2,
                    "h_max": 5,
                    "num_multihops_factor": 2,
                },
            },
            "single_shot_question_generation": {
                "run": True,
                "additional_instructions": "Generate questions to test a curious adult",
                "chunk_sampling": {"mode": "count", "value": 1, "random_seed": 123},
            },
            "multi_hop_question_generation": {
                "run": True,
                "additional_instructions": "Generate questions to test a curious adult",
                "chunk_sampling": {"mode": "count", "value": 1, "random_seed": 42},
            },
            "lighteval": {"run": True},
        },
    }


# Test for ingestion stage with mocked components
@pytest.mark.parametrize("mock_no_docs", [False, True])
def test_ingestion_stage(mock_config, temp_dir, mock_no_docs):
    """
    Test the ingestion stage of the YourBench pipeline.

    Verifies that the ingestion stage correctly processes source documents.
    """
    # Create test document structure
    raw_dir = mock_config["pipeline"]["ingestion"]["source_documents_dir"]
    output_dir = mock_config["pipeline"]["ingestion"]["output_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create a test document only if not testing the no-docs case
    if not mock_no_docs:
        with open(os.path.join(raw_dir, "test_doc.txt"), "w") as f:
            f.write("This is a test document for ingestion.")

    # Mock the core functionality instead of just the MarkItDown class
    with (
        patch("yourbench.pipeline.ingestion.MarkItDown") as mock_markitdown,
        patch("yourbench.pipeline.ingestion._convert_document_to_markdown") as mock_convert,
    ):
        # Configure mocks
        mock_markitdown_instance = MagicMock()
        mock_markitdown.return_value = mock_markitdown_instance
        mock_convert.return_value = True

        # Import the run function after mocking
        from yourbench.pipeline.ingestion import run

        # Run the ingestion stage
        run(mock_config)

        # Verify behavior
        if mock_no_docs:
            mock_convert.assert_not_called()
        else:
            mock_convert.assert_called()


# Test for summarization stage
def test_summarization_stage(mock_config):
    """
    Test the summarization stage of the YourBench pipeline.

    Verifies that summarization correctly calls inference and processes the results.
    """
    # Mock Dataset loading and saving
    mock_dataset = Dataset.from_dict({
        "document_id": ["doc1", "doc2"],
        "document_text": ["This is document 1", "This is document 2"],
        "document_filename": ["doc1.md", "doc2.md"],
    })

    # Setup mocks
    with (
        patch("yourbench.utils.dataset_engine.custom_load_dataset", return_value=mock_dataset) as mock_load,
        patch("yourbench.utils.dataset_engine.custom_save_dataset") as mock_save,
        patch("yourbench.pipeline.summarization.run_inference") as mock_run_inference,
        patch("yourbench.pipeline.summarization.extract_content_from_xml_tags") as mock_extract,
    ):
        # Configure mocks
        mock_run_inference.return_value = {
            "fake_model": [
                "<final_summary>Summary for doc1</final_summary>",
                "<final_summary>Summary for doc2</final_summary>",
            ]
        }
        mock_extract.side_effect = (
            lambda text, tag: f"Summary for doc{text.split('doc')[1].split('<')[0]}"
            if tag == "final_summary"
            else None
        )

        # Import the summarization run function
        from yourbench.pipeline.summarization import run

        # Run the summarization stage
        run(mock_config)

        # Verify the summarization stage ran as expected
        mock_load.assert_called_once()
        assert mock_run_inference.call_count == 1
        mock_save.assert_called_once()


# Test for chunking stage
def test_chunking_stage(mock_config):
    """
    Test the chunking stage of the YourBench pipeline.

    Verifies that documents are properly chunked according to the configuration.
    """
    # Mock Dataset loading and saving
    mock_dataset = Dataset.from_dict({
        "document_id": ["doc1", "doc2"],
        "document_text": [
            "This is document 1 with enough text to be chunked properly. " * 10,
            "This is document 2 which also has sufficient text for chunking. " * 10,
        ],
        "document_summary": ["Summary 1", "Summary 2"],
    })

    # Mock functions and dependencies
    with (
        patch("yourbench.utils.dataset_engine.custom_load_dataset", return_value=mock_dataset) as mock_load,
        patch("yourbench.utils.dataset_engine.custom_save_dataset") as mock_save,
        patch("yourbench.utils.chunking_utils.split_into_token_chunks") as mock_split,
    ):
        # Configure mock returns
        mock_split.return_value = ["Chunk 1", "Chunk 2"]

        # Import the chunking run function
        from yourbench.pipeline.chunking import run

        # Run the chunking stage
        run(mock_config)

        # Verify the chunking stage behavior
        mock_load.assert_called_once()
        assert mock_split.call_count == 2  # Called once for each document
        mock_save.assert_called_once()

        # Verify that the dataset was saved with the right subset
        saved_args = mock_save.call_args
        assert saved_args[1]["subset"] == "chunked"


# Test for single-shot question generation stage
def test_single_shot_question_generation_stage(mock_config):
    """
    Test the single-shot question generation stage of the YourBench pipeline.

    Verifies that questions are generated for single chunks of text.
    """
    # Mock dataset with chunks
    chunks = [{"chunk_id": "chunk1", "chunk_text": "This is chunk 1"}]
    mock_dataset = Dataset.from_dict({
        "document_id": ["doc1"],
        "document_summary": ["Document 1 summary"],
        "document_filename": ["doc1.md"],
        "chunks": [chunks],
    })

    # Setup mocks
    with (
        patch("yourbench.utils.dataset_engine.custom_load_dataset", return_value=mock_dataset) as mock_load,
        patch("yourbench.utils.dataset_engine.custom_save_dataset") as mock_save,
        patch("yourbench.utils.inference_engine.run_inference") as mock_run_inference,
        patch("yourbench.pipeline.single_shot_question_generation.parse_qa_pairs_from_response") as mock_parse,
    ):
        # Configure mocks
        mock_run_inference.return_value = {"fake_model": ["Question generation response"]}
        mock_parse.return_value = [
            {
                "question": "Test question?",
                "answer": "Test answer",
                "estimated_difficulty": 5,
                "question_type": "factual",
                "thought_process": "Reasoning",
                "citations": ["citation"],
            }
        ]

        # Import run function
        from yourbench.pipeline.single_shot_question_generation import run

        # Run the stage
        run(mock_config)

        # Verify behavior
        mock_load.assert_called_once()
        mock_run_inference.assert_called_once()
        mock_parse.assert_called_once()
        mock_save.assert_called_once()


# Test for multi-hop question generation stage
def test_multi_hop_question_generation_stage(mock_config):
    """
    Test the multi-hop question generation stage of the YourBench pipeline.

    Verifies that questions are generated requiring reasoning across multiple chunks.
    """
    # Mock dataset with chunks and valid multihop_chunks format
    chunks = [{"chunk_id": "chunk1", "chunk_text": "This is chunk 1"}]
    # Updated format for multihop_chunks based on the refactored code
    multihop_chunks = [
        {
            "chunk_ids": ["chunk1", "chunk2"],
            "chunks_text": ["Text 1", "Text 2"],
        }
    ]
    mock_dataset = Dataset.from_dict({
        "document_id": ["doc1"],
        "document_summary": ["Document 1 summary"],
        "chunks": [chunks],
        "multihop_chunks": [multihop_chunks],
    })

    # Setup mocks
    with (
        patch("yourbench.utils.dataset_engine.custom_load_dataset", return_value=mock_dataset) as mock_load,
        patch("yourbench.utils.dataset_engine.custom_save_dataset") as mock_save,
        patch("yourbench.utils.inference_engine.run_inference") as mock_run_inference,
        patch("yourbench.pipeline.multi_hop_question_generation.parse_qa_pairs_from_response") as mock_parse,
        # Mock the chunk sampling function to bypass the empty multihop check
        patch("yourbench.pipeline.multi_hop_question_generation._multihop_chunk_sampling_and_calls") as mock_sampling,
    ):
        # Configure mocks
        mock_run_inference.return_value = {"fake_model": ["Multi-hop question generation response"]}
        mock_parse.return_value = [
            {
                "question": "Multi-hop test question?",
                "answer": "Multi-hop test answer",
                "estimated_difficulty": 7,
                "question_type": "reasoning",
                "thought_process": "Complex reasoning",
                "citations": ["citation1", "citation2"],
            }
        ]
        mock_sampling.return_value = (
            [MagicMock()],  # Mock inference calls list
            [(0, "doc1", ["chunk1", "chunk2"])],  # Mock call index mapping
        )

        # Import run function
        from yourbench.pipeline.multi_hop_question_generation import run

        # Run the stage
        run(mock_config)

        # Verify behavior
        mock_load.assert_called_once()
        mock_run_inference.assert_called_once()
        mock_save.assert_called_once()


# Test for lighteval stage
def test_lighteval_stage(mock_config):
    """
    Test the lighteval stage of the YourBench pipeline.

    Verifies that the stage combines questions into a unified dataset for evaluation.
    """
    # Mock single-shot and multi-hop datasets
    single_shot_ds = Dataset.from_dict({
        "document_id": ["doc1"],
        "chunk_id": ["chunk1"],
        "question": ["Single-shot question?"],
        "self_answer": ["Single-shot answer"],
        "estimated_difficulty": [5],
        "self_assessed_question_type": ["factual"],
        "generating_model": ["fake_model"],
        "additional_instructions": ["Generate questions"],
    })

    multi_hop_ds = Dataset.from_dict({
        "document_id": ["doc1"],
        "source_chunk_ids": [["chunk1", "chunk2"]],
        "question": ["Multi-hop question?"],
        "self_answer": ["Multi-hop answer"],
        "estimated_difficulty": [7],
        "self_assessed_question_type": ["reasoning"],
        "generating_model": ["fake_model"],
        "additional_instructions": ["Generate questions"],
    })

    chunked_ds = Dataset.from_dict({
        "document_id": ["doc1"],
        "document_text": ["Full document text"],
        "chunks": [[{"chunk_id": "chunk1", "chunk_text": "Chunk 1 text"}]],
    })

    summarized_ds = Dataset.from_dict({"document_id": ["doc1"], "document_summary": ["Document 1 summary"]})

    # Setup mocks
    with (
        patch("yourbench.utils.dataset_engine.custom_load_dataset") as mock_load,
        patch("yourbench.utils.dataset_engine.custom_save_dataset") as mock_save,
    ):
        # Configure mock to return different datasets based on the subset parameter
        def load_dataset_side_effect(config, subset):
            if subset == "single_shot_questions":
                return single_shot_ds
            elif subset == "multi_hop_questions":
                return multi_hop_ds
            elif subset == "chunked":
                return chunked_ds
            elif subset == "summarized":
                return summarized_ds
            return Dataset.from_dict({})

        mock_load.side_effect = load_dataset_side_effect

        # Import run function
        from yourbench.pipeline.lighteval import run

        # Run the stage
        run(mock_config)

        # Verify behavior
        assert mock_load.call_count == 4
        mock_save.assert_called_once()
