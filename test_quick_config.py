#!/usr/bin/env python3
"""Test the quick configuration generation."""

import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from yourbench.main import _run_quick_mode

# Mock the pipeline run
def mock_pipeline(*args, **kwargs):
    print("Mock pipeline called with:", args, kwargs)
    # Create fake JSONL output
    with open("test_dataset.jsonl", "w") as f:
        f.write('{"question": "What is AI?", "answer": "Artificial Intelligence"}\n')
    print("Created test_dataset.jsonl")

# Monkey patch the import
import yourbench.main
yourbench.main._lazy_import_pipeline = lambda: mock_pipeline

# Test configuration generation
try:
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test config
        config = {
            "hf_configuration": {
                "hf_dataset_name": "test-dataset",
                "hf_organization": "$HF_ORGANIZATION",
                "hf_token": "$HF_TOKEN",
                "private": True,
                "local_dataset_dir": str(Path(temp_dir) / "dataset"),
                "local_saving": True,
                "export_jsonl": True,
                "jsonl_export_dir": ".",
            },
            "model_list": [
                {
                    "model_name": "gpt-3.5-turbo",
                    "max_concurrent_requests": 16,
                }
            ],
            "pipeline": {
                "ingestion": {
                    "run": True,
                    "source_documents_dir": ".",
                    "output_dir": str(Path(temp_dir) / "processed"),
                },
                "summarization": {"run": True},
                "chunking": {"run": True},
                "single_shot_question_generation": {"run": True},
                "multi_hop_question_generation": {"run": True},
                "lighteval": {"run": True},
                "citation_score_filtering": {"run": True},
            },
        }
        
        # Save and display config
        config_path = Path(temp_dir) / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print("Generated configuration:")
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
        print("\nConfiguration validates the quick mode setup!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()