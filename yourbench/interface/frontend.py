import hashlib
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import gradio as gr
from loguru import logger


class FileManager:
    def __init__(self):
        """Initialize file manager with storage directory and database."""
        self.storage_dir = Path("files")
        self.storage_dir.mkdir(exist_ok=True)

        self.db_path = Path("files/file_registry.db")
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with files table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    hash TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    upload_date TIMESTAMP NOT NULL,
                    file_path TEXT NOT NULL
                )
            """
            )

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def file_exists(self, file_hash: str) -> bool:
        """Check if a file with given hash exists in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_path FROM files WHERE hash = ?", (file_hash,)
            )
            result = cursor.fetchone()
            if result:
                return os.path.exists(result[0])
        return False

    def save_file(self, file: Union[str, Path]) -> tuple[bool, str]:
        """
        Save file to storage and register in database.
        Returns (is_new, file_hash).
        """
        temp_path = Path(file)
        file_hash = self._calculate_hash(temp_path)

        # Check if file already exists
        if self.file_exists(file_hash):
            return False, file_hash

        # Generate unique filename and save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{temp_path.name}"
        new_path = self.storage_dir / new_filename

        # Copy file to storage directory
        with open(temp_path, "rb") as src, open(new_path, "wb") as dst:
            dst.write(src.read())

        # Register in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO files (hash, filename, upload_date, file_path)
                VALUES (?, ?, ?, ?)
                """,
                (file_hash, temp_path.name, datetime.now(), str(new_path)),
            )

        return True, file_hash


def process_files(files: List[str]) -> str:
    """Process uploaded files and return status."""
    if not files:
        return "No files uploaded."

    file_manager = FileManager()
    results = []

    for file in files:
        try:
            is_new, file_hash = file_manager.save_file(file)
            filename = Path(file).name
            if is_new:
                results.append(f"✅ {filename} (New file)")
                logger.info(f"Saved new file: {filename} with hash: {file_hash}")
            else:
                results.append(f"ℹ️ {filename} (Duplicate)")
                logger.info(f"Duplicate file detected: {filename}")
        except Exception as e:
            results.append(f"❌ {filename} (Error: {str(e)})")
            logger.error(f"Error processing file {filename}: {str(e)}")

    return "\n".join(results)


def get_default_config() -> Dict[str, Any]:
    """Return default configuration parameters."""
    return {
        "configurations": {
            "huggingface": {
                "push_to_huggingface": False,
                "hf_organization": "",
                "set_hf_repo_visibility": "private",
                "concat_if_exists": True,
            },
            "model": {
                "model_name": "gpt-4",
                "model_type": "azure",
                "max_concurrent_requests": 32,
            },
        },
        "chunking_configuration": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "min_tokens": 256,
            "target_chunk_size": 512,
            "max_tokens": 1024,
            "similarity_threshold": 0.3,
            "device": "cpu",
        },
        "pairing_configuration": {
            "min_num_chunks": 2,
            "max_num_chunks": 5,
        },
        "cluster_configuration": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "alpha": 0.7,
            "batch_size": 128,
            "distance_threshold": 0.7,
            "top_k": 10,
            "lambda_val": 1.0,
            "w_max": 5.0,
        },
        "test_audience": "an expert in the field",
    }


def create_interface() -> gr.Blocks:
    """Create interface with file upload and configuration options."""

    theme = gr.themes.Base(
        primary_hue="indigo",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="white",
        block_background_fill="white",
        block_border_width="0px",
        container_radius="8px",
    )

    with gr.Blocks(theme=theme, title="YourBench") as interface:
        gr.Markdown("# 📁 YourBench")

        with gr.Row():
            # Left column for file upload
            with gr.Column(scale=1):
                files = gr.File(
                    label="Upload your documents",
                    file_count="multiple",
                    file_types=[".txt", ".pdf", ".md"],
                    height=200,
                )

                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    show_copy_button=True,
                    lines=5,
                )

                upload_btn = gr.Button("Process Files", size="lg", variant="primary")

            # Right column for configurations
            with gr.Column(scale=1):
                with gr.Accordion("🤖 Model Configuration", open=False):
                    model_name = gr.Textbox(
                        label="Model Name",
                        value="gpt-4",
                        info="Name of the model to use",
                    )
                    model_type = gr.Dropdown(
                        label="Model Type",
                        choices=["azure", "openai"],
                        value="azure",
                        info="Type of model service",
                    )
                    max_concurrent = gr.Number(
                        label="Max Concurrent Requests",
                        value=32,
                        minimum=1,
                        step=1,
                        info="Maximum number of concurrent requests",
                    )

                with gr.Accordion("📊 Chunking Configuration", open=False):
                    chunk_model = gr.Textbox(
                        label="Embedding Model",
                        value="sentence-transformers/all-MiniLM-L6-v2",
                    )
                    with gr.Row():
                        min_tokens = gr.Number(
                            label="Min Tokens",
                            value=256,
                            step=1,
                        )
                        target_tokens = gr.Number(
                            label="Target Tokens",
                            value=512,
                            step=1,
                        )
                        max_tokens = gr.Number(
                            label="Max Tokens",
                            value=1024,
                            step=1,
                        )
                    similarity_threshold = gr.Slider(
                        label="Similarity Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                    )
                    device = gr.Dropdown(
                        label="Device",
                        choices=["cpu", "cuda"],
                        value="cpu",
                    )

                with gr.Accordion("🔄 Multi-hop Configuration", open=False):
                    with gr.Row():
                        min_chunks = gr.Number(
                            label="Min Chunks",
                            value=2,
                            minimum=2,
                            step=1,
                        )
                        max_chunks = gr.Number(
                            label="Max Chunks",
                            value=5,
                            minimum=2,
                            step=1,
                        )

                with gr.Accordion("🎯 Clustering Configuration", open=False):
                    cluster_model = gr.Textbox(
                        label="Embedding Model",
                        value="sentence-transformers/all-MiniLM-L6-v2",
                    )
                    with gr.Row():
                        alpha = gr.Slider(
                            label="Alpha",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                        )
                        batch_size = gr.Number(
                            label="Batch Size",
                            value=128,
                            minimum=1,
                            step=1,
                        )
                    with gr.Row():
                        distance_threshold = gr.Slider(
                            label="Distance Threshold",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                        )
                        top_k = gr.Number(
                            label="Top K",
                            value=10,
                            minimum=1,
                            step=1,
                        )
                    with gr.Row():
                        lambda_val = gr.Number(
                            label="Lambda",
                            value=1.0,
                            step=0.1,
                        )
                        w_max = gr.Number(
                            label="W Max",
                            value=5.0,
                            step=0.1,
                        )

                with gr.Accordion("👥 Other Settings", open=False):
                    test_audience = gr.Textbox(
                        label="Test Audience",
                        value="an expert in the field",
                        info="Target audience for generated questions",
                    )

                    with gr.Row():
                        push_to_hf = gr.Checkbox(
                            label="Push to HuggingFace",
                            value=False,
                        )
                        concat_if_exists = gr.Checkbox(
                            label="Concatenate if Exists",
                            value=True,
                        )

                    hf_org = gr.Textbox(
                        label="HuggingFace Organization",
                        value="",
                        visible=False,
                    )

                    def toggle_hf_org(push_enabled):
                        return gr.update(visible=push_enabled)

                    push_to_hf.change(
                        fn=toggle_hf_org,
                        inputs=[push_to_hf],
                        outputs=[hf_org],
                    )

        upload_btn.click(
            fn=process_files,
            inputs=[files],
            outputs=[status],
        )

    return interface


def launch_frontend():
    """Launch the file upload interface."""
    interface = create_interface()
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
