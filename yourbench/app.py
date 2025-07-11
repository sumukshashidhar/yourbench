import io
import os
import re
import sys
import atexit
import shutil
import subprocess

import yaml
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from loguru import logger


load_dotenv()
logger.remove()
logger.add(sys.stderr, level="INFO")


STAGES = [
    "ingestion",
    "summarization",
    "chunking",
    "single_shot_question_generation",
    "multi_hop_question_generation",
    "lighteval",
    "citation_score_filtering",
]

STAGE_DISPLAY_MAP = {
    "ingestion": "Process Input Docs",
    "summarization": "Summarize Documents",
    "chunking": "Chunk Documents",
    "single_shot_question_generation": "Generate Single Shot Questions",
    "multi_hop_question_generation": "Generate Multi Hop Questions",
    "lighteval": "Generate Lighteval Subset",
    "citation_score_filtering": "Citation Score Filtering",
}

HF_DEFAULTS = {
    "hf_token": os.getenv("HF_TOKEN", ""),
    "hf_organization": os.getenv("HF_ORGANIZATION", ""),
    "hf_dataset_name": "yourbench_dataset",
    "private": True,
    "concat_if_exist": False,
    "upload_card": True,
}

PROVIDERS = {
    "Cerebras": "cerebras",
    "Cohere": "cohere",
    "Fal AI": "fal-ai",
    "Featherless AI": "featherless-ai",
    "Fireworks": "fireworks-ai",
    "Groq": "groq",
    "HF Inference": "hf-inference",
    "Hyperbolic": "hyperbolic",
    "Nebius": "nebius",
    "Novita": "novita",
    "Nscale": "nscale",
    "Replicate": "replicate",
    "SambaNova": "sambanova",
    "Together": "together",
}

RESULTS_PROCESSED_DIR = os.path.join("results", "processed")
LOCAL_DATASETS_DIR = os.path.join("results", "datasets")

# Session state for localhost usage
WORKING_DIR = os.path.join(os.getcwd(), "yourbench_workspace")
os.makedirs(WORKING_DIR, exist_ok=True)

SESSION_STATE = {
    "working_dir": WORKING_DIR,
    "subprocess": None,
    "files": [],
    "config": None,
    "pipeline_completed": False,
}


# Cleanup function for subprocess management
def cleanup_session():
    if SESSION_STATE["subprocess"]:
        SESSION_STATE["subprocess"].stop()


atexit.register(cleanup_session)


def validate_file_upload(files):
    """Validate uploaded files"""
    if not files:
        return False, "No files uploaded"

    allowed_extensions = {".txt", ".md", ".pdf", ".html"}
    for file in files:
        _, ext = os.path.splitext(file.name.lower())
        if ext not in allowed_extensions:
            return False, f"File {file.name} has unsupported extension {ext}"

    return True, "Files valid"


def save_uploaded_files(files):
    if not files:
        return "❌ No files to upload."

    raw_dir = os.path.join(SESSION_STATE["working_dir"], "raw")
    os.makedirs(raw_dir, exist_ok=True)
    uploaded = []

    for file in files:
        dest = os.path.join(raw_dir, os.path.basename(file.name))
        shutil.copy(file.name, dest)
        uploaded.append(os.path.basename(file.name))

    SESSION_STATE["files"] = uploaded
    return f"✅ Uploaded {len(uploaded)} files."


def clear_uploaded_files():
    raw_dir = os.path.join(SESSION_STATE["working_dir"], "raw")
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir, ignore_errors=True)
    SESSION_STATE["files"] = []
    return "🧹 Uploads cleared.", gr.update(value=None)


def save_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


class SubprocessManager:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.config_path = os.path.join(self.working_dir, "config.yaml")
        self.process = None
        self.output_stream = io.StringIO()
        self.exit_code = None
        self.completed = False

    def start(self):
        if self.is_running():
            return False, "Process already running"

        if not os.path.exists(self.config_path):
            return False, "Config file not found"

        self.output_stream = io.StringIO()
        self.completed = False

        save_dirs(RESULTS_PROCESSED_DIR, LOCAL_DATASETS_DIR)

        try:
            self.process = subprocess.Popen(
                ["uv", "run", "yourbench", "run", "--config", self.config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            os.set_blocking(self.process.stdout.fileno(), False)
            return True, "Process started successfully"
        except Exception as e:
            logger.error(f"Error starting subprocess: {e}")
            return False, f"Error starting process: {str(e)}"

    def is_running(self):
        if not self.process:
            return False
        return self.process.poll() is None

    def stop(self):
        if self.is_running():
            self.process.terminate()
            try:
                self.exit_code = self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.exit_code = self.process.wait()

    def read_output(self):
        if not self.process or not self.process.stdout:
            return "", [], False

        try:
            while True:
                line = self.process.stdout.readline()
                if line:
                    self.output_stream.write(line)
                else:
                    break
        except BlockingIOError:
            pass

        # Check if process completed
        if not self.is_running() and not self.completed:
            self.completed = True
            self.exit_code = self.process.poll()

        full_output = self.output_stream.getvalue()
        stages_completed = list(set(re.findall(r"Completed stage: '([^']*)'", full_output)))
        stages_display = [STAGE_DISPLAY_MAP[s] for s in stages_completed if s in STAGE_DISPLAY_MAP]

        return full_output, stages_display, self.completed


def validate_config_inputs(table_data, ingestion_model, summarization_model, single_model, multi_model):
    """Validate configuration inputs"""
    errors = []

    # Check if models are defined
    if not table_data or len(table_data) == 0:
        errors.append("At least one model must be defined")

    # Check if all required roles have models assigned
    required_models = [ingestion_model, summarization_model, single_model, multi_model]
    model_names = [row[0] for row in table_data if isinstance(row, list) and len(row) > 0]

    for i, model in enumerate(required_models):
        role_names = ["ingestion", "summarization", "single_shot_question_generation", "multi_hop_question_generation"]
        if not model:
            errors.append(f"Model for {role_names[i]} must be selected")
        elif model not in model_names:
            errors.append(f"Model '{model}' for {role_names[i]} is not in the model list")

    return errors


def launch_ui():
    with gr.Blocks(title="YourBench") as demo:
        gr.Markdown("# 🚀 YourBench")
        if not HF_DEFAULTS["hf_token"]:
            gr.Markdown("⚠️ **HF_TOKEN not set in `.env` file.**")

        # Add output locations info
        with gr.Row():
            gr.Markdown("""
            ### 📁 Output Locations
            - **Processed Data**: `results/processed/`
            - **Local Datasets**: `results/datasets/` (when local saving is enabled)
            """)

        with gr.Tabs():
            with gr.Tab("Upload Documents & Construct Config"):
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(
                            file_count="multiple",
                            file_types=[".txt", ".md", ".pdf", ".html"],
                            label="Upload Source Documents",
                        )
                        upload_log = gr.Textbox(label="Upload Log")
                        clear_btn = gr.Button("🧹 Clear Uploads")
                        file_input.upload(save_uploaded_files, inputs=file_input, outputs=upload_log)
                        clear_btn.click(fn=clear_uploaded_files, outputs=[upload_log, file_input])

                    with gr.Column():
                        hf_token = gr.Textbox(label="HF Token", value=HF_DEFAULTS["hf_token"], type="password")
                        hf_org = gr.Textbox(label="HF Organization", value=HF_DEFAULTS["hf_organization"])
                        hf_dataset_name = gr.Textbox(label="Dataset Name", value=HF_DEFAULTS["hf_dataset_name"])
                        private = gr.Checkbox(label="Private Dataset", value=HF_DEFAULTS["private"])
                        concat = gr.Checkbox(label="Concat if Exists", value=HF_DEFAULTS["concat_if_exist"])
                        upload_card = gr.Checkbox(label="Generate Dataset Card", value=HF_DEFAULTS["upload_card"])
                        local_saving = gr.Checkbox(label="Save Locally", value=False)

                model_name_input = gr.Textbox(label="Model Name", placeholder="e.g. meta-llama/Llama-3.3-70B-Instruct")
                provider_dropdown = gr.Dropdown(
                    label="Provider", choices=list(PROVIDERS.keys()), value="HF Inference", allow_custom_value=True
                )
                add_model_btn = gr.Button("➕ Add Model")
                model_table = gr.Dataframe(
                    headers=["Model Name", "Provider"],
                    datatype=["str", "str"],
                    row_count=(1, "dynamic"),
                    interactive=True,
                    value=[],  # Initialize with empty list
                )
                remove_model_btn = gr.Button("🗑️ Remove Last Model")

                def add_model(model_name, provider_key, table_data):
                    if not model_name.strip():
                        raise gr.Error("Model name is required.")

                    # Convert table_data to list format
                    if isinstance(table_data, pd.DataFrame):
                        new_data = table_data.values.tolist()
                    elif table_data is None:
                        new_data = []
                    else:
                        new_data = table_data

                    # Check for duplicates
                    for row in new_data:
                        if isinstance(row, list) and len(row) > 0 and row[0] == model_name:
                            raise gr.Error(f"Model '{model_name}' already exists.")

                    new_data.append([model_name, provider_key])
                    return new_data, "", "HF Inference"

                def remove_model(table_data):
                    if isinstance(table_data, pd.DataFrame):
                        new_data = table_data.values.tolist()
                    elif table_data is None:
                        new_data = []
                    else:
                        new_data = table_data

                    if new_data:
                        new_data = new_data[:-1]
                    return new_data

                add_model_btn.click(
                    add_model,
                    inputs=[model_name_input, provider_dropdown, model_table],
                    outputs=[model_table, model_name_input, provider_dropdown],
                )
                remove_model_btn.click(remove_model, inputs=[model_table], outputs=[model_table])

                role_fields = {}
                for role in [
                    "ingestion",
                    "summarization",
                    "single_shot_question_generation",
                    "multi_hop_question_generation",
                ]:
                    role_fields[role] = gr.Dropdown(
                        label=f"Model for {role.replace('_', ' ').title()}", interactive=True, choices=[]
                    )

                def update_roles(table_data):
                    if isinstance(table_data, pd.DataFrame):
                        data = table_data.values.tolist()
                    elif table_data is None:
                        data = []
                    else:
                        data = table_data

                    names = [
                        row[0] for row in data if isinstance(row, list) and len(row) > 0 and row[0] and row[0].strip()
                    ]

                    default_value = names[0] if len(names) == 1 else None
                    return [gr.update(choices=names, value=default_value) for _ in role_fields.values()]

                model_table.change(update_roles, inputs=[model_table], outputs=list(role_fields.values()))

                question_mode = gr.Dropdown(
                    label="Question Mode", choices=["open-ended", "multi-choice"], value="multi-choice"
                )
                additional_instructions = gr.Textbox(
                    label="Additional Instructions", value="Ask deep, evidence-based questions from the document."
                )
                cross_doc_enable = gr.Checkbox(label="Enable Cross-Document Question Generation", value=False)

                build_btn = gr.Button("🛠️ Build Config")
                config_display = gr.Code(label="Generated Config", language="yaml")
                save_config_btn = gr.Button("💾 Save Changes")
                config_file_output = gr.File(label="📥 Download Config")

                def build_config(
                    hf_token,
                    hf_org,
                    hf_dataset_name,
                    private,
                    concat,
                    upload_card,
                    local_saving,
                    table_data,
                    ingestion_model,
                    summarization_model,
                    single_model,
                    multi_model,
                    question_mode,
                    additional_instructions,
                    cross_doc_enable,
                ):
                    # Convert table_data to list format
                    if isinstance(table_data, pd.DataFrame):
                        rows = table_data.values.tolist()
                    elif table_data is None:
                        rows = []
                    else:
                        rows = table_data

                    # Basic validation
                    if not rows:
                        raise gr.Error("At least one model must be defined")

                    if not all([ingestion_model, summarization_model, single_model, multi_model]):
                        raise gr.Error("All model roles must be assigned")

                    model_list = []
                    for row in rows:
                        if isinstance(row, list) and len(row) >= 2 and row[0] and row[1]:
                            model_list.append({
                                "model_name": row[0],
                                "provider": PROVIDERS.get(row[1], row[1]),
                            })

                    # Use fixed safe paths
                    local_dataset_dir = LOCAL_DATASETS_DIR if local_saving else None

                    if local_saving:
                        os.makedirs(LOCAL_DATASETS_DIR, exist_ok=True)
                        logger.info(f"Local datasets will be saved to: {LOCAL_DATASETS_DIR}")

                    config = {
                        "settings": {"debug": False},
                        "hf_configuration": {
                            "token": "$HF_TOKEN",
                            "hf_organization": hf_org,
                            "hf_dataset_name": hf_dataset_name,
                            "private": private,
                            "upload_card": upload_card,
                            "concat_if_exist": concat,
                            "local_saving": local_saving,
                            "local_dataset_dir": local_dataset_dir,
                        },
                        "local_dataset_dir": local_dataset_dir,
                        "model_list": model_list,
                        "model_roles": {
                            "ingestion": [ingestion_model],
                            "summarization": [summarization_model],
                            "single_shot_question_generation": [single_model],
                            "multi_hop_question_generation": [multi_model],
                        },
                        "pipeline": {
                            "ingestion": {
                                "run": True,
                                "source_documents_dir": os.path.join(SESSION_STATE["working_dir"], "raw"),
                                "output_dir": "results/processed",
                                "upload_to_hub": True,
                                "llm_ingestion": False,
                            },
                            "summarization": {
                                "run": True,
                                "max_tokens": 16384,
                                "token_overlap": 128,
                                "encoding_name": "cl100k_base",
                            },
                            "chunking": {
                                "run": True,
                                "chunking_configuration": {
                                    "l_max_tokens": 512,
                                    "token_overlap": 64,
                                    "encoding_name": "cl100k_base",
                                },
                            },
                            "single_shot_question_generation": {
                                "run": True,
                                "question_mode": question_mode,
                                "additional_instructions": additional_instructions,
                                "chunk_sampling": {"mode": "count", "value": 5, "random_seed": 49},
                            },
                            "multi_hop_question_generation": {
                                "run": True,
                                "question_mode": question_mode,
                                "additional_instructions": additional_instructions,
                                "cross_document": {
                                    "enable": cross_doc_enable,
                                    "max_combinations": 5,
                                    "chunks_per_document": 1,
                                },
                                "chunk_sampling": {"mode": "percentage", "value": 0.3, "random_seed": 42},
                            },
                            "lighteval": {"run": True},
                            "citation_score_filtering": {"run": True},
                        },
                    }

                    config_yaml = yaml.dump(config, sort_keys=False)
                    config_path = os.path.join(SESSION_STATE["working_dir"], "config.yaml")
                    with open(config_path, "w") as f:
                        f.write(config_yaml)

                    SESSION_STATE["config"] = config
                    return config_yaml, gr.update(value=config_path)

                build_btn.click(
                    build_config,
                    inputs=[
                        hf_token,
                        hf_org,
                        hf_dataset_name,
                        private,
                        concat,
                        upload_card,
                        local_saving,
                        model_table,
                    ]
                    + list(role_fields.values())
                    + [question_mode, additional_instructions, cross_doc_enable],
                    outputs=[config_display, config_file_output],
                )

                def save_manual_config(yaml_text):
                    try:
                        # Validate YAML
                        config = yaml.safe_load(yaml_text)

                        config_path = os.path.join(SESSION_STATE["working_dir"], "config.yaml")
                        with open(config_path, "w") as f:
                            f.write(yaml_text)

                        SESSION_STATE["config"] = config
                        gr.Success("✅ Config updated and ready to run")
                        return gr.update(value=config_path)
                    except yaml.YAMLError as e:
                        raise gr.Error(f"Invalid YAML: {str(e)}")
                    except Exception as e:
                        raise gr.Error(f"Error saving config: {str(e)}")

                save_config_btn.click(save_manual_config, inputs=[config_display], outputs=[config_file_output])

            with gr.Tab("Run Benchmark Pipeline"):
                start_btn = gr.Button("▶️ Start Pipeline")
                stop_btn = gr.Button("🛑 Stop Pipeline")
                status_output = gr.Textbox(label="Pipeline Status", interactive=False)
                stages_output = gr.CheckboxGroup(
                    choices=list(STAGE_DISPLAY_MAP.values()), label="Stages Completed", interactive=False
                )
                log_output = gr.Code(label="Live Log Output", language=None, lines=20, interactive=False)
                timer = gr.Timer(1.0, active=False)

                def start_pipeline():
                    if not SESSION_STATE["config"]:
                        return gr.update(), "❌ Please build a config first"

                    if not SESSION_STATE["files"]:
                        return gr.update(), "❌ Please upload source documents first"

                    if SESSION_STATE["subprocess"] and SESSION_STATE["subprocess"].is_running():
                        return gr.update(), "⚠️ Pipeline already running"

                    # Ensure directories exist before starting pipeline
                    save_dirs(RESULTS_PROCESSED_DIR, LOCAL_DATASETS_DIR)

                    config = SESSION_STATE["config"]
                    hf_config = config.get("hf_configuration", {})
                    if hf_config.get("local_saving"):
                        logger.info(f"Local datasets will be saved to: {LOCAL_DATASETS_DIR}")

                    manager = SubprocessManager(SESSION_STATE["working_dir"])
                    success, message = manager.start()

                    if success:
                        SESSION_STATE["subprocess"] = manager
                        SESSION_STATE["pipeline_completed"] = False
                        return gr.update(active=True), "✅ Pipeline started successfully"
                    else:
                        return gr.update(), f"❌ {message}"

                def stop_pipeline():
                    if SESSION_STATE["subprocess"]:
                        SESSION_STATE["subprocess"].stop()
                        return gr.update(active=False), "🛑 Pipeline stopped."
                    return gr.update(active=False), "ℹ️ No pipeline running."

                def stream_logs():
                    if not SESSION_STATE["subprocess"]:
                        return "", [], ""

                    output, stages, completed = SESSION_STATE["subprocess"].read_output()
                    exit_code = SESSION_STATE["subprocess"].exit_code

                    if completed:
                        if not SESSION_STATE["pipeline_completed"]:
                            SESSION_STATE["pipeline_completed"] = True
                        status = (
                            "✅ Pipeline completed successfully!"
                            if exit_code == 0
                            else "❌ Pipeline failed. Check logs."
                        )
                        return output, stages, status

                    if SESSION_STATE["subprocess"].is_running():
                        return output, stages, "🔄 Pipeline running..."

                    return output, stages, "⏸️ Pipeline stopped."

                start_btn.click(start_pipeline, outputs=[timer, status_output])
                stop_btn.click(stop_pipeline, outputs=[timer, status_output])
                timer.tick(fn=stream_logs, outputs=[log_output, stages_output, status_output])

    demo.launch()
