import io
import os
import re
import sys
import time
import atexit
import shutil
import subprocess


# Early startup logging
from loguru import logger
logger.info("YourBench Gradio UI starting up...")
ui_startup_time = time.perf_counter()

import yaml  # noqa: E402


logger.info("Loading Gradio...")
import gradio as gr  # noqa: E402


logger.info("Gradio loaded")

from dotenv import load_dotenv  # noqa: E402


# Lazy import pandas - only when needed
pd = None


def _get_pandas():
    global pd
    if pd is None:
        import pandas

        pd = pandas
    return pd


logger.info("Loading Gradio components...")

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
    "Together AI": "together",
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


def validate_url(url):
    """Validate URL format"""
    if not url.strip():
        return True, ""  # Empty URL is valid (optional field)

    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return False, "Base URL must start with http:// or https://"

    return True, ""


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
    return f"✅ Uploaded {len(uploaded)} files: {', '.join(uploaded)}"


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
    logger.info("Building Gradio interface...")
    interface_start = time.perf_counter()

    with gr.Blocks(title="YourBench", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 🚀 YourBench")
        gr.Markdown("**Create custom benchmark datasets from your documents using AI-powered question generation**")

        if not HF_DEFAULTS["hf_token"]:
            gr.Markdown("⚠️ **Warning**: HF_TOKEN not set in `.env` file. Please add it to enable dataset uploading.")

        # Add output locations info
        with gr.Row():
            gr.Markdown("""
            ### 📁 Output Locations
            - **Processed Data**: `results/processed/`
            - **Local Datasets**: `results/datasets/` (when local saving is enabled)
            """)

        with gr.Tabs():
            with gr.Tab("📄 Upload Documents"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Step 1: Upload Your Documents")
                        gr.Markdown(
                            "Upload the source documents you want to create benchmarks from. Supported formats: `.txt`, `.md`, `.pdf`, `.html`"
                        )

                        file_input = gr.File(
                            file_count="multiple",
                            file_types=[".txt", ".md", ".pdf", ".html"],
                            label="Choose Files",
                        )

                        with gr.Row():
                            clear_btn = gr.Button("🧹 Clear All", variant="secondary", size="sm")

                        upload_log = gr.Textbox(label="📝 Upload Status", interactive=False, lines=3)

                        file_input.upload(save_uploaded_files, inputs=file_input, outputs=upload_log)
                        clear_btn.click(fn=clear_uploaded_files, outputs=[upload_log, file_input])

                    with gr.Column(scale=1):
                        gr.Markdown("### 💡 Tips")
                        gr.Markdown("""
                        **Quality matters**: Clean, well-structured documents produce better questions<br>
                        **Size limits**: Very large files may take longer to process
                        """)

            with gr.Tab("🤖 Configure Models"):
                gr.Markdown("### Step 2: Set Up Your AI Models")
                gr.Markdown(
                    "Configure the AI models that will power different stages of your benchmark creation pipeline."
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### Add Models")

                        with gr.Row():
                            with gr.Column():
                                model_name_input = gr.Textbox(
                                    label="Model Name",
                                    placeholder="e.g., meta-llama/Llama-3.3-70B-Instruct",
                                    info="Full model identifier as used by the provider",
                                )

                                provider_dropdown = gr.Dropdown(
                                    label="Provider",
                                    choices=list(PROVIDERS.keys()),
                                    value="HF Inference",
                                    interactive=True,
                                    info="Select your AI provider",
                                )

                            with gr.Column():
                                base_url_input = gr.Textbox(
                                    label="🔗 Custom Base URL (Optional)",
                                    placeholder="https://api.example.com/v1",
                                    info="For custom endpoints or local servers",
                                )

                                api_key_input = gr.Textbox(
                                    label="🔑 API Key Environment Variable (Optional)",
                                    placeholder="e.g., VLLM_API_KEY",
                                    info="Name of environment variable containing your API key (as set in .env file)",
                                    visible=False,
                                )

                        def toggle_custom_fields(base_url_value):
                            has_custom_url = base_url_value.strip() != ""
                            return (gr.update(interactive=not has_custom_url), gr.update(visible=has_custom_url))

                        base_url_input.change(
                            toggle_custom_fields, inputs=[base_url_input], outputs=[provider_dropdown, api_key_input]
                        )

                        with gr.Row():
                            add_model_btn = gr.Button("➕ Add Model", variant="primary")
                            remove_model_btn = gr.Button("🗑️ Remove Last", variant="secondary")

                        model_table = gr.Dataframe(
                            headers=["Model Name", "Provider", "Base URL", "API Key Env"],
                            datatype=["str", "str", "str", "str"],
                            row_count=(1, "dynamic"),
                            interactive=True,
                            value=[],
                            label="📋 Configured Models",
                        )

                        def add_model(model_name, provider_key, table_data, base_url, api_key_var):
                            if not model_name.strip():
                                gr.Warning("⚠️ Model name was empty — nothing added.")
                                return (table_data if table_data is not None else [], "", "HF Inference", "", "")

                            if base_url.strip():
                                is_valid, error_msg = validate_url(base_url)
                                if not is_valid:
                                    gr.Warning(f"⚠️ Invalid Base URL: {error_msg}")
                                    return table_data, model_name, provider_key, base_url, api_key_var

                            pandas = _get_pandas()
                            if isinstance(table_data, pandas.DataFrame):
                                new_data = table_data.values.tolist()
                            elif table_data is None:
                                new_data = []
                            else:
                                new_data = table_data

                            for row in new_data:
                                if isinstance(row, list) and len(row) > 0 and row[0] == model_name:
                                    gr.Warning(f"⚠️ Model '{model_name}' already exists — not added.")
                                    return table_data, "", "HF Inference", "", ""

                            # If custom base URL is provided, set provider to empty string
                            final_provider = "" if base_url.strip() else provider_key

                            new_data.append([model_name, final_provider, base_url, api_key_var])
                            gr.Info(f"✅ Model '{model_name}' added.")
                            return new_data, "", "HF Inference", "", ""

                        def remove_model(table_data):
                            pandas = _get_pandas()
                            if isinstance(table_data, pandas.DataFrame):
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
                            inputs=[model_name_input, provider_dropdown, model_table, base_url_input, api_key_input],
                            outputs=[model_table, model_name_input, provider_dropdown, base_url_input, api_key_input],
                        )

                        remove_model_btn.click(remove_model, inputs=[model_table], outputs=[model_table])

                    with gr.Column(scale=1):
                        gr.Markdown("### 🔧 Model Setup Guide")
                        gr.Markdown("""
                        **Provider Options:**
                        - **Provider**: Choose a provider to use HF Inference Providers
                        - **Custom URL**: Your own API endpoint or GPT, Gemini, etc

                        **Custom Endpoints:**
                        - Use `https://` or `http://` prefix
                        - Common for local servers or API providers (vLLM, Ollama)
                        - API key environment variable required
                        - Provider field will be disabled when using custom URL

                        **Best Practices:**
                        - Use larger models for complex tasks
                        - Smaller models for simple processing
                        - Same model for all stages works fine
                        """)

            with gr.Tab("⚙️ Pipeline Configuration"):
                gr.Markdown("### Step 3: Configure Your Pipeline")

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### Model Role Assignment")
                        gr.Markdown("Assign models to different pipeline stages based on their capabilities.")

                        role_infos = {
                            "ingestion": "Converts raw documents (PDFs, HTML, etc.) to clean text. Vision models recommended for complex layouts.",
                            "summarization": "Creates concise summaries of long documents to improve processing efficiency.",
                            "single_shot_question_generation": "Generates straightforward questions from individual text chunks.",
                            "multi_hop_question_generation": "Creates complex questions requiring reasoning across multiple sources.",
                        }

                        role_fields = {}
                        for role in [
                            "ingestion",
                            "summarization",
                            "single_shot_question_generation",
                            "multi_hop_question_generation",
                        ]:
                            role_fields[role] = gr.Dropdown(
                                label=f"{role.replace('_', ' ').title()}",
                                interactive=True,
                                choices=[],
                                info=role_infos.get(role, ""),
                            )

                        def update_role_choices(table_data):
                            pandas = _get_pandas()
                            if isinstance(table_data, pandas.DataFrame):
                                data = table_data.values.tolist()
                            elif table_data is None:
                                data = []
                            else:
                                data = table_data

                            names = [
                                row[0]
                                for row in data
                                if isinstance(row, list) and len(row) > 0 and row[0] and row[0].strip()
                            ]
                            default_value = names[0] if len(names) == 1 else None
                            return [gr.update(choices=names, value=default_value) for _ in role_fields.values()]

                        model_table.change(
                            update_role_choices, inputs=[model_table], outputs=list(role_fields.values())
                        )

                        gr.Markdown("#### Question Generation Settings")

                        question_mode = gr.Dropdown(
                            label="📝 Question Format",
                            choices=["multi-choice", "open-ended"],
                            value="multi-choice",
                            info="Multi-choice: A/B/C/D options | Open-ended: Free-form answers",
                        )

                        additional_instructions = gr.Textbox(
                            label="🎯 Additional Instructions",
                            value="Ask deep, evidence-based questions from the document.",
                            lines=3,
                            info="Guide the AI on question style, difficulty, or focus areas",
                        )

                        cross_doc_enable = gr.Checkbox(
                            label="🔗 Enable Cross-Document Questions",
                            value=False,
                            info="Generate questions that require information from multiple documents",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Dataset Output Settings")

                        hf_token = gr.Textbox(
                            label="🤗 HF Token",
                            value=HF_DEFAULTS["hf_token"],
                            type="password",
                            info="Your Hugging Face API token (from .env file)",
                        )

                        hf_org = gr.Textbox(
                            label="🏢 HF Organization",
                            value=HF_DEFAULTS["hf_organization"],
                            info="Your HF organization",
                        )

                        hf_dataset_name = gr.Textbox(
                            label="🏷️ Dataset Name", value=HF_DEFAULTS["hf_dataset_name"], info="Name for your dataset"
                        )

                        private = gr.Checkbox(
                            label="🔒 Private Dataset",
                            value=HF_DEFAULTS["private"],
                            info="Make dataset private on HF Hub",
                        )

                        local_saving = gr.Checkbox(
                            label="💾 Save Locally", value=False, info="Save dataset files to local directory"
                        )

                        concat = gr.Checkbox(
                            label="🔄 Concatenate if Exists",
                            value=HF_DEFAULTS["concat_if_exist"],
                            info="Append to existing dataset",
                        )

                        upload_card = gr.Checkbox(
                            label="📄 Generate Dataset Card",
                            value=HF_DEFAULTS["upload_card"],
                            info="Create documentation for your dataset",
                        )

                with gr.Row():
                    build_btn = gr.Button("🛠️ Build Configuration", variant="primary", size="lg")

                config_display = gr.Code(label="📋 Generated Configuration", language="yaml", lines=15)

                with gr.Row():
                    save_config_btn = gr.Button("💾 Save Configuration", variant="secondary")
                    config_file_output = gr.File(label="📥 Download Config File")

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
                    try:
                        # Convert table_data to list format
                        pandas = _get_pandas()
                        if isinstance(table_data, pandas.DataFrame):
                            rows = table_data.values.tolist()
                        elif table_data is None:
                            rows = []
                        else:
                            rows = table_data

                        # Validation
                        if not rows:
                            raise gr.Error("❌ At least one model must be configured")

                        if not all([ingestion_model, summarization_model, single_model, multi_model]):
                            raise gr.Error("❌ All model roles must be assigned")

                        # Build model list with validation
                        model_list = []
                        for row in rows:
                            if isinstance(row, list) and len(row) >= 2:
                                model_entry = {
                                    "model_name": row[0],
                                }

                                # Handle provider vs custom base URL logic
                                if len(row) > 2 and row[2]:  # Has base URL
                                    # Validate URL
                                    is_valid, error_msg = validate_url(row[2])
                                    if not is_valid:
                                        raise gr.Error(f"❌ Invalid URL for model {row[0]}: {error_msg}")
                                    model_entry["base_url"] = row[2]
                                    # Don't include provider when using custom base URL
                                else:
                                    # Use provider only when no custom base URL
                                    model_entry["provider"] = PROVIDERS.get(row[1], row[1])

                                if len(row) > 3 and row[3]:
                                    model_entry["api_key"] = f"${row[3]}"
                                model_list.append(model_entry)

                        # Setup directories
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
                        gr.Success("✅ Configuration built successfully, you can run the pipeline! 🚀")
                        return config_yaml, gr.update(value=config_path)

                    except Exception as e:
                        raise gr.Error(f"❌ Configuration error: {str(e)}")

                build_btn.click(
                    build_config,
                    inputs=[hf_token, hf_org, hf_dataset_name, private, concat, upload_card, local_saving, model_table]
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
                        gr.Info("✅ Configuration saved successfully!")
                        return gr.update(value=config_path)
                    except yaml.YAMLError as e:
                        raise gr.Error(f"❌ Invalid YAML: {str(e)}")
                    except Exception as e:
                        raise gr.Error(f"❌ Error saving config: {str(e)}")

                save_config_btn.click(save_manual_config, inputs=[config_display], outputs=[config_file_output])

            with gr.Tab("🚀 Run Pipeline"):
                gr.Markdown("### Step 4: Execute Your Pipeline")
                gr.Markdown("Start the benchmark generation process and monitor progress in real-time.")

                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            start_btn = gr.Button("▶️ Start Pipeline", variant="primary", size="lg")
                            stop_btn = gr.Button("🛑 Stop Pipeline", variant="stop", size="lg")

                        status_output = gr.Textbox(label="📊 Pipeline Status", interactive=False, lines=2)

                        stages_output = gr.CheckboxGroup(
                            choices=list(STAGE_DISPLAY_MAP.values()), label="✅ Completed Stages", interactive=False
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📈 Pipeline Progress")
                        gr.Markdown("""
                        **Stage Overview:**
                        1. **Process Input Docs** - Convert files to text
                        2. **Summarize Documents** - Create concise summaries
                        3. **Chunk Documents** - Split into manageable pieces
                        4. **Generate Single Shot Questions** - Simple Q&A pairs
                        5. **Generate Multi Hop Questions** - Complex reasoning
                        6. **Generate Lighteval Subset** - Evaluation dataset suitable for Lighteval evalution
                        7. **Citation Score Filtering** - Quality filtering
                        """)

                with gr.Accordion("📜 Live Logs", open=False):
                    log_output = gr.Code(
                        label="Pipeline Output",
                        language=None,
                        lines=20,
                        interactive=False,
                    )

                timer = gr.Timer(1.0, active=False)

                def start_pipeline():
                    if not SESSION_STATE["config"]:
                        return gr.update(), "❌ Please build a configuration first (Step 3)"

                    if not SESSION_STATE["files"]:
                        return gr.update(), "❌ Please upload source documents first (Step 1)"

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
                        gr.Info("🚀 Pipeline started successfully!")
                        return gr.update(active=True), "🔄 Pipeline starting..."
                    else:
                        return gr.update(), f"❌ {message}"

                def stop_pipeline():
                    if SESSION_STATE["subprocess"]:
                        SESSION_STATE["subprocess"].stop()
                        gr.Info("🛑 Pipeline stopped")
                        return gr.update(active=False), "🛑 Pipeline stopped by user"
                    return gr.update(active=False), "ℹ️ No pipeline running"

                def stream_logs():
                    if not SESSION_STATE["subprocess"]:
                        return "", [], ""

                    output, stages, completed = SESSION_STATE["subprocess"].read_output()
                    exit_code = SESSION_STATE["subprocess"].exit_code

                    if completed:
                        if not SESSION_STATE["pipeline_completed"]:
                            SESSION_STATE["pipeline_completed"] = True
                            if exit_code == 0:
                                gr.Info("🎉 Pipeline completed successfully!")
                            else:
                                gr.Warning("⚠️ Pipeline completed with errors")

                        status = (
                            "✅ Pipeline completed successfully! Check the output directories for your dataset."
                            if exit_code == 0
                            else "❌ Pipeline failed. Check logs for details."
                        )
                        return output, stages, status

                    if SESSION_STATE["subprocess"].is_running():
                        return output, stages, "🔄 Pipeline running... Check logs for detailed progress."

                    return output, stages, "⏸️ Pipeline stopped"

                start_btn.click(start_pipeline, outputs=[timer, status_output])
                stop_btn.click(stop_pipeline, outputs=[timer, status_output])
                timer.tick(fn=stream_logs, outputs=[log_output, stages_output, status_output])

        # Add some custom CSS for better styling
        demo.load(
            js="""
        function() {
            // Add some custom styling
            const style = document.createElement('style');
            style.textContent = `
                .gradio-container {
                    max-width: 1200px !important;
                }
                .tab-nav {
                    font-weight: 600;
                }
                .gr-button-primary {
                    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                    border: none;
                    font-weight: 600;
                }
                .gr-button-stop {
                    background: linear-gradient(45deg, #FF6B6B, #FF8E8E);
                    border: none;
                    font-weight: 600;
                }
                .upload-area {
                    border: 2px dashed #4ECDC4;
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                    background: rgba(78, 205, 196, 0.1);
                }
            `;
            document.head.appendChild(style);
        }
        """
        )

    logger.success(f"Gradio interface built in {time.perf_counter() - interface_start:.2f}s")
    logger.success(f"Total UI startup time: {time.perf_counter() - ui_startup_time:.2f}s")
    logger.info("Launching Gradio server...")

    demo.launch()
