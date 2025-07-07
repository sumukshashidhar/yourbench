import io
import os
import re
import sys
import shutil
import tempfile
import subprocess

import yaml
import gradio as gr
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


def save_uploaded_files(files):
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
        shutil.rmtree(raw_dir)
    SESSION_STATE["files"] = []
    return "🧹 Uploads cleared.", gr.update(value=None)


def load_config():
    config_path = os.path.join(SESSION_STATE["working_dir"], "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


class SubprocessManager:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.config_path = os.path.join(self.working_dir, "config.yaml")
        self.process = None
        self.output_stream = io.StringIO()
        self.exit_code = None

    def start(self):
        self.output_stream = io.StringIO()

        os.makedirs("results/processed", exist_ok=True)

        self.process = subprocess.Popen(
            ["uv", "run", "yourbench", "run", "--config", self.config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        os.set_blocking(self.process.stdout.fileno(), False)

    def is_running(self):
        return self.process and self.process.poll() is None

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
            return "", []
        try:
            while True:
                line = self.process.stdout.readline()
                if line:
                    self.output_stream.write(line)
                else:
                    break
        except BlockingIOError:
            pass
        full_output = self.output_stream.getvalue()
        stages_completed = list(set(re.findall(r"Completed stage: '([^']*)'", full_output)))
        return full_output, [STAGE_DISPLAY_MAP[s] for s in stages_completed if s in STAGE_DISPLAY_MAP]


SESSION_STATE = {"working_dir": tempfile.mkdtemp(prefix="yourbench_"), "subprocess": None, "files": [], "config": None}


def launch_ui():
    with gr.Blocks(title="YourBench") as demo:
        gr.Markdown("# 🚀 YourBench")
        if not HF_DEFAULTS["hf_token"]:
            gr.Markdown("⚠️ **HF_TOKEN not set in `.env` file.**")

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

                model_name_input = gr.Textbox(label="Model Name")
                provider_dropdown = gr.Dropdown(
                    label="Provider", choices=list(PROVIDERS.keys()), value="HF Inference", allow_custom_value=True
                )
                add_model_btn = gr.Button("➕ Add Model")
                model_table = gr.Dataframe(
                    headers=["Model Name", "Provider"],
                    datatype=["str", "str"],
                    row_count=(1, "dynamic"),
                    interactive=True,
                )
                remove_model_btn = gr.Button("🗑️ Remove Last Model")

                def add_model(model_name, provider_key, table_data):
                    if not model_name:
                        raise gr.Error("Model name is required.")
                    new_data = table_data if isinstance(table_data, list) else table_data.values.tolist()
                    new_data.append([model_name, provider_key])
                    return new_data, "", "", "HF Inference"

                def remove_model(table_data):
                    new_data = table_data.values.tolist() if hasattr(table_data, "values") else table_data
                    if new_data:
                        new_data = new_data[:-1]
                    return new_data

                add_model_btn.click(
                    add_model,
                    inputs=[model_name_input, provider_dropdown, model_table],
                    outputs=[model_table, model_name_input, provider_dropdown, provider_dropdown],
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
                        label=f"Model for {role.replace('_', ' ').title()}", interactive=True
                    )

                def update_roles(table_data):
                    names = [
                        row[0]
                        for row in (table_data.values.tolist() if hasattr(table_data, "values") else table_data)
                        if isinstance(row, list) and len(row) > 0 and row[0]
                    ]
                    return {
                        field: gr.update(choices=names, value=names[0] if len(names) == 1 else None)
                        for field in role_fields.values()
                    }

                model_table.change(update_roles, inputs=[model_table], outputs=list(role_fields.values()))

                question_mode = gr.Dropdown(
                    label="Question Mode", choices=["open-ended", "multi-choice"], value="multi-choice"
                )
                additional_instructions = gr.Textbox(
                    label="Additional Instructions", value="Ask deep, evidence-based questions from the document."
                )
                cross_doc_enable = gr.Checkbox(label="Enable Cross-Document Question Generation", value=False)

                build_btn = gr.Button("🛠️ Build Config")
                save_config_btn = gr.Button("💾 Save Config")
                save_status = gr.Textbox(label="Save Status", interactive=False)
                config_display = gr.Code(label="Generated Config", language="yaml")
                config_file_output = gr.File(label="Download Config")

                def build_config(
                    hf_token,
                    hf_org,
                    hf_dataset_name,
                    private,
                    concat,
                    upload_card,
                    table_data,
                    ingestion_model,
                    summarization_model,
                    single_model,
                    multi_model,
                    question_mode,
                    additional_instructions,
                    cross_doc_enable,
                ):
                    model_list = []
                    rows = table_data.values.tolist() if hasattr(table_data, "values") else table_data
                    for row in rows:
                        if isinstance(row, list) and len(row) >= 2 and row[0] and row[1]:
                            model_list.append({
                                "model_name": row[0],
                                "provider": PROVIDERS.get(row[1], row[1]),
                            })

                    config = {
                        "settings": {"debug": False},
                        "hf_configuration": {
                            "token": "$HF_TOKEN",
                            "hf_organization": hf_org,
                            "hf_dataset_name": hf_dataset_name,
                            "private": private,
                            "upload_card": upload_card,
                            "concat_if_exist": concat,
                        },
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
                            "upload_ingest_to_hub": {"run": True, "source_documents_dir": "results/processed"},
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

                    return config_yaml, config_path, gr.update(visible=False)

                build_btn.click(
                    build_config,
                    inputs=[hf_token, hf_org, hf_dataset_name, private, concat, upload_card, model_table]
                    + list(role_fields.values())
                    + [question_mode, additional_instructions, cross_doc_enable],
                    outputs=[config_display, config_file_output, save_status],
                )

                def save_manual_config(yaml_text):
                    config_path = os.path.join(SESSION_STATE["working_dir"], "config.yaml")
                    with open(config_path, "w") as f:
                        f.write(yaml_text)
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    SESSION_STATE["config"] = config
                    return gr.Success("✅ Config updated and ready for download."), config_path

                save_config_btn.click(
                    save_manual_config, inputs=[config_display], outputs=[save_status, config_file_output]
                )

            with gr.Tab("Run Benchmark Pipeline"):
                start_btn = gr.Button("▶️ Start Pipeline")
                stop_btn = gr.Button("🛑 Stop Pipeline")
                stages_output = gr.CheckboxGroup(
                    choices=list(STAGE_DISPLAY_MAP.values()), label="Stages Completed", interactive=False
                )
                log_output = gr.Code(label="Live Log Output", language=None, lines=20, interactive=False)
                timer = gr.Timer(1.0, active=False)

                def start_pipeline():
                    if SESSION_STATE["subprocess"] and SESSION_STATE["subprocess"].is_running():
                        return gr.Warning("Already running.")
                    manager = SubprocessManager(SESSION_STATE["working_dir"])
                    manager.start()
                    SESSION_STATE["subprocess"] = manager
                    return gr.update(active=True)

                def stop_pipeline():
                    if SESSION_STATE["subprocess"]:
                        SESSION_STATE["subprocess"].stop()
                    return gr.update()

                def stream_logs():
                    if not SESSION_STATE["subprocess"]:
                        return "", []
                    return SESSION_STATE["subprocess"].read_output()

                start_btn.click(start_pipeline, outputs=timer)
                stop_btn.click(stop_pipeline)
                timer.tick(fn=stream_logs, outputs=[log_output, stages_output])

    demo.launch()
