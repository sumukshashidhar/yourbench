<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/yourbench_banner_dark_mode.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/yourbench_banner_light_mode.svg">
  <img alt="YourBench Logo" src="docs/assets/yourbench_banner_light_mode.svg" width="50%">
</picture>

<h2>YourBench: A Dynamic Benchmark Generation Framework</h2>

<a href="https://github.com/huggingface/yourbench/stargazers">
  <img src="https://img.shields.io/github/stars/huggingface/yourbench?style=social" alt="GitHub Repo stars">
</a>

<p>
  <strong>
    [<a href="https://github.com/huggingface/yourbench">GitHub</a>] · 
    [<a href="https://huggingface.co/datasets/sumuks/tempora">Dataset</a>] · 
    [<a href="https://github.com/huggingface/yourbench/tree/main/docs">Documentation</a>] · 
    [<a href="https://arxiv.org/abs/2504.01833">Paper</a>]
  </strong>
</p>

</div>

---

Yourbench is a structured data generation library for building better AI systems. Generate high-quality QA pairs, training data, and evaluation datasets from any source documents with full control over the output format and complexity. The modular architecture lets you configure every aspect of the generation pipeline, from document parsing (with built-in converters for common formats to markdown) to chunking strategies to output schemas. Most eval frameworks force you into their structure; Yourbench adapts to yours. Use it to create domain-specific benchmarks, fine-tuning datasets, or systematic model evaluations. Peer-reviewed and appearing at COLM 2025. **100% free and open source, forever.**

## Quick Start

You can use yourbench without installation instantly with [uv](https://docs.astral.sh/uv/getting-started/installation/)! Simply run:

```
uvx yourbench --model gpt-4o-mini --docs <YOUR_FILE_DIRECTORY_HERE>
```

You will see the dataset appear locally! If a valid `HF_TOKEN` is set, you will also see the dataset appear on your Hugging Face Hub!

## Installation

YourBench is available on PyPI and requires **Python 3.12+**. You can install it as follows:

* **Install via PyPI (stable release):**

  ```bash
  # uv (recommended; get it here: https://docs.astral.sh/uv/getting-started/installation/)
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install yourbench

  # pip (standard support)
  pip install yourbench
  ```

  This will install the latest published version (e.g. `0.4.0`).

* **Install from source (development version):**

  ```bash
  git clone https://github.com/huggingface/yourbench.git
  cd yourbench
  
  # uv, recommended
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install -e .

  # pip
  pip install -e .
  ```

  Installing from source is recommended if you want the latest updates or to run the included example configuration.

> **Note:** If you plan to use models that require API access (e.g. OpenAI GPT-4o or Hugging Face Inference API), make sure to have the appropriate credentials. You’ll also need a Hugging Face token (to optionally to upload results). See below for how to configure these before running YourBench.

## Usage

Once installed, YourBench can be run from the command line to generate a custom evaluation set. Here’s a quick example:

```bash
# 1. (Optional) If not done already, install YourBench
pip install yourbench

# 2. Prepare your API credentials (for model inference and Hub access)
# For example, create a .env file with required keys:
# echo "OPENROUTER_API_KEY=<your_openrouter_api_key>" >> .env        # Example
echo "HF_TOKEN=<your_huggingface_api_token>" >> .env              # Hugging Face token (for Hub datasets & inference)
echo "HF_ORGANIZATION=<your_hf_username_or_org>" >> .env          # (Optional) Organization name for dataset pushing

# 3. Run the pipeline on the provided example config (uses sample docs and models)
yourbench run --config example/configs/simple_example.yaml

# 4. (Optional) Run the pipeline on your own documents:
yourbench run --config my_custom_config.yaml
```

The **example configuration** `example/configs/simple_example.yaml` (included in the repository) demonstrates a basic setup. It specifies sample documents and default models for each stage of the pipeline. In step 3 above, YourBench will automatically ingest the example documents, generate a set of Q\&A pairs, and output a Hugging Face Dataset containing the evaluation questions and answers.

For your own data, you can create a YAML config pointing to your documents and preferred models. For instance, you might specify a folder of PDFs or text files under a `documents` field, and choose which LLM to use for question generation. **YourBench is fully configurable** – you can easily **toggle stages** on or off and swap in different models. *For example: you could disable the summarization stage for very short texts, or use a powerful, large, API model for question generation while using a faster local model for summarization.* The possibilities are endless! Simply adjust the YAML, and the pipeline will accommodate it. (See the [usage example](https://github.com/huggingface/yourbench/blob/main/example/configs/advanced_example.yaml) for all available options!)

You may be interested in [How YourBench Works](./docs/PRINCIPLES.md)

## Try it Online (Hugging Face Spaces)

You can **try YourBench right away in your browser** – no installation needed:

* **[YourBench Demo Space](https://huggingface.co/spaces/yourbench/demo)** – Use our ready-to-go web demo to upload a document (or paste text) and generate a custom evaluation set with **one click**, complete with an instant model leaderboard. **(This free demo will use a default set of models to answer the questions and show how different models perform.)**
* **[YourBench Advanced Space](https://huggingface.co/spaces/yourbench/advanced)** – For power users, the advanced demo lets you provide a custom YAML config and plug in your own models or API endpoints. This gives you full control over the pipeline (choose specific models, adjust chunking parameters, etc.) via a convenient UI, right from the browser.

👉 Both hosted apps are available on Hugging Face Spaces under the **[yourbench](https://huggingface.co/yourbench)** organization. Give them a try to see how YourBench can generate benchmarks tailored to your use-case in minutes.

## Contributing

Contributions are welcome!

We actively review PRs and welcome improvements or fixes from the community. For major changes, feel free to open an issue first to discuss the idea.

## Highlights


* **Dynamic Benchmark Generation** – Produce diverse, up-to-date question-answer pairs derived from real-world source documents (PDF, Word, HTML, even multimedia).
* **Scalable & Structured** – Seamlessly handle ingestion, summarization, and multi-hop chunking for large or specialized datasets.
* **Extensible Pipeline** – Use out-of-the-box stages (ingestion, summarization, question generation) or plug in custom models and logic to accommodate domain-specific needs.
* **Robust Configuration** – Control the entire pipeline via a single YAML config (model choices, data paths, chunking parameters, generation prompts, deduplication thresholds, etc.).
* **Multi-Model Support** – Assign different LLMs for each stage (ingestion, summarization, QG, answering), fostering broader coverage and question-style diversity.
* **Deduplication & Quality Filtering** – Automatically group near-duplicates to prune questions and retain a curated set of high-quality queries.
* **Logging & Analysis** – Built-in metrics evaluate dataset coverage, question distribution, difficulty, and more.
* **Flexible Output** – Save generated benchmarks locally or push them to the Hugging Face Hub for sharing or public leaderboards.

<div align="center">

<a href="https://youtu.be/mhszO6kZSbI" target="_blank">
  <img src="https://img.youtube.com/vi/mhszO6kZSbI/maxresdefault.jpg" alt="YourBench Demo Video" width="600" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" />
  <br/>
  <img src="https://img.shields.io/badge/Watch%20Demo-YouTube-red?style=for-the-badge&logo=youtube" alt="Watch Demo on YouTube"/>
  <br/>
  <em>Watch our 3-minute demo of the YourBench pipeline</em>
</a>
</div>

## 📈 Progress

<div align="center">
  <a href="https://star-history.com/#huggingface/yourbench&Date">
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=huggingface/yourbench&type=Date">
  </a>
</div>

## 📜  License

This project is licensed under the Apache 2.0 License – see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute YourBench in either commercial or academic projects under the terms of this license.

## 📚 Citation

If you use **YourBench** in your research or applications, please consider citing our paper:

```bibtex
@misc{shashidhar2025yourbencheasycustomevaluation,
      title={YourBench: Easy Custom Evaluation Sets for Everyone},
      author={Sumuk Shashidhar and Clémentine Fourrier and Alina Lozovskia and Thomas Wolf and Gokhan Tur and Dilek Hakkani-Tür},
      year={2025},
      eprint={2504.01833},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.01833}
}
```