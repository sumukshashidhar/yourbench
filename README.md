<!--
  README.md (Partial Snippet)
  ===========================
  This is a work-in-progress README for YourBench. 
  There is more coming soon—stay tuned!
-->

<div align="center">

<!-- Replace the paths below with your actual SVG logo paths or PNGs 
     Make sure these files exist in docs/assets or an accessible directory -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/yourbench_banner_dark_mode.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/yourbench_banner_light_mode.svg">
  <img alt="YourBench Logo" src="docs/assets/yourbench_banner_light_mode.svg" width="50%" height="50%">
</picture>

<h2>YourBench: A Dynamic Benchmark Generation Framework</h2>

<p>
  <strong>
    [<a href="https://github.com/huggingface/yourbench">GitHub</a>] 
    &middot; 
    [<a href="https://huggingface.co/datasets/sumuks/yourbench_y1">Dataset</a>] 
    &middot; 
    [<a href="https://github.com/huggingface/yourbench/tree/main/docs">Documentation</a>]
  </strong>
</p>

<!-- Example badges -->
<a href="https://github.com/sumukshashidhar/yourbench/stargazers">
  <img src="https://img.shields.io/github/stars/sumukshashidhar/yourbench?style=social" alt="GitHub Repo stars">
</a>

</div>

---

> **YourBench** is an open-source framework for generating domain-specific benchmarks in a zero-shot manner, inspired by modern software testing practices. It aims to keep your large language models on their toes—even as new data sources, domains, and knowledge demands evolve.

**Highlights**:
- **Dynamic Benchmark Generation**: Produce diverse, up-to-date questions from real-world source documents (PDF, Word, HTML, even multimedia).
- **Scalable & Structured**: Seamlessly handles ingestion, summarization, and multi-hop chunking for large or specialized datasets.
- **Zero-Shot Focus**: Emulates real-world usage scenarios by creating fresh tasks that guard against memorized knowledge.
- **Extensible**: Out-of-the-box pipeline stages (ingestion, summarization, question generation), plus an easy plugin mechanism to accommodate custom models or domain constraints.

---

## Quick Start (Alpha)

```bash
# 1. Clone the repo
git clone https://github.com/huggingface/yourbench.git
cd yourbench

# Use uv to install the dependencies
uv sync

# 3. Get a key from https://openrouter.ai/ and add it to the .env file (or make your own config with a different model!)
touch .env
echo "OPENROUTER_API_KEY=<your_openrouter_api_key>" >> .env

# 4. Run the pipeline with an example config
yourbench --config configs/example.yaml
```
