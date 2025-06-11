---
{{ card_data }}
{{ config_data }}
---
[<img src="https://raw.githubusercontent.com/huggingface/yourbench/main/docs/assets/yourbench-badge-web.png"
     alt="Built with YourBench" width="200" height="32" />](https://github.com/huggingface/yourbench)

# {{ pretty_name }}

This dataset was generated using YourBench (v{{ yourbench_version }}), an open-source framework for generating domain-specific benchmarks from document collections.

## Pipeline Steps

{{ pipeline_subsets }}

## Reproducibility

To reproduce this dataset, use YourBench v{{ yourbench_version }} with the following configuration:

```yaml
{{ config_yaml }}
```

{{ footer | default("", true) }}