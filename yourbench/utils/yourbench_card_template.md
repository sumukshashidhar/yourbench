---
{{ card_data }}
{{ config_data }}
---

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