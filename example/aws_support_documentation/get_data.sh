#!/usr/bin/env bash
set -euo pipefail

# list of AWS Bedrock docs to download
LINKS=(
  "https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html"
  "https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-permissions.html"
  "https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-how.html"
  "https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-tiers.html"
  "https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-ds.html"
  "https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-setup.html"
  "https://docs.aws.amazon.com/bedrock/latest/userguide/kb-osm-permissions-slr-rbp.html"
  "https://docs.aws.amazon.com/bedrock/latest/userguide/kb-osm-permissions-console-fgap.html"
  "https://docs.aws.amazon.com/bedrock/latest/userguide/key-definitions-flow.html"
  "https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html"
)

# make sure data/ exists
mkdir -p data

# download each link
for url in "${LINKS[@]}"; do
  # extract the last path segment (e.g. "getting-started.html")
  filename=$(basename "${url}")
  # download, following redirects, into data/
  curl -L --fail "${url}" -o "data/${filename}"
  echo "Downloaded ${filename}"
done
