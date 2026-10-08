#!/usr/bin/env bash
# Create .venv with the vendored Python SDK so scripts/run_parity.py can act as the oracle.
# Requires uv (https://docs.astral.sh/uv/) and a vendored tree (bash scripts/sync_vendor.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d vendor/openai-agents-python ]; then
  echo "vendor/openai-agents-python is missing; run: bash scripts/sync_vendor.sh" >&2
  exit 1
fi
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

# The Python SDK needs Python >= 3.10; 3.12 is what the oracle is verified with.
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e vendor/openai-agents-python
.venv/bin/python -c "import agents; print('agents', agents.__file__)"
echo "Done. Run: .venv/bin/python scripts/run_parity.py --write-golden && cargo test --test parity_scenarios"
