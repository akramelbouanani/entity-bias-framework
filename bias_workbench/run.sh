#!/usr/bin/env bash
set -euo pipefail

WORKBENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${WORKBENCH_DIR}/.."

exec python3 -m uvicorn bias_workbench.app.main:app \
  --host "${BIAS_WORKBENCH_HOST:-127.0.0.1}" \
  --port "${BIAS_WORKBENCH_PORT:-8010}"
