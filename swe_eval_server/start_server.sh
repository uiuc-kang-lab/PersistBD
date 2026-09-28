#!/bin/bash
###############################################################################
# start_server.sh – Start the SWE-bench evaluation server on a dedicated machine
#
# This server evaluates model-generated patches by running SWE-bench test
# suites inside Docker containers. It should run on a machine with:
#   - Docker installed and accessible (docker ps works without sudo)
#   - ~4 CPUs and ~16 GB RAM per concurrent evaluation
#   - Internet access to pull Docker images (first run only)
#
# Usage (run from the repository root):
#   bash swe_eval_server/start_server.sh [port] [max_concurrent_evals]
#
# Examples:
#   bash swe_eval_server/start_server.sh          # defaults: port=8000, concurrency=8
#   bash swe_eval_server/start_server.sh 8000 16  # 16 parallel evaluations
#
# SECURITY: this server binds 0.0.0.0 with NO authentication and runs
# model-generated shell commands inside Docker. Only run it on a trusted network.
###############################################################################

set -euo pipefail

PORT="${1:-8000}"
MAX_CONCURRENT="${2:-8}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-300}"

PROJECT_DIR="${PERSISTBD_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# Activate conda env with swebench + fastapi
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate swe_eval

export MAX_CONCURRENT_EVALS=$MAX_CONCURRENT
export EVAL_TIMEOUT=$EVAL_TIMEOUT
export SWE_DATASET="${SWE_DATASET:-princeton-nlp/SWE-bench_Verified}"

echo "[$(date)] Starting SWE eval server on port $PORT"
echo "  MAX_CONCURRENT_EVALS = $MAX_CONCURRENT"
echo "  EVAL_TIMEOUT         = ${EVAL_TIMEOUT}s"
echo "  Dataset              = $SWE_DATASET"
echo ""

# Pre-pull SWE-bench base Docker images to avoid cold-start latency
# The images are large (~5-10 GB each) — only do this once
if [[ "${PREFETCH_IMAGES:-0}" == "1" ]]; then
    echo "Pre-fetching SWE-bench Docker images..."
    python -c "
from swebench.harness.docker_build import build_base_images
from datasets import load_dataset
import docker
client = docker.from_env()
dataset = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
build_base_images(client, dataset=list(dataset), force_rebuild=False, instance_image_tag='latest', env_image_tag='latest')
print('Base images ready.')
"
fi

# Launch server (binds 0.0.0.0, no auth — trusted networks only; see header)
PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" uvicorn swe_eval_server.server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --timeout-keep-alive 600 \
    --log-level info
