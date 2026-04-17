#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=gpu_partition_a
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --account=project_account
#SBATCH --time=7-00:00:00
#SBATCH --job-name=llm_eval_job

set -euo pipefail

PORT=9747

# Update this to your local model root if needed.
MODEL_ROOT="/path/to/models"

MODELS=(
  "Meta-Llama-3-8B-Instruct"
)

TASKS=(
  "law"
  "credibility"
)

FEW_SHOT=false
NUMERICAL=false
LANGUAGE="english"

module load cuda/12.5

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM server (PID: ${VLLM_PID})..."
    kill "${VLLM_PID}"
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for MODEL_NAME in "${MODELS[@]}"; do
  echo "Starting model server for ${MODEL_NAME} on port ${PORT}..."

  vllm serve "${MODEL_ROOT}/${MODEL_NAME}" \
    --tensor-parallel-size 4 \
    --disable-log-requests \
    --port "${PORT}" &
  VLLM_PID=$!

  echo "Waiting for server startup..."
  sleep 600

  echo "Running evaluation for ${MODEL_NAME}..."
  python3 ../src/run_companies.py \
    --tasks "${TASKS[@]}" \
    --few_shot "${FEW_SHOT}" \
    --numerical "${NUMERICAL}" \
    --port "${PORT}" \
    --languages "${LANGUAGE}" \
    --model_path "${MODEL_ROOT}" \
    --model "${MODEL_NAME}"

  cleanup
  unset VLLM_PID
done