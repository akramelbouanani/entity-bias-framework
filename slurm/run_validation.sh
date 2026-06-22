#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=gpuh200p,gpuh100p,prismgpup,gpu80G
#SBATCH --gres=gpu:4
#SBATCH --exclude=node57
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --account=lasti
#SBATCH --time=7-00:00:00
#SBATCH --job-name=entity-validation

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "${REPO_ROOT}/pyproject.toml" ]] || REPO_ROOT="$(dirname "${REPO_ROOT}")"
[[ -f "${REPO_ROOT}/pyproject.toml" ]] || { echo "Repository root not found." >&2; exit 1; }
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PORT=9747
MODEL_PATH_COMMON="/home/data/dataset/huggingface/LLMs/meta-llama"
MODELS=("Meta-Llama-3-70B")
TASKS=("madtsc" "madtsc-parties" "pstance" "finentity" "liar")
FEW_SHOT=("false" "true")
LANGUAGES=("english" "russian" "chinese")
VARIANTS=("original" "synthetic")

TENSOR_PARALLEL_SIZE=4

module load cuda/12.5

if ! command -v vllm >/dev/null 2>&1; then
    echo "vLLM is not installed in the active environment." >&2
    echo "Install it once with: python3 -m pip install -e '.[inference]'" >&2
    exit 1
fi

VLLM_PID=""

stop_server() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Stopping vLLM..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    VLLM_PID=""
}

wait_for_server() {
    echo "Waiting for vLLM to become ready..."
    until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
        sleep 10
    done
    echo "vLLM is ready."
}

trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 -m bias_audit.cli validate
mkdir -p "${REPO_ROOT}/outputs/validation" "${REPO_ROOT}/slurm/logs/vllm"

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL_PATH_COMMON}/${MODEL_NAME}"
    SERVER_LOG="${REPO_ROOT}/slurm/logs/vllm/${SLURM_JOB_ID:-local}-${MODEL_NAME}.log"

    echo "Serving ${MODEL_NAME} on port ${PORT}..."
    vllm serve "${MODEL_PATH}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --port "${PORT}" \
        >"${SERVER_LOG}" 2>&1 &
    VLLM_PID=$!

    wait_for_server

    python3 "${REPO_ROOT}/src/run_validation.py" \
        --tasks "${TASKS[@]}" \
        --few-shot "${FEW_SHOT[@]}" \
        --languages "${LANGUAGES[@]}" \
        --variants "${VARIANTS[@]}" \
        --port "${PORT}"

    stop_server
done

echo "All validation runs completed."
