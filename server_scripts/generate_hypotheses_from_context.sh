#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128gb
#SBATCH --time=0-03:00:00
#SBATCH --job-name=gen_hypotheses_from_context-2
#SBATCH --output=out/gen_hypotheses_from_context-dpo-2.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:4

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
source ~/disk/venvs/pnw/bin/activate

GET_CONFIG="python3 src/utils/config.py"

MY_DISK=$($GET_CONFIG paths.base_path)

GEN_MODEL=$($GET_CONFIG models.generation_model)
REASONING_PARSER=$($GET_CONFIG inference.reasoning_parser)
MODELS_DIR=$MY_DISK/$($GET_CONFIG paths.models_dir)
COMPILE_CACHE_DIR=$MY_DISK/$($GET_CONFIG paths.compile_cache_dir)
CACHE_DIR=$MY_DISK/$($GET_CONFIG paths.cache_dir)

PROMPTS_DIR=$MY_DISK/$($GET_CONFIG paths.prompts_dir) 
HYPOTHESES_DIR=$MY_DISK/$($GET_CONFIG paths.hypotheses_dir)

GPU_MEMORY_UTILIZATION=$($GET_CONFIG hardware.gpu_memory_utilization)
NUM_GPUS=$($GET_CONFIG hardware.num_gpus)
MAX_MODEL_LEN=$($GET_CONFIG inference.max_model_len)

_trimmed="${REASONING_PARSER//[[:space:]]/}"
_lower=$(printf '%s' "$_trimmed" | tr '[:upper:]' '[:lower:]')

if [[ -n "$_trimmed" && "$_lower" != "none" && "$_lower" != "null" && "$_lower" != "~" ]]; then
    REASONING_ARG=(--reasoning-parser "$REASONING_PARSER")
else
    REASONING_ARG=()
fi

# ---------------------------------------------------------------------------
# Parse --backend from the forwarded arguments (default: vllm)
# ---------------------------------------------------------------------------
BACKEND="vllm"
for arg in "$@"; do
    case "$arg" in
        --backend=*) BACKEND="${arg#--backend=}" ;;
        --backend)   ;;  # value is the next arg — handled below
    esac
done
# Two-argument form: --backend vllm
for i in $(seq 1 $#); do
    if [[ "${!i}" == "--backend" ]]; then
        j=$((i + 1))
        BACKEND="${!j}"
    fi
done

echo "Configuration Loaded:"
echo "  Root Path:                   $MY_DISK"
echo "  Generation Model:            $GEN_MODEL"
echo "  Backend:                     $BACKEND"
echo "  Max Context Lenght:          $MAX_MODEL_LEN"
echo "  Reasoning Parser:            $REASONING_PARSER"
echo "  Prompts Path:                $PROMPTS_DIR"
echo "  Hypotheses Path:             $HYPOTHESES_DIR"
echo "  GPUS:                        $NUM_GPUS"
echo "  GPU Memory Utilization:      $GPU_MEMORY_UTILIZATION"
echo "  Compile Cache:               $COMPILE_CACHE_DIR"
echo "  Cache Dir:                   $CACHE_DIR"



export TORCHINDUCTOR_CACHE_DIR=$MY_DISK/$CACHE_DIR/torch_inductor
export VLLM_CACHE_ROOT=$MY_DISK/$CACHE_DIR/vllm
export HF_HOME=$MY_DISK/$CACHE_DIR/hf
export TRITON_CACHE_DIR=$MY_DISK/$CACHE_DIR/triton
export FLASHINFER_CACHE_DIR=$MY_DISK/$CACHE_DIR/flashinfer
export TORCH_EXTENSIONS_DIR=$MY_DISK/$CACHE_DIR/torch_extensions

mkdir -p $MODELS_DIR
mkdir -p $HYPOTHESES_DIR

GENERATOR_PID=""

# ---------------------------------------------------------------------------
# Start vLLM generation server only when using the vllm backend
# ---------------------------------------------------------------------------
if [[ "$BACKEND" == "vllm" ]]; then
    GENERATOR_PORT=$($GET_CONFIG models.generator_port)

    echo "  Generator Port:   $GENERATOR_PORT"
    echo "  GPU Memory:       $GPU_MEMORY_UTILIZATION"
    echo "  Max Context Len:  $MAX_MODEL_LEN"

    rm -rf "$COMPILE_CACHE_DIR"
    mkdir -p $COMPILE_CACHE_DIR

    echo "Starting generation vLLM server, port $GENERATOR_PORT ..."
    vllm serve $GEN_MODEL \
        --served-model-name $GEN_MODEL \
        "${REASONING_ARG[@]}" \
        --trust-remote-code \
        --logprobs-mode processed_logprobs \
        --port $GENERATOR_PORT \
        --tensor-parallel-size $NUM_GPUS \
        --download-dir "$MODELS_DIR" \
        --compilation-config "{\"cache_dir\": \"$COMPILE_CACHE_DIR/generator\"}" \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --max-model-len $MAX_MODEL_LEN \
        --dtype bfloat16 \
        --gdn-prefill-backend triton \
        --host 0.0.0.0 &

    GENERATOR_PID=$!

    echo "Waiting for generation server to be ready (port $GENERATOR_PORT) ..."
    while ! curl -s http://localhost:$GENERATOR_PORT/health > /dev/null; do
        sleep 15
        echo "  Generator still loading ..."
    done
    echo "Generator is ready!"
fi

# ---------------------------------------------------------------------------
# Run the Python script.
# All arguments passed to this sbatch script are forwarded as-is,
# e.g.:  sbatch generate_hypotheses_from_context.sh --backend gemini --mode accepted_only
# ---------------------------------------------------------------------------
echo "Starting hypothesis generation from context ..."
python3 scripts/dataset_generation/generate_hypotheses_from_context.py "$@"
PYTHON_EXIT=$?

if [[ -n "$GENERATOR_PID" ]]; then
    echo "Shutting down vLLM server ..."
    kill $GENERATOR_PID
    wait $GENERATOR_PID 2>/dev/null
fi

exit $PYTHON_EXIT
