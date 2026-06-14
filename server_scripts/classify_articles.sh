#!/bin/bash

#SBATCH -N 1    # CPU nodes
#SBATCH -c 4    # number of CPU cores
#SBATCH --mem=128gb
#SBATCH --time=0-07:00:00    # format dd-hh:mm:ss np. 1-12:30:15
#SBATCH --job-name=classify_articles
#SBATCH --output=out/classify_articles.out
#SBATCH -p lem-gpu-normal    # partition
#SBATCH --gres=gpu:hopper:4 #format gpu:${nazwa_zasobu}:${ilość} np. gpu:tesla:1

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0
source ~/disk/venvs/pnw/bin/activate # z wykorzystaniem srodowiska wirtualnego

GET_CONFIG="python3 src/utils/config.py"

MY_DISK=$($GET_CONFIG paths.base_path) 
CACHE_DIR=$MY_DISK/$($GET_CONFIG paths.cache_dir)

GEN_MODEL=$($GET_CONFIG models.generation_model)
MODELS_DIR=$MY_DISK/$($GET_CONFIG paths.models_dir) 
COMPILE_CACHE_DIR=$MY_DISK/$($GET_CONFIG paths.compile_cache_dir) 
REASONING_PARSER=$($GET_CONFIG inference.reasoning_parser)

PROMPTS_DIR=$MY_DISK/$($GET_CONFIG paths.prompts_dir) 

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

echo "Configuration Loaded:"
echo "  Root Path:                   $MY_DISK"
echo "  Models Path:                 $MODELS_DIR"
echo "  Model:                       $GEN_MODEL"
echo "  Max Context Lenght:          $MAX_MODEL_LEN"
echo "  Reasoning Parser:            $REASONING_PARSER"
echo "  Prompts Path:                $PROMPTS_DIR"
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
rm -rf "$COMPILE_CACHE_DIR"
mkdir -p $COMPILE_CACHE_DIR
mkdir -p $PROMPTS_DIR

# Start nvidia-smi monitoring in background
while true; do nvidia-smi > out/nvidia_smi.log 2>&1; sleep 5; done &
WATCH_PID=$!

echo "Starting vLLM server..."
vllm serve $GEN_MODEL \
    --served-model-name $GEN_MODEL \
    "${REASONING_ARG[@]}" \
    --logprobs-mode processed_logprobs \
    --port 8000 \
    --download-dir "$MODELS_DIR" \
    --compilation-config "{\"cache_dir\": \"$COMPILE_CACHE_DIR\"}" \
    --tensor-parallel-size $NUM_GPUS \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --enable-expert-parallel \
    --language-model-only \
    --enable-prefix-caching \
    --gdn-prefill-backend triton \
    --host 0.0.0.0 &

VLLM_PID=$!

echo "Waiting for vLLM to load model..."
while ! curl -s http://localhost:8000/health > /dev/null; do
    sleep 10
    echo "Still loading..."
done
echo "vLLM is ready!"

echo "Starting Article Classification..."
python3 scripts/dataset_generation/classify_articles.py

kill $VLLM_PID
kill $WATCH_PID
