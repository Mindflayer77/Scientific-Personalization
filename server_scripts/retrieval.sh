#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=128gb
#SBATCH --time=0-02:00:00
#SBATCH --job-name=retrieval
#SBATCH --output=out/retrieval-dpo.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:2

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
source ~/disk/venvs/pnw/bin/activate

# ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GET_CONFIG="python3 src/utils/config.py"

MY_DISK=$($GET_CONFIG paths.base_path)
WEAVIATE_MODE=$($GET_CONFIG weaviate.mode 2>/dev/null | tr '[:upper:]' '[:lower:]')
WEAVIATE_MODE=${WEAVIATE_MODE:-remote}

if [[ "$WEAVIATE_MODE" != "remote" && "$WEAVIATE_MODE" != "local" ]]; then
    echo "ERROR: Invalid weaviate.mode='$WEAVIATE_MODE'. Expected 'remote' or 'local'."
    exit 1
fi

ENV_DB_FILE="$MY_DISK/.env_db"

if [ -f "$ENV_DB_FILE" ]; then
    set -a
    source "$ENV_DB_FILE"
    set +a
else
    if [ "$WEAVIATE_MODE" = "remote" ]; then
        echo "ERROR: .env_db file not found at $ENV_DB_FILE"
        exit 1
    fi
    echo "WARNING: .env_db file not found at $ENV_DB_FILE (continuing in local Weaviate mode)"
fi



EMBEDDING_MODEL=$($GET_CONFIG models.embedding_model)
RERANKER_MODEL=$($GET_CONFIG models.reranker_model)
MODELS_DIR=$MY_DISK/$($GET_CONFIG paths.models_dir)
COMPILE_CACHE_DIR=$MY_DISK/$($GET_CONFIG paths.compile_cache_dir)
CACHE_DIR=$MY_DISK/$($GET_CONFIG paths.cache_dir)
ARTICLES_DIR=$MY_DISK/$($GET_CONFIG paths.articles_dir)

PROMPTS_DIR=$MY_DISK/$($GET_CONFIG paths.prompts_dir)
HYPOTHESES_DIR=$MY_DISK/$($GET_CONFIG paths.hypotheses_dir)

# ---------------------------------------------------------------------------
# VRAM budget (2x H100, one model + dense index per GPU)
# ---------------------------------------------------------------------------
EMBEDDER_GPU_MEM=$($GET_CONFIG hardware.embedder_gpu_memory_utilization 2>/dev/null)
EMBEDDER_GPU_MEM=${EMBEDDER_GPU_MEM:-0.40}

RERANKER_GPU_MEM=$($GET_CONFIG hardware.reranker_gpu_memory_utilization 2>/dev/null)
RERANKER_GPU_MEM=${RERANKER_GPU_MEM:-0.70}

MAX_MODEL_LEN=$($GET_CONFIG inference.max_model_len)

EMBEDDER_PORT=$($GET_CONFIG models.embedder_port)
RERANKER_PORT=$($GET_CONFIG models.reranker_port)

if [ "$WEAVIATE_MODE" = "local" ]; then
    WEAVIATE_PORT=$($GET_CONFIG weaviate.local_port 2>/dev/null)
    if [ -z "$WEAVIATE_PORT" ]; then
        WEAVIATE_PORT=$($GET_CONFIG weaviate.port 2>/dev/null)
    fi
    WEAVIATE_PORT=${WEAVIATE_PORT:-8080}
    WEAVIATE_HOST="localhost"
else
    WEAVIATE_PORT=$($GET_CONFIG weaviate.port 2>/dev/null)
    WEAVIATE_PORT=${WEAVIATE_PORT:-8080}
    WEAVIATE_HOST=${VPS_IP}
fi

WEAVIATE_PID=""

echo "Configuration Loaded:"
echo "  Root Path:                     $MY_DISK"
echo "  Weaviate Mode:                 $WEAVIATE_MODE"
echo "  Models Path:                   $MODELS_DIR"
echo "  Embedding Model:               $EMBEDDING_MODEL"
echo "  Reranker Model:                $RERANKER_MODEL"
echo "  Embedder GPU (cuda:0) budget:  $EMBEDDER_GPU_MEM"
echo "  Reranker GPU (cuda:1) budget:  $RERANKER_GPU_MEM"
echo "  Weaviate API Host:             $WEAVIATE_HOST:$WEAVIATE_PORT"

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
mkdir -p $HYPOTHESES_DIR

# ---------------------------------------------------------------------------
# Initialize Weaviate (remote or local)
# ---------------------------------------------------------------------------
if [ "$WEAVIATE_MODE" = "remote" ]; then
    if [ -z "$WEAVIATE_HOST" ]; then
        echo "ERROR: VPS_IP is not set in .env_db"
        exit 1
    fi

    if [ -z "$WEAVIATE_API_KEY" ]; then
        echo "ERROR: WEAVIATE_API_KEY is not set in .env_db"
        exit 1
    fi

    echo "Checking remote Weaviate API readiness at $WEAVIATE_HOST:$WEAVIATE_PORT..."
    while ! curl -s -H "X-API-KEY: $WEAVIATE_API_KEY" \
        "http://$WEAVIATE_HOST:$WEAVIATE_PORT/v1/.well-known/ready" > /dev/null; do
        sleep 5
        echo "  Remote Weaviate API not ready yet..."
    done
    echo "Remote Weaviate API is reachable."
else
    echo "Starting local Weaviate database container with Apptainer..."

    export APPTAINER_CACHEDIR=$MY_DISK/$CACHE_DIR/apptainer
    export APPTAINERENV_DEFAULT_VECTORIZER_MODULE="none"
    export APPTAINERENV_ENABLE_MODULES="backup-filesystem"
    export APPTAINERENV_BACKUP_FILESYSTEM_PATH="/var/lib/weaviate/backups"
    export APPTAINERENV_QUERY_DEFAULTS_LIMIT=25
    export APPTAINERENV_BACKUP_TIMEOUT="3600"
    export APPTAINERENV_BACKUP_CHUNK_SIZE="536870912"
    export APPTAINERENV_AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED="true"
    export APPTAINERENV_PERSISTENCE_DATA_PATH="/var/lib/weaviate"
    export APPTAINERENV_GOGC=100
    export APPTAINERENV_CLUSTER_HOSTNAME="node1"
    export APPTAINERENV_GOMEMLIMIT="30GiB"
    export APPTAINERENV_DISABLE_TELEMETRY="true"
    export APPTAINERENV_LIMIT_RESOURCES="true"

    # WEAVIATE_DATA_DIR="$MY_DISK/weaviate_data"
    # WEAVIATE_BACKUP_DIR="$MY_DISK/weaviate_backups"

    mkdir -p ./weaviate_data
    mkdir -p ./weaviate_backups
    rm -rf ./weaviate_data/*

    apptainer run \
        --bind ./weaviate_data:/var/lib/weaviate \
        --bind ./weaviate_backups:/var/lib/weaviate/backups \
        docker://semitechnologies/weaviate:1.36.6 \
        --host 0.0.0.0 --port $WEAVIATE_PORT --scheme http &

    WEAVIATE_PID=$!

    echo "Waiting for local Weaviate to initialize (port $WEAVIATE_PORT)..."
    while ! curl -s "http://localhost:$WEAVIATE_PORT/v1/.well-known/ready" > /dev/null; do
        sleep 5
        echo "  Local Weaviate is still starting..."
    done
    echo "Local Weaviate is ready!"

    echo "Restoring Weaviate collection from backup..."
    python3 scripts/weaviate_db/import_backup.py
fi

# ---------------------------------------------------------------------------
# Start embedder server — pinned to GPU 0
# ---------------------------------------------------------------------------
echo "Starting embedder vLLM server on GPU 0, port $EMBEDDER_PORT..."
CUDA_VISIBLE_DEVICES=0 vllm serve $EMBEDDING_MODEL \
    --served-model-name $EMBEDDING_MODEL \
    --trust-remote-code \
    --runner pooling \
    --port $EMBEDDER_PORT \
    --tensor-parallel-size 1 \
    --download-dir "$MODELS_DIR" \
    --compilation-config "{\"cache_dir\": \"$COMPILE_CACHE_DIR/embedder\"}" \
    --gpu-memory-utilization $EMBEDDER_GPU_MEM \
    --max-model-len $MAX_MODEL_LEN \
    --dtype bfloat16 \
    --host 0.0.0.0 &

EMBEDDER_PID=$!

# ---------------------------------------------------------------------------
# Start reranker server — pinned to GPU 1
# ---------------------------------------------------------------------------
echo "Starting reranker vLLM server on GPU 1, port $RERANKER_PORT..."
CUDA_VISIBLE_DEVICES=1 vllm serve $RERANKER_MODEL \
    --served-model-name $RERANKER_MODEL \
    --trust-remote-code \
    --runner pooling \
    --port $RERANKER_PORT \
    --tensor-parallel-size 1 \
    --download-dir "$MODELS_DIR" \
    --compilation-config "{\"cache_dir\": \"$COMPILE_CACHE_DIR/reranker\"}" \
    --gpu-memory-utilization $RERANKER_GPU_MEM \
    --max-model-len $MAX_MODEL_LEN \
    --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
    --chat-template $MY_DISK/Personalization/chat_templates/qwen3_reranker.jinja \
    --dtype bfloat16 \
    --host 0.0.0.0 &

RERANKER_PID=$!

wait_for_server() {
    local name=$1
    local port=$2
    echo "Waiting for $name to load model (port $port)..."
    while ! curl -s http://localhost:$port/health > /dev/null; do
        sleep 10
        echo "  $name still loading..."
    done
    echo "$name is ready!"
}

wait_for_server "Embedder" $EMBEDDER_PORT &
WAIT_EMBEDDER_PID=$!
 
wait_for_server "Reranker" $RERANKER_PORT &
WAIT_RERANKER_PID=$!
 
wait $WAIT_EMBEDDER_PID $WAIT_RERANKER_PID

# ---------------------------------------------------------------------------
# Run Main Process
# ---------------------------------------------------------------------------
echo "Starting Retrieval..."
python3 scripts/dataset_generation/retrieval.py
PYTHON_EXIT=$?

# ---------------------------------------------------------------------------
# Shutdown and Cleanup
# ---------------------------------------------------------------------------
echo "Shutting down vLLM servers..."
if [ -n "$WEAVIATE_PID" ]; then
    kill $EMBEDDER_PID $RERANKER_PID $WEAVIATE_PID
    wait $EMBEDDER_PID $RERANKER_PID $WEAVIATE_PID 2>/dev/null
else
    kill $EMBEDDER_PID $RERANKER_PID
    wait $EMBEDDER_PID $RERANKER_PID 2>/dev/null
fi

exit $PYTHON_EXIT
