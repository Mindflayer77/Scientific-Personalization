#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64gb
#SBATCH --time=0-01:00:00
#SBATCH --job-name=gen_hypotheses_from_candidates
#SBATCH --output=out/gen_hypotheses_from_candidates_2.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:2

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0
source ~/disk/venvs/pnw/bin/activate

GET_CONFIG="python3 src/utils/config.py"

MY_DISK=$($GET_CONFIG paths.base_path)

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

echo "Configuration Loaded:"
echo "  Root Path:                     $MY_DISK"
echo "  Models Path:                   $MODELS_DIR"
echo "  Embedding Model:               $EMBEDDING_MODEL"
echo "  Reranker Model:                $RERANKER_MODEL"
echo "  Embedder GPU (cuda:0) budget:  $EMBEDDER_GPU_MEM"
echo "  Reranker GPU (cuda:1) budget:  $RERANKER_GPU_MEM"

export TORCHINDUCTOR_CACHE_DIR=$MY_DISK/$CACHE_DIR/torch_inductor
export VLLM_CACHE_ROOT=$MY_DISK/$CACHE_DIR/vllm
export HF_HOME=$MY_DISK/$CACHE_DIR/hf
export TRITON_CACHE_DIR=$MY_DISK/$CACHE_DIR/triton
export FLASHINFER_CACHE_DIR=$MY_DISK/$CACHE_DIR/flashinfer
export TORCH_EXTENSIONS_DIR=$MY_DISK/$CACHE_DIR/torch_extensions

# Set Apptainer Cache to prevent filling up the home directory
export APPTAINER_CACHEDIR=$MY_DISK/$CACHE_DIR/apptainer

rm -rf ./weaviate_data/*

mkdir -p $MODELS_DIR
rm -rf "$COMPILE_CACHE_DIR"
mkdir -p $COMPILE_CACHE_DIR
mkdir -p $PROMPTS_DIR
mkdir -p $HYPOTHESES_DIR
mkdir -p ./weaviate_data

# ---------------------------------------------------------------------------
# Start Weaviate Database via Apptainer
# ---------------------------------------------------------------------------
echo "Starting Weaviate database container with Apptainer..."

# Pass environment variables to the Apptainer container
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
export APPTAINERENV_GOMEMLIMIT="22GiB"
export APPTAINERENV_DISABLE_TELEMETRY="true"
export APPTAINERENV_LIMIT_RESOURCES="true"

# Run Apptainer in the background, mapping the backup dir and data dir
apptainer run \
    --bind ./weaviate_data:/var/lib/weaviate \
    --bind ./weaviate_backups:/var/lib/weaviate/backups \
    docker://semitechnologies/weaviate:1.36.6 \
    --host 0.0.0.0 --port 8080 --scheme http &
    
WEAVIATE_PID=$!

echo "Waiting for Weaviate to initialize (port 8080)..."
while ! curl -s http://localhost:8080/v1/.well-known/ready > /dev/null; do
    sleep 5
    echo "  Weaviate is still starting..."
done
echo "Weaviate is ready!"

# ---------------------------------------------------------------------------
# Restore Database from Backup
# ---------------------------------------------------------------------------
echo "Restoring Weaviate collection from backup..."
python3 scripts/weaviate_db/import_backup.py

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
echo "Starting Hypotheses Generation..."
python3 scripts/dataset_generation/generate_hypotheses_from_subqueries_weaviate_hybrid.py
PYTHON_EXIT=$?

# ---------------------------------------------------------------------------
# Shutdown and Cleanup
# ---------------------------------------------------------------------------
echo "Shutting down servers (vLLM and Weaviate)..."
kill $EMBEDDER_PID $RERANKER_PID $WEAVIATE_PID
wait $EMBEDDER_PID $RERANKER_PID $WEAVIATE_PID 2>/dev/null

exit $PYTHON_EXIT
