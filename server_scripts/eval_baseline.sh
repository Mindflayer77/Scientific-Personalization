#!/bin/bash

#SBATCH -N 1    # CPU nodes
#SBATCH -c 8    #number of CPU cores
#SBATCH --mem=64gb
#SBATCH --time=0-03:00:00    # format dd-hh:mm:ss np. 1-12:30:15
#SBATCH --job-name=eval_baselines
#SBATCH --output=out/eval_baselines.out
#SBATCH -p lem-gpu-short   # partition
#SBATCH --gres=gpu:hopper:1     #format gpu:${nazwa_zasobu}:${ilość}

MY_DISK=$1
WANDB_API_KEY=$2
EVALUATION_CONFIG_PATH=$3

echo "Setting up environment..."

source /usr/local/sbin/modules.sh

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.0.0
source ~/disk/venvs/train-2/bin/activate # z wykorzystaniem srodowiska wirtualnego

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

# export WANDB_API_KEY=$WANDB_API_KEY
export TORCHINDUCTOR_CACHE_DIR=$MY_DISK/.cache_3/torch_inductor
export VLLM_CACHE_ROOT=$MY_DISK/.cache_3/vllm
export HF_HOME=$MY_DISK/.cache_3/hf
export TRITON_CACHE_DIR=$MY_DISK/.cache_3/triton
export FLASHINFER_CACHE_DIR=$MY_DISK/.cache_3/flashinfer
export TORCH_EXTENSIONS_DIR=$MY_DISK/.cache_3/torch_extensions

# Start nvidia-smi monitoring in background
while true; do nvidia-smi > out/nvidia_smi.log 2>&1; sleep 5; done &
WATCH_PID=$!

echo "Starting Evaluation..."
python3 scripts/eval_baseline.py --config $EVALUATION_CONFIG_PATH

# Kill the watch process when training completes
kill $WATCH_PID 2>/dev/null
