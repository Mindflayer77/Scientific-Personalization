#!/bin/bash

#SBATCH -N 1    # CPU nodes
#SBATCH -c 16    #number of CPU cores
#SBATCH --mem=128gb
#SBATCH --time=0-01:00:00    # format dd-hh:mm:ss np. 1-12:30:15
#SBATCH --job-name=train_baselines-4
#SBATCH --output=out/train_baselines-4.out
#SBATCH -p lem-gpu-short    # partition
#SBATCH --gres=gpu:hopper:4     #format gpu:${nazwa_zasobu}:${ilość}

MY_DISK=$1
WANDB_API_KEY=$2
TRAINING_CONFIG_PATH=$3
ACCELERATE_CONFIG_PATH=$4

echo "Setting up environment..."

source /usr/local/sbin/modules.sh

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.0.0
# source ~/disk/venvs/train-2/bin/activate # z wykorzystaniem srodowiska wirtualnego
source ~/disk/venvs/train-2/bin/activate

unset PYTHONPATH
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

echo "Starting Training..."
# #torchrun --nproc_per_node=2 scripts/train_baseline.py --config $TRAINING_CONFIG_PATH
# #python3 scripts/train_baseline.py --config $TRAINING_CONFIG_PATH
# echo "=== Python path ==="
# which python
# echo "=== PYTHONPATH ==="
# echo $PYTHONPATH
# echo "=== torch/torchvision versions ==="
# python -c "import torch, torchvision; print('torch CUDA:', torch.version.cuda); print('torchvision CUDA:', torchvision.version.cuda)"
# echo "=== LD_LIBRARY_PATH ==="
# echo $LD_LIBRARY_PATH

# echo "Python: $(which python)"
# echo "PYTHONPATH: $PYTHONPATH"
# python -c "
# import torch, torchvision
# print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)
# print('torchvision CUDA:', torchvision.version.cuda)
# "

accelerate launch --config-file $ACCELERATE_CONFIG_PATH  scripts/train_baseline.py --config $TRAINING_CONFIG_PATH

# Kill the watch process when training completes
kill $WATCH_PID 2>/dev/null
