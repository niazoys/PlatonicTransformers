#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail

# ─── Environment ─────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plato-dali

module load 2025
module load CUDA/12.8.0

# Fix: ensure nvcc is on PATH ahead of any stale conda-bundled one
CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

export DALI_NO_MMAP=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Verify ──────────────────────────────────────────────────────────────────
echo "Node:  $(hostname)"
echo "GPUs:  $(nvidia-smi -L)"
echo "nvcc:  $(nvcc --version | tail -1)"
echo "CUDA_HOME: ${CUDA_HOME}"

# ─── Run ─────────────────────────────────────────────────────────────────────
cd /gpfs/work4/0/prjs1709/PlatonicTransformers
mkdir -p logs

python mains/main_imagenet.py \
    --config configs/imagenet_dali.yaml \
    --dataset.data_dir=/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder \
    --training.epochs=800 \
    --training.batch_size=256 \
    --training.accumulate_grad_batches=2 \
    --model.num_heads=8 \
    --system.gpus=4 \
    --logging.enabled=true 2>&1
