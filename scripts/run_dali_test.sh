#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail

# ─── Environment ─────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plato-dali

export DALI_NO_MMAP=1

# CUDA module
module load 2025
module load CUDA/12.8.0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Run ─────────────────────────────────────────────────────────────────────
cd /gpfs/work4/0/prjs1709/PlatonicTransformers
mkdir -p logs

python tests/test_imagenet_dali.py \
    --data_dir /scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder \
    --batch_size 32 \
    --image_size 224 \
    --patch_size 16 \
    --num_batches 3
