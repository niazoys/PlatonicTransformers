#!/bin/bash
#SBATCH --job-name=extract-imagenet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --account=all6000users
#SBATCH --partition=all
#SBATCH --output=logs/extract_imagenet_%j.out

set -eo pipefail

# Source environment variables (e.g. HF_TOKEN)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export IMAGENET_PATH=${IMAGENET_PATH:-/ivi/zfs/s0/original_homes/dwessel/data}

PYTHONPATH=. python scripts/extract_imagenet_to_folder.py
