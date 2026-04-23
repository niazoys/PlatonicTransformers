#!/bin/bash
#SBATCH --job-name=qm9regr
#SBATCH --partition=geodude
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --output=/home/ebekker/platonic-transformers/logs/%x_%j.out
#SBATCH --error=/home/ebekker/platonic-transformers/logs/%x_%j.err

# Usage:
#   sbatch --job-name=qm9regr-trivial scripts/ivi/qm9_regr_slurm.sh trivial_3
#   sbatch --job-name=qm9regr-tetra   scripts/ivi/qm9_regr_slurm.sh tetrahedron
# First positional arg = solid_name (trivial_3 | tetrahedron | octahedron | icosahedron)
# Second positional arg (optional) = extra CLI overrides

set -euo pipefail

SOLID="${1:-tetrahedron}"
EXTRA="${2:-}"

cd /home/ebekker/platonic-transformers
source .venv/bin/activate

module load gnu12/12.4.0 2>/dev/null || true
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export WANDB__SERVICE_WAIT=120
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

mkdir -p logs

echo "=== starting qm9 regr run: solid=$SOLID extra=$EXTRA ==="
nvidia-smi
echo "========================================"

python mains/main_qm9_regr.py \
    --config configs/qm9_regr.yaml \
    --model.solid_name=$SOLID \
    --dataset.data_dir=/home/ebekker/data/qm9 \
    --system.num_workers=4 \
    $EXTRA
