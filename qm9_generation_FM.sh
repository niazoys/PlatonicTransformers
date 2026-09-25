#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=qm9_generation_fm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=40:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"

PROJECT_DIR="$HOME/PlatonicTransformers"
QM9_DATA_DIR="${QM9_DATA_DIR:-$PROJECT_DIR/datasets/qm9_gen}"
S_CHURN="${S_CHURN:-0.0}"
COMPILE="${COMPILE:-false}"
LR="${LR:-1.5e-4}"  

cd "$PROJECT_DIR"

if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    echo "Missing Platonic Transformers environment: $PROJECT_DIR/.venv/bin/python" >&2
    echo "Create it first with setup.sh or uv venv -p 3.12.4 && uv pip install -r requirements.txt" >&2
    exit 1
fi

if [[ ! -f "$QM9_DATA_DIR/gdb9.sdf" || ! -f "$QM9_DATA_DIR/gdb9.sdf.csv" ]]; then
    echo "QM9 generation files were not found in: $QM9_DATA_DIR" >&2
    echo "Expected gdb9.sdf and gdb9.sdf.csv." >&2
    exit 1
fi

if [[ ! -f "$QM9_DATA_DIR/processed_qm9_data.pkl" ]]; then
    echo "Warning: processed_qm9_data.pkl is missing; the first run will preprocess QM9." >&2
fi

srun "$PROJECT_DIR/.venv/bin/python" mains/main_qm9_gen_FM.py \
    --dataset.data_dir="$QM9_DATA_DIR" \
    --system.gpus=1 \
    --logging.enabled=true \
    --logging.wandb_identity=null \
    --model.compile="$COMPILE" \
    --diffusion.S_churn="$S_CHURN" \
    --optimizer.lr="$LR"
