#!/bin/bash
# Launches the QM9-alpha regression sweep on IVI geodude.
# Reproduces/beats the paper's Trivial (0.028) and Tetrahedron (0.012) alpha rows
# while benchmarking value-RoPE + compile + flash attention, plus EMA yes/no and
# matmul high vs medium ablations.
#
# Sweep (6 runs):
#   1. trivial_3   EMA off  matmul high
#   2. trivial_3   EMA on   matmul high
#   3. trivial_3   EMA off  matmul medium
#   4. tetrahedron EMA off  matmul high
#   5. tetrahedron EMA on   matmul high
#   6. tetrahedron EMA off  matmul medium
#
# Run from the login node.

set -euo pipefail
cd ~/platonic-transformers

submit () {
    local name="$1"
    local solid="$2"
    local extra="$3"
    sbatch --job-name="$name" \
           --output=logs/%x_%j.out \
           --error=logs/%x_%j.err \
           scripts/ivi/qm9_regr_slurm.sh "$solid" "$extra"
}

# ---- trivial_3 (paper target: alpha 0.028) ----
submit alpha-trivial-baseline    trivial_3 "--logging.enabled=true"
submit alpha-trivial-ema         trivial_3 "--logging.enabled=true --training.ema_enabled=true"
submit alpha-trivial-mm-medium   trivial_3 "--logging.enabled=true --system.float32_matmul_precision=medium"

# ---- tetrahedron (paper target: alpha 0.012) ----
submit alpha-tetra-baseline      tetrahedron "--logging.enabled=true"
submit alpha-tetra-ema           tetrahedron "--logging.enabled=true --training.ema_enabled=true"
submit alpha-tetra-mm-medium     tetrahedron "--logging.enabled=true --system.float32_matmul_precision=medium"

squeue -u "$(whoami)"
