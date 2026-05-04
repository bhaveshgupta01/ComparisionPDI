#!/bin/bash
# =============================================================================
# Phase E4 — Causal ablations on Phase C checkpoints
# Inference-only. Loads each Phase C ckpt (random split, seed=42), runs:
#   - Layer ablation (zero each MultiheadAttention block)
#   - Head ablation  (zero each individual head)
#   - V3 <-> V4 drug encoder representation swap
#
# All 4 variants in one job since each is fast (~3-5 min on A100).
# Outputs: outputs/phase_e_ablations/{<variant>.json, rep_swap_v3_v4.json, SUMMARY.csv}
#
# Submit:  sbatch hpc_phase_e/run_phase_e4_ablations.sbatch
# =============================================================================
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:45:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase-e4
#SBATCH --output=logs/phase_e4_%j.out
#SBATCH --error=logs/phase_e4_%j.err

set -e
module purge
cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/phase_e_ablations
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print('torch:', torch.__version__, 'gpu:', torch.cuda.get_device_name(0))"

python scripts/e4_ablate.py --variant all --n_batches 4 --batch_size 32

echo "==> E4 done. Results in outputs/phase_e_ablations/"
