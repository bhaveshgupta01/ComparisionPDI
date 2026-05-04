#!/bin/bash
# =============================================================================
# Phase D re-extract on Phase E1 checkpoints (CKA at d=256)
# Hooks every nn.MultiheadAttention, captures attention weights and hidden
# states for the first 256 examples of random / seed=42 validation set.
# Outputs: phase_d_artifacts_deep/analysis_deep_e1/v{1..4}_phase_e_xl/
# =============================================================================
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=extract-e1
#SBATCH --output=logs/extract_e1_%j.out
#SBATCH --error=logs/extract_e1_%j.err

set -e
module purge
cd /scratch/$USER/ComparisionPDI
mkdir -p logs phase_d_artifacts_deep/analysis_deep_e1
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print('torch:', torch.__version__, 'gpu:', torch.cuda.get_device_name(0))"

python scripts/extract_deep_e1.py --variant all
echo "==> extraction done.  phase_d_artifacts_deep/analysis_deep_e1/"
