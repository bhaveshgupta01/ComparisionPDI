#!/bin/bash
# =============================================================================
# Phase E1 — XL SMOKE TEST (1 epoch, 1 variant, 1 split, 1 seed)
# Submit FIRST before the 60-job sweep — confirms d=256/n_layers=12 doesn't OOM,
# checks step time, validates checkpoint shape.
#
# Submit:  sbatch run_phase_e_xl_smoke.sbatch
# Expected: ~5-8 min wall-clock. If this passes, fire the full 60-job sweep.
# =============================================================================
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:20:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase-e-xl-smoke
#SBATCH --output=logs/phase_e_xl_smoke_%j.out
#SBATCH --error=logs/phase_e_xl_smoke_%j.err

set -e
module purge
cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/phase_e_xl_smoke
source .venv/bin/activate

# Reduce CUDA memory fragmentation (the previous OOM error explicitly suggested this)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print('torch:', torch.__version__, 'gpu:', torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader || true

# Smoke test — heaviest variant (V2 cross-attn) at REVISED XL config, 1 epoch.
# Width doubled (d=256, h=8) but depth held at Phase C level (n_layers=6) to fit.
# If V2 fits, all variants fit.
python scripts/train.py \
    --variant     early_crossattn \
    --split       random \
    --seed        42 \
    --output_dir  "outputs/phase_e_xl_smoke/v2_smoke" \
    --d_model     256 \
    --n_heads     8 \
    --n_layers    6 \
    --batch_size  32 \
    --lr          3e-4 \
    --dropout     0.1 \
    --epochs      1

echo "==> SMOKE OK — fire the full sweep:"
echo "    bash hpc_phase_e/submit_phase_e_xl_all.sh"
