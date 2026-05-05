#!/bin/bash
# =============================================================================
# Phase E1 — XL Full-Scale Re-Run (single job)
# Submit:  sbatch run_phase_e_xl.sbatch <variant> <split> <seed>
# Examples:
#   sbatch run_phase_e_xl.sbatch early_concat    random      42
#   sbatch run_phase_e_xl.sbatch late_crossattn  cold_target 2024
# =============================================================================
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase-e-xl
#SBATCH --output=logs/phase_e_xl_%x_%j.out
#SBATCH --error=logs/phase_e_xl_%x_%j.err

set -e

VARIANT="${1:?usage: sbatch run_phase_e_xl.sbatch <variant> <split> <seed>}"
SPLIT="${2:?missing split}"
SEED="${3:?missing seed}"

# Per-variant n_layers — XL doubles WIDTH only; depth held at Phase C level
# (we tried n_layers=12/6 and OOM'd at d=256+h=8+1300-tok seq; revised 2026-04-30)
case "$VARIANT" in
    early_concat|early_crossattn) NLAYERS=6 ;;
    late_crossattn|late_concat)   NLAYERS=3 ;;
    *) echo "Unknown variant: $VARIANT"; exit 1 ;;
esac

TAG="phase_e_xl_${VARIANT}_${SPLIT}_seed${SEED}"

module purge
cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/phase_e_xl
source .venv/bin/activate

# Reduce CUDA memory fragmentation (recommended by the OOM error from earlier smoke)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==================== ENV ===================="
which python
python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "==================== RUN ===================="
echo "Variant:  $VARIANT"
echo "Split:    $SPLIT"
echo "Seed:     $SEED"
echo "n_layers: $NLAYERS"
echo "Tag:      $TAG"
echo "============================================="

# Train with the Phase E XL config
python scripts/train.py \
    --variant     "$VARIANT" \
    --split       "$SPLIT" \
    --seed        "$SEED" \
    --output_dir  "outputs/phase_e_xl/${TAG}" \
    --d_model     256 \
    --n_heads     8 \
    --n_layers    "$NLAYERS" \
    --batch_size  32 \
    --lr          3e-4 \
    --dropout     0.1 \
    --epochs      45

echo "==================== DONE ===================="
echo "Results:  outputs/phase_e_xl/${TAG}/"
