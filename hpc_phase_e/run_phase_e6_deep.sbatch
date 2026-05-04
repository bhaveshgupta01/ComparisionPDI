#!/bin/bash
# =============================================================================
# Phase E6 — Depth-axis ablation at d=128
# Width held at Phase C level; depth doubled.
# Tests whether the late-fusion advantage we see at (d=256, n=6) also holds at
# (d=128, n=12) — i.e. is the reversal driven by width specifically or by
# capacity in general?
#
# Submit:  sbatch run_phase_e6_deep.sbatch <variant> <seed>
# Splits restricted to random for cost; 3 seeds.
# =============================================================================
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase-e6
#SBATCH --output=logs/phase_e6_%x_%j.out
#SBATCH --error=logs/phase_e6_%x_%j.err

set -e

VARIANT="${1:?usage: sbatch run_phase_e6_deep.sbatch <variant> <seed>}"
SEED="${2:?missing seed}"
SPLIT="random"

# n_layers doubled from Phase C: V1/V2 shared=12, V3/V4 per-side=6 (so 12 total)
# Early variants (V1/V2) need bs=32 because their concat sequence is 1300 tokens;
# at n_layers=12 + bs=64 + h=4, the attention matrix forward+backward = ~40 GiB OOM.
# Late variants (V3/V4) have separate encoders with smaller per-modality seq, fit at bs=64.
case "$VARIANT" in
    early_concat|early_crossattn) NLAYERS=12 ; BATCH=32 ;;
    late_crossattn|late_concat)   NLAYERS=6  ; BATCH=64 ;;
    *) echo "Unknown variant: $VARIANT"; exit 1 ;;
esac

TAG="phase_e6_${VARIANT}_${SPLIT}_seed${SEED}"

module purge
cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/phase_e6
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print('torch:', torch.__version__, 'gpu:', torch.cuda.get_device_name(0))"
echo "Variant: $VARIANT  Split: $SPLIT  Seed: $SEED  n_layers: $NLAYERS  d=128"

python scripts/train.py \
    --variant     "$VARIANT" \
    --split       "$SPLIT" \
    --seed        "$SEED" \
    --output_dir  "outputs/phase_e6/${TAG}" \
    --d_model     128 \
    --n_heads     4 \
    --n_layers    "$NLAYERS" \
    --batch_size  "$BATCH" \
    --lr          3e-4 \
    --dropout     0.1 \
    --epochs      30

echo "==> done.  outputs/phase_e6/${TAG}/"
