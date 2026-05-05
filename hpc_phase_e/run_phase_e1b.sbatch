#!/bin/bash
# =============================================================================
# Phase E1b — Mid-scale probe (d=192) — bracket the phase transition
# Submit:  sbatch run_phase_e1b.sbatch <variant> <split> <seed>
# =============================================================================
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:30:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase-e1b
#SBATCH --output=logs/phase_e1b_%x_%j.out
#SBATCH --error=logs/phase_e1b_%x_%j.err

set -e

VARIANT="${1:?usage: sbatch run_phase_e1b.sbatch <variant> <split> <seed>}"
SPLIT="${2:?missing split}"
SEED="${3:?missing seed}"

case "$VARIANT" in
    early_concat|early_crossattn) NLAYERS=6 ;;
    late_crossattn|late_concat)   NLAYERS=3 ;;
    *) echo "Unknown variant: $VARIANT"; exit 1 ;;
esac

TAG="phase_e1b_${VARIANT}_${SPLIT}_seed${SEED}"

module purge
cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/phase_e1b
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==================== ENV ===================="
which python
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0))"
echo "==================== RUN ===================="
echo "Variant:  $VARIANT  Split: $SPLIT  Seed: $SEED  n_layers: $NLAYERS  d_model: 192"

python scripts/train.py \
    --variant     "$VARIANT" \
    --split       "$SPLIT" \
    --seed        "$SEED" \
    --output_dir  "outputs/phase_e1b/${TAG}" \
    --d_model     192 \
    --n_heads     6 \
    --n_layers    "$NLAYERS" \
    --batch_size  32 \
    --lr          3e-4 \
    --dropout     0.1 \
    --epochs      45

echo "==================== DONE ===================="
echo "Results:  outputs/phase_e1b/${TAG}/"
