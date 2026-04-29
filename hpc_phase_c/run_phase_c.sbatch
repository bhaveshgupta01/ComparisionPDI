#!/bin/bash
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase-c
#SBATCH --output=logs/phase_c_%x_%j.out
#SBATCH --error=logs/phase_c_%x_%j.err

set -e

VARIANT="${1:?usage: sbatch run_phase_c.sbatch <variant> <split> <seed>}"
SPLIT="${2:?missing split}"
SEED="${3:?missing seed}"

case "$VARIANT" in
    early_concat|early_crossattn) NLAYERS=6 ;;
    late_crossattn|late_concat)   NLAYERS=3 ;;
    *) echo "Unknown variant: $VARIANT"; exit 1 ;;
esac

TAG="phase_c_${VARIANT}_${SPLIT}_seed${SEED}"

module purge
cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/phase_c
source .venv/bin/activate

echo "==================== ENV ===================="
which python
python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0))"
echo "==================== RUN ===================="
echo "Variant:  $VARIANT"
echo "Split:    $SPLIT"
echo "Seed:     $SEED"
echo "n_layers: $NLAYERS"
echo "Tag:      $TAG"
echo "============================================="

python scripts/train.py \
    --variant   "$VARIANT" \
    --split     "$SPLIT" \
    --seed      "$SEED" \
    --output_dir "outputs/phase_c/${TAG}" \
    --d_model   128 \
    --n_heads   4 \
    --n_layers  "$NLAYERS" \
    --batch_size 64 \
    --lr        3e-4 \
    --dropout   0.1 \
    --epochs    30

echo "==================== DONE ===================="
echo "Results:  outputs/phase_c/${TAG}/"
