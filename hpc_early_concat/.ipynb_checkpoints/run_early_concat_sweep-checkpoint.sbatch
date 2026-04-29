#!/bin/bash
# ==============================================================================
# Early Concat — single-GPU L4 sweep template (Phase A)
# Submit one job:
#   sbatch run_early_concat_sweep.sbatch <tag> <extra_args...>
# ==============================================================================
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:30:00
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=ec-sweep
#SBATCH --output=logs/ec_%x_%j.out
#SBATCH --error=logs/ec_%x_%j.err

TAG="${1:-untagged}"
shift  # rest of args are passed to train.py

module purge

cd /scratch/$USER/ComparisionPDI
mkdir -p logs

source .venv/bin/activate

echo "==================== ENV ===================="
which python
python --version
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
echo "==================== RUN ===================="
echo "Tag:      $TAG"
echo "Variant:  early_concat"
echo "Args:     $@"
echo "============================================="

# Tag the run via output_dir so configs don't overwrite each other
python scripts/train.py \
    --variant early_concat \
    --output_dir "outputs/sweeps/early_concat/${TAG}" \
    "$@"

echo "==================== DONE ===================="
echo "Results:  outputs/sweeps/early_concat/${TAG}/"
