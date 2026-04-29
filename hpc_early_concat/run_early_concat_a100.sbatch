#!/bin/bash
# ==============================================================================
# Early Concat — A100 High-Memory Template
# Use this for all Phase A runs to prevent OOM on large configs (dm256, bs128).
# ==============================================================================
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1     # Using A100 for higher VRAM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8              # Increased for A100
#SBATCH --time=3:00:00                 # Early Concat is slower, giving more time
#SBATCH --mem=64GB                     # Increased system memory
#SBATCH --gres=gpu:1                   # 1 High-capacity A100 GPU
#SBATCH --job-name=ec-a100
#SBATCH --output=logs/ec_a100_%x_%j.out
#SBATCH --error=logs/ec_a100_%x_%j.err

TAG="${1:-untagged}"
shift  # rest of args are passed to train.py

module purge

cd /scratch/$USER/ComparisionPDI
mkdir -p logs

source .venv/bin/activate

echo "==================== ENV ===================="
which python
python --version
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0))"
echo "==================== RUN ===================="
echo "Tag:      $TAG"
echo "Variant:  early_concat"
echo "Args:     $@"
echo "============================================="

python scripts/train.py \
    --variant early_concat \
    --output_dir "outputs/sweeps/early_concat/${TAG}" \
    "$@"

echo "==================== DONE ===================="
echo "Results:  outputs/sweeps/early_concat/${TAG}/"
