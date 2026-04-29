#!/bin/bash
# ==============================================================================
# Early Concat — Multi-GPU (2 GPUs) for High-Memory Configs
# Use this for dm256, l8, bs128, and h8 which OOM even on a single A100.
#
# Submit one job:
#   sbatch run_early_concat_multi_gpu.sbatch <tag> <per_gpu_batch> <extra_args...>
# ==============================================================================
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=g2-standard-24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=ec-mgpu
#SBATCH --output=logs/ec_mgpu_%x_%j.out
#SBATCH --error=logs/ec_mgpu_%x_%j.err

TAG="${1:-untagged}"
BSIZE="${2:-32}"
shift 2  # rest of args are passed to train.py

module purge

cd /scratch/$USER/ComparisionPDI
mkdir -p logs

source .venv/bin/activate

echo "==================== ENV ===================="
which python
python --version
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count())"
echo "==================== RUN ===================="
echo "Tag:          $TAG"
echo "Variant:      early_concat"
echo "Per-GPU BS:   $BSIZE"
echo "Extra Args:   $@"
echo "============================================="

# train.py automatically detects 2 GPUs and scales BSIZE to effective batch
python scripts/train.py \
    --variant early_concat \
    --output_dir "outputs/sweeps/early_concat/${TAG}" \
    --batch_size "$BSIZE" \
    "$@"

echo "==================== DONE ===================="
echo "Results:  outputs/sweeps/early_concat/${TAG}/"
