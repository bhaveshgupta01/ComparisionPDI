#!/bin/bash
# ==============================================================================
# Late Cross-Attention — Multi-GPU (2x L4) for High-Memory Configs
# Use this for dm256, bs128, and h8 which OOM on a single L4.
#
# Submit one job:
#   sbatch run_late_crossattn_multi_gpu.sbatch <tag> <per_gpu_batch> <extra_args...>
# ==============================================================================
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=g2-standard-24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=lca-mgpu
#SBATCH --output=logs/lca_mgpu_%x_%j.out
#SBATCH --error=logs/lca_mgpu_%x_%j.err

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
echo "Per-GPU BS:   $BSIZE"
echo "Extra Args:   $@"
echo "============================================="

# train.py automatically detects 2 GPUs and scales BSIZE to effective batch
python scripts/train.py \
    --variant late_crossattn \
    --output_dir "outputs/sweeps/late_crossattn/${TAG}" \
    --batch_size "$BSIZE" \
    "$@"

echo "==================== DONE ===================="
echo "Results:  outputs/sweeps/late_crossattn/${TAG}/"
