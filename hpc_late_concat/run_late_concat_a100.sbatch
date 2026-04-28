#!/bin/bash
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:45:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=lc-a100
#SBATCH --output=logs/lc_a100_%j.out
#SBATCH --error=logs/lc_a100_%j.err

TAG="${1:-untagged}"
shift

module purge
cd /scratch/$USER/ComparisionPDI
mkdir -p logs
source .venv/bin/activate

echo "==================== ENV ===================="
which python && python --version
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '| gpu:', torch.cuda.get_device_name(0))"
echo "==================== RUN ===================="
echo "Tag:  $TAG"
echo "Args: $@"
echo "============================================="

python scripts/train.py \
    --variant late_concat \
    --output_dir "outputs/sweeps/${TAG}" \
    "$@"

echo "==================== DONE ===================="
