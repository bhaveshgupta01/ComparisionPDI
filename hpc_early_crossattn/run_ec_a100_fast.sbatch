#!/bin/bash
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=ec-fast
#SBATCH --output=logs/ec_fast_%j.out
#SBATCH --error=logs/ec_fast_%j.err

TAG="${1:-untagged}"
shift

module purge
cd /scratch/$USER/ComparisionPDI
mkdir -p logs
source .venv/bin/activate
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
echo "Tag: $TAG | Args: $@"

python scripts/train.py \
    --variant early_crossattn \
    --output_dir "outputs/sweeps/v2_${TAG}_fast" \
    --epochs 15 \
    --max_rows 10000 \
    "$@"

echo "DONE"
