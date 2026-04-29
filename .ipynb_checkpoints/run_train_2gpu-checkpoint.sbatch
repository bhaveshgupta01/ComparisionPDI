#!/bin/bash

#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=early_concat
#SBATCH --output=logs/early_concat_%j.out
#SBATCH --error=logs/early_concat_%j.err

module purge

cd /scratch/$USER/ComparisionPDI
mkdir -p logs

source .venv/bin/activate

which python
python --version
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu count:', torch.cuda.device_count())"

python scripts/train.py --variant early_concat