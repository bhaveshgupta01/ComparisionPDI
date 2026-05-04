#!/bin/bash
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=n1s8-t4-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:15:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=dti-verify
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err

# ==============================================================================
# Sanity check: verify the singularity + conda environment works end-to-end
# ==============================================================================

mkdir -p logs
module purge

NETID="${USER}"
OVERLAY="/scratch/$NETID/dti-overlay.ext3"
SIF="/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

singularity exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
source /ext3/env.sh

echo '===== System ====='
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ''

echo '===== Python ====='
python --version
echo ''

echo '===== Core imports ====='
python << 'PYEOF'
import torch
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib
import seaborn
import rdkit
from rdkit import Chem
import Bio
import transformers
import captum
import torch_geometric
import wandb
import lifelines
import umap

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'SciPy: {scipy.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'Seaborn: {seaborn.__version__}')
print(f'RDKit: {rdkit.__version__}')
print(f'BioPython: {Bio.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'captum: {captum.__version__}')
print(f'torch_geometric: {torch_geometric.__version__}')
print(f'wandb: {wandb.__version__}')
print(f'lifelines: {lifelines.__version__}')

# Quick GPU sanity test
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')
z = x @ y
print(f'\nGPU matmul test: output shape {z.shape}, device {z.device} — PASSED')

# RDKit sanity
mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')
print(f'RDKit parse aspirin: {mol.GetNumAtoms()} atoms — PASSED')

print('\n===== ALL CHECKS PASSED =====')
PYEOF
"
