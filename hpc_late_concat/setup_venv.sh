#!/bin/bash
# ==============================================================================
# One-time venv setup for ComparisionPDI on NYU HPC
# Run from /scratch/$USER/ComparisionPDI/
# ==============================================================================
set -e

cd /scratch/$USER/ComparisionPDI

echo "==> Loading python module"
module purge
module load python/intel/3.10.10 2>/dev/null || module load python/3.10 2>/dev/null || true

echo "==> Building .venv"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip wheel

echo "==> Installing PyTorch (CUDA 12.1)"
pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing project requirements"
pip install --no-cache-dir -r requirements.txt

echo "==> Installing extras (sklearn, captum, wandb, rdkit) for analysis"
pip install --no-cache-dir scikit-learn==1.3.2 captum==0.7.0 wandb==0.16.0 rdkit==2023.9.2 lifelines==0.27.8

echo
echo "==> Quick sanity check"
python -c "import torch; import pandas; import numpy; import scipy; print('OK — torch:', torch.__version__)"

echo
echo "==> Done. Activate with:  source .venv/bin/activate"
