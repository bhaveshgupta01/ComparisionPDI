#!/bin/bash
# ==============================================================================
# One-time environment setup for DTI Project on NYU HPC
# Creates a 25 GB ext3 overlay + installs all dependencies via miniconda
# Run from /scratch/$USER/dti-project/hpc/
# ==============================================================================
set -e

NETID="${USER}"
OVERLAY_PATH="/scratch/$NETID/dti-overlay.ext3"
SIF_IMAGE="/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY_TEMPLATE="/scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz"

echo "==========================================="
echo "DTI Project — HPC Environment Setup"
echo "NetID: $NETID"
echo "==========================================="

# ---- 1. Check Singularity image exists ----
if [ ! -f "$SIF_IMAGE" ]; then
    echo "ERROR: Singularity image not found: $SIF_IMAGE"
    echo "Run 'ls /scratch/work/public/singularity/ | grep cuda' and update SIF_IMAGE in this script."
    exit 1
fi

# ---- 2. Create overlay if it doesn't exist ----
if [ -f "$OVERLAY_PATH" ]; then
    echo "Overlay already exists at $OVERLAY_PATH. Skipping creation."
    echo "   (Delete it first if you want a fresh one: rm $OVERLAY_PATH)"
else
    echo "Creating overlay at $OVERLAY_PATH (25 GB)..."
    cp "$OVERLAY_TEMPLATE" "${OVERLAY_PATH}.gz"
    gunzip "${OVERLAY_PATH}.gz"
    echo "Overlay created."
fi

# ---- 3. Launch Singularity to install miniconda + packages ----
echo ""
echo "Installing miniconda + dependencies inside the overlay..."
echo "This will take ~15-20 minutes."

singularity exec --overlay "$OVERLAY_PATH:rw" "$SIF_IMAGE" /bin/bash << 'EOF'
set -e

# Install miniconda inside the overlay
if [ ! -d /ext3/miniconda3 ]; then
    echo "Downloading miniconda..."
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
fi

# Create the environment activation script
cat > /ext3/env.sh << 'INNER'
#!/bin/bash
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PYTHONPATH=/scratch/$USER/dti-project:$PYTHONPATH
export WANDB_MODE=${WANDB_MODE:-offline}
export HF_HOME=/ext3/hf_cache
export TRANSFORMERS_CACHE=/ext3/hf_cache
export TORCH_HOME=/ext3/torch_cache
conda activate dti
INNER
chmod +x /ext3/env.sh

# Activate miniconda
source /ext3/miniconda3/etc/profile.d/conda.sh

# Create dti environment
if ! conda env list | grep -q "^dti "; then
    echo "Creating conda env 'dti'..."
    conda create -n dti python=3.10 -y
fi
conda activate dti

# Install core PyTorch stack (CUDA 12.1)
echo "Installing PyTorch..."
pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Scientific + data
echo "Installing scientific stack..."
pip install --no-cache-dir \
    numpy==1.24.4 \
    pandas==2.0.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    matplotlib==3.7.4 \
    seaborn==0.12.2 \
    umap-learn==0.5.5 \
    lifelines==0.27.8

# Cheminformatics + bio
echo "Installing RDKit + BioPython..."
pip install --no-cache-dir \
    rdkit==2023.9.2 \
    biopython==1.81

# DL frameworks + interpretability
echo "Installing transformers, captum, torch-geometric..."
pip install --no-cache-dir \
    transformers==4.36.0 \
    captum==0.7.0 \
    fair-esm==2.0.0

pip install --no-cache-dir torch-geometric==2.4.0

# Experiment tracking + utilities
echo "Installing utilities..."
pip install --no-cache-dir \
    wandb==0.16.0 \
    pyyaml==6.0.1 \
    omegaconf==2.3.0 \
    hydra-core==1.3.2 \
    rich==13.7.0 \
    tqdm==4.66.1 \
    pytest==7.4.3 \
    ipykernel==6.27.1 \
    jupyter==1.0.0

# Register kernel for OOD Jupyter
python -m ipykernel install --user --name dti --display-name "Python (DTI)"

echo ""
echo "=========================================="
echo "Environment 'dti' built successfully"
echo "Overlay: /scratch/<NETID>/dti-overlay.ext3"
echo "=========================================="
EOF

echo ""
echo "==========================================="
echo "Setup complete!"
echo "Next steps:"
echo "  1. Submit verify_env.sbatch to confirm everything works:"
echo "     sbatch verify_env.sbatch"
echo "  2. Then use train_single.sbatch or train_array.sbatch to run experiments"
echo "==========================================="
