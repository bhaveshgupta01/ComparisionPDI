#!/bin/bash
# ==============================================================================
# Enter the project's Singularity + conda environment interactively
# Usage: bash hpc/enter_env.sh
# Use this when you've srun'd onto a node and want to debug / run python manually
# ==============================================================================

NETID="${USER}"
OVERLAY="/scratch/$NETID/dti-overlay.ext3"
SIF="/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

# Use --nv to mount CUDA; :rw if you need to pip-install more, :ro otherwise
singularity exec --nv --overlay "$OVERLAY:rw" "$SIF" /bin/bash --init-file /ext3/env.sh
