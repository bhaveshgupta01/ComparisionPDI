#!/bin/bash
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=n1s8-t4-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=dti-analyze
#SBATCH --output=logs/analyze_%j.out
#SBATCH --error=logs/analyze_%j.err

# ==============================================================================
# Deep analysis job: runs all 6 parts of the analysis playbook
# (attention, representation, causal, biological, failures, dynamics)
# on all trained checkpoints. Produces figures + tables.
# ==============================================================================

mkdir -p logs outputs/figures outputs/analysis
module purge

NETID="${USER}"
OVERLAY="/scratch/$NETID/dti-overlay.ext3"
SIF="/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

singularity exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
source /ext3/env.sh
cd /scratch/$NETID/dti-project
python scripts/run_analysis.py \
    --models-dir outputs/checkpoints/ \
    --output-dir outputs/analysis/ \
    --config configs/analysis/full_analysis.yaml
"
