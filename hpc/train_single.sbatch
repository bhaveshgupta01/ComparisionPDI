#!/bin/bash
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=n1s8-t4-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=dti-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ==============================================================================
# Single training run.
# Usage:  sbatch hpc/train_single.sbatch <config> <dataset> <split> <seed>
# Ex:     sbatch hpc/train_single.sbatch configs/variant_late_crossattn.yaml davis random 42
# ==============================================================================

CONFIG="${1:-configs/variant_late_crossattn.yaml}"
DATASET="${2:-davis}"
SPLIT="${3:-random}"
SEED="${4:-42}"

mkdir -p logs outputs/checkpoints outputs/results wandb
module purge

NETID="${USER}"
OVERLAY="/scratch/$NETID/dti-overlay.ext3"
SIF="/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

echo "====================================="
echo "Job: $SLURM_JOB_ID"
echo "Config: $CONFIG"
echo "Dataset: $DATASET | Split: $SPLIT | Seed: $SEED"
echo "Node: $(hostname)"
echo "====================================="

singularity exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
source /ext3/env.sh
cd /scratch/$NETID/dti-project
python scripts/train.py \
    --config $CONFIG \
    --dataset $DATASET \
    --split $SPLIT \
    --seed $SEED \
    --output-dir outputs/
"

echo "====================================="
echo "Job complete: $SLURM_JOB_ID"
echo "====================================="
