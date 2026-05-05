#!/bin/bash
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=n1s8-t4-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=dti-matrix
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --array=0-119%8

# ==============================================================================
# Experiment matrix: 4 variants x 2 datasets x 3 splits x 5 seeds = 120 runs
# Array is capped at 8 concurrent jobs (%8) to respect cluster limits.
# Run as:   sbatch hpc/train_array.sbatch
# ==============================================================================

mkdir -p logs outputs/checkpoints outputs/results wandb
module purge

NETID="${USER}"
OVERLAY="/scratch/$NETID/dti-overlay.ext3"
SIF="/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

# ---- Build the experiment grid ----
VARIANTS=(early_concat early_crossattn late_concat late_crossattn)
DATASETS=(davis kiba)
SPLITS=(random cold_drug cold_target)
SEEDS=(42 123 456 789 2024)

IDX=$SLURM_ARRAY_TASK_ID
N_SEEDS=${#SEEDS[@]}
N_SPLITS=${#SPLITS[@]}
N_DATASETS=${#DATASETS[@]}

SEED_IDX=$(( IDX % N_SEEDS ))
SPLIT_IDX=$(( (IDX / N_SEEDS) % N_SPLITS ))
DATASET_IDX=$(( (IDX / (N_SEEDS * N_SPLITS)) % N_DATASETS ))
VARIANT_IDX=$(( IDX / (N_SEEDS * N_SPLITS * N_DATASETS) ))

VARIANT=${VARIANTS[$VARIANT_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}
SPLIT=${SPLITS[$SPLIT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

CONFIG="configs/variant_${VARIANT}.yaml"

echo "====================================="
echo "Array index: $IDX"
echo "Variant: $VARIANT | Dataset: $DATASET | Split: $SPLIT | Seed: $SEED"
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
