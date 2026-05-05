#!/bin/bash
# =============================================================================
# Phase E6 master submitter — depth-axis ablation
# 4 variants x 1 split (random) x 3 seeds = 12 jobs
# Per-run wallclock: ~30-40 min. Total ~6-8 GPU-hours.
# =============================================================================
set -e

cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/phase_e6

SBATCH="hpc_phase_e/run_phase_e6_deep.sbatch"
VARIANTS=(early_concat early_crossattn late_crossattn late_concat)
SEEDS=(42 123 456)

count=0
echo "==> Submitting Phase E6 (depth-axis at d=128, n_layers x2)..."
for v in "${VARIANTS[@]}"; do
    for sd in "${SEEDS[@]}"; do
        sbatch "$SBATCH" "$v" "$sd"
        count=$((count+1))
    done
done

echo
echo "==> Submitted $count Phase E6 jobs."
echo "==> Monitor:  squeue -u \$USER | grep phase-e6"
