#!/bin/bash
# =============================================================================
# Phase E1b master submitter — 36 jobs (4 variants x 3 splits x 3 seeds at d=192).
# Per-job wall-clock: ~30-40 min on A100. Total GPU-hours: ~20-25.
# =============================================================================
set -e

cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/phase_e1b

SBATCH="hpc_phase_e/run_phase_e1b.sbatch"
VARIANTS=(early_concat early_crossattn late_crossattn late_concat)
SPLITS=(random cold_drug cold_target)
SEEDS=(42 123 456)

count=0
echo "==> Submitting Phase E1b (mid-scale d=192)..."
for v in "${VARIANTS[@]}"; do
    for s in "${SPLITS[@]}"; do
        for sd in "${SEEDS[@]}"; do
            sbatch "$SBATCH" "$v" "$s" "$sd"
            count=$((count+1))
        done
    done
done

echo
echo "==> Submitted $count Phase E1b jobs."
echo "==> Monitor:  squeue -u \$USER | grep phase-e1b"
