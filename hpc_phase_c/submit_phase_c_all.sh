#!/bin/bash
# =============================================================================
# Phase C master submitter — kicks off all 36 jobs (4 variants x 3 splits x 3 seeds).
# Run from /scratch/$USER/ComparisionPDI/.
# Total wall-clock: ~25-35 min per job; jobs run in parallel as A100s free up.
# Total GPU-hours: ~15-20.
# =============================================================================
set -e

cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/phase_c

SBATCH="hpc_phase_c/run_phase_c.sbatch"
VARIANTS=(early_concat early_crossattn late_crossattn late_concat)
SPLITS=(random cold_drug cold_target)
SEEDS=(42 123 456)

count=0
echo "==> Submitting Phase C..."
for v in "${VARIANTS[@]}"; do
    for s in "${SPLITS[@]}"; do
        for sd in "${SEEDS[@]}"; do
            sbatch "$SBATCH" "$v" "$s" "$sd"
            count=$((count+1))
        done
    done
done

echo
echo "==> Submitted $count Phase C jobs."
echo "==> Monitor:  squeue -u \$USER"
echo "==> Tail:     tail -f \$(ls -t logs/phase_c_*.out | head -1)"
