#!/bin/bash
# =============================================================================
# Phase E1 master submitter — kicks off all 60 jobs (4 variants x 3 splits x 5 seeds).
# Run from /scratch/$USER/ComparisionPDI/.
# Per-job wall-clock: ~70-90 min. Total GPU-hours: ~75-90.
# Wall-clock with concurrent A100s: ~6-8 hours.
# =============================================================================
set -e

cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/phase_e_xl

SBATCH="hpc_phase_e/run_phase_e_xl.sbatch"
VARIANTS=(early_concat early_crossattn late_crossattn late_concat)
SPLITS=(random cold_drug cold_target)
SEEDS=(42 123 456 789 2024)

count=0
echo "==> Submitting Phase E1 (XL)..."
for v in "${VARIANTS[@]}"; do
    for s in "${SPLITS[@]}"; do
        for sd in "${SEEDS[@]}"; do
            sbatch "$SBATCH" "$v" "$s" "$sd"
            count=$((count+1))
        done
    done
done

echo
echo "==> Submitted $count Phase E1 jobs."
echo "==> Monitor:  squeue -u \$USER"
echo "==> Tail:     tail -f \$(ls -t logs/phase_e_xl_*.out | head -1)"
echo "==> Quota:    sacct -u \$USER -X --starttime=$(date +%Y-%m-%d) --format=Elapsed -P --noheader | awk -F'|' '{split(\$1,t,\":\");s+=t[1]*3600+t[2]*60+t[3]} END{printf \"Today: %.2f GPU-hrs\\n\", s/3600}'"
