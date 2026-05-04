#!/bin/bash
# =============================================================================
# Identify and resubmit missing (variant, split, seed) combos for a given phase.
# Usage:  bash hpc_phase_e/resubmit_missing.sh <phase>
#   phase = e1b | e6
#
# Reads outputs/phase_<phase>/ to figure out what's missing, then sbatches
# the corresponding template for each.
# =============================================================================
set -e
cd /scratch/$USER/ComparisionPDI

PHASE="${1:?usage: bash resubmit_missing.sh <e1b|e6>}"

case "$PHASE" in
    e1b)
        SBATCH="hpc_phase_e/run_phase_e1b.sbatch"
        OUT_DIR="outputs/phase_e1b"
        TAG_PREFIX="phase_e1b"
        VARIANTS=(early_concat early_crossattn late_crossattn late_concat)
        SPLITS=(random cold_drug cold_target)
        SEEDS=(42 123 456)
        ;;
    e6)
        SBATCH="hpc_phase_e/run_phase_e6_deep.sbatch"
        OUT_DIR="outputs/phase_e6"
        TAG_PREFIX="phase_e6"
        VARIANTS=(early_concat early_crossattn late_crossattn late_concat)
        SPLITS=(random)
        SEEDS=(42 123 456)
        ;;
    *)
        echo "Unknown phase: $PHASE (use e1b or e6)"; exit 1 ;;
esac

count=0
for v in "${VARIANTS[@]}"; do
    for s in "${SPLITS[@]}"; do
        for sd in "${SEEDS[@]}"; do
            tag="${TAG_PREFIX}_${v}_${s}_seed${sd}"
            if [ ! -f "${OUT_DIR}/${tag}/results/results.csv" ]; then
                echo "==> resubmitting: $v $s $sd"
                if [ "$PHASE" = "e1b" ]; then
                    sbatch "$SBATCH" "$v" "$s" "$sd"
                else
                    # E6 sbatch only takes <variant> <seed> (random split is hard-coded)
                    sbatch "$SBATCH" "$v" "$sd"
                fi
                count=$((count+1))
            fi
        done
    done
done

echo
echo "==> Resubmitted $count missing $PHASE jobs."
echo "==> Monitor:  squeue -u \$USER"
