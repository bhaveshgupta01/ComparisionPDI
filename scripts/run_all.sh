#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh — Train all four DTI variants in sequence
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   bash scripts/run_all.sh [--max_rows N] [--epochs N] [--split SPLIT]
#
# Defaults: 20000 rows, 10 epochs, random split (fast comparison run)
# For full training: bash scripts/run_all.sh --max_rows 0 --epochs 30
# ─────────────────────────────────────────────────────────────────────────────

set -e

MAX_ROWS=20000
EPOCHS=10
SPLIT=random
SEED=42

# Parse optional args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --max_rows) MAX_ROWS="$2"; shift ;;
        --epochs)   EPOCHS="$2"; shift ;;
        --split)    SPLIT="$2"; shift ;;
        --seed)     SEED="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

PYTHON=.venv/bin/python
VARIANTS=("early_concat" "early_crossattn" "late_concat" "late_crossattn")

echo "════════════════════════════════════════════════════════════"
echo "  DTI Comparison Run"
echo "  Variants : ${VARIANTS[*]}"
echo "  Max rows : ${MAX_ROWS} (0 = all)"
echo "  Epochs   : ${EPOCHS}"
echo "  Split    : ${SPLIT}"
echo "  Seed     : ${SEED}"
echo "════════════════════════════════════════════════════════════"

# Determine --max_rows flag (0 means no limit → omit flag)
MAX_ROWS_FLAG=""
if [ "$MAX_ROWS" -gt 0 ] 2>/dev/null; then
    MAX_ROWS_FLAG="--max_rows $MAX_ROWS"
fi

START=$(date +%s)

for VARIANT in "${VARIANTS[@]}"; do
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Training: $VARIANT"
    echo "────────────────────────────────────────────────────────────"
    $PYTHON scripts/train.py \
        --variant "$VARIANT" \
        --split "$SPLIT" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        $MAX_ROWS_FLAG
done

END=$(date +%s)
ELAPSED=$(( END - START ))

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All variants trained in ${ELAPSED}s."
echo "  Results → outputs/results/results.csv"
echo "════════════════════════════════════════════════════════════"

# Pretty-print summary table
echo ""
echo "Summary:"
if command -v column &> /dev/null; then
    column -t -s ',' outputs/results/results.csv 2>/dev/null || cat outputs/results/results.csv
else
    cat outputs/results/results.csv 2>/dev/null
fi
