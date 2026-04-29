#!/bin/bash
# ==============================================================================
# Phase A Sweep Submitter — Late Concat (V4)
# Submits ~20 sbatch jobs; each runs one config of late_concat.
# Run from /scratch/$USER/ComparisionPDI/hpc_late_concat/
# ==============================================================================
set -e

cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/sweeps/late_concat

SBATCH=hpc_late_concat/run_late_concat_sweep.sbatch

echo "==> Submitting Phase A sweep for late_concat …"
echo

# ──────────── Round 0 — Baseline ────────────
sbatch "$SBATCH" baseline

# ──────────── Round 1 — Learning Rate (3 runs) ────────────
sbatch "$SBATCH" lr5e5     --lr 5e-5
sbatch "$SBATCH" lr1e4     --lr 1e-4
sbatch "$SBATCH" lr3e4     --lr 3e-4

# ──────────── Round 2 — d_model capacity (3 runs) ────────────
sbatch "$SBATCH" dm64      --d_model 64  --n_heads 2
sbatch "$SBATCH" dm128     --d_model 128 --n_heads 4
sbatch "$SBATCH" dm256     --d_model 256 --n_heads 8

# ──────────── Round 3 — Depth (3 runs) ────────────
sbatch "$SBATCH" l4        --n_layers 4
sbatch "$SBATCH" l6        --n_layers 6
sbatch "$SBATCH" l8        --n_layers 8

# ──────────── Round 4 — Batch size (3 runs) ────────────
sbatch "$SBATCH" bs32      --batch_size 32
sbatch "$SBATCH" bs64      --batch_size 64
sbatch "$SBATCH" bs128     --batch_size 128

# ──────────── Round 5 — Dropout (3 runs) ────────────
sbatch "$SBATCH" do01      --dropout 0.1
sbatch "$SBATCH" do02      --dropout 0.2
sbatch "$SBATCH" do03      --dropout 0.3

# ──────────── Round 6 — Heads (2 runs) ────────────
sbatch "$SBATCH" h2        --n_heads 2
sbatch "$SBATCH" h8        --n_heads 8

# ──────────── Round 7 — Seed sanity on baseline (3 runs) ────────────
# Note: results will go into outputs/sweeps/seed{42|123|456}/
sbatch "$SBATCH" seed42    --seed 42
sbatch "$SBATCH" seed123   --seed 123
sbatch "$SBATCH" seed456   --seed 456

# ──────────── Round 8 — Split robustness (2 runs) ────────────
sbatch "$SBATCH" cold_drug    --split cold_drug
sbatch "$SBATCH" cold_target  --split cold_target

echo
echo "==> Submitted Phase A sweep."
echo "==> Total: ~22 jobs. Monitor with:  squeue -u \$USER"
