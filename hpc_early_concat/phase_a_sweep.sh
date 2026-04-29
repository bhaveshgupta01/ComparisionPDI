#!/bin/bash
# ==============================================================================
# Phase A Sweep Submitter — Early Concat
# Submits ~23 sbatch jobs for the early_concat variant.
# Run from repository root: bash hpc_early_concat/phase_a_sweep.sh
# ==============================================================================
set -e

# Ensure output directory exists
mkdir -p logs outputs/sweeps/early_concat

SBATCH=hpc_early_concat/run_early_concat_a100.sbatch

echo "==> Submitting Phase A sweep for early_concat..."
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
sbatch "$SBATCH" seed42    --seed 42
sbatch "$SBATCH" seed123   --seed 123
sbatch "$SBATCH" seed456   --seed 456

# ──────────── Round 8 — Split robustness (2 runs) ────────────
sbatch "$SBATCH" cold_drug    --split cold_drug
sbatch "$SBATCH" cold_target  --split cold_target

echo
echo "==> Submitted Phase A sweep for early_concat."
echo "==> Total: 23 jobs. Monitor with: squeue -u \$USER"
