#!/bin/bash
# Submits the three configurations that failed with OOM
# Run from repo root: bash hpc_late_crossattn/submit_oom_fix.sh

SBATCH=hpc_late_crossattn/run_late_crossattn_multi_gpu.sbatch

echo "==> Submitting OOM fixes for late_crossattn..."

# 1. dm256: 2 GPUs * 32 BS = 64 Effective BS (Matches Baseline)
sbatch "$SBATCH" dm256 32 --d_model 256 --n_heads 8

# 2. h8: 2 GPUs * 32 BS = 64 Effective BS (Matches Baseline)
sbatch "$SBATCH" h8 32 --n_heads 8

# 3. bs128: 2 GPUs * 64 BS = 128 Effective BS (Matches intended BS128)
sbatch "$SBATCH" bs128 64

echo "==> Jobs submitted. Monitor with squeue -u \$USER"
