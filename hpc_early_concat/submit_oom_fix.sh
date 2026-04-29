#!/bin/bash
# Submits the four early_concat configurations that failed with OOM
# Run from repository root: bash hpc_early_concat/submit_oom_fix.sh

SBATCH=hpc_early_concat/run_early_concat_multi_gpu.sbatch

echo "==> Submitting OOM fixes for early_concat (using 2 GPUs)..."

# 1. dm256: 2 GPUs * 32 BS = 64 Effective BS (Matches Baseline)
sbatch "$SBATCH" dm256 32 --d_model 256 --n_heads 8

# 2. l8: 2 GPUs * 32 BS = 64 Effective BS (Matches Baseline)
sbatch "$SBATCH" l8 32 --n_layers 8

# 3. bs128: 2 GPUs * 64 BS = 128 Effective BS (Matches intended BS128)
sbatch "$SBATCH" bs128 64

# 4. h8: 2 GPUs * 32 BS = 64 Effective BS (Matches Baseline)
sbatch "$SBATCH" h8 32 --n_heads 8

echo "==> Jobs submitted. Monitor with squeue -u \$USER"
