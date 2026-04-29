#!/bin/bash
# Submits the four early_concat configurations that failed with OOM
# Run from repository root: bash hpc_early_concat/submit_oom_fix.sh
# Uses 2x A100 GPUs for maximum VRAM.

SBATCH=hpc_early_concat/run_early_concat_multi_gpu.sbatch

echo "==> Submitting OOM fixes for early_concat (using 2x A100 GPUs)..."

# Configuration:
# We use a physical batch size of 32 per GPU.
# With 2 GPUs, the effective batch size is 64 (Matches Baseline).

# 1. dm256
sbatch "$SBATCH" dm256 32 --d_model 256 --n_heads 8

# 2. l8
sbatch "$SBATCH" l8 32 --n_layers 8

# 3. bs128
# For BS128, we use 64 per GPU to get 128 total.
sbatch "$SBATCH" bs128 64

# 4. h8
sbatch "$SBATCH" h8 32 --n_heads 8

echo "==> Jobs submitted. Monitor with squeue -u \$USER"
