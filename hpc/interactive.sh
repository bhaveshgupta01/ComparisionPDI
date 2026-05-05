#!/bin/bash
# ==============================================================================
# Request an interactive GPU session for debugging.
# Usage: bash hpc/interactive.sh [hours]
# ==============================================================================

HOURS="${1:-2}"

srun --account=csci_ga_2565-2026sp \
     --partition=n1s8-t4-1 \
     --gres=gpu:1 \
     --cpus-per-task=4 \
     --mem=16G \
     --time=${HOURS}:00:00 \
     --pty /bin/bash -l
