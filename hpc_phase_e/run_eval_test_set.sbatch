#!/bin/bash
# =============================================================================
# Test-set evaluation across Phase C / E1 / E1b checkpoints
# Loads each best_model.pt and runs the held-out test split.
# Outputs: outputs/test_eval/PHASE_<X>_TEST_RESULTS.csv
#
# Wall-clock estimate: ~1-1.5 GPU-hr for all 132 ckpts (60 E1 + 36 E1b + 36 C).
# Submit:  sbatch hpc_phase_e/run_eval_test_set.sbatch [c|e1|e1b|all]
# =============================================================================
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:30:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=test-eval
#SBATCH --output=logs/test_eval_%j.out
#SBATCH --error=logs/test_eval_%j.err

set -e
PHASE="${1:-all}"

module purge
cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/test_eval
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print('torch:', torch.__version__, 'gpu:', torch.cuda.get_device_name(0))"

python scripts/eval_test_set.py --phase "$PHASE"
echo "==> test eval done (phase=$PHASE)"
