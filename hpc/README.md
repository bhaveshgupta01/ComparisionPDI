# HPC Quickstart

All files in this folder for running on NYU HPC.

## First Time (one-time setup)
```bash
cd /scratch/$USER/dti-project/hpc
bash setup_env.sh              # Creates overlay + installs all deps (~20 min)
sbatch verify_env.sbatch       # Confirms everything works (~2 min)
cat logs/verify_*.out          # Should print ALL CHECKS PASSED
```

## Debug Interactively
```bash
bash interactive.sh 2          # Request 2-hour GPU session
# ... waits for allocation ...
bash enter_env.sh              # Drop into the singularity env
python scripts/train.py ...    # Run your code
```

## Submit One Training Run
```bash
sbatch train_single.sbatch configs/variant_late_crossattn.yaml davis random 42
squeue -u $USER
```

## Submit Full Experiment Matrix (120 runs)
```bash
sbatch train_array.sbatch
squeue -u $USER                # monitor
```

## Submit Analysis Job
```bash
sbatch analyze.sbatch          # after all training is done
```

## Files

| File | Purpose |
|------|---------|
| `HPC_SETUP_GUIDE.md` | Full walkthrough with explanations |
| `setup_env.sh` | One-time: build singularity overlay + install deps |
| `verify_env.sbatch` | Sanity-check env on a GPU node |
| `enter_env.sh` | Drop into the env interactively (after srun) |
| `interactive.sh` | Request an interactive GPU session |
| `train_single.sbatch` | Run one experiment |
| `train_array.sbatch` | Run the full 120-experiment matrix |
| `analyze.sbatch` | Run the deep analysis pipeline |
