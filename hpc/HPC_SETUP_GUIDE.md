# NYU HPC Setup Guide — DTI Project

End-to-end instructions for running this project on NYU HPC (Cloud Bursting, CSCI-GA-2565 spring 2026 allocation).

**Your allocation:** 300 GPU-hours. Budget accordingly.

---

## Accounts and Partitions

- **Slurm account:** `csci_ga_2565-2026sp`
- **Useful partitions for this project:**
  - `interactive` — for debugging, never for real training
  - `n1s8-t4-1` — T4 GPU, cheap, good for Davis experiments
  - `g2-standard-12` — 1 × L4 GPU (newer than T4, good default)
  - `c12m85-a100-1` — 1 × A100 40GB — use for KIBA/BindingDB
  - `c24m170-a100-2` — 2 × A100, only if you need multi-GPU

**Rule of thumb for this project:**
- Davis experiments → `n1s8-t4-1` or `g2-standard-12`
- KIBA experiments → `g2-standard-12` or `c12m85-a100-1`
- BindingDB / heavy analysis → `c12m85-a100-1`
- Never request multi-GPU unless you actually parallelize

---

## Phase 0 — First Login

### 0.1 Connect via VPN
Install Cisco Secure Client if off-campus. Then visit:
https://ood-burst-001.hpc.nyu.edu/

### 0.2 Open a Shell on the Cluster
In OOD: **Clusters -> _Burst Shell Access_** (or start an interactive Jupyter session then use its terminal).

### 0.3 Verify Your Scratch Directory
```bash
echo $USER                # your NetID
cd /scratch/$USER
pwd                       # should print /scratch/<your_netid>
df -h /scratch/$USER      # check quota
```

All real work lives in `/scratch/$USER/`. Home directory has tiny quota — don't install anything there.

### 0.4 Clone the Project Repo
```bash
cd /scratch/$USER
git clone <your-repo-url> dti-project
cd dti-project
```

If you haven't pushed to GitHub yet, copy the folder up via OOD's file browser or `scp` from your Mac:
```bash
# From your Mac (replace NETID):
scp -r ~/CodeFiles/DTI_MLFinalProject \
    NETID@gw.hpc.nyu.edu:/scratch/NETID/dti-project
```

---

## Phase 1 — One-Time Environment Setup

NYU HPC uses **Singularity containers + ext3 overlay files** for conda environments. The overlay is essentially a writable virtual disk mounted inside a read-only container. This isolates your environment and makes it reproducible.

We'll script the whole setup.

### 1.1 Pick a Singularity Image

List available images:
```bash
ls /scratch/work/public/singularity/ | grep cuda
```

For PyTorch 2.1 + CUDA 12.1, use:
```
/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
```

(If this exact name isn't available when you check, pick the closest CUDA 12.x image.)

### 1.2 Create the Overlay File

Run the setup script (provided in this folder):
```bash
cd /scratch/$USER/dti-project/hpc
bash setup_env.sh
```

This creates `/scratch/$USER/dti-overlay.ext3` (a 25 GB writable disk) and installs all dependencies.

**Why 25 GB?** PyTorch + CUDA + RDKit + ESM-2 + ChemBERTa weights + HuggingFace cache = ~15 GB realistic. 25 GB leaves headroom.

### 1.3 Verify Environment

After setup completes:
```bash
cd /scratch/$USER/dti-project/hpc
sbatch verify_env.sbatch
# Wait 1-2 min for queue
squeue -u $USER
# When it completes:
cat verify_env.out
```

Expected: prints "PyTorch X.Y.Z, CUDA available: True, GPU: NVIDIA ..." without errors.

---

## Phase 2 — Running Experiments

### 2.1 Interactive Session (for debugging)

Quickest way to test code is an interactive GPU session:
```bash
srun --account=csci_ga_2565-2026sp \
     --partition=n1s8-t4-1 \
     --gres=gpu:1 \
     --time=1:00:00 \
     --mem=16G \
     --pty /bin/bash

# Once on the node:
cd /scratch/$USER/dti-project
bash hpc/enter_env.sh   # enters the singularity container with env active
# Now you can run python scripts/train.py ...
```

### 2.2 Batch Job for Single Training Run

Submit a training job with `train_single.sbatch`:
```bash
sbatch hpc/train_single.sbatch \
    configs/variant_late_crossattn.yaml davis random 42
```

This queues one training run. It will run regardless of your laptop state. Check status:
```bash
squeue -u $USER                    # your queue
sacct -j <job_id>                  # details once started
tail -f logs/slurm-<job_id>.out    # live logs
```

### 2.3 Array Jobs for the Full Experiment Matrix

For 120 core runs, submitting one job at a time is painful. Use a SLURM array:
```bash
sbatch hpc/train_array.sbatch
```

This submits 120 jobs as a single array. They run in parallel up to your concurrent-job limit (typically 8-16 at a time).

### 2.4 Monitor Many Jobs

```bash
squeue -u $USER                    # all your jobs
squeue -u $USER -t RUNNING         # only running
sacct -u $USER --starttime=today   # today's history
scancel <job_id>                   # cancel one
scancel -u $USER                   # nuclear option: cancel everything
```

### 2.5 Check GPU Usage Mid-Run

SSH into the compute node running your job:
```bash
squeue -u $USER                    # find the nodelist
ssh <nodelist>                     # e.g. gpu-001
nvidia-smi                         # see GPU utilization
exit
```

Ideal: GPU should be at >80% utilization. Below 50% = you're bottlenecked on data loading.

---

## Phase 3 — Logging and Outputs

### 3.1 Directory Layout
```
/scratch/<NETID>/dti-project/
├── hpc/                   # Slurm scripts (this folder)
├── src/                   # Code
├── configs/               # YAML configs
├── data/                  # Datasets (processed)
├── outputs/
│   ├── checkpoints/       # Model weights
│   ├── results/           # CSVs
│   └── figures/           # Plots
├── logs/                  # Slurm stdout/stderr
└── wandb/                 # W&B offline logs
```

### 3.2 Weights & Biases on HPC
HPC compute nodes may have no outbound network. Use W&B **offline mode**:
```bash
export WANDB_MODE=offline     # set in sbatch scripts
```
Then sync later from the login node:
```bash
wandb sync wandb/offline-run-*
```

### 3.3 Moving Results Back to Your Mac
From your Mac:
```bash
# Pull all results + figures (small)
scp -r NETID@gw.hpc.nyu.edu:/scratch/NETID/dti-project/outputs/results ./
scp -r NETID@gw.hpc.nyu.edu:/scratch/NETID/dti-project/outputs/figures ./

# For large checkpoints, use the dedicated transfer node:
scp -r NETID@dtn.torch.hpc.nyu.edu:/scratch/NETID/dti-project/outputs/checkpoints ./
```

Or use Globus if the volume is large.

---

## Phase 4 — Budget Management

### 4.1 Check Remaining GPU Hours
```bash
myquota               # disk quota
sshare -A csci_ga_2565-2026sp -u $USER   # Slurm account usage
```

### 4.2 Rough Budget for This Project
- Full 120-run core matrix on T4: ~100 GPU-hours
- Ablations on T4: ~80 GPU-hours
- Extensions on L4/A100: ~120 GPU-hours
- **Total: ~300 GPU-hours** — exactly your allocation; be disciplined

### 4.3 Cost-Saving Tips
- Debug on `interactive` partition (non-GPU when possible)
- Davis runs take ~30 min on T4 — don't use A100 for them
- Kill runs early if loss isn't decreasing (early stopping helps)
- Don't run duplicate experiments by accident — always check `squeue` first

---

## Phase 5 — Common Pitfalls

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce batch size; add `torch.cuda.empty_cache()`; request bigger GPU |
| Job sits in queue forever | Check `sinfo -p <partition>` for node availability; pick less busy partition |
| Container can't find package | Did you `source /ext3/env.sh` inside the singularity exec? |
| W&B hangs | Set `WANDB_MODE=offline`; sync from login node later |
| "Disk quota exceeded" in container | Your overlay is full; recreate bigger |
| Training fails at epoch N | Log checkpoints every epoch so you can resume |
| Job killed with `TIME LIMIT` | Request more time in sbatch; or checkpoint + resume |
| Slow data loading | Preprocess once, save `.pt` files, load directly |

---

## Phase 6 — When You're Done

- Copy final checkpoints and results to long-term storage (your Mac or Google Drive)
- Delete massive intermediate files from scratch (it's not backed up, but quota matters)
- Keep your environment overlay in case you need to re-run anything

```bash
# Space audit
du -sh /scratch/$USER/*
```

---

## Quick Reference — Typical Workflow

```bash
# Start of work session
ssh NETID@gw.hpc.nyu.edu
cd /scratch/$USER/dti-project
git pull

# Launch experiments
sbatch hpc/train_array.sbatch

# Check progress
squeue -u $USER

# Go to class / sleep / close laptop — jobs keep running

# Come back later
squeue -u $USER                     # done yet?
cat logs/slurm-*.out | grep "Val"   # spot-check results
```

That's the full HPC workflow for this project.
