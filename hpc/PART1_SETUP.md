# Part 1 — HPC Setup (Bhavesh, V4 Late Concat)

**Goal:** Get NYU HPC ready end-to-end so that when the team finishes the shared scaffolding, you can immediately start running Late Concat experiments. By the end of Part 1 you will have:

1. Confirmed VPN + OOD access
2. Scratch directory set up
3. Singularity environment built with all dependencies
4. Verified everything works on a real GPU node
5. A compute budget plan for Phase A

**Estimated wall-clock:** 60–90 minutes (mostly waiting for the environment to install and the verify job to queue).

**GPU hours consumed in Part 1:** < 0.5 (just the verify job).

---

## Step 0 — Prerequisites

Before starting, confirm:
- [ ] NYU VPN (Cisco Secure Client) installed and connected OR you're on campus WiFi
- [ ] You can log in to https://ood-burst-001.hpc.nyu.edu/
- [ ] You see your NetID in the top-right when logged in
- [ ] You've been added to the course allocation `csci_ga_2565-2026sp`

**Quick check your allocation:** In any HPC terminal:
```bash
sshare -A csci_ga_2565-2026sp -u $USER
```
Should show 300 GPU-hours available.

---

## Step 1 — Open a Shell on HPC

You have three ways; pick whichever you like.

### Option A (Recommended): OOD Web Terminal
1. Go to https://ood-burst-001.hpc.nyu.edu/
2. Top menu → **Clusters** → **_Burst Shell Access_**
3. Terminal opens in a browser tab. You're on a login node.

### Option B: OOD JupyterLab
1. OOD → **Interactive Apps** → **Jupyter Notebook**
2. Pick `interactive` partition, 2 hours, 1 CPU, 4 GB RAM, **no GPU** (this is just for setup)
3. Launch. Wait 30 sec.
4. Click **Connect to Jupyter Notebook** → inside Jupyter, File → New → Terminal

### Option C: SSH from your Mac
```bash
ssh $USER@gw.hpc.nyu.edu
```
Where `$USER` = your NetID. You'll need VPN + Duo 2FA.

**For this setup, I recommend Option A or B** — no SSH config needed, just browser.

Once you have a shell, verify:
```bash
whoami                   # your NetID
hostname                 # a login node, e.g. log-burst-1
pwd                      # likely /home/<NetID>
```

---

## Step 2 — Move to Scratch

Home directory has a tiny quota. All real work lives in `/scratch/$USER/`.

```bash
cd /scratch/$USER
pwd                      # /scratch/<your_netid>
df -h .                  # check available space (should have TBs)
```

If `/scratch/$USER` doesn't exist, create it:
```bash
mkdir -p /scratch/$USER
cd /scratch/$USER
```

---

## Step 3 — Get the Project Code onto HPC

You have options. Pick one:

### Option A: Clone from GitHub (recommended once repo exists)
```bash
cd /scratch/$USER
git clone <team_repo_url> dti-project
cd dti-project
ls                       # should see hpc/, configs/, etc.
```

### Option B: Upload via OOD File Browser
1. OOD → **Files** → navigate to `/scratch/<NetID>/`
2. Click **Upload** → drag your local `DTI_MLFinalProject` folder
3. Rename to `dti-project` once uploaded

### Option C: scp from your Mac
From your Mac terminal:
```bash
cd ~/CodeFiles
scp -r DTI_MLFinalProject <NetID>@gw.hpc.nyu.edu:/scratch/<NetID>/dti-project
```

**After upload, verify on HPC:**
```bash
cd /scratch/$USER/dti-project
ls hpc/                  # should list setup_env.sh, verify_env.sbatch, etc.
```

---

## Step 4 — Find the Right Singularity Image

We need a Singularity image with CUDA 12.1 for PyTorch 2.1.

```bash
ls /scratch/work/public/singularity/ | grep cuda12
```

You'll see something like:
```
cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
```

If the exact name in `hpc/setup_env.sh` doesn't match what's available, update it:

```bash
cd /scratch/$USER/dti-project/hpc
# Check what setup_env.sh expects:
grep "^SIF_IMAGE" setup_env.sh
# If needed, edit:
nano setup_env.sh        # or: vim / code
```

Replace the `SIF_IMAGE` line with the exact filename from the `ls` output. Save.

---

## Step 5 — Build the Environment (ONE TIME, ~20 min)

This creates a 25 GB writable overlay disk inside `/scratch/$USER/` and installs miniconda + all Python packages.

```bash
cd /scratch/$USER/dti-project/hpc
bash setup_env.sh
```

**What's happening:**
- Copies a pre-made 25 GB empty overlay template
- Launches Singularity container
- Downloads miniconda inside the overlay
- Creates conda env `dti` with Python 3.10
- Pip-installs PyTorch 2.1 + CUDA 12.1, RDKit, transformers, captum, torch-geometric, wandb, etc.
- Registers a Jupyter kernel so OOD Jupyter can use this env

**Expected output:** Long install log. At the end:
```
==========================================
Environment 'dti' built successfully
Overlay: /scratch/<NETID>/dti-overlay.ext3
==========================================
```

**If it fails:**
- Network hiccup during pip install → just re-run `bash setup_env.sh`; it resumes from where miniconda exists
- Out of disk space → `df -h /scratch/$USER/` — you may have a quota issue
- Singularity not found → `module load singularity` then re-run

---

## Step 6 — Verify the Environment on a Real GPU

Submit the sanity check job:

```bash
cd /scratch/$USER/dti-project/hpc
mkdir -p logs           # if not already there
sbatch verify_env.sbatch
```

You'll get a job ID, e.g. `Submitted batch job 1234567`.

**Monitor the queue:**
```bash
squeue -u $USER
```

Expected states:
- `PD` (pending) — waiting for a T4 node to free up; typically <5 min
- `R` (running) — executing
- `CG` (completing)
- Gone from queue → done

**Once done, read the output:**
```bash
cat logs/verify_*.out
```

You should see something like:
```
===== System =====
gpu-xxx.hpc.nyu.edu
NVIDIA T4, 15109 MiB
===== Python =====
Python 3.10.x
===== Core imports =====
PyTorch: 2.1.0
CUDA available: True
CUDA version: 12.1
GPU: Tesla T4
GPU memory: 15.1 GB
...
RDKit: 2023.09.x
...
GPU matmul test: output shape torch.Size([1000, 1000]), device cuda:0 — PASSED
RDKit parse aspirin: 13 atoms — PASSED

===== ALL CHECKS PASSED =====
```

If you see "ALL CHECKS PASSED" → **you're done with Part 1**.

**If errors:**
- `ImportError: module X` → re-run `bash setup_env.sh`, may have missed a package
- `CUDA not available` → wrong Singularity image version, try a different CUDA one
- Job never starts → partition busy, check `sinfo -p n1s8-t4-1`

---

## Step 7 — Understand Your Compute Budget

You have **300 GPU-hours**. Here's how I recommend you spend them as V4 (Late Concat):

### Phase A — Individual Tuning (Weeks 1–2)
~20 runs on mini-BindingDB (30k pairs)

| Stage | Partition | Time/run | Runs | Budget |
|-------|-----------|----------|------|--------|
| Baseline | `n1s8-t4-1` | 45 min | 1 | 0.75h |
| Tier 1 sweep (LR, d_model, n_layers, BS) | `n1s8-t4-1` | 45 min | 8 | 6h |
| Tier 2 sweep (pos enc, activation, dropout) | `n1s8-t4-1` | 45 min | 6 | 4.5h |
| Tier 3+4 (variant-specific, pooling, misc) | `n1s8-t4-1` | 45 min | 6 | 4.5h |
| Seed sanity (3 seeds on best config) | `n1s8-t4-1` | 45 min | 3 | 2.25h |
| Debug buffer | `interactive` + `n1s8-t4-1` | — | — | 2h |
| **Phase A subtotal** | | | | **~20h** |

### Phase C — Final Runs (shared across team; split evenly each pays ~25%)
~180 runs total ÷ 4 teammates = ~45 runs each on full datasets

| Dataset | Partition | Time/run | Your share | Budget |
|---------|-----------|----------|------------|--------|
| Davis runs | `n1s8-t4-1` | 30 min | ~15 | 7.5h |
| KIBA runs | `g2-standard-12` (L4) | 90 min | ~15 | 22.5h |
| BindingDB full runs | `c12m85-a100-1` (A100) | 3h | ~15 | 45h |
| **Phase C subtotal (your share)** | | | | **~75h** |

### Phase D — Deep Analysis (shared, but some GPU time needed)
Attention extraction, CKA, head ablation, etc.

| Task | Partition | Time | Budget |
|------|-----------|------|--------|
| Attention + gradient extraction across all checkpoints | `n1s8-t4-1` | — | ~10h |
| Head ablation | `n1s8-t4-1` | — | ~10h |
| Representation analysis (CKA, t-SNE, probing) | `n1s8-t4-1` | — | ~5h |
| Integrated gradients | `g2-standard-12` | — | ~10h |
| **Phase D subtotal (your share)** | | | **~35h** |

### Extensions (optional, Phase E)
If we do GNN / ESM-2 / 3D extensions — budget another ~50h

### Grand Total Budget
| Phase | Hours |
|-------|-------|
| Part 1 setup + verify | <0.5 |
| Phase A | 20 |
| Phase C | 75 |
| Phase D | 35 |
| Extensions (optional) | 50 |
| **Total committed** | **~180h** |
| **Buffer** | ~120h |

Plenty of headroom for mistakes and re-runs.

---

## Partition Cheat Sheet (for later reference)

| Partition | GPU | VRAM | When to use |
|-----------|-----|------|-------------|
| `interactive` | None | — | Debug shell, edit files, no GPU |
| `n1s8-t4-1` | T4 | 16 GB | **Default for Phase A** — 30k pairs, small/medium models |
| `g2-standard-12` | L4 | 24 GB | Phase C mid-size (KIBA), bigger models |
| `g2-standard-24` | 2× L4 | 48 GB total | Multi-GPU; skip unless needed |
| `c12m85-a100-1` | A100 40GB | 40 GB | Full BindingDB, large-scale training |
| `c24m170-a100-2` | 2× A100 | 80 GB total | Only if DDP matters |

**Rule:** always use the smallest partition that fits. T4 is free-est; A100 is precious.

---

## What to Do After Part 1

Once `verify_env.sbatch` prints ALL CHECKS PASSED:

1. **Tell the team** in group chat: "HPC env verified, ready for V4 runs"
2. **Wait for shared scaffolding** — whoever owns `src/data/`, `src/models/base.py`, `src/training/trainer.py` needs to land their code on `main` branch
3. **Pull updated code** onto HPC: `cd /scratch/$USER/dti-project && git pull`
4. **Part 2 will be:** build the Late Concat variant module + first baseline training run

You don't need to re-run `setup_env.sh` when you pull new code — the environment persists. You only rebuild if you add new Python dependencies.

---

## Common Gotchas (save future-you some pain)

1. **Don't put data in `/home/`.** Quota is ~20 GB. All data → `/scratch/$USER/`.
2. **Scratch is NOT backed up.** Copy critical results back to your Mac or Google Drive weekly.
3. **OOD session ≠ compute node.** When you launch Jupyter/Shell in OOD, you're on a login node. Real training must go through `sbatch` or `srun`.
4. **Don't run GPU code on the login node.** If you type `python train.py` directly on a login node, you'll either have no GPU access or get a warning.
5. **W&B offline mode.** Compute nodes often can't reach wandb.ai. Set `WANDB_MODE=offline` (already in `setup_env.sh` env.sh). Sync from login node later with `wandb sync wandb/offline-run-*`.
6. **Time limits matter.** If you request 4 hours and training takes 5, SLURM kills it. Request generous times; unused time isn't wasted.
7. **Signing out doesn't kill jobs.** Closing your browser tab, laptop lid, going home — jobs keep running. Come back anytime.

---

## Troubleshooting Checklist

| Symptom | Fix |
|---------|-----|
| `squeue -u $USER` is empty right after submission | Did the sbatch command print a job ID? Re-submit. |
| Job stuck in `PD` (pending) for >15 min | `squeue -p n1s8-t4-1` to see queue length; maybe try `g2-standard-12` |
| `logs/verify_*.out` empty or missing | Job failed before writing; check `logs/verify_*.err` |
| `singularity: command not found` | `module load singularity` then re-run |
| Overlay too small | Re-create with bigger template: `/scratch/work/public/overlay-fs-ext3/overlay-50GB-500K.ext3.gz` |
| `CUDA error: out of memory` in verify | Weird for verify; check other jobs of yours aren't hogging the GPU |

---

## Part 1 Complete Checklist

Before calling Part 1 done:

- [ ] Logged into OOD successfully
- [ ] `/scratch/$USER/dti-project/` exists and contains project files
- [ ] `/scratch/$USER/dti-overlay.ext3` exists (25 GB)
- [ ] `sbatch verify_env.sbatch` submitted
- [ ] `logs/verify_*.out` contains "ALL CHECKS PASSED"
- [ ] I understand my Phase A budget (~20 GPU-hrs, target partition `n1s8-t4-1`)
- [ ] I know how to check my remaining quota: `sshare -A csci_ga_2565-2026sp -u $USER`

Once all boxes are checked, you're ready for Part 2.

---

## Part 2 Preview (what's next)

Once the team's shared scaffolding is on `main`:

1. Pull the code
2. Write your `V4LateConcat` model in `src/models/variants/late_concat.py`
3. Create your config `configs/variant_late_concat.yaml`
4. Submit your baseline: `sbatch hpc/train_single.sbatch configs/variant_late_concat.yaml bindingdb_mini random 42`
5. Check result, iterate

We'll walk through Part 2 when the scaffolding is ready. For now — get the environment built and verified. That's Part 1.
