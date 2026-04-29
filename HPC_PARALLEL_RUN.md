# HPC parallel-run checklist

Two independent things to launch on HPC right now. Both fit inside your remaining ~280 GPU-hour budget.

| Track | Purpose | Wall clock | GPU-hours | Output |
|---|---|---|---|---|
| 1. **Phase D bugfix re-extract** | fix `predictions ≈ 0` bug, add V1 + V3 attention | ~10 min | <0.2 | `outputs/analysis_v2/v{1,2,3,4}_*/` |
| 2. **Phase C controlled runs** | 36 jobs (4 variants × 3 splits × 3 seeds) | ~30 min/job, parallel queue | ~15-20 | `outputs/phase_c/phase_c_*/` |

Run them in parallel — they don't share files or GPU slots.

---

## Track 1 — Phase D bugfix (you said you're starting this)

Already documented in [HPC_PHASE_D_BUGFIX.md](HPC_PHASE_D_BUGFIX.md). Quick reminder:

```bash
cd /scratch/$USER/ComparisionPDI

# 1. Edit scripts/extract_for_analysis.py — apply the fixes from HPC_PHASE_D_BUGFIX.md
nano scripts/extract_for_analysis.py
# Or: copy the v2 template wholesale from that doc.

# 2. Submit
sbatch hpc_late_concat/run_phase_d_extract_v2.sbatch
squeue -u $USER
```

Expected: a single ~10-min job. When it finishes, `outputs/analysis_v2/` will have V1, V2 (×2), V3, V4 with corrected predictions.

---

## Track 2 — Phase C (run the moment Track 1 is submitted, doesn't wait)

The Phase C templates are now in your local repo. **Sync them up to HPC first**, then submit.

### Step 2a — sync the new files to HPC

You added two new dirs locally:
- `configs/phase_c_fair.yaml`
- `hpc_phase_c/run_phase_c.sbatch`
- `hpc_phase_c/submit_phase_c_all.sh`

Two ways to get them on HPC:

**Option A — push from Mac, pull from HPC** (cleaner):

```bash
# Mac:
cd ~/CodeFiles/DTI_MLFinalProject
# Wait — these aren't in the GitHub repo yet. Push them on a branch first:
cd  ~/CodeFiles/DTI_MLFinalProject  # if a git repo locally; otherwise skip
# (The local DTI_MLFinalProject/ folder is not a git repo on its own — only the
#  HPC's /scratch/bg2896/ComparisionPDI is. So use scp instead.)
```

**Option B — scp from Mac directly**:

```bash
# from Mac (a fresh terminal):
scp -r ~/CodeFiles/DTI_MLFinalProject/configs        bg2896@gw.hpc.nyu.edu:/scratch/bg2896/ComparisionPDI/
scp -r ~/CodeFiles/DTI_MLFinalProject/hpc_phase_c    bg2896@gw.hpc.nyu.edu:/scratch/bg2896/ComparisionPDI/
```

(If `gw.hpc.nyu.edu` isn't your jumpbox, use whatever hostname you used for the Phase D zip download — same principle.)

**Option C — paste them via OOD terminal**:

If scp is annoying, copy each file's contents and paste into nano on HPC:

```bash
cd /scratch/$USER/ComparisionPDI
mkdir -p configs hpc_phase_c
nano configs/phase_c_fair.yaml         # paste content from local file
nano hpc_phase_c/run_phase_c.sbatch    # paste, then chmod +x
nano hpc_phase_c/submit_phase_c_all.sh # paste, then chmod +x
```

### Step 2b — sanity check one job before submitting all 36

```bash
cd /scratch/bg2896/ComparisionPDI
chmod +x hpc_phase_c/run_phase_c.sbatch hpc_phase_c/submit_phase_c_all.sh

# Submit ONE pilot job to verify the new sbatch works
sbatch hpc_phase_c/run_phase_c.sbatch early_concat random 42
squeue -u $USER

# Wait ~5-10 min, then check
ls outputs/phase_c/
tail -50 logs/phase_c_*.out | tail -50
```

If the pilot job:
- crashes immediately → likely a flag mismatch (`--epochs` not supported, etc.). Fix `run_phase_c.sbatch` and resubmit.
- runs but `results.csv` empty → check the train script output for hints.
- runs and produces `results.csv` with a sensible val MSE (~1.3-1.6 range) → ✅ **submit the rest.**

### Step 2c — submit all 36

```bash
bash hpc_phase_c/submit_phase_c_all.sh
squeue -u $USER | head
# Or watch the queue progress:
watch -n 30 'squeue -u $USER | wc -l'
```

A100 cluster usually gives you ~3-5 jobs running concurrently, so wall-clock for all 36 is ~6-10 hours. You don't have to babysit it; come back tomorrow morning and check `ls outputs/phase_c/ | wc -l` (should be 36).

---

## Monitoring both tracks at once

```bash
# All your jobs:
squeue -u $USER

# What's done in Phase C:
ls outputs/phase_c/ | wc -l                           # count finished dirs
find outputs/phase_c -name results.csv | wc -l        # finished AND successful

# What's done in Phase D bugfix:
ls outputs/analysis_v2/ 2>/dev/null

# GPU usage so far this month:
sacct -u $USER -X --starttime=2026-04-01 --format=Elapsed -P --noheader \
  | awk -F'|' '{n=split($1,t,":");h=(n==3?t[1]+t[2]/60+t[3]/3600:t[1]/60+t[2]/3600);s+=h}END{printf "%.1f GPU-hours\n",s}'
```

---

## Things to do on Mac while HPC runs

While the cluster is grinding, useful things I can do locally without waiting:

1. **Build figure templates** that auto-fill once `outputs/phase_c/*/results.csv` is on Mac:
   - diagram_10 upgrade — bars become mean ± std over seeds
   - diagram_11 — concordance index per variant per split (NEW)
   - diagram_12 — train/val loss curves (NEW, needs per-epoch logs)
   - diagram_13 — predicted-vs-true scatter, 4-panel (NEW, needs test predictions)
2. **Push the Phase D summaries + new figures to GitHub** on a `bhavesh/phase-d-figures` branch.
3. **Sweep `src/` for the actual training-script CLI** to verify the `run_phase_c.sbatch` flags match `train.py` exactly. Tell me if you want this — I'll need the contents of `scripts/train.py`.

Tell me: `pilot worked` or `pilot failed: <error>` and I'll keep going.
