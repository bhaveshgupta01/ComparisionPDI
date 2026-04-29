# HPC Next Steps — copy-paste runbook

> Run these on the OOD terminal at `https://ood-burst-001.hpc.nyu.edu/`.
> Cluster login → `ssh bg2896@<jumpbox>` is fine too. Working dir is `/scratch/bg2896/ComparisionPDI`.

Use this script as a checklist. Do **Step 0 → 1 → 2 → 3** in order. Step 4 (poster figures push) waits until the local figures regenerate.

---

## Step 0 — Land in the right directory and sanity-check state

```bash
cd /scratch/bg2896/ComparisionPDI
source .venv/bin/activate
squeue -u $USER                      # see if 176079 (phase-d-extract) is still queued / running
ls outputs/sweeps | wc -l            # should be plenty (V1/V2/V3/V4 sweeps)
ls outputs/analysis 2>/dev/null      # empty until 176079 finishes
```

Expected: job `176079` either `PD (Priority)` or `R` on partition `c12m85-a1`. If it's `R`, tail the log:

```bash
tail -f logs/phase_d_extract_176079.out
```

---

## Step 1 — Rebuild the broken `PHASE_A_4VARIANT_COMPARISON.csv`

The previous heredoc paste collided with `PYEOF` and only the header landed. Use a **file**, not a heredoc.

```bash
cat > build_4variant_csv.py << 'PY_END'
"""Build PHASE_A_4VARIANT_COMPARISON.csv from sweep outputs."""
import csv, glob, os, re

leaderboards = {"V1": {}, "V2": {}, "V3": {}, "V4": {}}

def add_variant(prefix, lboard):
    for f in glob.glob(f"outputs/sweeps/{prefix}*/results/results.csv"):
        tag = os.path.basename(os.path.dirname(os.path.dirname(f)))
        base = tag
        for p in ["v1_", "v2_", "v3_"]:
            if base.startswith(p):
                base = base[len(p):]
                break
        base = re.sub(r"_fast$", "", base)
        with open(f) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if len(lines) >= 2:
            lboard[base] = lines[-1].split(",")[-1]

add_variant("v1_", leaderboards["V1"])
add_variant("v2_", leaderboards["V2"])
add_variant("v3_", leaderboards["V3"])

# V4 = anything *_fast that doesn't start with v1_/v2_/v3_
for f in glob.glob("outputs/sweeps/*_fast/results/results.csv"):
    tag = os.path.basename(os.path.dirname(os.path.dirname(f)))
    if any(tag.startswith(p) for p in ["v1_", "v2_", "v3_"]):
        continue
    base = re.sub(r"_fast$", "", tag)
    with open(f) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if len(lines) >= 2:
        leaderboards["V4"][base] = lines[-1].split(",")[-1]

all_configs = sorted(set().union(*[lb.keys() for lb in leaderboards.values()]))
with open("PHASE_A_4VARIANT_COMPARISON.csv", "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["config", "V1_early_concat", "V2_early_xattn",
                "V3_late_xattn", "V4_late_concat"])
    for cfg in all_configs:
        w.writerow([cfg] + [leaderboards[v].get(cfg, "") for v in ["V1","V2","V3","V4"]])

print(f"Wrote {len(all_configs)} rows.")
print(open("PHASE_A_4VARIANT_COMPARISON.csv").read())
PY_END

python3 build_4variant_csv.py
```

Verify:

```bash
wc -l PHASE_A_4VARIANT_COMPARISON.csv      # should be ~27 (header + 26 configs)
head -5 PHASE_A_4VARIANT_COMPARISON.csv
```

---

## Step 2 — Push V1 + V3 sweep results to GitHub

You ran V1 (Lingwei) and V3 (Tenzin) on your quota. Their outputs sit in `outputs/sweeps/v1_*` and `outputs/sweeps/v3_*` and aren't pushed yet.

> **Decision point:** put them on **separate branches** so each owner can rebase / take ownership cleanly. Recommended:
> - `bhavesh/v1-lingwei-sweep`
> - `bhavesh/v3-tenzin-sweep`

```bash
cd /scratch/bg2896/ComparisionPDI

# --- branch 1: V1 ---
git checkout -b bhavesh/v1-lingwei-sweep
git add outputs/sweeps/v1_*/results/results.csv \
        outputs/sweeps/v1_*/logs/*.out \
        hpc_early_concat/run_v1_a100_fast.sbatch \
        hpc_early_concat/sweep_v1.sh 2>/dev/null
git status                                      # eyeball what's staged
git commit -m "V1 (early concat) Phase A sweep results - run on bhavesh's quota"
git push -u origin bhavesh/v1-lingwei-sweep

# --- branch 2: V3 ---
git checkout main
git checkout -b bhavesh/v3-tenzin-sweep
git add outputs/sweeps/v3_*/results/results.csv \
        outputs/sweeps/v3_*/logs/*.out \
        hpc_late_crossattn/run_v3_a100_fast.sbatch \
        hpc_late_crossattn/sweep_v3.sh 2>/dev/null
git status
git commit -m "V3 (late cross-attn) Phase A sweep results - run on bhavesh's quota"
git push -u origin bhavesh/v3-tenzin-sweep

# --- back to main ---
git checkout main

# --- also: push the 4-variant CSV on a separate branch so it's safe to share ---
git checkout -b bhavesh/phase-a-comparison
git add PHASE_A_4VARIANT_COMPARISON.csv build_4variant_csv.py
git commit -m "Phase A: 4-variant comparison CSV (all 4 variants, fast-mode val MSE)"
git push -u origin bhavesh/phase-a-comparison
git checkout main
```

> ⚠️ If `git add` complains about LFS / large files, use:
> ```bash
> git lfs track "outputs/sweeps/**/checkpoints/*.pt"
> git add .gitattributes
> ```
> Or just exclude checkpoints from the commits — only the `results.csv` and logs need to be shared.

---

## Step 3 — Watch / collect Phase D extraction (job 176079)

Once 176079 finishes, you'll have artifacts in `outputs/analysis/v{2,4}_*/`. Check:

```bash
sacct -j 176079 --format=JobID,State,ExitCode,Elapsed,ReqMem,NodeList
ls -la outputs/analysis/                # should be populated
ls outputs/analysis/v4_baseline/        # expect predictions.npy, attention_*.npy, drug_repr.npy, prot_repr.npy, meta.json
```

If `State=COMPLETED` and the npy files are there, scp them back to your Mac for figure building:

```bash
# from your Mac (NOT inside the OOD terminal):
mkdir -p ~/CodeFiles/DTI_MLFinalProject/phase_d_artifacts
scp -r bg2896@<hpc-host>:/scratch/bg2896/ComparisionPDI/outputs/analysis \
       ~/CodeFiles/DTI_MLFinalProject/phase_d_artifacts/
```

If the job **failed** (likely culprits: heredoc-mangled `extract_for_analysis.py`, or model_kwargs mismatch), inspect:

```bash
cat logs/phase_d_extract_176079.err | tail -80
```

> The previous extract script paste in the OOD terminal looks corrupted (lines like `chmod +x scripts/extract_for_analysis.pyanalysis/")t}")model": ...`). If 176079 errors out for that reason, you'll need to rewrite `scripts/extract_for_analysis.py` cleanly. Tell me when it fails and I'll hand you a corrected version of the script in a single file (no heredoc).

---

## Step 4 — Pull the regenerated 4-variant poster figures (after I finish them locally)

I'm regenerating `poster_figures/diagram_15_sensitivity_4variant.{png,svg}`, `diagram_39_4variant_leaderboard.{png,svg}`, etc. on your Mac right now. After you commit them locally, push from Mac:

```bash
# on your Mac:
cd ~/CodeFiles/DTI_MLFinalProject
# (we'll commit + push in the next step once the figures are regenerated and verified)
```

---

## Quick reference

| Task | Command |
|---|---|
| Check job state | `squeue -u $USER` and `sacct -j <id>` |
| Real GPU usage so far this month | `sacct -u $USER -X --starttime=2026-04-01 --format=Elapsed -P --noheader \| awk -F'\|' '{n=split($1,t,":");h=(n==3?t[1]+t[2]/60+t[3]/3600:t[1]/60+t[2]/3600);s+=h}END{printf "%.1f GPU-hours\n",s}'` |
| Tail latest log | `tail -f $(ls -t logs/*.out \| head -1)` |
| Free disk in scratch | `df -h /scratch \| head -2` |
| Cancel a queued job | `scancel <jobid>` |
