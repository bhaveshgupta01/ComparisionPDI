# HPC follow-up fixes — V1/V3 push + scp Phase D artifacts

## What just happened (the good)

- **Job 176079 COMPLETED** in 6:16, exit 0:0. Phase D artifacts exist in:
  - `outputs/analysis/v2_baseline/`
  - `outputs/analysis/v2_dm256_bs16/`
  - `outputs/analysis/v4_baseline/`
  - Each contains: `predictions.npy`, `truth.npy`, `attn_drug_encoder_layer{0,1,2}.npy`, `attn_prot_encoder_layer{0,1,2}.npy`, `meta.json`.
- **`bhavesh/phase-a-comparison`** branch pushed cleanly with the CSV + build script.

## What went sideways (the fixable)

1. **V1 and V3 branches got pushed empty.** `outputs/` is in `.gitignore`, so `git add outputs/sweeps/v1_*/results/results.csv` silently matched nothing. The branches were created and pushed but contain no new commits (`Total 0 (delta 0)`).
2. **scp command failed** because `<hpc-host>` was pasted literally — the shell tried to read from a file named `hpc-host`.

---

## Step A — Fix V1/V3 branches with force-add

Run on HPC (you're already in `/scratch/bg2896/ComparisionPDI`):

```bash
cd /scratch/bg2896/ComparisionPDI

# Sanity: confirm what will be added
ls outputs/sweeps/v1_*/results/results.csv | wc -l        # ~19 expected
ls outputs/sweeps/v3_*/results/results.csv | wc -l        # ~22 expected
ls hpc_early_concat/ 2>/dev/null
ls hpc_late_crossattn/ 2>/dev/null

# --- V1 branch ---
git checkout bhavesh/v1-lingwei-sweep
git add -f outputs/sweeps/v1_*/results/results.csv
git add -f hpc_early_concat/
git status | head -40
git commit -m "V1 (early concat) Phase A sweep results - run on bhavesh's quota" \
  --author="Bhavesh Gupta <bhaveshgupta01@gmail.com>"
git push -u origin bhavesh/v1-lingwei-sweep

# --- V3 branch ---
git checkout main
git checkout bhavesh/v3-tenzin-sweep
git add -f outputs/sweeps/v3_*/results/results.csv
git add -f hpc_late_crossattn/
git status | head -40
git commit -m "V3 (late cross-attn) Phase A sweep results - run on bhavesh's quota" \
  --author="Bhavesh Gupta <bhaveshgupta01@gmail.com>"
git push -u origin bhavesh/v3-tenzin-sweep

git checkout main
```

> Why `--author=` — your default git committer is showing as `bg2896@b-9-46.c.hpc-slurm-9c75.internal` (a hostname). Setting your real email keeps the GitHub commit attribution clean. (You could also fix it permanently with `git config --global user.email bhaveshgupta01@gmail.com` and `user.name "Bhavesh Gupta"` once.)

---

## Step B — Pull Phase D artifacts to your Mac

The artifacts are on a cloud-burst node, not the main HPC fileserver. Two clean options:

### Option 1 (recommended) — OOD file browser

1. In your browser, open **https://ood-burst-001.hpc.nyu.edu/**.
2. Top menu → **Files** → **Home Directory** → navigate to `/scratch/bg2896/ComparisionPDI/outputs/analysis`.
3. Tick `v2_baseline`, `v2_dm256_bs16`, `v4_baseline` → **Download** (it'll zip them).
4. Unzip into `~/CodeFiles/DTI_MLFinalProject/phase_d_artifacts/`.

### Option 2 — `rsync` from HPC over to Mac via Greene gateway

If your Mac is reachable via SSH (it usually isn't from outside campus, so this only works if you've set up a tunnel), you can `rsync` from HPC. Easier: from your **Mac**, ssh to the HPC and pull.

NYU's general-HPC hostname is `gw.hpc.nyu.edu` (the Greene gateway). The cloud-burst nodes (`b-19-73`) are inside the cluster and not directly addressable; you have to bounce through Greene:

```bash
# On your Mac (a fresh terminal, NOT inside OOD):
mkdir -p ~/CodeFiles/DTI_MLFinalProject/phase_d_artifacts

# Direct scp via Greene (you'll be prompted for your NYU password / Duo):
scp -r bg2896@gw.hpc.nyu.edu:/scratch/bg2896/ComparisionPDI/outputs/analysis \
       ~/CodeFiles/DTI_MLFinalProject/phase_d_artifacts/
```

If `gw.hpc.nyu.edu` doesn't work, try `greene.hpc.nyu.edu` (the renamed legacy alias). Your NYU IT docs / OOD's "About / Connect via SSH" link will have the canonical hostname for your account.

### Option 3 — `tar` it on HPC, `git lfs` push, pull on Mac

Quickest reproducible path that doesn't need ssh-to-Mac:

```bash
# On HPC:
cd /scratch/bg2896/ComparisionPDI
tar -czf phase_d_artifacts.tar.gz outputs/analysis/
ls -lh phase_d_artifacts.tar.gz                # check size — likely <100 MB

# Track in LFS (the repo already uses LFS for the BindingDB tsv)
git lfs track "phase_d_artifacts.tar.gz"
git checkout -b bhavesh/phase-d-artifacts
git add -f phase_d_artifacts.tar.gz .gitattributes
git commit -m "Phase D extraction artifacts (V2 baseline, V2 dm256_bs16, V4 baseline)"
git push -u origin bhavesh/phase-d-artifacts

# On your Mac:
cd ~/CodeFiles/DTI_MLFinalProject
git fetch origin
git checkout bhavesh/phase-d-artifacts -- phase_d_artifacts.tar.gz
mkdir -p phase_d_artifacts
tar -xzf phase_d_artifacts.tar.gz -C phase_d_artifacts/
```

---

## Step C — once artifacts are on your Mac, ping me

I'll build the Phase D figures next:
- **diagram_16** attention entropy curves (per layer × per variant)
- **diagram_17** attention heatmap on a held-out pair
- **diagram_13** predicted-vs-true scatter (4-panel — though we only have V2+V4 so 2-panel for now)
- **diagram_27** error vs predicted-pKi (failure-mode preview)

We don't have V1 or V3 artifacts yet — we'd need to rerun the extraction script with V1 and V3 checkpoints. Worth doing once V1/V3 are pushed and you've got a stable extract path. Until then, V2-vs-V4 is a meaningful early-vs-late contrast.

---

## Quick reference

| Want | Command |
|---|---|
| Verify .npy files are non-empty | `for f in outputs/analysis/*/*.npy; do echo "$f $(du -h $f \| cut -f1)"; done` |
| Inspect meta.json | `cat outputs/analysis/v4_baseline/meta.json` |
| Check artifact total size | `du -sh outputs/analysis/` |
| Re-extract for V1/V3 | edit `scripts/extract_for_analysis.py` to add V1/V3 checkpoints, resubmit `hpc_late_concat/run_phase_d_extract.sbatch` |
