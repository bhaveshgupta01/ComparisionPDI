# Sync local artifacts to GitHub (single block)

While Phase D bugfix + Phase C are running, you can push everything we've built locally up to the repo. The local Mac folder isn't a git repo — the source of truth is `/scratch/bg2896/ComparisionPDI/` on HPC. So the flow is: scp/paste local files up to HPC, then commit & push from HPC.

This single bash block does it all.

---

## Block — paste into OOD terminal

```bash
cd /scratch/$USER/ComparisionPDI

# Make sure the configs + hpc_phase_c stuff is in place (it should be from
# HPC_PHASE_C_RUN.md Block 1; this is a no-op if already done)
ls -la configs/phase_c_fair.yaml hpc_phase_c/run_phase_c.sbatch hpc_phase_c/submit_phase_c_all.sh \
       scripts/extract_for_analysis_v2.py hpc_late_concat/run_phase_d_extract_v2.sbatch \
       2>&1 | head -10

# ---- Create the docs branch -----------------------------------------------
git fetch origin
git checkout main
git checkout -b bhavesh/phase-bcd-setup 2>/dev/null || git checkout bhavesh/phase-bcd-setup

# Stage the Phase B / C / D scripts and configs (force, since outputs/ is gitignored
# but configs/ and scripts/ shouldn't be — but use -f for safety)
git add -f \
    configs/phase_c_fair.yaml \
    hpc_phase_c/run_phase_c.sbatch \
    hpc_phase_c/submit_phase_c_all.sh \
    scripts/extract_for_analysis_v2.py \
    hpc_late_concat/run_phase_d_extract_v2.sbatch \
    2>/dev/null

git status | head -30

git commit -m "Phase B/C/D setup: locked fair config, Phase C sbatch + submitter, Phase D v2 extract" \
  --author="Bhavesh Gupta <bhaveshgupta01@gmail.com>" \
  --allow-empty
git push -u origin bhavesh/phase-bcd-setup
```

That gets the HPC-side scripts into the repo. The Mac-side artifacts (poster figures, decision docs, summarizer) need to come up too.

---

## Block 2 — push Mac-only artifacts via scp + commit on HPC

Run this on **your Mac** (NOT inside OOD):

```bash
cd ~/CodeFiles/DTI_MLFinalProject

# Tar up everything the HPC repo doesn't yet have
tar -czf /tmp/dti_mac_artifacts.tar.gz \
    poster_figures/ \
    scripts/extract_and_summarize_phase_d.py \
    PHASE_B_DECISION.md \
    PHASE_B_MEETING_AGENDA.md \
    HPC_NEXT_STEPS.md HPC_FOLLOWUP_FIXES.md HPC_PHASE_D_BUGFIX.md \
    HPC_PARALLEL_RUN.md HPC_PHASE_C_RUN.md HPC_GITHUB_SYNC.md \
    CLEANUP_AT_END.md \
    branch_docs/STATUS.md 2>/dev/null

ls -lh /tmp/dti_mac_artifacts.tar.gz

# Push to HPC (use whichever NYU HPC hostname your account uses)
scp /tmp/dti_mac_artifacts.tar.gz bg2896@gw.hpc.nyu.edu:/scratch/bg2896/ComparisionPDI/
```

Then back on **HPC** (OOD terminal):

```bash
cd /scratch/$USER/ComparisionPDI
git checkout bhavesh/phase-bcd-setup     # if not already on it

mkdir -p docs
tar -xzf dti_mac_artifacts.tar.gz
ls -la poster_figures/ | head -10        # verify

git add -f \
    poster_figures/build_all.py \
    poster_figures/diagram_*.svg \
    poster_figures/diagram_*.png \
    poster_figures/phase_a_4variant_csv.py \
    scripts/extract_and_summarize_phase_d.py \
    PHASE_B_DECISION.md \
    HPC_NEXT_STEPS.md HPC_FOLLOWUP_FIXES.md HPC_PHASE_D_BUGFIX.md \
    HPC_PARALLEL_RUN.md HPC_PHASE_C_RUN.md HPC_GITHUB_SYNC.md \
    CLEANUP_AT_END.md \
    2>/dev/null

git status | head -40

git commit -m "Add Mac-side artifacts: poster figures (16 diagrams), Phase B/C/D docs, summarizer" \
  --author="Bhavesh Gupta <bhaveshgupta01@gmail.com>"
git push origin bhavesh/phase-bcd-setup

# Cleanup the tarball
rm dti_mac_artifacts.tar.gz
```

---

## What ends up on GitHub

After both blocks, the `bhavesh/phase-bcd-setup` branch has:

**Configs / sbatch / scripts:**
- `configs/phase_c_fair.yaml`
- `hpc_phase_c/run_phase_c.sbatch`
- `hpc_phase_c/submit_phase_c_all.sh`
- `scripts/extract_for_analysis_v2.py`
- `scripts/extract_and_summarize_phase_d.py`
- `hpc_late_concat/run_phase_d_extract_v2.sbatch`

**Decision / runbook docs:**
- `PHASE_B_DECISION.md` — locked Phase C fair config
- `HPC_NEXT_STEPS.md`, `HPC_FOLLOWUP_FIXES.md`, `HPC_PHASE_D_BUGFIX.md`
- `HPC_PARALLEL_RUN.md`, `HPC_PHASE_C_RUN.md`, `HPC_GITHUB_SYNC.md`
- `CLEANUP_AT_END.md`

**Poster figures (16 diagrams, SVG + PNG):**
- `poster_figures/build_all.py`
- `poster_figures/diagram_{03,04,05,08,09,10,10b,11,13,14,15,16,17,31,32,33,36,37,38,39,40}_*.{svg,png}` (the ones that have data; placeholders skipped)

You can open a PR from `bhavesh/phase-bcd-setup` → `main` whenever you want to fold the work back into mainline.

---

## Skip this for now if HPC is busy

This isn't on the critical path. The work is **already saved locally on Mac**; pushing to GitHub is just for backup + sharing. If you're juggling the Phase D + Phase C terminal output, do this later.
