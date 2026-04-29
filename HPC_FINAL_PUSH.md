# Final GitHub push — README + comprehensive docs

Goal: get the comprehensive `README.md`, updated `POSTER.md`, all 27 figures, and the rest of the docs onto `main` so teammates have a single source of truth.

The local Mac folder isn't a git clone, so the flow is: **tar locally → upload tarball to HPC via OOD → untar + commit + push from HPC.**

---

## Block 1 — On Mac (run in a fresh terminal)

```bash
cd ~/CodeFiles/DTI_MLFinalProject

tar -czf /tmp/dti_final_push.tar.gz \
    README.md \
    POSTER.md \
    FINDINGS.md \
    INDEX.md \
    PHASE_B_DECISION.md \
    PHASE_B_MEETING_AGENDA.md \
    CLEANUP_AT_END.md \
    HPC_NEXT_STEPS.md \
    HPC_FOLLOWUP_FIXES.md \
    HPC_PARALLEL_RUN.md \
    HPC_PARALLEL_WHILE_WAITING.md \
    HPC_PHASE_C_RUN.md \
    HPC_PHASE_D_BUGFIX.md \
    HPC_GITHUB_SYNC.md \
    HPC_FINAL_PUSH.md \
    poster_figures/ \
    scripts/extract_and_summarize_phase_d.py \
    scripts/extract_for_analysis_v2.py \
    scripts/extract_for_analysis_v3.py \
    scripts/extract_for_analysis_v4.py \
    binding_db_stats/ \
    2>/dev/null

ls -lh /tmp/dti_final_push.tar.gz
```

---

## Block 2 — Upload via OOD file browser

1. Open https://ood-burst-001.hpc.nyu.edu/ → **Files** → navigate to `/scratch/bg2896/ComparisionPDI/`.
2. Click **Upload** → select `/tmp/dti_final_push.tar.gz` from your Mac.
3. After upload, the tarball will be at `/scratch/bg2896/ComparisionPDI/dti_final_push.tar.gz`.

---

## Block 3 — On HPC (OOD terminal)

Since the prior work is already merged into `main`, this commit lands directly on `main` (no PR needed).

```bash
cd /scratch/$USER/ComparisionPDI

# 1. Sanity: tarball is here
ls -lh dti_final_push.tar.gz

# 2. Extract (overwrites local copies — that's intended)
tar -xzf dti_final_push.tar.gz
ls README.md POSTER.md FINDINGS.md INDEX.md
ls poster_figures/ | wc -l    # should be ~54 (27 png + 27 svg)

# 3. Sync with origin/main and check out
git fetch origin
git checkout main
git pull --ff-only origin main

# 4. Stage everything in one shot
git add -f README.md
git add -f POSTER.md FINDINGS.md INDEX.md PHASE_B_DECISION.md PHASE_B_MEETING_AGENDA.md CLEANUP_AT_END.md
git add -f HPC_NEXT_STEPS.md HPC_FOLLOWUP_FIXES.md HPC_PARALLEL_RUN.md HPC_PARALLEL_WHILE_WAITING.md
git add -f HPC_PHASE_C_RUN.md HPC_PHASE_D_BUGFIX.md HPC_GITHUB_SYNC.md HPC_FINAL_PUSH.md
git add -f poster_figures/
git add -f scripts/extract_for_analysis_v2.py scripts/extract_for_analysis_v3.py scripts/extract_for_analysis_v4.py
git add -f scripts/extract_phase_d_from_phase_c.py 2>/dev/null
git add -f scripts/extract_and_summarize_phase_d.py
git add -f binding_db_stats/
git add -f configs/phase_c_fair.yaml hpc_phase_c/
find outputs/phase_c -name "results.csv" -print0 | xargs -0 -r git add -f
find outputs/phase_c -name "*_history.csv" -print0 | xargs -0 -r git add -f

# 5. Verify what's staged (should show ~80-100 files)
echo "=== Staged: $(git diff --cached --name-only | wc -l) files ==="
git diff --cached --name-only | head -30

# 6. Commit + push directly to main
git commit -m "Add comprehensive README, updated POSTER, all 27 figures, Phase C/D results" \
  --author="Bhavesh Gupta <bhaveshgupta01@gmail.com>"
git push origin main

# 7. Cleanup
rm dti_final_push.tar.gz
```

---

## Verification

After Block 3, visit:

1. https://github.com/bhaveshgupta01/ComparisionPDI — top-level `README.md` should render with the full table of contents.
2. https://github.com/bhaveshgupta01/ComparisionPDI/tree/main/poster_figures — 27 figures visible.
3. https://github.com/bhaveshgupta01/ComparisionPDI/blob/main/FINDINGS.md — headline table renders.
4. https://github.com/bhaveshgupta01/ComparisionPDI/blob/main/POSTER.md — full poster draft renders.

---

## What teammates do after this

```bash
git clone https://github.com/bhaveshgupta01/ComparisionPDI.git
cd ComparisionPDI
cat README.md      # entry point — has everything
```

The README explains every directory, every file, every workflow, plus quick-start commands and a glossary. They should be able to onboard themselves without further help.
