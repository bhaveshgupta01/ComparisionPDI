# Final GitHub push — README + everything else

Goal: get the comprehensive `README.md`, updated `POSTER.md`, all 27 figures, and every other doc onto the GitHub repo so teammates can read the full story.

The local Mac folder isn't a git clone, so the flow is: **tar locally → upload tarball to HPC via OOD → untar + commit + push from HPC.**

---

## Block 1 — On Mac (run in a fresh terminal)

```bash
cd ~/CodeFiles/DTI_MLFinalProject

# Tar everything that should land on GitHub
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

This produces a single ~10-15 MB tarball.

---

## Block 2 — Upload via OOD file browser

1. Open https://ood-burst-001.hpc.nyu.edu/.
2. Top menu → **Files** → navigate to `/scratch/bg2896/ComparisionPDI/`.
3. Click **Upload** → select `/tmp/dti_final_push.tar.gz` from your Mac.
4. After upload, the tarball will be at `/scratch/bg2896/ComparisionPDI/dti_final_push.tar.gz`.

---

## Block 3 — On HPC (OOD terminal)

Single paste-block:

```bash
cd /scratch/$USER/ComparisionPDI

# 1. Sanity: tarball is here
ls -lh dti_final_push.tar.gz

# 2. Extract (overwrites any existing copies of these files — that's intended)
tar -xzf dti_final_push.tar.gz

# 3. Confirm the new files are visible
ls README.md POSTER.md FINDINGS.md INDEX.md
ls poster_figures/ | head -10
ls poster_figures/ | wc -l    # should show ~54 (27 png + 27 svg)

# 4. Get on the right branch and pull latest
git fetch origin
git checkout main
git pull origin main

# 5. Create the final-push branch (or reuse if it exists)
git checkout -B bhavesh/final-push origin/main

# 6. Stage everything in one shot
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

# 7. Verify what's staged (should show ~80-100 files)
echo "=== Staged: $(git diff --cached --name-only | wc -l) files ==="
git diff --cached --name-only | head -40

# 8. Commit + push
git commit -m "Final: README + comprehensive docs + 27 poster figures + Phase C results + Phase D summaries" \
  --author="Bhavesh Gupta <bhaveshgupta01@gmail.com>"
git push -u origin bhavesh/final-push

# 9. Cleanup
rm dti_final_push.tar.gz
```

---

## Block 4 — Open a pull request to merge into main

After Block 3, the new branch is on GitHub. To merge into `main`:

```bash
# from HPC or Mac, using the gh CLI:
gh pr create --base main --head bhavesh/final-push \
  --title "Final project push: README + 27 figures + Phase C/D results" \
  --body "Lands the comprehensive README, the updated POSTER.md, all 27 poster figures (PNG+SVG), Phase C results.csv files (36/36), Phase D summaries, and all HPC runbooks. See README.md for entry point."
```

Or just do it via GitHub web UI: visit `https://github.com/bhaveshgupta01/ComparisionPDI/pull/new/bhavesh/final-push`.

---

## Verification checklist

After Block 3 finishes, visit:

1. `https://github.com/bhaveshgupta01/ComparisionPDI/tree/bhavesh/final-push` — top-level README.md should render with the table of contents you wrote.
2. `https://github.com/bhaveshgupta01/ComparisionPDI/tree/bhavesh/final-push/poster_figures` — 27 figures visible.
3. `https://github.com/bhaveshgupta01/ComparisionPDI/blob/bhavesh/final-push/FINDINGS.md` — headline table renders correctly.
4. `https://github.com/bhaveshgupta01/ComparisionPDI/blob/bhavesh/final-push/POSTER.md` — full poster draft, no [TBD] markers.

If any of those don't render, paste me the URL and the error.

---

## What teammates do

Once the branch is merged into `main`:

```bash
# On their machine:
git clone https://github.com/bhaveshgupta01/ComparisionPDI.git
cd ComparisionPDI
cat README.md      # entry point — has everything they need
```

The README explains what every file does, how the workflow goes, where to find what, glossary of terms, and quick-start commands. They should be able to onboard themselves without further help.
