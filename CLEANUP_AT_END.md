# Cleanup at end of project

> Running list. Anything that's safe to delete once the poster is printed and grading is done.
> Order: biggest reclaimable bytes first.

## On Mac — `~/CodeFiles/DTI_MLFinalProject/`

| Path | Size | Why kept now | Safe to delete when |
|---|---|---|---|
| `phase_d_artifacts/v2_baseline.zip` | 3.2 GB | original Phase D extraction; may need to re-summarize if bug found in summarizer | after Phase D figures finalized |
| `phase_d_artifacts/v2_dm256_bs16.zip` | 3.2 GB | same | same |
| `phase_d_artifacts/v4_baseline.zip` | 1.3 GB | same | same |
| `phase_d_artifacts/_work/` | 0 (transient) | scratch dir for the summarizer | already auto-cleaned by script; rm anytime |
| `phase_d_summaries/` | 16 MB | source for diagram_16, diagram_17 | KEEP through poster; small enough to commit |

**Total reclaimable on Mac:** ~7.7 GB once Phase D figures are finalized.

## On HPC — `/scratch/bg2896/ComparisionPDI/`

| Path | Approx size | Why kept now | Safe to delete when |
|---|---|---|---|
| `outputs/sweeps/*/checkpoints/*.pt` | ~63 MB × ~85 sweeps = ~5 GB | source checkpoints for Phase A/D extractions | after Phase D re-extract is done |
| `outputs/phase_c/*/checkpoints/*.pt` | ~63 MB × 36 = ~2.3 GB | source for any final figure regen | after poster + report submitted |
| `outputs/analysis/v*` | ~25 GB (the giant attn npys) | already shipped to Mac as zips; redundant | now (zips are on Mac) |
| `outputs/analysis_v2/*/attn_*.npy` | similar to above (~25 GB) | bugfix re-extract output | after summarized → keep summaries only |
| `logs/*.out`, `logs/*.err` | ~50 MB each variant × dozens | helpful while debugging | after grading |
| `src/__pycache__/`, `src/*/__pycache__/` | <50 MB | regenerable | anytime |
| `.venv/` | ~5 GB | Python environment | only if rebuilding from scratch |

**Total reclaimable on HPC:** ~50+ GB after grading.

## On GitHub — `bhaveshgupta01/ComparisionPDI`

Branches we should clean up at end of project:
- `bhavesh/late-concat-phase-a` — merge into main, then delete
- `bhavesh/v1-lingwei-sweep` — merge / delete
- `bhavesh/v2-manas-sweep` — merge / delete
- `bhavesh/v3-tenzin-sweep` — merge / delete
- `bhavesh/phase-a-comparison` — merge / delete
- `bhavesh/phase-d-artifacts` (if pushed) — keep as a tag, then delete branch

Or: leave them all as historical artifacts. No urgency to delete branches.

## Cleanup commands (when ready)

```bash
# On Mac
cd ~/CodeFiles/DTI_MLFinalProject
rm phase_d_artifacts/*.zip
rm -rf phase_d_artifacts/_work

# On HPC
cd /scratch/$USER/ComparisionPDI
# Big one: delete the giant attention .npys (zips already saved them)
rm -rf outputs/analysis/*/attn_*.npy
rm -rf outputs/analysis_v2/*/attn_*.npy
# Pycache anytime
find . -type d -name __pycache__ -exec rm -rf {} +
# Old log files (keep a copy of the most recent before cleaning)
rm logs/lc_a100_17{4,5}*.out logs/lc_a100_17{4,5}*.err  # adjust ranges
```
