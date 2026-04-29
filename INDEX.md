# Project Index — what's where

> Quick map of every doc / script / artifact in this folder. Sorted by what you'll actually need next.

## 🟢 Active runbooks (paste-and-go)

| File | Purpose | Status |
|---|---|---|
| [HPC_PHASE_D_BUGFIX.md](HPC_PHASE_D_BUGFIX.md) | Single-block: re-run Phase D extract with prediction-scaling fix | submitted |
| [HPC_PHASE_C_RUN.md](HPC_PHASE_C_RUN.md) | Block 1: write Phase C config + sbatch + pilot. Block 2: fire 35 more | next to run |
| [HPC_GITHUB_SYNC.md](HPC_GITHUB_SYNC.md) | Push all local artifacts to GitHub on `bhavesh/phase-bcd-setup` branch | optional, low priority |

## 🔵 Decisions / state docs

| File | Purpose |
|---|---|
| [PHASE_B_DECISION.md](PHASE_B_DECISION.md) | Locked Phase C fair config + rationale |
| [PHASE_A_4VARIANT_COMPARISON.csv](PHASE_A_4VARIANT_COMPARISON.csv) | Phase A val MSE for all 4 variants × 26 configs (built on HPC) |
| [POSTER.md](POSTER.md) | Poster master doc — sections, figures, all `[TBD]` placeholders |
| [CLEANUP_AT_END.md](CLEANUP_AT_END.md) | Running list of what to delete at project end |
| [branch_docs/STATUS.md](branch_docs/STATUS.md) | V4 branch status (legacy, V4-specific) |

## 🟡 Legacy / superseded docs (kept for context)

| File | Why still here | What replaced it |
|---|---|---|
| [CONTEXT_FOR_NEXT_CHAT.md](CONTEXT_FOR_NEXT_CHAT.md) | Original session handoff | conversation continuity now |
| [PHASE_B_MEETING_AGENDA.md](PHASE_B_MEETING_AGENDA.md) | Drafted before solo decision | PHASE_B_DECISION.md |
| [HPC_NEXT_STEPS.md](HPC_NEXT_STEPS.md) | First HPC runbook (CSV fix + V1/V3 push) | done |
| [HPC_FOLLOWUP_FIXES.md](HPC_FOLLOWUP_FIXES.md) | Second runbook (V1/V3 force-add fix) | done |
| [HPC_PARALLEL_RUN.md](HPC_PARALLEL_RUN.md) | Earlier two-track checklist | HPC_PHASE_C_RUN.md is more concrete |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Original team-level vision doc | poster + repo |
| [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) | Original engineering spec | code in repo |
| [TEAM_BLUEPRINT.md](TEAM_BLUEPRINT.md), [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md) | Multi-person workflow | soloing now |
| [GAMEPLAN.md](GAMEPLAN.md), [DEEP_ANALYSIS_PLAYBOOK.md](DEEP_ANALYSIS_PLAYBOOK.md) | Phase D plan | absorbed into POSTER.md §10 |

## 🟣 Build outputs

| Path | What's in it |
|---|---|
| [poster_figures/](poster_figures/) | 16 finished diagrams (SVG + PNG), `build_all.py` script |
| [phase_d_summaries/](phase_d_summaries/) | Compressed Phase D extraction (16 MB), used by diagram_16/17 |
| `phase_d_artifacts/*.zip` | Original Phase D zips (7.7 GB, kept for now per CLEANUP_AT_END.md) |
| `phase_d_artifacts_v2/` | Will hold the Phase D bugfix re-run output once you scp it |
| [configs/phase_c_fair.yaml](configs/phase_c_fair.yaml) | Locked Phase C config |
| [hpc_phase_c/](hpc_phase_c/) | Phase C sbatch + 36-job submitter |

## 🟠 Build scripts

| File | Purpose |
|---|---|
| [poster_figures/build_all.py](poster_figures/build_all.py) | Generates all 16+ diagrams. Auto-skips Phase C / Phase D figures if data isn't present yet. |
| [scripts/extract_and_summarize_phase_d.py](scripts/extract_and_summarize_phase_d.py) | Mac-side: unzip Phase D artifacts → entropy summary + sample heatmaps → discard giants |
| [poster_figures/phase_a_4variant_csv.py](poster_figures/phase_a_4variant_csv.py) | HPC-side: rebuild PHASE_A_4VARIANT_COMPARISON.csv from sweep outputs |

## Phase status snapshot

| Phase | Status | Source of truth |
|---|---|---|
| **A** Individual Tuning (~22 sweeps × 4 variants) | ✅ DONE — pushed to GitHub on `bhavesh/v{1,2,3,4}-*` branches | `PHASE_A_4VARIANT_COMPARISON.csv` |
| **B** Fair-Config Negotiation | ✅ LOCKED solo | `PHASE_B_DECISION.md` + `configs/phase_c_fair.yaml` |
| **C** Controlled Final Runs (4 × 3 × 3 = 36 jobs) | 🟡 about to fire | `HPC_PHASE_C_RUN.md` |
| **D** Deep Analysis | 🟡 first extraction had bugs; bugfix re-run in flight | `HPC_PHASE_D_BUGFIX.md` |
| **Poster figures** | 16 of ~31 built; remaining wait on Phase C + Phase D v2 | `poster_figures/` |
| **Cleanup** | running list maintained | `CLEANUP_AT_END.md` |

## When you come back next session

If conversation context is lost, the read order is:

1. `INDEX.md` (this file)
2. `PHASE_B_DECISION.md` — what we're trying to do in Phase C
3. Whatever HPC log file is currently relevant (job IDs change session-to-session)
4. `POSTER.md` if working on poster content
