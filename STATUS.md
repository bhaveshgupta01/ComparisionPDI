# Project Status — V4 Late Concat Branch

**Snapshot:** 2026-04-27
**Branch:** `bhavesh/late-concat-phase-a`
**Owner:** Bhavesh Gupta
**Variant:** V4 — Late Concatenation

---

## TL;DR

Phase A sweep for Late Concat (V4) is essentially complete. Best full-run val MSE: **1.231** (beats Tenzin's 2x A100 baseline of 1.347 by 8.6%). Sensitivity table below. Phase B is blocked on the other 3 teammates' Phase A sweeps.

---

## What's Done

### Repository Scaffolding (Tenzin, weeks before)

All 4 variants implemented:
- `early_concat`
- `early_crossattn`
- `late_concat`
- `late_crossattn`

Shared modules under `src/`: encoders, embeddings, cross-attention, prediction head, dataset, splits, trainer, metrics.

### Late Concat (V4) Phase A Sweeps (this branch)

About 25 experiments on BindingDB PDSPKi:

**Full runs (30 epochs, 21k rows):** baseline, repro_baseline, bs64, dm64, do03 (5 runs)

**Fast runs (15 epochs, 10k rows on A100):** LR sweep, d_model sweep, depth sweep, batch size sweep, dropout sweep, heads sweep, 3-seed sanity, cold_drug split, cold_target split (about 19 runs)

Total GPU-hours spent: roughly 10 to 15.

### HPC Infrastructure

- `hpc_late_concat/setup_venv.sh` — one-time venv install
- `hpc_late_concat/run_late_concat_sweep.sbatch` — single L4 GPU template
- `hpc_late_concat/run_late_concat_a100.sbatch` — single A100 template (full epochs)
- `hpc_late_concat/run_lc_a100_fast.sbatch` — fast A100 template (15 epochs, 10k rows)
- `hpc_late_concat/phase_a_sweep.sh` — submits the 22-run sweep

---

## Best Result

- val_MSE = **1.231** (defaults, seed 42)
- val_CI = approximately 0.68 (random baseline = 0.50)
- val_Pearson = approximately 0.55 at convergence
- Achieved on 1x L4, batch_size=64

---

## Current Best Config

| Hyperparameter | Best value | Notes |
|----------------|------------|-------|
| d_model | 128 | bigger (256) shows promise on fast — confirm on full |
| n_heads | 4 | 2 slightly better on fast, but within noise |
| n_layers | 6 | 4 worse, 8 no improvement |
| d_ff | 512 (4x d_model) | not swept |
| Dropout | 0.1 | 0.2 / 0.3 clearly worse |
| Optimizer | AdamW | locked |
| Learning rate | 1e-4 default; 3e-4 may be better | needs full-run confirmation |
| Batch size | 64; 32 looks better on fast | needs full-run confirmation |
| Weight decay | 1e-5 | locked |
| Warmup | 5% | locked |
| Schedule | cosine | locked |

---

## Sensitivity Table (Phase A Deliverable)

| Hyperparameter | Sensitivity | Best zone | Worst-case delta MSE |
|----------------|-------------|-----------|----------------------|
| Learning rate | HIGH | 1e-4 to 3e-4 | +0.25 (lr=5e-5) |
| d_model | HIGH | at least 128 (256 may help) | +0.16 (d=64) |
| Dropout | HIGH | at most 0.1 | +0.23 (do=0.3) |
| Split type | HIGH | random | +0.50 (cold_target) |
| Batch size | MEDIUM | 32 to 64 (smaller better) | +0.04 (bs=128) |
| n_layers | LOW | 6 | +0.08 (l=4) |
| n_heads | LOW | 2 to 4 | +0.05 (h=8) |
| Seed variance | n/a | sigma approximately 0.07 | min/max swing 0.16 |

**Critical:** any improvement smaller than approximately 0.10 MSE is within seed-variance noise. Phase B and Phase C must allow for this noise floor.

---

## What's NOT Done (project-wide)

### Phase A for the other 3 variants

- **V1 Early Concat (Lingwei):** sbatch script committed to repo; no results yet
- **V2 Early Cross-Attn (Manas):** not started
- **V3 Late Cross-Attn (Tenzin):** baseline only (1.295), no sweep

### Phase B — Fair Config Negotiation

Requires all 4 sensitivity tables. Team meeting decides shared d_model, n_layers, lr, dropout, batch size. Estimated 1-hour discussion.

### Phase C — Final Controlled Runs

Once fair config locked: 4 variants x 3 splits x 5 seeds = 60 runs minimum on full BindingDB plus Davis plus KIBA. About 75 GPU-hours per teammate.

### Phase D — Deep Analysis (centerpiece of the report)

Six categories:
- A. Information flow (attention entropy, mixing point)
- B. Representation geometry (CKA, probing, t-SNE)
- C. Causal interventions (head ablation, layer ablation)
- D. Biological validation (binding site recovery)
- E. Failure modes (error stratification)
- F. Training dynamics (loss curves, attention emergence)

Not started.

### Extensions (optional)

GNN drug encoder, ESM-2 protein encoder, agentic optimization loop. Skip unless time allows.

---

## How a Teammate Replicates This Sweep for Their Variant

```bash
# 1. Pull this branch
git fetch
git checkout bhavesh/late-concat-phase-a

# 2. Copy templates and adapt for your variant
cp -r hpc_late_concat hpc_YOUR_VARIANT

# 3. Replace variant references
sed -i 's/late_concat/YOUR_VARIANT/g' hpc_YOUR_VARIANT/run_late_concat_sweep.sbatch
sed -i 's/late_concat/YOUR_VARIANT/g' hpc_YOUR_VARIANT/run_late_concat_a100.sbatch
sed -i 's/late_concat/YOUR_VARIANT/g' hpc_YOUR_VARIANT/run_lc_a100_fast.sbatch
sed -i 's/late_concat/YOUR_VARIANT/g' hpc_YOUR_VARIANT/phase_a_sweep.sh

# 4. Submit the sweep
bash hpc_YOUR_VARIANT/phase_a_sweep.sh
```

Replace `YOUR_VARIANT` with `early_concat`, `early_crossattn`, or `late_crossattn`.

---

## Cluster Gotchas Learned the Hard Way

| Issue | Fix |
|-------|-----|
| L4 partition all `idle~` (cold-bursting) | Use `c12m85-a100-1` (single A100), usually has warm `idle%` nodes |
| 22 jobs queued, 0 running, reason=Priority | Switch partition; cold-spin with low priority can take 8+ hours |
| Same MSE for many runs | Reproducibility, not bug — defaults producing identical seed=42 results |
| MSE differences below 0.10 | Likely seed-variance noise, not a real signal |
| GitHub HTTPS push: 403 Permission denied | Use SSH key auth, not password (passwords disabled by GitHub) |
| Heredoc paste mangles in terminal | Use OOD file upload or `nano` for multi-line content |

---

## Next Steps

### Immediate (today / tomorrow)

1. Wait for any in-flight runs to finish
2. Verify `PHASE_A_LEADERBOARD.csv` is clean
3. Open PR to main with this STATUS.md and results

### This Week

4. Ping team: V1, V2, V3 owners need to start their Phase A sweeps
5. Share `hpc_late_concat/` templates with them
6. Schedule Phase B meeting for about 1 week out

### Phase B Meeting (next week)

7. Each owner brings their sensitivity table
8. Negotiate fair config (rules: pick from acceptable zone for all variants; total params within 10%)
9. Lock `configs/fair_comparison.yaml`

### Phase C (week of May 11)

10. Run all 4 variants with fair config across 3 splits x 5 seeds x full BindingDB
11. Distribute load across teammates' HPC quotas

### Phase D (week of May 18)

12. Set up analysis pipeline under `src/analysis/`
13. Run all 6 analysis categories
14. Generate figures

### Writing (week of May 25)

15. Draft paper with focus on the deep analysis story
16. Polish figures
17. Final review

---

## Files in This Branch

```
ComparisionPDI/
├── STATUS.md                           <- this file
├── PHASE_A_LEADERBOARD.csv             <- consolidated results
├── hpc_late_concat/
│   ├── setup_venv.sh
│   ├── run_late_concat_sweep.sbatch
│   ├── run_late_concat_a100.sbatch
│   ├── run_lc_a100_fast.sbatch
│   └── phase_a_sweep.sh
└── outputs/sweeps/
    ├── baseline/, repro_baseline/, bs64/, dm64/, do03/  <- full runs
    └── *_fast/                                           <- A100 fast runs
```

Each sweep folder contains:
- `checkpoints/late_concat/best_model.pt`
- `logs/late_concat_history.csv`
- `results/results.csv`

---

## Contact

If anything in this branch is unclear, ping me (Bhavesh) on the team chat.
