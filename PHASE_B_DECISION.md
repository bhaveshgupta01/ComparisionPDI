# Phase B — Locked Fair Config (2026-04-28)

> The locked configuration below is the single source of truth for all Phase C runs. It was derived from Phase A's 88 sweeps across all four variants and represents the hyperparameter set inside every variant's acceptable performance zone.

## The locked config

```yaml
# configs/phase_c_fair.yaml
d_model: 128
n_heads: 4
d_ff:   512
batch_size: 64
lr:     3e-4
dropout: 0.1
warmup_steps: 500
epochs: 30
optimizer: adamw
lr_schedule: cosine
n_layers_shared:   6   # V1, V2
n_layers_per_side: 3   # V3, V4
```

Sweep dimensions (the only things that vary across runs):
- variants  ∈ {early_concat, early_crossattn, late_crossattn, late_concat}
- splits    ∈ {random, cold_drug, cold_target}
- seeds     ∈ {42, 123, 456}

Total **36 Phase C runs**.

## Why these values (Phase A evidence)

| Knob | Value | Phase A evidence |
|---|---|---|
| `lr` | **3e-4** | Wins 3 of 4 variants (V1 1.50, V2 1.47, V3 1.43, V4 1.44). 1e-4 was 0.10–0.15 MSE worse — outside seed-noise range. |
| `d_model` | **128** | Only size all 4 ran without OOM. 256 needs bs=16 for cross-attn variants. |
| `n_heads` | **4** | Only h-value all 4 ran cleanly. h=8 needs bs=32 for cross-attn. |
| `batch_size` | **64** | Everyone's default. Drops to 32 OOM fallback if needed. |
| `dropout` | **0.1** | 0.2 hurt all variants (+0.1 MSE); 0.3 broke V1 entirely (→ 2.36). |
| `n_layers` | 6 (shared) / 3 (per side) | Matches the team's original architectural design — 3+3=6 total blocks for late variants matches V1/V2's shared 6. |
| `epochs` | **30** | Up from Phase A's 15 since we're using full BindingDB instead of the 10k subset. |
| `early_stop_patience` | 5 | Cuts wasted compute on bad seed-split combos. |

## Why a fair-config lock works without per-variant tuning

1. Phase A data shows clear signal — for every knob, the Phase A best is shared by ≥3 variants. The hyperparameter optima largely agree across variants.
2. Decisions are reversible — if Phase C reveals the config is sub-optimal for a specific variant, that's a finding to report (variant *requires* a different config), not a process failure.
3. Locking one config is essential for the controlled-comparison framing: any cross-variant gap we report is then attributable to architecture, not hyperparameter tuning.

## What happens after Phase C

When all 36 runs finish (~6–10 hours wall-clock with the A100 queue), each `outputs/phase_c/<tag>/results/results.csv` contains:
- best val MSE
- best val CI (concordance index)
- best epoch
- final test MSE on the variant's split

These feed directly into:
- **diagram_10** (best MSE per variant) → upgrade to mean ± std over 3 seeds × 3 splits
- **diagram_11** (CI per variant per split, NEW) → grouped bars, error bars from seeds
- **diagram_12** (train/val loss curves, NEW) → 4 lines, single split
- **diagram_13** (predicted-vs-true scatter, 4-panel, NEW) → from saved test predictions

After Phase C, Phase D re-runs (with bugfix) on the new checkpoints to populate diagrams 16–30.

## Acceptance criteria

A Phase C run "succeeds" if:
1. `results.csv` exists and has a non-NaN best val MSE.
2. Best val MSE on random split is within ±0.30 of the variant's Phase A best (sanity check).
3. Test MSE on random split is within +0.20 of best val MSE (no overfit catastrophe).

If 2 or 3 fail, mark the run as "investigate" and re-run with logged config.
