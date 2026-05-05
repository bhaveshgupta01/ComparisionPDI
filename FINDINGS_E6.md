# Phase E6 — Width-vs-depth decomposition

**Test:** keep model parameters roughly equal between (d=128, n_layers=12) [E6] and
(d=256, n_layers=6) [E1]. Both add capacity vs Phase C. Does doubling *either* axis
trigger the early-vs-late ranking reversal, or is the reversal width-specific?

## Random-split val MSE (mean ± std, 3 seeds)

| Variant | Phase C (d=128, n=6/3) | E6 depth × 2 (d=128, n=12/6) | E1 width × 2 (d=256, n=6/3) |
|---|---|---|---|
| V1 EC | 1.004 ± 0.031 | 0.988 ± 0.025 | 0.824 ± 0.031 |
| V2 EX | 0.948 ± 0.023 | 0.910 ± 0.020 | 0.784 ± 0.038 |
| V3 LX | 1.030 ± 0.039 | 0.949 ± 0.041 | 0.747 ± 0.030 |
| V4 LC | 1.119 ± 0.018 | 0.971 ± 0.034 | 0.753 ± 0.032 |

## Winners
- Phase C (d=128, n=6/3):  **V2 EX**
- E6 (d=128, n=12/6):       **V2 EX**
- E1 (d=256, n=6/3):        **V3 LX**

## Per-variant Δ from each axis

| Variant | Δ from depth × 2 (E6−C) | Δ from width × 2 (E1−C) | Δ-ratio (depth/width) |
|---|---|---|---|
| V1 EC | -0.016 | -0.180 | 0.09 |
| V2 EX | -0.038 | -0.164 | 0.23 |
| V3 LX | -0.081 | -0.283 | 0.29 |
| V4 LC | -0.148 | -0.366 | 0.40 |

## Interpretation

**Width-specific reversal.** The early-fusion winner (V2 EX) is preserved after depth-doubling at d=128 (V2 EX still wins) but flips after width-doubling at n=6/3 (V3 LX now wins). This isolates the reversal to the *width* axis: deeper-but-narrow models do not produce the late-fusion advantage observed at d=256.

## Late vs early benefit, per axis

- Depth × 2 → avg Δ MSE: **early variants -0.027**, **late variants -0.114**
- Width × 2 → avg Δ MSE: **early variants -0.172**, **late variants -0.325**

Late variants benefit *more* from width than from depth, and the late-vs-early asymmetry is larger for width. **Width is the more impactful axis.**
