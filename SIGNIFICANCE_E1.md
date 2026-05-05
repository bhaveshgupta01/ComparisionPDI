# Phase E1 — Statistical significance

Paired-seed t-test (n=5 seeds, identical across all variants).
Each cell = p-value for H0: variant-row MSE = variant-col MSE; the
**sign** of the mean difference (row − col) is shown alongside p so
row-wins-vs-col is easy to read.

## Random

| | V1 EC | V2 EX | V3 LX | V4 LC |
|---|---|---|---|---|
| V1 EC | — | **p=0.008 (↑0.040)** | **p=0.004 (↑0.077)** | **p=0.010 (↑0.071)** |
| V2 EX | **p=0.008 (↓0.040)** | — | **p=0.010 (↑0.037)** | **p=0.047 (↑0.031)** |
| V3 LX | **p=0.004 (↓0.077)** | **p=0.010 (↓0.037)** | — | p=0.385 (↓0.006) |
| V4 LC | **p=0.010 (↓0.071)** | **p=0.047 (↓0.031)** | p=0.385 (↑0.006) | — |

Interpretation: *row vs col*. ↓ means row's mean MSE is lower (better). **Bold** = p<0.05.

## Cold-Drug

| | V1 EC | V2 EX | V3 LX | V4 LC |
|---|---|---|---|---|
| V1 EC | — | p=0.775 (↓0.013) | p=0.293 (↑0.045) | p=0.059 (↑0.083) |
| V2 EX | p=0.775 (↑0.013) | — | p=0.082 (↑0.058) | p=0.076 (↑0.095) |
| V3 LX | p=0.293 (↓0.045) | p=0.082 (↓0.058) | — | p=0.424 (↑0.037) |
| V4 LC | p=0.059 (↓0.083) | p=0.076 (↓0.095) | p=0.424 (↓0.037) | — |

Interpretation: *row vs col*. ↓ means row's mean MSE is lower (better). **Bold** = p<0.05.

## Cold-Target

| | V1 EC | V2 EX | V3 LX | V4 LC |
|---|---|---|---|---|
| V1 EC | — | p=0.682 (↑0.023) | p=0.136 (↑0.072) | p=0.494 (↑0.057) |
| V2 EX | p=0.682 (↓0.023) | — | p=0.111 (↑0.049) | p=0.314 (↑0.033) |
| V3 LX | p=0.136 (↓0.072) | p=0.111 (↓0.049) | — | p=0.733 (↓0.016) |
| V4 LC | p=0.494 (↓0.057) | p=0.314 (↓0.033) | p=0.733 (↑0.016) | — |

Interpretation: *row vs col*. ↓ means row's mean MSE is lower (better). **Bold** = p<0.05.

## Headline significance results

- **V3 vs V2 on random (the reversal)**: mean Δ = -0.0368, t=-4.560, p=0.0103. Winner: **V3 LX** (p < 0.05).
- **V3 vs V2 on cold-target**: mean Δ = -0.0489, t=-2.037, p=0.1113. Winner: **V3 LX** (p ≥ 0.05).
- **V4 vs V3 on cold-drug (where V4 wins)**: mean Δ = -0.0374, t=-0.889, p=0.4241. Winner: **V4 LC** (p ≥ 0.05).
- **V4 vs V1 on random (concat-only ranking)**: mean Δ = -0.0714, t=-4.568, p=0.0103. Winner: **V4 LC** (p < 0.05).
- **V3 vs V4 on random (best two)**: mean Δ = -0.0056, t=-0.974, p=0.3850. Winner: **V3 LX** (p ≥ 0.05).
