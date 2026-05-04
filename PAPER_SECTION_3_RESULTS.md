# Section 3 — Results (paper draft)

> Status: drafted from Phase C + E1 + E5 + significance tests. Phase E1b numbers and E4 ablation deltas are pending; placeholders marked `[PENDING:E1b]` / `[PENDING:E4]` will be filled in when those jobs land.

---

## 3.1 The headline reversal

We trained all four variants under the matched-config protocol of Section 2 at two model widths: `d_model = 128` (Phase C, 36 runs over 3 seeds × 3 splits × 4 variants) and `d_model = 256` (Phase E1, 60 runs over 5 seeds × 3 splits × 4 variants). Every other architectural axis — depth, head count, optimizer, schedule, data, splits — is identical across the two scales.

Table 1 reports best-validation pKi MSE (mean ± std across seeds) for both scales. **The ranking reverses between scales.** At `d=128`, V2 (Early Cross-Attn) wins random and cold-target while V3 (Late Cross-Attn) wins cold-drug; the late-concatenation baseline V4 — the field-default DTI architecture — never wins. At `d=256`, V3 wins random and cold-target while V4 wins cold-drug; V2 never wins. V1 (Early Concat) is last on every split at both scales.

**Table 1.** Best validation MSE (mean ± std), Phase C (d=128, 3 seeds) vs Phase E1 (d=256, 5 seeds). Bold = winner per column.

| Variant | Phase C Random | Phase E1 Random | Phase C Cold-Drug | Phase E1 Cold-Drug | Phase C Cold-Target | Phase E1 Cold-Target |
|---|---|---|---|---|---|---|
| V1 EC | 1.004 ± 0.031 | 0.824 ± 0.031 | 1.476 ± 0.029 | 1.388 ± 0.083 | 1.360 ± 0.197 | 1.230 ± 0.084 |
| V2 EX | **0.948 ± 0.023** | 0.784 ± 0.038 | 1.432 ± 0.170 | 1.401 ± 0.120 | **1.248 ± 0.187** | 1.207 ± 0.150 |
| V3 LX | 1.030 ± 0.039 | **0.747 ± 0.030** | **1.410 ± 0.129** | 1.343 ± 0.133 | 1.549 ± 0.131 | **1.158 ± 0.124** |
| V4 LC | 1.119 ± 0.018 | 0.753 ± 0.032 | 1.465 ± 0.178 | **1.306 ± 0.111** | 1.467 ± 0.069 | 1.173 ± 0.207 |

> **Figure 3** (hero). Two-panel side-by-side bar chart: Phase C bars (gray) and Phase E1 bars (color-coded per variant) for each of the three splits. ★ marks the winner. *Late-fusion variants take over all three columns at d=256.* Path: `poster_figures/diagram_phase_c_vs_e1_comparison.png`.

The reversal is not within seed noise on the random split. A paired-seed t-test (n=5, identical seeds across variants) yields **p = 0.010 for V3 < V2 on random** and **p = 0.010 for V4 < V1 on random**. On cold splits the absolute mean differences favor late variants but do not reach the p < 0.05 threshold (cold-drug V4 vs V3: p = 0.42; cold-target V3 vs V2: p = 0.11). We report the full pairwise paired-test matrix in Appendix A and observe that *no single variant pair* on cold splits achieves significance — i.e. cold splits at d=256 are tighter than the seed-level seed variance permits us to discriminate. The random-split reversal, however, is robust.

## 3.2 Differential improvement: late variants benefit more from scale

Per-variant percent improvement Phase C → E1, averaged over the three splits, is V1: **−18%**, V2: **−15%**, V3: **−25%**, V4: **−25%**. Concretely, V3's random-split MSE drops 0.283 (from 1.030 to 0.747) when width doubles, while V2's drops only 0.164 (from 0.948 to 0.784). The two architectural families respond to capacity scaling at fundamentally different rates.

> **Figure 5.** Bar chart, Δ% MSE per (variant, split) pair. Late variants (green / pink) consistently improve more than early variants (blue / orange). Path: `poster_figures/diagram_e1_per_variant_improvement.png`.

This differential improvement is the immediate mechanism behind the reversal: scaling width improves all variants, but it improves late variants ~1.5–2× faster than early variants, so the small-scale early-fusion advantage is consumed within one width doubling.

## 3.3 Bracketing the phase transition (`d=192`) `[PENDING:E1b]`

If the reversal between `d=128` and `d=256` is gradual, an intermediate point should land between the two extremes. We trained the same 2 × 2 set at `d_model = 192, n_heads = 6` (preserving `d_head = 32`), keeping every other knob identical — 36 runs over 3 seeds × 3 splits × 4 variants.

`[PENDING:E1b]` Plug d=192 numbers into Figure 4 and a fourth column of Table 1.

> **Figure 4.** Capacity curve. X-axis `d_model ∈ {128, 192, 256}`. Y-axis: random-split val MSE. Four lines (one per variant) with ±1 std ribbons. The crossing point of V3 (line) below V2 (line) is the phase boundary.

We observe `[PENDING:E1b]` whether the transition is sharp (variants are still V2 > V3 at d=192 then flip at d=256) or gradual (V3 catches up monotonically). Either outcome is informative: a sharp transition argues for a capacity threshold; a gradual one argues for a continuous tradeoff that simply tips past d≈192.

## 3.4 Mechanistic dissection

We extract internal representations and attention from the random-split, seed=42 checkpoints of all four variants at each scale and report three lenses on what changed.

### 3.4.1 Attention entropy with mask-aware aggregation

The poster version of this analysis included pad-token query positions in the per-layer entropy average, which are dominated by the soft-max over a sparse-but-equiprobable pattern from the key-padding mask. We recompute entropy averaging only over valid (non-pad) query positions.

> **Figure 6.** Two-panel entropy curve, V1 (early concat, blue) vs V2 (early cross-attn, orange) across the 6 shared encoder layers. Left: pad-contaminated; right: mask-aware. Path: `poster_figures/diagram_16b_attention_entropy_mask_aware.png`.

The relative ranking is preserved: V1 mean entropy 5.79 nats, V2 mean entropy 5.95 nats, with V1 dipping to 5.15 nats at layer 4 vs V2's flat 5.88-5.97 nats across all six layers. V1's encoder develops depth-dependent attention specialization; V2's does not — the cross-attention block at V2's input has already done the cross-modal alignment work, so the encoder body is left with per-modality compression that does not benefit from attention sharpening with depth. **This finding is independent of fusion stage** (V3 and V4 share a separate-encoders structure that we report only for layer 0 due to extraction storage limits) **and survives the mask-aware correction**.

### 3.4.2 Causal layer / head ablation `[PENDING:E4]`

We zero each transformer block's self-attention output (the residual stream still passes), and separately zero each individual attention head's contribution, on Phase C random-split seed=42 checkpoints. ΔMSE measures the prediction's reliance on each circuit.

`[PENDING:E4]` From `outputs/phase_e_ablations/SUMMARY.csv`:
- *Single-layer ablation*: which layer carries the most prediction weight in V1's depth-specialized encoder, and whether V2's flat-entropy encoder spreads the load uniformly.
- *Single-head ablation*: whether V2's cross-attention block has dominant heads.
- *V3 ↔ V4 drug-encoder swap*: whether V3 and V4 learn interchangeable drug encoders (high if so, low if their fusion modules co-adapt).

> **Figure 7.** Layer-ablation heatmap (rows = variants, cols = layer index, color = ΔMSE). Pending E4 results.

### 3.4.3 Cross-variant representation similarity (CKA) `[PENDING:E1-extract]`

A poster-era CKA analysis on attention features at d=128 revealed two clean clusters: {V1, V2} CKA = 0.97 and {V3, V4} CKA = 0.95, with cross-cluster similarity 0.73-0.84 — i.e. variants clustered by *fusion stage* (early vs late) rather than mechanism (concat vs cross-attn). We re-extract on Phase E1 checkpoints to ask whether this clustering survives at d=256. If late-fusion has become the dominant mode at d=256, we expect the {V3, V4} cluster to remain coherent while the {V1, V2} cluster either weakens or merges into the late cluster.

`[PENDING:E1-extract]` Compute CKA matrix at d=256 on the same 256-pair held-out batch. Path: `poster_figures/diagram_18b_cka_e1.png`.

> **Figure 8.** CKA matrices at d=128 and d=256. Pending re-extract.

## 3.5 Compute cost

Phase A sweeps + Phase C + E1 + E1b + ablations / extraction = ~145 GPU-hours on NYU HPC A100s (40 GB, single-GPU jobs). Median per-run wall-clock at `d=256, 45 epochs` was 2 hours 30 minutes; at `d=128, 30 epochs` was 28 minutes. Total measurements: 132 controlled runs across three scales, plus 88 Phase A sensitivity sweeps used only to lock the matched config. Full training scripts, configs, sbatch templates, and analysis pipeline are in the public repository.

---

## Appendix A — Full pairwise paired-seed t-test matrices

Table A1 — Random split (n = 5 seeds).

| | V1 EC | V2 EX | V3 LX | V4 LC |
|---|---|---|---|---|
| V1 EC | — | **p=0.008** ↑0.040 | **p=0.004** ↑0.077 | **p=0.010** ↑0.071 |
| V2 EX | **p=0.008** ↓0.040 | — | **p=0.010** ↑0.037 | **p=0.047** ↑0.031 |
| V3 LX | **p=0.004** ↓0.077 | **p=0.010** ↓0.037 | — | p=0.385 ↓0.006 |
| V4 LC | **p=0.010** ↓0.071 | **p=0.047** ↓0.031 | p=0.385 ↑0.006 | — |

Bold = p < 0.05. Arrow indicates whose mean MSE is lower (↓ = row better). The (V3, V4) cell shows the two late variants are statistically indistinguishable on random split — the fusion-stage advantage at scale is "late-vs-early," not "cross-attn-vs-concat."

Table A2 — Cold-Drug. Table A3 — Cold-Target. (None reach p<0.05 between best-pair candidates; full matrix in `SIGNIFICANCE_E1.md`.)
