# Phase B Meeting — Fair-Config Negotiation

**Goal:** lock a single hyperparameter set that all 4 variants will use for Phase C controlled runs. The fair config must sit inside *every* variant's "acceptable performance" zone — no variant gets to be sandbagged, none gets to be hand-tuned.

**Duration:** ~45 min. **Attendees:** Lingwei (V1), Manas (V2), Tenzin (V3), Bhavesh (V4).

**Pre-read (5 min):** [PHASE_A_4VARIANT_COMPARISON.csv](PHASE_A_4VARIANT_COMPARISON.csv) and the 4-variant figures in [poster_figures/](poster_figures/) — especially [diagram_15_sensitivity_4variant](poster_figures/diagram_15_sensitivity_4variant.png), [diagram_39_4variant_leaderboard](poster_figures/diagram_39_4variant_leaderboard.png), [diagram_40_sensitivity_4variant](poster_figures/diagram_40_sensitivity_4variant.png).

---

## 1. Phase A recap (5 min) — Bhavesh leads

Quick walk through best val MSE per variant (fast mode, 15 epochs, 10k pairs):

| Variant | Best val MSE | Best config |
|---|---|---|
| V1 Early Concat | 1.4395 | seed=123 |
| V2 Early X-Attn | 1.2882 | d_model=256, bs=16 |
| V3 Late X-Attn | 1.2637 | d_model=256, bs=16 |
| V4 Late Concat | 1.231 | full-mode baseline (1.4414 in fast mode, seed=123) |

**Caveat:** seed-to-seed variance is ≈ 0.10–0.15 MSE. Any variant gap smaller than that is noise, not signal.

---

## 2. Constraints we're locking in (10 min)

**Discussion points — vote on each:**

| # | Question | Default proposal | Rationale |
|---|---|---|---|
| 1 | d_model | **128** | 256 OOMs the cross-attn variants at default batch size; 128 is the only value all 4 ran cleanly |
| 2 | n_layers | **6** (V1, V2 shared encoder) **/ 3 per side** (V3, V4 dual encoder) | Already the protocol — keeps total depth comparable |
| 3 | n_heads | **4** | All 4 ran clean at h=4; h=8 needs bs=32 for cross-attn variants |
| 4 | batch size | **64** | Default everyone ran. bs=32 is the only fallback if Phase C runs go OOM |
| 5 | learning rate | **3e-4** | Best for 3 of 4 variants in Phase A (1.43 / 1.47 / 1.43 / 1.44). 1e-4 is a safer fallback |
| 6 | dropout | **0.1** | Higher dropout broke V1 (0.3→2.36 MSE); 0.1 is the only value that's safe everywhere |
| 7 | optimizer | **AdamW + cosine LR + warmup** | Already shared scaffolding |
| 8 | warmup steps | **500** | Tenzin's default; keep |
| 9 | epochs | **30** (Phase C, full data) | Up from Phase A's 15 since we're using full BindingDB |
| 10 | seeds | **{42, 123, 456}** | Three seeds matches our pre-registered plan. Five is nice, three is sufficient |
| 11 | max_rows | **None (full BindingDB ~52k pairs)** | Phase A used 10k for speed — Phase C uses full data |

**Open question to settle live:** lr=3e-4 vs lr=1e-4. 3e-4 wins on Phase A but is closer to the divergence cliff. Do we want speed or safety?

---

## 3. Phase C run protocol (10 min)

Once config is locked, each owner runs:
- **3 splits** × **3 seeds** = 9 runs per variant. 4 variants = **36 runs total**.
- Compute estimate per run: full BindingDB at lr=3e-4, 30 epochs ≈ 25–35 min on A100. Total ≈ 18–21 GPU-hours across the team. Comfortably inside our remaining ~280 GPU-hour budget.

**Action items:**
- Each owner copies their variant's `_fast` sbatch template into a new `_phaseC` template, removes `--max_rows`, sets `--epochs 30`, parameterizes `--seed` and `--split`.
- Decide who runs which (suggested): each owner runs their own variant, all 9 configs. Splits as a wrapper script over the 3 seeds × 3 splits.
- Bhavesh shares his `sweep_v*.sh` wrapper format so the others can adapt.

---

## 4. Davis & KIBA — go / no-go (5 min)

POSTER.md lists Davis (442 × 379 kinases) and KIBA (2116 × 229) as "optional Phase C extensions."

**Recommendation:** punt to Phase C+. Run BindingDB Phase C first. If we have spare GPU-hours and one week before the poster deadline, add Davis + KIBA as a robustness check.

**Vote:** include now or punt?

---

## 5. Phase D division of labor (10 min)

Six analysis axes; one owner per axis or shared?

| Axis | Outputs needed | Lead |
|---|---|---|
| A — Information Flow | attention entropy curves, mixing-point analysis | `[?]` |
| B — Representation Geometry | CKA matrix, t-SNE, probing classifiers | `[?]` |
| C — Causal Interventions | head ablation, layer ablation, repr swap | `[?]` |
| D — Biological Validation | binding-site Precision@K, IG attribution | `[?]` |
| E — Failure Modes | error vs Tanimoto distance, calibration | `[?]` |
| F — Training Dynamics | loss landscape, repr evolution | `[?]` |

**Bhavesh's proposal:**
- Bhavesh → A + B (extraction script already runs; CKA / attention ready to plot once 176079 finishes)
- Tenzin → C (he wrote the encoder scaffolding, knows where to ablate)
- Manas → D (highest visual impact for poster; needs PDBbind + structure rendering)
- Lingwei → E + F (clean stats, less moving parts)

**Open: agree, swap, or split differently?**

---

## 6. Poster figure ownership (5 min)

POSTER.md lists 31 planned figures. We'll print maybe ~12 strongest. Bhavesh has built 16 conceptual + Phase A figures already (in [poster_figures/](poster_figures/)).

**Per-axis figure budget for Phase D:**
- Axis A: 1 entropy curve, 1 attention heatmap → owner builds
- Axis B: 1 CKA matrix, 1 t-SNE → owner builds
- Axis C: 1 ablation grid, 1 swap bar chart → owner builds
- Axis D: 1 structure overlay, 1 IG heatmap, 1 Precision@K curve → owner builds
- Axis E: 1 error-vs-distance, 1 calibration → owner builds
- Axis F: 1 representation-evolution → owner builds

Each owner commits their figures to `poster_figures/` on a branch by the agreed Phase D deadline.

---

## 7. Decisions & timeline (5 min)

End the meeting with explicit answers to:

- [ ] Locked Phase C config (single YAML committed to repo as `configs/phase_c_fair.yaml`)
- [ ] Each owner has their Phase C sbatch template by `[date + 2 days]`
- [ ] Phase C runs complete by `[date]`
- [ ] Davis/KIBA: in or out
- [ ] Phase D axis assignments
- [ ] Phase D deadline (recommended: 1 week before poster deadline)

---

## Side notes (for the meeting)

- **GPU budget left:** ~280 hours per student. Phase C ≈ 18–21 GPU-hours total. Lots of headroom.
- **Don't argue about "best" configs** — Phase A already showed seed variance dominates the fast-mode rankings. Pick a *defensible* fair config, not an *optimal* one.
- **If V1 owner pushes back on lr=3e-4** — V1 is the worst-affected by high LR (only 0.18 MSE gap to V2) but still wins at 3e-4. Acceptable.
- **If V3 owner wants d_model=256** — note it OOMs everyone else at default batch size; only feasible if we drop bs to 16, which inflates wall-clock 4×. Vote it down or accept the slowdown.
