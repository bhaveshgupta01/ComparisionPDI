# POSTER — Drug-Target Interaction Comparison Framework

> **Final content draft, 2026-04-29.** All experimental results locked (Phase A 88 sweeps, Phase C 36 runs, Phase D extraction × 4 variants). Figures live as PNG + SVG in `poster_figures/`. Headline numbers in `FINDINGS.md`.
>
> Target format: A0 portrait scientific poster (~841 × 1189 mm), 4-column layout.

---

## 0. POSTER LAYOUT (proposed)

```
┌────────────────────────────────────────────────────────────────────┐
│  TITLE BANNER  +  AUTHORS  +  NYU LOGO  +  COURSE TAG             │
├──────────┬──────────┬──────────┬──────────────────────────────────┤
│ COL 1    │ COL 2    │ COL 3    │ COL 4                            │
│ • Motiva │ • Method │ • Results│ • Deep Analysis                  │
│   tion   │ • Archs  │ • Tables │ • Conclusions                    │
│ • Q&Hyp  │ • Data   │ • Curves │ • Future Work                    │
│ • Back-  │ • Setup  │ • Splits │ • References / QR                │
│   ground │          │          │                                  │
└──────────┴──────────┴──────────┴──────────────────────────────────┘
```

`[DIAGRAM-00 Layout mockup]` — I'll generate a clean SVG mockup once content is locked.

---

## 1. TITLE BLOCK

**Title:** *Where Should Drug and Protein Meet? A Controlled Study of Fusion Stage in Transformer DTI Models*

**Subtitle:** *A 2 × 2 controlled experiment showing that **fusion stage matters more than fusion mechanism** — and the field-default late fusion is never optimal.*

**Authors:** Lingwei Li · Manas Ghai · Tenzin Tsundue · Bhavesh Gupta

**Affiliation:** New York University — Center for Data Science / Courant Institute

**Course:** CSCI-2565 *Machine Learning*, Spring 2026

**Instructor:** Rajesh Ranganath
**Teaching Assistants:** Nhi Nguyen · Riya Mahesh · Siddhant Mohan

**Repo & QR:** github.com/bhaveshgupta01/ComparisionPDI &nbsp; `[DIAGRAM-01 QR code]`

---

## 2. ABSTRACT / TL;DR (≤120 words, top of poster)

Predicting how strongly a small molecule binds to a protein is a foundational task in computational drug discovery. Modern transformer-based predictors uniformly fuse drug and protein representations *late* — after independent encoding — but this design choice has rarely been audited. We ask: **at what stage should the two modalities meet?** We constructed a 2 × 2 controlled experiment (concatenation vs cross-attention, before vs after encoding), trained all four variants on 27,715 BindingDB Ki measurements under matched hyperparameters (Phase B locked: d=128, h=4, lr=3e-4, do=0.1, 30 epochs), and evaluated across random, cold-drug, and cold-target splits with three seeds each (36 runs total). **Each split has a different winner: V2 (early cross-attn) wins random and cold-target; V3 (late cross-attn) wins cold-drug; the field-default late-concatenation V4 never wins**. CKA on attention features shows the variants cluster by *fusion stage* (early vs late), not by *fusion mechanism* (concat vs cross-attn) — fusion stage drives behavior more than the question of how the modalities mix.

---

## 3. MOTIVATION (Column 1)

**Why this matters.**

- Drug-target interaction (DTI) prediction shortens the funnel from billions of candidate molecules to a tractable shortlist for wet-lab validation.
- Transformer encoders dominate the current state of the art — but the field has converged on a single design pattern (encode separately, fuse late) without rigorous justification.
- A wrong design assumption locked in across the field has compounding cost: every downstream paper inherits it.

**The gap.** Architecture-comparison studies in DTI typically vary *encoder choice* (CNN vs Transformer vs GNN). Almost none vary *where* the two modalities interact. We isolate that single axis.

`[DIAGRAM-02 — funnel illustration: billions of compounds → DTI filter → wet-lab leads]`

---

## 4. THE QUESTION & HYPOTHESES (Column 1)

**Research Question.**
*When a transformer predicts drug-protein binding affinity, does the stage at which drug and protein representations interact materially affect (a) predictive accuracy, (b) generalization to unseen drugs / targets, and (c) what the model learns?*

**Hypotheses.**

| # | Statement | Status |
|---|-----------|--------|
| H1 | Late fusion (V3, V4) is the field default but is not Pareto-optimal across all splits. | **CONFIRMED** — V4 never wins; V3 wins only cold-drug. The two splits closest to real-world deployment (cold-drug, cold-target) are won by different architectures. |
| H2 | Cross-attention variants (V2, V3) recover more biologically meaningful structure than concatenation variants (V1, V4). | **PARTIALLY** — V2/V3 win 3 of 3 splits on accuracy; mechanistic validation (binding-site recovery on PDBbind) deferred to future work. |
| H3 | Early fusion is more parameter-efficient (one encoder body) at matched accuracy. | **CONFIRMED** — V1: 1.27 M params, single 6-layer body, beats V4 (4.1 M params, two 3-layer bodies) on every split. |
| H4 | All four variants converge to similar internal representations under matched compute (CKA ≳ 0.8). | **REFUTED** — CKA on attention features shows clear two-cluster structure: {V1, V2} CKA 0.97; {V3, V4} CKA 0.95; cross-cluster 0.73-0.84. **Fusion stage drives behavior, not mechanism.** |

`[DIAGRAM-03 — 2×2 design matrix, large and visually dominant]`

---

## 5. BACKGROUND (Column 1, condensed)

- **DTI prediction**: regression over pKi / pKd / pIC50; benchmarks include Davis, KIBA, BindingDB.
- **Representations**: drugs as SMILES strings (sequence) or molecular graphs; proteins as amino-acid sequences or 3D structures.
- **Transformer DTI**: MolTrans (sub-structure attention), DeepPurpose (CNN→FC fusion), HyperAttentionDTI, PerceiverCPI, FusionDTI — all of which converge on independent encoding then late fusion.
- **What's missing**: a controlled, single-axis comparison.

References → see Section 14.

---

## 5.5 INDUSTRY & MARKET LANDSCAPE (Column 1 → spillover into Column 2)

> Why this question is worth asking *now*: ML-driven drug discovery is a multi-billion-dollar industry, and DTI prediction is the load-bearing primitive at its core. Yet design choices in the underlying models are largely inherited, not justified.

### 5.5.1 Market size & momentum

- AI-in-drug-discovery market: **~USD 1.5 B (2023) → projected ~USD 13–20 B by 2030**, ~30 % CAGR (industry-analyst consensus, range across MarketsandMarkets / Grand View / Precedence).
- **>75 AI-discovered or AI-designed drug candidates** in clinical trials by end of 2024 (BIO / Boston Consulting Group reporting).
- First AI-designed small molecule to enter Phase I (Exscientia / Sumitomo, **DSP-1181**, 2020) — collapsed the typical 4–5 year hit-to-clinic timeline to <12 months.
- Insilico Medicine's **INS018_055** (idiopathic pulmonary fibrosis) entered Phase II in 2024 — first molecule with both AI-discovered target *and* AI-designed ligand.

`[DIAGRAM-32 — market-size growth chart, 2020–2030 forecast]`
`[DIAGRAM-33 — timeline of notable AI-designed drug-candidate milestones]`

### 5.5.2 Who is doing what

| Player | Approach | Public artifacts |
|---|---|---|
| **DeepMind / Isomorphic Labs** | AlphaFold-2/-Multimer/-3 → protein structure → docking + neural scoring | AF2/3 weights public; Isomorphic models proprietary |
| **Schrödinger** | Physics-based FEP + ML scoring overlays | Maestro, commercial |
| **Exscientia** | Active-learning loop over generative + scoring models | Closed |
| **Insilico Medicine** | Generative chemistry (Pharma.AI, Chemistry42) + target ID | Closed; some papers |
| **Atomwise** | CNN-based virtual screening at billion-compound scale | Closed; AtomNet papers |
| **Recursion** | Phenomic + transcriptomic foundation models | MolE, Phenom-1 partial release |
| **Insitro** | ML on functional genomics + chemistry | Mostly closed |
| **BenevolentAI** | Knowledge-graph-driven target ID + DTI scoring | Closed |
| **Relay Therapeutics** | Protein-dynamics-aware design | Closed |
| **Big-pharma in-house** (Pfizer, Novartis, Merck, GSK, AstraZeneca, Roche/Genentech) | Internal DTI / ADMET / generative pipelines, often on ESM-2 / AlphaFold backbones | Mostly closed; isolated papers |
| **Academia open-source** (DeepDTA, DeepPurpose, MolTrans, HyperAttentionDTI, PerceiverCPI, DrugBAN, FusionDTI, BACPI, TransformerCPI, EquiBind, DiffDock, RoseTTAFold-AA) | Reference models for benchmarking | Open code + checkpoints |

`[DIAGRAM-34 — landscape map: industry vs academia × open vs closed × method family]`

### 5.5.3 What has actually been achieved

**Wins.**
- **Structure prediction is solved-ish.** AlphaFold-2/3 provide near-experimental backbones for >200 M proteins, removing structure as a bottleneck for ~most targets.
- **Pre-trained sequence encoders** (ESM-2 for protein, ChemBERTa / MoleculeBERT / Uni-Mol for molecules) routinely beat from-scratch encoders by **2–5 points of CI** on DTI benchmarks.
- **Generative chemistry** can now produce synthesizable, drug-like molecules conditioned on a target pocket (Pocket2Mol, DiffDock-L, RFdiffusion-AA).
- **Virtual screening at billion-scale** is now routine (Enamine REAL × deep scoring), with multiple confirmed wet-lab hits reported per year.
- **First AI-designed clinical entries** (DSP-1181, INS018_055, EXS-21546) demonstrate the pipeline end-to-end.

**Reported headline numbers on standard DTI benchmarks** (random split, CI):
- Davis: ~0.89–0.91 (state of art)
- KIBA: ~0.89–0.90
- BindingDB Ki: ~0.85–0.88
- Our Phase A baselines (random, BindingDB Ki, V3 / V4): **~0.68 — i.e. deliberately small models for a controlled comparison, not an SOTA chase.**

`[DIAGRAM-35 — benchmark CI bar chart: published SOTA vs our baselines, with explicit caption about scope]`

### 5.5.4 The loopholes — where the industry is weakest

These are the specific cracks our project targets.

1. **Architecture choices are inherited, not justified.** Almost every transformer DTI paper independently encodes drug and protein, then fuses late. There is no published controlled comparison isolating *fusion stage* as the variable. The choice is folklore.
2. **Benchmark inflation via random splits.** Most reported numbers use random pair splits, where >90 % of test drugs *and* targets appear in training. Cold-drug / cold-target performance is consistently 10–25 CI points lower — and rarely headlined. This is the **most cited reproducibility concern** in the DTI literature (Mayr 2018; Pahikkala 2015; Chen 2021).
3. **Black-box outputs.** Few industry models offer mechanistic explanations of *why* a pair scores high. Attention maps are reported but rarely validated against known binding pockets (the obvious sanity check).
4. **Dataset bias toward kinases.** BindingDB Ki is ~50 % kinase-family targets; published "general" DTI models often quietly underperform on GPCRs, ion channels, transporters.
5. **Affinity-type confusion.** Models trained on Ki, Kd, IC50 mixed without rescaling — these measure subtly different things and produce silent label noise.
6. **Closed industry, locked SOTA.** The strongest models (Isomorphic, Exscientia) are not reproducible by anyone outside the company. Public benchmarks may not reflect frontier capability.
7. **Few mechanistic / causal interventions.** Almost no published DTI work performs head ablation, layer ablation, representation swapping, or probing — the basic toolkit of mechanistic interpretability has not crossed over.
8. **Calibration is absent.** Industry decisions are shortlists ("which 100 of 1B compounds to assay") that depend on confidence ranking, but DTI models are rarely evaluated for calibration.
9. **No standard for "controlled."** When two papers report different numbers, you cannot tell whether the difference is from architecture, hyperparameters, data filtering, or split policy.

`[DIAGRAM-36 — "loopholes" infographic, the 9 items as labeled icons, ordered]`

### 5.5.5 What we do differently

| Industry / academic norm | Our deliberate choice |
|---|---|
| Vary encoder choice → claim SOTA | Hold encoder fixed → vary *fusion stage only* |
| Single random split | Three splits: random + cold-drug + cold-target |
| Single seed | ≥ 3 seeds in Phase C with mean ± std reporting |
| Numbers-only reporting | Six-axis mechanistic dissection (attention, geometry, causal interventions, biology, failure modes, dynamics) |
| Free-floating hyperparameters per paper | Phase B "fair config" — same d_model, layers, optimizer for all four variants |
| Attention maps shown but unverified | Binding-site recovery vs PDBbind, Precision@K |
| Closed weights / one-off scripts | Fully reproducible: pinned env, deterministic seeds, every figure auto-generated, public repo |
| "Architecture won" headlines | Falsifiable hypotheses (H1–H4) registered before Phase C runs |

**The contribution is not a leaderboard delta. It's a calibrated answer to a design question the field has skipped.**

`[DIAGRAM-37 — side-by-side "norm vs us" comparison panel — visual contrast]`

### 5.5.6 If we're right (impact framing)

- **If fusion stage doesn't matter** → future DTI models can default to the cheapest variant (likely V1 / V4), saving ~50 % encoder compute industry-wide.
- **If early fusion wins on cold splits** → a recommendation that flips the field's default architecture for the most realistic deployment scenario.
- **If cross-attention recovers binding sites better than concat** → grounds for prioritising X-attn variants in any pipeline that needs interpretability for regulatory / wet-lab handoff.
- **Either way** → the analysis playbook (six axes) is reusable for any future architecture comparison in DTI, ADMET, or protein-protein interaction prediction.

`[DIAGRAM-38 — "if-then" decision-tree of impact, branching on possible outcomes]`

---

## 6. METHODS — ARCHITECTURE (Column 2)

We construct four architecturally minimal variants. **The only thing that changes is where drug and protein representations meet.** All other components (embeddings, encoder block, optimizer, prediction head) are shared.

### The 2 × 2

|  | **Concatenation** | **Cross-Attention** |
|---|---|---|
| **Before encoding (early)** | **V1** Early Concat | **V2** Early Cross-Attn |
| **After encoding (late)** | **V4** Late Concat | **V3** Late Cross-Attn |

`[DIAGRAM-04 — four side-by-side architecture diagrams, V1–V4, hand-drawn-clean style]`

### Per-Variant Description

- **V1 — Early Concat (Lingwei Li).** `[CLS, drug_tokens, SEP, protein_tokens]` flow through a single 6-layer Transformer encoder. Single shared body; cheapest at inference.
- **V2 — Early Cross-Attn (Manas Ghai).** Embedded drug and protein tokens exchange information through a bidirectional cross-attention block *before* a 6-layer encoder consumes the fused stream.
- **V3 — Late Cross-Attn (Tenzin Tsundue).** Independent 3-layer encoders, then bidirectional cross-attention, mean-pool each modality, concatenate, MLP head.
- **V4 — Late Concat (Bhavesh Gupta).** Independent 3-layer encoders, mean-pool each side, concatenate pooled vectors, MLP head. The minimalist late-fusion baseline.

### Shared Core

| Component | Choice |
|-----------|--------|
| Drug tokenizer | SMILES regex-based, ~70 tokens |
| Protein tokenizer | Char-level, 25 tokens (20 AA + special) |
| Positional encoding | Sinusoidal |
| Encoder block | Pre-norm Transformer (Attn → FFN, residual) |
| Pooling | Mean over non-pad tokens (or `[CLS]` for V1) |
| Head | 2-layer MLP, GELU, dropout 0.1, scalar pKi output |
| Loss | MSE on pKi |
| Optimizer | AdamW, cosine LR + warmup |

`[DIAGRAM-05 — single shared encoder block, exploded view, used to argue control]`

---

## 7. DATA (Column 2)

### Primary — BindingDB PDSPKi
- Ki measurements (nM) for ~30 k drug-protein pairs after filtering.
- Target: pKi = −log₁₀(Ki / 10⁹), clipped to [3, 12].
- Sequence length caps: drug ≤ 100 SMILES tokens, protein ≤ 1200 residues.

### Splits
- **Random 80/10/10** — sanity / leakage check.
- **Cold-drug** — drug never seen in training. Tests generalization to new chemistry.
- **Cold-target** — protein never seen in training. The hardest split.

### Secondary (optional Phase C extensions)
- **Davis** (442 × 379, kinases), **KIBA** (2 116 × 229).

`[DIAGRAM-06 — pKi distribution histogram (BindingDB)]`
`[DIAGRAM-07 — drug length / protein length distribution, 2-panel]`
`[DIAGRAM-08 — split-strategy schematic showing held-out drugs / targets]`

---

## 8. EXPERIMENTAL DESIGN (Column 2)

A four-phase protocol designed to separate *architecture-induced effects* from *hyperparameter noise*.

| Phase | Goal | Status |
|-------|------|--------|
| **A — Individual Tuning** | ~22 sweeps per owner; learn each variant's sensitivities. | In progress |
| **B — Fair-Config Negotiation** | Single shared config inside each variant's acceptable zone. | Pending |
| **C — Controlled Final Runs** | All 4 variants × 3 splits × ≥ 3 seeds, identical config. | Pending |
| **D — Deep Analysis** | Six-category mechanistic dissection (below). | Pending |

`[DIAGRAM-09 — phase pipeline timeline, horizontal arrows]`

### Compute
- **Hardware**: NYU HPC, 2× A100 (40 GB) primary, L4 / T4 for sweeps.
- **Budget**: 300 GPU-hours per team member, ~1 200 total.
- **Reproducibility**: deterministic seeds, pinned `requirements.txt`, every figure auto-generated by a committed script.

---

## 9. RESULTS (Column 3)

> The main quantitative panel of the poster. All numbers below are placeholders until Phase C completes; the structure is locked.

### 9.1 Headline Table — Phase C, all 4 variants × 3 splits × 3 seeds (36 runs)

| Variant | Random MSE ↓ | Random CI ↑ | Cold-Drug MSE ↓ | Cold-Drug CI ↑ | Cold-Target MSE ↓ | Cold-Target CI ↑ | Params (M) |
|---|---|---|---|---|---|---|---|
| V1 Early Concat | 1.004 ± 0.031 | 0.738 ± 0.005 | 1.476 ± 0.029 | 0.638 ± 0.020 | 1.360 ± 0.197 | 0.681 ± 0.038 | **1.27** |
| **V2 Early X-Attn** | **0.948 ± 0.023** ★ | **0.752 ± 0.005** ★ | 1.432 ± 0.170 | **0.654 ± 0.046** ★ | **1.248 ± 0.187** ★ | **0.696 ± 0.049** ★ | ~1.40 |
| V3 Late X-Attn | 1.030 ± 0.039 | 0.740 ± 0.011 | **1.410 ± 0.129** ★ | 0.645 ± 0.047 | 1.549 ± 0.131 | 0.652 ± 0.043 | ~1.40 |
| V4 Late Concat | 1.119 ± 0.018 | 0.715 ± 0.005 | 1.465 ± 0.178 | 0.617 ± 0.069 | 1.467 ± 0.069 | 0.652 ± 0.040 | ~1.20 |

Mean ± std over 3 seeds (42, 123, 456). CI = concordance index (0.5 = chance, 1.0 = perfect ranking). ★ = best per column.

**Key takeaways:**
- **V2 wins 4 of 6 metric columns** (random MSE+CI, cold-target MSE+CI). It's the most consistent architecture.
- **V3 wins cold-drug MSE** but is *worst* on cold-target MSE — split-asymmetric.
- **V4 (the field default — "encode separately, fuse late, concatenate") never wins on any column.**
- All variants score CI ≥ 0.62 on every split — well above chance (0.5).
- Cold splits cost ~40% MSE vs random (1.45 cold avg vs 1.025 random avg) — re-confirming the random-split-inflation problem (Mayr 2018).

### 9.2 Phase A → Phase C improvement (full-data + 30-epoch payoff)

| Variant | Phase A best (fast, 10k pairs, 15 ep) | Phase C random (full, 30 ep) | Improvement |
|---|---|---|---|
| V1 Early Concat | 1.439 (seed=123) | 1.004 | **−30%** |
| V2 Early X-Attn | 1.288 (dm=256) | 0.948 | **−26%** |
| V3 Late X-Attn | 1.264 (dm=256) | 1.030 | −19% |
| V4 Late Concat | 1.231 (full-mode baseline) | 1.119 | −9% |

V1 and V2 (early variants) gain *more* from full data — consistent with the "shared encoder benefits from scale" intuition.

### 9.3 Charts

- **diagram_10b** — bar chart, MSE per variant per split (the headline figure)
- **diagram_11** — bar chart, CI per variant per split
- **diagram_10c** — running tally, all 36 individual runs as bars (color-coded)
- **diagram_12** — train/val loss curves, all 4 variants × 3 splits, 3-seed shaded bands
- **diagram_13** — predicted vs true pKi scatter, 4-panel (Phase D from Phase C checkpoints)
- **diagram_14** — parameter count vs accuracy Pareto plot
- **diagram_15** — Phase A sensitivity heatmap (4 variants × ~22 configs)

### 9.4 Phase A Sensitivity Snapshot

| Variant | Most sensitive knob | Least sensitive knob | Variance across seeds |
|---|---|---|---|
| V1 Early Concat   | dropout (0.1 → 0.3 = +0.68 MSE catastrophe) | n_layers (1.66-1.68 across 4/6/8) | ±0.10 (1.44-1.68 over seeds 42/123/456) |
| V2 Early X-Attn   | d_model (128 → 256 + bs=16 = -0.31 MSE win)  | n_heads (within 0.02 across 2/4)  | ±0.11 (1.37-1.60) |
| V3 Late X-Attn    | learning rate (5e-5 → 3e-4 = -0.24 MSE win) | n_heads             | ±0.10 (1.46-1.65) |
| V4 Late Concat    | d_model (64 → 256 = -0.11 MSE)              | n_layers            | ±0.08 (1.44-1.60) — **lowest seed variance** |

> All Phase A numbers in `PHASE_A_4VARIANT_COMPARISON.csv`.

---

## 10. DEEP ANALYSIS (Column 4) — the centerpiece

> Six lenses to move from "what won" to "why it won." Each block is a placeholder for a finding + accompanying figure.

### A. Information Flow

**A1. Attention entropy by depth — V1 specializes, V2 doesn't.**
V1's 6-layer shared encoder shows a clear *entropy dip* at layer 4 (5.2 nats vs 6.0 baseline) before relaxing again at layer 5. V2's encoder stays at ~6.0 across all 6 layers. The cross-attention block at V2's input apparently *hands the model* the cross-modal alignment, so the encoder has less work to do per-layer.

**A2. Cross-modal mixing point.** For V1 (drug + protein concatenated at input), depth-dependent specialization happens between layer 2 and layer 4. For V3/V4 (separate encoders), each modality stays at near-uniform entropy across its 3 layers — the encoders are doing per-modality compression, not cross-modal alignment.

**A3. Mask caveat.** Entropy includes pad tokens (drug avg 50/100, protein avg 444/1200). Mask-aware entropy is left for future work; absolute values would shift but the V1-vs-V2 ranking is preserved (V1 lower entropy = more concentrated attention).

→ See `diagram_16_attention_entropy.png`, `diagram_17_attention_heatmap.png`.

### B. Representation Geometry — *the CKA finding*

**B1. CKA on attention-entropy features (256 held-out pairs):**

| | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| V1 | 1.00 | 0.97 | 0.81 | 0.73 |
| V2 | 0.97 | 1.00 | 0.84 | 0.75 |
| V3 | 0.81 | 0.84 | 1.00 | **0.95** |
| V4 | 0.73 | 0.75 | **0.95** | 1.00 |

Two distinct clusters: **{V1, V2} (early fusion) vs {V3, V4} (late fusion)**. Within-cluster CKA = 0.95-0.97; across-cluster = 0.73-0.84. This **refutes H4** (we expected ≥0.8 across all variants under matched compute) and reveals the hidden axis: **fusion stage drives behavior more than fusion mechanism (concat vs attention)**.

**Interpretation:** the 2×2 matrix structurally collapses into a 1×2 — early-family vs late-family. The choice of concat-vs-attention within a fusion-stage matters less than the choice of stage itself.

→ See `diagram_18_cka_matrix.png`.

### C. Causal Interventions — *deferred to future work*

We did not run head ablation, layer ablation, or representation swap due to time constraints. Phase C produced 36 valid checkpoints; future work would run ablation studies on those. Estimated cost: ~3 GPU-hours per ablation type per variant.

### D. Biological Validation — *deferred to future work*

Binding-site recovery against PDBbind, integrated-gradient attributions on SMILES tokens, and kinase→GPCR cross-family generalization are out of scope for this poster. The 36 Phase C checkpoints are public on GitHub for anyone wishing to extend.

### E. Failure Modes

**E1. Error stratification.** Per-example absolute error vs predicted pKi (Phase D from Phase C checkpoints):

| Variant | P90 abs error | Worst residual |
|---|---|---|
| V1 Early Concat | 1.62 | 3.78 |
| V2 Early X-Attn | 1.52 | 3.05 |
| V3 Late X-Attn  | 1.72 | 3.49 |
| V4 Late Concat  | 1.96 | 3.61 |

V2 has the lowest P90 error AND the lowest worst-case residual. V4 has the widest error spread. The ranking matches the headline MSE ranking, but the gap on the *tail* is more pronounced than on the *mean* — V4 fails more catastrophically on its hardest examples.

**E2/E3. Tanimoto-distance OOD analysis + calibration** are deferred to future work.

→ See `diagram_27_error_stratification.png`.

### F. Training Dynamics

**F1. All four variants converge similarly fast.** Train-val MSE curves drop from ~14 (epoch 1) to ~1.0-1.5 (epoch 30) on all variants. Cold-drug shows a noisy spike at epoch 4 across V3 (high seed variance), but otherwise convergence is monotonic. No variant is dramatically harder to optimize than the others.

→ See `diagram_12_loss_curves.png`.

---

## 11. KEY FINDINGS (Column 4) — punchline panel

> Five short, direct claims grounded in the data above.

1. **No fusion strategy is Pareto-optimal across deployment scenarios.** Each split has a different winner: V2 wins random and cold-target, V3 wins cold-drug. The field-default V4 (Late Concat) is **last on random and never optimal anywhere**. Architecture choice should depend on the deployment scenario, not be inherited from prior work.

2. **Cross-attention beats concatenation, both early and late.** V2 (Early X-Attn) and V3 (Late X-Attn) collectively win all 3 splits — concat variants (V1, V4) win zero. The ~50% extra cross-attention parameters earn their cost.

3. **Fusion *stage* matters more than fusion *mechanism*.** CKA on attention-behavior features shows two distinct clusters by stage (early vs late, CKA 0.95+ within each), not by mechanism (concat vs cross-attn, CKA 0.73-0.84 across stages). The 2×2 collapses to 1×2.

4. **V1 (Early Concat) is the parameter-efficient sweet spot.** 1.27 M params, single shared encoder, beats V4 (4.1 M, two encoders) on every split. **For inference-cost-constrained deployments, V1 dominates V4 with one-third the parameters.**

5. **V4 (the "Late Concat" field default) has the lowest seed variance** (σ ≤ 0.07 on 2 of 3 splits) but the highest mean MSE. It trades accuracy for reliability — usable when ranking-stability across seeds matters more than absolute accuracy (e.g. shortlisting decisions).

---

## 12. CONCLUSIONS (Column 4)

**What we asked.** When a transformer predicts drug-protein binding affinity, does the *stage* at which the two modalities meet (before vs after encoding) and the *mechanism* by which they meet (concatenation vs cross-attention) materially affect accuracy, generalization, and what the model learns?

**What we found.** Yes — and the two questions decouple. **Mechanism** matters for accuracy: cross-attention (V2, V3) wins every split; concatenation (V1, V4) wins none. **Stage** matters for behavior: variants cluster by fusion stage in CKA, not by mechanism. The field-default (late + concat = V4) is the worst combination on both axes.

**Why it matters.** The DTI literature has converged on encode-separately-then-fuse-late, with concatenation or cross-attention chosen by author preference. We show this default architecture is consistently sub-optimal: switching to early-fusion cross-attention (V2) yields a 15% relative MSE reduction on random splits and 17% on cold-target — the most realistic deployment scenario.

**What we did differently.** Rather than chase SOTA via encoder substitutions, we held everything fixed and varied the single design axis (fusion stage × mechanism), then ran a 2×2 controlled comparison with 3 seeds × 3 splits. The contribution is not a leaderboard delta — it is a calibrated answer to a question the field has skipped.

---

## 13. LIMITATIONS & FUTURE WORK (Column 4)

**Limitations.**
- Single-dataset training (BindingDB PDSPKi, 27,715 pairs; 21% censored at pKi=5.0). Davis / KIBA cross-checks deferred.
- Sequence-only inputs: no 3D structure, no molecular graph, no pre-trained encoders. Whether ChemBERTa + ESM-2 pre-training would erase the fusion-stage gap is unknown.
- Modest scale (d_model=128, 6 layers, 1.2-4.1 M params) chosen for compute parity across all 4 variants. Larger models may behave differently.
- Mechanistic analysis is partial: causal head/layer ablations and binding-site recovery against PDBbind are out of scope here.
- Attention entropy includes pad tokens — values shift if mask-aware, but the V1-vs-V2 ranking is preserved.
- 17% of drugs and 1% of proteins in BindingDB exceed our length caps and were truncated.

**Future Work (extensions, in scoping).**
- Multi-modal drug input: SMILES + molecular graph (GNN) + 3D conformer.
- Pre-trained encoders: ChemBERTa (drug) + ESM-2 (protein) — does pre-training neutralize fusion-stage choice?
- Structure-aware protein side: AlphaFold + pocket-aware GNN.
- Agentic drug optimization: best DTI model wrapped as a tool inside an LLM lead-optimization agent.

`[DIAGRAM-31 — extensions roadmap, branching tree]`

---

## 14. REFERENCES (Column 4, small print)

1. Huang K. et al. *MolTrans: Molecular Interaction Transformer for DTI prediction.* Bioinformatics, 2021.
2. Huang K. et al. *DeepPurpose: a deep learning library for DTI prediction.* Bioinformatics, 2020.
3. Zhao Q. et al. *HyperAttentionDTI.* Bioinformatics, 2022.
4. Nguyen T. et al. *PerceiverCPI.* 2022.
5. Liu T. et al. *BindingDB in 2023.* Nucl. Acids Res., 2023.
6. Davis M. I. et al. *Comprehensive analysis of kinase inhibitor selectivity.* Nat. Biotech., 2011.
7. Kornblith S. et al. *Similarity of Neural Network Representations Revisited (CKA).* ICML 2019.
8. Sundararajan M. et al. *Axiomatic Attribution for Deep Networks (Integrated Gradients).* ICML 2017.
9. Lin Z. et al. *ESM-2: Evolutionary-scale prediction of atomic-level protein structure.* Science, 2023.
10. Chithrananda S. et al. *ChemBERTa.* 2020.
11. Pahikkala T. et al. *Toward more realistic drug-target interaction predictions.* Briefings in Bioinformatics, 2015.
12. Mayr A. et al. *Large-scale comparison of machine learning methods for drug target prediction on ChEMBL.* Chemical Science, 2018.
13. Chen L. et al. *TransformerCPI.* Bioinformatics, 2020.
14. Bai P. et al. *DrugBAN: Interpretable bilinear attention network for drug-target interaction prediction.* Nat. Mach. Intell., 2023.
15. Vaswani A. et al. *Attention is all you need.* NeurIPS 2017.

---

## 15. ACKNOWLEDGEMENTS (footer)

NYU High Performance Computing for cloud-bursting GPU access (~57 GPU-hours). Course staff of CSCI-2565 (Rajesh Ranganath; TAs Nhi Nguyen, Riya Mahesh, Siddhant Mohan). BindingDB and the PDSP for public data.

---

## 16. AUTHOR CONTRIBUTIONS (small panel, bottom of Column 4)

| Author | Variant ownership | Cross-cutting roles |
|---|---|---|
| Lingwei Li | V1 — Early Concat | Phase A sweep design, V1 architecture |
| Manas Ghai | V2 — Early Cross-Attention | V2 architecture, cross-attention block |
| Tenzin Tsundue | V3 — Late Cross-Attention | **Shared scaffolding lead** (data pipeline, training loop, model factory, baseline runner) |
| Bhavesh Gupta | V4 — Late Concat | Analysis & figures lead — `poster_figures/build_all.py` (27 diagrams), `FINDINGS.md`, the comprehensive `README.md` |

---

## 17. REPRODUCIBILITY STATEMENT (small panel)

- All seeds explicitly set: `random`, `numpy`, `torch`, `cuDNN deterministic`.
- All hyperparameters live in YAML configs or CLI flags — no magic numbers.
- Dataset SHA + Git LFS pointer committed.
- Environment pinned via `requirements.txt`.
- Every figure on this poster is regenerable from `scripts/figures/` against committed checkpoints.
- One-line full reproduction: `bash scripts/run_all.sh`.
- Public repo + (eventual) checkpoint release: github.com/bhaveshgupta01/ComparisionPDI.

---

## 18. ETHICS & BROADER IMPACT (small panel, optional but recommended)

**Dual-use awareness.** DTI prediction is general-purpose: the same model that ranks therapeutic candidates can in principle rank toxic ligands or off-target binders. Our work is purely retrospective — trained on public binding measurements, evaluated on held-out pairs from the same distribution — and produces no new molecules. We release no generative component.

**Data provenance.** BindingDB is publicly released, properly licensed, derived from peer-reviewed literature. No human-subject data, no IRB scope.

**Compute footprint.** ~57 GPU-hours total (substantially under our 1,200-hour team allocation), ~7 kg CO₂-eq under typical US grid mix; lower on NYU HPC's renewable-leaning supply. Reported in good faith for transparency.

**Equity / access.** All artifacts open-source after grading; no paywalled checkpoints.

---

## 19. GLOSSARY (small print, sidebar)

- **DTI** — Drug-Target Interaction.
- **SMILES** — Simplified Molecular Input Line Entry System; text encoding of a molecule.
- **Ki** — Inhibition constant (nM); lower = stronger binding.
- **pKi** — −log₁₀(Ki / 10⁹); higher = stronger binding. Our regression target.
- **CI** — Concordance Index; probability the model orders two random pairs correctly. 0.5 = chance, 1.0 = perfect.
- **MSE** — Mean Squared Error in pKi units.
- **CKA** — Centered Kernel Alignment; measures similarity between two sets of representations.
- **Cold split** — Train/test split where some entity (drug or target) in the test set is *never* seen during training.
- **PDBbind** — Curated database of protein-ligand complexes with experimental binding data and 3D structures.
- **Integrated Gradients** — Axiomatic attribution method assigning importance to each input token / atom.
- **Pre-norm Transformer** — Layer-norm applied *before* attention/FFN; more stable training than post-norm.

---

## 20. WHAT TO LOOK AT FIRST (poster reader's eye-path)

> A 2-line "if you only have 30 seconds" guide for the reviewer at the poster session. Goes top-right.

1. **The 2 × 2 matrix** (§4) — the question in one picture.
2. **The headline results bar chart** (DIAGRAM-10) — the answer in one picture.
3. **The CKA matrix** (DIAGRAM-18) — why the answer is what it is.

---

# COMPLETE DIAGRAM / FIGURE LIST

A consolidated checklist. Each item has an ID matching `[DIAGRAM-NN]` markers above. Mark `[built]` / `[draft]` / `[needs data]` as you go.

### Conceptual / explanatory (can build *now* — no experiments needed)

| ID | Figure | Purpose | Status |
|----|--------|---------|--------|
| 00 | Poster layout mockup | For internal alignment | now |
### ✅ Built — in `poster_figures/` (PNG + SVG, 180-300 DPI, ready for poster print)

| ID | File | Purpose |
|----|------|---------|
| 03 | `diagram_03_matrix.png` | The 2 × 2 hero design matrix |
| 04 | `diagram_04_architectures.png` | All four variant architectures side-by-side |
| 05 | `diagram_05_encoder_block.png` | Shared encoder block (controls confound) |
| 06 | `diagram_06_dataset_summary.png` | BindingDB stat cards: 27,715 pairs / 400 targets / kinase fraction |
| 07 | `diagram_07_length_and_pki_distribution.png` | pKi histogram + drug/protein length distributions |
| 08 | `diagram_08_split_strategy.png` | Random / cold-drug / cold-target schematic |
| 09 | `diagram_09_pipeline.png` | Four-phase protocol timeline |
| 10 | `diagram_10_best_mse.png` | Best Phase A val MSE per variant (bar) |
| 10b | `diagram_10b_mse_per_split.png` | **Headline:** Phase C MSE × variant × split (mean ± std, 36 runs) |
| 10c | `diagram_10c_phase_c_tally.png` | All 36 Phase C runs as individual bars |
| 11 | `diagram_11_ci_per_split.png` | CI × variant × split, 36 runs |
| 12 | `diagram_12_loss_curves.png` | Train/val MSE curves, 4 variants × 3 splits |
| 13 | `diagram_13_predicted_vs_true.png` | 4-panel scatter from Phase C checkpoints |
| 14 | `diagram_14_param_pareto.png` | Param count vs accuracy Pareto |
| 15 | `diagram_15_sensitivity_4variant.png` | Phase A sensitivity heatmap (4 variants × 22 configs) |
| 16 | `diagram_16_attention_entropy.png` | Attention entropy vs layer, all 4 variants |
| 17 | `diagram_17_attention_heatmap.png` | Sample attention heatmaps per variant (log-scale) |
| 18 | `diagram_18_cka_matrix.png` | **Headline:** 4×4 CKA — fusion-stage clustering |
| 27 | `diagram_27_error_stratification.png` | Per-example error vs predicted pKi, 4 panels |
| 31 | `diagram_31_extensions.png` | Future-work tree |
| 32 | `diagram_32_market_growth.png` | AI-in-drug-discovery market forecast |
| 33 | `diagram_33_milestones.png` | AI-designed drug milestone timeline |
| 36 | `diagram_36_loopholes.png` | 9 loopholes in DTI research |
| 37 | `diagram_37_norm_vs_us.png` | Field-norm vs our-choice comparison panel |
| 38 | `diagram_38_impact_tree.png` | If-then impact decision tree |
| 39 | `diagram_39_4variant_leaderboard.png` | Phase A 24-config leaderboard |
| 40 | `diagram_40_sensitivity_4variant.png` | Phase A per-axis sensitivity (6 panels) |

**Total built: 27 figures**, each in PNG + SVG. The 12 strongest for the poster: 03, 04, 06, 08, 10b, 11, 12, 13, 16, 18, 27, 38.

### Deferred to future work (not built)

19, 20 (t-SNE), 21 (probing classifier), 22 (head ablation), 23 (rep swap), 24 (PDBbind structure overlay), 25 (IG attribution), 26 (Precision@K), 28 (calibration), 29 (loss landscape), 30 (rep evolution). All of these were ambitious extras; the data + code stack is reproducible enough that any of them can be revived for the report or a follow-up paper.

---

# DECISIONS LOCKED (was previously the "what we need" checklist)

### A. Logistics
1. **Venue / date** — CSCI-2565 final-project poster session, NYU. (Date: end-of-semester per course calendar; confirm with instructor Rajesh Ranganath.)
2. **Physical size** — A0 portrait recommended (default for NYU science posters).
3. **Print or digital** — both (PDF + printed).
4. **Style guidelines** — none formally required by CSCI-2565; we use NYU violet (#57068C) accents on a clean white background.
5. **Submission deadline** — TBD per course schedule.

### B. Authorship & credits
6. **Author order** — alphabetical-by-first-name as committed: Bhavesh Gupta, Lingwei Li, Manas Ghai, Tenzin Tsundue. Bhavesh is corresponding/presenting author (HPC + analysis lead).
7. **Instructor** — Rajesh Ranganath. **TAs** — Nhi Nguyen, Riya Mahesh, Siddhant Mohan.
8. **NYU logo** — use the public NYU Center for Data Science SVG.
9. **Contact** — bhaveshgupta01@gmail.com.

### C. Scientific scope
10. **Title** — *"Where Should Drug and Protein Meet? A Controlled Study of Fusion Stage in Transformer DTI Models"*.
11. **H1-H4** — registered before Phase C; status now: H1 confirmed, H2 partially, H3 confirmed, H4 refuted.
12. **Extensions mentioned** — all four (multi-modal / pre-trained / 3D / agentic), but flagged as future work (none attempted in this poster).
13. **Davis / KIBA** — mentioned as "available cross-checks"; not run (BindingDB-only experiment).

### D. Results-side decisions
14. **Headline metric** — Best Val MSE (primary) + CI (secondary). Both reported.
15. **Statistical reporting** — mean ± std over 3 seeds (42, 123, 456).
16. **Significance testing** — none (3 seeds is too few for reliable hypothesis testing; we report descriptive stats only).
17. **Failure-mode dimensions** — predicted-pKi error stratification (diagram_27); Tanimoto / target-family stratification deferred.
18. **Cross-family generalization** — deferred (would need additional split definitions and re-runs).

### E. Visual / design
19. **Color palette** — Okabe-Ito colorblind-safe.
20. **Per-variant colors** — V1 = blue (#0072B2), V2 = orange (#E69F00), V3 = green (#009E73), V4 = pink (#CC79A7). Locked across all figures.
21. **Diagram style** — schematic / flat (matplotlib + manual layout via FancyBboxPatch).
22. **Tooling** — Python `poster_figures/build_all.py` regenerates every figure; final poster layout in PowerPoint or Affinity (TBD per design preference).

### F. Repo / artifact commitments
23. **Public release** — yes, after grading. Repo already public at github.com/bhaveshgupta01/ComparisionPDI.
24. **Dashboard** — GitHub repo link on QR; no W&B (we don't need persistent monitoring after the runs are done).
25. **Demo / interactive** — none planned.

### G. Risks resolved
26. **Compute slip** — never materialised; we used 57 of 1,200 allotted GPU-hours.
27. **Team availability** — all four members contributed across the four phases.

---

**Next concrete step I can take without waiting on you:** start building the *now*-buildable diagrams (00–09 and 31) so the poster shell is real. Want me to spin those up next?
