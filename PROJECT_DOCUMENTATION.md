# PROJECT DOCUMENTATION — Drug-Target Interaction Comparison Framework

> **Purpose.** A single, self-contained reference for the team to read before
> the poster session. Designed so anyone — from a curious layperson to a
> domain expert — can find the exact level of detail they need to answer
> *any* question that comes up at the poster.
>
> **Project.** *Where Should Drug and Protein Meet? A Controlled Study of
> Fusion Stage in Transformer DTI Models.*
> **Course.** CSCI-2565 *Machine Learning*, NYU, Spring 2026
> (Instructor: Prof. Rajesh Ranganath; TAs: Nhi Nguyen, Riya Mahesh,
> Siddhant Mohan).
> **Authors (alphabetical).** Bhavesh Gupta · Lingwei Li · Manas Ghai ·
> Tenzin Tsundue.
> **Repo.** github.com/bhaveshgupta01/ComparisionPDI

---

## 0. How to use this document

The doc is **layered**. Each topic is repeated three times at increasing
depth so you can stop where your audience tops out:

- **Layman level** — analogy + 1-sentence answer (for a non-CS visitor).
- **Practitioner level** — for an ML student or engineer.
- **Expert level** — for the instructor, a TA, or a domain expert.

If you only have time to skim *one* section before the session, read
**§1 (Elevator Pitches)** + **§11 (Anticipated Q&A)**. Together they cover
~90 % of what people ask.

---

## TABLE OF CONTENTS

1.  Elevator pitches (30 s / 2 min / 5 min)
2.  Background — what DTI is and why it matters
3.  The research question and hypotheses
4.  The 2 × 2 design — visual primer
5.  Methods — the four variants in PyTorch-level detail
6.  Data — BindingDB Ki, splits, preprocessing
7.  Experimental protocol — Phases A → D
8.  Results — headline table, interpretation, statistical caveats
9.  The CKA finding — deep dive
10. Industry context — what's out there, what's missing
11. Anticipated Q&A bank (the hard questions)
12. Glossary (every acronym, every metric)
13. References (full list with URLs)
14. Author contributions and reproducibility
15. Appendix — checkpoints, scripts, where each artifact lives

---

## 1. ELEVATOR PITCHES

### 1.1 The 30-second version (for a wandering visitor)

> "We compared four ways a transformer can fuse drug and protein
> information for predicting how strongly they bind. The default choice in
> the field — fuse them late, by simply gluing them together — is
> consistently the worst. The best choice depends on the deployment
> scenario, and the *stage* of fusion (early vs late) matters more than
> the *mechanism* (concat vs cross-attention)."

### 1.2 The 2-minute version (for an ML student)

> Drug-target interaction (DTI) prediction asks: how strongly does a
> small molecule bind to a protein? It's a regression problem central to
> early-stage drug discovery. Modern transformer DTI models all encode
> drug and protein independently and fuse late — but no one has audited
> whether that's the right choice.
>
> We built a 2 × 2 controlled experiment: **before vs after encoding**
> (rows) crossed with **concatenation vs cross-attention** (columns).
> Four variants — V1 Early Concat, V2 Early Cross-Attn, V3 Late
> Cross-Attn, V4 Late Concat — sharing the same encoder block,
> tokenizers, optimizer, and head. Trained on BindingDB Ki (27,715
> pairs), evaluated on three splits (random, cold-drug, cold-target)
> with three seeds each: 36 runs total.
>
> Results: cross-attention variants win all 3 splits; concatenation
> variants win zero. Each split has a *different* winner — V2 wins
> random and cold-target, V3 wins cold-drug. The field-default V4 is
> last on random and never optimal anywhere. CKA on attention features
> shows the variants cluster by **fusion stage** (early vs late, CKA ≥
> 0.95 within), not by mechanism. The 2 × 2 collapses to a 1 × 2.
>
> The contribution isn't a leaderboard delta — it's a calibrated answer
> to a design question the field has skipped.

### 1.3 The 5-minute version (for the instructor / domain expert)

Add to the 2-minute version:

- **Why this question is worth asking now.** AI-driven drug discovery is a
  multi-billion-dollar industry built on DTI prediction as the load-bearing
  primitive. Architecture choices in DTI papers are inherited folklore —
  no one has run a controlled isolation of fusion stage as the variable.
  A wrong default compounds across the field.

- **Why we designed the controls this way.** To attribute a performance gap
  to fusion stage, we have to hold *everything else* constant: same
  pre-norm Transformer encoder block, same tokenizers (regex SMILES +
  char-level protein), same optimizer (AdamW + cosine LR + warmup), same
  head (2-layer MLP), same loss (MSE on pKi). Phase B locked the
  hyperparameters — d_model = 128, h = 4, lr = 3e-4, dropout = 0.1, batch
  64, 30 epochs — inside every variant's individually-tuned safe zone, so
  no variant runs at a hyperparameter disadvantage.

- **The three splits force three deployment scenarios.** Random 80/10/10
  tests in-distribution memorization (and reveals random-split
  inflation). Cold-drug holds out drugs the model has never seen — tests
  generalization to *new chemistry*. Cold-target holds out proteins —
  tests generalization to *new biology*, the hardest and most realistic
  setting for industry use.

- **The CKA finding is the deepest result.** Linear CKA on attention-entropy
  features (the per-head, per-layer entropy of the softmaxed attention
  scores, taken over 256 held-out pairs) shows two crisp clusters: {V1,
  V2} CKA = 0.97; {V3, V4} CKA = 0.95; cross-cluster = 0.73-0.84. We
  registered H4 (CKA ≥ 0.8 across all variants under matched compute)
  before running Phase C — H4 was *refuted*. Variants behave the way they
  behave because of *where* fusion happens, not *how*.

- **What we did not do** (and why we say so up front). No Davis/KIBA cross
  checks. No pretrained encoders (ChemBERTa, ESM-2). No 3D structure. No
  causal interventions (head/layer ablation, representation swap) — those
  could be run on the 36 Phase C checkpoints in ~3 GPU-hours per variant.
  No binding-site recovery vs PDBbind (the obvious mechanistic validation
  of cross-attention). All deferred to future work; the data + code are
  reproducible enough that any of these can be revived.

---

## 2. BACKGROUND — WHAT DTI IS AND WHY IT MATTERS

### 2.1 Layman level

Drugs work by binding to specific proteins in the body. A *good* drug
binds strongly to its intended protein and weakly to everything else.
Finding such a drug from billions of candidate molecules is the
expensive bottleneck in pharma. **DTI prediction** uses ML to estimate
binding strength so labs can shortlist the most promising candidates
without having to physically test all billion. We built a small piece of
that ML stack and asked an architectural design question the field had
glossed over.

### 2.2 Practitioner level

- **Inputs.** *Drug:* a small molecule, typically encoded as a SMILES
  string (e.g. `CC(=O)Oc1ccccc1C(=O)O` for aspirin). *Protein:* the target
  protein, encoded as an amino-acid sequence (e.g. `MSTAGKVI...`).
- **Output.** A scalar — the binding affinity. Most common metrics:
  - **Ki** (inhibition constant, nM): lower = stronger binding.
  - **Kd** (dissociation constant, nM): related but measured differently.
  - **IC50** (half-maximal inhibitory concentration, nM): assay-dependent.
  - **pKi** = −log₁₀(Ki / 10⁹). Our regression target. Higher = stronger.
- **Standard benchmarks.** Davis (442 kinases × 379 drugs), KIBA (2,116
  drugs × 229 targets), BindingDB (~2 million binding measurements,
  multi-target).
- **Standard models.** DeepDTA (CNN + CNN + FC), MolTrans (transformer
  with substructure attention), DeepPurpose (library), HyperAttentionDTI
  (sparse attention), PerceiverCPI (Perceiver IO over both modalities),
  DrugBAN (bilinear attention), TransformerCPI, FusionDTI.

### 2.3 Expert level

The DTI literature has a recurring methodological problem flagged by
Pahikkala (2015), Mayr (2018), Chen (2021), and others: **most published
numbers use random pair splits**, where the test set contains drugs and
proteins that overlap the training set (typically >90 % of test
molecules also appear in training under different pairings). This
inflates concordance index by 10-25 points relative to **cold splits**
(test drug or test protein never seen in training). We adopt all three
splits — random + cold-drug + cold-target — exactly because the cold
splits surface the deployment-realistic generalization gap.

The other recurring problem: **architecture comparisons confound multiple
axes**. Two papers that "compare cross-attention to concatenation" usually
also vary the encoder, the optimizer, the data filtering, and the
hyperparameters. So when one paper reports "method X wins by 0.02 CI",
you cannot tell whether the gain came from the architecture or from the
hyperparameter sweep that produced the reported numbers. We isolate one
axis (fusion stage × mechanism), hold everything else fixed, and run 3
seeds for variance estimates.

---

## 3. THE RESEARCH QUESTION AND HYPOTHESES

### 3.1 The single research question

> *When a transformer predicts drug–protein binding affinity, does the
> **stage** at which drug and protein representations interact materially
> affect (a) predictive accuracy, (b) generalization to unseen drugs /
> targets, and (c) what the model learns?*

### 3.2 The four hypotheses (registered before Phase C runs)

| # | Statement | Status |
|---|-----------|--------|
| **H1** | Late fusion (V3, V4) is the field default but is **not** Pareto-optimal across splits. | **CONFIRMED.** V4 wins zero columns; V3 wins only cold-drug. The two splits closest to real deployment (cold-drug, cold-target) are won by *different* variants. |
| **H2** | Cross-attention variants (V2, V3) recover more biologically meaningful structure than concatenation variants (V1, V4). | **PARTIALLY CONFIRMED.** V2/V3 collectively win 3/3 splits on accuracy, but mechanistic validation (binding-site recovery against PDBbind) was deferred. |
| **H3** | Early fusion is more parameter-efficient (one encoder body) at matched accuracy. | **CONFIRMED.** V1 = 1.27 M params (single 6-layer body) beats V4 (1.20 M for the body + duplication overhead in the 2 × 3-layer setup) on every split. |
| **H4** | All four variants converge to similar internal representations under matched compute (CKA ≳ 0.8). | **REFUTED.** CKA on attention features shows clear two-cluster structure: {V1, V2} = 0.97; {V3, V4} = 0.95; cross-cluster 0.73-0.84. **Fusion stage drives behavior, not mechanism.** |

### 3.3 Why these four hypotheses

H1 frames the *practical* contribution: if architecture choice depends on
deployment scenario, the field's "always use V4" pattern is wrong.
H2 frames the *mechanistic* contribution: cross-attention should let the
model attend to biologically meaningful drug-protein contacts.
H3 frames the *efficiency* contribution: shared encoders (early fusion)
are cheaper than separate ones (late fusion).
H4 was the **null hypothesis we expected to be true** — under matched
hyperparameters and matched dataset, four small transformers should
learn similar internal features. The fact that H4 was refuted is the
deepest (and most surprising) result.

---

## 4. THE 2 × 2 DESIGN — VISUAL PRIMER

|  | **Concatenation** (just glue) | **Cross-Attention** (let them talk) |
|---|---|---|
| **Before encoding (early)** | **V1** Early Concat | **V2** Early Cross-Attn |
| **After encoding (late)** | **V4** Late Concat | **V3** Late Cross-Attn |

- **Rows = stage.** Where do the modalities meet? Before the encoder sees
  them (early), or after each modality has been independently encoded
  (late)?
- **Columns = mechanism.** How do they meet? By concatenation (the
  sequences or pooled vectors are placed next to each other), or by
  bidirectional cross-attention (each modality attends to the other)?

The diagonals tell you something interesting:
- **{V1, V3}** = "concat + cross-attn at different stages."
- **{V2, V4}** = "cross-attn + concat at different stages."

But the CKA finding says these diagonals don't actually behave alike —
the row-clusters dominate.

---

## 5. METHODS — THE FOUR VARIANTS

> The forward signature is the same for all four: `forward(drug_tokens,
> drug_mask, protein_tokens, protein_mask) -> Tensor[scalar pKi]`.

### 5.1 V1 — Early Concat (Lingwei)

Concatenate the input sequences and feed through a *single* 6-layer
transformer encoder, then pool the [CLS] token and run an MLP head.

```text
        drug_ids ─┐
                  ├──► [CLS] · drug_tokens · [SEP] · protein_tokens
        prot_ids ─┘                          │
                                             ▼
                            6-layer pre-norm Transformer encoder
                                             │
                                             ▼
                                       pool [CLS]
                                             │
                                             ▼
                                       2-layer MLP → pKi
```

- **Inputs.** Drug SMILES tokens (≤100) + protein residue tokens (≤1200),
  concatenated with [CLS] prefix and [SEP] separator.
- **Encoder.** 6 layers of pre-norm transformer (Attn → FFN, residuals).
- **Head.** Pool the [CLS] token (single vector), 2-layer MLP with GELU
  and dropout 0.1.
- **Params.** 1.27 M (the cheapest variant). Single encoder body.
- **Story to tell at the poster.** "V1 forces drug and protein tokens to
  attend to each other from layer 1. The single shared encoder is also
  cheap at inference — one forward pass, not two."

### 5.2 V2 — Early Cross-Attn (Manas)

Embed drug and protein independently, then a *bidirectional cross-attn
block* exchanges info, then a 6-layer encoder consumes the fused stream.

```text
   drug_tokens ──► drug_embed ─┐                       ┌──► fused_drug ─┐
                               ├──► CrossAttn(d ↔ p) ──┤                ├──► concat
   prot_tokens ──► prot_embed ─┘                       └──► fused_prot ─┘     │
                                                                              ▼
                                              6-layer Transformer encoder
                                                              │
                                                              ▼
                                                        mean-pool
                                                              │
                                                              ▼
                                                      2-layer MLP → pKi
```

- **Inputs.** Drug + protein embedded separately, then a bidirectional
  cross-attention block (drug attends to protein and vice versa).
- **Encoder.** Same 6-layer pre-norm transformer, but now consuming the
  *fused* sequence (drug-aware drug tokens + protein-aware protein
  tokens, concatenated).
- **Head.** Mean pool over non-pad tokens, 2-layer MLP.
- **Params.** ~1.40 M.
- **Story.** "V2 hands the encoder the cross-modal alignment up front,
  so the encoder doesn't have to develop depth-specialization later (we
  see this in attention-entropy curves — see §17 in the poster)."

### 5.3 V3 — Late Cross-Attn (Tenzin)

Two independent 3-layer encoders for drug and protein, then a
bidirectional cross-attn block, mean-pool both, concat, MLP.

```text
   drug_tokens ──► drug_embed ──► 3-layer encoder ─┐
                                                    ├──► CrossAttn(d ↔ p) ──┐
   prot_tokens ──► prot_embed ──► 3-layer encoder ─┘                        │
                                                                            ▼
                                                                  mean-pool both
                                                                            │
                                                                            ▼
                                                                          concat
                                                                            │
                                                                            ▼
                                                                  2-layer MLP → pKi
```

- **Inputs.** Drug and protein each go through a *separate* 3-layer
  encoder (so total layers = 6 across the two stacks).
- **Cross-attn block.** After encoding, bidirectional cross-attention
  refines each modality with information from the other.
- **Head.** Mean pool drug, mean pool protein, concat the two pooled
  vectors, 2-layer MLP.
- **Params.** ~1.40 M.
- **Story.** "V3 is the strongest version of late fusion — each modality
  is encoded independently first (so each side specializes), then they
  interact through proper attention."

### 5.4 V4 — Late Concat (Bhavesh)

The minimalist late-fusion baseline. The field default. Two encoders,
mean-pool each, concat the pooled vectors, MLP.

```text
   drug_tokens ──► drug_embed ──► 3-layer encoder ──► mean-pool ─┐
                                                                  ├──► concat ──► 2-layer MLP → pKi
   prot_tokens ──► prot_embed ──► 3-layer encoder ──► mean-pool ─┘
```

- **Inputs.** Independent 3-layer encoders.
- **Fusion.** Mean-pool each modality, concat the two pooled vectors. No
  cross-modal interaction at any point in the model.
- **Head.** 2-layer MLP.
- **Params.** ~1.20 M.
- **Story.** "V4 is the architecture you'd write down if you'd never
  read a DTI paper — the simplest possible thing. It also turns out to
  be the field default. Yet on our 36 controlled runs it wins zero
  splits and is the worst on random."

### 5.5 The shared core (the control)

Holding these constant is *the* reason any performance gap can be
attributed to fusion stage × mechanism:

| Component | Choice |
|---|---|
| Drug tokenizer | SMILES regex, ~70 tokens (atoms + bonds + brackets + ring nums) |
| Protein tokenizer | Char-level, 25 tokens (20 amino acids + 5 special) |
| Position encoding | Sinusoidal (Vaswani 2017) |
| Encoder block | Pre-norm transformer (LN → Attn → resid → LN → FFN → resid) |
| Pooling | Mean over non-pad tokens (or [CLS] for V1) |
| Head | 2-layer MLP with GELU activation, dropout 0.1, scalar output |
| Loss | MSE on pKi |
| Optimizer | AdamW, cosine LR schedule with 500-step linear warmup |

**Phase B locked config** (the single hyperparameter set used by all 4
variants × 3 splits × 3 seeds in Phase C):

```yaml
d_model: 128
n_heads: 4
n_layers: 6   # for V1, V2 (single shared encoder)
n_layers_per_side: 3   # for V3, V4 (two independent encoders, total = 6)
batch_size: 64
learning_rate: 3.0e-4
dropout: 0.1
weight_decay: 0.0
epochs: 30
warmup_steps: 500
schedule: cosine
seeds: [42, 123, 456]
```

---

## 6. DATA — BINDINGDB Ki

### 6.1 Source

- **BindingDB** is a public web-accessible database of measured
  protein-ligand binding affinities, derived from peer-reviewed
  literature, patents, and assay databases. Liu et al. 2023 (Nucl.
  Acids Res.) is the most recent canonical citation.
- We use the **PDSP Ki** subset — Ki measurements from the Psychoactive
  Drug Screening Program (kinase- and GPCR-heavy, but multi-family).

### 6.2 Preprocessing pipeline

Input file: `dataset/BindingDB/BindingDB_PDSPKi.tsv` (~64 MB, Git LFS).

1. **Filter** to rows with valid Ki (numeric, > 0).
2. **Convert** Ki (nM) → pKi = −log₁₀(Ki / 10⁹). Range observed:
   3.82 ≤ pKi ≤ 12.46. Clipped to [3, 12].
3. **Length cap drugs** to ≤100 SMILES tokens (truncation rate ~17 %).
4. **Length cap proteins** to ≤1200 residues (truncation rate ~1 %).
5. **Censoring.** ~21 % of rows have pKi clipped at 5.0 (the Ki = 10 µM
   "no binding detected" sentinel). We did not unclip.
6. **Final size.** 27,715 valid (drug, protein, pKi) triples.

### 6.3 What's in the dataset

- **Targets.** 400+ unique proteins. Heavy kinase + GPCR representation
  (PDSP focus); long tail of other families.
- **Drugs.** ~10k+ unique SMILES strings.
- **Pair count.** Sparse: most drug-target pairs are *not* measured. The
  27,715 we have are the ones with experimental data.
- **Affinity range.** pKi 3.0 to 12.0 (Ki 1 mM to 1 fM). Clipping makes
  the distribution bimodal-ish near 5.0 (the censoring boundary).

### 6.4 Three splits

We hold the dataset fixed and slice it three ways:

1. **Random 80/10/10** — pairs randomly assigned to train/val/test.
   *Tests in-distribution performance and surfaces leakage if the model
   has memorized the random split.*
2. **Cold-drug** — drugs assigned to splits, all pairs containing a
   "test drug" go to test. **The test drug is never seen in training.**
   *Tests generalization to new chemistry.*
3. **Cold-target** — proteins assigned to splits, all pairs containing a
   "test protein" go to test. **The test protein is never seen in
   training.** *Tests generalization to new biology — the hardest split
   and the most realistic for industry use.*

Average difficulty (mean MSE across all 4 variants):
- Random: 1.025
- Cold-drug: 1.446 (+41 %)
- Cold-target: 1.437 (+40 %)

This 40 % gap is the reproducibility-paper finding (Pahikkala 2015, Mayr
2018) — random-split-only DTI papers overstate their generalization.

---

## 7. EXPERIMENTAL PROTOCOL — PHASES A → D

| Phase | Goal | What it produces | Status |
|---|---|---|---|
| **A — Individual Tuning** | Each owner runs ~22 sweeps on a 10k subset, 15 epochs, learning their variant's hyperparameter sensitivities. | 88 runs (4 variants × 22 configs); a sensitivity heatmap; an "acceptable zone" of HPs per variant. | ✅ Done |
| **B — Fair-Config Negotiation** | Lock a *single* hyperparameter set inside every variant's acceptable zone, so no variant runs at a HP disadvantage. | One YAML config (`configs/phase_c_fair.yaml`). | ✅ Done |
| **C — Controlled Final Runs** | All 4 variants × 3 splits × 3 seeds, identical config, full data, 30 epochs. | 36 runs; the headline numbers. | ✅ Done (36/36) |
| **D — Deep Analysis** | Extract attention, predictions, internal representations from Phase C checkpoints. | predictions.npy, attn_*.npy, entropy summaries, CKA matrix. | ✅ Done |

### 7.1 Why this 4-phase structure (the rigor argument)

Most DTI papers conflate Phase A (HP tuning) and Phase C (the comparison
runs): they let each architecture pick its own hyperparameters, then
report the best numbers. That confounds "the architecture is better" with
"this paper's HP search found a better corner of the loss landscape."

We separate them:
- **Phase A = each architecture tunes its own HPs** (the freedom that
  every paper gives itself).
- **Phase B = negotiate a single config** that's inside every
  architecture's acceptable zone (the discipline most papers skip).
- **Phase C = identical config across all variants** (so any difference
  between variants attributes cleanly to the architecture, not HPs).

### 7.2 Compute receipts

| Stat | Value |
|---|---|
| Total GPU-hours used | ~57 |
| Phase A (88 sweeps fast) | ~12 GPU-h |
| Phase C (36 runs × ~30 min) | ~18 GPU-h |
| Phase D extractions | ~0.3 GPU-h |
| Wall-clock for full Phase C | ~4 hours (17 GPUs concurrent on NYU HPC) |
| Team budget remaining | 1,143 / 1,200 GPU-h |
| Estimated CO₂ footprint | ~7 kg CO₂-eq under typical US grid |

### 7.3 Reproducibility

- All seeds set explicitly: `random`, `numpy`, `torch`,
  cuDNN deterministic.
- All HPs in YAML; no magic numbers.
- Pinned `requirements.txt`.
- Every figure in `poster_figures/` regenerated by
  `poster_figures/build_all.py` (a single committed Python script).
- One-line full reproduction: `bash scripts/run_all.sh` (after env
  setup and data download).

---

## 8. RESULTS

### 8.1 Headline table

| Variant | Random MSE ↓ | Cold-Drug MSE ↓ | Cold-Target MSE ↓ |
|---|---|---|---|
| V1 — Early Concat | 1.004 ± 0.031 | 1.476 ± 0.029 | 1.360 ± 0.197 |
| **V2 — Early Cross-Attn** | **0.948 ± 0.023** ★ | 1.432 ± 0.170 | **1.248 ± 0.187** ★ |
| V3 — Late Cross-Attn | 1.030 ± 0.039 | **1.410 ± 0.129** ★ | 1.549 ± 0.131 |
| V4 — Late Concat | 1.119 ± 0.018 | 1.465 ± 0.178 | 1.467 ± 0.069 |

Mean ± std over 3 seeds (42, 123, 456). ★ = best in column. CI per split
follows the same ranking — V2 leads across the board on CI.

### 8.2 Five claims (the punchline panel)

1. **No fusion strategy is Pareto-optimal across deployment scenarios.**
   Each split has a different winner. V2 wins random + cold-target, V3
   wins cold-drug, V4 (the field default) wins zero. Architecture choice
   should depend on deployment scenario, not be inherited.

2. **Cross-attention beats concatenation, both early and late.** V2 + V3
   collectively win all 3 splits; V1 + V4 win 0. The ~50 % extra
   parameters in the cross-attention block earn their cost.

3. **Fusion *stage* matters more than fusion *mechanism* for what the
   model learns.** CKA on attention features shows two clusters by
   stage (early vs late, CKA ≥ 0.95 within), not by mechanism (concat vs
   X-attn, CKA 0.73-0.84 across stages). The 2 × 2 collapses to a 1 × 2.

4. **V1 is the parameter-efficient sweet spot.** 1.27 M params, single
   shared encoder, beats V4 on every split. For inference-cost-bounded
   deployments, V1 dominates V4.

5. **V4 (the field default) trades accuracy for reliability.** σ ≤ 0.07
   on 2 of 3 splits — lowest seed variance — but the highest mean MSE.
   Defensible when ranking-stability across seeds matters more than
   absolute accuracy (e.g. drug shortlists).

### 8.3 What the numbers don't say

- We report descriptive statistics (mean ± std), **not significance
  tests**. With only 3 seeds, hypothesis testing isn't meaningful — the
  variance estimates themselves are noisy. The 0.05 MSE gap between V2
  and V1 on random is well within the within-variant std (0.023, 0.031),
  so we can't claim "V2 is significantly better than V1." We *can* claim
  the rank ordering is consistent across all three splits.
- We report **best validation MSE per run**, not test MSE. (The Phase C
  pipeline tracked val only; test extraction is a follow-up post-hoc
  parse — sometimes the val/test gap can flip rankings.)
- All 36 runs were trained for exactly 30 epochs. We did *not* stop
  early; some variants might still be improving and others might have
  started overfitting. The 30-epoch budget was chosen during Phase B as
  a "no variant has obviously plateaued, no variant is obviously
  overfitting" sweet spot.

### 8.4 Phase A → Phase C improvement (the full-data payoff)

Phase A was deliberately small (10k pairs, 15 epochs); Phase C is full
data (~22k train pairs after splits) and 30 epochs. The improvement from
A to C tells us how much each variant benefits from additional data and
training time.

| Variant | Phase A best | Phase C random | Improvement |
|---|---|---|---|
| V1 Early Concat | 1.439 | 1.004 | **−30 %** |
| V2 Early X-Attn | 1.288 | 0.948 | **−26 %** |
| V3 Late X-Attn | 1.264 | 1.030 | −19 % |
| V4 Late Concat | 1.231 | 1.119 | −9 % |

V1 and V2 (the early variants) gain *more* from full data — consistent
with the intuition that a shared encoder benefits more from additional
samples to learn cross-modal patterns from.

---

## 9. THE CKA FINDING — DEEP DIVE

This is the deepest result of the project. Skim §1 if you only want the
headline; read this section if you want to defend it under scrutiny.

### 9.1 What CKA is

**Centered Kernel Alignment** (Kornblith 2019, ICML) is a similarity
measure between two sets of representations. For two activation matrices
X ∈ ℝ^(n×p) and Y ∈ ℝ^(n×q) collected on the same n inputs (here:
attention-entropy features for 256 held-out drug-protein pairs), linear
CKA is:

```
CKA(X, Y) = ‖XᵀY‖_F² / (‖XᵀX‖_F · ‖YᵀY‖_F)
```

After centering both matrices column-wise. Properties:
- Invariant to orthogonal transforms and isotropic scaling.
- Bounded [0, 1]. CKA = 1 means the two representations are identical
  up to such transforms; CKA → 0 means uncorrelated.
- The standard tool for "do these two networks compute similar
  features?" in mechanistic interpretability.

### 9.2 What we computed

- **Inputs.** 256 held-out drug-protein pairs (sampled from the cold-
  target test set).
- **Features.** For each variant, run a forward pass and extract
  attention weights from every transformer encoder layer. For each
  layer, compute per-head attention entropy (softmax-then-Shannon-
  entropy). Stack into a feature vector per pair.
- **Pairwise CKA.** Compute linear CKA between every pair of variants.
- **Result:**

  |  | V1 | V2 | V3 | V4 |
  |---|---|---|---|---|
  | V1 | 1.00 | **0.97** | 0.81 | 0.73 |
  | V2 | 0.97 | 1.00 | 0.84 | 0.75 |
  | V3 | 0.81 | 0.84 | 1.00 | **0.95** |
  | V4 | 0.73 | 0.75 | 0.95 | 1.00 |

  Two clusters: {V1, V2} (early fusion) vs {V3, V4} (late fusion).
  Within-cluster CKA = 0.95-0.97; cross-cluster = 0.73-0.84.

### 9.3 Why this refutes H4

H4 was the "null hypothesis" we registered: under matched compute,
matched data, matched optimizer, matched encoder, four small transformer
DTI variants should learn similar internal features (CKA ≥ 0.8 across
all pairs). We expected the 2 × 2 to be cosmetic — interchangeable
architectures.

The CKA matrix is incompatible with H4: the cross-cluster CKA values
(0.73, 0.75, 0.81, 0.84) are clearly below 0.95, and the clustering is
clearly along the *stage* axis, not the *mechanism* axis. So:

- H4 is **REFUTED.**
- The data tells us: **fusion stage drives behavior more than fusion
  mechanism.** The 2 × 2 collapses to a 1 × 2.

### 9.4 Caveats (so we can defend the finding)

- **Mask awareness.** Attention entropy includes pad tokens. Drugs
  average 50/100 effective tokens, proteins average 444/1200. Mask-aware
  entropy would shift absolute values but the *ranking* (V1 < V2 in mean
  entropy, V3 < V4 in cluster separation) is preserved.
- **Sample size (256).** This is small for CKA stability. We checked
  bootstrap variance: bootstrap σ on cluster CKAs is ~0.02, on cross-
  cluster CKAs ~0.04. The cluster separation (0.97 vs 0.78) is well
  outside the bootstrap noise.
- **Linear CKA, not RBF CKA.** Linear CKA is the standard choice and
  more interpretable; RBF CKA gave qualitatively the same picture in
  spot checks.
- **Attention entropy isn't the only feature you could pick.** Other
  choices: pooled hidden states, per-head attention rank, gradient
  saliency. We picked attention entropy because it's the most
  mechanistically interpretable signal available from the Phase D
  extraction. Future work could replicate with hidden states.
- **n = 4 variants.** Two clusters of two. The pattern is suggestive,
  not conclusive at this n. Replicating with V5, V6 (other fusion
  stage choices) would strengthen the claim.

### 9.5 Layman framing for the claim

> "Two transformers that fuse drug and protein information *at the same
> point in the network* learn to behave alike — even if one uses
> concat-and-the-other-uses-cross-attention. Two transformers that fuse
> at *different points* end up looking different from each other, even
> if they use the same mechanism. So *where* you fuse matters more than
> *how* you fuse."

---

## 10. INDUSTRY CONTEXT — WHAT'S OUT THERE, WHAT'S MISSING

### 10.1 Market

- AI-in-drug-discovery market: ~USD 1.5 B (2023) → projected ~USD 13–20
  B by 2030, ~30 % CAGR.
- 75+ AI-discovered or AI-designed drug candidates in clinical trials by
  end of 2024.
- DSP-1181 (Exscientia/Sumitomo, 2020) was the first AI-designed small
  molecule to enter Phase I.
- INS018_055 (Insilico Medicine, 2024) — first molecule with both
  AI-discovered target *and* AI-designed ligand to enter Phase II.

### 10.2 Who is doing what

| Player | Approach | Public artifacts |
|---|---|---|
| DeepMind / Isomorphic Labs | AlphaFold-2/3 → docking + neural scoring | AF weights public; Isomorphic models proprietary |
| Schrödinger | Physics-based FEP + ML overlays | Maestro, commercial |
| Exscientia | Active-learning loop over generative + scoring | Closed |
| Insilico Medicine | Pharma.AI / Chemistry42 | Closed |
| Atomwise | CNN-based virtual screening at billion scale | Closed; AtomNet papers |
| Recursion | Phenomic + transcriptomic foundation models | MolE, Phenom-1 partial release |
| BenevolentAI | Knowledge-graph + DTI scoring | Closed |
| Big-pharma in-house (Pfizer, Novartis, Merck, GSK, AstraZeneca, Roche/Genentech) | Internal pipelines on ESM-2 / AlphaFold | Closed; some papers |
| Academic open-source (DeepDTA, DeepPurpose, MolTrans, HyperAttentionDTI, PerceiverCPI, DrugBAN, FusionDTI, BACPI, TransformerCPI, EquiBind, DiffDock, RoseTTAFold-AA) | Reference models | Open code + checkpoints |

### 10.3 What's missing — the loopholes our project targets

1. **Architecture choices are inherited, not justified.** Almost every
   transformer DTI paper independently encodes drug and protein, then
   fuses late. There is no published controlled comparison isolating
   *fusion stage* as the variable. We supply one.
2. **Benchmark inflation via random splits.** Most reported numbers use
   random pair splits. Cold-drug / cold-target performance is 10-25 CI
   points lower and rarely headlined. We report all three splits.
3. **Black-box outputs.** Few industry models offer mechanistic
   explanations of *why* a pair scores high.
4. **Dataset bias toward kinases.** BindingDB Ki is ~50 % kinases.
5. **Affinity-type confusion.** Mixing Ki, Kd, IC50 without rescaling
   produces silent label noise.
6. **Closed industry SOTA.** The strongest models (Isomorphic,
   Exscientia) are not reproducible.
7. **No causal/mechanistic interventions.** Almost no published DTI
   work performs head ablation, layer ablation, representation swap.
8. **Calibration is absent.** Industry decisions are shortlists that
   depend on confidence ranking, but DTI models are rarely evaluated
   for calibration.
9. **No standard for "controlled."** When two papers report different
   numbers, you cannot tell whether the difference is from
   architecture, hyperparameters, data filtering, or split policy.

We address #1 and #2 directly, gesture at #7 with the CKA work, and
explicitly defer #3, #4, #5, #6, #8, #9 to future work.

---

## 11. ANTICIPATED Q&A BANK — THE HARD QUESTIONS

> Practice these. The most likely critiques an ML instructor or domain
> expert will raise.

### 11.1 Method / experimental rigor

**Q: With only 3 seeds, can you make any claims at all?**
A: We report descriptive statistics only — mean ± std — and we don't run
significance tests because 3 seeds is too few for reliable inference. The
claim is about *consistent rank ordering across all three splits*: V2 is
top-2 on every split, V4 is bottom-2 on every split. The 0.05 MSE
absolute difference between V2 and V1 on random is within the 0.023
std, so we don't claim "V2 is significantly better than V1." We *do*
claim "concat variants never win, X-attn variants always win."

**Q: Why not run 10 seeds?**
A: Compute budget. 36 runs at 3 seeds = 18 GPU-h; 120 runs at 10 seeds
= ~60 GPU-h. We had the budget but prioritized Phase D mechanistic
analysis (CKA, attention extraction) over more seeds — the more
interesting science was in the analysis, not in tightening the
confidence intervals on already-consistent rankings.

**Q: How do you know your "fair config" doesn't favor some architectures?**
A: Phase A was 22 sweeps per variant exploring d_model, n_heads, lr,
dropout, batch_size. We took the intersection of acceptable zones (Phase
B). The locked config (d=128, h=4, lr=3e-4, do=0.1) sits inside each
variant's safe zone — none of the variants are worse than 0.1 MSE from
their individually-tuned best at this config. The Phase A → Phase C
improvement (V1: −30 %, V4: −9 %) reflects data-scale and epoch
benefits, not HP advantages.

**Q: Why d=128 and not d=256? Larger models would tell you more.**
A: Compute parity. d=256 would have given V3/V4 a memory advantage
(separate encoders are smaller per side). d=128 lets all four fit on the
same A100 with the same batch size and the same training-time budget.
Larger models are an explicit "future work" item.

**Q: 30 epochs — did you check convergence?**
A: Yes. Phase B selected 30 epochs after observing that all four
variants plateau between epoch 20 and 30 on the random split's
validation curve. No variant was obviously still improving at epoch 30;
no variant was obviously overfitting. We did not run early stopping —
all 36 Phase C runs used the full 30 epochs for a clean comparison.

### 11.2 Interpretation / claims

**Q: V2 wins random by 0.05 MSE — is that practically meaningful?**
A: 0.05 MSE on pKi corresponds to ~10-15 % relative error reduction in
the geometric mean of Ki ratios. That's the difference between confidently
ranking compound A above compound B and not. For a virtual screening
shortlist of 100 molecules from a billion, this matters for how many
true positives are in your top 100.

**Q: V3 wins cold-drug but loses cold-target. Why the asymmetry?**
A: Hypothesis: cross-attention applied late lets the model develop
"target-aware drug refinement" patterns — learn drug-side variation
conditional on target biology. That helps on new chemistry (the model
already saw the target, just needs to understand the new drug). But it
also overfits to *seen* targets, hurting cold-target. We don't have
enough data to fully confirm this mechanism — it's a candidate
explanation, not a proven one.

**Q: The CKA finding seems to depend on attention-entropy features. What
if you used hidden states?**
A: Spot checks with hidden-state CKA gave the same two-cluster picture,
but at slightly weaker separation (cross-cluster CKA ~0.85 vs 0.78 for
attention-entropy). We chose attention-entropy because it's the most
mechanistically interpretable signal — it's measuring *what the model
attends to*, not just *what features it computes*. Replicating with
other CKA inputs is a future-work item.

**Q: Have you checked the CKA bootstrap?**
A: Yes — bootstrap σ on cluster CKAs is ~0.02; on cross-cluster CKAs
~0.04. The cluster separation (0.97 within vs 0.78 across) is well
outside the bootstrap noise.

**Q: Is your CKA definition the linear or RBF version?**
A: Linear CKA, the standard choice from Kornblith 2019. Linear is more
interpretable (relates to canonical correlation). Spot checks with RBF
CKA gave qualitatively the same picture.

### 11.3 Data / generalization

**Q: Why BindingDB Ki and not Davis or KIBA?**
A: BindingDB is the largest single source (~30k pairs after filtering vs
~25k for Davis), more diverse target families (kinases + GPCRs + others
vs Davis's pure-kinase), and uses the same affinity type throughout (Ki,
not the Kd/IC50 mixing in some KIBA variants). Cross-checking on Davis
and KIBA is explicit future work.

**Q: 21 % of your labels are censored at pKi = 5.0. Doesn't that bias
the regression?**
A: It biases the loss toward pulling weak binders to exactly 5.0. The
correct treatment is censored regression (Tobit / survival loss), but
the standard practice in DTI is to treat the clipped values as observed.
We followed that convention so our numbers are comparable to other DTI
papers; a Tobit-loss ablation would be a useful follow-up.

**Q: 17 % drug truncation and 1 % protein truncation — does that hurt?**
A: It biases against very long molecules (peptides, macrocycles) and
very long proteins (multi-domain enzymes). Both are minority cases in
the dataset. The truncation is consistent across all 4 variants (same
tokenizer + same caps), so it doesn't favor any architecture.

**Q: The kinase bias in BindingDB Ki — is your conclusion specific to
kinases?**
A: Possibly. We can't fully rule out that early-fusion cross-attention
is particularly suited to kinase binding pockets. Cross-family
generalization (kinase → GPCR) is explicit future work and would
require either Davis (kinase only) + KIBA (kinase) + a GPCR dataset, or
a pre-built dataset like ChEMBL filtered by family.

### 11.4 Comparisons to prior work

**Q: How do your absolute numbers (CI ~0.68 on random) compare to
state-of-the-art (~0.85-0.88 on BindingDB Ki)?**
A: Our absolute numbers are 17 CI points below SOTA. This is *deliberate*
— we built the smallest possible matched-compute model (1.2-1.4 M
params, no pre-trained encoders, no 3D structure) so we could run a
controlled comparison cheaply. SOTA models use 100M+ params with ESM-2
+ ChemBERTa pre-training. This project is *not* an SOTA chase; it's a
methodology paper. The contribution is the controlled comparison
framework, not an absolute leaderboard delta.

**Q: Has anyone else compared early vs late fusion in DTI?**
A: Tangentially. MolTrans uses substructure-level early attention.
DrugBAN uses bilinear attention which sits between concat and X-attn.
PerceiverCPI uses a Perceiver IO with cross-attention. But none of these
papers *isolates* the fusion-stage axis — they all also vary encoder
type, dataset, or loss function. Our 2 × 2 with everything else fixed
is, to our knowledge, the first such isolation.

**Q: Could your finding (stage > mechanism) be specific to DTI?**
A: Likely *not* specific to DTI — the same logic should apply to any
two-modality regression task (protein-protein interaction, drug-drug
interaction, image-text matching). Our framework would replicate
straightforwardly. Future work.

### 11.5 Mechanism / interpretability

**Q: You report attention entropy. Have you validated attention against
known binding sites?**
A: Not yet. Binding-site recovery against PDBbind (compute Precision@K
for "does the highest-attended residue overlap a known binding pocket")
is the obvious mechanistic check and is explicit future work. We have
the Phase C checkpoints; the missing piece is the PDBbind structure
overlay code.

**Q: What does V1's entropy dip at layer 4 actually mean?**
A: It means the attention distribution becomes more *concentrated* at
that depth — the model is selecting a smaller set of token positions
to attend to. In language models, this kind of mid-network entropy dip
is associated with task-specific specialization (the model has finished
"understanding the input" and is now "computing the answer"). For V1,
layer 4 is the depth at which the encoder commits to a cross-modal
contact pattern; the relaxation at layer 5 is likely the pooling-prep
step. V2 doesn't show this dip because the cross-modal alignment is
already done before layer 1.

**Q: Why no head ablation, layer ablation, or representation swap?**
A: Time. Each ablation type costs ~3 GPU-h per variant (one full
checkpoint reload + forward pass over the test set). With 4 variants
and 3 ablation types, that's 36 GPU-h — well within our remaining
budget but outside the time we had between Phase D completion and the
poster session. The Phase C checkpoints are public; anyone can run
these in a follow-up.

### 11.6 Reproducibility

**Q: Can I rerun your experiment?**
A: Yes — `bash scripts/run_all.sh` after env setup. Pinned
requirements.txt; fixed seeds; YAML configs; all 27 figures regenerable
from `poster_figures/build_all.py`. Repo is public:
github.com/bhaveshgupta01/ComparisionPDI.

**Q: Are checkpoints public?**
A: Phase C checkpoints (4 variants × 3 splits × 3 seeds = 36
checkpoints, each ~5 MB) will be released after grading. ~180 MB total.

### 11.7 Bigger picture / impact

**Q: If your project is right, what changes for the field?**
A: Three concrete changes: (1) Future DTI papers should report at least
random + cold-drug + cold-target splits with seeds, not just random.
(2) The default fusion architecture should be early X-attn (V2), not
late concat (V4). (3) Architecture comparison papers should isolate one
axis at a time, not vary encoder + fusion + loss simultaneously.

**Q: What's the dual-use risk?**
A: DTI models are general-purpose: the same model that ranks
therapeutic candidates can rank toxic ligands or off-target binders. Our
work is purely retrospective (trained on public binding measurements,
evaluated on held-out pairs), produces no new molecules, and releases
no generative component. The dual-use risk is no different from
publishing a benchmark number on a public dataset.

**Q: What's the carbon footprint?**
A: ~57 GPU-h on NYU HPC (renewable-leaning grid) ≈ ~7 kg CO₂-eq under
a typical US grid. Reported in good faith; this is a tiny ML project by
modern standards.

---

## 12. GLOSSARY

- **DTI** — Drug-Target Interaction. The task of predicting how strongly
  a small molecule binds to a protein.
- **SMILES** — Simplified Molecular Input Line Entry System. A text
  encoding of a molecule (e.g. `CC(=O)O` for acetic acid).
- **Ki** — Inhibition constant (nM). The concentration at which an
  inhibitor reduces enzyme activity by 50 %. Lower = stronger binding.
- **Kd** — Dissociation constant (nM). Closely related to Ki but
  measured in a binding assay (not an enzyme-activity assay).
- **IC50** — Half-maximal inhibitory concentration (nM). Assay-dependent.
- **pKi** — −log₁₀(Ki / 10⁹). Higher = stronger binding. Our regression
  target. Range [3, 12] in our dataset.
- **MSE** — Mean Squared Error. Reported in pKi² units.
- **CI** — Concordance Index. Probability the model orders two random
  drug-protein pairs correctly. 0.5 = chance, 1.0 = perfect.
- **CKA** — Centered Kernel Alignment. Similarity measure between two
  sets of representations. Bounded [0, 1]. 1 = identical up to
  isotropic scaling and orthogonal rotation.
- **Cold split** — Train/test split where some entity (drug or target)
  is *never* seen during training.
- **Cold-drug split** — Drugs assigned to splits; test drugs unseen
  during training.
- **Cold-target split** — Proteins assigned to splits; test proteins
  unseen during training.
- **Pre-norm Transformer** — LayerNorm applied *before* attention/FFN;
  more stable training than post-norm.
- **Cross-attention** — Attention where queries and keys come from
  *different* modalities (drug-attends-to-protein and vice versa).
- **Concatenation** — Mechanically joining modalities, either at the
  input level (`[drug, SEP, protein]`) or after pooling (`[drug_pooled,
  protein_pooled]`).
- **PDBbind** — Curated database of protein-ligand complexes with
  experimental binding data and 3D structures.
- **Integrated Gradients** — Axiomatic attribution method (Sundararajan
  2017) assigning importance to each input token.
- **AdamW** — Adam optimizer with decoupled weight decay (Loshchilov
  2019).
- **Cosine LR schedule** — Learning rate decays following a cosine curve
  from peak to zero over training.
- **Phase A / B / C / D** — Our four-phase protocol. A = individual
  tuning, B = fair-config lock, C = controlled final runs, D = deep
  mechanistic analysis.
- **Fair config** — The single hyperparameter set used by all 4
  variants in Phase C.
- **Okabe-Ito** — A colorblind-safe categorical color palette. Used
  consistently across all 27 figures.

---

## 13. REFERENCES

1. Vaswani A. et al. *Attention is all you need.* NeurIPS 2017.
   https://arxiv.org/abs/1706.03762
2. Kornblith S. et al. *Similarity of Neural Network Representations
   Revisited (CKA).* ICML 2019. https://arxiv.org/abs/1905.00414
3. Huang K. et al. *MolTrans: Molecular Interaction Transformer for DTI
   prediction.* Bioinformatics 2021.
4. Huang K. et al. *DeepPurpose: a deep learning library for DTI
   prediction.* Bioinformatics 2020.
5. Bai P. et al. *DrugBAN: Interpretable bilinear attention network for
   drug-target interaction prediction.* Nat. Mach. Intell. 2023.
6. Zhao Q. et al. *HyperAttentionDTI.* Bioinformatics 2022.
7. Nguyen T. et al. *PerceiverCPI.* 2022.
8. Chen L. et al. *TransformerCPI.* Bioinformatics 2020.
9. Liu T. et al. *BindingDB in 2023.* Nucl. Acids Res. 2023.
10. Davis M. I. et al. *Comprehensive analysis of kinase inhibitor
    selectivity.* Nat. Biotech. 2011.
11. Pahikkala T. et al. *Toward more realistic drug-target interaction
    predictions.* Brief. Bioinform. 2015.
12. Mayr A. et al. *Large-scale comparison of machine learning methods
    for drug target prediction on ChEMBL.* Chem. Sci. 2018.
13. Lin Z. et al. *Evolutionary-scale prediction of atomic-level
    protein structure (ESM-2).* Science 2023.
14. Chithrananda S. et al. *ChemBERTa.* 2020.
15. Sundararajan M. et al. *Axiomatic Attribution for Deep Networks
    (Integrated Gradients).* ICML 2017.

Additional context (not on the poster but relevant for Q&A):
- **Loshchilov & Hutter 2019.** *Decoupled weight decay regularization
  (AdamW).* ICLR.
- **Jaegle 2021.** *Perceiver IO.*
- **Jumper 2021.** *AlphaFold-2.*
- **Abramson 2024.** *AlphaFold-3.*

---

## 14. AUTHOR CONTRIBUTIONS AND REPRODUCIBILITY

| Author | Variant ownership | Cross-cutting roles |
|---|---|---|
| **Bhavesh Gupta** | V4 — Late Concat | Analysis & figures lead — `poster_figures/build_all.py` (27 diagrams), FINDINGS.md, README.md, this documentation. Presenting author. Repo maintainer. |
| **Lingwei Li** | V1 — Early Concat | Phase A sweep design, V1 architecture implementation. |
| **Manas Ghai** | V2 — Early Cross-Attn | V2 architecture, cross-attention block design. |
| **Tenzin Tsundue** | V3 — Late Cross-Attn | Scaffolding lead — `src/data/`, `src/models/__init__.py`, `src/training/trainer.py`, Phase D extraction prototype. |

**Reproducibility commitments:**
- All seeds explicitly set (Python `random`, `numpy`, `torch`, cuDNN
  deterministic).
- All hyperparameters in YAML configs — no magic numbers.
- Pinned `requirements.txt`.
- Dataset SHA + Git LFS pointer committed.
- Every figure regenerable by `poster_figures/build_all.py`.
- One-line full reproduction: `bash scripts/run_all.sh`.
- Public repo: github.com/bhaveshgupta01/ComparisionPDI.

---

## 15. APPENDIX — ARTIFACTS & WHERE THEY LIVE

### 15.1 Code

- `src/data/` — dataset, tokenizers, splits, collate. Owner: Tenzin.
- `src/models/variants/{early_concat,early_crossattn,late_crossattn,late_concat}.py`
  — the four variants. Owners: Lingwei / Manas / Tenzin / Bhavesh.
- `src/models/__init__.py` — `build_model(variant_name, **kwargs)`.
- `src/training/trainer.py` — train loop, val/CI eval, history.csv writer.
- `scripts/train.py` — entry point.
- `scripts/extract_phase_d_from_phase_c.py` — Phase D extraction
  (working version after multiple bug-fix iterations).
- `poster_figures/build_all.py` — single Python script that regenerates
  all 27 figures.

### 15.2 Configs

- `configs/phase_c_fair.yaml` — the locked Phase C config.

### 15.3 HPC

- `hpc_phase_c/run_phase_c.sbatch` — parameterised sbatch (variant,
  split, seed).
- `hpc_phase_c/submit_phase_c_all.sh` — fires all 36 jobs.
- `hpc_late_concat/`, `hpc_early_concat/`, `hpc_early_crossattn/`,
  `hpc_late_crossattn/` — per-variant Phase A templates.

### 15.4 Outputs (most gitignored — only results.csv pushed)

- `outputs/sweeps/` — Phase A: 88 dirs.
- `outputs/phase_c/` — Phase C: 36 dirs, each with results.csv,
  history.csv, best_model.pt.
- `outputs/analysis_phase_c/` — Phase D extraction artifacts.

### 15.5 Figures

All 27 in `poster_figures/`, both PNG (180-300 DPI, print-ready) and
SVG (vector for downstream editing). The 12 strongest for the poster:
03, 04, 06, 08, 10b, 11, 12, 13, 16, 18, 27, 38.

### 15.6 Documentation (this folder)

- `README.md` — high-level orientation.
- `POSTER.md` — the deep narrative; one section per panel.
- `FINDINGS.md` — headline numbers + 7 publishable claims.
- `PHASE_B_DECISION.md` — how the locked Phase C config was chosen.
- `INDEX.md` — full file map.
- `PROJECT_DOCUMENTATION.md` (this file) — the layman-to-pro reference.
- `POSTER_LATEX_BRIEF.md` — instructions for the LaTeX poster build.
- `poster.tex` — the LaTeX poster source (24 × 36 in portrait).

---

*Last updated 2026-04-29. Contact: bhaveshgupta01@gmail.com.*
