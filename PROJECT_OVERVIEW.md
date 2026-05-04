# Project Overview — Drug-Target Interaction Comparison Framework

> **A systematic study of how transformer drug-target prediction models behave when interaction between drug and protein representations happens at different stages of the architecture.**

---

## Quick Facts

| Field | Value |
|-------|-------|
| Course | CSCI-GA-2565 (Machine Learning), NYU MSCS, Spring 2026 |
| Repo | https://github.com/bhaveshgupta01/ComparisionPDI |
| Team | 4 students (one variant each) |
| Compute | NYU HPC Cloud Bursting, 300 GPU-hours each |
| Primary dataset | BindingDB PDSPKi (Ki measurements, ~64 MB TSV) |
| Target metric | pKi prediction (regression) |
| Status as of 2026-04-26 | Scaffolding complete, baseline runs done for 2/4 variants, Phase A sweeps pending |

---

## 1. The Question We're Asking

When a transformer predicts how strongly a drug binds to a protein, **at what point in the model should drug and protein information meet?**

Existing DTI papers default to "fuse representations late" — encode drug and protein separately first, then combine. But almost no one has rigorously tested whether earlier fusion would be better, worse, or different.

We answer this with a **2 × 2 controlled experiment**:

| | Concatenation | Cross-Attention |
|---|---|---|
| **Before encoding (early)** | V1: Early Concat | V2: Early Cross-Attn |
| **After encoding (late)** | V4: Late Concat | V3: Late Cross-Attn |

Critically, our goal is **not** to win benchmarks. It's to **understand** why each variant behaves the way it does. The deep analysis (attention patterns, representation geometry, error modes, biological validation) is the heart of the contribution.

---

## 2. Why This Matters

- **Scientific:** Most DTI papers fix the architecture and report numbers. We're asking a more fundamental design question.
- **Practical:** If early fusion works as well or better, future models can save compute (one encoder vs two).
- **Educational:** A controlled comparison teaches us about transformers as much as it teaches us about DTI.

---

## 3. Team and Ownership

| Variant | Strategy | Stage | Owner |
|---------|----------|-------|-------|
| **V1** | Concatenation | Before encoder | Lingwei Li |
| **V2** | Cross-Attention | Before encoder | Manas Ghai |
| **V3** | Cross-Attention | After encoder | Tenzin Tsundue |
| **V4** | Concatenation | After encoder | Bhavesh Gupta (you) |

Each owner is responsible for:
- Phase A tuning of their variant
- Sensitivity analysis of their hyperparameters
- Bringing findings to the team merge meeting

---

## 4. The Three-Phase Plan

### Phase A — Individual Tuning (in progress)
Each owner runs ~20 hyperparameter sweeps on their variant, on the BindingDB Ki dataset (~30k Ki-only pairs). Goal: find each variant's best config and learn its sensitivities.

### Phase B — Team Merge
Together we negotiate a "fair config" — settings where all 4 variants perform reasonably (not biased toward any single one). Same d_model, layer count, optimizer, etc.

### Phase C — Final Controlled Runs
Run all 4 variants with the agreed fair config, across 3 splits (random / cold-drug / cold-target) and multiple seeds. **The only thing that varies is the interaction strategy.**

### Phase D — Deep Analysis
Six categories of analysis (see Section 9) to explain *why* each variant behaves the way it does — not just what won.

---

## 5. Datasets

### Primary: BindingDB PDSPKi
- Source: https://www.bindingdb.org/
- File: `dataset/BindingDB/BindingDB_PDSPKi.tsv` (~64 MB, tracked via Git LFS)
- Columns used: `Ligand SMILES`, `BindingDB Target Chain Sequence 1`, `Ki (nM)`
- Target: pKi = −log₁₀(Ki / 10⁹), clipped to [3, 12]
- Size: ~30k usable pairs after filtering invalid rows

### Secondary (Phase C extensions, optional)
- **Davis** — 442 drugs × 379 proteins (small, fast benchmark)
- **KIBA** — 2,116 drugs × 229 proteins (medium benchmark)
- **MolTrans** subset — already in repo at `dataset/MolTrans/` for binary classification compatibility

### Splits
Three split strategies, each in `src/data/splits.py`:
- **random** — 80/10/10 by pair (sanity check)
- **cold-drug** — drugs in test never seen in training (realistic generalization)
- **cold-target** — proteins in test never seen in training (hardest)

---

## 6. Architecture (What's Built)

### Repository Layout

```
ComparisionPDI/
├── dataset/
│   ├── BindingDB/             # Git LFS — Ki + Articles TSVs
│   └── MolTrans/              # train/val/test CSVs
├── src/
│   ├── data/
│   │   ├── tokenizers.py      # SMILES (regex) + protein (char) tokenizers
│   │   ├── dataset.py         # BindingDBKiDataset reader
│   │   ├── splits.py          # random / cold_drug / cold_target
│   │   └── collate.py         # dynamic padding
│   ├── models/
│   │   ├── base.py            # BaseDTIModel abstract class
│   │   ├── embeddings.py      # Sinusoidal positional + token embedding
│   │   ├── encoders.py        # Pre-norm Transformer encoder block
│   │   ├── cross_attention.py # Bidirectional cross-attention
│   │   ├── prediction_head.py # MLP head
│   │   └── variants/
│   │       ├── early_concat.py
│   │       ├── early_crossattn.py
│   │       ├── late_concat.py      # ← V4 (yours)
│   │       └── late_crossattn.py
│   ├── training/
│   │   ├── trainer.py         # AdamW + cosine LR + early stopping
│   │   └── metrics.py         # MSE / CI / Pearson / Spearman
│   └── utils/
│       └── seeds.py
├── configs/
│   ├── base.yaml              # shared hyperparameters
│   └── variant_*.yaml         # one per variant
├── scripts/
│   ├── train.py               # CLI entry point
│   ├── preprocess.py          # builds vocab + splits cache
│   └── run_all.sh             # quick all-variants run
├── tests/                     # pytest scaffolding
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── results/
├── requirements.txt
└── run_train_2gpu.sbatch      # team's HPC submission template (2× A100)
```

### Core Hyperparameters (current baseline)

| Knob | Value |
|------|-------|
| d_model | 128 |
| n_heads | 4 |
| n_layers (total) | 6 |
| d_ff | 512 (4× d_model) |
| Dropout | 0.1 |
| MAX_DRUG_LEN | 100 |
| MAX_PROT_LEN | 1200 |
| Optimizer | AdamW |
| Learning rate | 1e-4 (cosine + warmup) |
| Weight decay | 1e-5 |
| Batch size | 64 (per-GPU) |
| Max epochs | 30 |
| Early-stop patience | 15 |
| Gradient clip | 1.0 |

For late variants, layers split 3+3 (drug encoder / protein encoder) by default.

### Variant Architectures

**V1 — Early Concat:**
```
embed(drug), embed(prot) → [CLS, drug, SEP, prot] → 6-layer shared encoder → CLS pool → MLP head
```

**V2 — Early Cross-Attn:**
```
embed(drug), embed(prot) → bidirectional cross-attn → concat → 6-layer encoder → CLS pool → MLP head
```

**V3 — Late Cross-Attn:**
```
embed(d), embed(p) → 3-layer drug enc, 3-layer prot enc → cross-attn → mean pool both → concat → MLP head
```

**V4 — Late Concat (yours):**
```
embed(d), embed(p) → 3-layer drug enc, 3-layer prot enc → mean pool both → concat → MLP head
```

---

## 7. Current Status (2026-04-26)

### What's Complete

- ✅ **All shared scaffolding** built and merged (mostly Tenzin's work over ~10 days)
- ✅ **All 4 variants** coded and registered in `src/models/`
- ✅ **Training pipeline** works with multi-GPU support via `DataParallel`
- ✅ **Data pipeline** parses BindingDB Ki data, builds vocab, supports 3 split types
- ✅ **HPC integration** — 2× A100 sbatch template tested
- ✅ **Documentation** — README, TEAM_BLUEPRINT, TECHNICAL_SPECIFICATION committed

### Baseline Results So Far

Tenzin ran two single-seed baselines on the random split:

| Variant | Best Val MSE | Best Val CI | Epochs run |
|---------|--------------|-------------|------------|
| V3 — Late Cross-Attn (Tenzin) | **1.295** | ~0.679 | 19+ |
| V4 — Late Concat | **1.347** | ~0.675 | 19+ |
| V1 — Early Concat (Lingwei) | sbatch added today, no result yet | — | — |
| V2 — Early Cross-Attn (Manas) | not started | — | — |

CI ≈ 0.68 means the model ranks pairs correctly ~68% of the time (random = 0.50). MSE ≈ 1.3 in pKi units corresponds to ~1.1 log-unit RMSE.

Both V3 and V4 reached similar accuracy at the baseline, which is preliminary support for the hypothesis that fusion strategy matters less than commonly assumed — but we need Phase A and Phase C before drawing real conclusions.

### What's Outstanding

- ⏳ Phase A sweeps for all 4 variants (none complete)
- ⏳ Multi-seed runs (only seed=42 so far)
- ⏳ Cold-drug / cold-target splits (only random tested)
- ⏳ Phase B fair-config negotiation
- ⏳ Phase C final 4-way comparison
- ⏳ Phase D deep analysis (everything below)

---

## 8. Phase A Plan (Per Owner)

Each owner runs ~20 sweep configurations on their variant. The shared CLI in `scripts/train.py` already supports:
- `--lr`, `--d_model`, `--n_layers`, `--n_heads`, `--dropout`, `--batch_size`, `--seed`, `--split`

### Sweep Rounds (template)

| Round | What varies | Runs |
|-------|-------------|------|
| 0 | Baseline | 1 |
| 1 | Learning rate {5e-5, 1e-4, 3e-4} | 3 |
| 2 | d_model {64, 128, 256} | 3 |
| 3 | n_layers {4, 6, 8} | 3 |
| 4 | Batch size {32, 64, 128} | 3 |
| 5 | Dropout {0.1, 0.2, 0.3} | 3 |
| 6 | Heads {2, 4, 8} | 2 |
| 7 | Seed sanity {42, 123, 456} on best config | 3 |
| 8 | Splits {cold_drug, cold_target} | 2 |
| **Total** | | **~22 runs** |

### Variant-Specific Knobs to Add (need ~5 lines of code)
- **V1 Early Concat:** type/segment embeddings, concat order
- **V2 Early X-Attn:** X-attn depth (1/2/3 layers), bidirectional vs one-way
- **V3 Late X-Attn:** encoder split (3+3 vs 2+4 vs 4+2), X-attn placement
- **V4 Late Concat:** encoder split, pooling per encoder (mean/max/CLS), pool-then-concat vs concat-then-pool

### Per-Owner Deliverable

Each owner produces a 1-page report:
```
Variant:        V<X> — <name>
Best config:    d_model=…, n_layers=…, lr=…, batch=…, dropout=…
Best result:    val_mse = … ± … (3 seeds), val_ci = … ± …
Sensitivity table:
  Hyperparam | Best | Range tested | Sensitivity (H/M/L)
  ……
Findings: surprises, failure modes, what to keep / drop in Phase B
```

---

## 9. Phase D — Deep Analysis Playbook

This is the centerpiece of the report. Six categories of analysis, each pushing from "what happened" → "how the model computes it" → "why the architecture causes that."

### Part A — Information Flow
- **Attention entropy per layer** — focused vs diffuse?
- **Cross-modal mixing point** — at which layer does drug info influence protein representation?
- **Gradient flow analysis** — which input tokens get the most signal?

### Part B — Representation Geometry
- **Intrinsic dimensionality** (participation ratio) of learned embeddings
- **CKA between variants** — do they learn the same features or different ones?
- **Probing classifiers** — what properties (MW, LogP, secondary structure, family) are linearly accessible?
- **t-SNE / UMAP** of drug and protein representations

### Part C — Causal Interventions
- **Attention head ablation** — which heads matter? what do they look at?
- **Layer ablation** — which layers carry the prediction?
- **Representation swap** — swap drug encoders across variants; does perf survive?
- **Input perturbation sensitivity** — robustness to small SMILES / sequence changes

### Part D — Biological Validation
- **Binding-site recovery** — Precision@K of attended residues vs known PDBbind binding pockets
- **Functional group attribution** (Integrated Gradients) — does the model identify pharmacophores?
- **Cross-family generalization** — train on kinases, test on GPCRs, etc.

### Part E — Failure Modes
- **Error stratification** — which drug / protein / pair properties make each variant fail?
- **OOD curves** — error vs distance to training set
- **Confidence calibration** — does the model know when it's wrong?

### Part F — Training Dynamics
- **Loss landscape comparison** — does one variant optimize easier?
- **Representation evolution** — when do reps stabilize during training?
- **Attention emergence** — when does biologically meaningful attention appear?

---

## 10. Compute Plan

### Each Owner: 300 GPU-hours allocated

| Phase | GPU-hours used (per owner) |
|-------|---------------------------|
| Phase A — sweeps | ~15 |
| Phase C — final runs (your share) | ~75 |
| Phase D — analysis | ~35 |
| Buffer + extensions | ~50 |
| **Total committed** | **~175** |
| **Reserve** | ~125 |

### Partition Strategy

| Partition | GPU | When to use |
|-----------|-----|-------------|
| `interactive` | none | OOD shell, debugging, no GPU |
| `n1s8-t4-1` | T4 16GB | Small sweeps, smoke tests |
| `g2-standard-12` | L4 24GB | **Default for Phase A** |
| `c12m85-a100-1` | A100 40GB | Phase C full BindingDB |
| `c24m170-a100-2` | 2× A100 | Multi-GPU; team's existing template |

### Bhavesh's HPC Files (V4 specific)

In `hpc_late_concat/`:
- `setup_venv.sh` — one-time venv setup
- `run_late_concat_sweep.sbatch` — single-job template (1× L4, 1.5h)
- `phase_a_sweep.sh` — submits all 22 sweep jobs

---

## 11. Possible Extensions (after Phase D)

If time permits, we can layer additional dimensions onto the analysis:

### Multi-Modal Drug Representation
Replace SMILES-only with SMILES + molecular graph (GNN) + 3D conformer (SchNet/EGNN). Re-run the 4 variants. Tests: does multi-modal input change the early-vs-late conclusion?

### Pre-Trained Encoders
Swap learnable embeddings for ChemBERTa (drug) and ESM-2 (protein). Tests: does pre-training neutralize the fusion-stage choice?

### 3D Structure-Aware Protein Encoder
Use AlphaFold structures + pocket-aware GNN on protein side. Tests: does explicit structure help early or late more?

### Agentic Drug Optimization
Wrap the best DTI model as a tool inside an LLM agent that iteratively proposes and refines drug candidates for a given target. Compare to random virtual screening.

---

## 12. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| All 4 variants perform identically | Medium | High | "Architecture doesn't matter" narrative is also publishable; CKA + analysis explains why |
| Hyperparameter choices in Phase B bias one variant | Medium | High | Each owner brings sensitivity table; pick from common acceptable zone |
| Cold-target split fails for all variants | Medium | Medium | Use ESM-2 pre-training as fallback (extension) |
| GPU budget runs out | Low | Medium | T4/L4 for sweeps, save A100 for Phase C only |
| BindingDB filtering yields too few pairs | Low | Low | Add Davis + KIBA as supplementary |
| Team coordination breakdown | Medium | High | Daily async standup, shared W&B project, locked configs |

---

## 13. Reproducibility Commitments

- All seeds explicitly set (random, numpy, torch, cuDNN deterministic)
- All hyperparameters in YAML configs or CLI flags (no magic numbers in code)
- Dataset SHA / Git LFS hash committed
- Environment pinned via `requirements.txt`
- Every figure auto-generated by a script committed to the repo
- Final reproduction command documented: `bash scripts/run_all.sh`

---

## 14. Where Each Document Lives

| Document | Purpose |
|----------|---------|
| `README.md` (in repo) | Dataset details, Git LFS setup |
| `TEAM_BLUEPRINT.md` (in repo) | Team conventions, locked vs free hyperparameters |
| `TECHNICAL_SPECIFICATION.md` (in repo) | Full engineering spec |
| `implementation_plan_claude.md` (in repo) | Original implementation outline |
| `PROJECT_OVERVIEW.md` (this doc) | High-level perspective |
| `GAMEPLAN.md` (Bhavesh's local) | 6-week phase-by-phase plan |
| `DEEP_ANALYSIS_PLAYBOOK.md` (Bhavesh's local) | Detailed analysis recipes |
| `hpc/HPC_SETUP_GUIDE.md` (Bhavesh's local) | NYU HPC walkthrough |
| `hpc_late_concat/` (Bhavesh's local) | V4-specific sbatch + sweep scripts |

---

## 15. Timeline (working assumption)

| Week | Goal |
|------|------|
| ~~Week of Apr 13~~ | ~~Scaffolding (done)~~ |
| ~~Week of Apr 20~~ | ~~Baseline runs (done for V3, V4)~~ |
| Week of Apr 27 | Phase A sweeps — all 4 owners |
| Week of May 4 | Phase B merge meeting → fair config locked |
| Week of May 11 | Phase C — full controlled runs |
| Week of May 18 | Phase D — deep analysis |
| Week of May 25 | Extensions (if time) + writing |
| Week of Jun 1 | Final paper + presentation |

---

## 16. The One-Sentence Summary

> We're running a 4-architecture controlled experiment to learn not just *which* fusion strategy makes a transformer better at predicting drug-target binding, but *why* each strategy behaves the way it does — using attention analysis, representation geometry, causal interventions, biological validation, failure-mode dissection, and training-dynamics studies as our microscope.
