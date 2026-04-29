# ComparisionPDI — Drug–Target Interaction Comparison Framework

> A 2 × 2 controlled comparison of how transformer architectures fuse drug and protein information for binding-affinity prediction. We hold every other design decision constant (encoder, optimizer, data, head) and vary only **fusion stage × fusion mechanism**, then run 36 controlled runs (4 variants × 3 splits × 3 seeds) plus a deep mechanistic analysis.

**Course:** CSCI-2565 Machine Learning, Spring 2026 (NYU)
**Instructor:** Rajesh Ranganath. **TAs:** Nhi Nguyen, Riya Mahesh, Siddhant Mohan.
**Authors (alphabetical):** Bhavesh Gupta · Lingwei Li · Manas Ghai · Tenzin Tsundue.
**Repo:** github.com/bhaveshgupta01/ComparisionPDI · **Contact:** bhaveshgupta01@gmail.com.

---

## TL;DR

We trained four transformer variants of a drug-target affinity model under matched hyperparameters and evaluated on three splits with three seeds. **No fusion strategy is Pareto-optimal across deployment scenarios:** V2 (early cross-attn) wins random + cold-target, V3 (late cross-attn) wins cold-drug, and V4 (the field-default late concatenation) **never wins**. CKA on attention features reveals that variants cluster by *fusion stage* (early vs late, CKA = 0.95+ within), not by *fusion mechanism* (concat vs attention).

→ Read [FINDINGS.md](FINDINGS.md) for headline numbers and 7 publishable claims.
→ Read [POSTER.md](POSTER.md) for the full poster draft (one section per panel).

---

## The 2 × 2 Design

| | **Concatenation** | **Cross-Attention** |
|---|---|---|
| **Before encoder (early)** | **V1** Early Concat | **V2** Early Cross-Attn |
| **After encoder (late)**   | **V4** Late Concat  | **V3** Late Cross-Attn  |

- **V1 — Early Concat (Lingwei).** `[CLS, drug_tokens, SEP, protein_tokens]` → single 6-layer Transformer encoder → CLS pool → MLP head. **1.27 M params.**
- **V2 — Early Cross-Attn (Manas).** Embedded drug/protein streams exchange info via bidirectional cross-attention block, then a 6-layer encoder consumes the fused stream. **~1.40 M params.**
- **V3 — Late Cross-Attn (Tenzin).** Independent 3-layer encoders for drug and protein, then bidirectional cross-attention, mean-pool both modalities, concat, MLP head. **~1.40 M params.**
- **V4 — Late Concat (Bhavesh).** Independent 3-layer encoders, mean-pool each side, concatenate pooled vectors, MLP head. **~1.20 M params.** (The field-default architecture.)

All four variants share the **same encoder block** (pre-norm Transformer), **same tokenizers** (regex SMILES + char-level protein), **same optimizer** (AdamW + cosine + warmup), **same head** (2-layer MLP → scalar pKi), **same loss** (MSE on pKi). The only thing that changes is **where + how** the two modalities meet.

---

## Headline Results (Phase C, 36 runs, mean ± std over 3 seeds)

| Variant | Random MSE | Cold-Drug MSE | Cold-Target MSE |
|---|---|---|---|
| V1 Early Concat | 1.004 ± 0.031 | 1.476 ± 0.029 | 1.360 ± 0.197 |
| **V2 Early X-Attn** | **0.948** ± 0.023 ★ | 1.432 ± 0.170 | **1.248** ± 0.187 ★ |
| V3 Late X-Attn  | 1.030 ± 0.039 | **1.410** ± 0.129 ★ | 1.549 ± 0.131 |
| V4 Late Concat  | 1.119 ± 0.018 | 1.465 ± 0.178 | 1.467 ± 0.069 |

**V2 wins 2 of 3 splits, V3 wins 1, V1 + V4 (concat variants) win zero.** Late fusion (V3, V4 — the field default) is never optimal across all splits. → see [`poster_figures/diagram_10b_mse_per_split.png`](poster_figures/diagram_10b_mse_per_split.png) and [FINDINGS.md](FINDINGS.md).

---

## The 4 Phases

| Phase | What | Status | Output |
|---|---|---|---|
| **A — Individual Tuning** | Each variant runs ~22 sweeps on a 10k subset, 15 epochs, to learn its own sensitivities | ✅ DONE (88 sweeps total) | `outputs/sweeps/v{1,2,3,4}_*_fast/results/results.csv` |
| **B — Fair-Config Negotiation** | Lock a single hyperparameter set inside every variant's "acceptable zone" | ✅ DONE (solo, see [PHASE_B_DECISION.md](PHASE_B_DECISION.md)) | [`configs/phase_c_fair.yaml`](configs/phase_c_fair.yaml) |
| **C — Controlled Final Runs** | All 4 variants × 3 splits (random / cold-drug / cold-target) × 3 seeds = 36 runs on full BindingDB Ki | ✅ DONE (36 / 36) | `outputs/phase_c/phase_c_<variant>_<split>_seed<n>/results/results.csv` |
| **D — Deep Analysis** | Attention extraction + entropy + heatmaps + CKA + error stratification | ✅ DONE | `outputs/analysis_phase_c/v{1,2,3,4}_phase_c/` (zipped) |

The single locked Phase C config (from [PHASE_B_DECISION.md](PHASE_B_DECISION.md)):
```yaml
d_model: 128, n_heads: 4, batch_size: 64, lr: 3e-4, dropout: 0.1
n_layers: 6 (V1, V2 shared) / 3 per side (V3, V4)
epochs: 30, warmup: 500 steps, cosine LR schedule
```

---

## Repository Structure

```
ComparisionPDI/                          # the GitHub repo (cloned locally on HPC at /scratch/$USER/ComparisionPDI)
├── README.md                            # ← this file
├── POSTER.md                            # poster draft (sections, claims, figures)
├── FINDINGS.md                          # headline results + 7 claims
├── INDEX.md                             # map of every doc/script/artifact
├── CLEANUP_AT_END.md                    # what to delete at project end
├── PHASE_B_DECISION.md                  # locked fair config + rationale
├── PHASE_A_4VARIANT_COMPARISON.csv      # full Phase A leaderboard (4 vars × 26 configs)
│
├── configs/
│   └── phase_c_fair.yaml                # the single source of truth for Phase C runs
│
├── src/                                 # core code (Tenzin's scaffolding + variant impls)
│   ├── data/
│   │   ├── dataset.py                   # BindingDBKiDataset — reads TSV, tokenizes
│   │   ├── tokenizers.py                # SMILESTokenizer + ProteinTokenizer
│   │   ├── splits.py                    # random / cold-drug / cold-target split fns
│   │   └── collate.py                   # collate_fn → 5-tuple (drug_t, drug_m, prot_t, prot_m, y)
│   ├── models/
│   │   ├── __init__.py                  # build_model(variant_name, **kwargs)
│   │   └── variants/
│   │       ├── early_concat.py          # V1
│   │       ├── early_crossattn.py       # V2
│   │       ├── late_crossattn.py        # V3
│   │       └── late_concat.py           # V4
│   ├── training/
│   │   └── trainer.py                   # train loop, val/CI eval, history.csv writer
│   └── utils/                           # seeds, logging, metrics
│
├── scripts/
│   ├── train.py                         # entry point: --variant, --split, --seed, --d_model, --epochs ...
│   ├── extract_for_analysis.py          # ORIGINAL Phase D extract (had ckpt-key bug)
│   ├── extract_for_analysis_v3.py       # second attempt (still had bug)
│   ├── extract_for_analysis_v4.py       # FIXED — recognises model_state_dict key
│   ├── extract_phase_d_from_phase_c.py  # extract from Phase C ckpts (final pass, all 4 variants)
│   └── extract_and_summarize_phase_d.py # Mac-side: unzip → entropy summary → drop bulky npys
│
├── hpc_phase_c/                         # Phase C HPC templates
│   ├── run_phase_c.sbatch               # parameterised: <variant> <split> <seed>
│   └── submit_phase_c_all.sh            # fires all 36 jobs
│
├── hpc_late_concat/                     # V4 HPC templates (legacy from Phase A)
│   ├── run_lc_a100_fast.sbatch
│   ├── phase_a_sweep.sh                 # 22 V4 sweeps
│   └── run_phase_d_extract_v4.sbatch    # final Phase D job
│
├── hpc_early_concat/                    # V1 templates
│   ├── run_v1_a100_fast.sbatch
│   └── sweep_v1.sh
│
├── hpc_early_crossattn/                 # V2 templates
│   ├── run_ec_a100_fast.sbatch
│   └── sweep_v2.sh
│
├── hpc_late_crossattn/                  # V3 templates
│   ├── run_v3_a100_fast.sbatch
│   └── sweep_v3.sh
│
├── outputs/                             # ALL EXPERIMENT OUTPUTS (gitignored — only results.csv pushed)
│   ├── sweeps/                          # Phase A: 88 dirs, e.g. v1_baseline_fast/{results,checkpoints,logs}
│   ├── phase_c/                         # Phase C: 36 dirs, e.g. phase_c_late_concat_random_seed42/
│   ├── analysis_phase_c/                # Final Phase D extraction artifacts
│   └── binding_db_stats/                # pKi histogram + length distributions
│
├── dataset/BindingDB/
│   └── BindingDB_PDSPKi.tsv             # ~64 MB, Git LFS, 27,715 Ki measurements
│
├── poster_figures/                      # 27 figures, PNG + SVG
│   ├── build_all.py                     # SINGLE script that regenerates every figure
│   ├── diagram_03_matrix.png            # 2x2 hero
│   ├── diagram_04_architectures.png     # all 4 variants side-by-side
│   ├── diagram_06_dataset_summary.png   # BindingDB stat cards
│   ├── diagram_07_length_and_pki_distribution.png
│   ├── diagram_08_split_strategy.png    # random / cold-drug / cold-target schematic
│   ├── diagram_10b_mse_per_split.png    # ★ headline figure
│   ├── diagram_11_ci_per_split.png
│   ├── diagram_12_loss_curves.png
│   ├── diagram_13_predicted_vs_true.png
│   ├── diagram_15_sensitivity_4variant.png
│   ├── diagram_16_attention_entropy.png
│   ├── diagram_17_attention_heatmap.png
│   ├── diagram_18_cka_matrix.png        # ★ behavioral-similarity finding
│   ├── diagram_27_error_stratification.png
│   ├── diagram_36_loopholes.png         # 9 problems with DTI literature
│   ├── diagram_37_norm_vs_us.png        # field-norm vs our-choice
│   └── ...                              # 12 more (architecture, market, milestones, etc.)
│
├── phase_d_artifacts/                   # large extraction zips (~14 GB total) — Mac-only, NOT in repo
│   └── v{1,2,3,4}_phase_c.zip           # the valid extraction (~1-3.5 GB each)
│
├── phase_d_summaries/                   # summarised Phase D (entropy + sample heatmaps + predictions)
│   └── v{1,2,3,4}_phase_c/{entropy_summary.npz, sample_attn.npz, predictions.npy, truth.npy, meta.json}
│
└── HPC_*.md, *.md                       # runbooks (paste-and-go HPC blocks for reproducibility)
```

> **What's in git vs what isn't:**
> - **In git**: code, configs, sbatch templates, results.csv files (small), poster figures (PNG/SVG), all .md docs, BindingDB stat summaries.
> - **Gitignored**: `outputs/sweeps/*/checkpoints/`, `outputs/phase_c/*/checkpoints/`, `outputs/analysis*/attn_*.npy` (giants), `phase_d_artifacts/*.zip` (Mac-only).

---

## Workflow — How Data Flows Through the System

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  dataset/BindingDB/BindingDB_PDSPKi.tsv  (27,715 Ki measurements)    │
│                            │                                         │
│                            ▼                                         │
│              src/data/dataset.py  →  src/data/collate.py             │
│                            │                                         │
│                            ▼                                         │
│              ┌───────────────────────────────┐                       │
│              │  scripts/train.py              │                       │
│              │  --variant --split --seed      │                       │
│              │  --epochs --d_model --lr ...   │                       │
│              └───────────────────────────────┘                       │
│                            │                                         │
│                ┌───────────┼────────────┐                            │
│                ▼           ▼            ▼                            │
│      results.csv     history.csv    best_model.pt                    │
│   (best val MSE)  (per-epoch CI)   (full state_dict)                 │
│                                                                      │
│                            │                                         │
│         (Phase D)          ▼                                         │
│   scripts/extract_phase_d_from_phase_c.py                            │
│                            │                                         │
│                ┌───────────┼─────────────┐                           │
│                ▼           ▼             ▼                           │
│        predictions.npy  attn_*.npy    drug_mask.npy                  │
│                            │                                         │
│       (Mac-side)           ▼                                         │
│   scripts/extract_and_summarize_phase_d.py                           │
│                            │                                         │
│                            ▼                                         │
│           phase_d_summaries/v*/entropy_summary.npz                   │
│                            │                                         │
│                            ▼                                         │
│           poster_figures/build_all.py  →  27 PNG + SVG               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start — Reproducing Phase C (HPC)

```bash
# 1. Clone the repo on NYU HPC (cloud-burst scratch).
ssh bg2896@gw.hpc.nyu.edu
cd /scratch/$USER
git clone https://github.com/bhaveshgupta01/ComparisionPDI.git
cd ComparisionPDI

# 2. Set up the venv (one-time).
bash hpc_late_concat/setup_venv.sh
source .venv/bin/activate

# 3. Run a single Phase C job to verify the pipeline.
sbatch hpc_phase_c/run_phase_c.sbatch early_concat random 42
squeue -u $USER

# 4. Once that finishes, fire all 36 jobs.
bash hpc_phase_c/submit_phase_c_all.sh

# 5. After ~2 hours, check progress.
find outputs/phase_c -name results.csv | wc -l   # target: 36
```

GPU usage for the full sweep: ~18-20 GPU-hours. NYU HPC budget: 300 GPU-hours per student.

---

## Quick Start — Regenerating All 27 Figures (Mac, no GPU)

```bash
cd ~/CodeFiles/DTI_MLFinalProject
python3 -m pip install numpy matplotlib    # one-time
python3 poster_figures/build_all.py
ls poster_figures/diagram_*.png            # 27 PNGs (and 27 SVGs)
```

The script auto-skips Phase C / Phase D figures when their data isn't on Mac yet (e.g. running before `outputs/phase_c/` is rsync'd from HPC). It always builds the conceptual figures (architecture, design matrix, market panel, etc.) regardless.

---

## Where Each Component Lives — A File Index for Teammates

### "I want to add a new variant V5"
1. Write `src/models/variants/v5.py` — implement the model (forward signature: `(drug_tokens, drug_mask, prot_tokens, prot_mask) -> Tensor`).
2. Register it in `src/models/__init__.py`'s `build_model()`.
3. Add it to `hpc_phase_c/run_phase_c.sbatch`'s case statement (set `NLAYERS`).
4. Update [`configs/phase_c_fair.yaml`](configs/phase_c_fair.yaml)'s `variants:` list.
5. Re-fire Phase C: `sbatch hpc_phase_c/run_phase_c.sbatch v5 random 42` etc.

### "I want to add a new figure"
1. Write a `def diagram_NN_<name>()` function in `poster_figures/build_all.py`.
2. Use `_load_phase_c()` for Phase C results, `_load_summary(variant)` for Phase D summaries.
3. Add the call to `main()` near the bottom.
4. Run `python3 poster_figures/build_all.py` to regenerate.
5. The script auto-saves both PNG (180 DPI) and SVG.

### "I want to change the Phase C config"
1. Edit `configs/phase_c_fair.yaml`.
2. Edit the matching `python scripts/train.py` flags in `hpc_phase_c/run_phase_c.sbatch`.
3. Re-fire all 36 jobs (`bash hpc_phase_c/submit_phase_c_all.sh`).
4. After they finish, regenerate figures.

### "I want to add a new split"
1. Implement `<new_split>_split(ds, seed)` in `src/data/splits.py`.
2. Add `<new_split>` to the SPLITS array in `hpc_phase_c/submit_phase_c_all.sh`.
3. Re-fire Phase C.

### "I want to understand a specific finding from the poster"
- Open [FINDINGS.md](FINDINGS.md) — has 7 claims with the supporting numbers.
- Each claim points to a `diagram_NN_*.png` for the visual.

### "I want to debug Phase D extraction"
- The chain of attempts is documented in [HPC_PHASE_D_BUGFIX.md](HPC_PHASE_D_BUGFIX.md).
- Final working script: `scripts/extract_phase_d_from_phase_c.py` (uses Phase C checkpoints, loads cleanly).
- Earlier broken attempts (kept for context): `extract_for_analysis_v{2,3,4}.py`.

### "I want to see what HPC commands were used at each phase"
- [HPC_NEXT_STEPS.md](HPC_NEXT_STEPS.md) — first runbook
- [HPC_FOLLOWUP_FIXES.md](HPC_FOLLOWUP_FIXES.md) — V1/V3 push fixes
- [HPC_PHASE_C_RUN.md](HPC_PHASE_C_RUN.md) — Phase C full run
- [HPC_PARALLEL_WHILE_WAITING.md](HPC_PARALLEL_WHILE_WAITING.md) — parallel jobs while pilots ran
- [HPC_PHASE_D_BUGFIX.md](HPC_PHASE_D_BUGFIX.md) — Phase D extraction
- [HPC_GITHUB_SYNC.md](HPC_GITHUB_SYNC.md) — git push workflow

---

## Glossary

- **DTI** — Drug-Target Interaction. Predicting how strongly a drug binds to a protein.
- **pKi** — −log₁₀(Ki / 10⁹). Higher = stronger binding. Our regression target. Range in BindingDB: 3.82–12.46.
- **Ki** — Inhibition constant (nM). Lower = stronger binding.
- **CI** — Concordance Index. Probability the model orders two random drug-protein pairs correctly. 0.5 = chance, 1.0 = perfect.
- **MSE** — Mean Squared Error in pKi units.
- **Cold split** — Train/test split where some entity is *never* seen during training. Cold-drug = unseen chemistry. Cold-target = unseen biology.
- **CKA** — Centered Kernel Alignment. Measures behavioral similarity between two models' representations.
- **SMILES** — Text encoding of a molecule. Drug input format.
- **Pre-norm Transformer** — LayerNorm before attention/FFN; stabler than post-norm.
- **Cross-attention** — Attention where queries and keys come from *different* modalities (drug-attends-to-protein and vice versa).
- **Concatenation** — Mechanically appending modalities (either at input as `[drug, SEP, protein]` or after pooling as `[drug_pooled, protein_pooled]`).
- **Phase A / B / C / D** — Our four-phase protocol. A = individual tuning, B = fair-config lock, C = controlled final runs, D = deep mechanistic analysis.
- **Fair config** — The single hyperparameter set used by all 4 variants in Phase C, locked in [PHASE_B_DECISION.md](PHASE_B_DECISION.md).

---

## Key References

- **Pahikkala 2015**, **Mayr 2018**, **Chen 2021** — flag the random-split inflation problem in DTI literature (justifies our cold-drug + cold-target splits).
- **Vaswani 2017** — the original Transformer paper.
- **Kornblith 2019** — the CKA paper, defines the similarity measure we use in diagram 18.
- **Huang 2020/2021** (DeepPurpose, MolTrans), **Bai 2023** (DrugBAN) — DTI methods we compare against in spirit.
- **Liu 2023** (BindingDB) — our data source.

Full reference list in [POSTER.md §14](POSTER.md).

---

## Contributors & Roles

| Author | Variant ownership | Cross-cutting role |
|---|---|---|
| **Bhavesh Gupta** | V4 — Late Concat | **HPC orchestration & analysis lead** — ran all 88 Phase A sweeps + 36 Phase C runs + Phase D extractions on his quota; built `poster_figures/` (27 diagrams from a single `build_all.py` pipeline); authored [PHASE_B_DECISION.md](PHASE_B_DECISION.md) + [FINDINGS.md](FINDINGS.md) + this README. |
| **Lingwei Li** | V1 — Early Concat | Phase A sweep design, V1 architecture spec |
| **Manas Ghai** | V2 — Early Cross-Attention | V2 architecture, cross-attention block design |
| **Tenzin Tsundue** | V3 — Late Cross-Attention | **Shared scaffolding lead** — `src/data/`, `src/models/__init__.py`, `src/training/trainer.py`, original Phase D extraction prototype |

---

## Where to Look First (Reading Order)

1. **This README** — high-level orientation.
2. [POSTER.md](POSTER.md) — the deep narrative; one section per panel of the planned A0 poster.
3. [FINDINGS.md](FINDINGS.md) — headline numbers + 7 publishable claims.
4. [PHASE_B_DECISION.md](PHASE_B_DECISION.md) — how the locked Phase C config was chosen.
5. [INDEX.md](INDEX.md) — full file map (covers legacy / superseded docs too).
6. The figures in [poster_figures/](poster_figures/) — 27 finished PNGs.

If you're trying to **reproduce** results: jump to "Quick Start — Reproducing Phase C (HPC)" above.
If you're trying to **extend** the work: see "Where Each Component Lives" above.
If you're **debugging** something Phase-D-related: [HPC_PHASE_D_BUGFIX.md](HPC_PHASE_D_BUGFIX.md).

---

## Status (snapshot)

| Component | State |
|---|---|
| Phase A (88 sweeps × 4 variants) | ✅ done, results in [PHASE_A_4VARIANT_COMPARISON.csv](PHASE_A_4VARIANT_COMPARISON.csv) |
| Phase B (locked fair config) | ✅ in [configs/phase_c_fair.yaml](configs/phase_c_fair.yaml) |
| Phase C (36 runs) | ✅ all 36 complete, results in `outputs/phase_c/*/results/results.csv` |
| Phase D extraction (4 variants) | ✅ complete on Phase C checkpoints, summaries in `phase_d_summaries/v*_phase_c/` |
| Poster figures | ✅ 27 built, in [poster_figures/](poster_figures/) |
| FINDINGS.md | ✅ 7 claims locked |
| POSTER.md | ✅ 0 [TBD] markers, all sections filled |
| Poster layout (PowerPoint / Affinity / LaTeX) | ⏳ pending — figures are ready, needs design pass |
| Final report write-up | ⏳ pending |
| GPU-hours used | 57 / 300 budget (well under) |

---

*Last updated 2026-04-29. If anything's unclear, ping bhaveshgupta01@gmail.com or open an issue on the repo.*
