# Where Should Drug and Protein Meet?

> **A Width-Specific Phase Transition in Transformer Fusion for Drug-Target Interaction Prediction.**
>
> NYU CSCI-GA-2565 · Spring 2026 · final project + research paper.
> Authors: Bhavesh Gupta · Lingwei Li · Manas Ghai · Tenzin Tsundue.
> Repo: [github.com/bhaveshgupta01/ComparisionPDI](https://github.com/bhaveshgupta01/ComparisionPDI)

---

## TL;DR

We ran **144 controlled training experiments** to answer: *when a transformer predicts drug-protein binding affinity, where and how should the two modalities meet?* The 2 × 2 design crosses **fusion stage** (before vs after the encoder body) with **fusion mechanism** (concatenation vs cross-attention).

**Headline finding.** The "best" architecture flips with model width:
- At **`d_model = 128`** (small) → V2 Early Cross-Attention wins.
- At **`d_model = 256`** (production scale) → V3 Late Cross-Attention wins; V4 Late Concatenation (the field default) wins the cold-drug split.
- **Doubling depth alone does NOT trigger the reversal** (E6 control: V2 still wins at d=128, n_layers=12). The phase transition is *width-specific*.

The paper's central claim: the field's tacit consensus on "encode separately, fuse late" is **scale-dependent and width-dependent** — correct at production scale, wrong at small scale.

---

## Why this question matters

**Pre-COVID**, validating one drug-protein pair in the wet lab took **7-10 days** (dose-response curves, Ki / IC50 assays, selectivity panels). With ~10⁹ commercially-available compounds × ~20,000 human protein targets, brute-force assay was impossible.

**COVID-19 forced the timeline shorter** — repurposing pipelines, virtual screens, and target-based drug design suddenly needed ML surrogates that returned binding-affinity predictions in microseconds. The first AI-designed small molecule (Exscientia / Sumitomo's DSP-1181) entered Phase I trials in 2020.

**Pre-transformer ML on DTI plateaued at ~85 % concordance index** — early CNN (DeepDTA, DeepConv-DTI) and RNN approaches captured local sequence patterns but couldn't directly model long-range token-to-token interactions between, say, a SMILES atom and a distant binding-pocket residue.

**Transformers won** because *self-attention* lets every token attend to every other token, and *positional encoding* preserves token order. The transformer-DTI literature (MolTrans, DeepPurpose, HyperAttentionDTI, PerceiverCPI, FusionDTI, TransformerCPI, DrugBAN) routinely outperforms CNN baselines by 2-5 CI points and is now dominant.

**But the field defaulted to one architecture pattern — encode separately, fuse late — without justifying the choice.** Some papers concatenate; others cross-attend; some fuse before the encoder body, others after. No published controlled comparison has isolated *fusion stage* as the variable. The choice is folklore, not science.

We do the controlled comparison the field has skipped.

---

## What's in this repo (short version)

```
DTI_MLFinalProject/
├── README.md                      ← this file (start here)
├── REPO_GUIDE.md                  ← detailed file inventory + paths
├── PAPER_DRAFT.md                 ← consolidated 6-page paper draft (handoff to writers)
├── PAPER_OUTLINE.md               ← high-level outline + figure inventory
├── PAPER_SECTION_*.md             ← longer per-section drafts (alt source material)
├── POSTER.md                      ← original poster session content
├── poster.pdf, poster.tex         ← compiled poster (April 29 session)
├── ML_Project_Proposal.pdf        ← original proposal
├── PROJECT_OVERVIEW.md            ← high-level perspective doc
├── PROJECT_DOCUMENTATION.md       ← master engineering doc
├── TECHNICAL_SPECIFICATION.md     ← engineering spec
├── TEAM_BLUEPRINT.md              ← team conventions
├── TEAM_WORKFLOW.md               ← team workflow doc
│
│ ── Findings (numerical results) ────────────────────────────────
├── FINDINGS_E1.md                 ← scale-dependent reversal (d=128 vs d=256)
├── FINDINGS_E5.md                 ← mask-aware attention entropy
├── FINDINGS_E4.md                 ← causal layer / head / rep-swap ablations
├── FINDINGS_E6.md                 ← width-vs-depth decomposition
├── FINDINGS_CKA.md                ← CKA at d=128 and d=256
├── CAPACITY_FINDINGS.md           ← three-scale phase transition curve
├── SIGNIFICANCE_E1.md             ← paired-seed t-test matrices
│
│ ── Raw results CSVs ──────────────────────────────────────────────
├── PHASE_E1_RESULTS.csv           ← d=256, 5 seeds × 3 splits × 4 variants
├── PHASE_E1B_RESULTS.csv          ← d=192, 3 × 3 × 4 (36 rows)
├── PHASE_E6_RESULTS.csv           ← d=128 n=12/6, 3 × random × 4 (12 rows)
├── PHASE_E1_TEST_RESULTS.csv      ← test MSE + CI + Pearson on 60 E1 ckpts
│
│ ── Code (reproducibility) ────────────────────────────────────────
├── configs/                       ← YAML configs for each phase
├── scripts/                       ← Python build / analysis scripts
├── hpc_phase_e/                   ← HPC sbatch templates for Phase E
│
│ ── Compute artifacts ─────────────────────────────────────────────
├── outputs/                       ← per-run results.csv + history.csv (Phase C)
├── outputs/phase_e_ablations/     ← E4 ablation outputs (SUMMARY.csv + JSONs)
├── poster_figures/                ← all final paper / poster figures (PNG + SVG)
└── phase_d_artifacts_deep/        ← raw mechanistic-extraction artifacts
```

For a full, line-by-line file inventory, see [REPO_GUIDE.md](REPO_GUIDE.md).

---

## Final figure inventory (paper-ready)

| Fig | File | What it shows |
|---|---|---|
| 1 | `poster_figures/diagram_04_architectures.png` | The four V1-V4 architectures |
| 2 | `poster_figures/diagram_07_length_and_pki_distribution.png` | BindingDB Ki dataset summary |
| 3 (hero) | `poster_figures/diagram_phase_c_vs_e1_comparison.png` | Phase C vs Phase E1 side-by-side bars (the reversal) |
| 4 | `poster_figures/diagram_4_capacity_curve.png` | Capacity curve, d ∈ {128, 192, 256} (gradual) |
| 5 | `poster_figures/diagram_e1_per_variant_improvement.png` | Δ% improvement per variant per split |
| 6 | `poster_figures/diagram_16b_attention_entropy_mask_aware.png` | Mask-aware attention entropy |
| 7 | `poster_figures/diagram_7_layer_ablation.png` | Layer-ablation heatmap |
| 7b | `poster_figures/diagram_7b_head_ablation.png` | Head-ablation max vs sum |
| 7c | `poster_figures/diagram_7c_rep_swap.png` | V3 ↔ V4 drug-encoder swap |
| 8 | `poster_figures/diagram_8_cka_comparison.png` | CKA matrices at d=128 vs d=256 |
| 9 | `poster_figures/diagram_9_width_vs_depth.png` | **Width-vs-depth decomposition (width-specific)** |
| 10b | `poster_figures/diagram_10b_e1_mse_per_split.png` | Phase E1 headline bar chart |
| supp | `poster_figures/diagram_significance_e1.png` | Paired-seed t-test heatmap |

---

## Reproducing all results from scratch (~190 GPU-hours)

```bash
# 1. Configure
ssh bg2896@ood-burst-001.hpc.nyu.edu       # NYU HPC OOD
cd /scratch/$USER/ComparisionPDI
git pull && source .venv/bin/activate

# 2. Train (in order; each is independent)
bash hpc_phase_c/submit_phase_c_all.sh         # Phase C, 36 runs, ~18 GPU-h
bash hpc_phase_e/submit_phase_e_xl_all.sh      # Phase E1 (d=256), 60 runs, ~86 GPU-h
bash hpc_phase_e/submit_phase_e1b_all.sh       # Phase E1b (d=192), 36 runs, ~25 GPU-h
bash hpc_phase_e/submit_phase_e6_all.sh        # Phase E6 (n_layers×2), 12 runs, ~12 GPU-h

# 3. Mechanistic (inference-only, ~5 GPU-h total)
sbatch hpc_phase_e/run_phase_e4_ablations.sbatch   # ablations
sbatch hpc_phase_e/run_extract_e1.sbatch           # CKA-feature extraction at d=256
sbatch hpc_phase_e/run_eval_test_set.sbatch e1     # test-set MSE/CI/Pearson

# 4. Build figures + findings docs (Mac-side, no GPU)
python3 scripts/build_e1_findings.py            # Figs 3, 5, 10b
python3 scripts/build_capacity_curve.py         # Fig 4 + CAPACITY_FINDINGS.md
python3 scripts/build_e6_depth_axis.py          # Fig 9 + FINDINGS_E6.md
python3 scripts/e1_significance_tests.py        # significance heatmap + SIGNIFICANCE_E1.md
python3 scripts/build_ablation_figs.py          # Figs 7, 7b, 7c + FINDINGS_E4.md
python3 scripts/build_cka_comparison.py         # Fig 8 + FINDINGS_CKA.md
python3 scripts/e5_mask_aware_entropy.py        # Fig 6 + FINDINGS_E5.md
```

---

## How to read this repo (for teammates writing the paper)

**You'll want these files, in order:**

1. **`PAPER_DRAFT.md`** — the consolidated draft. Sections 1-6 with all numbers, tables, and figure references plugged in. Most of your work is prose polish + LaTeX conversion.
2. **`PAPER_OUTLINE.md`** — figure-by-figure inventory and section structure.
3. **Section drafts** (`PAPER_SECTION_*.md`) — longer, looser drafts for each section if you want alternative phrasing to pull from.
4. **All `FINDINGS_*.md`** — these are the numerical "punchlines" for each result. If you want to verify a number in the paper, this is where it came from. Each was auto-generated from a committed CSV.
5. **`POSTER.md`** — the April 29 poster session content. The narrative flow is similar (we did the poster *before* the E1 reversal landed, so the poster's framing is "fusion stage matters more than mechanism" while the paper's framing is "width-specific phase transition" — the latter is more accurate but the poster's structure is reusable for the paper's introduction and motivation).

**Suggested writing assignment split:**

| Section | Suggested writer | Source materials |
|---|---|---|
| §1 Intro + Motivation | Lingwei | `PAPER_DRAFT.md` §1 + `PAPER_SECTION_1_INTRO.md` + the COVID/transformer narrative in this README |
| §2 Method | Tenzin (scaffolding owner) | `PAPER_DRAFT.md` §2 + `PAPER_SECTION_2_METHOD.md` + `TECHNICAL_SPECIFICATION.md` |
| §3 Results | Bhavesh (analysis lead) | `PAPER_DRAFT.md` §3 + all `FINDINGS_*.md` files |
| §4 Discussion | Manas | `PAPER_DRAFT.md` §4 + `FINDINGS_E4.md` + `FINDINGS_CKA.md` for the mechanistic story |
| §5 Limitations + §6 Conclusion + References | whoever, ~1 hour | `PAPER_DRAFT.md` §5 / §6 |

Once your section is drafted, paste the consolidated text into `paper.tex` and converge on a final pass together.

---

## Key contacts

- **Bhavesh Gupta** — corresponding / presenting author, HPC + analysis lead. <bhaveshgupta01@gmail.com>
- **Course staff** — Rajesh Ranganath (instructor); Nhi Nguyen, Riya Mahesh, Siddhant Mohan (TAs).

---

## License

Code: MIT. Data: BindingDB's terms (publicly licensed, derived from peer-reviewed literature).
