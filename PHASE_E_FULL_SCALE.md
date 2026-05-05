# Phase E — Full-Scale Re-Run

> **Date:** 2026-04-29 (post-poster)
> **Goal:** re-establish the Phase C findings on a model and an evidence base big enough that no reviewer can dismiss the result as "small-model artifact". Convert the poster's *Limitations* section into *Achieved* where feasible.
> **Status of Phase C (the work this builds on):** ✅ done. Locked numbers in [FINDINGS.md](FINDINGS.md). 36 / 36 runs on full BindingDB (27,715 pairs, 30 epochs, d=128, 6 total layers, 3 seeds × 3 splits).

---

## 0. Memory correction (read this first)

The recall "subset of data + 3 layers" is half right.

| | Phase A (sweeps) | Phase C (final, locked) |
|---|---|---|
| Data | 10k subset | **Full 27,715 pairs** ✅ |
| Epochs | 15 | 30 |
| Seeds | 1 | 3 |
| Splits | random | random + cold_drug + cold_target |
| d_model | swept | 128 |
| n_layers | swept | V1/V2 = 6 shared, **V3/V4 = 3 per side (= 6 total)** |
| Params | varied | 1.27 – 4.1 M |

**Phase C was already at full data scale.** What was modest was **model size (d=128, 1.27-4.1M params)**, not dataset size. Phase E targets the model-size + pre-training + ablation axes.

---

## 1. Compute budget

| Bucket | GPU-hours |
|---|---|
| Allocation per student | 300 |
| Used through Apr 29 | ~55 |
| **Remaining** | **~245** |
| **Phase E budget (target)** | **~155** (leaves ~90-hr buffer) |

---

## 2. The five sub-experiments

| ID | Name | Scientific value | Code needed | GPU-hrs | Priority |
|---|---|---|---|---|---|
| **E1** | **XL Re-Run** — d=256, n_layers=12 (or 6/side), h=8, bs=32+grad-accum, 60 epochs, 5 seeds | Robustness check at 7-9× param count. Either confirms or refutes the "fusion stage > mechanism" thesis at scale. | None — config + sbatch only | **75-90** | **🔥 fire today** |
| **E2** | **Pretrained Encoders** — ChemBERTa (drug) + ESM-2 t6/t12 (protein), Phase C config rerun | Tests the most-cited limitation: does pretrained representation neutralize fusion-stage choice? | New encoder wrappers + transformers install | **40-60** | week 2 |
| **E3** | **Davis + KIBA cross-check** — same fair config on smaller benchmarks | Single-dataset limitation removed. Kinase-only (Davis) and broader (KIBA) cross-test. | New data loaders | **8-12** | week 1 (parallel with E1) |
| **E4** | **Causal Ablations** — head ablation, layer ablation, representation swap | Fills the *intentionally-deferred* mechanistic gap from POSTER §10C. Inference-only on Phase C ckpts. | New analysis script (no training) | **3-6** | week 1 (parallel) |
| **E5** | **Mask-aware Attention Entropy** — re-extract on Phase C ckpts ignoring pad tokens | Fixes pad-token contamination admitted in POSTER §13. Confirms or refutes V1-vs-V2 ranking. | 5-line patch to `extract_deep_v1.py` | **<1** | week 1 (parallel) |
| **Total** | | | | **~125-170** | |

Out-of-scope (would push past budget or need weeks of dev):
- **3D structure / AlphaFold pocket GNN** — would require wholesale architecture redesign + PDB data integration.
- **PDBbind binding-site recovery** — requires structural alignment, not retrofittable from sequence-only models.
- **Generative agentic loop** — orthogonal direction.

---

## 3. E1 — XL Re-Run (READY TO DISPATCH)

### What changes vs Phase C

| Knob | Phase C | Phase E1 (XL) | Why |
|---|---|---|---|
| d_model | 128 | **256** | Tests scale dependency of the finding |
| n_layers (V1/V2 shared) | 6 | **12** | Match d_model upgrade |
| n_layers (V3/V4 per side) | 3 | **6** | Same total = 12 transformer blocks |
| n_heads | 4 | **8** | Standard d/h ratio preserved |
| d_ff | 512 | **1024** | 4× d_model preserved |
| head_hidden | 256 | **512** | Doubled |
| batch_size | 64 | **32** + grad-accum 2 | d=256 OOMs cross-attn at bs=64; effective bs=64 preserved |
| epochs | 30 | **60** | Bigger model needs more steps |
| seeds | 3 | **5** {42, 123, 456, 789, 2024} | Tighter error bars |
| splits | 3 | 3 (unchanged) | random + cold_drug + cold_target |

**Param count** (estimated):
- V1 ~9.5M, V2 ~10.5M, V3 ~10.5M, V4 ~9.0M (vs Phase C 1.27-4.1M)

**Job count:** 4 variants × 3 splits × 5 seeds = **60 runs**.

### Files dropped (in repo)

- [`configs/phase_e_xl.yaml`](configs/phase_e_xl.yaml) — locked spec
- [`hpc_phase_e/run_phase_e_xl.sbatch`](hpc_phase_e/run_phase_e_xl.sbatch) — single-job template
- [`hpc_phase_e/run_phase_e_xl_smoke.sbatch`](hpc_phase_e/run_phase_e_xl_smoke.sbatch) — 1-epoch smoke test (V2 = heaviest variant)
- [`hpc_phase_e/submit_phase_e_xl_all.sh`](hpc_phase_e/submit_phase_e_xl_all.sh) — fires all 60

### Dispatch sequence

```bash
# 0. From local Mac — sync new files to HPC
cd /Users/bhaveshgupta01/CodeFiles/DTI_MLFinalProject
scp configs/phase_e_xl.yaml bg2896@dtn.hpc.nyu.edu:/scratch/bg2896/ComparisionPDI/configs/
scp hpc_phase_e/*.sbatch hpc_phase_e/*.sh bg2896@dtn.hpc.nyu.edu:/scratch/bg2896/ComparisionPDI/hpc_phase_e/
# (or: git push, then `git pull` on HPC)

# 1. On HPC — smoke test first (5-8 min)
ssh bg2896@ood-burst-001.hpc.nyu.edu
cd /scratch/bg2896/ComparisionPDI
sbatch hpc_phase_e/run_phase_e_xl_smoke.sbatch
# wait, then check:
tail -f $(ls -t logs/phase_e_xl_smoke_*.out | head -1)

# 2. If smoke passes (no OOM, step time looks sane) — fire the full sweep
bash hpc_phase_e/submit_phase_e_xl_all.sh

# 3. Monitor
squeue -u $USER
sacct -u $USER -X --starttime=$(date +%Y-%m-%d) --format=JobID,JobName,Elapsed,State -P
```

### Acceptance criteria for E1

A run "succeeds" if:
1. `outputs/phase_e_xl/<tag>/results/results.csv` exists with non-NaN best_val_mse.
2. Best val MSE on random split ≤ Phase C random + 0.10 (XL should match-or-beat — anything worse is investigated).
3. No OOM in the log.

If E1 confirms Phase C ranking → "the finding is robust to 7-9× scale-up." Strongest possible writeup.
If E1 reverses the ranking → "scale neutralizes fusion-stage effect." Also publishable, also valuable.

---

## 4. E2 — Pretrained Encoders (NEEDS NEW CODE)

### Why this is the highest-impact extension

The poster's most-quoted limitation is *"Whether ChemBERTa + ESM-2 pre-training would erase the fusion-stage gap is unknown."* If E2 holds the V2/V3 advantage with pretrained weights, it elevates the contribution from "small controlled study" to "robust to the strongest representation prior available."

### Plan

| Drug encoder | Protein encoder | Param footprint | Notes |
|---|---|---|---|
| **ChemBERTa-77M-MTR** (DeepChem) | **ESM-2 t6 8M** (`esm2_t6_8M_UR50D`) | ~85M frozen + small head | Cheapest, fits in A100 40GB easily |
| ChemBERTa-77M-MTR | ESM-2 t12 35M (`esm2_t12_35M_UR50D`) | ~112M frozen | If t6 looks compute-cheap, upgrade |

**Strategy:** *Frozen* encoders → only fusion module + head trains. Removes 90% of the gradient pass; per-run wall-clock should be **comparable to Phase C** despite the bigger model.

### New code needed (in repo)

1. `src/models/encoders_pretrained.py` — wraps `transformers.AutoModel` for ChemBERTa and `esm.pretrained.esm2_*` for ESM-2; exposes the same `(B, L, d_hidden)` API as the existing `TransformerEncoder`.
2. `src/models/variants/early_concat.py` etc. — accept a `--use_pretrained` flag; if set, swap out the from-scratch encoders.
3. `requirements.txt` — add `transformers>=4.40`, `fair-esm>=2.0`.
4. Tokenizer override: ChemBERTa needs its own SMILES tokenizer; ESM-2 ditto for protein. Add a `--tokenizer_mode {scratch, pretrained}` flag.

### Dispatch (after E1 finishes)

`configs/phase_e_pretrained.yaml` + `hpc_phase_e/run_phase_e_pretrained.sbatch` to be written **after the new encoder wrappers compile and pass smoke**. Job count: 4 × 3 × 3 = 36 runs (3 seeds is enough — frozen encoders give lower seed variance).

---

## 5. E3 — Davis + KIBA Cross-Check (NEEDS NEW DATA LOADERS)

### What

Drop-in replacement of BindingDB Ki with two well-known DTI benchmarks:

| Dataset | Drugs × Proteins | Pairs | Target | Notes |
|---|---|---|---|---|
| **Davis** | 442 × 379 | ~30k | pKd | Kinases only — narrow domain |
| **KIBA** | 2,116 × 229 | ~118k | KIBA score | Broader chemistry, integrated affinity |

### New code needed

- `src/data/davis.py`, `src/data/kiba.py` — public download URLs in DeepPurpose / DeepDTA repos; tracked via Git LFS.
- `--dataset {bindingdb,davis,kiba}` CLI flag in `train.py` (one-line dispatch).

### Dispatch (parallel with E1, after data prep)

4 variants × 3 splits × 3 seeds × 2 datasets = **72 runs**, but each is much smaller wall-clock than BindingDB (Davis is ~30k pairs, KIBA is ~118k). Estimated 8-12 GPU-hrs total.

---

## 6. E4 — Causal Ablations (NO NEW TRAINING)

### What's deferred in poster §10C

> *"We did not run head ablation, layer ablation, or representation swap due to time constraints."*

All three are inference-only on the existing 36 Phase C checkpoints. No retraining.

### Three studies

| Ablation | What we learn | Cost |
|---|---|---|
| **Head ablation** — zero-out one head at a time, measure ΔMSE | Which heads carry the prediction? Does V2's cross-attn block dominate? | ~1 GPU-hr (4 variants × N_heads × forward pass) |
| **Layer ablation** — zero-out residual stream at layer L | Where does cross-modal info enter? | ~1 GPU-hr |
| **Representation swap** — swap V3's drug encoder into V4 (and vice versa); measure ΔMSE | Are the encoders interchangeable, or is the fusion logic learning encoder-specific tricks? | ~2 GPU-hrs |

### File to add

`scripts/ablate.py` — single-pass inference, writes to `outputs/phase_e_ablations/`. Mac-side post-processing builds figures `diagram_22_head_ablation.png`, `diagram_23_layer_ablation.png`, `diagram_24_rep_swap.png` (the three "deferred" entries from POSTER §10).

---

## 7. E5 — Mask-Aware Attention Entropy (TRIVIAL)

Patch to `scripts/extract_deep_v1.py`:

```python
# old: H = attn.mean(dim=1)  ; H_entropy = -(H * log(H)).sum(dim=-1)
# new (mask-aware):
mask = (~pad_mask).float()                 # (B, L)
attn_masked = attn * mask[:, None, None, :]
attn_renorm = attn_masked / attn_masked.sum(dim=-1, keepdim=True).clamp(min=1e-9)
H = attn_renorm.mean(dim=1)
H_entropy = -(H * torch.log(H + 1e-9) * mask[:, None, :]).sum(dim=-1)
H_entropy = H_entropy / mask.sum(dim=-1).clamp(min=1.0)  # per-token average
```

Re-run Phase D extraction with the patched script. Compare entropy curves to the poster's `diagram_16`. The poster claims *"V1-vs-V2 ranking is preserved"* — this verifies it.

---

## 8. Suggested execution order (week-by-week)

### Week 1 (this week, 2026-04-29 → 2026-05-05)

| Day | Action |
|---|---|
| Today | scp Phase E1 files to HPC, fire smoke test, then full E1 sweep (60 jobs) |
| Today | Apply E5 mask-aware patch on local Mac, re-run on Phase C artifacts |
| Tomorrow | Write `scripts/ablate.py` for E4 — head + layer ablation. Submit one inference job per variant. |
| Day 3 | Start E3 data loaders for Davis (smallest); test locally on Mac before HPC submit. |

### Week 2 (2026-05-06 → 2026-05-12)

| Day | Action |
|---|---|
| Day 1 | Phase E1 results harvest → update FINDINGS.md → rebuild diagrams 10b/11/12/13 with XL numbers + 5 seeds |
| Day 2-3 | E2 encoder wrappers + smoke test — start with ChemBERTa+ESM-2 t6 (cheapest) |
| Day 4-5 | E2 full 36-run sweep |
| Day 6-7 | Final FINDINGS_E.md + updated poster figures + write-up |

### Updated poster (Phase E version)

After E1 + E5 + E4 land, update:
- §9.1 headline table → 5 seeds + d=256 + 12 layers (call it the "validated headline")
- §10A entropy figure → mask-aware version
- §10C *causal interventions* → no longer "deferred" — three subsections of real findings
- §13 limitations → drop "modest scale", drop "pad-token contamination", drop "deferred causal ablations"
- Add a §9.5 *Robustness* panel: "the ranking holds at 8× scale (Phase E1)"

If E2 lands too: §13 "no pretrained encoders" can be dropped; add a §9.6 *Pretrained-Encoder Sanity Check* panel.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| E1 OOMs cross-attn variants at d=256 | Smoke test catches it before full sweep. Fallbacks: bs=16 + grad-accum=4, or drop n_layers to 8 |
| E1 takes longer than 90 min/run | `--time=2:30:00` already set; if blowout, reduce epochs to 45 (early-stop will trigger earlier anyway) |
| E1 reverses the Phase C ranking | That IS the result — poster framing pivots to "scale neutralizes fusion-stage effect" (still publishable) |
| E2 transformers install conflicts with the existing venv | Create separate `.venv_pretrained/` — no need to break what works |
| Davis / KIBA download URLs rot | Both datasets are mirrored in DeepPurpose's HuggingFace repo — use that as fallback |
| Phase E results panic-rush past poster session date | All five experiments are independent. If only E1 + E5 + E4 finish, that's already a 60% better paper than the poster. |

---

## 10. What "real readings, real results" ends up looking like

Final numbers table (Phase E1, projected — to be filled):

| Variant | Random MSE (5 seeds) | Cold-Drug MSE | Cold-Target MSE | Params (M) |
|---|---|---|---|---|
| V1 Early Concat | TBD | TBD | TBD | ~9.5 |
| V2 Early X-Attn | TBD | TBD | TBD | ~10.5 |
| V3 Late X-Attn | TBD | TBD | TBD | ~10.5 |
| V4 Late Concat | TBD | TBD | TBD | ~9.0 |

If V2 still wins at 10× the parameters → headline: *"Early cross-attention is preferred independent of model scale."*

---

## 11. Files added by this plan

```
configs/
  phase_e_xl.yaml                       (E1 spec)
hpc_phase_e/
  run_phase_e_xl.sbatch                 (E1 single job)
  run_phase_e_xl_smoke.sbatch           (E1 smoke test)
  submit_phase_e_xl_all.sh              (E1 master submitter)
PHASE_E_FULL_SCALE.md                   (this file)
```

To be added in subsequent commits:
- `configs/phase_e_pretrained.yaml`, `hpc_phase_e/run_phase_e_pretrained.sbatch` (E2)
- `src/models/encoders_pretrained.py` (E2)
- `src/data/davis.py`, `src/data/kiba.py` (E3)
- `scripts/ablate.py` (E4)
- patch to `scripts/extract_deep_v1.py` (E5)
