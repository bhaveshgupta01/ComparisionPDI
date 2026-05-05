# Section 2 — Method (paper draft, ~1.0 page)

## 2.1 Four variants, one axis of variation

We construct four transformer DTI predictors that share embedding tables, transformer encoder block, optimizer, schedule, loss, and prediction head. **The only thing that changes across the four variants is where and how drug and protein representations interact.** Figure 1 shows all four side-by-side. Notation: `T_d` and `T_p` are the SMILES-tokenizer and amino-acid-tokenizer embeddings respectively; `Enc` is the shared pre-norm Transformer block (Vaswani et al., 2017); `MLP` is a 2-layer GELU head producing a scalar pKi.

- **V1 Early Concatenation.** Concatenate drug and protein token embeddings into a single sequence `[CLS, T_d(drug), SEP, T_p(prot)]` and pass through a single shared `Enc` body. Pool the `[CLS]` token. Cheapest at inference (one encoder body).
- **V2 Early Cross-Attention.** Run a bidirectional cross-attention block between `T_d(drug)` and `T_p(prot)` *before* feeding the concatenated stream to a shared `Enc` body. Pool `[CLS]`.
- **V3 Late Cross-Attention.** Two separate `Enc` bodies — one for drug, one for protein. After encoding, a bidirectional cross-attention block lets the two streams exchange information; mean-pool each side over non-pad tokens; concatenate; MLP head.
- **V4 Late Concatenation.** Two separate `Enc` bodies. Mean-pool each side; concatenate; MLP head. The minimalist late-fusion baseline — corresponds to the field-default architecture in published DTI transformers.

The 2 × 2 axes are *fusion stage* (before vs after encoding) × *fusion mechanism* (concatenation vs cross-attention). V4 is the field default. V1 (Early Concat) is virtually unused in published DTI work. V2 and V3 appear in slightly different forms in HyperAttentionDTI [Zhao22] and FusionDTI [...] but never in a controlled comparison against V1/V4.

**Total parameters per variant** are matched within ±10% at each scale. At `d=128`, V1 has 1.27 M parameters (single shared encoder body), V4 has 4.1 M (two encoder bodies + larger head); V2 and V3 fall between (~1.4 M). The variants are *not* matched on parameter count by design — they are matched on the *building blocks they share*, which is the architectural-question we are auditing. Total-parameter parity would force V1 to a deeper or wider encoder than V4 uses internally, conflating the fusion-stage question with depth/width.

## 2.2 Shared scaffolding

The same block, optimizer, schedule, head, and tokenization protocol is used for every variant at every scale.

| Component | Choice |
|---|---|
| Encoder block | Pre-norm Transformer (Attention → FFN, residual) [Xiong20] |
| Attention | Standard multi-head self-attention (`nn.MultiheadAttention`, `batch_first=True`) |
| FFN | Linear → GELU → Linear |
| Position encoding | Sinusoidal |
| Drug tokenizer | SMILES regex-based (BPE-free); vocab size 66 (built once from training corpus) |
| Protein tokenizer | Char-level: 20 standard amino acids + 4 special tokens (vocab 24) |
| Length caps | drug ≤ 100 tokens, protein ≤ 1200 tokens (truncated; covers 83% of drugs and 99% of proteins) |
| Pooling | Mean over non-pad tokens (or `[CLS]` for V1/V2) |
| Head | 2-layer MLP, GELU, dropout 0.1, scalar pKi |
| Loss | MSE on pKi |
| Optimizer | AdamW, β = (0.9, 0.999), weight decay 1e-2 |
| LR schedule | Cosine with 500-step warmup |
| Determinism | All seeds set (random / numpy / torch / cuDNN); `cudnn.benchmark=False` |

## 2.3 Three model scales — the controlled-capacity sweep

We instantiate the same 2 × 2 set at three model widths, holding every other architectural knob identical:

| Phase | `d_model` | `n_heads` | `d_ff` | `n_layers` (V1/V2 shared, V3/V4 per-side) | `head_hidden` | `bs` | epochs | seeds |
|---|---|---|---|---|---|---|---|---|
| C   | 128 | 4 | 512  | 6, 3+3 | 256 | 64 | 30 | 3 |
| E1b | 192 | 6 | 768  | 6, 3+3 | 384 | 32 | 45 | 3 |
| E1  | 256 | 8 | 1024 | 6, 3+3 | 512 | 32 | 45 | 5 |

`d_ff` = 4 × `d_model` and `d_head` = 32 (= `d_model` / `n_heads`) are preserved across scales. Seeds at each scale are identical *across variants*, enabling paired-seed t-tests. Total runs: 36 + 36 + 60 = 132 controlled experiments. We additionally ran a *depth-axis* control at `d=128` with `n_layers=12` (V1/V2) and `n_layers=6` per side (V3/V4) — same total-block count as `(d=256, n=6)`, isolating width-vs-depth contributions to the reversal — for 12 more runs.

## 2.4 Data

**BindingDB PSDPKi** [Liu23], November 2024 release. Filter: `Ki (nM)` present, valid `SMILES`, valid sequence; 27,715 raw rows → 21,382 valid pairs (6,333 dropped). Target = pKi = −log₁₀(Ki / 10⁹), clipped to [3, 12]; 21% of pairs are censored at pKi=5.0 from the database. We do not modify or re-weight censored measurements — they enter the regression as-is, consistent with prior practice. Sequence/string truncation is described above.

## 2.5 Splits

Three splits define increasing degrees of distribution shift:
- **Random 80/10/10** by pair. Both drug and target may appear in train and test. Sanity-check setting; *never* the most informative for deployment.
- **Cold-drug**. Drugs in test are *never* seen in train. Tests generalization to new chemistry.
- **Cold-target**. Targets in test are *never* seen in train. Hardest setting — tests generalization to new biology.

Splits are generated deterministically per seed; seed-paired splits are compared across variants at the same scale.

## 2.6 Reporting protocol

We report **best validation MSE** across epochs per run; mean ± std across seeds. **Significance tests** use paired-seed t-tests with n = 5 (E1) or n = 3 (Phase C, E1b) — paired because the same seed produces the same train/val/test partitioning across variants. Test-set MSE, concordance index (Davis et al., 2011 definition), and Pearson r are recomputed via a separate inference pass over the held-out test split for each Phase E1 checkpoint and reported in Section 3 alongside val numbers.

## 2.7 Mechanistic analyses

For interpretability we run three lenses, all reproducibly:
- **Mask-aware attention entropy** per layer per variant on a 256-pair held-out batch, averaging only over valid (non-pad) query positions (the existing literature reports pad-contaminated values that inflate entropy).
- **Centered Kernel Alignment** [Kornblith19] between attention/representation features across variant pairs at both `d=128` and `d=256`, asking whether the {early-fusion} vs {late-fusion} clustering reported at small scale survives at scale.
- **Causal ablations** on Phase C checkpoints: zero each `nn.MultiheadAttention` block's output (one at a time) and measure ΔMSE; same for individual heads; and a representation-swap experiment where V3 and V4 exchange drug encoders to test interchangeability.

## 2.8 Reproducibility

All hyperparameters live in YAML / CLI flags (no magic numbers). Vocabularies, splits, and seeds are deterministic. Environment is pinned (`requirements.txt`). Total compute footprint reported in Section 3.5. Public repository at `github.com/bhaveshgupta01/ComparisionPDI`.

## 2.9 Compute platform

NYU HPC Cloud Bursting cluster, single-A100 (40 GB) jobs on partition `c12m85-a100-1`. Phase C: ~28 min/run wall-clock. Phase E1: ~150 min/run. Phase E1b: ~110 min/run. Concurrent A100 cap ~17 jobs per submitter.
