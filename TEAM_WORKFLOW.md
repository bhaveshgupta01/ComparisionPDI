# Team Workflow — 3-Phase Experimental Design

## The Plan (Summary)

**Phase A — Individual Exploration** (each teammate owns one variant)
- Build a small BindingDB subset (~50k–100k pairs)
- Each person tunes their assigned variant to its best
- Learn each variant's sensitivity to hyperparameters

**Phase B — Team Merge**
- Identify hyperparameters that work well across all 4 variants (the "fair comparison" config)
- Lock everything in

**Phase C — Final Runs**
- Run all 4 variants with identical config on full BindingDB, Davis, KIBA
- Only the interaction strategy differs
- Deep analysis on the comparison

This is the scientifically correct way to compare architectures. Most papers skip Phase A entirely and produce biased results where "their" variant is over-tuned.

---

## Why Phase A Matters

If you just pick hyperparameters upfront and run all 4 variants, you're implicitly biasing the comparison. Example:
- A small hidden dim might favor Early Concat (fewer params)
- A larger batch might favor Late variants (more stable gradients)
- A high dropout might hurt Cross-Attn more than Concat

Phase A lets each teammate learn: **what does MY variant need to work at all, and where does it peak?** Then Phase B finds settings where all 4 are in a "reasonable" zone (not each at their individual peak — that would be unfair in the opposite direction).

---

## Variant Assignment

| Teammate | Variant | Focus |
|----------|---------|-------|
| Person 1 | Early Concatenation | Simplest; test if it's enough |
| Person 2 | Early Cross-Attention | Token-level interaction |
| Person 3 | Late Concatenation | Separate encoding story |
| Person 4 | Late Cross-Attention | Expected strongest |

---

## Phase A Protocol (Week 1-2)

### Shared Setup (do this together, Day 1)

1. **Build the mini BindingDB subset:**
   - Download raw BindingDB (~2.8M records)
   - Filter: keep only pairs with Kd measurements (log-transform to pKd)
   - Subset: random 50k–100k pairs, stratified by affinity strength
   - Create train/val/test splits (80/10/10, random split for Phase A is fine)
   - Save as `data/processed/bindingdb_mini/` — everyone uses the same file

2. **Share baseline scaffolding:**
   - Tokenizers, data loaders, training loop, metrics — built once, used by all
   - Only the `InteractionModule` differs per teammate

3. **Agree on reporting format:**
   - Each teammate logs to the same W&B project
   - Shared Google Sheet for final results table
   - Common naming: `{variant}_{hyperparam_tag}_{seed}`

### Individual Work (Days 2-7)

Each teammate tunes their variant. Budget: 15-20 runs per person on the mini dataset.

**What to explore — full design space below.**

### Phase A Deliverable (end of Week 2)

Each teammate produces:
1. A **sensitivity table** — for each hyperparameter, how much does perf change?
2. A **best config** for their variant
3. A **minimum viable config** — the smallest settings where it still works
4. A **notes doc** — surprising findings, failure modes, what they'd do differently

---

## What Can You Alter in a Transformer?

Comprehensive taxonomy. Use this as your exploration checklist.

### 1. Tokenization & Input

| Knob | Options | What It Affects |
|------|---------|-----------------|
| SMILES tokenization | char-level / atom-level regex / BPE / SMILES-PE | Vocab size, sequence length |
| Protein tokenization | char-level / k-mer (2-3) / learned BPE | Sequence length, vocab size |
| Max sequence length | Drug: 50/100/150; Prot: 500/1000/1500 | Truncation, memory |
| Truncation strategy | head / tail / middle / binding-site-aware | What info is kept |
| Special tokens | CLS / SEP / MASK usage | Where pooling happens |
| Data augmentation | SMILES enumeration (multiple valid strings per molecule), protein random crops | Generalization |

### 2. Embedding Layer

| Knob | Options | Notes |
|------|---------|-------|
| Embedding dim (d_model) | 64 / 128 / 256 / 512 | Biggest capacity knob |
| Positional encoding | Sinusoidal (fixed) / Learned / Rotary (RoPE) / ALiBi / Relative | RoPE often best for sequences |
| Type/segment embeddings | Add embedding that marks "drug" vs "protein" | Important for early concat |
| Embedding dropout | 0 / 0.1 / 0.2 | Mild regularization |
| Weight tying | Tie input embedding with output projection | Saves params |

### 3. Attention Mechanism

| Knob | Options | Notes |
|------|---------|-------|
| Number of heads (H) | 2 / 4 / 8 / 16 | d_model must be divisible by H |
| Head dimension (d_k) | d_model / H (usually 32 or 64) | Smaller heads = more diversity |
| Attention variant | Full MHA / Multi-Query / Grouped-Query | MQA/GQA save params |
| Attention implementation | Standard / Flash Attention / Memory-efficient | Flash is faster, same math |
| Dropout on attention weights | 0 / 0.1 / 0.2 | Regularizes attention |
| Masking | Padding mask / Causal mask / Custom (for binding pockets) | Usually just padding |

### 4. Feed-Forward Network (FFN)

| Knob | Options | Notes |
|------|---------|-------|
| FFN hidden dim | 2×d_model / 4×d_model / 8×d_model | 4× is standard |
| Activation | ReLU / GELU / SiLU (Swish) / GeGLU / SwiGLU | GELU most common, SwiGLU hot in 2024-25 |
| FFN dropout | 0 / 0.1 / 0.2 | Standard location for dropout |
| FFN layers | 2 / 3 (bottleneck variants) | 2 is standard |

### 5. Layer Structure

| Knob | Options | Notes |
|------|---------|-------|
| Number of transformer layers | 2 / 4 / 6 / 8 / 12 | Depth vs compute tradeoff |
| Pre-norm vs post-norm | Pre-norm (x + f(LN(x))) / Post-norm (LN(x + f(x))) | Pre-norm more stable |
| Normalization type | LayerNorm / RMSNorm / DeepNorm | RMSNorm slightly faster |
| LayerNorm epsilon | 1e-5 / 1e-6 | Rarely matters |
| Stochastic depth (LayerDrop) | 0 / 0.1 / 0.2 | Randomly skip layers in training |
| Residual scaling | 1.0 / learnable α / ReZero | Tiny effect usually |

### 6. Pooling / Readout

| Knob | Options | Notes |
|------|---------|-------|
| Pooling strategy | CLS token / Mean / Max / Attention-pooling / Sum | Mean or CLS most common |
| Pool before/after last norm | Before / After | Minor effect |
| Multi-pool concat | Combine mean+max+CLS | Can help |

### 7. Prediction Head

| Knob | Options | Notes |
|------|---------|-------|
| Head depth | 1 linear / 2-layer MLP / 3-layer MLP | 2-layer MLP default |
| Hidden dim | 128 / 256 / 512 | Usually ≥ d_model |
| Head dropout | 0.1 / 0.2 / 0.3 | More aggressive than encoder dropout |
| Output activation | None (regression) / Sigmoid (if classifying) | None for pKd |

### 8. Optimization

| Knob | Options | Notes |
|------|---------|-------|
| Optimizer | Adam / AdamW / LAMB / Lion | AdamW standard |
| Peak learning rate | 1e-5 / 5e-5 / 1e-4 / 3e-4 / 1e-3 | Biggest training knob |
| Weight decay | 0 / 1e-5 / 1e-4 / 0.01 | AdamW with 0.01 common |
| Warmup | None / 1% / 5% / 10% of steps | Cosine warmup then decay |
| LR schedule | Constant / Cosine / Step / Polynomial | Cosine most common |
| Batch size | 32 / 64 / 128 / 256 | Larger = stabler grads |
| Gradient accumulation | 1 / 2 / 4 steps | Simulate bigger batches |
| Gradient clipping | None / 1.0 / 5.0 | Clip at 1.0 is safe |
| Mixed precision | fp32 / fp16 / bf16 | bf16 if supported; fp32 for small models |
| Label smoothing | N/A for regression | — |

### 9. Regularization

| Knob | Options | Notes |
|------|---------|-------|
| Encoder dropout | 0 / 0.1 / 0.2 / 0.3 | Applied to attention + FFN + residual |
| Weight decay | See above | |
| Early stopping patience | 5 / 10 / 20 epochs | Avoid overfitting |
| SMILES augmentation | Random canonical variants per epoch | Data-level regularization |
| EMA of weights | Keep exponential moving average | Improves final checkpoint |

### 10. DTI-Specific Design Choices

| Knob | Options | Notes |
|------|---------|-------|
| Shared vs separate encoders (late) | Share weights / Separate | Late variants usually separate |
| Encoder depth split (late) | 3+3 / 2+4 / 4+2 (drug+prot) | Proteins longer → may need more |
| Cross-attn depth | 1 / 2 / 3 layers of X-attn | More = more interaction |
| Bidirectional vs unidirectional X-attn | Both ways / drug→prot only | Bidirectional standard |
| Residual in X-attn | With / Without | With is safer |
| Pocket-aware attention mask | Unmask known binding site / full | Helps if PDBbind info available |

### 11. Multi-Modal Add-Ons (Phase 4 stuff)

| Knob | Options |
|------|---------|
| Molecular graph branch | Add GIN/GAT/GraphTransformer encoder |
| 3D structure branch | SchNet/EGNN on AlphaFold coords |
| Pre-trained SMILES encoder | ChemBERTa frozen / fine-tuned |
| Pre-trained protein encoder | ESM-2 frozen / fine-tuned (LoRA) |

---

## Phase A Exploration Strategy (per teammate)

Don't grid-search everything — you'll run out of compute. Use **coarse-to-fine** with a budget of ~20 runs:

### Round 1 — Learning Rate + Batch Size (4 runs)
Fix everything else at sensible defaults. Sweep LR in {5e-5, 1e-4, 3e-4} × batch size {32, 64, 128}. Pick the best 2 combos.

### Round 2 — Capacity (4 runs)
Vary d_model ∈ {64, 128, 256} × total layers ∈ {4, 6, 8}. Note: the layer count must be the same across variants in Phase C, so log how your variant scales with capacity.

### Round 3 — Regularization (3 runs)
Vary dropout ∈ {0.1, 0.2, 0.3}. Pick the sweet spot.

### Round 4 — Variant-specific knobs (4 runs)
- **Early Concat:** type embeddings on/off; pool strategy
- **Early X-Attn:** cross-attn depth (1/2/3 layers); bidirectional or drug→prot only
- **Late Concat:** encoder depth split (3+3 vs 2+4 vs 4+2); pooling strategy
- **Late X-Attn:** X-attn placement (between encoders vs after both)

### Round 5 — Sensitivity study (5 runs)
For the 3-5 hyperparameters that mattered most, run each at {-1 std, mean, +1 std} to get sensitivity. This is what you'll bring to Phase B negotiation.

### Output per teammate
A table like:

| Hyperparameter | My best value | Sensitivity | Min viable | Max useful |
|----------------|---------------|-------------|------------|------------|
| d_model | 256 | High | 128 | 512 |
| n_layers | 6 | Medium | 4 | 8 |
| learning_rate | 1e-4 | High | 5e-5 | 3e-4 |
| dropout | 0.2 | Low | 0.1 | 0.3 |
| batch_size | 64 | Low | 32 | 128 |
| ... | ... | ... | ... | ... |

---

## Phase B Protocol (Week 2-3)

### The Team Meeting

Bring your sensitivity tables. Walk through each hyperparameter:

**Rule 1 (Fairness):** Pick values in each variant's "acceptable zone" (not at anyone's peak). If Early Concat peaks at d_model=128 and Late X-Attn at d_model=256, pick d_model=192 or check if 128 is acceptable for Late X-Attn.

**Rule 2 (Total params ≈ equal):** Since variants have inherently different param counts, aim for ~10% variance across variants. Tune d_model or FFN width per variant to match.

**Rule 3 (Document every decision):** "We chose dropout=0.2 because all 4 variants performed within 5% of their best there."

**Rule 4 (Negotiate, don't vote):** If one variant needs an unusual setting, either (a) accept the setting for everyone or (b) treat the variant as handicapped and flag it in the paper.

### Deliverable — The "Fair Config"

One YAML file: `configs/fair_comparison.yaml`. This is the only config used in Phase C, with only the `variant: {early_concat|early_crossattn|late_concat|late_crossattn}` field differing.

---

## Phase C Protocol (Week 3-5)

### The Experiment Matrix

4 variants × 3 datasets (BindingDB full / Davis / KIBA) × 3 splits (random / cold-drug / cold-target) × 5 seeds = **180 runs.**

Distribute across team members' HPC quotas (each of you has 300 GPU-hours).

### What to Vary in Phase C

**Only two things:**
1. The variant (obviously)
2. The dataset × split × seed (for statistical power)

Everything else = locked from `fair_comparison.yaml`.

### What NOT to do in Phase C

- Don't tweak hyperparameters mid-experiment. Even if you see one variant struggling, don't rescue it. That's a finding.
- Don't cherry-pick seeds. Report all 5.
- Don't compare against different paper's baselines with different splits. Use your own controlled Davis/KIBA splits.

---

## Concrete Next Steps for Your Team

### This Week
1. Fork the repo, everyone clones it, we set up GitHub / shared drive
2. **Person 1** builds the BindingDB subset script (shared asset)
3. **Person 2** builds the shared training scaffolding (shared asset)
4. **Person 3** builds the shared evaluation + logging (shared asset)
5. **Person 4** builds the 4 `InteractionModule` stubs (shared asset)
6. Run a baseline training with the default config for each variant on mini BindingDB — make sure everyone's pipeline is identical

### Next Week
7. Phase A individual exploration begins
8. Daily 15-min standups to share surprises
9. Shared W&B project so everyone can see each other's runs

### Week 3
10. Phase B meeting: negotiate fair config
11. Lock `configs/fair_comparison.yaml`
12. Split Phase C matrix across team members by HPC quota

### Weeks 4-5
13. Run Phase C on HPC
14. Begin Phase D (deep analysis) in parallel

---

## Questions to Get Aligned on Now

1. Who is which variant?
2. What size should the mini BindingDB subset be? (Suggest 50k; big enough to be meaningful, small enough to train in 30 min)
3. Shared W&B project name?
4. Shared GitHub repo URL?
5. How do you handle disagreements in Phase B? (Suggest: majority vote; if tied, simple-architecture wins to avoid param bloat)
6. Who writes which part of the final paper?

Get these answered before any code is written — they'll save you fights later.
