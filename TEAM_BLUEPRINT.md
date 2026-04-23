# Team Blueprint — DTI Project (Shareable)

> **Share this doc with the team.** It defines what we all do the same, and where each person has freedom. Read once carefully — saves us pain later.

---

## Who Owns What

| # | Variant | Interaction | Stage | Owner |
|---|---------|-------------|-------|-------|
| V1 | **Early Concat** | Concatenation | Before encoder | Lingwei |
| V2 | **Early Cross-Attn** | Cross-attention | Before encoder | Manas |
| V3 | **Late Cross-Attn** | Cross-attention | After encoder | Tenzin |
| V4 | **Late Concat** | Concatenation | After encoder | Bhavesh |

Shorthand in commits/runs: V1 / V2 / V3 / V4.

---

## The 3-Phase Plan (keep this in mind)

**Phase A — Solo (now):** each of us tunes our own variant on the shared mini-BindingDB subset.
**Phase B — Merge:** we meet, compare findings, agree on a "fair config" (shared settings everyone uses).
**Phase C — Final runs:** all 4 variants with identical config on full datasets. Only the interaction differs.

Phase A is about **learning your variant**, not about winning. Don't over-tune. Report what the variant NEEDS to work, what it's SENSITIVE to, where it PEAKS.

---

## Shared Conventions (everyone follows these)

### 1. Dataset
- **Shared mini-BindingDB subset** — one of us (TBD) creates it once; rest of us download from shared drive.
- Target size: **30k pairs**, Kd-only, stratified by affinity.
- Splits fixed in a single JSON file we all use: `data/splits/bindingdb_mini/random.json` (80/10/10 → 24k train / 3k val / 3k test)
- **Do NOT make your own subset.** Fair comparison requires identical data.

### 2. Repository
- One shared GitHub repo: `<repo_url_to_be_shared>`
- Branch per person: `variant/v1-early-concat`, `variant/v2-early-xattn`, etc.
- Merge only shared code (scaffolding) to `main`. Variant-specific experiments stay on your branch.

### 3. Shared Scaffolding (one of us writes each; not the variant owner)
| Module | What it does | Owner (TBD) |
|--------|-------------|-------------|
| `src/data/` | Dataset, tokenizers, splits, DataLoaders | — |
| `src/models/base.py` | BaseDTIModel abstract class, shared embedding, encoder blocks | — |
| `src/training/trainer.py` | Training loop, checkpointing, W&B logging | — |
| `src/training/metrics.py` | MSE, CI, Pearson, Spearman | — |
| `src/utils/` | Seeds, config loading, logging helpers | — |

Assign these 5 modules so one person owns each. Everyone else builds on top. **Write these before starting Phase A.**

### 4. Naming / Tagging
- W&B project: `dti-early-vs-late` (shared)
- Run name: `v{1|2|3|4}_{descriptor}_{seed}` e.g. `v4_d128_l6_lr1e-4_s42`
- Tag every run with your variant, tunable being tested, and phase (A/B/C)

### 5. Reporting
- Shared Google Sheet / Notion page with one row per run
- Columns: run_name, variant, d_model, n_layers, lr, batch_size, dropout, val_mse, val_ci, train_time_s
- Update after each run — don't batch at the end

---

## What's LOCKED (everyone uses the same)

Settings frozen for Phase A — do NOT change these:

| Thing | Value |
|-------|-------|
| Dataset | mini-BindingDB (shared subset) |
| Train/val/test split | shared JSON |
| Tokenizer (SMILES) | atom-level regex (shared tokenizer class) |
| Tokenizer (protein) | char-level, 25 tokens |
| Max drug length | 100 |
| Max protein length | 1200 |
| Loss function | MSE on pKd |
| Metrics | MSE + Concordance Index |
| Seeds to test | 42, 123, 456 (3 seeds minimum) |
| Optimizer | AdamW |
| Early stop patience | 15 epochs |
| Gradient clip | 1.0 |
| Framework | PyTorch 2.1, CUDA 12.1 |
| Mixed precision | bf16 (if HPC GPU supports) else fp32 |

If you have a strong reason to deviate from any of these, raise it in the team chat BEFORE running.

---

## What You CAN Tune (Phase A design space)

You have a budget of ~20 runs. Use them on your variant. Here's the taxonomy of everything you could touch.

### Tier 1 — Biggest Impact (explore these first, ~8 runs)

| Knob | Suggested Sweep | Notes |
|------|-----------------|-------|
| **Learning rate** | 5e-5, 1e-4, 3e-4 | Single most important training knob |
| **d_model** (embedding dim) | 64, 128, 256 | Biggest capacity knob |
| **n_layers** (total transformer layers) | 4, 6, 8 | Depth matters |
| **batch_size** | 32, 64, 128 | Larger = stabler gradients, more VRAM |

### Tier 2 — Medium Impact (~6 runs)

| Knob | Options | Notes |
|------|---------|-------|
| **Positional encoding** | sinusoidal / learned / RoPE | RoPE is modern default |
| **FFN activation** | GELU / SiLU / SwiGLU | GELU baseline; SwiGLU often better |
| **Number of attention heads** | 2, 4, 8 | d_model must be divisible by it |
| **FFN hidden ratio** | 2× / 4× d_model | 4× standard |
| **Dropout** (attn + FFN + residual) | 0.1, 0.2, 0.3 | Regularization sweet spot |
| **Weight decay** | 0.0, 0.01, 0.1 | AdamW default is 0.01 |
| **Warmup ratio** | 0%, 5%, 10% | 5% linear warmup common |
| **LR schedule** | constant / cosine / step | Cosine most common |

### Tier 3 — Small Impact, Still Worth Noting (~3 runs)

| Knob | Options |
|------|---------|
| **Pre-norm vs Post-norm** | Pre-norm more stable; default |
| **LayerNorm vs RMSNorm** | RMSNorm slightly faster |
| **Pooling** | CLS token / mean / attention pooling |
| **LayerDrop (stochastic depth)** | 0.0, 0.1 |
| **Gradient accumulation** | 1, 2, 4 |
| **EMA of weights** | on/off |
| **Label smoothing** | N/A (regression task) |

### Tier 4 — Your Variant-Specific Knobs (~3 runs)

**V1 — Early Concat (Lingwei):**
- Separator/type embeddings: add learnable embedding marking "drug" vs "protein"? (on/off)
- Concat order: `[CLS] drug [SEP] protein [SEP]` vs `[CLS] protein [SEP] drug [SEP]`
- Pooling location: CLS only / mean of all / mean of drug+mean of protein concat

**V2 — Early Cross-Attn (Manas):**
- Cross-attn depth: 1 / 2 / 3 X-attn layers before encoder
- Bidirectional vs single-direction (drug→prot only)
- Residual connection around X-attn: yes/no
- Share X-attn params between directions: yes/no

**V3 — Late Cross-Attn (Tenzin):**
- Encoder depth split (drug/prot): 3+3 / 2+4 / 4+2
- X-attn depth: 1 / 2 / 3 layers after encoders
- Bidirectional vs single-direction
- Pool before X-attn vs after X-attn

**V4 — Late Concat (Bhavesh):**
- Encoder depth split: 3+3 / 2+4 / 4+2
- Pooling strategy per encoder: mean / max / CLS / attention pooling
- Pool-then-concat vs concat-then-pool
- Projection layer before concat: yes/no

---

## What You DO NOT Tune in Phase A

- Dataset composition (locked subset)
- Tokenizers
- Loss function
- Evaluation metrics
- Seeds other than 42/123/456

Also don't mess with adding new modalities (GNN, ESM-2, 3D) yet — that's Phase D (extensions).

---

## Phase A Runbook (per teammate)

### Step 1 — Baseline (1 run)
Run your variant with these defaults. Confirm it trains.
```
d_model=128, n_layers=6, n_heads=4, ffn_ratio=4, dropout=0.1,
lr=1e-4, batch_size=64, warmup=5%, cosine schedule, pre-norm,
positional_encoding=sinusoidal, activation=GELU, pooling=mean
```
This is your reference point. Record val_mse + val_ci.

### Step 2 — Tier 1 Sweep (8 runs)
Sweep LR × d_model × n_layers. Use coordinate descent:
1. Fix d_model=128, n_layers=6. Try lr ∈ {5e-5, 1e-4, 3e-4}. Pick best.
2. Fix best lr, n_layers=6. Try d_model ∈ {64, 128, 256}. Pick best.
3. Fix best lr and d_model. Try n_layers ∈ {4, 6, 8}. Pick best.

### Step 3 — Tier 2 Sweep (~6 runs)
With Tier 1 best as base, try:
- Positional encoding options
- Activation (GELU vs SwiGLU)
- Dropout (0.1, 0.2, 0.3)
- A few more based on sensitivity you observed

### Step 4 — Variant-Specific Knobs (~3 runs)
Try the variant-specific options from Tier 4 above.

### Step 5 — Seed Sanity (3 runs)
Run your best config with 3 seeds (42, 123, 456). Compute mean ± std. This tells you how reliable your results are.

**Total budget: ~20 runs per person.**

---

## Deliverable from Phase A (bring to team meeting)

A 1-page report per teammate:

```
Variant: V<X> — <Name>
Owner: <Your name>

Best config:
  d_model = <X>
  n_layers = <X>
  n_heads = <X>
  ffn_ratio = <X>
  dropout = <X>
  lr = <X>
  batch_size = <X>
  positional_encoding = <X>
  activation = <X>
  <variant-specific setting> = <X>

Best result:
  val_mse = <mean> ± <std> (over 3 seeds)
  val_ci = <mean> ± <std>

Sensitivity table:
  Hyperparameter | Best value | Acceptable range | Sensitivity (H/M/L)
  d_model       | 128        | 64–256           | M
  lr            | 1e-4       | 5e-5 to 3e-4     | H
  ...

Key findings (surprises, failures, tips):
  - ...
  - ...
  - ...
```

---

## Ground Rules

1. **Test on val, never on test.** Test set is touched only in Phase C.
2. **Log EVERYTHING.** Train loss, val loss, every hyperparameter, wall-clock time, GPU, seed.
3. **No local-only runs.** All runs land in W&B; all configs in Git.
4. **Commit often.** Push your branch daily so the team can see what you're trying.
5. **Ask early.** If your variant is training weirdly, post in team chat same-day, don't burn a week debugging alone.
6. **Don't skip the baseline.** Everyone reports their baseline first, then improvements relative to it.

---

## Daily Standup Format (5 min, async OK)

Post in team chat each day:
- Runs I did yesterday: X
- Best so far: config + val MSE
- What I learned / what's weird
- Next plan

---

## Red Flags — Raise Immediately

- Val loss higher than train loss by >2× → overfit, reduce capacity / add dropout
- Val loss not decreasing at all → LR too high / bug in model
- NaN loss → reduce LR, check gradient clipping, check for /0 in cross-attn
- GPU OOM → reduce batch size OR reduce max_prot_len OR switch partition
- Your variant needs extreme settings (like lr=1e-6, or 500 epochs) → probably a bug

---

## Timeline

| Week | What |
|------|------|
| Week 1 (this week) | Shared scaffolding built; mini-BindingDB subset created |
| Week 2 | Phase A — individual tuning |
| Week 3 | Phase B meeting → lock `fair_comparison.yaml` |
| Week 4-5 | Phase C — full runs on all datasets |
| Week 6-7 | Analysis + writing |

---

## Questions to Resolve in Kickoff Meeting

1. Who owns each shared module (5 items above)?
2. Who creates the mini-BindingDB subset?
3. GitHub repo URL + W&B project + shared Google Sheet link?
4. Everyone set up on HPC (VPN, OOD access, 300 GPU-hours confirmed)?
5. Next checkpoint meeting date (end of Phase A)?

Answer these first. Everything else follows.
