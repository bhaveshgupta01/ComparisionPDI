# Implementation Plan: DTI Comparison Project (BindingDB PDSPKi, Ki Prediction)

## Goal

Build a complete, runnable DTI comparison framework that trains all **four model variants** on the `BindingDB_PDSPKi.tsv` dataset, predicting **pKi** (−log10(Ki/1e9)). Start with a **small model size** so runs complete quickly for comparison.

---

## Dataset Column Mapping

| Field | TSV Column |
|---|---|
| Drug SMILES | `Ligand SMILES` |
| Protein Sequence | `BindingDB Target Chain Sequence 1` |
| Target (Ki) | `Ki (nM)` → converted to pKi = −log10(Ki/1e9) |

---

## Small Model Hyperparameters (for speed)

| Parameter | Value |
|---|---|
| d_model | 64 |
| n_heads | 2 |
| n_layers total | 4 (2+2 for late variants) |
| d_ff | 128 |
| Prediction head hidden | 128 |
| MAX_DRUG_LEN | 64 |
| MAX_PROT_LEN | 512 |
| Batch size | 32 |
| Max epochs | 30 |
| Early stop patience | 5 |

---

## Proposed Changes

### Component 1 — Project Structure Bootstrap

Create all missing `__init__.py` and directory scaffolding.

#### [NEW] `src/__init__.py`
#### [NEW] `src/data/__init__.py`  
#### [NEW] `src/models/__init__.py`
#### [NEW] `src/models/variants/__init__.py`
#### [NEW] `src/training/__init__.py`
#### [NEW] `src/utils/__init__.py`
#### [NEW] `configs/` directory with YAML configs

---

### Component 2 — Data Pipeline

#### [MODIFY] [tokenizers.py](file:///Users/tenzin/Desktop/ComparisionPDI/src/data/tokenizers.py)
Already exists. No changes needed.

#### [NEW] `src/data/dataset.py`
- `BindingDBKiDataset(Dataset)` — reads the TSV, filters rows where `Ligand SMILES` and `BindingDB Target Chain Sequence 1` and `Ki (nM)` are all valid
- Converts Ki (nM) → pKi = −log10(Ki*1e-9)
- Clips pKi to [3, 12]
- Tokenizes drug and protein on the fly using existing tokenizers

#### [NEW] `src/data/splits.py`
- `random_split(dataset, train_frac=0.8, val_frac=0.1, seed=42)`
- `cold_drug_split(...)` 
- `cold_target_split(...)`

#### [NEW] `src/data/collate.py`
- Pad drug tokens and protein tokens to max-within-batch
- Return `drug_tokens`, `drug_mask`, `prot_tokens`, `prot_mask`, `affinity`

---

### Component 3 — Model Modules

#### [NEW] `src/models/embeddings.py`
- `SinusoidalPositionalEncoding`
- `TokenEmbedding` — spec §7.2 (d_model=64 for small)

#### [NEW] `src/models/encoders.py`
- `TransformerEncoderBlock` — spec §7.3 pre-norm with saved `attn_weights`
- `TransformerEncoder(nn.Module)` — stack of N blocks

#### [NEW] `src/models/cross_attention.py`
- `BidirectionalCrossAttention` — spec §7.4

#### [NEW] `src/models/prediction_head.py`
- `PredictionHead` — spec §7.5

#### [NEW] `src/models/base.py`
- `BaseDTIModel(nn.Module)` — abstract class with `forward(drug_tokens, drug_mask, prot_tokens, prot_mask)` and `count_parameters()`

#### [NEW] `src/models/variants/early_concat.py`
- Variant 1: embed both → concat with CLS → 4 shared encoder layers → CLS → head

#### [NEW] `src/models/variants/early_crossattn.py`
- Variant 2: embed both → bidirectional cross-attention → concat with CLS → 4 shared encoder layers → CLS → head

#### [NEW] `src/models/variants/late_concat.py`
- Variant 3: embed → 2 separate encoders → mean-pool both → concat → head

#### [NEW] `src/models/variants/late_crossattn.py`
- Variant 4: embed → 2 separate encoders → bidirectional cross-attention → mean-pool both → concat → head

---

### Component 4 — Training

#### [NEW] `src/training/metrics.py`
- `mse(y_pred, y_true)`
- `concordance_index(y_pred, y_true)` — pure Python/NumPy implementation (no lifelines dependency)
- `pearson_r(y_pred, y_true)`
- `spearman_r(y_pred, y_true)`

#### [NEW] `src/training/trainer.py`
- `Trainer` class:
  - AdamW optimizer, cosine+warmup LR schedule
  - Gradient clipping (1.0)
  - Saves best checkpoint (val MSE)
  - Logs train loss, val MSE, CI, Pearson, Spearman per epoch
  - Saves results CSV

#### [NEW] `src/utils/seeds.py`
- `set_seed(seed)` — seeds random, numpy, torch

---

### Component 5 — Config & Scripts

#### [NEW] `configs/base.yaml`
Small model shared config.

#### [NEW] `configs/variants/early_concat.yaml`, `early_crossattn.yaml`, `late_concat.yaml`, `late_crossattn.yaml`
Each variant config extends base.

#### [NEW] `scripts/preprocess.py`
- Reads TSV, filters invalid rows, builds SMILES vocab, saves processed data to `data/processed/bindingdb/`

#### [NEW] `scripts/train.py`
CLI: `python scripts/train.py --variant early_concat --seed 42 --split random`

#### [NEW] `scripts/run_all.sh`
Loops all 4 variants × 1 seed (quick run) with random split.

---

### Component 6 — Tests

#### [NEW] `tests/test_models.py`
Forward-pass shape tests for all 4 variants.

#### [NEW] `tests/test_metrics.py`
Correctness tests for MSE and CI.

---

## Verification Plan

### Automated Tests
```bash
.venv/bin/python -m pytest tests/ -v
```

### Preprocessing Smoke Test
```bash
.venv/bin/python scripts/preprocess.py --max_rows 5000
```

### Training Quick Run (all 4 variants, 3 epochs)
```bash
bash scripts/run_all.sh
```

### Expected Outputs
- `data/processed/bindingdb/data.pt`
- `outputs/checkpoints/<variant>/best_model.pt`
- `outputs/results/results.csv` — one row per variant with MSE, CI, Pearson, Spearman
