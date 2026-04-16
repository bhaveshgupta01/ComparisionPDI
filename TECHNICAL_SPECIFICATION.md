# Technical Specification

## Early vs Late Interaction in Transformer Models for Drug–Target Prediction
### A Multi-Modal, Interpretable, Systematically-Analyzed DTI Framework

**Version:** 1.0
**Target Audience:** Engineer/researcher implementing the project end-to-end with only this document as reference.
**Estimated Implementation Time:** 6–8 weeks for one developer.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Scientific Background](#2-scientific-background)
3. [System Architecture](#3-system-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Environment Setup](#5-environment-setup)
6. [Data Pipeline](#6-data-pipeline)
7. [Model Specifications](#7-model-specifications)
8. [Training Protocol](#8-training-protocol)
9. [Evaluation Protocol](#9-evaluation-protocol)
10. [Deep Analysis Module](#10-deep-analysis-module)
11. [Multi-Modal Extensions](#11-multi-modal-extensions)
12. [Agentic Optimization Module](#12-agentic-optimization-module)
13. [Experiment Matrix](#13-experiment-matrix)
14. [Reproducibility Checklist](#14-reproducibility-checklist)
15. [Timeline & Milestones](#15-timeline--milestones)
16. [Risk Register](#16-risk-register)
17. [Appendices](#17-appendices)

---

## 1. Project Overview

### 1.1 Motivation

Drug–target interaction (DTI) prediction is the task of predicting the binding affinity (or interaction probability) between a small-molecule drug and a protein target. Modern deep learning approaches use transformers to encode drugs (SMILES strings) and proteins (amino acid sequences) and then combine these representations to make a prediction.

A fundamental design choice — largely unexplored in the literature — is **when** and **how** drug and protein representations should interact within the model:

- **Early interaction** fuses representations at the embedding level, before deep encoding.
- **Late interaction** fuses representations after independent deep encoding.
- **Concatenation** joins representations without modeling cross-dependencies.
- **Cross-attention** models fine-grained interactions between drug tokens and protein residues.

Crossing these two dimensions yields four architectural variants. This project systematically compares all four, and — critically — explains the mechanism behind their behavior through a comprehensive analysis pipeline.

### 1.2 Research Questions

- **RQ1 (Performance):** Which fusion strategy achieves the best predictive accuracy on standard DTI benchmarks?
- **RQ2 (Mechanism):** Why does each variant behave the way it does? What does each one actually learn?
- **RQ3 (Biology):** Does the model's internal attention correspond to known biological binding sites?
- **RQ4 (Generalization):** Do conclusions hold across: different datasets, different evaluation splits, different input modalities (sequence vs graph vs 3D)?
- **RQ5 (Application):** Can the best model be wrapped in an agentic optimization loop to iteratively refine drug candidates?

### 1.3 Deliverables

- Reproducible training code for four interaction variants
- Comprehensive analysis notebooks producing all figures
- Trained model checkpoints on Davis, KIBA, BindingDB
- Final report (20–30 pages) with all findings
- Optional: agentic refinement demo

---

## 2. Scientific Background

### 2.1 Drug Representation — SMILES

SMILES (Simplified Molecular Input Line Entry System) encodes molecular structure as an ASCII string. Example: aspirin is `CC(=O)Oc1ccccc1C(=O)O`.

Properties:
- Characters represent atoms (`C`, `N`, `O`, `S`, etc.), bonds (`=`, `#`), and structural features (`(`, `)`, digits for ring closures, `c` lowercase for aromatic carbon).
- Typical length: 20–100 characters.
- Vocabulary: ~50–100 distinct tokens depending on tokenization scheme.

### 2.2 Protein Representation — Amino Acid Sequence

Proteins are sequences of 20 standard amino acids. Example start of a kinase: `MGSSHHHHHHSSGLVPRGSHMAS...`

Properties:
- Vocabulary: 20 amino acids + special tokens (padding, start, end, unknown) = 25 tokens.
- Typical length in DTI datasets: 200–2,000 residues. Truncation to 1,000–1,200 is standard.

### 2.3 Binding Affinity — Target Variable

Affinity is measured by multiple experimental assays:
- **Kd (dissociation constant):** Lower = stronger binding. Used in Davis.
- **Ki (inhibition constant):** Similar to Kd.
- **IC50:** Concentration that inhibits 50% of activity.
- **KIBA score:** Integrated score combining Kd, Ki, and IC50. Used in KIBA.

In log-transformed form (pKd = −log10(Kd)): higher = stronger binding. This is the standard regression target.

### 2.4 Evaluation Metrics

- **Mean Squared Error (MSE):** Standard regression metric. Lower is better.
- **Concordance Index (CI):** Probability that the model correctly ranks a random pair of test examples. CI = 1 is perfect ranking; CI = 0.5 is random. Standard for ranking-oriented tasks.
- **Pearson correlation (r):** Linear correlation between predicted and true affinity.
- **Spearman correlation (ρ):** Rank correlation.

---

## 3. System Architecture

### 3.1 High-Level Flow

```
                 ┌──────────────┐                    ┌──────────────┐
                 │ Drug SMILES  │                    │ Protein Seq  │
                 └──────┬───────┘                    └──────┬───────┘
                        │                                   │
                        ▼                                   ▼
                 ┌──────────────┐                    ┌──────────────┐
                 │  Tokenizer   │                    │  Tokenizer   │
                 └──────┬───────┘                    └──────┬───────┘
                        │                                   │
                        ▼                                   ▼
                 ┌──────────────┐                    ┌──────────────┐
                 │  Embedding   │                    │  Embedding   │
                 │  + Pos Enc   │                    │  + Pos Enc   │
                 └──────┬───────┘                    └──────┬───────┘
                        │                                   │
                        └─────────────┬─────────────────────┘
                                      │
                                      ▼
                 ┌────────────────────────────────────────────┐
                 │         INTERACTION MODULE                 │
                 │  {Early|Late} × {Concat|Cross-Attention}   │
                 └──────────────────┬─────────────────────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │  Prediction Head │
                           └──────┬───────────┘
                                  │
                                  ▼
                           ┌──────────────────┐
                           │ Affinity Score   │
                           └──────────────────┘
```

### 3.2 Module Boundaries

- **DataModule:** Loads raw data, tokenizes, produces DataLoaders.
- **EmbeddingModule:** Learnable token embeddings + positional encoding.
- **EncoderModule:** Transformer encoder blocks (shared or separate depending on variant).
- **InteractionModule:** Implements one of four fusion strategies.
- **PredictionHead:** MLP that maps pooled representation to scalar affinity.
- **AnalysisModule:** Extracts attention, gradients, representations for deep analysis.

Every variant uses the same modules; only the `InteractionModule` changes.

---

## 4. Repository Structure

```
DTI_MLFinalProject/
├── README.md
├── TECHNICAL_SPECIFICATION.md          (this file)
├── GAMEPLAN.md
├── DEEP_ANALYSIS_PLAYBOOK.md
├── environment.yml                      (conda environment)
├── requirements.txt                     (pip dependencies)
├── configs/
│   ├── base.yaml                        (shared hyperparameters)
│   ├── variant_early_concat.yaml
│   ├── variant_early_crossattn.yaml
│   ├── variant_late_concat.yaml
│   ├── variant_late_crossattn.yaml
│   ├── datasets/
│   │   ├── davis.yaml
│   │   ├── kiba.yaml
│   │   └── bindingdb.yaml
│   └── analysis/
│       └── full_analysis.yaml
├── data/
│   ├── raw/                             (downloaded datasets)
│   ├── processed/                       (preprocessed, tokenized)
│   ├── splits/                          (train/val/test indices per split strategy)
│   └── pdbbind/                         (binding site annotations for validation)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py                  (Davis, KIBA, BindingDB dataset classes)
│   │   ├── tokenizers.py                (SMILES + protein tokenizers)
│   │   ├── splits.py                    (random, cold-drug, cold-target)
│   │   └── download.py                  (fetch scripts)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                      (BaseDTIModel)
│   │   ├── embeddings.py                (token + positional embeddings)
│   │   ├── encoders.py                  (TransformerEncoder block)
│   │   ├── cross_attention.py           (bidirectional cross-attention)
│   │   ├── prediction_head.py
│   │   ├── variants/
│   │   │   ├── __init__.py
│   │   │   ├── early_concat.py
│   │   │   ├── early_crossattn.py
│   │   │   ├── late_concat.py
│   │   │   └── late_crossattn.py
│   │   └── extensions/
│   │       ├── gnn_drug_encoder.py      (GNN molecular graph encoder)
│   │       ├── structure_protein_encoder.py  (AlphaFold 3D encoder)
│   │       └── pretrained_encoders.py   (ESM-2 / ChemBERTa wrappers)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                   (main training loop)
│   │   ├── losses.py
│   │   ├── metrics.py                   (MSE, CI, Pearson, Spearman)
│   │   ├── optimizers.py
│   │   └── callbacks.py                 (logging, checkpointing, early stop)
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── attention_extraction.py      (Part A)
│   │   ├── representation_geometry.py   (Part B: CKA, probing, t-SNE)
│   │   ├── causal_interventions.py      (Part C: ablations)
│   │   ├── biological_validation.py     (Part D: binding site overlap)
│   │   ├── failure_modes.py             (Part E: error stratification)
│   │   ├── training_dynamics.py         (Part F: checkpoint analysis)
│   │   └── report_generator.py          (produces all figures)
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── react_agent.py
│   │   ├── tools.py                     (DTI tool, RDKit tools)
│   │   └── optimization_loop.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py                   (W&B integration)
│       ├── seeds.py
│       └── visualization.py
├── scripts/
│   ├── download_data.sh
│   ├── preprocess.py
│   ├── train.py                         (main CLI training script)
│   ├── evaluate.py
│   ├── run_analysis.py
│   ├── run_all_experiments.sh           (launches full experiment matrix)
│   └── generate_report.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_sanity_check.ipynb
│   ├── 03_attention_visualization.ipynb
│   ├── 04_representation_analysis.ipynb
│   ├── 05_binding_site_validation.ipynb
│   ├── 06_error_analysis.ipynb
│   └── 07_final_figures.ipynb
├── tests/
│   ├── test_tokenizers.py
│   ├── test_models.py
│   ├── test_splits.py
│   └── test_metrics.py
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── results/
└── report/
    ├── main.tex
    ├── figures/
    └── references.bib
```

---

## 5. Environment Setup

### 5.1 Hardware Requirements

- **Minimum:** 1 × GPU with 16 GB VRAM (e.g., RTX 4080, T4, V100)
- **Recommended:** 1 × GPU with 24+ GB VRAM (RTX 3090, RTX 4090, A5000, A100)
- **CPU:** 8+ cores
- **RAM:** 32 GB minimum, 64 GB for BindingDB-scale training
- **Disk:** 100 GB free (datasets + checkpoints)

### 5.2 Software Dependencies

**Python version:** 3.10

**environment.yml:**
```yaml
name: dti-project
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.10
  - pytorch=2.1
  - pytorch-cuda=12.1
  - numpy=1.24
  - pandas=2.0
  - scipy=1.11
  - scikit-learn=1.3
  - matplotlib=3.7
  - seaborn=0.12
  - rdkit=2023.09
  - biopython=1.81
  - jupyter
  - pip
  - pip:
    - torch-geometric==2.4.0
    - transformers==4.36.0
    - fair-esm==2.0.0
    - wandb==0.16.0
    - captum==0.7.0
    - pyyaml==6.0
    - omegaconf==2.3.0
    - hydra-core==1.3.0
    - rich==13.7.0
    - umap-learn==0.5.5
    - lifelines==0.27.8
    - pytest==7.4.0
    - black==23.11.0
    - isort==5.12.0
```

### 5.3 Installation Commands

```bash
conda env create -f environment.yml
conda activate dti-project
python -m pip install -e .
pytest tests/ -v   # verify setup
```

### 5.4 External Accounts

- **Weights & Biases:** For experiment tracking (free academic tier).
- **Anthropic/OpenAI API key:** Only needed for agentic module (Phase 4).

---

## 6. Data Pipeline

### 6.1 Datasets

#### 6.1.1 Davis

- **Source:** https://github.com/hkmztrk/DeepDTA/tree/master/data/davis
- **Size:** 442 drugs × 379 proteins = 30,056 pairs (all measured)
- **Target:** pKd = −log10(Kd / 1e9), range typically 5–10
- **Use:** Primary development benchmark (small, fast to train)

#### 6.1.2 KIBA

- **Source:** https://github.com/hkmztrk/DeepDTA/tree/master/data/kiba
- **Size:** 2,116 drugs × 229 proteins = 118,254 pairs
- **Target:** KIBA score, range typically 8–18
- **Use:** Secondary benchmark (medium-scale)

#### 6.1.3 BindingDB

- **Source:** https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp
- **Size:** ~2.8M records (filter to Kd/Ki measurements, ~1M usable)
- **Target:** pKd or pKi
- **Use:** Large-scale training / pretraining (optional, resource-intensive)

#### 6.1.4 PDBbind (for binding site validation)

- **Source:** http://www.pdbbind.org.cn/
- **Use:** Provides known binding site residues for ~20,000 protein-ligand complexes.
- **Integration:** Map test proteins to PDBbind entries; extract residues within 5 Å of the ligand as "true binding site."

### 6.2 Raw Data Schema

After download, each dataset lives in `data/raw/<dataset_name>/` and contains:

```
davis/
├── smiles.json           # {drug_id: smiles_string}
├── proteins.json         # {protein_id: aa_sequence}
├── affinity.csv          # drug_id, protein_id, pKd
```

### 6.3 Preprocessing (`scripts/preprocess.py`)

**Steps:**

1. **Clean SMILES**
   - Parse with RDKit; discard invalid SMILES.
   - Canonicalize: `Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)`
   - Compute descriptors (MW, LogP, QED, #rings, #rotatable bonds) and store in metadata for error analysis.

2. **Clean Proteins**
   - Remove non-standard amino acids (replace with `X`).
   - Truncate sequences longer than `MAX_PROT_LEN = 1200` from the C-terminus.
   - Keep metadata: original length, protein family (if available via UniProt).

3. **Filter Affinity**
   - Discard pairs with missing values.
   - For Davis: clip pKd to range [5, 10].
   - For KIBA: clip to reasonable range [5, 20].

4. **Store processed**
   - `data/processed/<dataset>/data.pt` as a PyTorch dict:
     ```python
     {
       'drug_smiles': List[str],
       'drug_tokens': Tensor[N_drugs, max_smiles_len],
       'drug_descriptors': Tensor[N_drugs, 5],
       'protein_seqs': List[str],
       'protein_tokens': Tensor[N_prots, max_prot_len],
       'protein_metadata': List[dict],
       'pairs': Tensor[N_pairs, 3],  # (drug_idx, protein_idx, affinity)
     }
     ```

### 6.4 Tokenization

#### 6.4.1 SMILES Tokenizer

**Strategy:** Regex-based atom-level tokenization (preserves multi-character atoms like `Cl`, `Br`).

```python
SMILES_REGEX = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
```

**Vocabulary:** Build by iterating over all training SMILES; expect ~60–80 tokens. Add special tokens: `<pad>`, `<cls>`, `<sep>`, `<unk>`.

**Max length:** `MAX_DRUG_LEN = 100`. Drugs exceeding this are truncated (rare).

#### 6.4.2 Protein Tokenizer

**Strategy:** Character-level (one token per amino acid).

**Vocabulary:** 20 standard AAs + `X` (unknown) + `<pad>`, `<cls>`, `<sep>` = 25 tokens.

**Max length:** `MAX_PROT_LEN = 1200`.

### 6.5 Splits (`src/data/splits.py`)

Three split strategies, each producing `(train_idx, val_idx, test_idx)` tuples:

#### 6.5.1 Random Split

80% / 10% / 10% of all pairs, randomly sampled. Purpose: sanity check.

#### 6.5.2 Cold-Drug Split

1. Collect all unique drug IDs.
2. Randomly partition drugs into train-drugs (80%), val-drugs (10%), test-drugs (10%).
3. Pairs inherit the split from their drug.

Purpose: Test generalization to unseen drugs.

#### 6.5.3 Cold-Target Split

Analogous to cold-drug but for proteins.

Purpose: Test generalization to unseen proteins (more realistic for drug discovery).

#### 6.5.4 Cold-Both (stretch goal)

Neither drug nor protein seen in training. Hardest setting.

#### 6.5.5 Split Fixing

- All splits seeded with `SPLIT_SEED = 42`
- Splits saved to `data/splits/<dataset>/<split_type>.json`
- Loaded deterministically in every experiment

### 6.6 DataLoader

**Collate function** pads sequences within a batch:
- Drug tokens padded to max length in batch (or to 100, whichever is smaller)
- Protein tokens padded to max length in batch (or to 1200)
- Returns attention masks for each modality

**Batch size:** 64 (default), 128 for late variants if GPU allows.

**Num workers:** 4

---

## 7. Model Specifications

### 7.1 Shared Hyperparameters

All four variants share these to ensure fair comparison:

| Parameter | Value |
|-----------|-------|
| Embedding dimension (d_model) | 128 |
| FFN hidden dim | 512 |
| Num attention heads | 4 |
| Dropout | 0.1 |
| Activation | GELU |
| LayerNorm eps | 1e-5 |
| Total transformer layers (combined) | 6 |
| Prediction head hidden dim | 256 |
| Pooling | [CLS] token or mean pooling |

**Critical fairness rule:** Total number of transformer encoder layers is 6 across all variants. For late variants, this is 3 drug-encoder layers + 3 protein-encoder layers. For early variants, this is 6 shared-encoder layers. Any cross-attention module counts as 1 additional "interaction layer" with its own separate parameter budget, which is equal across both cross-attention variants.

### 7.2 Embedding Module (`src/models/embeddings.py`)

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_len=1200, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = SinusoidalPositionalEncoding(d_model, max_len)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        x = self.token_embed(token_ids)
        x = x + self.pos_embed(x)
        x = self.layer_norm(x)
        return self.dropout(x)
```

Sinusoidal positional encoding (standard transformer formulation); alternative: learned positional embeddings for ablation.

### 7.3 Transformer Encoder Block (`src/models/encoders.py`)

Standard pre-norm transformer block:

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # save for analysis

    def forward(self, x, mask=None):
        x_norm = self.ln1(x)
        attn_out, attn_w = self.self_attn(x_norm, x_norm, x_norm,
                                           key_padding_mask=mask, need_weights=True)
        self.attn_weights = attn_w.detach()
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x
```

Stacking multiple blocks forms the encoder.

### 7.4 Cross-Attention Module (`src/models/cross_attention.py`)

Bidirectional cross-attention:

```python
class BidirectionalCrossAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.drug_attends_protein = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.protein_attends_drug = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_drug = nn.LayerNorm(d_model)
        self.ln_prot = nn.LayerNorm(d_model)

    def forward(self, drug, prot, drug_mask=None, prot_mask=None):
        drug_out, dap_w = self.drug_attends_protein(
            self.ln_drug(drug), self.ln_prot(prot), self.ln_prot(prot),
            key_padding_mask=prot_mask, need_weights=True)
        prot_out, pad_w = self.protein_attends_drug(
            self.ln_prot(prot), self.ln_drug(drug), self.ln_drug(drug),
            key_padding_mask=drug_mask, need_weights=True)
        self.drug_attn_weights = dap_w.detach()
        self.prot_attn_weights = pad_w.detach()
        return drug + drug_out, prot + prot_out
```

### 7.5 Prediction Head

```python
class PredictionHead(nn.Module):
    def __init__(self, d_in, d_hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
```

### 7.6 Variant 1 — Early Concatenation

```
Input: drug_tokens [B, L_d], prot_tokens [B, L_p]
  drug_emb = DrugEmbedding(drug_tokens)        # [B, L_d, D]
  prot_emb = ProtEmbedding(prot_tokens)         # [B, L_p, D]
  combined = concat([CLS, drug_emb, SEP, prot_emb], dim=1)  # [B, L_d+L_p+2, D]
  encoded = SharedEncoder(combined)             # 6 layers
  cls_repr = encoded[:, 0]                      # [B, D]
  pred = PredictionHead(cls_repr)
```

**Total parameters:** ~1.2M

### 7.7 Variant 2 — Early Cross-Attention

```
drug_emb = DrugEmbedding(drug_tokens)
prot_emb = ProtEmbedding(prot_tokens)
drug_attn, prot_attn = CrossAttention(drug_emb, prot_emb)   # interaction at embedding level
combined = concat([CLS, drug_attn, SEP, prot_attn], dim=1)
encoded = SharedEncoder(combined)             # 6 layers
pred = PredictionHead(encoded[:, 0])
```

**Total parameters:** ~1.3M (extra cross-attn module)

### 7.8 Variant 3 — Late Concatenation

```
drug_emb = DrugEmbedding(drug_tokens)
prot_emb = ProtEmbedding(prot_tokens)
drug_enc = DrugEncoder(drug_emb)              # 3 layers
prot_enc = ProtEncoder(prot_emb)              # 3 layers
drug_pool = mean_pool(drug_enc, drug_mask)    # [B, D]
prot_pool = mean_pool(prot_enc, prot_mask)    # [B, D]
combined = concat([drug_pool, prot_pool])     # [B, 2D]
pred = PredictionHead(combined)
```

**Total parameters:** ~1.2M

### 7.9 Variant 4 — Late Cross-Attention

```
drug_emb = DrugEmbedding(drug_tokens)
prot_emb = ProtEmbedding(prot_tokens)
drug_enc = DrugEncoder(drug_emb)              # 3 layers
prot_enc = ProtEncoder(prot_emb)              # 3 layers
drug_fused, prot_fused = CrossAttention(drug_enc, prot_enc)
drug_pool = mean_pool(drug_fused, drug_mask)
prot_pool = mean_pool(prot_fused, prot_mask)
combined = concat([drug_pool, prot_pool])
pred = PredictionHead(combined)
```

**Total parameters:** ~1.3M

### 7.10 Parameter Count Table

Report exact param counts per variant in the paper:

| Variant | Drug Enc | Prot Enc | Fusion | Head | Total |
|---------|----------|----------|--------|------|-------|
| Early Concat | — | — | 6 layers shared | MLP | ~1.2M |
| Early X-Attn | — | — | X-attn + 6 layers shared | MLP | ~1.3M |
| Late Concat | 3 layers | 3 layers | concat | MLP | ~1.2M |
| Late X-Attn | 3 layers | 3 layers | X-attn | MLP | ~1.3M |

Within 10% across variants = fair comparison.

---

## 8. Training Protocol

### 8.1 Loss Function

Mean Squared Error on pKd (or KIBA score):

```
L = mean((y_pred - y_true) ** 2)
```

### 8.2 Optimizer

- **Optimizer:** AdamW
- **Learning rate:** 1e-4 (peak)
- **Weight decay:** 1e-5
- **Betas:** (0.9, 0.999)
- **Epsilon:** 1e-8

### 8.3 Learning Rate Schedule

- Warmup for 5% of total steps (linear)
- Cosine decay to 0 over remaining steps

### 8.4 Training Configuration

| Parameter | Davis | KIBA | BindingDB |
|-----------|-------|------|-----------|
| Batch size | 64 | 128 | 256 |
| Max epochs | 200 | 100 | 30 |
| Early stop patience | 20 | 10 | 5 |
| Validation frequency | Every epoch | Every epoch | Every 0.5 epoch |
| Gradient clip | 1.0 | 1.0 | 1.0 |
| Seeds (min) | 5 | 3 | 2 |

### 8.5 Checkpointing

- Save best model based on validation MSE
- Save every 10 epochs for representation evolution analysis (Part F.2)
- Save final attention weight snapshots
- Directory: `outputs/checkpoints/<variant>/<dataset>/<split>/<seed>/`

### 8.6 Logging (Weights & Biases)

Log every training step:
- Train loss
- Learning rate
- Gradient norm

Log every validation pass:
- Val MSE, CI, Pearson, Spearman
- Per-layer attention entropy (for analysis)
- GPU memory usage
- Wall-clock time per epoch

### 8.7 Reproducibility

Set seeds for:
- `random.seed(seed)`
- `numpy.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

Seeds to use: [42, 123, 456, 789, 2024]

### 8.8 Training Script CLI

```bash
python scripts/train.py \
  --config configs/variant_late_crossattn.yaml \
  --dataset davis \
  --split cold_drug \
  --seed 42 \
  --output-dir outputs/
```

---

## 9. Evaluation Protocol

### 9.1 Primary Metrics

- **MSE** (primary): `mean((y_pred - y_true) ** 2)`
- **CI** (Concordance Index): computed via `lifelines.utils.concordance_index`
  - For each pair (i, j), check if sign(y_pred_i - y_pred_j) matches sign(y_true_i - y_true_j)
- **Pearson r**: `scipy.stats.pearsonr(y_pred, y_true)[0]`
- **Spearman ρ**: `scipy.stats.spearmanr(y_pred, y_true)[0]`

### 9.2 Secondary Metrics

- **RMSE:** `sqrt(MSE)` for interpretability
- **R²:** Coefficient of determination
- **Precision@K:** For binding site validation (Part D)
- **Expected Calibration Error (ECE):** Confidence calibration (Part E.3)

### 9.3 Statistical Testing

For each comparison between variants:
- Run N ≥ 5 seeds
- Paired t-test on MSE and CI
- Report p-value and effect size (Cohen's d)
- Adjust for multiple comparisons with Bonferroni correction (since we do 6 pairwise comparisons among 4 variants)

### 9.4 Reporting Format

For each experiment, produce a row in results table:

```
variant | dataset | split | seed | mse | ci | pearson | spearman | train_time_s | gpu_mem_mb
```

Aggregate across seeds: mean ± std for each metric.

---

## 10. Deep Analysis Module

This is the core research contribution. All analysis code lives in `src/analysis/`.

### 10.1 Part A — Information Flow

#### A.1 Attention Entropy (`attention_extraction.py`)

**Method:**
```python
def compute_attention_entropy(attn_weights):
    # attn_weights: [B, H, L_q, L_k]
    eps = 1e-10
    entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)
    return entropy  # [B, H, L_q]
```

**Procedure:**
1. Run each trained model on test set.
2. Extract attention weights from every layer / every head.
3. Compute per-token entropy, then per-layer mean entropy.
4. Plot mean ± std entropy as function of layer depth, stratified by variant.

**Expected output:** Line chart, 4 variants on same axes, x=layer depth, y=attention entropy.

#### A.2 Mixing Point Detection (`attention_extraction.py`)

**Method:** For early-fusion variants where drug + protein co-exist in combined sequence:
```python
def cross_modal_attention_ratio(attn_weights, drug_len, prot_len):
    # attn_weights: [B, H, L, L] with L = L_d + L_p
    drug_rows = attn_weights[:, :, :drug_len, :]
    drug_to_prot = drug_rows[:, :, :, drug_len:].sum(dim=-1)
    drug_to_drug = drug_rows[:, :, :, :drug_len].sum(dim=-1)
    return (drug_to_prot / (drug_to_drug + drug_to_prot)).mean()
```

Plot this ratio per layer. "Effective mixing point" = layer where ratio first exceeds 0.5.

#### A.3 Gradient Flow Analysis

**Method:** Use Captum's `IntegratedGradients` on trained model to compute input attribution:

```python
from captum.attr import IntegratedGradients
ig = IntegratedGradients(model)
attributions = ig.attribute(inputs=(drug_tokens_emb, prot_tokens_emb), target=0)
```

Aggregate attribution magnitudes by drug/protein side per variant. Report mean attribution ratio (drug vs protein) per variant.

### 10.2 Part B — Representation Geometry

#### B.1 Intrinsic Dimensionality (`representation_geometry.py`)

Participation ratio:
```python
def participation_ratio(X):
    # X: [N, D]
    X_centered = X - X.mean(dim=0)
    cov = X_centered.T @ X_centered / X.shape[0]
    eigenvalues = torch.linalg.eigvalsh(cov)
    pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    return pr.item()
```

Report PR for drug representations and protein representations per variant.

#### B.2 CKA (Centered Kernel Alignment)

Linear CKA:
```python
def linear_cka(X, Y):
    X_c = X - X.mean(0)
    Y_c = Y - Y.mean(0)
    numerator = (X_c.T @ Y_c).pow(2).sum()
    denominator = (X_c.T @ X_c).pow(2).sum().sqrt() * (Y_c.T @ Y_c).pow(2).sum().sqrt()
    return (numerator / denominator).item()
```

Build 4x4 CKA matrix comparing all variants' drug representations and another for protein representations. Plot as heatmaps.

#### B.3 Probing Classifiers

For each variant, freeze representations, train linear probes to predict:
- **Drug-side properties:** MW, LogP, QED, #rings, #rotatable bonds (regression); drug class (classification, if labels available)
- **Protein-side properties:** sequence length, protein family, secondary structure content
- **Interaction-side properties:** binding affinity bin (classification)

Report probe accuracy per property per variant. Use 80/20 split, single linear layer, SGD + weight decay.

#### B.4 t-SNE / UMAP Visualization

Project drug representations to 2D. Create 4-panel plot (one per variant). Color by:
- Drug class
- Molecular weight
- pKd affinity
- Protein family (for protein reps)

Use fixed random state for reproducibility.

### 10.3 Part C — Causal Interventions

#### C.1 Attention Head Ablation (`causal_interventions.py`)

```python
def ablate_head(model, layer_idx, head_idx):
    # Zero out one head's output in forward pass by monkey-patching
    ...
```

For each of the top model's attention heads:
1. Zero it out; re-evaluate on held-out validation set.
2. Record performance drop (ΔMSE).
3. Rank heads by importance.

For top-5 most important heads, visualize attention patterns on representative examples.

#### C.2 Layer Ablation

Replace each layer with identity (residual-only path) one at a time; measure ΔMSE per layer. Plot curve.

#### C.3 Representation Swap

Take Variant A's drug encoder; combine with Variant B's protein encoder; retrain only fusion + prediction head on training set (frozen encoders). Compare to both parent variants' performance.

Do pairwise swaps across all 4 variants (where applicable — only late variants have separate encoders). Report as a swap-compatibility matrix.

#### C.4 Input Perturbation Sensitivity

**Drug perturbation:** Use RDKit to generate isomorphic SMILES (same molecule, different string). Also replace 10% of atoms with chemically similar ones using a substitution table. Measure ΔPrediction / ||Δinput||.

**Protein perturbation:** Substitute 5% of residues with conservative substitutions (BLOSUM62 positive-score swaps).

Report sensitivity scores per variant; lower = more robust.

### 10.4 Part D — Biological Validation

#### D.1 Binding Site Recovery (`biological_validation.py`)

**Input data:** PDBbind crystal structures with annotated binding residues (residues within 5 Å of the bound ligand).

**Procedure:**
1. For each test protein that has a PDBbind match, extract known binding residues.
2. For each drug-protein pair with that protein, extract top-K attended residues from model's attention (aggregate drug→protein cross-attention over all heads and layers).
3. Compute Precision@K = |top-K attended ∩ known binding| / K.

Report K = 5, 10, 20 per variant.

#### D.2 Functional Group Attribution

1. Use RDKit to identify functional groups in each drug (carbonyls, amines, hydroxyls, aromatic rings, halogens).
2. Use Integrated Gradients to compute per-token attribution.
3. Aggregate attribution magnitudes per functional group.
4. Cross-reference with medicinal chemistry literature for each protein family (e.g., kinase hinge-binders should highlight amide N-H).

Produce a "functional group importance heatmap" per variant per protein family.

#### D.3 Cross-Family Generalization

1. Annotate proteins by family (kinase, GPCR, protease, nuclear receptor, etc.) using UniProt or manual mapping.
2. Perform leave-one-family-out: train on all families except one, test on held-out family.
3. Report MSE and CI per held-out family per variant.
4. Build a variant × family matrix showing where each variant succeeds and fails.

### 10.5 Part E — Failure Modes

#### E.1 Error Stratification (`failure_modes.py`)

For each variant, on test set, record absolute error per pair. Stratify by:
- **Drug property bins:** MW (<300, 300–500, >500), #rotatable bonds (<5, 5–10, >10)
- **Protein property bins:** length (<200, 200–500, >500), family
- **Pair property bins:** affinity strength (pKd <6, 6–8, >8), training similarity (max Tanimoto to training drugs)

Plot: 2D heatmap of error by (stratification axis × variant). Identify statistically significant high-error cells via z-score.

#### E.2 Out-of-Distribution Analysis

**OOD score for a test pair:**
- Drug OOD: 1 − max(Tanimoto to training drugs)
- Protein OOD: 1 − max(BLAST identity to training proteins, using Biopython)
- Pair OOD: sum of the two

**Procedure:**
1. Compute OOD score for every test pair.
2. Bin into deciles.
3. Plot mean error per decile per variant.
4. Slope of error vs OOD = brittleness score.

#### E.3 Calibration Analysis

Use MC Dropout (enable dropout at test time, run 20 forward passes) to estimate predictive uncertainty.

Binned calibration:
1. Bin predictions by confidence (variance).
2. For each bin, compute expected error (from variance) vs actual error.
3. Plot reliability diagram.
4. Compute ECE = Σ |expected − actual| weighted by bin size.

### 10.6 Part F — Training Dynamics

#### F.1 Loss Landscape Comparison

Track during training:
- Loss curves (train and val)
- Gradient norms per layer
- Learning rate

Plot loss curves with error bars across seeds for all 4 variants on same axes.

#### F.2 Representation Evolution via CKA

Save checkpoints every 10 epochs. For each variant:
- Extract representations at each checkpoint.
- Compute CKA(checkpoint_i, checkpoint_final) for each i.
- Plot CKA vs epoch: "when do representations stabilize?"

#### F.3 Attention Emergence

At each checkpoint:
- Compute binding-site Precision@10 (Part D.1).
- Plot precision vs epoch per variant.
- Identify the epoch at which each variant "learns" biologically meaningful attention.

### 10.7 Analysis Report Generation (`report_generator.py`)

Automated pipeline that, given a set of trained models, produces all figures and tables:

```bash
python scripts/run_analysis.py --models outputs/checkpoints/ --output outputs/analysis/
```

Outputs:
- `figures/`: all plots as PDF and PNG
- `tables/`: CSV tables for all numerical results
- `summary.md`: auto-generated findings report

---

## 11. Multi-Modal Extensions

Extensions to be implemented after the core 4-variant analysis is complete.

### 11.1 GNN Molecular Graph Encoder (`extensions/gnn_drug_encoder.py`)

**Molecular graph construction:**
- Nodes: atoms (with feature vector: atom type, degree, formal charge, hybridization, is_aromatic, H count)
- Edges: bonds (with features: bond type, is_conjugated, is_in_ring, stereo)

**Node feature dim:** 40 (one-hot encoded + scalars)
**Edge feature dim:** 10

**Model:** Graph Isomorphism Network (GIN) with edge features, 4 layers, hidden dim 128.

**Output:** Graph-level representation via readout (mean + max + sum pooling, concatenated).

**Integration:** Concatenate GNN output with SMILES transformer output (add cross-modal attention between them optional). Re-run all 4 variants with multi-modal drug input.

### 11.2 Pre-trained Encoders (`extensions/pretrained_encoders.py`)

- **Drug:** ChemBERTa-2 (from HuggingFace `DeepChem/ChemBERTa-77M-MLM`)
- **Protein:** ESM-2 150M parameters (from `facebookresearch/esm`)

Use as frozen or lightly fine-tuned encoders. Re-run the 4 variants.

**Hypothesis to test:** Does pre-training reduce the importance of fusion strategy?

### 11.3 3D Structure Encoder (`extensions/structure_protein_encoder.py`)

**Input:** AlphaFold-predicted structure for each protein (download from https://alphafold.ebi.ac.uk/).
**Features:** Residue 3D coordinates + one-hot amino acid identity.
**Model:** SchNet or EGNN (equivariant GNN) on Cα coordinates, focused on binding pocket (known or predicted).

Pocket prediction (if unknown): Use fpocket or P2Rank.

---

## 12. Agentic Optimization Module

### 12.1 Goal

Given a target protein, iteratively propose and refine candidate drug SMILES to maximize predicted binding affinity while maintaining chemical validity and drug-likeness.

### 12.2 Architecture

```
          ┌────────────────┐
          │  Target Spec   │
          └───────┬────────┘
                  ▼
         ┌─────────────────┐
         │   LLM Agent     │◄─────────┐
         │  (ReAct loop)   │          │
         └────────┬────────┘          │
                  ▼                   │
         ┌─────────────────┐          │
         │ Propose SMILES  │          │
         └────────┬────────┘          │
                  ▼                   │
         ┌─────────────────┐          │
         │  RDKit Validate │          │
         └────────┬────────┘          │
                  ▼                   │
         ┌─────────────────┐          │
         │  DTI Predictor  │          │
         │ (our best model)│          │
         └────────┬────────┘          │
                  ▼                   │
         ┌─────────────────┐          │
         │  Attention Map  │          │
         │  Interpretation │          │
         └────────┬────────┘          │
                  └──────────────────┘
```

### 12.3 Tools Exposed to Agent

- `predict_binding(smiles, protein_seq) -> (affinity, attention_map)`
- `validate_smiles(smiles) -> bool`
- `compute_qed(smiles) -> float` (drug-likeness)
- `propose_modification(smiles, instruction) -> new_smiles` (LLM-based or rule-based)
- `compute_similarity(smiles1, smiles2) -> float` (Tanimoto)

### 12.4 Agent Implementation

Use Anthropic Claude API (or equivalent) with system prompt describing the drug discovery task and available tools. Run 5–10 iterations per target.

### 12.5 Evaluation

- Random baseline: sample 100 drugs from ChEMBL; predict against target; take top-K.
- Agent: run 10 iterations; take best candidate.
- Compare: agent's best affinity vs random baseline's best.
- Metric: improvement ratio, success rate across 20 test targets.

---

## 13. Experiment Matrix

### 13.1 Core Experiments

| # | Variant | Dataset | Split | Seeds |
|---|---------|---------|-------|-------|
| 1–5 | Early Concat | Davis | Random | 5 |
| 6–10 | Early X-Attn | Davis | Random | 5 |
| 11–15 | Late Concat | Davis | Random | 5 |
| 16–20 | Late X-Attn | Davis | Random | 5 |
| 21–25 | Early Concat | Davis | Cold-Drug | 5 |
| ... | ... | ... | ... | ... |
| 116–120 | Late X-Attn | KIBA | Cold-Target | 5 |

Total: 4 variants × 2 datasets × 3 splits × 5 seeds = **120 runs**.

### 13.2 Ablation Experiments

| Variable | Values | Runs |
|----------|--------|------|
| Encoder depth | 2, 4, 6, 8 layers | 4 × 4 × 3 seeds = 48 |
| Embedding dim | 64, 128, 256 | 4 × 3 × 3 = 36 |
| Max prot length | 500, 1000, 1200 | 4 × 3 × 3 = 36 |
| Training data % | 10, 25, 50, 100 | 4 × 4 × 3 = 48 |

Total ablations: **168 runs** (Davis only).

### 13.3 Extension Experiments

- GNN-augmented: 4 × 2 × 3 × 3 = 72 runs
- Pre-trained encoders: 4 × 2 × 3 × 3 = 72 runs
- 3D structure (late variants only): 2 × 2 × 3 × 3 = 36 runs

Total extensions: **180 runs**.

### 13.4 Total Budget

- Core: 120 runs × ~30 min/run (Davis) or ~2h/run (KIBA) → ~100 GPU-hours
- Ablations: 168 × 30 min → ~84 GPU-hours
- Extensions: 180 × 45 min → ~135 GPU-hours

**Total: ~320 GPU-hours.** Feasible on a single A100 over 2 weeks, or parallelizable across 4 GPUs in 3 days.

### 13.5 Analysis Runs

Analysis uses trained checkpoints; no additional training compute. Budget ~40 hours of CPU time for CKA, probing, attribution, etc.

---

## 14. Reproducibility Checklist

- [ ] All random seeds explicitly set; seed list documented.
- [ ] All hyperparameters in config YAMLs; no hardcoded values in code.
- [ ] All dataset versions documented with download URLs and checksums.
- [ ] Preprocessing fully scripted; re-running produces identical `data.pt` files.
- [ ] Environment pinned via `environment.yml` with exact versions.
- [ ] Every figure auto-generated by a script (no manual matplotlib in notebooks for final figures).
- [ ] Results CSVs committed to repo; plots regenerate from them.
- [ ] W&B logs archived for all experiments.
- [ ] Model checkpoints stored with metadata (training config, git commit hash).
- [ ] All statistical tests reported with N, effect size, p-value.
- [ ] README includes one-command reproduction: `bash scripts/reproduce_all.sh`.

---

## 15. Timeline & Milestones

**Assumes:** Single developer, 20–25 hours/week, 8-week horizon.

### Week 1 — Foundation
- Repository setup, environment installed
- Data download scripts working for Davis and KIBA
- Preprocessing pipeline complete
- Tokenizers tested
- Unit tests passing
- **Milestone:** `python scripts/preprocess.py --dataset davis` produces clean `data.pt`

### Week 2 — Core Models
- All 4 variants implemented
- Training loop working
- Logging to W&B
- First end-to-end training on Davis random split
- **Milestone:** Late X-Attn achieves val MSE < 0.5 on Davis random split

### Week 3 — Full Experiment Matrix
- All 120 core experiments running (parallelized if possible)
- Results table populated
- Statistical tests implemented
- **Milestone:** Main results table with means + std devs for all 4 variants on both datasets.

### Week 4 — Analysis Part 1 (Information Flow + Representations)
- Attention extraction working
- Part A (entropy, mixing point, gradient flow) complete
- Part B (PR, CKA, probing, t-SNE) complete
- **Milestone:** First 8 figures drafted

### Week 5 — Analysis Part 2 (Causal + Biological)
- Part C (head ablation, layer ablation, representation swap, perturbation) complete
- Part D (binding site recovery, functional groups, cross-family) complete
- **Milestone:** Biological validation figures show meaningful results

### Week 6 — Analysis Part 3 (Failures + Dynamics) + Ablations
- Part E (error stratification, OOD, calibration) complete
- Part F (training dynamics) complete
- All ablation experiments run and analyzed
- **Milestone:** Complete analysis deck; narrative framing locked in.

### Week 7 — Extensions
- GNN encoder integrated; 4 variants re-run
- Pre-trained encoders integrated; 4 variants re-run
- (Optional) 3D structure encoder integrated
- (Optional) Agentic demo built
- **Milestone:** Extension results demonstrate robustness of main findings.

### Week 8 — Report & Polish
- Paper draft complete
- All figures publication-quality
- Code cleaned up; README complete
- Reproducibility script tested end-to-end on fresh machine
- **Milestone:** Final submission ready.

---

## 16. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| All 4 variants perform identically | Medium | High | Have "architecture doesn't matter" narrative pre-planned; use CKA to show they learn same thing |
| Models don't converge on cold-target split | Medium | Medium | Reduce protein max length, use pre-trained ESM-2 |
| BindingDB too large for available compute | High | Low | Skip BindingDB; Davis+KIBA sufficient for main claims |
| Binding site validation data unavailable for most proteins | High | Medium | Use only subset with PDB matches; be transparent about N |
| Integrated Gradients too slow on full test set | Medium | Low | Sample 100 examples for IG; aggregate statistics |
| GPU runs OOM on Late X-Attn + KIBA | Medium | Low | Reduce batch size to 32; use gradient accumulation |
| Agentic module requires API budget | Low | Low | Use local LLM or cap to 20 target evaluations |
| Statistical significance not achieved with 5 seeds | Medium | Medium | Increase to 10 seeds for top 2 variants |
| Reproducibility fails on different hardware | Low | Medium | Fix CUDNN flags; document precise GPU + CUDA version |

---

## 17. Appendices

### Appendix A — Full Hyperparameter Table

```yaml
# base.yaml
seed: 42
model:
  d_model: 128
  n_heads: 4
  d_ff: 512
  dropout: 0.1
  n_layers_total: 6
  max_drug_len: 100
  max_prot_len: 1200
  drug_vocab_size: 80
  prot_vocab_size: 25
  pred_head_hidden: 256
  pred_head_dropout: 0.2
  pooling: mean  # or 'cls'

training:
  optimizer: adamw
  lr: 1e-4
  weight_decay: 1e-5
  betas: [0.9, 0.999]
  warmup_ratio: 0.05
  schedule: cosine
  batch_size: 64
  max_epochs: 200
  early_stop_patience: 20
  gradient_clip: 1.0
  num_workers: 4

data:
  dataset: davis
  split: random  # random | cold_drug | cold_target
  val_ratio: 0.1
  test_ratio: 0.1

logging:
  project: dti-early-vs-late
  save_every_n_epochs: 10
  log_attention_every_n_steps: 500
```

### Appendix B — Variant-Specific Configs

Each variant config overrides `model.variant`:

```yaml
# variant_early_concat.yaml
defaults:
  - base
model:
  variant: early_concat
  shared_encoder_layers: 6

# variant_late_crossattn.yaml
defaults:
  - base
model:
  variant: late_crossattn
  drug_encoder_layers: 3
  prot_encoder_layers: 3
  use_cross_attn: true
```

### Appendix C — Metric Implementations

```python
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr

def concordance_index_fn(y_true, y_pred):
    return concordance_index(y_true, y_pred)

def mse_fn(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean().item()

def pearson_fn(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def spearman_fn(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]
```

### Appendix D — Data Download URLs

```bash
# Davis
wget https://github.com/hkmztrk/DeepDTA/raw/master/data/davis/SMILES.txt
wget https://github.com/hkmztrk/DeepDTA/raw/master/data/davis/target_seq.txt
wget https://github.com/hkmztrk/DeepDTA/raw/master/data/davis/Y.txt

# KIBA
wget https://github.com/hkmztrk/DeepDTA/raw/master/data/kiba/SMILES.txt
wget https://github.com/hkmztrk/DeepDTA/raw/master/data/kiba/target_seq.txt
wget https://github.com/hkmztrk/DeepDTA/raw/master/data/kiba/Y.txt

# PDBbind (requires registration)
# http://www.pdbbind.org.cn/download.php

# BindingDB (optional, large)
# https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp
```

### Appendix E — References

Primary references used in this spec:

1. Öztürk, H., Özgür, A., & Ozkirimli, E. (2018). DeepDTA: deep drug–target binding affinity prediction. Bioinformatics, 34(17), i821-i829.
2. Huang, K., Xiao, C., Glass, L. M., & Sun, J. (2021). MolTrans: molecular interaction transformer for drug–target interaction prediction. Bioinformatics, 37(6), 830-836.
3. Bai, P., Miljković, F., John, B., & Lu, H. (2023). Interpretable bilinear attention network with domain adaptation improves drug–target prediction. Nature Machine Intelligence.
4. Singh, R., Sledzieski, S., Bryson, B., Cowen, L., & Berger, B. (2023). Contrastive learning in protein language space predicts interactions between drugs and protein targets. PNAS.
5. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. ICML.
6. Abramson et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature.
7. Bran, A. M., Cox, S., Schilter, O., Baldassari, C., White, A. D., & Schwaller, P. (2023). ChemCrow: Augmenting large language models with chemistry tools.

---

## End of Technical Specification

This document is self-contained. A developer with ML + Python experience should be able to implement the full project from this specification alone. Any ambiguity should be resolved by defaulting to the most commonly-used approach in transformer-based DTI literature (e.g., DeepDTA, MolTrans as baselines).
