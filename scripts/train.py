#!/usr/bin/env python
"""
Training Script
===============
Trains a single DTI variant on BindingDB PDSPKi data.

Usage:
    .venv/bin/python scripts/train.py \\
        --variant early_concat \\
        --split random \\
        --seed 42 \\
        --max_rows 20000 \\
        --epochs 30

Variants: early_concat | early_crossattn | late_concat | late_crossattn
"""
import argparse
import json
import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from src.data.collate import collate_fn
from src.data.dataset import BindingDBKiDataset
from src.data.splits import random_split, cold_drug_split, cold_target_split
from src.data.tokenizers import SMILESTokenizer, ProteinTokenizer
from src.models import build_model
from src.training.trainer import Trainer
from src.utils.seeds import set_seed


# ────────────────────────────────────────────────────────────────────────────
# Phase A Baseline defaults (override via CLI for sweeps)
# ────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    d_model=128,
    n_heads=4,
    n_layers=6,
    d_ff=512,
    dropout=0.1,
    max_drug_len=100,
    max_prot_len=1200,
    head_hidden=256,
    head_dropout=0.2,
    lr=1e-4,
    weight_decay=1e-5,
    batch_size=64,
    num_workers=4,
    max_epochs=30,
    patience=15,
    grad_clip=1.0,
)

TSV_PATH = "dataset/BindingDB/BindingDB_PDSPKi.tsv"
PROCESSED_DIR = "data/processed/bindingdb"


def parse_args():
    p = argparse.ArgumentParser(description="Train a DTI model variant")
    p.add_argument("--variant", required=True,
                   choices=["early_concat", "early_crossattn", "late_concat", "late_crossattn"])
    p.add_argument("--split", default="random",
                   choices=["random", "cold_drug", "cold_target"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tsv", default=TSV_PATH)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--max_rows", type=int, default=None,
                   help="Limit dataset size (smoke test). None = use all.")
    p.add_argument("--epochs", type=int, default=DEFAULTS["max_epochs"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--d_model", type=int, default=DEFAULTS["d_model"])
    p.add_argument("--n_layers", type=int, default=DEFAULTS["n_layers"])
    p.add_argument("--n_heads", type=int, default=DEFAULTS["n_heads"])
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--vocab_file", default=None,
                   help="Pre-built SMILES vocab JSON. Auto-built if not provided.")
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated GPU indices to use, e.g. '0,1'. "
                        "Defaults to all visible CUDA GPUs.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ── GPU selection ───────────────────────────────────────────────────────
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"[train.py] Restricting to GPUs: {args.gpus}")

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        # Scale total batch_size so each GPU sees the configured per-GPU size.
        # e.g. 2 GPUs × 64 per-GPU = 128 total → DataParallel splits evenly.
        effective_batch = args.batch_size * num_gpus
        effective_workers = DEFAULTS["num_workers"] * num_gpus
        print(f"[train.py] {num_gpus} GPUs detected → "
              f"batch_size {args.batch_size} → {effective_batch} (total), "
              f"num_workers {DEFAULTS['num_workers']} → {effective_workers}")
    else:
        effective_batch = args.batch_size
        effective_workers = DEFAULTS["num_workers"]

    # ── Tokenizers ─────────────────────────────────────────────────────────
    vocab_file = args.vocab_file
    if vocab_file is None:
        vocab_file = os.path.join(PROCESSED_DIR, "smiles_vocab.json")

    smiles_tok = SMILESTokenizer(
        vocab_file=vocab_file if os.path.exists(vocab_file) else None,
        max_len=DEFAULTS["max_drug_len"],
    )
    prot_tok = ProteinTokenizer(max_len=DEFAULTS["max_prot_len"])

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = BindingDBKiDataset(
        tsv_path=args.tsv,
        smiles_tokenizer=smiles_tok,
        protein_tokenizer=prot_tok,
        max_rows=args.max_rows,
    )

    # Build vocab if not pre-existing
    if not smiles_tok.vocab or len(smiles_tok.vocab) <= 4:
        print("[train.py] Building SMILES vocab from training data …")
        smiles_tok.build_vocab(dataset.smiles_list)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        smiles_tok.save_vocab(vocab_file)
        print(f"[train.py] SMILES vocab size: {len(smiles_tok.vocab)}")

    # ── Splits ─────────────────────────────────────────────────────────────
    splits_cache = os.path.join(PROCESSED_DIR, f"splits_{args.split}.json")
    # Only use split cache if we're using the full dataset (no max_rows limit).
    # When max_rows is set the dataset is smaller and the cached indices may be out of range.
    use_cache = os.path.exists(splits_cache) and args.max_rows is None
    if use_cache:
        with open(splits_cache) as f:
            d = json.load(f)
        train_idx, val_idx, test_idx = d["train"], d["val"], d["test"]
        print(f"[train.py] Loaded splits from cache: {splits_cache}")
    else:
        split_fn = {"random": random_split, "cold_drug": cold_drug_split,
                    "cold_target": cold_target_split}[args.split]
        train_idx, val_idx, test_idx = split_fn(dataset, seed=args.seed)

    print(
        f"[train.py] Split '{args.split}': "
        f"train={len(train_idx):,} | val={len(val_idx):,} | test={len(test_idx):,}"
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=effective_batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=effective_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=effective_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=effective_batch * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=effective_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=effective_workers > 0,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    # Combine DEFAULTS with CLI overrides
    model_kwargs = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": 4 * args.d_model,  # 4x d_model standard
        "dropout": args.dropout,
        "max_drug_len": DEFAULTS["max_drug_len"],
        "max_prot_len": DEFAULTS["max_prot_len"],
        "head_hidden": DEFAULTS["head_hidden"],
        "head_dropout": DEFAULTS["head_dropout"],
    }

    model = build_model(
        args.variant,
        drug_vocab_size=len(smiles_tok.vocab),
        prot_vocab_size=len(prot_tok.vocab),
        **model_kwargs
    )
    print(f"[train.py] {model.parameter_summary()}")

    # ── Train ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        variant_name=args.variant,
        output_dir=args.output_dir,
        lr=args.lr,
        weight_decay=DEFAULTS["weight_decay"],
        max_epochs=args.epochs,
        patience=args.patience,
        grad_clip=DEFAULTS["grad_clip"],
    )
    trainer.train(train_loader, val_loader, split_name=args.split, seed=args.seed)

    print(f"\n[train.py] Done. Results → {trainer.results_path}")


if __name__ == "__main__":
    main()
