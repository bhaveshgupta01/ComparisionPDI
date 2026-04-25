#!/usr/bin/env python
"""
Preprocessing Script
====================
Reads BindingDB_PDSPKi.tsv, builds the SMILES vocabulary from the training
split, and saves a processed data bundle to data/processed/bindingdb/.

Usage:
    .venv/bin/python scripts/preprocess.py [--max_rows N] [--seed 42]

Outputs:
    data/processed/bindingdb/smiles_vocab.json  — SMILES tokenizer vocabulary
    data/processed/bindingdb/splits.json        — {train, val, test} index lists
    data/processed/bindingdb/stats.json         — dataset statistics
"""
import argparse
import json
import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import BindingDBKiDataset
from src.data.splits import random_split, cold_drug_split, cold_target_split
from src.data.tokenizers import SMILESTokenizer, ProteinTokenizer
from src.utils.seeds import set_seed


TSV_PATH = "dataset/BindingDB/BindingDB_PDSPKi.tsv"
OUT_DIR = "data/processed/bindingdb"


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess BindingDB PDSPKi dataset")
    p.add_argument("--tsv", default=TSV_PATH, help="Path to BindingDB_PDSPKi.tsv")
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--max_rows", type=int, default=None,
                   help="Limit rows read (for smoke testing, e.g. 5000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="random",
                   choices=["random", "cold_drug", "cold_target"])
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Step 1: Build dataset with default tokenizers ──────────────────────
    smiles_tok = SMILESTokenizer(max_len=64)
    prot_tok = ProteinTokenizer(max_len=512)

    dataset = BindingDBKiDataset(
        tsv_path=args.tsv,
        smiles_tokenizer=smiles_tok,
        protein_tokenizer=prot_tok,
        max_rows=args.max_rows,
    )

    n = len(dataset)
    print(f"[Preprocess] Total valid samples: {n}")

    # ── Step 2: Build SMILES vocabulary from *all* samples ─────────────────
    # (In production use only training SMILES; for simplicity we build
    #  from all since there are no string-level leakage concerns for vocab.)
    print("[Preprocess] Building SMILES vocabulary …")
    smiles_tok.build_vocab(dataset.smiles_list)

    vocab_path = os.path.join(args.out_dir, "smiles_vocab.json")
    smiles_tok.save_vocab(vocab_path)
    print(f"[Preprocess] SMILES vocab size: {len(smiles_tok.vocab)} → {vocab_path}")

    # ── Step 3: Create split ───────────────────────────────────────────────
    split_fn = {
        "random": random_split,
        "cold_drug": cold_drug_split,
        "cold_target": cold_target_split,
    }[args.split]

    train_idx, val_idx, test_idx = split_fn(dataset, seed=args.seed)
    print(
        f"[Preprocess] Split '{args.split}': "
        f"train={len(train_idx):,} | val={len(val_idx):,} | test={len(test_idx):,}"
    )

    splits_path = os.path.join(args.out_dir, f"splits_{args.split}.json")
    with open(splits_path, "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)
    print(f"[Preprocess] Splits saved → {splits_path}")

    # ── Step 4: Dataset statistics ─────────────────────────────────────────
    import numpy as np
    pki_arr = np.array(dataset.pki_list)
    stats = {
        "n_samples": n,
        "pki_mean": float(pki_arr.mean()),
        "pki_std": float(pki_arr.std()),
        "pki_min": float(pki_arr.min()),
        "pki_max": float(pki_arr.max()),
        "n_unique_drugs": len(set(dataset.smiles_list)),
        "n_unique_proteins": len(set(dataset.prot_list)),
        "smiles_vocab_size": len(smiles_tok.vocab),
    }
    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("[Preprocess] Dataset statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"[Preprocess] Stats saved → {stats_path}")
    print("[Preprocess] Done ✓")


if __name__ == "__main__":
    main()
