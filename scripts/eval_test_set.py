#!/usr/bin/env python3
"""
Evaluate every Phase C / Phase E1 / Phase E1b checkpoint on the held-out test
split (the one the model NEVER saw during training).

Why: train.py logs val MSE per epoch but doesn't write test MSE to results.csv.
For the paper we want test numbers (the standard reporting metric).

Usage on HPC:
  python scripts/eval_test_set.py --phase c       # Phase C (d=128)
  python scripts/eval_test_set.py --phase e1      # Phase E1 (d=256)
  python scripts/eval_test_set.py --phase e1b     # Phase E1b (d=192)
  python scripts/eval_test_set.py --phase all     # all three

Outputs:
  outputs/test_eval/PHASE_<X>_TEST_RESULTS.csv

Each row: variant, split, seed, val_mse, test_mse, test_ci, test_pearson
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import BindingDBKiDataset
from src.data.splits import random_split, cold_drug_split, cold_target_split
from src.data.collate import collate_fn
from src.data.tokenizers import SMILESTokenizer, ProteinTokenizer
from src.models.variants.early_concat   import EarlyConcatDTI
from src.models.variants.early_crossattn import EarlyCrossAttnDTI
from src.models.variants.late_concat    import LateConcatDTI
from src.models.variants.late_crossattn import LateCrossAttnDTI

REGISTRY = {
    "early_concat":     EarlyConcatDTI,
    "early_crossattn":  EarlyCrossAttnDTI,
    "late_concat":      LateConcatDTI,
    "late_crossattn":   LateCrossAttnDTI,
}
SPLIT_FN = {"random": random_split, "cold_drug": cold_drug_split, "cold_target": cold_target_split}

# Per-phase model configs (must match the training run)
PHASE_CONFIGS = {
    "c":   dict(d_model=128, n_heads=4, d_ff=512, dir_glob="outputs/phase_c/phase_c_*"),
    "e1":  dict(d_model=256, n_heads=8, d_ff=1024, dir_glob="outputs/phase_e_xl/phase_e_xl_*"),
    "e1b": dict(d_model=192, n_heads=6, d_ff=768, dir_glob="outputs/phase_e1b/phase_e1b_*"),
}
COMMON = dict(dropout=0.1, max_drug_len=100, max_prot_len=1200,
              head_hidden=256, head_dropout=0.2)

DATA_ROOT     = "dataset/BindingDB"
TSV_PATH      = os.path.join(DATA_ROOT, "BindingDB_PDSPKi.tsv")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
VOCAB_FILE    = os.path.join(PROCESSED_DIR, "smiles_vocab.json")


# ---------- shared dataset (build once, reuse for all 60+ ckpts) -----------
_DATASET = None
_VOCAB_CACHE = {}
def get_dataset():
    global _DATASET
    if _DATASET is None:
        smiles_tok = SMILESTokenizer(
            vocab_file=VOCAB_FILE if os.path.exists(VOCAB_FILE) else None,
            max_len=100,
        )
        prot_tok = ProteinTokenizer(max_len=1200)
        _DATASET = BindingDBKiDataset(
            tsv_path=TSV_PATH, smiles_tokenizer=smiles_tok,
            protein_tokenizer=prot_tok, max_rows=None,
        )
        if not smiles_tok.vocab or len(smiles_tok.vocab) <= 4:
            smiles_tok.build_vocab(_DATASET.smiles_list)
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            smiles_tok.save_vocab(VOCAB_FILE)
        _VOCAB_CACHE["drug"] = len(smiles_tok.vocab)
        _VOCAB_CACHE["prot"] = (len(prot_tok.vocab) if hasattr(prot_tok, "vocab")
                                else getattr(prot_tok, "vocab_size", 30))
        print(f"  vocab sizes: drug={_VOCAB_CACHE['drug']} prot={_VOCAB_CACHE['prot']}", flush=True)
    return _DATASET


_TEST_LOADERS: dict[tuple[str, int], DataLoader] = {}
def get_test_loader(split: str, seed: int, batch_size: int = 32):
    key = (split, seed)
    if key in _TEST_LOADERS:
        return _TEST_LOADERS[key]
    ds = get_dataset()
    train_idx, val_idx, test_idx = SPLIT_FN[split](ds, seed=seed)
    test_set = Subset(ds, test_idx)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=0)
    _TEST_LOADERS[key] = loader
    return loader


# ---------- evaluation primitives ------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    sse, n = 0.0, 0
    preds_all, y_all = [], []
    dvs = getattr(model, "_drug_vsize", 10**6) - 1
    pvs = getattr(model, "_prot_vsize", 10**6) - 1
    for batch in loader:
        if isinstance(batch, dict):
            d_tok  = batch["drug_tokens"].to(device)
            d_mask = batch["drug_mask"].to(device)
            p_tok  = batch["protein_tokens"].to(device)
            p_mask = batch["protein_mask"].to(device)
            y      = batch["target"].to(device)
        else:
            d_tok, d_mask, p_tok, p_mask, y = (b.to(device) for b in batch)
        d_tok = d_tok.clamp(max=dvs)
        p_tok = p_tok.clamp(max=pvs)
        preds = model(d_tok, d_mask, p_tok, p_mask).squeeze(-1)
        sse += ((preds - y) ** 2).sum().item()
        n   += y.numel()
        preds_all.append(preds.cpu()); y_all.append(y.cpu())
    mse = sse / n
    preds = torch.cat(preds_all); y = torch.cat(y_all)
    # Concordance Index (Davis et al definition)
    ci = concordance_index(preds.numpy(), y.numpy())
    pearson = float(torch.corrcoef(torch.stack([preds, y]))[0, 1])
    return mse, ci, pearson


def concordance_index(preds, y):
    """Davis-style CI: probability that (preds, y) order agrees on a random pair."""
    import numpy as np
    n = len(y)
    idx_pairs_concordant = 0
    n_pairs = 0
    # Vectorized with broadcasting; restrict to upper triangle
    diff_y     = y[None, :]     - y[:, None]
    diff_preds = preds[None, :] - preds[:, None]
    mask = (diff_y > 0)  # only count pairs where y[j] > y[i]
    n_pairs = int(mask.sum())
    if n_pairs == 0:
        return 0.5
    same = (diff_preds[mask] == 0).sum() * 0.5
    pos  = (diff_preds[mask] > 0).sum()
    return float((pos + same) / n_pairs)


def parse_run_dir(run_dir: Path, phase_prefix: str):
    """Extract (variant, split, seed) from a run directory name."""
    name = run_dir.name.replace(f"{phase_prefix}_", "")
    m = re.match(r"(early_concat|early_crossattn|late_crossattn|late_concat)_"
                 r"(random|cold_drug|cold_target)_seed(\d+)", name)
    if not m: return None
    return m.group(1), m.group(2), int(m.group(3))


def _vocab_sizes_from_ckpt(sd: dict) -> tuple[int, int]:
    drug_keys = [k for k in sd if "drug" in k and "embed" in k and "weight" in k]
    prot_keys = [k for k in sd if "prot" in k and "embed" in k and "weight" in k]
    drug_vsize = sd[drug_keys[0]].shape[0] if drug_keys else 66
    prot_vsize = sd[prot_keys[0]].shape[0] if prot_keys else 24
    return drug_vsize, prot_vsize


def load_model_for_run(variant: str, phase_cfg: dict, device, sd: dict, n_layers_override: int | None = None):
    if n_layers_override is not None:
        n_layers = n_layers_override
    else:
        n_layers = (6 if variant in ("early_concat", "early_crossattn") else 3)
    drug_vsize, prot_vsize = _vocab_sizes_from_ckpt(sd)
    cls = REGISTRY[variant]
    kwargs = {**COMMON,
              "drug_vocab_size": drug_vsize,
              "prot_vocab_size": prot_vsize,
              "d_model": phase_cfg["d_model"],
              "n_heads": phase_cfg["n_heads"], "d_ff": phase_cfg["d_ff"],
              "n_layers": n_layers}
    return cls(**kwargs).to(device)


def evaluate_run(run_dir: Path, phase_cfg: dict, device, phase_prefix: str):
    parsed = parse_run_dir(run_dir, phase_prefix)
    if parsed is None: return None
    variant, split, seed = parsed
    ckpt = run_dir / "checkpoints" / variant / "best_model.pt"
    if not ckpt.exists():
        print(f"  [skip] {run_dir.name}: no ckpt", flush=True)
        return None

    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict):
        if "model_state_dict" in sd: sd = sd["model_state_dict"]
        elif "state_dict" in sd:    sd = sd["state_dict"]
    drug_vsize, prot_vsize = _vocab_sizes_from_ckpt(sd)
    model = load_model_for_run(variant, phase_cfg, device, sd)
    res = model.load_state_dict(sd, strict=False)
    if len(res.missing_keys) > 5:
        print(f"  WARNING: {variant} ckpt missing {len(res.missing_keys)} keys", flush=True)
    model._drug_vsize = drug_vsize
    model._prot_vsize = prot_vsize
    model.eval()

    # Read the val MSE from results.csv to keep the row complete
    val_mse = None
    rcsv = run_dir / "results" / "results.csv"
    if rcsv.exists():
        with open(rcsv) as fh:
            for r in csv.DictReader(fh):
                val_mse = float(r["best_val_mse"]); break

    loader = get_test_loader(split, seed)
    test_mse, ci, pear = evaluate(model, loader, device)
    print(f"  {variant:18s} {split:12s} seed={seed:5d}  "
          f"val={val_mse:.3f}  test={test_mse:.3f}  CI={ci:.3f}  r={pear:.3f}",
          flush=True)
    return dict(variant=variant, split=split, seed=seed,
                val_mse=val_mse, test_mse=test_mse, test_ci=ci, test_pearson=pear)


def run_phase(phase: str, device):
    cfg = PHASE_CONFIGS[phase]
    prefix = f"phase_{phase if phase != 'c' else 'c'}"
    if phase == "e1":  prefix = "phase_e_xl"
    elif phase == "e1b": prefix = "phase_e1b"
    elif phase == "c":   prefix = "phase_c"
    out_dir = ROOT / "outputs" / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"PHASE_{phase.upper()}_TEST_RESULTS.csv"

    runs = sorted(Path(".").glob(cfg["dir_glob"]))
    print(f"\n=== Phase {phase.upper()}: {len(runs)} run dirs found ===", flush=True)

    rows = []
    for r in runs:
        try:
            row = evaluate_run(r, cfg, device, prefix)
            if row: rows.append(row)
        except Exception as e:
            print(f"  [FAIL] {r.name}: {type(e).__name__}: {e}", flush=True)

    with open(out_csv, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "split", "seed",
                                            "val_mse", "test_mse", "test_ci", "test_pearson"])
        w.writeheader()
        for row in rows: w.writerow(row)
    print(f"\n[saved] {out_csv}  ({len(rows)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["c", "e1", "e1b", "all"], default="all")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}", flush=True)

    phases = ["c", "e1", "e1b"] if args.phase == "all" else [args.phase]
    for p in phases:
        run_phase(p, device)


if __name__ == "__main__":
    main()
