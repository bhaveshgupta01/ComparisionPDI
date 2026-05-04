#!/usr/bin/env python3
"""
E4 — Causal ablations on Phase C checkpoints.

Three studies, all inference-only (no training, no gradient updates):
  1. Layer ablation — zero each transformer block's self-attention output, leaving residual.
  2. Head ablation  — zero each individual attention head's output (per-layer).
  3. Rep swap (V3↔V4) — swap V3's drug_encoder weights into V4 (and vice versa);
                         measure delta vs original.

Inputs:
  outputs/phase_c/phase_c_<variant>_random_seed42/checkpoints/<variant>/best_model.pt

Outputs:
  outputs/phase_e_ablations/<variant>.json
  outputs/phase_e_ablations/rep_swap_v3_v4.json
  outputs/phase_e_ablations/SUMMARY.csv

Usage on HPC:
  python scripts/e4_ablate.py --variant all
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- repo imports (matched against scripts/train.py) ------------------------
from src.data.dataset import BindingDBKiDataset
from src.data.splits import random_split, cold_drug_split, cold_target_split
from src.data.collate import collate_fn
from src.data.tokenizers import SMILESTokenizer, ProteinTokenizer
from src.models.variants.early_concat   import EarlyConcatDTI
from src.models.variants.early_crossattn import EarlyCrossAttnDTI
from src.models.variants.late_concat    import LateConcatDTI
from src.models.variants.late_crossattn import LateCrossAttnDTI

VARIANT_REGISTRY = {
    "early_concat":     EarlyConcatDTI,
    "early_crossattn":  EarlyCrossAttnDTI,
    "late_concat":      LateConcatDTI,
    "late_crossattn":   LateCrossAttnDTI,
}

SPLIT_FN = {
    "random":      random_split,
    "cold_drug":   cold_drug_split,
    "cold_target": cold_target_split,
}

# Paths / defaults match train.py
DATA_ROOT      = "dataset/BindingDB"
TSV_PATH       = os.path.join(DATA_ROOT, "BindingDB_PDSPKi.tsv")
PROCESSED_DIR  = os.path.join(DATA_ROOT, "processed")
VOCAB_FILE     = os.path.join(PROCESSED_DIR, "smiles_vocab.json")
MAX_DRUG_LEN   = 100
MAX_PROT_LEN   = 1200

PHASE_C_KWARGS = dict(d_model=128, n_heads=4, d_ff=512, dropout=0.1,
                      max_drug_len=MAX_DRUG_LEN, max_prot_len=MAX_PROT_LEN,
                      head_hidden=256, head_dropout=0.2)


_VOCAB_CACHE = {}
def build_dataset_and_tokenizers():
    """Reconstruct the same dataset train.py uses; cache vocab sizes for model construction."""
    smiles_tok = SMILESTokenizer(
        vocab_file=VOCAB_FILE if os.path.exists(VOCAB_FILE) else None,
        max_len=MAX_DRUG_LEN,
    )
    prot_tok = ProteinTokenizer(max_len=MAX_PROT_LEN)

    dataset = BindingDBKiDataset(
        tsv_path=TSV_PATH,
        smiles_tokenizer=smiles_tok,
        protein_tokenizer=prot_tok,
        max_rows=None,
    )
    if not smiles_tok.vocab or len(smiles_tok.vocab) <= 4:
        smiles_tok.build_vocab(dataset.smiles_list)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        smiles_tok.save_vocab(VOCAB_FILE)

    _VOCAB_CACHE["drug"] = len(smiles_tok.vocab)
    _VOCAB_CACHE["prot"] = (len(prot_tok.vocab) if hasattr(prot_tok, "vocab")
                            else getattr(prot_tok, "vocab_size", 30))
    print(f"  vocab sizes: drug={_VOCAB_CACHE['drug']} prot={_VOCAB_CACHE['prot']}", flush=True)
    return dataset


def get_val_loader(split: str = "random", seed: int = 42,
                   n_batches: int = 4, batch_size: int = 32):
    """Build the validation Subset for (split, seed) and return n_batches of it."""
    dataset = build_dataset_and_tokenizers()
    train_idx, val_idx, test_idx = SPLIT_FN[split](dataset, seed=seed)
    val_set = Subset(dataset, val_idx)
    loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=0)
    out = []
    for i, b in enumerate(loader):
        if i >= n_batches: break
        out.append(b)
    return out


def _vocab_sizes_from_ckpt(sd: dict) -> tuple[int, int]:
    """Read drug + protein vocab sizes from embedding weight shapes in the saved
    state_dict. Avoids vocab-drift issues between training and inference."""
    drug_keys = [k for k in sd if "drug" in k and "embed" in k and "weight" in k]
    prot_keys = [k for k in sd if "prot" in k and "embed" in k and "weight" in k]
    drug_vsize = sd[drug_keys[0]].shape[0] if drug_keys else 66
    prot_vsize = sd[prot_keys[0]].shape[0] if prot_keys else 24
    return drug_vsize, prot_vsize


def load_variant(variant: str, device: torch.device):
    n_layers = 6 if variant in ("early_concat", "early_crossattn") else 3
    cls = VARIANT_REGISTRY[variant]
    ckpt_path = ROOT / "outputs" / "phase_c" / f"phase_c_{variant}_random_seed42" / "checkpoints" / variant / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Phase C ckpt missing: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict):
        if "model_state_dict" in sd: sd = sd["model_state_dict"]
        elif "state_dict" in sd:    sd = sd["state_dict"]
    drug_vsize, prot_vsize = _vocab_sizes_from_ckpt(sd)
    model = cls(
        drug_vocab_size=drug_vsize,
        prot_vocab_size=prot_vsize,
        n_layers=n_layers,
        **PHASE_C_KWARGS,
    ).to(device)
    res = model.load_state_dict(sd, strict=False)
    print(f"  loaded {variant}: drug_vocab={drug_vsize} prot_vocab={prot_vsize} "
          f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}", flush=True)
    model._drug_vsize = drug_vsize
    model._prot_vsize = prot_vsize
    model.eval()
    return model


@torch.no_grad()
def eval_mse(model, batches, device):
    sse, n = 0.0, 0
    # The training-time vocab may have been smaller than the current tokenizer
    # vocab — clamp token IDs to the size the model's embedding can actually
    # handle. Out-of-range IDs were missing from the training set anyway.
    dvs = getattr(model, "_drug_vsize", 10**6) - 1
    pvs = getattr(model, "_prot_vsize", 10**6) - 1
    for batch in batches:
        if isinstance(batch, dict):
            d_tok  = batch.get("drug_tokens",   batch.get("drug_input_ids")).to(device)
            d_mask = batch.get("drug_mask",     batch.get("drug_attention_mask")).to(device)
            p_tok  = batch.get("protein_tokens", batch.get("protein_input_ids")).to(device)
            p_mask = batch.get("protein_mask",  batch.get("protein_attention_mask")).to(device)
            y      = batch.get("target", batch.get("y")).to(device)
        else:
            d_tok, d_mask, p_tok, p_mask, y = (b.to(device) for b in batch)
        d_tok = d_tok.clamp(max=dvs)
        p_tok = p_tok.clamp(max=pvs)
        preds = model(d_tok, d_mask, p_tok, p_mask).squeeze(-1)
        sse += ((preds - y) ** 2).sum().item()
        n   += y.numel()
    return sse / n


def find_attention_blocks(model):
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, nn.MultiheadAttention)]


def ablate_layer_mse(model, batches, device, attn_module):
    orig = attn_module.forward
    def zero_forward(query, key, value, *a, **k):
        # The custom encoder layer in src/models/encoders.py expects attn_w to NOT be None
        # (it stores `attn_w.detach()` for downstream extraction). Return a zero attn matrix.
        # batch_first=True so query is (B, L, d).
        B = query.shape[0]
        L = query.shape[1] if query.dim() == 3 else query.shape[0]
        zero_out  = torch.zeros_like(query)
        zero_attn = torch.zeros(B, L, L, device=query.device, dtype=query.dtype)
        return zero_out, zero_attn
    attn_module.forward = zero_forward
    mse = eval_mse(model, batches, device)
    attn_module.forward = orig
    return mse


def ablate_head_mse(model, batches, device, attn_module, head_idx):
    n_heads = attn_module.num_heads
    d_model = attn_module.embed_dim
    head_dim = d_model // n_heads
    def hook(module, inputs, output):
        attn_out, attn_w = output
        new_out = attn_out.clone()
        lo = head_idx * head_dim
        hi = (head_idx + 1) * head_dim
        new_out[..., lo:hi] = 0.0
        return (new_out, attn_w)
    handle = attn_module.register_forward_hook(hook)
    mse = eval_mse(model, batches, device)
    handle.remove()
    return mse


def run_ablations_for(variant: str, device, batches, out_dir: Path):
    print(f"\n=== Ablations on {variant} ===", flush=True)
    model = load_variant(variant, device)
    baseline = eval_mse(model, batches, device)
    print(f"  baseline MSE: {baseline:.4f}", flush=True)

    blocks = find_attention_blocks(model)
    print(f"  {len(blocks)} nn.MultiheadAttention modules", flush=True)

    layer_results, head_results = [], []
    for name, mod in blocks:
        mse_layer = ablate_layer_mse(model, batches, device, mod)
        d_layer = mse_layer - baseline
        layer_results.append({"module": name, "ablated_mse": mse_layer, "delta": d_layer})
        print(f"    LAYER {name:55s}  Δ MSE = {d_layer:+.4f}", flush=True)

        n_h = mod.num_heads
        for h in range(n_h):
            mse_h = ablate_head_mse(model, batches, device, mod, h)
            d_h = mse_h - baseline
            head_results.append({"module": name, "head": h, "ablated_mse": mse_h, "delta": d_h})
            print(f"    HEAD  {name:55s} h={h}  Δ MSE = {d_h:+.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"variant": variant, "baseline_mse": baseline,
               "layer_ablations": layer_results, "head_ablations": head_results}
    (out_dir / f"{variant}.json").write_text(json.dumps(payload, indent=2))
    print(f"  -> wrote {out_dir / variant}.json", flush=True)
    return payload


def run_rep_swap(device, batches, out_dir: Path):
    print("\n=== V3 ↔ V4 representation swap ===", flush=True)
    v3 = load_variant("late_crossattn", device)
    v4 = load_variant("late_concat",    device)

    base_v3 = eval_mse(v3, batches, device)
    base_v4 = eval_mse(v4, batches, device)
    print(f"  V3 baseline: {base_v3:.4f}", flush=True)
    print(f"  V4 baseline: {base_v4:.4f}", flush=True)

    if not hasattr(v3, "drug_encoder") or not hasattr(v4, "drug_encoder"):
        print("  [skip] one of V3/V4 has no .drug_encoder attr — repo API differs", flush=True)
        return None

    v3_drug_sd = copy.deepcopy(v3.drug_encoder.state_dict())
    v4_drug_sd = copy.deepcopy(v4.drug_encoder.state_dict())
    try:
        v3.drug_encoder.load_state_dict(v4_drug_sd, strict=True)
        v4.drug_encoder.load_state_dict(v3_drug_sd, strict=True)
    except Exception as e:
        print(f"  [skip] swap failed: {e}", flush=True)
        return None

    swapped_v3 = eval_mse(v3, batches, device)
    swapped_v4 = eval_mse(v4, batches, device)
    print(f"  V3 with V4's drug_encoder: {swapped_v3:.4f}  (Δ = {swapped_v3 - base_v3:+.4f})", flush=True)
    print(f"  V4 with V3's drug_encoder: {swapped_v4:.4f}  (Δ = {swapped_v4 - base_v4:+.4f})", flush=True)

    payload = {"v3_baseline": base_v3, "v4_baseline": base_v4,
               "v3_with_v4_drug": swapped_v3, "v4_with_v3_drug": swapped_v4,
               "delta_v3": swapped_v3 - base_v3, "delta_v4": swapped_v4 - base_v4}
    (out_dir / "rep_swap_v3_v4.json").write_text(json.dumps(payload, indent=2))
    print(f"  -> wrote rep_swap_v3_v4.json", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", choices=list(VARIANT_REGISTRY) + ["all"])
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--skip_rep_swap", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}", flush=True)

    print("Loading eval batches…", flush=True)
    batches = get_val_loader(split="random", seed=42,
                              n_batches=args.n_batches, batch_size=args.batch_size)
    n_examples = sum(b["target"].numel() if isinstance(b, dict) else b[-1].numel() for b in batches)
    print(f"  {len(batches)} batches × {args.batch_size} = {n_examples} examples", flush=True)

    out_dir = ROOT / "outputs" / "phase_e_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = list(VARIANT_REGISTRY) if args.variant == "all" else [args.variant]
    summaries = []
    for v in targets:
        try:
            r = run_ablations_for(v, device, batches, out_dir)
            summaries.append(r)
        except FileNotFoundError as e:
            print(f"[skip] {v}: {e}", flush=True)
        except Exception as e:
            print(f"[FAIL] {v}: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()

    if not args.skip_rep_swap and ("late_crossattn" in targets and "late_concat" in targets):
        try:
            run_rep_swap(device, batches, out_dir)
        except Exception as e:
            print(f"[FAIL] rep_swap: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()

    import csv
    with open(out_dir / "SUMMARY.csv", "w") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "kind", "module", "head", "baseline_mse", "ablated_mse", "delta"])
        for s in summaries:
            for r in s["layer_ablations"]:
                w.writerow([s["variant"], "layer", r["module"], "",
                            s["baseline_mse"], r["ablated_mse"], r["delta"]])
            for r in s["head_ablations"]:
                w.writerow([s["variant"], "head", r["module"], r["head"],
                            s["baseline_mse"], r["ablated_mse"], r["delta"]])
    print(f"\nWrote {out_dir / 'SUMMARY.csv'}", flush=True)


if __name__ == "__main__":
    main()
