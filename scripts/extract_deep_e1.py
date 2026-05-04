#!/usr/bin/env python3
"""
Extract attention weights, hidden states, masks, and predictions for the first
256 examples of the random-split / seed=42 validation set, on each Phase E1
checkpoint (V1-V4 at d=256). Mirrors the artifact layout of the original Phase D
extraction so analyze_deep.py can chew on it.

Outputs:
  phase_d_artifacts_deep/analysis_deep_e1/v{1..4}_phase_e_xl/
    - meta.json
    - attn_<encoder|drug_encoder|prot_encoder>_layer<L>.npy   shape (256, L, L) fp16
    - hidden_subset_..._layer<L>.npy                          shape (256, 200, d) fp32
    - pooled_..._layer<L>.npy                                 shape (256, d)     fp32
    - drug_mask.npy / prot_mask.npy                           bool, 1=PADDING
    - predictions.npy / truth.npy                             fp32

This file is HPC-side; depends on the same src/* tree as train.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import BindingDBKiDataset
from src.data.splits import random_split
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

# Phase E1 model config — matches training. head_hidden=256 because the actual
# training used train.py defaults (256), not the aspirational 512 in the config yaml.
E1_KWARGS = dict(d_model=256, n_heads=8, d_ff=1024, dropout=0.1,
                  max_drug_len=100, max_prot_len=1200,
                  head_hidden=256, head_dropout=0.2)

VARIANT_FOLDER = {
    "early_concat":     "v1_phase_e_xl",
    "early_crossattn":  "v2_phase_e_xl",
    "late_crossattn":   "v3_phase_e_xl",
    "late_concat":      "v4_phase_e_xl",
}

DATA_ROOT     = "dataset/BindingDB"
TSV_PATH      = os.path.join(DATA_ROOT, "BindingDB_PDSPKi.tsv")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
VOCAB_FILE    = os.path.join(PROCESSED_DIR, "smiles_vocab.json")

N_ANALYZED  = 64        # reduced from 256 — at d=256/h=8/seq=1300 the (B*h*L^2)
                         # attention matrix is 13.8 GB at B=256; 64 keeps it ~3.5 GB
                         # so total fwd memory stays under 25 GB on a 40 GB A100.
SUBSET_N    = 32        # smaller subset for hidden states (memory)
SUBSET_LCAP = 200       # token cap for hidden_subset arrays


_VOCAB_CACHE = {}
def build_loader_and_vocabs():
    smiles_tok = SMILESTokenizer(
        vocab_file=VOCAB_FILE if os.path.exists(VOCAB_FILE) else None,
        max_len=100,
    )
    prot_tok = ProteinTokenizer(max_len=1200)
    ds = BindingDBKiDataset(tsv_path=TSV_PATH, smiles_tokenizer=smiles_tok,
                             protein_tokenizer=prot_tok, max_rows=None)
    if not smiles_tok.vocab or len(smiles_tok.vocab) <= 4:
        smiles_tok.build_vocab(ds.smiles_list)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        smiles_tok.save_vocab(VOCAB_FILE)

    drug_vocab_size = len(smiles_tok.vocab)
    prot_vocab_size = (len(prot_tok.vocab) if hasattr(prot_tok, "vocab")
                       else getattr(prot_tok, "vocab_size", 30))
    _VOCAB_CACHE["drug"] = drug_vocab_size
    _VOCAB_CACHE["prot"] = prot_vocab_size
    print(f"  vocab sizes: drug={drug_vocab_size} prot={prot_vocab_size}", flush=True)

    train_idx, val_idx, _ = random_split(ds, seed=42)
    val_set = Subset(ds, val_idx[:N_ANALYZED])
    loader = DataLoader(val_set, batch_size=N_ANALYZED, shuffle=False,
                       collate_fn=collate_fn, num_workers=0)
    return loader


def _vocab_sizes_from_ckpt(sd: dict) -> tuple[int, int]:
    drug_keys = [k for k in sd if "drug" in k and "embed" in k and "weight" in k]
    prot_keys = [k for k in sd if "prot" in k and "embed" in k and "weight" in k]
    drug_vsize = sd[drug_keys[0]].shape[0] if drug_keys else 66
    prot_vsize = sd[prot_keys[0]].shape[0] if prot_keys else 24
    return drug_vsize, prot_vsize


def load_model(variant: str, device):
    n_layers = 6 if variant in ("early_concat", "early_crossattn") else 3
    cls = REGISTRY[variant]
    ckpt = ROOT / "outputs" / "phase_e_xl" / f"phase_e_xl_{variant}_random_seed42" / "checkpoints" / variant / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"E1 ckpt missing: {ckpt}")
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict):
        if "model_state_dict" in sd: sd = sd["model_state_dict"]
        elif "state_dict" in sd:    sd = sd["state_dict"]
    drug_vsize, prot_vsize = _vocab_sizes_from_ckpt(sd)
    model = cls(
        drug_vocab_size=drug_vsize,
        prot_vocab_size=prot_vsize,
        n_layers=n_layers,
        **E1_KWARGS,
    ).to(device)
    res = model.load_state_dict(sd, strict=False)
    print(f"  loaded {variant}: drug_vocab={drug_vsize} prot_vocab={prot_vsize} "
          f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}", flush=True)
    if len(res.missing_keys) > 5:
        print(f"  WARNING: {len(res.missing_keys)} missing keys — load may be broken", flush=True)
    model._drug_vsize = drug_vsize
    model._prot_vsize = prot_vsize
    model.eval()
    return model


class AttentionRecorder:
    """Hook every nn.MultiheadAttention with a forward hook that captures
    attention weights and hidden inputs/outputs. Aggregates head dimension."""

    def __init__(self):
        self.handles = []
        self.attn = {}     # name -> (B, L, L) fp16
        self.hidden_in = {}  # name -> input hidden, fp32
        self.hidden_out = {}  # name -> output hidden, fp32

    def attach(self, model: nn.Module):
        for name, m in model.named_modules():
            if isinstance(m, nn.MultiheadAttention):
                # Force need_weights=True so we get attention back
                # We patch its forward to keep the original signature but always set need_weights
                orig_forward = m.forward
                def patched(query, key, value, *a, _orig=orig_forward, **k):
                    k["need_weights"] = True
                    k["average_attn_weights"] = True
                    return _orig(query, key, value, *a, **k)
                m.forward = patched
                # Pre-hook: record input hidden (the query tensor)
                self.handles.append(m.register_forward_pre_hook(self._pre_hook(name)))
                # Forward hook: record output (attn_output, attn_weights)
                self.handles.append(m.register_forward_hook(self._post_hook(name)))

    def _pre_hook(self, name):
        def hook(module, inputs):
            if len(inputs) >= 1:
                self.hidden_in[name] = inputs[0].detach()
        return hook

    def _post_hook(self, name):
        def hook(module, inputs, output):
            attn_out, attn_w = output
            if attn_w is not None:
                self.attn[name] = attn_w.detach()
            self.hidden_out[name] = attn_out.detach()
        return hook

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def extract_for_variant(variant: str, device, loader, out_dir: Path):
    print(f"\n=== Extract: {variant} ===", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(variant, device)
    recorder = AttentionRecorder()
    recorder.attach(model)

    batch = next(iter(loader))
    if isinstance(batch, dict):
        d_tok  = batch["drug_tokens"].to(device)
        d_mask = batch["drug_mask"].to(device)
        p_tok  = batch["protein_tokens"].to(device)
        p_mask = batch["protein_mask"].to(device)
        y      = batch["target"]
    else:
        d_tok, d_mask, p_tok, p_mask, y = (
            batch[0].to(device), batch[1].to(device),
            batch[2].to(device), batch[3].to(device),
            batch[4],
        )
    # Clamp token IDs to the model's training-time vocab size; current tokenizer
    # has more tokens than the saved embedding can handle.
    d_tok = d_tok.clamp(max=model._drug_vsize - 1)
    p_tok = p_tok.clamp(max=model._prot_vsize - 1)

    with torch.no_grad():
        preds = model(d_tok, d_mask, p_tok, p_mask).squeeze(-1)

    # Save masks (1 = padding convention)
    np.save(out_dir / "drug_mask.npy", d_mask.cpu().numpy().astype(bool))
    np.save(out_dir / "prot_mask.npy", p_mask.cpu().numpy().astype(bool))
    np.save(out_dir / "predictions.npy", preds.cpu().numpy().astype(np.float32))
    np.save(out_dir / "truth.npy", y.cpu().numpy().astype(np.float32))

    # Naming convention: rename module path to <component>_layer<L>
    # named_modules paths depend on the variant. We map them to short names
    # matching the v1/v2/v3/v4 conventions used by analyze_deep.py.
    attn_keys, hidden_keys = [], []
    for name, attn_w in recorder.attn.items():
        short, layer = canonical_name(name, variant)
        if short is None: continue
        outpath = out_dir / f"attn_{short}_layer{layer}.npy"
        np.save(outpath, attn_w.cpu().numpy().astype(np.float16))
        attn_keys.append(f"{short}_layer{layer}")

        # Hidden subset: first SUBSET_N examples, first SUBSET_LCAP tokens
        h = recorder.hidden_out[name]
        if h.dim() == 3:
            # nn.MultiheadAttention output is (B, L, d) when batch_first=True
            h_subset = h[:SUBSET_N, :SUBSET_LCAP, :].cpu().numpy().astype(np.float32)
        else:
            h_subset = h[:SUBSET_LCAP, :SUBSET_N, :].permute(1, 0, 2).cpu().numpy().astype(np.float32)
        np.save(out_dir / f"hidden_subset_{short}_layer{layer}.npy", h_subset)
        hidden_keys.append(f"{short}_layer{layer}")

        # Mean-pool hidden over valid tokens for a quick summary
        if "drug" in short:
            valid = (~d_mask[:SUBSET_N]).float()
        elif "prot" in short:
            valid = (~p_mask[:SUBSET_N]).float()
        else:
            valid = torch.ones(SUBSET_N, h.shape[1], device=h.device)
        m = valid[..., None]
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = (h[:SUBSET_N, :SUBSET_LCAP] * m[:, :SUBSET_LCAP]).sum(dim=1) / denom
        np.save(out_dir / f"pooled_{short}_layer{layer}.npy", pooled.cpu().numpy().astype(np.float32))

    recorder.detach()

    # MSE on the analysis batch
    mse = float(((preds.cpu() - y) ** 2).mean())
    meta = {
        "variant": variant,
        "ckpt": str(ROOT / "outputs" / "phase_e_xl" / f"phase_e_xl_{variant}_random_seed42" / "checkpoints" / variant / "best_model.pt"),
        "n_analyzed": N_ANALYZED, "subset_n": SUBSET_N, "subset_lcap": SUBSET_LCAP,
        "phase": "e1", "d_model": 256, "n_heads": 8,
        "n_layers": 6 if variant in ("early_concat", "early_crossattn") else 3,
        "mse_on_analysis_batch": mse,
        "attn_keys": sorted(attn_keys),
        "hidden_keys": sorted(hidden_keys),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  -> {len(attn_keys)} attn arrays, mse={mse:.4f}", flush=True)


def canonical_name(module_path: str, variant: str):
    """Map nn.MultiheadAttention module path to a (component, layer) tuple.
    Mirrors the original Phase D naming conventions:
      V1/V2 single concat encoder        -> 'encoder', layer N
      V3/V4 separate drug + prot encoders -> 'drug_encoder' / 'prot_encoder', layer N
    """
    # Heuristics: pull layer index from the path
    import re
    m = re.search(r"layers?\.(\d+)\.self_attn$", module_path) or \
        re.search(r"\.(\d+)\.self_attn$", module_path)
    if not m: return None, None
    layer = int(m.group(1))

    if variant in ("early_concat", "early_crossattn"):
        return "encoder", layer
    # late variants: differentiate drug vs protein by path prefix
    if "drug" in module_path:
        return "drug_encoder", layer
    if "prot" in module_path:
        return "prot_encoder", layer
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", choices=list(REGISTRY) + ["all"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}", flush=True)

    print("Loading data…", flush=True)
    loader = build_loader_and_vocabs()
    print(f"  loaded loader, batch_size={N_ANALYZED}", flush=True)

    base = ROOT / "phase_d_artifacts_deep" / "analysis_deep_e1"
    base.mkdir(parents=True, exist_ok=True)

    targets = list(REGISTRY) if args.variant == "all" else [args.variant]
    for v in targets:
        try:
            out_dir = base / VARIANT_FOLDER[v]
            extract_for_variant(v, device, loader, out_dir)
            torch.cuda.empty_cache()  # free between variants
        except Exception as e:
            print(f"[FAIL] {v}: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
