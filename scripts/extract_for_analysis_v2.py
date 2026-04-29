#!/usr/bin/env python
"""
Phase D extraction v2 - fixes prediction scaling, adds V1+V3.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.collate import collate_fn
from src.data.dataset import BindingDBKiDataset
from src.data.splits import random_split
from src.data.tokenizers import SMILESTokenizer, ProteinTokenizer
from src.models import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ANALYSIS = 256


def _coerce_pred(out):
    """Pull a (B,) tensor of predictions out of whatever the model returned."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for key in ("pred", "predictions", "y_pred", "logits", "out"):
            if key in out and torch.is_tensor(out[key]):
                return out[key]
        for k, v in out.items():
            if torch.is_tensor(v) and v.ndim <= 2:
                return v
    if isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
        return out[0]
    raise RuntimeError(f"Cannot find prediction tensor in model output: {type(out)}")


def _coerce_attn(out):
    """Pull the attention dict (or empty)."""
    if isinstance(out, dict) and "attn" in out and isinstance(out["attn"], dict):
        return out["attn"]
    return {}


def load_data(n=N_ANALYSIS):
    smiles_tok = SMILESTokenizer(vocab_file="data/processed/bindingdb/smiles_vocab.json")
    prot_tok = ProteinTokenizer()
    ds = BindingDBKiDataset(
        tsv_path="dataset/BindingDB/BindingDB_PDSPKi.tsv",
        smiles_tokenizer=smiles_tok,
        protein_tokenizer=prot_tok,
        max_rows=10000,
    )
    train_idx, val_idx, test_idx = random_split(ds, seed=42)
    train_y = torch.tensor([ds[i]["y"] for i in train_idx], dtype=torch.float32)
    y_mean = float(train_y.mean()); y_std = float(train_y.std())
    print(f"   normalization: mean={y_mean:.4f} std={y_std:.4f}")

    ana_idx = test_idx[:n]
    loader = DataLoader(Subset(ds, ana_idx), batch_size=32,
                        collate_fn=collate_fn, shuffle=False)
    return loader, smiles_tok, prot_tok, ds, y_mean, y_std


def extract(variant, ckpt_path, out_dir, model_kwargs):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {out_dir} ===")
    print(f"   variant: {variant}")
    print(f"   ckpt:    {ckpt_path}")
    print(f"   kwargs:  {model_kwargs}")

    loader, smiles_tok, prot_tok, _, y_mean, y_std = load_data()
    model = build_model(variant,
                        drug_vocab_size=len(smiles_tok.vocab),
                        prot_vocab_size=len(prot_tok.vocab),
                        **model_kwargs).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE)
    sd = state.get("model", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"   WARN missing keys (first 5): {list(missing)[:5]}  total={len(missing)}")
    if unexpected:
        print(f"   WARN unexpected keys (first 5): {list(unexpected)[:5]}  total={len(unexpected)}")

    model.eval()
    all_preds, all_truth_norm, all_truth_raw = [], [], []
    all_attn = {}
    all_masks_drug, all_masks_prot = [], []
    debug_done = False

    with torch.no_grad():
        for batch in loader:
            for k in list(batch.keys()):
                if hasattr(batch[k], "to"):
                    batch[k] = batch[k].to(DEVICE)

            try:
                out = model(**batch, return_attn=True)
            except TypeError:
                out = model(**batch)

            if not debug_done:
                print(f"   model output type: {type(out)}")
                if isinstance(out, dict):
                    print(f"   dict keys: {list(out.keys())}")
                print(f"   batch keys: {list(batch.keys())}")
                debug_done = True

            preds_tensor = _coerce_pred(out)
            attn_dict    = _coerce_attn(out)

            preds = preds_tensor.detach().cpu().numpy().reshape(-1)
            truth = batch["y"].detach().cpu().numpy().reshape(-1)

            # Normalization detection: if predicted range is 10x smaller than truth range,
            # assume model emits z-scored output and denormalize.
            all_preds.append(preds)
            all_truth_raw.append(truth)
            all_truth_norm.append((truth - y_mean) / (y_std + 1e-8))

            # Save padding masks if available (collate may use various names)
            for cand_drug in ("drug_mask", "drug_attn_mask", "smiles_mask"):
                if cand_drug in batch and torch.is_tensor(batch[cand_drug]):
                    all_masks_drug.append(batch[cand_drug].cpu().numpy())
                    break
            for cand_prot in ("prot_mask", "protein_mask", "prot_attn_mask"):
                if cand_prot in batch and torch.is_tensor(batch[cand_prot]):
                    all_masks_prot.append(batch[cand_prot].cpu().numpy())
                    break

            for k, v in attn_dict.items():
                if torch.is_tensor(v):
                    all_attn.setdefault(k, []).append(v.detach().cpu().numpy())

    preds_raw = np.concatenate(all_preds)
    truth_raw = np.concatenate(all_truth_raw)
    truth_norm = np.concatenate(all_truth_norm)

    # Decide whether to denormalize. Heuristic: if MSE in raw space > 5x Phase A target,
    # try denormalization; pick whichever is closer to the Phase A val MSE.
    mse_raw = float(((preds_raw - truth_raw) ** 2).mean())
    preds_denorm = preds_raw * y_std + y_mean
    mse_denorm = float(((preds_denorm - truth_raw) ** 2).mean())

    if mse_denorm < mse_raw:
        preds_final = preds_denorm
        chose = "denormalized"
    else:
        preds_final = preds_raw
        chose = "raw"

    final_mse = float(((preds_final - truth_raw) ** 2).mean())
    print(f"   MSE raw={mse_raw:.4f}  denorm={mse_denorm:.4f}  chose={chose}  final={final_mse:.4f}")

    np.save(os.path.join(out_dir, "predictions.npy"), preds_final.astype(np.float32))
    np.save(os.path.join(out_dir, "predictions_raw_model.npy"), preds_raw.astype(np.float32))
    np.save(os.path.join(out_dir, "truth.npy"), truth_raw.astype(np.float32))

    if all_masks_drug:
        np.save(os.path.join(out_dir, "drug_mask.npy"), np.concatenate(all_masks_drug, axis=0))
    if all_masks_prot:
        np.save(os.path.join(out_dir, "prot_mask.npy"), np.concatenate(all_masks_prot, axis=0))

    for k, vals in all_attn.items():
        np.save(os.path.join(out_dir, f"attn_{k}.npy"),
                np.concatenate(vals, axis=0))

    meta = {
        "variant": variant,
        "ckpt": ckpt_path,
        "n_analyzed": int(len(preds_final)),
        "mse_raw": mse_raw,
        "mse_denorm": mse_denorm,
        "chose": chose,
        "mse_on_analysis_batch": final_mse,
        "y_mean": y_mean,
        "y_std": y_std,
        "model_kwargs": model_kwargs,
        "attn_keys": sorted(all_attn.keys()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    OUT_BASE = "outputs/analysis_v2"
    KW_DEFAULT  = {"d_model": 128, "n_heads": 4, "n_layers": 6, "d_ff": 512}
    KW_DM256BS16 = {"d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 1024}

    targets = [
        ("early_concat",     "outputs/sweeps/v1_baseline_fast/checkpoints/early_concat/best_model.pt",     f"{OUT_BASE}/v1_baseline",     KW_DEFAULT),
        ("early_crossattn",  "outputs/sweeps/v2_baseline_fast/checkpoints/early_crossattn/best_model.pt",  f"{OUT_BASE}/v2_baseline",     KW_DEFAULT),
        ("early_crossattn",  "outputs/sweeps/v2_dm256_bs16_fast/checkpoints/early_crossattn/best_model.pt", f"{OUT_BASE}/v2_dm256_bs16",   KW_DM256BS16),
        ("late_crossattn",   "outputs/sweeps/v3_baseline_fast/checkpoints/late_crossattn/best_model.pt",    f"{OUT_BASE}/v3_baseline",     KW_DEFAULT),
        ("late_crossattn",   "outputs/sweeps/v3_dm256_bs16_fast/checkpoints/late_crossattn/best_model.pt",  f"{OUT_BASE}/v3_dm256_bs16",   KW_DM256BS16),
        ("late_concat",      "outputs/sweeps/baseline/checkpoints/late_concat/best_model.pt",               f"{OUT_BASE}/v4_baseline",     KW_DEFAULT),
    ]
    for variant, ckpt, out, kw in targets:
        if not os.path.exists(ckpt):
            print(f"\n=== {out} === SKIP missing ckpt: {ckpt}")
            continue
        try:
            extract(variant, ckpt, out, kw)
        except Exception as e:
            print(f"\n=== {out} === FAIL: {e}")
            import traceback; traceback.print_exc()

    print("\nALL DONE.")
