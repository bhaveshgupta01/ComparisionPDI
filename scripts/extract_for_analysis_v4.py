#!/usr/bin/env python
"""Phase D extraction v4 - fixes ckpt key (was looking for 'state_dict', actual key is 'model_state_dict')."""
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

def load_data(n=N_ANALYSIS):
    smiles_tok = SMILESTokenizer(vocab_file="data/processed/bindingdb/smiles_vocab.json")
    prot_tok   = ProteinTokenizer()
    ds = BindingDBKiDataset(
        tsv_path="dataset/BindingDB/BindingDB_PDSPKi.tsv",
        smiles_tokenizer=smiles_tok, protein_tokenizer=prot_tok, max_rows=10000)
    train_idx, _, test_idx = random_split(ds, seed=42)
    train_y = torch.stack([ds[i][2] for i in train_idx]).float()
    y_mean, y_std = float(train_y.mean()), float(train_y.std())
    print(f"   normalization: mean={y_mean:.4f} std={y_std:.4f}")
    ana_idx = test_idx[:n]
    loader = DataLoader(Subset(ds, ana_idx), batch_size=32, collate_fn=collate_fn, shuffle=False)
    return loader, smiles_tok, prot_tok, y_mean, y_std

def _pick_sd(state):
    if isinstance(state, dict):
        for k in ("model_state_dict", "state_dict", "model", "net"):
            if k in state and isinstance(state[k], dict):
                return state[k], k
    return state, "<root>"

def extract(variant, ckpt_path, out_dir, model_kwargs):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {out_dir}  ({variant}) ===")
    print(f"   ckpt: {ckpt_path}")

    loader, st, pt, y_mean, y_std = load_data()
    model = build_model(variant, drug_vocab_size=len(st.vocab),
                        prot_vocab_size=len(pt.vocab), **model_kwargs).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    sd, key_used = _pick_sd(state)
    print(f"   ckpt key used: {key_used}, sd has {len(sd)} entries")
    res = model.load_state_dict(sd, strict=False)
    print(f"   load_state_dict missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    if res.missing_keys:
        print(f"     first 3 missing: {res.missing_keys[:3]}")
    if res.unexpected_keys:
        print(f"     first 3 unexpected: {res.unexpected_keys[:3]}")

    model.eval()
    all_preds, all_truth = [], []
    all_drug_masks, all_prot_masks = [], []
    layer_attns = {}
    with torch.no_grad():
        for batch in loader:
            drug_t, drug_m, prot_t, prot_m, y = [x.to(DEVICE) for x in batch]
            preds = model(drug_t, drug_m, prot_t, prot_m)
            if preds.ndim > 1: preds = preds.squeeze(-1)
            all_preds.append(preds.cpu().numpy())
            all_truth.append(y.cpu().numpy())
            all_drug_masks.append(drug_m.cpu().numpy())
            all_prot_masks.append(prot_m.cpu().numpy())
            for attr in ["encoder", "drug_encoder", "prot_encoder"]:
                if hasattr(model, attr):
                    enc = getattr(model, attr)
                    layers = getattr(enc, "layers", None)
                    if layers is None: continue
                    for li, layer in enumerate(layers):
                        aw = getattr(layer, "attn_weights", None)
                        if aw is not None:
                            layer_attns.setdefault(f"{attr}_layer{li}", []).append(aw.detach().cpu().numpy())

    preds_raw = np.concatenate(all_preds).astype(np.float32)
    truth_raw = np.concatenate(all_truth).astype(np.float32)
    mse_raw = float(((preds_raw - truth_raw) ** 2).mean())
    preds_dn = preds_raw * y_std + y_mean
    mse_dn = float(((preds_dn - truth_raw) ** 2).mean())
    if mse_dn < mse_raw:
        preds_final, chose, final_mse = preds_dn, "denormalized", mse_dn
    else:
        preds_final, chose, final_mse = preds_raw, "raw", mse_raw
    print(f"   MSE raw={mse_raw:.4f}  denorm={mse_dn:.4f}  chose={chose}  final={final_mse:.4f}")
    print(f"   preds range [{preds_final.min():.3f}, {preds_final.max():.3f}], std={preds_final.std():.3f}")

    np.save(os.path.join(out_dir, "predictions.npy"), preds_final)
    np.save(os.path.join(out_dir, "predictions_raw_model.npy"), preds_raw)
    np.save(os.path.join(out_dir, "truth.npy"), truth_raw)
    np.save(os.path.join(out_dir, "drug_mask.npy"), np.concatenate(all_drug_masks, axis=0))
    np.save(os.path.join(out_dir, "prot_mask.npy"), np.concatenate(all_prot_masks, axis=0))
    for k, vals in layer_attns.items():
        np.save(os.path.join(out_dir, f"attn_{k}.npy"), np.concatenate(vals, axis=0))

    meta = dict(variant=variant, ckpt=ckpt_path, n_analyzed=int(len(preds_final)),
                y_mean=y_mean, y_std=y_std, mse_raw=mse_raw, mse_denorm=mse_dn,
                chose=chose, mse_on_analysis_batch=final_mse,
                model_kwargs=model_kwargs, ckpt_key_used=key_used,
                load_n_missing=len(res.missing_keys),
                load_n_unexpected=len(res.unexpected_keys),
                attn_keys=sorted(layer_attns.keys()))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    OUT = "outputs/analysis_v4"
    KW = dict(d_model=128, n_heads=4, n_layers=6, d_ff=512, dropout=0.1,
              max_drug_len=100, max_prot_len=1200,
              head_hidden=256, head_dropout=0.2)
    KW256 = {**KW, "d_model": 256, "n_heads": 8, "d_ff": 1024}
    targets = [
        ("early_concat",    "outputs/sweeps/v1_baseline_fast/checkpoints/early_concat/best_model.pt",    f"{OUT}/v1_baseline",    KW),
        ("late_crossattn",  "outputs/sweeps/v3_baseline_fast/checkpoints/late_crossattn/best_model.pt",  f"{OUT}/v3_baseline",    KW),
        ("late_crossattn",  "outputs/sweeps/v3_dm256_bs16_fast/checkpoints/late_crossattn/best_model.pt", f"{OUT}/v3_dm256_bs16",   KW256),
    ]
    for v, c, o, kw in targets:
        if not os.path.exists(c):
            print(f"\n=== {o} === SKIP missing: {c}"); continue
        try:
            extract(v, c, o, kw)
        except Exception as e:
            print(f"\n=== {o} === FAIL: {e}")
            import traceback; traceback.print_exc()
    print("\nALL DONE (v4).")
