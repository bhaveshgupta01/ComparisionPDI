#!/usr/bin/env python
"""Phase D extraction from Phase C checkpoints — fills V2 and V4 gap."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from src.data.collate import collate_fn
from src.data.dataset import BindingDBKiDataset
from src.data.splits import random_split
from src.data.tokenizers import SMILESTokenizer, ProteinTokenizer
from src.models import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ANALYSIS = 256

def load_data():
    st = SMILESTokenizer(vocab_file="data/processed/bindingdb/smiles_vocab.json")
    pt = ProteinTokenizer()
    ds = BindingDBKiDataset(tsv_path="dataset/BindingDB/BindingDB_PDSPKi.tsv",
        smiles_tokenizer=st, protein_tokenizer=pt, max_rows=10000)
    train_idx, _, test_idx = random_split(ds, seed=42)
    train_y = torch.stack([ds[i][2] for i in train_idx]).float()
    return DataLoader(Subset(ds, test_idx[:N_ANALYSIS]), batch_size=32, collate_fn=collate_fn, shuffle=False), st, pt, float(train_y.mean()), float(train_y.std())

def extract(variant, ckpt, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {out_dir}  ({variant}) ===")
    loader, st, pt, y_mean, y_std = load_data()
    KW = dict(d_model=128, n_heads=4, n_layers=3 if variant.startswith("late") else 6,
              d_ff=512, dropout=0.1, max_drug_len=100, max_prot_len=1200,
              head_hidden=256, head_dropout=0.2)
    model = build_model(variant, drug_vocab_size=len(st.vocab),
                        prot_vocab_size=len(pt.vocab), **KW).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    sd = state.get("model_state_dict", state.get("state_dict", state))
    res = model.load_state_dict(sd, strict=False)
    print(f"   loaded missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")

    model.eval()
    preds, truth, drug_masks, prot_masks = [], [], [], []
    layer_attns = {}
    with torch.no_grad():
        for d_t, d_m, p_t, p_m, y in loader:
            d_t, d_m, p_t, p_m, y = [x.to(DEVICE) for x in (d_t, d_m, p_t, p_m, y)]
            out = model(d_t, d_m, p_t, p_m)
            if out.ndim > 1: out = out.squeeze(-1)
            preds.append(out.cpu().numpy())
            truth.append(y.cpu().numpy())
            drug_masks.append(d_m.cpu().numpy())
            prot_masks.append(p_m.cpu().numpy())
            for attr in ["encoder", "drug_encoder", "prot_encoder"]:
                if hasattr(model, attr):
                    enc = getattr(model, attr)
                    layers = getattr(enc, "layers", None)
                    if layers is None: continue
                    for li, layer in enumerate(layers):
                        aw = getattr(layer, "attn_weights", None)
                        if aw is not None:
                            layer_attns.setdefault(f"{attr}_layer{li}", []).append(aw.detach().cpu().numpy())

    p = np.concatenate(preds).astype(np.float32)
    t = np.concatenate(truth).astype(np.float32)
    p_dn = p * y_std + y_mean
    mse_raw, mse_dn = float(((p - t) ** 2).mean()), float(((p_dn - t) ** 2).mean())
    final, chose = (p_dn, "denormalized") if mse_dn < mse_raw else (p, "raw")
    print(f"   MSE raw={mse_raw:.3f}  denorm={mse_dn:.3f}  chose={chose}")
    print(f"   preds range [{final.min():.2f}, {final.max():.2f}], std={final.std():.3f}")

    np.save(os.path.join(out_dir, "predictions.npy"), final)
    np.save(os.path.join(out_dir, "truth.npy"), t)
    np.save(os.path.join(out_dir, "drug_mask.npy"), np.concatenate(drug_masks, axis=0))
    np.save(os.path.join(out_dir, "prot_mask.npy"), np.concatenate(prot_masks, axis=0))
    for k, vs in layer_attns.items():
        np.save(os.path.join(out_dir, f"attn_{k}.npy"), np.concatenate(vs, axis=0))
    json.dump(dict(variant=variant, ckpt=ckpt, mse_raw=mse_raw, mse_denorm=mse_dn,
                   chose=chose, mse_on_analysis_batch=min(mse_raw, mse_dn),
                   y_mean=y_mean, y_std=y_std, n=int(len(p)),
                   load_n_missing=len(res.missing_keys),
                   load_n_unexpected=len(res.unexpected_keys),
                   attn_keys=sorted(layer_attns.keys())),
              open(os.path.join(out_dir, "meta.json"), "w"), indent=2)


if __name__ == "__main__":
    OUT = "outputs/analysis_phase_c"
    targets = [
        ("early_concat",    "outputs/phase_c/phase_c_early_concat_random_seed42/checkpoints/early_concat/best_model.pt",     f"{OUT}/v1_phase_c"),
        ("early_crossattn", "outputs/phase_c/phase_c_early_crossattn_random_seed42/checkpoints/early_crossattn/best_model.pt", f"{OUT}/v2_phase_c"),
        ("late_crossattn",  "outputs/phase_c/phase_c_late_crossattn_random_seed42/checkpoints/late_crossattn/best_model.pt",   f"{OUT}/v3_phase_c"),
        ("late_concat",     "outputs/phase_c/phase_c_late_concat_random_seed42/checkpoints/late_concat/best_model.pt",         f"{OUT}/v4_phase_c"),
    ]
    for v, c, o in targets:
        if os.path.exists(c):
            try: extract(v, c, o)
            except Exception as e:
                print(f"\n=== {o} === FAIL: {e}")
                import traceback; traceback.print_exc()
        else:
            print(f"\n=== {o} === SKIP missing: {c}")
    print("\nALL DONE.")
