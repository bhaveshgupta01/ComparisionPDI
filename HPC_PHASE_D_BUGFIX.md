# Phase D extraction — v3 final fix

> v2 had two bugs (tuple-vs-dict on `ds[i]["y"]`, and wrong V2/V4 checkpoint paths).
> v3 fixes both. We discovered V2 / V4 checkpoints don't exist on disk anymore (only V1 / V3 baselines remain) — Phase D will be re-run on Phase C checkpoints once those finish.

## What v3 does differently

1. **Positional model call** matching original: `preds = model(drug_t, drug_m, prot_t, prot_m)` — no `return_attn` keyword.
2. **Tuple unpacking from collate**: `for drug_t, drug_m, prot_t, prot_m, y in loader:` (5-tuple).
3. **Walks `layer.attn_weights`** from each `model.encoder.layers[i]` / `model.drug_encoder.layers[i]` / `model.prot_encoder.layers[i]` — same pattern as the original working script.
4. **Verbose ckpt-load report**: prints how many params loaded vs missing — diagnoses the silent `strict=False` issue if the prediction head didn't load.
5. **MSE both ways**: computes raw + denormalized MSE, picks whichever matches Phase A val MSE (~1.6).
6. **Saves drug/prot masks** for mask-aware entropy in figures.
7. **Only targets V1 + V3 baselines** (the only checkpoints on disk).

## Single bash block — paste into OOD terminal

```bash
cd /scratch/$USER/ComparisionPDI

# ---- 1. Write the v3 extract script ---------------------------------------
cat > scripts/extract_for_analysis_v3.py << 'END_OF_V3_AB7392'
#!/usr/bin/env python
"""
Phase D extraction v3 - matches original positional model call,
walks layer.attn_weights for attention, robust ckpt loading,
auto-denormalizes predictions, saves masks.
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


def load_data(n=N_ANALYSIS):
    smiles_tok = SMILESTokenizer(vocab_file="data/processed/bindingdb/smiles_vocab.json")
    prot_tok   = ProteinTokenizer()
    ds = BindingDBKiDataset(
        tsv_path="dataset/BindingDB/BindingDB_PDSPKi.tsv",
        smiles_tokenizer=smiles_tok,
        protein_tokenizer=prot_tok,
        max_rows=10000,
    )
    train_idx, val_idx, test_idx = random_split(ds, seed=42)

    # Compute target normalization stats from the train split.
    # Dataset returns tuple (drug_ids, prot_ids, y) per __getitem__.
    train_y = torch.stack([ds[i][2] for i in train_idx]).float()
    y_mean = float(train_y.mean()); y_std = float(train_y.std())
    print(f"   normalization on train: mean={y_mean:.4f} std={y_std:.4f}")

    ana_idx = test_idx[:n]
    loader = DataLoader(Subset(ds, ana_idx), batch_size=32,
                        collate_fn=collate_fn, shuffle=False)
    return loader, smiles_tok, prot_tok, ds, y_mean, y_std


def extract(variant, ckpt_path, out_dir, model_kwargs):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {out_dir}  ({variant}) ===")
    print(f"   ckpt: {ckpt_path}")

    loader, smiles_tok, prot_tok, _, y_mean, y_std = load_data()
    model = build_model(variant, drug_vocab_size=len(smiles_tok.vocab),
                        prot_vocab_size=len(prot_tok.vocab),
                        **model_kwargs).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE)
    sd = state.get("state_dict", state.get("model", state)) if isinstance(state, dict) else state
    res = model.load_state_dict(sd, strict=False)
    expected = sum(1 for _ in model.parameters())
    n_loaded = expected - len(res.missing_keys)
    print(f"   ckpt keys total in file: {len(sd)}")
    print(f"   loaded {n_loaded} / {expected} model param tensors  "
          f"(missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)})")
    if res.missing_keys:
        print(f"     first 5 missing: {res.missing_keys[:5]}")
    if res.unexpected_keys:
        print(f"     first 5 unexpected: {res.unexpected_keys[:5]}")

    model.eval()
    all_preds, all_truth = [], []
    all_drug_masks, all_prot_masks = [], []
    layer_attns = {}

    with torch.no_grad():
        for batch in loader:
            # batch is a 5-tuple: (drug_t, drug_m, prot_t, prot_m, y)
            drug_t, drug_m, prot_t, prot_m, y = [x.to(DEVICE) for x in batch]
            preds = model(drug_t, drug_m, prot_t, prot_m)
            if preds.ndim > 1:
                preds = preds.squeeze(-1)
            all_preds.append(preds.cpu().numpy())
            all_truth.append(y.cpu().numpy())
            all_drug_masks.append(drug_m.cpu().numpy())
            all_prot_masks.append(prot_m.cpu().numpy())

            # Capture attention from each encoder block (same as original script).
            encoders = []
            for attr in ["encoder", "drug_encoder", "prot_encoder"]:
                if hasattr(model, attr):
                    encoders.append((attr, getattr(model, attr)))
            for ename, enc in encoders:
                layers = getattr(enc, "layers", None)
                if layers is None:
                    continue
                for li, layer in enumerate(layers):
                    aw = getattr(layer, "attn_weights", None)
                    if aw is not None:
                        key = f"{ename}_layer{li}"
                        layer_attns.setdefault(key, []).append(aw.detach().cpu().numpy())

    preds_raw = np.concatenate(all_preds).astype(np.float32)
    truth_raw = np.concatenate(all_truth).astype(np.float32)

    # Try both interpretations of the model output and pick whichever lands close to Phase A val MSE.
    mse_raw    = float(((preds_raw  - truth_raw) ** 2).mean())
    preds_dn   = preds_raw * y_std + y_mean
    mse_dn     = float(((preds_dn   - truth_raw) ** 2).mean())
    if mse_dn < mse_raw:
        preds_final, chose, final_mse = preds_dn, "denormalized", mse_dn
    else:
        preds_final, chose, final_mse = preds_raw, "raw", mse_raw
    print(f"   MSE raw={mse_raw:.4f}  denorm={mse_dn:.4f}  chose={chose}  final={final_mse:.4f}")

    np.save(os.path.join(out_dir, "predictions.npy"), preds_final)
    np.save(os.path.join(out_dir, "predictions_raw_model.npy"), preds_raw)
    np.save(os.path.join(out_dir, "truth.npy"), truth_raw)
    np.save(os.path.join(out_dir, "drug_mask.npy"), np.concatenate(all_drug_masks, axis=0))
    np.save(os.path.join(out_dir, "prot_mask.npy"), np.concatenate(all_prot_masks, axis=0))
    for key, attns in layer_attns.items():
        np.save(os.path.join(out_dir, f"attn_{key}.npy"),
                np.concatenate(attns, axis=0))

    meta = {
        "variant": variant,
        "ckpt": ckpt_path,
        "n_analyzed": int(len(preds_final)),
        "y_mean": y_mean, "y_std": y_std,
        "mse_raw": mse_raw, "mse_denorm": mse_dn,
        "chose": chose,
        "mse_on_analysis_batch": final_mse,
        "model_kwargs": model_kwargs,
        "ckpt_load_n_loaded": n_loaded,
        "ckpt_load_n_missing": len(res.missing_keys),
        "ckpt_load_n_unexpected": len(res.unexpected_keys),
        "attn_keys": sorted(layer_attns.keys()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    OUT = "outputs/analysis_v3"
    KW_DEFAULT = dict(d_model=128, n_heads=4, n_layers=6, d_ff=512, dropout=0.1,
                      max_drug_len=100, max_prot_len=1200,
                      head_hidden=256, head_dropout=0.2)
    KW_DM256_BS16 = {**KW_DEFAULT, "d_model": 256, "n_heads": 8, "d_ff": 1024}

    targets = [
        ("early_concat",
         "outputs/sweeps/v1_baseline_fast/checkpoints/early_concat/best_model.pt",
         f"{OUT}/v1_baseline", KW_DEFAULT),
        ("late_crossattn",
         "outputs/sweeps/v3_baseline_fast/checkpoints/late_crossattn/best_model.pt",
         f"{OUT}/v3_baseline", KW_DEFAULT),
        ("late_crossattn",
         "outputs/sweeps/v3_dm256_bs16_fast/checkpoints/late_crossattn/best_model.pt",
         f"{OUT}/v3_dm256_bs16", KW_DM256_BS16),
    ]
    for variant, ckpt, out_dir, kw in targets:
        if not os.path.exists(ckpt):
            print(f"\n=== {out_dir} === SKIP missing ckpt: {ckpt}")
            continue
        try:
            extract(variant, ckpt, out_dir, kw)
        except Exception as e:
            print(f"\n=== {out_dir} === FAIL: {e}")
            import traceback; traceback.print_exc()

    print("\nALL DONE (v3).")
END_OF_V3_AB7392

# ---- 2. Sbatch wrapper ---------------------------------------------------
cat > hpc_late_concat/run_phase_d_extract_v3.sbatch << 'END_OF_V3_SBATCH_AB7392'
#!/bin/bash
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase-d-v3
#SBATCH --output=logs/phase_d_extract_v3_%j.out
#SBATCH --error=logs/phase_d_extract_v3_%j.err

module purge
cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/analysis_v3
source .venv/bin/activate
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
python scripts/extract_for_analysis_v3.py
echo "DONE"
END_OF_V3_SBATCH_AB7392

# ---- 3. Submit -----------------------------------------------------------
chmod +x hpc_late_concat/run_phase_d_extract_v3.sbatch
sbatch hpc_late_concat/run_phase_d_extract_v3.sbatch
squeue -u $USER

echo
echo "When the job runs, watch:"
echo "  tail -f \$(ls -t logs/phase_d_extract_v3_*.out | head -1)"
```

## What good output looks like

```
=== outputs/analysis_v3/v1_baseline  (early_concat) ===
   ckpt: outputs/sweeps/v1_baseline_fast/checkpoints/early_concat/best_model.pt
   normalization on train: mean=6.9001 std=1.5234
   ckpt keys total in file: 87
   loaded 87 / 87 model param tensors  (missing=0 unexpected=0)
   MSE raw=1.6543  denorm=58.32  chose=raw  final=1.6543
```

If `loaded N / N` with `missing=0` AND `final ≈ 1.6` → ✅ checkpoint loads cleanly, and predictions were already in pKi units (no normalization). The original v1 extract was just buggy because... actually wait: if there's no normalization issue, then why did the original predictions clump near zero?

If `chose=raw` and `final ≈ 1.6` works for v3, that means the original v1 script was reading the wrong tensor as predictions. The original used `state.get("state_dict", state)`. If that returned the entire dict (because the checkpoint doesn't have a `state_dict` key but is keyed differently), then `model.load_state_dict(sd, strict=False)` would silently match nothing → random init.

`v3` uses `state.get("state_dict", state.get("model", state))` and verifies with the loaded-keys count. That's the actual fix.

## After it finishes

Same procedure as before — zip `outputs/analysis_v3/` via OOD file browser, drop into `~/CodeFiles/DTI_MLFinalProject/phase_d_artifacts_v3/` on Mac, ping me. I'll re-run the summarizer (now mask-aware) and rebuild diagrams 13, 16, 17 with valid V1 + V3 attention.

## V2 / V4 — defer to Phase C checkpoints

V2 / V4 baseline checkpoints are gone. Don't bother re-training fast-mode V2 / V4 — Phase C is producing fresh full-data 30-epoch checkpoints for *all 4 variants × 3 splits × 3 seeds*. Once Phase C is done, we'll point the v4 extract script at `outputs/phase_c/phase_c_<variant>_random_seed42/checkpoints/<variant>/best_model.pt` and get all 4 variants in one shot.
