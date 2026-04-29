#!/usr/bin/env python3
"""
Extract Phase D artifact zips on-the-fly, compute summary stats, save them,
delete the bulky .npy files. Outputs land in phase_d_summaries/.

For each variant artifact dir (v2_baseline, v2_dm256_bs16, v4_baseline):
  - predictions.npy + truth.npy  -> kept (small)
  - meta.json                    -> kept
  - attn_*_layer*.npy            -> entropy_per_layer.npz (mean entropy per head)
                                  -> example_attn_layer{0,mid,last}.npz (1 example, full)
                                  -> THEN deleted to free disk
"""
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ZIP_DIR = ROOT / "phase_d_artifacts"
WORK_DIR = ROOT / "phase_d_artifacts" / "_work"
OUT_DIR = ROOT / "phase_d_summaries"
OUT_DIR.mkdir(exist_ok=True)
WORK_DIR.mkdir(exist_ok=True)


def attention_entropy(attn):
    """Shannon entropy of an attention distribution along the last axis.

    attn: array of shape (..., L_q, L_k) — last axis sums to ~1.
    Returns: (..., L_q) — mean across L_q gives a scalar per (batch, head).
    """
    # Numerical safety
    p = np.clip(attn, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1)  # nats


def summarize_attn_file(npy_path, sample_idx=0):
    """Load one big attn .npy and return summary stats.

    Accepts either (B,H,Lq,Lk) or (B,Lq,Lk) — the HPC extraction script
    averaged over heads, so we get 3D arrays in practice.
    """
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim == 4:
        B, H, Lq, Lk = arr.shape
    elif arr.ndim == 3:
        B, Lq, Lk = arr.shape
        H = 1
    else:
        raise ValueError(f"{npy_path}: shape {arr.shape}, want 3D or 4D")

    # Per-example mean entropy (averaged over query tokens AND heads if present).
    entropy_B = np.zeros((B,), dtype=np.float32)
    # Per-head mean entropy across the batch (only meaningful if 4D)
    entropy_H = np.zeros((H,), dtype=np.float32) if H > 1 else None

    for b in range(B):
        slab = arr[b]                                # (H, Lq, Lk) or (Lq, Lk)
        ent = attention_entropy(slab)               # (H, Lq) or (Lq,)
        entropy_B[b] = ent.mean()
        if H > 1:
            entropy_H += ent.mean(axis=-1)
    if H > 1:
        entropy_H /= B

    # Sample slice for visualisation: a single example, mean over heads if needed.
    sample = arr[sample_idx].astype(np.float32)
    if sample.ndim == 3:                             # (H, Lq, Lk)
        sample_mean_heads = sample.mean(axis=0)
        sample_per_head = sample
    else:                                            # (Lq, Lk) — already mean
        sample_mean_heads = sample
        sample_per_head = None

    return {
        "shape": tuple(arr.shape),
        "entropy_per_example": entropy_B,            # (B,)
        "entropy_per_head": entropy_H,               # (H,) or None
        "sample_attn_mean_heads": sample_mean_heads, # (Lq, Lk)
        "sample_attn_per_head": sample_per_head,     # (H, Lq, Lk) or None
    }


def process_variant_zip(zip_path, summary_subdir, kind):
    """kind in {'shared', 'dual'}.

    'shared' -> V1/V2: one stream of attn_encoder_layer{0..n}.npy
    'dual'   -> V3/V4: attn_drug_encoder_layer*.npy + attn_prot_encoder_layer*.npy
    """
    out_dir = OUT_DIR / summary_subdir
    out_dir.mkdir(exist_ok=True)

    extract_root = WORK_DIR / summary_subdir
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)

    print(f"\n=== {summary_subdir}  ({kind})  =====================")
    print(f"   zip:  {zip_path}")
    print(f"   work: {extract_root}")

    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
        # Pull small files first, in full.
        small = [n for n in names if not n.endswith(".npy") or "attn_" not in n]
        big = [n for n in names if n.endswith(".npy") and "attn_" in n]
        for n in small:
            if n.endswith("/"):
                continue
            print(f"   extract small: {n}")
            zf.extract(n, path=extract_root)
        # Copy small files into out_dir
        for fname in ("predictions.npy", "truth.npy", "meta.json"):
            src = extract_root / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)

        layer_summaries = []  # list of dicts
        for n in sorted(big):
            print(f"   extract big:   {n}  -- summarize -- delete")
            zf.extract(n, path=extract_root)
            local = extract_root / n
            try:
                summary = summarize_attn_file(local, sample_idx=0)
            except Exception as e:
                print(f"      !! failed: {e}")
                local.unlink(missing_ok=True)
                continue

            label = Path(n).stem  # e.g. "attn_encoder_layer3" or "attn_prot_encoder_layer1"
            layer_summaries.append({
                "label": label,
                "entropy_B": summary["entropy_per_example"],
                "sample_mean_heads": summary["sample_attn_mean_heads"],
                "sample_per_head": summary["sample_attn_per_head"],
                "shape": summary["shape"],
            })

            # Delete the giant file immediately to free disk.
            local.unlink(missing_ok=True)

    # Write entropy summary npz (lightweight)
    entropy_data = {ls["label"] + "__entropy_B": ls["entropy_B"] for ls in layer_summaries}
    shapes = {ls["label"]: list(ls["shape"]) for ls in layer_summaries}
    np.savez(out_dir / "entropy_summary.npz", **entropy_data,
             _shapes=np.array([json.dumps(shapes)]))

    # Write sample attention for first / mid / last layer (mean-over-heads, small)
    sample_data = {}
    if layer_summaries:
        keep_ix = [0, len(layer_summaries) // 2, len(layer_summaries) - 1]
        keep_ix = sorted(set(keep_ix))
        for i in keep_ix:
            ls = layer_summaries[i]
            sample_data[ls["label"] + "__mean_heads"] = ls["sample_mean_heads"]
            if ls["sample_per_head"] is not None and i == keep_ix[-1]:
                sample_data[ls["label"] + "__per_head"] = ls["sample_per_head"]
    np.savez_compressed(out_dir / "sample_attn.npz", **sample_data)

    # Cleanup work dir
    shutil.rmtree(extract_root, ignore_errors=True)
    print(f"   wrote: {out_dir}/entropy_summary.npz, sample_attn.npz, predictions.npy, truth.npy, meta.json")


def main():
    targets = [
        # Phase C extraction (correctly-loaded checkpoints) -- USE THESE
        ("v1_phase_c.zip",      "v1_phase_c",      "shared"),
        ("v2_phase_c.zip",      "v2_phase_c",      "shared"),
        ("v3_phase_c.zip",      "v3_phase_c",      "dual"),
        ("v4_phase_c.zip",      "v4_phase_c",      "dual"),
        # Old fast-mode extractions (V1/V3 valid, V2/V4 broken)
        ("v1_baseline.zip",     "v1_baseline",     "shared"),
        ("v2_baseline.zip",     "v2_baseline",     "shared"),
        ("v2_dm256_bs16.zip",   "v2_dm256_bs16",   "shared"),
        ("v3_baseline.zip",     "v3_baseline",     "dual"),
        ("v3_dm256_bs16.zip",   "v3_dm256_bs16",   "dual"),
        ("v4_baseline.zip",     "v4_baseline",     "dual"),
    ]
    for zname, subdir, kind in targets:
        zpath = ZIP_DIR / zname
        if not zpath.exists():
            print(f"!! missing zip: {zpath}")
            continue
        try:
            process_variant_zip(zpath, subdir, kind)
        except Exception as e:
            print(f"!! {zname} failed: {e}")
            import traceback
            traceback.print_exc()

    # Final cleanup
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    print("\nDone. Summaries:")
    for sd in OUT_DIR.iterdir():
        if sd.is_dir():
            files = sorted(p.name for p in sd.iterdir())
            sz = sum(p.stat().st_size for p in sd.iterdir()) / 1024 / 1024
            print(f"  {sd.name}: {sz:.1f} MB  ({', '.join(files)})")


if __name__ == "__main__":
    main()
