#!/usr/bin/env python3
"""
E5 — Mask-aware attention entropy.

The poster's diagram_16 reported attention entropy per layer across V1-V4 but
included pad-token query positions in the per-layer average. This script
recomputes entropy ignoring pad queries and writes:
  - poster_figures/diagram_16b_attention_entropy_mask_aware.png/.svg
  - FINDINGS_E5.md  (numerical comparison: old vs mask-aware)

Inputs read from phase_d_artifacts_deep/analysis_deep_v1/v{1..4}_phase_c/.
No GPU required. Pure NumPy.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART  = ROOT / "phase_d_artifacts_deep" / "analysis_deep_v1"
FIG  = ROOT / "poster_figures"
OUT  = ROOT / "FINDINGS_E5.md"

# Locked colors (Okabe-Ito) — match the poster
COLORS = {
    "v1_phase_c": ("V1 Early Concat",   "#0072B2"),
    "v2_phase_c": ("V2 Early X-Attn",   "#E69F00"),
    "v3_phase_c": ("V3 Late X-Attn",    "#009E73"),
    "v4_phase_c": ("V4 Late Concat",    "#CC79A7"),
}


def attn_entropy_old(p_qk: np.ndarray) -> np.ndarray:
    """Original poster computation: per-row entropy, averaged over ALL query
    positions including pads. Returns shape (B,).
    """
    eps = 1e-9
    H = -(p_qk * np.log(p_qk + eps)).sum(axis=-1)   # (B, L_q)
    return H.mean(axis=-1)                          # (B,)


def attn_entropy_masked(p_qk: np.ndarray, valid_q: np.ndarray) -> np.ndarray:
    """Mask-aware: per-row entropy, averaged only over VALID (non-pad) query
    positions. p_qk shape (B, L_q, L_k); valid_q shape (B, L_q) bool.
    """
    eps = 1e-9
    H = -(p_qk * np.log(p_qk + eps)).sum(axis=-1)   # (B, L_q)
    valid = valid_q.astype(np.float32)
    denom = valid.sum(axis=-1).clip(min=1.0)
    return (H * valid).sum(axis=-1) / denom         # (B,)


def variant_layer_entropies(variant_dir: Path):
    """Returns dict layer_idx -> (old_mean, masked_mean)."""
    drug_mask = np.load(variant_dir / "drug_mask.npy").astype(bool)   # 1 = pad
    prot_mask = np.load(variant_dir / "prot_mask.npy").astype(bool)
    valid_drug = ~drug_mask
    valid_prot = ~prot_mask

    out = {}

    # V1 / V2: concatenated sequence [CLS, drug, SEP, protein]
    encoder_files = sorted(variant_dir.glob("attn_encoder_layer*.npy"))
    if encoder_files:
        Ldrug = drug_mask.shape[1]
        Lprot = prot_mask.shape[1]
        for f in encoder_files:
            layer = int(f.stem.split("layer")[-1])
            attn = np.load(f).astype(np.float32)              # (B, Lcat, Lcat)
            B, Lcat, _ = attn.shape
            extra = Lcat - Ldrug - Lprot                      # CLS + SEP = 2 typically
            valid_q = np.zeros((B, Lcat), dtype=bool)
            # First `extra` positions are special tokens (always valid)
            valid_q[:, :extra] = True
            valid_q[:, extra:extra+Ldrug] = valid_drug[:B]
            valid_q[:, extra+Ldrug:]      = valid_prot[:B, :Lcat - extra - Ldrug]
            old = attn_entropy_old(attn).mean()
            new = attn_entropy_masked(attn, valid_q).mean()
            out[("encoder", layer)] = (float(old), float(new))
        return out

    # V3 / V4: separate drug and protein encoders, layer 0 only available
    for kind, mask_valid, n_kind in (("drug_encoder", valid_drug, drug_mask.shape[1]),
                                     ("prot_encoder", valid_prot, prot_mask.shape[1])):
        for f in sorted(variant_dir.glob(f"attn_{kind}_layer*.npy")):
            layer = int(f.stem.split("layer")[-1])
            attn = np.load(f).astype(np.float32)              # (B, L, L)
            B = attn.shape[0]
            valid_q = mask_valid[:B]
            old = attn_entropy_old(attn).mean()
            new = attn_entropy_masked(attn, valid_q).mean()
            out[(kind, layer)] = (float(old), float(new))
    return out


def main():
    all_results = {}
    for vkey in COLORS:
        vdir = ART / vkey
        if not vdir.exists():
            print(f"[skip] {vkey}: dir missing")
            continue
        all_results[vkey] = variant_layer_entropies(vdir)
        print(f"[ok] {vkey}: {len(all_results[vkey])} (component, layer) entries")

    # ----- figure 16b: V1/V2 multi-layer entropy curve, old vs masked
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for idx, (label, vkeys) in enumerate([("Pad-contaminated (poster diagram_16)", ["v1_phase_c", "v2_phase_c"]),
                                          ("Mask-aware (E5 fix)",                   ["v1_phase_c", "v2_phase_c"])]):
        ax = axes[idx]
        for vkey in vkeys:
            res = all_results.get(vkey, {})
            layers = sorted([l for (k, l) in res.keys() if k == "encoder"])
            if not layers:
                continue
            old_vals = [res[("encoder", l)][0] for l in layers]
            new_vals = [res[("encoder", l)][1] for l in layers]
            vals = old_vals if idx == 0 else new_vals
            name, color = COLORS[vkey]
            ax.plot(layers, vals, "o-", color=color, label=name, linewidth=2, markersize=7)
        ax.set_xlabel("Encoder layer")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Attention entropy (nats)")
    fig.suptitle("E5 — Mask-aware attention entropy: V1 vs V2", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "diagram_16b_attention_entropy_mask_aware.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "diagram_16b_attention_entropy_mask_aware.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] diagram_16b_attention_entropy_mask_aware.png/.svg")

    # ----- numeric summary
    lines = ["# Phase E5 — Mask-aware attention entropy", "",
             "Recomputes attention entropy on the existing Phase D artifacts, but averages",
             "over *valid* query positions only (excludes pad tokens that artificially",
             "inflated the original poster numbers).", "",
             "## Per-variant per-layer entropy (nats)", "",
             "| Variant | Component | Layer | Pad-contaminated (old) | Mask-aware (new) | Δ (new − old) |",
             "|---|---|---|---|---|---|"]
    for vkey, res in all_results.items():
        name = COLORS[vkey][0]
        for (kind, layer), (old, new) in sorted(res.items()):
            d = new - old
            lines.append(f"| {name} | {kind} | {layer} | {old:.3f} | {new:.3f} | {d:+.3f} |")

    # Headline finding: V1 vs V2 ranking preservation
    v1 = all_results.get("v1_phase_c", {})
    v2 = all_results.get("v2_phase_c", {})
    if v1 and v2:
        lines += ["", "## V1 vs V2 ranking check (poster claim: V1 < V2 in entropy preserved under mask-aware)", ""]
        v1_min_old = min(v1[(k, l)][0] for (k, l) in v1 if k == "encoder")
        v1_min_new = min(v1[(k, l)][1] for (k, l) in v1 if k == "encoder")
        v2_min_old = min(v2[(k, l)][0] for (k, l) in v2 if k == "encoder")
        v2_min_new = min(v2[(k, l)][1] for (k, l) in v2 if k == "encoder")
        v1_mean_old = np.mean([v1[(k, l)][0] for (k, l) in v1 if k == "encoder"])
        v1_mean_new = np.mean([v1[(k, l)][1] for (k, l) in v1 if k == "encoder"])
        v2_mean_old = np.mean([v2[(k, l)][0] for (k, l) in v2 if k == "encoder"])
        v2_mean_new = np.mean([v2[(k, l)][1] for (k, l) in v2 if k == "encoder"])
        lines += [f"- V1 mean entropy across layers: {v1_mean_old:.3f} (old) → {v1_mean_new:.3f} (new)",
                  f"- V2 mean entropy across layers: {v2_mean_old:.3f} (old) → {v2_mean_new:.3f} (new)",
                  f"- V1 minimum entropy (the layer-4 dip claim): {v1_min_old:.3f} (old) → {v1_min_new:.3f} (new)",
                  f"- V2 minimum entropy: {v2_min_old:.3f} (old) → {v2_min_new:.3f} (new)",
                  ""]
        if v1_mean_new < v2_mean_new:
            lines.append(f"**Ranking preserved**: V1 < V2 in mean entropy (V1={v1_mean_new:.3f} < V2={v2_mean_new:.3f}). The poster's central claim — V1 attention is more concentrated than V2 — survives mask-aware reanalysis.")
        else:
            lines.append(f"**Ranking REVERSED under mask-aware**: V1={v1_mean_new:.3f}, V2={v2_mean_new:.3f}. The poster's claim about V1 specialization needs revision — the original 'dip' was a pad-token artifact.")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"[saved] {OUT.name}")


if __name__ == "__main__":
    main()
