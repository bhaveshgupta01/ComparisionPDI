#!/usr/bin/env python3
"""
Build the CKA matrices at d=128 (existing Phase D artifacts) and d=256 (Phase E1
re-extract). Two side-by-side panels.

Inputs:
  phase_d_artifacts_deep/analysis_deep_v1/v{1..4}_phase_c/    (already on Mac)
  phase_d_artifacts_deep/analysis_deep_e1/v{1..4}_phase_e_xl/ (after extract-e1 lands + scp)

Output:
  poster_figures/diagram_8_cka_comparison.png/.svg
  FINDINGS_CKA.md
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART  = ROOT / "phase_d_artifacts_deep"
FIG  = ROOT / "poster_figures"

VARIANTS = ["v1", "v2", "v3", "v4"]
LABELS   = {"v1": "V1 EC", "v2": "V2 EX", "v3": "V3 LX", "v4": "V4 LC"}

PHASE_C_DIR  = ART / "analysis_deep_v1"
PHASE_E1_DIR = ART / "analysis_deep_e1"

DIR_MAP_C  = {"v1": "v1_phase_c",     "v2": "v2_phase_c",
              "v3": "v3_phase_c",     "v4": "v4_phase_c"}
DIR_MAP_E1 = {"v1": "v1_phase_e_xl",  "v2": "v2_phase_e_xl",
              "v3": "v3_phase_e_xl",  "v4": "v4_phase_e_xl"}


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between (n, d_x) and (n, d_y) feature matrices.
    Kornblith et al., 2019. Robust to differing dimensions."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XtY = X.T @ Y
    num = (XtY ** 2).sum()
    den = np.sqrt((X.T @ X).sum() ** 2 * (Y.T @ Y).sum() ** 2)
    den = np.sqrt(((X.T @ X) ** 2).sum()) * np.sqrt(((Y.T @ Y) ** 2).sum())
    return float(num / (den + 1e-9))


def attention_features(variant_dir: Path) -> np.ndarray | None:
    """Build per-example attention-entropy feature vector. Concatenate per-layer
    mean entropy across whichever attention modules were extracted, giving a
    fixed-size per-example feature for CKA."""
    files = sorted(variant_dir.glob("attn_*.npy"))
    if not files: return None

    drug_mask = np.load(variant_dir / "drug_mask.npy").astype(bool)
    prot_mask = np.load(variant_dir / "prot_mask.npy").astype(bool)

    feats = []
    for f in files:
        attn = np.load(f).astype(np.float32)  # (B, L, L) head-averaged
        B, L, _ = attn.shape

        # build valid-query mask appropriate to component
        name = f.stem.replace("attn_", "")
        if name.startswith("drug"):
            Lvalid = min(L, drug_mask.shape[1])
            valid = np.zeros((B, L), dtype=bool)
            valid[:, :Lvalid] = ~drug_mask[:B, :Lvalid]
        elif name.startswith("prot"):
            Lvalid = min(L, prot_mask.shape[1])
            valid = np.zeros((B, L), dtype=bool)
            valid[:, :Lvalid] = ~prot_mask[:B, :Lvalid]
        else:
            # V1/V2 concat seq: [extra special tokens (CLS/SEP), drug, protein]
            Ldrug = drug_mask.shape[1]   # 100
            Lprot = prot_mask.shape[1]   # 1200
            extra = L - Ldrug - Lprot    # CLS+SEP usually = 2 (or 0 if fits exactly)
            extra = max(extra, 0)
            valid = np.zeros((B, L), dtype=bool)
            valid[:, :extra] = True       # special tokens always valid
            d_end = min(extra + Ldrug, L)
            valid[:, extra:d_end] = ~drug_mask[:B, :d_end - extra]
            p_end = min(d_end + Lprot, L)
            valid[:, d_end:p_end] = ~prot_mask[:B, :p_end - d_end]
        eps = 1e-9
        H = -(attn * np.log(attn + eps)).sum(axis=-1)             # (B, L)
        denom = valid.astype(np.float32).sum(axis=-1).clip(min=1) # (B,)
        per_layer = (H * valid.astype(np.float32)).sum(axis=-1) / denom  # (B,)
        feats.append(per_layer)

    return np.stack(feats, axis=1)  # (B, n_layers)


def compute_cka_matrix(variant_dirs: dict[str, Path]) -> np.ndarray | None:
    feats = {}
    for v, d in variant_dirs.items():
        if not d.exists():
            print(f"  [skip] {v}: dir missing ({d})")
            return None
        f = attention_features(d)
        if f is None:
            print(f"  [skip] {v}: no attn arrays in {d}")
            return None
        feats[v] = f

    M = np.eye(len(VARIANTS))
    for i, vi in enumerate(VARIANTS):
        for j, vj in enumerate(VARIANTS):
            if i == j: continue
            M[i, j] = linear_cka(feats[vi], feats[vj])
    return M


def render(M_c, M_e1, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, M, title in zip(axes, [M_c, M_e1], ["Phase C (d=128)", "Phase E1 (d=256)"]):
        if M is None:
            ax.text(0.5, 0.5, f"{title}\n(missing)", ha="center", va="center")
            ax.axis("off"); continue
        im = ax.imshow(M, cmap="Blues", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([LABELS[v] for v in VARIANTS])
        ax.set_yticklabels([LABELS[v] for v in VARIANTS])
        ax.set_title(title)
        for i in range(4):
            for j in range(4):
                color = "white" if M[i, j] > 0.85 else "black"
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color=color, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.suptitle("Linear CKA on attention-entropy features\n"
                 "Phase C clustered by fusion stage ({V1,V2} vs {V3,V4}); Phase E1 ?", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_findings(M_c, M_e1):
    lines = ["# CKA findings — d=128 vs d=256", ""]
    if M_c is not None:
        lines.append("## Phase C (d=128) CKA matrix\n")
        lines.append("| | " + " | ".join(LABELS[v] for v in VARIANTS) + " |")
        lines.append("|---|" + "---|" * 4)
        for i, vi in enumerate(VARIANTS):
            row = [LABELS[vi]] + [f"{M_c[i, j]:.3f}" for j in range(4)]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    if M_e1 is not None:
        lines.append("## Phase E1 (d=256) CKA matrix\n")
        lines.append("| | " + " | ".join(LABELS[v] for v in VARIANTS) + " |")
        lines.append("|---|" + "---|" * 4)
        for i, vi in enumerate(VARIANTS):
            row = [LABELS[vi]] + [f"{M_e1[i, j]:.3f}" for j in range(4)]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Cluster analysis
        lines += ["## Cluster check", ""]
        in_early = M_e1[0, 1]   # V1-V2
        in_late  = M_e1[2, 3]   # V3-V4
        across   = (M_e1[0, 2] + M_e1[0, 3] + M_e1[1, 2] + M_e1[1, 3]) / 4
        lines.append(f"- Within-early {{V1, V2}} CKA = {in_early:.3f}")
        lines.append(f"- Within-late  {{V3, V4}} CKA = {in_late:.3f}")
        lines.append(f"- Across stages mean CKA      = {across:.3f}")
        if max(in_early, in_late) - across > 0.05:
            lines.append("\n**Fusion-stage clustering preserved at d=256.** Within-cluster similarity exceeds cross-cluster by >0.05.")
        else:
            lines.append("\n**Fusion-stage clustering disappears at d=256.** Variants converge to similar internal representations once capacity is sufficient — interesting in light of the headline reversal.")

    (ROOT / "FINDINGS_CKA.md").write_text("\n".join(lines) + "\n")
    print("[saved] FINDINGS_CKA.md")


def main():
    print("Computing CKA at d=128 …")
    M_c  = compute_cka_matrix({v: PHASE_C_DIR  / DIR_MAP_C[v]  for v in VARIANTS})
    print("Computing CKA at d=256 …")
    M_e1 = compute_cka_matrix({v: PHASE_E1_DIR / DIR_MAP_E1[v] for v in VARIANTS})
    render(M_c, M_e1, FIG / "diagram_8_cka_comparison.png")
    print(f"[saved] diagram_8_cka_comparison.png/.svg")
    write_findings(M_c, M_e1)


if __name__ == "__main__":
    main()
