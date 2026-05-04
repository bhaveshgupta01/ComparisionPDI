#!/usr/bin/env python3
"""
Mac-side deep-analysis on artifacts produced by extract_deep_v1.py.

Inputs:  phase_d_artifacts_deep/analysis_deep_v1/v{1,2,3,4}_phase_c/
Outputs: poster_figures/diagram_{19,20,21,22,23}_*.{png,svg}
         FINDINGS_DEEP.md  (numerical summary)

Run order (no HPC needed):
  1. unzip analysis_deep_v1.zip into phase_d_artifacts_deep/
  2. python scripts/analyze_deep.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART  = ROOT / "phase_d_artifacts_deep" / "analysis_deep_v1"
FIG  = ROOT / "poster_figures"
OUT_FINDINGS = ROOT / "FINDINGS_DEEP.md"

VARIANTS = [
    ("v1_phase_c", "V1 Early Concat",    "encoder"),
    ("v2_phase_c", "V2 Early X-Attn",    "encoder"),
    ("v3_phase_c", "V3 Late X-Attn",     "dual"),
    ("v4_phase_c", "V4 Late Concat",     "dual"),
]
COLORS = {"v1_phase_c": "#1f77b4", "v2_phase_c": "#ff7f0e",
          "v3_phase_c": "#2ca02c", "v4_phase_c": "#d62728"}


# -------------------- helpers --------------------------------------------

def load_hidden_subset(variant_dir: Path) -> dict[str, np.ndarray]:
    """Per-token hidden states for the first SUBSET_N=32 examples,
    first SUBSET_LCAP=200 tokens. Returns float32."""
    out = {}
    for p in sorted(variant_dir.glob("hidden_subset_*.npy")):
        out[p.stem.replace("hidden_subset_", "")] = np.load(p).astype(np.float32)
    return out


def correctly_pooled_from_subset(variant_dir: Path) -> dict[str, np.ndarray]:
    """Re-pool hidden_subset_* arrays with the *correct* mask polarity.

    Convention: drug_mask / prot_mask use 1=PADDING. Valid tokens are where
    mask == False. V1/V2 hidden_subset is over the concat sequence; the
    first 100 cols are drug, the next 100 are protein. V3/V4 are split.
    """
    hs = load_hidden_subset(variant_dir)
    if not hs:
        return {}
    drug_mask = np.load(variant_dir / "drug_mask.npy").astype(bool)
    prot_mask = np.load(variant_dir / "prot_mask.npy").astype(bool)
    SN = min(32, drug_mask.shape[0])

    out = {}
    for key, H in hs.items():
        # H: (SN, L_subset, d)
        SN_, L, _ = H.shape
        if key.startswith("drug_encoder"):
            valid = ~drug_mask[:SN_, :L]                 # (SN, L)
        elif key.startswith("prot_encoder"):
            valid = ~prot_mask[:SN_, :L]
        elif key.startswith("encoder"):
            # V1/V2 concat: first 100 = drug tokens, next L-100 = protein.
            Ldrug = min(100, L)
            valid = np.zeros((SN_, L), dtype=bool)
            valid[:, :Ldrug] = ~drug_mask[:SN_, :Ldrug]
            valid[:, Ldrug:] = ~prot_mask[:SN_, : L - Ldrug]
        else:
            valid = np.ones((SN_, L), dtype=bool)
        m = valid[..., None].astype(np.float32)
        denom = m.sum(axis=1).clip(min=1.0)              # (SN, 1)
        pooled = (H * m).sum(axis=1) / denom             # (SN, d)
        out[key] = pooled
    return out


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (N, d) matrices. Robust + centered."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den > 0 else 0.0


def participation_ratio(X: np.ndarray) -> float:
    """PR = (sum λ)^2 / sum λ^2 of the covariance eigenvalues."""
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    eigs = np.linalg.eigvalsh(cov)
    eigs = np.clip(eigs, 0, None)
    s = eigs.sum()
    return float((s ** 2) / (eigs ** 2).sum()) if s > 0 else 0.0


def layer_index(key: str) -> int:
    m = re.search(r"layer(\d+)", key)
    return int(m.group(1)) if m else -1


# -------------------- A.2 mixing-point ----------------------------------

def _pool_with_mask(H: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """H: (N, L, d), valid: (N, L) bool. Mean over valid positions per row."""
    m = valid[..., None].astype(np.float32)
    return (H * m).sum(axis=1) / m.sum(axis=1).clip(min=1.0)


def analyze_mixing_point(findings: list[str]) -> None:
    """Per-layer drug↔prot CKA on the held-out 32-example subset.
    V1/V2: slice concat seq into drug-half / prot-half using mask widths.
    V3/V4: pair drug_encoder_layer_i vs prot_encoder_layer_i.
    Mask polarity: valid = ~mask (mask=1 means PAD).
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, (vdir, vname, kind) in zip(axes, VARIANTS):
        vp = ART / vdir
        if not vp.exists():
            ax.set_title(f"{vname}\n(missing)"); continue
        hs = load_hidden_subset(vp)
        drug_mask = np.load(vp / "drug_mask.npy").astype(bool)[:32]
        prot_mask = np.load(vp / "prot_mask.npy").astype(bool)[:32]

        cka_vals, layer_idxs = [], []
        if kind == "encoder":
            for li in sorted({layer_index(k) for k in hs if k.startswith("encoder_")}):
                key = f"encoder_layer{li}"
                if key not in hs: continue
                H = hs[key]                              # (32, L, d) L≈200
                Ldrug = min(100, H.shape[1])
                drug_valid = ~drug_mask[:H.shape[0], :Ldrug]
                prot_valid = ~prot_mask[:H.shape[0], : H.shape[1] - Ldrug]
                Hd = _pool_with_mask(H[:, :Ldrug],  drug_valid)
                Hp = _pool_with_mask(H[:, Ldrug:],  prot_valid)
                cka_vals.append(linear_cka(Hd, Hp))
                layer_idxs.append(li)
        else:
            # Independent encoders: pair by layer index.
            ds = {layer_index(k): hs[k] for k in hs if k.startswith("drug_encoder_")}
            ps = {layer_index(k): hs[k] for k in hs if k.startswith("prot_encoder_")}
            for li in sorted(ds.keys() & ps.keys()):
                Hd_full, Hp_full = ds[li], ps[li]
                Hd = _pool_with_mask(Hd_full, ~drug_mask[:Hd_full.shape[0], :Hd_full.shape[1]])
                Hp = _pool_with_mask(Hp_full, ~prot_mask[:Hp_full.shape[0], :Hp_full.shape[1]])
                cka_vals.append(linear_cka(Hd, Hp))
                layer_idxs.append(li)

        ax.plot(layer_idxs, cka_vals, "o-", color=COLORS[vdir], lw=2)
        ax.set_title(vname)
        ax.set_xlabel("layer")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        findings.append(f"- **{vname}** mixing-point CKA per layer: " +
                        ", ".join(f"L{l}={c:.2f}" for l, c in zip(layer_idxs, cka_vals)))
    axes[0].set_ylabel("CKA(drug rep, protein rep)")
    fig.suptitle("A.2 — Mixing-point: when do drug & protein representations align?",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "diagram_19_mixing_point")


# -------------------- A.3 gradient flow ---------------------------------

def analyze_gradient_flow(findings: list[str]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, (vdir, vname, _kind) in zip(axes, VARIANTS):
        vp = ART / vdir
        if not vp.exists(): ax.set_title(f"{vname}\n(missing)"); continue
        gd = vp / "grad_drug_emb.npy"
        gp = vp / "grad_prot_emb.npy"
        if not gd.exists() or not gp.exists():
            ax.set_title(f"{vname}\n(no grads)"); continue
        gdrug = np.load(gd).astype(np.float32)      # (B, Ld, d)
        gprot = np.load(gp).astype(np.float32)      # (B, Lp, d)
        # Mask polarity: stored masks use 1=PAD; valid = ~mask.
        drug_valid = ~np.load(vp / "drug_mask.npy").astype(bool)
        prot_valid = ~np.load(vp / "prot_mask.npy").astype(bool)
        nd = np.linalg.norm(gdrug, axis=-1)
        np_ = np.linalg.norm(gprot, axis=-1)
        drug_total = float(nd[drug_valid].sum())
        prot_total = float(np_[prot_valid].sum())
        ratio = drug_total / max(drug_total + prot_total, 1e-9)

        ax.bar(["drug", "prot"], [drug_total, prot_total], color=COLORS[vdir])
        ax.set_title(f"{vname}\ndrug share = {ratio:.2f}")
        ax.set_ylabel("Σ ‖∂preds/∂emb‖₂")
        findings.append(f"- **{vname}** gradient share: drug={ratio:.2f}, "
                        f"protein={1-ratio:.2f}  "
                        f"(total drug={drug_total:.1f}, prot={prot_total:.1f})")
    fig.suptitle("A.3 — Gradient flow: which modality drives predictions?", fontsize=12)
    fig.tight_layout()
    save(fig, "diagram_20_gradient_flow")


# -------------------- B.1 participation ratio ---------------------------

def analyze_participation_ratio(findings: list[str]) -> None:
    """PR per layer using subset-pooled reps (correct mask polarity, N=32)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    table = []
    for vdir, vname, _kind in VARIANTS:
        vp = ART / vdir
        if not vp.exists(): continue
        pooled = correctly_pooled_from_subset(vp)
        layer_to_X = {}
        for key, X in pooled.items():
            li = layer_index(key)
            if li < 0: continue
            layer_to_X.setdefault(li, []).append(X)
        layers = sorted(layer_to_X.keys())
        prs = []
        for li in layers:
            stacked = np.concatenate(layer_to_X[li], axis=0)
            prs.append(participation_ratio(stacked))
        ax.plot(layers, prs, "o-", lw=2, color=COLORS[vdir], label=vname)
        table.append((vname, prs[-1] if prs else float("nan")))
        findings.append(f"- **{vname}** PR per layer: " +
                        ", ".join(f"L{l}={p:.1f}" for l, p in zip(layers, prs)))
    ax.set_xlabel("layer")
    ax.set_ylabel("Participation ratio (effective dim)")
    ax.set_title("B.1 — Representation complexity (effective dim per layer)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, "diagram_21_participation_ratio")
    findings.append("\n**Final-layer PR table:** " +
                    ", ".join(f"{v}={p:.1f}" for v, p in table))


# -------------------- B.3 probing ---------------------------------------

def analyze_probing(findings: list[str]) -> None:
    """Linear probe pooled[final layer] -> truth (binding affinity).

    A trained representation should make affinity at least *partially*
    linearly readable. Higher R^2 = more affinity-relevant info packed
    into the final pooled rep.

    We deliberately do NOT load RDKit here (avoids extra dep on Mac); MW /
    LogP probes can be added by the user if RDKit is installed locally.
    """
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    names, scores = [], []
    rng = np.random.default_rng(0)
    for vdir, vname, _kind in VARIANTS:
        vp = ART / vdir
        if not vp.exists(): continue
        pooled = correctly_pooled_from_subset(vp)
        last = max(layer_index(k) for k in pooled.keys() if layer_index(k) >= 0)
        feats = [X for key, X in pooled.items() if layer_index(key) == last]
        X = np.concatenate(feats, axis=1)            # (32, d) or (32, 2d) for dual
        truth = np.load(vp / "truth.npy")[:X.shape[0]]
        # 5-fold CV ridge to avoid trivial overfit at N=32.
        idx = rng.permutation(X.shape[0])
        K = 5
        folds = np.array_split(idx, K)
        ss_res, ss_tot = 0.0, 0.0
        ymean = float(truth.mean())
        for k in range(K):
            test = folds[k]
            train = np.concatenate([folds[j] for j in range(K) if j != k])
            Xt, yt = X[train], truth[train]
            Xs, ys = X[test],  truth[test]
            mu_x = Xt.mean(0, keepdims=True); mu_y = float(yt.mean())
            Xtc = Xt - mu_x; ytc = yt - mu_y
            lam = 10.0
            w = np.linalg.solve(Xtc.T @ Xtc + lam*np.eye(Xtc.shape[1]), Xtc.T @ ytc)
            yhat = (Xs - mu_x) @ w + mu_y
            ss_res += float(((ys - yhat) ** 2).sum())
            ss_tot += float(((ys - ymean) ** 2).sum())
        r2 = float(1 - ss_res / max(ss_tot, 1e-9))
        rows.append((vname, r2))
        names.append(vname); scores.append(r2)
        findings.append(f"- **{vname}** linear-probe 5-fold CV R^2 to pKi (last layer, N=32): {r2:.3f}")
    ax.bar(names, scores, color=[COLORS[v[0]] for v in VARIANTS if (ART/v[0]).exists()])
    ax.set_ylabel("R² (probe → pKi)")
    ax.set_title("B.3 — Linear probing: how affinity-readable is each rep?")
    ax.set_ylim(0, 1)
    for i, s in enumerate(scores):
        ax.text(i, s + 0.02, f"{s:.2f}", ha="center")
    fig.tight_layout()
    save(fig, "diagram_22_probing")


# -------------------- B.4 t-SNE / UMAP -----------------------------------

def analyze_embedding_geometry(findings: list[str]) -> None:
    """2-D PCA panels coloured by binding affinity. (We use PCA instead of
    t-SNE to avoid the sklearn dependency mess; PCA is enough for a
    qualitative geometry check.)"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (vdir, vname, _kind) in zip(axes, VARIANTS):
        vp = ART / vdir
        if not vp.exists(): ax.set_title(f"{vname}\n(missing)"); continue
        pooled = correctly_pooled_from_subset(vp)
        last = max(layer_index(k) for k in pooled.keys() if layer_index(k) >= 0)
        feats = [X for k, X in pooled.items() if layer_index(k) == last]
        X = np.concatenate(feats, axis=1)
        truth = np.load(vp / "truth.npy")[:X.shape[0]]
        # PCA via SVD.
        Xc = X - X.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ Vt[:2].T
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=truth, cmap="viridis",
                        s=12, alpha=0.85)
        ax.set_title(vname); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_xticks([]); ax.set_yticks([])
        # Variance explained by PC1+PC2.
        ev = (S ** 2) / (S ** 2).sum()
        findings.append(f"- **{vname}** PC1+PC2 variance share: "
                        f"{ev[:2].sum():.2f}; total feat dim={X.shape[1]}")
    fig.suptitle("B.4 — 2-D PCA of final pooled reps, coloured by pKi", fontsize=12)
    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02, label="pKi")
    save(fig, "diagram_23_pca_panels")


# -------------------- IO -------------------------------------------------

def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg",          bbox_inches="tight")
    plt.close(fig)
    print(f"   saved {FIG/name}.png + .svg")


def main() -> None:
    if not ART.exists():
        print(f"!! artifacts not found at {ART}")
        print("   Pull analysis_deep_v1.zip from HPC and unzip it first.")
        return

    findings = ["# Deep analysis (Mac-side)\n"]
    print("→ A.2 mixing-point");                analyze_mixing_point(findings)
    print("→ A.3 gradient flow");               analyze_gradient_flow(findings)
    print("→ B.1 participation ratio");         analyze_participation_ratio(findings)
    print("→ B.3 probing (R² to pKi)");         analyze_probing(findings)
    print("→ B.4 PCA geometry");                analyze_embedding_geometry(findings)

    OUT_FINDINGS.write_text("\n".join(findings) + "\n")
    print(f"\nWrote {OUT_FINDINGS}")
    print("Done.")


if __name__ == "__main__":
    main()
