#!/usr/bin/env python3
"""
Build ablation figures from outputs/phase_e_ablations/SUMMARY.csv (after E4 lands).

Inputs (after `scp` from HPC):
  outputs/phase_e_ablations/SUMMARY.csv
  outputs/phase_e_ablations/<variant>.json
  outputs/phase_e_ablations/rep_swap_v3_v4.json (optional)

Outputs:
  poster_figures/diagram_7_layer_ablation.png/.svg
  poster_figures/diagram_7b_head_ablation.png/.svg
  poster_figures/diagram_7c_rep_swap.png/.svg (if rep_swap_v3_v4.json present)
  FINDINGS_E4.md
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG  = ROOT / "poster_figures"
SUM  = ROOT / "outputs" / "phase_e_ablations" / "SUMMARY.csv"
ABL_DIR = ROOT / "outputs" / "phase_e_ablations"

VARIANTS = ["early_concat", "early_crossattn", "late_crossattn", "late_concat"]
LABELS   = {"early_concat": "V1 EC", "early_crossattn": "V2 EX",
            "late_crossattn": "V3 LX", "late_concat":    "V4 LC"}
COLORS   = {"early_concat": "#0072B2", "early_crossattn": "#E69F00",
            "late_crossattn": "#009E73", "late_concat":    "#CC79A7"}


def load():
    if not SUM.exists():
        print(f"[ERROR] {SUM} not found — pull from HPC first.")
        return None
    rows = list(csv.DictReader(open(SUM)))
    return rows


def parse_layer_idx(module_name: str) -> tuple[str, int]:
    """Extract (component, layer_idx) from a module name like
    'encoder.layers.3.self_attn' or 'drug_encoder.layers.2.self_attn'."""
    import re
    m = re.search(r"(drug_encoder|prot_encoder|encoder)\.layers?\.(\d+)\.self_attn", module_name)
    if not m: return ("?", -1)
    return (m.group(1), int(m.group(2)))


def fig_layer_ablation(rows):
    """Heatmap: variants × layer index, color = ΔMSE.
    For V1/V2 (single shared encoder, 6 layers) and V3/V4 (drug + prot, 3 layers each)
    we plot two stacked panels."""
    layer_data = [r for r in rows if r["kind"] == "layer"]
    by_var = defaultdict(list)  # variant -> list of (component, layer, delta)
    for r in layer_data:
        comp, layer = parse_layer_idx(r["module"])
        if layer < 0: continue
        by_var[r["variant"]].append((comp, layer, float(r["delta"])))

    # Build a layout: 4 variants horizontally, x-axis = layer index, y-axis = component
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    axes = axes.flatten()
    for ax_idx, v in enumerate(VARIANTS):
        ax = axes[ax_idx]
        items = by_var.get(v, [])
        if not items:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(LABELS[v])
            ax.axis("off"); continue
        components = sorted(set(c for c, _, _ in items))
        max_layer  = max(l for _, l, _ in items)
        grid = np.zeros((len(components), max_layer + 1))
        grid[:] = np.nan
        for c, l, d in items:
            grid[components.index(c), l] = d
        im = ax.imshow(grid, aspect="auto", cmap="Reds", vmin=0,
                       vmax=max(0.05, np.nanmax(grid)))
        ax.set_yticks(range(len(components))); ax.set_yticklabels(components)
        ax.set_xticks(range(max_layer + 1))
        ax.set_xlabel("Layer index")
        ax.set_title(f"{LABELS[v]} — Δ MSE per layer (zero attn output)")
        for i in range(len(components)):
            for j in range(max_layer + 1):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Layer ablation on Phase C (d=128) — Δ MSE when self-attn output is zeroed (residual still passes)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"diagram_7_layer_ablation.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] diagram_7_layer_ablation.png/.svg")


def fig_head_ablation(rows):
    """Per (variant, layer): max-head ΔMSE, sum-of-heads ΔMSE — shows how concentrated head importance is."""
    head_data = [r for r in rows if r["kind"] == "head"]
    if not head_data: return
    by = defaultdict(list)  # (variant, component, layer) -> [head_delta, ...]
    for r in head_data:
        comp, layer = parse_layer_idx(r["module"])
        if layer < 0: continue
        by[(r["variant"], comp, layer)].append(float(r["delta"]))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    for ax_idx, v in enumerate(VARIANTS):
        ax = axes[ax_idx]
        keys = sorted([(c, l) for vv, c, l in by if vv == v], key=lambda x: (x[0], x[1]))
        if not keys:
            ax.text(0.5, 0.5, "no data", ha="center", va="center"); ax.axis("off"); continue
        labels, max_d, sum_d = [], [], []
        for c, l in keys:
            heads = by[(v, c, l)]
            labels.append(f"{c}\nL{l}")
            max_d.append(max(heads))
            sum_d.append(sum(heads))
        x = np.arange(len(labels))
        ax.bar(x - 0.2, max_d, 0.4, label="max head Δ", color=COLORS[v], alpha=0.9, edgecolor="black")
        ax.bar(x + 0.2, sum_d, 0.4, label="sum heads Δ", color=COLORS[v], alpha=0.4, edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=45)
        ax.set_title(LABELS[v])
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0: ax.set_ylabel("Δ MSE")
        ax.legend(fontsize=7)
    fig.suptitle("Head ablation: max-head importance vs total head contribution per layer", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"diagram_7b_head_ablation.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] diagram_7b_head_ablation.png/.svg")


def fig_rep_swap():
    p = ABL_DIR / "rep_swap_v3_v4.json"
    if not p.exists(): return
    d = json.loads(p.read_text())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cats = ["V3 baseline", "V3 with V4's drug enc.", "V4 baseline", "V4 with V3's drug enc."]
    vals = [d["v3_baseline"], d["v3_with_v4_drug"], d["v4_baseline"], d["v4_with_v3_drug"]]
    cols = ["#009E73", "#9be8c8", "#CC79A7", "#e8b4cd"]
    bars = ax.bar(cats, vals, color=cols, edgecolor="black")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("MSE on held-out batch")
    ax.set_title(f"V3 ↔ V4 drug encoder swap\n"
                 f"V3 Δ = {d['delta_v3']:+.3f}   V4 Δ = {d['delta_v4']:+.3f}")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"diagram_7c_rep_swap.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] diagram_7c_rep_swap.png/.svg")


def write_findings(rows):
    layer_data = [r for r in rows if r["kind"] == "layer"]
    head_data  = [r for r in rows if r["kind"] == "head"]

    lines = ["# Phase E4 Findings — Causal ablations on Phase C (d=128) checkpoints", "",
             "Each row of the SUMMARY.csv is a single inference pass on the same fixed",
             "held-out batch with one circuit (a layer's attention output, or one head's",
             "contribution) zeroed. Δ MSE = ablated - baseline. Larger positive Δ ⇒",
             "the circuit is more important to the prediction.", "",
             "## Per-variant most-important layer", ""]
    by_var_layer = defaultdict(list)
    for r in layer_data:
        c, l = parse_layer_idx(r["module"])
        if l < 0: continue
        by_var_layer[r["variant"]].append((c, l, float(r["delta"])))
    for v in VARIANTS:
        items = sorted(by_var_layer.get(v, []), key=lambda x: -x[2])
        if not items: continue
        c, l, d = items[0]
        lines.append(f"- **{LABELS[v]}**: most-load-bearing layer = `{c}.layer{l}` (Δ MSE = {d:+.3f}). "
                     f"Top-3: {', '.join(f'{x[0]}.L{x[1]} ({x[2]:+.3f})' for x in items[:3])}")

    lines += ["", "## Per-variant most-important head", ""]
    by_var_head = defaultdict(list)
    for r in head_data:
        c, l = parse_layer_idx(r["module"])
        if l < 0: continue
        by_var_head[r["variant"]].append((c, l, int(r["head"]), float(r["delta"])))
    for v in VARIANTS:
        items = sorted(by_var_head.get(v, []), key=lambda x: -x[3])
        if not items: continue
        c, l, h, d = items[0]
        lines.append(f"- **{LABELS[v]}**: most-load-bearing head = `{c}.L{l} h={h}` (Δ MSE = {d:+.3f}). "
                     f"Top-3: {', '.join(f'{x[0]}.L{x[1]}/h{x[2]} ({x[3]:+.3f})' for x in items[:3])}")

    p = ABL_DIR / "rep_swap_v3_v4.json"
    if p.exists():
        d = json.loads(p.read_text())
        lines += ["", "## V3 ↔ V4 drug-encoder representation swap", "",
                  f"- V3 baseline MSE: {d['v3_baseline']:.4f}",
                  f"- V3 with V4's drug encoder: {d['v3_with_v4_drug']:.4f} (Δ = {d['delta_v3']:+.4f})",
                  f"- V4 baseline MSE: {d['v4_baseline']:.4f}",
                  f"- V4 with V3's drug encoder: {d['v4_with_v3_drug']:.4f} (Δ = {d['delta_v4']:+.4f})",
                  "",
                  ("**Drug encoders are interchangeable** between V3 and V4 — the swapped MSE is within ~5% of baseline."
                   if abs(d['delta_v3']) < 0.05 * d['v3_baseline'] and abs(d['delta_v4']) < 0.05 * d['v4_baseline']
                   else "**Drug encoders are co-adapted with their fusion module** — swapping yields >5% MSE degradation.")]

    (ROOT / "FINDINGS_E4.md").write_text("\n".join(lines) + "\n")
    print("[saved] FINDINGS_E4.md")


def main():
    rows = load()
    if rows is None: return
    fig_layer_ablation(rows)
    fig_head_ablation(rows)
    fig_rep_swap()
    write_findings(rows)


if __name__ == "__main__":
    main()
