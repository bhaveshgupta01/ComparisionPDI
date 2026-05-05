#!/usr/bin/env python3
"""
Phase E6 width-vs-depth decomposition.

Tests: does doubling depth at d=128 also reverse the ranking, or is the reversal
width-specific?

Compares random split, 3 seeds:
  Phase C:  d=128, n=6/3   → V2 wins (early X-attn)
  Phase E6: d=128, n=12/6  → ?                     (depth doubled)
  Phase E1: d=256, n=6/3   → V3 wins (late X-attn) (width doubled)

Output:
  poster_figures/diagram_9_width_vs_depth.png/.svg
  FINDINGS_E6.md  (summarizing whether the reversal is width- or capacity-driven)
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG  = ROOT / "poster_figures"

VARIANTS = ["early_concat", "early_crossattn", "late_crossattn", "late_concat"]
LABELS   = {"early_concat": "V1 EC", "early_crossattn": "V2 EX",
            "late_crossattn": "V3 LX", "late_concat":    "V4 LC"}
COLORS   = {"early_concat": "#0072B2", "early_crossattn": "#E69F00",
            "late_crossattn": "#009E73", "late_concat":    "#CC79A7"}

# Phase C (d=128, n=6/3) random-split numbers from FINDINGS.md
PHASE_C_RAND = {
    "early_concat":    (1.004, 0.031),
    "early_crossattn": (0.948, 0.023),
    "late_crossattn":  (1.030, 0.039),
    "late_concat":     (1.119, 0.018),
}


def agg(csv_path: Path, split_filter: str = "random"):
    by = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["split"] != split_filter: continue
            by[r["variant"]].append(float(r["best_val_mse"]))
    return {v: (st.mean(vs), st.stdev(vs) if len(vs) > 1 else 0.0)
            for v, vs in by.items()}


def main():
    e6 = agg(ROOT / "PHASE_E6_RESULTS.csv")
    e1 = agg(ROOT / "PHASE_E1_RESULTS.csv")

    print("=== Random-split val MSE (mean ± std), 3 seeds ===")
    print(f"{'Variant':18s} {'Phase C (d=128)':>20s} {'E6 (depth x2)':>20s} {'E1 (width x2)':>20s}")
    for v in VARIANTS:
        pc = PHASE_C_RAND[v]
        e6m, e6s = e6.get(v, (None, None))
        e1m, e1s = e1.get(v, (None, None))
        print(f"{LABELS[v]:18s}  {pc[0]:.3f} ± {pc[1]:.3f}     "
              f"{e6m:.3f} ± {e6s:.3f}     {e1m:.3f} ± {e1s:.3f}")

    # Determine winners
    pc_winner = min(VARIANTS, key=lambda v: PHASE_C_RAND[v][0])
    e6_winner = min(VARIANTS, key=lambda v: e6[v][0])
    e1_winner = min(VARIANTS, key=lambda v: e1[v][0])
    print(f"\nWinners: Phase C={LABELS[pc_winner]}, E6 (depth)={LABELS[e6_winner]}, E1 (width)={LABELS[e1_winner]}")

    # ---- Figure 9: width-vs-depth bar comparison ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(VARIANTS))
    w = 0.27
    pc_vals = [PHASE_C_RAND[v][0] for v in VARIANTS]
    pc_errs = [PHASE_C_RAND[v][1] for v in VARIANTS]
    e6_vals = [e6[v][0] for v in VARIANTS]
    e6_errs = [e6[v][1] for v in VARIANTS]
    e1_vals = [e1[v][0] for v in VARIANTS]
    e1_errs = [e1[v][1] for v in VARIANTS]

    b1 = ax.bar(x - w, pc_vals, w, yerr=pc_errs, capsize=4, label="Phase C  (d=128, n=6/3)",
                color="#bbbbbb", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x,     e6_vals, w, yerr=e6_errs, capsize=4, label="E6 depth × 2  (d=128, n=12/6)",
                color="#888888", edgecolor="black", linewidth=0.5)
    b3 = ax.bar(x + w, e1_vals, w, yerr=e1_errs, capsize=4, label="E1 width × 2  (d=256, n=6/3)",
                color="#444444", edgecolor="black", linewidth=0.5)
    # Color-code E1 bars by variant
    for i, bar in enumerate(b3):
        bar.set_color(COLORS[VARIANTS[i]])
        bar.set_edgecolor("black")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[v] for v in VARIANTS])
    ax.set_ylabel("Best Val MSE  ↓ better  (random split, 3-5 seeds)")
    ax.set_title("Width-vs-depth decomposition\n"
                 f"Winner: Phase C={LABELS[pc_winner]}, E6 (depth doubled)={LABELS[e6_winner]}, E1 (width doubled)={LABELS[e1_winner]}",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    # Annotate winners
    for cfg, vals, off in [("PC", pc_vals, -w), ("E6", e6_vals, 0), ("E1", e1_vals, w)]:
        wi = np.argmin(vals)
        ax.annotate("★", xy=(wi + off, vals[wi] + 0.04), ha="center", color="darkred", fontsize=14)

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"diagram_9_width_vs_depth.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] diagram_9_width_vs_depth.png/.svg")

    # ---- Findings summary ----
    lines = [
        "# Phase E6 — Width-vs-depth decomposition",
        "",
        "**Test:** keep model parameters roughly equal between (d=128, n_layers=12) [E6] and",
        "(d=256, n_layers=6) [E1]. Both add capacity vs Phase C. Does doubling *either* axis",
        "trigger the early-vs-late ranking reversal, or is the reversal width-specific?",
        "",
        "## Random-split val MSE (mean ± std, 3 seeds)",
        "",
        "| Variant | Phase C (d=128, n=6/3) | E6 depth × 2 (d=128, n=12/6) | E1 width × 2 (d=256, n=6/3) |",
        "|---|---|---|---|",
    ]
    for v in VARIANTS:
        pc = PHASE_C_RAND[v]
        e6m, e6s = e6[v]
        e1m, e1s = e1[v]
        lines.append(f"| {LABELS[v]} | {pc[0]:.3f} ± {pc[1]:.3f} | {e6m:.3f} ± {e6s:.3f} | {e1m:.3f} ± {e1s:.3f} |")
    lines += ["",
              "## Winners",
              f"- Phase C (d=128, n=6/3):  **{LABELS[pc_winner]}**",
              f"- E6 (d=128, n=12/6):       **{LABELS[e6_winner]}**",
              f"- E1 (d=256, n=6/3):        **{LABELS[e1_winner]}**",
              ""]

    # Δ analysis: how much each variant improved Phase C → E6 vs Phase C → E1
    lines += ["## Per-variant Δ from each axis", "",
              "| Variant | Δ from depth × 2 (E6−C) | Δ from width × 2 (E1−C) | Δ-ratio (depth/width) |",
              "|---|---|---|---|"]
    for v in VARIANTS:
        d_depth = e6[v][0] - PHASE_C_RAND[v][0]
        d_width = e1[v][0] - PHASE_C_RAND[v][0]
        ratio = d_depth / d_width if d_width != 0 else float("nan")
        lines.append(f"| {LABELS[v]} | {d_depth:+.3f} | {d_width:+.3f} | {ratio:.2f} |")
    lines.append("")

    # Interpretation
    if pc_winner == e6_winner and pc_winner != e1_winner:
        finding = (f"**Width-specific reversal.** The early-fusion winner ({LABELS[pc_winner]}) is preserved "
                   f"after depth-doubling at d=128 ({LABELS[e6_winner]} still wins) but flips after width-doubling "
                   f"at n=6/3 ({LABELS[e1_winner]} now wins). This isolates the reversal to the *width* axis: "
                   f"deeper-but-narrow models do not produce the late-fusion advantage observed at d=256.")
    elif pc_winner != e6_winner and pc_winner != e1_winner:
        finding = (f"**Capacity-driven reversal.** Both depth- and width-doubling trigger a winner change "
                   f"({LABELS[pc_winner]} → {LABELS[e6_winner]} via depth, → {LABELS[e1_winner]} via width). "
                   f"The reversal is driven by total capacity, not width specifically.")
    elif pc_winner == e6_winner == e1_winner:
        finding = "Neither axis reverses the ranking — this would refute the headline finding, which our other data does not support."
    else:
        finding = (f"**Mixed signal.** E6 winner = {LABELS[e6_winner]}, E1 winner = {LABELS[e1_winner]}. "
                   f"Width and depth interact non-trivially.")

    lines += ["## Interpretation", "", finding, ""]

    # Late-vs-early sensitivity
    avg_depth_late  = ((e6["late_crossattn"][0] - PHASE_C_RAND["late_crossattn"][0])
                       + (e6["late_concat"][0]    - PHASE_C_RAND["late_concat"][0])) / 2
    avg_depth_early = ((e6["early_concat"][0]    - PHASE_C_RAND["early_concat"][0])
                       + (e6["early_crossattn"][0] - PHASE_C_RAND["early_crossattn"][0])) / 2
    avg_width_late  = ((e1["late_crossattn"][0] - PHASE_C_RAND["late_crossattn"][0])
                       + (e1["late_concat"][0]    - PHASE_C_RAND["late_concat"][0])) / 2
    avg_width_early = ((e1["early_concat"][0]    - PHASE_C_RAND["early_concat"][0])
                       + (e1["early_crossattn"][0] - PHASE_C_RAND["early_crossattn"][0])) / 2

    lines += ["## Late vs early benefit, per axis", "",
              f"- Depth × 2 → avg Δ MSE: **early variants {avg_depth_early:+.3f}**, **late variants {avg_depth_late:+.3f}**",
              f"- Width × 2 → avg Δ MSE: **early variants {avg_width_early:+.3f}**, **late variants {avg_width_late:+.3f}**",
              ""]
    if avg_width_late < avg_depth_late and abs(avg_width_late - avg_width_early) > abs(avg_depth_late - avg_depth_early):
        lines.append("Late variants benefit *more* from width than from depth, and the late-vs-early "
                     "asymmetry is larger for width. **Width is the more impactful axis.**")
    elif avg_depth_late < avg_width_late:
        lines.append("Late variants benefit *more* from depth than width — unexpected.")
    else:
        lines.append("Both axes produce similar late-vs-early differential improvement.")

    (ROOT / "FINDINGS_E6.md").write_text("\n".join(lines) + "\n")
    print(f"[saved] FINDINGS_E6.md")


if __name__ == "__main__":
    main()
