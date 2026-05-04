#!/usr/bin/env python3
"""
Phase E1b results have landed. Build the capacity-curve figure (Fig 4 of paper).

Inputs:
  PHASE_E1_RESULTS.csv     d=256, 5 seeds, 60 runs
  PHASE_E1B_RESULTS.csv    d=192, 3 seeds, 36 runs (after harvesting on HPC + scp)
  Phase C numbers          d=128 hardcoded from FINDINGS.md

Output:
  poster_figures/diagram_4_capacity_curve.png/.svg
  CAPACITY_FINDINGS.md  (text summary of the phase transition behavior)
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

# Phase C numbers from FINDINGS.md (mean ± std across 3 seeds)
PHASE_C = {
    ("early_concat",    "random"):      (1.004, 0.031),
    ("early_concat",    "cold_drug"):   (1.476, 0.029),
    ("early_concat",    "cold_target"): (1.360, 0.197),
    ("early_crossattn", "random"):      (0.948, 0.023),
    ("early_crossattn", "cold_drug"):   (1.432, 0.170),
    ("early_crossattn", "cold_target"): (1.248, 0.187),
    ("late_crossattn",  "random"):      (1.030, 0.039),
    ("late_crossattn",  "cold_drug"):   (1.410, 0.129),
    ("late_crossattn",  "cold_target"): (1.549, 0.131),
    ("late_concat",     "random"):      (1.119, 0.018),
    ("late_concat",     "cold_drug"):   (1.465, 0.178),
    ("late_concat",     "cold_target"): (1.467, 0.069),
}

VARIANTS = ["early_concat", "early_crossattn", "late_crossattn", "late_concat"]
LABELS   = {"early_concat": "V1 Early Concat", "early_crossattn": "V2 Early X-Attn",
            "late_crossattn": "V3 Late X-Attn", "late_concat":    "V4 Late Concat"}
SPLITS   = ["random", "cold_drug", "cold_target"]
SPLIT_LABELS = {"random": "Random", "cold_drug": "Cold-Drug", "cold_target": "Cold-Target"}
COLORS   = {"early_concat": "#0072B2", "early_crossattn": "#E69F00",
            "late_crossattn": "#009E73", "late_concat":    "#CC79A7"}
MARKERS  = {"early_concat": "o", "early_crossattn": "s",
            "late_crossattn": "^", "late_concat":    "D"}


def load_results_csv(path: Path):
    by = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            v, s = row["variant"], row["split"]
            by[(v, s)].append(float(row["best_val_mse"]))
    return {k: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) for k, v in by.items()}


def main():
    e1   = load_results_csv(ROOT / "PHASE_E1_RESULTS.csv")
    e1b_path = ROOT / "PHASE_E1B_RESULTS.csv"
    if not e1b_path.exists():
        print(f"[ERROR] {e1b_path} not found — pull from HPC first:")
        print("        scp bg2896@gw.hpc.nyu.edu:/scratch/bg2896/ComparisionPDI/PHASE_E1B_RESULTS.csv ./")
        return
    e1b = load_results_csv(e1b_path)

    # Build per (variant, split) curves indexed by d_model
    SCALES = [128, 192, 256]
    series = {(v, s): {} for v in VARIANTS for s in SPLITS}
    for k, val in PHASE_C.items(): series[k][128] = val
    for k, val in e1b.items():    series[k][192] = val
    for k, val in e1.items():     series[k][256] = val

    # --- Three-panel figure: one per split ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for sidx, split in enumerate(SPLITS):
        ax = axes[sidx]
        for v in VARIANTS:
            xs = [d for d in SCALES if d in series[(v, split)]]
            means = [series[(v, split)][d][0] for d in xs]
            stds  = [series[(v, split)][d][1] for d in xs]
            ax.plot(xs, means, "-", color=COLORS[v], linewidth=2,
                    marker=MARKERS[v], markersize=9, label=LABELS[v])
            ax.fill_between(xs, np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=COLORS[v], alpha=0.15)
        ax.set_xlabel("d_model")
        ax.set_xticks(SCALES)
        ax.set_title(SPLIT_LABELS[split])
        ax.grid(alpha=0.3)
        if sidx == 0:
            ax.set_ylabel("Best Val MSE  ↓ better")
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle("Capacity curve — three model widths, four variants\n"
                 "Late-fusion variants (green / pink) overtake early-fusion (blue / orange) as scale grows",
                 fontsize=12)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"diagram_4_capacity_curve.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {FIG}/diagram_4_capacity_curve.png/.svg")

    # --- Findings summary ----
    lines = ["# Capacity-curve findings (E1b bracket)", "",
             "| Variant | d=128 (Phase C) | d=192 (E1b) | d=256 (E1) | Δ Phase C → E1 |",
             "|---|---|---|---|---|"]
    for split in SPLITS:
        lines.append(f"\n## {SPLIT_LABELS[split]}\n")
        lines.append("| Variant | d=128 | d=192 | d=256 | Δ |")
        lines.append("|---|---|---|---|---|")
        for v in VARIANTS:
            cells = []
            for d in SCALES:
                if d in series[(v, split)]:
                    m, s = series[(v, split)][d]
                    cells.append(f"{m:.3f} ± {s:.3f}")
                else:
                    cells.append("—")
            d_total = (series[(v, split)][256][0] - series[(v, split)][128][0]
                       if 128 in series[(v, split)] and 256 in series[(v, split)] else None)
            d_str = f"{d_total:+.3f}" if d_total is not None else "—"
            lines.append(f"| {LABELS[v]} | {cells[0]} | {cells[1]} | {cells[2]} | {d_str} |")

    # Determine where the V2/V3 crossover happens on random split
    if all(d in series[("late_crossattn", "random")] and d in series[("early_crossattn", "random")]
           for d in SCALES):
        v3_curve = [series[("late_crossattn", "random")][d][0] for d in SCALES]
        v2_curve = [series[("early_crossattn", "random")][d][0] for d in SCALES]
        lines += ["", "## V3 vs V2 crossover (random split)", "",
                  f"- d=128: V3={v3_curve[0]:.3f}, V2={v2_curve[0]:.3f} → V2 wins by {v2_curve[0]-v3_curve[0]:+.3f}",
                  f"- d=192: V3={v3_curve[1]:.3f}, V2={v2_curve[1]:.3f} → "
                  + ("V2 wins" if v2_curve[1] < v3_curve[1] else "V3 wins")
                  + f" by {abs(v2_curve[1]-v3_curve[1]):.3f}",
                  f"- d=256: V3={v3_curve[2]:.3f}, V2={v2_curve[2]:.3f} → V3 wins by {v2_curve[2]-v3_curve[2]:+.3f}",
                  "",
                  "**The phase transition is " +
                  ("sharp (between d=192 and d=256)" if v2_curve[1] < v3_curve[1]
                   else "gradual (V3 already ahead at d=192)") + ".**",
                  ""]

    (ROOT / "CAPACITY_FINDINGS.md").write_text("\n".join(lines) + "\n")
    print(f"[saved] CAPACITY_FINDINGS.md")


if __name__ == "__main__":
    main()
