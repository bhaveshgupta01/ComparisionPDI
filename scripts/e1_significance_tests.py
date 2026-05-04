#!/usr/bin/env python3
"""
Statistical significance for Phase E1 (and Phase C, where seeds match).

Computes paired-seed t-tests for every variant pair × every split, since seeds
{42, 123, 456, 789, 2024} are identical across variants in E1 (and {42, 123, 456}
in Phase C). With matched seeds we can use a paired t-test which is more powerful
than Welch's.

Outputs:
  - SIGNIFICANCE_E1.md          markdown tables for inclusion in the paper
  - poster_figures/diagram_significance_e1.png  heatmap of paired p-values
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
CSV  = ROOT / "PHASE_E1_RESULTS.csv"
OUT  = ROOT / "SIGNIFICANCE_E1.md"
FIG  = ROOT / "poster_figures"

VARIANTS = ["early_concat", "early_crossattn", "late_crossattn", "late_concat"]
LABELS   = {"early_concat": "V1 EC", "early_crossattn": "V2 EX",
            "late_crossattn": "V3 LX", "late_concat":    "V4 LC"}
SPLITS   = ["random", "cold_drug", "cold_target"]
SPLIT_LABELS = {"random": "Random", "cold_drug": "Cold-Drug", "cold_target": "Cold-Target"}

# Phase C raw seed-level data (3 seeds: 42, 123, 456) — extracted from
# the original Phase C runs (we have these on local Mac too if needed).
# For this script we focus on E1 since it has 5 seeds and the headline finding
# is reversed at scale. Phase C significance can be added the same way later.


def load_e1():
    by_vss = defaultdict(dict)  # (variant, split) -> {seed: mse}
    with open(CSV) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            v, s, sd = row["variant"], row["split"], int(row["seed"])
            by_vss[(v, s)][sd] = float(row["best_val_mse"])
    return by_vss


def paired_t(a: list, b: list):
    """Paired t-test; a and b are matched-seed arrays. Two-sided."""
    diffs = np.array(a) - np.array(b)
    n = len(diffs)
    mean = diffs.mean()
    sd = diffs.std(ddof=1)
    if sd == 0:
        return mean, 0.0, 1.0
    t = mean / (sd / math.sqrt(n))
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return mean, t, p


def main():
    e1 = load_e1()
    seeds = [42, 123, 456, 789, 2024]
    n_seeds = len(seeds)

    # --- For each split, build a 4x4 paired-p-value matrix ---
    pmats = {}
    diffmats = {}
    for split in SPLITS:
        P = np.ones((4, 4))
        D = np.zeros((4, 4))
        for i, vi in enumerate(VARIANTS):
            for j, vj in enumerate(VARIANTS):
                if i == j: continue
                ai = [e1[(vi, split)][s] for s in seeds]
                aj = [e1[(vj, split)][s] for s in seeds]
                mean_diff, t, p = paired_t(ai, aj)
                P[i, j] = p
                D[i, j] = mean_diff
        pmats[split] = P
        diffmats[split] = D

    # --- Markdown report ---
    lines = ["# Phase E1 — Statistical significance",
             "",
             f"Paired-seed t-test (n={n_seeds} seeds, identical across all variants).",
             "Each cell = p-value for H0: variant-row MSE = variant-col MSE; the",
             "**sign** of the mean difference (row − col) is shown alongside p so",
             "row-wins-vs-col is easy to read.",
             ""]

    for split in SPLITS:
        P = pmats[split]; D = diffmats[split]
        lines += [f"## {SPLIT_LABELS[split]}", "",
                  "| | " + " | ".join(LABELS[v] for v in VARIANTS) + " |",
                  "|---|" + "---|" * 4]
        for i, vi in enumerate(VARIANTS):
            row_cells = []
            for j, vj in enumerate(VARIANTS):
                if i == j:
                    row_cells.append("—")
                else:
                    sig = "**" if P[i, j] < 0.05 else ""
                    arrow = "↓" if D[i, j] < 0 else "↑"  # row better if ↓
                    row_cells.append(f"{sig}p={P[i, j]:.3f} ({arrow}{abs(D[i, j]):.3f}){sig}")
            lines.append(f"| {LABELS[vi]} | " + " | ".join(row_cells) + " |")
        lines.append("")
        lines.append("Interpretation: *row vs col*. ↓ means row's mean MSE is lower (better). **Bold** = p<0.05.")
        lines.append("")

    # --- Headline highlights ---
    lines += ["## Headline significance results", ""]
    pairs_to_check = [
        ("late_crossattn", "early_crossattn", "random",      "V3 vs V2 on random (the reversal)"),
        ("late_crossattn", "early_crossattn", "cold_target", "V3 vs V2 on cold-target"),
        ("late_concat",    "late_crossattn",  "cold_drug",   "V4 vs V3 on cold-drug (where V4 wins)"),
        ("late_concat",    "early_concat",    "random",      "V4 vs V1 on random (concat-only ranking)"),
        ("late_crossattn", "late_concat",     "random",      "V3 vs V4 on random (best two)"),
    ]
    for vi, vj, sp, label in pairs_to_check:
        ai = [e1[(vi, sp)][s] for s in seeds]
        aj = [e1[(vj, sp)][s] for s in seeds]
        mean_diff, t, p = paired_t(ai, aj)
        winner = LABELS[vi] if mean_diff < 0 else LABELS[vj]
        lines.append(f"- **{label}**: mean Δ = {mean_diff:+.4f}, t={t:+.3f}, p={p:.4f}. "
                     f"Winner: **{winner}** (p {'<' if p < 0.05 else '≥'} 0.05).")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"[saved] {OUT.name}")

    # --- Heatmap figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, split in enumerate(SPLITS):
        ax = axes[idx]
        P = pmats[split]
        # Use -log10(p) for display, capped at 3
        with np.errstate(divide="ignore"):
            display = -np.log10(np.where(P > 0, P, 1e-10))
            np.fill_diagonal(display, 0)
        display = np.clip(display, 0, 3)
        im = ax.imshow(display, cmap="Blues", vmin=0, vmax=3)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([LABELS[v] for v in VARIANTS], rotation=45)
        ax.set_yticklabels([LABELS[v] for v in VARIANTS])
        ax.set_title(SPLIT_LABELS[split])
        for i in range(4):
            for j in range(4):
                if i == j: continue
                txt = f"{P[i,j]:.3f}"
                color = "white" if display[i, j] > 1.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    fig.suptitle("Phase E1 paired-seed t-test p-values\n(darker = more significant; **bold** = p < 0.05)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("-log10(p), capped at 3")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"diagram_significance_e1.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] diagram_significance_e1.png/.svg")


if __name__ == "__main__":
    main()
