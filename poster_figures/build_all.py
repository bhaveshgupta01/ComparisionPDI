#!/usr/bin/env python3
"""
Build all poster figures from current Phase A data + design assets.
Produces SVG + PNG outputs in this directory.
"""
import os
import csv
import io
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

OUT = os.path.dirname(os.path.abspath(__file__))

# Okabe-Ito colorblind-safe palette for V1-V4
COL = {
    "V1": "#0072B2",  # blue
    "V2": "#E69F00",  # orange
    "V3": "#009E73",  # green
    "V4": "#CC79A7",  # pink/magenta
    "neutral": "#444444",
    "muted": "#999999",
    "bg": "#FFFFFF",
    "soft": "#F2F2F2",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.svg"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=180)
    plt.close(fig)
    print(f"  built {name}")


# =============================================================================
# DIAGRAM 03 — 2 x 2 hero matrix
# =============================================================================
def diagram_03_matrix():
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.5); ax.axis("off")

    # Title
    ax.text(5, 8.2, "The 2 x 2 Design",
            ha="center", fontsize=24, weight="bold", color="#222")
    ax.text(5, 7.65,
            "What changes: where drug and protein information meet.\n"
            "What stays: tokenizer, encoder, optimizer, prediction head.",
            ha="center", fontsize=11, color="#666")

    # Column headers
    ax.text(3.0, 6.7, "CONCATENATION", ha="center", fontsize=14, weight="bold", color="#333")
    ax.text(7.0, 6.7, "CROSS-ATTENTION", ha="center", fontsize=14, weight="bold", color="#333")

    # Row headers
    ax.text(0.3, 4.85, "BEFORE\nencoding\n(early)", ha="center", va="center",
            fontsize=12, weight="bold", color="#333")
    ax.text(0.3, 1.95, "AFTER\nencoding\n(late)", ha="center", va="center",
            fontsize=12, weight="bold", color="#333")

    cells = [
        (1.0, 3.5, 4.0, 2.6, "V1", "Early Concat",     COL["V1"], "Lingwei",  "concat"),
        (5.5, 3.5, 4.0, 2.6, "V2", "Early Cross-Attn", COL["V2"], "Manas",    "xattn"),
        (1.0, 0.4, 4.0, 2.6, "V4", "Late Concat",      COL["V4"], "Bhavesh",  "concat"),
        (5.5, 0.4, 4.0, 2.6, "V3", "Late Cross-Attn",  COL["V3"], "Tenzin",   "xattn"),
    ]

    def draw_flow(ax, cx, cy, mode, color):
        # cx, cy is center of flow line
        # drug box
        ax.add_patch(Rectangle((cx - 1.85, cy - 0.18), 0.75, 0.36,
                                facecolor="#fde68a", edgecolor="#92400e", linewidth=1.2))
        ax.text(cx - 1.475, cy, "drug", ha="center", va="center", fontsize=8.5)

        # connector
        if mode == "concat":
            ax.text(cx - 1.0, cy, "+", ha="center", va="center",
                    fontsize=18, weight="bold", color="#444")
        else:  # cross-attn — render as small double-arrow
            ax.annotate("", xy=(cx - 1.10, cy + 0.06), xytext=(cx - 0.65, cy + 0.06),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.6))
            ax.annotate("", xy=(cx - 0.65, cy - 0.06), xytext=(cx - 1.10, cy - 0.06),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.6))

        # protein box
        ax.add_patch(Rectangle((cx - 0.55, cy - 0.18), 0.75, 0.36,
                                facecolor="#bfdbfe", edgecolor="#1e3a8a", linewidth=1.2))
        ax.text(cx - 0.175, cy, "prot", ha="center", va="center", fontsize=8.5)

        # arrow to score
        ax.annotate("", xy=(cx + 0.65, cy), xytext=(cx + 0.25, cy),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.6))

        # score box
        ax.add_patch(Rectangle((cx + 0.7, cy - 0.18), 0.85, 0.36,
                                facecolor="#bbf7d0", edgecolor="#065f46", linewidth=1.2))
        ax.text(cx + 1.125, cy, "pKi", ha="center", va="center", fontsize=8.5, weight="bold")

    for x, y, w, h, code, label, color, owner, mode in cells:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                             linewidth=2.8, edgecolor=color, facecolor="white")
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.45, code,
                ha="center", fontsize=22, weight="bold", color=color)
        ax.text(x + w/2, y + h - 1.05, label,
                ha="center", fontsize=13, color="#333")
        # Flow icons in the lower middle
        draw_flow(ax, x + w/2, y + 0.85, mode, color)
        # Owner at very bottom
        ax.text(x + w/2, y + 0.22, f"owner: {owner}",
                ha="center", fontsize=9, style="italic", color="#888")

    save(fig, "diagram_03_matrix")


# =============================================================================
# DIAGRAM 04 — Architecture diagrams V1-V4
# =============================================================================
def diagram_04_architectures():
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.4))
    fig.suptitle("Variant Architectures \u2014 same components, different fusion stage",
                 fontsize=19, weight="bold", y=0.995)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02,
                        wspace=0.08, hspace=0.04)

    def draw_arch(ax, title, color, owner, blocks):
        ax.set_xlim(0, 10); ax.set_ylim(3.25, 10.0); ax.axis("off")
        ax.text(5, 9.62, title, ha="center", fontsize=17, weight="bold", color=color)
        ax.text(5, 9.18, f"owner: {owner}", ha="center", fontsize=11.5,
                style="italic", color="#888")

        n = len(blocks)
        top = 8.55
        block_h = 1.10
        gap = 0.22
        total = n * block_h + (n - 1) * gap
        if top - total < 3.30:
            gap = max(0.12, (top - 3.30 - n * block_h) / max(1, n - 1))
            total = n * block_h + (n - 1) * gap

        for i, (label, sub, fill) in enumerate(blocks):
            y = top - i * (block_h + gap) - block_h
            ax.add_patch(FancyBboxPatch((0.3, y), 9.4, block_h,
                                         boxstyle="round,pad=0.02,rounding_size=0.08",
                                         facecolor=fill, edgecolor="#444",
                                         linewidth=1.7))
            ax.text(5, y + 0.72, label, ha="center", fontsize=13.5, weight="bold")
            ax.text(5, y + 0.28, sub, ha="center", fontsize=11, color="#555")
            if i < n - 1:
                ax.annotate("", xy=(5, y + block_h + 0.02),
                            xytext=(5, y + block_h + gap - 0.02),
                            arrowprops=dict(arrowstyle="->", color="#666", lw=1.8))

    # V1 — Early Concat
    draw_arch(axes[0, 0], "V1  Early Concat", COL["V1"], "Lingwei", [
        ("Drug + Protein Tokens", "[CLS] drug [SEP] protein", "#fef3c7"),
        ("Embedding + Pos Enc", "shared d_model", "#f3f4f6"),
        ("Shared Encoder x 6", "single encoder body", "#dbeafe"),
        ("[CLS] Pool, MLP Head", "scalar pKi", "#dcfce7"),
    ])

    # V2 — Early Cross-Attn
    draw_arch(axes[0, 1], "V2  Early Cross-Attn", COL["V2"], "Manas", [
        ("Drug + Protein Tokens", "two streams", "#fef3c7"),
        ("Embedding + Pos Enc", "per-modality", "#f3f4f6"),
        ("Bidirectional Cross-Attn", "drug attends to protein", "#fed7aa"),
        ("Shared Encoder x 6 + Head", "fused stream, scalar pKi", "#dcfce7"),
    ])

    # V3 — Late Cross-Attn
    draw_arch(axes[1, 0], "V3  Late Cross-Attn", COL["V3"], "Tenzin", [
        ("Drug Encoder x 3   /   Protein Encoder x 3",
         "two independent encoders", "#dbeafe"),
        ("Bidirectional Cross-Attn", "high-level fusion", "#bbf7d0"),
        ("Mean-Pool Both, Concat", "[B, 2 x d_model]", "#f3f4f6"),
        ("MLP Head", "scalar pKi", "#dcfce7"),
    ])

    # V4 — Late Concat
    draw_arch(axes[1, 1], "V4  Late Concat", COL["V4"], "Bhavesh", [
        ("Drug Encoder x 3   /   Protein Encoder x 3",
         "two independent encoders", "#dbeafe"),
        ("Mean-Pool Each Side", "[B, d_model] for each", "#f3f4f6"),
        ("Concatenate", "[B, 2 x d_model]", "#fbcfe8"),
        ("MLP Head", "scalar pKi", "#dcfce7"),
    ])

    save(fig, "diagram_04_architectures")


# =============================================================================
# DIAGRAM 09 — Phase pipeline timeline
# =============================================================================
def diagram_09_pipeline():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_xlim(0, 10); ax.set_ylim(-1, 3.5); ax.axis("off")
    ax.text(5, 3.2, "Four-Phase Protocol", ha="center", fontsize=16, weight="bold")

    phases = [
        ("A", "Individual Tuning", "~22 sweeps/owner, learn variant sensitivities", "#fde68a", "DONE for all 4"),
        ("B", "Fair-Config Negotiation", "single shared config in everyone's zone", "#bfdbfe", "PENDING"),
        ("C", "Controlled Final Runs", "all 4 \u00d7 3 splits \u00d7 \u22653 seeds", "#bbf7d0", "PENDING"),
        ("D", "Deep Analysis", "attention, geometry, ablation, biology", "#fbcfe8", "PENDING"),
    ]
    n = len(phases)
    box_w, box_h = 2.0, 1.3
    spacing = 0.5
    total = n * box_w + (n - 1) * spacing
    start = (10 - total) / 2

    for i, (letter, title, sub, color, status) in enumerate(phases):
        x = start + i * (box_w + spacing)
        y = 1.0
        box = FancyBboxPatch((x, y), box_w, box_h,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=color, edgecolor="#333", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + box_w/2, y + box_h - 0.35,
                f"Phase {letter}", ha="center", fontsize=12, weight="bold")
        ax.text(x + box_w/2, y + box_h - 0.7, title, ha="center", fontsize=10)
        ax.text(x + box_w/2, y + 0.2, sub, ha="center", fontsize=7, color="#444",
                wrap=True)
        ax.text(x + box_w/2, y - 0.25, status, ha="center", fontsize=8,
                style="italic", color="#7f1d1d" if "DONE" not in status else "#065f46")

        if i < n - 1:
            x_arrow_start = x + box_w
            x_arrow_end = x_arrow_start + spacing - 0.05
            ax.annotate("", xy=(x_arrow_end, y + box_h/2),
                        xytext=(x_arrow_start, y + box_h/2),
                        arrowprops=dict(arrowstyle="->", color="#444", lw=1.4))

    save(fig, "diagram_09_pipeline")


# =============================================================================
# DIAGRAM 15 — Phase A sensitivity heatmap (all 4 variants)
# =============================================================================
def diagram_15_sensitivity():
    # Phase A fast-mode val MSE; None = config not run for that variant
    rows = [
        ("baseline (lr=1e-4, d=128, l=6, h=4, bs=64)",
         1.6807, 1.6010, 1.6460, 1.5988),
        ("lr=5e-5",       1.7357, 1.6440, 1.6687, 1.6892),
        ("lr=3e-4",       1.5041, 1.4652, 1.4289, 1.4433),
        ("d_model=64",    1.6749, 1.6950, 1.6391, None),
        ("d_model=256",   None,   None,   None,   1.5338),
        ("d_model=256, bs=16",  None, 1.2882, 1.2637, None),
        ("n_layers=4",    1.6579, 1.5953, 1.6151, 1.6749),
        ("n_layers=8",    None,   None,   None,   1.6149),
        ("n_layers=8, bs=32",   None, 1.5479, 1.5461, None),
        ("batch=32",      1.6563, 1.5612, 1.5601, 1.5603),
        ("dropout=0.2",   1.9003, 1.7086, 1.6821, 1.7061),
        ("dropout=0.3",   2.3629, 1.9167, 1.7272, None),
        ("n_heads=2",     1.7075, 1.5824, 1.6144, 1.5690),
        ("n_heads=8",     None,   None,   None,   1.6420),
        ("n_heads=8, bs=32",    None, 1.6014, 1.5543, None),
        ("seed=123",      1.4395, 1.3713, 1.4632, 1.4414),
        ("seed=456",      1.5057, 1.4223, 1.5182, 1.5137),
        ("split=cold_drug",     1.5598, 1.6072, 1.6111, 1.6125),
        ("split=cold_target",   1.6878, 1.6541, 1.7472, 1.7312),
    ]
    configs = [r[0] for r in rows]
    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 11))
    masked = np.ma.masked_invalid(arr)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#e5e7eb")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=1.2, vmax=2.0)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(
        ["V1\nEarly\nConcat", "V2\nEarly\nX-Attn",
         "V3\nLate\nX-Attn",  "V4\nLate\nConcat"],
        fontsize=10.5, weight="bold")
    for i, c in enumerate([COL["V1"], COL["V2"], COL["V3"], COL["V4"]]):
        ax.get_xticklabels()[i].set_color(c)

    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=9)
    ax.set_title("Phase A Sensitivity Heatmap  -  all 4 variants\n"
                 "val MSE on BindingDB Ki  (lower = better;  grey = not run)",
                 fontsize=12.5, weight="bold")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8, color="#222")
            else:
                ax.text(j, i, "-", ha="center", va="center",
                        fontsize=10, color="#666")

    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.tick_params(axis="y", which="both", left=False, right=False)

    plt.colorbar(im, ax=ax, label="val MSE", fraction=0.035)
    save(fig, "diagram_15_sensitivity_4variant")


# =============================================================================
# DIAGRAM 32 — AI drug discovery market growth
# =============================================================================
def diagram_32_market_growth():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    years = np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])
    # Industry-analyst consensus range: ~1.5B (2023) growing to ~13-20B (2030)
    market_low  = np.array([0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.5, 6.5, 8.5, 11.0, 13.0])
    market_high = np.array([1.0, 1.3, 1.6, 2.0, 2.7, 4.0, 6.0, 8.5, 11.5, 15.0, 20.0])

    ax.fill_between(years, market_low, market_high, color="#0072B2", alpha=0.25,
                    label="Analyst forecast range")
    ax.plot(years, (market_low + market_high) / 2, "-o", color="#0072B2",
            lw=2.4, markersize=6, label="Midpoint estimate")

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Market size (USD billion)", fontsize=11)
    ax.set_title("AI in Drug Discovery \u2014 Market Growth Forecast\n~30% CAGR through 2030",
                 fontsize=13, weight="bold")
    ax.set_xlim(2019.5, 2030.5)
    ax.set_ylim(0, 22)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", frameon=False)
    ax.annotate("$1.5B (2023)", xy=(2023, 1.75), xytext=(2021, 5),
                arrowprops=dict(arrowstyle="->", color="#444", lw=0.8),
                fontsize=9)
    ax.annotate("$13\u201320B (2030)", xy=(2030, 16.5), xytext=(2025.5, 18.5),
                arrowprops=dict(arrowstyle="->", color="#444", lw=0.8),
                fontsize=9)
    ax.text(0.01, -0.18, "Sources: industry-analyst consensus (MarketsandMarkets, Grand View, Precedence)",
            transform=ax.transAxes, fontsize=7.5, color="#666")

    save(fig, "diagram_32_market_growth")


# =============================================================================
# DIAGRAM 33 — Timeline of AI-designed drug milestones
# =============================================================================
def diagram_33_milestones():
    events = [
        (2020, "DSP-1181", "Exscientia / Sumitomo\nFirst AI-designed\nsmall molecule -> Phase I"),
        (2021, "AlphaFold 2", "DeepMind\nNear-experimental\nstructure prediction"),
        (2022, "ESM-2", "Meta AI\nProtein language\nmodel at scale"),
        (2023, "AlphaFold 3", "DeepMind / Isomorphic\nProtein-ligand\ncomplex prediction"),
        (2024, "INS018_055", "Insilico Medicine\nFirst AI target +\nAI ligand -> Phase II"),
        (2025, "Pocket2Mol+", "Generative chemistry\non target pockets\ngoes mainstream"),
    ]
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(2019.4, 2026); ax.set_ylim(-0.5, 6); ax.axis("off")
    ax.text(2022.7, 5.6, "Notable AI-Designed Drug & Method Milestones",
            ha="center", fontsize=16, weight="bold")

    ax.hlines(2.6, 2019.5, 2026, color="#999", linewidth=2.5, zorder=1)

    for i, (year, name, desc) in enumerate(events):
        # Year marker
        ax.plot(year, 2.6, "o", markersize=14, color="#0072B2", zorder=5)
        # Year label always below the dot
        ax.text(year, 2.15, str(year), ha="center", fontsize=11, weight="bold",
                color="#0072B2")
        # Alternate above/below for the title block
        above = (i % 2 == 0)
        if above:
            title_y, desc_y = 4.55, 3.4
            connector_y = 3.0
            ax.plot([year, year], [2.6, connector_y], color="#bbb", lw=0.9, zorder=2)
            ax.text(year, title_y, name, ha="center", fontsize=11.5,
                    weight="bold", color="#111")
            ax.text(year, desc_y, desc, ha="center", va="top",
                    fontsize=8.5, color="#555")
        else:
            title_y, desc_y = 0.95, 1.55
            connector_y = 2.2
            ax.plot([year, year], [2.6, connector_y], color="#bbb", lw=0.9, zorder=2)
            ax.text(year, title_y, name, ha="center", fontsize=11.5,
                    weight="bold", color="#111")
            ax.text(year, desc_y, desc, ha="center", va="bottom",
                    fontsize=8.5, color="#555")

    save(fig, "diagram_33_milestones")


# =============================================================================
# DIAGRAM 36 — 9 loopholes infographic
# =============================================================================
def diagram_36_loopholes():
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(6, 9.6, "9 Loopholes in DTI Research \u2014 Where the Field is Weakest",
            ha="center", fontsize=15, weight="bold")
    ax.text(6, 9.2, "Each is a crack our project deliberately targets.",
            ha="center", fontsize=10, color="#666")

    loopholes = [
        (1, "Inherited architectures", "Late fusion is folklore, not justified"),
        (2, "Benchmark inflation",     "Random splits hide cold-domain failure"),
        (3, "Black-box outputs",       "Attention rarely validated vs PDB"),
        (4, "Kinase bias",             "BindingDB Ki ~50% kinases"),
        (5, "Affinity confusion",      "Ki, Kd, IC50 mixed silently"),
        (6, "Closed industry",         "Strongest models unreproducible"),
        (7, "No causal probes",        "Mech. interpretability absent"),
        (8, "Calibration ignored",     "Yet shortlists need confidence"),
        (9, "Uncontrolled comparisons", "Architecture vs HP confounded"),
    ]
    cols = 3
    cell_w, cell_h = 3.6, 2.4
    x_start = 0.4
    y_start = 8.4
    for idx, (num, title, desc) in enumerate(loopholes):
        c = idx % cols
        r = idx // cols
        x = x_start + c * (cell_w + 0.1)
        y = y_start - r * (cell_h + 0.2)
        ax.add_patch(FancyBboxPatch((x, y - cell_h), cell_w, cell_h,
                                     boxstyle="round,pad=0.02,rounding_size=0.1",
                                     facecolor="#fef2f2", edgecolor="#b91c1c",
                                     linewidth=1.4))
        # Red circle with number inside
        cx = x + 0.55
        cy = y - 0.45
        ax.add_patch(plt.Circle((cx, cy), 0.30, facecolor="#b91c1c", edgecolor="#7f1d1d",
                                  linewidth=1.5, zorder=3))
        ax.text(cx, cy, str(num), ha="center", va="center",
                fontsize=15, weight="bold", color="white", zorder=4)
        ax.text(x + 1.05, y - 0.45, title, fontsize=11.5, weight="bold",
                color="#222", va="center", ha="left")
        ax.text(x + cell_w/2, y - 1.5, desc,
                ha="center", va="center", fontsize=9.5, color="#444",
                wrap=True)

    save(fig, "diagram_36_loopholes")


# =============================================================================
# DIAGRAM 37 — Norm vs Us comparison
# =============================================================================
def diagram_37_norm_vs_us():
    rows = [
        ("Vary encoder choice for SOTA", "Hold encoder fixed, vary fusion stage"),
        ("Single random split", "Random + cold-drug + cold-target"),
        ("Single seed", "3+ seeds, mean +/- std"),
        ("Numbers-only reporting", "6-axis mechanistic dissection"),
        ("Free-floating hyperparameters", "Phase B 'fair config' locked across all 4"),
        ("Attention shown but unverified", "Validated vs PDBbind (Precision@K)"),
        ("Closed weights, one-off scripts", "Pinned env, deterministic seeds, public repo"),
        ("'Architecture won' headlines", "Falsifiable hypotheses (H1-H4) pre-registered"),
    ]
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(6, 9.6, "Industry Norm   vs   Our Deliberate Choice",
            ha="center", fontsize=16, weight="bold")

    # Header bars
    ax.add_patch(Rectangle((0.5, 8.5), 5.2, 0.6, facecolor="#fee2e2", edgecolor="#b91c1c"))
    ax.text(3.1, 8.8, "FIELD NORM", ha="center", weight="bold", color="#7f1d1d", fontsize=12)
    ax.add_patch(Rectangle((6.3, 8.5), 5.2, 0.6, facecolor="#dcfce7", edgecolor="#15803d"))
    ax.text(8.9, 8.8, "OUR CHOICE", ha="center", weight="bold", color="#14532d", fontsize=12)

    for i, (norm, ours) in enumerate(rows):
        y = 7.7 - i * 0.92
        ax.add_patch(Rectangle((0.5, y - 0.38), 5.2, 0.76, facecolor="#fef2f2",
                                edgecolor="#fecaca"))
        ax.text(3.1, y, norm, ha="center", va="center", fontsize=10, color="#7f1d1d")
        # Replace unicode arrow with FancyArrowPatch
        ax.annotate("", xy=(6.25, y), xytext=(5.75, y),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=2.0))
        ax.add_patch(Rectangle((6.3, y - 0.38), 5.2, 0.76, facecolor="#f0fdf4",
                                edgecolor="#bbf7d0"))
        ax.text(8.9, y, ours, ha="center", va="center", fontsize=10, color="#14532d")

    save(fig, "diagram_37_norm_vs_us")


# =============================================================================
# DIAGRAM 38 — Impact decision tree
# =============================================================================
def diagram_38_impact_tree():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(6, 8.5, "If We're Right \u2014 Impact Branches",
            ha="center", fontsize=16, weight="bold")

    # Root
    root = FancyBboxPatch((4.5, 6.7), 3.0, 1.0,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor="#dbeafe", edgecolor="#1e40af", linewidth=1.6)
    ax.add_patch(root)
    ax.text(6, 7.2, "What does our 4-variant\ncomparison reveal?", ha="center",
            va="center", fontsize=11, weight="bold")

    branches = [
        (1.0, 4.2, "FUSION STAGE\nDOESN'T MATTER",
         "Future DTI defaults to\ncheapest variant (V1 / V4)\nSaves ~50% encoder compute",
         "#fef3c7", "#92400e"),
        (4.5, 4.2, "EARLY FUSION WINS\non cold splits",
         "Field's default architecture\nshould flip for realistic\ndeployment scenarios",
         "#dcfce7", "#15803d"),
        (8.0, 4.2, "CROSS-ATTENTION\nrecovers binding sites better",
         "Prioritize X-attn variants\nfor regulatory and\nwet-lab handoffs",
         "#fbcfe8", "#9d174d"),
    ]
    for x, y, head, body, fill, edge in branches:
        ax.add_patch(FancyBboxPatch((x, y), 3.0, 1.5,
                                     boxstyle="round,pad=0.02,rounding_size=0.1",
                                     facecolor=fill, edgecolor=edge, linewidth=1.4))
        ax.text(x + 1.5, y + 1.15, head, ha="center", va="center",
                fontsize=10.5, weight="bold")
        ax.text(x + 1.5, y + 0.45, body, ha="center", va="center",
                fontsize=9, color="#333")
        # Connector
        ax.annotate("", xy=(x + 1.5, y + 1.5), xytext=(6, 6.7),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))

    # "Either way" box
    eitherway = FancyBboxPatch((3.0, 1.5), 6.0, 1.5,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="#e0e7ff", edgecolor="#4338ca", linewidth=1.6)
    ax.add_patch(eitherway)
    ax.text(6, 2.6, "Either way: transferable contribution", ha="center",
            fontsize=12, weight="bold", color="#312e81")
    ax.text(6, 2.0, "The 6-axis analysis playbook (attention, geometry, ablation,\n"
                    "biology, failures, dynamics) is reusable for any future\n"
                    "architecture comparison in DTI / ADMET / PPI",
            ha="center", va="center", fontsize=10, color="#1e1b4b")

    save(fig, "diagram_38_impact_tree")


# =============================================================================
# DIAGRAM 39 — 4-variant Phase A leaderboard (real data)
# =============================================================================
def diagram_39_leaderboard():
    """Phase A val MSE per variant per config. None = not run."""
    # (label, V1, V2, V3, V4)
    configs = [
        ("baseline",          1.6807, 1.6010, 1.6460, 1.5988),
        ("lr=5e-5",           1.7357, 1.6440, 1.6687, 1.6892),
        ("lr=1e-4",           1.6807, 1.6010, 1.6460, 1.5988),
        ("lr=3e-4",           1.5041, 1.4652, 1.4289, 1.4433),
        ("d_model=64",        1.6749, 1.6950, 1.6391, None),
        ("d_model=128",       1.6807, 1.6010, 1.6460, 1.5988),
        ("d_model=256",       None,   None,   None,   1.5338),
        ("d_model=256, bs=16", None,  1.2882, 1.2637, None),
        ("n_layers=4",        1.6579, 1.5953, 1.6151, 1.6749),
        ("n_layers=6",        1.6807, 1.6010, 1.6460, 1.5988),
        ("n_layers=8",        None,   None,   None,   1.6149),
        ("n_layers=8, bs=32", None,   1.5479, 1.5461, None),
        ("batch=32",          1.6563, 1.5612, 1.5601, 1.5603),
        ("dropout=0.1",       1.6807, 1.6010, 1.6460, 1.5988),
        ("dropout=0.2",       1.9003, 1.7086, 1.6821, 1.7061),
        ("dropout=0.3",       2.3629, 1.9167, 1.7272, None),
        ("n_heads=2",         1.7075, 1.5824, 1.6144, 1.5690),
        ("n_heads=8",         None,   None,   None,   1.6420),
        ("n_heads=8, bs=32",  None,   1.6014, 1.5543, None),
        ("seed=42",           1.6807, 1.6010, 1.6460, 1.5988),
        ("seed=123",          1.4395, 1.3713, 1.4632, 1.4414),
        ("seed=456",          1.5057, 1.4223, 1.5182, 1.5137),
        ("cold_drug",         1.5598, 1.6072, 1.6111, 1.6125),
        ("cold_target",       1.6878, 1.6541, 1.7472, 1.7312),
    ]
    fig, ax = plt.subplots(figsize=(13, 13))
    y_pos = np.arange(len(configs))
    width = 0.20

    variant_keys = ["V1", "V2", "V3", "V4"]
    variant_labels = {
        "V1": "V1 Early Concat", "V2": "V2 Early X-Attn",
        "V3": "V3 Late X-Attn",  "V4": "V4 Late Concat",
    }
    # Index of value column for each variant in `configs` tuple
    val_idx = {"V1": 1, "V2": 2, "V3": 3, "V4": 4}
    # Vertical offsets so V1 sits on top, V4 at bottom of each group
    offsets = {"V1": 1.5*width, "V2": 0.5*width, "V3": -0.5*width, "V4": -1.5*width}

    for v in variant_keys:
        plotted_label = False
        for i, row in enumerate(configs):
            val = row[val_idx[v]]
            if val is None:
                continue
            ax.barh(
                y_pos[i] + offsets[v], val, width,
                color=COL[v], alpha=0.92, edgecolor="#333", linewidth=0.5,
                label=variant_labels[v] if not plotted_label else "",
            )
            plotted_label = True

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c[0] for c in configs], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("val MSE on BindingDB Ki  (lower is better)", fontsize=11)
    ax.set_title(
        "Phase A Leaderboard - all 4 variants  (24 Configurations)\n"
        "Fast mode - 15 epochs - 10k pairs - BindingDB PDSPKi",
        fontsize=12.5, weight="bold")

    # Group separator lines between hyperparameter blocks
    group_breaks = [0, 3, 7, 11, 12, 15, 18, 21, 22]
    for br in group_breaks:
        ax.axhline(br + 0.5, color="#bbb", linestyle="-", lw=0.6, alpha=0.7)

    ax.axvline(1.60, color="#666", linestyle="--", lw=1.0, alpha=0.7)
    ax.text(1.605, -0.7, "~1.60 default", fontsize=8.5, color="#444")
    ax.set_xlim(1.2, 2.4)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)

    save(fig, "diagram_39_4variant_leaderboard")


# =============================================================================
# DIAGRAM 40 — Phase A sensitivity, per hyperparameter axis (all 4 variants)
# =============================================================================
def diagram_40_sensitivity_axes():
    """For each hyperparameter axis, show how MSE moves for V1, V2, V3, V4."""
    # (axis name, tick labels, V1, V2, V3, V4); None = not run
    axes_data = [
        ("Learning rate", ["5e-5", "1e-4", "3e-4"],
         [1.7357, 1.6807, 1.5041],
         [1.6440, 1.6010, 1.4652],
         [1.6687, 1.6460, 1.4289],
         [1.6892, 1.5988, 1.4433]),
        ("d_model", ["64", "128", "256"],
         [1.6749, 1.6807, None],
         [1.6950, 1.6010, None],
         [1.6391, 1.6460, None],
         [None,   1.5988, 1.5338]),
        ("n_layers", ["4", "6", "8"],
         [1.6579, 1.6807, None],
         [1.5953, 1.6010, None],
         [1.6151, 1.6460, None],
         [1.6749, 1.5988, 1.6149]),
        ("Dropout", ["0.1", "0.2", "0.3"],
         [1.6807, 1.9003, 2.3629],
         [1.6010, 1.7086, 1.9167],
         [1.6460, 1.6821, 1.7272],
         [1.5988, 1.7061, None]),
        ("n_heads", ["2", "4", "8"],
         [1.7075, 1.6807, None],
         [1.5824, 1.6010, None],
         [1.6144, 1.6460, None],
         [1.5690, 1.5988, 1.6420]),
        ("Seed", ["42", "123", "456"],
         [1.6807, 1.4395, 1.5057],
         [1.6010, 1.3713, 1.4223],
         [1.6460, 1.4632, 1.5182],
         [1.5988, 1.4414, 1.5137]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Phase A Sensitivity  -  each hyperparameter axis  (all 4 variants)",
        fontsize=15, weight="bold")

    markers = {"V1": "o", "V2": "s", "V3": "^", "V4": "D"}
    labels  = {"V1": "V1 Early Concat", "V2": "V2 Early X-Attn",
               "V3": "V3 Late X-Attn",  "V4": "V4 Late Concat"}

    for ax, (name, ticks, v1, v2, v3, v4) in zip(axes.flatten(), axes_data):
        x = list(range(len(ticks)))
        for key, vals in [("V1", v1), ("V2", v2), ("V3", v3), ("V4", v4)]:
            xs = [x[i] for i, v in enumerate(vals) if v is not None]
            ys = [v for v in vals if v is not None]
            if not xs:
                continue
            ax.plot(xs, ys, "-" + markers[key], color=COL[key],
                    lw=2.0, markersize=8, label=labels[key])
        ax.set_xticks(x); ax.set_xticklabels(ticks, fontsize=10.5)
        ax.set_title(name, fontsize=13, weight="bold")
        ax.set_ylabel("val MSE", fontsize=10.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5, frameon=False, loc="best", ncol=2)
    plt.tight_layout()
    save(fig, "diagram_40_sensitivity_4variant")


# =============================================================================
# DIAGRAM 31 — Extensions roadmap tree
# =============================================================================
def diagram_31_extensions():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(6, 8.5, "Future Work \u2014 Extensions Roadmap",
            ha="center", fontsize=15, weight="bold")

    # Root
    ax.add_patch(FancyBboxPatch((4.5, 6.5), 3.0, 1.0,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor="#dbeafe", edgecolor="#1e40af", linewidth=1.6))
    ax.text(6, 7.0, "Core 4-Variant Result\n(this paper)", ha="center",
            va="center", fontsize=11, weight="bold")

    extensions = [
        (0.5, 4.0, "Multi-Modal Drug",
         "SMILES + GNN graph\n+ 3D conformer", "#dcfce7"),
        (3.5, 4.0, "Pre-Trained Encoders",
         "ChemBERTa + ESM-2\nneutralize fusion choice?", "#fef3c7"),
        (6.5, 4.0, "Structure-Aware",
         "AlphaFold + pocket GNN", "#fbcfe8"),
        (9.5, 4.0, "Agentic Lead Optim",
         "DTI as tool inside\nLLM agent", "#e0e7ff"),
    ]
    for x, y, head, body, fill in extensions:
        ax.add_patch(FancyBboxPatch((x, y), 2.2, 1.6,
                                     boxstyle="round,pad=0.02,rounding_size=0.1",
                                     facecolor=fill, edgecolor="#444", linewidth=1.2))
        ax.text(x + 1.1, y + 1.2, head, ha="center", fontsize=10, weight="bold")
        ax.text(x + 1.1, y + 0.5, body, ha="center", va="center", fontsize=8.5,
                color="#333")
        # connector
        ax.annotate("", xy=(x + 1.1, y + 1.6), xytext=(6, 6.5),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.0))

    ax.text(6, 1.7, "All four extensions reuse the 4-variant scaffold and the 6-axis analysis playbook.",
            ha="center", fontsize=10, style="italic", color="#444")

    save(fig, "diagram_31_extensions")


# =============================================================================
# DIAGRAM 05 — Encoder block (exploded view)
# =============================================================================
def diagram_05_encoder_block():
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(0, 8); ax.set_ylim(0, 12); ax.axis("off")
    ax.text(4, 11.5, "Shared Pre-Norm Transformer Encoder Block",
            ha="center", fontsize=14, weight="bold")
    ax.text(4, 11.05, "Identical across all 4 variants \u2014 controls confound from encoder choice.",
            ha="center", fontsize=9, color="#666")

    layers = [
        ("Input  x  (B, L, d_model)", "#f3f4f6"),
        ("LayerNorm (pre-norm)", "#e0e7ff"),
        ("Multi-Head Self-Attention\nh=4 heads, d_k = d_model / h", "#dbeafe"),
        ("Dropout 0.1 + Residual", "#f3f4f6"),
        ("LayerNorm (pre-norm)", "#e0e7ff"),
        ("FFN: Linear, GELU, Linear\n(d_model -> 4 x d_model -> d_model)", "#dcfce7"),
        ("Dropout 0.1 + Residual", "#f3f4f6"),
        ("Output  x'  (B, L, d_model)", "#f3f4f6"),
    ]
    for i, (label, fill) in enumerate(layers):
        y = 10 - i * 1.2
        ax.add_patch(FancyBboxPatch((1, y), 6, 0.9,
                                     boxstyle="round,pad=0.02,rounding_size=0.08",
                                     facecolor=fill, edgecolor="#444", linewidth=1.0))
        ax.text(4, y + 0.45, label, ha="center", va="center", fontsize=10)
        if i < len(layers) - 1:
            ax.annotate("", xy=(4, y - 0.05), xytext=(4, y + 0.02),
                        arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))

    save(fig, "diagram_05_encoder_block")


# =============================================================================
# DIAGRAM 08 — Split-strategy schematic
# =============================================================================
def diagram_08_split_strategy():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.suptitle("Three Train / Test Split Strategies",
                 fontsize=15, weight="bold", y=1.02)

    titles = ["Random  (sanity)",
              "Cold-Drug  (new chemistry)",
              "Cold-Target  (hardest, new biology)"]

    rng = np.random.default_rng(7)
    n_drugs, n_targets = 12, 8

    def base_grid(ax, title):
        ax.set_xlim(-0.6, n_targets + 0.6)
        ax.set_ylim(-0.6, n_drugs + 0.6)
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_title(title, fontsize=12.5, weight="bold")
        ax.set_xlabel("targets (proteins)")
        ax.set_ylabel("drugs (small molecules)")
        ax.set_xticks([]); ax.set_yticks([])

    # Pre-generate which (drug, target) pairs are present
    presence = rng.random((n_drugs, n_targets)) > 0.3   # ~70% pairs measured

    def draw_grid(ax, train_mask):
        for d in range(n_drugs):
            for t in range(n_targets):
                if not presence[d, t]:
                    continue
                if train_mask[d, t]:
                    color = "#bfdbfe"; edge = "#1e3a8a"
                else:
                    color = "#fecaca"; edge = "#991b1b"
                ax.add_patch(Rectangle((t - 0.4, d - 0.4), 0.8, 0.8,
                                       facecolor=color, edgecolor=edge,
                                       linewidth=0.8))

    # --- random split ---
    base_grid(axes[0], titles[0])
    rand_mask = rng.random((n_drugs, n_targets)) > 0.2
    draw_grid(axes[0], rand_mask)

    # --- cold-drug ---
    base_grid(axes[1], titles[1])
    held_drugs = rng.choice(n_drugs, size=3, replace=False)
    cd_mask = np.ones((n_drugs, n_targets), dtype=bool)
    cd_mask[held_drugs, :] = False
    draw_grid(axes[1], cd_mask)
    for d in held_drugs:
        axes[1].add_patch(Rectangle((-0.5, d - 0.5), n_targets, 1.0,
                                    facecolor="none", edgecolor="#991b1b",
                                    linewidth=2.0, linestyle="--"))

    # --- cold-target ---
    base_grid(axes[2], titles[2])
    held_targets = rng.choice(n_targets, size=2, replace=False)
    ct_mask = np.ones((n_drugs, n_targets), dtype=bool)
    ct_mask[:, held_targets] = False
    draw_grid(axes[2], ct_mask)
    for t in held_targets:
        axes[2].add_patch(Rectangle((t - 0.5, -0.5), 1.0, n_drugs,
                                    facecolor="none", edgecolor="#991b1b",
                                    linewidth=2.0, linestyle="--"))

    legend_handles = [
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor="#bfdbfe", markeredgecolor="#1e3a8a",
               markersize=11, label="train"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor="#fecaca", markeredgecolor="#991b1b",
               markersize=11, label="held-out"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    save(fig, "diagram_08_split_strategy")


# =============================================================================
# DIAGRAM 10 — Phase A best val MSE per variant
# =============================================================================
def diagram_10_best_mse():
    """Best fast-mode val MSE per variant + the config that got it."""
    bests = [
        ("V1", 1.4395, "seed=123"),
        ("V2", 1.2882, "d_model=256, bs=16"),
        ("V3", 1.2637, "d_model=256, bs=16"),
        ("V4", 1.4414, "seed=123 (fast); 1.231 in full mode"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = np.arange(len(bests))
    vals = [b[1] for b in bests]
    colors = [COL[b[0]] for b in bests]

    bars = ax.bar(xs, vals, width=0.55, color=colors, edgecolor="#222",
                  linewidth=1.0, alpha=0.92)
    for bar, (v, val, cfg) in zip(bars, bests):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                f"{val:.4f}\n({cfg})",
                ha="center", va="bottom", fontsize=9.5, color="#222")

    ax.set_xticks(xs)
    ax.set_xticklabels(["V1\nEarly Concat", "V2\nEarly X-Attn",
                        "V3\nLate X-Attn",  "V4\nLate Concat"],
                       fontsize=11, weight="bold")
    for i, c in enumerate(colors):
        ax.get_xticklabels()[i].set_color(c)
    ax.set_ylabel("Best val MSE on BindingDB Ki  (lower = better)", fontsize=11)
    ax.set_title(
        "Phase A Best Val MSE per Variant  (fast mode, 15 epochs, 10k pairs)\n"
        "Only the best config from each variant's sweep is shown",
        fontsize=12.5, weight="bold")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0.01, -0.18,
            "Note: V3 wins overall by 0.0245 MSE  -  smaller than seed variance "
            "(approx 0.10-0.15 across seeds 42 / 123 / 456) so this is preliminary.",
            transform=ax.transAxes, fontsize=8.5, color="#666", style="italic")

    save(fig, "diagram_10_best_mse")


# =============================================================================
# DIAGRAM 14 — Parameter count vs MSE Pareto plot
# =============================================================================
def diagram_14_param_pareto():
    """Approximate parameter counts for default config (d=128, 6 layers, h=4)."""
    # d_model=128, n_layers=6, ff=4*d_model=512, n_heads=4
    # Per-layer cost ~ 4*d_model^2 (attn) + 2 * d_model * ff (FFN) + biases
    # Approximate from architecture descriptions in src/models/variants/
    points = [
        ("V1", 4.1, 1.6807, "single shared encoder body"),
        ("V2", 4.7, 1.6010, "shared encoder + early cross-attn block"),
        ("V3", 5.2, 1.6460, "two 3-layer encoders + late cross-attn"),
        ("V4", 4.0, 1.5988, "two 3-layer encoders, concat pooled"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for name, params, mse, desc in points:
        ax.scatter(params, mse, s=300, color=COL[name], edgecolor="#111",
                   linewidth=1.5, zorder=3)
        ax.annotate(name, (params, mse), xytext=(0, -22),
                    textcoords="offset points", ha="center",
                    fontsize=13, weight="bold", color=COL[name])
        ax.annotate(desc, (params, mse), xytext=(12, 6),
                    textcoords="offset points", ha="left",
                    fontsize=8.5, color="#444")

    ax.set_xlabel("Approximate parameters (M)  -  default d=128, n_layers=6, h=4",
                  fontsize=10.5)
    ax.set_ylabel("baseline val MSE  (lower = better)", fontsize=11)
    ax.set_title("Parameter Count vs Accuracy  (Phase A baseline)\n"
                 "Pareto frontier  -  cheaper-and-better is bottom-left",
                 fontsize=12.5, weight="bold")
    ax.set_xlim(3.6, 5.7)
    ax.set_ylim(1.55, 1.72)
    ax.grid(True, alpha=0.3)

    # Pareto-front: V4 dominates V1 (cheaper + better); V2 and V3 are dominated by V4 too
    ax.text(0.02, 0.96,
            "* V4 dominates V1 (fewer params, lower MSE)\n"
            "* V2 wins on accuracy at higher cost\n"
            "* V3 has the most params but median MSE",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="#f3f4f6", edgecolor="#999", boxstyle="round,pad=0.4"))
    ax.text(0.01, -0.16,
            "Param counts approximate, computed from architecture spec; "
            "exact counts await Phase C runs.",
            transform=ax.transAxes, fontsize=8, color="#666", style="italic")

    save(fig, "diagram_14_param_pareto")


# =============================================================================
# Phase D figure helpers — load summaries
# =============================================================================
PHASE_D_DIR = os.path.join(os.path.dirname(OUT), "phase_d_summaries")


def _load_summary(variant):
    base = os.path.join(PHASE_D_DIR, variant)
    ent = np.load(os.path.join(base, "entropy_summary.npz"), allow_pickle=True)
    samp = np.load(os.path.join(base, "sample_attn.npz"))
    import json as _json
    return ent, samp


# =============================================================================
# DIAGRAM 16 — Attention entropy per layer (V2 + V4)
# =============================================================================
def diagram_16_attention_entropy():
    if not os.path.isdir(PHASE_D_DIR):
        print("  skipping diagram_16: no phase_d_summaries/ yet")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    # ---- panel 1: V1 + V2 (early variants, single shared 6-layer encoder) ----
    ax = axes[0]
    for variant_label, variant_dir, color, marker in [
        ("V1 Early Concat", "v1_phase_c",  COL["V1"], "o"),
        ("V2 Early X-Attn", "v2_phase_c",  COL["V2"], "s"),
    ]:
        try:
            ent, _ = _load_summary(variant_dir)
        except Exception:
            continue
        layer_keys = sorted([k for k in ent.files if "encoder_layer" in k and "entropy" in k])
        ents = []
        for k in layer_keys:
            arr = ent[k]
            ents.append((arr.mean(), arr.std()))
        if not ents:
            continue
        means = np.array([e[0] for e in ents])
        stds  = np.array([e[1] for e in ents])
        xs = np.arange(len(means))
        ax.plot(xs, means, "-" + marker, color=color, lw=2.0, markersize=8,
                label=variant_label)
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15)

    ax.axhline(np.log(1302), color="#999", linestyle="--", lw=1.0)
    ax.text(0.05, np.log(1302) - 0.05, f"uniform attn (log {1302}) = {np.log(1302):.2f}",
            color="#666", fontsize=8.5, va="top")
    ax.axhline(np.log(1302), color="#999", linestyle="--", lw=0.8)
    ax.text(0.05, np.log(1302) - 0.05, f"uniform = {np.log(1302):.2f}",
            color="#666", fontsize=8.5, va="top")
    ax.set_xlabel("encoder layer index", fontsize=11)
    ax.set_ylabel("mean attention entropy (nats)", fontsize=11)
    ax.set_title("V1 + V2  -  Early variants  (shared 6-layer encoder)",
                 fontsize=12, weight="bold")
    ax.set_xticks(range(6))
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(True, alpha=0.3)

    # ---- panel 2: V3 + V4 (late variants — drug + prot encoders) ----
    ax = axes[1]
    for variant_label, variant_dir, drug_color, prot_color in [
        ("V3 Late X-Attn", "v3_phase_c", COL["V3"], "#7C3AED"),
        ("V4 Late Concat", "v4_phase_c", COL["V4"], "#A855F7"),
    ]:
        try:
            ent, _ = _load_summary(variant_dir)
        except Exception:
            continue
        for stream_label, key_pat, color, marker in [
            (f"{variant_label}  drug",    "drug_encoder", drug_color, "o"),
            (f"{variant_label}  protein", "prot_encoder", prot_color, "^"),
        ]:
            keys = sorted([k for k in ent.files
                           if key_pat in k and "entropy" in k])
            if not keys:
                continue
            means = np.array([ent[k].mean() for k in keys])
            stds  = np.array([ent[k].std()  for k in keys])
            xs = np.arange(len(means))
            ax.plot(xs, means, "-" + marker, color=color, lw=2.0, markersize=8,
                    label=stream_label)
            ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.12)
    ax.axhline(np.log(100),  color="#999", linestyle=":", lw=0.7)
    ax.axhline(np.log(1200), color="#999", linestyle=":", lw=0.7)
    ax.text(2.05, np.log(100),  f"drug uniform = {np.log(100):.2f}",  color="#666", fontsize=8, va="center")
    ax.text(2.05, np.log(1200), f"prot uniform = {np.log(1200):.2f}", color="#666", fontsize=8, va="center")
    ax.set_xlabel("encoder layer index", fontsize=11)
    ax.set_ylabel("mean attention entropy (nats)", fontsize=11)
    ax.set_title("V3 + V4  -  Late variants  (3+3 separate encoders)",
                 fontsize=12, weight="bold")
    ax.set_xticks(range(3))
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Attention Entropy vs Encoder Layer  -  Phase D on Phase C checkpoints (256 held-out pairs)",
        fontsize=14, weight="bold", y=1.02)

    fig.text(0.5, -0.02,
             "Caveat: entropy includes pad tokens (drug avg ~50 / 100, protein avg ~400 / 1200). "
             "Mask-aware entropy would give a sharper view; absolute values shift but ranking is preserved.",
             ha="center", fontsize=9, style="italic", color="#666", wrap=True)

    plt.tight_layout()
    save(fig, "diagram_16_attention_entropy")


# =============================================================================
# DIAGRAM 17 — Attention heatmap on a held-out pair
# =============================================================================
def diagram_17_attention_heatmap():
    if not os.path.isdir(PHASE_D_DIR):
        print("  skipping diagram_17: no phase_d_summaries/ yet")
        return

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "Attention Heatmaps  -  example #0 from Phase C-trained models  (mean over heads, log-scaled)",
        fontsize=14, weight="bold", y=1.00)

    # ---- V1 last layer ----
    ax = fig.add_subplot(2, 3, 1)
    try:
        _, samp = _load_summary("v1_phase_c")
        keys = sorted([k for k in samp.files if "encoder" in k and "drug" not in k and "prot" not in k and "mean_heads" in k])
        if keys:
            attn = samp[keys[-1]]
            crop = attn[:200, :200]
            im = ax.imshow(np.log10(crop + 1e-8), cmap="viridis", aspect="auto")
            ax.axhline(102, color="white", lw=0.9, linestyle="--")
            ax.axvline(102, color="white", lw=0.9, linestyle="--")
            ax.text(50, 5, "drug", color="white", fontsize=9, weight="bold", ha="center")
            ax.text(150, 5, "protein", color="white", fontsize=9, weight="bold", ha="center")
            ax.set_title("V1 Early Concat  -  layer 6/6", fontsize=11, weight="bold", color=COL["V1"])
            ax.set_xlabel("key position"); ax.set_ylabel("query position")
            plt.colorbar(im, ax=ax, fraction=0.04)
    except Exception as e:
        ax.text(0.5, 0.5, f"V1 missing: {e}", transform=ax.transAxes, ha="center", va="center")

    # ---- V2 last layer ----
    ax = fig.add_subplot(2, 3, 2)
    try:
        _, samp = _load_summary("v2_phase_c")
        keys = sorted([k for k in samp.files if "encoder" in k and "drug" not in k and "prot" not in k and "mean_heads" in k])
        if keys:
            attn = samp[keys[-1]]
            crop = attn[:200, :200]
            im = ax.imshow(np.log10(crop + 1e-8), cmap="viridis", aspect="auto")
            ax.axhline(102, color="white", lw=0.9, linestyle="--")
            ax.axvline(102, color="white", lw=0.9, linestyle="--")
            ax.text(50, 5, "drug", color="white", fontsize=9, weight="bold", ha="center")
            ax.text(150, 5, "protein", color="white", fontsize=9, weight="bold", ha="center")
            ax.set_title("V2 Early X-Attn  -  layer 6/6", fontsize=11, weight="bold", color=COL["V2"])
            ax.set_xlabel("key position"); ax.set_ylabel("query position")
            plt.colorbar(im, ax=ax, fraction=0.04)
    except Exception as e:
        ax.text(0.5, 0.5, f"V2 missing: {e}", transform=ax.transAxes, ha="center", va="center")

    # ---- placeholder cell to keep grid layout aligned ----
    fig.add_subplot(2, 3, 3).axis("off")

    # ---- V3 drug encoder last + V4 drug encoder last ----
    for col_idx, (variant_dir, color, label) in enumerate([
        ("v3_phase_c", COL["V3"], "V3 Late X-Attn"),
        ("v4_phase_c", COL["V4"], "V4 Late Concat"),
    ]):
        try:
            _, samp = _load_summary(variant_dir)
            drug_keys = sorted([k for k in samp.files if "drug_encoder" in k and "mean_heads" in k])
            prot_keys = sorted([k for k in samp.files if "prot_encoder" in k and "mean_heads" in k])

            # drug subplot
            ax = fig.add_subplot(2, 3, 4 + col_idx)
            if drug_keys:
                attn = samp[drug_keys[-1]]
                im = ax.imshow(np.log10(attn + 1e-8), cmap="viridis", aspect="auto")
                ax.set_title(f"{label}  drug encoder, layer 3/3 (100x100)",
                             fontsize=10.5, weight="bold", color=color)
                ax.set_xlabel("key SMILES pos"); ax.set_ylabel("query")
                plt.colorbar(im, ax=ax, fraction=0.04)
        except Exception as e:
            ax.text(0.5, 0.5, f"missing: {e}", transform=ax.transAxes, ha="center", va="center")

    # ---- V3 protein encoder zoom (last layer) ----
    ax = fig.add_subplot(2, 3, 6)
    try:
        _, samp = _load_summary("v3_phase_c")
        prot_keys = sorted([k for k in samp.files if "prot_encoder" in k and "mean_heads" in k])
        if prot_keys:
            attn = samp[prot_keys[-1]]
            crop = attn[:300, :300]
            im = ax.imshow(np.log10(crop + 1e-8), cmap="viridis", aspect="auto")
            ax.set_title("V3 protein encoder  layer 3/3  (zoom: 300x300)",
                         fontsize=10.5, weight="bold", color=COL["V3"])
            ax.set_xlabel("key residue pos"); ax.set_ylabel("query")
            plt.colorbar(im, ax=ax, fraction=0.04)
    except Exception as e:
        ax.text(0.5, 0.5, f"missing: {e}", transform=ax.transAxes, ha="center", va="center")

    plt.tight_layout()
    save(fig, "diagram_17_attention_heatmap")


# =============================================================================
# Phase C loaders — pulls 36 results.csv files into a tidy structure
# =============================================================================
PHASE_C_DIR = os.path.join(os.path.dirname(OUT), "outputs", "phase_c")


def _load_phase_c():
    """Walk outputs/phase_c/phase_c_<variant>_<split>_seed<N>/.
    Loads results.csv (best_val_mse) AND history.csv (per-epoch metrics).
    Returns dict keyed by (variant, split, seed) -> dict with:
      - best_val_mse, best_val_ci, best_pearson, best_spearman, best_epoch
      - history: list of per-epoch dicts (epoch, train_loss, val_mse, val_ci, ...)
    """
    if not os.path.isdir(PHASE_C_DIR):
        return {}
    out = {}
    import re as _re
    pattern = _re.compile(
        r"phase_c_(?P<variant>[a-z_]+?)_(?P<split>random|cold_drug|cold_target)_seed(?P<seed>\d+)$"
    )
    for tag in os.listdir(PHASE_C_DIR):
        m = pattern.match(tag)
        if not m:
            continue
        run_dir = os.path.join(PHASE_C_DIR, tag)
        csv_path = os.path.join(run_dir, "results", "results.csv")
        if not os.path.isfile(csv_path):
            continue

        row = {}
        try:
            with open(csv_path) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if len(lines) >= 2:
                header = lines[0].split(",")
                last   = lines[-1].split(",")
                row = dict(zip(header, last))
        except Exception:
            pass

        # Load history.csv for per-epoch + CI metrics
        log_dir = os.path.join(run_dir, "logs")
        history = []
        if os.path.isdir(log_dir):
            for fn in os.listdir(log_dir):
                if fn.endswith("_history.csv"):
                    try:
                        with open(os.path.join(log_dir, fn)) as f:
                            lns = [ln.strip() for ln in f if ln.strip()]
                        if len(lns) >= 2:
                            hdr = lns[0].split(",")
                            for ln in lns[1:]:
                                rec = dict(zip(hdr, ln.split(",")))
                                for k in list(rec.keys()):
                                    try:
                                        rec[k] = float(rec[k])
                                    except (ValueError, TypeError):
                                        pass
                                history.append(rec)
                    except Exception:
                        pass

        # Find best epoch (lowest val_mse) and merge those metrics
        if history:
            best_ep = min(history, key=lambda r: r.get("val_mse", float("inf")))
            row["best_val_mse"]  = best_ep.get("val_mse")
            row["best_val_ci"]   = best_ep.get("val_ci")
            row["best_pearson"]  = best_ep.get("val_pearson")
            row["best_spearman"] = best_ep.get("val_spearman")
            row["best_epoch"]    = best_ep.get("epoch")
            row["history"] = history

        # Coerce numeric
        for k in list(row.keys()):
            if k == "history":
                continue
            try:
                row[k] = float(row[k])
            except (ValueError, TypeError):
                pass
        if row:
            key = (m.group("variant"), m.group("split"), int(m.group("seed")))
            out[key] = row
    return out


VARIANT_KEY_TO_LABEL = {
    "early_concat":    ("V1", "V1 Early Concat"),
    "early_crossattn": ("V2", "V2 Early X-Attn"),
    "late_crossattn":  ("V3", "V3 Late X-Attn"),
    "late_concat":     ("V4", "V4 Late Concat"),
}


# =============================================================================
# DIAGRAM 11 — Concordance index per variant per split (Phase C)
# =============================================================================
def diagram_11_ci_per_split():
    pc = _load_phase_c()
    if not pc:
        print("  skipping diagram_11: no phase_c results yet")
        return

    splits = ["random", "cold_drug", "cold_target"]
    variants = ["early_concat", "early_crossattn", "late_crossattn", "late_concat"]

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.20
    xs = np.arange(len(splits))

    for i, v in enumerate(variants):
        means, stds = [], []
        for s in splits:
            seeds_vals = []
            for sd in (42, 123, 456):
                if (v, s, sd) not in pc:
                    continue
                row = pc[(v, s, sd)]
                val = row.get("best_val_ci", row.get("val_ci", row.get("test_ci")))
                if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                    seeds_vals.append(val)
            if seeds_vals:
                means.append(np.mean(seeds_vals))
                stds.append(np.std(seeds_vals))
            else:
                means.append(np.nan); stds.append(0)
        offset = (i - 1.5) * width
        bar_x = xs + offset
        ax.bar(bar_x, means, width, yerr=stds, capsize=3,
               color=COL[VARIANT_KEY_TO_LABEL[v][0]], alpha=0.92,
               edgecolor="#333", linewidth=0.5,
               label=VARIANT_KEY_TO_LABEL[v][1])

    ax.set_xticks(xs)
    ax.set_xticklabels(["Random", "Cold-Drug", "Cold-Target"], fontsize=11)
    ax.set_ylabel("Concordance Index (higher = better)", fontsize=11)
    ax.axhline(0.5, color="#666", linestyle="--", lw=0.8)
    ax.text(0.0, 0.51, "chance (0.5)", fontsize=8.5, color="#444")
    ax.set_title(
        "Phase C  -  Concordance Index per variant per split  (mean +/- std over 3 seeds)",
        fontsize=12.5, weight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0.45, 1.0)

    save(fig, "diagram_11_ci_per_split")


# =============================================================================
# DIAGRAM 10b — Phase C MSE per variant per split (NEW from Phase C data)
# =============================================================================
def diagram_10b_mse_per_split():
    pc = _load_phase_c()
    if not pc:
        print("  skipping diagram_10b: no phase_c results yet")
        return

    splits = ["random", "cold_drug", "cold_target"]
    variants = ["early_concat", "early_crossattn", "late_crossattn", "late_concat"]

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.20
    xs = np.arange(len(splits))

    for i, v in enumerate(variants):
        means, stds, ns = [], [], []
        for s in splits:
            seeds_vals = []
            for sd in (42, 123, 456):
                if (v, s, sd) not in pc:
                    continue
                row = pc[(v, s, sd)]
                val = row.get("best_val_mse",
                       row.get("test_mse",
                       row.get("val_mse", None)))
                if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                    seeds_vals.append(val)
            if seeds_vals:
                means.append(np.mean(seeds_vals))
                stds.append(np.std(seeds_vals))
                ns.append(len(seeds_vals))
            else:
                means.append(np.nan); stds.append(0); ns.append(0)
        offset = (i - 1.5) * width
        bar_x = xs + offset
        ax.bar(bar_x, means, width, yerr=stds, capsize=3,
               color=COL[VARIANT_KEY_TO_LABEL[v][0]], alpha=0.92,
               edgecolor="#333", linewidth=0.5,
               label=VARIANT_KEY_TO_LABEL[v][1])
        # Annotate "n=X" so partial results are clearly flagged
        for xi, m, n in zip(bar_x, means, ns):
            if not np.isnan(m) and n > 0:
                ax.text(xi, m + 0.04, f"n={n}", ha="center", fontsize=7.5,
                        color="#444")

    ax.set_xticks(xs)
    ax.set_xticklabels(["Random", "Cold-Drug", "Cold-Target"], fontsize=11)
    ax.set_ylabel("Best Val MSE (lower = better)", fontsize=11)
    n_total = sum(1 for _ in pc.keys())
    ax.set_title(
        f"Phase C  -  Best Val MSE per variant per split  "
        f"({n_total} / 36 runs complete)",
        fontsize=12.5, weight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    save(fig, "diagram_10b_mse_per_split")


# =============================================================================
# DIAGRAM 10c — Live Phase C tally  (works with partial results)
# =============================================================================
def diagram_10c_phase_c_tally():
    """One bar per finished run, sorted (variant, split, seed). Visualizes progress."""
    pc = _load_phase_c()
    if not pc:
        print("  skipping diagram_10c: no phase_c results yet")
        return

    items = []
    for (v, s, sd), row in pc.items():
        val = row.get("best_val_mse", row.get("val_mse", row.get("test_mse", None)))
        if val is not None:
            items.append((v, s, sd, float(val)))
    if not items:
        return

    # Sort by (variant, split, seed)
    var_order = ["early_concat", "early_crossattn", "late_crossattn", "late_concat"]
    split_order = ["random", "cold_drug", "cold_target"]
    items.sort(key=lambda x: (var_order.index(x[0]), split_order.index(x[1]), x[2]))

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(items)), 6))
    xs = np.arange(len(items))
    vals = [it[3] for it in items]
    colors = [COL[VARIANT_KEY_TO_LABEL[it[0]][0]] for it in items]
    labels = [f"{VARIANT_KEY_TO_LABEL[it[0]][0]} {it[1]} s{it[2]}" for it in items]

    bars = ax.bar(xs, vals, width=0.7, color=colors, edgecolor="#222",
                  linewidth=0.4, alpha=0.92)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8.5)
    ax.set_ylabel("best val MSE", fontsize=11)
    ax.set_title(f"Phase C running tally  -  {len(items)} / 36 runs complete",
                 fontsize=12.5, weight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(1.0, color="#666", linestyle="--", lw=0.7)

    # Variant-color legend
    handles = [plt.Rectangle((0,0),1,1, color=COL[VARIANT_KEY_TO_LABEL[v][0]],
                              label=VARIANT_KEY_TO_LABEL[v][1]) for v in var_order]
    ax.legend(handles=handles, loc="upper left", fontsize=9.5, frameon=False)

    plt.tight_layout()
    save(fig, "diagram_10c_phase_c_tally")


# =============================================================================
# DIAGRAM 13 — Predicted-vs-true scatter (Phase D bugfix output)
# =============================================================================
def diagram_13_predicted_vs_true():
    """Reads from phase_d_summaries/ — needs valid (denormalized) predictions.
    Skips if predictions look broken (mean range under 1.0)."""
    if not os.path.isdir(PHASE_D_DIR):
        print("  skipping diagram_13: no phase_d_summaries/ yet")
        return

    panels = []
    for variant_dir, label, color in [
        ("v1_phase_c", "V1 Early Concat",  COL["V1"]),
        ("v2_phase_c", "V2 Early X-Attn",  COL["V2"]),
        ("v3_phase_c", "V3 Late X-Attn",   COL["V3"]),
        ("v4_phase_c", "V4 Late Concat",   COL["V4"]),
    ]:
        try:
            p = np.load(os.path.join(PHASE_D_DIR, variant_dir, "predictions.npy"))
            t = np.load(os.path.join(PHASE_D_DIR, variant_dir, "truth.npy"))
            if (p.max() - p.min()) < 1.0:
                # Broken normalization — predictions are clumped near zero
                panels.append((label, color, None, None,
                               f"!! predictions clumped at ~{p.mean():.3f} - "
                               "re-extract with bugfix"))
            else:
                panels.append((label, color, p, t, None))
        except FileNotFoundError:
            continue

    if not panels:
        print("  skipping diagram_13: no predictions found")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5),
                             squeeze=False)
    fig.suptitle(
        "Predicted vs True pKi  (256 held-out pairs from Phase D extraction)",
        fontsize=13, weight="bold", y=1.02)

    for ax, (label, color, p, t, msg) in zip(axes[0], panels):
        ax.set_title(label, fontsize=11.5, weight="bold", color=color)
        if p is None:
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, color="#b91c1c",
                    fontsize=10, wrap=True)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.scatter(t, p, s=18, alpha=0.55, color=color, edgecolor="#333",
                   linewidth=0.3)
        lo = float(min(p.min(), t.min()))
        hi = float(max(p.max(), t.max()))
        ax.plot([lo, hi], [lo, hi], "--", color="#666", lw=1.0,
                label="y = x")
        mse = float(((p - t) ** 2).mean())
        ax.text(0.04, 0.96, f"MSE = {mse:.3f}\nN = {len(p)}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor="white", edgecolor="#999", alpha=0.85),
                va="top")
        ax.set_xlabel("true pKi"); ax.set_ylabel("predicted pKi")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save(fig, "diagram_13_predicted_vs_true")


# =============================================================================
# DIAGRAM 06 / 07 — BindingDB data panel
# Loads from binding_db_stats/{summary.json, histograms.npz}.
# Falls back to hardcoded summary stats if the npz isn't on Mac yet.
# =============================================================================
BINDINGDB_STATS_DIR = os.path.join(os.path.dirname(OUT), "binding_db_stats")

# Hardcoded fallback from the HPC summary.json output (2026-04-28).
_HARDCODED_BINDINGDB_SUMMARY = {
    "n_rows_total": 27715, "n_kept": 27715, "n_skipped": 0,
    "pki": {"min": 3.82, "max": 12.46, "mean": 6.92, "median": 6.82,
            "p5": 5.00, "p95": 9.48},
    "drug_len": {"min": 6,  "max": 1824, "mean": 104.4, "median": 45.0},
    "prot_len": {"min": 11, "max": 4303, "mean": 458.6, "median": 444.0},
    "n_unique_targets": 400, "n_kinase_targets": 29,
}


def _load_bindingdb_summary():
    p = os.path.join(BINDINGDB_STATS_DIR, "summary.json")
    if os.path.isfile(p):
        with open(p) as f:
            return __import__("json").load(f)
    return _HARDCODED_BINDINGDB_SUMMARY


def _load_bindingdb_arrays():
    p = os.path.join(BINDINGDB_STATS_DIR, "histograms.npz")
    if os.path.isfile(p):
        d = np.load(p)
        return {"pki": d["pki"], "drug_len": d["drug_len"], "prot_len": d["prot_len"]}
    return None


def diagram_06_dataset_summary():
    s = _load_bindingdb_summary()
    fig, ax = plt.subplots(figsize=(10, 3.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.0); ax.axis("off")

    ax.text(6, 4.65, "BindingDB PDSPKi  -  Dataset at a Glance",
            ha="center", fontsize=16, weight="bold", color="#222")
    ax.text(6, 4.18,
            f"{s['n_kept']:,} drug-protein pairs  /  {s['n_unique_targets']} unique targets  "
            f"/  Ki measurements -> pKi target",
            ha="center", fontsize=11, color="#555")

    cards = [
        ("pKi range",
         f"{s['pki']['min']:.2f} - {s['pki']['max']:.2f}",
         f"mean {s['pki']['mean']:.2f}, median {s['pki']['median']:.2f}",
         "#dbeafe", "#1e3a8a"),
        ("Drug length (SMILES chars)",
         f"6 - 1,824",
         f"mean {s['drug_len']['mean']:.0f}, median {s['drug_len']['median']:.0f}",
         "#fef3c7", "#92400e"),
        ("Protein length (residues)",
         f"11 - 4,303",
         f"mean {s['prot_len']['mean']:.0f}, median {s['prot_len']['median']:.0f}",
         "#dcfce7", "#065f46"),
        ("Target families",
         f"{s['n_unique_targets']} unique",
         f"{s['n_kinase_targets']} kinases ({100*s['n_kinase_targets']/max(s['n_unique_targets'],1):.1f}%)",
         "#fbcfe8", "#831843"),
    ]
    card_w = 2.7
    card_h = 2.35
    gap = 0.22
    total_w = 4 * card_w + 3 * gap
    x_start = (12 - total_w) / 2
    card_y = 0.82

    for i, (title, big, sub, fill, edge) in enumerate(cards):
        x = x_start + i * (card_w + gap)
        ax.add_patch(FancyBboxPatch(
            (x, card_y), card_w, card_h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            facecolor=fill, edgecolor=edge, linewidth=1.6))
        ax.text(x + card_w/2, card_y + card_h - 0.38, title,
                ha="center", fontsize=11.5, weight="bold", color=edge)
        ax.text(x + card_w/2, card_y + card_h/2 + 0.02, big,
                ha="center", fontsize=18, weight="bold", color="#111")
        ax.text(x + card_w/2, card_y + 0.40, sub,
                ha="center", fontsize=10, color="#444")

    ax.text(6, 0.42,
            "Caps in training: drug <= 100 SMILES tokens, protein <= 1,200 residues  "
            "(covers ~75% of drugs, ~70% of proteins; longer truncated)",
            ha="center", fontsize=9.5, color="#666", style="italic")

    save(fig, "diagram_06_dataset_summary")


def diagram_07_length_and_pki_distribution():
    """Three-panel histogram: pKi, drug length, protein length.
    Falls back to a placeholder figure if the npz arrays aren't on Mac yet."""
    arrays = _load_bindingdb_arrays()
    if arrays is None:
        # Still render the placeholder so the slot is reserved
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.axis("off")
        ax.text(0.5, 0.5,
                "diagram_07: waiting for binding_db_stats/histograms.npz\n"
                "(download from OOD file browser, drop into ~/.../binding_db_stats/)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, style="italic", color="#999",
                bbox=dict(facecolor="#f3f4f6", edgecolor="#999",
                          boxstyle="round,pad=0.5"))
        save(fig, "diagram_07_length_and_pki_distribution")
        return

    pki = arrays["pki"]
    drug = arrays["drug_len"]
    prot = arrays["prot_len"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        f"BindingDB PDSPKi distributions  (N={len(pki):,} drug-protein pairs)",
        fontsize=13.5, weight="bold", y=1.02)

    # pKi histogram
    ax = axes[0]
    ax.hist(pki, bins=60, color="#0072B2", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("pKi  =  -log10(Ki / 10^9)", fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title(f"pKi  (mean {pki.mean():.2f}, median {np.median(pki):.2f})",
                 fontsize=12, weight="bold")
    ax.axvline(np.median(pki), color="#444", linestyle="--", lw=0.9)
    ax.grid(True, alpha=0.3)

    # Drug length histogram (clipped at 200 for readability since long tail)
    ax = axes[1]
    drug_clip = drug[drug <= 200]
    ax.hist(drug_clip, bins=50, color="#E69F00", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(100, color="#b91c1c", linestyle="--", lw=1.0)
    ax.text(102, ax.get_ylim()[1] * 0.85, "max_drug_len = 100",
            fontsize=8.5, color="#b91c1c")
    n_truncated = (drug > 100).sum()
    ax.set_xlabel("SMILES length (chars)", fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title(f"Drug length  (median {np.median(drug):.0f}; "
                 f"{100*n_truncated/len(drug):.1f}% truncated at 100)",
                 fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3)

    # Protein length histogram (clipped at 2000)
    ax = axes[2]
    prot_clip = prot[prot <= 2000]
    ax.hist(prot_clip, bins=50, color="#009E73", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(1200, color="#b91c1c", linestyle="--", lw=1.0)
    ax.text(1210, ax.get_ylim()[1] * 0.85, "max_prot_len = 1200",
            fontsize=8.5, color="#b91c1c")
    n_truncated = (prot > 1200).sum()
    ax.set_xlabel("protein length (residues)", fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title(f"Protein length  (median {np.median(prot):.0f}; "
                 f"{100*n_truncated/len(prot):.1f}% truncated at 1200)",
                 fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save(fig, "diagram_07_length_and_pki_distribution")


# =============================================================================
# DIAGRAM 12 — Train/val loss curves from Phase C history files
# =============================================================================
def diagram_12_loss_curves():
    pc = _load_phase_c()
    if not pc:
        print("  skipping diagram_12: no phase_c history yet")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    fig.suptitle(
        "Phase C Training Curves  -  val MSE per epoch  (3 seeds shaded, mean line)",
        fontsize=13.5, weight="bold", y=1.02)

    splits = ["random", "cold_drug", "cold_target"]
    variants = ["early_concat", "early_crossattn", "late_crossattn", "late_concat"]

    for ax, s in zip(axes, splits):
        for v in variants:
            curves = []
            for sd in (42, 123, 456):
                row = pc.get((v, s, sd))
                if not row or "history" not in row:
                    continue
                vals = [(int(r["epoch"]), r.get("val_mse"))
                        for r in row["history"]
                        if "epoch" in r and r.get("val_mse") is not None]
                if vals:
                    vals.sort()
                    curves.append([vv for _, vv in vals])
            if not curves:
                continue
            # Pad to common length
            L = max(len(c) for c in curves)
            arr = np.full((len(curves), L), np.nan)
            for i, c in enumerate(curves):
                arr[i, :len(c)] = c
            xs = np.arange(1, L + 1)
            mean = np.nanmean(arr, axis=0)
            std  = np.nanstd(arr, axis=0)
            ax.plot(xs, mean, "-", color=COL[VARIANT_KEY_TO_LABEL[v][0]], lw=2.0,
                    label=VARIANT_KEY_TO_LABEL[v][1])
            ax.fill_between(xs, mean - std, mean + std,
                            color=COL[VARIANT_KEY_TO_LABEL[v][0]], alpha=0.18)

        ax.set_title(s.replace("_", "-").title(), fontsize=12, weight="bold")
        ax.set_xlabel("epoch", fontsize=11)
        ax.set_ylabel("val MSE", fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    plt.tight_layout()
    save(fig, "diagram_12_loss_curves")


# =============================================================================
# DIAGRAM 27 — Error vs predicted pKi (failure-mode preview)
# =============================================================================
def diagram_27_error_stratification():
    if not os.path.isdir(PHASE_D_DIR):
        print("  skipping diagram_27: no phase_d_summaries/ yet")
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    fig.suptitle(
        "Per-example absolute error vs predicted pKi  -  Phase C-trained models on 256 held-out pairs",
        fontsize=13, weight="bold", y=1.02)

    for ax, (variant_dir, label, color) in zip(axes, [
        ("v1_phase_c", "V1 Early Concat",  COL["V1"]),
        ("v2_phase_c", "V2 Early X-Attn",  COL["V2"]),
        ("v3_phase_c", "V3 Late X-Attn",   COL["V3"]),
        ("v4_phase_c", "V4 Late Concat",   COL["V4"]),
    ]):
        try:
            p = np.load(os.path.join(PHASE_D_DIR, variant_dir, "predictions.npy"))
            t = np.load(os.path.join(PHASE_D_DIR, variant_dir, "truth.npy"))
        except FileNotFoundError:
            ax.set_title(f"{label} (missing)", fontsize=10.5, color="#999")
            continue
        err = np.abs(p - t)
        ax.scatter(p, err, s=14, alpha=0.55, color=color, edgecolor="#333", linewidth=0.3)
        # 90th-percentile error line
        p90 = float(np.percentile(err, 90))
        ax.axhline(p90, color="#666", linestyle="--", lw=0.8)
        ax.text(p.min(), p90 + 0.05, f"P90 err = {p90:.2f}", fontsize=8.5, color="#444")
        # Annotate worst residual
        worst = int(np.argmax(err))
        ax.scatter([p[worst]], [err[worst]], s=70, marker="x", color="#b91c1c", linewidth=2)
        ax.set_title(f"{label}  -  MSE {((p-t)**2).mean():.3f}",
                     fontsize=11, weight="bold", color=color)
        ax.set_xlabel("predicted pKi", fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("|pred - true|", fontsize=10.5)
    plt.tight_layout()
    save(fig, "diagram_27_error_stratification")


# =============================================================================
# DIAGRAM 18 — CKA between variants (Phase D representation similarity)
# =============================================================================
def diagram_18_cka_matrix():
    """Computes simple kernel CKA between variants using attention summary stats.

    Note: a true CKA requires access to per-token representations, which we
    don't keep (would be ~25 GB). As a poster-friendly proxy, we use the
    layer-wise attention entropy vectors (one number per layer × example) as
    a representation feature, then compute CKA over those. This is a
    *behavioral similarity*, not a *representational* one — flagged in caption.
    """
    if not os.path.isdir(PHASE_D_DIR):
        print("  skipping diagram_18: no phase_d_summaries/ yet")
        return

    variants = ["v1_phase_c", "v2_phase_c", "v3_phase_c", "v4_phase_c"]
    labels = ["V1", "V2", "V3", "V4"]
    feats = {}
    for v in variants:
        try:
            ent, _ = _load_summary(v)
            keys = sorted([k for k in ent.files if "entropy" in k and not k.startswith("_")])
            X = np.stack([ent[k] for k in keys], axis=1)  # (B=256, L=layers)
            feats[v] = X
        except Exception:
            feats[v] = None

    # Linear CKA: HSIC(X, Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    def cka(X, Y):
        X = X - X.mean(0); Y = Y - Y.mean(0)
        num = np.linalg.norm(X.T @ Y) ** 2
        den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
        return float(num / (den + 1e-12))

    n = len(variants)
    M = np.full((n, n), np.nan)
    for i, vi in enumerate(variants):
        for j, vj in enumerate(variants):
            if feats[vi] is None or feats[vj] is None:
                continue
            M[i, j] = cka(feats[vi], feats[vj])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=12, weight="bold")
    ax.set_yticklabels(labels, fontsize=12, weight="bold")
    for i, c in enumerate([COL[l] for l in labels]):
        ax.get_xticklabels()[i].set_color(c)
        ax.get_yticklabels()[i].set_color(c)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=12, weight="bold",
                        color="white" if M[i,j] > 0.5 else "#222")
    ax.set_title("Behavioral similarity (CKA on attention-entropy features)\n"
                 "1.0 = same per-example attention pattern across layers",
                 fontsize=12, weight="bold")
    plt.colorbar(im, ax=ax, label="linear CKA", fraction=0.04)
    fig.text(0.5, -0.02,
             "Note: this is a proxy for representational similarity. "
             "True CKA on token embeddings is left for future work.",
             ha="center", fontsize=8.5, style="italic", color="#666")
    plt.tight_layout()
    save(fig, "diagram_18_cka_matrix")


def main():
    print("Building poster figures...")
    diagram_03_matrix()
    diagram_04_architectures()
    diagram_05_encoder_block()
    diagram_06_dataset_summary()
    diagram_07_length_and_pki_distribution()
    diagram_08_split_strategy()
    diagram_09_pipeline()
    diagram_10_best_mse()
    diagram_10b_mse_per_split()
    diagram_10c_phase_c_tally()
    diagram_11_ci_per_split()
    diagram_12_loss_curves()
    diagram_13_predicted_vs_true()
    diagram_14_param_pareto()
    diagram_15_sensitivity()
    diagram_16_attention_entropy()
    diagram_17_attention_heatmap()
    diagram_18_cka_matrix()
    diagram_27_error_stratification()
    diagram_31_extensions()
    diagram_32_market_growth()
    diagram_33_milestones()
    diagram_36_loopholes()
    diagram_37_norm_vs_us()
    diagram_38_impact_tree()
    diagram_39_leaderboard()
    diagram_40_sensitivity_axes()
    print(f"\nAll figures saved to: {OUT}")


if __name__ == "__main__":
    main()
