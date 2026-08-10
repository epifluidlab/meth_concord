#!/usr/bin/env python3
"""Why C and D are both needed: coordination magnitude vs realized direction.

Four scenarios, all against a marginal-matched null:

    scenario                D        C        reads as
    coordinated co-open    +0.151   0.304     co-open
    coordinated co-closed  -0.151   0.304     co-closed, SAME C
    uncoordinated           0.000   0.000     mixed
    coordinated balanced    0.000   0.405     D~0 but C large -- ambiguous

Panel a draws each scenario's co-state composition (observed vs the null) so
the numbers are traceable to the mass that produced them.  Panel b is the
point of the figure: in the (C, D) plane the four scenarios are distinct, but
each axis ALONE collapses a different pair --

    C alone  merges co-open with co-closed   (identical magnitude)
    D alone  merges uncoordinated with balanced (both sit at D = 0)

...so neither is sufficient and the pair is.

Values are the exact fixture numbers from
tests/test_signed_concordance.py (soft posteriors 0.95 / 0.05), so the figure
and the regression test cannot drift apart.

Emits a vector PDF + a 300-dpi PNG into figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- palette (open = accessible = teal, matching the GCH white->teal maps) -
OPEN_C = "#2A9D8F"
MIX_C = "#CBD1D8"
CLOSE_C = "#34495E"
COORD_C = "#7E6FCB"
INK = "#23262B"
MUT = "#6B7280"
EDGE = "#D5D8DF"
PANEL = "#F7F8FA"

# Composition is (co-open, mixed, co-closed).  O = observed long-range
# contacts, E = the marginal-matched null.  pairs = the cartoon, 1 = open.
SCEN = [
    dict(
        key="co-open", title="coordinated\nco-open",
        pairs=[(1, 1)] * 6 + [(0, 0)] * 2,
        O=(0.6775, 0.0950, 0.2275), E=(0.5256, 0.3987, 0.0756),
        C=0.304, D=+0.151, verdict="co-open",
    ),
    dict(
        key="co-closed", title="coordinated\nco-closed",
        pairs=[(1, 1)] * 2 + [(0, 0)] * 6,
        O=(0.2275, 0.0950, 0.6775), E=(0.0756, 0.3987, 0.5256),
        C=0.304, D=-0.151, verdict="co-closed  (same C)",
    ),
    dict(
        key="uncoordinated", title="uncoordinated",
        pairs=[(1, 1)] * 4 + [(1, 0), (0, 1), (1, 0)] + [(0, 0)],
        O=(0.5256, 0.3987, 0.0756), E=(0.5256, 0.3987, 0.0756),
        C=0.000, D=0.000, verdict="mixed",
    ),
    dict(
        key="balanced", title="coordinated\nbalanced",
        pairs=[(1, 1)] * 4 + [(0, 0)] * 4,
        O=(0.4525, 0.0950, 0.4525), E=(0.2500, 0.5000, 0.2500),
        C=0.405, D=0.000, verdict="ambiguous direction",
    ),
]


def _stack(ax, x, y, w, h, comp, label):
    """One horizontal co-open | mixed | co-closed composition bar."""
    left = x
    for frac, col in zip(comp, (OPEN_C, MIX_C, CLOSE_C)):
        ax.add_patch(plt.Rectangle((left, y), w * frac, h, facecolor=col,
                                   edgecolor="white", linewidth=0.6))
        left += w * frac
    ax.text(x - 1.4, y + h / 2, label, ha="right", va="center",
            fontsize=8.5, color=MUT, style="italic")


def _panel_a(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    centres = [13.5, 37.5, 61.5, 85.5]
    col_w = 21.0

    for cx, s in zip(centres, SCEN):
        x0 = cx - col_w / 2
        ax.add_patch(plt.Rectangle((x0 - 1.5, 4), col_w + 3, 90,
                                   facecolor=PANEL, edgecolor=EDGE,
                                   linewidth=0.8, zorder=0))
        ax.text(cx, 90, s["title"], ha="center", va="top", fontsize=10.5,
                fontweight="bold", color=INK, linespacing=1.35)

        # --- contact cartoons: 2 rows x 4 pairs, filled = open ------------
        for i, (a, b) in enumerate(s["pairs"]):
            row, col = divmod(i, 4)
            px = x0 + 2.6 + col * 4.6
            py = 74.5 - row * 9.0
            ax.plot([px, px + 2.4], [py, py], color="#9AA0A6", lw=0.8,
                    zorder=1, solid_capstyle="round")
            t = np.linspace(0, np.pi, 24)
            ax.plot(px + 1.2 - 1.2 * np.cos(t), py + 2.6 * np.sin(t),
                    color="#B9BEC6", lw=0.7, zorder=1)
            for dx, st in ((0.0, a), (2.4, b)):
                ax.scatter([px + dx], [py], s=26, zorder=3,
                           facecolor=OPEN_C if st else "white",
                           edgecolor=OPEN_C if st else CLOSE_C, linewidth=1.0)

        # --- composition bars --------------------------------------------
        _stack(ax, x0 + 4.5, 48.0, col_w - 6.0, 4.4, s["O"], "O")
        _stack(ax, x0 + 4.5, 41.0, col_w - 6.0, 4.4, s["E"], "E")

        # --- C and D read-outs --------------------------------------------
        bx, bw = x0 + 5.0, col_w - 7.0
        ax.text(x0 + 1.0, 32.5, "C", fontsize=9.5, color=MUT, va="center",
                style="italic")
        ax.add_patch(plt.Rectangle((bx, 30.6), bw, 3.6, facecolor="white",
                                   edgecolor=EDGE, linewidth=0.7))
        ax.add_patch(plt.Rectangle((bx, 30.6), bw * s["C"] / 0.45, 3.6,
                                   facecolor=COORD_C, edgecolor="none"))
        ax.text(bx + bw + 0.8, 32.4, f"{s['C']:.3f}", fontsize=9,
                va="center", color=INK)

        # D is diverging: drawn from the bar's midline, left = co-closed.
        mid = bx + bw / 2
        ax.text(x0 + 1.0, 22.5, "D", fontsize=9.5, color=MUT, va="center",
                style="italic")
        ax.add_patch(plt.Rectangle((bx, 20.6), bw, 3.6, facecolor="white",
                                   edgecolor=EDGE, linewidth=0.7))
        ax.plot([mid, mid], [20.6, 24.2], color=EDGE, lw=0.7, zorder=2)
        if abs(s["D"]) > 1e-9:
            span = (bw / 2) * (s["D"] / 0.22)
            ax.add_patch(plt.Rectangle((min(mid, mid + span), 20.6),
                                       abs(span), 3.6, zorder=3,
                                       facecolor=OPEN_C if s["D"] > 0
                                       else CLOSE_C, edgecolor="none"))
        ax.text(bx + bw + 0.8, 22.4, f"{s['D']:+.3f}", fontsize=9,
                va="center", color=INK)

        ax.text(cx, 12.0, s["verdict"], ha="center", va="center",
                fontsize=9.2, color=INK, style="italic")

    # shared legend for the composition bars
    lx = 3.0
    for lab, col in (("co-open", OPEN_C), ("mixed", MIX_C),
                     ("co-closed", CLOSE_C)):
        ax.add_patch(plt.Rectangle((lx, 0.0), 3.0, 2.8, facecolor=col,
                                   edgecolor="white", linewidth=0.5))
        ax.text(lx + 3.8, 1.4, lab, fontsize=8.6, va="center", color=MUT)
        lx += 15.0
    ax.text(97, 1.4, "O = long-range contacts     E = marginal-matched null",
            fontsize=8.6, ha="right", va="center", color=MUT, style="italic")


def _panel_b(ax):
    """The (C, D) plane — where each axis alone collapses a pair.

    The two collapses are drawn as PROJECTIONS onto the axes rather than as
    braces between the points: "what this metric alone sees" is literally the
    projection, and it keeps the annotation out of the interior, where the
    two indicators would otherwise cross at (0.304, 0) on top of three
    labels.
    """
    red = "#B0453A"
    y_lo = -0.30
    ax.set_xlim(-0.05, 0.56)
    ax.set_ylim(y_lo, 0.27)

    # --- collapse 1: co-open and co-closed project to the SAME C ----------
    for sign in (+1, -1):
        ax.plot([0.304, 0.304], [sign * 0.151, y_lo + 0.055],
                color=red, lw=0.8, ls=":", zorder=2)
    ax.scatter([0.304], [y_lo + 0.055], marker="^", s=45, color=red, zorder=3)
    ax.text(0.304, y_lo + 0.028, "C alone: both project here",
            fontsize=8.2, ha="center", va="center", color=red)

    # --- collapse 2: uncoordinated and balanced BOTH sit on D = 0 ---------
    ax.plot([0.0, 0.405], [0, 0], color=red, lw=2.4, alpha=0.30, zorder=1,
            solid_capstyle="butt")
    ax.text(0.185, 0.028, "D alone: both sit at 0", fontsize=8.2,
            ha="center", va="center", color=red)

    ax.axhline(0, color="#9AA0A6", lw=0.9, zorder=0)
    ax.axvline(0, color="#9AA0A6", lw=0.9, zorder=0)

    for s, dx, dy, ha in ((SCEN[0], 0.020, 0.034, "left"),
                          (SCEN[1], 0.020, -0.034, "left"),
                          (SCEN[2], 0.012, -0.050, "left"),
                          (SCEN[3], 0.000, -0.050, "center")):
        col = (OPEN_C if s["D"] > 1e-9 else
               CLOSE_C if s["D"] < -1e-9 else COORD_C)
        ax.scatter([s["C"]], [s["D"]], s=95, zorder=5, facecolor=col,
                   edgecolor="white", linewidth=1.4)
        ax.text(s["C"] + dx, s["D"] + dy, s["title"].replace("\n", " "),
                fontsize=8.8, ha=ha, va="center", color=INK, zorder=5)

    ax.set_xlabel("C   —   coordination magnitude", fontsize=9.5)
    ax.set_ylabel("D   —   directional coordination", fontsize=9.5)
    ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax.tick_params(labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.text(0.555, 0.235, "co-open", fontsize=8.6, color=OPEN_C, ha="right",
            fontweight="bold")
    ax.text(0.555, -0.245, "co-closed", fontsize=8.6, color=CLOSE_C,
            ha="right", fontweight="bold")


TAKEAWAY = [
    ("C  answers  “how strongly coordinated?”",
     "Blind to direction by construction: Cov is unchanged by\n"
     "open ↔ closed, so co-open and co-closed score identically."),
    ("D  answers  “coordinated which way?”",
     "Signs C by where the concordant mass sits, F. Collapses to 0\n"
     "when there is nothing to sign — and also when F ≈ 0.5."),
    ("Report both.",
     "D ≈ 0 is ambiguous alone: with C ≈ 0 it means mixed, with C\n"
     "large it means coordinated but balanced. Only the pair separates them."),
]


def build():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(13.2, 9.0), facecolor="white")

    ax_a = fig.add_axes([0.035, 0.435, 0.93, 0.525])
    _panel_a(ax_a)
    fig.text(0.035, 0.975, "a", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.058, 0.975, "the same four scenarios, as co-state mass",
             fontsize=11.5, color=MUT, va="baseline")

    ax_b = fig.add_axes([0.075, 0.075, 0.42, 0.30])
    _panel_b(ax_b)
    fig.text(0.035, 0.395, "b", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.058, 0.395, "neither axis is sufficient alone",
             fontsize=11.5, color=MUT, va="baseline")

    y = 0.345
    for head, body in TAKEAWAY:
        fig.text(0.565, y, head, fontsize=10.6, fontweight="bold", color=INK)
        fig.text(0.565, y - 0.032, body, fontsize=9.3, color=MUT,
                 linespacing=1.5, va="top")
        y -= 0.108
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=str(Path(__file__).parent))
    ap.add_argument("--stem", default="coordination_vs_direction")
    args = ap.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = build()
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        path = out / f"{args.stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
