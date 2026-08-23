#!/usr/bin/env python3
"""Mechanistic scheme: where long-range epigenetic coordination goes in tumour.

A MODEL, not a measurement.  The panel it belongs beside reports what was
observed -- GCH-GCH and HCG-HCG concordance rising inside cLADs, late-replicating
domains and PMDs, and falling outside them.  This draws the arrangement that
would produce that, and the caption should say so; nothing here is fitted to
data and the arrow thicknesses are chosen, not measured.

The two nuclei differ in exactly one respect -- which compartment carries the
coordination -- because a schematic that changes several things at once lets a
reader attribute the effect to whichever one they already believed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TUMOUR = "#c0392b"
OTHER = "#2c3e50"
LAMINA = "#7f8c8d"
B_FILL = "#d5dbdb"
A_FILL = "#fdf2e9"
GENE = "#b9770e"


def _arc(ax, p0, p1, bulge, color, lw, alpha=1.0, ls="-", zorder=5):
    """A quadratic curve between two loci -- one coordinated pair."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    mid = (p0 + p1) / 2.0
    d = p1 - p0
    normal = np.array([-d[1], d[0]])
    n = np.linalg.norm(normal)
    ctrl = mid + (normal / n * bulge if n else 0.0)
    ax.add_patch(PathPatch(
        MPath([tuple(p0), tuple(ctrl), tuple(p1)],
              [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]),
        fc="none", ec=color, lw=lw, alpha=alpha, ls=ls,
        zorder=zorder, capstyle="round"))


def _nucleus(ax, cx, cy, rx, ry):
    from matplotlib.patches import Ellipse

    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry, fc=A_FILL, ec="none",
                         zorder=0))
    # the lamina: drawn as a ring, because the B domains are tethered TO it
    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry, fc="none", ec=LAMINA,
                         lw=3.2, zorder=1))
    ax.add_patch(Ellipse((cx, cy), 2 * rx * 0.995, 2 * ry * 0.99, fc="none",
                         ec="white", lw=0.8, zorder=1.1))


def _b_domain(ax, cx, cy, rx, ry, angle_deg, width_deg, depth):
    """A heterochromatin domain pressed against the lamina.

    Returns the loci on it that the coordination arcs connect.
    """
    from matplotlib.patches import Polygon

    a0 = np.radians(angle_deg - width_deg / 2)
    a1 = np.radians(angle_deg + width_deg / 2)
    ts = np.linspace(a0, a1, 40)
    outer = np.column_stack([cx + rx * np.cos(ts), cy + ry * np.sin(ts)])
    inner = np.column_stack([cx + (rx - depth) * np.cos(ts[::-1]),
                             cy + (ry - depth) * np.sin(ts[::-1])])
    ax.add_patch(Polygon(np.vstack([outer, inner]), closed=True, fc=B_FILL,
                         ec="#95a5a6", lw=0.8, zorder=2))
    mid = (rx + (rx - depth)) / 2.0, (ry + (ry - depth)) / 2.0
    loci = []
    for f in (0.22, 0.5, 0.78):
        t = a0 + f * (a1 - a0)
        loci.append((cx + mid[0] * np.cos(t), cy + mid[1] * np.sin(t)))
    return loci


def _gene_hub(ax, pts, color=GENE, r=0.030):
    from matplotlib.patches import Circle
    for p in pts:
        ax.add_patch(Circle(p, r, fc=color, ec="white", lw=0.7, zorder=6))


def draw(ax, title, tag, b_strength, a_strength, note, cross=False):
    """One nucleus.  b_strength / a_strength scale the arcs, nothing else."""
    cx, cy, rx, ry = 0.5, 0.50, 0.40, 0.34
    _nucleus(ax, cx, cy, rx, ry)

    b_loci = []
    for ang, wid in ((122, 74), (248, 70), (356, 58)):
        b_loci.append(_b_domain(ax, cx, cy, rx, ry, ang, wid, 0.115))

    # A-compartment loci: interior, arranged as a regulatory hub
    a_loci = [(cx - 0.145, cy + 0.075), (cx + 0.020, cy + 0.140),
              (cx + 0.155, cy + 0.040), (cx + 0.075, cy - 0.120),
              (cx - 0.115, cy - 0.095)]
    _gene_hub(ax, a_loci)
    for d in b_loci:
        _gene_hub(ax, d, color="#5d6d7e", r=0.020)

    # --- coordination WITHIN heterochromatin domains -------------------
    for d in b_loci:
        for i in range(len(d) - 1):
            _arc(ax, d[i], d[i + 1], 0.030, OTHER,
                 0.7 + 3.0 * b_strength, alpha=0.40 + 0.50 * b_strength)
    # and BETWEEN them -- the long-range part of the claim
    for p, q in ((b_loci[0][1], b_loci[1][1]), (b_loci[1][1], b_loci[2][1]),
                 (b_loci[0][1], b_loci[2][1])):
        _arc(ax, p, q, 0.10, OTHER, 0.6 + 3.4 * b_strength,
             alpha=0.30 + 0.58 * b_strength)

    # --- coordination within the A-compartment regulatory hub ----------
    for i in range(len(a_loci)):
        for j in range(i + 1, len(a_loci)):
            _arc(ax, a_loci[i], a_loci[j], 0.045, GENE,
                 0.6 + 2.5 * a_strength, alpha=0.42 + 0.45 * a_strength,
                 ls="-" if not cross else (0, (2.4, 1.8)))

    ax.text(0.02, 0.975, tag, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top")
    ax.set_title(title, fontsize=11, pad=6)
    ax.text(0.5, 0.045, note, transform=ax.transAxes, fontsize=8.2,
            ha="center", va="center", color="0.25")
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(-0.02, 0.96)
    ax.set_aspect("equal")
    ax.axis("off")


def legend_panel(ax):
    """Horizontal key at fixed columns.

    Square swatches, not ellipses stretched by the axes aspect, and fixed
    positions rather than a cumulative advance -- text width is not knowable
    before rendering, so advancing by a guess is how legend items end up on
    top of one another.
    """
    from matplotlib.patches import Rectangle

    ax.axis("off")
    items = [
        (0.015, "patch", B_FILL, "cLAD / late-RT / PMD"),
        (0.290, "line", LAMINA, "nuclear lamina"),
        (0.490, "line", OTHER, "B-domain co-accessibility"),
        (0.760, "line", GENE, "A-hub co-accessibility"),
    ]
    for x, kind, col, lab in items:
        if kind == "patch":
            ax.add_patch(Rectangle((x, 0.36), 0.020, 0.34, fc=col,
                                   ec="#95a5a6", lw=0.8,
                                   transform=ax.transAxes))
        else:
            ax.plot([x, x + 0.020], [0.53, 0.53], color=col, lw=2.6,
                    solid_capstyle="round", transform=ax.transAxes)
        ax.text(x + 0.028, 0.53, lab, fontsize=7.8, va="center",
                transform=ax.transAxes)


def _direction(ax, xy, text, color, up=True):
    """Name the measured direction on the drawing itself.

    A schematic that shows a rearrangement without saying which way the
    number moved leaves the reader to infer it from line weight, which is a
    choice made by whoever drew it.
    """
    ax.annotate(text, xy=xy, xycoords="axes fraction", fontsize=9,
                fontweight="bold", color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color,
                          lw=1.0, alpha=0.92), zorder=20)


def figure(path, title, subtitle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.4, 4.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.13], hspace=0.02,
                          wspace=0.01, left=0.01, right=0.99,
                          top=0.855, bottom=0.02)
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axl = fig.add_subplot(gs[1, :])

    draw(ax0, "Reference cell types", "a", b_strength=0.10, a_strength=0.95,
         note="coordination held in the active\nregulatory compartment")
    draw(ax1, "Meningioma", "b", b_strength=0.95, a_strength=0.14,
         note="coordination shifts into heterochromatin;\n"
              "the regulatory hub decoheres", cross=True)
    _direction(ax1, (0.235, 0.845), "D \u2191", OTHER)
    _direction(ax1, (0.500, 0.530), "D \u2193", GENE)
    legend_panel(axl)

    fig.suptitle(title, fontsize=11.5, y=0.985)
    fig.text(0.5, 0.905, subtitle, ha="center", fontsize=7.8, color="0.32")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True,
                    help="Output figure (.pdf keeps it vector for the "
                         "journal; .png only for a draft).")
    ap.add_argument("--title",
                    default="A compartment-scale redistribution of long-range "
                            "epigenetic coordination")
    ap.add_argument("--subtitle",
                    default="Model consistent with GCH-GCH and HCG-HCG "
                            "concordance rising inside cLADs, late-replicating "
                            "domains and PMDs, and falling outside them")
    args = ap.parse_args(argv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure(out, args.title, args.subtitle)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
