#!/usr/bin/env python3
"""Supplementary figure: Fig. 5a-d with accessibility read at GCY alone.

NOMe-seq reads M.CviPI's GpC label at GCH as accessibility, but a GCA is
also a CpA, and neurons' endogenous mCA there reads as the label.  Here the
GCH sites are narrowed to GCY, GCC and GCT (split_context_bed.py), the
pipeline's steps 1-4 re-run on them and end_posterior_distance_correlation.py
run on their end posteriors as for Fig. 5, with the HCG posteriors as
published.  The panels are Fig. 5a-d's, drawn by figure5_cortex.py's own
functions, each as large and where it is in Fig. 5:

  a  the end posteriors' correlation by distance, linked pairs against
     non-linked ones, and their difference, for GCY-GCY, HCG-HCG and
     HCG-GCY (--distance-ab: its null keeping the contacts' near-copies)
  b  per 10-Mb region, HCG-HCG against GCY-GCY Δr (--distance-ab)
  c  per cell, concordance against the negative control (--distance: its
     null refusing near-copies)
  d  the same by cell type

The runs' tables call the GCY sites GCH, their --posteriors key; the
figure calls them --label.

    python scripts/figure_s_gcy.py \\
        --distance gcy/end_posterior_dist.null_link_shuffled_match_per_cell.cell_center.excl1kb \\
        --distance-ab gcy/end_posterior_dist.null_link_shuffled_match_per_cell.cell_center \\
        --samples mouse_cortex.samples.tsv --output FigureS_GCY.pdf
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


f5 = _load("figure5_cortex")

#: Fig. 5's page without h, as figure5_cortex.build_figure lays it out: its
#: height in inches, its rows' heights (ratios) and the space between them
#: (GridSpec's hspace), its top and bottom margins (shares of the page).
F5_H, F5_ROWS, F5_HSPACE = 10.9, (2.5, 1.75, 2.15, 1.95), 0.46
F5_TOP, F5_BOTTOM = 0.03, 0.04
#: Beneath d, at least this room in inches for its cell types and b's and
#: c's axis labels; the page then rounded up to a quarter inch, at the
#: usual resolutions a whole number of pixels.
FOOT = 0.4


def ad_inches() -> tuple:
    """Fig. 5's rows of a to d in inches: the room above them for the
    letters and titles, the two rows with the space between them, and
    GridSpec's hspace for that space."""
    rows = np.asarray(F5_ROWS)
    n = rows.size
    # GridSpec's own arithmetic: an average row, the space between rows
    cell = (1 - F5_TOP - F5_BOTTOM) * F5_H / (n + F5_HSPACE * (n - 1))
    ad = cell * n * rows[:2] / rows.sum()
    gap = F5_HSPACE * cell
    return F5_TOP * F5_H, ad.sum() + gap, gap / ad.mean()


def layout() -> tuple:
    """Fig. 5's rows of a to d, as tall as there and as far apart: the
    page's height, and GridSpec's hspace, top and bottom for the two."""
    above, span, hspace = ad_inches()
    body = above + span
    H = float(np.ceil((body + FOOT) * 4) / 4)
    return H, hspace, 1 - above / H, (H - body) / H


def ad_grid(fig, top: float, bottom: float, hspace: float):
    """Fig. 5's grid for a to d, its two rows between ``top`` and
    ``bottom`` (shares of the page)."""
    from matplotlib.gridspec import GridSpec
    return GridSpec(2, 12, figure=fig, hspace=hspace, wspace=1.6,
                    height_ratios=list(F5_ROWS[:2]), left=0.075, right=0.975,
                    top=top, bottom=bottom)


def draw_ad(fig, gs, d: dict, order, colors, schematics: bool = True) -> list:
    """Fig. 5a-d in :func:`ad_grid`'s ``gs``, by figure5_cortex.py's own
    functions: a across the first row, b, c and d along the second.  With
    ``schematics`` false, a's reshuffle schematics are left out.  The axes
    each letter goes on, and how far left of them it sits (a share of the
    page's width): [(letter, axes, dx)], a to d."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    # as in Fig. 5: a across the page, then b, c and d
    ga = GridSpecFromSubplotSpec(2, 3, gs[0, :], hspace=0.1, wspace=0.26,
                                 height_ratios=[1.5, 1])
    ax_ar = [fig.add_subplot(ga[0, j]) for j in range(3)]
    ax_ad = [fig.add_subplot(ga[1, j], sharex=ax_ar[j]) for j in range(3)]
    f5.panel_a(ax_ar, ax_ad, d["dist"])
    if not schematics:
        for ax in ax_ar:
            for inset in list(ax.child_axes):
                inset.remove()
    gb = GridSpecFromSubplotSpec(1, 2, gs[1, 0:4], width_ratios=[0.12, 1],
                                 wspace=0.0)
    ax_b = fig.add_subplot(gb[1])
    f5.panel_b(ax_b, d["regions"], d["regions_summary"])
    gc = GridSpecFromSubplotSpec(2, 2, gs[1, 4:7], hspace=0.95, wspace=0.0,
                                 width_ratios=[0.15, 1])
    ax_c = [fig.add_subplot(gc[i, 1]) for i in range(2)]
    f5.panel_c(ax_c, d["percell"])
    gd = GridSpecFromSubplotSpec(2, 1, gs[1, 7:12], hspace=0.22)
    ax_d = [fig.add_subplot(gd[0])]
    ax_d.append(fig.add_subplot(gd[1], sharex=ax_d[0]))
    f5.panel_d(ax_d, d["percell"], order, colors)
    return [("a", ax_ar[0], 0.06), ("b", ax_b, 0.06), ("c", ax_c[0], 0.035),
            ("d", ax_d[0], 0.06)]


def put_letters(fig, letters, H: float):
    """``letters`` (:func:`draw_ad`'s) as Fig. 5's, a tenth of an inch and
    a bit above their axes, on a page ``H`` inches tall."""
    fig.canvas.draw()
    for lab, ax, dx in letters:
        bb = ax.get_position()
        fig.text(bb.x0 - dx, bb.y1 + 0.01 * F5_H / H, lab, fontsize=8,
                 fontweight="bold", va="bottom", ha="left")


def read_tables(args) -> dict:
    """a and b's tables from --distance-ab, c and d's from --distance,
    named as figure5_cortex.read_tables names them; or stop, naming the
    missing ones."""
    dd = Path(args.distance)
    ab = Path(args.distance_ab) if args.distance_ab else dd
    need = [dd / "end_posterior_percell.tsv"] + [ab / f for f in f5.AB_TABLES]
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        sys.exit("missing: " + ", ".join(miss))
    return {"percell": pd.read_csv(need[0], sep="\t"),
            "dist": pd.read_csv(need[1], sep="\t"),
            "regions": pd.read_csv(need[2], sep="\t"),
            "regions_summary": pd.read_csv(need[3], sep="\t").iloc[0]}


def relabel(fig, name: str):
    """Every GCH of ``fig``'s text as ``name``."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        s = t.get_text()
        if "GCH" in s:
            t.set_text(s.replace("GCH", name))
    return fig


def build_figure(d: dict, order, colors, label: str = "GCY"):
    plt = f5._style()
    H, hspace, top, bottom = layout()
    fig = plt.figure(figsize=(7.2, H))
    gs = ad_grid(fig, top, bottom, hspace)
    put_letters(fig, draw_ad(fig, gs, d, order, colors), H)
    return relabel(fig, label)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--distance", required=True,
                    help="end_posterior_distance_correlation.py's output on "
                         "the GCY posteriors, its null refusing the "
                         "contacts' near-copies (--null-exclude 1000): c "
                         "and d, as Fig. 5's.")
    ap.add_argument("--distance-ab", default=None,
                    help="The same with near-copies kept: a and b, as Fig. "
                         "5's.  Default --distance.")
    ap.add_argument("--samples", default=None,
                    help="A samples table giving each cell type's colour "
                         "(--group-column, --color-column); else the "
                         "cortex defaults.")
    ap.add_argument("--group-column", default="group")
    ap.add_argument("--color-column", default="color")
    ap.add_argument("--label", default="GCY",
                    help="The sites' name in the figure (default "
                         "%(default)s); the runs' tables call them GCH.")
    ap.add_argument("--output", required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    d = read_tables(args)
    order, colors = f5.cell_colors(args, d["percell"]["group"].dropna())
    fig = build_figure(d, order, colors, args.label)
    import matplotlib.pyplot as plt
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
