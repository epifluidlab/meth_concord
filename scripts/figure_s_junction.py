#!/usr/bin/env python3
"""Supplementary figure: concordance against the DNA between a contact's
two ends when labelled, from junction_concordance.py's tables.

M.CviPI labels the nuclei after proximity ligation, when a contact's two
ends are one molecule.  Were their likeness made by access to that molecule
at labelling time, it would fall with the DNA between them, as it does
along one unligated molecule.  The panels, in Fig. 5's style:

  a  s, that DNA: on a ligation product, each end's mean site distance to
     the junction -- the first DpnII site in its read's 3' direction --
     summed; on one unligated molecule, the distance between its two ends'
     mean sites
  b  GCH: the end posteriors' r against s, for one unligated molecule's two
     ends and for contacts (cis, >= 10 kb, linked) and their stand-ins
     (non-linked: control 1 of Fig. 5c,d); beneath, Δr, linked against
     non-linked, cis and trans, with its cell-bootstrap interval
  c  the same for HCG

Shaded, s below JUNCTION_BP: both ends' sites by the junction.  Beside each
axis, the contacts with s of 1 kb or more or no junction found.

    python scripts/figure_s_junction.py \\
        --junction output/junction_concordance --output FigureS_junction.pdf
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

#: The marks: (mark, title).
MARKS = (("GCH", "GCH accessibility"), ("HCG", "HCG methylation"))
#: Below it (bp), shaded: both ends' sites by the junction.
JUNCTION_BP = 100
#: One unligated molecule's two ends.
MOLECULE_C = "0.3"
#: a: the partner locus's fragment, beside figure5_cortex's SEGMENT_C.
SEGMENT2_C = "#f2c28f"
#: The page in inches: its margins, a's height, the gap beneath it and b's
#: and c's -- r above Δr, as tall as these.  4.5 inches in all, at the usual
#: resolutions a whole number of pixels.
TOP, A_H, GAP, BC_H, FOOT = 0.08, 0.95, 0.42, 2.55, 0.5
BC_ROWS = (1.45, 1)
#: Beside each axis, the strip for the contacts without s: its width as a
#: share of the axis's.
STRIP = 0.075


def read_table(junction) -> pd.DataFrame:
    """junction_concordance.tsv, every mark in it; or stop, naming what is
    missing."""
    p = Path(junction) / "junction_concordance.tsv"
    if not p.exists():
        sys.exit(f"missing: {p}")
    t = pd.read_csv(p, sep="\t")
    miss = [m for m, _ in MARKS if not (t["mark"] == m).any()]
    if miss:
        sys.exit(f"{p} has no {', '.join(miss)}")
    return t


def series(t: pd.DataFrame, mark: str, set_: str):
    """A set's bins of s in order, their centres, and its last bin: s of
    1 kb or more, or no junction found."""
    u = t[(t["mark"] == mark) & (t["set"] == set_)]
    b = u[u["s_lo"].notna()].sort_values("s_lo")
    return b, (b["s_lo"] + b["s_hi"]).to_numpy(float) / 2, \
        u[u["s_lo"].isna()]


def junction_share(t: pd.DataFrame, mark: str,
                   set_: str = "contacts") -> float:
    """The share of a set's pairs with s below JUNCTION_BP."""
    u = t[(t["mark"] == mark) & (t["set"] == set_)]
    n = u["pairs"].sum()
    return float(u.loc[u["s_hi"] <= JUNCTION_BP, "pairs"].sum() / n) \
        if n else np.nan


def pct(frac: float) -> str:
    """A small share to one decimal (0.4%), to two figures below 0.1%."""
    return f"{100 * frac:.1f}%" if frac >= 0.001 else f"{100 * frac:.2g}%"


def _schematic(ax):
    """a: a ligation product's two ends, each read pointing at the junction,
    s the DNA between their sites' means through it; one unligated
    molecule's two ends, s the DNA between theirs."""
    mk, on, off, *_ = f5.SITE_STYLE["GCH"]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    y_read, y_dna, y_span = 7.3, 5.6, 3.3

    def dna(x0, x1, c):
        ax.plot([x0, x1], [y_dna, y_dna], color=c, lw=3.0,
                solid_capstyle="butt", zorder=1)

    def read(x5, x3, sites, states) -> float:
        """A read from its 5' end to its 3' end, its sites on it; their
        mean, dropped to the spans."""
        ax.annotate("", xy=(x3, y_read), xytext=(x5, y_read),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.2",
                                    mutation_scale=5, shrinkA=0, shrinkB=0))
        for x_, st in zip(sites, states):
            ax.plot(x_, y_read, mk, ms=2.6, mfc=on if st else off,
                    mec="black", mew=0.4, zorder=3)
        c = float(np.mean(sites))
        ax.plot([c, c], [y_span, y_read - 0.5], ls=(0, (1, 1)), color="0.45",
                lw=0.4, zorder=0)
        return c

    def span(x0, x1, label):
        ax.annotate("", xy=(x1, y_span), xytext=(x0, y_span),
                    arrowprops=dict(arrowstyle="<|-|>", lw=0.45,
                                    color="0.25", mutation_scale=3.5,
                                    shrinkA=0, shrinkB=0))
        ax.text((x0 + x1) / 2, y_span - 0.45, label, ha="center", va="top",
                fontsize=5)

    def locus(x, ha, label):
        ax.text(x, y_dna - 0.75, label, ha=ha, va="top", fontsize=4.6,
                color="0.4")

    # a ligation product: two loci's fragments, joined where DpnII cut
    j = 25
    ax.text(1, 10, "ligation product: one molecule when labelled",
            ha="left", va="top", fontsize=5.2)
    dna(2, j, f5.SEGMENT_C)
    dna(j, 48, SEGMENT2_C)
    ax.plot([j, j], [y_dna - 0.9, y_dna + 0.9], color="black", lw=0.7,
            zorder=2)
    ax.text(j, y_read + 0.7, "junction (DpnII)", ha="center", va="bottom",
            fontsize=4.8)
    locus(2, "left", "locus 1")
    locus(48, "right", "locus 2")
    c1 = read(9, j - 2, (11.5, 15.0, 19.5), (1, 0, 1))
    c2 = read(41, j + 2, (29.5, 33.0, 38.0), (0, 1, 1))
    span(c1, j, "$d_1$")
    span(j, c2, "$d_2$")
    ax.text(j, 0.9, "$s$ = $d_1$ + $d_2$", ha="center", va="center",
            fontsize=5.2)
    # one unligated molecule: an inward pair on one fragment
    ax.text(53, 10, "one unligated molecule", ha="left", va="top",
            fontsize=5.2)
    dna(54, 98, f5.SEGMENT_C)
    locus(54, "left", "locus 1")
    c1 = read(58, 71, (60.5, 64.0, 68.5), (1, 1, 0))
    c2 = read(95, 82, (84.0, 88.5, 92.0), (0, 1, 0))
    span(c1, c2, "$s$")


def panel_mark(axes, t: pd.DataFrame, mark: str, title: str):
    """``axes``: r and its strip, Δr and its strip, for one mark."""
    ax, ax_n, axd, axd_n = axes
    col = f5.MARK_COLOR[mark]
    m, x, _ = series(t, mark, "molecules")
    if len(m):
        ax.fill_between(x, m["r_lo"], m["r_hi"], color=MOLECULE_C,
                        alpha=0.18, lw=0)
        ax.plot(x, m["r"], "-o", color=MOLECULE_C, ms=1.6,
                label="one unligated molecule")
    c, x, none = series(t, mark, "contacts")
    ax.plot(x, c["r"], "-o", color=col, ms=1.6, label="contacts (linked)")
    ax.plot(x, c["r_null"], "--o", color=col, ms=1.6, mfc="white", mew=0.5,
            lw=0.8, alpha=0.9, label="non-linked")
    if len(none):
        ax_n.plot(0, none["r"].iloc[0], "o", color=col, ms=1.6)
        ax_n.plot(0, none["r_null"].iloc[0], "o", color=col, ms=1.6,
                  mfc="white", mew=0.5, alpha=0.9)
    ax.set_title(title, loc="left", color=col)
    ax.legend(frameon=False, loc="upper right", handlelength=1.8)
    for set_, ls, mk_, alpha, at, label in (
            ("contacts", "-", "o", 0.2, -0.35, "cis contacts"),
            ("trans", (0, (1.2, 1.2)), "s", 0.1, 0.35, "trans contacts")):
        u, x, none = series(t, mark, set_)
        if u.empty:
            continue
        axd.fill_between(x, u["delta_lo"], u["delta_hi"], color=col,
                         alpha=alpha, lw=0)
        axd.plot(x, u["delta"], ls=ls, marker=mk_, color=col, ms=1.6,
                 label=label)
        if len(none) and np.isfinite(none["delta"].iloc[0]):
            r = none.iloc[0]
            axd_n.errorbar(at, r["delta"],
                           yerr=[[r["delta"] - r["delta_lo"]],
                                 [r["delta_hi"] - r["delta"]]],
                           fmt=mk_, color=col, ms=1.6, elinewidth=0.6,
                           capsize=0)
    axd.legend(frameon=False, loc="upper right", handlelength=1.8)
    share = junction_share(t, mark)
    if np.isfinite(share):
        axd.text(JUNCTION_BP, 0.97, f"  s < {JUNCTION_BP} bp: {pct(share)} "
                 "of contacts", transform=axd.get_xaxis_transform(),
                 ha="left", va="top", fontsize=4.8, color="0.4")
    s_max = float(t["s_hi"].max())
    for a in (ax, axd):
        a.axvspan(0, JUNCTION_BP, color="0.93", lw=0, zorder=0)
        a.set_xlim(0, s_max)
        f5._zero(a)
    ax.text(JUNCTION_BP / 2, 0.2, "junction", rotation=90, ha="center",
            va="bottom", fontsize=4.6, color="0.4",
            transform=ax.get_xaxis_transform())
    ax.tick_params(labelbottom=False)
    axd.set_xlabel("s, DNA between the two ends' sites (bp)")
    for a in (ax_n, axd_n):
        a.set_xlim(-1, 1)
        f5._zero(a)
        a.spines["left"].set_visible(False)
        a.tick_params(left=False, labelleft=False)
    ax_n.set_xticks([])
    axd_n.set_xticks([0])
    axd_n.set_xticklabels(["≥1 kb\nor none"])


def build_figure(t: pd.DataFrame):
    plt = f5._style()
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    H = TOP + A_H + GAP + BC_H + FOOT
    fig = plt.figure(figsize=(7.2, H))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[A_H, BC_H],
                  hspace=GAP / ((A_H + BC_H) / 2), left=0.075, right=0.975,
                  top=1 - TOP / H, bottom=FOOT / H)
    ax_a = fig.add_subplot(gs[0])
    _schematic(ax_a)
    gbc = GridSpecFromSubplotSpec(1, 2, gs[1], wspace=0.2)
    first = []
    for j, (mark, title) in enumerate(MARKS):
        g = GridSpecFromSubplotSpec(2, 2, gbc[j], hspace=0.1, wspace=0.16,
                                    width_ratios=[1, STRIP],
                                    height_ratios=list(BC_ROWS))
        ax = fig.add_subplot(g[0, 0])
        axd = fig.add_subplot(g[1, 0], sharex=ax)
        axes = (ax, fig.add_subplot(g[0, 1], sharey=ax), axd,
                fig.add_subplot(g[1, 1], sharey=axd))
        panel_mark(axes, t, mark, title)
        if j == 0:
            ax.set_ylabel("Pearson r of end posteriors")
            axd.set_ylabel("Δr, linked − non-linked")
        first.append(ax)
    fig.canvas.draw()
    # the letters as Fig. 5's, a tenth of an inch above the axes; a's at
    # the top of its schematic, over b's
    x_b = first[0].get_position().x0 - 0.06
    fig.text(x_b, ax_a.get_position().y1, "a", fontsize=8,
             fontweight="bold", va="top", ha="left")
    for lab, ax, x in (("b", first[0], x_b),
                       ("c", first[1], first[1].get_position().x0 - 0.045)):
        fig.text(x, ax.get_position().y1 + 0.1 / H, lab, fontsize=8,
                 fontweight="bold", va="bottom", ha="left")
    return fig


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--junction", required=True,
                    help="junction_concordance.py's --output-dir, its "
                         "junction_concordance.tsv.")
    ap.add_argument("--output", required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    fig = build_figure(read_table(args.junction))
    import matplotlib.pyplot as plt
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
