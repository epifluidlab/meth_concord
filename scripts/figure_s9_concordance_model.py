#!/usr/bin/env python3
"""Supplementary Figure 9: how long-range concordance is measured in single
cells -- a schematic of the model behind Figs. 5 and 6, no data drawn.

  a  a contact and its two ends: each mate of a read pair, k of its n GCH
     (or HCG) sites methylated
  b  the end posterior: a two-component beta-binomial mixture, P(open or
     methylated | k, n), centred on the cell's mean
  c  the within-cell null: the anchor kept, the partner replaced by an end
     of the same cell in the partner's 250-kb bin and distance bin, near-
     copies refused
  d  concordance by distance: Δr = r(linked) − r(non-linked)
  e  per cell, against a second null draw, across cells and cell types
  f  per region, GCH against HCG, with a circular genomic permutation
  g  by chromatin class: C, ΔC and the direction weight w
  h  along the genome: hot and cold bins and what they hold
  i  each X copy: stand-ins of the same cell and copy, and the excess that
     pooling the copies makes, ½ δ_a δ_p

    python scripts/figure_s9_concordance_model.py --output s9.v2.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

LINK = "#2f6db5"          # a contact
NULL = "0.45"             # a stand-in
SEG = "#9fd3ee"           # DNA
PANEL_FC, PANEL_EC = "#f6f7f9", "#cfd5dc"
INK, MUTED = "#1f2933", "#5b6573"
HOT, COLD = "#c0392b", "#2c6fbb"
XI, XA = "#b2182b", "#2166ac"
CELL_C, CTL_C = "#2f6db5", "0.7"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6, "axes.linewidth": 0.5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "mathtext.default": "regular",
        "xtick.major.width": 0.4, "ytick.major.width": 0.4,
        "xtick.major.size": 1.5, "ytick.major.size": 1.5,
        "xtick.labelsize": 4.6, "ytick.labelsize": 4.6})
    from nome_hic_concordance.figure_style import pdf_images_8bit
    pdf_images_8bit()
    return plt


# ------------------------------------------------------------- drawing --

def _panel(fig, rect, letter, title):
    """A rounded panel in figure coordinates with its letter and title; an
    axes over it, 0-10 each way, for the drawing."""
    from matplotlib.patches import FancyBboxPatch
    x0, y0, w, h = rect
    fig.patches.append(FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.012",
        transform=fig.transFigure, fc=PANEL_FC, ec=PANEL_EC, lw=0.6,
        zorder=-10))
    fig.text(x0 - 0.012, y0 + h + 0.004, letter, fontsize=8,
             fontweight="bold", va="bottom", ha="left", color="black")
    fig.text(x0 + 0.012, y0 + h - 0.012, title, fontsize=6.6,
             fontweight="bold", va="top", ha="left", color=INK)
    ax = fig.add_axes([x0, y0, w, h])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.patch.set_visible(False)
    return ax


def _note(ax, y, lines, size=5.1, color=MUTED, x=0.35, gap=0.62):
    """Lines of small text from y downwards, left-aligned."""
    for i, t in enumerate(lines):
        ax.text(x, y - i * gap, t, fontsize=size, color=color, ha="left",
                va="top")


def _segment(ax, x0, x1, y, states, lw=3.0, ms=3.2, mk="o"):
    """A stretch of DNA with sites: filled = methylated/accessible."""
    ax.plot([x0, x1], [y, y], color=SEG, lw=lw, solid_capstyle="butt",
            zorder=1)
    xs = np.linspace(x0 + 0.3, x1 - 0.3, len(states))
    for x_, s in zip(xs, states):
        ax.plot(x_, y, mk, ms=ms, mfc="black" if s else "white",
                mec="black", mew=0.45, zorder=2)
    return xs


def _arc(ax, x0, x1, y, h, color=LINK, ls="-", lw=0.9):
    from matplotlib.patches import Arc
    ax.add_patch(Arc(((x0 + x1) / 2, y), x1 - x0, 2 * h, theta1=0,
                     theta2=180, color=color, ls=ls, lw=lw, zorder=1))


def panel_a(ax):
    l = _segment(ax, 0.6, 3.6, 6.1, (1, 1, 0, 1))
    r = _segment(ax, 6.4, 9.4, 6.1, (1, 0, 1, 1))
    _arc(ax, 2.1, 7.9, 6.45, 1.55)
    ax.text(5.0, 8.2, "contact: the two ends of one read pair",
            fontsize=5.2, color=LINK, ha="center", va="bottom")
    ax.text(2.1, 5.35, "end 1:  $k_1$ of $n_1$ sites", fontsize=5.2,
            ha="center", va="top", color=INK)
    ax.text(7.9, 5.35, "end 2:  $k_2$ of $n_2$ sites", fontsize=5.2,
            ha="center", va="top", color=INK)
    ax.annotate("", xy=(7.9, 4.25), xytext=(2.1, 4.25),
                arrowprops=dict(arrowstyle="<->", lw=0.5, color=MUTED))
    ax.text(5.0, 4.05, "distance $d$", fontsize=5, ha="center", va="top",
            color=MUTED)
    _note(ax, 3.1, ["Each mate is an end: its own GCH (accessibility) or",
                    "HCG (methylation) sites, read on its bisulfite strand.",
                    "One nucleus; the two marks are analysed separately."])


def panel_b(fig, ax, rect):
    from scipy.stats import beta
    x0, y0, w, h = rect
    ia = fig.add_axes([x0 + 0.13 * w, y0 + 0.5 * h, 0.8 * w, 0.31 * h])
    p = np.linspace(0.001, 0.999, 200)
    ia.fill_between(p, beta.pdf(p, 2, 7), color="0.75", alpha=0.6, lw=0)
    ia.plot(p, beta.pdf(p, 2, 7), color="0.35", lw=0.8)
    ia.fill_between(p, beta.pdf(p, 7, 2.2), color="#f2a7a0", alpha=0.6,
                    lw=0)
    ia.plot(p, beta.pdf(p, 7, 2.2), color=HOT, lw=0.8)
    ia.text(0.16, 3.25, "closed /\nunmethylated", fontsize=4.6,
            color="0.3", ha="center", va="bottom", linespacing=0.95)
    ia.text(0.8, 3.25, "open /\nmethylated", fontsize=4.6, color=HOT,
            ha="center", va="bottom", linespacing=0.95)
    ia.set_ylim(0, 4.6)
    ia.set_xlim(0, 1)
    ia.set_yticks([])
    ia.set_xticks([0, 0.5, 1])
    ia.set_xlabel("level of an end's sites", fontsize=4.8, labelpad=1)
    for s in ("top", "right", "left"):
        ia.spines[s].set_visible(False)
    ax.text(5.0, 3.75, "$p$ = P(open or methylated | $k$, $n$)",
            fontsize=5.6, ha="center", va="top", color=INK)
    _note(ax, 2.85, ["Two-component beta-binomial mixture, fitted per mark",
                     "over all cells.  Each end is then centred on its",
                     "cell's mean:  $x = p - m_c$."])


def panel_c(ax):
    y = 4.9
    ax.plot([0.3, 9.7], [y, y], color="0.6", lw=0.6, zorder=0)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((6.05, y - 0.55), 2.3, 1.1, fc="#e8eef7",
                           ec="none", zorder=0))
    ax.text(7.2, y - 0.7, "partner's 250-kb bin", fontsize=4.6,
            color=MUTED, ha="center", va="top")
    # the anchor, the partner and two stand-ins
    for x_, mfc, mec in ((1.3, "black", "black"), (6.5, "black", "black")):
        ax.plot(x_, y, "s", ms=3.6, mfc=mfc, mec=mec, mew=0.5, zorder=3)
    for x_ in (7.35, 7.9):
        ax.plot(x_, y, "s", ms=3.6, mfc="white", mec=NULL, mew=0.6,
                zorder=3)
    _arc(ax, 1.3, 6.5, y + 0.15, 2.4)
    _arc(ax, 1.3, 7.35, y + 0.15, 1.75, color=NULL, ls=(0, (2.2, 1.6)))
    ax.text(3.9, y + 2.65, "linked", fontsize=5, color=LINK, ha="center",
            va="bottom")
    ax.text(4.4, y + 0.95, "non-linked", fontsize=5, color=NULL,
            ha="center", va="bottom")
    ax.text(1.3, y - 0.35, "anchor\n(kept)", fontsize=4.6, ha="center",
            va="top", color=INK, linespacing=0.95)
    # a near-copy: its mate sits on the anchor
    ax.plot(7.9, y, "x", ms=4.2, color=HOT, mew=0.9, zorder=4)
    ax.annotate("", xy=(1.55, y - 0.25), xytext=(7.85, y - 0.25),
                arrowprops=dict(arrowstyle="-", lw=0.5, color=HOT, ls=":",
                                connectionstyle="arc3,rad=0.25"))
    ax.text(8.15, y + 0.42, "near-copy\nrefused", fontsize=4.5, color=HOT,
            ha="left", va="bottom", linespacing=0.95)
    _note(ax, 2.55, ["Partner replaced by an end of another read pair of",
                     "the same cell, in the partner's 250-kb bin and its",
                     "distance bin.  Refused: within 1 kb of the partner,",
                     "or with its own mate within 1 kb of the anchor."])


def _curves(d):
    """Illustrative r by distance: alike below 1 kb, where a pair's ends are
    mostly one molecule; apart beyond 10 kb."""
    lg = np.log10(d)
    short = 0.28 * np.exp(-(lg - 2) / 0.3) + 0.018
    rise = 1 / (1 + np.exp(-(lg - 3.7) / 0.22))
    linked = short + 0.09 * rise
    null = short * 0.93 + 0.014 * rise * np.exp(-(lg - 3.7) / 1.6)
    return linked, null


def panel_d(fig, ax, rect):
    x0, y0, w, h = rect
    ia = fig.add_axes([x0 + 0.17 * w, y0 + 0.5 * h, 0.76 * w, 0.31 * h])
    d = np.logspace(2, 7, 120)
    lk, nl = _curves(d)
    ia.axvspan(1e2, 1e3, color="0.92", lw=0, zorder=0)
    ia.plot(d, lk, color=LINK, lw=0.9, label="linked")
    ia.plot(d, nl, color=NULL, lw=0.8, ls=(0, (2.2, 1.4)),
            label="non-linked")
    k = 95
    ia.annotate("", xy=(d[k], lk[k]), xytext=(d[k], nl[k]),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK))
    ia.text(d[k] * 1.25, (lk[k] + nl[k]) / 2, "Δr", fontsize=5.4,
            va="center", color=INK)
    ia.text(1.15e2, 0.27, "one\nmolecule", fontsize=4.3, color="0.45",
            va="top", linespacing=0.9)
    ia.set_xscale("log")
    ia.set_xlim(1e2, 1e7)
    ia.set_ylim(0, 0.32)
    ia.set_yticks([0, 0.1, 0.2, 0.3])
    ia.set_ylabel("r", fontsize=5, labelpad=1)
    ia.set_xlabel("distance (bp)", fontsize=4.8, labelpad=1)
    for s in ("top", "right"):
        ia.spines[s].set_visible(False)
    ia.legend(frameon=False, fontsize=4.5, loc="upper right",
              handlelength=1.6, borderaxespad=0.1)
    ax.text(5.0, 3.75, "Δr(d) = r(linked) − r(non-linked)", fontsize=5.6,
            ha="center", va="top", color=INK)
    _note(ax, 2.85, ["Pearson r over all cells' pairs in each distance bin.",
                    "95% CI and P from resampling cells (500×),",
                    "Benjamini–Hochberg across bins (Fig. 5a)."])


def panel_e(fig, ax, rect):
    x0, y0, w, h = rect
    ia = fig.add_axes([x0 + 0.13 * w, y0 + 0.5 * h, 0.8 * w, 0.31 * h])
    rng = np.random.default_rng(3)
    ctl = rng.normal(0, 0.012, 1500)
    obs = rng.normal(0.06, 0.02, 1500)
    bins = np.linspace(-0.06, 0.14, 40)
    ia.hist(ctl, bins=bins, color=CTL_C, alpha=0.8, lw=0, label="control")
    ia.hist(obs, bins=bins, histtype="step", color=CELL_C, lw=0.9,
            label="cells")
    ia.axvline(0, color="0.3", lw=0.5, ls="--")
    ia.set_yticks([])
    ia.set_xticks([0, 0.1])
    ia.set_xlabel("per-cell Δr", fontsize=4.8, labelpad=1)
    for s in ("top", "right", "left"):
        ia.spines[s].set_visible(False)
    ia.legend(frameon=False, fontsize=4.5, loc="upper right",
              handlelength=1.2, borderaxespad=0.1)
    _note(ax, 3.75, ["Each cell's own Δr (pairs 1 kb–10 Mb); s.e. from",
                     "resampling 10-Mb blocks within the cell.  Control:",
                     "a second non-linked draw against the first.  Sign test",
                     "across cells; Kruskal–Wallis across types (Fig. 5c,d)."])


def panel_f(fig, ax, rect):
    x0, y0, w, h = rect
    rng = np.random.default_rng(7)
    n = 28
    base = np.cumsum(rng.normal(0, 0.6, n))
    gch = 0.08 + 0.012 * (base - base.mean()) / base.std()
    hcg = 0.02 + 0.6 * (gch - 0.08) + rng.normal(0, 0.004, n)
    for k, (v, col, lab) in enumerate(((gch, "#1f77b4", "GCH Δr"),
                                        (hcg, "#d62728", "HCG Δr"))):
        ia = fig.add_axes([x0 + 0.1 * w, y0 + (0.6 - 0.17 * k) * h,
                           0.62 * w, 0.14 * h])
        ia.bar(np.arange(n), v - v.min() + 0.002, width=0.85, color=col,
               lw=0)
        ia.set_xlim(-0.6, n - 0.4)
        ia.axis("off")
        ia.text(-0.02, 0.5, lab, transform=ia.transAxes, fontsize=4.6,
                ha="right", va="center", color=col)
    # the shift
    ax.annotate("", xy=(8.95, 4.9), xytext=(8.95, 6.9),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=INK,
                                connectionstyle="arc3,rad=-0.9"))
    ax.text(9.25, 5.9, "shift", fontsize=4.6, color=INK, ha="left",
            va="center", rotation=90)
    ax.text(4.2, 4.15, "10-Mb regions along the genome", fontsize=4.6,
            color=MUTED, ha="center", va="top")
    _note(ax, 3.3, ["Pearson r between the two marks across regions.",
                    "P: circular genomic permutation, one profile shifted",
                    "around the genome by ≥50 Mb, keeping each profile's",
                    "own runs of similar regions (Fig. 5b)."])


def panel_g(ax):
    # a contact between two classed ends, and a stand-in of the same class
    y = 7.35
    l = _segment(ax, 0.5, 2.7, y, (1, 1, 0))
    r = _segment(ax, 4.5, 6.7, y, (1, 0, 1))
    _arc(ax, 1.6, 5.6, y + 0.3, 0.95)
    ax.text(1.6, y - 0.45, "class $K_1$", fontsize=4.6, ha="center",
            va="top", color=MUTED)
    ax.text(5.6, y - 0.45, "class $K_2$", fontsize=4.6, ha="center",
            va="top", color=MUTED)
    ax.text(8.45, y + 0.55, "A/B · replication\ntiming · PMD/HMD\n"
            "· chromatin state", fontsize=4.4, ha="center", va="center",
            color=MUTED, linespacing=1.0)
    ax.text(0.35, 5.75, r"$x = a - \mu_{c,K_1}$,   $y = p - \mu_{c,K_2}$",
            fontsize=5.6, va="center", color=INK)
    ax.text(0.35, 4.7, r"$C = \dfrac{2}{N}\sum_i x_i y_i$,"
            r"   $\Delta C = C - C_{\rm null}$", fontsize=5.6,
            va="center", color=INK)
    ax.text(0.35, 3.35, r"$F = \dfrac{\sum ap}{\sum ap + \sum(1-a)(1-p)}$",
            fontsize=5.4, va="center", color=INK)
    ax.text(5.55, 3.35, r"$w = \dfrac{F - F_{\rm ref}}"
            r"{F + F_{\rm ref} - 2FF_{\rm ref}}$", fontsize=5.4,
            va="center", color=INK)
    _note(ax, 2.3, ["Stand-ins from the same cell and class.  ΔC × w > 0:",
                     "agreeing ends more often both open (GCH) or both",
                     "methylated (HCG) than genome-wide; < 0: both closed",
                     "or both unmethylated (Fig. 5e,f)."])


def panel_h(fig, ax, rect):
    x0, y0, w, h = rect
    ia = fig.add_axes([x0 + 0.08 * w, y0 + 0.44 * h, 0.86 * w, 0.33 * h])
    rng = np.random.default_rng(11)
    n = 160
    v = 0.8 + rng.normal(0, 0.35, n)
    hot = rng.choice(n, 9, replace=False)
    v[hot] += rng.uniform(1.4, 2.4, hot.size)
    cold = rng.choice(np.setdiff1d(np.arange(n), hot), 1)
    v[cold] -= 1.6
    col = np.array(["0.55"] * n, dtype=object)
    col[hot] = HOT
    col[cold] = COLD
    ia.scatter(np.arange(n), v, s=1.6, c=list(col), lw=0)
    ia.axhline(np.median(v), color="0.3", lw=0.5, ls="--")
    ia.text(n + 1, np.median(v), "median\nbin", fontsize=4.2, va="center",
            ha="left", color="0.35", linespacing=0.9)
    ia.set_xlim(-2, n + 2)
    ia.set_xticks([])
    ia.set_yticks([])
    ia.set_ylabel("ΔC", fontsize=4.8, labelpad=1)
    ia.set_xlabel("1-Mb bins along the genome", fontsize=4.6, labelpad=1)
    for s in ("top", "right"):
        ia.spines[s].set_visible(False)
    _note(ax, 3.45, ["Hot and cold bins: z against the median bin, null",
                     "fitted to the control map, 5% FDR.  What they hold:",
                     "each annotation's share in hot over all tested bins,",
                     "P by circular genomic permutation (Fig. 5g)."])


def panel_i(ax):
    for y, col, lab in ((7.3, XI, "Xi"), (5.2, XA, "Xa")):
        ax.plot([0.9, 8.7], [y, y], color=col, lw=2.2, alpha=0.85,
                solid_capstyle="butt")
        ax.text(0.6, y, lab, fontsize=5.4, color=col, ha="right",
                va="center", fontweight="bold")
        _arc(ax, 2.0, 5.4, y + 0.2, 1.0, color=col)
        _arc(ax, 2.0, 6.0, y + 0.2, 0.7, color=col, ls=(0, (2.2, 1.6)))
        for x_ in (2.0, 5.4):
            ax.plot(x_, y, "o", ms=2.8, mfc="black", mec="black", mew=0.4)
        ax.plot(6.0, y, "o", ms=2.8, mfc="white", mec=col, mew=0.6)
    ax.text(8.9, 6.25, "stand-ins\nfrom the same\ncell and copy",
            fontsize=4.5, ha="right", va="center", color=MUTED,
            linespacing=1.0)
    ax.text(0.35, 3.85, r"pooled copies:  excess $= \frac{1}{2}\,\delta_a"
            r"\,\delta_p$", fontsize=5.3, va="center", color=INK)
    _note(ax, 3.0, ["A contact's ends lie on one copy; a pooled stand-in",
                    "comes from either.  δ: Xi − Xa level at each end",
                    "(Fig. 6h).  Read pairs are put on a copy by phased",
                    "SNVs (Fig. 6i)."])


def build_figure():
    plt = _style()
    fig = plt.figure(figsize=(7.2, 7.4))
    cols = (0.035, 0.36, 0.685)
    rows = (0.675, 0.355, 0.035)
    w, h = 0.295, 0.29
    R = {k: (cols[i % 3], rows[i // 3], w, h) for i, k in enumerate("abcdefghi")}
    titles = {"a": "A contact and its ends",
              "b": "End posterior",
              "c": "Non-linked pairs from the same cell",
              "d": "Concordance by distance",
              "e": "Concordance in each cell",
              "f": "Concordance across regions",
              "g": "Concordance by chromatin class",
              "h": "Concordance along the genome",
              "i": "Concordance on each X copy"}
    ax = {k: _panel(fig, R[k], k, titles[k]) for k in "abcdefghi"}
    panel_a(ax["a"])
    panel_b(fig, ax["b"], R["b"])
    panel_c(ax["c"])
    panel_d(fig, ax["d"], R["d"])
    panel_e(fig, ax["e"], R["e"])
    panel_f(fig, ax["f"], R["f"])
    panel_g(ax["g"])
    panel_h(fig, ax["h"], R["h"])
    panel_i(ax["i"])
    # the sites, once for all panels
    fig.text(0.035, 0.012, "●  methylated / accessible     ○  unmethylated / "
             "closed     ━  linked (a contact)     ┅  non-linked (a stand-in)",
             fontsize=5.2, color=MUTED, ha="left", va="bottom")
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    fig = build_figure()
    import matplotlib.pyplot as plt
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
