#!/usr/bin/env python3
"""Figure 5: long-range concordance in mouse cortex, from the runs' tables.

The panels:

  a  the end posteriors' correlation by distance, linked pairs against
     non-linked ones -- another end of the same cell at the partner's
     place and distance -- and their difference with its interval, for
     GCH-GCH, HCG-HCG and HCG-GCH, over the bins where a tenth of the
     linked pairs or more found one (end_posterior_dist_all.tsv); below
     1 kb, shaded, a pair's two ends are mostly one molecule
  b  per 10-Mb region, HCG-HCG against GCH-GCH concordance
     (end_posterior_regions.tsv, _summary.tsv)

a and b come from --distance-ab where given: a run whose null keeps every
stand-in, so a reaches down to 100 bp, where a null excluding the contact's
near-copies (--null-exclude) finds none.  --supplementary then draws
--distance's own a and b, that stricter null's, as a figure of their own.
  c  per cell, concordance against the negative control
     (end_posterior_percell.tsv)
  d  the same by cell type
  e  concordance by class pair -- A/B compartments, replication timing,
     PMD / HMD: ΔC x w, ΔC against reshuffled links signed by where the
     concordant mass sits against the cell's genome (summary.tsv of
     annotation_class_concordance.py)
  f  ΔC x w between chromHMM states
  g  concordance along the genome per 1-Mb bin, its hot bins, and what
     the hot bins are -- each annotation the run tested, its share in
     them over all tested bins, with p against genome rotations
     (locus_hotspots.tsv.gz)

The cells' own values only, nothing imputed: the imputed matrices' cell-type
signal is reproduced by links with no coordination (design §6b.6).

    python scripts/figure5_cortex.py \\
        --distance end_posterior_distance_correlation/end_posterior_dist.null_link_shuffled_match_per_cell.cell_center.excl1kb \\
        --distance-ab end_posterior_distance_correlation/null_link_shuffled_match_per_cell.cell_center \\
        --classes annotation_class_concordance_v14_samecell_excl \\
        --output Figure5.pdf --supplementary FigureS5_near_copies.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance.percell_existence import (  # noqa: E402
    MARK_COLOR, NULL_GREY, fmt_p, sign_test)

#: The marks: (key in the distance table, mark, label).
MARKS = (("gg", "GCH", "GCH accessibility"),
         ("hh", "HCG", "HCG methylation"))
#: Panel a's measures: (key in the distance table, title, colour).
GH_C = "#2ca02c"
A_MEASURES = (("gg", "GCH–GCH: accessibility", MARK_COLOR["GCH"]),
              ("hh", "HCG–HCG: methylation", MARK_COLOR["HCG"]),
              ("gh", "HCG–GCH: methylation × accessibility", GH_C))
#: Cell types in their order, and their colours where the samples table
#: gives none.
CELL_TYPES = {"IT-L2/3": "#f19f5c", "IT-L4": "#d75846", "IT-L5/6": "#834d35",
              "CT-L6": "#aa3d28", "Inhib": "#b659ac", "ODC": "#7dc0a6",
              "OPC": "#4b9c7a", "ASC": "#63a1f8", "MGC": "#f9da56",
              "PC/EC": "#b0d767"}
#: Panel e's class pairs: (system, anchor, partner, label); a rule between
#: the systems.
E_PAIRS = (("ab", "A", "A", "A–A"), ("ab", "A", "B", "A–B"),
           ("ab", "B", "B", "B–B"),
           ("rt", "early", "early", "early–early"),
           ("rt", "early", "late", "early–late"),
           ("rt", "late", "late", "late–late"),
           ("pmd", "HMD", "HMD", "HMD–HMD"), ("pmd", "HMD", "PMD", "HMD–PMD"),
           ("pmd", "PMD", "PMD", "PMD–PMD"))
#: Panel f's states, active to silent.
STATES = ("promoter", "enhancer", "transcribed", "polycomb",
          "heterochromatin", "quiescent")
#: What + and - of ΔC x w mean, per mark.
SIGN_WORDS = {"GCH": ("co-closed", "co-open"),
              "HCG": ("co-unmethylated", "co-methylated")}
#: Panel g's annotations, (track, label), in groups -- domains, chromHMM
#: states, gene density, gene families; any other track the run tested (a
#: --hotspot-annotation BED) follows them under its own name.
TSS_TRACK = "TSSs (count)"
G_GROUPS = (
    (("compartment: A", "A compartment"),
     ("compartment: B", "B compartment"),
     ("replication timing: early", "early replicating"),
     ("replication timing: late", "late replicating"),
     ("PMD / HMD: HMD", "HMD"),
     ("PMD / HMD: PMD", "PMD"),
     ("PMD borders", "PMD borders")),
    tuple((f"chromHMM: {st}", st) for st in STATES),
    ((TSS_TRACK, "TSS density"),
     ("CpG islands", "CpG islands")),
    (("zinc-finger (Zfp / ZNF) genes", "Zfp genes"),
     ("olfactory receptor genes", "olfactory receptors"),
     ("protocadherin cluster", "protocadherin cluster"),
     ("imprinted genes", "imprinted genes")))
G_TRACKS = tuple(t for g in G_GROUPS for t in g)
#: The hot-bin table's own columns; every other one is a track the run
#: tested the bins against.
HOT_FIELDS = frozenset({
    "context", "system", "chrom", "start", "end", "band", "contacts",
    "cells", "delta", "control", "se", "se_control", "z", "z_control",
    "tested", "excess", "z_excess", "q_hot", "q_cold", "hot", "cold",
    "grid_index", "code", "band_index", "x"})
#: Panel a draws a bin where this share of its linked pairs or more found
#: a stand-in: below it, a few pairs, and those the ones with a stand-in.
MIN_MATCHED = 0.1
#: Below it, shaded in a, most read pairs are the two ends of one molecule
#: (an unligated fragment), not a contact.
ONE_MOLECULE_BP = 1000
#: The whole distance range of the class tables.
ALL_BAND = "all 1kb-10Mb"
#: Panel e's dot, as a share of an axis's range (2.2 pt of ~56): intervals
#: narrower than it hide behind it.
DOT_FRACTION = 0.05
HOT_C, COLD_C = "#c0392b", "#2166ac"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6, "axes.titlesize": 6.2, "axes.labelsize": 6,
        "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
        "legend.fontsize": 5.2, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "lines.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.default": "regular",
        "axes.spines.top": False, "axes.spines.right": False})
    from nome_hic_concordance.figure_style import pdf_images_8bit
    pdf_images_8bit()
    return plt


def _zero(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color="0.7", lw=0.4,
                                                zorder=0)


def _p(p) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ----------------------------------------------------------------- panels --

#: The schematic's sites per mark: (marker, the set state's colour, the
#: unset one's, their names).
SITE_STYLE = {"GCH": ("s", "#1a9850", "white", "accessible", "inaccessible"),
              "HCG": ("o", "black", "white", "methylated", "unmethylated")}
SEGMENT_C = "#7cc5ea"


def _reshuffle_schematic(ax, mark: str):
    """Inset: three linked pairs, their two ends alike in ``mark``, against
    the same ends shuffled -- each anchor with another pair's partner, at
    the same distance and in the same cell -- their ends unalike."""
    mk, on, off, lab_on, lab_off = SITE_STYLE[mark]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def segment(x0, y, states):
        xs = x0 + 0.45 + 0.75 * np.arange(len(states))
        ax.plot([x0, xs[-1] + 0.45], [y, y], color=SEGMENT_C, lw=2.6,
                solid_capstyle="butt", zorder=1)
        for x_, st in zip(xs, states):
            ax.plot(x_, y, mk, ms=3.0, mfc=on if st else off, mec="black",
                    mew=0.4, zorder=2)
        return xs[-1] + 0.45

    def tie(x0, y0, x1, y1):
        ax.plot([x0, x1], [y0, y1], ls=(0, (1, 1.2)), color="0.4", lw=0.45,
                zorder=0)

    def d0(x_, y):
        ax.text(x_, y, "$d$ = $d_0$", ha="center", va="center", fontsize=4.4,
                zorder=3, bbox=dict(fc="white", ec="none", pad=0.2))

    ax.text(3.4, 10.0, "Linked pairs", ha="center", va="top", fontsize=4.8)
    # three contacts, each pair's ends alike
    for y, lx, ls_, rx, rs in ((8.4, 0.4, (1, 1), 4.2, (1, 1, 1)),
                               (7.4, 0.1, (0, 0), 4.6, (0, 0)),
                               (6.4, 0.4, (1, 1), 4.2, (1, 1, 1))):
        tie(segment(lx, y, ls_), y, rx, y)
        segment(rx, y, rs)
    d0(3.15, 7.4)
    for y, st, lab in ((8.4, 0, lab_off), (7.4, 1, lab_on)):
        ax.plot(7.3, y, mk, ms=3.0, mfc=on if st else off, mec="black",
                mew=0.4)
        ax.text(7.7, y, lab, ha="left", va="center", fontsize=4.6)
    ax.text(0.9, 5.8, "vs.", ha="center", va="center", fontsize=4.6,
            style="italic")
    ax.text(5.0, 4.75, "Shuffled pairs (matched distance, same cell)",
            ha="center", va="center", fontsize=4.8)
    # the same anchors, each with another pair's partner
    rows = (3.0, 2.0, 1.0)
    left = [segment(x0, y, st) for x0, y, st in
            ((0.6, rows[0], (1, 0)), (0.3, rows[1], (1, 0)),
             (0.6, rows[2], (0, 1)))]
    rx = (4.4, 4.8, 4.4)
    for (x0, st), y in zip(((rx[0], (0, 1, 1)), (rx[1], (0, 0)),
                            (rx[2], (0, 1, 0))), rows):
        segment(x0, y, st)
    for i, j in enumerate((2, 0, 1)):
        tie(left[i], rows[i], rx[j], rows[j])
    d0(3.35, 3.65)


def matched_bins(dist: pd.DataFrame, k: str,
                 min_share: float = MIN_MATCHED) -> np.ndarray:
    """The distance bins where measure ``k`` has both correlations and
    ``min_share`` of its linked pairs or more were matched."""
    lk, mt = (dist[f"{k}_{c}_n"].to_numpy(float) for c in ("linked", "matched"))
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(lk > 0, mt / lk, 0.0)
    return (share >= min_share) \
        & np.isfinite(dist[f"{k}_matched_r"].to_numpy(float)) \
        & np.isfinite(dist[f"{k}_null_r"].to_numpy(float))


def panel_a(axes_r, axes_d, dist: pd.DataFrame):
    """``axes_r``, ``axes_d``: per measure of A_MEASURES, the correlations
    and their difference, over its matched_bins."""
    x = dist["center_bp"].to_numpy(float)
    have = np.zeros(x.size, bool)
    for k, *_ in A_MEASURES:
        have |= matched_bins(dist, k)
    span = (x[have].min() / 1.3, x[have].max() * 1.3) if have.any() else \
        (x.min() / 1.3, x.max() * 1.3)
    for j, (k, title, col) in enumerate(A_MEASURES):
        ax, axd = axes_r[j], axes_d[j]
        keep = matched_bins(dist, k)
        lk, nl = (np.where(keep, dist[f"{k}_{c}_r"].to_numpy(float), np.nan)
                  for c in ("matched", "null"))
        ax.plot(x, lk, "-o", color=col, ms=1.6, lw=0.9, label="linked")
        ax.plot(x, nl, "--o", color=col, ms=1.6, mfc="white", mew=0.5,
                lw=0.8, alpha=0.9, label="non-linked")
        _zero(ax)
        ax.set_title(title, loc="left", color=col)
        d = lk - nl
        axd.fill_between(x, np.where(keep, dist[f"{k}_delta_lo"], np.nan),
                         np.where(keep, dist[f"{k}_delta_hi"], np.nan),
                         color=col, alpha=0.2, lw=0)
        axd.plot(x, d, "-o", color=col, ms=1.6, lw=0.9)
        q = dist[f"{k}_delta_q"].to_numpy(float)
        sig = keep & np.isfinite(q) & (q < 0.05)
        n_ok = int(keep.sum())
        axd.plot(x[sig], d[sig], "o", ms=3.6, mfc="none", mec=col, mew=0.5,
                 label=f"q < 0.05 ({int(sig.sum())} of {n_ok} bins)")
        _zero(axd)
        for a in (ax, axd):
            a.set_xscale("log")
            a.set_xlim(*span)
            if span[0] < ONE_MOLECULE_BP:
                a.axvspan(span[0], ONE_MOLECULE_BP, color="0.93", lw=0,
                          zorder=0)
        ax.tick_params(labelbottom=False)
        axd.set_xlabel("distance between the two ends (bp)")
        axd.legend(frameon=False, loc="best", handletextpad=0.2)
    axes_r[0].set_ylabel("Pearson r of end posteriors")
    axes_d[0].set_ylabel("Δr, linked − non-linked")
    if span[0] < ONE_MOLECULE_BP:
        # the short pairs' correlation is the molecule's own: the curves
        # start high on the left and the long range sits low, leaving the
        # top right to the schematics and the gap between the curves on
        # the right to the keys
        axes_d[0].text(0.012, 0.97, "one\nmolecule", fontsize=4.6,
                       color="0.4", transform=axes_d[0].transAxes,
                       ha="left", va="top")
        for ax, (k, *_), at in zip(axes_r, A_MEASURES,
                                   ((1.0, 0.31), (1.0, 0.36), None)):
            if at is None:
                ax.legend(frameon=False, loc="lower right", handlelength=1.8)
            else:
                ax.legend(frameon=False, loc="upper right", handlelength=1.8,
                          bbox_to_anchor=at)
        _reshuffle_schematic(axes_r[0].inset_axes([0.17, 0.36, 0.83, 0.64]),
                             "GCH")
        _reshuffle_schematic(axes_r[1].inset_axes([0.25, 0.4, 0.75, 0.6]),
                             "HCG")
    else:
        # between the curves; the main figure's a shows the shuffle
        axes_r[0].legend(frameon=False, loc="center left", handlelength=1.8,
                         bbox_to_anchor=(0.0, 0.45))


def region_table(regions: pd.DataFrame, summ: pd.Series,
                 min_pairs: int = 500) -> pd.DataFrame:
    """Per region of the summary's distance band, with both marks at
    ``min_pairs`` observed and null pairs or more (as the run's summary
    took them): Δ and the control's Δ, GCH and HCG."""
    d = regions[(regions["band_lo"] == summ["band_lo"])
                & (regions["band_hi"] == summ["band_hi"])
                & (regions["n_obs"] >= min_pairs)
                & (regions["n_null"] >= min_pairs)]
    key = ["chrom", "start", "end"]
    g = d[d["mark"] == "GCH"].set_index(key)
    h = d[d["mark"] == "HCG"].set_index(key)
    return g[["delta", "delta_ctl"]].join(
        h[["delta", "delta_ctl"]], lsuffix="_gch", rsuffix="_hcg",
        how="inner").dropna()


def _scatter(ax, x, y, cmap="Blues", sparse_frac=0.1):
    """As the run draws it, smoothScatter-style: the density on a fourth-
    root scale, the sparsest ``sparse_frac`` of points, least squares with
    its 95% band and Theil-Sen."""
    from scipy.stats import gaussian_kde, t as tdist, theilslopes
    x, y = np.asarray(x, float), np.asarray(y, float)
    px, py = 0.06 * (np.ptp(x) or 1.0), 0.06 * (np.ptp(y) or 1.0)
    x0, x1, y0, y1 = x.min() - px, x.max() + px, y.min() - py, y.max() + py
    kde = gaussian_kde(np.vstack([x, y]))
    gx, gy = np.mgrid[x0:x1:160j, y0:y1:160j]
    z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.imshow((z / z.max()).T ** 0.25, origin="lower", aspect="auto",
              extent=[x0, x1, y0, y1], cmap=cmap, interpolation="bilinear",
              rasterized=True)
    dens = kde(np.vstack([x, y]))
    sp = dens <= np.quantile(dens, sparse_frac)
    ax.scatter(x[sp], y[sp], s=1.5, color="black", lw=0, zorder=3)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x0, x1, 100)
    n, xm = x.size, x.mean()
    s = np.sqrt(np.sum((y - (a + b * x)) ** 2) / max(n - 2, 1))
    half = (tdist.ppf(0.975, max(n - 2, 1)) * s
            * np.sqrt(1 / n + (xs - xm) ** 2 / np.sum((x - xm) ** 2)))
    ax.fill_between(xs, a + b * xs - half, a + b * xs + half,
                    color="#d62728", alpha=0.18, lw=0, zorder=4)
    ax.plot(xs, a + b * xs, color="#d62728", lw=0.8, zorder=5,
            label=f"least squares, slope {b:+.2f}")
    ts = theilslopes(y, x)
    ax.plot(xs, ts[1] + ts[0] * xs, color="#d62728", lw=0.7, ls="--",
            zorder=5, label=f"Theil–Sen, slope {ts[0]:+.2f}")
    for f_ in (ax.axhline, ax.axvline):
        f_(0, color="#9AA0A6", lw=0.4, zorder=2)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def panel_b(ax, regions: pd.DataFrame, summ: pd.Series):
    """Per region, HCG-HCG against GCH-GCH Δr."""
    t = region_table(regions, summ)
    _scatter(ax, t["delta_gch"], t["delta_hcg"])
    ax.set_box_aspect(1)
    ax.set_title(f"r = {summ['pearson_r']:+.2f}, ρ = "
                 f"{summ['spearman_rho']:+.2f}\ngenome rotation "
                 f"{_p(summ['rotation_p'])}", loc="left")
    ax.set_ylabel("HCG–HCG Δr")
    ax.set_xlabel(f"GCH–GCH Δr\n({len(t)} regions of 10 Mb)")
    ax.legend(frameon=False, loc="upper left", fontsize=4.4,
              handlelength=1.6, borderaxespad=0.1)


def panel_c(axes, percell: pd.DataFrame):
    for ax, (_, mark, label) in zip(axes, MARKS):
        d = percell[(percell["mark"] == mark)
                    & np.isfinite(percell["delta"])]
        col = MARK_COLOR[mark]
        obs, ctl = sign_test(d["delta"]), sign_test(d["delta_null"])
        both = np.concatenate([d["delta"], d["delta_null"].dropna()])
        lo, hi = np.percentile(both, [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        ax.hist(d["delta_null"].dropna(), bins=bins, density=True,
                histtype="stepfilled", color=NULL_GREY, alpha=0.45, lw=0)
        ax.hist(d["delta"], bins=bins, density=True, histtype="stepfilled",
                color=col, alpha=0.3, lw=0)
        ax.hist(d["delta"], bins=bins, density=True, histtype="step",
                color=col, lw=0.7)
        ax.axvline(0, color="0.25", lw=0.5, ls="--")
        # two short lines, one long one runs into d; spaced in points, as
        # the p's exponent rises into the line above
        ax.annotate(f"{100 * obs['frac']:.0f}% of {obs['n']:,} cells Δ > 0",
                    xy=(0, 1), xycoords="axes fraction", xytext=(0, 10.5),
                    textcoords="offset points", ha="left", va="bottom",
                    color=col, fontsize=5.4, fontweight="bold")
        ax.annotate(f"{fmt_p(obs['p'])}; control (grey) "
                    f"{100 * ctl['frac']:.0f}%",
                    xy=(0, 1), xycoords="axes fraction", xytext=(0, 1.5),
                    textcoords="offset points", ha="left", va="bottom",
                    color="0.4", fontsize=5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylabel(label, color=col)
    axes[-1].set_xlabel("per-cell Δr (contacts − null)")


def panel_d(axes, percell: pd.DataFrame, order, colors):
    rng = np.random.default_rng(1)
    for r, (ax, (_, mark, label)) in enumerate(zip(axes, MARKS)):
        d = percell[(percell["mark"] == mark)
                    & np.isfinite(percell["delta"])]
        present = [g for g in order if (d["group"] == g).any()]
        ctl = d["delta_null"].dropna().to_numpy()
        if ctl.size:
            lo, hi = np.percentile(ctl, [2.5, 97.5])
            ax.axhspan(lo, hi, color=NULL_GREY, alpha=0.25, lw=0, zorder=0)
            # inside, past the last type: beside the axes it meets e's rows
            ax.text(len(present) - 0.35, (lo + hi) / 2, "control\n95%",
                    ha="left", va="center", fontsize=4.6, color="0.4",
                    linespacing=1)
        ax.axhline(0, color="0.55", lw=0.4, ls="--", zorder=0)
        data = [d.loc[d["group"] == g, "delta"].to_numpy() for g in present]
        bp = ax.boxplot(data, positions=range(len(present)), widths=0.55,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color="0.1", lw=0.8),
                        whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                        boxprops=dict(lw=0.5))
        for i, (g, patch) in enumerate(zip(present, bp["boxes"])):
            c = colors.get(g, "0.5")
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
            patch.set_edgecolor(c)
            v = data[i]
            ax.scatter(i + rng.uniform(-0.2, 0.2, v.size), v, s=0.8,
                       color=c, alpha=0.4, lw=0, zorder=1, rasterized=True)
            ax.text(i, 1.0, f"{100 * (v > 0).mean():.0f}%",
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=4.8, color="0.3")
        ax.set_xlim(-0.6, len(present) + 0.45)
        ax.set_ylabel(f"{mark}: per-cell Δr", color=MARK_COLOR[mark])
        ax.set_xticks(range(len(present)))
        if r < len(axes) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticklabels(present, rotation=35, ha="right")
            for lab, g in zip(ax.get_xticklabels(), present):
                lab.set_color(colors.get(g, "0.2"))


def _pairs(summ: pd.DataFrame, context: str, rows) -> pd.DataFrame:
    """Each (system, anchor, partner) of ``rows`` over all distances, in
    order, either orientation of the pair."""
    s = summ[(summ["context"] == context) & (summ["band"] == ALL_BAND)]
    out = []
    for sy, a_, b_, *_ in rows:
        m = s[(s["system"] == sy) & (((s["anchor"] == a_)
                                      & (s["partner"] == b_))
                                     | ((s["anchor"] == b_)
                                        & (s["partner"] == a_)))]
        out.append(m.iloc[0] if len(m) else pd.Series(dtype=float))
    return pd.DataFrame(out).reset_index(drop=True)


def panel_e(axes, summ: pd.DataFrame):
    """``axes``: per mark, ΔC x w by class pair (rows E_PAIRS), the
    reshuffled links' beneath each."""
    from matplotlib.ticker import MaxNLocator
    y = np.arange(len(E_PAIRS))[::-1].astype(float)
    narrow = []
    for ax, (_, mark, _) in zip(axes, MARKS):
        t = _pairs(summ, mark, E_PAIRS)
        col = MARK_COLOR[mark]
        for stat, dy, fmt, ms, c_, ew in (
                ("signed_rel", 0.13, "o", 2.2, col, 0.6),
                ("signed_rel_control", -0.13, "D", 1.6, NULL_GREY, 0.5)):
            v, lo, hi = (100 * t[c].to_numpy(float)
                         for c in (stat, f"{stat}_lo", f"{stat}_hi"))
            ax.errorbar(v, y + dy, xerr=[v - lo, hi - v], fmt=fmt, ms=ms,
                        color=c_, elinewidth=ew, capsize=0)
        _zero(ax, "x")
        for b in (2.5, 5.5):
            ax.axhline(b, color="0.85", lw=0.4, zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel("ΔC × w (pp)")
        # what each side means, at its end, beneath the label
        neg, pos = SIGN_WORDS[mark]
        for x_, ha, word in ((0.0, "left", f"← {neg}"),
                             (1.0, "right", f"{pos} →")):
            ax.annotate(word, xy=(x_, 0), xycoords="axes fraction",
                        xytext=(0, -23), textcoords="offset points", ha=ha,
                        va="top", fontsize=4.8, color="0.3")
        ax.set_title(label_of(mark), loc="left", color=col)
        # millions of contacts: the intervals can be narrower than the dots
        # -- say so, or they read as missing
        x0, x1 = ax.get_xlim()
        w_ = 100 * (t["signed_rel_hi"] - t["signed_rel_lo"]).to_numpy(float)
        if np.nanmedian(w_) / (x1 - x0) < DOT_FRACTION:
            narrow.append(mark)
    # the axes share y: the labels once, on the first
    axes[0].set_ylim(-0.6, len(E_PAIRS) - 0.4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for *_, lab in E_PAIRS])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    from matplotlib.lines import Line2D
    # beneath the axis labels: the dots fill both sides of every axis
    ci = "95% CI" + (f"; {' and '.join(narrow)}'s within the dots"
                     if narrow else "")
    axes[0].legend([Line2D([], [], marker="o", ms=2.2, color="0.3", ls=""),
                    Line2D([], [], marker="D", ms=1.6, color=NULL_GREY,
                           ls="")],
                   [f"contacts ({ci})", "reshuffled links"],
                   frameon=False, loc="upper left", fontsize=4.8, ncol=2,
                   handletextpad=0.1, columnspacing=1.2,
                   bbox_to_anchor=(0.0, -0.25), borderaxespad=0.0)


def label_of(mark: str) -> str:
    return dict((m, lab) for _, m, lab in MARKS)[mark]


def chromhmm_matrix(summ: pd.DataFrame, context: str, stat="signed_rel"):
    """ΔC x w between STATES (pp, lower triangle; NaN where sparse) and
    whether its interval spans 0."""
    s = summ[(summ["context"] == context) & (summ["system"] == "chromhmm")
             & (summ["band"] == ALL_BAND)]
    n = len(STATES)
    v = np.full((n, n), np.nan)
    spans = np.zeros((n, n), bool)
    for i, a_ in enumerate(STATES):
        for j, b_ in enumerate(STATES[:i + 1]):
            m = s[((s["anchor"] == a_) & (s["partner"] == b_))
                  | ((s["anchor"] == b_) & (s["partner"] == a_))]
            if not len(m) or bool(m["sparse"].iloc[0]):
                continue
            r = m.iloc[0]
            v[i, j] = 100 * r[stat]
            spans[i, j] = r[f"{stat}_lo"] <= 0 <= r[f"{stat}_hi"]
    return v, spans


def _limit(v) -> float:
    """A colour scale's half-range: the largest magnitude, or the next one
    where the largest is over twice it -- one pair (HCG's promoters) would
    otherwise wash out the rest; its number is printed."""
    a = np.sort(np.abs(v[np.isfinite(v)]))[::-1]
    if not a.size:
        return 1.0
    return float(a[1] * 1.1) if a.size > 1 and a[0] > 2 * a[1] else float(a[0])


def panel_f(axes, summ: pd.DataFrame):
    """Per mark, ΔC x w between the states: each row's state on the left,
    each column's at its head, along the diagonal.  The cells are squares
    drawn one by one, not an image: a few-colour image is stored at a few
    bits per sample, which some readers draw as one colour."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    cmap = plt.get_cmap("RdBu_r")
    for ax, (_, mark, _) in zip(axes, MARKS):
        v, spans = chromhmm_matrix(summ, mark)
        lim = _limit(v)
        norm = Normalize(-lim, lim)
        n = len(STATES)
        for i in range(n):
            for j in range(i + 1):
                if not np.isfinite(v[i, j]):
                    continue
                # beyond the scale, its end colour
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, lw=0,
                                       facecolor=cmap(norm(v[i, j])),
                                       zorder=1))
                dark = abs(v[i, j]) > 0.6 * lim
                ax.text(j, i, f"{v[i, j]:.2f}".replace("-", "−"),
                        ha="center", va="center", fontsize=4.2,
                        color="white" if dark else "0.15",
                        alpha=0.4 if spans[i, j] else 1.0)
            ax.text(i - 0.25, i - 0.62, STATES[i], rotation=40, ha="left",
                    va="bottom", fontsize=4.8, rotation_mode="anchor")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks(range(n))
        # the rows' states once, on the first; the heads name the columns
        ax.set_yticklabels(STATES if ax is axes[0] else [], fontsize=4.8)
        ax.tick_params(length=0)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        # the last head runs past the axes: no background to cover it
        ax.patch.set_visible(False)
        ax.text(1.0, 1.04, label_of(mark), transform=ax.transAxes,
                ha="right", va="top", color=MARK_COLOR[mark], fontsize=6)
        cax = ax.inset_axes([0.42, -0.1, 0.58, 0.045])
        cb = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                                cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=4.4, length=1.5, pad=1)
        cb.set_label(f"ΔC × w (pp), + {SIGN_WORDS[mark][1]}", fontsize=4.6,
                     labelpad=1)


def _genome_x(hot: pd.DataFrame):
    """Each bin's position along the genome, chromosomes laid end to end in
    order, and where each starts."""
    key = hot["chrom"].str.replace("chr", "", regex=False)
    order = sorted(key.unique(), key=lambda c: (not c.isdigit(),
                                                int(c) if c.isdigit() else c))
    ends = hot.groupby(key)["end"].max()
    off = dict(zip(order, np.r_[0, np.cumsum([ends[c] for c in order])[:-1]]))
    return key.map(off) + (hot["start"] + hot["end"]) / 2, order, off, ends


def fdr_bh(p) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size:
        o = ok[np.argsort(p[ok])]
        v = p[o] * o.size / np.arange(1, o.size + 1)
        q[o] = np.minimum(np.minimum.accumulate(v[::-1])[::-1], 1.0)
    return q


def _rotation_p(h, x, min_shift: int = 50) -> float:
    """Two-sided p of where the hot bins ``h`` sit on a track ``x``, both
    in genome order -- their covariance, which moves with the fold --
    against every circular shift of the labels by ``min_shift`` bins or
    more (a tenth of the series when that is fewer): neighbouring bins are
    not independent, and a shift keeps runs of hot bins and the track's
    own stretches while breaking their alignment.  Twice the nearer tail,
    as a sparse track's null is lopsided.  NaN when too few shifts remain
    to reach p < 0.05."""
    h, x = np.asarray(h, float), np.asarray(x, float)
    n = h.size
    ms = max(1, min(min_shift, n // 10))
    if n - 2 * ms + 1 < 39 or h.std() == 0 or x.std() == 0:
        return np.nan
    a, b = h - h.mean(), x - x.mean()
    # every circular shift at once: cc[s] = sum_i a[i] b[i + s]
    cc = np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(b), n)
    null = cc[ms:n - ms + 1]
    tol = 1e-9 * max(1.0, abs(cc[0]))
    tail = min(np.sum(null >= cc[0] - tol), np.sum(null <= cc[0] + tol))
    return float(min(1.0, 2 * (1 + tail) / (1 + null.size)))


def tested_tracks(hot: pd.DataFrame) -> list:
    """The tracks the run tested the bins against: every numeric column of
    its hot-bin table but the table's own (HOT_FIELDS), --hotspot-annotation
    BEDs among them."""
    return [c for c in hot.columns if c not in HOT_FIELDS
            and pd.api.types.is_numeric_dtype(hot[c])]


def g_rows(tracks) -> tuple:
    """Panel g's rows, one per track the run tested: G_GROUPS' in order,
    then any others under their own names; and where each group starts."""
    tested = list(tracks)
    groups = [[(t, lab) for t, lab in g if t in tested] for g in G_GROUPS]
    groups.append([(t, t) for t in tested if t not in dict(G_TRACKS)])
    groups = [g for g in groups if g]
    return ([r for g in groups for r in g],
            [int(b) for b in np.cumsum([len(g) for g in groups])[:-1]])


def hot_enrichment(t: pd.DataFrame, tracks) -> pd.DataFrame:
    """Per track of ``tracks``, over bins ``t`` (one mark's tested bins, in
    genome order): the track's mean over the hot bins against its mean
    over all of them (fold); the same for as many of the control map's
    most extreme bins (control_fold: by its z against the median bin, as
    the hot calls are made); and the fold's p against genome rotations
    with BH q over ``tracks`` -- every track the run tested, drawn or not.
    Not the run's hot_odds_ratio: whether a bin touches a track at all
    says nothing where a track is in every bin, as heterochromatin is.  A
    track no tested bin touches has no value, as in the run."""
    hot_ = t["hot"].to_numpy(bool)
    k = int(hot_.sum())
    z0 = ((t["control"] - t["control"].median())
          / t["se_control"]).to_numpy(float)
    ctl = np.zeros(len(t), bool)
    ctl[np.argsort(-np.nan_to_num(z0, nan=-np.inf), kind="stable")[:k]] = True
    rows = []
    for name in tracks:
        x = t[name].to_numpy(float)
        m = x.mean() if x.size else np.nan
        if not k or not np.ptp(x) > 0 or not m > 0:
            rows.append({"track": name})
            continue
        rows.append({"track": name, "fold": x[hot_].mean() / m,
                     "control_fold": x[ctl].mean() / m,
                     "p": _rotation_p(hot_, x)})
    out = pd.DataFrame(rows).reindex(
        columns=["track", "fold", "control_fold", "p"])
    out["q"] = fdr_bh(out["p"])
    return out


def panel_g(ax_tracks, ax_enr, hot: pd.DataFrame):
    """``ax_tracks``: per mark, ΔC per bin along the genome; ``ax_enr``:
    what the bins are, the marks side by side per track."""
    every = hot[hot["band"] == ALL_BAND].copy()
    every["x"], order, off, ends = _genome_x(every)
    every = every.sort_values("x")
    h = every[every["tested"].astype(bool)]
    for r, (ax, (_, mark, label)) in enumerate(zip(ax_tracks, MARKS)):
        t = h[h["context"] == mark]
        # bins short of contacts are not tested: ticked along the foot, so
        # gaps read as untested bins, not as wider ones
        un = every[(every["context"] == mark) & ~every["tested"].astype(bool)]
        v = 100 * t["delta"].to_numpy(float)
        top = np.sort(v[np.isfinite(v)])[::-1]
        hi = 1.25 * top[0]
        # one bin far above the rest would flatten them: it sits at the top
        # edge, its value printed
        if top.size > 1 and top[0] > 2 * top[1]:
            hi = 1.35 * top[1]
        far = v > hi
        yv = np.where(far, 0.93 * hi, v)
        xv = t["x"].to_numpy(float) / 1e6
        key = t["chrom"].str.replace("chr", "", regex=False)
        alt = key.map({c: i % 2 for i, c in enumerate(order)}).to_numpy()
        hot_, cold_ = (t[f].to_numpy(bool) for f in ("hot", "cold"))
        ax.scatter(xv, 100 * t["control"], s=0.6, color="0.88", lw=0,
                   rasterized=True, zorder=0)
        for k, c in ((0, "0.35"), (1, "0.62")):
            m = (alt == k) & ~hot_ & ~cold_
            ax.scatter(xv[m], yv[m], s=0.8, color=c, lw=0, rasterized=True,
                       zorder=1)
        for m, c in ((hot_, HOT_C), (cold_, COLD_C)):
            ax.scatter(xv[m & ~far], yv[m & ~far], s=2.2, color=c, lw=0,
                       rasterized=True, zorder=2)
        for xf, yf, vf, hf in zip(xv[far], yv[far], v[far], hot_[far]):
            c = HOT_C if hf else "0.35"
            ax.scatter([xf], [yf], s=5, marker="^", color=c, lw=0, zorder=2)
            ax.text(xf, yf, f"  {vf:.0f}", ha="left", va="center",
                    fontsize=4.4, color=c)
        n_bins = len(t) + len(un)
        ax.text(0.005, 0.98, f"hot {int(hot_.sum())} of {len(t):,} tested "
                             f"bins"
                + (f" (of {n_bins:,}; grey ticks: too few contacts)"
                   if len(un) > 0.05 * n_bins else ""),
                transform=ax.transAxes, ha="left", va="top", fontsize=4.8,
                color=HOT_C)
        ax.text(0.995, 0.98, f"cold {int(cold_.sum())}",
                transform=ax.transAxes, ha="right", va="top", fontsize=4.8,
                color=COLD_C)
        _zero(ax)
        ax.set_xlim(0, (off[order[-1]] + ends[order[-1]]) / 1e6)
        lo = np.nanpercentile(v, 0.5)
        lo = min(lo, np.nanpercentile(100 * t["control"], 0.5))
        if len(un):
            foot = lo - 0.06 * (hi - lo)
            ax.scatter(un["x"] / 1e6, np.full(len(un), foot), s=1.2,
                       marker="|", color="0.7", lw=0.4, rasterized=True,
                       zorder=0)
            lo = foot - 0.04 * (hi - lo)
        ax.set_ylim(lo, hi)
        ax.set_ylabel(f"{mark}\nΔC (pp)", color=MARK_COLOR[mark])
        # the tracks share x, and with it the labels: set once, hidden above
        if r < len(ax_tracks) - 1:
            ax.tick_params(labelbottom=False)
        ax.tick_params(axis="x", length=0)
    mids = [(off[c] + ends[c] / 2) / 1e6 for c in order]
    ax_tracks[-1].set_xticks(mids)
    ax_tracks[-1].set_xticklabels(order, fontsize=4.6)
    ax_tracks[-1].set_xlabel("chromosome (1-Mb bins)")
    # what the hot bins are: each track's share in them over all tested
    # bins, on a log scale, faded where q >= 0.05
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullLocator
    tracks = tested_tracks(h)
    rows, breaks = g_rows(tracks)
    y = np.arange(len(rows))[::-1].astype(float)
    shown = [t for t, _ in rows]
    res = {mark: hot_enrichment(h[h["context"] == mark], tracks)
           .set_index("track").reindex(shown) for _, mark, _ in MARKS}
    v = np.concatenate([r_[c].to_numpy(float) for r_ in res.values()
                        for c in ("fold", "control_fold")])
    v = v[np.isfinite(v) & (v > 0)]
    lim = max(1.0, 1.15 * np.abs(np.log2(v)).max()) if v.size else 1.0
    lo, hi = 2.0 ** -lim, 2.0 ** lim
    for k, (_, mark, _) in enumerate(MARKS):
        e = res[mark]
        f = e["fold"].to_numpy(float)
        sig = (e["q"] < 0.05).to_numpy()
        yk = y + (0.19 if k == 0 else -0.19)
        ok = np.isfinite(f) & (f > 0)
        ax_enr.barh(yk[ok], f[ok] - 1, left=1, height=0.36, lw=0,
                    color=[to_rgba(MARK_COLOR[mark], 0.85 if s_ else 0.22)
                           for s_ in sig[ok]])
        # no hot bin on a sparse track: no bar to the axis's edge, a 0
        for yy in yk[np.isfinite(f) & (f <= 0)]:
            ax_enr.text(1 / 1.08, yy, "0", ha="right", va="center",
                        fontsize=4.2, color="0.45")
        c = e["control_fold"].to_numpy(float)
        okc = np.isfinite(c) & (c > 0)
        ax_enr.scatter(c[okc], yk[okc], s=5, marker="|", color="0.35",
                       lw=0.6, zorder=3)
        for yy, f_, s_ in zip(yk, f, sig):
            if s_:
                # an asterisk rides high in its line: set down to the bar
                ax_enr.text(f_ * 1.06 if f_ >= 1 else f_ / 1.06, yy - 0.24,
                            "*", ha="left" if f_ >= 1 else "right",
                            va="center", fontsize=6.5,
                            color=MARK_COLOR[mark])
    ax_enr.axvline(1, color="0.7", lw=0.4, zorder=0)
    ax_enr.set_xscale("log", base=2)
    ax_enr.set_xlim(lo, hi)
    ticks = [2.0 ** i for i in range(-int(lim), int(lim) + 1)]
    ax_enr.set_xticks(ticks)
    ax_enr.set_xticklabels([f"{t_:g}×" for t_ in ticks])
    ax_enr.xaxis.set_minor_locator(NullLocator())
    for b in breaks:
        ax_enr.axhline(len(rows) - b - 0.5, color="0.85", lw=0.4, zorder=0)
    ax_enr.set_yticks(y)
    ax_enr.set_yticklabels([lab for _, lab in rows], fontsize=5)
    ax_enr.set_ylim(-0.7, len(rows) - 0.3)
    ax_enr.tick_params(axis="y", length=0)
    ax_enr.set_xlabel("annotation in hot bins, fold over all tested bins\n"
                      "(* q < 0.05, genome rotations)")
    handles = [Patch(color=to_rgba(MARK_COLOR[m], 0.85), label=m)
               for _, m, _ in MARKS]
    handles += [Patch(color=to_rgba("0.5", 0.22), label="q ≥ 0.05"),
                Line2D([], [], marker="|", ls="", color="0.35",
                       label="control")]
    ax_enr.legend(handles=handles, frameon=False, loc="lower right",
                  fontsize=4.6, ncol=4, handletextpad=0.3, handlelength=1.2,
                  borderaxespad=0.1, columnspacing=0.8,
                  bbox_to_anchor=(1.0, 1.0))


# ----------------------------------------------------------------- layout --

def build_figure(d: dict, order, colors):
    plt = _style()
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(7.2, 10.9))
    gs = GridSpec(4, 12, figure=fig, hspace=0.46, wspace=1.6,
                  height_ratios=[2.5, 1.75, 2.15, 1.95],
                  left=0.075, right=0.975, top=0.97, bottom=0.04)
    letters = []
    # a: the three measures across the page, the correlations tall enough
    # for the shuffle's schematic above their curves
    ga = GridSpecFromSubplotSpec(2, 3, gs[0, :], hspace=0.1, wspace=0.26,
                                 height_ratios=[1.5, 1])
    ax_ar = [fig.add_subplot(ga[0, j]) for j in range(3)]
    ax_ad = [fig.add_subplot(ga[1, j], sharex=ax_ar[j]) for j in range(3)]
    panel_a(ax_ar, ax_ad, d["dist"])
    letters.append(("a", ax_ar[0], 0.06))
    # b, c, d: the regions, then the cells, overall and by type
    gb = GridSpecFromSubplotSpec(1, 2, gs[1, 0:4], width_ratios=[0.12, 1],
                                 wspace=0.0)
    ax_b = fig.add_subplot(gb[1])
    panel_b(ax_b, d["regions"], d["regions_summary"])
    letters.append(("b", ax_b, 0.06))
    gc = GridSpecFromSubplotSpec(2, 2, gs[1, 4:7], hspace=0.95, wspace=0.0,
                                 width_ratios=[0.15, 1])
    ax_c = [fig.add_subplot(gc[i, 1]) for i in range(2)]
    panel_c(ax_c, d["percell"])
    letters.append(("c", ax_c[0], 0.035))
    gd = GridSpecFromSubplotSpec(2, 1, gs[1, 7:12], hspace=0.22)
    ax_d = [fig.add_subplot(gd[0])]
    ax_d.append(fig.add_subplot(gd[1], sharex=ax_d[0]))
    panel_d(ax_d, d["percell"], order, colors)
    letters.append(("d", ax_d[0], 0.06))
    # e, f: by class pair
    ge = GridSpecFromSubplotSpec(1, 3, gs[2, 0:6], wspace=0.12,
                                 width_ratios=[0.3, 1, 1])
    ax_e = [fig.add_subplot(ge[1])]
    ax_e.append(fig.add_subplot(ge[2], sharey=ax_e[0]))
    panel_e(ax_e, d["summary"])
    letters.append(("e", ax_e[0], 0.075))
    gf = GridSpecFromSubplotSpec(1, 4, gs[2, 6:12], wspace=0.06,
                                 width_ratios=[0.3, 1, 1, 0.25])
    ax_f = [fig.add_subplot(gf[1]), fig.add_subplot(gf[2])]
    panel_f(ax_f, d["summary"])
    letters.append(("f", ax_f[0], 0.075))
    # g: along the genome, and what the bins are
    gt = GridSpecFromSubplotSpec(2, 1, gs[3, 0:7], hspace=0.14)
    ax_gt = [fig.add_subplot(gt[0])]
    ax_gt.append(fig.add_subplot(gt[1], sharex=ax_gt[0]))
    ge2 = GridSpecFromSubplotSpec(1, 2, gs[3, 7:12], wspace=0.0,
                                  width_ratios=[0.42, 1])
    ax_ge = fig.add_subplot(ge2[1])
    panel_g(ax_gt, ax_ge, d["hotspots"])
    letters.append(("g", ax_gt[0], 0.06))
    fig.canvas.draw()
    for lab, ax, dx in letters:
        bb = ax.get_position()
        fig.text(bb.x0 - dx, bb.y1 + 0.01, lab, fontsize=8,
                 fontweight="bold", va="bottom", ha="left")
    return fig


def build_supplementary(d: dict):
    """--distance's own a and b, its null excluding the contacts'
    near-copies, where the main figure's come from --distance-ab."""
    plt = _style()
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(7.2, 3.3))
    # a's three measures, b beside them
    gs = GridSpec(1, 2, figure=fig, wspace=0.16, width_ratios=[3.1, 1],
                  left=0.07, right=0.985, top=0.9, bottom=0.14)
    ga = GridSpecFromSubplotSpec(2, 3, gs[0], hspace=0.12, wspace=0.34)
    ax_ar = [fig.add_subplot(ga[0, j]) for j in range(3)]
    ax_ad = [fig.add_subplot(ga[1, j], sharex=ax_ar[j]) for j in range(3)]
    panel_a(ax_ar, ax_ad, d["dist_supp"])
    gb = GridSpecFromSubplotSpec(3, 2, gs[1], width_ratios=[0.28, 1],
                                 height_ratios=[0.12, 1, 0.12], wspace=0.0)
    ax_b = fig.add_subplot(gb[1, 1])
    panel_b(ax_b, d["regions_supp"], d["regions_summary_supp"])
    fig.canvas.draw()
    for lab, ax, dx in (("a", ax_ar[0], 0.06), ("b", ax_b, 0.06)):
        bb = ax.get_position()
        fig.text(bb.x0 - dx, bb.y1 + 0.015, lab, fontsize=8,
                 fontweight="bold", va="bottom", ha="left")
    return fig


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--distance", required=True,
                    help="end_posterior_distance_correlation.py's output "
                         "directory (per cell, link-shuffled null).")
    ap.add_argument("--distance-ab", default=None,
                    help="Panels a and b from this run instead: one whose "
                         "null keeps every stand-in (no --null-exclude), so "
                         "a reaches below 1 kb.  Default --distance.")
    ap.add_argument("--supplementary", default=None, metavar="PDF",
                    help="With --distance-ab: --distance's own a and b, its "
                         "null without the contacts' near-copies, drawn "
                         "here as a figure of their own.")
    ap.add_argument("--classes", required=True,
                    help="annotation_class_concordance.py's output directory "
                         "(--systems ab,rt,chromhmm,pmd).")
    ap.add_argument("--samples", default=None,
                    help="A samples table giving each cell type's colour "
                         "(--group-column, --color-column); else the "
                         "cortex defaults.")
    ap.add_argument("--group-column", default="group")
    ap.add_argument("--color-column", default="color")
    ap.add_argument("--output", required=True)
    return ap.parse_args(argv)


#: A run's tables for a and b.
AB_TABLES = ("end_posterior_dist_all.tsv", "end_posterior_regions.tsv",
             "end_posterior_regions_summary.tsv")


def read_tables(args) -> dict:
    dd, cc = Path(args.distance), Path(args.classes)
    ab = Path(args.distance_ab) if getattr(args, "distance_ab", None) else dd
    if getattr(args, "supplementary", None) and ab == dd:
        sys.exit("--supplementary draws --distance's a and b beside "
                 "--distance-ab's: give --distance-ab")
    need = [dd / "end_posterior_percell.tsv"]
    need += [r / f for r in dict.fromkeys((ab, dd)) for f in AB_TABLES]
    need += [cc / f for f in ("summary.tsv", "locus_hotspots.tsv.gz")]
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        sys.exit("missing: " + ", ".join(miss))

    def ab_of(r: Path, suffix=""):
        return {f"dist{suffix}": pd.read_csv(r / AB_TABLES[0], sep="\t"),
                f"regions{suffix}": pd.read_csv(r / AB_TABLES[1], sep="\t"),
                f"regions_summary{suffix}": pd.read_csv(
                    r / AB_TABLES[2], sep="\t").iloc[0]}
    d = {"percell": pd.read_csv(need[0], sep="\t"),
         "summary": pd.read_csv(cc / "summary.tsv", sep="\t"),
         "hotspots": pd.read_csv(cc / "locus_hotspots.tsv.gz", sep="\t")}
    d.update(ab_of(ab))
    if ab != dd:
        d.update(ab_of(dd, "_supp"))
    return d


def cell_colors(args, groups) -> tuple:
    """The cell types in order -- CELL_TYPES's, then any others -- and their
    colours, from --samples where it has them."""
    colors = dict(CELL_TYPES)
    if args.samples:
        s = pd.read_csv(args.samples, sep="\t")
        if {args.group_column, args.color_column} <= set(s.columns):
            colors.update(s.dropna(subset=[args.color_column])
                          .groupby(args.group_column)[args.color_column]
                          .first().to_dict())
    order = [g for g in CELL_TYPES if g in set(groups)]
    order += sorted(set(groups) - set(order))
    return order, colors


def figure_from_args(args, d=None):
    d = read_tables(args) if d is None else d
    order, colors = cell_colors(args, d["percell"]["group"].dropna())
    return build_figure(d, order, colors)


def main(argv=None) -> int:
    args = parse_args(argv)
    d = read_tables(args)
    fig = figure_from_args(args, d)         # Agg first (_style)
    import matplotlib.pyplot as plt
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")
    if args.supplementary:
        sup = Path(args.supplementary)
        sup.parent.mkdir(parents=True, exist_ok=True)
        fig = build_supplementary(d)
        fig.savefig(sup, dpi=300)
        plt.close(fig)
        print(f"wrote {sup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
