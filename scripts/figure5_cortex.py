#!/usr/bin/env python3
"""Figure 5: long-range concordance in mouse cortex, from the runs' tables.

The panels:

  a  the end posteriors' correlation by distance, linked pairs against
     non-linked ones at the same distance in the same cell, and their
     difference with its interval (end_posterior_dist_all.tsv)
  b  per 10-Mb region, HCG-HCG against GCH-GCH concordance, and the same
     for the negative control (end_posterior_regions.tsv, _summary.tsv)
  c  per cell, concordance against the negative control
     (end_posterior_percell.tsv)
  d  the same by cell type
  e  concordance by class pair -- A/B compartments, replication timing,
     PMD / HMD: ΔC against reshuffled links, and ΔC x w, signed by where
     the concordant mass sits against the cell's genome (summary.tsv of
     annotation_class_concordance.py)
  f  ΔC x w between chromHMM states
  g  concordance along the genome per 1-Mb bin, its hot bins, and what
     the bins are (locus_hotspots.tsv.gz, locus_hotspot_enrichment.tsv)

The cells' own values only, nothing imputed: the imputed matrices' cell-type
signal is reproduced by links with no coordination (design §6b.6).

    python scripts/figure5_cortex.py \\
        --distance end_posterior_distance_correlation/null_link_shuffled_match_per_cell.cell_center \\
        --classes annotation_class_concordance_v12 --output Figure5.pdf
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
#: Panel g's tracks, (track, label); TSS density has no value given itself.
G_TRACKS = (("compartment: B", "B compartment"),
            ("replication timing: late", "late replicating"),
            ("PMD / HMD: PMD", "PMD"),
            ("chromHMM: heterochromatin", "heterochromatin"),
            ("chromHMM: quiescent", "quiescent"),
            ("chromHMM: polycomb", "polycomb"),
            ("chromHMM: transcribed", "transcribed"),
            ("chromHMM: enhancer", "enhancer"),
            ("chromHMM: promoter", "promoter"),
            ("zinc-finger (Zfp / ZNF) genes", "Zfp genes"),
            ("olfactory receptor genes", "olfactory receptors"),
            ("TSSs (count)", "TSS density"))
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
    return plt


def _zero(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color="0.7", lw=0.4,
                                                zorder=0)


def _p(p) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ----------------------------------------------------------------- panels --

def _schematic(ax):
    """Inset: a linked pair, and its non-linked match -- the same anchor,
    a partner at the same distance in the same cell, no ligation."""
    from matplotlib.patches import Arc
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    for y, ls, lab in ((6.6, "-", "linked"), (1.6, "--", "non-linked")):
        ax.plot([0.3, 9.7], [y, y], color="0.55", lw=0.6)
        ax.add_patch(Arc((5.0, y), 7.0, 5.6, theta1=0, theta2=180,
                         color="0.15", lw=0.6, ls=ls))
        for x in (1.5, 8.5):
            ax.plot(x, y, "s", ms=2.6, color="0.15")
        ax.text(5.0, y + 3.1, lab, ha="center", va="bottom", fontsize=4.6)


def panel_a(axes_r, axes_d, dist: pd.DataFrame):
    """``axes_r``, ``axes_d``: per mark, the correlations and their
    difference."""
    x = dist["center_bp"].to_numpy(float)
    for j, (k, mark, title) in enumerate(MARKS):
        col = MARK_COLOR[mark]
        ax, axd = axes_r[j], axes_d[j]
        lk, nl = (dist[f"{k}_{c}_r"].to_numpy(float)
                  for c in ("matched", "null"))
        ax.plot(x, lk, "-o", color=col, ms=1.6, lw=0.9, label="linked")
        ax.plot(x, nl, "--o", color=col, ms=1.6, mfc="white", mew=0.5,
                lw=0.8, alpha=0.9, label="non-linked")
        _zero(ax)
        ax.set_title(title, loc="left", color=col)
        d = lk - nl
        axd.fill_between(x, dist[f"{k}_delta_lo"], dist[f"{k}_delta_hi"],
                         color=col, alpha=0.2, lw=0)
        axd.plot(x, d, "-o", color=col, ms=1.6, lw=0.9)
        q = dist[f"{k}_delta_q"].to_numpy(float)
        sig = np.isfinite(q) & (q < 0.05)
        axd.plot(x[sig], d[sig], "o", ms=3.6, mfc="none", mec=col, mew=0.5,
                 label=f"q < 0.05 ({int(sig.sum())} of {sig.size} bins)")
        _zero(axd)
        for a in (ax, axd):
            a.set_xscale("log")
            a.set_xlim(x.min() / 1.3, x.max() * 1.3)
        ax.tick_params(labelbottom=False)
        axd.set_xlabel("distance between the two ends (bp)")
        axd.legend(frameon=False, loc="upper left", handletextpad=0.2)
    axes_r[0].set_ylabel("r of end posteriors")
    axes_d[0].set_ylabel("Δr, linked − non-linked")
    axes_r[1].legend(frameon=False, loc="upper right", handlelength=1.8)
    _schematic(axes_r[0].inset_axes([0.5, 0.5, 0.48, 0.48]))


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


def panel_b(ax, ax_c, regions: pd.DataFrame, summ: pd.Series):
    t = region_table(regions, summ)
    _scatter(ax, t["delta_gch"], t["delta_hcg"])
    ax.set_title(f"r = {summ['pearson_r']:+.2f}, ρ = "
                 f"{summ['spearman_rho']:+.2f}\ngenome rotation "
                 f"{_p(summ['rotation_p'])}", loc="left")
    ax.set_ylabel("HCG–HCG Δr")
    ax.set_xlabel(f"GCH–GCH Δr ({len(t)} regions of 10 Mb)")
    ax.legend(frameon=False, loc="upper left", fontsize=4.4,
              handlelength=1.6, borderaxespad=0.1)
    _scatter(ax_c, t["delta_ctl_gch"], t["delta_ctl_hcg"], cmap="Greys")
    ax_c.set_title(f"control (second null − null): r = "
                   f"{summ['control_r']:+.2f}", loc="left")
    ax_c.set_ylabel("HCG–HCG control")
    ax_c.set_xlabel("GCH–GCH control")


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
        ax.text(0.0, 1.17, f"{100 * obs['frac']:.0f}% of {obs['n']:,} cells "
                           f"Δ > 0, {fmt_p(obs['p'])}",
                transform=ax.transAxes, ha="left", va="bottom", color=col,
                fontsize=5.4, fontweight="bold")
        ax.text(0.0, 1.03, f"negative control (grey): "
                           f"{100 * ctl['frac']:.0f}% > 0",
                transform=ax.transAxes, ha="left", va="bottom", color="0.4",
                fontsize=5)
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
    """``axes``: GCH ΔC, GCH ΔC x w, HCG ΔC, HCG ΔC x w, rows E_PAIRS."""
    y = np.arange(len(E_PAIRS))[::-1].astype(float)
    for k, (_, mark, _) in enumerate(MARKS):
        t = _pairs(summ, mark, E_PAIRS)
        col = MARK_COLOR[mark]
        for ax, stat, ctl, xlab in (
                (axes[2 * k], "delta", "control", "ΔC (pp)"),
                (axes[2 * k + 1], "signed_rel", "signed_rel_control",
                 "ΔC × w (pp)")):
            v, lo, hi = (100 * t[c].to_numpy(float)
                         for c in (stat, f"{stat}_lo", f"{stat}_hi"))
            ax.errorbar(v, y + 0.13, xerr=[v - lo, hi - v], fmt="o", ms=2.2,
                        color=col, elinewidth=0.6, capsize=0)
            c, clo, chi = (100 * t[c_].to_numpy(float)
                           for c_ in (ctl, f"{ctl}_lo", f"{ctl}_hi"))
            ax.errorbar(c, y - 0.13, xerr=[c - clo, chi - c], fmt="D",
                        ms=1.6, color=NULL_GREY, elinewidth=0.5, capsize=0)
            _zero(ax, "x")
            for b in (2.5, 5.5):
                ax.axhline(b, color="0.85", lw=0.4, zorder=0)
            if stat == "signed_rel":
                xlab += f"\n+ {SIGN_WORDS[mark][1]}"
            ax.set_xlabel(xlab)
        axes[2 * k].set_title(label_of(mark), loc="left", color=col)
        # millions of contacts: the intervals can be narrower than the dots
        # -- say so, or they read as missing
        narrow = []
        for ax, stat in ((axes[2 * k], "delta"),
                         (axes[2 * k + 1], "signed_rel")):
            x0, x1 = ax.get_xlim()
            w_ = 100 * (t[f"{stat}_hi"] - t[f"{stat}_lo"]).to_numpy(float)
            narrow.append(np.nanmedian(w_) / (x1 - x0) < DOT_FRACTION)
        if any(narrow):
            axes[2 * k + 1].text(1.0, 1.02, "95% CIs within the dots",
                                 transform=axes[2 * k + 1].transAxes,
                                 ha="right", va="bottom", fontsize=4.6,
                                 color="0.4")
    # the axes share y: the labels once, on the first
    axes[0].set_ylim(-0.6, len(E_PAIRS) - 0.4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for *_, lab in E_PAIRS])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    from matplotlib.lines import Line2D
    axes[0].legend([Line2D([], [], marker="o", ms=2.2, color="0.3", ls=""),
                    Line2D([], [], marker="D", ms=1.6, color=NULL_GREY,
                           ls="")],
                   ["contacts", "reshuffled\nlinks"],
                   frameon=False, loc="center", fontsize=4.6,
                   handletextpad=0.1, bbox_to_anchor=(0.42, 0.5),
                   borderaxespad=0.0)


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
    each column's at its head, along the diagonal."""
    for ax, (_, mark, _) in zip(axes, MARKS):
        v, spans = chromhmm_matrix(summ, mark)
        lim = _limit(v)
        im = ax.imshow(np.ma.masked_invalid(v), cmap="RdBu_r", vmin=-lim,
                       vmax=lim)
        n = len(STATES)
        for i in range(n):
            for j in range(i + 1):
                if not np.isfinite(v[i, j]):
                    continue
                dark = abs(v[i, j]) > 0.6 * lim
                ax.text(j, i, f"{v[i, j]:.2f}".replace("-", "−"),
                        ha="center", va="center", fontsize=4.2,
                        color="white" if dark else "0.15",
                        alpha=0.4 if spans[i, j] else 1.0)
            ax.text(i - 0.25, i - 0.62, STATES[i], rotation=40, ha="left",
                    va="bottom", fontsize=4.8, rotation_mode="anchor")
        ax.set_xticks([])
        ax.set_yticks(range(n))
        # the rows' states once, on the first; the heads name the columns
        ax.set_yticklabels(STATES if ax is axes[0] else [], fontsize=4.8)
        ax.tick_params(length=0)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(1.0, 1.04, label_of(mark), transform=ax.transAxes,
                ha="right", va="top", color=MARK_COLOR[mark], fontsize=6)
        cax = ax.inset_axes([0.42, -0.1, 0.58, 0.045])
        cb = ax.figure.colorbar(im, cax=cax, orientation="horizontal")
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


def panel_g(ax_tracks, ax_enr, hot: pd.DataFrame, enr: pd.DataFrame):
    """``ax_tracks``: per mark, ΔC per bin along the genome; ``ax_enr``:
    what the bins are, the marks side by side per track."""
    h = hot[(hot["band"] == ALL_BAND) & hot["tested"].astype(bool)].copy()
    h["x"], order, off, ends = _genome_x(h)
    for r, (ax, (_, mark, label)) in enumerate(zip(ax_tracks, MARKS)):
        t = h[h["context"] == mark]
        key = t["chrom"].str.replace("chr", "", regex=False)
        alt = key.map({c: i % 2 for i, c in enumerate(order)})
        ax.scatter(t["x"] / 1e6, 100 * t["control"], s=0.6, color="0.88",
                   lw=0, rasterized=True, zorder=0)
        for k, c in ((0, "0.35"), (1, "0.62")):
            m = (alt == k) & ~t["hot"] & ~t["cold"]
            ax.scatter(t.loc[m, "x"] / 1e6, 100 * t.loc[m, "delta"], s=0.8,
                       color=c, lw=0, rasterized=True, zorder=1)
        n_ = {}
        for flag, c in (("hot", HOT_C), ("cold", COLD_C)):
            m = t[flag].astype(bool)
            ax.scatter(t.loc[m, "x"] / 1e6, 100 * t.loc[m, "delta"], s=2.2,
                       color=c, lw=0, rasterized=True, zorder=2)
            n_[flag] = int(m.sum())
        ax.text(0.005, 0.98, f"hot {n_['hot']} of {len(t):,} bins",
                transform=ax.transAxes, ha="left", va="top", fontsize=4.8,
                color=HOT_C)
        ax.text(0.995, 0.98, f"cold {n_['cold']}", transform=ax.transAxes,
                ha="right", va="top", fontsize=4.8, color=COLD_C)
        _zero(ax)
        ax.set_xlim(0, (off[order[-1]] + ends[order[-1]]) / 1e6)
        lo = np.nanpercentile(100 * t["delta"], 0.5)
        ax.set_ylim(min(lo, np.nanpercentile(100 * t["control"], 0.5)),
                    1.25 * np.nanmax(100 * t["delta"]))
        ax.set_ylabel(f"{mark}\nΔC (pp)", color=MARK_COLOR[mark])
        mids = [(off[c] + ends[c] / 2) / 1e6 for c in order]
        ax.set_xticks(mids)
        ax.set_xticklabels(order if r == len(ax_tracks) - 1 else [],
                           fontsize=4.6)
        ax.tick_params(axis="x", length=0)
    ax_tracks[-1].set_xlabel("chromosome (1-Mb bins)")
    y = np.arange(len(G_TRACKS))[::-1].astype(float)
    for k, (_, mark, _) in enumerate(MARKS):
        e = enr[enr["context"] == mark].set_index("track")
        e = e.reindex([t for t, _ in G_TRACKS])
        given = e["rho_given_tss"].to_numpy(float)
        raw = e["rho"].to_numpy(float)
        yk = y + (0.19 if k == 0 else -0.19)
        ok = np.isfinite(given)
        ax_enr.barh(yk[ok], given[ok], height=0.36, color=MARK_COLOR[mark],
                    alpha=0.8, lw=0, label=mark)
        ax_enr.scatter(raw, yk, s=4, facecolor="white", edgecolor="0.2",
                       lw=0.45, zorder=3, label="raw" if k else None)
        ax_enr.scatter(e["rho_control"], yk, s=5, marker="|", color="0.45",
                       lw=0.6, zorder=3, label="control" if k else None)
    _zero(ax_enr, "x")
    ax_enr.set_yticks(y)
    ax_enr.set_yticklabels([lab for _, lab in G_TRACKS], fontsize=5)
    ax_enr.set_ylim(-0.7, len(G_TRACKS) - 0.3)
    ax_enr.set_xlabel("Spearman ρ, bins' ΔC against the track\n"
                      "(bars: given TSS density)")
    ax_enr.legend(frameon=False, loc="lower right", fontsize=4.6, ncol=4,
                  handletextpad=0.2, borderaxespad=0.1, columnspacing=0.9,
                  bbox_to_anchor=(1.0, 1.0))


# ----------------------------------------------------------------- layout --

def build_figure(d: dict, order, colors):
    plt = _style()
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(7.2, 8.8))
    gs = GridSpec(3, 12, figure=fig, hspace=0.36, wspace=1.6,
                  height_ratios=[2.25, 1.75, 2.25],
                  left=0.075, right=0.975, top=0.965, bottom=0.05)
    letters = []
    ga = GridSpecFromSubplotSpec(2, 2, gs[0, 0:6], hspace=0.12, wspace=0.32)
    ax_ar = [fig.add_subplot(ga[0, j]) for j in range(2)]
    ax_ad = [fig.add_subplot(ga[1, j], sharex=ax_ar[j]) for j in range(2)]
    panel_a(ax_ar, ax_ad, d["dist"])
    letters.append(("a", ax_ar[0], 0.06))
    # b: a spacer on its left keeps its axis labels off a's
    gb = GridSpecFromSubplotSpec(2, 2, gs[0, 6:9], hspace=0.75, wspace=0.0,
                                 width_ratios=[0.16, 1])
    ax_b = [fig.add_subplot(gb[i, 1]) for i in range(2)]
    panel_b(*ax_b, d["regions"], d["regions_summary"])
    letters.append(("b", ax_b[0], 0.06))
    gc = GridSpecFromSubplotSpec(2, 1, gs[0, 9:12], hspace=0.75)
    ax_c = [fig.add_subplot(gc[i]) for i in range(2)]
    panel_c(ax_c, d["percell"])
    letters.append(("c", ax_c[0], 0.03))
    gd = GridSpecFromSubplotSpec(2, 1, gs[1, 0:6], hspace=0.22)
    ax_d = [fig.add_subplot(gd[0])]
    ax_d.append(fig.add_subplot(gd[1], sharex=ax_d[0]))
    panel_d(ax_d, d["percell"], order, colors)
    letters.append(("d", ax_d[0], 0.06))
    # e: a spacer on its left keeps its pairs' names off d
    ge = GridSpecFromSubplotSpec(1, 5, gs[1, 6:12], wspace=0.18,
                                 width_ratios=[0.22, 1, 1, 1, 1])
    ax_e = [fig.add_subplot(ge[1])]
    ax_e += [fig.add_subplot(ge[i], sharey=ax_e[0]) for i in range(2, 5)]
    panel_e(ax_e, d["summary"])
    letters.append(("e", ax_e[0], 0.075))
    # f: room on the left for the states, on the right for the heads
    gf = GridSpecFromSubplotSpec(1, 4, gs[2, 0:6], wspace=0.06,
                                 width_ratios=[0.12, 1, 1, 0.3])
    ax_f = [fig.add_subplot(gf[1]), fig.add_subplot(gf[2])]
    panel_f(ax_f, d["summary"])
    letters.append(("f", ax_f[0], 0.075))
    gg = GridSpecFromSubplotSpec(2, 1, gs[2, 6:12], hspace=0.5,
                                 height_ratios=[1, 1.45])
    gt = GridSpecFromSubplotSpec(2, 1, gg[0], hspace=0.12)
    ax_gt = [fig.add_subplot(gt[0])]
    ax_gt.append(fig.add_subplot(gt[1], sharex=ax_gt[0]))
    # g's enrichment inset from the left, its track names inside g
    ge2 = GridSpecFromSubplotSpec(1, 2, gg[1], wspace=0.0,
                                  width_ratios=[0.3, 1])
    ax_ge = fig.add_subplot(ge2[1])
    panel_g(ax_gt, ax_ge, d["hotspots"], d["enrichment"])
    letters.append(("g", ax_gt[0], 0.06))
    fig.canvas.draw()
    for lab, ax, dx in letters:
        bb = ax.get_position()
        fig.text(bb.x0 - dx, bb.y1 + 0.012, lab, fontsize=8,
                 fontweight="bold", va="bottom", ha="left")
    return fig


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--distance", required=True,
                    help="end_posterior_distance_correlation.py's output "
                         "directory (per cell, link-shuffled null).")
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


def read_tables(args) -> dict:
    dd, cc = Path(args.distance), Path(args.classes)
    need = [dd / f for f in ("end_posterior_dist_all.tsv",
                             "end_posterior_percell.tsv",
                             "end_posterior_regions.tsv",
                             "end_posterior_regions_summary.tsv")]
    need += [cc / f for f in ("summary.tsv", "locus_hotspots.tsv.gz",
                              "locus_hotspot_enrichment.tsv")]
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        sys.exit("missing: " + ", ".join(miss))
    rs = pd.read_csv(need[3], sep="\t")
    return {"dist": pd.read_csv(need[0], sep="\t"),
            "percell": pd.read_csv(need[1], sep="\t"),
            "regions": pd.read_csv(need[2], sep="\t"),
            "regions_summary": rs.iloc[0],
            "summary": pd.read_csv(need[4], sep="\t"),
            "hotspots": pd.read_csv(need[5], sep="\t"),
            "enrichment": pd.read_csv(need[6], sep="\t")}


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


def figure_from_args(args):
    d = read_tables(args)
    order, colors = cell_colors(args, d["percell"]["group"].dropna())
    return build_figure(d, order, colors)


def main(argv=None) -> int:
    args = parse_args(argv)
    fig = figure_from_args(args)            # Agg first (_style)
    import matplotlib.pyplot as plt
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
