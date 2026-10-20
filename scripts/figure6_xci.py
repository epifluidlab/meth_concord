#!/usr/bin/env python3
"""Figure 6: X inactivation copy by copy, from allele-resolved scNOMe-HiC.

The panels, from the runs' tables:

  a  how a read pair is put on one copy, and which copy a cell silences
  b  X copy by copy, as a genome browser shows it: each copy's contact map
     (haplotype_locus_<line>_chrX_map.tsv.gz), then along X the contacts
     1 Mb or more spanning each position, over those expected of their
     lengths placed at random (haplotype_contacts_profile.tsv), the
     compartment of each copy's long-range partners about its mean along X
     over the bulk A/B, and each copy's HCG and GCH level
     (haplotype_locus_<line>_chrX.tsv); X's elements dotted through all, at
     one x on every panel
  c  P(s): log2 Xi over Xa by distance, chrX against the autosomes' copies
     (haplotype_contacts_ps_ratio.tsv)
  d  Xi less Xa methylation and accessibility per kind of element
     (allelic_imbalance.py's allelic_summary.tsv), and beneath it the
     share of the contacts with an end in each that span 1 Mb or more
     (haplotype_contacts.py --element), the autosomes' copies the control
  e  a mosaic line: how many cells silence each haplotype
     (haplotype_xi_calls_<line>.tsv or --mosaic-calls), and its copies'
     differences as bulk data and as each cell sees them
     (haplotype_contacts.tsv)
  f  escape against inactivated genes: levels (haplotype_genes_<line>.tsv),
     contacts 1 Mb or more (haplotype_contacts.tsv)
  g  the clonal line's cells by their own XCI strength
     (haplotype_xci_cells/_strata/_trend)
  h  X's excess concordance over the autosomes, observed and expected from
     the copies' level difference alone (copy_mixing_excess.py), with p of
     the clonal line's observed against expected
  i  each copy's concordance against non-contacting pairs of the same cell
     and copy, with p of X against the autosomes and of Xi against Xa
     (end_posterior_distance_correlation.py --haplotypes)

    python scripts/figure6_xci.py --contacts haplotype_contacts_v4 \\
        --allelic-summary allelic_imbalance_gm12878_elements/allelic_summary.tsv \\
        --copy-mixing copy_mixing_excess \\
        --units haplotype_contacts_v4/end_posterior_units_v2 \\
        --clonal GM12878 --mosaic IMR90 --output Figure6.pdf
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

from nome_hic_concordance.xci import XI_ELEMENTS  # noqa: E402

#: Colours: the copies, the autosomes' copies, the bulk view, the marks,
#: the kinds of gene.
XI, XA, AUTO, BULK = "#b2182b", "#2166ac", "0.45", "0.6"
HCG_C, GCH_C = "#762a83", "#1b7837"
ESC_C, INACT_C = "#e08214", "0.25"


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
        "axes.spines.top": False, "axes.spines.right": False})
    from nome_hic_concordance.figure_style import pdf_images_8bit
    pdf_images_8bit()
    return plt


def _zero(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color="0.7", lw=0.4,
                                                zorder=0)


def _err(ax, x, y, lo, hi, color, label=None, marker="o", ms=3, **kw):
    y, lo, hi = (np.asarray(v, float) for v in (y, lo, hi))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=marker, ms=ms,
                color=color, ecolor=color, elinewidth=0.7, capsize=1.2,
                label=label, **kw)


def _row(stats, group, measure, sname, xi=None):
    t = stats[(stats["group"] == group) & (stats["measure"] == measure)
              & (stats["set"] == sname)]
    if xi is not None:
        t = t[t["xi"].astype(str) == xi]
    else:
        t = t[t["xi"].astype(str) != "haplotypes (bulk view)"]
    return t.iloc[0] if len(t) else None


# ----------------------------------------------------------------- panels --

def panel_a(ax):
    """The schematic: a read pair put on one copy by a SNP, its two ends'
    marks; which copy each cell silences, clonal and mosaic."""
    from matplotlib.patches import FancyBboxPatch, Circle, Arc
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 10.4)
    ax.axis("off")
    ax.text(0.0, 10.2, "SNP puts a read pair on one copy;\n"
                       "each end: CpG methylation, GpC accessibility",
            ha="left", va="top", fontsize=5.3)
    for y, col, lab, allele in ((7.9, XA, "Xa", "A"), (6.4, XI, "Xi", "G")):
        ax.add_patch(FancyBboxPatch((1.0, y - 0.3), 8.2, 0.6,
                                    boxstyle="round,pad=0,rounding_size=0.3",
                                    fc=col, ec="none", alpha=0.85))
        ax.text(0.8, y, lab, ha="right", va="center", fontsize=6,
                color=col, fontweight="bold")
        ax.text(2.6, y, allele, ha="center", va="center", fontsize=5.2,
                color="white", fontweight="bold")
    ax.text(2.6, 8.45, "SNP", ha="center", va="bottom", fontsize=5)
    # a contact on Xi: two read ends, joined below the copy
    for x in (2.6, 7.4):
        ax.add_patch(FancyBboxPatch((x - 0.5, 5.75), 1.0, 0.22,
                                    boxstyle="square,pad=0", fc="black",
                                    ec="none"))
        ax.add_patch(Circle((x - 0.22, 5.25), 0.2, fc="black", ec="black",
                            lw=0.4))
        ax.add_patch(Circle((x + 0.25, 5.25), 0.2,
                            fc=GCH_C if x > 5 else "white", ec=GCH_C,
                            lw=0.5))
    ax.add_patch(Arc((5.0, 5.86), 4.8, 2.2, theta1=180, theta2=360,
                     color="black", lw=0.7))
    ax.text(5.0, 4.45, "contact", ha="center", va="center", fontsize=5)
    # cells: which haplotype each silences (Xi red)
    for y, lab, kinds in ((2.7, "clonal (GM12878)", [1] * 7),
                          (0.45, "mosaic (IMR-90)", [1, 2, 2, 1, 2, 2, 1])):
        ax.text(0.0, y + 0.75, lab, fontsize=5.3, va="bottom")
        for i, k in enumerate(kinds):
            cx = 0.8 + i * 1.3
            ax.add_patch(Circle((cx, y), 0.52, fc="white", ec="0.4",
                                lw=0.5))
            for j, h in enumerate((1, 2)):
                ax.add_patch(FancyBboxPatch(
                    (cx - 0.33, y + 0.08 - j * 0.3), 0.66, 0.17,
                    boxstyle="square,pad=0", ec="none",
                    fc=XI if h == k else XA))
    ax.text(10.0, 3.35, "hap1\nhap2", ha="right", va="center", fontsize=4.6,
            color="0.3")


#: Panel b's tracks under the maps: (the locus table's column stem, or
#: 'crossing' from the profile; label).
B_TRACKS = (("crossing", "≥1 Mb\ncontacts\nspanning"),
            ("partners_compartment", "partners'\nA/B\n(grey: A)"),
            ("HCG", "HCG\nlevel"), ("GCH", "GCH\nlevel"))


def locus_maps(m: pd.DataFrame, edges: np.ndarray, labels=("Xi", "Xa")):
    """haplotype_contacts.py's map table (the upper triangle, pixel by
    pixel) as each copy's matrix over the bins of ``edges``, NaN below the
    diagonal."""
    at = {int(s): k for k, s in enumerate(edges[:-1])}
    i, j = m["start_1"].map(at), m["start_2"].map(at)
    ok = (i.notna() & j.notna()).to_numpy()
    i, j = i[ok].astype(int).to_numpy(), j[ok].astype(int).to_numpy()
    out = []
    for lab in labels:
        a = np.full((edges.size - 1,) * 2, np.nan)
        a[i, j] = m.loc[ok, lab].to_numpy(float)
        out.append(a)
    return out


def _triangle(ax, a, edges, up: bool, vmin, vmax, depth):
    """A copy's map as a triangle along X (Mb), up or down, to ``depth`` bp
    apart: log10 of each pixel's share of the copy's contacts."""
    x_, y_ = np.meshgrid(edges / 1e6, edges / 1e6)
    xs, ys = (x_ + y_) / 2, (x_ - y_) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = np.ma.masked_invalid(np.log10(a))
    qm = ax.pcolormesh(xs, ys if up else -ys, lm, cmap="YlOrRd", vmin=vmin,
                       vmax=vmax, shading="flat", rasterized=True)
    h = depth / 2e6
    ax.set_ylim((0, h) if up else (-h, 0))
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return qm


def panel_b(axes, maps, edges, tracks: pd.DataFrame, prof: pd.DataFrame,
            group: str, elements: dict):
    """``axes``: the two maps, then B_TRACKS's, sharing x.  The colour
    scale goes in the corner the Xi map leaves empty, not beside it, so
    every panel keeps the one width and an element's line runs straight
    down them all."""
    from matplotlib.ticker import MaxNLocator
    ax_xi, ax_xa, *tr_axes = axes
    pos = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in maps])
    vmax = np.log10(np.percentile(pos, 99.5)) if pos.size else 0.0
    depth = 0.5 * (edges[-1] - edges[0])
    qm = _triangle(ax_xi, maps[0], edges, True, vmax - 2.5, vmax, depth)
    _triangle(ax_xa, maps[1], edges, False, vmax - 2.5, vmax, depth)
    for ax, lab, col in ((ax_xi, "Xi", XI), (ax_xa, "Xa", XA)):
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", color=col,
                      fontsize=7, fontweight="bold", labelpad=4)
    ax_xi.text(0.0, 1.0, f"{group} chrX\n{(edges[1] - edges[0]) / 1e6:g} Mb "
                         f"bins", transform=ax_xi.transAxes, ha="left",
               va="top", fontsize=5.3)
    cax = ax_xi.inset_axes([0.962, 0.46, 0.011, 0.5])
    cb = ax_xi.figure.colorbar(qm, cax=cax)
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.tick_params(labelsize=4.6, length=1.5, pad=1, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.ax.set_title("log$_{10}$ share", fontsize=4.6, pad=2, loc="right")
    mid = (tracks["start"] + tracks["end"]).to_numpy() / 2e6
    p = prof[prof["group"] == group]
    for ax, (key, lab_) in zip(tr_axes, B_TRACKS):
        if key == "crossing":
            for lab, col in (("Xi", XI), ("Xa", XA)):
                ax.plot(p["pos"] / 1e6, p[f"log2_obs_exp_{lab}"], color=col,
                        lw=0.7, label=lab)
            _zero(ax)
        else:
            if key == "partners_compartment" and "compartment_bulk" in tracks:
                # the bulk A bins shaded, runs merged: the lines keep the
                # track's height, where bars of the track itself (to +/-1)
                # would flatten them
                a = np.r_[False, tracks["compartment_bulk"].to_numpy(float)
                          > 0, False]
                ed = np.flatnonzero(np.diff(a.astype(int)))
                for i, j in zip(ed[::2], ed[1::2]):
                    ax.axvspan(tracks["start"].iloc[i] / 1e6,
                               tracks["end"].iloc[j - 1] / 1e6, color="0.89",
                               lw=0, zorder=0)
                _zero(ax)
            for lab, col in (("Xi", XI), ("Xa", XA)):
                if f"{key}_{lab}" not in tracks:
                    continue
                v = tracks[f"{key}_{lab}"].to_numpy(float)
                if key == "partners_compartment":
                    # about the copy's own mean along X: partners spread
                    # over X pull every bin toward X's B-leaning average;
                    # what tells is whether a bin's partners are more A or
                    # more B than that
                    v = v - np.nanmean(v)
                ax.plot(mid, v, color=col, lw=0.6, marker="o", ms=0.9)
        ax.set_ylabel(lab_, rotation=0, ha="right", va="center", fontsize=5.3,
                      labelpad=2)
        ax.yaxis.set_major_locator(MaxNLocator(2))
        ax.tick_params(axis="y", labelsize=4.8, length=1.5, pad=1)
    # X's elements: one dotted line through every panel, named above
    lo, hi = edges[0] / 1e6, edges[-1] / 1e6
    for nm, (a_, b_) in elements.items():
        x = (a_ + b_) / 2e6
        if not lo < x < hi:
            continue
        for ax in axes:
            ax.axvline(x, color="0.3", lw=0.45, ls=(0, (1, 1.5)), zorder=3)
        ax_xi.text(x, 1.01, nm, transform=ax_xi.get_xaxis_transform(),
                   ha="center", va="bottom", fontsize=5)
    ax_xi.set_xlim(lo, hi)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", length=0, labelbottom=False)
    tr_axes[-1].set_xlabel("chrX (Mb)")


def panel_c(ax, ratio: pd.DataFrame, group: str):
    for sname, col, lab in (("chrX", XI, "chrX"),
                            ("autosomes", AUTO, "autosomes' copies")):
        t = ratio[(ratio["group"] == group) & (ratio["set"] == sname)
                  & (ratio["log10_lo"] >= 4) & (ratio["log10_hi"] <= 8.0)
                  & (ratio[["contacts_Xi", "contacts_Xa"]].min(axis=1) >= 50)]
        x = 10 ** ((t["log10_lo"] + t["log10_hi"]) / 2)
        ax.fill_between(x, t["ci_lo"], t["ci_hi"], color=col, alpha=0.2,
                        lw=0)
        ax.plot(x, t["log2_ratio"], color=col, label=lab)
    _zero(ax)
    ax.set_xscale("log")
    ax.set_xlabel("distance between contact ends (bp)")
    ax.set_ylabel("log$_2$ P(s), Xi / Xa")
    ax.set_title(f"Contact distance ({group})", loc="left")
    ax.legend(frameon=False, loc="upper left")


#: Panel d's kinds of element: (label, allelic_summary's context suffix,
#: haplotype_contacts.py's --element name; None: every contact).
D_ELEMENTS = (("all", "", None), ("CpG\nislands", "_island", "island"),
              ("promoters", "_promoter", "promoter"),
              ("enhancers", "_enhancer", "enhancer"),
              ("CTCF\nsites", "_ctcf", "ctcf"))


def panel_d(ax, summ: pd.DataFrame, group: str, ax_c=None, stats=None):
    """``ax``: each kind of element's Xi − Xa level, with each one's p;
    ``ax_c``, beneath it, its contacts' (:func:`panel_d_contacts`)."""
    s = summ[summ["group"] == group]
    sname = "chrX rest" if (s["set"] == "chrX rest").any() else "chrX"
    s = s[s["set"] == sname].set_index("context")
    cats = [(lab, suf) for lab, suf, _ in D_ELEMENTS]
    rows = []
    for j, (mark, col) in enumerate((("HCG", HCG_C), ("GCH", GCH_C))):
        xs, ys, los, his, ps = [], [], [], [], []
        for i, (_, suf) in enumerate(cats):
            k = f"{mark}{suf}"
            if k not in s.index:
                ps.append(np.nan)
                continue
            r = s.loc[k]
            xs.append(i + (j - 0.5) * 0.3)
            ys.append(r["xi_minus_xa"])
            los.append(r["xi_ci_lo"])
            his.append(r["xi_ci_hi"])
            ps.append(float(r["p"]) if "p" in r else np.nan)
        _err(ax, xs, ys, los, his, col)
        rows.append(([_p_or_blank(v) for v in ps], col))
    _zero(ax)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c for c, _ in cats])
    ax.set_ylabel("Xi − Xa level")
    # the marks named in their colours, the top left for each element's p
    _title_parts(ax, [("Methylation", HCG_C), (" and ", "black"),
                      ("accessibility", GCH_C), (f" ({group})", "black")])
    # half a category each side, so the first p clears the axis
    ax.set_xlim(-0.6, len(cats) - 0.4)
    _tops(ax, range(len(cats)), rows, gap=0.1)
    if ax_c is not None and stats is not None:
        panel_d_contacts(ax_c, stats, group)
        # the axes share the elements: named once, beneath
        ax.tick_params(labelbottom=False)
        ax_c.set_xticks(range(len(cats)))
        ax_c.set_xticklabels([c for c, _ in cats], fontsize=5)


def panel_d_contacts(ax, stats: pd.DataFrame, clonal: str):
    """Xi − Xa share of contacts 1 Mb or more apart, of all X's contacts
    and of those with an end in each kind of element (haplotype_contacts.py
    --element sets 'chrX <name>'), the clonal line's as the levels above,
    each with its p."""
    m = "P(s): share ≥ 1 Mb"
    xs, ys, los, his, ps = [], [], [], [], []
    for i, (_, _, el) in enumerate(D_ELEMENTS):
        r = _row(stats, clonal, m, "chrX" if el is None else f"chrX {el}")
        if r is None:
            ps.append(np.nan)
            continue
        xs.append(i)
        ys.append(r["difference"])
        los.append(r["ci_lo"])
        his.append(r["ci_hi"])
        ps.append(float(r.get("p", np.nan)))
    if xs:
        _err(ax, xs, ys, los, his, "black", ms=2.6)
    _zero(ax)
    ax.set_ylabel("Xi − Xa, share of\ncontacts ≥1 Mb")
    _tops(ax, range(len(D_ELEMENTS)), [([_p_or_blank(v) for v in ps],
                                        "black")], gap=0.12)


#: Panel e's measures: (measure, short name).
E_MEASURES = (("P(s): share ≥ 1 Mb", "contacts ≥1 Mb"),
              ("A/B compartments: ends' correlation", "compartments"),
              ("crossing DXZ4: log2 obs/exp", "DXZ4 crossing"))


def panel_e(axes, stats, clonal, mosaic, calls):
    """``calls``: (hap1 cells, hap2 cells) of the mosaic line, said above
    the panels -- two numbers, no plot of their own."""
    n1, n2 = calls
    axes[0].annotate(f"{mosaic}: hap1 silenced in {n1} cells, hap2 in {n2}",
                     xy=(0, 1), xycoords="axes fraction", xytext=(0, 13),
                     textcoords="offset points", ha="left", va="bottom",
                     fontsize=6.2)
    for ax, (m, nm) in zip(axes, E_MEASURES):
        pts = [(_row(stats, clonal, m, "chrX"), "black", "o"),
               (_row(stats, mosaic, m, "chrX", "haplotypes (bulk view)"),
                BULK, "s"),
               (_row(stats, mosaic, m, "chrX"), XI, "o")]
        for i, (r, col, mk) in enumerate(pts):
            if r is not None:
                _err(ax, [i], [r["difference"]], [r["ci_lo"]], [r["ci_hi"]],
                     col, marker=mk)
        _zero(ax)
        # each copy difference against none
        for i, (r, col, _) in enumerate(pts):
            if r is not None and np.isfinite(r.get("p", np.nan)):
                ax.text(i, 0.99, _p(float(r["p"])), color=col,
                        transform=ax.get_xaxis_transform(), ha="center",
                        va="top", fontsize=4.6)
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.14 * (y1 - y0))
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([clonal, f"{mosaic}\nbulk", f"{mosaic}\nper cell"],
                           fontsize=5)
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(nm, loc="left")
    axes[0].set_ylabel("Xi − Xa")


def panel_f(ax_lv, ax_3d, genes, stats, clonal, mosaic):
    cats = [("HCG_island", "CGI\nHCG"), ("HCG_promoter", "promoter\nHCG"),
            ("GCH_promoter", "promoter\nGCH")]
    for j, (cls, col) in enumerate((("escape genes", ESC_C),
                                    ("inactivated genes", INACT_C))):
        xs, ys, los, his = [], [], [], []
        for i, (ctx, _) in enumerate(cats):
            r = genes[(genes["context"] == ctx) & (genes["class"] == cls)]
            if not len(r) or not np.isfinite(r["difference"].iloc[0]):
                continue
            r = r.iloc[0]
            xs.append(i + (j - 0.5) * 0.3)
            ys.append(r["difference"])
            los.append(r["ci_lo"])
            his.append(r["ci_hi"])
        _err(ax_lv, xs, ys, los, his, col, cls)
    _zero(ax_lv)
    ax_lv.set_xticks(range(len(cats)))
    ax_lv.set_xticklabels([c for _, c in cats], fontsize=5)
    ax_lv.set_xlim(-0.5, len(cats) - 0.5)
    ax_lv.set_ylabel("Xi − Xa level")
    ax_lv.set_title("Escape genes: levels like Xa…", loc="left")
    ax_lv.legend(frameon=False, loc="center right")
    # escape against inactivated genes, from the difference's interval
    ps = []
    for ctx, _ in cats:
        r = genes[(genes["context"] == ctx)
                  & (genes["class"] == "escape − inactivated genes")]
        ps.append(p_from_ci(*r[["difference", "ci_lo", "ci_hi"]].iloc[0])
                  if len(r) else np.nan)
    _tops(ax_lv, range(len(cats)), [([_p_or_blank(v, True) for v in ps],
                                     "black")], fontsize=4.4)
    for i, g in enumerate((clonal, mosaic)):
        for j, (cls, col) in enumerate((("escape genes", ESC_C),
                                        ("inactivated genes", INACT_C))):
            r = _row(stats, g, "P(s): share ≥ 1 Mb", f"chrX {cls}")
            if r is not None:
                _err(ax_3d, [i + (j - 0.5) * 0.3], [r["difference"]],
                     [r["ci_lo"]], [r["ci_hi"]], col)
    _zero(ax_3d)
    ps = []
    for g in (clonal, mosaic):
        r = _row(stats, g, "P(s): share ≥ 1 Mb", "escape − inactivated genes")
        ps.append(float(r.get("p", np.nan)) if r is not None else np.nan)
    ax_3d.set_xlim(-0.4, 1.4)
    _tops(ax_3d, [0, 1], [([_p_or_blank(v, True) for v in ps], "black")],
          fontsize=4.4)
    ax_3d.set_xticks([0, 1])
    ax_3d.set_xticklabels([clonal, f"{mosaic}\n(per cell)"], fontsize=5)
    ax_3d.set_ylabel("Xi − Xa, contacts ≥1 Mb")
    ax_3d.set_title("…but fold like Xi", loc="left")


def panel_g(ax_h, ax1, ax2, cells, strata, trend, clonal):
    from scipy.stats import norm
    s = cells["strength"].to_numpy(float)
    e = cells["noise_sd"].to_numpy(float)
    ok = np.isfinite(s)
    s = s[ok]
    bins = np.linspace(np.nanmin(s) - 0.05, np.nanmax(s) + 0.05, 28)
    ax_h.hist(s, bins=bins, color="0.7")
    e = e[ok & np.isfinite(cells["noise_sd"].to_numpy(float))]
    e = e[np.isfinite(e) & (e > 0)]
    if e.size:
        xs = np.linspace(bins[0], bins[-1], 200)
        dens = np.mean([norm.pdf(xs, s.mean(), v) for v in e], axis=0)
        ax_h.plot(xs, dens * s.size * (bins[1] - bins[0]), color="black",
                  lw=0.8, label="sampling alone")
        # the key above the curve's peak, not across it
        ax_h.set_ylim(0, ax_h.get_ylim()[1] * 1.3)
        ax_h.legend(frameon=False, loc="upper right")
    ax_h.axvline(0, color="0.5", lw=0.4)
    ax_h.set_xlabel("XCI strength (Xi − Xa\nat CpG islands)")
    ax_h.set_ylabel("cells")
    # the clonal line alone: a mosaic line's Xi is called from this same
    # difference, so its strength would be positive by construction
    ax_h.set_title(f"{clonal}, {len(s)} of {len(cells)} cells", loc="left")
    for ax, (m, nm) in ((ax1, ("P(s): share ≥ 1 Mb", "contacts ≥1 Mb")),
                        (ax2, ("A/B compartments: ends' correlation",
                               "compartment correlation"))):
        sub = []
        for sname, col in (("chrX", XI), ("autosomes", AUTO)):
            t = strata[(strata["measure"] == m) & (strata["set"] == sname)]
            if not len(t):
                continue
            ax.errorbar(t["strength_mean"], t["difference"],
                        yerr=[t["difference"] - t["ci_lo"],
                              t["ci_hi"] - t["difference"]],
                        fmt="o-", ms=2.5, lw=0.7, capsize=1.2, color=col,
                        label=sname)
            r = trend[(trend["measure"] == m) & (trend["set"] == sname)]
            if len(r) and sname == "chrX":
                sub.append(f"slope P = {r['p_perm'].iloc[0]:.2g}")
        _zero(ax)
        ax.set_xlabel("quartile's mean strength")
        ax.set_ylabel(f"Xi − Xa, {nm}")
        ax.set_title(f"{nm.split(' correlation')[0]}"
                     + (f" ({sub[0].replace('slope ', '')})" if sub else ""),
                     loc="left")
    top = ax1.get_ylim()[1]
    ax1.set_ylim(ax1.get_ylim()[0], top + 0.35 * (top - ax1.get_ylim()[0]))
    ax1.legend(frameon=False, loc="upper center", ncol=2)


def _p(p) -> str:
    """'p = 0.10', 'p = 0.004', 'p < 0.001'."""
    if not np.isfinite(p):
        return "p n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}"


def p_from_ci(d, lo, hi) -> float:
    """Two-sided p of a difference from its 95% interval, normal: for
    tables that keep the interval alone (haplotype_genes_<line>.tsv)."""
    from scipy.stats import norm
    se = (hi - lo) / (2 * norm.ppf(0.975))
    return float(2 * norm.sf(abs(d) / se)) if se > 0 and np.isfinite(
        d) else np.nan


def _p_or_blank(p, compact: bool = False) -> str:
    """_p, or nothing where there is no p; ``compact``: without spaces, for
    narrow panels."""
    if not np.isfinite(p):
        return ""
    return _p(p).replace(" ", "") if compact else _p(p)


def _title_parts(ax, parts, fontsize=6.2):
    """A left-aligned title in pieces, each in its colour [(text, colour)]
    -- the marks' colours named in it, in place of a legend."""
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    box = HPacker(children=[TextArea(t, textprops=dict(color=c,
                                                       fontsize=fontsize))
                            for t, c in parts], align="baseline", pad=0,
                  sep=0)
    ax.add_artist(AnchoredOffsetbox(
        loc="lower left", child=box, pad=0, borderpad=0, frameon=False,
        bbox_to_anchor=(0, 1.02), bbox_transform=ax.transAxes))


def vs_expected(cme: pd.DataFrame) -> pd.Series:
    """Per row of copy_mixing_excess.tsv, p of observed against expected:
    their difference over both intervals' standard errors (95% interval /
    3.92), taken as independent -- cells resampled for the one, SNP blocks
    for the other."""
    from scipy.stats import norm
    z = norm.ppf(0.975)
    se = np.hypot((cme["observed_hi"] - cme["observed_lo"]) / (2 * z),
                  (cme["predicted_hi"] - cme["predicted_lo"]) / (2 * z))
    return pd.Series(2 * norm.sf(np.abs(cme["observed"] - cme["predicted"])
                                 / se), index=cme.index)


def _tops(ax, xs, rows, gap=0.075, fontsize=4.6):
    """Rows of text along the top of ``ax``, one per (texts, colour), each
    text at its x; the y range opened to make room."""
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + (gap * len(rows) + 0.03) * (y1 - y0) / (
        1 - gap * len(rows) - 0.03))
    for k, (texts, col) in enumerate(rows):
        for x, t in zip(xs, texts):
            ax.text(x, 0.99 - gap * k, t, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=fontsize, color=col)


def panel_h(axes, cme: pd.DataFrame, fine: pd.DataFrame, clonal, mosaic):
    for ax, ctx in zip(axes, ("HCG", "GCH")):
        f = fine[fine["context"] == ctx]
        xm = np.sqrt(f["dist_lo"] * f["dist_hi"])
        ok = np.isfinite(f["predicted"])
        ax.fill_between(xm[ok], 100 * f["predicted_lo"][ok],
                        100 * f["predicted_hi"][ok], color=XI, alpha=0.18,
                        lw=0)
        ax.plot(xm[ok], 100 * f["predicted"][ok], color=XI,
                label="expected from the copies' level\ndifference alone")
        o = cme[cme["context"] == ctx]
        for k, (g, col, mk) in enumerate(((clonal, "black", "o"),
                                          (mosaic, BULK, "s"))):
            u = o[o["group"] == g]
            if not len(u):
                continue
            xc = np.sqrt(u["dist_lo"] * u["dist_hi"]) * (1.15 ** (k - 0.5))
            _err(ax, xc, 100 * u["observed"], 100 * u["observed_lo"],
                 100 * u["observed_hi"], col, f"observed, {g}", marker=mk)
        _zero(ax)
        ax.set_xscale("log")
        ax.set_xlim(1e3, 1e7)
        ax.set_xlabel("distance between contact ends (bp)")
        ax.set_title(f"{ctx}–{ctx}", loc="left")
        # the clonal line's observed against expected, band by band
        u = o[o["group"] == clonal].sort_values("dist_lo")
        if len(u) and {"predicted_lo", "predicted_hi"} <= set(u.columns):
            _tops(ax, np.sqrt(u["dist_lo"] * u["dist_hi"]),
                  [([_p(v) for v in vs_expected(u)], "black")])
            ax.set_title(f"p: {clonal} against expected", loc="right",
                         fontsize=5, color="0.3")
    axes[0].set_ylabel("ΔC, chrX − autosomes (pp)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=5,
                   bbox_to_anchor=(1.0, 0.9))


def panel_i(ax, units: pd.DataFrame):
    u = units[units["measure"] == "gg"]
    # every contact, band by band: the run also writes --unit-element's
    # rows and all the bands together
    if "element" in u.columns:
        u = u[u["element"] == "all"]
    if "band" in u.columns:
        u = u[~u["band"].astype(str).str.startswith("all")]
    series = (("chrX Xi", XI, "Xi"), ("chrX Xa", XA, "Xa"),
              ("autosomes same copy", AUTO, "autosomes"))
    for k, (cmp_, col, lab) in enumerate(series):
        t = u[u["comparison"] == cmp_].sort_values("band_lo")
        if not len(t):
            continue
        xc = np.sqrt(t["band_lo"] * t["band_hi"]) * (1.25 ** (k - 1))
        _err(ax, xc, t["value"], t["lo"], t["hi"], col, lab)
    _zero(ax)
    ax.set_xscale("log")
    # room left of the first band for its p
    ax.set_xlim(1.4e3, 7e6)
    ax.set_xlabel("distance between ends (bp)")
    ax.set_ylabel("Δr, contacts − same-cell, same-copy null")
    ax.set_title("GCH–GCH concordance per copy", loc="left")
    lo = np.nanmin(u.loc[u["comparison"].isin([c for c, _, _ in series]),
                         "lo"])
    # each band's p, X against the autosomes and Xi against Xa, the cells
    # resampled -- in place of a word that says what they say; their key a
    # coloured p in the legend
    from matplotlib.lines import Line2D
    rows, keys = [], []
    for cmp_, col, lab in (("chrX − autosomes, same copy", AUTO,
                            "X − autosomes"),
                           ("chrX Xi − chrX Xa", XI, "Xi − Xa")):
        d = u[u["comparison"] == cmp_].sort_values("band_lo")
        if len(d) and "p" in d:
            xs = np.sqrt(d["band_lo"] * d["band_hi"])
            rows.append(([_p(v) if np.isfinite(v) else "" for v in d["p"]],
                         col))
            keys.append((Line2D([], [], ls="", marker="$p$", ms=4.5,
                                color=col), lab))
    ax.set_ylim(min(lo, 0) - (0.12 if keys else 0.06), ax.get_ylim()[1])
    if rows:
        _tops(ax, xs, rows)
    h, lab = ax.get_legend_handles_labels()
    h, lab = h + [k for k, _ in keys], lab + [t for _, t in keys]
    if len(h) == 5:
        # filled column by column: the copies on one row, the p's below
        h, lab = ([v[i] for i in (0, 3, 1, 4, 2)] for v in (h, lab))
    ax.legend(h, lab, frameon=False, loc="lower center", ncol=3,
              handletextpad=0.2, columnspacing=0.8)


# ----------------------------------------------------------------- layout --

def build_figure(d: dict, clonal: str, mosaic: str, calls: tuple,
                 elements: dict):
    plt = _style()
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(7.2, 10.0))
    gs = GridSpec(4, 6, figure=fig, hspace=0.5, wspace=1.15,
                  height_ratios=[2.35, 1, 1, 1],
                  left=0.075, right=0.985, top=0.975, bottom=0.045)
    letters = []
    gl = GridSpecFromSubplotSpec(2, 1, gs[0, 0:2], height_ratios=[1.45, 1],
                                 hspace=0.3)
    ax_a = fig.add_subplot(gl[0])
    panel_a(ax_a)
    letters.append(("a", ax_a, 0.0))
    # b's panels in one column of one grid: one width, one x per position
    gb = GridSpecFromSubplotSpec(6, 1, gs[0, 2:6], hspace=0.14,
                                 height_ratios=[2.6, 2.6, 1, 1, 1, 1])
    ax_b = [fig.add_subplot(gb[0])]
    ax_b += [fig.add_subplot(gb[i], sharex=ax_b[0]) for i in range(1, 6)]
    panel_b(ax_b, d["maps"], d["edges"], d["tracks"], d["profile"], clonal,
            elements)
    letters.append(("b", ax_b[0], 0.06))
    ax_c = fig.add_subplot(gl[1])
    panel_c(ax_c, d["ratio"], clonal)
    letters.append(("c", ax_c, 0.045))
    # d and f name more categories than a third of the width holds: their
    # rows split further right, across the grid's own gap
    r1, r2 = (GridSpecFromSubplotSpec(1, 2, gs[r, :], width_ratios=[1, 1.75],
                                      wspace=0.22) for r in (1, 2))
    gd = GridSpecFromSubplotSpec(2, 1, r1[0], hspace=0.14,
                                 height_ratios=[1.25, 1])
    ax_d = fig.add_subplot(gd[0])
    ax_dc = fig.add_subplot(gd[1], sharex=ax_d)
    panel_d(ax_d, d["summary"], clonal, ax_dc, d["stats"])
    letters.append(("d", ax_d, 0.045))
    ge = GridSpecFromSubplotSpec(1, len(E_MEASURES), r1[1], wspace=0.55)
    ax_e = [fig.add_subplot(ge[i]) for i in range(len(E_MEASURES))]
    panel_e(ax_e, d["stats"], clonal, mosaic, calls)
    letters.append(("e", ax_e[0], 0.06))
    gf = GridSpecFromSubplotSpec(1, 2, r2[0], width_ratios=[1.7, 1],
                                 wspace=0.6)
    ax_f1, ax_f2 = fig.add_subplot(gf[0]), fig.add_subplot(gf[1])
    panel_f(ax_f1, ax_f2, d["genes"], d["stats"], clonal, mosaic)
    letters.append(("f", ax_f1, 0.045))
    gg = GridSpecFromSubplotSpec(1, 3, r2[1], wspace=0.65)
    ax_g = [fig.add_subplot(gg[i]) for i in range(3)]
    panel_g(*ax_g, d["cells"], d["strata"], d["trend"], clonal)
    letters.append(("g", ax_g[0], 0.045))
    gh = GridSpecFromSubplotSpec(1, 2, gs[3, 0:4], wspace=0.3)
    ax_h = [fig.add_subplot(gh[i]) for i in range(2)]
    panel_h(ax_h, d["cme"], d["fine"], clonal, mosaic)
    letters.append(("h", ax_h[0], 0.045))
    ax_i = fig.add_subplot(gs[3, 4:6])
    panel_i(ax_i, d["units"])
    letters.append(("i", ax_i, 0.05))
    fig.canvas.draw()
    for lab, ax, dx in letters:
        bb = ax.get_position()
        fig.text(bb.x0 - dx, bb.y1 + 0.014, lab, fontsize=8,
                 fontweight="bold", va="bottom", ha="left")
    return fig


def _calls(args, contacts: Path):
    """The mosaic line's cells by silenced haplotype: from
    haplotype_xi_calls_<line>.tsv if there, else --mosaic-calls."""
    p = contacts / f"haplotype_xi_calls_{args.mosaic}.tsv"
    if p.exists():
        t = pd.read_csv(p, sep="\t")
        return int((t["xi_hap"] == 1).sum()), int((t["xi_hap"] == 2).sum())
    if not args.mosaic_calls:
        sys.exit(f"no {p}: give --mosaic-calls hap1=N,hap2=M")
    kv = dict(x.split("=") for x in args.mosaic_calls.split(","))
    return int(kv["hap1"]), int(kv["hap2"])


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--contacts", required=True,
                    help="haplotype_contacts.py's output directory (run with "
                         "--xi-cells, --genes and the mosaic line).")
    ap.add_argument("--allelic-summary", required=True,
                    help="allelic_imbalance.py's allelic_summary.tsv (with "
                         "--region island, promoter, enhancer, ctcf).")
    ap.add_argument("--copy-mixing", required=True,
                    help="copy_mixing_excess.py's output directory.")
    ap.add_argument("--units", required=True,
                    help="end_posterior_distance_correlation.py "
                         "--haplotypes output directory.")
    ap.add_argument("--clonal", default="GM12878")
    ap.add_argument("--mosaic", default="IMR90")
    ap.add_argument("--mosaic-calls", default=None,
                    metavar="hap1=N,hap2=M",
                    help="The mosaic line's cells by silenced haplotype, "
                         "where the run wrote no haplotype_xi_calls_<line>"
                         ".tsv.")
    ap.add_argument("--locus", default="chrX",
                    help="b's locus: haplotype_contacts.py's haplotype_locus_"
                         "<clonal>_<locus>.tsv and _map.tsv.gz in --contacts "
                         "(default all of X, as its runs name it).")
    ap.add_argument("--genome", default="hg38",
                    choices=sorted(XI_ELEMENTS) + ["none"],
                    help="X's elements dotted through b (XI_ELEMENTS), or "
                         "none.")
    ap.add_argument("--output", required=True)
    return ap.parse_args(argv)


def read_tables(args) -> dict:
    c = Path(args.contacts)
    cm = Path(args.copy_mixing)
    lt = c / f"haplotype_locus_{args.clonal}_{args.locus}.tsv"
    lm = c / f"haplotype_locus_{args.clonal}_{args.locus}_map.tsv.gz"
    miss = [str(p) for p in (lt, lm) if not p.exists()]
    if miss:
        sys.exit(f"no {', '.join(miss)}: haplotype_contacts.py writes them "
                 f"for each of its --locus (all of X by default); name "
                 f"another with --locus")
    tracks = pd.read_csv(lt, sep="\t")
    edges = np.r_[tracks["start"].to_numpy(np.int64), tracks["end"].iloc[-1]]
    d = {"tracks": tracks, "edges": edges,
         "maps": locus_maps(pd.read_csv(lm, sep="\t"), edges),
         "ratio": pd.read_csv(c / "haplotype_contacts_ps_ratio.tsv",
                              sep="\t"),
         "profile": pd.read_csv(c / "haplotype_contacts_profile.tsv",
                                sep="\t"),
         "stats": pd.read_csv(c / "haplotype_contacts.tsv", sep="\t",
                              dtype={"xi": str}),
         "summary": pd.read_csv(args.allelic_summary, sep="\t"),
         "genes": pd.read_csv(c / f"haplotype_genes_{args.clonal}.tsv",
                              sep="\t"),
         "cells": pd.read_csv(c / f"haplotype_xci_cells_{args.clonal}.tsv",
                              sep="\t"),
         "strata": pd.read_csv(c / f"haplotype_xci_strata_{args.clonal}.tsv",
                               sep="\t"),
         "trend": pd.read_csv(c / f"haplotype_xci_trend_{args.clonal}.tsv",
                              sep="\t"),
         "cme": pd.read_csv(cm / "copy_mixing_excess.tsv", sep="\t"),
         "fine": pd.read_csv(cm / "copy_mixing_excess_fine.tsv", sep="\t"),
         "units": pd.read_csv(Path(args.units)
                              / f"end_posterior_units_{args.clonal}.tsv",
                              sep="\t")}
    return d


def figure_from_args(args):
    d = read_tables(args)
    calls = _calls(args, Path(args.contacts))
    return build_figure(d, args.clonal, args.mosaic, calls,
                        XI_ELEMENTS.get(args.genome, {}))


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
