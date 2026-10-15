#!/usr/bin/env python3
"""Figure 6: X inactivation copy by copy, from allele-resolved scNOMe-HiC.

The panels, from the runs' tables:

  a  how a read pair is put on one copy, and which copy a cell silences
  b  P(s): log2 Xi over Xa by distance, chrX against the autosomes' copies
     (haplotype_contacts_ps_ratio.tsv)
  c  contacts 1 Mb or more crossing each position along X, each copy, DXZ4
     marked (haplotype_contacts_profile.tsv); the correlation of the
     compartment track at contacts' ends (haplotype_contacts.tsv)
  d  Xi less Xa methylation and accessibility per kind of element
     (allelic_imbalance.py's allelic_summary.tsv)
  e  a mosaic line: each cell's inactive X (haplotype_xi_calls_<line>.tsv
     or --mosaic-calls), and its copies' differences as bulk data and as
     each cell sees them (haplotype_contacts.tsv)
  f  escape against inactivated genes: levels (haplotype_genes_<line>.tsv),
     contacts 1 Mb or more (haplotype_contacts.tsv)
  g  cells by their own XCI strength (haplotype_xci_cells/_strata/_trend)
  h  X's excess concordance over the autosomes, observed and predicted from
     the copies' levels alone (copy_mixing_excess.py)
  i  each copy's concordance against non-contacting pairs of the same cell
     and copy (end_posterior_distance_correlation.py --haplotypes)

    python scripts/figure6_xci.py --contacts haplotype_contacts_v2 \\
        --allelic-summary allelic_imbalance_gm12878_elements/allelic_summary.tsv \\
        --copy-mixing copy_mixing_excess \\
        --units haplotype_contacts_v2/end_posterior_units \\
        --clonal GM12878 --mosaic IMR90 --mosaic-calls hap1=49,hap2=92 \\
        --output Figure6.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: Colours: the copies, the autosomes' copies, the bulk view, the marks,
#: the kinds of gene.
XI, XA, AUTO, BULK = "#b2182b", "#2166ac", "0.45", "0.6"
HCG_C, GCH_C = "#762a83", "#1b7837"
ESC_C, INACT_C = "#e08214", "0.25"
#: DXZ4 on hg38 (Xi's megadomain boundary), Mb.
DXZ4_MB = (115_630_000 + 115_970_000) / 2e6


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


def panel_b(ax, ratio: pd.DataFrame, group: str):
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


def panel_c(ax, axb, prof: pd.DataFrame, stats: pd.DataFrame, group: str):
    p = prof[prof["group"] == group]
    x = p["pos"] / 1e6
    ax.plot(x, p["log2_obs_exp_Xi"], color=XI, label="Xi")
    ax.plot(x, p["log2_obs_exp_Xa"], color=XA, label="Xa")
    ax.plot(x, p["difference"], color="black", lw=0.7, label="Xi − Xa")
    ax.axvline(DXZ4_MB, color="0.4", lw=0.5, ls=":")
    ax.text(DXZ4_MB - 3, ax.get_ylim()[1] * 0.98, "DXZ4", ha="right",
            va="top", fontsize=5)
    _zero(ax)
    ax.set_xlabel("chrX (Mb)")
    ax.set_ylabel("contacts ≥1 Mb crossing\n(log$_2$ obs/exp)")
    ax.set_title("Crossing each position", loc="left")
    ax.legend(frameon=False, loc="lower right", ncol=1, handlelength=1.2)
    # compartments: correlation of the track at a contact's two ends
    vals, cols, labs = [], [], []
    for sname, c1, c2 in (("chrX", XI, XA), ("autosomes", "0.35", "0.65")):
        r = _row(stats, group, "A/B compartments: ends' correlation", sname)
        if r is None:
            continue
        vals += [r["copy1"], r["copy2"]]
        cols += [c1, c2]
        labs += (["Xi", "Xa"] if sname == "chrX" else ["hap1", "hap2"])
    if vals:
        pos = [0, 1, 2.5, 3.5][:len(vals)]
        axb.bar(pos, vals, color=cols, width=0.85)
        axb.set_xticks(pos)
        axb.set_xticklabels(labs, rotation=90, fontsize=5)
        axb.set_xlabel("chrX   auto.", fontsize=5.2)
        axb.set_title("A/B", loc="left")
        axb.set_ylabel("correlation at contact ends")


def panel_d(ax, summ: pd.DataFrame, group: str):
    s = summ[summ["group"] == group]
    sname = "chrX rest" if (s["set"] == "chrX rest").any() else "chrX"
    s = s[s["set"] == sname].set_index("context")
    cats = [("all", ""), ("CpG\nislands", "_island"),
            ("promoters", "_promoter"), ("enhancers", "_enhancer"),
            ("CTCF\nsites", "_ctcf")]
    for j, (mark, col, lab) in enumerate((("HCG", HCG_C,
                                           "methylation (HCG)"),
                                          ("GCH", GCH_C,
                                           "accessibility (GCH)"))):
        xs, ys, los, his = [], [], [], []
        for i, (_, suf) in enumerate(cats):
            k = f"{mark}{suf}"
            if k not in s.index:
                continue
            r = s.loc[k]
            xs.append(i + (j - 0.5) * 0.3)
            ys.append(r["xi_minus_xa"])
            los.append(r["xi_ci_lo"])
            his.append(r["xi_ci_hi"])
        _err(ax, xs, ys, los, his, col, lab)
    _zero(ax)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c for c, _ in cats])
    ax.set_ylabel("Xi − Xa level")
    ax.set_title(f"Methylation and accessibility ({group})", loc="left")
    ax.legend(frameon=False, loc="upper right")


#: Panel e's measures: (measure, short name).
E_MEASURES = (("P(s): share ≥ 1 Mb", "contacts ≥1 Mb"),
              ("P(s): share ≥ 10 Mb", "contacts ≥10 Mb"),
              ("A/B compartments: ends' correlation", "compartments"),
              ("crossing DXZ4: log2 obs/exp", "DXZ4 crossing"))


def panel_e(ax_calls, axes, stats, clonal, mosaic, calls):
    """``calls``: (hap1 cells, hap2 cells) of the mosaic line."""
    n1, n2 = calls
    ax_calls.bar([0, 1], [n1, n2], color=[XI, "#f4a582"], width=0.7)
    ax_calls.set_xticks([0, 1])
    ax_calls.set_xticklabels(["hap1", "hap2"])
    ax_calls.set_ylabel("cells")
    ax_calls.set_title(f"{mosaic} cells:\nhaplotype silenced",
                       loc="left")
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
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["GM", "IMR\nbulk", "IMR\ncell"])
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
    ax_lv.set_xticklabels([c for _, c in cats])
    ax_lv.set_ylabel("Xi − Xa level")
    ax_lv.set_title("Escape genes: levels like Xa…", loc="left")
    ax_lv.legend(frameon=False, loc="center right")
    for i, g in enumerate((clonal, mosaic)):
        for j, (cls, col) in enumerate((("escape genes", ESC_C),
                                        ("inactivated genes", INACT_C))):
            r = _row(stats, g, "P(s): share ≥ 1 Mb", f"chrX {cls}")
            if r is not None:
                _err(ax_3d, [i + (j - 0.5) * 0.3], [r["difference"]],
                     [r["ci_lo"]], [r["ci_hi"]], col)
    _zero(ax_3d)
    ax_3d.set_xticks([0, 1])
    ax_3d.set_xticklabels([clonal, f"{mosaic}\n(per cell)"])
    ax_3d.set_ylabel("Xi − Xa, contacts ≥1 Mb")
    ax_3d.set_title("…but fold like Xi", loc="left")


def panel_g(ax_h, ax1, ax2, cells, strata, trend):
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
        ax_h.legend(frameon=False, loc="upper left")
    ax_h.axvline(0, color="0.5", lw=0.4)
    ax_h.set_xlabel("XCI strength (Xi − Xa\nat CpG islands)")
    ax_h.set_ylabel("cells")
    ax_h.set_title(f"{len(s)} cells", loc="left")
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


def panel_h(axes, cme: pd.DataFrame, fine: pd.DataFrame, clonal, mosaic):
    for ax, ctx in zip(axes, ("HCG", "GCH")):
        f = fine[fine["context"] == ctx]
        xm = np.sqrt(f["dist_lo"] * f["dist_hi"])
        ok = np.isfinite(f["predicted"])
        ax.fill_between(xm[ok], 100 * f["predicted_lo"][ok],
                        100 * f["predicted_hi"][ok], color=XI, alpha=0.18,
                        lw=0)
        ax.plot(xm[ok], 100 * f["predicted"][ok], color=XI,
                label="predicted: copies' levels alone")
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
    axes[0].set_ylabel("ΔC, chrX − autosomes (pp)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=5)


def panel_i(ax, units: pd.DataFrame):
    u = units[units["measure"] == "gg"]
    series = (("chrX Xi", XI, "Xi"), ("chrX Xa", XA, "Xa"),
              ("autosomes same copy", AUTO, "autosomes (per copy)"))
    for k, (cmp_, col, lab) in enumerate(series):
        t = u[u["comparison"] == cmp_].sort_values("band_lo")
        if not len(t):
            continue
        xc = np.sqrt(t["band_lo"] * t["band_hi"]) * (1.25 ** (k - 1))
        _err(ax, xc, t["value"], t["lo"], t["hi"], col, lab)
    d = u[u["comparison"] == "chrX − autosomes, same copy"]
    if len(d):
        inc = bool(((d["lo"] < 0) & (d["hi"] > 0)).all())
        ax.text(0.97, 0.97, "chrX − autosomes, per copy:\n"
                + ("no difference at any distance" if inc
                   else "differs in some band"),
                transform=ax.transAxes, va="top", ha="right", fontsize=5.2)
    _zero(ax)
    ax.set_xscale("log")
    ax.set_xlabel("distance between ends (bp)")
    ax.set_ylabel("Δr, contacts − same-cell, same-copy null")
    ax.set_title("GCH–GCH concordance per copy", loc="left")
    lo = np.nanmin(u.loc[u["comparison"].isin([c for c, _, _ in series]),
                         "lo"])
    ax.set_ylim(min(lo, 0) - 0.06, ax.get_ylim()[1] + 0.06)
    ax.legend(frameon=False, loc="lower center", ncol=3, handletextpad=0.2,
              columnspacing=0.8)


# ----------------------------------------------------------------- layout --

def make_figure(d: dict, out: Path, clonal: str, mosaic: str,
                calls: tuple):
    plt = _style()
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(7.2, 9.4))
    gs = GridSpec(4, 6, figure=fig, hspace=0.85, wspace=1.15,
                  height_ratios=[1.12, 1, 1, 1],
                  left=0.075, right=0.985, top=0.975, bottom=0.05)
    letters = []
    ax_a = fig.add_subplot(gs[0, 0:2])
    panel_a(ax_a)
    letters.append(("a", ax_a, 0.0))
    ax_b = fig.add_subplot(gs[0, 2:4])
    panel_b(ax_b, d["ratio"], clonal)
    letters.append(("b", ax_b, 0.045))
    gc = GridSpecFromSubplotSpec(1, 2, gs[0, 4:6], width_ratios=[2.3, 1],
                                 wspace=0.75)
    ax_c, ax_cb = fig.add_subplot(gc[0]), fig.add_subplot(gc[1])
    panel_c(ax_c, ax_cb, d["profile"], d["stats"], clonal)
    letters.append(("c", ax_c, 0.05))
    ax_d = fig.add_subplot(gs[1, 0:2])
    panel_d(ax_d, d["summary"], clonal)
    letters.append(("d", ax_d, 0.045))
    ge = GridSpecFromSubplotSpec(1, 5, gs[1, 2:6],
                                 width_ratios=[0.85, 1, 1, 1, 1], wspace=0.75)
    ax_e0 = fig.add_subplot(ge[0])
    ax_e = [fig.add_subplot(ge[i]) for i in range(1, 5)]
    panel_e(ax_e0, ax_e, d["stats"], clonal, mosaic, calls)
    letters.append(("e", ax_e0, 0.045))
    gf = GridSpecFromSubplotSpec(1, 2, gs[2, 0:2], width_ratios=[1.6, 1],
                                 wspace=0.7)
    ax_f1, ax_f2 = fig.add_subplot(gf[0]), fig.add_subplot(gf[1])
    panel_f(ax_f1, ax_f2, d["genes"], d["stats"], clonal, mosaic)
    letters.append(("f", ax_f1, 0.045))
    gg = GridSpecFromSubplotSpec(1, 3, gs[2, 2:6], wspace=0.65)
    ax_g = [fig.add_subplot(gg[i]) for i in range(3)]
    panel_g(*ax_g, d["cells"], d["strata"], d["trend"])
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
    fig.savefig(out, dpi=300)
    plt.close(fig)


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


def main(argv=None) -> int:
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
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    c = Path(args.contacts)
    cm = Path(args.copy_mixing)
    d = {"ratio": pd.read_csv(c / "haplotype_contacts_ps_ratio.tsv",
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
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_figure(d, out, args.clonal, args.mosaic, _calls(args, c))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
