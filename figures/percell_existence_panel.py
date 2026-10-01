#!/usr/bin/env python3
"""Figure 5c: is long-range concordance detectable inside individual cells?

Panel a pools pairs from every cell into one correlation per distance, which
shows concordance exists on average.  This panel asks the single-cell question:
for EACH cell, are its linked pairs more concordant than its own matched
unlinked pairs?  Same statistic as panel a, computed per cell:

    delta_c = r(linked pairs of cell c) - r(matched unlinked pairs of cell c)
    z_c     = delta_c / se_c

with a within-cell mate-shuffle giving the null.  A correlation subtracts each
end's own mean, so a difference in marginal between linked and unlinked ends
cannot fake a signal -- which is why this panel uses panel a's statistic and
not the marginal-corrected C (that matrix is centred to zero per bin, so the
level this panel tests is gone from it by construction).

Layout, one row per mark:
  (i)   per-cell z, observed vs within-cell null, with how many cells pass
  (ii)  per-cell delta by cell type, in --group-order, with % significant
  (iii) z against linked pairs per cell: how deep a cell must be sequenced

Input is a per-cell table (one row per cell x mark):
    cell_id, group, mark, n_pairs, delta, se, z, z_null[, simulated]

    figures/percell_existence_panel.py --simulate -o figures/percell_existence_panel.simulated
    figures/percell_existence_panel.py --table per_cell_existence.tsv -o fig5c

Anything drawn from simulated input carries a stamp saying so, on the figure
itself, so a mock-up can never be mistaken for a result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nome_hic_concordance import figure_style                 # noqa: E402
from nome_hic_concordance.distance_stats import fdr           # noqa: E402

GROUP_ORDER = ["IT-L2/3", "IT-L4", "IT-L5/6", "CT-L6", "Inhib",
               "ODC", "OPC", "ASC", "MGC", "PC/EC"]
#: Placeholders.  The real run takes colours from the manifest's
#: --color-column so this panel matches the UMAP and heatmap.
PLACEHOLDER_COLORS = ["#E69F00", "#D55E00", "#CC79A7", "#882255", "#0072B2",
                      "#009E73", "#56B4E9", "#999933", "#44AA99", "#777777"]
MARKS = ("GCH", "HCG")
MARK_COLOR = {"GCH": "#3D7AB8", "HCG": "#D4504A"}     # panel a's blue / red
MARK_LABEL = {"GCH": "GCH accessibility", "HCG": "HCG methylation"}
SIM_STAMP = "SIMULATED DATA — layout mock-up, not results"
NULL_GREY = "#9AA0A8"
REQUIRED = ("cell_id", "group", "mark", "n_pairs", "delta", "se", "z", "z_null")


# ----------------------------------------------------------------- statistics

def one_sided_p(z) -> np.ndarray:
    """Upper-tail normal p.  Concordance is tested in one direction: the
    question is whether linked pairs are MORE alike than matched unlinked."""
    from scipy.stats import norm
    return norm.sf(np.asarray(z, float))


def add_q(df: pd.DataFrame) -> pd.DataFrame:
    """BH q-values across cells, separately per mark, for observed and null.

    Per mark because the two marks are separate families of tests; pooling
    them would let the deeper-covered mark set the threshold for the other.
    """
    out = df.copy()
    for col, src in (("q", "z"), ("q_null", "z_null")):
        out[col] = np.nan
        for m, idx in out.groupby("mark").groups.items():
            out.loc[idx, col] = fdr(one_sided_p(out.loc[idx, src]))
    return out


def z_threshold(df_mark: pd.DataFrame, alpha: float = 0.05) -> float:
    """The smallest observed z that still passes BH at ``alpha``."""
    passing = df_mark.loc[df_mark["q"] < alpha, "z"]
    return float(passing.min()) if len(passing) else float("nan")


def running_median(x, y, n_bins: int = 14, min_n: int = 15):
    """Median of y in log-spaced bins of x, dropping thin bins."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    mid, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_n:
            mid.append(np.sqrt(lo * hi))
            med.append(float(np.median(y[m])))
    return np.array(mid), np.array(med)


def pass_depth(n_pairs, significant, target: float = 0.8,
               n_bins: int = 14, min_n: int = 15):
    """Depth above which at least ``target`` of cells pass, and how it reads.

    Returns ``(depth, kind)`` with kind one of ``"crossing"`` (the running
    fraction rises through ``target``), ``"always"`` (it is above ``target`` at
    every depth with enough cells) or ``"never"``.  A median-z crossing was the
    first version of this, and it misled: when the median already passes in the
    shallowest bin it returned that bin's depth, which reads as a threshold
    that the data never showed.
    """
    x = np.asarray(n_pairs, float)
    sig = np.asarray(significant, bool)
    mid, frac = [], []
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_n:
            mid.append(np.sqrt(lo * hi))
            frac.append(float(sig[m].mean()))
    mid, frac = np.array(mid), np.array(frac)
    if not len(mid):
        return float("nan"), "never"
    if (frac >= target).all():
        return float(mid[0]), "always"
    below = np.flatnonzero(frac < target)
    last_below = int(below[-1])
    if last_below == len(mid) - 1:
        return float("nan"), "never"
    i = last_below + 1
    x0, x1 = np.log10(mid[i - 1]), np.log10(mid[i])
    y0, y1 = frac[i - 1], frac[i]
    t = (target - y0) / (y1 - y0) if y1 != y0 else 0.0
    return float(10 ** (x0 + t * (x1 - x0))), "crossing"


# ----------------------------------------------------------------- simulation

def simulate(seed: int = 0) -> pd.DataFrame:
    """Per-cell table with assumed effect sizes and depths.  For LAYOUT only.

    The assumptions, so it is clear what drives the picture:
      * genome-wide delta per cell: GCH ~0.03-0.05, HCG ~0.09-0.12, set from
        panel a's beyond-1-kb range, slightly higher in neurons;
      * informative linked pairs per cell: log-normal, GCH median 12-18 k,
        HCG 4-6 k (both ends must carry the mark, and CpGs are sparse);
      * se = sqrt(1/n + 1/(3n)): three matched unlinked draws per linked pair.
    The % of significant cells the figure shows follows directly from these
    numbers and says nothing about the real data.
    """
    rng = np.random.default_rng(seed)
    counts = [260, 140, 210, 160, 215, 290, 75, 120, 60, 47]      # sum 1577
    neuron = {"IT-L2/3", "IT-L4", "IT-L5/6", "CT-L6", "Inhib"}
    delta_mu = {
        "GCH": {"IT-L2/3": .046, "IT-L4": .044, "IT-L5/6": .048, "CT-L6": .050,
                "Inhib": .041, "ODC": .034, "OPC": .039, "ASC": .031,
                "MGC": .027, "PC/EC": .030},
        "HCG": {"IT-L2/3": .118, "IT-L4": .112, "IT-L5/6": .121, "CT-L6": .124,
                "Inhib": .108, "ODC": .094, "OPC": .101, "ASC": .090,
                "MGC": .085, "PC/EC": .088},
    }
    delta_sd = {"GCH": 0.012, "HCG": 0.028}
    depth_median = {"GCH": (18_000, 12_000), "HCG": (6_000, 4_000)}

    rows = []
    cell_i = 0
    for g, n_g in zip(GROUP_ORDER, counts):
        for _ in range(n_g):
            cid = f"cell{cell_i:04d}"
            cell_i += 1
            depth_factor = rng.lognormal(0.0, 0.6)   # shared by both marks
            for mark in MARKS:
                med = depth_median[mark][0 if g in neuron else 1]
                n = max(150, int(med * depth_factor * rng.lognormal(0.0, 0.15)))
                se = np.sqrt(1.0 / n + 1.0 / (3 * n))
                true = rng.normal(delta_mu[mark][g], delta_sd[mark])
                delta = true + rng.normal(0.0, se)
                rows.append({
                    "cell_id": cid, "group": g, "mark": mark, "n_pairs": n,
                    "delta": delta, "se": se, "z": delta / se,
                    "z_null": rng.normal(0.0, 1.0), "simulated": True,
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- plotting

def plot(df: pd.DataFrame, group_order=None, group_colors=None,
         out_prefix: str | None = None, formats=("pdf", "png")):
    """Draw panel c and return the Figure.

    The stamp is driven by the DATA, not by a flag the caller could forget: if
    any row is marked simulated, the figure says so.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"per-cell table lacks {missing}")
    if "q" not in df.columns:
        df = add_q(df)
    group_order = list(group_order or GROUP_ORDER)
    present = [g for g in group_order if g in set(df["group"])]
    if group_colors is None:
        group_colors = dict(zip(GROUP_ORDER, PLACEHOLDER_COLORS))
    simulated = bool(df.get("simulated", pd.Series(False)).astype(bool).any())
    marks = [m for m in MARKS if m in set(df["mark"])]

    with plt.rc_context(figure_style.RC):
        fig = plt.figure(figsize=(figure_style.FIG_W_IN, 1.65 * len(marks) + 0.55))
        gs = fig.add_gridspec(len(marks), 3, width_ratios=[1.0, 1.65, 1.05],
                              hspace=0.62, wspace=0.34,
                              left=0.085, right=0.985, top=0.86, bottom=0.12)
        for r, mark in enumerate(marks):
            d = df[df["mark"] == mark]
            col = MARK_COLOR.get(mark, "#333333")
            z_star = z_threshold(d)
            n_cells = len(d)
            n_sig = int((d["q"] < 0.05).sum())
            n_null = int((d["q_null"] < 0.05).sum())

            # (i) observed vs within-cell null ---------------------------
            ax = fig.add_subplot(gs[r, 0])
            hi = float(np.nanpercentile(d["z"], 99.5)) + 1.0
            bins = np.linspace(-4.0, max(hi, 6.0), 46)
            hn, _, _ = ax.hist(d["z_null"], bins=bins, density=True,
                               histtype="stepfilled", color=NULL_GREY,
                               alpha=0.45, lw=0)
            ho, _, _ = ax.hist(d["z"], bins=bins, density=True,
                               histtype="stepfilled", color=col, alpha=0.35, lw=0)
            ax.hist(d["z"], bins=bins, density=True, histtype="step",
                    color=col, lw=0.9)
            ax.set_ylim(0, max(hn.max(), ho.max()) * 1.18)
            # Label the two histograms where they are, not in a legend box.
            # The null label sits LEFT of the null peak at three-quarters
            # height, where the null's own left tail is already low.
            centres = 0.5 * (bins[:-1] + bins[1:])
            ax.text(centres[np.argmax(hn)] - 1.3, hn.max() * 0.78,
                    f"within-cell\nnull\n{n_null:,} cells\nq < 0.05",
                    ha="right", va="center", fontsize=5.4, color="0.4",
                    linespacing=1.05)
            ax.text(centres[np.argmax(ho)], ho.max() * 1.05, "observed",
                    ha="center", va="bottom", fontsize=5.8, color=col)
            if np.isfinite(z_star):
                ax.axvline(z_star, color="0.25", lw=0.6, ls="--")
                ax.text(z_star + 0.2, 0.99, "q = 0.05",
                        transform=ax.get_xaxis_transform(), ha="left",
                        va="top", fontsize=5.2, color="0.3")
            ax.text(0.99, 0.80,
                    f"{n_sig:,} / {n_cells:,} cells\nq < 0.05 ({100 * n_sig / n_cells:.0f}%)",
                    transform=ax.transAxes, ha="right", va="top", color=col,
                    fontsize=6.2, fontweight="bold")
            ax.set_xlabel("per-cell z (linked vs matched unlinked)")
            # No y label: the axis carries no ticks, and the space left of it
            # is where the null histogram's label has to go.
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            if r == 0:
                ax.set_title("Observed vs within-cell null", pad=6)
            ax.text(-0.30, 0.5, MARK_LABEL.get(mark, mark), rotation=90,
                    transform=ax.transAxes, ha="center", va="center",
                    color=col, fontsize=7, fontweight="bold")

            # (ii) by cell type ------------------------------------------
            ax = fig.add_subplot(gs[r, 1])
            null_delta = d["z_null"] * d["se"]
            lo, hi95 = np.percentile(null_delta, [2.5, 97.5])
            ax.axhspan(lo, hi95, color=NULL_GREY, alpha=0.22, lw=0, zorder=0)
            ax.text(1.01, 0.5 * (lo + hi95), "null\n95%",
                    transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=5.2, color="0.45", linespacing=1.0)
            ax.axhline(0.0, color="0.55", lw=0.5, ls="--", zorder=0)
            rng = np.random.default_rng(1 + r)
            data = [d.loc[d["group"] == g, "delta"].to_numpy() for g in present]
            bp = ax.boxplot(data, positions=np.arange(len(present)), widths=0.55,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color="0.1", lw=0.9),
                            whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                            boxprops=dict(lw=0.6))
            for i, (g, patch) in enumerate(zip(present, bp["boxes"])):
                gc = group_colors.get(g, "#888888")
                patch.set_facecolor(gc)
                patch.set_alpha(0.30)
                patch.set_edgecolor(gc)
                vals = data[i]
                ax.scatter(i + rng.uniform(-0.2, 0.2, len(vals)), vals, s=1.2,
                           color=gc, alpha=0.35, lw=0, zorder=1, rasterized=True)
                frac = float((d.loc[d["group"] == g, "q"] < 0.05).mean())
                ax.text(i, 1.0, f"{100 * frac:.0f}%", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=5.4, color="0.3")
            ax.set_xticks(np.arange(len(present)))
            ax.set_xticklabels(present, rotation=40, ha="right")
            for lab, g in zip(ax.get_xticklabels(), present):
                lab.set_color(group_colors.get(g, "#333333"))
            ax.set_ylabel("per-cell Δr")
            ax.set_xlim(-0.6, len(present) - 0.4)
            if r == 0:
                ax.set_title("By cell type (% cells with q < 0.05)", pad=11)

            # (iii) detection vs depth -----------------------------------
            ax = fig.add_subplot(gs[r, 2])
            sig = (d["q"] < 0.05).to_numpy()
            ax.scatter(d.loc[~sig, "n_pairs"], d.loc[~sig, "z"], s=1.5,
                       color=NULL_GREY, alpha=0.55, lw=0, rasterized=True,
                       label="q ≥ 0.05")
            ax.scatter(d.loc[sig, "n_pairs"], d.loc[sig, "z"], s=1.5,
                       color=col, alpha=0.45, lw=0, rasterized=True,
                       label="q < 0.05")
            mid, med = running_median(d["n_pairs"], d["z"])
            ax.plot(mid, med, color="0.1", lw=1.0, label="running median")
            ax.set_xscale("log")
            if np.isfinite(z_star):
                ax.axhline(z_star, color="0.25", lw=0.6, ls="--")
            depth, kind = pass_depth(d["n_pairs"], sig)
            if kind == "crossing":
                ax.axvline(depth, color="0.25", lw=0.6, ls=":")
                note = f"\u226580% of cells pass\nabove ~{depth:,.0f} pairs"
                ax.text(depth * 1.1, 0.97, note, transform=ax.get_xaxis_transform(),
                        ha="left", va="top", fontsize=5.6, color="0.25")
            else:
                note = ("\u226580% of cells pass\nat every depth" if kind == "always"
                        else "<80% of cells pass\nat any depth")
                ax.text(0.03, 0.97, note, transform=ax.transAxes, ha="left",
                        va="top", fontsize=5.6, color="0.25")
            ax.set_xlabel("informative linked pairs per cell")
            ax.set_ylabel("per-cell z")
            if r == 0:
                ax.set_title("Detection vs sequencing depth", pad=6)
                ax.legend(loc="upper left", fontsize=5.2, markerscale=3,
                          handlelength=0.9, handletextpad=0.3,
                          borderaxespad=0.1, labelspacing=0.25)

        fig.text(0.006, 0.985, "c", fontsize=9, fontweight="bold",
                 ha="left", va="top")
        fig.text(0.04, 0.985, "Long-range concordance is detectable in individual cells",
                 fontsize=8, fontweight="bold", ha="left", va="top")
        if simulated:
            fig.text(0.985, 0.985, SIM_STAMP, fontsize=6.5, fontweight="bold",
                     color="#C0392B", ha="right", va="top")

        if out_prefix:
            for fmt in formats:
                fig.savefig(f"{out_prefix}.{fmt}", dpi=600 if fmt == "png" else None,
                            bbox_inches="tight")
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--table", help="per-cell TSV (columns: %s)" % ", ".join(REQUIRED))
    src.add_argument("--simulate", action="store_true",
                     help="draw from simulated input; the figure is stamped")
    ap.add_argument("-o", "--out", required=True, help="output prefix")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    df = simulate(args.seed) if args.simulate else pd.read_csv(args.table, sep="\t")
    df = add_q(df)
    plot(df, out_prefix=args.out)
    for mark, d in df.groupby("mark"):
        n_sig = int((d["q"] < 0.05).sum())
        print(f"{mark}: {n_sig}/{len(d)} cells q<0.05 "
              f"(null {int((d['q_null'] < 0.05).sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
