"""Per-cell existence of long-range concordance: statistic, tests and figure.

Panel a pools pairs from every cell into one correlation per distance, which
shows concordance exists on average but not that it can be detected inside one
cell.  This module answers the single-cell question with panel a's own
contrast, computed per cell:

    delta_c = r(linked pairs of c) - r(matched unlinked pairs of c)

and is shared by ``scripts/end_posterior_distance_correlation.py``, which
computes it from real data, and ``figures/percell_existence_panel.py``, which
draws the same figure from simulated input for layout.

Two choices decide whether the per-cell test means anything:

**The standard error comes from a block bootstrap over the genome, within the
cell.**  Pairs inside one cell are not independent observations: loci near
each other share compartment and domain state, so an analytic 1/sqrt(n) would
call nearly every cell significant.  Resampling ~10 Mb blocks carries that
spatial dependence into the interval.  Linked and null pairs in a block are
resampled together, so the comparison stays paired by region.

**The negative control is a second, independent null re-pairing.**  delta_null =
r(null draw 2) - r(null draw 1), put through exactly the same bootstrap.  It is
the per-cell test applied to pairs that have no ligation between them, so the
number of cells it passes is the calibration of the whole procedure, read off
the figure rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import figure_style
from .distance_stats import fdr

MARKS = ("GCH", "HCG")
#: Measure keys in end_posterior_distance_correlation -> the mark they test.
MEASURE_TO_MARK = {"gg": "GCH", "hh": "HCG", "gh": "HCG-GCH"}
MARK_COLOR = {"GCH": "#3D7AB8", "HCG": "#D4504A"}     # panel a's blue / red
MARK_LABEL = {"GCH": "GCH accessibility", "HCG": "HCG methylation"}
SIM_STAMP = "SIMULATED DATA \u2014 layout mock-up, not results"
NULL_GREY = "#9AA0A8"
REQUIRED = ("cell_id", "group", "mark", "n_pairs", "delta", "se", "z",
            "z_null", "delta_null")


# ------------------------------------------------------- per-cell estimator ---

def corr_from_sums(s) -> np.ndarray:
    """Pearson r from raw sums ``(..., 6)`` = (n, Sx, Sy, Sxx, Syy, Sxy)."""
    s = np.asarray(s, dtype=np.float64)
    n, sx, sy, sxx, syy, sxy = (s[..., i] for i in range(6))
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        r = cov / np.sqrt(vx * vy)
    return np.where((n >= 2) & (vx > 0) & (vy > 0), r, np.nan)


def block_sums(block_ids, x, y):
    """``(blocks, sums)``: per-block raw sums of one arm, ``sums`` is (k, 6)."""
    b = np.asarray(block_ids)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    b, x, y = b[ok], x[ok], y[ok]
    if b.size == 0:
        return np.empty(0, np.int64), np.empty((0, 6))
    uniq, inv = np.unique(b, return_inverse=True)
    k = uniq.size
    out = np.column_stack([
        np.bincount(inv, minlength=k).astype(np.float64),
        np.bincount(inv, weights=x, minlength=k),
        np.bincount(inv, weights=y, minlength=k),
        np.bincount(inv, weights=x * x, minlength=k),
        np.bincount(inv, weights=y * y, minlength=k),
        np.bincount(inv, weights=x * y, minlength=k),
    ])
    return uniq.astype(np.int64), out


def _align(arms):
    """Put several arms' block sums on one shared block axis (zeros if absent)."""
    allb = np.unique(np.concatenate([a[0] for a in arms])) if arms else np.empty(0)
    out = []
    for blocks, sums in arms:
        m = np.zeros((allb.size, 6))
        if blocks.size:
            m[np.searchsorted(allb, blocks)] = sums
        out.append(m)
    return allb, out


def percell_stats(linked, null, null2=None, n_boot: int = 200, seed: int = 0,
                  min_pairs: int = 100) -> dict:
    """Panel a's delta for ONE cell, with a block-bootstrap SE and a control.

    ``linked``, ``null`` and ``null2`` are :func:`block_sums` outputs for the
    linked arm, the matched-unlinked arm, and an independent second
    matched-unlinked draw.  Returns NaNs (never zeros) when a cell is too thin
    to say anything: below ``min_pairs`` in either arm, or fewer than 5 blocks
    to resample.
    """
    nan = float("nan")
    out = dict(n_pairs=0, n_blocks=0, r_linked=nan, r_null=nan, delta=nan,
               se=nan, z=nan, r_null2=nan, delta_null=nan, se_null=nan,
               z_null=nan)
    arms = [linked, null] + ([null2] if null2 is not None else [])
    blocks, mats = _align(arms)
    L, N = mats[0], mats[1]
    N2 = mats[2] if null2 is not None else None
    nL, nN = L[:, 0].sum(), N[:, 0].sum()
    out.update(n_pairs=int(nL), n_blocks=int(blocks.size))
    if nL < min_pairs or nN < min_pairs:
        return out
    rL, rN = corr_from_sums(L.sum(0)), corr_from_sums(N.sum(0))
    out.update(r_linked=float(rL), r_null=float(rN), delta=float(rL - rN))
    if N2 is not None and N2[:, 0].sum() >= min_pairs:
        rN2 = corr_from_sums(N2.sum(0))
        out.update(r_null2=float(rN2), delta_null=float(rN2 - rN))
    k = blocks.size
    if k < 5 or n_boot < 2:
        return out

    rng = np.random.default_rng(seed)
    W = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(np.float64)
    bL, bN = corr_from_sums(W @ L), corr_from_sums(W @ N)
    d = bL - bN
    se = float(np.nanstd(d, ddof=1)) if np.isfinite(d).sum() > 2 else nan
    out["se"] = se
    if np.isfinite(se) and se > 0:
        out["z"] = out["delta"] / se
    if N2 is not None and np.isfinite(out["delta_null"]):
        d0 = corr_from_sums(W @ N2) - bN
        se0 = float(np.nanstd(d0, ddof=1)) if np.isfinite(d0).sum() > 2 else nan
        out["se_null"] = se0
        if np.isfinite(se0) and se0 > 0:
            out["z_null"] = out["delta_null"] / se0
    return out


# --------------------------------------------------------------- statistics ---

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


def sign_test(delta) -> dict:
    """Fraction of cells with delta > 0 and a one-sided binomial p vs 50%.

    The population-level reading of the per-cell statistic: whether
    concordance is carried by MOST cells, which a cell-by-cell test cannot
    show at single-cell depth.  Under the null a cell's delta is symmetric
    about zero, so 50% is the reference -- and the negative-control arm,
    reported beside it, is the empirical check that 50% is right.
    """
    from scipy.stats import binomtest
    d = np.asarray(delta, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = int(d.size)
    k = int((d > 0).sum())
    if n == 0:
        return {"n": 0, "k": 0, "frac": float("nan"), "p": float("nan")}
    return {"n": n, "k": k, "frac": k / n,
            "p": float(binomtest(k, n, 0.5, alternative="greater").pvalue)}


def fmt_p(p: float) -> str:
    """'p = 3x10^-40' as mathtext.

    Not Unicode superscripts: Arial has no superscript digits, so they print
    as empty boxes in the PDF.  Mathtext in the regular font draws the
    exponent from the same face as the rest of the label.
    """
    if not np.isfinite(p):
        return "p = n/a"
    if p >= 1e-3:
        return f"p = {p:.3f}"
    if p <= 0:
        return r"p < 10$^{-300}$"
    e = int(np.floor(np.log10(p)))
    m = p / 10 ** e
    return f"p = {m:.0f}\u00d710$^{{{e}}}$"


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



# ------------------------------------------------------------------- figure ---

#: Relative widths of the three columns, and the full panel's title.
_COL_WIDTH = (1.0, 1.65, 1.05)
FULL_TITLE = "Long-range concordance is carried by most individual cells"
#: Panel c in the main figure is column (i) alone; (ii) and (iii) go to
#: Extended Data.  (columns, letter, title, file suffix) for each.
LAYOUTS = {
    "full": ((0, 1, 2), "c", FULL_TITLE, ""),
    "main": ((0,), "c", "Concordance in individual cells", "_main"),
    "ED": ((1, 2), None, "Per-cell concordance by cell type and depth", "_ED"),
}


def plot(df: pd.DataFrame, group_order=None, group_colors=None,
         out_prefix: str | None = None, formats=("pdf", "png"),
         columns=(0, 1, 2), letter: str | None = "c",
         title: str | None = FULL_TITLE):
    """Draw panel c and return the Figure.

    Read at the level of CELLS, not of each cell's p-value.  At single-cell
    depth a cell holds a few hundred to a few thousand usable pairs, which is
    rarely enough to call one cell on its own; what the data can show is
    whether the effect is carried by most cells.  So:

      (i)   per-cell delta, observed vs the negative control, headed by the
            share of cells with delta > 0 and a sign test against 50%;
      (ii)  the same per cell type, with each type's share above zero;
      (iii) per-cell z against depth -- where individual significance does
            start, if it does.

    ``columns`` draws any subset, in order: ``(0,)`` is the main-figure panel,
    ``(1, 2)`` its Extended Data companion.  The figure width follows the
    columns kept, with fixed margins in inches, so a one-column panel is not
    a three-column canvas with two holes in it.

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
    if group_order is None:
        group_order = list(pd.unique(df["group"]))
    present = [g for g in group_order if g in set(df["group"])]
    if not group_colors:
        group_colors = {g: figure_style.PALETTE[i % len(figure_style.PALETTE)]
                        for i, g in enumerate(present)}
    simulated = bool(df.get("simulated", pd.Series(False)).astype(bool).any())
    marks = [m for m in MARKS if m in set(df["mark"])]

    rc = dict(figure_style.RC, **{"mathtext.default": "regular"})
    with plt.rc_context(rc):
        columns = tuple(columns)
        if not columns or any(c not in (0, 1, 2) for c in columns):
            raise ValueError("columns must be a non-empty subset of (0, 1, 2)")
        ratios = [_COL_WIDTH[c] for c in columns]
        # Width scales with the columns kept; margins are fixed in inches.
        unit = (figure_style.FIG_W_IN - 0.95) / (sum(_COL_WIDTH) + 0.7)
        width = 0.95 + unit * (sum(ratios) + 0.35 * (len(ratios) - 1))
        # Top margin in inches: the column titles sit 24 pt above the axes
        # (headline lines go between), and the figure title needs its own line
        # above them -- at 0.55 in the two overprinted.
        top_in = 0.74 if title else 0.52
        height = 1.65 * max(len(marks), 1) + top_in - 0.1
        fig = plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(max(len(marks), 1), len(columns),
                              width_ratios=ratios,
                              hspace=0.78, wspace=0.34,
                              left=0.62 / width, right=1 - 0.12 / width,
                              top=1 - top_in / height,
                              bottom=0.12)
        slot = {c: i for i, c in enumerate(columns)}
        for r, mark in enumerate(marks):
            d = df[(df["mark"] == mark) & np.isfinite(df["delta"])]
            col = MARK_COLOR.get(mark, "#333333")
            obs, ctl = sign_test(d["delta"]), sign_test(d["delta_null"])
            row_axes = []

            if 0 in slot:
                # (i) observed vs control, read by sign ------------------------
                ax = fig.add_subplot(gs[r, slot[0]])
                row_axes.append(ax)
                both = np.concatenate([d["delta"].to_numpy(),
                                       d["delta_null"].dropna().to_numpy()])
                both = both[np.isfinite(both)]
                lo_x, hi_x = (np.percentile(both, [0.5, 99.5]) if both.size
                              else (-1.0, 1.0))
                pad = 0.05 * (hi_x - lo_x or 1.0)
                bins = np.linspace(lo_x - pad, hi_x + pad, 41)
                ax.hist(d["delta_null"].dropna(), bins=bins, density=True,
                        histtype="stepfilled", color=NULL_GREY, alpha=0.45, lw=0)
                ax.hist(d["delta"], bins=bins, density=True, histtype="stepfilled",
                        color=col, alpha=0.35, lw=0)
                ax.hist(d["delta"], bins=bins, density=True, histtype="step",
                        color=col, lw=0.9)
                ax.axvline(0.0, color="0.25", lw=0.6, ls="--")
                # The headline sits ABOVE the axes: with real data the two
                # histograms overlap around zero and any in-axes text covers them.
                # Colour names the curves -- mark colour observed, grey control --
                # so no legend box is needed.
                ax.text(0.5, 1.13,
                        f"{100 * obs['frac']:.0f}% of {obs['n']:,} cells "
                        f"\u0394 > 0, {fmt_p(obs['p'])}",
                        transform=ax.transAxes, ha="center", va="bottom",
                        color=col, fontsize=6.2, fontweight="bold")
                ax.text(0.5, 1.02,
                        f"negative control (grey): {100 * ctl['frac']:.0f}% > 0",
                        transform=ax.transAxes, ha="center", va="bottom",
                        color="0.4", fontsize=5.8)
                ax.set_xlabel("per-cell \u0394r (contacts \u2212 null)")
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                if r == 0:
                    ax.set_title("Observed vs negative control", pad=24)

            if 1 in slot:
                # (ii) by cell type -------------------------------------------
                ax = fig.add_subplot(gs[r, slot[1]])
                row_axes.append(ax)
                ctl_d = d["delta_null"].dropna().to_numpy()
                if ctl_d.size:
                    lo, hi95 = np.percentile(ctl_d, [2.5, 97.5])
                    ax.axhspan(lo, hi95, color=NULL_GREY, alpha=0.22, lw=0, zorder=0)
                    ax.text(1.01, 0.5 * (lo + hi95), "control\n95%",
                            transform=ax.get_yaxis_transform(), ha="left",
                            va="center", fontsize=5.2, color="0.45",
                            linespacing=1.0)
                ax.axhline(0.0, color="0.55", lw=0.5, ls="--", zorder=0)
                rng = np.random.default_rng(1 + r)
                data = [d.loc[d["group"] == g, "delta"].to_numpy() for g in present]
                keep = [i for i, v in enumerate(data) if v.size]
                if keep:
                    bp = ax.boxplot([data[i] for i in keep], positions=keep,
                                    widths=0.55, showfliers=False,
                                    patch_artist=True,
                                    medianprops=dict(color="0.1", lw=0.9),
                                    whiskerprops=dict(lw=0.6),
                                    capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
                    for i, patch in zip(keep, bp["boxes"]):
                        gc = group_colors.get(present[i], "#888888")
                        patch.set_facecolor(gc)
                        patch.set_alpha(0.30)
                        patch.set_edgecolor(gc)
                for i, g in enumerate(present):
                    vals = data[i]
                    if not vals.size:
                        continue
                    gc = group_colors.get(g, "#888888")
                    ax.scatter(i + rng.uniform(-0.2, 0.2, len(vals)), vals, s=1.2,
                               color=gc, alpha=0.35, lw=0, zorder=1,
                               rasterized=True)
                    frac = float((vals > 0).mean())
                    ax.text(i, 1.0, f"{100 * frac:.0f}%",
                            transform=ax.get_xaxis_transform(), ha="center",
                            va="bottom", fontsize=5.4, color="0.3")
                ax.set_xticks(np.arange(len(present)))
                ax.set_xticklabels(present, rotation=40, ha="right")
                for lab, g in zip(ax.get_xticklabels(), present):
                    lab.set_color(group_colors.get(g, "#333333"))
                ax.set_ylabel("per-cell \u0394r")
                ax.set_xlim(-0.6, len(present) - 0.4)
                if r == 0:
                    ax.set_title("By cell type (% cells with \u0394 > 0)", pad=24)

            if 2 in slot:
                # (iii) individual significance vs depth ------------------------
                ax = fig.add_subplot(gs[r, slot[2]])
                row_axes.append(ax)
                dz = d[np.isfinite(d["z"])]
                if len(dz):
                    sig = (dz["q"] < 0.05).to_numpy()
                    ax.scatter(dz.loc[~sig, "n_pairs"], dz.loc[~sig, "z"], s=1.5,
                               color=NULL_GREY, alpha=0.55, lw=0, rasterized=True,
                               label="q \u2265 0.05")
                    ax.scatter(dz.loc[sig, "n_pairs"], dz.loc[sig, "z"], s=1.5,
                               color=col, alpha=0.45, lw=0, rasterized=True,
                               label="q < 0.05")
                    if len(dz) >= 30:
                        mid, med = running_median(dz["n_pairs"], dz["z"])
                        ax.plot(mid, med, color="0.1", lw=1.0,
                                label="running median")
                    ax.set_xscale("log")
                    z_star = z_threshold(dz)
                    if np.isfinite(z_star):
                        ax.axhline(z_star, color="0.25", lw=0.6, ls="--")
                    depth, kind = pass_depth(dz["n_pairs"], sig)
                    if kind == "crossing":
                        ax.axvline(depth, color="0.25", lw=0.6, ls=":")
                        note = f"\u226580% pass above\n~{depth:,.0f} pairs"
                    elif kind == "always":
                        note = "\u226580% pass\nat every depth"
                    else:
                        note = (f"{100 * sig.mean():.0f}% pass\n"
                                f"individually")
                    ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right",
                            va="top", fontsize=5.6, color="0.25",
                            bbox=dict(facecolor="white", alpha=0.75, lw=0,
                                      pad=1.0))
                    if r == 0:
                        ax.legend(loc="upper left", fontsize=5.2, markerscale=3,
                                  handlelength=0.9, handletextpad=0.3,
                                  borderaxespad=0.1, labelspacing=0.25)
                else:
                    ax.text(0.5, 0.5, "no cell has enough\nblocks for a z",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=6, color="0.4")
                ax.set_xlabel("pairs used per cell")
                ax.set_ylabel("per-cell z")
                if r == 0:
                    ax.set_title("Individual significance vs depth", pad=24)


            # The mark names the row on whichever column is drawn first.
            # Offset in inches: column (i) has no y ticks, the others do.
            off = 0.16 if columns[0] == 0 else 0.5
            row_axes[0].text(-off / (unit * _COL_WIDTH[columns[0]]), 0.5,
                             MARK_LABEL.get(mark, mark), rotation=90,
                             transform=row_axes[0].transAxes, ha="center",
                             va="center", color=col, fontsize=7,
                             fontweight="bold")

        # Letter and title are placed in inches, not figure fractions, so they
        # sit the same distance in on a one-column panel as on three.
        if letter:
            fig.text(0.05 / width, 0.985, letter, fontsize=9, fontweight="bold",
                     ha="left", va="top")
        if title:
            fig.text((0.22 if letter else 0.05) / width, 0.985, title,
                     fontsize=8, fontweight="bold", ha="left", va="top")
        if simulated:
            # Top right on the wide figure; on a narrow one the title fills
            # that line, so the stamp goes along the bottom edge instead.
            if width >= 4.0:
                fig.text(0.985, 0.985, SIM_STAMP, fontsize=6.5,
                         fontweight="bold", color="#C0392B", ha="right",
                         va="top")
            else:
                fig.text(0.05 / width, 0.004, SIM_STAMP, fontsize=6.5,
                         fontweight="bold", color="#C0392B", ha="left",
                         va="bottom")

        if out_prefix:
            for fmt in formats:
                fig.savefig(f"{out_prefix}.{fmt}",
                            dpi=600 if fmt == "png" else None,
                            bbox_inches="tight")
    return fig


def plot_layouts(df: pd.DataFrame, out_prefix: str, group_order=None,
                 group_colors=None, formats=("pdf", "png"),
                 layouts=("full", "main", "ED")) -> dict:
    """Write each layout to ``<out_prefix><suffix>.<fmt>``; returns the figures.

    The main and Extended Data panels are drawn from the same table by the same
    code as the full panel, so the three can never disagree.
    """
    import matplotlib.pyplot as plt
    figs = {}
    for name in layouts:
        columns, letter, title, suffix = LAYOUTS[name]
        figs[name] = plot(df, group_order=group_order,
                          group_colors=group_colors,
                          out_prefix=f"{out_prefix}{suffix}", formats=formats,
                          columns=columns, letter=letter, title=title)
        plt.close(figs[name])
    return figs
