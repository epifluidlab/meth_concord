#!/usr/bin/env python3
"""Per-bin scatter of GCH-GCH against HCG-HCG concordance.

One dot per genomic bin: x is the bin's GCH concordance, y is its HCG
concordance, each averaged over the cells the two contexts share.  Fits an OLS
line and reports Pearson and Spearman correlation with p-values in the caption.

Answers whether accessibility coordination and methylation coordination pick
out the SAME regions -- a bin-level question, which is why the cells are
collapsed rather than correlated within bins.

TWO COMPANION VIEWS, because collapsing the cells first can hide the answer
--------------------------------------------------------------------------
* PER-BIN ACROSS-CELL CORRELATION (--per-bin-correlation, on by default).  For
  each bin, the paired (GCH, HCG) values across cells give one Pearson r and
  one Spearman rho; the genome-wide density of those values says whether the
  two marks move together WITHIN bins.  A bin whose GCH and HCG track each
  other cell by cell averages to the same two numbers as a bin where they do
  not, so the scatter cannot see this, and averaging over cell types is
  precisely what removes the variation it is made of.  A permuted-cell null is
  drawn alongside: cells are shuffled per bin WITHIN batch, so a batch x bin
  effect shared by the two contexts stays in the null rather than counting as
  signal.
* STATE FLOW (--state-flow, on by default for D).  Each bin gets a GCH state
  (co-open / mixed / co-closed) and an HCG state (co-unmethylated / mixed /
  co-methylated), drawn as two stacked bars with ribbons between them, so the
  percentage keeping the canonical pairing -- open with unmethylated, closed
  with methylated -- and the percentage switching are read off directly.  A
  state is assigned only when a bin's cells agree on a sign (one-sample t,
  BH-adjusted); otherwise it is 'mixed'.

Give it the D pair, the C pair, or both; one figure per pair:

    python scripts/gch_hcg_concordance_scatter.py \\
        --gch-d out/context_GCH/05_concordance/concordance_directional.parquet \\
        --hcg-d out/context_HCG/05_concordance/concordance_directional.parquet \\
        --gch-c out/context_GCH/05_concordance/concordance_coordination.parquet \\
        --hcg-c out/context_HCG/05_concordance/concordance_coordination.parquet \\
        --output-dir out/10_gch_vs_hcg

Cells are intersected before averaging so the two axes summarize the same
population; bins are intersected so every dot exists on both.  Bins with fewer
than ``--min-cells`` observed cells in either context are dropped rather than
averaged from a handful of values.

BATCH EFFECTS ON A BIN-LEVEL SCATTER
------------------------------------
Sharing cells and bins between the axes removes one confound but not all of
them, and the correction that helps here is NOT the one the cell-level
pipelines use:

* A per-cell global shift (cell i offset in every bin) survives the average as
  a single constant added to every bin.  A constant shifts neither Pearson nor
  Spearman, so this class of batch effect is already harmless -- which is the
  part the "same cells" intuition gets right.
* A batch x bin interaction does NOT cancel: it adds a per-bin offset that
  varies across bins, so it moves the dots.  Worse, it is plausibly SHARED
  between the two contexts -- both are read off the same libraries, so a bin
  that is badly covered in one batch is badly covered for GCH and HCG alike --
  and shared technical structure inflates the very correlation being measured.

Harmony is the wrong instrument for this figure: it aligns cells in a
low-dimensional embedding, and there is no way back from a harmonized
embedding to a per-bin value.  ``--batch-correction center`` instead subtracts
each batch's per-bin mean before aggregating, which is the operation that
matches a bin-level summary.

``--per-batch-check`` is the stronger evidence: it recomputes the correlation
inside each batch separately.  If r is similar in every batch, no
between-batch effect can be producing it, and no correction is needed at all.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("gch_hcg_scatter")

STAT_STYLE = {
    "D": ("#5F4BB6", "directional coordination  D"),
    "C": ("#1F8A7B", "coordination magnitude  C"),
}


def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    df = (pd.read_parquet(p) if p.suffix in (".parquet", ".pq")
          else pd.read_csv(p, sep="\t", index_col=0))
    return df.astype(float)


def _drop_chroms(df: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    if not exclude:
        return df
    return df[[c for c in df.columns
               if str(c).split(":", 1)[0] not in exclude]]


def batch_series(samples_path: str, cell_column: str, batch_column: str):
    """cell -> batch label (composite when several columns are named)."""
    samples = pd.read_csv(samples_path, sep="\t")
    if cell_column not in samples.columns:
        raise SystemExit(f"--cell-column {cell_column!r} not in {samples_path}")
    idx = samples.set_index(cell_column)
    cols = [c.strip() for c in batch_column.split(",") if c.strip()]
    missing = [c for c in cols if c not in idx.columns]
    if missing:
        raise SystemExit(f"--batch-column not in {samples_path}: {missing}")
    return idx[cols[0]] if len(cols) == 1 else idx[cols]


def per_batch_correlations(gch, hcg, batch, how, min_cells) -> pd.DataFrame:
    """Recompute the correlation independently inside each batch.

    A between-batch effect cannot manufacture agreement that is already
    present within every batch, so consistent within-batch r is stronger
    evidence than any correction.
    """
    from nome_hic_concordance.batch_correction import (_coerce_batch_df,
                                                       _composite_label)
    cells = gch.index.intersection(hcg.index)
    lab = _composite_label(_coerce_batch_df(batch, cells))
    rows = []
    for level, idx in lab.groupby(lab).groups.items():
        sub = [c for c in idx if c in cells]
        if len(sub) < max(3, min_cells):
            logger.info(f"  batch {level!r}: {len(sub)} cells — skipped.")
            continue
        d = bin_summary(gch.loc[sub], hcg.loc[sub], how,
                        min_cells=min(min_cells, len(sub)))
        s = correlations(d["gch"].to_numpy(), d["hcg"].to_numpy())
        rows.append({"batch": str(level), "n_cells": len(sub), **s})
        logger.info(f"  batch {level!r} ({len(sub)} cells): "
                    f"Pearson r = {s['pearson_r']:.3f}, "
                    f"Spearman ρ = {s['spearman_rho']:.3f}, n = {s['n']} bins")
    return pd.DataFrame(rows)


def bin_summary(gch: pd.DataFrame, hcg: pd.DataFrame, how: str,
                min_cells: int) -> pd.DataFrame:
    """Collapse both matrices to one value per shared bin.

    Cells are intersected first: averaging different cell sets per axis would
    make the two coordinates summaries of different populations.
    """
    cells = gch.index.intersection(hcg.index)
    bins = gch.columns.intersection(hcg.columns)
    if len(cells) == 0 or len(bins) == 0:
        raise SystemExit(
            f"No overlap: {len(cells)} shared cells, {len(bins)} shared bins. "
            f"Check the two contexts come from the same run and bin grid.")
    g = gch.loc[cells, bins]
    h = hcg.loc[cells, bins]
    agg = (lambda d: d.mean(axis=0)) if how == "mean" else \
          (lambda d: d.median(axis=0))
    out = pd.DataFrame({
        "bin": bins,
        "gch": agg(g).to_numpy(),
        "hcg": agg(h).to_numpy(),
        "n_cells_gch": g.notna().sum(axis=0).to_numpy(),
        "n_cells_hcg": h.notna().sum(axis=0).to_numpy(),
    })
    keep = ((out["n_cells_gch"] >= min_cells)
            & (out["n_cells_hcg"] >= min_cells)
            & np.isfinite(out["gch"]) & np.isfinite(out["hcg"]))
    logger.info(f"{len(cells)} shared cells, {len(bins)} shared bins; "
                f"{int(keep.sum())} bins pass min_cells={min_cells}.")
    return out.loc[keep].reset_index(drop=True)


def _aligned(gch: pd.DataFrame, hcg: pd.DataFrame):
    cells = gch.index.intersection(hcg.index)
    bins = gch.columns.intersection(hcg.columns)
    if len(cells) == 0 or len(bins) == 0:
        raise SystemExit(
            f"No overlap: {len(cells)} shared cells, {len(bins)} shared bins. "
            f"Check the two contexts come from the same run and bin grid.")
    return gch.loc[cells, bins], hcg.loc[cells, bins]


def _column_pearson(G: np.ndarray, H: np.ndarray, min_cells: int):
    """Per-COLUMN Pearson r over the pairwise-complete rows → (r, n).

    Masked to the cells where both contexts are observed, per bin: a cell that
    covers a bin in GCH but not HCG carries no paired information there, and
    dropping it column-wise keeps more bins than dropping it everywhere.
    """
    ok = np.isfinite(G) & np.isfinite(H)
    x = np.where(ok, G, 0.0)
    y = np.where(ok, H, 0.0)
    n = ok.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        sx, sy = x.sum(axis=0), y.sum(axis=0)
        cov = (x * y).sum(axis=0) / n - (sx / n) * (sy / n)
        vx = (x * x).sum(axis=0) / n - (sx / n) ** 2
        vy = (y * y).sum(axis=0) / n - (sy / n) ** 2
        r = cov / np.sqrt(vx * vy)
    bad = (n < max(min_cells, 3)) | ~np.isfinite(r) | (vx <= 0) | (vy <= 0)
    return np.where(bad, np.nan, np.clip(r, -1.0, 1.0)), n


def _corr_p(r: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Two-sided p for a correlation, via the usual t transform."""
    from scipy.stats import t as _t
    with np.errstate(invalid="ignore", divide="ignore"):
        tt = r * np.sqrt((n - 2) / np.maximum(1.0 - r ** 2, 1e-300))
        p = 2.0 * _t.sf(np.abs(tt), np.maximum(n - 2, 1))
    return np.where(np.isfinite(r) & (n > 2), p, np.nan)


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if idx.size == 0:
        return out
    pv = p[idx]
    order = np.argsort(pv)
    m = pv.size
    q = pv[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[idx[order]] = np.clip(q, 0.0, 1.0)
    return out


def _shuffle_rows(H: np.ndarray, groups, rng):
    """Permute cells independently per bin, within batch when one is given.

    Within batch, because the confound this null has to absorb is a batch x bin
    effect shared by the two contexts: both are read off the same libraries, so
    a bin poorly covered in one batch is poorly covered for GCH and HCG alike,
    and that alone correlates them across cells.  A global shuffle would break
    that structure too and hand back a null that only describes finite-n noise.
    """
    out = np.empty_like(H)
    for rows in groups:
        block = H[rows]
        for j in range(block.shape[1]):
            block[:, j] = block[rng.permutation(block.shape[0]), j]
        out[rows] = block
    return out


def per_bin_correlations(gch, hcg, min_cells, batch=None, n_null=1, seed=42):
    """Correlate the two contexts ACROSS CELLS, one value per bin.

    The bin-level scatter collapses each bin to a mean first, which answers a
    different question and can hide this one entirely: bins whose GCH and HCG
    track each other cell by cell average to the same two numbers as bins where
    they do not.  Averaging over cell types is exactly what removes the
    variation this correlation is made of.

    Returns (table, null_pearson, null_spearman).  The null permutes cells
    independently per bin, so it carries the same bin count, the same per-bin
    cell counts and the same marginal distributions as the observed curve, and
    differs from it only in whether a cell's GCH sits beside its own HCG.
    """
    G, H = _aligned(gch, hcg)
    bins = list(G.columns)
    # Rank ONCE, on the pairwise-complete mask, so Spearman is Pearson on ranks
    # with exactly the same cells the Pearson used.
    both = G.notna() & H.notna()
    Gm, Hm = G.where(both), H.where(both)
    Gr = Gm.rank(axis=0).to_numpy()
    Hr = Hm.rank(axis=0).to_numpy()
    Ga, Ha = Gm.to_numpy(), Hm.to_numpy()

    r, n = _column_pearson(Ga, Ha, min_cells)
    rho, _ = _column_pearson(Gr, Hr, min_cells)
    p_r, p_s = _corr_p(r, n), _corr_p(rho, n)
    table = pd.DataFrame({
        "bin": bins, "n_cells": n.astype(int),
        "pearson_r": r, "pearson_p": p_r, "pearson_q": _bh(p_r),
        "spearman_rho": rho, "spearman_p": p_s, "spearman_q": _bh(p_s)})

    rng = np.random.default_rng(seed)
    if batch is not None:
        from nome_hic_concordance.batch_correction import (_coerce_batch_df,
                                                           _composite_label)
        lab = _composite_label(_coerce_batch_df(batch, G.index)).to_numpy()
        groups = [np.flatnonzero(lab == u) for u in pd.unique(lab)]
    else:
        groups = [np.arange(len(G.index))]
    null_r, null_rho = [], []
    for _ in range(max(int(n_null), 0)):
        Hs = _shuffle_rows(Ha.copy(), groups, rng)
        Hrs = _shuffle_rows(Hr.copy(), groups, rng)
        null_r.append(_column_pearson(Ga, Hs, min_cells)[0])
        null_rho.append(_column_pearson(Gr, Hrs, min_cells)[0])
    cat = lambda v: (np.concatenate(v) if v else np.array([]))   # noqa: E731
    return table, cat(null_r), cat(null_rho)


def correlations(x: np.ndarray, y: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr
    if x.size < 3:
        nan = float("nan")
        return {"pearson_r": nan, "pearson_p": nan,
                "spearman_rho": nan, "spearman_p": nan,
                "slope": nan, "intercept": nan, "n": int(x.size)}
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp),
            "slope": float(slope), "intercept": float(intercept),
            "n": int(x.size)}


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "P = n/a"
    if p == 0.0:                      # underflow, not a real zero
        return "P < 1e-308"
    if p < 1e-4:
        return f"P = {p:.1e}"
    return f"P = {p:.4f}"


def caption(stats: dict) -> str:
    return (f"Pearson r = {stats['pearson_r']:.3f} ({_fmt_p(stats['pearson_p'])})"
            f"    ·    Spearman ρ = {stats['spearman_rho']:.3f} "
            f"({_fmt_p(stats['spearman_p'])})    ·    n = {stats['n']} bins")


def scatter(df, stats, stat_key, out_path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    color, pretty = STAT_STYLE[stat_key]
    x = df["gch"].to_numpy()
    y = df["hcg"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    if args.density and x.size >= 200:
        # Colour by local density: at several thousand bins a flat scatter is
        # a solid blob and the shape of the relationship is invisible.
        from scipy.stats import gaussian_kde
        try:
            z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
            order = np.argsort(z)     # densest last, so they are not buried
            ax.scatter(x[order], y[order], c=z[order], s=9, cmap="viridis",
                       linewidths=0, alpha=0.85)
        except Exception:             # singular covariance -> fall back
            ax.scatter(x, y, s=9, color=color, alpha=0.35, linewidths=0)
    else:
        ax.scatter(x, y, s=12, color=color, alpha=0.45, linewidths=0)

    xs = np.linspace(np.min(x), np.max(x), 100)
    ax.plot(xs, stats["slope"] * xs + stats["intercept"], color="#B0453A",
            lw=1.8, label=f"OLS fit  (slope = {stats['slope']:.3f})")
    ax.axhline(0, color="#C9CED6", lw=0.8, zorder=0)
    ax.axvline(0, color="#C9CED6", lw=0.8, zorder=0)

    ax.set_xlabel(f"GCH–GCH concordance   {pretty}   (mean over cells)")
    ax.set_ylabel(f"HCG–HCG concordance   {pretty}   (mean over cells)")
    ax.set_title(f"{stat_key}: GCH vs HCG long-range concordance per bin",
                 fontsize=11.5, pad=18)
    ax.annotate(caption(stats), xy=(0.5, 1.0), xytext=(0, 5),
                xycoords="axes fraction", textcoords="offset points",
                ha="center", va="bottom", fontsize=8.4, color="#4B5563")
    ax.legend(fontsize=8.5, frameon=False, loc="best")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _kde_curve(v, grid):
    from scipy.stats import gaussian_kde
    v = v[np.isfinite(v)]
    if v.size < 5 or np.allclose(v, v[0]):
        return None
    try:
        return gaussian_kde(v)(grid)
    except Exception:
        return None


def per_bin_density(table, null_r, null_rho, stat_key, out_path, args):
    """Genome-wide distribution of the per-bin across-cell correlation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    _, pretty = STAT_STYLE[stat_key]
    grid = np.linspace(-1, 1, 401)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)
    panels = (("pearson_r", null_r, "Pearson r", "#5F4BB6"),
              ("spearman_rho", null_rho, "Spearman ρ", "#1F8A7B"))
    lines = []
    for ax, (col, null, label, color) in zip(axes, panels):
        v = table[col].to_numpy()
        v = v[np.isfinite(v)]
        obs = _kde_curve(v, grid)
        if obs is not None:
            ax.fill_between(grid, obs, color=color, alpha=0.25, lw=0)
            ax.plot(grid, obs, color=color, lw=1.8, label=f"observed (n={v.size})")
        nul = _kde_curve(null, grid) if null.size else None
        if nul is not None:
            ax.plot(grid, nul, color="#6B7280", lw=1.4, ls="--",
                    label="cells permuted within batch"
                          if args.batch_column else "cells permuted")
        med = float(np.median(v)) if v.size else np.nan
        ax.axvline(0, color="#C9CED6", lw=0.9, zorder=0)
        ax.axvline(med, color=color, lw=1.2, ls=":")
        pos = float(np.mean(v > 0)) if v.size else np.nan
        sig_col = "pearson_q" if col == "pearson_r" else "spearman_q"
        q = table[sig_col].to_numpy()
        up = int(np.sum((q < 0.05) & (table[col].to_numpy() > 0)))
        dn = int(np.sum((q < 0.05) & (table[col].to_numpy() < 0)))
        ax.set_xlabel(f"{label} across cells, per bin")
        ax.set_title(f"{label}   median = {med:+.3f},   {100 * pos:.1f}% > 0",
                     fontsize=10.5)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        lines.append(f"{label}: median {med:+.3f}, {100 * pos:.1f}% positive, "
                     f"{up} bins q<0.05 positive vs {dn} negative")
    axes[0].set_ylabel("density (bins)")
    fig.suptitle(f"{stat_key}: GCH–GCH vs HCG–HCG {pretty}, correlated across "
                 f"cells within each bin", fontsize=11.5)
    fig.text(0.5, -0.02, "   ·   ".join(lines), ha="center", fontsize=8.4,
             color="#4B5563")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return lines


# ------------------------------------------------------------- state flow ----

_GCH_STATES = ("co-open", "mixed", "co-closed")
_HCG_STATES = ("co-unmethylated", "mixed", "co-methylated")
_STATE_COLOR = {"co-open": "#D1495B", "co-closed": "#4A6FA5",
                "co-unmethylated": "#3E8E7E", "co-methylated": "#8C5E8A",
                "mixed": "#B8BDC4"}
# The pairing the biology predicts: open chromatin is unmethylated.
_CANONICAL = {("co-open", "co-unmethylated"), ("co-closed", "co-methylated")}


def bin_states(M: pd.DataFrame, names, args):
    """Label each bin by the sign its cells agree on: (labels, mean, q).

    Under the default 'fdr' rule a bin is called only when its cells actually
    agree — a one-sample t of the per-cell values against zero, BH-adjusted
    over bins — so 'mixed' means "no consistent direction", not "close to an
    arbitrary line".  --state-rule threshold restores the arbitrary line for
    anyone who wants a fixed effect size instead.
    """
    from scipy.stats import t as _t
    X = M.to_numpy(float)
    n = np.isfinite(X).sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1)
        tt = mean / (sd / np.sqrt(np.maximum(n, 1)))
        p = 2.0 * _t.sf(np.abs(tt), np.maximum(n - 1, 1))
    p = np.where((n >= max(args.min_cells, 3)) & np.isfinite(sd) & (sd > 0),
                 p, np.nan)
    q = _bh(p)
    pos, neg = names
    lab = np.full(X.shape[1], "mixed", dtype=object)
    if args.state_rule == "fdr":
        called = np.isfinite(q) & (q < args.state_fdr)
    else:
        called = np.isfinite(mean) & (np.abs(mean) > args.state_threshold)
    lab[called & (mean > 0)] = pos
    lab[called & (mean < 0)] = neg
    lab[~np.isfinite(mean)] = "unusable"
    return lab, mean, q


def flow_table(g_lab, h_lab) -> pd.DataFrame:
    keep = (g_lab != "unusable") & (h_lab != "unusable")
    tab = pd.crosstab(pd.Series(g_lab[keep], name="gch_state"),
                      pd.Series(h_lab[keep], name="hcg_state"))
    return tab.reindex(index=_GCH_STATES, columns=_HCG_STATES).fillna(0).astype(int)


def flow_figure(tab: pd.DataFrame, stat_key, out_path, args):
    """Two stacked bars with ribbons: where each GCH state goes in HCG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    total = int(tab.to_numpy().sum())
    if total == 0:
        logger.warning(f"{stat_key}: no bins classified — no flow figure.")
        return None
    left = tab.sum(axis=1)
    right = tab.sum(axis=0)
    gap = 0.02 * total

    def spans(counts, order):
        y, out = 0.0, {}
        for k in order:
            out[k] = (y, y + float(counts[k]))
            y += float(counts[k]) + gap
        return out, y - gap

    lspan, ltop = spans(left, _GCH_STATES)
    rspan, rtop = spans(right, _HCG_STATES)
    top = max(ltop, rtop)

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    bar_w, x0, x1 = 0.12, 0.0, 1.0
    for span, order, xa in ((lspan, _GCH_STATES, x0), (rspan, _HCG_STATES, x1)):
        for k in order:
            lo, hi = span[k]
            if hi <= lo:
                continue
            ax.add_patch(plt.Rectangle((xa - bar_w / 2, lo), bar_w, hi - lo,
                                       color=_STATE_COLOR[k], lw=0))
            ax.text(xa + (-0.10 if xa == x0 else 0.10), (lo + hi) / 2,
                    f"{k}\n{int(hi - lo):,} ({100 * (hi - lo) / total:.1f}%)",
                    ha="right" if xa == x0 else "left", va="center",
                    fontsize=8.6)

    t = np.linspace(0, 1, 120)
    ease = 0.5 * (1 - np.cos(np.pi * t))          # smooth, monotone
    lcur = {k: lspan[k][0] for k in _GCH_STATES}
    rcur = {k: rspan[k][0] for k in _HCG_STATES}
    flows = [(g, h, int(tab.loc[g, h])) for g in _GCH_STATES for h in _HCG_STATES]
    drawn = [f for f in flows if f[2] > 0]
    for i, (g, h, cnt) in enumerate(drawn):
        y0a, y0b = lcur[g], lcur[g] + cnt
        y1a, y1b = rcur[h], rcur[h] + cnt
        lcur[g], rcur[h] = y0b, y1b
        xs = x0 + bar_w / 2 + (x1 - x0 - bar_w) * t
        canon = (g, h) in _CANONICAL
        ax.fill_between(xs, y0a + (y1a - y0a) * ease, y0b + (y1b - y0b) * ease,
                        color=_STATE_COLOR[g], alpha=0.55 if canon else 0.22,
                        lw=0.6 if canon else 0.0,
                        edgecolor="#374151" if canon else "none")
        if cnt / total > 0.02:
            # Labels are staggered ALONG the ribbon rather than all pinned to
            # the midline: crossing ribbons pass through the same point there,
            # and two percentages printed on top of each other are worse than
            # none.  y follows the ribbon's own path at its label position.
            ti = 0.28 + 0.44 * (i / max(len(drawn) - 1, 1))
            e = 0.5 * (1 - np.cos(np.pi * ti))
            ymid = (y0a + y0b) / 2 + ((y1a + y1b) / 2 - (y0a + y0b) / 2) * e
            ax.text(x0 + bar_w / 2 + (x1 - x0 - bar_w) * ti, ymid,
                    f"{100 * cnt / total:.1f}%", ha="center", va="center",
                    fontsize=7.6, color="#111827")

    canon_n = sum(c for g, h, c in flows if (g, h) in _CANONICAL)
    from scipy.stats import chi2_contingency
    sub = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
    if sub.shape[0] > 1 and sub.shape[1] > 1:
        chi2, pval, _, _ = chi2_contingency(sub.to_numpy())
        v = float(np.sqrt(chi2 / (total * (min(sub.shape) - 1))))
        assoc = f"Cramér's V = {v:.3f}, {_fmt_p(pval)}"
    else:
        assoc = "association undefined (one state occupied)"
    ax.set_xlim(-0.42, 1.42)
    # Inverted so the states read top-to-bottom in the order they are declared
    # (co-open first), which is also the order the legend text implies.
    ax.set_ylim(top * 1.02, -0.02 * top)
    ax.axis("off")
    ax.set_title(f"{stat_key}: GCH–GCH state → HCG–HCG state, per bin",
                 fontsize=11.5, pad=16)
    ax.annotate(f"{total:,} bins   ·   canonical (co-open→co-unmethylated, "
                f"co-closed→co-methylated) = {100 * canon_n / total:.1f}%"
                f"   ·   {assoc}",
                xy=(0.5, 1.0), xytext=(0, 4), xycoords="axes fraction",
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8.4, color="#4B5563")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"n_bins": total, "canonical_frac": canon_n / total,
            "association": assoc}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gch-d"), ap.add_argument("--hcg-d")
    ap.add_argument("--gch-c"), ap.add_argument("--hcg-c")
    ap.add_argument("--aggregate", choices=["mean", "median"], default="mean",
                    help="How each bin's cells are collapsed to one value.")
    ap.add_argument("--min-cells", type=int, default=10,
                    help="Minimum observed cells per bin, per context (10).")
    ap.add_argument("--exclude-chrom", default="")
    ap.add_argument("--samples", default=None,
                    help="Sample sheet, required by --batch-correction and "
                         "--per-batch-check.")
    ap.add_argument("--cell-column", default="cell_name")
    ap.add_argument("--batch-column", default=None,
                    help="Comma-separated batch columns in --samples.")
    ap.add_argument("--batch-correction",
                    choices=["none", "center", "zscore", "combat"],
                    default="none",
                    help="Subtract each batch's PER-BIN effect from both "
                         "matrices before aggregating.  'center' is the right "
                         "default if you want one.  Harmony is deliberately "
                         "absent: it corrects a cell embedding, which cannot "
                         "be mapped back to a per-bin value.")
    ap.add_argument("--per-batch-check", action="store_true",
                    help="Also recompute the correlation within each batch "
                         "separately and write it to <stat>_per_batch.tsv.  "
                         "Consistent within-batch r is stronger evidence than "
                         "correction that batch is not driving the result.")
    ap.add_argument("--per-bin-correlation", choices=["on", "off"],
                    default="on",
                    help="Also correlate the two contexts ACROSS CELLS within "
                         "each bin, one r per bin, and plot the genome-wide "
                         "density of those values.  The scatter averages the "
                         "cells away first, which answers a different question "
                         "and hides this one: a bin whose GCH and HCG track "
                         "each other cell by cell averages to the same two "
                         "numbers as a bin where they do not.")
    ap.add_argument("--null-permutations", type=int, default=1,
                    help="Permuted-cell null curves for that density (1; 0 "
                         "disables).  Cells are shuffled per bin WITHIN batch "
                         "when --batch-column is given, so a batch x bin "
                         "effect shared by the two contexts stays in the null "
                         "instead of being counted as signal.")
    ap.add_argument("--state-flow", choices=["on", "off"], default="on",
                    help="Also emit the two stacked bars with linking ribbons: "
                         "each bin's GCH state (co-open / mixed / co-closed) "
                         "against its HCG state (co-unmethylated / mixed / "
                         "co-methylated), so the percentage that keeps the "
                         "canonical pairing and the percentage that switches "
                         "are read off directly.  Directional only — for C the "
                         "sign is concordant vs anti-concordant, not open vs "
                         "closed, so it is skipped unless --state-flow-c on.")
    ap.add_argument("--state-flow-c", choices=["on", "off"], default="off",
                    help="Draw the state flow for C too.  Off by default "
                         "because C's sign does not mean open vs closed.")
    ap.add_argument("--state-rule", choices=["fdr", "threshold"], default="fdr",
                    help="How a bin earns a direction.  'fdr': a one-sample t "
                         "of its cells against zero, BH-adjusted over bins, so "
                         "'mixed' means the cells do not agree rather than "
                         "'near an arbitrary line'.  'threshold': |mean| > "
                         "--state-threshold.")
    ap.add_argument("--state-fdr", type=float, default=0.05)
    ap.add_argument("--state-threshold", type=float, default=0.0,
                    help="Only with --state-rule threshold.")
    ap.add_argument("--density", action="store_true", default=True,
                    help="Colour points by local density (default on).")
    ap.add_argument("--no-density", dest="density", action="store_false")
    ap.add_argument("--plot-format", choices=["pdf", "png", "svg"],
                    default="pdf")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    pairs = [(k, g, h) for k, g, h in
             (("D", args.gch_d, args.hcg_d), ("C", args.gch_c, args.hcg_c))
             if g and h]
    if not pairs:
        raise SystemExit("Give at least one complete pair: "
                         "--gch-d/--hcg-d and/or --gch-c/--hcg-c.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    excl = {c.strip() for c in args.exclude_chrom.split(",") if c.strip()}

    needs_batch = args.batch_correction != "none" or args.per_batch_check
    batch = None
    if needs_batch:
        if not (args.samples and args.batch_column):
            raise SystemExit("--batch-correction / --per-batch-check need "
                             "--samples and --batch-column.")
        batch = batch_series(args.samples, args.cell_column, args.batch_column)

    for stat_key, gch_path, hcg_path in pairs:
        logger.info(f"--- {stat_key} ---")
        gch = _drop_chroms(_load(gch_path), excl)
        hcg = _drop_chroms(_load(hcg_path), excl)

        if args.per_batch_check:
            logger.info(f"{stat_key}: per-batch correlations")
            pb = per_batch_correlations(gch, hcg, batch, args.aggregate,
                                        args.min_cells)
            pb.to_csv(out_dir / f"{stat_key}_gch_vs_hcg_per_batch.tsv",
                      sep="\t", index=False)

        if args.batch_correction != "none":
            from nome_hic_concordance import batch_correction as _bc
            gch = _bc.apply_batch_correction(
                gch, batch.reindex(gch.index), method=args.batch_correction)
            hcg = _bc.apply_batch_correction(
                hcg, batch.reindex(hcg.index), method=args.batch_correction)
            logger.info(f"{stat_key}: applied per-bin '{args.batch_correction}' "
                        f"batch correction to both matrices.")

        df = bin_summary(gch, hcg, args.aggregate, args.min_cells)
        stats = correlations(df["gch"].to_numpy(), df["hcg"].to_numpy())
        logger.info(f"{stat_key}: {caption(stats)}")

        df.to_csv(out_dir / f"{stat_key}_gch_vs_hcg_per_bin.tsv",
                  sep="\t", index=False)
        pd.DataFrame([{"statistic": stat_key, **stats}]).to_csv(
            out_dir / f"{stat_key}_gch_vs_hcg_stats.tsv", sep="\t", index=False)
        path = out_dir / f"{stat_key}_gch_vs_hcg_scatter.{args.plot_format}"
        scatter(df, stats, stat_key, path, args)
        logger.info(f"wrote {path}")

        # Across-cell correlation within each bin: the variation the scatter
        # averages away is exactly what this is made of.
        if args.per_bin_correlation == "on":
            pb, nr, nrho = per_bin_correlations(
                gch, hcg, args.min_cells, batch=batch,
                n_null=args.null_permutations)
            pb.to_csv(out_dir / f"{stat_key}_per_bin_across_cell_corr.tsv",
                      sep="\t", index=False)
            dpath = (out_dir /
                     f"{stat_key}_per_bin_across_cell_density.{args.plot_format}")
            for line in per_bin_density(pb, nr, nrho, stat_key, dpath, args):
                logger.info(f"{stat_key}: {line}")
            logger.info(f"wrote {dpath}")

        want_flow = (args.state_flow == "on"
                     and (stat_key == "D" or args.state_flow_c == "on"))
        if want_flow:
            G, H = _aligned(gch, hcg)
            g_lab, g_mean, g_q = bin_states(G, ("co-open", "co-closed"), args)
            h_lab, h_mean, h_q = bin_states(
                H, ("co-methylated", "co-unmethylated"), args)
            tab = flow_table(g_lab, h_lab)
            tab.to_csv(out_dir / f"{stat_key}_state_flow_counts.tsv", sep="\t")
            pd.DataFrame({"bin": G.columns, "gch_state": g_lab,
                          "gch_mean": g_mean, "gch_q": g_q,
                          "hcg_state": h_lab, "hcg_mean": h_mean,
                          "hcg_q": h_q}).to_csv(
                out_dir / f"{stat_key}_state_flow_per_bin.tsv", sep="\t",
                index=False)
            fpath = out_dir / f"{stat_key}_state_flow.{args.plot_format}"
            info = flow_figure(tab, stat_key, fpath, args)
            if info:
                logger.info(f"{stat_key}: {info['n_bins']:,} bins classified, "
                            f"canonical {100 * info['canonical_frac']:.1f}%, "
                            f"{info['association']} → {fpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
