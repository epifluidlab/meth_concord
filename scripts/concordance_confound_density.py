#!/usr/bin/env python3
"""Overlaid densities of per-bin correlation between concordance and its 1D/3D
baselines, before and after regressing the baselines out.

Four curves on one axes, for one context (GCH or HCG):

    corr(C, 1D)                 coordination vs mean accessibility/methylation
    corr(D, 3D)                 directional  vs compartment score
    corr(C residual, 1D)        the same, after the baselines are regressed out
    corr(D residual, 3D)

Hue encodes the pairing (C/1D vs D/3D), line style encodes raw vs residual, so
the before/after shift is readable per pairing.

READ THE RESIDUAL CURVES CORRECTLY
----------------------------------
Under the default ``--residual-mode both`` the residual is the ordinary
least-squares residual of the concordance on BOTH baselines, fit per bin over
cells.  An OLS residual is orthogonal to its own regressors, so those two
curves collapse onto a spike at r = 0 *by construction*.  That is a quality
check ("the linear 1D/3D dependence really is gone"), not a result -- exactly
how nome_hic_compare's own decorrelation panel frames it.  The informative
half of the figure is the raw pair: how much correlation there was to remove.

``--residual-mode other`` regresses out only the OTHER baseline (C on 3D, then
correlate with 1D; D on 1D, then correlate with 3D).  Those residuals are not
orthogonal to the baseline being plotted, so the curves are non-trivial and
answer a real question: does C still track accessibility once compartment is
accounted for, and vice versa.

Correlations are computed per bin across cells by default (the convention used
by nome_hic_compare); ``--axis cell`` flips it to per cell across bins.

Example
-------
    python scripts/concordance_confound_density.py \\
        --concordance-c out/context_GCH/05_concordance/concordance_coordination.parquet \\
        --concordance-d out/context_GCH/05_concordance/concordance_directional.parquet \\
        --baseline-1d  out/context_GCH/05_concordance/accessibility_mean.parquet \\
        --baseline-3d  out/compartments/compartment_scores.parquet \\
        --context GCH --output-dir out/confound_density

Run it again with the HCG paths for the second figure.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("confound_density")

C_RAW, C_RES = "#1F8A7B", "#7FCFC4"      # coordination / 1D pairing
D_RAW, D_RES = "#5F4BB6", "#B3A8E3"      # directional  / 3D pairing
# Phase-shifted dashes: under --residual-mode both the two residual curves are
# delta spikes at exactly the same x, so identical dashes would hide one of
# them entirely.  Offsetting the phase interleaves them.
C_RES_DASH = (0, (7, 5))
D_RES_DASH = (6, (7, 5))


# --------------------------------------------------------------- data prep --

def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    df = (pd.read_parquet(p) if p.suffix in (".parquet", ".pq")
          else pd.read_csv(p, sep="\t", index_col=0))
    return df.astype(float)


def _align(mats: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every matrix to the shared cells and bins, same order.

    Intersecting rather than reindex-with-NaN keeps the per-bin fits honest:
    a bin present in only one input would otherwise contribute an all-NaN
    column that silently drops out later.
    """
    cells = sorted(set.intersection(*(set(m.index) for m in mats.values())))
    bins = sorted(set.intersection(*(set(m.columns) for m in mats.values())))
    if not cells or not bins:
        raise SystemExit(
            f"No overlap between inputs: {len(cells)} shared cells, "
            f"{len(bins)} shared bins.  Check that the matrices come from the "
            f"same run (and that the compartment matrix uses the same bins).")
    logger.info(f"Aligned on {len(cells)} cells x {len(bins)} bins.")
    return {k: m.reindex(index=cells, columns=bins) for k, m in mats.items()}


# ------------------------------------------------------------- statistics --

def _rank(a: np.ndarray) -> np.ndarray:
    """Column-wise rank transform, NaN-preserving (for Spearman)."""
    from scipy.stats import rankdata
    out = np.full_like(a, np.nan, dtype=float)
    for j in range(a.shape[1]):
        col = a[:, j]
        m = np.isfinite(col)
        if m.sum():
            out[m, j] = rankdata(col[m])
    return out


def _corr_1d(y: np.ndarray, x: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 3:
        return np.nan
    yy, xx = y[m], x[m]
    if np.std(yy) < 1e-12 or np.std(xx) < 1e-12:
        return np.nan
    return float(np.corrcoef(yy, xx)[0, 1])


def per_column_corr(Y: np.ndarray, X: np.ndarray,
                    min_obs: int = 20) -> np.ndarray:
    """corr(Y[:, j], X[:, j]) for every column j."""
    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        if (np.isfinite(Y[:, j]) & np.isfinite(X[:, j])).sum() >= min_obs:
            out[j] = _corr_1d(Y[:, j], X[:, j])
    return out


def residualize(Y: np.ndarray, preds: list[np.ndarray],
                min_obs: int = 20) -> np.ndarray:
    """Per-column OLS residual of Y on ``preds`` (with intercept).

    Columns with too few complete observations are returned all-NaN rather
    than un-residualized, so they drop out of the density instead of leaking
    in as if the baselines had been removed.
    """
    R = np.full_like(Y, np.nan, dtype=float)
    for j in range(Y.shape[1]):
        y = Y[:, j]
        cols = [p[:, j] for p in preds]
        Xj = np.column_stack(cols)
        good = np.isfinite(y) & np.all(np.isfinite(Xj), axis=1)
        if good.sum() < max(min_obs, Xj.shape[1] + 2):
            continue
        if np.std(y[good]) < 1e-12:
            continue
        A = np.column_stack([np.ones(good.sum()), Xj[good]])
        beta, *_ = np.linalg.lstsq(A, y[good], rcond=None)
        R[good, j] = y[good] - A @ beta
    return R


# ----------------------------------------------------------------- plotting --

def _density(ax, vals, label, color, ls, fill):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 0
    med = float(np.median(np.abs(v)))
    lab = f"{label}   (median |r| = {med:.3f}, n = {v.size})"
    grid = np.linspace(-1, 1, 512)
    if np.std(v) < 1e-9:
        # A degenerate spike (the by-construction residual case) has no KDE.
        # Draw it as a line at the value -- thin, so a "there is nothing here"
        # result does not out-shout the real densities.  The two residual
        # curves land on exactly the same x, so their dash patterns are
        # phase-shifted (see _RES_DASH) to keep both visible.
        ax.axvline(float(v[0]), color=color, ls=ls, lw=1.6, alpha=0.9,
                   label=lab)
        return v.size
    from scipy.stats import gaussian_kde
    dens = gaussian_kde(v)(grid)
    ax.plot(grid, dens, color=color, ls=ls, lw=2.0, label=lab)
    if fill:
        ax.fill_between(grid, dens, color=color, alpha=0.11, lw=0)
    return v.size


def build_figure(curves, context, axis, residual_mode, plot_hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    fig, ax = plt.subplots(figsize=(8.8, 5.9))
    for (label, vals, color, ls, fill) in curves:
        if plot_hist:
            v = np.asarray(vals, float)
            v = v[np.isfinite(v)]
            if v.size:
                ax.hist(v, bins=np.linspace(-1, 1, 61), density=True,
                        histtype="step", color=color, lw=0.9, alpha=0.5)
        _density(ax, vals, label, color, ls, fill)

    ax.axvline(0, color="#9AA0A6", lw=0.9, zorder=0)
    unit = "bin" if axis == "bin" else "cell"
    other = "cells" if axis == "bin" else "bins"
    ax.set_xlabel(f"Pearson r per {unit}  (across {other})")
    ax.set_ylabel("density")
    # Title gets extra pad so the mode note can sit under it as a subtitle.
    ax.set_title(f"{context}: concordance vs 1D / 3D baselines, "
                 f"before and after regression", fontsize=11.5, pad=20)
    note = ("residual = OLS on BOTH baselines → orthogonal by construction, "
            "so r = 0 here is a QC, not a result"
            if residual_mode == "both" else
            "residual = OLS on the OTHER baseline only → not orthogonal, "
            "so these curves are informative")
    ax.annotate(note, xy=(0.5, 1.0), xytext=(0, 5), xycoords="axes fraction",
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8, color="#6B7280", style="italic")
    ax.set_xlim(-1, 1)
    # Legend below the axes: with four long labels an in-axes legend occludes
    # whichever side the densities happen to fall on.
    ax.legend(fontsize=8.4, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.135), handlelength=2.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- main --

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concordance-c", required=True,
                    help="concordance_coordination.parquet (C)")
    ap.add_argument("--concordance-d", required=True,
                    help="concordance_directional.parquet (D)")
    ap.add_argument("--baseline-1d", required=True,
                    help="accessibility_mean.parquet (GCH) or the HCG "
                         "equivalent — the 1D level signal")
    ap.add_argument("--baseline-3d", required=True,
                    help="compartment_scores.parquet — the 3D signal")
    ap.add_argument("--context", default="GCH", help="Label for the title.")
    ap.add_argument("--axis", choices=["bin", "cell"], default="bin",
                    help="'bin' (default) correlates across cells within each "
                         "bin, matching nome_hic_compare; 'cell' correlates "
                         "across bins within each cell.")
    ap.add_argument("--residual-mode", choices=["both", "other"],
                    default="both",
                    help="'both' (default, as in nome_hic_compare) regresses "
                         "out 1D AND 3D — the residual is then orthogonal to "
                         "both, so those curves sit at r=0 BY CONSTRUCTION and "
                         "are a QC, not a result.  'other' regresses out only "
                         "the other baseline, giving non-trivial curves.")
    ap.add_argument("--method", choices=["pearson", "spearman"],
                    default="pearson")
    ap.add_argument("--min-obs", type=int, default=20,
                    help="Minimum complete observations per bin/cell (20).")
    ap.add_argument("--hist", action="store_true",
                    help="Overlay step histograms under the density curves.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--stem", default=None,
                    help="Output basename (default: "
                         "<context>_confound_density).")
    ap.add_argument("--plot-format", choices=["pdf", "png", "svg"],
                    default="pdf")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    mats = _align({
        "C": _load(args.concordance_c),
        "D": _load(args.concordance_d),
        "1D": _load(args.baseline_1d),
        "3D": _load(args.baseline_3d),
    })
    # Work column-wise throughout; transpose once when the unit is the cell.
    arrs = {k: (m.to_numpy() if args.axis == "bin" else m.to_numpy().T)
            for k, m in mats.items()}
    if args.method == "spearman":
        arrs = {k: _rank(a) for k, a in arrs.items()}

    C, D, B1, B3 = arrs["C"], arrs["D"], arrs["1D"], arrs["3D"]
    if args.residual_mode == "both":
        c_res = residualize(C, [B1, B3], args.min_obs)
        d_res = residualize(D, [B1, B3], args.min_obs)
    else:                              # regress out only the OTHER baseline
        c_res = residualize(C, [B3], args.min_obs)
        d_res = residualize(D, [B1], args.min_obs)

    tag = "1D+3D" if args.residual_mode == "both" else "the other baseline"
    curves = [
        ("C  vs  1D", per_column_corr(C, B1, args.min_obs), C_RAW, "-", True),
        ("D  vs  3D", per_column_corr(D, B3, args.min_obs), D_RAW, "-", True),
        (f"C residual ({tag})  vs  1D",
         per_column_corr(c_res, B1, args.min_obs), C_RES, C_RES_DASH, False),
        (f"D residual ({tag})  vs  3D",
         per_column_corr(d_res, B3, args.min_obs), D_RES, D_RES_DASH, False),
    ]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"{args.context}_confound_density"

    unit = mats["C"].columns if args.axis == "bin" else mats["C"].index
    pd.DataFrame({args.axis: unit,
                  **{lab: vals for lab, vals, *_ in curves}}
                 ).to_csv(out / f"{stem}.tsv", sep="\t", index=False)

    for label, vals, *_ in curves:
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        logger.info(f"{label:44s} median |r| = "
                    f"{np.median(np.abs(v)) if v.size else float('nan'):.3f} "
                    f"(n = {v.size})")

    fig = build_figure(curves, args.context, args.axis, args.residual_mode,
                       args.hist)
    path = out / f"{stem}.{args.plot_format}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info(f"wrote {path}")
    logger.info(f"wrote {out / f'{stem}.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
