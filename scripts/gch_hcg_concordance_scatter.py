#!/usr/bin/env python3
"""Per-bin scatter of GCH-GCH against HCG-HCG concordance.

One dot per genomic bin: x is the bin's GCH concordance, y is its HCG
concordance, each averaged over the cells the two contexts share.  Fits an OLS
line and reports Pearson and Spearman correlation with p-values in the caption.

Answers whether accessibility coordination and methylation coordination pick
out the SAME regions -- a bin-level question, which is why the cells are
collapsed rather than correlated within bins.

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
