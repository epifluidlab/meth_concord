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

    for stat_key, gch_path, hcg_path in pairs:
        logger.info(f"--- {stat_key} ---")
        gch = _drop_chroms(_load(gch_path), excl)
        hcg = _drop_chroms(_load(hcg_path), excl)
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
