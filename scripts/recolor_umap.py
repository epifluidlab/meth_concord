#!/usr/bin/env python3
"""Recolor existing UMAPs by technical factors, to check for bias.

nome_hic_concordance and nome_hic_compare both save the embedding as
``<prefix>_coordinates.tsv`` (columns: cell, UMAP1, UMAP2, group).  This reads
those coordinates back and recolors them by columns from a metadata file --
without re-running UMAP, so the geometry is byte-for-byte the one in the paper
figure and only the colour changes.

The point is diagnostic: if a technical factor (sequencing depth, mapping rate,
prep date, sample) paints a smooth gradient across the embedding, the clustering
may be tracking that factor rather than biology.  Each panel therefore also
prints how strongly the factor aligns with the embedding -- Spearman rho against
each UMAP axis for a numeric factor, and the between-group share of position
variance (an R^2) for a categorical one -- so "looks fine" is backed by a number.

Cells are matched on a metadata column (``--cell-column``, default ``cell``).
Coordinates with no metadata row are drawn in light grey rather than dropped, so
a coverage gap is visible instead of silently shrinking the cloud.

Usage
-----
    python scripts/recolor_umap.py \\
        --coords out/context_GCH/06_umap/umap_quantile_svd10_skip1_coordinates.tsv \\
        --coords out/context_HCG/06_umap/umap_quantile_svd10_skip1_coordinates.tsv \\
        --coords cmp/umap_gch_residual_coordinates.tsv \\
        --metadata metadata_with_cell_type.txt --cell-column cell \\
        --color-by sample --color-by library_prep_date \\
        --color-by mapq30rate --color-by 'log10(mapq30num)' \\
        --color-by noncpg_chrM --color-by cis1kbrate \\
        --output-dir output/umap_technical
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_TRANSFORMS = {
    "log10": lambda s: np.log10(s.where(s > 0)),
    "log2": lambda s: np.log2(s.where(s > 0)),
    "log1p": lambda s: np.log1p(s.where(s >= 0)),
}
_FUNC_RE = re.compile(r"^\s*(log10|log2|log1p)\s*\(\s*(.+?)\s*\)\s*$")


def parse_color_spec(spec):
    """``col`` | ``log10(col)`` | ``label=expr`` -> (label, column, transform).

    The transform is applied to the raw column; ``label`` is what the file name
    and colourbar show.
    """
    label = None
    if "=" in spec and not _FUNC_RE.match(spec):
        label, spec = spec.split("=", 1)
        label = label.strip()
    spec = spec.strip()
    m = _FUNC_RE.match(spec)
    if m:
        fn_name, column = m.group(1), m.group(2).strip()
        return (label or f"{fn_name}({column})", column, _TRANSFORMS[fn_name])
    return (label or spec, spec, None)


def _safe_name(label):
    return re.sub(r"[^A-Za-z0-9._+-]", "_", str(label)).strip("._") or "factor"


def load_metadata(path, cell_column):
    """Metadata keyed by cell.  Tolerates the leading unnamed index column."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    if cell_column not in df.columns:
        sys.exit(f"--cell-column {cell_column!r} not in {path}\n"
                 f"  columns: {df.columns.tolist()}")
    df[cell_column] = df[cell_column].astype(str)
    dups = int(df[cell_column].duplicated().sum())
    if dups:
        print(f"  NOTE: {dups} duplicate {cell_column!r} rows in metadata — "
              f"keeping the first of each.", file=sys.stderr)
        df = df.drop_duplicates(cell_column, keep="first")
    return df.set_index(cell_column)


def _as_numeric(values):
    """Coerce to float; return (numeric, n_failed) so a mixed column is caught."""
    num = pd.to_numeric(values, errors="coerce")
    failed = int(num.isna().sum() - values.isna().sum())
    return num, failed


def is_categorical(values, transform, force, max_levels):
    if force == "categorical":
        return True
    if force == "continuous":
        return False
    if transform is not None:
        return False                       # a log of a category is nonsense
    num, failed = _as_numeric(values)
    if failed > 0:                         # any non-numeric entry -> categorical
        return True
    clean = num.dropna()
    if clean.empty:
        return True
    # Few distinct INTEGER codes (cluster ids, plate numbers) are categorical;
    # a float is continuous however few values happen to be observed, so a rate
    # with a handful of distinct values is not mistaken for a category.
    all_integer = bool(np.all(clean == np.floor(clean)))
    return all_integer and clean.nunique() <= max_levels


def assoc_continuous(coords, v):
    """Spearman rho of a numeric factor against each UMAP axis."""
    from scipy.stats import spearmanr
    ok = np.isfinite(v)
    if ok.sum() < 10:
        return None
    r1 = spearmanr(coords["UMAP1"][ok], v[ok]).correlation
    r2 = spearmanr(coords["UMAP2"][ok], v[ok]).correlation
    return float(r1), float(r2)


def assoc_categorical(coords, labels):
    """Between-group share of embedding-position variance (a 2-D eta^2/R^2).

    High means the categories occupy distinct regions of the embedding -- i.e.
    the factor is spatially structured, which for a TECHNICAL factor is the
    warning sign.
    """
    ok = labels.notna().to_numpy()
    if ok.sum() < 10 or labels[ok].nunique() < 2:
        return None
    P = coords.loc[ok, ["UMAP1", "UMAP2"]].to_numpy()
    lab = labels[ok].to_numpy()
    tot = ((P - P.mean(0)) ** 2).sum()
    within = 0.0
    for g in np.unique(lab):
        Q = P[lab == g]
        within += ((Q - Q.mean(0)) ** 2).sum()
    return float(1.0 - within / tot) if tot > 0 else None


def _plot(coords, values, label, categorical, out, args, note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 7))
    missing = values.isna().to_numpy()
    if missing.any():
        ax.scatter(coords["UMAP1"][missing], coords["UMAP2"][missing],
                   s=10, c="0.85", edgecolors="none", zorder=1,
                   label=f"n/a ({int(missing.sum())})")

    vis = ~missing
    if categorical:
        levels = pd.Index(sorted(values[vis].unique(), key=str))
        cmap = plt.get_cmap("tab20" if len(levels) > 10 else "tab10")
        for i, lv in enumerate(levels):
            m = (values == lv).to_numpy()
            ax.scatter(coords["UMAP1"][m], coords["UMAP2"][m], s=14,
                       color=cmap(i % cmap.N), edgecolors="none", alpha=0.8,
                       label=str(lv), zorder=2)
        ncol = 1 if len(levels) <= 20 else 2
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
                  ncol=ncol, title=label, title_fontsize=8)
    else:
        v = values[vis].to_numpy(float)
        lo, hi = np.nanpercentile(v, [2, 98]) if v.size else (0, 1)
        if lo == hi:
            lo, hi = (lo - 0.5, hi + 0.5)
        sc = ax.scatter(coords["UMAP1"][vis], coords["UMAP2"][vis], s=14, c=v,
                        cmap="viridis", vmin=lo, vmax=hi, edgecolors="none",
                        alpha=0.85, zorder=2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label + "  (colour clipped to 2–98th pct)", fontsize=9)
        if missing.any():
            ax.legend(loc="lower right", fontsize=7)

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"{out.stem}\ncoloured by {label}", fontsize=10)
    if note:
        ax.text(0.5, -0.09, note, transform=ax.transAxes, ha="center",
                va="top", fontsize=8, color="0.35")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords", action="append", required=True,
                    help="A *_coordinates.tsv from nome_hic_concordance or "
                         "nome_hic_compare. Repeatable.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--cell-column", default="cell",
                    help="Metadata column matching the coordinates' 'cell'.")
    ap.add_argument("--color-by", action="append", required=True,
                    metavar="COL|log10(COL)|LABEL=EXPR",
                    help="Factor to colour by. Repeatable. log10()/log2()/"
                         "log1p() wrap a column; LABEL= renames the panel.")
    ap.add_argument("--type", action="append", default=None,
                    metavar="COL=categorical|continuous",
                    help="Force a factor's type instead of auto-detecting.")
    ap.add_argument("--max-levels", type=int, default=20,
                    help="A numeric column with more distinct values than this "
                         "is treated as continuous.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--plot-format", choices=["png", "pdf", "svg"],
                    default="pdf")
    args = ap.parse_args()

    forced = {}
    for t in (args.type or []):
        if "=" not in t:
            sys.exit(f"--type must be COL=categorical|continuous, got {t!r}")
        k, v = t.split("=", 1)
        if v not in ("categorical", "continuous"):
            sys.exit(f"--type value must be categorical|continuous, got {v!r}")
        forced[k.strip()] = v

    meta = load_metadata(args.metadata, args.cell_column)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [parse_color_spec(s) for s in args.color_by]
    for _label, col, _fn in specs:
        if col not in meta.columns:
            sys.exit(f"column {col!r} not in metadata "
                     f"(columns: {meta.columns.tolist()})")

    n_written = 0
    for coords_path in args.coords:
        cp = Path(coords_path)
        if not cp.exists():
            print(f"  skip {cp}: not found", file=sys.stderr)
            continue
        coords = pd.read_csv(cp, sep="\t")
        if not {"cell", "UMAP1", "UMAP2"} <= set(coords.columns):
            print(f"  skip {cp}: not a *_coordinates.tsv (need cell/UMAP1/UMAP2)",
                  file=sys.stderr)
            continue
        coords["cell"] = coords["cell"].astype(str)
        matched = coords["cell"].isin(meta.index).sum()
        print(f"\n{cp.name}: {len(coords)} cells, {matched} matched to metadata")
        stem = cp.name.replace("_coordinates.tsv", "") or cp.stem

        for label, col, fn in specs:
            raw = meta[col].reindex(coords["cell"])
            values = raw.reset_index(drop=True)
            cat = is_categorical(values, fn, forced.get(col), args.max_levels)
            if not cat:
                num, _ = _as_numeric(values)
                values = fn(num) if fn is not None else num
            elif fn is not None:
                print(f"  {label}: transform ignored (categorical)",
                      file=sys.stderr)

            if cat:
                r2 = assoc_categorical(coords, values)
                note = (f"between-group share of UMAP-position variance: "
                        f"R² = {r2:.3f}" if r2 is not None
                        else "association: too few labelled cells")
            else:
                rr = assoc_continuous(coords, values.to_numpy(float))
                note = (f"Spearman vs UMAP1/2: ρ = {rr[0]:+.2f} / "
                        f"{rr[1]:+.2f}" if rr is not None
                        else "association: too few finite values")

            out = out_dir / f"{stem}.by_{_safe_name(label)}.{args.plot_format}"
            _plot(coords, values, label, cat, out, args, note)
            n_written += 1
            kind = "categorical" if cat else "continuous"
            print(f"  {label:<22} {kind:<12} {note}")
            print(f"      -> {out}")

    if n_written == 0:
        sys.exit("no panels written — check --coords paths and --cell-column.")
    print(f"\nwrote {n_written} panels to {out_dir}")
    print("Read them: a technical factor forming a SMOOTH GRADIENT across the "
          "embedding\n  (high Spearman rho, or high categorical R²) means "
          "the clustering may track\n  that factor.  Scattered colour = the "
          "factor is not driving the layout.")


if __name__ == "__main__":
    main()
