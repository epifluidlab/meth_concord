#!/usr/bin/env python3
"""Is a concordance matrix one per-cell number repeated across bins?

A marker heatmap whose columns are solid stripes -- each cell the same colour
in nearly every selected bin -- is not showing bins that differ between cell
types.  It is showing ONE per-cell factor that the bins share, picked up by
the cell-type ranking because that factor differs between cell types.  The
per-cell rank in the normalisation removes a per-cell offset, but not a
per-cell STRENGTH of a genome-wide bin pattern: a cell with more information
ranks the bins in the common order more cleanly, and per-bin quantile
normalisation then puts it at the same end of every bin -- one colour in the
bins on one side of the pattern, the other colour in the rest.

This finds that factor -- the first component of the matrix normalised
exactly as celltype_marker_bins.py normalises it before drawing -- and asks
what it tracks:

  per cell   log10 linked pairs and the fraction of bins with any pair
             (pair_counts_per_bin.parquet, the RAW pre-imputation counts),
             the mean level and 2p(1-p) (accessibility_mean.parquet), the
             cell's mean raw value, its share of |value| > 0.5 and its share
             of exact zeros.  |value| > 0.5 is the --concordance-formula mle
             edge: an imputed linked n of exactly 0 sets the observed rate to
             0 while the expected side keeps its value, so C = -E, around
             -0.6 for HCG.  An exact 0 after RWR means nothing reached that
             cell in that bin -- both sides empty -- which happens when the
             cell graph splits into disconnected pieces.  Either one matters
             far beyond its own bins: the per-cell rank makes every value of
             a cell depend on all its others, so a cell with many ties or
             extremes has every remaining bin shifted, which is a stripe.
  per bin    fraction of cells with any pair, mean level, mean raw value

Every per-cell association is reported twice: across all cells, and WITHIN
cell type (ranks, each cell type's mean removed from both sides).  A factor
that tracks depth within a cell type is technical however well it also
separates cell types; one that separates cell types and tracks nothing
within them is not, or not detectably so.

A heatmap without stripes is not cleared by that.  The level can also enter
bin by bin, and then it draws cell-type BLOCKS, which look like biology.  For
pure coordination r between two ends at level p,

    C = 2 r p(1-p)                     0.84x from p = 0.52 to 0.70
    L ~ r (1 - 2p) / (p(1-p))          11.5x over the same range

so at equal coordination L is set largely by the bin's level, and wherever
cell types differ in level -- most bins, for methylation -- L differs with
them.  (C's level term is a different one: a marginal mismatch d between the
linked and the expected ends adds 2d(2p-1) to C, 10x over that range, but
only 2d/(p(1-p)) to L, 1.2x.  Its weight follows the cell's genome-wide
level, which is a stripe.)  So, per bin, the statistic's cell-type means are
correlated with the level's cell-type means -- means, because a cell has
level data in too few bins to regress on cell by cell -- and set against two
references:

  other bins       the level of a DIFFERENT bin: what a genome-wide level
                   difference between cell types gives on its own
  permuted types   the level with cell-type labels shuffled per bin: chance

over all bins and over the --top-bins most cell-type-variable ones (the kind
a marker heatmap draws).  Same bin well above other bins means the
statistic's blocks follow each bin's own methylation.

None of this says whether the structure is COORDINATION.  The drawn null
does: step 5 builds it from as many random NON-contact end pairs as the cell
had contacts in that bin, and carries it through the same prior, imputation,
expected side and normalisation.  It keeps every cell's marginal, depth,
missingness and the co-state of ends that merely share a region -- all but
contact-specific coordination.  --null NAME=PATH pairs a matrix with its
null and every analysis reports kNN cell-type purity on its leading
components against chance.  The comparison that answers the question is
REAL against NULL: a matrix that separates cell types no better than its
null shows no cell-type structure of its own.

NAME_minus_null (real - null, cell-bin by cell-bin) is reported too, but its
purity is NOT evidence on its own whenever the matrices were imputed over a
cell graph.  Both were smoothed over the same graph, so their difference is
smoothed over it as well and takes on the graph's cell-type structure
whatever it contains -- RWR of pure noise does the same.  For sparse data
the difference also keeps the shared pattern of empty cell-bins (both sides
empty gives an exact 0 in both).  On the 250 kb mouse cortex run real -
null still scored 0.79 (HCG C) and 0.53 (GCH) while real and null tied
(0.82 / 0.82, 0.53 / 0.52).  A cell-type contrast that can tell coordination
from the graph has to be made without the graph: counts pooled by cell type.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance.utils import (  # noqa: E402
    bin_label_chrom, chrom_excluded, parse_exclude_chroms)

#: Above this |C| a value is not a coordination estimate but the mle edge.
EDGE = 0.5

CELL_COVARIATES = ("log10_pairs", "observed_frac", "mean_level",
                   "two_p_q", "mean_value", "edge_frac", "zero_frac")
BIN_COVARIATES = ("bin_coverage", "bin_level", "bin_mean_value")


def _sidecar(matrix_path: Path, filename: str) -> Path | None:
    for d in (matrix_path.parent, matrix_path.parent.parent):
        p = d / filename
        if p.exists():
            return p
    return None


def _aligned(path: Path | None, cells, bins) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_parquet(path)
    df.index = df.index.astype(str)
    return df.reindex(index=cells, columns=bins)


def cell_covariates(raw: pd.DataFrame, pairs: pd.DataFrame | None,
                    level: pd.DataFrame | None) -> pd.DataFrame:
    """One row per cell; a covariate whose sidecar is missing is NaN."""
    X = raw.to_numpy(float)
    out = pd.DataFrame(index=raw.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        if pairs is not None:
            P = np.nan_to_num(pairs.to_numpy(float), nan=0.0)
            out["log10_pairs"] = np.log10(1.0 + P.sum(1))
            out["observed_frac"] = (P > 0).mean(1)
        else:
            out["log10_pairs"] = np.nan
            out["observed_frac"] = np.nan
        if level is not None:
            L = level.to_numpy(float)
            out["mean_level"] = np.nanmean(L, 1)
            out["two_p_q"] = np.nanmean(2.0 * L * (1.0 - L), 1)
        else:
            out["mean_level"] = np.nan
            out["two_p_q"] = np.nan
        out["mean_value"] = np.nanmean(X, 1)
        fin = np.isfinite(X)
        n_fin = np.maximum(fin.sum(1), 1)
        out["edge_frac"] = (np.abs(np.where(fin, X, 0.0)) > EDGE).sum(1) / n_fin
        out["zero_frac"] = (fin & (X == 0.0)).sum(1) / n_fin
    return out


def bin_covariates(raw: pd.DataFrame, pairs: pd.DataFrame | None,
                   level: pd.DataFrame | None) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.columns)
    with np.errstate(invalid="ignore"):
        out["bin_coverage"] = (
            (np.nan_to_num(pairs.to_numpy(float), nan=0.0) > 0).mean(0)
            if pairs is not None else np.nan)
        out["bin_level"] = (np.nanmean(level.to_numpy(float), 0)
                            if level is not None else np.nan)
        out["bin_mean_value"] = np.nanmean(raw.to_numpy(float), 0)
    return out


def leading_factors(normed: pd.DataFrame, n_pcs: int = 5, seed: int = 0):
    """Column-standardise then SVD: the bins are compared on the scale the
    heatmap shows them (z per bin).  Returns scores, loadings, var ratio."""
    from sklearn.decomposition import TruncatedSVD
    X = normed.to_numpy(np.float64)
    sd = X.std(0, keepdims=True)
    Z = (X - X.mean(0, keepdims=True)) / np.where(sd < 1e-12, 1.0, sd)
    k = max(1, min(n_pcs, Z.shape[0] - 1, Z.shape[1] - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed).fit(Z)
    # Variance shares of the TOTAL, so PC1's number means what it says.
    total = float((Z ** 2).sum())
    share = (svd.singular_values_ ** 2) / total if total > 0 else \
        np.zeros(k)
    return svd.transform(Z), svd.components_, share


def _colcorr(A: np.ndarray, B: np.ndarray, min_n: int = 4) -> np.ndarray:
    """Pearson r per column over the rows finite in both; NaN under min_n."""
    ok = np.isfinite(A) & np.isfinite(B)
    n = ok.sum(0)
    A0, B0 = np.where(ok, A, 0.0), np.where(ok, B, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = A0.sum(0) / n, B0.sum(0) / n
        da, db = np.where(ok, A - ma, 0.0), np.where(ok, B - mb, 0.0)
        r = (da * db).sum(0) / np.sqrt((da ** 2).sum(0) * (db ** 2).sum(0))
    return np.where(n >= min_n, r, np.nan)


def type_variable_bins(normed: pd.DataFrame, cell_type, top: int) -> list:
    """The ``top`` bins with the largest between-cell-type share (eta^2) of
    the normalised values -- the kind of bin a marker heatmap selects."""
    X = normed.to_numpy(float)
    g = pd.Series(np.asarray(cell_type))
    means = pd.DataFrame(X).groupby(g.values).transform("mean").to_numpy()
    mu = X.mean(0, keepdims=True)
    tot = ((X - mu) ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        e2 = np.where(tot > 0, ((means - mu) ** 2).sum(0) / tot, 0.0)
    order = np.argsort(-e2)[:max(1, int(top))]
    return [normed.columns[i] for i in order]


def level_tracking(raw: pd.DataFrame, level: pd.DataFrame | None,
                   cell_type, top_bins: list, seed: int = 0) -> dict:
    """Does the statistic's cell-type pattern in a bin follow that bin's level?

    Returns ``{scope: {same_abs_r, same_r, other_bins_abs_r,
    permuted_types_abs_r, n_bins}}`` for scope ``all`` and ``top``, medians
    over bins.  Empty when there is no level or fewer than 4 cell types.
    """
    if level is None:
        return {}
    g = np.asarray(cell_type).astype(str)
    S = raw.groupby(g).mean()
    Lv = level.reindex(index=raw.index, columns=raw.columns).groupby(g).mean()
    if len(S) < 4:
        return {}
    Y, X = S.to_numpy(float), Lv.reindex(S.index).to_numpy(float)
    rng = np.random.default_rng(seed)
    nb = X.shape[1]
    # Another bin's level: one random non-zero offset, so no bin meets itself.
    other = (np.arange(nb) + int(rng.integers(1, max(nb, 2)))) % nb
    # Cell-type labels shuffled independently in every bin.
    perm = np.argsort(rng.random(X.shape), axis=0)
    r_same = _colcorr(X, Y)
    r_other = _colcorr(X[:, other], Y)
    r_perm = _colcorr(np.take_along_axis(X, perm, axis=0), Y)
    pos = {b: i for i, b in enumerate(S.columns)}
    top_idx = np.array([pos[b] for b in top_bins if b in pos], int)
    out = {}
    for scope, idx in (("all", np.arange(nb)), ("top", top_idx)):
        if not len(idx):
            continue
        out[scope] = {
            "same_abs_r": float(np.nanmedian(np.abs(r_same[idx]))),
            "same_r": float(np.nanmedian(r_same[idx])),
            "other_bins_abs_r": float(np.nanmedian(np.abs(r_other[idx]))),
            "permuted_types_abs_r": float(np.nanmedian(np.abs(r_perm[idx]))),
            "n_bins": int(np.isfinite(r_same[idx]).sum()),
        }
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy(float)


def spearman(a, b, groups=None) -> float:
    """Spearman rho; with ``groups``, on ranks centred within each group."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    if groups is not None:
        g = pd.Series(np.asarray(groups)[ok])
        ra = ra - pd.Series(ra).groupby(g.values).transform("mean").to_numpy()
        rb = rb - pd.Series(rb).groupby(g.values).transform("mean").to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def eta2(x, groups) -> float:
    """Share of x's variance between groups (one-way ANOVA R^2)."""
    x = np.asarray(x, float)
    s = pd.Series(x)
    ok = s.notna().to_numpy()
    s, g = s[ok], pd.Series(np.asarray(groups))[ok]
    tot = float(((s - s.mean()) ** 2).sum())
    if tot == 0:
        return float("nan")
    means = s.groupby(g.values).transform("mean")
    return float(((means - s.mean()) ** 2).sum() / tot)


def load_matrix(path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    raw.index = raw.index.astype(str)
    return raw


def minus(real: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    """real - null on the cells and bins both carry; NaN where either is."""
    cells = real.index.intersection(null.index)
    bins = real.columns.intersection(null.columns)
    return real.loc[cells, bins] - null.loc[cells, bins]


def analyse(name: str, path: str, cell_type: pd.Series,
            batches: pd.DataFrame, args, out_dir: Path,
            raw: pd.DataFrame | None = None):
    """Factor + covariates for one matrix -> (cells, bins, summary rows).

    ``raw`` overrides the matrix read from ``path`` (used for real - null);
    the step-5 sidecars are always looked up from ``path``.
    """
    from nome_hic_concordance.comparison import _prep_matrix
    from nome_hic_concordance.metrics import knn_chance, knn_purity
    mpath = Path(path)
    raw = load_matrix(mpath) if raw is None else raw.copy()
    n_matrix = raw.shape[0]
    raw = raw.loc[[c for c in raw.index if c in cell_type.index]]
    excl = parse_exclude_chroms(args.exclude_chrom)
    if excl:
        raw = raw[[b for b in raw.columns
                   if not chrom_excluded(bin_label_chrom(b), excl)]]
    cells, bins = list(raw.index), list(raw.columns)
    pairs = _aligned(_sidecar(mpath, "pair_counts_per_bin.parquet"),
                     cells, bins)
    level = _aligned(_sidecar(mpath, "accessibility_mean.parquet"),
                     cells, bins)

    # Normalised exactly as celltype_marker_bins.py does before it draws.
    normed = _prep_matrix(raw, name, str(out_dir), args.matrix_norm,
                          args.min_cell_coverage, args.per_cell_norm,
                          args.post_norm_recenter == "on")
    normed = normed.loc[cells]
    scores, loadings, share = leading_factors(normed, args.n_pcs)

    cov = cell_covariates(raw, pairs, level)
    pc1 = scores[:, 0]
    # The SVD sign is arbitrary: orient PC1 so that higher means a higher
    # mean raw value, which is the only reading the plots need.
    if spearman(pc1, cov["mean_value"]) < 0:
        pc1, loadings = -pc1, loadings.copy()
        loadings[0] = -loadings[0]
        scores = scores.copy()
        scores[:, 0] = pc1
    ct = cell_type.reindex(cells).astype(str).to_numpy()

    cells_df = cov.copy()
    cells_df.insert(0, "cell_type", ct)
    for j in range(scores.shape[1]):
        cells_df[f"PC{j + 1}"] = scores[:, j]
    for c in batches.columns:
        cells_df[c] = batches[c].reindex(cells).astype(str).to_numpy()
    cells_df.insert(0, "matrix", name)

    bins_df = bin_covariates(raw, pairs, level)
    retained = list(normed.columns)
    bins_df = bins_df.reindex(retained)
    bins_df["PC1_loading"] = loadings[0]
    bins_df.insert(0, "matrix", name)

    rows = [{"matrix": name, "kind": "shape", "covariate": "cells",
             "value": len(cells), "note": f"{n_matrix} in the matrix"},
            {"matrix": name, "kind": "shape", "covariate": "bins",
             "value": len(retained), "note": ""}]
    for j, s in enumerate(share):
        rows.append({"matrix": name, "kind": "variance_share",
                     "covariate": f"PC{j + 1}", "value": float(s),
                     "note": ""})
    rows.append({"matrix": name, "kind": "eta2", "covariate": "cell_type",
                 "value": eta2(pc1, ct), "note": "PC1 between cell types"})
    for c in batches.columns:
        rows.append({"matrix": name, "kind": "eta2", "covariate": c,
                     "value": eta2(pc1, cells_df[c].to_numpy()),
                     "note": "PC1 between batches"})
    for c in CELL_COVARIATES:
        rows.append({"matrix": name, "kind": "rho_all", "covariate": c,
                     "value": spearman(pc1, cov[c]), "note": "PC1 vs cell"})
        rows.append({"matrix": name, "kind": "rho_within_type",
                     "covariate": c,
                     "value": spearman(pc1, cov[c], groups=ct),
                     "note": "PC1 vs cell, within cell type"})
    for c in BIN_COVARIATES:
        rows.append({"matrix": name, "kind": "rho_bins", "covariate": c,
                     "value": spearman(bins_df["PC1_loading"], bins_df[c]),
                     "note": "PC1 loading vs bin"})
    k_pcs = max(1, min(args.n_pcs, scores.shape[1]))
    rows.append({"matrix": name, "kind": "purity", "covariate": "knn_purity",
                 "value": knn_purity(scores[:, :k_pcs], ct),
                 "note": f"cell type, k=15 on {k_pcs} components"})
    rows.append({"matrix": name, "kind": "purity", "covariate": "knn_chance",
                 "value": knn_chance(ct), "note": "sum of squared shares"})
    top = type_variable_bins(normed, ct, args.top_bins)
    for scope, vals in level_tracking(raw.loc[:, retained], level, ct,
                                      top).items():
        label = ("all bins" if scope == "all"
                 else f"top {args.top_bins} cell-type-variable bins")
        for k, v in vals.items():
            rows.append({"matrix": name, "kind": f"level_tracking_{scope}",
                         "covariate": k, "value": v, "note": label})
    return cells_df, bins_df, rows


def verdict(summary: pd.DataFrame, name: str) -> str:
    s = summary[summary["matrix"] == name]

    def get(kind, cov):
        v = s[(s["kind"] == kind) & (s["covariate"] == cov)]["value"]
        return float(v.iloc[0]) if len(v) else float("nan")

    lines = [f"{name}: {int(get('shape', 'cells')):,} cells x "
             f"{int(get('shape', 'bins')):,} bins; PC1 "
             f"{100 * get('variance_share', 'PC1'):.1f}% of variance "
             f"(PC2 {100 * get('variance_share', 'PC2'):.1f}%)"]
    eta = s[s["kind"] == "eta2"]
    lines.append("  PC1 between groups (eta2): " + ", ".join(
        f"{r.covariate} {r.value:.2f}" for r in eta.itertuples()))
    if (s["kind"] == "purity").any():
        lines.append(f"  cell-type kNN purity {get('purity', 'knn_purity'):.2f}"
                     f" (chance {get('purity', 'knn_chance'):.2f})")
    within = s[s["kind"] == "rho_within_type"].dropna(subset=["value"])
    within = within.reindex(within["value"].abs().sort_values(
        ascending=False).index)
    lines.append("  within cell type, PC1 tracks: " + ", ".join(
        f"{r.covariate} {r.value:+.2f}" for r in within.itertuples()))
    bins = s[s["kind"] == "rho_bins"].dropna(subset=["value"])
    lines.append("  PC1 loading per bin tracks: " + ", ".join(
        f"{r.covariate} {r.value:+.2f}" for r in bins.itertuples()))
    for scope in ("top", "all"):
        k = f"level_tracking_{scope}"
        if not (s["kind"] == k).any():
            continue
        note = s[s["kind"] == k]["note"].iloc[0]
        lines.append(
            f"  cell-type pattern vs the bin's own level ({note}, "
            f"n={int(get(k, 'n_bins'))}): |r| same bin "
            f"{get(k, 'same_abs_r'):.2f} (signed {get(k, 'same_r'):+.2f}), "
            f"other bins {get(k, 'other_bins_abs_r'):.2f}, permuted types "
            f"{get(k, 'permuted_types_abs_r'):.2f}")
    return "\n".join(lines)


def plot(cells: pd.DataFrame, bins: pd.DataFrame, out_prefix: str,
         formats=("pdf", "png")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    names = list(dict.fromkeys(cells["matrix"]))
    panels = [("log10_pairs", "log10 linked pairs (raw)"),
              ("observed_frac", "fraction of bins with a pair"),
              ("mean_level", "mean level (accessibility / methylation)")]
    with plt.rc_context(figure_style.RC):
        fig, axes = plt.subplots(len(names), 4, squeeze=False,
                                 figsize=(10.5, 2.5 * len(names)))
        for r, nm in enumerate(names):
            d = cells[cells["matrix"] == nm]
            types = sorted(d["cell_type"].unique())
            cmap = plt.get_cmap("tab20")
            col = {t: cmap(i % 20) for i, t in enumerate(types)}
            for k, (c, lab) in enumerate(panels):
                ax = axes[r, k]
                if d[c].notna().any():
                    for t in types:
                        m = d["cell_type"] == t
                        ax.scatter(d.loc[m, c], d.loc[m, "PC1"], s=3,
                                   color=col[t], lw=0, alpha=0.7, label=t,
                                   rasterized=True)
                else:
                    ax.text(0.5, 0.5, "sidecar not found",
                            transform=ax.transAxes, ha="center")
                ax.set_xlabel(lab)
                if k == 0:
                    ax.set_ylabel(f"{nm}: PC1 score")
            b = bins[bins["matrix"] == nm]
            ax = axes[r, 3]
            if b["bin_coverage"].notna().any():
                ax.scatter(b["bin_coverage"], b["PC1_loading"], s=2,
                           color="0.3", lw=0, alpha=0.5, rasterized=True)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.set_xlabel("bin: fraction of cells with a pair")
            ax.set_ylabel("PC1 loading")
            if r == 0:
                axes[r, 0].legend(fontsize=5, markerscale=2, ncol=2,
                                  loc="best", frameon=False)
        fig.tight_layout()
        for fmt in formats:
            fig.savefig(f"{out_prefix}.{fmt}", dpi=300 if fmt == "png"
                        else None, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", action="append", required=True,
                    metavar="NAME=PATH")
    ap.add_argument("--null", action="append", default=None,
                    metavar="NAME=PATH",
                    help="The drawn null for --matrix NAME (e.g. "
                         "concordance_direction_logor_null.parquet).  Adds "
                         "NAME_null and NAME_minus_null to the analysis.")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-column", default=None,
                    help="Cell id column (default: cell_name, cell or "
                         "cell_id, whichever is present).")
    ap.add_argument("--cell-type-column", required=True)
    ap.add_argument("--batch-column", default="",
                    help="Comma list of batch columns to test PC1 against.")
    ap.add_argument("--exclude-chrom", action="append", default=None)
    ap.add_argument("--matrix-norm", default="quantile",
                    choices=["quantile", "zscore", "none"],
                    help="As celltype_marker_bins.py (quantile).")
    ap.add_argument("--per-cell-norm", default="rank",
                    choices=["off", "rank", "center"],
                    help="As celltype_marker_bins.py (rank).")
    ap.add_argument("--post-norm-recenter", default="on",
                    choices=["on", "off"])
    ap.add_argument("--min-cell-coverage", type=float, default=0.0)
    ap.add_argument("--n-pcs", type=int, default=5)
    ap.add_argument("--top-bins", type=int, default=500,
                    help="Cell-type-variable bins for the level test (500, "
                         "the size of the marker heatmap).")
    ap.add_argument("-o", "--output-prefix", required=True)
    ap.add_argument("--plot-format", default="pdf,png")
    args = ap.parse_args(argv)

    s = pd.read_csv(args.samples, sep="\t")
    cell_col = args.cell_column or next(
        (c for c in ("cell_name", "cell", "cell_id") if c in s.columns),
        s.columns[0])
    s[cell_col] = s[cell_col].astype(str)
    s = s.set_index(cell_col)
    if args.cell_type_column not in s.columns:
        sys.exit(f"--cell-type-column {args.cell_type_column!r} not in "
                 f"{args.samples}")
    cell_type = s[args.cell_type_column].dropna().astype(str)
    bcols = [c.strip() for c in args.batch_column.split(",") if c.strip()]
    missing = [c for c in bcols if c not in s.columns]
    if missing:
        sys.exit(f"--batch-column {missing} not in {args.samples}")
    batches = s[bcols]

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    all_cells, all_bins, rows = [], [], []
    def _specs(values, flag):
        out = {}
        for spec in values or []:
            if "=" not in spec:
                sys.exit(f"{flag} expects NAME=PATH, got {spec!r}")
            k, v = spec.split("=", 1)
            out[k] = v
        return out

    matrices = _specs(args.matrix, "--matrix")
    nulls = _specs(args.null, "--null")
    stray = [k for k in nulls if k not in matrices]
    if stray:
        sys.exit(f"--null given for {stray}, which no --matrix names")
    jobs = []
    for name, path in matrices.items():
        jobs.append((name, path, None))
        if name in nulls:
            null = load_matrix(nulls[name])
            jobs.append((f"{name}_null", path, null))
            jobs.append((f"{name}_minus_null", path,
                         minus(load_matrix(path), null)))
    for name, path, raw in jobs:
        c, b, r = analyse(name, path, cell_type, batches, args,
                          out_prefix.parent, raw=raw)
        all_cells.append(c)
        all_bins.append(b)
        rows += r
    cells = pd.concat(all_cells)
    bins = pd.concat(all_bins)
    summary = pd.DataFrame(rows)
    cells.to_csv(f"{out_prefix}.cells.tsv", sep="\t", index_label="cell")
    bins.to_csv(f"{out_prefix}.bins.tsv", sep="\t", index_label="bin")
    summary.to_csv(f"{out_prefix}.summary.tsv", sep="\t", index=False)
    plot(cells, bins, str(out_prefix),
         formats=[f for f in args.plot_format.split(",") if f])
    for name in dict.fromkeys(summary["matrix"]):
        print(verdict(summary, name))
    for name in nulls:
        def pur(m):
            v = summary[(summary["matrix"] == m) & (summary["kind"] == "purity")
                        & (summary["covariate"] == "knn_purity")]["value"]
            return float(v.iloc[0]) if len(v) else float("nan")
        chance = summary[(summary["matrix"] == name)
                         & (summary["covariate"] == "knn_chance")]["value"]
        print(f"{name} against its null: cell-type purity real "
              f"{pur(name):.2f}, null {pur(name + '_null'):.2f} (chance "
              f"{float(chance.iloc[0]):.2f}); real - null "
              f"{pur(name + '_minus_null'):.2f}, which is not evidence on its "
              f"own if both were imputed over a cell graph")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
