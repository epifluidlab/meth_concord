#!/usr/bin/env python3
"""Is the residual cell-type mixing in the signed UMAP a COVERAGE problem?

After the MLE switch the cell-type axis lands on PC0 (AUC ~0.88), but ~12%
of cells stay mis-separated and the UMAP shows a mixed cluster.  This
script tests whether those mis-separated cells are explained by per-cell
COVERAGE (or mean accessibility, or batch) — i.e. whether a cell-level QC
filter would clean it up — vs the cell-type feature simply being too weak
for them.

Method
------
1. SVD-reduce concordance_quantile.parquet to --n-svd components (the same
   space the UMAP consumes).
2. Leave-one-out cosine kNN (k=15) in that space; predict each cell's HCG
   label by neighbour majority vote; flag misclassified cells.
3. Load per-cell covariates:
     coverage           = sum over bins of pair_counts_per_bin (raw n_linked)
     mean_accessibility = row mean of accessibility_mean.parquet
     batch              = --batch-column of --samples (optional)
4. Compare misclassified vs correctly-separated cells:
     coverage / accessibility — medians + Mann-Whitney U
     batch                    — composition + chi-square
   Also correlate per-cell PC0 score with coverage.
5. Optional: if --umap-coords is given, write the UMAP colored by coverage,
   accessibility, batch, and correct/misclassified.

Verdict (printed)
-----------------
• Misclassified cells have significantly LOWER coverage  -> a cell-level
  coverage filter will help; a suggested threshold is printed.
• Coverage indistinguishable                              -> mixing is a
  feature-strength problem, not coverage; a filter won't fix it.

Usage
-----
    python scripts/umap_residual_mixing_diagnostic.py \\
        --concordance-dir output/signed_v8_final/context_GCH/05_concordance \\
        --hcg-labels /tmp/hcg_labels.umap_coords.tsv \\
        --samples samples.filtered.tsv --batch-column batch \\
        --n-svd 5 \\
        --umap-coords output/signed_v8_final/context_GCH/06_umap/umap_quantile_svd5_coordinates.tsv \\
        --out-prefix /tmp/mixing_diag
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("mixing_diag")


def _load_samples(path: str) -> pd.DataFrame:
    from nome_hic_concordance.utils import manifest_text

    first = manifest_text(path).readline().strip()   # first non-comment line
    first_field = first.split("\t")[0].strip().lower()
    has_header = first_field in (
        "cell_name", "cellname", "cell", "sample", "sample_name",
    )
    if has_header:
        df = pd.read_csv(manifest_text(path), sep="\t", dtype=str)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "cellname" in df.columns:
            df = df.rename(columns={"cellname": "cell_name"})
        if "cell" in df.columns and "cell_name" not in df.columns:
            df = df.rename(columns={"cell": "cell_name"})
        if "sample" in df.columns and "cell_name" not in df.columns:
            df = df.rename(columns={"sample": "cell_name"})
    else:
        df = pd.read_csv(manifest_text(path), sep="\t", header=None,
                         names=["cell_name", "bam_path", "group"], dtype=str)
    return df


def _loo_knn_predict(X: np.ndarray, y: np.ndarray, k: int = 15) -> np.ndarray:
    """Leave-one-out cosine kNN majority-vote prediction (self excluded)."""
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X, return_distance=True)
    idx = idx[:, 1:]  # drop self
    pred = np.empty(n, dtype=y.dtype)
    for i in range(n):
        vals, counts = np.unique(y[idx[i]], return_counts=True)
        pred[i] = vals[np.argmax(counts)]
    return pred


def _compare(name: str, values: pd.Series, mis: np.ndarray):
    from scipy.stats import mannwhitneyu
    a = values[~mis].dropna()
    b = values[mis].dropna()
    if len(a) < 3 or len(b) < 3:
        print(f"  {name}: too few cells to compare")
        return
    try:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        p = float("nan")
    print(f"  {name:>20s}: correct median={a.median():.4g} (n={len(a)}), "
          f"misclassified median={b.median():.4g} (n={len(b)}), "
          f"Mann-Whitney p={p:.2e}")
    return p


def _scatter(coords: pd.DataFrame, color: pd.Series, title: str, path: Path,
             categorical: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    c = color.reindex(coords["cell"]).values
    if categorical:
        cats = pd.Series(c).fillna("NA").astype(str)
        groups = sorted(cats.unique())
        cmap = plt.cm.tab10 if len(groups) <= 10 else plt.cm.tab20
        for g, col in zip(groups, cmap(np.linspace(0, 1, max(len(groups), 1)))):
            m = (cats == g).values
            ax.scatter(coords["UMAP1"][m], coords["UMAP2"][m], c=[col],
                       label=g, s=14, alpha=0.75, edgecolors="none")
        ax.legend(loc="best", fontsize=8)
    else:
        sc = ax.scatter(coords["UMAP1"], coords["UMAP2"], c=c, cmap="viridis",
                        s=14, alpha=0.8, edgecolors="none")
        fig.colorbar(sc, ax=ax)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concordance-dir", required=True)
    ap.add_argument("--hcg-labels", required=True)
    ap.add_argument("--samples", default=None)
    ap.add_argument("--batch-column", default=None)
    ap.add_argument("--n-svd", type=int, default=5,
                    help="SVD components for the kNN space (match the UMAP).")
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--umap-coords", default=None,
                    help="Optional UMAP *_coordinates.tsv for colored plots.")
    ap.add_argument("--out-prefix", default="/tmp/mixing_diag")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cdir = Path(args.concordance_dir)

    # ---- matrix + SVD space ----
    mat = pd.read_parquet(cdir / "concordance_quantile.parquet")
    from sklearn.decomposition import TruncatedSVD
    n_comp = max(1, min(args.n_svd, mat.shape[0] - 1, mat.shape[1]))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    emb = svd.fit_transform(mat.values)
    pc0 = pd.Series(emb[:, 0], index=mat.index, name="PC0")
    logger.info(f"matrix {mat.shape} -> SVD {emb.shape} "
                f"(PC0 var {svd.explained_variance_ratio_[0]*100:.2f}%)")

    # ---- HCG labels ----
    hcg = pd.read_csv(args.hcg_labels, sep="\t")
    hcg_map = dict(zip(hcg["cell"], hcg["predicted_label"]))
    keep = [c for c in mat.index if c in hcg_map]
    Xk = emb[[mat.index.get_loc(c) for c in keep]]
    y = np.array([str(hcg_map[c]) for c in keep])
    logger.info(f"HCG-labelled cells: {len(keep)} / {len(mat)}")

    # ---- LOO kNN -> misclassified mask ----
    pred = _loo_knn_predict(Xk, y, k=args.knn_k)
    mis = pred != y
    acc = float((~mis).mean())
    print(f"\nLeave-one-out kNN (k={args.knn_k}, cosine, SVD-{n_comp}) "
          f"cell-type accuracy: {acc:.3f}  ({int(mis.sum())} misclassified)")

    # ---- covariates (all PRE-imputation / pre-batch, like --min-linked-
    # pairs-per-cell) ----
    cov_df = pd.DataFrame(index=keep)
    pair_path = cdir / "pair_counts_per_bin.parquet"
    if pair_path.exists():
        pc = pd.read_parquet(pair_path)  # raw n_linked per (cell, bin)
        # Depth: total raw linked pairs (== --min-linked-pairs-per-cell metric).
        cov_df["coverage"] = pc.sum(axis=1).reindex(keep)
        # Breadth: number of bins with any linked pair (a deep-but-narrow cell
        # may mis-impute even at high depth).
        cov_df["breadth_bins"] = (pc > 0).sum(axis=1).reindex(keep)
    # Unlinked depth: short-range baseline that feeds S_u (from soft counts).
    soft_path = cdir / "soft_counts_raw.parquet"
    if soft_path.exists():
        soft = pd.read_parquet(soft_path, columns=["cell", "n_unlinked"])
        cov_df["unlinked_depth"] = (
            soft.groupby("cell")["n_unlinked"].sum().reindex(keep)
        )
    acc_path = cdir / "accessibility_mean.parquet"
    if acc_path.exists():
        am = pd.read_parquet(acc_path)
        cov_df["mean_accessibility"] = am.mean(axis=1).reindex(keep)
    cov_df["PC0"] = pc0.reindex(keep)

    batch = None
    if args.samples and args.batch_column:
        samples = _load_samples(args.samples)
        if args.batch_column in samples.columns:
            batch = samples.set_index("cell_name")[args.batch_column].reindex(keep)

    # ---- compare misclassified vs correct ----
    print("\nMisclassified vs correctly-separated cells:")
    cov_p = None
    if "coverage" in cov_df:
        cov_p = _compare("coverage", cov_df["coverage"], mis)
    if "breadth_bins" in cov_df:
        _compare("breadth_bins", cov_df["breadth_bins"], mis)
    if "unlinked_depth" in cov_df:
        _compare("unlinked_depth", cov_df["unlinked_depth"], mis)
    if "mean_accessibility" in cov_df:
        _compare("mean_accessibility", cov_df["mean_accessibility"], mis)
    # PC0 boundary check: do misclassified cells sit near PC0=median?
    pc0_abs = (cov_df["PC0"] - cov_df["PC0"].median()).abs()
    _compare("|PC0 - median|", pc0_abs, mis)

    if "coverage" in cov_df:
        r = np.corrcoef(cov_df["coverage"].fillna(cov_df["coverage"].median()),
                        cov_df["PC0"])[0, 1]
        print(f"\n  corr(coverage, PC0) = {r:+.3f}  "
              f"(strong |r| means PC0 partly tracks depth, not cell type)")

    if batch is not None:
        from scipy.stats import chi2_contingency
        tab = pd.crosstab(pd.Series(mis, index=keep, name="misclassified"), batch)
        print(f"\n  batch composition (misclassified rows):\n{tab}")
        try:
            chi2, p, _, _ = chi2_contingency(tab)
            print(f"  batch x misclassified chi-square p = {p:.2e}")
        except ValueError:
            pass

    # ---- verdict ----
    print("\n" + "=" * 70)
    if cov_p is not None and cov_p < 0.01:
        lo = cov_df.loc[mis, "coverage"].median()
        hi = cov_df.loc[~mis, "coverage"].median()
        if lo < hi:
            q = float(cov_df["coverage"].quantile(0.15))
            print("VERDICT: misclassified cells have significantly LOWER "
                  "coverage.\n  A cell-level coverage filter should help. "
                  f"Suggested starting threshold (~15th pct): coverage >= {q:.0f}.")
        else:
            print("VERDICT: misclassified cells have significantly HIGHER "
                  "coverage (unusual) — investigate before filtering.")
    else:
        print("VERDICT: coverage does NOT separate misclassified from correct "
              "cells.\n  The residual mixing is a feature-strength problem, "
              "not coverage — a coverage filter will not fix it.  Consider a "
              "stronger initial RWR basis or more refinement passes instead.")
    print("=" * 70)

    # ---- optional colored UMAPs ----
    if args.umap_coords:
        coords = pd.read_csv(args.umap_coords, sep="\t")
        outp = Path(args.out_prefix)
        if "coverage" in cov_df:
            _scatter(coords, cov_df["coverage"], "UMAP colored by coverage",
                     Path(f"{outp}.umap_coverage.png"))
        if "mean_accessibility" in cov_df:
            _scatter(coords, cov_df["mean_accessibility"],
                     "UMAP colored by mean accessibility",
                     Path(f"{outp}.umap_accessibility.png"))
        mis_series = pd.Series(np.where(mis, "misclassified", "correct"), index=keep)
        _scatter(coords, mis_series, "UMAP: kNN correct vs misclassified",
                 Path(f"{outp}.umap_misclassified.png"), categorical=True)
        if batch is not None:
            _scatter(coords, batch, "UMAP colored by batch",
                     Path(f"{outp}.umap_batch.png"), categorical=True)


if __name__ == "__main__":
    main()
