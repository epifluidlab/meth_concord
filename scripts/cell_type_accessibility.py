#!/usr/bin/env python3
"""Per-cell-type mean accessibility + mislabeling diagnostic.

Given an accessibility matrix (cells x bins) and a samples manifest with
cell-type labels, computes:

  (A) Per-claimed-cell-type mean accessibility per bin
      → bins x groups TSV.

  (B) Top differentially-accessible bins between the claimed groups
      → can be loaded into IGV against reference ENCODE ATAC-seq tracks
        to verify which group corresponds to which true cell type.

  (C) If --umap-coords is provided: per-UMAP-cluster mean accessibility,
      plus a confusion / correlation matrix showing how the K UMAP
      clusters line up against the K labeled groups.  If labelling is
      correct, you'd expect cluster <-> group correlations to be
      block-diagonal (cluster 1 ↔ group A, cluster 2 ↔ group B).  If
      labelling is scrambled, the correlations look mixed.

Example:

    python scripts/cell_type_accessibility.py \\
        --accessibility-parquet output/.../05_concordance/accessibility_mean.parquet \\
        --samples samples.filtered.tsv \\
        --umap-coords output/.../06_umap/umap_quantile_svd5_coordinates.tsv \\
        --output-prefix /tmp/cell_type_acc \\
        --top-n-diff 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_samples(path: str) -> pd.DataFrame:
    """Load the manifest TSV.  Accepts headered or headerless format."""
    with open(path) as fh:
        first = fh.readline().strip()
    first_field = first.split("\t")[0].strip().lower()
    has_header = first_field in (
        "cell_name", "cellname", "cell", "sample", "sample_name",
    )
    if has_header:
        df = pd.read_csv(path, sep="\t", comment="#", dtype=str)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "cellname" in df.columns:
            df = df.rename(columns={"cellname": "cell_name"})
        if "sample" in df.columns and "cell_name" not in df.columns:
            df = df.rename(columns={"sample": "cell_name"})
    else:
        df = pd.read_csv(
            path, sep="\t", header=None, comment="#",
            names=["cell_name", "bam_path", "group"], dtype=str,
        )
    if "cell_name" not in df.columns:
        raise SystemExit(f"manifest {path} has no cell_name column.")
    if "group" not in df.columns:
        raise SystemExit(
            f"manifest {path} has no group column — can't do per-group means."
        )
    df["group"] = df["group"].fillna("unknown")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--accessibility-parquet", required=True,
                        help="Path to accessibility_mean.parquet from "
                             "output/.../05_concordance/.")
    parser.add_argument("--samples", required=True,
                        help="Samples manifest TSV with cell_name + group columns.")
    parser.add_argument("--umap-coords", default=None,
                        help="Optional path to umap_*_coordinates.tsv "
                             "(cell, UMAP1, UMAP2, ...) for the "
                             "per-cluster vs per-group comparison.")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="Number of UMAP clusters to identify "
                             "(default: number of unique groups).")
    parser.add_argument("--top-n-diff", type=int, default=200,
                        help="Number of top differentially-accessible bins "
                             "to save (default: 200).")
    parser.add_argument("--output-prefix", required=True,
                        help="Output prefix; produces "
                             "<prefix>.per_group.tsv, "
                             "<prefix>.diff_bins.tsv, "
                             "<prefix>.per_cluster.tsv (if UMAP given), "
                             "<prefix>.mislabel_diagnostic.tsv.")
    args = parser.parse_args()

    acc = pd.read_parquet(args.accessibility_parquet)
    samples = load_samples(args.samples)
    cell_to_group = dict(zip(samples["cell_name"], samples["group"]))

    # Align accessibility to manifest cells (drop cells not in manifest).
    in_manifest = acc.index.isin(cell_to_group)
    if not in_manifest.all():
        n_drop = int((~in_manifest).sum())
        print(
            f"WARNING: {n_drop} of {len(acc)} cells in accessibility matrix "
            f"have no manifest entry; dropping.",
            file=sys.stderr,
        )
    acc = acc.loc[in_manifest]
    group_series = pd.Series(
        [cell_to_group[c] for c in acc.index], index=acc.index, name="group",
    )

    groups = sorted(group_series.unique())
    print(f"\nCells: {len(acc)}  |  Bins: {acc.shape[1]}  |  Groups: {groups}")
    print(group_series.value_counts().to_string())

    # ---- (A) per-claimed-cell-type mean accessibility ---------------------
    per_group_mean = acc.groupby(group_series).mean()   # groups × bins
    per_group_mean = per_group_mean.T                   # bins × groups
    per_group_mean.index.name = "bin"
    per_group_mean.to_csv(f"{args.output_prefix}.per_group.tsv", sep="\t")
    print(f"\nWrote per-group means: {args.output_prefix}.per_group.tsv "
          f"({per_group_mean.shape[0]} bins × {per_group_mean.shape[1]} groups)")

    # ---- (B) top differentially-accessible bins between groups ------------
    # For 2 groups: signed log2(A/B) of mean openness.
    # For >2 groups: between-group variance / total variance.
    if len(groups) == 2:
        a, b = groups
        eps = 1e-3
        diff = np.log2((per_group_mean[a] + eps) / (per_group_mean[b] + eps))
        diff_abs = diff.abs()
        diff_df = pd.DataFrame({
            f"mean_{a}": per_group_mean[a],
            f"mean_{b}": per_group_mean[b],
            f"log2_{a}_over_{b}": diff,
        }).sort_values(f"log2_{a}_over_{b}", key=lambda s: s.abs(),
                       ascending=False)
    else:
        # Between-group / total variance: simple "discriminative" score.
        var_between = per_group_mean.var(axis=1)
        diff_df = (
            per_group_mean.assign(var_between=var_between)
            .sort_values("var_between", ascending=False)
        )
        diff_abs = var_between

    diff_top = diff_df.head(args.top_n_diff)
    diff_top.to_csv(f"{args.output_prefix}.diff_bins.tsv", sep="\t")
    print(
        f"Top {len(diff_top)} differentially-accessible bins: "
        f"{args.output_prefix}.diff_bins.tsv"
    )
    if len(groups) == 2:
        n_up_in_a = int((diff_df[f"log2_{a}_over_{b}"] > 0).sum())
        n_up_in_b = int((diff_df[f"log2_{a}_over_{b}"] < 0).sum())
        median_abs = float(diff_abs.median())
        print(f"  log2 fold-change distribution: "
              f"{n_up_in_a} bins up in '{a}', {n_up_in_b} up in '{b}'; "
              f"median |log2FC| = {median_abs:.3f}")

    # ---- (C) UMAP-cluster vs claimed group mislabel diagnostic ------------
    if args.umap_coords is not None:
        umap_df = pd.read_csv(args.umap_coords, sep="\t")
        if "cell" not in umap_df.columns:
            raise SystemExit(
                f"UMAP coordinates file {args.umap_coords} must have a "
                f"'cell' column."
            )
        umap_df = umap_df.set_index("cell")
        umap_df = umap_df.loc[umap_df.index.isin(acc.index)]
        umap_df = umap_df.loc[acc.index]  # align order

        coord_cols = [c for c in umap_df.columns if c.startswith("UMAP")]
        if len(coord_cols) < 2:
            raise SystemExit(
                f"UMAP file must have at least UMAP1 + UMAP2 columns; "
                f"found {umap_df.columns.tolist()}."
            )

        from sklearn.cluster import KMeans
        k = args.n_clusters or len(groups)
        clusters = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(
            umap_df[coord_cols].to_numpy()
        )
        cluster_series = pd.Series(
            [f"cluster_{i}" for i in clusters], index=acc.index, name="cluster",
        )

        # Per-cluster mean accessibility.
        per_cluster_mean = (
            acc.groupby(cluster_series).mean().T
        )
        per_cluster_mean.index.name = "bin"
        per_cluster_mean.to_csv(
            f"{args.output_prefix}.per_cluster.tsv", sep="\t",
        )
        print(f"\nWrote per-cluster means: "
              f"{args.output_prefix}.per_cluster.tsv "
              f"({per_cluster_mean.shape[0]} bins × "
              f"{per_cluster_mean.shape[1]} clusters)")

        # Confusion matrix: how do clusters split by claimed group?
        confusion = pd.crosstab(cluster_series, group_series,
                                margins=True, margins_name="total")
        confusion.to_csv(
            f"{args.output_prefix}.cluster_vs_group_confusion.tsv",
            sep="\t",
        )

        # Correlation between per-cluster mean profile and per-group mean
        # profile (bin-by-bin Pearson r).
        # Drop bins that are all-NaN in either set.
        common_bins = (
            per_group_mean.dropna(how="all").index
            .intersection(per_cluster_mean.dropna(how="all").index)
        )
        pg = per_group_mean.loc[common_bins].fillna(0.5)
        pc = per_cluster_mean.loc[common_bins].fillna(0.5)
        # Correlation matrix: clusters (rows) × groups (cols).
        corr = pd.DataFrame(
            index=pc.columns, columns=pg.columns, dtype=float,
        )
        for c in pc.columns:
            for g in pg.columns:
                corr.loc[c, g] = float(
                    np.corrcoef(pc[c].to_numpy(), pg[g].to_numpy())[0, 1]
                )
        corr.to_csv(
            f"{args.output_prefix}.mislabel_diagnostic.tsv", sep="\t",
        )

        print("\n" + "=" * 60)
        print("MISLABELING DIAGNOSTIC")
        print("=" * 60)
        print("\nCells per (UMAP cluster, claimed group):")
        print(confusion.to_string())
        print("\nBin-by-bin Pearson r between cluster profiles and group profiles:")
        print(corr.to_string(float_format=lambda x: f"{x:.3f}"))
        print("\nINTERPRETATION:")
        if k == len(groups) == 2:
            # 2 clusters × 2 groups: a clean labelling gives high diagonal
            # correlations.  Mislabeling shows near-equal off-diagonal corrs.
            diag = corr.values.diagonal()
            off = corr.values[~np.eye(k, dtype=bool)].reshape(k, -1).flatten()
            print(f"  Mean diagonal r:     {diag.mean():.3f}")
            print(f"  Mean off-diagonal r: {off.mean():.3f}")
            if abs(diag.mean() - off.mean()) < 0.05:
                print(
                    "  → Clusters and groups DO NOT align — likely either "
                    "mislabeling or the UMAP isn't picking up cell type."
                )
            elif diag.mean() > off.mean():
                print(
                    "  → Clusters and groups DO align cleanly (diagonal "
                    "dominates) — labels look correct."
                )
            else:
                print(
                    "  → Off-diagonal dominates — labels are likely scrambled "
                    "(cluster 0 looks more like the OTHER group's profile, etc.)."
                )
        else:
            print("  (For >2 groups, look at the correlation matrix for "
                  "block-diagonal vs scrambled patterns.)")

    print("\nNext step: load <prefix>.diff_bins.tsv into IGV alongside "
          "ENCODE bulk ATAC-seq for GM12878 and IMR90; visually check "
          "which group corresponds to which bulk reference.")


if __name__ == "__main__":
    main()
