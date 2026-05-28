#!/usr/bin/env python3
"""Infer per-cell labels from any accessibility / methylation matrix.

Workflow:
  1. Load a cells × bins matrix (e.g., accessibility_mean.parquet — works
     for either GCH or HCG context: GCH gives mean accessibility, HCG
     gives mean methylation).
  2. Coverage-filter bins, per-bin quantile normalize, top-K SVD, UMAP.
  3. KMeans-cluster in UMAP space.
  4. Assign each cluster a predicted label by majority vote from the
     claimed labels (or by Hungarian matching when the user wants a
     1:1 cluster↔label assignment).
  5. Output:
       - <prefix>.umap_coords.tsv : per-cell UMAP coords + cluster +
                                    predicted_label + claimed_label + agree
       - <prefix>.umap_claimed.png : UMAP colored by claimed label
       - <prefix>.umap_predicted.png : UMAP colored by predicted label
       - <prefix>.confusion.tsv : claimed × predicted counts
       - <prefix>.disagreements.tsv : cells where claimed ≠ predicted

For the GCH-vs-HCG cross-check you want:

    # First, on the HCG matrix (lineage-stable signal):
    python scripts/infer_cell_labels_from_accessibility.py \\
        --accessibility-parquet output/.../context_HCG/05_concordance/accessibility_mean.parquet \\
        --samples samples.filtered.tsv \\
        --output-prefix /tmp/hcg_labels

    # Same on the GCH matrix (for comparison):
    python scripts/infer_cell_labels_from_accessibility.py \\
        --accessibility-parquet output/.../context_GCH/05_concordance/accessibility_mean.parquet \\
        --samples samples.filtered.tsv \\
        --output-prefix /tmp/gch_labels

    # Then compare the two predicted-label files:
    join -j 1 -t $'\\t' \\
         <(sort /tmp/gch_labels.umap_coords.tsv) \\
         <(sort /tmp/hcg_labels.umap_coords.tsv) \\
       | awk -F'\\t' 'NR>1 {print $5, $11}' | sort | uniq -c

If the GCH labels and HCG labels agree → real cell types, claimed labels
are the wrong axis.  If GCH labels agree with claimed but HCG doesn't →
the HCG signal is in something other than cell type (cell-cycle, etc.).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_samples(path: str) -> pd.DataFrame:
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
    if "group" not in df.columns:
        df["group"] = "unknown"
    df["group"] = df["group"].fillna("unknown")
    return df


def _quantile_normalize_per_bin(matrix: np.ndarray) -> np.ndarray:
    """Per-column rank/(n+1) → centred at 0; NaNs preserved."""
    from scipy.stats import rankdata
    out = np.full_like(matrix, np.nan, dtype=np.float64)
    mask = ~np.isnan(matrix)
    for j in range(matrix.shape[1]):
        col_mask = mask[:, j]
        n_obs = int(col_mask.sum())
        if n_obs == 0:
            continue
        if n_obs == 1:
            out[col_mask, j] = 0.5
            continue
        ranks = rankdata(matrix[col_mask, j], method="average")
        out[col_mask, j] = ranks / (n_obs + 1)
    return out


def _scatter(coords: pd.DataFrame, color_col: str, title: str, out_path: str):
    """Save a UMAP scatter plot colored by ``color_col``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    groups = coords[color_col].fillna("unknown").unique()
    cmap = plt.cm.tab10 if len(groups) <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, max(len(groups), 1)))
    for g, c in zip(sorted(groups), colors):
        mask = coords[color_col].fillna("unknown") == g
        ax.scatter(
            coords.loc[mask, "UMAP1"], coords.loc[mask, "UMAP2"],
            c=[c], label=str(g), s=15, alpha=0.75, edgecolors="none",
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _match_clusters_to_labels(
    cluster_assignments: np.ndarray,
    claimed_labels: pd.Series,
    strategy: str = "hungarian",
) -> dict[int, str]:
    """Map cluster IDs (integers) to claimed labels.

    Hungarian: best 1:1 matching (if K = number of distinct claimed labels).
    Majority: each cluster takes the most common claimed label among its
              members; can produce many clusters mapped to the same label.
    """
    cluster_to_label: dict[int, str] = {}
    unique_clusters = sorted(set(cluster_assignments.tolist()))
    unique_labels = sorted(claimed_labels.dropna().unique())

    if strategy == "hungarian" and len(unique_clusters) == len(unique_labels):
        # Build cost matrix: cost = -count(cluster i, label j).
        cost = np.zeros((len(unique_clusters), len(unique_labels)))
        for i, c in enumerate(unique_clusters):
            mask = cluster_assignments == c
            for j, lab in enumerate(unique_labels):
                cost[i, j] = -int(((claimed_labels.iloc[mask.nonzero()[0]] == lab)).sum())
        from scipy.optimize import linear_sum_assignment
        row_idx, col_idx = linear_sum_assignment(cost)
        for r, c in zip(row_idx, col_idx):
            cluster_to_label[unique_clusters[r]] = unique_labels[c]
    else:
        # Majority vote per cluster.
        for c in unique_clusters:
            mask = cluster_assignments == c
            sub = claimed_labels.iloc[mask.nonzero()[0]]
            if len(sub) == 0:
                cluster_to_label[c] = "unknown"
            else:
                cluster_to_label[c] = sub.value_counts().idxmax()
    return cluster_to_label


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--accessibility-parquet", required=True,
                        help="Path to a cells × bins parquet (e.g. "
                             "accessibility_mean.parquet from GCH or HCG context).")
    parser.add_argument("--samples", required=True,
                        help="Samples manifest TSV with cell_name + group columns.")
    parser.add_argument("--output-prefix", required=True,
                        help="Output path prefix.")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="K for KMeans (default: number of unique claimed labels).")
    parser.add_argument("--n-svd", type=int, default=5,
                        help="Top-K SVD components used as input to UMAP (default: 5).")
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors (default: 15).")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="UMAP min_dist (default: 0.1).")
    parser.add_argument("--min-cell-coverage", type=float, default=0.5,
                        help="Drop bins where < this fraction of cells "
                             "have a value (default: 0.5).")
    parser.add_argument("--cluster-match", choices=["hungarian", "majority"],
                        default="hungarian",
                        help="How to map KMeans clusters to claimed labels. "
                             "'hungarian' (default): best 1:1 matching; "
                             "'majority': each cluster takes the most common "
                             "claimed label.")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    X = pd.read_parquet(args.accessibility_parquet)
    print(f"Loaded accessibility matrix: {X.shape[0]} cells × {X.shape[1]} bins")

    samples = load_samples(args.samples)
    cell_to_group = dict(zip(samples["cell_name"], samples["group"]))
    claimed = pd.Series(
        [cell_to_group.get(c, "unknown") for c in X.index],
        index=X.index, name="claimed_label",
    )

    # --- Bin coverage filter -----------------------------------------------
    coverage = X.notna().mean(axis=0)
    keep_bins = coverage >= args.min_cell_coverage
    n_kept = int(keep_bins.sum())
    print(f"Bin filter (coverage >= {args.min_cell_coverage:.0%}): "
          f"{X.shape[1]} → {n_kept} bins")
    if n_kept == 0:
        sys.exit("ERROR: no bins survive the coverage filter; lower --min-cell-coverage.")
    X = X.loc[:, keep_bins]

    # --- Quantile normalize per bin ----------------------------------------
    qn = _quantile_normalize_per_bin(X.to_numpy(dtype=np.float64))
    # Center at 0 and impute NaNs to centered median (0).
    qn_centered = np.nan_to_num(qn - 0.5, nan=0.0)

    # --- Top-K SVD ---------------------------------------------------------
    from sklearn.decomposition import TruncatedSVD
    k_svd = min(args.n_svd, X.shape[0] - 1, X.shape[1])
    svd = TruncatedSVD(n_components=k_svd, random_state=args.random_seed)
    svd_emb = svd.fit_transform(qn_centered)
    print(f"SVD: top {k_svd} components explain "
          f"{svd.explained_variance_ratio_.sum():.1%} variance")

    # --- UMAP --------------------------------------------------------------
    import umap
    n_neighbors = min(args.n_neighbors, max(2, X.shape[0] - 1))
    print(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={args.min_dist})...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=args.min_dist,
        random_state=args.random_seed,
    )
    embedding = reducer.fit_transform(svd_emb)

    # --- KMeans clustering -------------------------------------------------
    from sklearn.cluster import KMeans
    n_clusters = args.n_clusters or len(claimed.dropna().unique())
    if n_clusters < 2:
        sys.exit(f"Need >= 2 clusters; got n_clusters={n_clusters}.")
    print(f"KMeans clustering with k={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=args.random_seed, n_init=10)
    clusters = km.fit_predict(svd_emb)

    cluster_to_label = _match_clusters_to_labels(
        clusters, claimed, strategy=args.cluster_match,
    )
    predicted = pd.Series(
        [cluster_to_label[c] for c in clusters],
        index=X.index, name="predicted_label",
    )

    # --- Assemble output table ---------------------------------------------
    coords = pd.DataFrame({
        "cell": X.index,
        "UMAP1": embedding[:, 0],
        "UMAP2": embedding[:, 1],
        "cluster": clusters,
        "predicted_label": predicted.values,
        "claimed_label": claimed.values,
    })
    coords["agree"] = (coords["predicted_label"] == coords["claimed_label"]).astype(int)
    coords.to_csv(f"{args.output_prefix}.umap_coords.tsv", sep="\t", index=False)

    # --- Plots -------------------------------------------------------------
    _scatter(
        coords, "claimed_label",
        title=f"UMAP — colored by claimed label "
              f"({Path(args.accessibility_parquet).parent.parent.name})",
        out_path=f"{args.output_prefix}.umap_claimed.png",
    )
    _scatter(
        coords, "predicted_label",
        title=f"UMAP — colored by predicted label "
              f"({Path(args.accessibility_parquet).parent.parent.name})",
        out_path=f"{args.output_prefix}.umap_predicted.png",
    )

    # --- Confusion + disagreements -----------------------------------------
    confusion = pd.crosstab(
        coords["claimed_label"], coords["predicted_label"],
        margins=True, margins_name="total",
    )
    confusion.to_csv(f"{args.output_prefix}.confusion.tsv", sep="\t")

    disagreements = coords.loc[coords["agree"] == 0, [
        "cell", "cluster", "predicted_label", "claimed_label",
    ]]
    disagreements.to_csv(
        f"{args.output_prefix}.disagreements.tsv", sep="\t", index=False,
    )

    # --- Console report ----------------------------------------------------
    print("\n" + "=" * 60)
    print("LABEL INFERENCE SUMMARY")
    print("=" * 60)
    print(f"\nMatching strategy: {args.cluster_match}")
    print(f"Cluster → label mapping:")
    for c, lab in sorted(cluster_to_label.items()):
        n = int((clusters == c).sum())
        print(f"  cluster_{c} → '{lab}' ({n} cells)")

    print(f"\nConfusion (claimed × predicted):")
    print(confusion.to_string())

    n_total = len(coords)
    n_agree = int(coords["agree"].sum())
    n_disagree = n_total - n_agree
    print(f"\nAgreement: {n_agree}/{n_total} ({100*n_agree/n_total:.1f}%) cells")
    print(f"Disagreements: {n_disagree} cells — saved to "
          f"{args.output_prefix}.disagreements.tsv")

    if n_disagree > 0.3 * n_total:
        print("\n  ⚠  Over 30% of cells disagree between claimed and predicted "
              "labels.  Possible interpretations:")
        print("     - claimed labels are wrong (per-cell mislabeling)")
        print("     - the matrix-derived clusters reflect something OTHER than")
        print("       cell type (e.g. cell cycle, batch, technical artefact)")
        print("     - sample mixture (one well had two cell types)")

    print(f"\nOutputs:")
    for f in (".umap_coords.tsv", ".umap_claimed.png", ".umap_predicted.png",
              ".confusion.tsv", ".disagreements.tsv"):
        print(f"  {args.output_prefix}{f}")


if __name__ == "__main__":
    main()
