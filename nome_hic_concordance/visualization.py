"""UMAP embedding and scatter-plot visualisation of the cells x bins matrix."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def run_svd_scree(
    matrix: pd.DataFrame,
    max_components: int = 50,
    output_prefix: str = "svd_scree",
    output_dir: str = "output/06_umap",
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit TruncatedSVD on ``matrix`` and emit a scree plot + table.

    Helps decide ``--n-svd`` for the UMAP step.  The plot shows
    per-component variance explained (bars) and cumulative variance
    explained (line) for up to ``max_components`` components.

    Parameters
    ----------
    matrix : DataFrame, cells x bins, NaN-free.
    max_components : int
        Upper bound on number of components to fit.  Clamped to
        min(max_components, n_cells - 1, n_bins).
    output_prefix : str
        Filename prefix for the .tsv and .png outputs.
    output_dir : str

    Returns
    -------
    DataFrame with columns ``component, singular_value, explained_var,
    cumulative_var`` (1-indexed components).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD

    out_path = ensure_dir(output_dir)
    n_cells, n_bins = matrix.shape
    k = max(1, min(max_components, n_cells - 1, n_bins))
    if k < 1:
        logger.warning(
            f"Cannot fit SVD: matrix shape {matrix.shape} has no components."
        )
        return pd.DataFrame()

    logger.info(
        f"Fitting TruncatedSVD with k={k} components on {n_cells} cells "
        f"x {n_bins} bins for scree plot."
    )
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    svd.fit(matrix.values)

    explained = svd.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    df = pd.DataFrame({
        "component": np.arange(1, k + 1),
        "singular_value": svd.singular_values_,
        "explained_var": explained,
        "cumulative_var": cumulative,
    })
    df.to_csv(out_path / f"{output_prefix}.tsv", sep="\t", index=False)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(df["component"], df["explained_var"] * 100.0,
            color="steelblue", alpha=0.8, edgecolor="none",
            label="per-component")
    ax1.set_xlabel("SVD component")
    ax1.set_ylabel("Variance explained (%)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xlim(0.5, k + 0.5)

    ax2 = ax1.twinx()
    ax2.plot(df["component"], df["cumulative_var"] * 100.0,
             color="darkorange", marker="o", markersize=4,
             linewidth=1.5, label="cumulative")
    ax2.set_ylabel("Cumulative variance explained (%)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(0, 105)

    # Reference lines at 70 / 80 / 90 % cumulative.
    for thresh in (70, 80, 90):
        ax2.axhline(thresh, ls="--", color="gray", lw=0.5, alpha=0.5)

    ax1.set_title(f"SVD scree: top {k} components")
    fig.tight_layout()
    fig.savefig(out_path / f"{output_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Log the K needed to hit common variance thresholds.
    for thresh in (0.70, 0.80, 0.90, 0.95):
        if cumulative[-1] >= thresh:
            k_thresh = int(np.searchsorted(cumulative, thresh) + 1)
            logger.info(
                f"  -> {k_thresh} components explain >= {thresh:.0%} variance"
            )
        else:
            logger.info(
                f"  -> first {k} components explain "
                f"{cumulative[-1]:.1%} (< {thresh:.0%})"
            )
            break

    return df


def run_umap_and_plot(
    matrix: pd.DataFrame,
    group_labels: pd.Series | None = None,
    title: str = "UMAP",
    output_prefix: str = "umap",
    output_dir: str = "output/06_umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    n_svd: int = 0,
    skip_top_svd: int = 0,
    n_threads: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Embed ``matrix`` via UMAP and save a coordinate table + scatter plot.

    Parameters
    ----------
    matrix : DataFrame (cells x bins)
        Must be numeric and NaN-free.
    group_labels : Series, optional
        ``cell_name -> group``. If given, coloured by group in the plot.
    title : str
        Title of the generated figure.
    output_prefix : str
        Filename prefix for coordinates and PNG.
    output_dir : str
    n_neighbors, min_dist, metric, n_threads, random_state :
        Passed through to ``umap.UMAP``.
    n_svd : int
        If > 0, first reduce the matrix to ``n_svd`` components via
        TruncatedSVD before running UMAP (bulk-RNA / scATAC-seq style).
        0 (default) runs UMAP directly on the bin matrix.
    skip_top_svd : int
        Drop the first N SVD components before UMAP.  Useful when the
        top components carry non-target signal (per-cell depth, batch,
        k-NN community structure).  Requires ``n_svd > skip_top_svd``.
        Default 0.

    Returns
    -------
    DataFrame with columns ``cell, UMAP1, UMAP2, group``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    out_path = ensure_dir(output_dir)

    # UMAP cannot handle the degenerate case of fewer than 2 cells.
    if matrix.shape[0] < 2:
        raise ValueError(
            f"UMAP requires >= 2 cells; received {matrix.shape[0]}."
        )

    # Optional TruncatedSVD pre-reduction.
    X = matrix.values
    n_svd_used = 0
    if n_svd and n_svd > 0:
        from sklearn.decomposition import TruncatedSVD
        # Fit enough components to satisfy --n-svd PLUS --skip-top-svd.
        skip = max(0, int(skip_top_svd))
        k_total = max(1, min(int(n_svd) + skip, matrix.shape[0] - 1, matrix.shape[1]))
        svd = TruncatedSVD(n_components=k_total, random_state=random_state)
        emb = svd.fit_transform(X)
        # Drop the top-`skip` components.
        X = emb[:, skip:]
        n_svd_used = X.shape[1]
        if skip > 0:
            logger.info(
                f"SVD: dropped top {skip} components "
                f"(--skip-top-svd); using components {skip}..{k_total - 1} "
                f"({n_svd_used} components in total)"
            )
        var_explained = float(svd.explained_variance_ratio_[skip:].sum())
        logger.info(
            f"TruncatedSVD reduced {matrix.shape[1]} bins -> {k} components "
            f"(cumulative variance explained: {var_explained:.1%})"
        )

    logger.info(
        f"Running UMAP on {matrix.shape[0]} cells x {X.shape[1]} "
        f"{'SVD components' if n_svd_used else 'bins'} "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})..."
    )

    # Clamp n_neighbors to the number of cells - 1 (umap-learn requirement).
    effective_neighbors = min(n_neighbors, max(2, matrix.shape[0] - 1))

    reducer = umap.UMAP(
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_jobs=n_threads,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)

    result = pd.DataFrame({
        "cell": matrix.index,
        "UMAP1": embedding[:, 0],
        "UMAP2": embedding[:, 1],
    })

    if group_labels is not None:
        result["group"] = group_labels.reindex(matrix.index).values
    else:
        result["group"] = "unknown"

    result.to_csv(out_path / f"{output_prefix}_coordinates.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    unique_groups = result["group"].dropna().unique()
    if group_labels is not None and 0 < len(unique_groups) <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_groups), 1)))
        for g, c in zip(unique_groups, colors):
            mask = result["group"] == g
            ax.scatter(
                result.loc[mask, "UMAP1"], result.loc[mask, "UMAP2"],
                c=[c], label=str(g), s=15, alpha=0.75, edgecolors="none",
            )
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    else:
        ax.scatter(
            result["UMAP1"], result["UMAP2"], s=15, alpha=0.75,
            c="steelblue", edgecolors="none",
        )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path / f"{output_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return result
