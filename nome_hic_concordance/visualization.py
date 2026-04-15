"""UMAP embedding and scatter-plot visualisation of the cells x bins matrix."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def run_umap_and_plot(
    matrix: pd.DataFrame,
    group_labels: pd.Series | None = None,
    title: str = "UMAP",
    output_prefix: str = "umap",
    output_dir: str = "output/06_umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
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

    Returns
    -------
    DataFrame with columns ``cell, UMAP1, UMAP2, group``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    out_path = ensure_dir(output_dir)

    logger.info(
        f"Running UMAP on {matrix.shape[0]} cells x {matrix.shape[1]} bins "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})..."
    )

    # UMAP cannot handle the degenerate case of fewer than 2 cells.
    if matrix.shape[0] < 2:
        raise ValueError(
            f"UMAP requires >= 2 cells; received {matrix.shape[0]}."
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
    embedding = reducer.fit_transform(matrix.values)

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
