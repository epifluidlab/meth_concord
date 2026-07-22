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
    plot_format: str = "png",
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
        Filename prefix for the .tsv and figure outputs.
    output_dir : str
    plot_format : str
        Figure extension: ``png`` (default), or a vector format such as
        ``pdf`` / ``svg`` for publication figures.

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
    fig.savefig(out_path / f"{output_prefix}.{plot_format}", dpi=150,
                bbox_inches="tight")
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


def order_groups(groups, group_order) -> list:
    """Groups arranged for the legend: those named in ``group_order`` first, in
    that order, then any remaining groups in data-encounter order.

    Names in ``group_order`` that don't occur in the data are ignored, so a
    single project-wide order string can be reused across cell subsets.
    """
    seq = list(groups)
    if not group_order:
        return seq
    present = set(seq)
    ordered = [g for g in group_order if g in present]
    named = set(ordered)
    return ordered + [g for g in seq if g not in named]


def resolve_group_colors(groups, group_colors) -> list:
    """One colour per group: the explicit ``group_colors`` entry where present,
    else that slot's tab20 fallback (so a partial colour column still works)."""
    import matplotlib.pyplot as plt

    n = len(groups)
    fallback = list(plt.cm.tab20(np.linspace(0, 1, max(n, 1))))
    if not group_colors:
        return fallback
    return [group_colors.get(g, fallback[i]) for i, g in enumerate(groups)]


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
    group_colors: dict | None = None,
    group_order: list | None = None,
    plot_format: str = "png",
    batch_labels=None,
    batch_correction: str = "none",
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
    group_colors : dict, optional
        ``group -> matplotlib colour`` (hex or named).  Overrides the default
        tab20 palette, and lifts the 20-group cap (with explicit colours there
        is no palette to run out of).  Groups missing from the dict fall back
        to tab20.
    group_order : list, optional
        Explicit group order for the legend (and plotting/z-order).  Groups
        present in the data but absent from the list are appended in
        data-encounter order; names in the list that aren't in the data are
        ignored, so one order can be reused across subsets.
    plot_format : str
        Figure extension: ``png`` (default), or a vector format such as
        ``pdf`` / ``svg`` for publication figures.

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
    # SVD runs whenever EITHER --n-svd > 0 OR --skip-top-svd > 0.
    # When the user specifies --skip-top-svd without --n-svd, default
    # the number of retained components to 10 — otherwise the option
    # would silently no-op.
    X = matrix.values
    n_svd_used = 0
    skip = max(0, int(skip_top_svd or 0))
    n_svd_requested = int(n_svd or 0)
    if n_svd_requested > 0 or skip > 0:
        from sklearn.decomposition import TruncatedSVD
        # If user only specified --skip-top-svd, retain 10 components by default.
        effective_n_svd = n_svd_requested if n_svd_requested > 0 else 10
        # Fit enough components to satisfy retained + skipped.
        k_total = max(
            1,
            min(effective_n_svd + skip, matrix.shape[0] - 1, matrix.shape[1]),
        )
        svd = TruncatedSVD(n_components=k_total, random_state=random_state)
        emb = svd.fit_transform(X)
        # Drop the top-`skip` components.
        X = emb[:, skip:]
        n_svd_used = X.shape[1]
        var_explained = float(svd.explained_variance_ratio_[skip:].sum())

        # Batch-correct HERE -- on the SVD scores that feed UMAP, after all
        # normalisation and after the top-component skip.  This is the standard
        # harmony workflow; correcting the raw matrix before per-cell rank /
        # quantile normalisation (as the callers used to) is undone by those
        # transforms, so the UMAP still splits by batch.
        if batch_labels is not None and batch_correction not in (None, "none"):
            from . import batch_correction as _bc
            logger.info(
                f"Batch correction ({batch_correction}) on {X.shape[1]} SVD "
                f"scores ({matrix.shape[0]} cells) before UMAP.")
            if batch_correction == "harmony":
                # Align to the matrix's cell order and make it POSITIONAL:
                # harmonize_embedding coerces against a RangeIndex, so a
                # cell-indexed frame would reindex to all-NaN (one batch, no-op).
                bl = batch_labels
                if isinstance(bl, (pd.Series, pd.DataFrame)):
                    bl = bl.reindex(matrix.index).reset_index(drop=True)
                X = _bc.harmonize_embedding(X, bl, random_state=random_state)
            else:
                # center / zscore / combat: per-batch mean (or z) removal per
                # component.  For BALANCED composition this removes the batch
                # shift without moving cell-type centroids, and unlike harmony
                # it cannot fail when batch dominates a component.
                bl = (batch_labels.reindex(matrix.index)
                      if isinstance(batch_labels, (pd.Series, pd.DataFrame))
                      else batch_labels)
                score_df = pd.DataFrame(X, index=matrix.index)
                X = _bc.apply_batch_correction(
                    score_df, bl, method=batch_correction,
                    random_state=random_state).to_numpy()
        if skip > 0:
            logger.info(
                f"SVD: fitted {k_total} components, dropped top {skip} "
                f"(--skip-top-svd), retained components {skip}..{k_total - 1} "
                f"({n_svd_used} components in total; "
                f"cumulative variance explained by retained: {var_explained:.1%})"
            )
        else:
            logger.info(
                f"TruncatedSVD reduced {matrix.shape[1]} bins → "
                f"{k_total} components "
                f"(cumulative variance explained: {var_explained:.1%})"
            )

    if (batch_labels is not None and batch_correction not in (None, "none")
            and n_svd_used == 0):
        logger.warning(
            "Batch correction on the UMAP scores needs an SVD reduction; pass "
            "--n-svd (or --skip-top-svd). No SVD ran, so batch correction was "
            "SKIPPED and the embedding is uncorrected.")

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
    unique_groups = order_groups(result["group"].dropna().unique(), group_order)
    n_g = len(unique_groups)
    # explicit colours also lift the tab20 cap — there is no palette to exhaust
    if group_labels is not None and n_g > 0 and (n_g <= 20 or group_colors):
        colors = resolve_group_colors(unique_groups, group_colors)
        for g, c in zip(unique_groups, colors):
            mask = result["group"] == g
            ax.scatter(
                result.loc[mask, "UMAP1"], result.loc[mask, "UMAP2"],
                color=c, label=str(g), s=15, alpha=0.75, edgecolors="none",
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
    plt.savefig(out_path / f"{output_prefix}.{plot_format}", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return result
