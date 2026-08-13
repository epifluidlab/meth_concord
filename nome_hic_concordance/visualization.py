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


def umap_input_scores(
    matrix: pd.DataFrame,
    n_svd: int = 0,
    skip_top_svd: int = 0,
    n_threads: int = 4,
    random_state: int = 42,
    batch_labels=None,
    batch_correction: str = "none",
    harmony_theta=None,
) -> tuple[np.ndarray, int]:
    """Everything :func:`run_umap_and_plot` does BEFORE calling UMAP.

    Returns ``(X, n_svd_used)`` — the exact array UMAP would be handed, after
    the optional TruncatedSVD pre-reduction, the top-component drop, and the
    batch correction.  Factored out so a parameter sweep can score the real
    pipeline input instead of a re-implementation of it that could drift.
    """
    X = matrix.values
    n_svd_used = 0
    skip = max(0, int(skip_top_svd or 0))
    n_svd_requested = int(n_svd or 0)
    if n_svd_requested > 0 or skip > 0:
        from sklearn.decomposition import TruncatedSVD
        effective_n_svd = n_svd_requested if n_svd_requested > 0 else 10
        k_total = max(
            1,
            min(effective_n_svd + skip, matrix.shape[0] - 1, matrix.shape[1]),
        )
        svd = TruncatedSVD(n_components=k_total, random_state=random_state)
        emb = svd.fit_transform(X)
        X = emb[:, skip:]
        n_svd_used = X.shape[1]
        var_explained = float(svd.explained_variance_ratio_[skip:].sum())
        if skip > 0:
            logger.info(
                f"SVD: fitted {k_total} components, dropped top {skip} "
                f"(--skip-top-svd), retained components {skip}..{k_total - 1} "
                f"({n_svd_used} components in total; "
                f"cumulative variance explained by retained: "
                f"{var_explained:.1%})")
        else:
            logger.info(
                f"TruncatedSVD reduced {matrix.shape[1]} bins → "
                f"{k_total} components "
                f"(cumulative variance explained: {var_explained:.1%})")

    if batch_labels is not None and batch_correction not in (None, "none"):
        space = (f"{X.shape[1]} SVD scores" if n_svd_used
                 else f"{X.shape[1]} bins (full matrix, no SVD)")
        logger.info(
            f"Batch correction ({batch_correction}) on {space}, "
            f"{matrix.shape[0]} cells before UMAP.")
        X = correct_scores(X, matrix.index, batch_labels, batch_correction,
                           random_state=random_state, n_threads=n_threads,
                           harmony_theta=harmony_theta)
    return X, n_svd_used


def correct_scores(X, index, batch_labels, method: str,
                   random_state: int = 42, n_threads: int = 4,
                   harmony_theta=None):
    """Batch-correct an already-reduced score matrix.

    Split out of :func:`umap_input_scores` so a parameter sweep can reuse ONE
    SVD across many score-level corrections — the SVD is the expensive half and
    does not depend on the correction — without duplicating this branch, where
    harmony's index handling is easy to get subtly wrong.
    """
    if batch_labels is None or method in (None, "none"):
        return X
    from . import batch_correction as _bc
    if method == "harmony":
        # harmonize_embedding coerces against a RangeIndex, so a cell-indexed
        # frame would reindex to all-NaN (one batch, silently a no-op).
        bl = batch_labels
        if isinstance(bl, (pd.Series, pd.DataFrame)):
            bl = bl.reindex(index).reset_index(drop=True)
        return _bc.harmonize_embedding(X, bl, random_state=random_state,
                                       theta=harmony_theta,
                                       n_threads=n_threads)
    bl = (batch_labels.reindex(index)
          if isinstance(batch_labels, (pd.Series, pd.DataFrame))
          else batch_labels)
    return _bc.apply_batch_correction(
        pd.DataFrame(X, index=index), bl, method=method,
        random_state=random_state).to_numpy()


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
    harmony_theta=None,
    annotate_metrics: bool = True,
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
    annotate_metrics : bool
        When ``group_labels`` is given, score how well the representation
        separates those labels (ASW / kNN-purity / ARI / NMI, see
        :mod:`nome_hic_concordance.metrics`), caption the figure with them,
        and write ``<output_prefix>_metrics.tsv``.  Scored on BOTH the
        feature space UMAP actually consumed (post-SVD, post-batch-
        correction) and the 2-D embedding; the caption reports the feature
        space, since the 2-D layout is a lossy view of it.

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

    # SVD pre-reduction + batch correction of WHAT FEEDS UMAP, shared with
    # the parameter sweep via umap_input_scores so the two cannot drift.
    X, n_svd_used = umap_input_scores(
        matrix, n_svd=n_svd, skip_top_svd=skip_top_svd, n_threads=n_threads,
        random_state=random_state, batch_labels=batch_labels,
        batch_correction=batch_correction, harmony_theta=harmony_theta)

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

    # ---- Label-agreement metrics -------------------------------------
    # Scored on the space UMAP consumed (X: post-SVD, post-batch-correction)
    # as well as the 2-D layout.  The caption reports the feature space: the
    # 2-D embedding is a lossy projection of it, so judging a representation
    # by its picture flatters whatever UMAP happened to lay out neatly.
    metrics_line = ""
    if annotate_metrics and group_labels is not None:
        from . import metrics as _metrics
        y = result["group"].where(result["group"] != "unknown")
        space = (f"{X.shape[1]} SVD comps" if n_svd_used
                 else f"{X.shape[1]} bins")
        feat = _metrics.label_agreement(X, y, random_state=random_state)
        emb = _metrics.label_agreement(embedding, y, random_state=random_state)
        metrics_line = _metrics.format_metrics(feat, space)
        pd.DataFrame([{"space": "feature", "detail": space, **feat},
                      {"space": "umap2d", "detail": "2 UMAP axes", **emb}]
                     ).to_csv(out_path / f"{output_prefix}_metrics.tsv",
                              sep="\t", index=False)
        if metrics_line:
            logger.info(f"{output_prefix} [feature space]: {metrics_line}")

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
    # Extra title pad leaves room for the metrics caption, which is drawn
    # just above the axes so it reads as a subtitle at a smaller size.
    ax.set_title(title, pad=20 if metrics_line else None)
    if metrics_line:
        ax.annotate(
            metrics_line, xy=(0.5, 1.0), xytext=(0, 5),
            xycoords="axes fraction", textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color="0.35",
        )
    plt.tight_layout()
    plt.savefig(out_path / f"{output_prefix}.{plot_format}", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return result
