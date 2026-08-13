"""Matrix-level operations on the cells x bins concordance matrix.

Applies bin filtering, NaN imputation, and per-bin normalization
(z-score or quantile-rank) across cells.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import blacklist as blacklist_mod
from .utils import ensure_dir

logger = logging.getLogger(__name__)

# Re-exports for backward compatibility with any external callers /
# tests that imported these private symbols from matrix_ops.
_infer_bin_size = blacklist_mod.infer_bin_size
_load_blacklist_bed = blacklist_mod.load_blacklist_bed
_bin_overlaps_blacklist = blacklist_mod.interval_overlaps_blacklist


def _quantile_normalize_per_bin(
    matrix: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-column rank-based quantile normalization to a uniform target.

    Replaces each column's values with ``rank / (n+1)`` (in (0,1)) so every
    column has an identical uniform distribution across cells.  NaN/masked
    entries are preserved as NaN and ignored during rank computation.
    """
    from scipy.stats import rankdata

    out = np.full_like(matrix, np.nan, dtype=np.float64)
    if mask is None:
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


def process_concordance_matrix(
    concordance_shrunk: pd.DataFrame,
    min_cell_coverage: float = 0.8,
    output_dir: str = "output/05_concordance",
    matrix_norm: str = "zscore",
    per_cell_center: bool = False,
    per_cell_norm: str = "off",
    post_norm_recenter: bool = False,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Filter bins, impute NaNs, and normalize per bin.

    Parameters
    ----------
    concordance_shrunk : DataFrame
        cells × bins matrix (Pearson r, shrunk r, or Δ values; NaN-tolerant).
        Bin column labels look like ``"chr1:0"``, ``"chr1:250000"``, ...
    min_cell_coverage : float
        Drop bins where < ``min_cell_coverage`` of cells have a value.
    output_dir : str
    matrix_norm : "zscore" | "quantile"
        Per-bin normalization across cells.  ``"zscore"`` (default) for the
        classical pipeline; ``"quantile"`` for rank-based QN to a uniform
        [-0.5, 0.5] target — robust to outliers, matches bulk-RNA /
        scATAC-seq style preprocessing.

    Returns
    -------
    dict with keys:
        - 'retained_bins' : list[str]
        - 'imputed' : DataFrame (cells x retained_bins), NaN-free
        - 'zscore'  : DataFrame (cells x retained_bins), normalized per column.
                      The label "zscore" is kept for backward compatibility
                      regardless of which ``matrix_norm`` was used; check
                      ``matrix_norm`` to know how the values were produced.

    Notes
    -----
    Blacklist filtering happens at the *read* level in Step 2 (see
    :func:`nome_hic_concordance.bam_parser.parse_all_cells`).  There is
    no longer a bin-level blacklist filter here; if you need it, re-run
    Step 2 with ``--blacklist-bed``.
    """
    # save_outputs=False keeps this in memory: the normalized matrix is ~126 MB
    # for 1577 cells x 10k bins, and a parameter sweep that re-normalizes tens
    # of times would otherwise write gigabytes it never reads back.
    out_path = ensure_dir(output_dir) if save_outputs else None
    n_cells, n_bins_orig = concordance_shrunk.shape

    # Filter bins by fraction of non-NA cells.
    if n_cells == 0:
        raise ValueError("concordance_shrunk has zero cells")
    coverage_frac = concordance_shrunk.notna().mean(axis=0)
    retained_mask = coverage_frac >= min_cell_coverage
    retained_bins = concordance_shrunk.columns[retained_mask].tolist()

    filtered = concordance_shrunk.loc[:, retained_bins].copy()
    logger.info(
        f"Bin filter: {n_bins_orig} -> {len(retained_bins)} bins "
        f"(coverage >= {min_cell_coverage:.0%})"
    )

    if save_outputs:
        with open(out_path / "retained_bins.txt", "w") as f:
            for b in retained_bins:
                f.write(f"{b}\n")

    # Step 2: impute remaining NaNs with 0.
    na_count = int(filtered.isna().sum().sum())
    total_entries = filtered.shape[0] * filtered.shape[1]
    if total_entries > 0:
        logger.info(
            f"Imputing {na_count} NaNs ({na_count / total_entries:.1%}) with 0"
        )
    imputed = filtered.fillna(0.0)

    # Step 2.5: optional per-cell normalization.
    # Removes per-cell global shift / scale effects (driven by depth,
    # conversion efficiency, library complexity, k-NN connectivity
    # in the RWR step, etc.) that otherwise dominate the top SVD
    # component without carrying cell-type information.  Standard
    # scATAC-seq / scRNA-seq preprocessing trick.
    #
    # Modes:
    #   "off"     — no per-cell normalization (legacy behaviour).
    #   "center"  — subtract per-cell mean only (location shift).
    #   "zscore"  — subtract mean AND divide by per-cell std (location + scale).
    #   "rank"    — replace each cell's row with within-cell ranks scaled
    #               to (-0.5, 0.5).  Aggressive: removes both location
    #               and scale and is robust to outliers.  Closest
    #               analogue to TF-normalisation in LSI / scATAC-seq.
    #
    # The legacy ``per_cell_center=True`` parameter is treated as a
    # shorthand for ``per_cell_norm="center"`` when ``per_cell_norm``
    # is at its default "off".
    if per_cell_center and per_cell_norm == "off":
        per_cell_norm = "center"

    if per_cell_norm == "off":
        pass
    elif per_cell_norm == "center":
        cell_means = imputed.mean(axis=1)
        imputed = imputed.sub(cell_means, axis=0)
        filtered = filtered.sub(cell_means, axis=0)
        logger.info(
            f"Per-cell normalization: 'center' "
            f"(mean shift over cells = {cell_means.mean():.4f}, "
            f"std = {cell_means.std():.4f})"
        )
    elif per_cell_norm == "zscore":
        cell_means = imputed.mean(axis=1)
        cell_stds = imputed.std(axis=1).replace(0.0, 1.0)
        imputed = imputed.sub(cell_means, axis=0).div(cell_stds, axis=0)
        # Apply the same transform to `filtered` (NaN-safe).
        filt_means = filtered.mean(axis=1)
        filt_stds = filtered.std(axis=1).replace(0.0, 1.0)
        filtered = filtered.sub(filt_means, axis=0).div(filt_stds, axis=0)
        logger.info(
            f"Per-cell normalization: 'zscore' "
            f"(mean cell std = {cell_stds.mean():.4f})"
        )
    elif per_cell_norm == "rank":
        from scipy.stats import rankdata
        def _rank_row(row: np.ndarray) -> np.ndarray:
            mask = ~np.isnan(row)
            out = np.full_like(row, np.nan, dtype=np.float64)
            if mask.sum() == 0:
                return out
            ranks = rankdata(row[mask], method="average")
            out[mask] = ranks / (mask.sum() + 1) - 0.5
            return out
        imputed_arr = np.apply_along_axis(_rank_row, 1, imputed.to_numpy())
        filtered_arr = np.apply_along_axis(_rank_row, 1, filtered.to_numpy())
        imputed = pd.DataFrame(
            np.nan_to_num(imputed_arr, nan=0.0),
            index=imputed.index, columns=imputed.columns,
        )
        filtered = pd.DataFrame(
            filtered_arr, index=filtered.index, columns=filtered.columns,
        )
        logger.info(
            "Per-cell normalization: 'rank' (within-cell rank scaled to (-0.5, 0.5))"
        )
    else:
        raise ValueError(
            f"per_cell_norm must be one of 'off', 'center', 'zscore', 'rank'; "
            f"got {per_cell_norm!r}."
        )

    # Step 3: per-bin normalization.
    if matrix_norm == "zscore":
        def _zscore_col(col: pd.Series) -> pd.Series:
            std = col.std()
            if std is None or not np.isfinite(std) or std < 1e-10:
                return col * 0.0
            return (col - col.mean()) / std
        normalized = imputed.apply(_zscore_col, axis=0)
        logger.info("Per-bin normalization: z-score")
    elif matrix_norm == "quantile":
        # Use the un-imputed matrix so NaN positions don't bias the ranks.
        X = filtered.values.astype(float)
        mask = ~np.isnan(X)
        qn = _quantile_normalize_per_bin(X, mask=mask)
        # Center around 0 by subtracting 0.5; impute NaN to 0 (the centered median).
        qn_centered = np.nan_to_num(qn - 0.5, nan=0.0)
        normalized = pd.DataFrame(
            qn_centered, index=filtered.index, columns=filtered.columns,
        )
        logger.info("Per-bin normalization: quantile (rank-based)")
    else:
        raise ValueError(
            f"matrix_norm must be 'zscore' or 'quantile', got {matrix_norm!r}"
        )

    # Step 4 (optional): per-cell mean re-centering AFTER per-bin normalization.
    #
    # Per-cell rank → per-bin QN doesn't commute: the per-bin QN step
    # re-ranks values within each column, which shuffles the per-row
    # distribution and breaks per-cell mean=0.  Empirically this drives
    # the dominant SVD component on Δ_signed (correlation r = −0.87 with
    # per-cell median post-normalization), even though batch and other
    # confounders have been addressed upstream.
    #
    # The fix is one additional row-mean subtraction.  Since the matrix
    # is already per-bin-centered (col means ≈ 0 by construction of QN),
    # this is exactly the "double centering" formula
    # ``M' = M - row_means - col_means + grand_mean``
    # specialised to the case col_means ≈ 0, giving simply
    # ``M' = M - row_means``.  Closed-form, one shot, ~5 lines.
    if post_norm_recenter:
        row_shifts = normalized.mean(axis=1)
        normalized = normalized.sub(row_shifts, axis=0)
        logger.info(
            f"Post-normalization per-cell re-centering: "
            f"median absolute row shift = {float(row_shifts.abs().median()):.4f}"
        )
        # Apply the same to ``imputed`` so the saved imputed matrix
        # reflects the same row-mean correction (downstream UMAP also
        # uses this matrix for the 'unnormalized' diagnostic UMAP).
        imp_row_shifts = imputed.mean(axis=1)
        imputed = imputed.sub(imp_row_shifts, axis=0)

    if save_outputs:
        imputed.to_parquet(out_path / "concordance_imputed.parquet")
        imputed.to_csv(out_path / "concordance_imputed.tsv", sep="\t")
    # Filename reflects the actual normalization used.
    if save_outputs:
        normalized.to_parquet(out_path / f"concordance_{matrix_norm}.parquet")
        normalized.to_csv(out_path / f"concordance_{matrix_norm}.tsv", sep="\t")

    return {
        "retained_bins": retained_bins,
        "imputed": imputed,
        "zscore": normalized,
    }
