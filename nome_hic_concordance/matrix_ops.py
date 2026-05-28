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
    out_path = ensure_dir(output_dir)
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

    imputed.to_parquet(out_path / "concordance_imputed.parquet")
    imputed.to_csv(out_path / "concordance_imputed.tsv", sep="\t")
    # Filename reflects the actual normalization used.
    normalized.to_parquet(out_path / f"concordance_{matrix_norm}.parquet")
    normalized.to_csv(out_path / f"concordance_{matrix_norm}.tsv", sep="\t")

    return {
        "retained_bins": retained_bins,
        "imputed": imputed,
        "zscore": normalized,
    }
