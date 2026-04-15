"""Matrix-level operations on the cells x bins concordance matrix.

Applies bin filtering, NaN imputation (fill to 0, the shrinkage target),
and per-bin z-scoring across cells.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def process_concordance_matrix(
    concordance_shrunk: pd.DataFrame,
    min_cell_coverage: float = 0.8,
    output_dir: str = "output/05_concordance",
) -> dict[str, object]:
    """Filter bins, impute NaNs with 0, and z-score per bin.

    Returns
    -------
    dict with keys:
        - 'retained_bins' : list[str]
        - 'imputed' : DataFrame (cells x retained_bins), NaN-free
        - 'zscore'  : DataFrame (cells x retained_bins), z-scored per column
    """
    out_path = ensure_dir(output_dir)
    n_cells, n_bins_orig = concordance_shrunk.shape

    # Step 1: filter bins by fraction of non-NA cells.
    if n_cells == 0:
        raise ValueError("concordance_shrunk has zero cells")
    coverage_frac = concordance_shrunk.notna().mean(axis=0)
    retained_mask = coverage_frac >= min_cell_coverage
    retained_bins = concordance_shrunk.columns[retained_mask].tolist()

    filtered = concordance_shrunk.loc[:, retained_bins].copy()
    logger.info(
        f"Bin filter: {n_bins_orig} -> {len(retained_bins)} bins "
        f"(coverage >= {min_cell_coverage:.0%} of cells)"
    )

    with open(out_path / "retained_bins.txt", "w") as f:
        for b in retained_bins:
            f.write(f"{b}\n")

    # Step 2: impute remaining NaNs with 0 (the shrinkage target).
    na_count = int(filtered.isna().sum().sum())
    total_entries = filtered.shape[0] * filtered.shape[1]
    if total_entries > 0:
        logger.info(
            f"Imputing {na_count} NaNs ({na_count / total_entries:.1%}) with 0"
        )
    imputed = filtered.fillna(0.0)

    # Step 3: per-bin z-score across cells. Bins with zero variance stay at 0.
    def _zscore_col(col: pd.Series) -> pd.Series:
        std = col.std()
        if std is None or not np.isfinite(std) or std < 1e-10:
            return col * 0.0
        return (col - col.mean()) / std

    zscore = imputed.apply(_zscore_col, axis=0)

    imputed.to_parquet(out_path / "concordance_imputed.parquet")
    imputed.to_csv(out_path / "concordance_imputed.tsv", sep="\t")
    zscore.to_parquet(out_path / "concordance_zscore.parquet")
    zscore.to_csv(out_path / "concordance_zscore.tsv", sep="\t")

    return {
        "retained_bins": retained_bins,
        "imputed": imputed,
        "zscore": zscore,
    }
