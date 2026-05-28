"""Matrix-level operations on the cells x bins concordance matrix.

Applies bin filtering, NaN imputation, and per-bin normalization
(z-score or quantile-rank) across cells.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def _infer_bin_size(columns: pd.Index) -> int | None:
    """Infer bin_size in bp from column labels like ``"chr1:0"``, ``"chr1:250000"``...

    Returns the GCD of all consecutive start-position differences within
    each chromosome (the only number that consistently divides every gap).
    Returns ``None`` if there aren't enough labels to infer.
    """
    import math
    from functools import reduce

    starts_by_chrom: dict[str, list[int]] = {}
    for col in columns:
        s = str(col)
        if ":" not in s:
            continue
        chrom, start_str = s.split(":", 1)
        try:
            starts_by_chrom.setdefault(chrom, []).append(int(start_str))
        except ValueError:
            continue

    diffs: list[int] = []
    for starts in starts_by_chrom.values():
        unique_sorted = sorted(set(starts))
        for a, b in zip(unique_sorted, unique_sorted[1:]):
            d = b - a
            if d > 0:
                diffs.append(d)
    if not diffs:
        return None
    return int(reduce(math.gcd, diffs))


def _load_blacklist_bed(path: str) -> dict[str, np.ndarray]:
    """Read a BED file and return ``{chrom: ndarray of shape (n, 2)}``.

    Overlapping intervals on the same chromosome are merged so that the
    per-chrom array is non-overlapping and sorted by start, which lets
    bin-overlap checks use a single ``np.searchsorted`` lookup.

    The BED is expected to be plain text or gzipped.  Header / comment /
    track / browser lines are skipped.  Only the first three columns
    (chrom, start, end) are read.
    """
    import gzip
    raw: dict[str, list[tuple[int, int]]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for ln in fh:
            if not ln or ln.startswith(("#", "track", "browser")):
                continue
            fields = ln.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            chrom = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                continue
            if end <= start:
                continue
            raw.setdefault(chrom, []).append((start, end))

    merged: dict[str, np.ndarray] = {}
    n_in, n_out = 0, 0
    for chrom, ivals in raw.items():
        ivals.sort()
        out: list[tuple[int, int]] = []
        for s, e in ivals:
            if out and s <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        n_in += len(ivals)
        n_out += len(out)
        merged[chrom] = np.array(out, dtype=np.int64)
    logger.info(
        f"Loaded blacklist BED: {n_in:,} intervals across {len(merged)} "
        f"chromosomes, merged to {n_out:,} non-overlapping intervals."
    )
    return merged


def _bin_overlaps_blacklist(
    chrom: str, bin_start: int, bin_end: int,
    blacklist: dict[str, np.ndarray],
) -> bool:
    """Return True iff ``[bin_start, bin_end)`` overlaps any interval in ``blacklist[chrom]``.

    Assumes ``blacklist[chrom]`` is sorted and non-overlapping (as produced
    by ``_load_blacklist_bed``).
    """
    ivals = blacklist.get(chrom)
    if ivals is None or len(ivals) == 0:
        return False
    starts = ivals[:, 0]
    ends = ivals[:, 1]
    # Index of last interval whose start < bin_end (the only candidate
    # for overlap, given merged non-overlapping intervals).
    # side="left" so that an interval starting exactly at bin_end (which
    # would touch but not overlap) is excluded.
    idx = int(np.searchsorted(starts, bin_end, side="left") - 1)
    if idx < 0:
        return False
    return bool(ends[idx] > bin_start)


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
    blacklist_bed: str | None = None,
    bin_size: int | None = None,
) -> dict[str, object]:
    """Filter bins, impute NaNs, and normalize per bin.

    Parameters
    ----------
    concordance_shrunk : DataFrame
        cells × bins matrix (Pearson r, shrunk r, or Δ values; NaN-tolerant).
        Bin column labels must look like ``"chr1:0"``, ``"chr1:250000"``, ...
    min_cell_coverage : float
        Drop bins where < ``min_cell_coverage`` of cells have a value.
    output_dir : str
    matrix_norm : "zscore" | "quantile"
        Per-bin normalization across cells.  ``"zscore"`` (default) for the
        classical pipeline; ``"quantile"`` for rank-based QN to a uniform
        [-0.5, 0.5] target — robust to outliers, matches bulk-RNA /
        scATAC-seq style preprocessing.
    blacklist_bed : str, optional
        Path to a BED file of blacklist regions (e.g. ENCODE).  Bins whose
        ``[start, start+bin_size)`` interval overlaps any blacklist record
        are dropped.  Requires ``bin_size`` to be set.
    bin_size : int, optional
        Bin size in bp.  Only used when ``blacklist_bed`` is given.

    Returns
    -------
    dict with keys:
        - 'retained_bins' : list[str]
        - 'imputed' : DataFrame (cells x retained_bins), NaN-free
        - 'zscore'  : DataFrame (cells x retained_bins), normalized per column.
                      The label "zscore" is kept for backward compatibility
                      regardless of which ``matrix_norm`` was used; check
                      ``matrix_norm`` to know how the values were produced.
    """
    out_path = ensure_dir(output_dir)
    n_cells, n_bins_orig = concordance_shrunk.shape

    # Step 1a: filter bins by fraction of non-NA cells.
    if n_cells == 0:
        raise ValueError("concordance_shrunk has zero cells")
    coverage_frac = concordance_shrunk.notna().mean(axis=0)
    retained_mask = coverage_frac >= min_cell_coverage

    # Step 1b: drop bins overlapping any blacklist interval (optional).
    if blacklist_bed:
        # Infer bin_size from the matrix labels — this is the ground truth.
        # If the caller also passed bin_size, validate it matches.
        inferred_bin_size = _infer_bin_size(concordance_shrunk.columns)
        if inferred_bin_size is None:
            if not bin_size:
                raise ValueError(
                    "process_concordance_matrix: blacklist_bed needs bin_size "
                    "and could not infer it from the matrix column labels "
                    "(expected labels like 'chr1:0', 'chr1:250000', ...)."
                )
            effective_bin_size = int(bin_size)
            logger.info(
                f"Could not infer bin_size from matrix labels; using "
                f"bin_size={effective_bin_size} from the caller."
            )
        else:
            effective_bin_size = inferred_bin_size
            if bin_size and int(bin_size) != inferred_bin_size:
                logger.warning(
                    f"--bin-size={bin_size} disagrees with bin_size inferred "
                    f"from the matrix labels ({inferred_bin_size}). Using the "
                    f"inferred value for the blacklist overlap check. (This "
                    f"typically happens when you re-run --steps 7 without "
                    f"passing the same --bin-size used to build the matrix.)"
                )
            else:
                logger.info(
                    f"Blacklist overlap check using bin_size="
                    f"{effective_bin_size} (inferred from matrix labels)."
                )

        blacklist = _load_blacklist_bed(blacklist_bed)
        bin_blacklisted = []
        for bid in concordance_shrunk.columns:
            chrom, start_str = str(bid).split(":", 1)
            start = int(start_str)
            bin_blacklisted.append(
                _bin_overlaps_blacklist(
                    chrom, start, start + effective_bin_size, blacklist,
                )
            )
        bl_mask = pd.Series(bin_blacklisted, index=concordance_shrunk.columns)
        n_bl = int(bl_mask.sum())
        n_bl_overlapping_kept = int((bl_mask & retained_mask).sum())
        retained_mask = retained_mask & ~bl_mask
        logger.info(
            f"Blacklist filter: {n_bl:,} of {len(concordance_shrunk.columns):,} "
            f"bins overlap blacklist regions (of which {n_bl_overlapping_kept:,} "
            f"would have passed the coverage filter); dropping all."
        )

    retained_bins = concordance_shrunk.columns[retained_mask].tolist()

    filtered = concordance_shrunk.loc[:, retained_bins].copy()
    logger.info(
        f"Bin filter: {n_bins_orig} -> {len(retained_bins)} bins "
        f"(coverage >= {min_cell_coverage:.0%}"
        f"{', not in blacklist' if blacklist_bed else ''})"
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
