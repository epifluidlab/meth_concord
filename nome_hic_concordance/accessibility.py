"""Per-(cell, bin) mean GCH accessibility.

For each cell × bin, computes the mean of the Layer-1 per-end posterior
``P(end is open)`` across all informative ends (n_GCH >= min_gch_sites,
posterior not NaN) that fall in the bin.

This is the *dense* companion feature to the (sparse) concordance matrix:
each (cell, bin) typically accumulates 100s–1000s of informative ends
(because we count *every* informative end, linked or unlinked, regardless
of distance to its mate), so the resulting cells × bins matrix has
near-100% coverage with low per-(cell, bin) variance.

Two uses:

1. **k-NN basis for RWR imputation** of the soft-count matrices (see
   :mod:`nome_hic_concordance.imputation`).  The QN'd version of this
   matrix gives stable cell-cell similarities, which the concordance
   matrix at pre-Layer-2 is too sparse to provide on its own.

2. **Baseline feature in the comparison harness** — the question "does
   concordance add information beyond bin-mean accessibility?" can only
   be asked once both matrices exist.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def _compute_cell_accessibility(
    posteriors_path: str,
    cell_name: str,
    bin_size: int,
    min_gch_sites: int,
) -> tuple[str, dict[str, tuple[float, int]]]:
    """For one cell, return ``{bin_id: (sum_posterior, n_ends)}``.

    Each pair's two ends are processed independently. An end contributes
    its posterior to its own bin iff ``n >= min_gch_sites`` AND
    ``posterior`` is finite.
    """
    df = pd.read_parquet(
        posteriors_path,
        columns=["chrom", "pos1", "pos2", "n1", "n2", "posterior1", "posterior2"],
    )

    if len(df) == 0:
        return cell_name, {}

    chrom = df["chrom"].astype(str).values
    pos1 = df["pos1"].to_numpy(dtype=np.int64)
    pos2 = df["pos2"].to_numpy(dtype=np.int64)
    n1 = df["n1"].to_numpy(dtype=np.int32)
    n2 = df["n2"].to_numpy(dtype=np.int32)
    p1 = df["posterior1"].to_numpy(dtype=np.float64, na_value=np.nan)
    p2 = df["posterior2"].to_numpy(dtype=np.float64, na_value=np.nan)

    bsz = int(bin_size)
    bin1 = (pos1 // bsz) * bsz
    bin2 = (pos2 // bsz) * bsz

    inf1 = (n1 >= min_gch_sites) & np.isfinite(p1)
    inf2 = (n2 >= min_gch_sites) & np.isfinite(p2)

    if not (inf1.any() or inf2.any()):
        return cell_name, {}

    # Stack both ends into a single (chrom, bin, posterior) triple table
    # so the per-bin aggregation runs once via pandas groupby (faster than
    # a per-bin Python loop on dicts).
    rows = []
    if inf1.any():
        rows.append(pd.DataFrame({
            "chrom": chrom[inf1],
            "bin": bin1[inf1],
            "posterior": p1[inf1],
        }))
    if inf2.any():
        rows.append(pd.DataFrame({
            "chrom": chrom[inf2],
            "bin": bin2[inf2],
            "posterior": p2[inf2],
        }))
    ends_df = pd.concat(rows, ignore_index=True) if len(rows) > 1 else rows[0]

    agg = ends_df.groupby(["chrom", "bin"], sort=False)["posterior"].agg(["sum", "size"])
    counts: dict[str, tuple[float, int]] = {
        f"{chrom_}:{int(bin_)}": (float(s), int(c))
        for (chrom_, bin_), s, c in zip(agg.index, agg["sum"].values, agg["size"].values)
    }
    return cell_name, counts


def compute_accessibility_matrix(
    cell_posterior_paths: dict[str, str],
    chrom_sizes: dict[str, int] | None = None,
    bin_size: int = 250_000,
    min_gch_sites: int = 2,
    n_threads: int = 4,
    output_dir: str = "output/05_concordance",
    save_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute a cells × bins matrix of mean per-end posterior.

    Parameters
    ----------
    cell_posterior_paths : dict
        ``cell_name -> *.posteriors.parquet`` from Step 4.
    chrom_sizes : dict, optional
        Used only for sorting bin columns; if None, natural-sort on
        chromosome name.
    bin_size : int
        Bin width in bp (default 250 kb, matching the concordance bin size).
    min_gch_sites : int
        Min GCH sites per end to be counted (default 2, per design-doc §6 #14).
    n_threads : int
    output_dir : str
        Where to write ``accessibility_mean.parquet`` and
        ``accessibility_n_ends.parquet``.  Ignored if ``save_outputs=False``.
    save_outputs : bool
        Set False when calling this function from tests or in-memory pipelines.

    Returns
    -------
    (accessibility_df, n_ends_df) :
        Both DataFrames are cells × bins.  ``accessibility_df`` holds
        mean posterior per (cell, bin) — NaN where the bin has zero
        informative ends in that cell.  ``n_ends_df`` holds the
        denominator (number of informative ends per (cell, bin)) for
        downstream weighting or QC.
    """
    from joblib import Parallel, delayed

    items = list(cell_posterior_paths.items())
    if n_threads > 1 and len(items) > 1:
        results = Parallel(n_jobs=n_threads)(
            delayed(_compute_cell_accessibility)(
                path, cell_name, bin_size, min_gch_sites,
            )
            for cell_name, path in items
        )
    else:
        results = [
            _compute_cell_accessibility(path, cell_name, bin_size, min_gch_sites)
            for cell_name, path in items
        ]

    # Collect all unique bins seen across cells.
    all_bins: set[str] = set()
    for _, counts in results:
        all_bins.update(counts.keys())
    sorted_bins = _sort_bins(all_bins, chrom_sizes)

    cell_names = [c for c, _ in results]
    n_cells = len(cell_names)
    n_bins = len(sorted_bins)
    bin_to_col = {b: i for i, b in enumerate(sorted_bins)}

    mean_arr = np.full((n_cells, n_bins), np.nan, dtype=np.float64)
    n_ends_arr = np.zeros((n_cells, n_bins), dtype=np.int32)

    for row_idx, (_, counts) in enumerate(results):
        for bid, (s, c) in counts.items():
            j = bin_to_col[bid]
            mean_arr[row_idx, j] = s / c if c > 0 else np.nan
            n_ends_arr[row_idx, j] = c

    accessibility_df = pd.DataFrame(mean_arr, index=cell_names, columns=sorted_bins)
    n_ends_df = pd.DataFrame(n_ends_arr, index=cell_names, columns=sorted_bins)
    accessibility_df.index.name = "cell"
    accessibility_df.columns.name = "bin"
    n_ends_df.index.name = "cell"
    n_ends_df.columns.name = "bin"

    if save_outputs:
        out_path = ensure_dir(output_dir)
        accessibility_df.to_parquet(out_path / "accessibility_mean.parquet")
        n_ends_df.to_parquet(out_path / "accessibility_n_ends.parquet")
        # CSV mirrors only for interactive inspection — parquets are the source of truth.
        accessibility_df.to_csv(out_path / "accessibility_mean.tsv", sep="\t")

    positive_ends = n_ends_arr[n_ends_arr > 0]
    median_ends = int(np.median(positive_ends)) if positive_ends.size > 0 else 0
    logger.info(
        f"Accessibility matrix: {n_cells} cells x {n_bins} bins "
        f"({accessibility_df.notna().sum().sum():,} non-NaN entries; "
        f"median ends/bin/cell = {median_ends})"
    )
    return accessibility_df, n_ends_df


def _sort_bins(bins: set[str], chrom_sizes: dict[str, int] | None) -> list[str]:
    """Sort bin identifiers like 'chr1:0' by (chrom, start).

    Identical to the helper in concordance.py — kept here so accessibility.py
    has no upstream dependency on concordance.py.  If we add a third module
    needing this, factor into a shared utility.
    """
    parsed: list[tuple[str, int, str]] = []
    for b in bins:
        chrom, start_str = b.split(":", 1)
        try:
            start = int(start_str)
        except ValueError:
            start = 0
        parsed.append((chrom, start, b))

    if chrom_sizes is not None:
        order = {c: i for i, c in enumerate(chrom_sizes.keys())}
        parsed.sort(key=lambda t: (order.get(t[0], 1_000_000), t[1]))
    else:
        def _chrom_key(chrom: str) -> tuple[int, int | str]:
            base = chrom[3:] if chrom.startswith("chr") else chrom
            if base.isdigit():
                return (0, int(base))
            return (1, base)
        parsed.sort(key=lambda t: (_chrom_key(t[0]), t[1]))

    return [b for _, _, b in parsed]
