"""Per-bin concordance: Pearson r of paired read-end posteriors.

For each (cell, bin) with at least ``min_pairs_per_bin`` qualifying pairs,
we compute the Pearson correlation of the two posterior columns. Bins are
assigned using the midpoint of the two read ends. Pairs below the distance
or GCH-site thresholds are discarded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def _compute_cell_concordance(
    posteriors_path: str,
    cell_name: str,
    bin_size: int,
    min_distance: int,
    min_gch_sites: int,
    min_pairs_per_bin: int,
) -> tuple[str, dict[str, float], dict[str, int]]:
    """Compute per-bin concordance and pair counts for one cell."""
    df = pd.read_parquet(posteriors_path)

    # Required columns:
    required = {"chrom", "pos1", "pos2", "n1", "n2", "distance", "posterior1", "posterior2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Posteriors file for {cell_name} is missing columns {sorted(missing)}"
        )

    df = df[
        (df["distance"] >= min_distance)
        & (df["n1"] >= min_gch_sites)
        & (df["n2"] >= min_gch_sites)
        & df["posterior1"].notna()
        & df["posterior2"].notna()
    ]

    if len(df) == 0:
        return cell_name, {}, {}

    midpoint = (df["pos1"].astype(np.int64) + df["pos2"].astype(np.int64)) // 2
    bin_start = (midpoint // int(bin_size)) * int(bin_size)
    bin_id = df["chrom"].astype(str) + ":" + bin_start.astype(str)

    df = df.assign(bin_id=bin_id.values)

    concordance: dict[str, float] = {}
    pair_counts: dict[str, int] = {}

    for bid, group in df.groupby("bin_id", sort=False):
        n_pairs = len(group)
        pair_counts[bid] = int(n_pairs)

        if n_pairs < min_pairs_per_bin:
            concordance[bid] = np.nan
            continue

        p1 = group["posterior1"].to_numpy(dtype=float)
        p2 = group["posterior2"].to_numpy(dtype=float)

        if np.std(p1) < 1e-10 or np.std(p2) < 1e-10:
            concordance[bid] = np.nan
            continue

        r = float(np.corrcoef(p1, p2)[0, 1])
        concordance[bid] = r

    return cell_name, concordance, pair_counts


def compute_concordance_matrix(
    cell_posterior_paths: dict[str, str],
    chrom_sizes: dict[str, int] | None = None,
    bin_size: int = 5_000_000,
    min_distance: int = 20_000,
    min_gch_sites: int = 3,
    min_pairs_per_bin: int = 5,
    n_threads: int = 4,
    output_dir: str = "output/05_concordance",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute a (cells x bins) matrix of Pearson concordance values.

    Parameters
    ----------
    cell_posterior_paths : dict
        Mapping ``cell_name -> posteriors.parquet``.
    chrom_sizes : dict, optional
        Mapping ``chrom -> length`` used for sorting bin columns. If None,
        bins are sorted lexicographically by chromosome, then by start.
    bin_size : int
        Genomic bin size in bp.
    min_distance, min_gch_sites, min_pairs_per_bin : int
        Filtering thresholds (see design doc).
    n_threads : int
    output_dir : str

    Returns
    -------
    (concordance_df, pair_counts_df) : both DataFrames indexed by cell,
        columns are bin identifiers like ``"chr1:0"``.
    """
    from joblib import Parallel, delayed

    out_path = ensure_dir(output_dir)

    items = list(cell_posterior_paths.items())
    if n_threads > 1 and len(items) > 1:
        results = Parallel(n_jobs=n_threads)(
            delayed(_compute_cell_concordance)(
                path, cell_name, bin_size, min_distance, min_gch_sites, min_pairs_per_bin,
            )
            for cell_name, path in items
        )
    else:
        results = [
            _compute_cell_concordance(
                path, cell_name, bin_size, min_distance, min_gch_sites, min_pairs_per_bin,
            )
            for cell_name, path in items
        ]

    all_bins: set[str] = set()
    for _, conc, _ in results:
        all_bins.update(conc.keys())

    sorted_bins = _sort_bins(all_bins, chrom_sizes)

    concordance_data: dict[str, dict[str, float]] = {}
    pair_count_data: dict[str, dict[str, int]] = {}
    for cell_name, conc, counts in results:
        concordance_data[cell_name] = {b: conc.get(b, np.nan) for b in sorted_bins}
        pair_count_data[cell_name] = {b: counts.get(b, 0) for b in sorted_bins}

    concordance_df = pd.DataFrame.from_dict(concordance_data, orient="index")
    pair_counts_df = pd.DataFrame.from_dict(pair_count_data, orient="index")

    # Preserve column order
    concordance_df = concordance_df.reindex(columns=sorted_bins)
    pair_counts_df = pair_counts_df.reindex(columns=sorted_bins).fillna(0).astype(np.int64)

    concordance_df.columns.name = "bin"
    concordance_df.index.name = "cell"
    pair_counts_df.columns.name = "bin"
    pair_counts_df.index.name = "cell"

    concordance_df.to_parquet(out_path / "concordance_raw.parquet")
    pair_counts_df.to_parquet(out_path / "pair_counts_per_bin.parquet")
    concordance_df.to_csv(out_path / "concordance_raw.tsv", sep="\t")
    pair_counts_df.to_csv(out_path / "pair_counts_per_bin.tsv", sep="\t")

    logger.info(
        f"Concordance matrix: {concordance_df.shape[0]} cells x {concordance_df.shape[1]} bins"
    )
    return concordance_df, pair_counts_df


def _sort_bins(bins: set[str], chrom_sizes: dict[str, int] | None) -> list[str]:
    """Sort bin identifiers like ``'chr1:0'`` by chromosome and start."""
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
        # Natural sort on chromosome name, then start.
        def _chrom_key(chrom: str) -> tuple[int, int | str]:
            base = chrom[3:] if chrom.startswith("chr") else chrom
            if base.isdigit():
                return (0, int(base))
            return (1, base)

        parsed.sort(key=lambda t: (_chrom_key(t[0]), t[1]))

    return [t[2] for t in parsed]
