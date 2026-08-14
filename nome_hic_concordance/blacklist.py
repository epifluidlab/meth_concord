"""Shared blacklist (e.g. ENCODE) handling: BED loading + read/bin filters.

Used by:
  - bam_parser.parse_all_cells: drop read pairs where either end overlaps.
  - matrix_ops.process_concordance_matrix: drop bins whose interval overlaps.
"""

from __future__ import annotations

import gzip
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_blacklist_bed(path: str) -> dict[str, np.ndarray]:
    """Read a BED file and return ``{chrom: ndarray of shape (n, 2)}``.

    Overlapping intervals on the same chromosome are merged so that the
    per-chrom array is non-overlapping and sorted by start, which lets
    membership / overlap checks use a single ``np.searchsorted`` lookup.

    Plain text or gzipped BED is accepted.  Header / comment / track /
    browser lines are skipped.  Only the first three columns (chrom,
    start, end) are read.
    """
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


def interval_overlaps_blacklist(
    chrom: str, start: int, end: int,
    blacklist: dict[str, np.ndarray],
) -> bool:
    """Return True iff ``[start, end)`` overlaps any interval in ``blacklist[chrom]``.

    Assumes ``blacklist[chrom]`` is sorted and non-overlapping (as produced
    by :func:`load_blacklist_bed`).
    """
    ivals = blacklist.get(chrom)
    if ivals is None or len(ivals) == 0:
        return False
    starts = ivals[:, 0]
    ends = ivals[:, 1]
    # Index of last interval whose start < end.
    # side="left" so an interval starting exactly at `end` (touching but
    # not overlapping the half-open [start, end)) is excluded.
    idx = int(np.searchsorted(starts, end, side="left") - 1)
    if idx < 0:
        return False
    return bool(ends[idx] > start)


def _positions_in_blacklist(
    chroms: np.ndarray,
    positions: np.ndarray,
    blacklist: dict[str, np.ndarray],
) -> np.ndarray:
    """Vectorized per-chrom check: is each ``(chrom, pos)`` inside any
    blacklist interval?

    Returns a boolean ndarray of the same length as ``positions``.
    """
    in_bl = np.zeros(len(positions), dtype=bool)
    if len(positions) == 0 or not blacklist:
        return in_bl

    # Process each chromosome that appears in both df and blacklist.
    unique_chroms = pd.unique(chroms)
    for chrom in unique_chroms:
        ivals = blacklist.get(chrom)
        if ivals is None or len(ivals) == 0:
            continue
        mask_c = chroms == chrom
        pos_c = positions[mask_c]
        starts = ivals[:, 0]
        ends = ivals[:, 1]
        # For each position, find the index of the last interval whose
        # start <= pos.  Then it's "in blacklist" iff ends[idx] > pos.
        # side="right" so that a position equal to a start (i.e., the
        # first base of an interval) IS considered inside.
        idx = np.searchsorted(starts, pos_c, side="right") - 1
        valid = idx >= 0
        if valid.any():
            ends_at_idx = np.where(valid, ends[np.clip(idx, 0, len(ends) - 1)], -1)
            hit = valid & (ends_at_idx > pos_c)
            in_bl[mask_c] = hit
    return in_bl


def filter_read_pairs_blacklist(
    df: pd.DataFrame,
    blacklist: dict[str, np.ndarray],
    chrom_col: str = "chrom",
    pos_cols: tuple[str, str] = ("pos1", "pos2"),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows whose either end position falls inside a blacklist interval.

    A read pair has columns ``chrom``, ``pos1``, ``pos2`` (and possibly
    a different chrom for pos2, but in current cis-only pipelines both
    ends share a chromosome).  A row is dropped if **either** ``pos1``
    or ``pos2`` lies inside any blacklist interval on ``chrom``.

    Returns
    -------
    (filtered_df, stats)
        ``stats = {"n_in", "n_out", "n_dropped_pos1", "n_dropped_pos2", "n_dropped_either"}``
    """
    n_in = len(df)
    if n_in == 0 or not blacklist:
        return df, {
            "n_in": n_in, "n_out": n_in,
            "n_dropped_pos1": 0, "n_dropped_pos2": 0, "n_dropped_either": 0,
        }

    chroms = df[chrom_col].astype(str).to_numpy()
    p1 = df[pos_cols[0]].to_numpy(dtype=np.int64, copy=False)
    p2 = df[pos_cols[1]].to_numpy(dtype=np.int64, copy=False)

    in_bl_1 = _positions_in_blacklist(chroms, p1, blacklist)
    in_bl_2 = _positions_in_blacklist(chroms, p2, blacklist)
    drop = in_bl_1 | in_bl_2
    keep = ~drop

    out = df.loc[keep].reset_index(drop=True)
    return out, {
        "n_in": n_in,
        "n_out": len(out),
        "n_dropped_pos1": int(in_bl_1.sum()),
        "n_dropped_pos2": int(in_bl_2.sum()),
        "n_dropped_either": int(drop.sum()),
    }


def infer_bin_size(columns: pd.Index) -> int | None:
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


def blacklisted_bin_mask(columns, blacklist: dict, bin_size: int | None = None):
    """Boolean array, True where a ``chrom:start`` bin overlaps the blacklist.

    Bin-level rather than read-level: these matrices are already built, so the
    only thing left to do about a blacklisted region is drop the bins covering
    it.  That is coarser than the Step-2 read filter -- a 250 kb bin is dropped
    for overlapping a 1 kb blacklist interval -- so prefer re-running Step 2
    with ``--blacklist-bed`` when the matrices can be rebuilt, and use this
    when they cannot.

    Chromosome names are matched with and without a ``chr`` prefix, since a
    BED and a matrix that disagree about it are otherwise silently disjoint --
    which looks like a clean blacklist rather than a broken one.
    """
    import numpy as _np

    cols = list(columns)
    if bin_size is None:
        bin_size = infer_bin_size(pd.Index(cols))
    if not bin_size:
        raise ValueError(
            "Cannot infer the bin size from the column labels, so a bin's "
            "interval is unknown; pass bin_size explicitly.")
    out = _np.zeros(len(cols), dtype=bool)
    for i, col in enumerate(cols):
        s = str(col)
        if ":" not in s:
            continue
        chrom, start_str = s.split(":", 1)
        try:
            start = int(start_str)
        except ValueError:
            continue
        alt = chrom[3:] if chrom.lower().startswith("chr") else "chr" + chrom
        for name in (chrom, alt):
            if name in blacklist and interval_overlaps_blacklist(
                    name, start, start + int(bin_size), blacklist):
                out[i] = True
                break
    return out


def drop_blacklisted_bins(df: pd.DataFrame, bed_paths, bin_size=None,
                          who: str = "matrix"):
    """Drop the columns of a cells × bins frame that overlap any of the BEDs."""
    paths = [p for p in (bed_paths or []) if p]
    if not paths:
        return df
    merged: dict = {}
    for p in paths:
        for chrom, ivals in load_blacklist_bed(p).items():
            merged[chrom] = (ivals if chrom not in merged
                             else np.vstack([merged[chrom], ivals]))
    for chrom, ivals in merged.items():                 # re-merge across files
        order = np.argsort(ivals[:, 0], kind="stable")
        ivals = ivals[order]
        keep = [ivals[0]]
        for s, e in ivals[1:]:
            if s <= keep[-1][1]:
                keep[-1] = (keep[-1][0], max(keep[-1][1], e))
            else:
                keep.append((s, e))
        merged[chrom] = np.asarray(keep, dtype=np.int64)
    hit = blacklisted_bin_mask(df.columns, merged, bin_size)
    if hit.any():
        logger.info(f"{who}: dropped {int(hit.sum()):,} of {df.shape[1]:,} "
                    f"bins overlapping {len(paths)} blacklist file(s).")
    else:
        logger.warning(
            f"{who}: the blacklist matched NO bin.  Check the chromosome "
            f"naming and the coordinates — an empty filter and a clean "
            f"genome look identical downstream.")
    return df.loc[:, ~hit]
