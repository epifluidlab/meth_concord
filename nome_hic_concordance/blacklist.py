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
