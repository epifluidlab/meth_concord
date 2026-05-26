"""Per-cell linked/unlinked aggregation for pair-concordance analysis.

For a given pair (region_i, region_j) from the panel, classifies each read
pair from a cell's posteriors.parquet into one of three categories:

* **Linked** — pos1 in region_i AND pos2 in region_j (or the symmetric
  placement), with distance between the two read ends >= ``min_distance``.
  Both ends must be informative (>= ``min_gch_sites``).
* **Unlinked-i** — one read end in region_i, mate NOT in region_j,
  AND mate within ``unlinked_max_distance`` of anchor_i's center.
  Only the in-anchor end must be informative.
* **Unlinked-j** — one read end in region_j, mate NOT in region_i,
  AND mate within ``unlinked_max_distance`` of anchor_j's center.
  Only the in-anchor end must be informative.

Read pairs are further filtered by a posterior threshold and minimum GCH
site count before being counted.

For efficiency, per-chromosome sorted anchor arrays and ``np.searchsorted``
are used instead of full interval trees, allowing a single sweep over the
posteriors DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interval index: sorted-array lookup with searchsorted
# ---------------------------------------------------------------------------


def _build_anchor_index(
    panel: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    """Build per-chromosome sorted anchor arrays for fast overlap queries.

    For each chromosome, produces four parallel sorted arrays plus the
    maximum anchor size on that chromosome (used to bound searchsorted
    range queries):

    * ``starts`` — anchor start positions (sorted).
    * ``ends`` — anchor end positions (matching order).
    * ``pair_indices`` — row index into the panel DataFrame.
    * ``sides`` — 0 for anchor_i, 1 for anchor_j.
    * ``max_size`` — max(end - start) over all anchors on this chromosome.

    Returns
    -------
    dict
        ``chrom -> (starts, ends, pair_indices, sides, max_size)``
    """
    records: dict[str, list[tuple[int, int, int, int]]] = {}

    # Vectorised collection.
    chrom_i = panel["chrom_i"].to_numpy()
    start_i = panel["start_i"].to_numpy(dtype=np.int64)
    end_i = panel["end_i"].to_numpy(dtype=np.int64)
    chrom_j = panel["chrom_j"].to_numpy()
    start_j = panel["start_j"].to_numpy(dtype=np.int64)
    end_j = panel["end_j"].to_numpy(dtype=np.int64)

    for row_idx in range(len(panel)):
        records.setdefault(chrom_i[row_idx], []).append(
            (int(start_i[row_idx]), int(end_i[row_idx]), row_idx, 0)
        )
        records.setdefault(chrom_j[row_idx], []).append(
            (int(start_j[row_idx]), int(end_j[row_idx]), row_idx, 1)
        )

    index: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for chrom, recs in records.items():
        recs.sort(key=lambda t: t[0])
        starts = np.array([r[0] for r in recs], dtype=np.int64)
        ends = np.array([r[1] for r in recs], dtype=np.int64)
        pair_indices = np.array([r[2] for r in recs], dtype=np.int64)
        sides = np.array([r[3] for r in recs], dtype=np.int8)
        max_size = int((ends - starts).max()) if len(ends) else 0
        index[chrom] = (starts, ends, pair_indices, sides, max_size)

    return index


def _find_overlapping_anchors(
    pos: int,
    chrom_starts: np.ndarray,
    chrom_ends: np.ndarray,
    chrom_pair_indices: np.ndarray,
    chrom_sides: np.ndarray,
    max_size: int,
) -> list[tuple[int, int]]:
    """Find all anchors overlapping a genomic position — O(log N) + O(k).

    Any anchor with start <= pos < end must satisfy
    ``start in (pos - max_size, pos]``.  We use two ``searchsorted`` calls
    to bracket this range to a tight window of size O(max_size / density),
    then linearly check the few candidates' end coordinates.

    For a 5 kb-anchor panel with ~5 kb spacing, this is ~1 candidate per
    query, vs. the previous O(N) backward scan that examined ALL
    anchors with start <= pos on the chromosome.
    """
    if len(chrom_starts) == 0:
        return []

    # Restrict candidates to start in (pos - max_size, pos].
    lo = int(np.searchsorted(chrom_starts, pos - max_size, side="left"))
    hi = int(np.searchsorted(chrom_starts, pos, side="right"))
    if lo >= hi:
        return []

    hits: list[tuple[int, int]] = []
    for k in range(lo, hi):
        if chrom_ends[k] > pos:
            hits.append((int(chrom_pair_indices[k]), int(chrom_sides[k])))
    return hits


# ---------------------------------------------------------------------------
# Per-cell aggregation
# ---------------------------------------------------------------------------


def aggregate_pair_counts(
    posteriors_parquet: str,
    cell_name: str,
    panel: pd.DataFrame,
    posterior_threshold: float = 0.5,
    min_gch_sites: int = 3,
    min_distance: int = 20_000,
    unlinked_max_distance: int = 20_000,
) -> pd.DataFrame:
    """Aggregate linked and unlinked counts at every panel pair for one cell.

    For each read pair in the posteriors file, the two read-end positions
    (``pos1``, ``pos2``) are tested against every anchor in the panel.  Read
    pairs are classified as:

    * **Linked** — one end in anchor_i and the other in anchor_j of the
      same panel pair, with distance between the two read ends >=
      ``min_distance``.  Both ends must be informative (n >= min_gch_sites).
      ``k_linked_active`` counts pairs where *both* posteriors exceed
      ``posterior_threshold``.  ``k_linked_symmetric`` counts pairs where
      both posteriors agree (both above or both below threshold).
    * **Unlinked-i** — one end in anchor_i, mate not in anchor_j, AND
      mate within ``unlinked_max_distance`` of anchor_i's center.  Only
      the in-anchor end must be informative.
      ``k_unlinked_i_open`` counts those where the anchor_i end's
      posterior exceeds the threshold.
    * **Unlinked-j** — one end in anchor_j, mate not in anchor_i, AND
      mate within ``unlinked_max_distance`` of anchor_j's center.  Only
      the in-anchor end must be informative.
      ``k_unlinked_j_open`` counts those where the anchor_j end's
      posterior exceeds the threshold.

    Parameters
    ----------
    posteriors_parquet : str
        Path to a cell's posteriors parquet file.  Expected columns:
        chrom, pos1, pos2, n1, n2, posterior1, posterior2.
    cell_name : str
        Identifier for the cell.
    panel : pd.DataFrame
        Panel DataFrame with standard ``PANEL_COLUMNS``.
    posterior_threshold : float
        Threshold for calling a read end "accessible" (default 0.5).
    min_gch_sites : int
        Minimum informative GCH sites for a read end to be counted
        (default 3).
    min_distance : int
        Minimum genomic distance (bp) between the two read ends for
        a read pair to qualify as linked (default 20,000).
    unlinked_max_distance : int
        Maximum distance (bp) from the relevant anchor's center for
        the mate to qualify as an unlinked read (default 20,000).

    Returns
    -------
    pd.DataFrame
        One row per panel pair with columns: cell_name, pair_id,
        n_linked, k_linked_active, k_linked_symmetric,
        n_unlinked_i, k_unlinked_i_open,
        n_unlinked_j, k_unlinked_j_open.
    """
    df = pd.read_parquet(posteriors_parquet)

    required = {"chrom", "pos1", "pos2", "n1", "n2", "posterior1", "posterior2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Posteriors file for {cell_name} is missing columns {sorted(missing)}"
        )

    # Pre-filter: at least one end needs sufficient GCH sites and a valid
    # posterior.  (Linked requires *both* informative; unlinked only the
    # in-anchor end — we check per-end informativeness during classification.)
    df = df[
        (
            ((df["n1"] >= min_gch_sites) & df["posterior1"].notna())
            | ((df["n2"] >= min_gch_sites) & df["posterior2"].notna())
        )
    ].copy()

    n_pairs_panel = len(panel)

    # Accumulators.
    n_linked = np.zeros(n_pairs_panel, dtype=np.int64)
    k_linked_active = np.zeros(n_pairs_panel, dtype=np.int64)
    k_linked_symmetric = np.zeros(n_pairs_panel, dtype=np.int64)
    n_unlinked_i = np.zeros(n_pairs_panel, dtype=np.int64)
    k_unlinked_i_open = np.zeros(n_pairs_panel, dtype=np.int64)
    n_unlinked_j = np.zeros(n_pairs_panel, dtype=np.int64)
    k_unlinked_j_open = np.zeros(n_pairs_panel, dtype=np.int64)

    # Build anchor index.
    anchor_idx = _build_anchor_index(panel)

    # Precompute anchor centers for unlinked distance checks.
    # anchor_center_i[pair_idx] = midpoint of anchor_i for that panel pair.
    anchor_center_i = ((panel["start_i"].values + panel["end_i"].values) // 2).astype(
        np.int64
    )
    anchor_center_j = ((panel["start_j"].values + panel["end_j"].values) // 2).astype(
        np.int64
    )

    # Sweep each read pair once.
    for row in df.itertuples(index=False):
        chrom = row.chrom
        if chrom not in anchor_idx:
            continue

        chrom_starts, chrom_ends, chrom_pair_indices, chrom_sides, chrom_max_size = (
            anchor_idx[chrom]
        )

        pos1 = row.pos1
        pos2 = row.pos2
        n1 = row.n1
        n2 = row.n2
        p1 = row.posterior1
        p2 = row.posterior2
        distance = abs(pos2 - pos1)

        # Find which anchors each read end overlaps.
        hits1 = _find_overlapping_anchors(
            pos1, chrom_starts, chrom_ends, chrom_pair_indices, chrom_sides,
            chrom_max_size,
        )
        hits2 = _find_overlapping_anchors(
            pos2, chrom_starts, chrom_ends, chrom_pair_indices, chrom_sides,
            chrom_max_size,
        )

        if not hits1 and not hits2:
            continue

        # Index hits by (pair_index, side) for fast lookup.
        hit1_set: dict[int, set[int]] = {}
        for pair_idx, side in hits1:
            hit1_set.setdefault(pair_idx, set()).add(side)

        hit2_set: dict[int, set[int]] = {}
        for pair_idx, side in hits2:
            hit2_set.setdefault(pair_idx, set()).add(side)

        # Check for linked pairs: pos1 in anchor_i & pos2 in anchor_j
        # (or symmetric), with distance >= min_distance and both ends
        # informative.
        linked_pairs: set[int] = set()
        both_informative = (
            n1 >= min_gch_sites
            and n2 >= min_gch_sites
            and not (np.isnan(p1) if isinstance(p1, float) else False)
            and not (np.isnan(p2) if isinstance(p2, float) else False)
        )

        all_pair_indices = set(hit1_set.keys()) | set(hit2_set.keys())
        for pair_idx in all_pair_indices:
            sides1 = hit1_set.get(pair_idx, set())
            sides2 = hit2_set.get(pair_idx, set())

            is_linked = (0 in sides1 and 1 in sides2) or (1 in sides1 and 0 in sides2)

            if is_linked and distance >= min_distance and both_informative:
                linked_pairs.add(pair_idx)
                n_linked[pair_idx] += 1

                active1 = p1 >= posterior_threshold
                active2 = p2 >= posterior_threshold
                if active1 and active2:
                    k_linked_active[pair_idx] += 1
                if active1 == active2:
                    k_linked_symmetric[pair_idx] += 1

        # Unlinked-i: pos in anchor_i of some pair, mate NOT in anchor_j,
        # AND mate within unlinked_max_distance of anchor_i's center.
        # The in-anchor end must be informative.
        for pair_idx, side in hits1:
            if pair_idx in linked_pairs:
                continue
            if side == 0:
                # pos1 is in anchor_i; in-anchor end must be informative.
                if n1 < min_gch_sites or (isinstance(p1, float) and np.isnan(p1)):
                    continue
                # Check mate (pos2) is NOT in anchor_j.
                sides2 = hit2_set.get(pair_idx, set())
                if 1 not in sides2:
                    # Mate must be within unlinked_max_distance of anchor_i center.
                    if abs(pos2 - anchor_center_i[pair_idx]) <= unlinked_max_distance:
                        n_unlinked_i[pair_idx] += 1
                        if p1 >= posterior_threshold:
                            k_unlinked_i_open[pair_idx] += 1
            elif side == 1:
                # pos1 is in anchor_j; in-anchor end must be informative.
                if n1 < min_gch_sites or (isinstance(p1, float) and np.isnan(p1)):
                    continue
                # Check mate (pos2) is NOT in anchor_i.
                sides2 = hit2_set.get(pair_idx, set())
                if 0 not in sides2:
                    # Mate must be within unlinked_max_distance of anchor_j center.
                    if abs(pos2 - anchor_center_j[pair_idx]) <= unlinked_max_distance:
                        n_unlinked_j[pair_idx] += 1
                        if p1 >= posterior_threshold:
                            k_unlinked_j_open[pair_idx] += 1

        for pair_idx, side in hits2:
            if pair_idx in linked_pairs:
                continue
            if side == 0:
                # pos2 is in anchor_i; in-anchor end must be informative.
                if n2 < min_gch_sites or (isinstance(p2, float) and np.isnan(p2)):
                    continue
                # Check mate (pos1) is NOT in anchor_j.
                sides1 = hit1_set.get(pair_idx, set())
                if 1 not in sides1:
                    # Mate must be within unlinked_max_distance of anchor_i center.
                    if abs(pos1 - anchor_center_i[pair_idx]) <= unlinked_max_distance:
                        n_unlinked_i[pair_idx] += 1
                        if p2 >= posterior_threshold:
                            k_unlinked_i_open[pair_idx] += 1
            elif side == 1:
                # pos2 is in anchor_j; in-anchor end must be informative.
                if n2 < min_gch_sites or (isinstance(p2, float) and np.isnan(p2)):
                    continue
                # Check mate (pos1) is NOT in anchor_i.
                sides1 = hit1_set.get(pair_idx, set())
                if 0 not in sides1:
                    # Mate must be within unlinked_max_distance of anchor_j center.
                    if abs(pos1 - anchor_center_j[pair_idx]) <= unlinked_max_distance:
                        n_unlinked_j[pair_idx] += 1
                        if p2 >= posterior_threshold:
                            k_unlinked_j_open[pair_idx] += 1

    result = pd.DataFrame({
        "cell_name": cell_name,
        "pair_id": panel["pair_id"].values,
        "n_linked": n_linked,
        "k_linked_active": k_linked_active,
        "k_linked_symmetric": k_linked_symmetric,
        "n_unlinked_i": n_unlinked_i,
        "k_unlinked_i_open": k_unlinked_i_open,
        "n_unlinked_j": n_unlinked_j,
        "k_unlinked_j_open": k_unlinked_j_open,
    })

    return result


def _aggregate_one_cell(
    posteriors_parquet: str,
    cell_name: str,
    panel: pd.DataFrame,
    output_dir: str,
    cell_idx: int = 0,
    total_cells: int = 0,
    **kwargs,
) -> tuple[str, str]:
    """Wrapper for parallel execution: aggregate one cell and write parquet.

    Prints per-cell start/done progress to stderr (bypasses Python
    logging, which doesn't work from joblib loky workers).
    """
    import sys
    import time

    if total_cells > 0:
        print(
            f"  [{cell_idx}/{total_cells}] Starting {cell_name} ...",
            file=sys.stderr, flush=True,
        )

    t0 = time.time()
    result = aggregate_pair_counts(
        posteriors_parquet, cell_name, panel, **kwargs,
    )
    out_path = Path(output_dir) / f"{cell_name}.pair_counts.parquet"
    result.to_parquet(out_path, index=False)
    elapsed = time.time() - t0

    if total_cells > 0:
        pct = 100.0 * cell_idx / total_cells
        n_pairs_with_data = int((result["n_linked"] > 0).sum()) if len(result) else 0
        print(
            f"  [{cell_idx}/{total_cells}] ({pct:.0f}%) Done {cell_name}: "
            f"{n_pairs_with_data:,} pairs with linked data ({elapsed:.0f}s)",
            file=sys.stderr, flush=True,
        )
    return cell_name, str(out_path)


def aggregate_all_cells(
    posteriors_dir: str,
    cell_names: list[str],
    panel: pd.DataFrame,
    output_dir: str,
    n_threads: int = 4,
    **kwargs,
) -> dict[str, str]:
    """Aggregate pair counts for all cells in parallel.

    Looks for posteriors files at ``{posteriors_dir}/{cell_name}.posteriors.parquet``.

    Parameters
    ----------
    posteriors_dir : str
        Directory containing per-cell posteriors parquet files.
    cell_names : list[str]
        List of cell identifiers to process.
    panel : pd.DataFrame
        Panel DataFrame with standard ``PANEL_COLUMNS``.
    output_dir : str
        Directory for per-cell output parquet files.
    n_threads : int
        Number of parallel workers (default 4).
    **kwargs
        Extra keyword arguments passed to :func:`aggregate_pair_counts`
        (e.g. ``posterior_threshold``, ``min_gch_sites``,
        ``min_distance``, ``unlinked_max_distance``).

    Returns
    -------
    dict
        Mapping ``cell_name -> parquet_path`` for each processed cell.
    """
    from joblib import Parallel, delayed

    out_path = ensure_dir(output_dir)
    posteriors_path = Path(posteriors_dir)

    # Resolve per-cell parquet paths.
    items: list[tuple[str, str]] = []
    for cell in cell_names:
        pq = posteriors_path / f"{cell}.posteriors.parquet"
        if not pq.exists():
            logger.warning(f"Posteriors file not found for {cell}: {pq}")
            continue
        items.append((cell, str(pq)))

    if not items:
        logger.warning("No posteriors files found — nothing to aggregate")
        return {}

    total = len(items)
    logger.info(
        f"Aggregating pair counts for {total} cells "
        f"({len(panel)} panel pairs) using {n_threads} threads"
    )

    import time
    t_start = time.time()

    # Number the cells (1-based) so workers can print [i/total] progress.
    indexed = [(i + 1, cell_name, pq_path)
               for i, (cell_name, pq_path) in enumerate(items)]

    if n_threads > 1 and len(items) > 1:
        results = Parallel(n_jobs=n_threads)(
            delayed(_aggregate_one_cell)(
                pq_path, cell_name, panel, str(out_path),
                cell_idx=cell_idx, total_cells=total,
                **kwargs,
            )
            for cell_idx, cell_name, pq_path in indexed
        )
    else:
        results = [
            _aggregate_one_cell(
                pq_path, cell_name, panel, str(out_path),
                cell_idx=cell_idx, total_cells=total,
                **kwargs,
            )
            for cell_idx, cell_name, pq_path in indexed
        ]

    cell_to_path = dict(results)
    elapsed = time.time() - t_start
    logger.info(
        f"Pair-count aggregation complete: {len(cell_to_path)} cells "
        f"in {elapsed:.0f}s ({elapsed/max(len(cell_to_path),1):.1f}s/cell)"
    )
    return cell_to_path
