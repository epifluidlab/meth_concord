"""BAM parsing: per-cell extraction of read-pair methylation counts.

Uses reference-comparison methylation calling, which works without a
bisulfite strand tag. At each context cytosine:
  - Ref base 'C' (+ strand site): read C = methylated, T = unmethylated,
    other bases (G, A, N) are skipped (read came from opposite conversion
    strand or contains a SNP/error).
  - Ref base 'G' (- strand site): read G = methylated, A = unmethylated,
    other bases (C, T, N) are skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)


READ_PAIR_COLUMNS = [
    "chrom",
    "pos1", "k1", "n1",
    "pos2", "k2", "n2",
    "distance",
]


@dataclass
class ReadPairContext:
    """Per-read-pair methylation counts at one cytosine context."""
    chrom: str
    pos1: int     # midpoint of read 1
    k1: int       # methylated context count, end 1
    n1: int       # total informative context sites, end 1
    pos2: int     # midpoint of read 2
    k2: int
    n2: int
    distance: int  # genomic distance (bp) between midpoints


def _count_context_in_read(read, context_sites_chrom, trim_ends: int = 5) -> tuple[int, int]:
    """Count methylated / total context sites covered by a single read.

    Parameters
    ----------
    read : pysam.AlignedSegment
    context_sites_chrom : numpy structured array
        Fields ``pos`` (int64) and ``ref_base`` (uint8, ord('C') or ord('G')).
    trim_ends : int
        Bases to exclude from each end of the read to avoid end-repair artifacts.

    Returns
    -------
    (k_methylated, n_total)
    """
    ref_start = read.reference_start + trim_ends
    ref_end = read.reference_end - trim_ends
    if ref_start >= ref_end:
        return (0, 0)

    positions = context_sites_chrom["pos"]
    idx_start = np.searchsorted(positions, ref_start, side="left")
    idx_end = np.searchsorted(positions, ref_end, side="right")
    sites_in_read = context_sites_chrom[idx_start:idx_end]
    if len(sites_in_read) == 0:
        return (0, 0)

    # Map reference positions within this read to the corresponding query
    # (sequence) positions. ``pysam.AlignedSegment.get_aligned_pairs`` yields
    # ``(query_position, reference_position)`` tuples; we invert that here.
    aligned_pairs = {
        ref_pos: query_pos
        for query_pos, ref_pos in read.get_aligned_pairs(matches_only=True)
    }

    query_seq = read.query_sequence
    if query_seq is None:
        return (0, 0)

    k = 0
    n = 0
    c_ord = ord("C")
    g_ord = ord("G")

    for site in sites_in_read:
        ref_pos = int(site["pos"])
        query_pos = aligned_pairs.get(ref_pos)
        if query_pos is None:
            continue

        read_base = query_seq[query_pos]

        if site["ref_base"] == c_ord:
            # + strand context site
            if read_base == "C":
                k += 1
                n += 1
            elif read_base == "T":
                n += 1
            # G, A, N -> uninformative, skip
        elif site["ref_base"] == g_ord:
            # - strand context site
            if read_base == "G":
                k += 1
                n += 1
            elif read_base == "A":
                n += 1
            # C, T, N -> uninformative, skip

    return (k, n)


def parse_cell_bam(
    bam_path: str,
    context_positions: dict[str, np.ndarray],
    trim_ends: int = 5,
    mapq_min: int = 30,
) -> pd.DataFrame:
    """Parse a single cell's BAM into a DataFrame of read-pair context counts.

    Parameters
    ----------
    bam_path : str
        Coordinate-sorted, indexed BAM.
    context_positions : dict
        Mapping ``chrom -> structured array`` (as produced by
        :func:`reference.extract_context_positions`).
    trim_ends : int
        Bases to trim from each read end.
    mapq_min : int
        Minimum mapping quality. **Both** ends of a pair must meet this
        threshold (default: 30).

    Returns
    -------
    DataFrame with columns ``chrom, pos1, k1, n1, pos2, k2, n2, distance``.

    Filtering applied to every read before buffering:
      - unmapped or mate-unmapped → skip
      - secondary or supplementary → skip
      - PCR / optical duplicate (``is_duplicate``) → skip
      - not properly paired → skip
      - inter-chromosomal → skip
      - MAPQ < ``mapq_min`` → skip

    After pairing, an additional check ensures that **both** mates satisfy
    MAPQ ≥ ``mapq_min`` (the per-read filter already covers the first-seen
    mate, but the second mate's MAPQ is verified here for safety).
    """
    import pysam

    records: list[dict] = []

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        # Buffer reads by query name so we can pair read1 with read2 in
        # coordinate-sorted traversal without loading the whole file.
        read_buffer: dict[str, "pysam.AlignedSegment"] = {}

        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.mate_is_unmapped:
                continue
            if read.is_secondary or read.is_supplementary:
                continue
            if read.is_duplicate:
                continue
            if read.mapping_quality < mapq_min:
                continue
            if not read.is_proper_pair:
                continue
            if read.reference_name != read.next_reference_name:
                continue  # intra-chromosomal only

            qname = read.query_name
            mate = read_buffer.pop(qname, None)
            if mate is None:
                # Only buffer the first-seen mate. Keeping a single reference
                # bounds memory in pathological cases where a read's mate is
                # far downstream.
                read_buffer[qname] = read
                continue

            # Both ends must be uniquely mapped with MAPQ >= mapq_min.
            # The per-read filter above already guarantees the current read
            # passes, but the buffered mate was only checked when *it* was
            # the loop variable.  Re-check here in case the mate's flags or
            # MAPQ changed between buffering and now (defensive).
            if mate.mapping_quality < mapq_min:
                continue

            # Order so r1 has the smaller reference_start (deterministic).
            if read.reference_start <= mate.reference_start:
                r1, r2 = read, mate
            else:
                r1, r2 = mate, read

            chrom = r1.reference_name
            ctx_chrom = context_positions.get(chrom)
            if ctx_chrom is None or len(ctx_chrom) == 0:
                continue

            k1, n1 = _count_context_in_read(r1, ctx_chrom, trim_ends)
            k2, n2 = _count_context_in_read(r2, ctx_chrom, trim_ends)

            pos1 = (r1.reference_start + r1.reference_end) // 2
            pos2 = (r2.reference_start + r2.reference_end) // 2
            distance = abs(pos2 - pos1)

            records.append({
                "chrom": chrom,
                "pos1": pos1, "k1": k1, "n1": n1,
                "pos2": pos2, "k2": k2, "n2": n2,
                "distance": distance,
            })

    if records:
        df = pd.DataFrame.from_records(records, columns=READ_PAIR_COLUMNS)
    else:
        df = pd.DataFrame(columns=READ_PAIR_COLUMNS)

    # Downcast for on-disk compactness.
    for col in ("pos1", "pos2", "distance"):
        if col in df.columns and len(df) > 0:
            df[col] = df[col].astype(np.int64)
    for col in ("k1", "n1", "k2", "n2"):
        if col in df.columns and len(df) > 0:
            df[col] = df[col].astype(np.int32)

    logger.info(f"  {bam_path}: {len(df):,} intra-chromosomal pairs extracted")
    return df


def parse_all_cells(
    samples: pd.DataFrame,
    context_positions: dict[str, np.ndarray],
    output_dir: str,
    n_threads: int = 4,
    trim_ends: int = 5,
    mapq_min: int = 30,
    **_unused_kwargs,
) -> dict[str, str]:
    """Parse all cell BAMs in parallel and save per-cell parquet outputs.

    Parameters
    ----------
    samples : DataFrame
        Columns ``cell_name``, ``bam_path`` (and optionally ``group``).
    context_positions : dict
        Already flattened to a single context: ``chrom -> structured array``.
    output_dir : str
        Directory for ``{cell_name}.read_pairs.parquet`` outputs.
    n_threads : int
    trim_ends, mapq_min : int
        ``mapq_min`` applies to **both** ends of a pair (default: 30).

    Returns
    -------
    dict : cell_name -> parquet path
    """
    from joblib import Parallel, delayed

    out_path = ensure_dir(output_dir)

    def _process_one(row) -> tuple[str, str]:
        cell = row["cell_name"]
        bam = row["bam_path"]
        logger.info(f"Processing {cell}...")
        df = parse_cell_bam(bam, context_positions, trim_ends=trim_ends, mapq_min=mapq_min)
        parquet_path = str(out_path / f"{cell}.read_pairs.parquet")
        df.to_parquet(parquet_path, index=False)
        return cell, parquet_path

    rows: Iterable = [row for _, row in samples.iterrows()]

    if n_threads > 1 and len(samples) > 1:
        results = Parallel(n_jobs=n_threads)(
            delayed(_process_one)(row) for row in rows
        )
    else:
        results = [_process_one(row) for row in rows]

    return dict(results)
