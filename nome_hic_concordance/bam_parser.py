"""BAM parsing: per-cell extraction of read-pair methylation counts.

Uses reference-comparison methylation calling, which works without a
bisulfite strand tag. At each context cytosine:
  - Ref base 'C' (+ strand site): read C = methylated, T = unmethylated,
    other bases (G, A, N) are skipped (read came from opposite conversion
    strand or contains a SNP/error).
  - Ref base 'G' (- strand site): read G = methylated, A = unmethylated,
    other bases (C, T, N) are skipped.

Performance notes
-----------------
The inner loop is performance-critical (~2–8 M reads per cell).  Three
optimisations over the naïve implementation:

1. **Simple-CIGAR fast path** — for reads whose CIGAR contains only M/=/X
   and S/H operations (no indels), the query position at any reference
   position is a constant offset: ``ref_pos - reference_start +
   query_alignment_start``.  This skips the expensive
   ``get_aligned_pairs()`` call and Python dict construction for >90% of
   reads.
2. **Vectorised base comparison** — numpy arrays for context-site
   positions and read bases; boolean masks replace the Python-level loop.
3. **Per-chromosome iteration** — ``bam.fetch(chrom)`` bounds the
   mate-pairing buffer to a single chromosome at a time.
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

# BAM flag bits used for fast filtering.
_FLAG_UNMAPPED       = 0x4
_FLAG_MATE_UNMAPPED  = 0x8
_FLAG_PROPER_PAIR    = 0x2
_FLAG_SECONDARY      = 0x100
_FLAG_DUPLICATE      = 0x400
_FLAG_SUPPLEMENTARY  = 0x800

# CIGAR operation codes (from the SAM spec).
_CIGAR_I = 1  # insertion to reference
_CIGAR_D = 2  # deletion from reference

# ASCII ordinals for base comparison.
_C = ord("C")   # 67
_T = ord("T")   # 84
_G = ord("G")   # 71
_A = ord("A")   # 65

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


# ---------------------------------------------------------------------------
# Per-read context counting (vectorised with simple-CIGAR fast path)
# ---------------------------------------------------------------------------


def _has_indel(read) -> bool:
    """Return True if the CIGAR contains I or D operations."""
    cigar = read.cigartuples
    if cigar is None:
        return False
    for op, _ in cigar:
        if op == _CIGAR_I or op == _CIGAR_D:
            return True
    return False


def _count_context_in_read(
    read,
    ctx_pos: np.ndarray,
    ctx_ref_base: np.ndarray,
    trim_ends: int = 5,
) -> tuple[int, int]:
    """Count methylated / total context sites covered by *read*.

    Parameters
    ----------
    read : pysam.AlignedSegment (or duck-type with the same attributes)
    ctx_pos : int64 array
        Sorted context-site positions for this chromosome.
    ctx_ref_base : uint8 array
        Reference base at each context site (ord('C') or ord('G')).
    trim_ends : int
        Bases to exclude from each end to avoid end-repair artefacts.

    Returns
    -------
    (k_methylated, n_total)
    """
    ref_start = read.reference_start + trim_ends
    ref_end = read.reference_end - trim_ends
    if ref_start >= ref_end:
        return 0, 0

    # Binary search for overlapping context sites.
    i0 = int(np.searchsorted(ctx_pos, ref_start, side="left"))
    i1 = int(np.searchsorted(ctx_pos, ref_end, side="right"))
    if i0 == i1:
        return 0, 0

    site_pos = ctx_pos[i0:i1]        # positions of context sites in range
    site_ref = ctx_ref_base[i0:i1]   # ref base at each site

    seq = read.query_sequence
    if seq is None:
        return 0, 0

    # ----- Resolve query positions for each context site --------------------
    if not _has_indel(read):
        # Fast path: direct arithmetic (no get_aligned_pairs needed).
        query_offset = read.query_alignment_start - read.reference_start
        query_positions = (site_pos.astype(np.intp) + int(query_offset))

        # Bounds check against the actual query sequence length.
        seq_len = len(seq)
        valid = (query_positions >= 0) & (query_positions < seq_len)
        if not valid.any():
            return 0, 0
        site_ref = site_ref[valid]
        query_positions = query_positions[valid]
    else:
        # Slow path: walk CIGAR via get_aligned_pairs (reads with indels).
        pairs_dict = {
            ref_p: query_p
            for query_p, ref_p in read.get_aligned_pairs(matches_only=True)
        }
        raw_qp = np.array(
            [pairs_dict.get(int(p), -1) for p in site_pos], dtype=np.intp,
        )
        valid = raw_qp >= 0
        if not valid.any():
            return 0, 0
        site_ref = site_ref[valid]
        query_positions = raw_qp[valid]

    # ----- Vectorised base comparison ---------------------------------------
    seq_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    read_bases = seq_arr[query_positions]

    plus_mask = site_ref == _C         # + strand context sites
    minus_mask = site_ref == _G        # − strand context sites

    meth = (
        (plus_mask & (read_bases == _C))
        | (minus_mask & (read_bases == _G))
    )
    informative = (
        (plus_mask & ((read_bases == _C) | (read_bases == _T)))
        | (minus_mask & ((read_bases == _G) | (read_bases == _A)))
    )

    return int(meth.sum()), int(informative.sum())


# ---------------------------------------------------------------------------
# Cell-level BAM parsing
# ---------------------------------------------------------------------------


def _passes_filters(read, mapq_min: int) -> bool:
    """Fast flag + MAPQ filter for a single read."""
    flag = read.flag
    if flag & (_FLAG_UNMAPPED | _FLAG_MATE_UNMAPPED):
        return False
    if flag & (_FLAG_SECONDARY | _FLAG_SUPPLEMENTARY):
        return False
    if flag & _FLAG_DUPLICATE:
        return False
    if not (flag & _FLAG_PROPER_PAIR):
        return False
    if read.mapping_quality < mapq_min:
        return False
    return True


def parse_cell_bam(
    bam_path: str,
    context_positions: dict[str, np.ndarray],
    trim_ends: int = 5,
    mapq_min: int = 30,
) -> pd.DataFrame:
    """Parse a single cell's BAM into a DataFrame of read-pair context counts.

    Processes one chromosome at a time via ``bam.fetch(chrom)`` to bound
    the mate-pairing buffer. Uses the simple-CIGAR fast path for reads
    without indels (see :func:`_count_context_in_read`).

    Parameters
    ----------
    bam_path : str
        Coordinate-sorted, indexed BAM.
    context_positions : dict
        ``chrom -> structured array`` with fields ``pos`` (int64) and
        ``ref_base`` (uint8).
    trim_ends : int
        Bases to trim from each read end.
    mapq_min : int
        Both ends of a pair must have MAPQ >= this value (default: 30).

    Returns
    -------
    DataFrame with columns ``chrom, pos1, k1, n1, pos2, k2, n2, distance``.
    """
    import pysam

    # Output collectors — lists of primitives (faster than list of dicts).
    out_chrom: list[str] = []
    out_pos1: list[int] = []
    out_k1: list[int] = []
    out_n1: list[int] = []
    out_pos2: list[int] = []
    out_k2: list[int] = []
    out_n2: list[int] = []
    out_dist: list[int] = []

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        # Process per-chromosome to keep read_buffer bounded.
        for chrom, ctx_arr in context_positions.items():
            if len(ctx_arr) == 0:
                continue

            # Pre-split structured array into contiguous views (avoids
            # repeated structured-array field lookups in the inner loop).
            ctx_pos = ctx_arr["pos"]
            ctx_ref = ctx_arr["ref_base"]

            read_buffer: dict[str, object] = {}

            try:
                read_iter = bam.fetch(chrom)
            except ValueError:
                # Chromosome not in BAM header.
                continue

            for read in read_iter:
                if not _passes_filters(read, mapq_min):
                    continue
                # Intra-chromosomal only.
                if read.next_reference_name != chrom:
                    continue

                qname = read.query_name
                mate = read_buffer.pop(qname, None)
                if mate is None:
                    read_buffer[qname] = read
                    continue

                # Re-check mate MAPQ (defensive — it passed the filter when
                # it was the loop variable, but we verify both ends).
                if mate.mapping_quality < mapq_min:
                    continue

                # Canonical ordering: r1 has smaller reference_start.
                if read.reference_start <= mate.reference_start:
                    r1, r2 = read, mate
                else:
                    r1, r2 = mate, read

                k1, n1 = _count_context_in_read(r1, ctx_pos, ctx_ref, trim_ends)
                k2, n2 = _count_context_in_read(r2, ctx_pos, ctx_ref, trim_ends)

                pos1 = (r1.reference_start + r1.reference_end) // 2
                pos2 = (r2.reference_start + r2.reference_end) // 2

                out_chrom.append(chrom)
                out_pos1.append(pos1)
                out_k1.append(k1)
                out_n1.append(n1)
                out_pos2.append(pos2)
                out_k2.append(k2)
                out_n2.append(n2)
                out_dist.append(abs(pos2 - pos1))

            del read_buffer  # free mate buffer between chromosomes

    n_pairs = len(out_chrom)
    if n_pairs > 0:
        df = pd.DataFrame({
            "chrom": out_chrom,
            "pos1": np.array(out_pos1, dtype=np.int64),
            "k1": np.array(out_k1, dtype=np.int32),
            "n1": np.array(out_n1, dtype=np.int32),
            "pos2": np.array(out_pos2, dtype=np.int64),
            "k2": np.array(out_k2, dtype=np.int32),
            "n2": np.array(out_n2, dtype=np.int32),
            "distance": np.array(out_dist, dtype=np.int64),
        })
    else:
        df = pd.DataFrame(columns=READ_PAIR_COLUMNS)

    logger.info(f"  {bam_path}: {n_pairs:,} intra-chromosomal pairs extracted")
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

    total = len(samples)

    def _process_one(cell_idx: int, row) -> tuple[str, str, int]:
        """Parse one cell and print progress to stderr.

        ``logger.info()`` does not work from joblib/loky workers because
        the logging handlers are not inherited.  ``print(..., file=stderr,
        flush=True)`` writes directly to the file descriptor and is
        always visible.
        """
        import sys

        cell = row["cell_name"]
        bam = row["bam_path"]
        df = parse_cell_bam(bam, context_positions, trim_ends=trim_ends, mapq_min=mapq_min)
        parquet_path = str(out_path / f"{cell}.read_pairs.parquet")
        df.to_parquet(parquet_path, index=False)
        n_pairs = len(df)

        print(
            f"  [{cell_idx}/{total}] ({100 * cell_idx / total:.0f}%) "
            f"{cell}: {n_pairs:,} pairs",
            file=sys.stderr, flush=True,
        )
        return cell, parquet_path, n_pairs

    rows = [(i + 1, row) for i, (_, row) in enumerate(samples.iterrows())]

    if n_threads > 1 and len(samples) > 1:
        raw = Parallel(n_jobs=n_threads)(
            delayed(_process_one)(idx, row) for idx, row in rows
        )
    else:
        raw = [_process_one(idx, row) for idx, row in rows]

    total_pairs = sum(r[2] for r in raw)
    logger.info(f"  All {total} cells done — {total_pairs:,} total pairs")
    return {cell: path for cell, path, _ in raw}
