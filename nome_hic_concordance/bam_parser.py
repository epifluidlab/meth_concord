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
The inner loop is performance-critical (~2–8 M reads per cell).

1. **Pure Python inner loop** — for the typical 5–15 context sites per
   read, a tight Python loop with direct string indexing beats numpy's
   per-operation overhead (array creation, dtype casting) by ~5×.
2. **Fast indel check** — ``reference_length != query_alignment_length``
   (two C int properties) replaces iterating ``cigartuples``.
3. **Inlined filter logic** — avoids function-call overhead for the
   8M-call-per-BAM filter check.
4. **Per-chromosome iteration** — ``bam.fetch(chrom)`` bounds the
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

# Try to import the C++/htslib extension.
try:
    from . import _bam_parser_core  # type: ignore[attr-defined]
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False

# The C++ backend now uses a simple two-step MD parser:
# Step 1: CIGAR → ref_to_query[] mapping
# Step 2: Walk MD mismatches, look up query position, compare bases

# BAM flag bits — combined into a single skip-mask for fast filtering.
_FLAG_UNMAPPED       = 0x4
_FLAG_MATE_UNMAPPED  = 0x8
_FLAG_PROPER_PAIR    = 0x2
_FLAG_SECONDARY      = 0x100
_FLAG_DUPLICATE      = 0x400
_FLAG_SUPPLEMENTARY  = 0x800
_SKIP_FLAGS_BASE = _FLAG_UNMAPPED | _FLAG_MATE_UNMAPPED | _FLAG_SECONDARY | _FLAG_DUPLICATE
_SKIP_FLAGS = _SKIP_FLAGS_BASE | _FLAG_SUPPLEMENTARY  # default: skip supplementary too

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
# Bisulfite conversion strand inference
# ---------------------------------------------------------------------------

# In non-directional bisulfite (bhmem), each read comes from either a
# C→T or G→A conversion strand, but no tag records which.  A read from
# the C→T strand is informative ONLY at ref=C sites (C=meth, T=unmeth);
# at ref=G sites it always shows G (the unconverted reference base),
# which would be falsely called "methylated" if counted.  The reverse
# holds for G→A reads at ref=C sites.
#
# We infer the conversion direction per read by counting C→T vs G→A
# mismatches between the read and the reference (reconstructed from the
# MD tag via ``get_reference_sequence``).  The dominant mismatch type
# tells us the conversion strand, and we only count context sites from
# the matching strand.


def _infer_conversion_strand(read) -> str:
    """Infer whether *read* is from a C→T or G→A bisulfite conversion strand.

    Counts C→T mismatches (ref=C, read=T) and G→A mismatches (ref=G,
    read=A) across the aligned span.  The dominant type determines the
    conversion direction.

    For reads without indels (>90%), a direct 1:1 loop is used.  For
    reads with indels, ``get_aligned_pairs(matches_only=True)`` maps
    reference to query positions correctly.

    Requires the MD tag (present in calmd BAMs).

    Returns
    -------
    ``'CT'``, ``'GA'``, or ``'unknown'`` (if counts are tied or both zero).
    """
    seq = read.query_sequence
    if seq is None:
        return "unknown"

    try:
        ref_seq = read.get_reference_sequence().upper()
    except ValueError:
        return "unknown"

    if ref_seq is None:
        return "unknown"

    ct = 0
    ga = 0

    if read.reference_length == read.query_alignment_length:
        # Fast path: no indels — direct 1:1 correspondence between
        # ref_seq[i] and seq[query_alignment_start + i].
        offset = read.query_alignment_start
        for i in range(len(ref_seq)):
            rb = ref_seq[i]
            qb = seq[offset + i]
            if rb == "C" and qb == "T":
                ct += 1
            elif rb == "G" and qb == "A":
                ga += 1
    else:
        # Slow path: indels break the 1:1 correspondence.  Use
        # aligned pairs to correctly map ref → query positions.
        ref_start = read.reference_start
        ref_len = len(ref_seq)
        seq_len = len(seq)
        for qpos, rpos in read.get_aligned_pairs(matches_only=True):
            ri = rpos - ref_start
            if 0 <= ri < ref_len and 0 <= qpos < seq_len:
                rb = ref_seq[ri]
                qb = seq[qpos]
                if rb == "C" and qb == "T":
                    ct += 1
                elif rb == "G" and qb == "A":
                    ga += 1

    if ct > ga:
        return "CT"
    elif ga > ct:
        return "GA"
    return "unknown"


# ---------------------------------------------------------------------------
# Per-read context counting
# ---------------------------------------------------------------------------


def _count_context_in_read(
    read,
    ctx_pos: np.ndarray,
    ctx_ref_base: np.ndarray,
    trim_ends: int = 5,
) -> tuple[int, int]:
    """Count methylated / total context sites covered by *read*.

    First infers the bisulfite conversion direction (C→T or G→A) from
    the read's mismatch pattern, then counts only sites from the
    matching strand:

    * C→T reads → only ref=C sites are informative (C=meth, T=unmeth)
    * G→A reads → only ref=G sites are informative (G=meth, A=unmeth)

    Sites from the non-matching strand are skipped (they always show
    the reference base and would be falsely called "methylated").

    Parameters
    ----------
    read : pysam.AlignedSegment (or duck-type)
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
    i1 = int(np.searchsorted(ctx_pos, ref_end, side="left"))
    if i0 >= i1:
        return 0, 0

    seq = read.query_sequence
    if seq is None:
        return 0, 0

    # Determine which context sites are informative for this read.
    strand = _infer_conversion_strand(read)
    if strand == "CT":
        informative_ref = 67   # ord('C') — only + strand sites
    elif strand == "GA":
        informative_ref = 71   # ord('G') — only − strand sites
    else:
        # Cannot determine strand (no mismatches, or tied).
        # Skip this read — we can't tell which sites are real.
        return 0, 0

    # Convert the small slice to Python lists for fast element access.
    site_positions = ctx_pos[i0:i1].tolist()
    site_ref_bases = ctx_ref_base[i0:i1].tolist()

    k = 0
    n = 0

    if read.reference_length == read.query_alignment_length:
        # ---- Fast path: no indels ----
        offset = read.query_alignment_start - read.reference_start
        seq_len = len(seq)
        for j in range(len(site_positions)):
            rb = site_ref_bases[j]
            if rb != informative_ref:
                continue  # wrong strand for this read
            qp = site_positions[j] + offset
            if qp < 0 or qp >= seq_len:
                continue
            base = seq[qp]
            if rb == 67:    # ref=C, C→T strand
                if base == "C":
                    k += 1; n += 1
                elif base == "T":
                    n += 1
            elif rb == 71:  # ref=G, G→A strand
                if base == "G":
                    k += 1; n += 1
                elif base == "A":
                    n += 1
    else:
        # ---- Slow path: has indels ----
        pairs_dict = {
            ref_p: query_p
            for query_p, ref_p in read.get_aligned_pairs(matches_only=True)
        }
        for j in range(len(site_positions)):
            rb = site_ref_bases[j]
            if rb != informative_ref:
                continue
            qp = pairs_dict.get(site_positions[j])
            if qp is None:
                continue
            base = seq[qp]
            if rb == 67:
                if base == "C":
                    k += 1; n += 1
                elif base == "T":
                    n += 1
            elif rb == 71:
                if base == "G":
                    k += 1; n += 1
                elif base == "A":
                    n += 1

    return k, n


# ---------------------------------------------------------------------------
# Cell-level BAM parsing
# ---------------------------------------------------------------------------


def _parse_cell_bam_cpp(
    bam_path: str,
    context_positions: dict[str, np.ndarray],
    trim_ends: int = 5,
    mapq_min: int = 30,
    require_proper_pair: bool = False,
    keep_supplementary: bool = False,
) -> pd.DataFrame:
    """Parse a BAM using the C++/htslib backend (fast path)."""
    # The C++ function expects {chrom: (pos_array, ref_base_array)}.
    # CRITICAL: arr["pos"] and arr["ref_base"] are views into a structured
    # numpy array with stride = record_size (9 bytes), NOT element_size.
    # The C++ code uses raw pointer arithmetic (ctx_pos[j]) which assumes
    # contiguous layout.  np.ascontiguousarray() copies them into packed
    # arrays with stride = element_size.
    cpp_ctx = {
        chrom: (
            np.ascontiguousarray(arr["pos"]),
            np.ascontiguousarray(arr["ref_base"]),
        )
        for chrom, arr in context_positions.items()
        if len(arr) > 0
    }
    # Pass require_proper_pair / keep_supplementary as keywords; older
    # compiled C++ extensions may not accept them.  On TypeError, fall
    # back to the Python backend (which honors all flags).
    try:
        raw = _bam_parser_core.parse_cell_bam(
            bam_path, cpp_ctx, trim_ends, mapq_min,
            require_proper_pair=require_proper_pair,
            keep_supplementary=keep_supplementary,
        )
    except TypeError:
        if not require_proper_pair or keep_supplementary:
            logger.warning(
                "Installed C++ extension is out of date "
                "(missing require_proper_pair / keep_supplementary kwargs); "
                "falling back to Python backend for this cell. "
                "Rebuild via 'pip install -e .' to update."
            )
            return _parse_cell_bam_python(
                bam_path, context_positions, trim_ends, mapq_min,
                require_proper_pair=require_proper_pair,
                keep_supplementary=keep_supplementary,
            )
        raw = _bam_parser_core.parse_cell_bam(bam_path, cpp_ctx, trim_ends, mapq_min)

    n_pairs = len(raw["pos1"])
    if n_pairs > 0:
        df = pd.DataFrame({
            "chrom": raw["chrom"],
            "pos1": np.asarray(raw["pos1"], dtype=np.int64),
            "k1": np.asarray(raw["k1"], dtype=np.int32),
            "n1": np.asarray(raw["n1"], dtype=np.int32),
            "pos2": np.asarray(raw["pos2"], dtype=np.int64),
            "k2": np.asarray(raw["k2"], dtype=np.int32),
            "n2": np.asarray(raw["n2"], dtype=np.int32),
            "distance": np.asarray(raw["distance"], dtype=np.int64),
        })
    else:
        df = pd.DataFrame(columns=READ_PAIR_COLUMNS)
    return df


def _parse_cell_bam_python(
    bam_path: str,
    context_positions: dict[str, np.ndarray],
    trim_ends: int = 5,
    mapq_min: int = 30,
    require_proper_pair: bool = False,
    keep_supplementary: bool = False,
) -> pd.DataFrame:
    """Parse a single cell's BAM using pure Python/pysam (fallback).

    Processes one chromosome at a time via ``bam.fetch(chrom)`` to bound
    the mate-pairing buffer.

    For Hi-C data, ``require_proper_pair`` should be ``False`` (default)
    because BWA-MEM flags long-range Hi-C ligation products as
    *improperly paired* (insert size violates short-insert expectation).

    ``keep_supplementary`` (default False) controls whether chimeric-read
    supplementary alignments are included.  Standard Hi-C practice is to
    exclude them (they're either redundant with primary or should be
    handled upstream by pairtools).  Enable only if you suspect bhmem
    assigns "primary" to the wrong half of a chimera.
    """
    import pysam
    pysam.set_verbosity(0)

    # Output collectors — lists of primitives.
    out_chrom: list[str] = []
    out_pos1: list[int] = []
    out_k1: list[int] = []
    out_n1: list[int] = []
    out_pos2: list[int] = []
    out_k2: list[int] = []
    out_n2: list[int] = []
    out_dist: list[int] = []

    # Local aliases for hot-path speed.
    skip_flags = _SKIP_FLAGS_BASE if keep_supplementary else _SKIP_FLAGS
    proper_flag = _FLAG_PROPER_PAIR
    searchsorted = np.searchsorted

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for chrom, ctx_arr in context_positions.items():
            if len(ctx_arr) == 0:
                continue

            ctx_pos = ctx_arr["pos"]
            ctx_ref = ctx_arr["ref_base"]

            read_buffer: dict[str, object] = {}

            try:
                read_iter = bam.fetch(chrom)
            except ValueError:
                continue

            for read in read_iter:
                # --- Inlined filter (avoids function-call overhead) ---
                flag = read.flag
                if flag & skip_flags:
                    continue
                if require_proper_pair and not (flag & proper_flag):
                    continue
                if read.mapping_quality < mapq_min:
                    continue
                mate_chrom = read.next_reference_name
                if mate_chrom != chrom:
                    continue

                # --- Mate pairing ---
                qname = read.query_name
                mate = read_buffer.pop(qname, None)
                if mate is None:
                    read_buffer[qname] = read
                    continue

                if mate.mapping_quality < mapq_min:
                    continue

                # Canonical ordering.
                r1_start = read.reference_start
                m_start = mate.reference_start
                if r1_start <= m_start:
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

            del read_buffer

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

    return df


def parse_cell_bam(
    bam_path: str,
    context_positions: dict[str, np.ndarray],
    trim_ends: int = 5,
    mapq_min: int = 30,
    require_proper_pair: bool = False,
    keep_supplementary: bool = False,
) -> pd.DataFrame:
    """Parse a single cell's BAM into a DataFrame of read-pair context counts.

    Automatically uses the C++/htslib backend if the extension was
    compiled; otherwise falls back to the pure-Python/pysam implementation.

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
    require_proper_pair : bool
        If True, require BAM proper-pair flag (default: False).
        For Hi-C data, this should be False — BWA-MEM does NOT flag
        long-range ligation products as proper pairs, so requiring it
        would silently discard most cis Hi-C contacts.
    keep_supplementary : bool
        If True, keep supplementary alignments (chimeric-read splits).
        Default False — standard Hi-C practice to avoid double-counting.

    Returns
    -------
    DataFrame with columns ``chrom, pos1, k1, n1, pos2, k2, n2, distance``.
    """
    if _CPP_AVAILABLE:
        return _parse_cell_bam_cpp(
            bam_path, context_positions, trim_ends, mapq_min,
            require_proper_pair=require_proper_pair,
            keep_supplementary=keep_supplementary,
        )
    return _parse_cell_bam_python(
        bam_path, context_positions, trim_ends, mapq_min,
        require_proper_pair=require_proper_pair,
        keep_supplementary=keep_supplementary,
    )


def parse_all_cells(
    samples: pd.DataFrame,
    context_positions: dict[str, np.ndarray],
    output_dir: str,
    n_threads: int = 4,
    trim_ends: int = 5,
    mapq_min: int = 30,
    require_proper_pair: bool = False,
    keep_supplementary: bool = False,
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
    backend = "C++/htslib" if _CPP_AVAILABLE else "Python/pysam"
    logger.info(f"  BAM parser backend: {backend}")

    def _process_one(cell_idx: int, row) -> tuple[str, str, int]:
        import sys
        import time

        if not _CPP_AVAILABLE:
            import pysam
            pysam.set_verbosity(0)

        cell = row["cell_name"]
        bam = row["bam_path"]

        print(
            f"  [{cell_idx}/{total}] Starting {cell} ...",
            file=sys.stderr, flush=True,
        )

        t0 = time.time()
        df = parse_cell_bam(
            bam, context_positions, trim_ends=trim_ends, mapq_min=mapq_min,
            require_proper_pair=require_proper_pair,
            keep_supplementary=keep_supplementary,
        )
        parquet_path = str(out_path / f"{cell}.read_pairs.parquet")
        df.to_parquet(parquet_path, index=False)
        n_pairs = len(df)
        elapsed = time.time() - t0

        print(
            f"  [{cell_idx}/{total}] Done {cell}: "
            f"{n_pairs:,} pairs ({elapsed:.0f}s)",
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
