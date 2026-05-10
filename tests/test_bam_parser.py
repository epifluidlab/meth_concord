"""Tests for bam_parser._count_context_in_read using a fake AlignedSegment.

This test does not require pysam, since we mimic the small subset of the
AlignedSegment interface that ``_count_context_in_read`` actually touches.
"""

from __future__ import annotations

import numpy as np

from nome_hic_concordance.bam_parser import _count_context_in_read
from nome_hic_concordance.reference import CONTEXT_DTYPE


class FakeRead:
    """Minimal duck-type matching pysam.AlignedSegment for our parser.

    Parameters
    ----------
    reference_start : int
        0-based leftmost reference position of the alignment.
    query_sequence : str
        Full query sequence (including soft-clipped bases).
    aligned_pairs : list[(int, int)]
        (query_pos, ref_pos) tuples — only matched/mismatched positions.
    cigartuples : list[(int, int)] or None
        CIGAR operations as (op_code, length) tuples.
        If None, defaults to a single M block covering the whole alignment.
    query_alignment_start : int
        Index of the first aligned (non-clipped) base in query_sequence.
    """

    def __init__(
        self,
        reference_start: int,
        query_sequence: str,
        aligned_pairs: list[tuple[int, int]],
        cigartuples: list[tuple[int, int]] | None = None,
        query_alignment_start: int = 0,
    ):
        self.reference_start = reference_start
        self.query_sequence = query_sequence
        self._aligned_pairs = aligned_pairs
        self.query_alignment_start = query_alignment_start

        if cigartuples is None:
            # Default: simple M-only CIGAR.
            ref_positions = [p for _, p in aligned_pairs if p is not None]
            span = max(ref_positions) - min(ref_positions) + 1 if ref_positions else 0
            self.cigartuples = [(0, span)]  # 0 = BAM_CMATCH
        else:
            self.cigartuples = cigartuples

    @property
    def flag(self):
        return 0x2  # proper pair

    @property
    def reference_end(self):
        ref_positions = [p for _, p in self._aligned_pairs if p is not None]
        return max(ref_positions) + 1

    def get_aligned_pairs(self, matches_only: bool = False):
        return list(self._aligned_pairs)


def _ctx_arrays(positions_and_bases: list[tuple[int, str]]):
    """Return (pos_array, ref_base_array) pair for _count_context_in_read."""
    arr = np.empty(len(positions_and_bases), dtype=CONTEXT_DTYPE)
    arr["pos"] = [p for p, _ in positions_and_bases]
    arr["ref_base"] = [ord(b) for _, b in positions_and_bases]
    return arr["pos"], arr["ref_base"]


def test_plus_strand_methylated_and_unmethylated():
    ctx_pos, ctx_ref = _ctx_arrays([(101, "C")])

    # read C at ref C => methylated
    read = FakeRead(
        reference_start=100,
        query_sequence="ACT",
        aligned_pairs=[(0, 100), (1, 101), (2, 102)],
    )
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (1, 1)

    # read T at ref C => unmethylated
    read2 = FakeRead(
        reference_start=100,
        query_sequence="ATT",
        aligned_pairs=[(0, 100), (1, 101), (2, 102)],
    )
    k, n = _count_context_in_read(read2, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (0, 1)

    # read G at C site => uninformative
    read3 = FakeRead(
        reference_start=100,
        query_sequence="AGT",
        aligned_pairs=[(0, 100), (1, 101), (2, 102)],
    )
    k, n = _count_context_in_read(read3, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (0, 0)


def test_minus_strand_methylated_and_unmethylated():
    ctx_pos, ctx_ref = _ctx_arrays([(101, "G")])

    read = FakeRead(
        reference_start=100,
        query_sequence="AGT",
        aligned_pairs=[(0, 100), (1, 101), (2, 102)],
    )
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (1, 1)

    read2 = FakeRead(
        reference_start=100,
        query_sequence="AAT",
        aligned_pairs=[(0, 100), (1, 101), (2, 102)],
    )
    k, n = _count_context_in_read(read2, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (0, 1)

    read3 = FakeRead(
        reference_start=100,
        query_sequence="ACT",
        aligned_pairs=[(0, 100), (1, 101), (2, 102)],
    )
    k, n = _count_context_in_read(read3, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (0, 0)


def test_trim_ends_excludes_sites():
    ctx_pos, ctx_ref = _ctx_arrays([(101, "C"), (105, "C")])
    seq = "C" * 11
    pairs = [(i, 100 + i) for i in range(11)]
    read = FakeRead(reference_start=100, query_sequence=seq, aligned_pairs=pairs)

    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=3)
    assert (k, n) == (1, 1)


def test_no_sites_in_range_returns_zero():
    ctx_pos, ctx_ref = _ctx_arrays([(500, "C")])
    read = FakeRead(
        reference_start=100,
        query_sequence="C" * 10,
        aligned_pairs=[(i, 100 + i) for i in range(10)],
    )
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (0, 0)


def test_multiple_sites_mixed():
    ctx_pos, ctx_ref = _ctx_arrays([(100, "C"), (102, "G"), (105, "C")])
    read = FakeRead(
        reference_start=100,
        query_sequence="CTAATT",
        aligned_pairs=[(0, 100), (1, 101), (2, 102), (3, 103), (4, 104), (5, 105)],
    )
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (1, 3)


def test_indel_falls_back_to_aligned_pairs():
    """Reads with I or D in CIGAR must use the slow path."""
    ctx_pos, ctx_ref = _ctx_arrays([(101, "C"), (103, "C")])
    # CIGAR: 2M 1I 2M — insertion shifts query positions after the insertion.
    # aligned_pairs (matches_only): (0,100), (1,101), (3,103), (4,104)
    # The inserted base at query_pos=2 has no reference position.
    read = FakeRead(
        reference_start=100,
        query_sequence="ACCCT",  # 5 query bases
        aligned_pairs=[(0, 100), (1, 101), (3, 103), (4, 104)],
        cigartuples=[(0, 2), (1, 1), (0, 2)],  # 2M 1I 2M
        query_alignment_start=0,
    )
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    # pos 101: read base = 'C' (query_pos 1) => methylated
    # pos 103: read base = 'C' (query_pos 3) => methylated
    assert (k, n) == (2, 2)


def test_soft_clip_fast_path():
    """Soft-clipped reads should use the fast path with correct offset."""
    ctx_pos, ctx_ref = _ctx_arrays([(101, "C")])
    # CIGAR: 2S 3M — soft clip of 2 bases, then 3 matched.
    # reference_start = 100, query_alignment_start = 2
    # ref 100 => query 2, ref 101 => query 3, ref 102 => query 4
    read = FakeRead(
        reference_start=100,
        query_sequence="NNCAT",  # 5 bases; first 2 are soft-clipped
        aligned_pairs=[(2, 100), (3, 101), (4, 102)],
        cigartuples=[(4, 2), (0, 3)],  # 4 = BAM_CSOFT_CLIP, 0 = BAM_CMATCH
        query_alignment_start=2,
    )
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    # pos 101: query_pos = 101 - 100 + 2 = 3 => seq[3] = 'A' => uninformative for C site
    assert (k, n) == (0, 0)

    # Same but with 'C' at the right position
    read2 = FakeRead(
        reference_start=100,
        query_sequence="NNCCT",
        aligned_pairs=[(2, 100), (3, 101), (4, 102)],
        cigartuples=[(4, 2), (0, 3)],
        query_alignment_start=2,
    )
    k, n = _count_context_in_read(read2, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (1, 1)
