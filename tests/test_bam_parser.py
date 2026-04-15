"""Tests for bam_parser._count_context_in_read using a fake AlignedSegment.

This test does not require pysam, since we mimic the small subset of the
AlignedSegment interface that ``_count_context_in_read`` actually touches.
"""

from __future__ import annotations

import numpy as np

from nome_hic_concordance.bam_parser import _count_context_in_read
from nome_hic_concordance.reference import CONTEXT_DTYPE


class FakeRead:
    """Minimal duck-type matching pysam.AlignedSegment for our parser."""

    def __init__(
        self,
        reference_start: int,
        query_sequence: str,
        # pairs: list of (query_pos, ref_pos)
        aligned_pairs: list[tuple[int, int]],
    ):
        self.reference_start = reference_start
        self.query_sequence = query_sequence
        self._aligned_pairs = aligned_pairs

    @property
    def reference_end(self):
        # reference_end is exclusive in pysam: last ref pos covered + 1
        ref_positions = [p for _, p in self._aligned_pairs if p is not None]
        return max(ref_positions) + 1

    def get_aligned_pairs(self, matches_only: bool = False):
        return list(self._aligned_pairs)


def _ctx_array(positions_and_bases: list[tuple[int, str]]) -> np.ndarray:
    arr = np.empty(len(positions_and_bases), dtype=CONTEXT_DTYPE)
    arr["pos"] = [p for p, _ in positions_and_bases]
    arr["ref_base"] = [ord(b) for _, b in positions_and_bases]
    return arr


def test_plus_strand_methylated_and_unmethylated():
    # Reference:    pos:  100 101 102
    # ref bases:           ref A   C   T  -> only pos 101 is a context site
    # read bases:           A   C   T -> read C at ref C => methylated
    read = FakeRead(
        reference_start=100,
        query_sequence="ACT",
        aligned_pairs=[(0, 100), (1, 101), (2, 102)],
    )
    sites = _ctx_array([(101, "C")])
    k, n = _count_context_in_read(read, sites, trim_ends=0)
    assert (k, n) == (1, 1)

    # Same setup but read shows T at pos 101 -> unmethylated
    read2 = FakeRead(
        reference_start=100,
        query_sequence="ATT",
        aligned_pairs=[(0, 100), (1, 101), (2, 102)],
    )
    k, n = _count_context_in_read(read2, sites, trim_ends=0)
    assert (k, n) == (0, 1)

    # Read shows G at the C site -> uninformative (skipped)
    read3 = FakeRead(
        reference_start=100,
        query_sequence="AGT",
        aligned_pairs=[(0, 100), (1, 101), (2, 102)],
    )
    k, n = _count_context_in_read(read3, sites, trim_ends=0)
    assert (k, n) == (0, 0)


def test_minus_strand_methylated_and_unmethylated():
    # Site at pos 101 with ref_base='G' (=> C on - strand)
    sites = _ctx_array([(101, "G")])
    # Read shows G at pos 101 -> methylated
    read = FakeRead(
        reference_start=100,
        query_sequence="AGT",
        aligned_pairs=[(0, 100), (1, 101), (2, 102)],
    )
    k, n = _count_context_in_read(read, sites, trim_ends=0)
    assert (k, n) == (1, 1)

    # Read shows A at pos 101 -> unmethylated
    read2 = FakeRead(
        reference_start=100,
        query_sequence="AAT",
        aligned_pairs=[(0, 100), (1, 101), (2, 102)],
    )
    k, n = _count_context_in_read(read2, sites, trim_ends=0)
    assert (k, n) == (0, 1)

    # Read shows C at pos 101 -> uninformative for - strand site
    read3 = FakeRead(
        reference_start=100,
        query_sequence="ACT",
        aligned_pairs=[(0, 100), (1, 101), (2, 102)],
    )
    k, n = _count_context_in_read(read3, sites, trim_ends=0)
    assert (k, n) == (0, 0)


def test_trim_ends_excludes_sites():
    # Read covers ref 100-110 (11 bases). Trim 3 -> usable 103..107.
    # Site at pos 101 (in trimmed region) should be excluded.
    sites = _ctx_array([(101, "C"), (105, "C")])
    seq = "C" * 11  # all methylated where applicable
    pairs = [(i, 100 + i) for i in range(11)]
    read = FakeRead(reference_start=100, query_sequence=seq, aligned_pairs=pairs)

    k, n = _count_context_in_read(read, sites, trim_ends=3)
    # Only site at 105 should be counted.
    assert (k, n) == (1, 1)


def test_no_sites_in_range_returns_zero():
    sites = _ctx_array([(500, "C")])
    read = FakeRead(
        reference_start=100,
        query_sequence="C" * 10,
        aligned_pairs=[(i, 100 + i) for i in range(10)],
    )
    k, n = _count_context_in_read(read, sites, trim_ends=0)
    assert (k, n) == (0, 0)


def test_multiple_sites_mixed():
    sites = _ctx_array([(100, "C"), (102, "G"), (105, "C")])
    # ref pos 100 -> read C (methylated +)
    # ref pos 102 -> read A (unmethylated -)
    # ref pos 105 -> read T (unmethylated +)
    read = FakeRead(
        reference_start=100,
        query_sequence="CTAATT",  # idx 0=C, 1=T, 2=A, 3=A, 4=T, 5=T
        aligned_pairs=[(0, 100), (1, 101), (2, 102), (3, 103), (4, 104), (5, 105)],
    )
    k, n = _count_context_in_read(read, sites, trim_ends=0)
    # methylated: only pos 100; unmethylated: pos 102 and 105 -> (1, 3)
    assert (k, n) == (1, 3)
