"""Tests for bam_parser._count_context_in_read with strand inference.

FakeRead mimics the pysam.AlignedSegment interface.  Each test provides
a ``reference_sequence`` so that ``_infer_conversion_strand`` can count
C→T vs G→A mismatches and choose the correct strand.
"""

from __future__ import annotations

import numpy as np

from nome_hic_concordance.bam_parser import (
    _count_context_in_read,
    _infer_conversion_strand,
)
from nome_hic_concordance.reference import CONTEXT_DTYPE


class FakeRead:
    """Minimal duck-type matching pysam.AlignedSegment."""

    def __init__(
        self,
        reference_start: int,
        query_sequence: str,
        aligned_pairs: list[tuple[int, int]],
        reference_sequence: str,
        cigartuples: list[tuple[int, int]] | None = None,
        query_alignment_start: int = 0,
    ):
        self.reference_start = reference_start
        self.query_sequence = query_sequence
        self._aligned_pairs = aligned_pairs
        self._reference_sequence = reference_sequence
        self.query_alignment_start = query_alignment_start

        if cigartuples is None:
            ref_positions = [p for _, p in aligned_pairs if p is not None]
            span = max(ref_positions) - min(ref_positions) + 1 if ref_positions else 0
            self.cigartuples = [(0, span)]
        else:
            self.cigartuples = cigartuples

    @property
    def flag(self):
        return 0x2

    @property
    def reference_end(self):
        ref_positions = [p for _, p in self._aligned_pairs if p is not None]
        return max(ref_positions) + 1

    @property
    def reference_length(self) -> int:
        return sum(ln for op, ln in self.cigartuples if op in (0, 2, 3, 7, 8))

    @property
    def query_alignment_length(self) -> int:
        return sum(ln for op, ln in self.cigartuples if op in (0, 1, 7, 8))

    def get_aligned_pairs(self, matches_only: bool = False):
        return list(self._aligned_pairs)

    def get_reference_sequence(self) -> str:
        return self._reference_sequence


def _ctx_arrays(positions_and_bases: list[tuple[int, str]]):
    arr = np.empty(len(positions_and_bases), dtype=CONTEXT_DTYPE)
    arr["pos"] = [p for p, _ in positions_and_bases]
    arr["ref_base"] = [ord(b) for _, b in positions_and_bases]
    return arr["pos"], arr["ref_base"]


# ---------------------------------------------------------------------------
# Strand inference tests
# ---------------------------------------------------------------------------


def test_infer_ct_strand():
    """Read with C→T mismatches is classified as CT strand."""
    # ref=ACGCGA, read=ATGTGA → C→T at pos 1 and 3
    read = FakeRead(
        reference_start=100, query_sequence="ATGTGA",
        aligned_pairs=[(i, 100 + i) for i in range(6)],
        reference_sequence="ACGCGA",
    )
    assert _infer_conversion_strand(read) == "CT"


def test_infer_ga_strand():
    """Read with G→A mismatches is classified as GA strand."""
    # ref=ACGCGA, read=ACACCA → G→A at pos 2 and 4
    read = FakeRead(
        reference_start=100, query_sequence="ACACCA",
        aligned_pairs=[(i, 100 + i) for i in range(6)],
        reference_sequence="ACGCGA",
    )
    assert _infer_conversion_strand(read) == "GA"


def test_infer_unknown_strand():
    """No mismatches → unknown strand."""
    read = FakeRead(
        reference_start=100, query_sequence="ACGCGA",
        aligned_pairs=[(i, 100 + i) for i in range(6)],
        reference_sequence="ACGCGA",
    )
    assert _infer_conversion_strand(read) == "unknown"


# ---------------------------------------------------------------------------
# Context counting with strand awareness
# ---------------------------------------------------------------------------


def test_ct_read_at_plus_strand_site():
    """C→T read at ref=C site: C=methylated, T=unmethylated."""
    ctx_pos, ctx_ref = _ctx_arrays([(103, "C")])

    # ref=ACGACACT, read=ATGATACT → C→T at pos 1,3 → CT strand
    # At pos 103 (ref=C): read has 'A' → ref has 'C', read 'A' → but
    # let me construct more carefully:
    # Positions 100-107, ref = "CCGCCACT", read = "TCGTCACT"
    # C→T at pos 100 and 103 (2 C→T), 0 G→A → CT strand
    # At context site pos 103 (ref=C): read has 'C' → methylated? No,
    # read at pos 103 is 'C' (no conversion here = methylated)

    # Let me make it simpler:
    # 6 bases starting at pos 100
    # ref = "ACCCGA", read = "ATCTGA"
    # C→T at pos 101 and 103 (2 C→T), G→A at... pos 104: ref G read G (no). → CT strand
    # Context site at pos 102 (ref=C): read shows 'C' → methylated

    read = FakeRead(
        reference_start=100,
        query_sequence="ATCTGA",
        aligned_pairs=[(i, 100 + i) for i in range(6)],
        reference_sequence="ACCCGA",
    )
    # C→T at pos 101 (ref C, read T) and pos 103 (ref C, read T)
    # → CT strand. Context site at pos 102 (ref=C): read='C' → methylated.
    ctx_pos, ctx_ref = _ctx_arrays([(102, "C")])
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (1, 1)


def test_ct_read_at_plus_strand_site_unmethylated():
    """C→T read at ref=C site with T → unmethylated."""
    # ref = "ACCCGA", read = "ATTTGA"
    # C→T at 101, 102, 103 → CT strand
    # Context site at 102 (ref=C): read='T' → unmethylated
    read = FakeRead(
        reference_start=100,
        query_sequence="ATTTGA",
        aligned_pairs=[(i, 100 + i) for i in range(6)],
        reference_sequence="ACCCGA",
    )
    ctx_pos, ctx_ref = _ctx_arrays([(102, "C")])
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (0, 1)


def test_ct_read_skips_minus_strand_sites():
    """C→T read at ref=G (− strand) site is skipped — not informative."""
    # ref = "ACCCGA", read = "ATCTGA"
    # C→T at 101, 103 → CT strand
    # Context site at 104 (ref=G): should be SKIPPED for CT reads
    read = FakeRead(
        reference_start=100,
        query_sequence="ATCTGA",
        aligned_pairs=[(i, 100 + i) for i in range(6)],
        reference_sequence="ACCCGA",
    )
    ctx_pos, ctx_ref = _ctx_arrays([(104, "G")])
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (0, 0)  # skipped, NOT (1, 1)


def test_ga_read_at_minus_strand_site():
    """G→A read at ref=G site: G=methylated, A=unmethylated."""
    # ref = "TGAGCT", read = "TAACCT"
    # G→A at pos 101 and 103 → GA strand
    # No C→T mismatches (ref C at pos 104 → read C, no change)
    # Context site at 101 (ref=G): read='A' → unmethylated
    read = FakeRead(
        reference_start=100,
        query_sequence="TAACCT",
        aligned_pairs=[(i, 100 + i) for i in range(6)],
        reference_sequence="TGAGCT",
    )
    ctx_pos, ctx_ref = _ctx_arrays([(101, "G")])
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (0, 1)


def test_ga_read_skips_plus_strand_sites():
    """G→A read at ref=C (+ strand) site is skipped."""
    # Same read as above: GA strand
    # Context site at 104 (ref=C): should be SKIPPED for GA reads
    read = FakeRead(
        reference_start=100,
        query_sequence="TAACCT",
        aligned_pairs=[(i, 100 + i) for i in range(6)],
        reference_sequence="TGAGCT",
    )
    ctx_pos, ctx_ref = _ctx_arrays([(104, "C")])
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (0, 0)


def test_unknown_strand_returns_zero():
    """Read with no mismatches → unknown strand → (0, 0)."""
    read = FakeRead(
        reference_start=100,
        query_sequence="ACGCGA",
        aligned_pairs=[(i, 100 + i) for i in range(6)],
        reference_sequence="ACGCGA",
    )
    ctx_pos, ctx_ref = _ctx_arrays([(102, "C")])
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (0, 0)


def test_trim_ends_excludes_sites():
    """Sites within trim_ends of read boundaries are excluded."""
    # ref = "CCCCCCCCCCCC" (12 C's), read has T at pos 5
    # ref = "CCCCCCCCCCCC", read = "CCCCCTTCCCCC"
    # C→T at pos 105, 106 → CT strand
    ref_seq = "C" * 12
    query = "CCCCCTTCCCCC"
    read = FakeRead(
        reference_start=100,
        query_sequence=query,
        aligned_pairs=[(i, 100 + i) for i in range(12)],
        reference_sequence=ref_seq,
    )
    # Context site at 101 (within trim=3, should be excluded)
    # Context site at 107 (outside trim, ref=C, read=C → methylated)
    ctx_pos, ctx_ref = _ctx_arrays([(101, "C"), (107, "C")])
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=3)
    # Only site at 107 is within trimmed range [103, 109)
    assert (k, n) == (1, 1)


def test_no_sites_in_range():
    """No context sites in the read's span → (0, 0)."""
    read = FakeRead(
        reference_start=100,
        query_sequence="ATCCGA",
        aligned_pairs=[(i, 100 + i) for i in range(6)],
        reference_sequence="ACCCGA",
    )
    ctx_pos, ctx_ref = _ctx_arrays([(500, "C")])
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (0, 0)


def test_mixed_sites_ct_strand():
    """CT read with multiple + strand sites, some meth some unmeth."""
    # ref = "ACGCGATCG", read = "ATGTAATTG"
    # 9 bases at positions 100-108
    # C→T at: 101 (C→T), 103 (C→T), 107 (C→T) → 3 C→T, 0 G→A → CT strand
    # Context sites: 101 (ref=C, read=T → unmeth), 103 (ref=C, read=A → skip),
    #                105 (ref=G, skip for CT), 107 (ref=C, read=T → unmeth)
    # Wait, let me be more careful with the sequence:
    # ref = "ACGCGATCG"
    # pos:   100 101 102 103 104 105 106 107 108
    # ref:    A   C   G   C   G   A   T   C   G
    # read:   A   T   G   T   G   A   T   T   G
    # Mismatches: pos 101 C→T, pos 103 C→T, pos 107 C→T → CT strand
    # Context sites at ref=C positions: 101, 103, 107
    read = FakeRead(
        reference_start=100,
        query_sequence="ATGTGATTG",
        aligned_pairs=[(i, 100 + i) for i in range(9)],
        reference_sequence="ACGCGATCG",
    )
    ctx_pos, ctx_ref = _ctx_arrays([(101, "C"), (103, "C"), (107, "C")])
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    # All 3 sites: read shows T → unmethylated
    assert (k, n) == (0, 3)


def test_indel_with_ct_strand():
    """Read with indels uses the slow path and respects strand inference."""
    # ref at 100-104: "ACCGA", read: "ACTCGA" (1 insertion after pos 101)
    # aligned_pairs: (0,100), (1,101), (2,None), (3,102), (4,103), (5,104)
    # matches_only: (0,100), (1,101), (3,102), (4,103), (5,104)
    # ref_seq for get_reference_sequence: based on MD tag, covers the aligned
    # portion.  For an insertion, the reference portion is the same as without
    # the inserted base.  ref_seq = "ACCGA" (5 ref bases).
    # Mismatches: pos 101 ref=C read=C (no mismatch)... hmm, need a C→T.
    # Let me redo: ref="ACCGA", read="ATTGA" with 1I after pos 101
    # aligned_pairs (matches_only): (0,100), (1,101), (3,103), (4,104)
    # ref_seq = "ACGA" (4 ref bases consumed by matches)
    # Wait, this is getting complex. Let me simplify:

    # ref = "ACGA" at positions 100-103
    # CIGAR: 1M 1I 2M → ref consumes pos 100, skip, pos 101, 102
    # Wait no, 1M at pos 100, 1I (no ref consumed), 2M at pos 101-102
    # That's only 3 ref bases (100, 101, 102)
    # ref_seq = "ACG" (for positions 100, 101, 102)
    # read = "ATCG" (4 query bases: A at 100, T inserted, C at 101, G at 102)
    # aligned_pairs (matches_only): (0,100), (2,101), (3,102)
    # C→T? ref_seq[0]='A' read[0]='A' no. ref_seq[1]='C' read[2]='C' no.
    # No C→T mismatches. Need to add one elsewhere.

    # OK let me make a longer read:
    # ref = "ACCGAT" at positions 100-105
    # CIGAR: 3M 1I 3M → positions 100-102 matched, 1 inserted, 103-105 matched
    # ref_seq = "ACCGAT"
    # read = "ATCXGAT" where X is inserted (7 query bases)
    # aligned_pairs (matches_only): (0,100),(1,101),(2,102),(4,103),(5,104),(6,105)
    # ref_seq = "ACCGAT", read aligned = "A T C G A T"
    # C→T at pos 101 (ref C, read T) → CT strand
    read = FakeRead(
        reference_start=100,
        query_sequence="ATCNGAT",  # N is the inserted base
        aligned_pairs=[(0, 100), (1, 101), (2, 102), (4, 103), (5, 104), (6, 105)],
        reference_sequence="ACCGAT",
        cigartuples=[(0, 3), (1, 1), (0, 3)],  # 3M1I3M
        query_alignment_start=0,
    )
    # CT strand. Context site at 102 (ref=C): read[2]='C' → methylated
    ctx_pos, ctx_ref = _ctx_arrays([(102, "C")])
    k, n = _count_context_in_read(read, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (1, 1)
