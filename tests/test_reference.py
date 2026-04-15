"""Unit tests for the pure-Python logic in reference.py (no .2bit / pysam)."""

from __future__ import annotations

from nome_hic_concordance.reference import (
    _matches_context,
    _reverse_complement_context,
)


def test_matches_context_gch():
    # GCH matches GCA, GCC, GCT. GCG fails because H={A,C,T} excludes G.
    assert _matches_context("GCA", "GCH")
    assert _matches_context("GCC", "GCH")
    assert _matches_context("GCT", "GCH")
    assert not _matches_context("GCG", "GCH")


def test_matches_context_hcg():
    assert _matches_context("ACG", "HCG")
    assert _matches_context("CCG", "HCG")
    assert _matches_context("TCG", "HCG")
    assert not _matches_context("GCG", "HCG")


def test_matches_context_wcg():
    # W = {A, T}
    assert _matches_context("ACG", "WCG")
    assert _matches_context("TCG", "WCG")
    assert not _matches_context("CCG", "WCG")
    assert not _matches_context("GCG", "WCG")


def test_matches_context_rejects_wrong_length():
    assert not _matches_context("AC", "HCG")
    assert not _matches_context("ACGT", "HCG")


def test_reverse_complement_context():
    # GCH -> complement each base, reverse: G->C, C->G, H->D => HGC reversed = CGH... hmm
    # Let's trace: GCH reversed = HCG, complement HCG letter-by-letter = DGC.
    # Actually we reverse FIRST then complement per the implementation order:
    # "".join(IUPAC_COMP[c] for c in reversed(context))
    # reversed("GCH") = "HCG", complementing H->D, C->G, G->C gives "DGC".
    assert _reverse_complement_context("GCH") == "DGC"

    # HCG -> reversed = GCH, complement G->C, C->G, H->D -> CGD
    assert _reverse_complement_context("HCG") == "CGD"

    # WCG -> reversed = GCW, complement G->C, C->G, W->W -> CGW
    assert _reverse_complement_context("WCG") == "CGW"

    # Palindrome: CCGG -> reversed = GGCC, complement C,C,G,G -> CCGG. But we use 3-letter.
    # ACG -> reversed = GCA, complement = CGT
    assert _reverse_complement_context("ACG") == "CGT"
