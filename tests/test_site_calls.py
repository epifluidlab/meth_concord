"""Tests for the per-site cytosine extractor and the distance-correlation math.

The extractor must agree EXACTLY with bam_parser's aggregate counter — a read
that contributes (k, n) upstream must yield n sites here of which k are
methylated — otherwise the site-level curves describe different data than the
rest of the pipeline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from nome_hic_concordance.bam_parser import _count_context_in_read
from nome_hic_concordance.site_calls import site_calls_in_read

from .test_bam_parser import FakeRead

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "site_distance_correlation.py")


def _mod():
    spec = importlib.util.spec_from_file_location("sdc", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sdc"] = m
    spec.loader.exec_module(m)
    return m


def _read(ref_seq: str, query_seq: str, start: int = 0):
    pairs = [(i, start + i) for i in range(len(query_seq))]
    return FakeRead(reference_start=start, query_sequence=query_seq,
                    aligned_pairs=pairs, reference_sequence=ref_seq)


# --------------------------------------------------------------------------
# The extractor agrees with the aggregate counter
# --------------------------------------------------------------------------

def test_site_calls_match_aggregate_counts():
    #                0123456789
    ref = "AACGAACGAA"
    # C→T read: pos2 C stays methylated, pos6 C converted to T
    qry = "AACGAATGAA"
    r = _read(ref, qry)
    ctx_pos = np.array([2, 6], dtype=np.int64)
    ctx_ref = np.array([67, 67], dtype=np.uint8)      # both ref=C

    k, n = _count_context_in_read(r, ctx_pos, ctx_ref, trim_ends=0)
    pos, st = site_calls_in_read(r, ctx_pos, ctx_ref, trim_ends=0)
    assert n == pos.size, "site count must equal the aggregate n"
    assert k == int(st.sum()), "methylated count must equal the aggregate k"
    assert list(pos) == [2, 6]
    assert list(st) == [1, 0]                          # meth, unmeth


def test_site_calls_respect_trim_ends():
    ref = "AACGAACGAA"
    qry = "AACGAATGAA"          # C→T at pos 6 — also fixes the strand call
    r = _read(ref, qry)
    ctx_pos = np.array([2, 6], dtype=np.int64)
    ctx_ref = np.array([67, 67], dtype=np.uint8)
    pos_all, _ = site_calls_in_read(r, ctx_pos, ctx_ref, trim_ends=0)
    pos_trim, _ = site_calls_in_read(r, ctx_pos, ctx_ref, trim_ends=3)
    assert list(pos_all) == [2, 6]
    assert list(pos_trim) == [6]      # pos 2 falls inside the trimmed margin


def test_site_calls_skip_wrong_strand_sites():
    """A C→T read must not score ref=G sites (they'd look falsely methylated)."""
    ref = "AACGAACGAA"
    qry = "AATGAACGAA"                                 # a C→T conversion present
    r = _read(ref, qry)
    ctx_pos = np.array([2, 3], dtype=np.int64)
    ctx_ref = np.array([67, 71], dtype=np.uint8)       # one ref=C, one ref=G
    pos, _ = site_calls_in_read(r, ctx_pos, ctx_ref, trim_ends=0)
    assert list(pos) == [2], "only the ref=C site is informative for a C→T read"


def test_site_calls_empty_when_strand_undeterminable():
    ref = "AACGAACGAA"
    r = _read(ref, ref)                                # no mismatches at all
    ctx_pos = np.array([2, 6], dtype=np.int64)
    ctx_ref = np.array([67, 67], dtype=np.uint8)
    k, n = _count_context_in_read(r, ctx_pos, ctx_ref, trim_ends=0)
    pos, st = site_calls_in_read(r, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (0, 0) and pos.size == 0 and st.size == 0


# --------------------------------------------------------------------------
# Distance binning / pairing / correlation math
# --------------------------------------------------------------------------

def test_pair_sites_within_read_is_upper_triangle():
    m = _mod()
    pos = np.array([0, 10, 100], np.int64)
    st = np.array([1, 0, 1], np.uint8)
    edges = m._build_edges(1, 1000, 3)
    rng = np.random.default_rng(0)
    bins, x, y = m._pair_sites(pos, st, pos, st, edges, True, rng, 0)
    assert bins.size == 3            # 3 choose 2 — never a site with itself
    # Full cross product is 9, but the 3 diagonal (i==j) entries have distance 0,
    # which is below --min-distance and so is filtered out: a site is never
    # correlated against itself even on the cross-product path.
    b2, _, _ = m._pair_sites(pos, st, pos, st, edges, False, rng, 0)
    assert b2.size == 6


def test_pair_sites_cap_bounds_the_blowup():
    m = _mod()
    pos = np.arange(40, dtype=np.int64) * 3
    st = (np.arange(40) % 2).astype(np.uint8)
    edges = m._build_edges(1, 1000, 3)
    rng = np.random.default_rng(0)
    bins, _, _ = m._pair_sites(pos, st, pos, st, edges, True, rng, cap=25)
    assert bins.size <= 25


def test_corr_matches_numpy_pearson():
    m = _mod()
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, 400).astype(np.uint8)
    y = ((x + rng.integers(0, 2, 400)) % 2).astype(np.uint8)
    acc = m._new_acc(1)
    m._accumulate(acc, np.zeros(400, np.int64), x, y)
    r, n = m._corr(acc, min_pairs=1)
    assert n[0] == 400
    assert abs(r[0] - np.corrcoef(x, y)[0, 1]) < 1e-9


def test_corr_respects_min_pairs():
    m = _mod()
    acc = m._new_acc(1)
    x = np.array([1, 0, 1, 0], np.uint8)
    m._accumulate(acc, np.zeros(4, np.int64), x, x)
    assert np.isnan(m._corr(acc, min_pairs=100)[0][0])   # too few → NaN
    assert m._corr(acc, min_pairs=1)[0][0] == pytest.approx(1.0)


def test_build_edges_exact_bounds():
    m = _mod()
    e = m._build_edges(1, 1_000_000, 8)
    assert e[0] == 1 and e[-1] == 1_000_000
    assert len(e) == 6 * 8 + 1


def test_perfectly_correlated_sites_give_r_one():
    """End-to-end on the accumulator: identical states → phi = 1."""
    m = _mod()
    edges = m._build_edges(1, 1000, 3)
    rng = np.random.default_rng(0)
    accs = {("gg", "linked"): m._new_acc(len(edges) - 1)}
    pos = np.array([0, 50], np.int64)
    for s in (0, 1, 0, 1, 1, 0):                       # varying but identical
        st = np.array([s, s], np.uint8)
        b, x, y = m._pair_sites(pos, st, pos, st, edges, True, rng, 0)
        m._accumulate(accs[("gg", "linked")], b, x, y)
    r, n = m._corr(accs[("gg", "linked")], min_pairs=1)
    assert n.sum() == 6
    assert np.nanmax(r) == pytest.approx(1.0)
