"""Distance-matched null, built during the per-cell parse.

The locus-matched null swaps end 2 for another end within ±window of the TRUE
partner, so it needs a read end beside one specific position — inside a single
cell that essentially never exists past a few kb, and the null comes back
empty.  The distance-matched null keeps the SEPARATION and redraws the locus,
matching to the observed pair's own distance BIN.

These tests drive the real extension over a synthetic BAM, because the bug
this replaced was invisible at the unit level: the first implementation read
from the pooled donor reservoir (~1e4 ends/cell) instead of the parse's
complete end set (~1e6), a density at which no end ever lands at a prescribed
separation.  Only an end-to-end fixture shows that.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pysam = pytest.importorskip("pysam")            # also loads the bundled libhts
_CPP = pytest.importorskip("nome_hic_concordance._site_dist_core")

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "site_distance_correlation.py")
_EDGES = list(np.logspace(0, 6, 49))
_READ_LEN = 100


def _driver():
    spec = importlib.util.spec_from_file_location("sdc_dm", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sdc_dm"] = m
    spec.loader.exec_module(m)
    return m


def _write_bam(path, pairs, chrom_len=2_000_000):
    """One BAM, one cell.  Each (p1, p2) is a read pair at those start
    coordinates; reads carry an MD tag with a C->T mismatch so the conversion
    strand can be inferred, and are otherwise all C at the context positions."""
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"},
           "SQ": [{"SN": "chr1", "LN": chrom_len}]}
    recs = []
    for k, (p1, p2) in enumerate(pairs):
        for name, pos, mpos, flag in ((f"r{k}", p1, p2, 99),
                                      (f"r{k}", p2, p1, 147)):
            a = pysam.AlignedSegment()
            a.query_name = name
            a.flag = flag
            a.reference_id = 0
            a.reference_start = pos
            a.mapping_quality = 60
            a.cigarstring = f"{_READ_LEN}M"
            a.query_sequence = "T" + "C" * (_READ_LEN - 1)
            a.query_qualities = pysam.qualitystring_to_array("I" * _READ_LEN)
            a.next_reference_id = 0
            a.next_reference_start = mpos
            a.set_tag("MD", f"0C{_READ_LEN - 1}")
            recs.append(a)
    with pysam.AlignmentFile(str(path), "wb", header=hdr) as out:
        for a in sorted(recs, key=lambda x: x.reference_start):
            out.write(a)
    pysam.index(str(path))
    return str(path)


def _ctx(chrom_len=2_000_000):
    pos = np.ascontiguousarray(np.arange(0, chrom_len, 2), np.int64)
    ref = np.ascontiguousarray(np.full(len(pos), ord("C")), np.uint8)
    return {"chr1": (pos, ref)}


def _parse(bam, null_type, **kw):
    kw.setdefault("null_window", 100)
    kw.setdefault("anchor_window", 10_000_000)
    return _CPP.parse_cell_site_distance(
        bam, _ctx(), {}, _EDGES, trim_ends=0, mapq_min=0,
        max_site_pairs_per_read=64, seed=7, null_type=null_type, **kw)


def _n(out, key):
    return float(np.asarray(out[key])[0].sum())


# --------------------------------------------------------------- end to end --

def test_distance_matched_finds_a_null_where_locus_matched_finds_none(tmp_path):
    """The whole point.  A ladder of ends 5 kb apart: no end sits within 100 bp
    of any true partner, so the locus-matched null is empty — but cross-pair
    combinations DO sit at 5 kb, so the distance-matched null is not."""
    bam = _write_bam(tmp_path / "a.bam",
                     [(1000, 6000), (11000, 16000), (21000, 26000)])
    locus = _parse(bam, 0)
    dist = _parse(bam, 1)
    assert locus["n_null_eligible"] == 0 and _n(locus, "gg_null") == 0
    assert dist["n_null_eligible"] == 3, "every pair should be matchable"
    assert _n(dist, "gg_null") > 0
    assert _n(dist, "gg_matched") > 0, "MATCHED accumulates with the null"


def test_a_true_ligation_is_never_used_as_its_own_null(tmp_path):
    """Isolated pairs whose only same-separation partners are their OWN two
    ends must yield nothing — using them would compare a pair to itself."""
    bam = _write_bam(tmp_path / "b.bam",
                     [(1000, 6000), (500_000, 505_000)])
    out = _parse(bam, 1)
    assert out["n_null_eligible"] == 0
    assert _n(out, "gg_null") == 0


def test_separation_is_matched_to_the_observed_bin(tmp_path):
    """The null pair must land in the same distance bin as the pair it stands
    in for — the bin is the resolution the figure has."""
    bam = _write_bam(tmp_path / "c.bam",
                     [(1000, 6000), (11000, 16000), (21000, 26000)])
    out = _parse(bam, 1)
    null_n = np.asarray(out["gg_null"])[0]
    linked_n = np.asarray(out["gg_linked"])[0]
    # The cross-end linked arm and the null both sit at ~5 kb; the linked arm
    # additionally has the within-read (short) bins, so compare only where the
    # null is non-zero.
    occupied = np.flatnonzero(null_n > 0)
    assert occupied.size > 0
    for b in occupied:
        assert linked_n[b] > 0, (
            f"null occupies bin {b} where the linked cross-end arm does not")


def test_anchor_window_confines_the_synthetic_pair(tmp_path):
    """A tight window leaves no usable anchor near the far pair; widening it
    reaches the ladder."""
    bam = _write_bam(tmp_path / "d.bam",
                     [(1000, 6000), (11000, 16000), (900_000, 905_000)])
    tight = _parse(bam, 1, anchor_window=5_000)
    wide = _parse(bam, 1, anchor_window=10_000_000)
    assert wide["n_null_eligible"] > tight["n_null_eligible"]


def test_zero_anchor_window_means_no_locality_restriction(tmp_path):
    bam = _write_bam(tmp_path / "e.bam",
                     [(1000, 6000), (900_000, 905_000), (910_000, 915_000)])
    out = _parse(bam, 1, anchor_window=0)
    assert out["n_null_eligible"] > 0


def test_locus_matched_still_works_when_ends_are_close(tmp_path):
    """The old null is untouched: with two partners within ±100 bp it fires."""
    bam = _write_bam(tmp_path / "f.bam",
                     [(1000, 6000), (1200, 6050)])
    out = _parse(bam, 0, null_window=100)
    assert out["n_null_eligible"] > 0
    assert _n(out, "gg_null") > 0


# ----------------------------------------------------------------- routing --

def test_distance_null_uses_the_parse_not_the_reservoir():
    """Regression: routing it through the reservoir gave an empty null at every
    distance, because the reservoir holds far too few ends per cell."""
    m = _driver()
    res, dist, notes = m._resolve_null_path("distance-matched", "cell", True)
    assert dist is True
    assert res is False, "must NOT take the donor-reservoir path"


def test_pooled_scope_is_reported_as_inapplicable_to_the_distance_null():
    m = _driver()
    res, dist, notes = m._resolve_null_path("distance-matched", "pooled", True)
    assert res is False and dist is True
    assert any("within-cell" in n for n in notes)


def test_locus_matched_routing_is_unchanged():
    m = _driver()
    assert m._resolve_null_path("locus-matched", "cell", True)[0] is False
    assert m._resolve_null_path("locus-matched", "pooled", True)[0] is True


def test_distance_null_without_the_backend_exits():
    m = _driver()
    with pytest.raises(SystemExit, match="htslib"):
        m._resolve_null_path("distance-matched", "cell", False)


def test_driver_forwards_null_type_and_anchor_window(tmp_path):
    """process_cell_cpp must pass both through to the extension."""
    m = _driver()
    bam = _write_bam(tmp_path / "g.bam",
                     [(1000, 6000), (11000, 16000), (21000, 26000)])
    prepared = (_ctx(), {}, {})
    accs, n_pairs, n_elig, *_ = m.process_cell_cpp(
        bam, prepared, np.asarray(_EDGES), 0, 0, 64, 100, 7,
        null_type=1, anchor_window=10_000_000)
    assert n_pairs == 3 and n_elig == 3
    assert accs[("gg", "null")][0].sum() > 0


# ------------------------------------------------------------ log reporting --

def test_distance_counts_table_flags_a_null_below_min_pairs(capsys):
    m = _driver()
    centers = np.array([100.0, 10_000.0])
    accs = {}
    for key, _ in m._MEASURES:
        for cond, _l in m._CONDS:
            a = m._new_acc(2)
            a[0] = [5000, 4000] if cond == "linked" else [900, 5]
            accs[(key, cond)] = a

    class A:
        min_pairs = 100
        log_distance_counts = True
    m._print_distance_counts(accs, centers, A())
    out = capsys.readouterr().out
    assert "100 bp" in out and "10 kb" in out and "5,000" in out
    assert "*" in out, "a null below min_pairs must be flagged as masked"

    class B(A):
        log_distance_counts = False
    m._print_distance_counts(accs, centers, B())
    assert capsys.readouterr().out == ""
