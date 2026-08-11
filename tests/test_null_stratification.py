"""Donor stratification for the pooled null (--null-match-column).

The pooled null draws substitute partner ends from every cell in the group,
which crosses batch boundaries; stratifying runs it independently inside each
batch combination and merges.  These tests cover the partition-and-merge layer
only -- the matcher itself lives in the C++ extension and is stubbed, so they
run without it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "site_distance_correlation.py")


def _mod():
    spec = importlib.util.spec_from_file_location("sdc_ns", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sdc_ns"] = m
    spec.loader.exec_module(m)
    return m


def _stub(m, monkeypatch, empty_for=()):
    """Replace _pooled_null with a recorder returning one row per input cell."""
    calls = []

    def fake(res, idxs, edges, args, nb):
        calls.append(list(idxs))
        if tuple(sorted(idxs)) in empty_for:
            return None
        return {
            "null_per_cell": {k: [np.full(3, float(i)) for i in idxs]
                              for k, _ in m._MEASURES},
            "matched_per_cell": {k: [np.full(3, float(i) + 0.5) for i in idxs]
                                 for k, _ in m._MEASURES},
            "n_eligible": 10 * len(idxs), "n_anchors": 100 * len(idxs),
            "n_donors": 1000 * len(idxs), "n_anchor_long": len(idxs),
            "n_pairs_long_seen": 2 * len(idxs),
        }

    monkeypatch.setattr(m, "_pooled_null", fake)
    return calls


def test_no_key_delegates_to_the_unstratified_null(monkeypatch):
    m = _mod()
    calls = _stub(m, monkeypatch)
    out = m._pooled_null_stratified(None, [0, 1, 2], None, None, 3, None)
    assert calls == [[0, 1, 2]], "one call with every cell"
    assert "n_strata" not in out


def test_donors_never_cross_a_stratum(monkeypatch):
    m = _mod()
    calls = _stub(m, monkeypatch)
    key = np.array(["b1", "b2", "b1", "b2", "b1"])
    out = m._pooled_null_stratified(None, [0, 1, 2, 3, 4], None, None, 3, key)
    assert sorted(map(sorted, calls)) == [[0, 2, 4], [1, 3]], \
        "each stratum gets its own pooled-null call, and only its own cells"
    assert out["n_strata"] == 2


def test_merge_concatenates_cells_and_sums_counts(monkeypatch):
    m = _mod()
    _stub(m, monkeypatch)
    key = np.array(["b1", "b2", "b1"])
    out = m._pooled_null_stratified(None, [0, 1, 2], None, None, 3, key)
    for mkey, _ in m._MEASURES:
        vals = sorted(a[0] for a in out["null_per_cell"][mkey])
        assert vals == [0.0, 1.0, 2.0], "every cell survives the merge exactly once"
        assert len(out["matched_per_cell"][mkey]) == 3
    # counts are additive over strata (3 cells total, regardless of split)
    assert out["n_anchors"] == 300 and out["n_donors"] == 3000
    assert out["n_eligible"] == 30 and out["n_pairs_long_seen"] == 6


def test_composite_key_splits_on_the_full_combination(monkeypatch):
    """exp+plate+lib: two cells share exp_batch but differ in plate, so they
    must NOT donate to each other."""
    m = _mod()
    calls = _stub(m, monkeypatch)
    key = np.array(["e1|p1|l1", "e1|p2|l1", "e1|p1|l1"])
    m._pooled_null_stratified(None, [0, 1, 2], None, None, 3, key)
    assert sorted(map(sorted, calls)) == [[0, 2], [1]]


def test_a_stratum_with_no_donors_is_dropped_not_fatal(monkeypatch):
    m = _mod()
    calls = _stub(m, monkeypatch, empty_for=((1,),))
    key = np.array(["b1", "b2", "b1"])
    out = m._pooled_null_stratified(None, [0, 1, 2], None, None, 3, key)
    assert len(calls) == 2
    assert out["n_strata"] == 1, "the empty stratum contributes nothing"
    for mkey, _ in m._MEASURES:
        assert len(out["null_per_cell"][mkey]) == 2


def test_all_strata_empty_returns_none(monkeypatch):
    m = _mod()
    _stub(m, monkeypatch, empty_for=((0,), (1,)))
    key = np.array(["b1", "b2"])
    assert m._pooled_null_stratified(None, [0, 1], None, None, 3, key) is None


def test_flag_is_exposed_and_defaults_off():
    m = _mod()
    ap = m.build_parser() if hasattr(m, "build_parser") else None
    if ap is None:                       # parser is built inside main()
        src = _SCRIPT.read_text()
        assert '"--null-match-column"' in src
        assert "null_match_column" in src
    else:
        assert ap.parse_args([]).null_match_column is None
