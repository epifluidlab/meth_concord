"""Correlating the two CONTEXTS with each other, per read pair.

The third panel used to be HCG-GCH: a CpG on one end against a GpC on the
other, which asks whether methylation at one locus tracks accessibility at
another.  That is a cross-context question and its curve hugs zero an order
of magnitude below the other two.

The question the panel was wanted for is different: among read pairs that
carry BOTH contexts on BOTH ends, does a pair that is co-accessible also
tend to be co-methylated, and how does that vary with distance.  That is a
property of the read PAIR, so a pair contributes one observation rather
than one per site pair -- a different unit from every other measure here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / \
    "site_distance_correlation.py"


def _mod():
    spec = importlib.util.spec_from_file_location("_sdc", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EDGES = np.array([1e2, 1e3, 1e4, 1e5, 1e6])


def _end(gpos, gcalls, hpos, hcalls):
    return {"GCH": (np.asarray(gpos, float), np.asarray(gcalls)),
            "HCG": (np.asarray(hpos, float), np.asarray(hcalls))}


def test_the_concordance_is_the_pipelines_own_F():
    """gg = a1 a2 + (1-a1)(1-a2), the probability the two ends are in the
    SAME state -- the same F the concordance pipeline computes, formed here
    from site calls instead of end posteriors."""
    m = _mod()
    e1 = _end([100, 200], [1, 1], [150], [0])
    e2 = _end([20100, 20200], [1, 0], [20150], [0])
    got = m._pair_concordance(e1, e2, EDGES)
    assert got is not None
    b, gg, hh = got
    # a1 = 1.0, a2 = 0.5 -> 1*0.5 + 0*0.5
    assert gg == pytest.approx(0.5)
    # m1 = 0, m2 = 0 -> 0*0 + 1*1
    assert hh == pytest.approx(1.0)
    # ~20 kb apart -> the [1e4, 1e5) bin
    assert b == 2


def test_both_contexts_on_both_ends_or_nothing():
    """A pair missing either context cannot answer the question, so it is
    dropped rather than filled in -- which is why this measure's n is
    smaller than the site-level ones' and reported separately."""
    m = _mod()
    full = _end([100], [1], [150], [0])
    no_h = {"GCH": (np.array([20100.]), np.array([1]))}
    no_g = {"HCG": (np.array([20150.]), np.array([0]))}
    empty_g = {"GCH": (np.empty(0), np.empty(0)),
               "HCG": (np.array([20150.]), np.array([0]))}
    assert m._pair_concordance(full, no_h, EDGES) is None
    assert m._pair_concordance(full, no_g, EDGES) is None
    assert m._pair_concordance(full, empty_g, EDGES) is None
    assert m._pair_concordance(no_h, full, EDGES) is None


def test_the_separation_uses_every_site_the_pair_carries():
    """The two contexts tile the same read, so taking the midpoint from one
    of them alone would shift it."""
    m = _mod()
    e1 = _end([0], [1], [1000], [1])          # midpoint 500
    e2 = _end([50_000], [1], [51_000], [1])   # midpoint 50,500
    b, _, _ = m._pair_concordance(e1, e2, EDGES)
    assert b == int(np.digitize([50_000.0], EDGES)[0] - 1)
    # Using GCH alone would give exactly 50,000 and HCG alone 50,000 too --
    # so build a case where they disagree.
    e3 = _end([0], [1], [90_000], [1])        # midpoint 45,000
    e4 = _end([100], [1], [100], [1])         # midpoint 100
    b2, _, _ = m._pair_concordance(e3, e4, EDGES)
    assert b2 == int(np.digitize([44_900.0], EDGES)[0] - 1)


def test_a_pair_outside_the_axis_is_dropped():
    m = _mod()
    near = _end([0], [1], [0], [1])
    also_near = _end([10], [1], [10], [1])    # 10 bp, below edges[0]
    assert m._pair_concordance(near, also_near, EDGES) is None
    far = _end([10 ** 9], [1], [10 ** 9], [1])
    assert m._pair_concordance(near, far, EDGES) is None


def test_it_accumulates_one_observation_per_pair():
    """Every other measure adds one observation per SITE pair.  This one
    adds one per read pair, which is what makes it a correlation between
    contexts rather than between sites."""
    m = _mod()
    accs = {(k, c): m._new_acc(len(EDGES) - 1)
            for k, _ in m._MEASURES for c, _ in m._CONDS}
    e1 = _end([100, 200, 300], [1, 1, 0], [150, 250], [1, 0])
    e2 = _end([20100, 20200, 20300], [1, 0, 0], [20150, 20250], [1, 1])
    m._add_pair(accs, "linked", EDGES, np.random.default_rng(0), 100, e1, e2)
    cc = accs[("cc", "linked")]
    assert cc[0].sum() == 1.0                      # exactly one observation
    # The site-level measures saw many more than one.
    assert accs[("gg", "linked")][0].sum() > 1.0


def test_the_third_panel_is_the_pair_measure_and_gh_is_still_computed():
    m = _mod()
    assert m._PLOTTED == ("gg", "hh", "cc")
    keys = [k for k, _ in m._MEASURES]
    assert keys == ["gg", "hh", "gh", "cc"]
    # 'gh' keeps its accumulator and its TSV columns, the way 'matched'
    # does -- dropped from the figure, not from the output.
    assert "gh" not in m._PLOTTED
    assert "gh" in keys


def test_the_compiled_core_computes_it_too():
    """It is the same computation the C++ core already does for GCH-GCH and
    HCG-HCG, with one extra requirement -- both contexts on both ends -- so
    there was no reason to leave it to the slow backend.  What it could NOT
    do was share the stride-3 pack: those three measures are 0/1, so the
    core stores four rows and rebuilds Sxx from Sx, and a continuous x
    breaks that identity."""
    m = _mod()
    assert m._CPP_STRIDE3 == ("gg", "hh", "gh")
    assert "cc" not in m._CPP_STRIDE3
    src = _SCRIPT.read_text()
    assert "3 * c + mi" not in src            # the old hard-coded stride
    assert "_nm * c + mi" in src
    # No backend-dependent gap left in the Python side.
    assert "needs --backend python" not in src
    assert "_new_acc(nb_)" not in src


def test_the_cpp_core_carries_the_new_measure():
    """Every path the core exposes has to produce it, or a run picks up an
    all-zero panel depending on which null it was asked for."""
    cpp = (Path(__file__).resolve().parents[1] / "nome_hic_concordance"
           / "ext" / "site_dist_core.cpp").read_text()
    for key in ('out["cc_linked"]', 'out["cc_null"]', 'out["cc_matched"]',
                'out["cc_per_cell"]', 'out["null_cc_per_cell"]',
                'out["matched_cc_per_cell"]'):
        assert key in cpp, key
    # A six-row accumulator, because Sxx == Sx only holds for 0/1 data.
    assert "struct AccF" in cpp
    assert "v(6 * (size_t)n, 0.0)" in cpp
    assert "add_pair_concordance" in cpp
