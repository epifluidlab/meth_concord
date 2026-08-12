"""Distance-matched null in the C++ backend (same_cell_distance_null).

cross_cell_null holds BOTH loci fixed and swaps the molecule; this one holds
the SEPARATION fixed and redraws the locus, which is what makes a within-cell
null usable at long range.  These tests drive the extension directly with
hand-built CSR inputs, so they check the contract that matters: the synthetic
pair spans the observed distance, both ends come from the anchor's own cell,
the anchor stays inside its locality window, and the observed pair is never
handed back as its own null.
"""

from __future__ import annotations

import numpy as np
import pytest

pysam = pytest.importorskip("pysam")            # loads the bundled libhts
_CPP = pytest.importorskip("nome_hic_concordance._site_dist_core")


def _csr(rows):
    """rows = list of [(pos, state), ...] -> (pos, st, ptr) CSR arrays."""
    pos, st, ptr = [], [], [0]
    for r in rows:
        for p, s in r:
            pos.append(p), st.append(s)
        ptr.append(len(pos))
    return (np.asarray(pos, np.int32), np.asarray(st, np.int8),
            np.asarray(ptr, np.int32))


def _donors(mids, cells, chrom=0, n_sites=4):
    """One donor end per midpoint, each carrying n_sites GCH calls.

    Site states alternate 0/1 by end index so a pair of ends has a definite,
    checkable correlation; positions sit right at the end's midpoint so the
    site-pair separation equals the end separation.
    """
    order = np.lexsort((np.asarray(mids), np.zeros(len(mids), int)))
    mids = np.asarray(mids, np.int32)[order]
    cells = np.asarray(cells, np.int32)[order]
    g_rows = [[(int(m) + k, (i + k) % 2) for k in range(n_sites)]
              for i, m in enumerate(mids)]
    h_rows = [[] for _ in mids]
    d = {"chrom": np.full(len(mids), chrom, np.int32),
         "mid": mids, "cell": cells}
    d["g_pos"], d["g_st"], d["g_ptr"] = _csr(g_rows)
    d["h_pos"], d["h_st"], d["h_ptr"] = _csr(h_rows)
    return d


def _anchors(mid1s, mid2s, cells, chrom=0):
    n = len(mid1s)
    a = {"chrom": np.full(n, chrom, np.int32),
         "mid1": np.asarray(mid1s, np.int32),
         "mid2": np.asarray(mid2s, np.int32),
         "cell": np.asarray(cells, np.int32)}
    for tag in ("g1", "h1", "g2", "h2"):
        rows = [[(int(m), 1)] for m in mid1s] if tag.endswith("1") else \
               [[(int(m), 1)] for m in mid2s]
        if tag.startswith("h"):
            rows = [[] for _ in mid1s]
        a[f"{tag}_pos"], a[f"{tag}_st"], a[f"{tag}_ptr"] = _csr(rows)
    return a


_EDGES = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]


def _run(anchors, donors, **kw):
    kw.setdefault("anchor_window", 10_000_000)
    kw.setdefault("dist_tol", 100)
    kw.setdefault("n_cells", int(max(anchors["cell"]) + 1))
    return _CPP.same_cell_distance_null(anchors, donors, _EDGES, seed=7, **kw)


def test_places_a_null_pair_when_the_separation_exists():
    """Ends at 0, 5000, 10000 in one cell; an observed pair spanning 5000 has
    plenty of same-separation alternatives."""
    mids = [0, 5000, 10000, 15000, 20000]
    d = _donors(mids, [0] * len(mids))
    a = _anchors([0], [5000], [0])
    out = _run(a, d)
    assert out["n_anchors"] == 1
    assert out["n_eligible"] == 1, "a 5 kb separation is available"
    tot = sum(np.asarray(out["null_per_cell"][i]).sum() for i in range(3))
    assert tot > 0, "the null must accumulate site pairs"


def test_no_null_when_the_separation_is_unavailable():
    """Only two ends 100 bp apart cannot supply a 500 kb separation."""
    d = _donors([0, 100], [0, 0])
    a = _anchors([0], [500_000], [0])
    out = _run(a, d)
    assert out["n_eligible"] == 0


def test_partner_never_comes_from_another_cell():
    """Cell 0 has a lone end; cell 1 has the full ladder.  The cell-0 anchor
    must find nothing rather than borrow cell 1's ends."""
    mids = [0, 5000, 10000, 15000]
    d = _donors([0] + mids, [0] + [1] * len(mids))
    a = _anchors([0], [5000], [0])            # anchor belongs to cell 0
    out = _run(a, d)
    assert out["n_eligible"] == 0, "must not cross the cell boundary"


def test_anchor_window_confines_the_synthetic_pair():
    """A tight window admits only ends near the real anchor; widening it reaches
    a ladder a megabase away."""
    # 995 kb is the escape hatch: with a 1 kb window the only usable anchor is
    # the real one at 1.0 Mb, whose UPWARD partner (1.005 Mb) is the observed
    # pair and is rejected — so the null can only be placed downward.
    mids = [995_000, 1_000_000, 1_005_000, 1_010_000]
    d = _donors(mids, [0] * len(mids))
    a = _anchors([1_000_000], [1_005_000], [0])
    assert _run(a, d, anchor_window=1000)["n_eligible"] >= 1, \
        "the downward partner at 995 kb is a valid non-observed pair"

    # An anchor at 0 spanning 5 kb: no end lies within 1 kb of 0, so the window
    # blocks it entirely; widening it reaches the distant ladder.
    a2 = _anchors([0], [5000], [0])
    assert _run(a2, d, anchor_window=1000)["n_eligible"] == 0
    assert _run(a2, d, anchor_window=2_000_000)["n_eligible"] >= 1


def test_zero_window_removes_the_locality_restriction():
    mids = [0, 1_000_000, 1_005_000]
    d = _donors(mids, [0] * len(mids))
    a = _anchors([0], [5000], [0])
    assert _run(a, d, anchor_window=0)["n_eligible"] >= 1


def test_observed_pair_is_rejected_as_its_own_null():
    """Exactly two ends, and the observed pair uses both: the only candidate
    partner IS the true partner, so nothing may be placed."""
    d = _donors([0, 5000], [0, 0])
    a = _anchors([0], [5000], [0])
    out = _run(a, d)
    assert out["n_eligible"] == 0, "the real pair cannot serve as its own null"


def test_dist_tol_controls_how_exact_the_match_must_be():
    """Ends at 0 and 5300 can stand in for a 5000 bp pair only if the
    tolerance admits the 300 bp discrepancy."""
    d = _donors([0, 5300, 20000], [0, 0, 0])
    a = _anchors([0], [5000], [0])
    assert _run(a, d, dist_tol=50)["n_eligible"] == 0
    assert _run(a, d, dist_tol=500)["n_eligible"] >= 1


def test_matched_arm_is_populated_alongside_the_null():
    mids = [0, 5000, 10000, 15000]
    d = _donors(mids, [0] * len(mids))
    a = _anchors([0], [5000], [0])
    out = _run(a, d)
    got = sum(np.asarray(out["matched_per_cell"][i]).sum() for i in range(3))
    assert got > 0, "MATCHED must accumulate exactly when the null does"


def test_driver_routes_null_type_and_forwards_mid1(tmp_path):
    """The script's _pooled_null must pass mid1 through and dispatch on
    --null-type; a reservoir lacking mid1 must fail loudly, not silently fall
    back to the locus-matched null."""
    import importlib.util
    import sys as _sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "sdc_dm", Path(__file__).resolve().parent.parent / "scripts"
        / "site_distance_correlation.py")
    m = importlib.util.module_from_spec(spec)
    _sys.modules["sdc_dm"] = m
    spec.loader.exec_module(m)

    mids = [0, 5000, 10000, 15000]
    d = _donors(mids, [0] * len(mids))
    a = _anchors([0], [5000], [0])
    # One cell's reservoir: (donor_parts, anchor_parts, stats)
    part_d = {k: d[k] for k in d if k != "cell"}
    part_a = {k: a[k] for k in a if k != "cell"}
    res = {0: (None, 0, 0, (part_d, part_a, {"anchor_long": 0,
                                             "pairs_long_seen": 0}))}

    class Args:
        null_window = 100
        null_anchor_window = 10_000_000
        max_site_pairs_per_read = 64
        seed = 7
        null_type = "distance-matched"

    out = m._pooled_null(res, [0], _EDGES, Args(), len(_EDGES) - 1)
    assert out is not None and out["n_eligible"] >= 1

    # Same call without mid1 in the reservoir must exit, not silently switch.
    part_a_no_mid1 = {k: v for k, v in part_a.items() if k != "mid1"}
    res2 = {0: (None, 0, 0, (part_d, part_a_no_mid1,
                             {"anchor_long": 0, "pairs_long_seen": 0}))}
    with pytest.raises(SystemExit, match="mid1"):
        m._pooled_null(res2, [0], _EDGES, Args(), len(_EDGES) - 1)


def test_per_cell_accumulators_are_keyed_by_the_anchor_cell():
    mids = [0, 5000, 10000, 15000]
    d = _donors(list(mids) + list(mids), [0] * 4 + [1] * 4)
    a = _anchors([0, 0], [5000, 5000], [0, 1])
    out = _run(a, d, n_cells=2)
    # 3 measures x 2 cells, laid out as 3*cell + measure.
    assert len(out["null_per_cell"]) == 6
    c0 = sum(np.asarray(out["null_per_cell"][i]).sum() for i in range(3))
    c1 = sum(np.asarray(out["null_per_cell"][3 + i]).sum() for i in range(3))
    assert c0 > 0 and c1 > 0, "both cells must contribute their own rows"
