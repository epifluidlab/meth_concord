"""Density of bin sets around domain boundaries."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "boundary_profile.py")


def _mod():
    spec = importlib.util.spec_from_file_location("bp", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["bp"] = m
    spec.loader.exec_module(m)
    return m


def _bed(path, rows, header=True):
    df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    df.to_csv(path, sep="\t", index=False, header=header)
    return str(path)


def _grid(n, chrom="chr1", step=250_000):
    return pd.DataFrame({"chrom": [chrom] * n,
                         "start": np.arange(n) * step,
                         "end": (np.arange(n) + 1) * step})


# ------------------------------------------------------------- reading -----

def test_a_bin_grid_is_not_merged(tmp_path):
    """Every bin abuts its neighbour, so merging collapses a chromosome into
    one interval -- silently, leaving a universe of one row per contig."""
    m = _mod()
    p = _bed(tmp_path / "u.bed", [("chr1", i * 250_000, (i + 1) * 250_000)
                                  for i in range(5)])
    assert len(m.read_bed(p)) == 5
    assert len(m.read_bed(p, merge=True)) == 1


def test_domains_are_merged_when_asked(tmp_path):
    """Touching domains are one domain and their shared edge is not a
    boundary."""
    m = _mod()
    p = _bed(tmp_path / "d.bed", [("chr1", 0, 100), ("chr1", 100, 200),
                                  ("chr1", 500, 600)])
    got = m.read_bed(p, merge=True)
    assert len(got) == 2
    assert list(got.end) == [200, 600]


def test_a_headerless_bed_is_read(tmp_path):
    m = _mod()
    p = _bed(tmp_path / "n.bed", [("chr1", 0, 100)], header=False)
    assert len(m.read_bed(p)) == 1


# ------------------------------------------------------------ distance ----

def test_distance_is_positive_inside_and_negative_outside():
    """Both edges of a domain are boundaries; folding them without flipping
    the sign at one cancels the asymmetry the plot exists for."""
    m = _mod()
    bins = _grid(8, step=100)      # centres 50, 150, ... 750
    dom = pd.DataFrame({"chrom": ["chr1"], "start": [300], "end": [500]})
    d = m.signed_distance(bins, dom)
    assert d[1] < 0, "centre 150 is outside"
    assert d[3] > 0, "centre 350 is inside"
    assert d[4] > 0, "centre 450 is inside"
    assert d[6] < 0, "centre 650 is outside"


def test_distance_is_to_the_nearest_edge():
    m = _mod()
    bins = pd.DataFrame({"chrom": ["chr1"], "start": [380], "end": [420]})
    dom = pd.DataFrame({"chrom": ["chr1"], "start": [300], "end": [500]})
    d = m.signed_distance(bins, dom)
    assert d[0] == pytest.approx(100.0)      # centre 400 -> 100 from each


def test_a_contig_without_domains_gets_nan():
    """A bin on a chromosome the domains never mention has no distance, and
    NaN keeps it out of the profile instead of placing it at zero."""
    m = _mod()
    bins = pd.concat([_grid(2, "chr1", 100), _grid(2, "chr9", 100)],
                     ignore_index=True)
    dom = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [200]})
    d = m.signed_distance(bins, dom)
    assert np.isfinite(d[:2]).all()
    assert np.isnan(d[2:]).all()


# ----------------------------------------------------------- occupancy ----

def _iv(rows):
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def test_occupancy_asks_whether_the_set_is_there_per_boundary():
    """One row per boundary edge, 1 where a bin of the set overlaps that
    offset window.  This is the denominator that makes a random set flat."""
    m = _mod()
    dom = _iv([("chr1", 1_000_000, 3_000_000)])
    # a single set bin sitting just inside the left edge
    bed = _iv([("chr1", 1_000_000, 1_250_000)])
    occ, centres = m.occupancy(dom, bed, flank=1_000_000, step=250_000)
    assert occ.shape == (2, 8)          # two edges, eight offsets
    left = occ[0]
    assert left[centres == 125_000][0]  # just inside the left edge
    assert not left[centres == -875_000][0]


def test_offsets_are_oriented_so_positive_is_inside():
    """For a right edge that means walking left; without the flip the two
    edges cancel."""
    m = _mod()
    dom = _iv([("chr1", 1_000_000, 3_000_000)])
    bed = _iv([("chr1", 2_750_000, 3_000_000)])   # just inside the RIGHT edge
    occ, centres = m.occupancy(dom, bed, flank=1_000_000, step=250_000)
    right = occ[1]
    assert right[centres == 125_000][0], "inside should be positive"
    assert not right[centres == -125_000][0]


def test_a_uniformly_placed_set_is_flat():
    """The whole point.  Histogramming a set's own bins by distance peaks at
    the boundary for ANY set, because the positions available at each
    distance are not uniform; occupancy per boundary does not."""
    m = _mod()
    step = 250_000
    dom = _iv([("chr1", i * 20_000_000, i * 20_000_000 + 4_000_000)
               for i in range(1, 9)])
    rng = np.random.default_rng(0)
    starts = np.sort(rng.choice(np.arange(0, 200_000_000, step),
                                size=400, replace=False))
    bed = _iv([("chr1", int(s), int(s) + step) for s in starts])
    occ, centres = m.occupancy(dom, bed, flank=5_000_000, step=step)
    dens, n = m.profile_from_occupancy(occ)
    assert n == 16
    # flat: the centre is no higher than the rest, within sampling noise
    near = dens[np.abs(centres) < 1_000_000].mean()
    far = dens[np.abs(centres) >= 3_000_000].mean()
    assert abs(near - far) < 0.12, f"near {near:.3f} vs far {far:.3f}"


def test_a_boundary_hugging_set_is_not_flat():
    m = _mod()
    step = 250_000
    # 12 Mb wide, so a domain's own far edge falls OUTSIDE the 5 Mb window
    # and the test measures boundary-hugging rather than domain width.
    dom = _iv([("chr1", i * 30_000_000, i * 30_000_000 + 12_000_000)
               for i in range(1, 7)])
    rows = []
    for i in range(1, 7):
        for edge in (i * 30_000_000, i * 30_000_000 + 12_000_000):
            # one bin each side of the edge, so both nearest offsets are hit
            rows.append(("chr1", edge - step, edge))
            rows.append(("chr1", edge, edge + step))
    occ, centres = m.occupancy(dom, _iv(rows), flank=5_000_000, step=step)
    dens, _ = m.profile_from_occupancy(occ)
    assert dens[np.abs(centres) < 300_000].mean() > 0.9
    assert dens[np.abs(centres) > 3_000_000].mean() < 0.1


def test_profile_from_empty_occupancy_is_nan():
    m = _mod()
    dens, n = m.profile_from_occupancy(np.zeros((0, 8), bool))
    assert n == 0 and np.isnan(dens).all()


def test_occupancy_ignores_a_chromosome_the_set_never_touches():
    m = _mod()
    dom = _iv([("chr9", 1_000_000, 3_000_000)])
    bed = _iv([("chr1", 1_000_000, 1_250_000)])
    occ, _ = m.occupancy(dom, bed, flank=1_000_000, step=250_000)
    assert occ.shape[0] == 2 and not occ.any()


# ---------------------------------------------------------------- test ----

def test_the_occupancy_test_compares_near_offsets_with_far_ones():
    m = _mod()
    centres = np.array([-750_000.0, -250_000.0, 250_000.0, 750_000.0])
    occ = np.array([[False, True, True, False],
                    [False, True, True, False],
                    [True, True, True, True]])
    st = m.occupancy_test(occ, centres, near=500_000)
    assert (st["n_occupied_near"], st["n_empty_near"]) == (6, 0)
    assert (st["n_occupied_far"], st["n_empty_far"]) == (2, 4)
    assert st["frac_near"] == pytest.approx(1.0)
    assert st["odds_ratio"] > 1


def test_a_flat_set_gives_an_odds_ratio_near_one():
    m = _mod()
    rng = np.random.default_rng(1)
    centres = np.linspace(-5e6, 5e6, 40)
    occ = rng.random((200, 40)) < 0.2
    st = m.occupancy_test(occ, centres, near=5e5)
    assert 0.7 < st["odds_ratio"] < 1.4
    assert st["pvalue"] > 0.05


def test_the_occupancy_test_on_nothing_returns_nothing():
    m = _mod()
    assert m.occupancy_test(np.zeros((0, 4), bool),
                            np.zeros(4), near=1.0) == {}
