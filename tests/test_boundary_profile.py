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


# ------------------------------------------------------------- profile ----

def test_the_profile_is_a_share_of_the_universe_not_a_count():
    """Counting would draw the distribution of domain lengths, which appears
    identically in every set and reads as biology."""
    m = _mod()
    dist = np.array([-300.0, -100.0, 100.0, 300.0, 100.0, 300.0])
    in_set = np.array([True, True, False, True, False, False])
    uni = np.ones(6, bool)
    x, dens, n = m.profile(dist, in_set, uni, flank=400, step=200)
    assert list(x) == [-300.0, -100.0, 100.0, 300.0]
    assert dens[1] == pytest.approx(1.0)      # one bin, in the set
    assert dens[2] == pytest.approx(0.0)      # two bins, neither in the set
    assert dens[3] == pytest.approx(0.5)      # two bins, one in the set
    assert list(n) == [1, 1, 2, 2]


def test_offsets_with_no_bins_are_nan_not_zero():
    """Zero would draw a gap as a real absence of the set."""
    m = _mod()
    dist = np.array([100.0])
    x, dens, n = m.profile(dist, np.array([True]), np.ones(1, bool), 400, 200)
    assert n[0] == 0 and np.isnan(dens[0])


# ---------------------------------------------------------------- test ----

def test_edge_test_builds_the_two_by_two_on_proximity():
    """Near means |distance| < near on EITHER side: a boundary effect is
    about proximity to the edge, not which side of it a bin sits on."""
    m = _mod()
    dist = np.array([-50.0, 50.0, -500.0, 500.0, -500.0, 500.0])
    in_set = np.array([True, True, False, False, True, False])
    st = m.edge_test(dist, in_set, np.ones(6, bool), near=100)
    assert (st["n_near_in_set"], st["n_near_out"]) == (2, 0)
    assert (st["n_far_in_set"], st["n_far_out"]) == (1, 3)
    assert st["frac_near"] == pytest.approx(1.0)
    assert st["frac_far"] == pytest.approx(0.25)


def test_a_uniform_set_gives_an_odds_ratio_near_one():
    m = _mod()
    rng = np.random.default_rng(0)
    dist = rng.uniform(-5e6, 5e6, 4000)
    in_set = rng.random(4000) < 0.2
    st = m.edge_test(dist, in_set, np.ones(4000, bool), near=1e6)
    assert 0.7 < st["odds_ratio"] < 1.4
    assert st["pvalue"] > 0.01
