"""Genome-browser panel for one locus."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "locus_track_figure.py")


def _mod():
    spec = importlib.util.spec_from_file_location("ltf", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ltf"] = m
    spec.loader.exec_module(m)
    return m


def _bins(n, chrom="chr1", step=250_000):
    return pd.DataFrame({"chrom": [chrom] * n,
                         "start": np.arange(n) * step,
                         "end": (np.arange(n) + 1) * step})


# ---------------------------------------------------------- separation -----

def test_a_bin_scores_only_when_every_low_group_is_below_every_high_one():
    """An average of the two sides would score a bin one tumour population
    drives alone -- which is the thing this figure would be read as ruling
    out."""
    m = _mod()
    means = {"t1": pd.Series([0.1, 0.1, 0.9]),
             "t2": pd.Series([0.2, 0.9, 0.2]),
             "ref": pd.Series([0.8, 0.8, 0.8])}
    got = m.separation_score(means, ["t1", "t2"], ["ref"])
    assert got.iloc[0] == pytest.approx(0.6)      # both low
    assert got.iloc[1] < 0                        # t2 is not low
    assert got.iloc[2] < 0                        # t1 is not low


def test_separation_takes_the_worst_high_group_not_their_mean():
    m = _mod()
    means = {"t": pd.Series([0.2]),
             "a": pd.Series([0.9]), "b": pd.Series([0.25])}
    got = m.separation_score(means, ["t"], ["a", "b"])
    assert got.iloc[0] == pytest.approx(0.05)     # against b, not (a+b)/2


def test_separation_needs_a_group_on_each_side():
    m = _mod()
    with pytest.raises(SystemExit, match="each side"):
        m.separation_score({"t": pd.Series([1.0])}, ["t"], ["missing"])


# ----------------------------------------------------------------- runs ----

def test_best_run_finds_the_longest_strongest_contiguous_stretch():
    m = _mod()
    b = _bins(10)
    score = np.array([-1, 2, 3, 4, -1, 9, -1, -1, -1, -1], float)
    run = m.best_run(b, score, min_bins=2)
    assert (run["start"], run["n"]) == (250_000, 3)
    assert run["total"] == pytest.approx(9.0)


def test_a_run_cannot_span_two_chromosomes():
    """Adjacent in the table is not adjacent in the genome."""
    m = _mod()
    b = pd.concat([_bins(3, "chr1"), _bins(3, "chr2")], ignore_index=True)
    run = m.best_run(b, np.ones(6), min_bins=2)
    assert run["n"] == 3
    assert run["chrom"] in ("chr1", "chr2")


def test_a_run_cannot_jump_a_gap_in_the_bins():
    m = _mod()
    b = _bins(6)
    b = b.drop(index=3).reset_index(drop=True)     # a missing bin
    run = m.best_run(b, np.ones(len(b)), min_bins=2)
    assert run["n"] == 3, "the run should stop at the gap"


def test_significance_can_be_required_as_well_as_separation():
    m = _mod()
    b = _bins(6)
    score = np.ones(6)
    require = np.array([False, True, True, False, False, False])
    run = m.best_run(b, score, min_bins=2, require=require)
    assert (run["start"], run["n"]) == (250_000, 2)


def test_no_qualifying_run_returns_none():
    m = _mod()
    assert m.best_run(_bins(4), np.full(4, -1.0), min_bins=2) is None


# ------------------------------------------------------------ smoothing ----

def test_smoothing_is_a_rolling_mean_over_the_existing_bins():
    """It cannot sharpen a boundary the binning already blurred; the point is
    a continuous track, not added resolution."""
    m = _mod()
    starts = np.arange(4) * 250_000
    vals = np.array([0.0, 1.0, 1.0, 0.0])
    grid, out = m.smooth(vals, starts, 10_000, 250_000)
    assert grid[1] - grid[0] == 10_000
    assert np.nanmax(out) <= 1.0 + 1e-9
    assert np.nanmin(out) >= 0.0 - 1e-9
    # the middle of the block averages only the two 1.0 bins
    mid = out[np.argmin(np.abs(grid - 375_000))]
    assert mid == pytest.approx(1.0)


def test_smoothing_ignores_nan_bins():
    m = _mod()
    starts = np.arange(3) * 250_000
    grid, out = m.smooth(np.array([1.0, np.nan, 1.0]), starts, 50_000,
                         250_000)
    assert np.isfinite(out).any()
    assert np.nanmax(out) == pytest.approx(1.0)


# ------------------------------------------------------------ gene rows ----

def test_genes_are_packed_into_rows_without_overlapping():
    m = _mod()
    g = pd.DataFrame({"chrom": ["chr1"] * 3,
                      "start": [0, 10, 10_000],
                      "end": [100, 110, 10_100],
                      "name": ["a", "b", "c"]})
    got = m._gene_rows(g, "chr1", 0, 20_000)
    assert len(got) == 3
    assert got.loc[0, "row"] != got.loc[1, "row"], "a and b overlap"


def test_genes_off_the_window_are_dropped():
    m = _mod()
    g = pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [0, 0],
                      "end": [100, 100], "name": ["a", "b"]})
    assert list(m._gene_rows(g, "chr1", 0, 1000)["name"]) == ["a"]
