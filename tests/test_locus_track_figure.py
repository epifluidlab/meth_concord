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
    grid, out = m.smooth(vals, starts, 10_000, 250_000, bin_bp=250_000)
    assert grid[1] - grid[0] == 10_000
    assert np.nanmax(out) <= 1.0 + 1e-9
    assert np.nanmin(out) >= 0.0 - 1e-9
    # Centred on the join between the two 1.0 bins, a 250 kb window sees only
    # those two.  A point 130 kb earlier straddles the 0.0 bin below and
    # correctly reads the average -- the window is 250 kb WIDE, not 250 kb
    # each way.
    assert out[np.argmin(np.abs(grid - 500_000))] == pytest.approx(1.0)
    assert out[np.argmin(np.abs(grid - 370_000))] == pytest.approx(0.5)


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


# ------------------------------------------------------- colour limits -----

def test_limits_follow_the_data_when_it_does_not_straddle_zero():
    """D over one locus often sits in a narrow positive band; a forced
    -|max|..+|max| scale then spends half the range on values that do not
    exist and paints every cell alike."""
    m = _mod()
    v = np.random.default_rng(0).normal(0.55, 0.05, 5000)
    lo, hi = m.robust_limits(v)
    assert lo > 0.3, f"lower limit {lo} should track the data, not zero"
    assert hi < 0.8
    assert hi - lo < 0.5


def test_limits_stay_symmetric_when_the_data_spans_both_signs():
    """There zero is a meaningful midpoint and the diverging map should have
    its neutral colour on it."""
    m = _mod()
    v = np.concatenate([np.full(500, -0.8), np.full(500, 0.4)])
    lo, hi = m.robust_limits(v)
    assert lo == pytest.approx(-hi)


def test_explicit_limits_win():
    m = _mod()
    assert m.robust_limits(np.array([0.0, 1.0]), vmin=-2, vmax=3) == (-2, 3)


def test_limits_survive_an_all_nan_input():
    m = _mod()
    lo, hi = m.robust_limits(np.array([np.nan, np.nan]))
    assert np.isfinite([lo, hi]).all() and hi > lo


def test_limits_never_collapse_to_a_point():
    """A constant panel would otherwise give vmin == vmax and imshow would
    raise rather than draw a flat colour."""
    m = _mod()
    lo, hi = m.robust_limits(np.full(100, 0.5))
    assert hi > lo


def test_min_gap_rejects_a_run_that_only_just_qualifies():
    """The ordering alone is weak: arms differing by 0.0001 satisfy 'every low
    below every high' and are indistinguishable from noise.  Ranking is by
    total gap, so a real locus still comes first -- the trivial runs are what
    fill positions 2..N when the top N are asked for, and --min-gap is what
    keeps them out."""
    m = _mod()
    b = _bins(8)
    score = np.array([0.0001, 0.0001, 0.0001, -1, 0.9, 0.9, -1, -1])
    strong = m.best_run(b, score, min_bins=2)
    assert (strong["start"], strong["n"]) == (4 * 250_000, 2)

    # With the real locus taken, the trivial run is what the next request
    # returns -- unless a floor excludes it.
    rest = score.copy()
    rest[4:6] = np.nan
    assert m.best_run(b, rest, min_bins=2)["n"] == 3
    assert m.best_run(b, rest, min_bins=2, min_gap=0.01) is None


def test_min_gap_can_reject_everything():
    m = _mod()
    b = _bins(5)
    assert m.best_run(b, np.full(5, 0.01), min_bins=2, min_gap=0.5) is None


# ------------------------------------------------------- mean and band -----

def _sub(vals_by_group, n_bins=3):
    cells, rows, ct = [], [], []
    for g, vals in vals_by_group.items():
        for i, v in enumerate(vals):
            cells.append(f"{g}_{i}")
            rows.append(np.full(n_bins, float(v)))
            ct.append(g)
    M = pd.DataFrame(rows, index=cells,
                     columns=[f"chr1:{i * 250_000}" for i in range(n_bins)])
    return M, pd.Series(ct, index=cells)


def test_group_mean_ci_gives_the_mean_and_a_band_around_it():
    m = _mod()
    M, ct = _sub({"a": [0.4, 0.5, 0.6], "b": [0.9, 1.0, 1.1]})
    got = m.group_mean_ci(M, ct, ["a", "b"])
    assert got["a"]["mean"][0] == pytest.approx(0.5)
    assert got["b"]["mean"][0] == pytest.approx(1.0)
    assert got["a"]["lo"][0] < 0.5 < got["a"]["hi"][0]


def test_the_band_uses_t_not_a_normal_approximation():
    """With a dozen cells the normal interval is visibly too narrow, and the
    band exists to show when two groups are NOT resolved."""
    m = _mod()
    M, ct = _sub({"a": [0.0, 1.0, 2.0]})
    got = m.group_mean_ci(M, ct, ["a"])
    half = got["a"]["hi"][0] - got["a"]["mean"][0]
    se = 1.0 / np.sqrt(3)                        # sd = 1 with ddof=1
    assert half > 1.96 * se, "t is wider than the normal at n=3"
    assert half == pytest.approx(4.302653 * se, rel=1e-4)   # t(0.975, df=2)


def test_the_band_widens_where_coverage_thins():
    """n is counted per BIN: a band drawn on the group's total cell count
    would be too tight exactly where the data is thinnest."""
    m = _mod()
    M, ct = _sub({"a": [0.2, 0.4, 0.6, 0.8]}, n_bins=2)
    M.iloc[1:, 1] = np.nan                       # bin 1 keeps one cell
    got = m.group_mean_ci(M, ct, ["a"])
    assert got["a"]["n"][0] == 4 and got["a"]["n"][1] == 1
    assert np.isfinite(got["a"]["hi"][0])
    assert np.isnan(got["a"]["hi"][1]), "one cell cannot carry an interval"


def test_a_group_with_no_cells_is_skipped():
    m = _mod()
    M, ct = _sub({"a": [1.0, 2.0]})
    assert list(m.group_mean_ci(M, ct, ["a", "absent"])) == ["a"]


def test_significant_spans_merges_adjacent_bins():
    m = _mod()
    starts = np.arange(6) * 250_000
    sig = np.array([False, True, True, False, True, False])
    got = m.significant_spans(starts, sig, 250_000)
    assert got == [(250_000.0, 750_000.0), (1_000_000.0, 1_250_000.0)]


def test_significant_spans_closes_a_run_at_the_end():
    m = _mod()
    got = m.significant_spans([0, 100], [False, True], 100)
    assert got == [(100.0, 200.0)]


def test_no_significant_bins_gives_no_spans():
    m = _mod()
    assert m.significant_spans([0, 100], [False, False], 100) == []


# --------------------------------------------------------- ranking ---------

def test_all_runs_reports_the_statistics_needed_to_rank():
    m = _mod()
    b = _bins(8)
    score = np.array([-1, 0.5, 0.5, 0.5, -1, 0.2, 0.2, -1])
    fdr = np.array([1, 1e-4, 1e-4, 1e-2, 1, 1e-9, 1e-9, 1], float)
    runs = m.all_runs(b, score, min_bins=2, fdr=fdr)
    assert len(runs) == 2
    a, c = runs
    assert (a["n"], a["mean_gap"], a["total"]) == (3, pytest.approx(0.5),
                                                   pytest.approx(1.5))
    assert a["median_fdr"] == pytest.approx(1e-4)
    assert c["n"] == 2 and c["mean_gap"] == pytest.approx(0.2)


def test_gap_mean_puts_the_most_different_locus_first():
    """Narrow and strong beats wide and modest -- which is what "the most
    different locus" means, and what summing over the run gets backwards."""
    m = _mod()
    runs = [{"chrom": "chr1", "start": 0, "end": 1, "n": 14, "total": 2.0,
             "mean_gap": 0.143, "min_gap": 0.13, "median_fdr": 1e-12,
             "evidence": 12.0},
            {"chrom": "chr1", "start": 9, "end": 10, "n": 3, "total": 1.6,
             "mean_gap": 0.538, "min_gap": 0.53, "median_fdr": 0.04,
             "evidence": 1.4}]
    assert m.rank_runs(runs, "gap-mean")[0]["n"] == 3
    assert m.rank_runs(runs, "gap-total")[0]["n"] == 14


def test_combined_lets_evidence_outweigh_a_slightly_larger_gap():
    m = _mod()
    runs = [{"n": 14, "total": 2.0, "mean_gap": 0.143, "evidence": 12.0},
            {"n": 3, "total": 1.6, "mean_gap": 0.538, "evidence": 1.4}]
    #  0.143*12 = 1.72  vs  0.538*1.4 = 0.75
    assert m.rank_runs(runs, "combined")[0]["n"] == 14


def test_combined_falls_back_when_no_fdr_is_available():
    m = _mod()
    runs = [{"n": 2, "mean_gap": 0.4, "total": 0.8, "evidence": float("nan")},
            {"n": 2, "mean_gap": 0.9, "total": 1.8, "evidence": float("nan")}]
    assert m.rank_runs(runs, "combined")[0]["mean_gap"] == pytest.approx(0.9)


def test_an_unknown_ranking_is_refused():
    m = _mod()
    with pytest.raises(ValueError, match="unknown ranking"):
        m.rank_runs([], "whatever")


def test_all_runs_honours_min_gap_and_min_bins():
    m = _mod()
    b = _bins(8)
    score = np.array([0.01, 0.01, 0.01, -1, 0.9, 0.9, -1, -1])
    assert len(m.all_runs(b, score, min_bins=2)) == 2
    assert len(m.all_runs(b, score, min_bins=2, min_gap=0.1)) == 1
    assert len(m.all_runs(b, score, min_bins=4, min_gap=0.1)) == 0


def test_all_runs_does_not_join_across_a_chromosome():
    m = _mod()
    b = pd.concat([_bins(3, "chr1"), _bins(3, "chr2")], ignore_index=True)
    runs = m.all_runs(b, np.ones(6), min_bins=2)
    assert len(runs) == 2 and {r["chrom"] for r in runs} == {"chr1", "chr2"}


def test_the_window_is_its_stated_width_not_twice_it():
    """--window is the FULL width of a centred window.  Treating it as the
    bin's extent too made the window reach that far EACH WAY, so a '250 kb
    window' smoothed over 500 kb."""
    m = _mod()
    starts = np.arange(9) * 250_000
    vals = np.array([0.6] * 4 + [0.0] * 5)
    for win in (250_000, 500_000, 1_000_000):
        _, out = m.smooth(vals, starts, 10_000, win, bin_bp=250_000)
        ramp = np.sum((out > 0.02) & (out < 0.58)) * 10_000
        assert abs(ramp - win) <= 20_000, f"{win}: ramp came out {ramp}"


def test_a_window_equal_to_the_bin_barely_smooths():
    """Which is why a 250 kb window over 250 kb bins looks unsmoothed: each
    point averages one or two bins."""
    m = _mod()
    starts = np.arange(10) * 250_000
    vals = np.array([0.6] * 5 + [0.0] * 5)
    _, narrow = m.smooth(vals, starts, 10_000, 250_000, bin_bp=250_000)
    _, wide = m.smooth(vals, starts, 10_000, 1_000_000, bin_bp=250_000)
    assert len(np.unique(np.round(narrow, 4))) <= 4
    assert len(np.unique(np.round(wide, 4))) >= 8


def test_the_bin_width_is_inferred_when_not_given():
    m = _mod()
    starts = np.arange(6) * 100_000
    vals = np.arange(6, dtype=float)
    a = m.smooth(vals, starts, 10_000, 300_000)[1]
    b = m.smooth(vals, starts, 10_000, 300_000, bin_bp=100_000)[1]
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_smoothing_preserves_the_plateau_values():
    """It interpolates between bins; it must not move a level that is flat
    across the window."""
    m = _mod()
    starts = np.arange(12) * 250_000
    vals = np.full(12, 0.42)
    _, out = m.smooth(vals, starts, 10_000, 1_000_000, bin_bp=250_000)
    ok = np.isfinite(out)
    np.testing.assert_allclose(out[ok], 0.42)
