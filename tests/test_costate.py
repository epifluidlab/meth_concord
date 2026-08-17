"""Cross-mark co-state analysis.

The identities this module is built around are exact, so they are checked
directly: coordination cannot prefer one co-state over the other, and any
observed direction is the marginal level.  Everything else here exists because
of that, and the tests are written to fail if a level confound creeps back in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import costate as cs


def test_the_two_identities_that_shape_the_module():
    """pi_open - pi_closed = p1 + p2 - 1, and the two excesses are equal."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        p11, p10, p01, p00 = rng.dirichlet([2, 2, 2, 2])
        m1, m2 = p11 + p10, p11 + p01
        assert p11 - p00 == pytest.approx(m1 + m2 - 1, abs=1e-12)
        excess_open = p11 - m1 * m2
        excess_closed = p00 - (1 - m1) * (1 - m2)
        assert excess_open == pytest.approx(excess_closed, abs=1e-12)


# ------------------------------------------------------------- classification --

def test_codes_need_coordination_before_a_direction():
    C = np.array([[1.0, 1.0, -1.0, 0.0]])
    F = np.array([[0.9, 0.1, 0.9, 0.9]])
    got = cs.costate_codes(C, F)[0]
    assert list(got) == [cs.CO_HIGH, cs.CO_LOW, cs.CO_MIXED, cs.CO_MIXED], (
        "F alone must not assign a side when there is no coordination")


def test_codes_mark_missing_data_unusable():
    C = np.array([[np.nan, 1.0]])
    F = np.array([[0.9, np.nan]])
    assert list(cs.costate_codes(C, F)[0]) == [cs.UNUSABLE, cs.UNUSABLE]


def test_min_coordination_raises_the_bar():
    C = np.array([[0.05, 1.0]])
    F = np.array([[0.9, 0.9]])
    loose = cs.costate_codes(C, F, min_coordination=0.0)[0]
    strict = cs.costate_codes(C, F, min_coordination=0.5)[0]
    assert list(loose) == [cs.CO_HIGH, cs.CO_HIGH]
    assert list(strict) == [cs.CO_MIXED, cs.CO_HIGH]


# ------------------------------------------------------------- stratification --

def _codes_from_pairs(pairs_per_bin):
    """pairs_per_bin[b] = list of (gch_code, hcg_code) for that bin's cells."""
    n_bins = len(pairs_per_bin)
    n_cells = len(pairs_per_bin[0])
    g = np.zeros((n_cells, n_bins), np.int8)
    h = np.zeros((n_cells, n_bins), np.int8)
    for b, pairs in enumerate(pairs_per_bin):
        for c, (gc, hc) in enumerate(pairs):
            g[c, b], h[c, b] = gc, hc
    return g, h


def test_stratified_counts_are_one_table_per_bin():
    g, h = _codes_from_pairs([
        [(cs.CO_HIGH, cs.CO_LOW)] * 3 + [(cs.CO_LOW, cs.CO_HIGH)] * 2,
        [(cs.CO_MIXED, cs.CO_MIXED)] * 5,
    ])
    tab = cs.stratified_counts(g, h)
    assert tab.shape == (2, 3, 3)
    assert tab[0, cs.CO_HIGH, cs.CO_LOW] == 3
    assert tab[0, cs.CO_LOW, cs.CO_HIGH] == 2
    assert tab[1, cs.CO_MIXED, cs.CO_MIXED] == 5
    assert tab.sum() == 10


def test_unusable_observations_are_counted_nowhere():
    g = np.array([[cs.UNUSABLE], [cs.CO_HIGH]], np.int8)
    h = np.array([[cs.CO_LOW], [cs.CO_LOW]], np.int8)
    assert cs.stratified_counts(g, h).sum() == 1


def test_mixed_is_dropped_from_the_two_by_two_core():
    """An observation with no coordination has no direction to contribute."""
    g, h = _codes_from_pairs([
        [(cs.CO_HIGH, cs.CO_LOW)] * 10 + [(cs.CO_MIXED, cs.CO_MIXED)] * 990])
    tab = cs.stratified_counts(g, h)
    a, b, c, d = cs._abcd(tab)
    assert (a[0], b[0], c[0], d[0]) == (10, 0, 0, 0)


# --------------------------------------------------------- the level confound --

def _confounded(n_cells=60, n_bins=40, seed=0, within_or=1.0):
    """Bins differ wildly in level; WITHIN a bin the two marks may or may not
    be associated.  The crude OR sees the level, the stratified one does not.

    Half the bins are 'open' bins where most cells are co-open in GCH and most
    are co-unmethylated in HCG; the other half are the mirror image.  That
    alone produces a huge crude odds ratio with no within-bin association at
    all -- which is exactly the artifact the stratification removes.
    """
    rng = np.random.default_rng(seed)
    g = np.zeros((n_cells, n_bins), np.int8)
    h = np.zeros((n_cells, n_bins), np.int8)
    for b in range(n_bins):
        open_bin = b < n_bins // 2
        pg = 0.9 if open_bin else 0.1          # P(cell is co-open here)
        ph = 0.1 if open_bin else 0.9          # P(cell is co-methylated here)
        gc = rng.random(n_cells) < pg
        # within_or > 1 couples the two marks WITHIN the bin, on top of level
        tilt = np.where(gc, 1.0 / within_or, within_or)
        hc = rng.random(n_cells) < np.clip(ph * tilt, 0, 1)
        g[:, b] = np.where(gc, cs.CO_HIGH, cs.CO_LOW)
        h[:, b] = np.where(hc, cs.CO_HIGH, cs.CO_LOW)
    return g, h


def test_crude_odds_ratio_is_driven_by_level_and_mh_is_not():
    """The headline result: with NO within-bin association, the crude OR is
    enormous and the stratified one sits at 1."""
    g, h = _confounded(within_or=1.0)
    tab = cs.stratified_counts(g, h)
    crude = cs.odds_ratio(tab)
    mh = cs.mantel_haenszel(tab)
    assert crude > 20, f"level alone should inflate the crude OR ({crude})"
    assert 0.8 < mh < 1.25, f"stratified must find nothing ({mh})"


def test_mh_still_finds_a_real_within_bin_association():
    g, h = _confounded(within_or=2.5, seed=3)
    mh = cs.mantel_haenszel(cs.stratified_counts(g, h))
    assert mh > 1.5, mh


def test_per_stratum_log_or_survives_an_empty_cell():
    g, h = _codes_from_pairs([[(cs.CO_HIGH, cs.CO_LOW)] * 5])
    lor = cs.per_stratum_log_or(cs.stratified_counts(g, h))
    assert np.isfinite(lor[0]) and lor[0] > 0, "Haldane keeps the stratum"


def test_a_bin_with_no_usable_observations_is_dropped_not_counted():
    g = np.full((4, 2), cs.CO_MIXED, np.int8)
    h = np.full((4, 2), cs.CO_MIXED, np.int8)
    lor = cs.per_stratum_log_or(cs.stratified_counts(g, h))
    assert np.isnan(lor).all()


# ------------------------------------------------------------------ bootstrap --

def test_bootstrap_resamples_cells_and_brackets_the_estimate():
    g, h = _confounded(within_or=2.5, seed=1)
    tab = cs.stratified_counts(g, h)
    mh = cs.mantel_haenszel(tab)
    lo, hi, p = cs.bootstrap_mh(g, h, n_boot=120, seed=0)
    assert lo < mh < hi
    assert lo > 1.0 and p < 0.05, "a real association should exclude OR = 1"


def test_bootstrap_says_nothing_when_there_is_nothing():
    g, h = _confounded(within_or=1.0, seed=2)
    lo, hi, p = cs.bootstrap_mh(g, h, n_boot=120, seed=0)
    assert lo < 1.0 < hi and p > 0.05


def test_bootstrap_needs_enough_cells():
    g, h = _confounded(n_cells=2, n_bins=4)
    assert cs.bootstrap_mh(g, h, n_boot=50) is None


# ----------------------------------------------------------------- level grid --

def test_level_grid_uses_quantile_bands_and_summarises_each_cell():
    rng = np.random.default_rng(0)
    nb = 900
    x = rng.uniform(0, 1, nb)
    y = rng.uniform(0, 1, nb)
    # Coordination is strongest where x is high and y is low.
    c = 2.0 * x - 2.0 * y + rng.normal(0, 0.1, nb)
    grid, xe, ye = cs.level_grid(x, y, {"C": c}, n_bins=3)
    assert len(grid) == 9
    assert grid["n_bins"].sum() == nb
    assert len(xe) == 2 and len(ye) == 2, "interior cuts only"
    # Bands are roughly equal in size, being quantiles.
    assert grid["n_bins"].min() > 0.5 * grid["n_bins"].max()
    hi = grid[(grid.x_band == "Q3") & (grid.y_band == "Q1")]["mean_C"].iloc[0]
    lo = grid[(grid.x_band == "Q1") & (grid.y_band == "Q3")]["mean_C"].iloc[0]
    assert hi > lo + 1.5
    assert grid["sem_C"].notna().all()


def test_level_grid_labels_can_be_named():
    x = np.linspace(0, 1, 300)
    y = np.linspace(1, 0, 300)
    grid, _, _ = cs.level_grid(x, y, {"C": x}, n_bins=3,
                               labels=["low", "mid", "high"])
    assert set(grid["x_band"]) == {"low", "mid", "high"}


def test_level_grid_refuses_a_grid_it_cannot_fill():
    with pytest.raises(ValueError, match="usable bins"):
        cs.level_grid(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                      {"C": np.array([1.0, 2.0])}, n_bins=3)


def test_batch_stratification_splits_the_strata_further():
    g, h = _confounded(n_cells=60, n_bins=10, within_or=1.0)
    batch = np.repeat([0, 1], 30)
    plain = cs.stratified_counts(g, h)
    split = cs.stratified_counts(g, h, batch_codes=batch)
    assert plain.shape == (10, 3, 3)
    assert split.shape == (20, 3, 3), "one stratum per (bin, batch)"
    assert plain.sum() == split.sum(), "every observation still counted once"


def test_batch_stratification_removes_a_batch_driven_association():
    """Batch is a CELL property, so it survives stratifying by bin alone."""
    rng = np.random.default_rng(0)
    n_cells, n_bins = 80, 30
    batch = np.repeat([0, 1], n_cells // 2)
    g = np.zeros((n_cells, n_bins), np.int8)
    h = np.zeros((n_cells, n_bins), np.int8)
    for b in range(n_bins):
        # No within-bin biology at all: batch only shifts each mark's OWN
        # probability, and the two draws are independent given the batch.  So
        # inside a (bin, batch) stratum there is nothing to find, while pooling
        # the batches makes the two marks look strongly coupled.
        pg = np.where(batch == 0, 0.85, 0.15)
        ph = np.where(batch == 0, 0.15, 0.85)
        g[:, b] = np.where(rng.random(n_cells) < pg, cs.CO_HIGH, cs.CO_LOW)
        h[:, b] = np.where(rng.random(n_cells) < ph, cs.CO_HIGH, cs.CO_LOW)
    by_bin = cs.mantel_haenszel(cs.stratified_counts(g, h))
    by_bin_batch = cs.mantel_haenszel(
        cs.stratified_counts(g, h, batch_codes=batch))
    assert by_bin > 5, f"batch alone drives the bin-stratified OR ({by_bin})"
    assert 0.5 < by_bin_batch < 2.0, (
        f"stratifying by batch too must remove it ({by_bin_batch})")


def test_bootstrap_ors_returns_both_from_one_set_of_resamples():
    g, h = _confounded(within_or=2.5, seed=1)
    got = cs.bootstrap_ors(g, h, n_boot=120, seed=0)
    assert set(got) == {"mh", "crude"}
    mh_lo, mh_hi, _ = got["mh"]
    cr_lo, cr_hi, _ = got["crude"]
    assert mh_hi < cr_lo, "the crude OR is inflated by level, and separably so"
    # bootstrap_mh is the same numbers, so the two cannot drift.
    assert cs.bootstrap_mh(g, h, n_boot=120, seed=0) == got["mh"]


def test_a_non_estimable_stratification_says_so_rather_than_inventing_one():
    """When a stratum has no variation left in one mark there is no odds ratio
    in it, and the pooled estimate must come back NaN rather than 1.0 -- the
    shape of a perfectly confounded design, which is worth seeing."""
    n_cells, n_bins = 40, 5
    batch = np.repeat([0, 1], n_cells // 2)
    hi = batch == 0
    g = np.tile(np.where(hi, cs.CO_HIGH, cs.CO_LOW)[:, None], (1, n_bins))
    h = np.tile(np.where(hi, cs.CO_LOW, cs.CO_HIGH)[:, None], (1, n_bins))
    assert cs.mantel_haenszel(cs.stratified_counts(g, h)) > 5
    assert np.isnan(cs.mantel_haenszel(
        cs.stratified_counts(g.astype(np.int8), h.astype(np.int8),
                             batch_codes=batch)))
