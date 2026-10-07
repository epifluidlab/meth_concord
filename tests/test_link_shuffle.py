"""The link-shuffle donor draw shared by step 5 and ep_coordination_existence:
a contact keeps its left end; its partner is a contact end from ANOTHER cell
of the stratum, on the partner's side at a distance in the contact's own
bin."""

from __future__ import annotations

import numpy as np

from nome_hic_concordance import link_shuffle as ls


def _pool(spec):
    """{cell: [(pos, p), ...]} on chromosome code 0 -> pool."""
    return ls.build_pool({c: (np.array([q for q, _ in v], np.int64),
                              np.array([p for _, p in v], float))
                          for c, v in spec.items()})


def test_the_donor_is_another_cells_end_at_the_partners_distance():
    dbins = ls.log_dbins(20_000, 2_000_000, 4)
    pool = _pool({0: [(150_000, 0.9)],                # the anchor's own cell
                  1: [(149_000, 0.2), (900_000, 0.7)],
                  2: [(60_000, 0.4)]})                # left of the anchor
    own = np.array([150_000], np.int64)
    donor, level = ls.draw_link_shuffle(
        np.array([100_000]), np.array([50_000]), 0, pool, own, dbins,
        np.random.default_rng(0), n_draws=20)
    assert level.tolist() == [0]
    assert set(donor[:, 0]) == {0.2}       # cell 1's end, never cell 0's own


def test_a_window_widens_only_when_its_own_bin_is_empty():
    dbins = ls.log_dbins(20_000, 2_000_000, 4)
    pool = _pool({1: [(100_000 + 80_000, 0.3)]})      # one bin up from 50 kb
    donor, level = ls.draw_link_shuffle(
        np.array([100_000, 100_000]), np.array([50_000, 1_900_000]), 0,
        pool, np.empty(0, np.int64), dbins, np.random.default_rng(0),
        max_widen=1)
    assert level.tolist() == [1, -1]
    assert donor[0, 0] == 0.3 and np.isnan(donor[0, 1])


def test_bounds_keep_the_donor_where_the_partner_is():
    dbins = ls.log_dbins(1_000, 2_500_000, 4)
    pool = _pool({1: [(2_510_000, 0.8), (2_480_000, 0.1)]})
    donor, level = ls.draw_link_shuffle(
        np.array([2_470_000]), np.array([20_000]), 0, pool,
        np.empty(0, np.int64), dbins, np.random.default_rng(0),
        bounds=(np.array([0]), np.array([2_500_000])), max_widen=3)
    assert donor[0, 0] == 0.1               # 2.51 Mb is past the bin's end


def test_many_candidates_give_independent_draws():
    dbins = ls.log_dbins(20_000, 2_000_000, 4)
    rng = np.random.default_rng(1)
    pool = _pool({c: [(int(x), float(p)) for x, p in
                      zip(rng.integers(140_000, 160_000, 5), rng.random(5))]
                  for c in range(1, 30)})
    donor, level = ls.draw_link_shuffle(
        np.array([100_000]), np.array([50_000]), 0, pool,
        np.empty(0, np.int64), dbins, np.random.default_rng(0), n_draws=10)
    assert level[0] == 0 and np.unique(donor[:, 0]).size >= 8
