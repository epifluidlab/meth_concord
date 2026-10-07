"""The link-shuffle E and null shared by step 5 and ep_coordination_existence:
a contact keeps its left end; E is the anchor with the mean donor -- a
long-range read end of ANOTHER cell of the stratum, on the partner's side at
a distance in the contact's own bin -- of one half of the stratum's cells,
the null one donor per draw from the other half, and within a draw no null
donor goes to two contacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import link_shuffle as ls

DBINS = ls.log_dbins(20_000, 2_000_000, 4)


def _pool(spec):
    """{cell: [(pos, p), ...]} on chromosome code 0 -> pool."""
    return ls.build_pool({c: (np.array([q for q, _ in v], np.int64),
                              np.array([p for _, p in v], float))
                          for c, v in spec.items()})


def _assign(anchors, cells, pool_e, pool_n, **kw):
    a = np.array([g for g, _ in anchors], np.int64)
    d = np.array([x for _, x in anchors], np.int64)
    return ls.assign_stratum(a, d, np.asarray(cells), pool_e, pool_n, DBINS,
                             np.random.default_rng(0), **kw)


# ------------------------------------------------------------ the windows --


def test_e_and_the_null_come_from_other_cells_at_the_partners_distance():
    pool_e = _pool({0: [(150_000, 0.9)],              # the anchor's own cell
                    1: [(149_000, 0.2), (155_000, 0.4), (900_000, 0.7)],
                    2: [(60_000, 0.1)]})              # left of the anchor
    pool_n = _pool({3: [(152_000, 0.6)], 0: [(151_000, 0.95)]})
    level, m, donor, reused = _assign([(100_000, 50_000)], [0], pool_e,
                                      pool_n, n_draws=20)
    assert level.tolist() == [0]
    assert m[0] == pytest.approx(0.3)       # cells 1's two ends, not 0's own
    assert set(donor[:, 0]) == {0.6}        # cell 3's end, never cell 0's own
    assert reused == 0


def test_a_contact_needs_a_donor_in_both_halves():
    pool_e = _pool({1: [(150_000, 0.2)]})
    for pool_n in (_pool({0: [(150_000, 0.9)]}),       # only its own cell's
                   _pool({2: [(900_000, 0.9)]})):      # none at its distance
        level, m, donor, _ = _assign([(100_000, 50_000)], [0], pool_e, pool_n)
        assert level.tolist() == [-1]
        assert np.isnan(m[0]) and np.isnan(donor[0, 0])


def test_a_window_widens_only_when_its_own_bin_is_empty():
    pool = _pool({1: [(100_000 + 80_000, 0.3)]})      # one bin up from 50 kb
    level, m, donor, _ = _assign([(100_000, 50_000), (100_000, 1_900_000)],
                                 [0, 0], pool, pool, max_widen=1)
    assert level.tolist() == [1, -1]
    assert m[0] == 0.3 and donor[0, 0] == 0.3 and np.isnan(donor[0, 1])


def test_bounds_keep_donors_where_the_partner_is():
    dbins = ls.log_dbins(1_000, 2_500_000, 4)
    pool = _pool({1: [(2_510_000, 0.8), (2_480_000, 0.1)]})
    level, m, donor, _ = ls.assign_stratum(
        np.array([2_470_000]), np.array([20_000]), np.array([0]), pool, pool,
        dbins, np.random.default_rng(0),
        bounds=(np.array([0]), np.array([2_500_000])), max_widen=3)
    assert m[0] == 0.1 and donor[0, 0] == 0.1   # 2.51 Mb is past the bin


# -------------------------------------------------------------- the draws --


def test_no_null_donor_goes_to_two_contacts_in_a_draw():
    """Forty contacts of forty cells share one window holding sixty donors:
    every draw gives each contact its own."""
    rng = np.random.default_rng(1)
    pos = rng.integers(140_000, 160_000, 60)
    pool_n = ls.build_pool({100 + j: (pos[j:j + 1], np.array([j / 100]))
                            for j in range(60)})
    pool_e = _pool({99: [(150_000, 0.5)]})
    level, _, donor, reused = _assign([(100_000, 50_000)] * 40, range(40),
                                      pool_e, pool_n, n_draws=5)
    assert (level == 0).all() and reused == 0
    for draw in donor:
        assert np.unique(draw).size == 40
    assert np.unique(donor[:, 0]).size >= 3       # and the draws differ


def test_a_contact_borrows_only_when_its_window_is_used_up():
    pool_n = _pool({5: [(150_000, 0.7)]})               # one donor, two asks
    pool_e = _pool({6: [(150_000, 0.5)]})
    level, _, donor, reused = _assign([(100_000, 50_000), (101_000, 50_000)],
                                      [0, 1], pool_e, pool_n, n_draws=3)
    assert (level == 0).all()
    assert (donor == 0.7).all() and reused == 3        # one per draw


def test_many_candidates_give_independent_draws():
    rng = np.random.default_rng(1)
    pool = _pool({c: [(int(x), float(p)) for x, p in
                      zip(rng.integers(140_000, 160_000, 5), rng.random(5))]
                  for c in range(1, 30)})
    level, _, donor, _ = _assign([(100_000, 50_000)], [0], pool, pool,
                                 n_draws=10)
    assert level[0] == 0 and np.unique(donor[:, 0]).size >= 8


# ------------------------------------------------------- cells to strata --


def _cell(path, rows):
    df = pd.DataFrame(rows, columns=["chrom", "pos1", "pos2", "posterior1",
                                     "posterior2"])
    df["n1"] = np.int32(3)
    df["n2"] = np.int32(3)
    df.to_parquet(path)
    return path


def test_read_cell_keeps_contacts_by_row_and_long_range_ends(tmp_path):
    p = _cell(tmp_path / "c.parquet", [
        ("chr1", 100_000, 150_000, 0.9, 0.8),       # contact, row 0
        ("chr1", 200_000, 200_500, 0.1, 0.2),       # short: neither
        ("chr2", 990_000, 1_030_000, 0.4, 0.3),     # crosses a 1 Mb bin
        ("chr2", 5_000_000, 9_000_000, 0.6, 0.5)])  # past max_distance
    x = ls.read_cell(p, 20_000, 2, max_distance=2_000_000)
    assert x["contacts"]["chr1"][2].tolist() == [0]
    assert x["contacts"]["chr2"][2].tolist() == [2]
    assert x["contacts"]["chr1"][1].tolist() == [50_000]
    assert sorted(x["ends"]["chr2"][0].tolist()) == [
        990_000, 1_030_000, 5_000_000, 9_000_000]   # any long pair's ends
    assert 200_000 not in x["ends"]["chr1"][0]
    in_bin = ls.read_cell(p, 20_000, 2, max_distance=2_000_000,
                          bin_size=1_000_000)
    assert "chr2" not in in_bin["contacts"]


def test_assign_all_splits_each_stratum_and_finds_contacts_by_row(tmp_path):
    rng = np.random.default_rng(3)
    cells, strata = [], []
    for c in range(12):
        rows = []
        for _ in range(60):
            a = int(rng.integers(0, 5_000_000))
            rows.append(("chr1", a, a + int(rng.integers(20_000, 300_000)),
                         float(rng.random()), float(rng.random())))
        cells.append(ls.read_cell(_cell(tmp_path / f"c{c}.parquet", rows),
                                  20_000, 2, 2_000_000))
        strata.append("AB"[c % 2])
    out, summary = ls.assign_all(cells, strata, DBINS, n_draws=2, seed=7)
    assert summary["strata"] == 2 and summary["contacts"] == 12 * 60
    assert summary["own_bin"] + summary["widened"] + summary["none"] == 720
    for x, o in zip(cells, out):
        assert np.array_equal(o["row"], np.sort(x["contacts"]["chr1"][2]))
        assert o["donor_p"].shape == (2, o["row"].size)
        level, m, donor = ls.lookup(o, np.array([o["row"][3], 10_000]), 2)
        assert level[0] == o["level"][3] and level[1] == -1
        assert np.isnan(donor[:, 1]).all()
    again, _ = ls.assign_all(cells, strata, DBINS, n_draws=2, seed=7)
    assert all(np.array_equal(a["donor_p"], b["donor_p"], equal_nan=True)
               for a, b in zip(out, again))          # reproducible


def test_e_comes_from_the_other_half_and_the_null_from_the_contacts_own():
    """A real partner is an end of the contact's own half that also sits in
    the other half's E; a null donor from the contact's own half does too.
    One half lending to every E and the other to every null coupled real
    partners to E and null donors to nothing -- and made a null with no
    coordination more separable than the real."""
    import zlib
    cells = []
    for c in range(6):                  # cell c's every end has posterior c/10
        ends = np.array([150_000 + c * 100, 151_000 + c * 100], np.int64)
        cells.append({"ends": {"chr1": (ends, np.full(2, c / 10))},
                      "contacts": {"chr1": (np.array([100_000]),
                                            np.array([50_000]),
                                            np.array([0]))}})
    out, _ = ls.assign_all(cells, ["S"] * 6, DBINS, n_draws=8, seed=3)
    half = ls.split_cells(range(6), seed=3 + zlib.crc32(b"S"))
    for c in range(6):
        own = [k / 10 for k in range(6) if half[k] == half[c] and k != c]
        other = [k / 10 for k in range(6) if half[k] != half[c]]
        assert out[c]["m"][0] == pytest.approx(np.mean(other))
        assert set(out[c]["donor_p"][:, 0]) <= set(own)
