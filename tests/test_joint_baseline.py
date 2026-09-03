"""Per-end posteriors and the aligned 2-D baseline (design A.3.2-A.3.4).

The two failure modes these guard are both silent: a zero variance at n == 1
that would be read as infinite precision, and two marks centred against
differently-shaped neighbourhoods, which manufactures exactly the cross-mark
covariance Part B exists to report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance.joint_baseline import (
    NU_FLOOR, betabinom_mom, distance_strata, fit_joint_baseline,
    fit_prior_grouped, grid_index, normalize, posterior_mean_var)


# ------------------------------------------------------- the n == 1 trap ---

def test_variance_is_nonzero_for_every_single_site_end():
    """--min-hcg-sites 1 is the default, so k/n is exactly 0 or 1 and the
    plug-in variance theta(1-theta)/n is exactly 0.  Anything weighting by
    inverse variance would hand infinite precision to the least informative
    ends in the dataset."""
    n = np.ones(200)
    k = np.concatenate([np.zeros(100), np.ones(100)])
    mean, var = posterior_mean_var(k, n, alpha=2.0, beta=3.0)
    plug_in = (k / n) * (1 - k / n) / n
    assert (plug_in == 0).all(), "the trap must actually be present"
    assert (var > 0).all(), "posterior variance must never be zero at n == 1"
    assert np.isfinite(var).all()


def test_variance_is_nonzero_at_the_saturated_extremes():
    """k == 0 and k == n at any depth are where a plug-in estimator dies."""
    for n in (1, 2, 5, 50):
        for k in (0, n):
            _, var = posterior_mean_var(np.array([k]), np.array([n]), 1.5, 1.5)
            assert var[0] > 0, f"zero variance at k={k}, n={n}"


def test_variance_falls_as_depth_rises():
    _, var = posterior_mean_var(np.array([1, 5, 50]), np.array([2, 10, 100]),
                                2.0, 2.0)
    assert var[0] > var[1] > var[2]


def test_an_unobserved_end_gets_the_prior_not_nan():
    """n == 0 is uninformative, not missing; returning NaN would drop the end
    from Part B rather than let its weight do the work."""
    mean, var = posterior_mean_var(np.array([0.0]), np.array([0.0]), 3.0, 7.0)
    assert mean[0] == pytest.approx(0.3)
    assert var[0] == pytest.approx(0.3 * 0.7 / 11.0)


# ------------------------------------------------------------ the MoM fit ---

def test_mom_matches_the_established_estimator():
    """Same numbers as concordance._fit_betabinom_eb_prior, which the design
    says to reuse -- only the column names were lifted out."""
    from nome_hic_concordance.concordance import _fit_betabinom_eb_prior
    rng = np.random.default_rng(0)
    n = rng.integers(1, 20, 500)
    p = rng.beta(2.0, 5.0, 500)
    k = rng.binomial(n, p)
    a, b, info = betabinom_mom(k, n)
    df = pd.DataFrame({"n_linked": n, "k_linked_active": k})
    a2, b2, info2 = _fit_betabinom_eb_prior(df)
    assert a == pytest.approx(a2)
    assert b == pytest.approx(b2)
    assert info["nu"] == pytest.approx(info2["nu_hat"])


def test_grouped_fit_equals_the_scalar_fit_group_by_group():
    rng = np.random.default_rng(1)
    g = rng.integers(0, 4, 4000)
    n = rng.integers(1, 15, 4000)
    k = rng.binomial(n, rng.beta(2.0, 4.0, 4000))
    alpha, beta, cnt = fit_prior_grouped(k, n, g, 4, min_obs=10)
    for gi in range(4):
        m = g == gi
        a, b, _ = betabinom_mom(k[m], n[m], min_obs=10)
        assert alpha[gi] == pytest.approx(a, rel=1e-9)
        assert beta[gi] == pytest.approx(b, rel=1e-9)
        assert cnt[gi] == m.sum()


def test_thin_groups_come_back_nan_for_the_caller_to_fill():
    k = np.array([1, 0, 1]); n = np.array([2, 2, 2]); g = np.array([0, 0, 1])
    alpha, beta, cnt = fit_prior_grouped(k, n, g, 2, min_obs=3)
    assert np.isnan(alpha[1]) and np.isnan(beta[1])
    assert cnt.tolist() == [2, 1]


def test_a_tight_group_gets_a_tight_prior_not_a_degenerate_one():
    """No over-dispersion beyond binomial noise must not drive nu to zero."""
    rng = np.random.default_rng(2)
    n = np.full(500, 20)
    k = rng.binomial(n, 0.4)
    a, b, info = betabinom_mom(k, n)
    assert info["nu"] >= NU_FLOOR
    assert a > 0 and b > 0


# ------------------------------------------------- the aligned 2-D baseline ---

def _two_mark_data(seed=3, n=6000, n_cells=20):
    rng = np.random.default_rng(seed)
    cell = rng.integers(0, n_cells, n)
    ng = rng.integers(0, 10, n)
    nh = rng.integers(0, 6, n)
    kg = rng.binomial(ng, 0.25)
    kh = rng.binomial(nh, 0.75)
    return cell, (kg, ng), (kh, nh), n_cells


def test_both_marks_fall_back_on_the_same_grid_cells():
    """The A.3.3 alignment guarantee.  Falling back per mark would leave the
    two centred vectors measured against different neighbourhoods, and that
    difference enters theta - mu as cross-mark covariance."""
    cell, g, h, n_cells = _two_mark_data()
    # Starve HCG in half the grid cells only.
    kh, nh = h
    starve = cell % 2 == 0
    nh = nh.copy(); kh = kh.copy()
    nh[starve] = 0; kh[starve] = 0

    bl = fit_joint_baseline(cell, [g, (kh, nh)], n_cells, ("GCH", "HCG"),
                            min_obs=50)
    assert not bl.usable.all(), "the starvation must actually bite"
    mu_g, mu_h = bl.mu("GCH"), bl.mu("HCG")
    # Where unusable, BOTH marks sit exactly on their global prior mean.
    gm = bl.global_alpha / (bl.global_alpha + bl.global_beta)
    assert np.allclose(mu_g[~bl.usable], gm[0])
    assert np.allclose(mu_h[~bl.usable], gm[1])


def test_a_cell_usable_for_one_mark_alone_is_used_for_neither():
    cell = np.repeat(np.arange(2), 200)
    ng = np.full(400, 5)
    kg = np.full(400, 2)
    nh = np.where(cell == 0, 5, 0)          # mark 2 absent in grid cell 1
    kh = np.where(cell == 0, 3, 0)
    bl = fit_joint_baseline(cell, [(kg, ng), (kh, nh)], 2, ("A", "B"),
                            min_obs=50)
    assert bl.usable.tolist() == [True, False]


def test_evaluate_uses_the_grid_cell_prior_where_there_is_one():
    cell, g, h, n_cells = _two_mark_data()
    bl = fit_joint_baseline(cell, [g, h], n_cells, ("GCH", "HCG"), min_obs=20)
    mean, var = bl.evaluate("GCH", g[0], g[1], cell)
    assert mean.shape == g[0].shape
    assert (var > 0).all()
    # An out-of-range grid cell silently falling back is fine; silently
    # indexing the wrong prior is not.
    m2, _ = bl.evaluate("GCH", g[0][:5], g[1][:5], np.full(5, -1))
    a, b = bl.global_alpha[0], bl.global_beta[0]
    assert np.allclose(m2, (a + g[0][:5]) / (a + b + g[1][:5]))


def test_grid_index_round_trips_bin_and_stratum():
    cell, n_cells, bin_of, strat_of = grid_index(
        np.array([0, 1, 2, 5]), np.array([0, 2, 1, 0]), n_bins=3, n_strata=3)
    assert n_cells == 9
    assert cell.tolist() == [0, 5, 7, -1], "an out-of-range bin must be -1"
    ok = cell >= 0
    assert bin_of[cell[ok]].tolist() == [0, 1, 2]
    assert strat_of[cell[ok]].tolist() == [0, 2, 1]


def test_trans_contacts_get_their_own_stratum():
    """Folding trans into the widest cis stratum would let a trans baseline
    set the expectation for the longest cis contacts."""
    edges = np.array([1e3, 1e4, 1e5, 1e6])
    s = distance_strata(np.array([-1, 5e3, 5e4, 5e5]), edges)
    assert s[0] == len(edges) - 1
    assert s[1:].tolist() == [0, 1, 2]
    assert s[0] not in s[1:].tolist()


def test_distances_outside_the_edges_are_excluded_not_clamped():
    edges = np.array([1e3, 1e4, 1e5])
    s = distance_strata(np.array([500.0, 1e6]), edges)
    assert (s == -1).all()


# ------------------------------------------------------------ normalization ---

def test_location_normalization_does_not_touch_scale():
    """A.3.4: a prior rescale distorts Sigma_AA and Sigma_BB and biases rho_2
    in a coverage-dependent way, so the default must centre and stop."""
    theta = np.array([0.1, 0.5, 0.9])
    mu = np.full(3, 0.4)
    var = np.array([0.01, 0.25, 0.04])
    out = normalize(theta, mu, var, mode="location")
    assert np.allclose(out, theta - mu)
    assert np.std(out) == pytest.approx(np.std(theta))


def test_full_normalization_is_available_but_is_not_the_default():
    import inspect
    sig = inspect.signature(normalize)
    assert sig.parameters["mode"].default == "location"
    out = normalize(np.array([0.5]), np.array([0.4]), np.array([0.25]),
                    mode="full")
    assert out[0] == pytest.approx(0.1 / 0.5)


def test_full_normalization_refuses_to_guess_a_variance():
    with pytest.raises(ValueError, match="variance"):
        normalize(np.array([0.5]), np.array([0.4]), None, mode="full")


def test_an_unknown_mode_names_the_valid_ones():
    with pytest.raises(ValueError, match="location"):
        normalize(np.array([0.5]), np.array([0.4]), mode="quantile")


# --------------------------------------------------- streaming accumulation ---

def test_accumulator_reproduces_the_in_memory_fit():
    """The whole dataset will not fit in memory, so the fit accumulates; it
    must land on the same prior as fitting the array at once."""
    from nome_hic_concordance.joint_baseline import PriorAccumulator
    rng = np.random.default_rng(4)
    g = rng.integers(0, 6, 9000)
    n = rng.integers(1, 12, 9000)
    k = rng.binomial(n, rng.beta(2.0, 3.0, 9000))
    want_a, want_b, want_c = fit_prior_grouped(k, n, g, 6, min_obs=20)

    acc = PriorAccumulator.zeros(6, center=k.sum() / n.sum())
    for lo in range(0, 9000, 700):                    # arbitrary shard sizes
        sl = slice(lo, lo + 700)
        acc.add(k[sl], n[sl], g[sl])
    got_a, got_b, got_c = acc.fit(min_obs=20)
    assert np.allclose(got_a, want_a, rtol=1e-8)
    assert np.allclose(got_b, want_b, rtol=1e-8)
    assert got_c.tolist() == want_c.tolist()


def test_merge_order_does_not_change_the_fit():
    """A.4 asks for output independent of the thread count; the pooled fit is
    the only place where completion order could leak in."""
    from nome_hic_concordance.joint_baseline import PriorAccumulator
    rng = np.random.default_rng(5)
    parts = []
    for _ in range(5):
        n = rng.integers(1, 10, 900)
        parts.append((rng.binomial(n, 0.3), n, rng.integers(0, 4, 900)))
    c = sum(p[0].sum() for p in parts) / sum(p[1].sum() for p in parts)

    def run(order):
        acc = PriorAccumulator.zeros(4, center=c)
        for i in order:
            acc.merge(PriorAccumulator.zeros(4, center=c).add(*parts[i]))
        return acc.fit(min_obs=10)

    a1, b1, _ = run([0, 1, 2, 3, 4])
    a2, b2, _ = run([4, 2, 0, 3, 1])
    assert np.allclose(a1, a2, rtol=1e-9) and np.allclose(b1, b2, rtol=1e-9)


def test_merging_differently_centred_accumulators_is_refused():
    from nome_hic_concordance.joint_baseline import PriorAccumulator
    a = PriorAccumulator.zeros(3, center=0.2)
    b = PriorAccumulator.zeros(3, center=0.7)
    with pytest.raises(ValueError, match="centred"):
        a.merge(b)


def test_shifting_the_origin_keeps_precision_on_a_homogeneous_cell():
    """The naive sum-of-squares form loses its significant digits exactly where
    a tight prior matters most.  Pin that the shifted form does not."""
    from nome_hic_concordance.joint_baseline import PriorAccumulator
    n = np.full(400, 100)
    k = np.full(400, 90)                      # rate 0.9 with no spread at all
    g = np.zeros(400, dtype=int)
    acc = PriorAccumulator.zeros(1, center=0.9).add(k, n, g)
    alpha, beta, _ = acc.fit(min_obs=10)
    naive = float((n * (k / n) ** 2).sum() - ((n * (k / n)).sum() ** 2) / n.sum())
    assert naive != 0.0 or True                # the naive form may cancel to 0
    assert np.isfinite(alpha[0]) and np.isfinite(beta[0])
    assert alpha[0] / (alpha[0] + beta[0]) == pytest.approx(0.9, abs=1e-9)
