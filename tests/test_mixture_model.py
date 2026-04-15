"""Unit tests for BetaBinomialMixture and subsample_for_em."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import betabinom as sp_betabinom

from nome_hic_concordance.mixture_model import (
    BetaBinomialMixture,
    betabinom_logpmf,
    subsample_for_em,
)


def _simulate_mixture(
    n_samples: int,
    pi: float,
    alpha_closed: float,
    beta_closed: float,
    alpha_accessible: float,
    beta_accessible: float,
    n_per_end: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.random(n_samples) < pi  # True = accessible
    k = np.empty(n_samples, dtype=int)
    n_arr = np.full(n_samples, n_per_end, dtype=int)

    acc_mask = z
    k[acc_mask] = sp_betabinom.rvs(
        n_per_end, alpha_accessible, beta_accessible,
        size=int(acc_mask.sum()), random_state=rng.integers(0, 2**31 - 1),
    )
    k[~acc_mask] = sp_betabinom.rvs(
        n_per_end, alpha_closed, beta_closed,
        size=int((~acc_mask).sum()), random_state=rng.integers(0, 2**31 - 1),
    )
    return k, n_arr


def test_betabinom_logpmf_matches_scipy():
    """Our log-PMF matches scipy.stats.betabinom.logpmf."""
    k = np.array([0, 1, 2, 3, 4, 5])
    n = np.array([5, 5, 5, 5, 5, 5])
    alpha = 3.0
    beta = 4.0
    ours = betabinom_logpmf(k, n, alpha, beta)
    scipy_vals = sp_betabinom.logpmf(k, n, alpha, beta)
    np.testing.assert_allclose(ours, scipy_vals, rtol=1e-10, atol=1e-12)


def test_mixture_recovers_known_parameters():
    """EM should recover ground-truth parameters on clean synthetic data."""
    true_pi = 0.3
    true_alpha_c, true_beta_c = 2.0, 8.0        # closed: mean = 0.2
    true_alpha_a, true_beta_a = 8.0, 2.0        # accessible: mean = 0.8

    k, n = _simulate_mixture(
        n_samples=20_000,
        pi=true_pi,
        alpha_closed=true_alpha_c, beta_closed=true_beta_c,
        alpha_accessible=true_alpha_a, beta_accessible=true_beta_a,
        n_per_end=10, seed=123,
    )

    model = BetaBinomialMixture(max_iter=200, tol=1e-6, n_restarts=2, random_state=7)
    model.fit(k, n)

    # Component 1 = accessible, component 0 = closed.
    mean_c = model.alpha_[0] / (model.alpha_[0] + model.beta_[0])
    mean_a = model.alpha_[1] / (model.alpha_[1] + model.beta_[1])

    assert abs(mean_c - 0.2) < 0.05, f"mean_closed off: {mean_c}"
    assert abs(mean_a - 0.8) < 0.05, f"mean_accessible off: {mean_a}"
    assert abs(model.pi_ - true_pi) < 0.05, f"pi off: {model.pi_}"


def test_mixture_component_swap_logic():
    """If EM initialises with component 0 > component 1 mean, it should swap."""
    # Simulate with very clear separation.
    k, n = _simulate_mixture(
        n_samples=5_000,
        pi=0.5,
        alpha_closed=1.0, beta_closed=9.0,      # mean ~0.1
        alpha_accessible=9.0, beta_accessible=1.0,  # mean ~0.9
        n_per_end=8, seed=42,
    )
    model = BetaBinomialMixture(max_iter=100, n_restarts=3, random_state=1)
    model.fit(k, n)

    mean_0 = model.alpha_[0] / (model.alpha_[0] + model.beta_[0])
    mean_1 = model.alpha_[1] / (model.alpha_[1] + model.beta_[1])
    assert mean_0 < mean_1  # canonical ordering


def test_predict_proba_handles_n_zero():
    """Posterior should be NaN for n=0 and in [0, 1] otherwise."""
    k = np.array([0, 1, 2, 0, 5, 3])
    n = np.array([0, 5, 5, 5, 10, 10])

    # Fit on a tiny dataset first.
    k_train, n_train = _simulate_mixture(
        n_samples=2_000, pi=0.4,
        alpha_closed=2.0, beta_closed=8.0,
        alpha_accessible=8.0, beta_accessible=2.0,
        n_per_end=8, seed=5,
    )
    model = BetaBinomialMixture(max_iter=50, n_restarts=1).fit(k_train, n_train)

    post = model.predict_proba(k, n)
    assert np.isnan(post[0])  # n=0 -> NaN
    assert np.all((post[1:] >= 0.0) & (post[1:] <= 1.0))


def test_subsample_for_em_preserves_n_distribution():
    """Stratified subsampling should keep the marginal distribution of n close."""
    rng = np.random.default_rng(0)
    # Construct a heterogeneous n distribution.
    n_all = np.concatenate([
        np.full(10_000, 3),
        np.full(5_000, 10),
        np.full(1_000, 30),
    ])
    k_all = rng.integers(0, n_all + 1)

    k_sub, n_sub = subsample_for_em(k_all, n_all, max_samples=4_000, random_state=0)
    assert len(k_sub) <= 4_000

    # Fractions per stratum should be approximately preserved.
    for n_val in (3, 10, 30):
        frac_before = float(np.mean(n_all == n_val))
        frac_after = float(np.mean(n_sub == n_val))
        assert abs(frac_after - frac_before) < 0.05, (
            f"stratum {n_val}: {frac_before} -> {frac_after}"
        )


def test_subsample_for_em_noop_below_threshold():
    k = np.array([0, 1, 2, 3])
    n = np.array([5, 5, 5, 5])
    k_sub, n_sub = subsample_for_em(k, n, max_samples=100)
    np.testing.assert_array_equal(k, k_sub)
    np.testing.assert_array_equal(n, n_sub)


def test_to_dict_from_dict_round_trip():
    k, n = _simulate_mixture(
        n_samples=1_500, pi=0.35,
        alpha_closed=2.0, beta_closed=8.0,
        alpha_accessible=8.0, beta_accessible=2.0,
        n_per_end=8, seed=17,
    )
    model = BetaBinomialMixture(max_iter=50, n_restarts=1).fit(k, n)
    d = model.to_dict()
    restored = BetaBinomialMixture.from_dict(d)

    np.testing.assert_allclose(model.alpha_, restored.alpha_)
    np.testing.assert_allclose(model.beta_, restored.beta_)
    assert model.pi_ == restored.pi_

    # predictions should match
    test_k = np.array([0, 3, 5, 7])
    test_n = np.array([8, 8, 8, 8])
    np.testing.assert_allclose(
        model.predict_proba(test_k, test_n),
        restored.predict_proba(test_k, test_n),
    )


def test_fit_raises_on_empty():
    model = BetaBinomialMixture()
    with pytest.raises(ValueError):
        model.fit(np.array([]), np.array([]))
