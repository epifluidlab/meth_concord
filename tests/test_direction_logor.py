"""Direction on the log-odds scale, and what it does and does not measure.

L = logit(F) - logit(F_exp) = log[(O_open/O_closed) / (E_open/E_closed)] is
the natural parameter of the 2x2 table: additive, and with a variance free of
F, so bins of different depth and different marginal are comparable in a way
F - F_exp is not.

It is not a second axis beside F - F_exp -- both are monotone in the same F --
and it is not a magnitude.  Both of those are pinned here, because both are
easy to assume and expensive to assume wrongly.
"""

from __future__ import annotations

import numpy as np
import pytest

from nome_hic_concordance.concordance import decompose_components as dc


def _rates(rng, n):
    """Four DM-style rates per bin, summing under 1 with mixed mass left."""
    a = rng.dirichlet([2.0, 2.0, 4.0], size=n)
    o_open, o_closed = a[:, 0], a[:, 1]
    b = rng.dirichlet([2.0, 2.0, 4.0], size=n)
    return o_open, o_closed, b[:, 0], b[:, 1]


def test_it_is_the_log_odds_ratio_of_the_two_tables():
    d = dc(np.array([0.30]), np.array([0.50]),
           np.array([0.20]), np.array([0.60]))
    assert d["direction_logor"][0] == pytest.approx(
        np.log(0.30 / 0.50) - np.log(0.20 / 0.60))


def test_it_is_zero_exactly_where_the_excess_is_zero():
    """Both call the marginal-matched null the null.  If they disagreed
    about where zero is, one of them would be reporting the marginal."""
    rng = np.random.default_rng(0)
    o_open, o_closed, _e_o, _e_c = _rates(rng, 500)
    d = dc(o_open, o_closed, o_open * 0.5, o_closed * 0.5)   # same ratio
    np.testing.assert_allclose(d["direction_excess"], 0.0, atol=1e-12)
    np.testing.assert_allclose(d["direction_logor"], 0.0, atol=1e-12)


def test_it_agrees_with_the_excess_on_sign_everywhere():
    rng = np.random.default_rng(1)
    d = dc(*_rates(rng, 4_000))
    ex, lo = d["direction_excess"], d["direction_logor"]
    move = np.abs(ex) > 1e-9
    assert np.all(np.sign(ex[move]) == np.sign(lo[move]))


def test_it_carries_the_same_information_as_the_excess():
    """The proposal to embed BOTH would supply one axis twice.  To first
    order L = (F - F_exp) / [F_exp(1 - F_exp)], so what separates them is
    F_exp -- the marginal -- which is already emitted as 'marginal'."""
    rng = np.random.default_rng(2)
    d = dc(*_rates(rng, 20_000))
    ex, lo, fx = (d["direction_excess"], d["direction_logor"],
                  d["direction_expected"])
    assert np.corrcoef(ex, lo)[0, 1] > 0.85
    # The first-order relation, checked where a first-order relation is
    # meant to hold: a small step, and a null not pinned against 0 or 1.
    small = (np.abs(ex) < 0.02) & (fx > 0.2) & (fx < 0.8)
    assert small.sum() > 200
    approx = ex[small] / (fx[small] * (1 - fx[small]))
    assert np.corrcoef(approx, lo[small])[0, 1] > 0.99


def test_the_variance_stabilisation_is_the_point():
    """Equal excess at different F is a different amount of evidence; equal
    log-OR is the same odds ratio wherever it sits.  This is what makes the
    axis comparable across bins with different marginals."""
    near_half = dc(np.array([0.30]), np.array([0.30]),
                   np.array([0.25]), np.array([0.35]))
    near_edge = dc(np.array([0.05]), np.array([0.55]),
                   np.array([0.02]), np.array([0.58]))
    # the same shift in F means very different odds ratios
    assert abs(near_half["direction_excess"][0]) == pytest.approx(
        0.0833333, abs=1e-4)
    assert abs(near_edge["direction_excess"][0]) == pytest.approx(
        0.0501, abs=1e-3)
    assert abs(near_edge["direction_logor"][0]) > \
        abs(near_half["direction_logor"][0])


def test_it_says_nothing_about_how_much_coordination_there_is():
    """Scaling the concordant mass up at a fixed co-open:co-closed ratio
    raises C and leaves L alone.  A claim that coordination INCREASED is a
    claim about C, and L cannot support or refute it."""
    weak = dc(np.array([0.12]), np.array([0.06]),
              np.array([0.10]), np.array([0.10]))
    strong = dc(np.array([0.48]), np.array([0.24]),
                np.array([0.10]), np.array([0.10]))
    assert strong["coordination"][0] > 5 * weak["coordination"][0]
    assert strong["direction_logor"][0] == pytest.approx(
        weak["direction_logor"][0])


def test_an_empty_pool_gets_no_direction_rather_than_a_default():
    """With no observed or no expected concordant mass there is nothing to
    take a ratio of, and a clipped log would report a huge direction from
    the clip rather than from the data."""
    d = dc(np.array([0.0, 0.3]), np.array([0.0, 0.4]),
           np.array([0.2, 0.0]), np.array([0.3, 0.0]))
    np.testing.assert_array_equal(d["direction_logor"], [0.0, 0.0])


def test_tiny_rates_are_clipped_rather_than_infinite():
    d = dc(np.array([1e-300]), np.array([0.4]),
           np.array([0.2]), np.array([0.3]))
    assert np.isfinite(d["direction_logor"]).all()


def test_the_component_is_emitted_under_its_own_name():
    d = dc(np.array([0.3]), np.array([0.4]),
           np.array([0.2]), np.array([0.3]))
    assert "direction_logor" in d


def test_it_is_the_same_function_the_pipeline_calls():
    """Lifted to module level so the test pins the code rather than a
    re-implementation of it that can drift."""
    import inspect
    from nome_hic_concordance import concordance as cc
    src = inspect.getsource(cc.compute_signed_concordance_matrix)
    assert "decompose_components(" in src
    assert "def _decompose" not in src
