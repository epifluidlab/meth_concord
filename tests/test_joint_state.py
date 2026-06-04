"""Tests for the joint GCH+HCG multi-state core (design §2–§3)."""

from __future__ import annotations

import numpy as np
import pytest

from nome_hic_concordance.joint_state import (
    JOINT_STATES,
    co_state_agreement,
    impute_missing_modality,
    pair_co_state,
    per_end_joint_state,
)


# --------------------------------------------------------------------------
# per_end_joint_state — the independence-approximation product (§2)
# --------------------------------------------------------------------------

def test_per_end_states_sum_to_one():
    rng = np.random.default_rng(0)
    a = rng.random(1000)
    m = rng.random(1000)
    q = per_end_joint_state(a, m)
    total = sum(q)
    assert np.allclose(total, 1.0)


def test_per_end_states_corner_one_hots():
    # Each corner of the (a, m) unit square is a pure joint state.
    # order: open, closed, poised, active_methylated
    cases = {
        (1.0, 0.0): (1.0, 0.0, 0.0, 0.0),   # accessible + unmethylated -> open
        (0.0, 1.0): (0.0, 1.0, 0.0, 0.0),   # occupied + methylated -> closed
        (0.0, 0.0): (0.0, 0.0, 1.0, 0.0),   # occupied + unmethylated -> poised
        (1.0, 1.0): (0.0, 0.0, 0.0, 1.0),   # accessible + methylated -> active-meth
    }
    for (a, m), expected in cases.items():
        q = per_end_joint_state(a, m)
        assert np.allclose([float(x) for x in q], expected), f"a={a} m={m}"


def test_per_end_states_center_is_uniform():
    q = per_end_joint_state(0.5, 0.5)
    assert np.allclose([float(x) for x in q], [0.25, 0.25, 0.25, 0.25])


def test_per_end_states_match_explicit_formula():
    a, m = 0.7, 0.2
    q_open, q_closed, q_poised, q_am = per_end_joint_state(a, m)
    assert q_open == pytest.approx(a * (1 - m))
    assert q_closed == pytest.approx((1 - a) * m)
    assert q_poised == pytest.approx((1 - a) * (1 - m))
    assert q_am == pytest.approx(a * m)


def test_per_end_states_preserve_shape():
    a = np.full((3, 4), 0.6)
    m = np.full((3, 4), 0.3)
    q = per_end_joint_state(a, m)
    for ch in q:
        assert ch.shape == (3, 4)


def test_joint_states_constant_order():
    assert JOINT_STATES == ("open", "closed", "poised", "active_methylated")


# --------------------------------------------------------------------------
# impute_missing_modality — lenient marginalization (§2.1)
# --------------------------------------------------------------------------

def test_impute_uses_prior_where_uninformative():
    a = np.array([0.9, 0.8, np.nan, 0.1])
    m = np.array([0.2, np.nan, 0.5, 0.4])
    n_gpc = np.array([5, 5, 5, 0])     # end 3 has no GpC sites
    n_cpg = np.array([5, 0, 5, 5])     # end 1 has no CpG sites
    a_eff, m_eff, a_obs, m_obs = impute_missing_modality(
        a, n_gpc, m, n_cpg, min_gpc=2, min_cpg=2, a_prior=0.33, m_prior=0.77)

    # end 0: both observed -> unchanged
    assert a_eff[0] == pytest.approx(0.9) and m_eff[0] == pytest.approx(0.2)
    # end 1: CpG missing -> m imputed to prior, a kept
    assert a_eff[1] == pytest.approx(0.8) and m_eff[1] == pytest.approx(0.77)
    # end 2: GpC posterior NaN -> a imputed to prior, m kept
    assert a_eff[2] == pytest.approx(0.33) and m_eff[2] == pytest.approx(0.5)
    # end 3: GpC count below threshold -> a imputed; m kept
    assert a_eff[3] == pytest.approx(0.33) and m_eff[3] == pytest.approx(0.4)

    assert a_obs.tolist() == [True, True, False, False]
    assert m_obs.tolist() == [True, False, True, True]


def test_impute_nan_count_treated_as_missing():
    # A NaN posterior with enough sites is still "not observed".
    a = np.array([np.nan])
    a_eff, _, a_obs, _ = impute_missing_modality(
        a, np.array([10]), np.array([0.5]), np.array([10]),
        min_gpc=2, min_cpg=2, a_prior=0.4, m_prior=0.5)
    assert a_eff[0] == pytest.approx(0.4)
    assert not a_obs[0]


def test_impute_array_prior_broadcasts():
    # Locally-informed per-region prior (the optional upgrade): array prior.
    a = np.array([np.nan, np.nan])
    a_prior = np.array([0.1, 0.9])
    a_eff, _, _, _ = impute_missing_modality(
        a, np.array([0, 0]), np.array([0.5, 0.5]), np.array([5, 5]),
        min_gpc=2, min_cpg=2, a_prior=a_prior, m_prior=0.5)
    assert a_eff.tolist() == [0.1, 0.9]


def test_impute_fully_observed_is_identity():
    rng = np.random.default_rng(1)
    a = rng.random(50)
    m = rng.random(50)
    n = np.full(50, 9)
    a_eff, m_eff, a_obs, m_obs = impute_missing_modality(
        a, n, m, n, min_gpc=2, min_cpg=2, a_prior=0.0, m_prior=0.0)
    assert np.allclose(a_eff, a) and np.allclose(m_eff, m)
    assert a_obs.all() and m_obs.all()


# --------------------------------------------------------------------------
# pair_co_state — multi-state soft co-state (§3)
# --------------------------------------------------------------------------

def test_co_state_five_sum_to_one():
    rng = np.random.default_rng(2)
    a_i, m_i = rng.random(500), rng.random(500)
    a_j, m_j = rng.random(500), rng.random(500)
    qi = per_end_joint_state(a_i, m_i)
    qj = per_end_joint_state(a_j, m_j)
    co = pair_co_state(qi, qj)
    assert len(co) == 5
    assert np.allclose(sum(co), 1.0)


def test_co_state_identical_pure_states_agree():
    # Both ends pure-open -> co_open = 1, everything else 0.
    qi = per_end_joint_state(1.0, 0.0)
    qj = per_end_joint_state(1.0, 0.0)
    co_open, co_closed, co_poised, co_am, mixed = pair_co_state(qi, qj)
    assert co_open == pytest.approx(1.0)
    assert co_closed == pytest.approx(0.0)
    assert co_poised == pytest.approx(0.0)
    assert co_am == pytest.approx(0.0)
    assert mixed == pytest.approx(0.0)


def test_co_state_orthogonal_pure_states_all_mixed():
    # End i pure-open, end j pure-closed -> all co_s = 0, mixed = 1.
    qi = per_end_joint_state(1.0, 0.0)   # open
    qj = per_end_joint_state(0.0, 1.0)   # closed
    co = pair_co_state(qi, qj)
    assert np.allclose([float(x) for x in co[:4]], 0.0)
    assert co[4] == pytest.approx(1.0)


def test_co_state_mixed_is_nonnegative():
    rng = np.random.default_rng(3)
    qi = per_end_joint_state(rng.random(2000), rng.random(2000))
    qj = per_end_joint_state(rng.random(2000), rng.random(2000))
    mixed = pair_co_state(qi, qj)[4]
    assert (mixed >= 0).all()


def test_co_state_channels_match_products():
    qi = per_end_joint_state(0.7, 0.1)
    qj = per_end_joint_state(0.4, 0.6)
    co = pair_co_state(qi, qj)
    for s in range(4):
        assert co[s] == pytest.approx(float(qi[s]) * float(qj[s]))


def test_co_state_agreement_matches_sum_of_channels():
    rng = np.random.default_rng(4)
    qi = per_end_joint_state(rng.random(300), rng.random(300))
    qj = per_end_joint_state(rng.random(300), rng.random(300))
    agree = co_state_agreement(qi, qj)
    via_channels = sum(pair_co_state(qi, qj)[:4])
    assert np.allclose(agree, via_channels)


def test_co_state_agreement_reduces_to_single_context_when_methylation_flat():
    # If methylation is uninformative (m = 0 at every end), the joint state
    # collapses to {open=a, poised=1-a}; the co-state agreement then equals the
    # single-context GCH agreement  p_i p_j + (1-p_i)(1-p_j).
    rng = np.random.default_rng(5)
    a_i, a_j = rng.random(400), rng.random(400)
    qi = per_end_joint_state(a_i, 0.0)
    qj = per_end_joint_state(a_j, 0.0)
    agree = co_state_agreement(qi, qj)
    single = a_i * a_j + (1 - a_i) * (1 - a_j)
    assert np.allclose(agree, single)
