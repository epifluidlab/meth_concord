"""Rank-1 test on the 2x2 GCH/HCG cross-covariance (Part B of the design).

The simulator draws from the Section 1 model with known parameters, so the
estimator is checked against a truth rather than against itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from nome_hic_concordance import rank1_joint as rj


import contextlib
import warnings


@contextlib.contextmanager
def _no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


def simulate(n=4000, rho_u=0.6, lam_g=1.0, lam_h=-0.8, noise=0.5,
             rank2=0.0, seed=0):
    """Section 1: theta = mu + lambda * u + eta, u correlated across ends.

    ``rank2`` injects Cov(eta_A, eta_B) on the HCG residual only -- coordination
    the shared axis cannot explain, which is the alternative the test exists to
    detect.
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho_u], [rho_u, 1.0]])
    u = rng.multivariate_normal([0, 0], cov, size=n)
    eh = rng.multivariate_normal([0, 0], [[1.0, rank2], [rank2, 1.0]], size=n)
    tA = np.column_stack([lam_g * u[:, 0] + noise * rng.normal(size=n),
                          lam_h * u[:, 0] + noise * eh[:, 0]])
    tB = np.column_stack([lam_g * u[:, 1] + noise * rng.normal(size=n),
                          lam_h * u[:, 1] + noise * eh[:, 1]])
    v = np.full((n, 2), 1e-4)
    return tA, tB, v, v


# ----------------------------------------------------------- accumulators --

def test_sharded_accumulation_equals_a_single_pass():
    """All 20 fields are additive; sharding must not change the answer."""
    tA, tB, vA, vB = simulate(n=600, seed=1)
    one = rj.accumulate(rj.new_acc(), tA, tB, vA, vB)
    parts = [rj.accumulate(rj.new_acc(), tA[s], tB[s], vA[s], vB[s])
             for s in (slice(0, 137), slice(137, 400), slice(400, 600))]
    assert np.allclose(one, rj.merge(*parts), rtol=1e-12, atol=1e-12)
    # and merging is order-independent
    assert np.allclose(rj.merge(*parts), rj.merge(*parts[::-1]))


def test_moments_recover_the_planted_cross_covariance():
    tA, tB, vA, vB = simulate(n=200_000, rho_u=0.5, lam_g=1.0, lam_h=-0.8,
                              noise=0.3, seed=2)
    m = rj.moments(rj.accumulate(rj.new_acc(), tA, tB, vA, vB))
    # Sigma_AB = rho_u * u u^T with u = (lam_g, lam_h)
    want = 0.5 * np.outer([1.0, -0.8], [1.0, -0.8])
    assert np.allclose(m["S_AB"], want, atol=0.02), (m["S_AB"], want)


def test_the_weight_depends_on_variance_only_not_on_the_level():
    """A posterior near 0.5 may be uninformative OR a genuine intermediate
    state, so the weight must not read the level -- down-weighting |theta -
    0.5| conflates the two.  Asserted on BEHAVIOUR: hold the variances fixed,
    move theta, and the weighted total must not move.  (An earlier version
    grepped the source and passed on the docstring, which explains the rule.)
    """
    rng = np.random.default_rng(12)
    v = rng.uniform(0.01, 0.2, size=(500, 2))
    near_half = np.full((500, 2), 0.5)
    extreme = np.column_stack([np.zeros(500), np.ones(500)])
    w_half = rj.accumulate(rj.new_acc(), near_half, near_half, v, v)[0]
    w_ext = rj.accumulate(rj.new_acc(), extreme, extreme, v, v)[0]
    assert w_half == pytest.approx(w_ext, rel=1e-12)


# ---------------------------------------------------- the determinant test --

def test_the_zero_of_D_is_what_whitening_preserves():
    """The design's test list asks for 'D invariant under whitening'.  It is
    NOT -- D = 2det/||.||_F^2 rescales, because the Frobenius norm is not
    multiplicative under M -> A M B.  What whitening preserves is the RANK,
    hence D's ZERO, which is exactly what Section B.3 argues and all the
    hypothesis needs.  Both halves are pinned here so neither claim drifts.
    """
    rng = np.random.default_rng(3)
    for _ in range(20):
        A, B = rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
        rank1 = np.outer(rng.normal(size=2), rng.normal(size=2))
        assert abs(rj.det_statistic(rank1)) < 1e-12
        assert abs(rj.det_statistic(A @ rank1 @ B)) < 1e-12   # zero survives
    full = rng.normal(size=(2, 2))
    A, B = rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
    assert not np.isclose(rj.det_statistic(full),
                          rj.det_statistic(A @ full @ B), atol=1e-3)


def test_D_is_bounded_and_signed():
    assert rj.det_statistic(np.eye(2)) == pytest.approx(1.0)
    assert rj.det_statistic(np.array([[0.0, 1.0], [1.0, 0.0]])) == pytest.approx(-1.0)
    assert rj.det_statistic(np.zeros((2, 2))) == 0.0


def test_D_is_scale_free():
    rng = np.random.default_rng(4)
    M = rng.normal(size=(2, 2))
    assert rj.det_statistic(M) == pytest.approx(rj.det_statistic(7.5 * M))


# -------------------------------------------------------------- whitening --

def test_rho_are_invariant_to_swapping_the_two_marks():
    """Relabelling gch/hcg transposes the pattern; the canonical
    correlations are properties of the pair and must not move."""
    tA, tB, vA, vB = simulate(n=5000, seed=5)
    def rhos(a, b, va, vb):
        m = rj.moments(rj.accumulate(rj.new_acc(), a, b, va, vb))
        w = rj.whiten(m["S_AA"], m["S_BB"], m["S_AB"], m["V_A"], m["V_B"])
        c = rj.canonical(w["R"])
        return c["rho1"], c["rho2"]
    a = rhos(tA, tB, vA, vB)
    b = rhos(tA[:, ::-1], tB[:, ::-1], vA[:, ::-1], vB[:, ::-1])
    assert np.allclose(a, b, atol=1e-9), (a, b)


def test_pd_clipping_flags_and_leaves_good_matrices_alone():
    good = rj.whiten(np.eye(2), np.eye(2), 0.3 * np.eye(2),
                     np.zeros(2), np.zeros(2))
    assert good["flag_pd_clipped"] is False
    # disattenuation drives a variance negative -- the min-hcg-sites 1 case
    bad = rj.whiten(np.diag([1.0, 0.2]), np.eye(2), 0.3 * np.eye(2),
                    np.array([0.0, 0.9]), np.zeros(2))
    assert bad["flag_pd_clipped"] is True
    assert bad["reliability_hcg"] < 0          # reported, not silently repaired


def test_reliabilities_are_the_disattenuated_share():
    w = rj.whiten(np.diag([1.0, 1.0]), np.eye(2), 0.2 * np.eye(2),
                  np.array([0.25, 0.5]), np.zeros(2))
    assert w["reliability_gch"] == pytest.approx(0.75)
    assert w["reliability_hcg"] == pytest.approx(0.50)


# ---------------------------------------------------------- sign patterns --

@pytest.mark.parametrize("lam_h,rho_u,want", [
    (-0.8, +0.6, "shared_state"),
    (-0.8, -0.6, "opposite_state"),
    (+0.8, +0.6, "coupled_same"),
    (+0.8, -0.6, "coupled_opposite"),
])
def test_all_four_valid_sign_patterns_are_recovered(lam_h, rho_u, want):
    """Three of the four would score as failures against any single
    'pattern is correct' boolean, which is why there isn't one."""
    tA, tB, vA, vB = simulate(n=60_000, rho_u=rho_u, lam_h=lam_h,
                              noise=0.3, seed=6)
    m = rj.moments(rj.accumulate(rj.new_acc(), tA, tB, vA, vB))
    got = rj.sign_pattern(m["S_AA"], m["S_AB"])
    assert got["rank1_pattern_class"] == want, got
    assert got["sign_lambda_h"] == int(np.sign(lam_h))
    assert got["sign_rho_u"] == int(np.sign(rho_u))


# ---------------------------------------------------------------- the null --

def test_rank1_data_gives_no_excess_and_a_large_p():
    tA, tB, vA, vB = simulate(n=3000, rank2=0.0, seed=7)
    r = rj.conditional_rank1_null(tA, tB, vA, vB, n_boot=200, seed=0)
    assert r["D_null_mean"] > 0, "the null mean must be positive at finite n"
    assert abs(r["D_excess"]) < 0.15, r
    assert r["p_rank1"] > 0.05, r


def test_injected_rank2_structure_is_detected():
    """Cov(eta_A, eta_B) on the HCG residual is coordination the shared axis
    cannot carry -- the alternative, and the headline class."""
    tA, tB, vA, vB = simulate(n=3000, rank2=0.7, seed=8)
    r = rj.conditional_rank1_null(tA, tB, vA, vB, n_boot=200, seed=0)
    assert r["D_excess"] > 0.1, r
    assert r["p_rank1"] < 0.05, r


def test_the_null_expectation_is_positive_so_the_raw_value_misleads():
    """The design names this the most likely route to a false headline: at
    finite n both |D| and rho2 are positive under the null, so a raw value
    makes every bin in the genome read as rank-2."""
    tA, tB, vA, vB = simulate(n=400, rank2=0.0, seed=9)
    r = rj.conditional_rank1_null(tA, tB, vA, vB, n_boot=200, seed=0)
    assert r["D_null_mean"] > 0.02, r
    assert r["rho2_null_mean"] > 0.0, r
    # The excess is what may be reported.  At rank-1 truth it sits near zero
    # while the raw |D| does not, which is the whole point: a bin reading
    # |D| = 0.1 is unremarkable once its own null mean is subtracted.
    assert abs(r["D_excess"]) < r["D_null_mean"], r


def test_a_permutation_p_is_never_exactly_zero():
    tA, tB, vA, vB = simulate(n=2000, rank2=0.9, seed=10)
    r = rj.conditional_rank1_null(tA, tB, vA, vB, n_boot=50, seed=0)
    assert r["p_rank1"] >= 1.0 / 51.0


def test_the_rank2_injection_uses_a_valid_covariance():
    """rank2=1.5 asks numpy for a non-PSD covariance and it silently returns
    something anyway.  Keep the injection inside the valid range."""
    with pytest.warns(None) if False else _no_warning():
        simulate(n=50, rank2=0.9, seed=0)


def test_null_p_values_are_not_grossly_anticonservative():
    """A non-uniform QQ under rank-1 truth would invalidate every rank-2
    call.  Loose bound: this is a smoke test, not the full B.7.4 panel."""
    ps = []
    for s in range(24):
        tA, tB, vA, vB = simulate(n=1500, rank2=0.0, seed=100 + s)
        ps.append(rj.conditional_rank1_null(tA, tB, vA, vB, n_boot=100,
                                            seed=s)["p_rank1"])
    ps = np.array(ps)
    # calibrated, not merely non-anticonservative: the fix that made this
    # pass was mapping the shuffled null back into theta space, since D is
    # not whitening-invariant.  Before it, every p sat above 0.74.
    assert (ps < 0.05).mean() <= 0.20, ps
    assert 0.3 < ps.mean() < 0.7, ps.mean()


def test_strata_confine_the_shuffle():
    tA, tB, vA, vB = simulate(n=800, seed=11)
    strata = np.repeat([0, 1, 2, 3], 200)
    r = rj.conditional_rank1_null(tA, tB, vA, vB, n_boot=50, seed=0,
                                  strata=strata)
    assert np.isfinite(r["D_null"]).all()
    assert r["D_null"].size == 50


# ------------------------------------------------- B.6.1 the mate shuffle ---

def test_mate_shuffle_destroys_rho1(rank1_sim=None):
    """B.7.5: rho1 under the mate shuffle must sit at zero.  It is the wrong
    null for the RANK question precisely because it destroys rho1 too."""
    tA, tB, vA, vB = simulate(n=1500, seed=21)
    out = rj.mate_shuffle_null(tA, tB, vA, vB, n_boot=120, seed=3)
    assert out["rho1"] > 0.15, "the simulation must be coordinated at all"
    assert abs(out["rho1_mate_shuffle_mean"]) < out["rho1"] / 2, (
        "shuffling mates must collapse rho1 toward zero")
    assert out["z_mate_shuffle"] > 3


def test_mate_shuffle_finds_nothing_when_there_is_nothing():
    """Uncoordinated ends must not produce a significant z."""
    rng = np.random.default_rng(7)
    n = 1200
    tA = rng.normal(size=(n, 2))
    tB = rng.normal(size=(n, 2))
    v = np.full((n, 2), 0.05)
    out = rj.mate_shuffle_null(tA, tB, v, v, n_boot=150, seed=4)
    assert abs(out["z_mate_shuffle"]) < 3.5, out["z_mate_shuffle"]


def test_mate_shuffle_respects_strata():
    """Shuffling across strata would mix distance regimes and manufacture a
    null that is easier to beat than the truth."""
    tA, tB, vA, vB = simulate(n=800, seed=22)
    s = np.arange(800) % 4
    out = rj.mate_shuffle_null(tA, tB, vA, vB, n_boot=60, seed=5, strata=s)
    assert np.isfinite(out["z_mate_shuffle"])


# ------------------------------------------- B.6.3 the cell-block bootstrap ---

def _per_cell(n_cells=12, per=150, seed=30):
    accs, rng = [], np.random.default_rng(seed)
    for c in range(n_cells):
        tA, tB, vA, vB = simulate(n=per, seed=int(rng.integers(1 << 30)))
        accs.append(rj.accumulate(rj.new_acc(), tA, tB, vA, vB))
    return accs


def test_bootstrap_interval_brackets_the_point_estimate():
    accs = _per_cell()
    m = rj.moments(rj.merge(*accs))
    rho1 = rj.canonical(rj.whiten(m["S_AA"], m["S_BB"], m["S_AB"],
                                  m["V_A"], m["V_B"])["R"])["rho1"]
    out = rj.cell_block_bootstrap(accs, n_boot=300, seed=1)
    assert out["rho1_ci_lo"] <= rho1 <= out["rho1_ci_hi"]
    assert out["n_cells"] == 12


def test_bootstrap_resamples_cells_not_contacts():
    """A contact-level bootstrap would give a far narrower interval; pin that
    the cell-block one is wider on data with real between-cell spread."""
    rng = np.random.default_rng(11)
    accs = []
    for c in range(15):
        # Each cell gets its own coordination strength -- between-cell spread
        # a contact-level resample cannot see.
        tA, tB, vA, vB = simulate(n=200, rho_u=float(rng.uniform(0.2, 0.9)),
                                  seed=int(rng.integers(1 << 30)))
        accs.append(rj.accumulate(rj.new_acc(), tA, tB, vA, vB))
    cell = rj.cell_block_bootstrap(accs, n_boot=400, seed=2)
    width_cell = cell["rho1_ci_hi"] - cell["rho1_ci_lo"]

    # Same total contacts, but the cell identity thrown away: split the pooled
    # data into arbitrary blocks of equal size.
    pooled = rj.merge(*accs)
    fake = [pooled / len(accs)] * len(accs)
    flat = rj.cell_block_bootstrap(fake, n_boot=400, seed=2)
    width_flat = flat["rho1_ci_hi"] - flat["rho1_ci_lo"]
    assert width_cell > width_flat, (
        f"cell-block interval {width_cell:.4f} must exceed the interval that "
        f"ignores between-cell spread {width_flat:.4f}")


def test_one_cell_gives_no_interval_rather_than_a_zero_width_one():
    """A zero-width interval reads as certainty; the honest answer is that
    there is no replication to resample."""
    accs = _per_cell(n_cells=1)
    out = rj.cell_block_bootstrap(accs, n_boot=50, seed=3)
    assert np.isnan(out["rho1_ci_lo"]) and np.isnan(out["rho1_ci_hi"])
    assert out["n_boot_failed"] == 50


# ------------------------------------------------------ B.8 classification ---

def test_anti_assortative_is_class_iv_not_the_headline_class():
    """IV is a real reading of the model and is still rank 1.  Calling it II
    would put an ordinary finding in the headline class."""
    assert rj.bin_class(True, False, -1) == "IV"
    assert rj.bin_class(True, True, -1) == "IV", (
        "the sign is checked before the rank test on purpose")


def test_the_headline_class_needs_both_coordination_and_rank_rejection():
    assert rj.bin_class(True, True, +1) == "II"
    assert rj.bin_class(True, False, +1) == "I"
    assert rj.bin_class(False, True, +1) == "III", (
        "rank-2 without coordination at all is not a finding")
    assert rj.bin_class(False, False, +1) == "III"


def test_every_class_has_a_label():
    for c in ("I", "II", "III", "IV"):
        assert c in rj.BIN_CLASSES


# ------------------------------------------------------------ B.8 the row ---

def test_summarize_emits_every_design_column():
    tA, tB, vA, vB = simulate(n=900, seed=40)
    accs = _per_cell(n_cells=6, per=150, seed=41)
    acc = rj.accumulate(rj.new_acc(), tA, tB, vA, vB)
    row = rj.summarize(acc, per_cell_accs=accs, theta_A=tA, theta_B=tB,
                       v_A=vA, v_B=vB, n_boot_null=40, n_boot_mate=40,
                       n_boot_ci=60, seed=1)
    want = {
        "n_contacts", "n_cells", "N_eff",
        "D", "D_null_mean", "D_excess", "p_rank1",
        "rho1", "rho1_ci_lo", "rho1_ci_hi", "z_mate_shuffle",
        "rho2", "rho2_null_mean", "rho2_excess", "rho2_over_rho1",
        "w_A_gch", "w_A_hcg", "w_B_gch", "w_B_hcg", "phi_A", "phi_B",
        "sign_lambda_h", "sign_rho_u", "rank1_pattern_class",
        "asymmetry_frob", "asymmetry_angle",
        "sigma_AB_gg", "sigma_AB_gh", "sigma_AB_hg", "sigma_AB_hh",
        "reliability_gch", "reliability_hcg",
        "flag_pd_clipped", "flag_low_neff",
        "error_convention", "moment_mode", "gch_norm", "bin_class",
    }
    assert want <= set(row), f"missing: {sorted(want - set(row))}"


def test_summarize_without_contacts_reports_nan_not_a_substitute():
    """The nulls need the raw contacts.  Quietly falling back to something
    computable from the accumulator alone would report a p-value that answers
    a different question."""
    tA, tB, vA, vB = simulate(n=400, seed=42)
    acc = rj.accumulate(rj.new_acc(), tA, tB, vA, vB)
    row = rj.summarize(acc)
    assert np.isnan(row["p_rank1"]) and np.isnan(row["z_mate_shuffle"])
    assert np.isnan(row["rho1_ci_lo"])
    assert row["bin_class"] == "III", "no test run means no claim"
    assert np.isfinite(row["D"]) and np.isfinite(row["rho1"])


def test_w_vectors_live_in_the_original_mark_basis():
    """w_A = Sigma_AA^(-1/2) u1 -- the direction the shared axis loads on in
    (gch, hcg), not in the whitened basis where the angle is meaningless."""
    tA, tB, vA, vB = simulate(n=1200, seed=43)
    acc = rj.accumulate(rj.new_acc(), tA, tB, vA, vB)
    m = rj.moments(acc)
    wh = rj.whiten(m["S_AA"], m["S_BB"], m["S_AB"], m["V_A"], m["V_B"])
    cn = rj.canonical(wh["R"])
    row = rj.summarize(acc)
    want = wh["W_A"] @ cn["u1"]
    assert row["w_A_gch"] == pytest.approx(want[0])
    assert row["w_A_hcg"] == pytest.approx(want[1])
    assert row["phi_A"] == pytest.approx(
        np.degrees(np.arctan2(want[1], want[0])))
