"""A summary that separates coordinated-balanced from uncoordinated.

C and (F - F_exp) are each blind where the other sees.  C cannot tell
coordinated co-open from coordinated co-closed, because Cov is unchanged by
swapping open for closed.  (F - F_exp) cannot tell coordinated co-open from
UNCOORDINATED, because it is a ratio describing where the concordant mass
sits and does not know whether that mass exceeds chance.

No monotone function of one recovers the other -- the state is genuinely
two-dimensional -- so the summary changes COORDINATES rather than reducing
dimension: R is the distance from the null in either sense, theta is which
kind of departure.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from nome_hic_concordance.concordance import (COORD_CLASSES,
                                              coordination_class,
                                              coordination_polar,
                                              _robust_sd)

# the four cells of the taxonomy, in standardised units
CASES = {
    "uncoordinated":         (0.00,  0.00),
    "coordinated balanced":  (1.00,  0.00),
    "co-open shifted":       (0.70,  0.70),
    "co-closed shifted":     (0.70, -0.70),
    "anti-coordinated":      (-1.00, 0.00),
    "direction only":        (0.00,  1.00),
}


def _polar(cases, spread=0.30, n=400, seed=0):
    """Embed the cases in a cloud so the robust scales have something to
    estimate, then read each case back out."""
    rng = np.random.default_rng(seed)
    bg_c = rng.normal(0, spread, n)
    bg_f = rng.normal(0, spread, n)
    C = np.concatenate([bg_c, [v[0] for v in cases.values()]])
    F = np.concatenate([bg_f, [v[1] for v in cases.values()]])
    out = coordination_polar(C, F)
    k = len(cases)
    return {name: (out["R"][-k + i], out["theta"][-k + i], out["p"][-k + i])
            for i, name in enumerate(cases)}, out


# ------------------------------------------------- the asked-for property --

def test_R_separates_coordinated_balanced_from_uncoordinated():
    """The whole request.  Neither C nor dF alone does this: C sees them as
    different but cannot name the difference for the other two cells, and dF
    puts BOTH at zero."""
    got, _ = _polar(CASES)
    assert got["uncoordinated"][0] < 1.0
    assert got["coordinated balanced"][0] > 2.5
    assert got["coordinated balanced"][0] > 3 * got["uncoordinated"][0]


def test_R_is_large_for_every_coordinated_state():
    """A magnitude that missed one of them would be no better than C."""
    got, _ = _polar(CASES)
    for name in ("coordinated balanced", "co-open shifted",
                 "co-closed shifted", "anti-coordinated", "direction only"):
        assert got[name][0] > 2.5, name


def test_theta_names_which_kind_of_departure():
    got, _ = _polar(CASES)
    assert abs(got["coordinated balanced"][1]) < 15          # ~0 deg
    assert 30 < got["co-open shifted"][1] < 60                # ~+45
    assert -60 < got["co-closed shifted"][1] < -30            # ~-45
    assert abs(abs(got["anti-coordinated"][1]) - 180) < 15    # ~180
    assert abs(got["direction only"][1] - 90) < 15            # ~+90


def test_the_two_states_C_cannot_separate_get_opposite_angles():
    """C's blind spot, closed."""
    got, _ = _polar(CASES)
    assert got["co-open shifted"][0] == pytest.approx(
        got["co-closed shifted"][0], rel=0.2)
    assert got["co-open shifted"][1] * got["co-closed shifted"][1] < 0


def test_each_axis_alone_collapses_a_pair_the_plane_separates():
    """The complementarity, stated as the two collapses it repairs.

    Not a claim about radius: 'coordinated co-open' at (0.7, 0.7) and
    'direction only' at (0, 1.0) are equidistant from the origin by
    construction, and should be -- both depart from the null by the same
    amount.  What differs is the KIND of departure, which is the angle.
    """
    got, _ = _polar(CASES)

    # C is blind here: co-open and co-closed carry identical C ...
    assert CASES["co-open shifted"][0] == CASES["co-closed shifted"][0]
    # ... and the plane splits them by sign of angle
    assert got["co-open shifted"][1] * got["co-closed shifted"][1] < 0

    # dF is blind here: balanced and uncoordinated both sit at dF = 0 ...
    assert CASES["coordinated balanced"][1] == CASES["uncoordinated"][1] == 0
    # ... and the plane splits them by radius
    assert got["coordinated balanced"][0] > 3 * got["uncoordinated"][0]


def test_a_pure_direction_departure_is_a_right_angle_from_a_pure_magnitude_one():
    """'Coordinated balanced' and 'direction only' are the two axes of the
    plane.  Each is exactly what the OTHER summary statistic cannot see, so
    they had better not land in the same place."""
    got, _ = _polar(CASES)
    sep = abs(got["direction only"][1] - got["coordinated balanced"][1])
    assert 70 < sep < 110


# --------------------------------------------------------- the class codes --

def test_classes_recover_the_taxonomy():
    got, out = _polar(CASES)
    for name, want in (("uncoordinated", "uncoordinated"),
                       ("coordinated balanced", "coordinated balanced"),
                       ("co-open shifted", "co-open shifted"),
                       ("co-closed shifted", "co-closed shifted"),
                       ("anti-coordinated", "anti-coordinated")):
        R, th, _ = got[name]
        code = coordination_class(np.array([R]), np.array([th]))[0]
        assert COORD_CLASSES[code] == want, f"{name} -> {COORD_CLASSES[code]}"


def test_a_bin_at_the_origin_is_never_given_a_direction():
    """The failure mode dF alone has.  A bin with no coordination has no
    direction to report, whatever angle its noise happens to point."""
    rng = np.random.default_rng(3)
    th = rng.uniform(-180, 180, 500)
    R = rng.uniform(0, 1.5, 500)          # all inside r_cut
    code = coordination_class(R, th, r_cut=2.0)
    assert (code == 0).all()


def test_nan_in_either_axis_stays_nan():
    C = np.array([1.0, np.nan, 2.0, 3.0] * 40)
    F = np.array([0.5, 0.5, np.nan, 0.2] * 40)
    out = coordination_polar(C, F)
    bad = ~(np.isfinite(C) & np.isfinite(F))
    assert np.isnan(out["R"][bad]).all()
    assert np.isnan(out["theta"][bad]).all()
    assert (coordination_class(out["R"], out["theta"])[bad] == -1).all()


# ------------------------------------------------------------ calibration --

def test_p_is_calibrated_against_a_null():
    """R^2 is chi-square(2) when both axes are null, so p should be uniform
    -- that is what makes it a test rather than a score."""
    rng = np.random.default_rng(7)
    n = 60000
    out = coordination_polar(rng.normal(0, 1, n), rng.normal(0, 1, n))
    p = out["p"]
    assert abs(float(np.mean(p < 0.05)) - 0.05) < 0.01
    assert abs(float(np.mean(p < 0.01)) - 0.01) < 0.005
    assert abs(float(np.median(p)) - 0.5) < 0.03


def test_correlated_axes_are_whitened_before_the_radius():
    """Unwhitened, a bin that is ordinary along the correlated direction
    would be called extreme for being ordinary."""
    rng = np.random.default_rng(11)
    n = 40000
    a = rng.normal(0, 1, n)
    b = 0.9 * a + np.sqrt(1 - 0.81) * rng.normal(0, 1, n)
    out = coordination_polar(a, b)
    assert out["rho"] == pytest.approx(0.9, abs=0.05)
    # whitened, the null is still calibrated despite the correlation
    assert abs(float(np.mean(out["p"] < 0.05)) - 0.05) < 0.012


def test_the_scale_is_robust_to_a_heavy_tail():
    """Thin bins produce huge C; a plain sd would let them set the units for
    the whole matrix and push every ordinary bin toward the origin."""
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 5000)
    x[:20] = 500.0
    assert _robust_sd(x) == pytest.approx(1.0, abs=0.1)
    assert np.std(x) > 5


def test_passing_a_calibration_reuses_it():
    """So two runs can be put on one plane rather than each setting its own
    units and being silently incomparable."""
    rng = np.random.default_rng(13)
    a = coordination_polar(rng.normal(0, 1, 3000), rng.normal(0, 1, 3000))
    b = coordination_polar(np.array([2.0, 0.0]), np.array([0.0, 2.0]),
                           scale_c=a["scale_c"], scale_f=a["scale_f"],
                           rho=a["rho"])
    assert b["scale_c"] == a["scale_c"] and b["rho"] == a["rho"]


def test_a_flat_axis_is_reported_not_divided_by():
    out = coordination_polar(np.ones(500), np.arange(500.0))
    assert np.isnan(out["R"]).all()
    assert "note" in out


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="shape"):
        coordination_polar(np.zeros((4, 5)), np.zeros((4, 6)))


# ------------------------------------------- what reaches the UMAP step --

_CLI = (Path(__file__).resolve().parent.parent / "nome_hic_concordance"
        / "cli.py")


def _umap_components():
    """The component names step 6/7 builds companion UMAPs for."""
    import re
    src = _CLI.read_text()
    block = src[src.index("for comp, comp_title in ("):]
    block = block[:block.index("):")]
    return re.findall(r'\("([A-Za-z_]+)",', block)


def test_polar_R_reaches_the_umap_step():
    """Emitting a matrix that step 6 never reads would mean the statistic
    exists only in the parquet nobody opens."""
    assert "polar_R" in _umap_components()


def test_polar_p_is_deliberately_not_embedded():
    """p = exp(-R^2/2) is a monotone function of R, so a UMAP on it answers
    the same question with worse conditioning: it is crushed toward zero for
    everything interesting, leaving the null bins to set the distances."""
    assert "polar_p" not in _umap_components()
    R = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    p = np.exp(-R ** 2 / 2)
    assert np.all(np.diff(p) < 0)            # strictly monotone in R
    assert p[-1] < 1e-3 and p[-2] < 0.02     # and crushed where it matters


def test_polar_theta_degrees_are_deliberately_not_embedded():
    """Circular.  Euclidean distance on degrees tears the circle at +-180,
    which is exactly the anti-coordinated sector."""
    assert "polar_theta" not in _umap_components()
    assert abs(179.0 - (-179.0)) == 358.0            # what Euclid sees
    d = abs(179.0 - (-179.0))
    assert min(d, 360.0 - d) == 2.0                  # what is true


def test_the_sine_and_cosine_pair_restores_the_true_separation():
    """The representation a direction embedding has to use instead."""
    def xy(deg):
        r = np.radians(deg)
        return np.array([np.sin(r), np.cos(r)])
    near = np.linalg.norm(xy(179) - xy(-179))
    far = np.linalg.norm(xy(90) - xy(-90))
    assert near < 0.1                # adjacent on the circle, adjacent here
    assert far == pytest.approx(2.0, abs=1e-9)       # genuinely opposite
    assert near < far
