"""Where the zero of the direction axis belongs.

D = C·(2F−1) puts "no directional preference" at F = 0.5, which is the right
zero only when the marginal is p-bar = 0.5.  Under independence at marginal
p-bar the concordant mass already splits F_indep = p^2/(p^2 + (1-p)^2) -- 0.31
at p-bar = 0.40.  So on sparse accessibility data a bin with NO preference
scores 2F-1 ~ -0.38, every coordinated bin reports co-closed, and D collapses
to a scaled -C.

D' = C·(F − F_exp) with F_exp = E_open/(E_open + E_closed) puts the zero
where the marginal-matched null puts it, which is what "balanced" is supposed
to mean.
"""

from __future__ import annotations

import numpy as np
import pytest

from nome_hic_concordance import concordance as cc


def _decompose(o_open, o_closed, e_open, e_closed):
    """Reach the nested _decompose through a tiny stand-in with the same
    arithmetic, so the test pins the FORMULA rather than a closure."""
    o_open, o_closed = np.atleast_1d(o_open), np.atleast_1d(o_closed)
    e_open, e_closed = np.atleast_1d(e_open), np.atleast_1d(e_closed)
    coord = (o_open + o_closed) - (e_open + e_closed)
    mass = o_open + o_closed
    frac = np.where(mass > 1e-12, o_open / np.where(mass > 1e-12, mass, 1.0), 0.5)
    e_mass = e_open + e_closed
    fexp = np.where(e_mass > 1e-12,
                    e_open / np.where(e_mass > 1e-12, e_mass, 1.0), 0.5)
    ok = (mass > 1e-12) & (e_mass > 1e-12)
    excess = np.where(ok, frac - fexp, 0.0)
    return {"coordination": coord, "direction": frac,
            "direction_expected": fexp, "direction_excess": excess,
            "directional": coord * (2.0 * frac - 1.0),
            "directional_corrected": coord * excess}


def _null(p, extra_concordant=0.0, toward_open=0.0):
    """Co-state rates at marginal p under independence, optionally with
    concordant mass added and shifted toward co-open.

    "No directional preference" means the added mass splits in the NULL'S OWN
    o:c ratio, not 50/50.  Adding equal absolute amounts pulls F toward 0.5
    and is itself a shift toward co-open -- which is the same confusion
    between "balanced" and "marginal-matched" that the F = 0.5 zero makes.
    """
    o, c = p * p, (1 - p) ** 2
    m = 1.0 - o - c
    e_open, e_closed = o, c
    take = min(extra_concordant, m)
    r = o / (o + c)                      # the null's own split
    o_open = o + take * (r + toward_open)
    o_closed = c + take * (1.0 - r - toward_open)
    return o_open, o_closed, e_open, e_closed


# ------------------------------------------------------- the failure mode --

@pytest.mark.parametrize("p", [0.10, 0.20, 0.40])
def test_the_old_axis_calls_an_unbiased_bin_co_closed(p):
    """Coordination added with NO directional preference.  The legacy D still
    reports co-closed, and more strongly the sparser the data."""
    r = _decompose(*_null(p, extra_concordant=0.15, toward_open=0.0))
    assert r["coordination"][0] > 0
    assert r["directional"][0] < 0, "legacy D should mislabel this"
    expected_bias = 2 * (p ** 2 / (p ** 2 + (1 - p) ** 2)) - 1
    assert expected_bias < 0


@pytest.mark.parametrize("p", [0.10, 0.20, 0.40])
def test_the_new_axis_calls_it_balanced(p):
    """Same bin, corrected axis: no preference reads as no preference."""
    r = _decompose(*_null(p, extra_concordant=0.15, toward_open=0.0))
    assert r["coordination"][0] > 0
    assert abs(r["direction_excess"][0]) < 1e-9
    assert abs(r["directional_corrected"][0]) < 1e-9


def test_the_expected_split_is_the_independence_value():
    for p in (0.05, 0.20, 0.4009):
        r = _decompose(*_null(p))
        want = p ** 2 / (p ** 2 + (1 - p) ** 2)
        assert r["direction_expected"][0] == pytest.approx(want, rel=1e-9)


# --------------------------------------------------------- it still signs --

def test_a_co_open_shift_is_positive_and_co_closed_negative():
    """The taxonomy the axis exists for: same C, opposite direction."""
    op = _decompose(*_null(0.40, extra_concordant=0.15, toward_open=+0.30))
    cl = _decompose(*_null(0.40, extra_concordant=0.15, toward_open=-0.30))
    assert op["coordination"][0] == pytest.approx(cl["coordination"][0])
    assert op["directional_corrected"][0] > 0
    assert cl["directional_corrected"][0] < 0
    # and they are symmetric about the null, which 2F-1 is not
    assert op["direction_excess"][0] == pytest.approx(
        -cl["direction_excess"][0], rel=1e-9)


def test_the_legacy_axis_is_not_symmetric_about_its_own_zero():
    """Why the old one cannot be rescued by a threshold: at p-bar = 0.40 both
    a co-open and a co-closed shift land on the same side of zero."""
    op = _decompose(*_null(0.40, extra_concordant=0.15, toward_open=+0.15))
    cl = _decompose(*_null(0.40, extra_concordant=0.15, toward_open=-0.15))
    assert op["directional"][0] < 0 and cl["directional"][0] < 0


def test_uncoordinated_stays_zero_on_both_axes():
    r = _decompose(*_null(0.40, extra_concordant=0.0))
    assert r["coordination"][0] == pytest.approx(0.0, abs=1e-12)
    assert r["directional_corrected"][0] == pytest.approx(0.0, abs=1e-12)


def test_coordinated_but_balanced_is_distinguishable_from_uncoordinated():
    """The pair of axes has to separate these two; that is why both are
    reported."""
    bal = _decompose(*_null(0.40, extra_concordant=0.15, toward_open=0.0))
    unc = _decompose(*_null(0.40, extra_concordant=0.0))
    assert bal["directional_corrected"][0] == pytest.approx(
        unc["directional_corrected"][0], abs=1e-12)
    assert bal["coordination"][0] > unc["coordination"][0] + 0.05


def test_an_empty_pool_reports_no_direction_rather_than_half():
    """F defaults to 0.5 with no concordant mass.  Subtracting F_exp from
    that default would invent a direction, so the excess is pinned to 0."""
    r = _decompose(0.0, 0.0, 0.16, 0.36)
    assert r["direction_excess"][0] == 0.0
    assert r["directional_corrected"][0] == 0.0
    r2 = _decompose(0.16, 0.36, 0.0, 0.0)
    assert r2["direction_excess"][0] == 0.0


# ---------------------------------------------------------- the real data --

def test_it_reproduces_the_measured_meningioma_numbers():
    """p-bar = 0.4009 and F = 0.3230 were measured on the GCH run.  The old
    axis says co-closed, the new one says co-open -- they disagree on
    DIRECTION, not merely on scale, which is what makes the correction worth
    re-running the pipeline for."""
    p, f_obs = 0.4009, 0.3230
    f_exp = p ** 2 / (p ** 2 + (1 - p) ** 2)
    assert f_exp == pytest.approx(0.3093, abs=5e-4)
    assert (2 * f_obs - 1) < 0
    assert (f_obs - f_exp) > 0


def test_the_module_emits_both_axes():
    src = (cc.__file__ and open(cc.__file__).read())
    for key in ("direction_expected", "direction_excess",
                "directional_corrected"):
        assert f'"{key}"' in src, key


# ------------------------------------------------- the co-state classifier --

def test_thresholding_F_at_half_empties_the_co_open_class():
    """The same bug in costate_codes, which is where it does visible damage:
    at p-bar = 0.40 the concordant mass splits 0.31 under independence, so a
    0.5 threshold puts every coordinated cell-bin in CO_LOW."""
    from nome_hic_concordance import costate as cs
    p = 0.4009
    f_exp = p ** 2 / (p ** 2 + (1 - p) ** 2)
    rng = np.random.default_rng(0)
    F = np.clip(rng.normal(f_exp, 0.02, (40, 60)), 0.0, 1.0)
    C = np.full(F.shape, 0.3)
    codes = cs.costate_codes(C, F)
    assert (codes == cs.CO_HIGH).sum() == 0, "0.5 threshold is unreachable"
    assert (codes == cs.CO_LOW).sum() == F.size


def test_thresholding_against_the_null_recovers_both_classes():
    from nome_hic_concordance import costate as cs
    p = 0.4009
    f_exp = p ** 2 / (p ** 2 + (1 - p) ** 2)
    rng = np.random.default_rng(0)
    F = np.clip(rng.normal(f_exp, 0.02, (40, 60)), 0.0, 1.0)
    C = np.full(F.shape, 0.3)
    codes = cs.costate_codes(C, F, expected=f_exp)
    hi = int((codes == cs.CO_HIGH).sum())
    lo = int((codes == cs.CO_LOW).sum())
    assert hi > 0 and lo > 0
    assert 0.3 < hi / (hi + lo) < 0.7, "symmetric noise should split evenly"


def test_the_null_reference_can_vary_per_cell_bin():
    """F_exp is not one number: the marginal differs by bin and by cell, so
    the reference has to be an array, not a constant."""
    from nome_hic_concordance import costate as cs
    C = np.full((2, 3), 0.3)
    F = np.array([[0.20, 0.35, 0.50], [0.20, 0.35, 0.50]])
    ref = np.array([[0.30, 0.30, 0.30], [0.10, 0.40, 0.60]])
    codes = cs.costate_codes(C, F, expected=ref)
    assert list(codes[0]) == [cs.CO_LOW, cs.CO_HIGH, cs.CO_HIGH]
    assert list(codes[1]) == [cs.CO_HIGH, cs.CO_LOW, cs.CO_LOW]


def test_an_uncoordinated_cell_bin_still_gets_no_direction():
    """The guard that was already right must survive the change."""
    from nome_hic_concordance import costate as cs
    codes = cs.costate_codes(np.array([[-0.1, 0.0]]), np.array([[0.9, 0.9]]),
                             expected=0.31)
    assert list(codes[0]) == [cs.CO_MIXED, cs.CO_MIXED]


def test_the_default_is_unchanged():
    """Existing callers must get exactly what they got before."""
    from nome_hic_concordance import costate as cs
    C = np.array([[0.3, 0.3, 0.3]])
    F = np.array([[0.2, 0.5, 0.8]])
    np.testing.assert_array_equal(cs.costate_codes(C, F),
                                  cs.costate_codes(C, F, expected=0.5))
