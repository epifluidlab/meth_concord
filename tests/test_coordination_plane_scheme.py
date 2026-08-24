"""The five-state panel: R and theta against every scalar that collapses.

The figure's whole claim is arithmetic, not artistic: each scalar summary
gives two different states the same value, and only the pair does not.  That
claim is checkable, so it is checked here rather than trusted to whoever
last edited the drawing.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "coordination_plane_scheme.py")


def _mod():
    spec = importlib.util.spec_from_file_location("plane", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["plane"] = m
    spec.loader.exec_module(m)
    return m


def _table(m):
    """(label, C, dF, D, R, theta) for each drawn state."""
    out = []
    for lab, c, f, _col, _o, _h, _v in m.STATES:
        R = float(np.hypot(c, f))
        th = np.degrees(np.arctan2(f, c)) if R > 1e-9 else np.nan
        out.append((lab, c, f, c * m.D_MULT, R, th))
    return out


def test_five_states_one_per_sector():
    m = _mod()
    assert len(m.STATES) == 5
    rows = _table(m)
    far = [(lab, th) for lab, _c, _f, _d, R, th in rows if R >= m.R_CUT]
    assert len(far) == 4                       # the fifth is the origin
    sectors = set()
    for lab, th in far:
        a = abs(th)
        sectors.add("balanced" if a <= m.SECTOR else
                    "anti" if a >= 180 - m.SECTOR else
                    "co-open" if th > 0 else "co-closed")
    assert sectors == {"balanced", "anti", "co-open", "co-closed"}


@pytest.mark.parametrize("col,idx,collide", [
    ("C", 1, {"co-open shifted", "co-closed shifted"}),
    ("dF", 2, {"uncoordinated", "coordinated balanced", "anti-coordinated"}),
])
def test_each_scalar_collapses_the_pair_the_figure_says_it_does(col, idx,
                                                                collide):
    """If a scalar stopped collapsing its pair the panel would be making a
    claim that is no longer true."""
    m = _mod()
    rows = _table(m)
    seen = {}
    for r in rows:
        seen.setdefault(round(r[idx], 6), set()).add(r[0])
    clashed = set().union(*[v for v in seen.values() if len(v) > 1]) \
        if any(len(v) > 1 for v in seen.values()) else set()
    assert collide <= clashed, f"{col}: expected {collide}, got {clashed}"


def test_R_alone_also_collapses_two_pairs():
    """The reason theta is needed: equidistant states are indistinguishable
    on a radius."""
    m = _mod()
    rows = _table(m)
    by_r = {}
    for lab, _c, _f, _d, R, _th in rows:
        by_r.setdefault(round(R, 2), set()).add(lab)
    pairs = [v for v in by_r.values() if len(v) > 1]
    assert {"co-open shifted", "co-closed shifted"} in pairs
    assert {"coordinated balanced", "anti-coordinated"} in pairs


def test_the_pair_separates_all_five():
    """The claim the figure exists to make."""
    m = _mod()
    keys = set()
    for lab, _c, _f, _d, R, th in _table(m):
        keys.add((round(R, 2), "-" if not np.isfinite(th) else round(th, 0)))
    assert len(keys) == 5


def test_polar_p_collapses_too_which_is_why_theta_is_shown_beside_it():
    """p answers 'how far from the null', never 'which way'.  Balanced and
    anti-coordinated are equidistant, so they share a p exactly -- and a
    reader given only p would call them the same state."""
    m = _mod()
    pv = {}
    for lab, _c, _f, _d, R, _th in _table(m):
        pv[lab] = float(np.exp(-R * R / 2))
    assert pv["coordinated balanced"] == pytest.approx(
        pv["anti-coordinated"], rel=1e-9)
    assert pv["co-open shifted"] == pytest.approx(
        pv["co-closed shifted"], rel=1e-9)
    assert pv["uncoordinated"] > 0.9          # the null looks like the null


def test_the_radius_cut_is_a_weak_threshold_and_the_figure_says_so():
    """R_CUT = 2 is p = 0.135 -- not significant by any standard.  It is a
    naming default for the sector diagram, and the panel must not imply it
    is a test, because with millions of cell-bins the real cut is FDR on
    polar_p and lands near R = 5.3."""
    m = _mod()
    p_at_cut = float(np.exp(-m.R_CUT ** 2 / 2))
    assert 0.10 < p_at_cut < 0.20
    src = _SCRIPT.read_text()
    assert "class default, not a test" in src
    assert "p=0.05" in src or "0.05" in src


def test_D_inverts_the_sign_of_C_at_the_measured_marginal():
    """The trap the panel warns about: with F = 0.323, D is a negative
    multiple of C, so coordination falling shows up as D rising."""
    m = _mod()
    assert m.D_MULT < 0
    assert m.D_MULT == pytest.approx(-0.354, abs=0.01)
    for lab, c, _f, d, _R, _th in _table(m):
        if abs(c) > 1e-9:
            assert np.sign(d) == -np.sign(c), lab


def test_it_renders_a_vector_pdf(tmp_path):
    out = tmp_path / "plane.pdf"
    r = subprocess.run([sys.executable, str(_SCRIPT), "--output", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.read_bytes().startswith(b"%PDF")
    assert b"/Subtype /Image" not in out.read_bytes()


def test_it_renders_png(tmp_path):
    out = tmp_path / "plane.png"
    r = subprocess.run([sys.executable, str(_SCRIPT), "--output", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.stat().st_size > 4000


def test_the_sector_bounds_match_the_pipeline_classifier():
    """A figure whose sectors disagreed with coordination_class would be
    documenting a classifier nobody runs."""
    from nome_hic_concordance.concordance import (COORD_CLASSES,
                                                  coordination_class)
    m = _mod()
    for lab, _c, _f, _d, R, th in _table(m):
        code = coordination_class(np.array([R]),
                                  np.array([0.0 if not np.isfinite(th)
                                            else th]),
                                  r_cut=m.R_CUT, sector=m.SECTOR)[0]
        want = {"uncoordinated": "uncoordinated",
                "coordinated balanced": "coordinated balanced",
                "co-open shifted": "co-open shifted",
                "co-closed shifted": "co-closed shifted",
                "anti-coordinated": "anti-coordinated"}[lab]
        assert COORD_CLASSES[code] == want, lab


# --------------------------------------------------- the pair cartoons --

def _counts(layout):
    """(co-open, co-closed, mixed) from a drawing-order layout string."""
    return layout.count("O"), layout.count("C"), layout.count("M")


def _implied(layout, null):
    """(C, F, dF) implied by a state's pairs, read against the null."""
    o, c, _m = _counts(layout)
    M = o + c
    no, nc, _ = _counts(null)
    Me = no + nc
    Fe = no / Me
    F = o / M if M else float("nan")
    return M - Me, F, F - Fe


def test_every_state_draws_ten_pairs():
    """The cartoons are comparable only if each shows the same total."""
    m = _mod()
    assert len(m.PAIRS_NULL) == 10
    for lab, layout in m.PAIRS.items():
        assert len(layout) == 10, lab
        assert set(layout) <= set("OCM"), lab


def test_the_cartoons_imply_the_points_drawn_in_panel_a():
    """The two panels must be the same figure.  Cartoons chosen by eye
    would illustrate states that are not the ones plotted."""
    m = _mod()
    want_sign = {"uncoordinated": (0, 0), "coordinated balanced": (+1, 0),
                 "co-open shifted": (+1, +1), "co-closed shifted": (+1, -1),
                 "anti-coordinated": (-1, 0)}
    for lab, c, f, _col, _o, _h, _v in m.STATES:
        C, _F, dF = _implied(m.PAIRS[lab], m.PAIRS_NULL)
        sc, sf = want_sign[lab]
        assert np.sign(C) == sc, f"{lab}: C sign"
        assert np.sign(round(dF, 6)) == sf, f"{lab}: dF sign"
        # and the signs agree with the point plotted in panel a
        assert np.sign(C) == np.sign(c)
        assert np.sign(round(dF, 6)) == np.sign(round(f, 6))


def test_the_cartoons_are_the_extreme_of_each_category():
    """Chosen so a reader can see the category at a glance: every pair
    co-open, every pair co-closed, every pair concordant at the null ratio,
    and the smallest concordant count that still keeps that ratio."""
    m = _mod()
    assert _counts(m.PAIRS["co-open shifted"])[1] == 0     # no co-closed
    assert _counts(m.PAIRS["co-closed shifted"])[0] == 0   # no co-open
    no, nc, nm = _counts(m.PAIRS_NULL)
    for lab in ("coordinated balanced", "co-open shifted",
                "co-closed shifted"):
        assert _counts(m.PAIRS[lab])[2] < nm, lab          # less mixed
    o, c, mm = _counts(m.PAIRS["anti-coordinated"])
    assert o + c < no + nc                                 # a deficit
    assert mm > nm                                         # visibly so


def test_balanced_keeps_the_nulls_ratio_rather_than_splitting_fifty_fifty():
    """The distinction the whole re-referencing exists for.  Drawing
    'balanced' as half co-open would make it a co-open SHIFT at any marginal
    other than 0.5 -- which is the error the old D = C(2F-1) encodes."""
    m = _mod()
    o, c, _ = _counts(m.PAIRS["coordinated balanced"])
    no, nc, _ = _counts(m.PAIRS_NULL)
    assert o / (o + c) == pytest.approx(no / (no + nc))     # same ratio
    assert o / (o + c) != pytest.approx(0.5)                # not 50/50
    _C, _F, dF = _implied(m.PAIRS["coordinated balanced"], m.PAIRS_NULL)
    assert dF == pytest.approx(0.0, abs=1e-9)


def test_anti_coordinated_shows_fewer_concordant_pairs_than_the_null():
    """Anti-coordination is a deficit, not a direction.  A cartoon with more
    concordant pairs than the null would be showing coordination."""
    m = _mod()
    o, c, _ = _counts(m.PAIRS["anti-coordinated"])
    no, nc, _ = _counts(m.PAIRS_NULL)
    assert o + c < no + nc


def test_uncoordinated_is_drawn_identical_to_the_null():
    m = _mod()
    assert m.PAIRS["uncoordinated"] == m.PAIRS_NULL


def test_every_drawn_state_has_a_cartoon():
    m = _mod()
    assert {lab for lab, *_ in m.STATES} == set(m.PAIRS)
