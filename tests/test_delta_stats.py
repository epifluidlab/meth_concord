"""Δ = linked − null: masking of thin bins, and the cell-level bootstrap.

Two things are being pinned here.

Masking: a bin can hold millions of linked site pairs and a few hundred null
ones.  Differencing an unmasked noisy null is exactly what puts a lone spike
into an otherwise smooth curve, so Δ is masked when EITHER side is thin.

The test: site pairs are not independent -- tens come off one read and ~1e6
reads off one cell -- so a Fisher z-test on the site-pair count would treat
~1e8 correlated observations as independent.  The resampling unit is the CELL.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "site_distance_correlation.py")


def _mod():
    spec = importlib.util.spec_from_file_location("sdc_delta", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sdc_delta"] = m
    spec.loader.exec_module(m)
    return m


def _acc(x, y, nb=1, b=0):
    """Pearson sufficient statistics (6, nb) from raw 0/1 vectors in bin `b`."""
    a = np.zeros((6, nb))
    x = np.asarray(x, float); y = np.asarray(y, float)
    a[0, b] = len(x); a[1, b] = x.sum(); a[2, b] = y.sum()
    a[3, b] = (x * x).sum(); a[4, b] = (y * y).sum(); a[5, b] = (x * y).sum()
    return a


def _corr_pair(rng, n, rho):
    """n draws of correlated 0/1 pairs with roughly the requested phi."""
    x = rng.integers(0, 2, n)
    flip = rng.random(n) > rho
    y = np.where(flip, rng.integers(0, 2, n), x)
    return x, y


# ---------------------------------------------------------------- masking ---

def test_delta_is_masked_when_the_null_side_is_thin():
    m = _mod()
    curves = {("gg", "linked"): (np.array([0.50, 0.40]), np.array([1e6, 1e6])),
              ("gg", "null"): (np.array([0.10, 0.20]), np.array([50.0, 1e6]))}
    d, ok = m._delta(curves, "gg", min_pairs=1000)
    assert not np.isfinite(d[0]), "thin null bin must be masked, not differenced"
    assert not ok[0] and ok[1]
    assert np.isclose(d[1], 0.20)


def test_delta_is_masked_when_the_linked_side_is_thin():
    m = _mod()
    curves = {("gg", "linked"): (np.array([0.50]), np.array([10.0])),
              ("gg", "null"): (np.array([0.10]), np.array([1e6]))}
    d, ok = m._delta(curves, "gg", min_pairs=1000)
    assert not np.isfinite(d[0]) and not ok[0]


def test_delta_survives_when_both_sides_are_thick():
    m = _mod()
    curves = {("gg", "linked"): (np.array([0.50]), np.array([1e6])),
              ("gg", "null"): (np.array([0.12]), np.array([1e6]))}
    d, ok = m._delta(curves, "gg", min_pairs=1000)
    assert ok[0] and np.isclose(d[0], 0.38)


# -------------------------------------------------------------- bootstrap ---

def test_bootstrap_needs_at_least_three_cells():
    m = _mod()
    rng = np.random.default_rng(0)
    a = [_acc(*_corr_pair(rng, 5000, 0.5)) for _ in range(2)]
    b = [_acc(*_corr_pair(rng, 5000, 0.0)) for _ in range(2)]
    assert m._bootstrap_delta(a, b, 100, 200, 42) is None


def test_bootstrap_flags_a_real_difference():
    """linked strongly correlated, null not: Δ should be significant."""
    m = _mod()
    rng = np.random.default_rng(1)
    a = [_acc(*_corr_pair(rng, 20_000, 0.60)) for _ in range(12)]
    b = [_acc(*_corr_pair(rng, 20_000, 0.00)) for _ in range(12)]
    lo, hi, p = m._bootstrap_delta(a, b, 100, 500, 42)
    assert lo[0] > 0, "95% CI for Δ should exclude zero"
    assert p[0] < 0.05, f"expected a significant Δ, got p={p[0]}"


def test_bootstrap_does_not_flag_a_null_difference():
    """Both arms from the SAME distribution: Δ must not come out significant."""
    m = _mod()
    rng = np.random.default_rng(2)
    a = [_acc(*_corr_pair(rng, 20_000, 0.30)) for _ in range(12)]
    b = [_acc(*_corr_pair(rng, 20_000, 0.30)) for _ in range(12)]
    lo, hi, p = m._bootstrap_delta(a, b, 100, 500, 42)
    assert lo[0] <= 0 <= hi[0], "CI should straddle zero when there is no effect"
    assert p[0] > 0.05, f"false positive: p={p[0]}"


def test_bootstrap_respects_min_pairs():
    """A bin too thin to correlate stays NaN through the bootstrap."""
    m = _mod()
    rng = np.random.default_rng(3)
    a = [_acc(*_corr_pair(rng, 20, 0.6)) for _ in range(8)]
    b = [_acc(*_corr_pair(rng, 20, 0.0)) for _ in range(8)]
    out = m._bootstrap_delta(a, b, 10_000, 200, 42)   # min_pairs far above n
    assert out is not None
    _lo, _hi, p = out
    assert not np.isfinite(p[0]), "p must not be reported for a masked bin"


def test_stars_thresholds():
    m = _mod()
    assert m._stars(0.0005) == "***"
    assert m._stars(0.005) == "**"
    assert m._stars(0.03) == "*"
    assert m._stars(0.2) == ""
    assert m._stars(np.nan) == ""


# ------------------------------------------------------------- axis floor ---

def test_plot_starts_at_the_requested_distance(tmp_path):
    """--plot-min-distance must clip the x-axis, not just hide markers.

    Below ~10bp the site pairs are adjacent-motif artefacts; the distance-1
    HCG-HCG point in particular is a single CpG dyad, visible only when the two
    mates disagree on conversion strand.  Those bins stay in the TSV.
    """
    import matplotlib
    matplotlib.use("Agg")
    m = _mod()
    edges = m._build_edges(1, 1_000_000, 8)
    centers = np.sqrt(edges[:-1] * edges[1:])
    nb = len(centers)
    curves = {}
    for key, _ in m._MEASURES:
        for cond, _lab in m._CONDS:
            curves[(key, cond)] = (np.full(nb, 0.3), np.full(nb, 1e6))
    out = tmp_path / "f.png"
    m._plot(curves, None, centers, edges, "t", 3, 100, out, 100, plot_min=10.0)
    assert out.exists()

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m._logx(ax, 10.0, 1e6)
    assert ax.get_xlim()[0] == 10.0, "axis must start exactly at the floor"
    plt.close(fig)


def test_matched_is_kept_out_of_the_plotted_conditions():
    """It stays in _CONDS (so it reaches the TSV) but must not be drawn."""
    m = _mod()
    assert "matched" in dict(m._CONDS)
    assert "matched" not in m._PLOT_CONDS
    assert m._PLOT_CONDS == ("linked", "null")
