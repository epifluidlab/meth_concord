"""Association measures for 0/1 calls.

Pearson r on binary data IS the phi coefficient, and phi is bounded by the
marginals: with HCG ~0.78 and GCH ~0.25 it can only occupy [-0.92, +0.31], and
a FIXED odds ratio of 9 reads anywhere from 0.10 to 0.50 depending on the rates
alone.  That makes phi incomparable across measures with different methylation
rates -- which is exactly the GCH-GCH vs HCG-GCH comparison the figures invite.

Yule's Q and the log odds ratio are functions of the odds ratio and are
invariant to the marginals, so these tests pin that invariance and the phi
ceiling that motivates it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "site_distance_correlation.py")


def _mod():
    spec = importlib.util.spec_from_file_location("sdc_metrics", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sdc_metrics"] = m
    spec.loader.exec_module(m)
    return m


def _acc_from_xy(x, y, nb=1, b=0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    a = np.zeros((6, nb))
    a[0, b] = len(x); a[1, b] = x.sum(); a[2, b] = y.sum()
    a[3, b] = (x * x).sum(); a[4, b] = (y * y).sum(); a[5, b] = (x * y).sum()
    return a


def _acc_from_table(n11, n10, n01, n00, nb=1, b=0):
    """The accumulator IS the 2x2 table for binary data."""
    a = np.zeros((6, nb))
    a[0, b] = n11 + n10 + n01 + n00
    a[1, b] = n11 + n10          # Sx = row total
    a[2, b] = n11 + n01          # Sy = column total
    a[3, b] = n11 + n10          # x*x == x
    a[4, b] = n11 + n01
    a[5, b] = n11                # Sxy = n11
    return a


# --------------------------------------------------------------- identities --

def test_phi_equals_pearson_r_on_binary_data():
    m = _mod()
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 20_000)
    y = np.where(rng.random(20_000) < 0.7, x, 1 - x)
    acc = _acc_from_xy(x, y)
    phi = m._metric(acc, 1, "phi")[0][0]
    assert phi == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-12)


def test_table_is_recovered_from_the_sufficient_statistics():
    """No extra storage was needed — Sxy is n11 and the margins close it."""
    m = _mod()
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, 5_000)
    y = np.where(rng.random(5_000) < 0.6, x, 1 - x)
    a, b, c, d, n = m._table(_acc_from_xy(x, y))
    assert int(a[0]) == int(((x == 1) & (y == 1)).sum())
    assert int(b[0]) == int(((x == 1) & (y == 0)).sum())
    assert int(c[0]) == int(((x == 0) & (y == 1)).sum())
    assert int(d[0]) == int(((x == 0) & (y == 0)).sum())
    assert int(n[0]) == len(x)


# ------------------------------------------------------------- the ceiling --

def test_phi_cannot_reach_one_when_marginals_differ():
    """Perfect achievable association, skewed marginals: phi stays far below 1."""
    m = _mod()
    # p(x=1)=0.8, p(y=1)=0.2, association as strong as those margins allow
    acc = _acc_from_table(n11=2000, n10=6000, n01=0, n00=2000)
    phi = m._metric(acc, 1, "phi")[0][0]
    _pmin, pmax = m._phi_envelope(acc)
    assert phi < 0.55, f"phi={phi} — the point is that it cannot approach 1"
    assert phi == pytest.approx(pmax[0], rel=1e-9), \
        "with an empty cell phi should sit exactly on its attainable bound"


def test_envelope_matches_the_closed_form():
    m = _mod()
    acc = _acc_from_table(n11=780, n10=220, n01=250, n00=750)
    a, b, c, d, n = m._table(acc)
    p1 = float((a + b)[0] / n[0]); p2 = float((a + c)[0] / n[0])
    lo, hi = min(p1, p2), max(p1, p2)
    want_max = np.sqrt(lo * (1 - hi) / ((1 - lo) * hi))
    want_min = (-np.sqrt(p1 * p2 / ((1 - p1) * (1 - p2))) if p1 + p2 <= 1
                else -np.sqrt((1 - p1) * (1 - p2) / (p1 * p2)))
    pmin, pmax = m._phi_envelope(acc)
    assert pmax[0] == pytest.approx(want_max)
    assert pmin[0] == pytest.approx(want_min)


def test_phi_is_always_inside_its_envelope():
    m = _mod()
    rng = np.random.default_rng(5)
    for _ in range(200):
        t = rng.integers(1, 500, 4)
        acc = _acc_from_table(*t)
        phi = m._metric(acc, 1, "phi")[0][0]
        pmin, pmax = m._phi_envelope(acc)
        assert pmin[0] - 1e-9 <= phi <= pmax[0] + 1e-9, (t, phi, pmin, pmax)


# --------------------------------------------------- marginal independence --

def _table_with(p1, p2, odds):
    """2x2 with the given marginals and odds ratio (proportions)."""
    from scipy.optimize import brentq

    def f(a):
        b, c, d = p1 - a, p2 - a, 1 - p1 - p2 + a
        if min(a, b, c, d) <= 1e-12:
            return 1e9
        return (a * d) / (b * c) - odds
    lo = max(1e-9, p1 + p2 - 1) + 1e-9
    hi = min(p1, p2) - 1e-9
    a = brentq(f, lo, hi)
    return a, p1 - a, p2 - a, 1 - p1 - p2 + a


def test_yule_q_is_invariant_to_the_marginals_but_phi_is_not():
    """The whole reason the alternative metrics exist.

    Same odds ratio throughout; only the methylation rates change.
    """
    pytest.importorskip("scipy")
    m = _mod()
    phis, qs = [], []
    for p1, p2 in ((0.5, 0.5), (0.25, 0.25), (0.78, 0.78), (0.78, 0.25)):
        a, b, c, d = _table_with(p1, p2, odds=9.0)
        acc = _acc_from_table(a * 1e6, b * 1e6, c * 1e6, d * 1e6)
        phis.append(m._metric(acc, 1, "phi")[0][0])
        qs.append(m._metric(acc, 1, "yule_q")[0][0])
    assert max(qs) - min(qs) < 0.01, f"Yule's Q should be constant, got {qs}"
    assert max(phis) - min(phis) > 0.20, (
        f"phi should swing with the marginals, got {phis}")


def test_log_or_is_the_log_of_the_odds_ratio():
    m = _mod()
    acc = _acc_from_table(400, 100, 100, 400)      # OR = 16
    got = m._metric(acc, 1, "log_or")[0][0]
    a, b, c, d = 400.5, 100.5, 100.5, 400.5       # Haldane-Anscombe
    assert got == pytest.approx(np.log(a * d / (b * c)))


def test_yule_q_and_log_or_agree_in_sign_with_phi():
    """The metrics disagree in MAGNITUDE by design, but not in direction.

    Only checked away from zero: phi uses raw counts while Q and log-OR carry
    the Haldane +0.5, so on a table with essentially no association (|phi| of a
    few 1e-4) the correction alone can flip the sign of a number that is zero
    either way.  That is the correction working, not a disagreement.
    """
    m = _mod()
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(300):
        t = rng.integers(5, 400, 4)
        acc = _acc_from_table(*t)
        phi = m._metric(acc, 1, "phi")[0][0]
        if abs(phi) < 0.02:                     # numerically no association
            continue
        checked += 1
        q = m._metric(acc, 1, "yule_q")[0][0]
        lor = m._metric(acc, 1, "log_or")[0][0]
        assert np.sign(phi) == np.sign(q) == np.sign(lor), (t, phi, q, lor)
    assert checked > 50, "the sample should contain real associations to check"


def test_haldane_correction_survives_an_empty_cell():
    """A zero cell must not send the odds ratio to 0 or infinity."""
    m = _mod()
    acc = _acc_from_table(500, 0, 100, 400)
    for metric in ("yule_q", "log_or"):
        v = m._metric(acc, 1, metric)[0][0]
        assert np.isfinite(v), f"{metric} blew up on an empty cell"


def test_phi_norm_is_phi_scaled_to_its_bound():
    m = _mod()
    acc = _acc_from_table(780, 220, 250, 750)
    phi = m._metric(acc, 1, "phi")[0][0]
    pn = m._metric(acc, 1, "phi_norm")[0][0]
    _pmin, pmax = m._phi_envelope(acc)
    assert pn == pytest.approx(phi / pmax[0])
    assert abs(pn) <= 1.0 + 1e-9


# --------------------------------------------------------------- plumbing ---

def test_metric_handles_the_stacked_bootstrap_shape():
    """The bootstrap must use the same code path as the point estimate."""
    m = _mod()
    rng = np.random.default_rng(11)
    accs = np.stack([_acc_from_table(*rng.integers(50, 500, 4)) for _ in range(7)])
    assert accs.shape == (7, 6, 1)
    for metric in m._METRICS:
        v, n = m._metric(accs, 1, metric)
        assert v.shape == (7, 1)
        one = m._metric(accs[3], 1, metric)[0]
        assert v[3] == pytest.approx(one)


def test_min_pairs_masks_every_metric():
    m = _mod()
    acc = _acc_from_table(5, 5, 5, 5)
    for metric in m._METRICS:
        assert not np.isfinite(m._metric(acc, 10_000, metric)[0][0])


def test_unknown_metric_is_rejected():
    m = _mod()
    with pytest.raises(ValueError, match="unknown metric"):
        m._metric(_acc_from_table(10, 10, 10, 10), 1, "spearman")


# --------------------------------------------------------- contig exclusion --

def test_exclude_chrom_drops_only_the_named_contigs():
    """A short high-copy contig (chrM) can dominate the bins below its length."""
    m = _mod()
    ctx = {"GCH": {"chr1": np.arange(10), "chrM": np.arange(5),
                   "chr2": np.arange(3)},
           "HCG": {"chr1": np.arange(7), "chrM": np.arange(4)}}
    out, removed = m._drop_chroms(ctx, "chrM")
    assert sorted(out["GCH"]) == ["chr1", "chr2"]
    assert sorted(out["HCG"]) == ["chr1"]
    assert removed == {"GCH:chrM": 5, "HCG:chrM": 4}


def test_exclude_chrom_matches_naming_variants():
    """chrM / M / MT are the same contig; a miss would silently do nothing."""
    m = _mod()
    for present in ("chrM", "M", "MT"):
        ctx = {"GCH": {"chr1": np.arange(4), present: np.arange(2)}}
        for asked in ("chrM", "M", "MT"):
            out, removed = m._drop_chroms(ctx, asked)
            assert sorted(out["GCH"]) == ["chr1"], (present, asked)
            assert removed, (present, asked)


def test_exclude_chrom_accepts_several_and_is_a_noop_when_empty():
    m = _mod()
    ctx = {"GCH": {"chr1": np.arange(4), "chrM": np.arange(2),
                   "chrY": np.arange(3)}}
    out, _ = m._drop_chroms(ctx, "chrM,chrY")
    assert sorted(out["GCH"]) == ["chr1"]
    out2, removed2 = m._drop_chroms(ctx, "")
    assert out2 is ctx and removed2 == {}


def test_exclude_chrom_reports_nothing_for_an_absent_contig():
    """Empty `removed` is what drives the 'matched no contig' warning."""
    m = _mod()
    ctx = {"GCH": {"chr1": np.arange(4)}}
    out, removed = m._drop_chroms(ctx, "chrM")
    assert removed == {}
    assert sorted(out["GCH"]) == ["chr1"]
