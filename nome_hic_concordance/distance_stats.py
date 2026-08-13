"""Per-distance-bin Pearson sufficient statistics.

The same six running sums serve both distance analyses, because Pearson's r
needs nothing else: ``n, Sx, Sy, Sxx, Syy, Sxy``.  site_distance_correlation.py
feeds them binary site calls (where they double as a 2x2 table);
end_posterior_distance_correlation.py feeds them continuous per-end posteriors.
Keeping one implementation here means a change to the estimator cannot reach
one analysis and miss the other.
"""

from __future__ import annotations

import warnings

import numpy as np


def build_edges(min_d: int, max_d: int, per_decade: int) -> np.ndarray:
    """Log-spaced bin edges with the exact bounds preserved at the ends."""
    lo, hi = np.log10(max(min_d, 1)), np.log10(max_d)
    n_edges = max(2, int(round((hi - lo) * per_decade)) + 1)
    e = np.logspace(lo, hi, n_edges)
    e[0], e[-1] = float(min_d), float(max_d)      # exact bounds (FP-safe)
    return e


def new_acc(nb: int) -> np.ndarray:
    """(6, nb) raw sums: n, Sx, Sy, Sxx, Syy, Sxy."""
    return np.zeros((6, nb))


def accumulate(acc: np.ndarray, bins: np.ndarray, x, y) -> None:
    """Add already-binned observations into the running sums."""
    bins = np.asarray(bins)
    if bins.size == 0:
        return
    nb = acc.shape[1]
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.all():
        bins, x, y = bins[ok], x[ok], y[ok]
        if bins.size == 0:
            return
    acc[0] += np.bincount(bins, minlength=nb)
    acc[1] += np.bincount(bins, weights=x, minlength=nb)
    acc[2] += np.bincount(bins, weights=y, minlength=nb)
    acc[3] += np.bincount(bins, weights=x * x, minlength=nb)
    acc[4] += np.bincount(bins, weights=y * y, minlength=nb)
    acc[5] += np.bincount(bins, weights=x * y, minlength=nb)


def corr_from_acc(acc: np.ndarray, min_pairs: int):
    """(r, n) per bin; NaN where n < min_pairs or either side has no variance.

    A bin whose x or y is constant has an undefined correlation rather than a
    zero one, so it is masked instead of being reported as "no relationship".

    Accepts a stack of accumulators too — ``(..., 6, nb)`` — so a bootstrap can
    evaluate every resample in one call.
    """
    acc = np.asarray(acc, dtype=np.float64)
    take = (lambda i: acc[i]) if acc.ndim == 2 else (lambda i: acc[..., i, :])
    n, sx, sy = take(0), take(1), take(2)
    sxx, syy, sxy = take(3), take(4), take(5)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        r = cov / np.sqrt(vx * vy)
    bad = (n < max(min_pairs, 2)) | ~np.isfinite(r) | (vx <= 0) | (vy <= 0)
    r = np.where(bad, np.nan, r)
    return r, n


def bootstrap_delta(per_cell_a, per_cell_b, min_pairs, n_boot, seed,
                    corr_fn=None):
    """Cell-level bootstrap of r(a) − r(b) per bin → (lo, hi, p).

    The unit of resampling is the CELL, not the observation.  End pairs are
    massively non-independent — hundreds of thousands come off one cell — so a
    test treating them as independent would return significance for everything.

    ``corr_fn(acc, min_pairs) -> r`` defaults to Pearson via
    :func:`corr_from_acc`; the site-level script passes its own so the same
    resampling serves the 2x2 measures.
    """
    if corr_fn is None:
        def corr_fn(acc, mp):
            return corr_from_acc(acc, mp)[0]
    a = np.stack(per_cell_a).astype(np.float64)          # (k, 6, nb)
    b = np.stack(per_cell_b).astype(np.float64)
    k = a.shape[0]
    if k < 3 or n_boot <= 0:                             # too few cells
        return None
    rng = np.random.default_rng(seed)
    w = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(np.float64)
    da = (w @ a.reshape(k, -1)).reshape(n_boot, 6, -1)
    db = (w @ b.reshape(k, -1)).reshape(n_boot, 6, -1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # A bin masked in every resample is an all-NaN slice — expected, not an
        # error; it simply has no Δ to report.
        warnings.simplefilter("ignore", RuntimeWarning)
        d = corr_fn(da, min_pairs) - corr_fn(db, min_pairs)
        lo, hi = np.nanpercentile(d, [2.5, 97.5], axis=0)
        n_ok = np.sum(np.isfinite(d), axis=0)
        # (1 + count) / (1 + B), NOT count / B: a finite bootstrap can never
        # justify p = 0, and an exact zero would survive any FDR procedure
        # unconditionally.  Floors the two-sided p at 2/(B+1).
        p = 2.0 * (1.0 + np.minimum(np.nansum(d <= 0, axis=0),
                                    np.nansum(d >= 0, axis=0))) / (n_ok + 1.0)
    # Below ~20 usable resamples the tail is unresolvable; report nothing.
    p = np.where(n_ok >= 20, np.minimum(p, 1.0), np.nan)
    return lo, hi, p


def fdr(p, method="bh"):
    """Adjusted p (q) over the finite entries of ``p``; NaN stays NaN.

    One test per distance bin per measure means dozens of tests per figure, so
    at raw p<0.05 several bins would be starred by chance alone.

    'bh' (Benjamini-Hochberg) is valid under positive regression dependency,
    the right assumption here: adjacent distance bins share cells and trace a
    smooth curve, so the tests are positively correlated rather than
    independent.  'by' holds under arbitrary dependence at a log(m) cost.
    """
    p = np.asarray(p, float)
    if method == "none":
        return p.copy()
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    ps = p[ok]
    order = np.argsort(ps)
    c = np.sum(1.0 / np.arange(1, m + 1)) if method == "by" else 1.0
    q = ps[order] * c * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]      # monotone in rank
    adj = np.empty(m)
    adj[order] = np.clip(q, 0.0, 1.0)
    out[ok] = adj
    return out


def stars(p) -> str:
    """*** / ** / * for q < 0.001 / 0.01 / 0.05; empty otherwise."""
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
