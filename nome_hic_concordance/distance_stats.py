"""Per-distance-bin Pearson sufficient statistics.

The same six running sums serve both distance analyses, because Pearson's r
needs nothing else: ``n, Sx, Sy, Sxx, Syy, Sxy``.  site_distance_correlation.py
feeds them binary site calls (where they double as a 2x2 table);
end_posterior_distance_correlation.py feeds them continuous per-end posteriors.
Keeping one implementation here means a change to the estimator cannot reach
one analysis and miss the other.
"""

from __future__ import annotations

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
    """
    n, sx, sy, sxx, syy, sxy = acc
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        r = cov / np.sqrt(vx * vy)
    bad = (n < max(min_pairs, 2)) | ~np.isfinite(r) | (vx <= 0) | (vy <= 0)
    r = np.where(bad, np.nan, r)
    return r, n
