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


def _fisher_z(r, n):
    """Fisher's variance-stabilising z with the O(1/n) bias correction.

    E[z] ~ zeta + rho / (2(n-1)), and a bin's linked and null arms need not
    carry the same n, so without the correction unequal counts alone would
    leave a systematic difference behind when the two true correlations are
    equal.
    """
    r = np.clip(np.asarray(r, dtype=np.float64), -0.999999, 0.999999)
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.arctanh(r) - r / (2.0 * np.maximum(n - 1.0, 1.0))


def fisher_z_cell(per_cell_a, per_cell_b, min_pairs, min_cells=3):
    """Paired Fisher z within cells, one-sample t across them → (dz,lo,hi,p,k).

    Fisher's two-sample z compares two correlations through n = the number of
    OBSERVATIONS.  Applied to the pooled curves that n is the number of end
    pairs — hundreds of thousands per bin, nearly all of them non-independent,
    since one cell contributes most of them and one end sits in many — so it
    would return p ~ 1e-12 for a difference of 0.01.  Here the transform is
    taken WITHIN each cell and the per-cell differences are tested with a
    one-sample t, which puts the unit of inference back on the cell, exactly
    where :func:`bootstrap_delta` puts it.  The two tests then differ only in
    how the tail is obtained — parametric rather than resampled, so this one
    resolves p below the bootstrap's 2/(B+1) floor, at the cost of assuming
    the cell-level differences are roughly normal.

    Pairing inside the cell also repairs the other flaw in the textbook form:
    the linked and null arms are not independent samples (same cells, and
    under a locus-matched null the same anchor ends), which the two-sample
    standard error assumes.

    Cells are weighted equally rather than by their pair count — the standard
    cluster-summary analysis.  Inverse-variance weights would be more
    efficient only if the within-cell 1/(n-3) model were exact, which is the
    assumption this whole function exists to avoid relying on.
    """
    from scipy.stats import t as _t

    a = np.stack(per_cell_a).astype(np.float64)          # (k, 6, nb)
    b = np.stack(per_cell_b).astype(np.float64)
    with warnings.catch_warnings():
        # Bins a cell never covers are all-NaN slices — expected, not an error.
        warnings.simplefilter("ignore", RuntimeWarning)
        ra, na = corr_from_acc(a, min_pairs)
        rb, nbp = corr_from_acc(b, min_pairs)
        d = _fisher_z(ra, na) - _fisher_z(rb, nbp)       # (k, nbins)
        k = np.sum(np.isfinite(d), axis=0)
        mean = np.nanmean(d, axis=0)
        sd = np.nanstd(d, axis=0, ddof=1)
    ok = (k >= max(min_cells, 3)) & np.isfinite(mean) & np.isfinite(sd)
    ok &= sd > 0
    df = np.maximum(k - 1, 1)
    se = np.where(ok, sd / np.sqrt(np.maximum(k, 1)), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = 2.0 * _t.sf(np.abs(np.where(ok, mean / se, np.nan)), df)
        crit = np.where(ok, _t.ppf(0.975, df), np.nan)
    mean = np.where(ok, mean, np.nan)
    return mean, mean - crit * se, mean + crit * se, p, k


def fisher_z_pooled(acc_a, acc_b, min_pairs):
    """Textbook two-sample Fisher z on the pooled curves → (dz, lo, hi, p, n).

    Offered for comparison only.  Its n is the number of end pairs, which are
    emphatically not independent observations, so it is anti-conservative by
    orders of magnitude: at n ~ 1e6 per arm the standard error is ~0.0014 and
    a Δz of 0.01 lands at p ~ 1e-12.  It also assumes the two samples are
    independent, which linked and null are not.  Use :func:`fisher_z_cell`.
    """
    from scipy.stats import norm

    ra, na = corr_from_acc(acc_a, min_pairs)
    rb, nb = corr_from_acc(acc_b, min_pairs)
    dz = _fisher_z(ra, na) - _fisher_z(rb, nb)
    na = np.where(na > 3, na, np.nan)
    nb = np.where(nb > 3, nb, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(1.0 / (na - 3.0) + 1.0 / (nb - 3.0))
        p = 2.0 * norm.sf(np.abs(dz / se))
    return dz, dz - 1.959964 * se, dz + 1.959964 * se, p, np.minimum(na, nb)


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
