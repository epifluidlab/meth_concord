"""Per-end beta-binomial posteriors and the 2-D distance-pooled baseline.

Design sections A.3.2 to A.3.4.  Three things here are not the obvious ones:

**The variance may not come from the observed rate.**  With ``--min-hcg-sites
1`` the default, a read end's ``k/n`` is exactly 0 or 1, and the plug-in
variance ``theta_hat (1 - theta_hat) / n`` evaluates to ZERO.  Anything
weighting by inverse variance would then assign infinite precision to the
least informative observations in the dataset -- silently, and in the direction
that manufactures signal.  Every variance here is the beta-binomial posterior
variance ``mu (1 - mu) / (alpha + beta + n + 1)``, which is strictly positive
for any finite n because the prior contributes ``alpha + beta`` to the
denominator and the mean is strictly interior.

**Both marks share one grid, and one eligibility rule.**  If GCH's baseline
were pooled over grid cells where GCH is well observed and HCG's over cells
where HCG is, the two centred vectors would be measured against neighbourhoods
of different shape and composition -- and the difference enters ``theta - mu``
as covariance between the marks, which is precisely the quantity Part B
reports.  So a grid cell is usable only when BOTH marks clear ``min_obs``, and
when it is not, BOTH fall back to the global prior together.

**Normalization is location only** (A.3.4).  Part B whitens each end by its own
disattenuated covariance, which handles scale correctly and with measurement
error subtracted; a prior rescale distorts Sigma_AA and Sigma_BB, propagates
into R, and biases rho_2 in a coverage-dependent way.  ``full`` exists solely
as the B.7.6 ablation arm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

#: Smallest allowed prior concentration alpha + beta.  Also what keeps the
#: posterior variance away from zero at n = 1.
NU_FLOOR = 2.0

#: Concentration used when a grid cell shows no over-dispersion at all.
NU_TIGHT = 100.0

GCH_NORM_CHOICES = ("location", "full", "none")


def betabinom_mom(k, n, min_obs: int = 50, nu_floor: float = NU_FLOOR):
    """Method-of-moments beta-binomial prior from one pool of ``(k, n)``.

    The same estimator as :func:`~.concordance._fit_betabinom_eb_prior`, with
    the column names lifted out so it can be applied per grid cell.
    ``test_joint_baseline.py`` pins the two against each other.

    Returns ``(alpha, beta, info)``.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    ok = n >= 1
    k, n = k[ok], n[ok]
    if len(n) < min_obs or n.sum() <= 0:
        return 1.0, 1.0, {"mu": 0.5, "nu": 2.0, "n_obs": int(len(n)),
                          "fallback": True}
    mu = float(k.sum() / n.sum())
    w = n / n.sum()
    s2 = float(np.sum(w * (k / n - mu) ** 2))
    n_bar = float(np.sum(w * n))
    nu = _solve_nu(mu, s2, n_bar, nu_floor)
    return mu * nu, (1.0 - mu) * nu, {
        "mu": mu, "nu": nu, "n_obs": int(len(n)), "fallback": False}


def _solve_nu(mu, s2, n_bar, nu_floor):
    """Scalar branch of the MoM solve, shared with the grouped path."""
    mu1mu = mu * (1.0 - mu)
    numer = mu1mu - s2
    denom = s2 - mu1mu / max(n_bar, 1.0)
    if denom <= 0:
        return NU_TIGHT          # no over-dispersion beyond binomial noise
    if numer <= 0:
        return nu_floor          # super-binomial; let the data dominate
    return max(float(numer / denom), nu_floor)


def fit_prior_grouped(k, n, group, n_groups: int, min_obs: int = 50,
                      nu_floor: float = NU_FLOOR):
    """:func:`betabinom_mom` per group, vectorized.

    Parameters
    ----------
    group : int array
        Grid-cell index per observation, in ``[0, n_groups)``; negatives are
        dropped.

    Returns
    -------
    (alpha, beta, n_obs) : arrays of length ``n_groups``
        ``n_obs`` is the count of informative observations behind each fit;
        cells below ``min_obs`` come back as ``NaN`` for the caller to fill.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    g = np.asarray(group)
    keep = (n >= 1) & (g >= 0) & (g < n_groups)
    k, n, g = k[keep], n[keep], g[keep].astype(np.intp)

    cnt = np.bincount(g, minlength=n_groups)
    sk = np.bincount(g, weights=k, minlength=n_groups)
    sn = np.bincount(g, weights=n, minlength=n_groups)

    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(sn > 0, sk / np.maximum(sn, 1e-12), np.nan)
        w = n / np.maximum(sn[g], 1e-12)
        s2 = np.bincount(g, weights=w * (k / n - mu[g]) ** 2, minlength=n_groups)
        n_bar = np.bincount(g, weights=w * n, minlength=n_groups)

    mu1mu = mu * (1.0 - mu)
    denom = s2 - mu1mu / np.maximum(n_bar, 1.0)
    numer = mu1mu - s2
    with np.errstate(invalid="ignore", divide="ignore"):
        nu = np.where(denom <= 0, NU_TIGHT,
             np.where(numer <= 0, nu_floor,
                      np.maximum(numer / np.maximum(denom, 1e-300), nu_floor)))
    alpha = mu * nu
    beta = (1.0 - mu) * nu
    thin = cnt < min_obs
    alpha[thin] = np.nan
    beta[thin] = np.nan
    return alpha, beta, cnt


@dataclass
class PriorAccumulator:
    """Streaming sufficient statistics for :func:`fit_prior_grouped`.

    1,577 cells at ~1 M read ends each will not fit in memory, so the fit has
    to accumulate.  The five statistics below are additive, which makes shard
    merging a summation and the result independent of the order cells finish
    in -- the A.4 requirement that output not depend on the thread count.

    The accumulation is taken about ``center``, the mark's global mean, rather
    than about zero.  ``sum(n r^2) - (sum n r)^2 / sum n`` is algebraically the
    same number but computes a small variance as the difference of two large
    sums, and loses most of its significant digits when the grid cell is
    homogeneous -- which is exactly where a tight prior matters most.  Shifting
    to the global mean makes both terms small and the cancellation mild.
    """

    center: float
    cnt: np.ndarray
    sn: np.ndarray
    b: np.ndarray            # sum n (rate - center)
    a: np.ndarray            # sum n (rate - center)^2
    snn: np.ndarray          # sum n^2

    @classmethod
    def zeros(cls, n_cells: int, center: float) -> "PriorAccumulator":
        z = lambda: np.zeros(n_cells, dtype=np.float64)   # noqa: E731
        return cls(float(center), np.zeros(n_cells, dtype=np.int64),
                   z(), z(), z(), z())

    def add(self, k, n, cell) -> "PriorAccumulator":
        k = np.asarray(k, dtype=float)
        n = np.asarray(n, dtype=float)
        g = np.asarray(cell)
        keep = (n >= 1) & (g >= 0) & (g < self.cnt.size)
        k, n, g = k[keep], n[keep], g[keep].astype(np.intp)
        m = self.cnt.size
        d = k / n - self.center
        self.cnt += np.bincount(g, minlength=m).astype(np.int64)
        self.sn += np.bincount(g, weights=n, minlength=m)
        self.b += np.bincount(g, weights=n * d, minlength=m)
        self.a += np.bincount(g, weights=n * d * d, minlength=m)
        self.snn += np.bincount(g, weights=n * n, minlength=m)
        return self

    def merge(self, other: "PriorAccumulator") -> "PriorAccumulator":
        if other.center != self.center:
            raise ValueError("accumulators were centred differently; merging "
                             "them would mix two variance origins")
        self.cnt += other.cnt
        self.sn += other.sn
        self.b += other.b
        self.a += other.a
        self.snn += other.snn
        return self

    def fit(self, min_obs: int = 50, nu_floor: float = NU_FLOOR):
        """``(alpha, beta, n_obs)``, thin cells NaN, as fit_prior_grouped."""
        sn = np.maximum(self.sn, 1e-12)
        mu = self.center + self.b / sn
        # sum n (rate - mu)^2 = a - b^2 / sn, about the shifted origin.
        s2 = np.maximum(self.a - self.b ** 2 / sn, 0.0) / sn
        n_bar = self.snn / sn
        mu1mu = mu * (1.0 - mu)
        denom = s2 - mu1mu / np.maximum(n_bar, 1.0)
        numer = mu1mu - s2
        with np.errstate(invalid="ignore", divide="ignore"):
            nu = np.where(denom <= 0, NU_TIGHT,
                 np.where(numer <= 0, nu_floor,
                          np.maximum(numer / np.maximum(denom, 1e-300), nu_floor)))
        alpha = mu * nu
        beta = (1.0 - mu) * nu
        thin = self.cnt < min_obs
        alpha = np.where(thin, np.nan, alpha)
        beta = np.where(thin, np.nan, beta)
        return alpha, beta, self.cnt.copy()


def posterior_mean_var(k, n, alpha, beta):
    """Beta-binomial posterior mean and variance per end.

    ``Beta(alpha + k, beta + n - k)``, so

        mean = (alpha + k) / (alpha + beta + n)
        var  = mean (1 - mean) / (alpha + beta + n + 1)

    The variance is strictly positive for every finite ``n``, INCLUDING
    ``n == 1`` and including ``k == 0`` or ``k == n`` -- which is the whole
    reason the posterior is used instead of the observed rate (A.3.2).
    Ends with ``n == 0`` return the prior mean and the prior variance, not
    NaN: an unobserved end is uninformative, not missing, and Part B's weight
    already handles a wide posterior correctly.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    alpha = np.maximum(np.asarray(alpha, dtype=float), 1e-6)
    beta = np.maximum(np.asarray(beta, dtype=float), 1e-6)
    nu = alpha + beta + n
    mean = (alpha + k) / nu
    var = mean * (1.0 - mean) / (nu + 1.0)
    return mean, var


@dataclass
class JointBaseline:
    """Aligned per-grid-cell priors for both marks (A.3.3)."""

    marks: tuple[str, ...]
    alpha: np.ndarray            # (n_marks, n_cells)
    beta: np.ndarray             # (n_marks, n_cells)
    n_obs: np.ndarray            # (n_marks, n_cells)
    usable: np.ndarray           # (n_cells,) bool -- both marks cleared min_obs
    global_alpha: np.ndarray     # (n_marks,)
    global_beta: np.ndarray      # (n_marks,)
    bin_of_cell: np.ndarray
    stratum_of_cell: np.ndarray

    def mu(self, mark: str) -> np.ndarray:
        """Prior mean per grid cell -- the ``mu(bin, delta)`` of A.3.3."""
        i = self.marks.index(mark)
        return self.alpha[i] / (self.alpha[i] + self.beta[i])

    def evaluate(self, mark: str, k, n, cell_id):
        """Per-end posterior mean and variance against this baseline."""
        i = self.marks.index(mark)
        cell_id = np.asarray(cell_id)
        valid = (cell_id >= 0) & (cell_id < self.alpha.shape[1])
        idx = np.where(valid, cell_id, 0)
        a = np.where(valid, self.alpha[i][idx], self.global_alpha[i])
        b = np.where(valid, self.beta[i][idx], self.global_beta[i])
        return posterior_mean_var(k, n, a, b)

    def metadata(self) -> dict:
        return {
            "marks": list(self.marks),
            "n_grid_cells": int(self.usable.size),
            "n_usable_grid_cells": int(self.usable.sum()),
            "global_prior": {
                m: {"alpha": float(self.global_alpha[i]),
                    "beta": float(self.global_beta[i])}
                for i, m in enumerate(self.marks)},
        }


def fit_joint_baseline(cell_id, counts, n_cells: int, marks,
                       bin_of_cell=None, stratum_of_cell=None,
                       min_obs: int = 50, nu_floor: float = NU_FLOOR
                       ) -> JointBaseline:
    """Fit both marks' priors on ONE grid with ONE eligibility rule (A.3.3).

    Parameters
    ----------
    cell_id : int array
        Grid-cell index per read end (see :func:`grid_index`).
    counts : sequence of ``(k, n)`` pairs
        One per entry of *marks*, aligned to *cell_id*.
    """
    marks = tuple(marks)
    n_marks = len(marks)
    alpha = np.empty((n_marks, n_cells))
    beta = np.empty((n_marks, n_cells))
    n_obs = np.empty((n_marks, n_cells))
    g_alpha = np.empty(n_marks)
    g_beta = np.empty(n_marks)

    for i, (k, n) in enumerate(counts):
        a, b, cnt = fit_prior_grouped(k, n, cell_id, n_cells, min_obs, nu_floor)
        alpha[i], beta[i], n_obs[i] = a, b, cnt
        ga, gb, _ = betabinom_mom(k, n, min_obs=1, nu_floor=nu_floor)
        g_alpha[i], g_beta[i] = ga, gb

    # A grid cell is usable only where EVERY mark cleared min_obs.  Falling
    # back per mark would leave the two centred vectors measured against
    # different neighbourhoods -- the exact spurious covariance A.3.3 warns of.
    usable = np.isfinite(alpha).all(axis=0) & np.isfinite(beta).all(axis=0)
    for i in range(n_marks):
        alpha[i] = np.where(usable, alpha[i], g_alpha[i])
        beta[i] = np.where(usable, beta[i], g_beta[i])

    n_fell_back = int((~usable).sum())
    if n_fell_back:
        logger.info(
            "  baseline: %d/%d grid cells below min_obs=%d for at least one "
            "mark; BOTH marks use the global prior there, so the pair stays "
            "aligned", n_fell_back, n_cells, min_obs)
    return JointBaseline(
        marks=marks, alpha=alpha, beta=beta, n_obs=n_obs, usable=usable,
        global_alpha=g_alpha, global_beta=g_beta,
        bin_of_cell=(np.asarray(bin_of_cell) if bin_of_cell is not None
                     else np.arange(n_cells)),
        stratum_of_cell=(np.asarray(stratum_of_cell)
                         if stratum_of_cell is not None else np.zeros(n_cells, int)),
    )


def distance_strata(distance, edges) -> np.ndarray:
    """Log-spaced distance stratum per contact; ``-1`` outside the edges.

    Trans contacts (``distance < 0``) get their own stratum, ``len(edges) - 1``,
    rather than being folded into the widest cis one.
    """
    d = np.asarray(distance)
    edges = np.asarray(edges, dtype=float)
    s = np.searchsorted(edges, d, side="right") - 1
    s = np.where((d < edges[0]) | (d >= edges[-1]), -1, s)
    return np.where(np.asarray(distance) < 0, len(edges) - 1, s).astype(np.int32)


def grid_index(bin_id, stratum, n_bins: int, n_strata: int):
    """Flatten ``(bin, distance stratum)`` into one grid-cell index.

    Returns ``(cell_id, n_cells, bin_of_cell, stratum_of_cell)``; ``cell_id``
    is ``-1`` wherever either coordinate is out of range.
    """
    b = np.asarray(bin_id)
    s = np.asarray(stratum)
    ok = (b >= 0) & (b < n_bins) & (s >= 0) & (s < n_strata)
    cell = np.where(ok, b * n_strata + s, -1).astype(np.int64)
    n_cells = n_bins * n_strata
    bin_of_cell = np.repeat(np.arange(n_bins), n_strata)
    stratum_of_cell = np.tile(np.arange(n_strata), n_bins)
    return cell, n_cells, bin_of_cell, stratum_of_cell


def prior_variance(alpha, beta):
    """Variance the fitted prior implies for a grid cell: ``mu(1-mu)/(nu+1)``.

    This -- not the per-END posterior variance -- is the "bin-level variance"
    A.3.4 names.  Dividing by the posterior sd would rescale each end by its
    own coverage, which is a per-observation reweighting, not a normalization
    of the bin; the ablation would then be testing the wrong thing.
    """
    alpha = np.maximum(np.asarray(alpha, dtype=float), 1e-12)
    beta = np.maximum(np.asarray(beta, dtype=float), 1e-12)
    nu = alpha + beta
    mu = alpha / nu
    return mu * (1.0 - mu) / (nu + 1.0)


def normalize(theta, mu, var=None, mode: str = "location"):
    """Centre (and, only as the B.7.6 ablation, rescale) a per-end value.

    ``location`` -- the default and the one A.3.4 argues for -- subtracts the
    distance-pooled expectation and stops.  ``full`` additionally divides by
    the GRID-CELL standard deviation (see :func:`prior_variance`); it exists so
    the ablation can be run, not because it is a sensible default.  ``none``
    returns ``theta`` unchanged.
    """
    if mode not in GCH_NORM_CHOICES:
        raise ValueError(f"gch_norm must be one of {GCH_NORM_CHOICES}; "
                         f"got {mode!r}")
    theta = np.asarray(theta, dtype=float)
    if mode == "none":
        return theta
    centred = theta - np.asarray(mu, dtype=float)
    if mode == "location":
        return centred
    if var is None:
        raise ValueError("gch_norm='full' needs the grid-cell variance")
    sd = np.sqrt(np.maximum(np.asarray(var, dtype=float), 1e-12))
    return centred / sd
