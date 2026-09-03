"""Part C numerics: the per-cell scalar coordination and what is built on it.

A per-cell 2x2 is not estimable at 250 kb.  A per-cell **scalar** is, provided
the DIRECTION is fixed from pooled data and only the magnitude is estimated per
cell (C.1).  Everything here follows from that split, and three consequences of
it are easy to get wrong:

**Numerator per contact, denominator pooled.**  ``kappa_r`` divides a per-contact
product by the POOLED disattenuated SD from Part B, not by a per-cell one.  That
is what makes rho-hat an average of well-behaved terms rather than a correlation
estimated from a handful of points -- a per-cell denominator would be a ratio of
two noisy quantities and would blow up exactly where coverage is thin.

**The per-cell genome-wide mean is removed and kept separately.**  It is real
signal -- cell cycle, global hypomethylation -- but if it stayed in the matrix
every bin would be correlated through one per-cell scalar, and the first PC
would be that scalar wearing a genomic costume.

**Raw variance across cells is the wrong ranking for C.5.**  It is dominated by
estimation noise, so the "most variable" list would be the lowest-coverage bins.
The noise term is subtracted, and calibrated against the B.6.1 mate-shuffle
floor rather than trusted from the analytic SE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Below this median per-(cell, bin) N_eff the design says go coarser before
#: proceeding; the power curve from Part B decides how much coarser.
NEFF_MEDIAN_FLOOR = 10.0

#: Stands in for an infinite shrinkage weight: a bin whose between-cell spread
#: is entirely sampling noise has no per-cell signal to preserve.
TAU_MAX = 1e6

#: What C.1's feasibility note calls workable at 250 kb.
NEFF_MEDIAN_TARGET = (20.0, 60.0)


# ------------------------------------------------------------------- C.1 ----

def project_ends(theta, v, w):
    """``s = w^T theta`` and ``v_s = w^T diag(v) w`` for one end.

    ``w`` may be one direction (2,) or one per contact (n, 2) -- per-BIN
    directions are the point: a single global direction would erase the
    lambda_h sign flips that gene bodies and PMDs actually show.
    """
    theta = np.atleast_2d(np.asarray(theta, float))
    v = np.atleast_2d(np.asarray(v, float))
    w = np.asarray(w, float)
    if w.ndim == 1:
        w = np.broadcast_to(w, theta.shape)
    s = (theta * w).sum(axis=1)
    v_s = (v * w ** 2).sum(axis=1)
    return s, v_s


def contact_kernel(s_A, s_B, sigma_A, sigma_B, eps=1e-12):
    """``kappa = s_A s_B / (sigma_A sigma_B)``, with ``E[kappa] = rho1``.

    ``sigma_*`` are the POOLED disattenuated SDs from Part B for the two ends'
    (bin, stratum).  Passing per-cell SDs here would make every term a ratio of
    two noisy quantities.
    """
    return (np.asarray(s_A, float) * np.asarray(s_B, float)
            / (np.asarray(sigma_A, float) * np.asarray(sigma_B, float) + eps))


def aggregate(kappa, weights, keys) -> pd.DataFrame:
    """Weighted mean kernel per key, with ``N_eff`` and the C.1 step-4 SE.

    ``SE^2 = (1 + rho^2) / N_eff`` -- the large-sample variance of a
    correlation, evaluated at the estimate.  It is the ANALYTIC floor; C.5
    prefers the mate-shuffle floor over it and this is what that gets compared
    against.
    """
    k = np.asarray(kappa, float)
    w = np.asarray(weights, float)
    ok = np.isfinite(k) & np.isfinite(w) & (w > 0)
    df = pd.DataFrame(dict(keys))[ok] if isinstance(keys, dict) else keys[ok]
    df = df.copy()
    df["_wk"] = w[ok] * k[ok]
    df["_w"] = w[ok]
    df["_w2"] = w[ok] ** 2
    cols = [c for c in df.columns if not c.startswith("_")]
    g = df.groupby(cols, observed=True, sort=True).agg(
        _wk=("_wk", "sum"), _w=("_w", "sum"), _w2=("_w2", "sum"),
        n_contacts=("_w", "size")).reset_index()
    g["rho_hat"] = g["_wk"] / g["_w"]
    g["n_eff"] = np.where(g["_w2"] > 0, g["_w"] ** 2 / g["_w2"], 0.0)
    g["se"] = np.sqrt((1.0 + g["rho_hat"] ** 2)
                      / np.maximum(g["n_eff"], 1e-12))
    return g.drop(columns=["_wk", "_w", "_w2"])


def tau_moment(rho_hat, n_eff, bin_id, min_cells: int = 3) -> pd.Series:
    """Method-of-moments shrinkage weight ``tau_b`` per bin (C.1 step 4).

    ``tau_b = mean(SE^2) / sigma^2_between``.  A bin whose between-cell spread
    is no larger than its sampling noise has no per-cell signal to preserve, so
    tau is large and every cell is pulled to the bin mean; a bin with real
    spread keeps its cells apart.  Bins with too few cells get the pooled tau
    rather than an estimate from two points.
    """
    df = pd.DataFrame({"bin": np.asarray(bin_id), "rho": np.asarray(rho_hat, float),
                       "n_eff": np.asarray(n_eff, float)})
    df["se2"] = (1.0 + df["rho"] ** 2) / np.maximum(df["n_eff"], 1e-12)
    g = df.groupby("bin", observed=True)
    var_obs = g["rho"].var(ddof=1)
    mean_se2 = g["se2"].mean()
    n_cells = g["rho"].size()
    between = (var_obs - mean_se2).clip(lower=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = mean_se2 / between
    pooled = float(np.nanmedian(tau[np.isfinite(tau)])) if np.isfinite(tau).any() \
        else 1.0
    # A bin with no between-cell spread at all gives tau = inf.  Filling that
    # from the largest FINITE tau would make "no signal" indistinguishable from
    # "a little signal" -- it has to be a hard shrink, so a sentinel.
    infinite = ~np.isfinite(tau)
    tau = tau.where(np.isfinite(tau), TAU_MAX)
    tau = tau.where((n_cells >= min_cells) | infinite, pooled)
    return tau


def shrink(rho_hat, n_eff, rho_bar, tau):
    """``(N_eff rho_hat + tau rho_bar) / (N_eff + tau)`` -- C.1 step 4."""
    n = np.asarray(n_eff, float)
    t = np.asarray(tau, float)
    return (n * np.asarray(rho_hat, float) + t * np.asarray(rho_bar, float)) \
        / np.maximum(n + t, 1e-12)


def feasibility(n_eff, bin_size: int) -> dict:
    """The C.1 check that has to run BEFORE anything else.

    If the median per-(cell, bin) ``N_eff`` is under ~10 the design says go to
    500 kb or 1 Mb before proceeding.  Returning a recommendation rather than
    silently re-binning: the bin size changes every downstream file, and that
    is the user's call.
    """
    x = np.asarray(n_eff, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"median": float("nan"), "verdict": "no_data",
                "message": "no (cell, bin) had any weight"}
    med = float(np.median(x))
    lo, hi = NEFF_MEDIAN_TARGET
    out = {"median": med, "q25": float(np.percentile(x, 25)),
           "q75": float(np.percentile(x, 75)),
           "frac_below_5": float((x < 5).mean()), "bin_size": int(bin_size)}
    if med < NEFF_MEDIAN_FLOOR:
        out["verdict"] = "too_thin"
        out["message"] = (
            f"median N_eff {med:.1f} at {bin_size:,} bp is below {NEFF_MEDIAN_FLOOR:.0f}. "
            f"Go to {2 * bin_size:,} or {4 * bin_size:,} bp before proceeding -- "
            f"the Part B power curve decides how far.")
    elif med < lo:
        out["verdict"] = "marginal"
        out["message"] = (f"median N_eff {med:.1f} is workable but below the "
                          f"{lo:.0f}-{hi:.0f} the design expects at this size")
    else:
        out["verdict"] = "ok"
        out["message"] = f"median N_eff {med:.1f}"
    return out


# ------------------------------------------------------------------- C.2 ----

@dataclass
class CellBinMatrix:
    """cells x bins, with the mask and the per-cell covariates kept alongside.

    Two value arrays, and using the wrong one is the trap.  ``values`` is the
    C.2 output: per-cell centred AND per-bin standardised, which is what the
    embedding wants.  ``values_centred`` stops after the centring.

    C.5 must use ``values_centred``.  Per-bin standardization sets every bin's
    across-cell variance to 1 by construction, so ``Var_cells`` on ``values``
    is the same number everywhere and the whole sigma2_bio decomposition --
    the thing that stops the ranking being dominated by estimation noise --
    returns a constant.
    """

    values: np.ndarray            # (n_cells, n_bins), NaN where masked
    values_centred: np.ndarray    # the same, before per-bin standardization
    cells: list
    bins: list
    se: np.ndarray
    n_eff: np.ndarray
    obs: pd.DataFrame             # per-cell covariates, indexed like `cells`
    steps: list

    @property
    def mask(self) -> np.ndarray:
        return ~np.isfinite(self.values)


def build_matrix(per_cell: pd.DataFrame, cells, bins, min_neff_cell: float = 5.0,
                 value_col: str = "rho_shrunk") -> CellBinMatrix:
    """C.2, in the order the design gives -- and the order matters.

    1. shrunk rho; 2. per-cell centring (removes depth, conversion and global
    coordination); 3. per-bin standardization weighted by 1/SE^2; 4. mask thin
    (cell, bin) entries as MISSING.

    The mask comes last on purpose.  Masking before centring would let a cell's
    row mean be computed over a different bin set than its neighbours', which
    puts the coverage back into the very step meant to remove it.

    One consequence worth knowing: per-cell centring redistributes a cell-type
    signal across the OTHER bins with the opposite sign, in proportion to the
    fraction of bins carrying it.  At genome scale (a handful of specific bins
    in ~10,800) that is negligible; on a few dozen bins it is not, and a small
    test fixture will show every non-specific bin as strongly group-associated.
    """
    cells = list(cells)
    bins = list(bins)
    ci = {c: i for i, c in enumerate(cells)}
    bi = {b: i for i, b in enumerate(bins)}
    shape = (len(cells), len(bins))
    val = np.full(shape, np.nan)
    se = np.full(shape, np.nan)
    neff = np.zeros(shape)

    r = per_cell[per_cell["cell_id"].isin(ci) & per_cell["bin"].isin(bi)]
    ri = r["cell_id"].map(ci).to_numpy()
    rj = r["bin"].map(bi).to_numpy()
    val[ri, rj] = r[value_col].to_numpy()
    se[ri, rj] = r["se"].to_numpy()
    neff[ri, rj] = r["n_eff"].to_numpy()

    steps = ["shrunk"]
    row_mean = np.nanmean(np.where(np.isfinite(val), val, np.nan), axis=1)
    val = val - row_mean[:, None]
    steps.append("per_cell_centred")
    centred = val.copy()

    w = 1.0 / np.maximum(se ** 2, 1e-12)
    w[~np.isfinite(val)] = 0.0
    sw = w.sum(axis=0)
    mu = np.where(sw > 0, np.nansum(np.where(np.isfinite(val), w * val, 0), axis=0)
                  / np.maximum(sw, 1e-12), 0.0)
    dev = np.where(np.isfinite(val), val - mu, 0.0)
    var = np.where(sw > 0, (w * dev ** 2).sum(axis=0) / np.maximum(sw, 1e-12), 1.0)
    sd = np.sqrt(np.maximum(var, 1e-12))
    val = (val - mu) / sd
    steps.append("per_bin_standardised")

    thin = neff < float(min_neff_cell)
    val[thin] = np.nan
    centred[thin] = np.nan
    steps.append(f"masked_neff_lt_{min_neff_cell:g}")

    obs = pd.DataFrame(index=pd.Index(cells, name="cell_id"))
    obs["row_mean_rho"] = row_mean
    obs["n_bins_observed"] = np.isfinite(val).sum(axis=1)
    return CellBinMatrix(values=val, values_centred=centred, cells=cells,
                         bins=bins, se=se, n_eff=neff, obs=obs, steps=steps)


# ------------------------------------------------------------------- C.5 ----

def sigma2_bio(values, se) -> pd.DataFrame:
    """``max(0, Var_cells - mean_c(SE^2))`` per bin, with its parts kept.

    Ranking on raw variance would put the lowest-coverage bins on top, because
    their variance IS the sampling noise.
    """
    v = np.asarray(values, float)
    s = np.asarray(se, float)
    obs = np.isfinite(v)
    n = obs.sum(axis=0)
    with np.errstate(invalid="ignore"):
        var = np.nanvar(np.where(obs, v, np.nan), axis=0, ddof=1)
        mse = np.nanmean(np.where(obs & np.isfinite(s), s ** 2, np.nan), axis=0)
    return pd.DataFrame({
        "n_cells": n, "var_total": var, "mean_se2": mse,
        "sigma2_bio": np.maximum(var - mse, 0.0),
        "snr": np.where(mse > 0, np.maximum(var - mse, 0.0) / mse, np.nan),
    })


def noise_floor_from_null_matrix(null_values) -> np.ndarray:
    """Per-bin noise floor: the across-cell variance of a MATE-SHUFFLED matrix.

    The floor has to come from a matrix rebuilt on shuffled contacts (B.6.1),
    not from permuting the finished matrix.  Permuting cells within a bin
    reorders that bin's column and leaves its variance EXACTLY unchanged, so a
    floor computed that way equals the observed variance in every bin and no
    bin ever clears it -- a bug that looks like "nothing is variable" rather
    than like an error.  ``test_a_column_permutation_cannot_be_a_noise_floor``
    keeps it from coming back.
    """
    v = np.asarray(null_values, float)
    with np.errstate(invalid="ignore"):
        return np.nanvar(np.where(np.isfinite(v), v, np.nan), axis=0, ddof=1)


def empirical_sigma2_bio(stats: pd.DataFrame, noise_floor) -> pd.DataFrame:
    """Attach the empirical floor and the variance that clears it.

    The design prefers this to the analytic ``mean(SE^2)``: the shuffled matrix
    carries whatever the SE formula does not -- weight caps, shrinkage, the
    mask -- and is therefore the more trustworthy denominator.
    """
    out = stats.copy()
    out["noise_floor"] = np.asarray(noise_floor, float)
    out["sigma2_bio_emp"] = np.maximum(
        out["var_total"].to_numpy() - out["noise_floor"].to_numpy(), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["snr_emp"] = np.where(out["noise_floor"] > 0,
                                  out["sigma2_bio_emp"] / out["noise_floor"],
                                  np.nan)
    return out


def group_f_statistic(values, groups) -> pd.DataFrame:
    """Between- vs within-group variance per bin, over the manifest groups."""
    v = np.asarray(values, float)
    g = pd.Series(np.asarray(groups)).astype(str).to_numpy()
    levels = [l for l in pd.unique(g) if l == l]
    n_bins = v.shape[1]
    ssb = np.zeros(n_bins)
    ssw = np.zeros(n_bins)
    dfb = np.zeros(n_bins)
    dfw = np.zeros(n_bins)
    with np.errstate(invalid="ignore"):
        grand = np.nanmean(v, axis=0)
        for l in levels:
            m = g == l
            sub = v[m]
            n = np.isfinite(sub).sum(axis=0)
            mean = np.nanmean(sub, axis=0)
            ok = n > 0
            ssb[ok] += n[ok] * (mean[ok] - grand[ok]) ** 2
            dfb[ok] += 1
            ssw[ok] += np.nansum((sub - mean) ** 2, axis=0)[ok]
            dfw[ok] += np.maximum(n[ok] - 1, 0)
    dfb = np.maximum(dfb - 1, 1)
    dfw = np.maximum(dfw, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        F = (ssb / dfb) / (ssw / dfw)
    return pd.DataFrame({"F": F, "df_between": dfb, "df_within": dfw})


def specificity_index(values, groups) -> pd.DataFrame:
    """Max group mean minus the median of the rest, and the group achieving it.

    A bin can have a large F because one group is LOW; the index and the named
    group are what say which, and in which direction.
    """
    v = np.asarray(values, float)
    g = pd.Series(np.asarray(groups)).astype(str).to_numpy()
    levels = [l for l in pd.unique(g) if l == l]
    with np.errstate(invalid="ignore"):
        means = np.vstack([np.nanmean(v[g == l], axis=0) for l in levels])
    idx = np.nanargmax(np.abs(means - np.nanmedian(means, axis=0)), axis=0)
    top = means[idx, np.arange(means.shape[1])]
    rest = np.nanmedian(means, axis=0)
    return pd.DataFrame({
        "specificity_index": top - rest,
        "top_group": [levels[i] for i in idx],
        "top_group_mean": top, "other_group_median": rest,
    })


def permutation_p(values, groups, n_perm: int = 1000, seed: int = 0
                  ) -> np.ndarray:
    """F-statistic p-value by permuting CELL labels, never reads.

    A read-level permutation would break the within-cell correlation that makes
    a cell one observation, and would give p-values far too small.
    """
    v = np.asarray(values, float)
    g = np.asarray(groups)
    obs = group_f_statistic(v, g)["F"].to_numpy()
    rng = np.random.default_rng(seed)
    ge = np.zeros(v.shape[1])
    for _ in range(int(n_perm)):
        f = group_f_statistic(v, rng.permutation(g))["F"].to_numpy()
        ge += (f >= obs) | (~np.isfinite(obs) & ~np.isfinite(f))
    return (1.0 + ge) / (1.0 + int(n_perm))


def classify_regions(stats: pd.DataFrame, top_n: int | None = 2000,
                     fdr: float | None = 0.05,
                     stable_rho_quantile: float = 0.75,
                     stable_neff_quantile: float = 0.5) -> pd.DataFrame:
    """The four C.5 sets, which are NOT the same ranking.

    "Most variable" is not "cell-type specific": a bin can vary a great deal
    across cells within one type.  And a bin that is stable because it is
    UNCOORDINATED is background, not stable -- the ``|mean_rho|`` condition is
    the whole thing that separates them, so it is required, not a tiebreak.

    ``top_n`` and ``fdr`` intersect when both are given, which is the design's
    rule; passing neither leaves every category empty rather than guessing.
    """
    s = stats.copy()
    qc = s["passes_qc"] if "passes_qc" in s else pd.Series(True, index=s.index)

    def pick(order_col, extra=None, ascending=False):
        m = qc.copy()
        if extra is not None:
            m &= extra
        cand = s[m]
        chosen = pd.Series(False, index=s.index)
        by_n = by_q = None
        if top_n:
            by_n = cand[order_col].sort_values(ascending=ascending).head(top_n).index
        if fdr is not None and "q_value" in cand:
            by_q = cand.index[cand["q_value"] < fdr]
        if by_n is not None and by_q is not None:
            keep = by_n.intersection(by_q)
        elif by_n is not None:
            keep = by_n
        elif by_q is not None:
            keep = by_q
        else:
            keep = s.index[[]]
        chosen.loc[keep] = True
        return chosen

    # Prefer the empirical floor when a mate-shuffled matrix supplied one.
    rank_col = "sigma2_bio_emp" if "sigma2_bio_emp" in s else "sigma2_bio"
    if "noise_floor" in s and "var_total" in s:
        above_floor = s["var_total"] > s["noise_floor"]
    else:
        above_floor = s["sigma2_bio"] > 0
    s["is_variable"] = pick(rank_col, above_floor)
    s["is_celltype_specific"] = pick("F", above_floor)

    rho_hi = s["mean_rho"].abs() >= s["mean_rho"].abs().quantile(stable_rho_quantile)
    neff_hi = s["mean_neff"] >= s["mean_neff"].quantile(stable_neff_quantile)
    low_var = s[rank_col] <= s[rank_col].quantile(0.25)
    s["is_stable"] = qc & rho_hi & neff_hi & low_var
    s["is_background"] = qc

    s["category"] = np.select(
        [s["is_celltype_specific"], s["is_variable"], s["is_stable"]],
        ["celltype_specific", "variable", "stable"], default="background")
    return s
