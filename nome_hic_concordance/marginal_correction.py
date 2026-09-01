"""Two- and three-stage marginal correction for GCH concordance matrices.

WHY THIS EXISTS.  Long-range GCH concordance profiles did not transfer
between biological replicates -- cross-replicate cell-type purity 0.31 of the
within-replicate value -- while mean accessibility measured on the SAME cells
transferred at 0.93.  The carrier was found to be the pair-derived marginal

    p_bar = (O_open - O_closed + 1) / 2

which transferred at 0.27 and differed between preparations (mean 0.267
against 0.346), while the dense accessibility estimate of the same biological
quantity did not.  Both statistics inherit it: the exact per-pair identity
D = p1 + p2 - 1 pins the marginal inside L, so pure coordination reads +1.73
at p_bar 0.10 and -1.73 at 0.90.  HCG concordance, which involves no
exogenous enzyme step, transferred at 0.79 and needs none of this.

WHAT WAS RULED OUT FIRST, each measured rather than argued -- listed because
the natural fixes all fail here and someone will try them again:

  per-cell sequencing depth      |corr(depth, PC1)| = 0.17; 0.244 -> 0.246
  the per-(cell, bin) coverage   canonical corr 0.81 but under 5% of the
    matrix, ranks 1..20          variance; 0.244 -> 0.255
  differential bin coverage      98.3% of bins usable in both replicates
  plate / library batch          no effect
  RWR refinement passes          no effect
  per-bin centre, ComBat,        all location-based; the difference is not a
    z-score, harmony             shift at matched biology
  local matching anchored on     0.244 -> 0.290; accessibility neighbourhoods
    accessibility                do not match L's structure

THE CORRECTION.  Stage 1 removes the leading factors of the marginal matrix
while PROTECTING the accessibility-derived components, so only the technical
part of p_bar goes and the real accessibility biology stays.  Stage 2 then
uses the stage-1 output as the anchor space for local cross-batch matching:
the raw matrix cannot anchor itself, because at cross-purity 0.14 it cannot
find its own cross-replicate neighbours -- removing the marginal first is
what makes them findable.  Stage 3 is the variance-partition residual, the
part not predicted by accessibility and compartment, which is both the
batch-robust quantity and the sharper scientific one.

Measured end to end, inside the pipeline, identical settings, only the input
matrix differing: residual cross/within 0.309 -> 0.703 against HCG's 0.788
ceiling, with within-replicate purity unchanged at 0.577-0.583.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Default number of marginal factors removed in stage 1.  Swept on the real
#: data: cross/within peaked at 0.575 (k=5) and 0.581 (k=8), but k=8 already
#: cost within-purity (0.565 against 0.577), and past k=8 it erodes steadily.
#: k=5 is the last point where cross-purity more than doubles with within
#: fully intact.
DEFAULT_K = 5

#: Neighbours per replicate for the stage-2 local matching.
DEFAULT_ANCHOR_K = 30

#: Baseline components retained to protect biology.
DEFAULT_PROTECT_PCS = 20


def _as_float(df: pd.DataFrame, fill: float = 0.0) -> np.ndarray:
    return np.nan_to_num(df.to_numpy(float), nan=fill)


def _too_small(X: np.ndarray, what: str) -> bool:
    """A factorisation needs at least 2 cells AND 2 bins.

    Sparse contexts really do reduce to one usable bin after coverage
    filtering, and TruncatedSVD raises on a single feature.  Returning the
    matrix untouched with a warning is the honest response: there is no
    factor structure to remove, and a correction that silently did nothing
    would be indistinguishable from one that worked.
    """
    if min(X.shape) < 2:
        logger.warning(
            "%s is %d x %d; a factorisation needs at least 2 cells and 2 "
            "bins, so this stage is skipped and the matrix returned "
            "unchanged.", what, X.shape[0], X.shape[1])
        return True
    return False


def marginal_transform(P: np.ndarray, how: str = "d") -> np.ndarray:
    """Put the marginal on the scale the correction fits.

    ``"d"`` is D = 2*p_bar - 1, the term that actually appears in the
    identity, and it beat ``"logit"`` at every k on the real data (cross
    0.332 against 0.240 at k=5).  A reparameterisation of the same quantity
    is not automatically as good as the quantity itself.
    """
    if how == "d":
        return 2.0 * P - 1.0
    if how == "logit":
        q = np.clip(P, 1e-4, 1 - 1e-4)
        return np.log(q / (1 - q))
    if how == "raw":
        return P
    raise ValueError(f"transform must be 'd', 'logit' or 'raw'; got {how!r}")


def protect_design(baseline: pd.DataFrame | None, index: pd.Index,
                   n_pcs: int = DEFAULT_PROTECT_PCS,
                   seed: int = 0) -> np.ndarray | None:
    """Standardised baseline components, as the biology to be preserved.

    Returns ``None`` when no baseline is given, which makes every stage fall
    back to its unprotected form -- measurably worse (stage 1 unprotected
    took cross-purity DOWN, 0.142 -> 0.083), so a baseline is strongly
    recommended rather than optional in spirit.
    """
    if baseline is None:
        return None
    from sklearn.decomposition import TruncatedSVD
    X = _as_float(baseline.reindex(index))
    if _too_small(X, "protect_design baseline"):
        return None
    X = X - X.mean(0, keepdims=True)
    k = max(1, min(int(n_pcs), min(X.shape) - 1))
    Z = TruncatedSVD(n_components=k, random_state=seed).fit_transform(X)
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-12)


def stage1_marginal(matrix: pd.DataFrame, marginal: pd.DataFrame,
                    protect: np.ndarray | None = None,
                    k: int = DEFAULT_K, transform: str = "d",
                    seed: int = 0) -> pd.DataFrame:
    """Remove the leading marginal factors, keeping what the baseline explains.

    The fit is ``M ~ [1, protect, U]`` per bin and only ``U``'s contribution
    is subtracted, so the accessibility-explained part of p_bar survives.
    That distinction is the whole correction: unprotected, this makes things
    WORSE, because p_bar carries real accessibility biology as well as the
    technical drift.
    """
    from sklearn.decomposition import TruncatedSVD
    idx = matrix.index
    P = _as_float(marginal.reindex(index=idx, columns=matrix.columns), fill=0.5)
    X = marginal_transform(P, transform)
    if _too_small(X, "stage1 marginal"):
        return matrix.copy()
    Xc = X - X.mean(0, keepdims=True)
    kk = max(1, min(int(k), min(Xc.shape) - 1))
    U = TruncatedSVD(n_components=kk, random_state=seed).fit_transform(Xc)
    U = (U - U.mean(0)) / (U.std(0) + 1e-12)
    cols = [np.ones(len(idx))] + ([protect] if protect is not None else []) + [U]
    D = np.column_stack(cols)
    Y = _as_float(matrix)
    beta, *_ = np.linalg.lstsq(D, Y, rcond=None)
    out = Y - U @ beta[-U.shape[1]:, :]
    return pd.DataFrame(out, index=idx, columns=matrix.columns).where(
        matrix.notna())


def stage2_local(matrix: pd.DataFrame, reference: pd.DataFrame,
                 batch: pd.Series, k: int = DEFAULT_ANCHOR_K,
                 n_pcs: int = DEFAULT_PROTECT_PCS,
                 seed: int = 0) -> pd.DataFrame:
    """Align replicates using neighbourhoods defined in ``reference`` space.

    Each cell is shifted by the gap between its own replicate's local mean
    and the consensus across replicates, so the correction is allowed to
    DEPEND ON CELL STATE -- which a single per-bin term cannot express, and
    which is why the global form gains almost nothing here (0.244 -> 0.259).

    ``reference`` should be the stage-1 output, not the raw matrix and not
    the accessibility baseline.  Anchored on accessibility this reached only
    0.290, because accessibility neighbourhoods do not match the target's
    structure; anchored on the stage-1 output it reached 0.724.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import TruncatedSVD
    idx = matrix.index
    b = batch.reindex(idx).astype(str).values
    levels = sorted(pd.unique(b))
    if len(levels) < 2:
        logger.warning("stage2_local: one replicate level (%s); nothing to "
                       "align, returning the matrix unchanged.", levels)
        return matrix.copy()
    R = _as_float(reference.reindex(idx))
    if _too_small(R, "stage2 reference"):
        return matrix.copy()
    R = R - R.mean(0, keepdims=True)
    kk = max(1, min(int(n_pcs), min(R.shape) - 1))
    Z = TruncatedSVD(n_components=kk, random_state=seed).fit_transform(R)
    Y = _as_float(matrix)
    loc = np.zeros((len(levels), *Y.shape))
    for li, lv in enumerate(levels):
        sel = np.flatnonzero(b == lv)
        nn = min(int(k), len(sel))
        _, nbr = NearestNeighbors(n_neighbors=nn).fit(Z[sel]).kneighbors(Z)
        loc[li] = Y[sel][nbr].mean(1)
    own = loc[np.array([levels.index(x) for x in b]), np.arange(len(b)), :]
    out = Y - (own - loc.mean(0))
    return pd.DataFrame(out, index=idx, columns=matrix.columns).where(
        matrix.notna())


def stage3_residual(matrix: pd.DataFrame,
                    baselines: list[pd.DataFrame]) -> pd.DataFrame:
    """The part not linearly predicted by the baselines, per bin.

    Mirrors the variance-partition residual the comparison pipeline already
    computes, so a matrix corrected here and one routed through
    ``nome_hic_compare`` answer the same question.  This is the quantity to
    lead with: raw concordance partly restates accessibility, whereas the
    residual is coordination BEYOND it -- and it is the batch-robust one
    (0.309 -> 0.703 against 0.266 -> 0.479 for the un-residualised matrix).
    """
    idx = matrix.index
    cols = [np.ones(len(idx))]
    for b in baselines:
        X = _as_float(b.reindex(idx))
        X = (X - X.mean(0, keepdims=True))
        sd = X.std(0, keepdims=True)
        cols.append(np.divide(X, sd, out=np.zeros_like(X), where=sd > 1e-12))
    D = np.column_stack(cols)
    Y = _as_float(matrix)
    beta, *_ = np.linalg.lstsq(D, Y, rcond=None)
    return pd.DataFrame(Y - D @ beta, index=idx, columns=matrix.columns
                        ).where(matrix.notna())


def two_stage(matrix: pd.DataFrame, marginal: pd.DataFrame,
              batch: pd.Series, baseline: pd.DataFrame | None = None,
              k: int = DEFAULT_K, anchor_k: int = DEFAULT_ANCHOR_K,
              transform: str = "d", n_pcs: int = DEFAULT_PROTECT_PCS,
              seed: int = 0) -> pd.DataFrame:
    """Stage 1 then stage 2, the configuration measured at cross/within 0.724."""
    protect = protect_design(baseline, matrix.index, n_pcs, seed)
    s1 = stage1_marginal(matrix, marginal, protect, k, transform, seed)
    return stage2_local(matrix, s1, batch, anchor_k, n_pcs, seed)


def three_stage(matrix: pd.DataFrame, marginal: pd.DataFrame,
                batch: pd.Series, baseline: pd.DataFrame | None = None,
                extra_baselines: list[pd.DataFrame] | None = None,
                k: int = DEFAULT_K, anchor_k: int = DEFAULT_ANCHOR_K,
                transform: str = "d", n_pcs: int = DEFAULT_PROTECT_PCS,
                seed: int = 0) -> pd.DataFrame:
    """:func:`two_stage` followed by the residual against the baselines."""
    out = two_stage(matrix, marginal, batch, baseline, k, anchor_k,
                    transform, n_pcs, seed)
    bases = ([baseline] if baseline is not None else []) + list(
        extra_baselines or [])
    if not bases:
        logger.warning("three_stage: no baselines given; stage 3 is a no-op.")
        return out
    return stage3_residual(out, bases)
