"""Does a matrix reproduce across biological replicates?

ORDINARY kNN PURITY CANNOT ANSWER THIS, which is the reason this module
exists.  A cell whose neighbours are all its own cell type FROM ITS OWN
REPLICATE scores 1.0 -- and that is precisely the failure being looked for.
On the GCH concordance, purity against cell type was 0.58 against a chance
level of 0.115 while the replicates did not overlap at all; the number said
the embedding was excellent.

Restricting each cell's neighbours to the OTHER replicate separates the two
situations.  Validated on synthetic matrices where cell types are aligned
across replicates versus duplicated per replicate: BOTH have within-replicate
purity 1.000, and only the cross-replicate number distinguishes them (1.000
against 0.301).

Read ``cross`` and ``within`` together.  A correction can always raise the
ratio by flattening cell state, so a rising ratio with ``within`` collapsing
toward chance is a correction eating the biology, not fixing the batch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["per_cell_rank", "svd_scores", "purity", "batch_separation"]


def per_cell_rank(matrix) -> np.ndarray:
    """Within-cell ranks in (-0.5, 0.5), matching ``--per-cell-norm rank``."""
    from scipy.stats import rankdata
    X = np.asarray(getattr(matrix, "to_numpy", lambda *_: matrix)(float),
                   dtype=float)
    R = np.full_like(X, np.nan)
    for i in range(X.shape[0]):
        ok = ~np.isnan(X[i])
        if ok.sum():
            R[i, ok] = rankdata(X[i, ok]) / (ok.sum() + 1) - 0.5
    return np.nan_to_num(R, nan=0.0)


def svd_scores(matrix, n_svd: int = 20, seed: int = 0) -> np.ndarray:
    """The space the pipeline embeds: per-cell rank, then truncated SVD."""
    from sklearn.decomposition import TruncatedSVD
    R = per_cell_rank(matrix)
    n = max(1, min(int(n_svd), min(R.shape) - 1))
    return TruncatedSVD(n_components=n, random_state=seed).fit_transform(R)


def purity(scores, cell_type, replicate, k: int = 30) -> dict:
    """Cell-type purity within and ACROSS replicates.

    ``ratio = cross / within`` near 1 means the cell types correspond across
    replicates; near ``chance / within`` means each replicate recovered them
    separately.
    """
    from sklearn.neighbors import NearestNeighbors
    ct = np.asarray(pd.Series(cell_type).astype(str))
    rp = np.asarray(pd.Series(replicate).astype(str))
    within = np.full(len(ct), np.nan)
    cross = np.full(len(ct), np.nan)
    for lv in sorted(pd.unique(rp)):
        own = np.flatnonzero(rp == lv)
        oth = np.flatnonzero(rp != lv)
        for dst, out in ((own, within), (oth, cross)):
            if len(dst) < 2:
                continue
            kk = min(int(k), len(dst) - (1 if dst is own else 0))
            if kk < 1:
                continue
            nn = NearestNeighbors(n_neighbors=min(kk + 1, len(dst))).fit(
                scores[dst])
            _, idx = nn.kneighbors(scores[own])
            for r, i in enumerate(own):
                nb = dst[idx[r]]
                nb = nb[nb != i][:kk]
                out[i] = float(np.mean(ct[nb] == ct[i])) if len(nb) else np.nan
    w = float(np.nanmean(within)) if np.isfinite(within).any() else float("nan")
    c = float(np.nanmean(cross)) if np.isfinite(cross).any() else float("nan")
    chance = float((pd.Series(ct).value_counts(normalize=True) ** 2).sum())
    return {"within": w, "cross": c,
            "ratio": (c / w) if w else float("nan"), "chance": chance}


def batch_separation(scores, replicate) -> float:
    """Between-replicate share of embedding variance.

    A proxy, and gameable in both directions: a correction that reduces the
    embedding toward noise drives this to ~0 while destroying cell-type
    correspondence.  Report it beside :func:`purity`, never instead of it.
    """
    lab = pd.Series(np.asarray(pd.Series(replicate).astype(str)))
    S = np.asarray(scores, float)
    tot = float(((S - S.mean(0)) ** 2).sum())
    if not tot:
        return float("nan")
    btw = sum(len(g) * float(((S[g.index].mean(0) - S.mean(0)) ** 2).sum())
              for _, g in lab.groupby(lab))
    return btw / tot
