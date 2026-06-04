"""J3 — per-channel embedding of the joint multi-state concordance tensor.

Design §6.  The joint per-cell feature object is a tensor **cells × regions × 4
channels** (the four co-state Δ's: open / closed / poised / active_methylated).
Because the channels live on very different scales (co-poised large,
co-active-methylated small), per-channel normalization is *mandatory first*;
then the default embedding is **per-channel SVD → concatenate scores → (joint
SVD/skip-top →) single UMAP**, which bounds each channel's contribution, is
scale-robust, and is *diagnosable*: the per-channel component AUC reveals
**which co-state channel carries cell-type identity** (a result, not just a
method knob).

This module is the numeric core (numpy + scikit-learn only — no UMAP / tensorly
dependency, so it is importable and unit-testable everywhere).  The final joint
SVD + UMAP + scatter is delegated to
:func:`nome_hic_concordance.visualization.run_umap_and_plot` (lazy ``umap``
import), and the optional tensor decomposition to ``tensorly``, both wired in
``scripts/joint_multistate_embedding.py``.

This is **not** the RWR path: SVD/UMAP of the Δ tensor is unsupervised
dimensionality reduction, not cross-cell concordance smoothing on a marginal
graph — so it does not reintroduce the leakage the existence test avoids.  NaNs
(regions a cell has no contact in) are filled with the per-bin mean — innocuous
mean-imputation, not similarity-based borrowing.
"""

from __future__ import annotations

import numpy as np

from .matrix_ops import _quantile_normalize_per_bin


def per_cell_rank(M: np.ndarray) -> np.ndarray:
    """Rank each cell's regions into (0, 1), preserving NaN (per-cell-rank step).

    Removes per-cell scale/coverage effects: a cell's Δ values become within-cell
    ranks, so a cell that simply has larger-magnitude Δ everywhere is put on the
    same footing as the others before cross-cell normalization."""
    from scipy.stats import rankdata
    out = np.full_like(M, np.nan, dtype=np.float64)
    for i in range(M.shape[0]):
        obs = np.isfinite(M[i])
        n = int(obs.sum())
        if n == 0:
            continue
        if n == 1:
            out[i, obs] = 0.5
        else:
            out[i, obs] = rankdata(M[i, obs], method="average") / (n + 1)
    return out


def normalize_channel(M: np.ndarray, do_rank: bool = True) -> np.ndarray:
    """Per-channel normalization stack: per-cell-rank → per-bin-QN → recenter.

    ``M`` is one channel's ``cells × regions`` matrix (NaN where unobserved).
    Returns a NaN-free, per-bin-centered matrix on a common footing across
    channels.  Per-bin mean-imputation fills the remaining NaN (innocuous; not
    similarity-based)."""
    X = per_cell_rank(M) if do_rank else M.astype(np.float64)
    X = _quantile_normalize_per_bin(X)            # per-bin rank → uniform(0,1)
    mu = np.nanmean(X, axis=0, keepdims=True)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    X = np.where(np.isfinite(X), X, mu)           # per-bin mean impute
    return X - X.mean(axis=0, keepdims=True)      # recenter per bin


def per_channel_svd(channels_norm: list, k_per_channel: int, seed: int = 0):
    """Truncated-SVD each normalized channel → ``cells × k`` scores; concatenate.

    Returns ``(scores_concat, blocks)`` where ``scores_concat`` is
    ``cells × (K·k)`` (column-blocks in channel order) and ``blocks`` is the list
    of per-channel ``cells × k`` score matrices (for the per-channel AUC
    diagnostic).  ``k`` is capped at each channel's rank."""
    from sklearn.decomposition import TruncatedSVD
    blocks = []
    for M in channels_norm:
        k = max(1, min(k_per_channel, M.shape[0] - 1, M.shape[1]))
        if M.shape[1] <= k:
            # too few regions to reduce — standardize columns and pad to k
            sd = M.std(axis=0, keepdims=True); sd = np.where(sd < 1e-12, 1.0, sd)
            block = (M - M.mean(axis=0, keepdims=True)) / sd
            if block.shape[1] < k:
                block = np.hstack([block, np.zeros((block.shape[0],
                                                     k - block.shape[1]))])
        else:
            block = TruncatedSVD(n_components=k,
                                 random_state=seed).fit_transform(M)
        blocks.append(block)
    return np.hstack(blocks) if blocks else np.empty((0, 0)), blocks


def channel_auc(blocks: list, channel_names: list, y, method: str = "lr",
                n_folds: int = 5, seed: int = 0) -> dict:
    """Per-channel cross-validated AUC of each channel's SVD-score block vs ``y``.

    The §6 diagnostic — WHICH co-state channel carries cell-type identity.
    Returns ``{channel_name: (auc_mean, auc_std)}``."""
    from . import coordination_eval as ce
    out = {}
    for name, block in zip(channel_names, blocks):
        out[name] = ce.score(block, np.asarray(y), method, n_folds, seed)
    return out


def build_embedding(channels: list, channel_names: list, k_per_channel: int = 10,
                    do_rank: bool = True, seed: int = 0):
    """Full per-channel embedding: normalize each channel, per-channel SVD,
    concatenate the scores.

    ``channels`` is a list of K ``cells × regions`` Δ matrices (one per
    co-state), ``channel_names`` their labels.  Returns
    ``(scores_concat, blocks, normed)`` — the concatenated ``cells × (K·k)``
    score matrix to feed UMAP (joint SVD/skip-top + plot), the per-channel score
    blocks (for :func:`channel_auc`), and the normalized channel matrices."""
    normed = [normalize_channel(M, do_rank=do_rank) for M in channels]
    scores, blocks = per_channel_svd(normed, k_per_channel, seed=seed)
    return scores, blocks, normed


def concordance_rwr_embed(channels_norm: list, channel_names: list,
                          k_per_channel: int = 10, n_neighbors: int = 15,
                          restart_prob: float = 0.5, n_passes: int = 1,
                          seed: int = 0):
    """LEAKAGE-FREE concordance-graph RWR densification, then re-embed.

    Random-walk-with-restart densifies the sparse per-(cell, region) Δ by
    borrowing across cells — but the leak the single-context investigation
    caught (`SIGNED_CONCORDANCE_AND_RWR_DESIGN.md` §6b) was never RWR itself, it
    was the **graph basis**: the original pass-0 graph was built from
    ``QN(mean accessibility) ⊕ QN(mean methylation)`` — the *marginals* — which
    injects marginal cell-type structure into the smoothed concordance.

    Here the k-NN graph is built from the **concordance embedding itself** (the
    per-channel SVD of the Δ tensor), never from the marginals, so it cannot
    inject that structure.  Each pass: graph ← concordance scores → RWR-smooth
    the ORIGINAL normalized channels on that graph (scHiCluster-style: refine
    the graph, always smooth the raw data) → re-embed.  Reuses the validated
    :func:`nome_hic_concordance.imputation.build_knn_graph` /
    :func:`rwr_impute_matrices`.

    **Caveat:** a concordance-built graph can *amplify / confirm* whatever the
    SVD already separated — validate the result against the permutation /
    split-half checks, do not trust it standalone.

    Returns ``(scores, blocks, smoothed_channels, info)``."""
    from .imputation import build_knn_graph, rwr_impute_matrices
    scores, blocks = per_channel_svd(channels_norm, k_per_channel, seed=seed)
    smoothed = channels_norm
    info = {"n_passes": int(max(1, n_passes)), "n_neighbors": int(n_neighbors),
            "restart_prob": float(restart_prob),
            "graph_basis": "concordance_scores (leakage-free)"}
    for _ in range(max(1, int(n_passes))):
        _, P = build_knn_graph(scores, k=n_neighbors)          # graph ← concordance
        mats = {nm: M for nm, M in zip(channel_names, channels_norm)}
        imputed, _ = rwr_impute_matrices(mats, P, restart_prob=restart_prob)
        smoothed = [imputed[nm] for nm in channel_names]
        scores, blocks = per_channel_svd(smoothed, k_per_channel, seed=seed)
    return scores, blocks, smoothed, info
