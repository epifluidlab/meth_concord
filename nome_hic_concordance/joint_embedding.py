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


def per_channel_svd(channels_norm: list, k_per_channel: int, seed: int = 0,
                    skip_top: int = 0):
    """Truncated-SVD each normalized channel → ``cells × k`` scores; concatenate.

    Returns ``(scores_concat, blocks)`` where ``scores_concat`` is
    ``cells × (K·k)`` (column-blocks in channel order) and ``blocks`` is the list
    of per-channel ``cells × k`` score matrices (for the per-channel AUC
    diagnostic).  ``k`` is capped at each channel's rank.

    ``skip_top`` drops the first ``skip_top`` SVD components **of each channel**
    before keeping the next ``k_per_channel`` — the per-channel analogue of the
    single-context ``--skip-top-svd``, for when a channel's leading component is
    a technical axis rather than cell type (use :func:`per_component_auc` to
    decide).  The per-cell-rank normalization usually removes per-cell depth, so
    the default is 0."""
    from sklearn.decomposition import TruncatedSVD
    blocks = []
    for M in channels_norm:
        if M.shape[1] <= k_per_channel:
            # too few regions to reduce — standardize columns and pad to k
            sd = M.std(axis=0, keepdims=True); sd = np.where(sd < 1e-12, 1.0, sd)
            block = (M - M.mean(axis=0, keepdims=True)) / sd
            if block.shape[1] < k_per_channel:
                block = np.hstack([block, np.zeros(
                    (block.shape[0], k_per_channel - block.shape[1]))])
        else:
            want = min(k_per_channel + max(0, skip_top), M.shape[0] - 1, M.shape[1])
            full = TruncatedSVD(n_components=max(1, want),
                                random_state=seed).fit_transform(M)
            s = min(max(0, skip_top), max(0, full.shape[1] - 1))
            block = full[:, s:s + k_per_channel]
            if block.shape[1] < k_per_channel:       # pad if skip ate too many
                block = np.hstack([block, np.zeros(
                    (block.shape[0], k_per_channel - block.shape[1]))])
        blocks.append(block)
    return np.hstack(blocks) if blocks else np.empty((0, 0)), blocks


def per_component_auc(blocks: list, channel_names: list, y, max_comp: int = 10):
    """Univariate cell-type AUC of **each** of the top SVD components, per channel.

    For each channel and each of its first ``max_comp`` SVD components, the
    single-component separation of the labels (sign-corrected; OvR-macro for
    >2 classes).  This is the §6 / single-context check that decides whether a
    channel's leading component is a technical axis to skip (low AUC) versus
    cell-type signal (high AUC).  Returns ``{channel_name: [auc_comp0, ...]}``."""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder
    yy = LabelEncoder().fit_transform(np.asarray(y))
    nc = len(np.unique(yy))
    out = {}
    for name, block in zip(channel_names, blocks):
        aucs = []
        for j in range(min(block.shape[1], max_comp)):
            comp = block[:, j]
            if np.std(comp) < 1e-12:
                aucs.append(0.5); continue
            if nc == 2:
                a = roc_auc_score(yy, comp); aucs.append(float(max(a, 1 - a)))
            else:
                per = [max(a, 1 - a) for a in
                       (roc_auc_score((yy == c).astype(int), comp)
                        for c in range(nc))]
                aucs.append(float(np.mean(per)))
        out[name] = aucs
    return out


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
                    do_rank: bool = True, seed: int = 0, skip_top: int = 0):
    """Full per-channel embedding: normalize each channel, per-channel SVD,
    concatenate the scores.

    ``channels`` is a list of K ``cells × regions`` Δ matrices (one per
    co-state), ``channel_names`` their labels.  Returns
    ``(scores_concat, blocks, normed)`` — the concatenated ``cells × (K·k)``
    score matrix to feed UMAP (joint SVD/skip-top + plot), the per-channel score
    blocks (for :func:`channel_auc`), and the normalized channel matrices.
    ``skip_top`` drops the top N SVD components of each channel (see
    :func:`per_channel_svd`)."""
    normed = [normalize_channel(M, do_rank=do_rank) for M in channels]
    scores, blocks = per_channel_svd(normed, k_per_channel, seed=seed,
                                     skip_top=skip_top)
    return scores, blocks, normed


def rwr_embed(channels_norm: list, channel_names: list, graph_basis: np.ndarray,
              k_per_channel: int = 10, n_neighbors: int = 15,
              restart_prob: float = 0.5, n_refine_passes: int = 0,
              auto_select: bool = False, max_passes: int = 5,
              refinement_basis: str = "postnorm", seed: int = 0,
              skip_top: int = 0):
    """Multi-pass RWR densification — the SAME scheme as
    :func:`nome_hic_concordance.imputation.run_rwr_pipeline` — then re-embed.

    * **Pass 0** builds the k-NN graph from ``graph_basis`` =
      ``QN(mean accessibility ⊕ mean methylation)`` (the marginal / pass-0 basis).
    * **Pass ≥1 (refinement)** rebuilds the graph from the *concordance* of the
      previous pass — ``refinement_basis`` of the concatenated smoothed channels
      (``"postnorm"`` = per-cell-rank → per-bin-QN → row-recenter then QN, which
      strips the per-cell shift; ``"raw"`` = QN only) — and **re-imputes from the
      ORIGINAL channels** (only the graph evolves between passes).
    * **Auto-select** (``auto_select=True``): run up to ``max_passes`` refinement
      passes and keep the one whose post-norm concordance has the largest top
      SVD eigengap (``_top_eigengap``, the label-free proxy validated to track
      cell-type separation among *refined* passes).  Pass 0 is never selected
      (its dominant axis is the accessibility/batch confound).  Patience 1.

    Reuses ``build_knn_graph`` / ``rwr_impute_matrices`` / ``post_norm_for_knn``
    / ``quantile_normalize_for_knn`` / ``_top_eigengap`` unchanged.  RWR is
    appropriate here (this is the *embedding*, not the beyond-marginal
    hypothesis test the existence test already settled, §6b).

    Returns ``(scores, blocks, smoothed_channels, info)``; ``info['selected_pass']``
    and ``info['passes']`` record the per-pass bases + eigengaps."""
    from .imputation import (_top_eigengap, build_knn_graph, post_norm_for_knn,
                             quantile_normalize_for_knn, rwr_impute_matrices)
    mats = {nm: M for nm, M in zip(channel_names, channels_norm)}

    def _impute(P):
        imp, _ = rwr_impute_matrices(mats, P, restart_prob=restart_prob)
        return imp

    def _concord(imp):                       # joint concordance for the refine graph
        return np.hstack([imp[nm] for nm in channel_names])

    # Pass 0: marginal graph.
    _, P = build_knn_graph(
        quantile_normalize_for_knn(np.asarray(graph_basis, np.float64)),
        k=n_neighbors)
    imputed = _impute(P)
    passes = [{"pass": 0, "basis": "QN(mean accessibility ⊕ mean methylation)"}]
    selected, sel_pass = imputed, 0

    total = max_passes if auto_select else n_refine_passes
    best_eg, no_improve, patience = -np.inf, 0, 1
    prev = _concord(imputed)
    for p in range(1, int(total) + 1):
        basis = (quantile_normalize_for_knn(post_norm_for_knn(prev))
                 if refinement_basis == "postnorm"
                 else quantile_normalize_for_knn(prev))
        _, P = build_knn_graph(basis, k=n_neighbors)
        imputed = _impute(P)                 # re-impute from ORIGINAL channels
        cur = _concord(imputed)
        rec = {"pass": p,
               "basis": f"{refinement_basis}(concordance pass {p - 1})"}
        if auto_select:
            eg = _top_eigengap(post_norm_for_knn(cur))
            rec["eigengap"] = float(eg)
            if eg > best_eg:
                best_eg, selected, sel_pass, no_improve = (
                    eg, {k_: v.copy() for k_, v in imputed.items()}, p, 0)
            else:
                no_improve += 1
        else:
            selected, sel_pass = imputed, p
        passes.append(rec)
        prev = cur
        if auto_select and no_improve >= patience:
            break

    smoothed = [selected[nm] for nm in channel_names]
    scores, blocks = per_channel_svd(smoothed, k_per_channel, seed=seed,
                                     skip_top=skip_top)
    info = {"n_neighbors": int(n_neighbors), "restart_prob": float(restart_prob),
            "graph_basis": "QN(mean accessibility ⊕ mean methylation)",
            "refinement_basis": refinement_basis, "auto_select": bool(auto_select),
            "selected_pass": int(sel_pass), "passes": passes}
    return scores, blocks, smoothed, info
