"""Cell embeddings from pair-concordance data.

Builds three parallel embeddings:

- **E_contact**: depth-normalised linked-read counts per pair, z-scored,
  PCA top-20, UMAP.
- **E_coord**: precision-weighted delta z-scores (from ``hierarchical.py``),
  masked-loss PCA (iterative SVD), UMAP.
- **E_full**: concatenated standardised E_contact + E_coord features, PCA, UMAP.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature normalization (z-score or quantile)
# ---------------------------------------------------------------------------


def _quantile_normalize_per_feature(
    matrix: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-column rank-based quantile normalization to a uniform target.

    For each column (feature/pair), replaces values with their empirical
    quantile in [0, 1].  This makes every feature have an identical
    uniform distribution across cells, making the embedding robust to
    outliers and matching the bulk-RNA / scATAC-seq QN convention.

    NaN entries (or entries where mask=False) are preserved as NaN.
    Ranks are computed within observed values only.

    Parameters
    ----------
    matrix : ndarray of shape (n_cells, n_features)
    mask : ndarray of shape (n_cells, n_features), optional
        True where the value is observed; False/NaN positions are
        excluded from rank computation.  Defaults to ``~np.isnan(matrix)``.

    Returns
    -------
    ndarray of shape (n_cells, n_features)
        Quantile-normalized values in [0, 1]; NaN preserved.
    """
    from scipy.stats import rankdata

    out = np.full_like(matrix, np.nan, dtype=np.float64)
    if mask is None:
        mask = ~np.isnan(matrix)

    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        col_mask = mask[:, j]
        n_obs = int(col_mask.sum())
        if n_obs == 0:
            continue
        if n_obs == 1:
            out[col_mask, j] = 0.5
            continue
        ranks = rankdata(col[col_mask], method="average")
        # Map ranks to (0, 1) uniformly.  rank/(n+1) avoids hitting 0 or 1.
        out[col_mask, j] = ranks / (n_obs + 1)
    return out


def _normalize_features(
    matrix: np.ndarray,
    method: str = "zscore",
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Standardize a (cells x features) matrix per-feature (per column).

    Parameters
    ----------
    matrix : ndarray
    method : "zscore" | "quantile"
        ``"zscore"`` — per-column z-score via ``StandardScaler``.
        ``"quantile"`` — per-column rank-based QN to a uniform [0,1] target.
    mask : ndarray, optional
        Used only by ``"quantile"`` for handling missing entries.
    """
    if method == "zscore":
        m = StandardScaler(with_mean=True, with_std=True).fit_transform(matrix)
        return np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    if method == "quantile":
        m = _quantile_normalize_per_feature(matrix, mask=mask)
        # Center on 0 for PCA stability — QN gives values in (0,1), so subtract 0.5.
        m = m - 0.5
        return np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    raise ValueError(f"Unknown feature normalization method: {method!r}")


# ---------------------------------------------------------------------------
# Masked-loss PCA via iterative SVD
# ---------------------------------------------------------------------------


def sparse_pca_em(
    X: np.ndarray,
    n_components: int = 20,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Masked-loss PCA via iterative SVD.

    Missing values (NaN) are initialised with column means, then iteratively
    replaced by the rank-``n_components`` reconstruction until convergence.

    Parameters
    ----------
    X : ndarray, shape (n_cells, n_features)
        Data matrix with NaN for missing entries.
    n_components : int
        Number of principal components to retain.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on relative change in imputed values.

    Returns
    -------
    scores : ndarray, shape (n_cells, n_components)
        Cell scores in PC space.
    components : ndarray, shape (n_components, n_features)
        Principal component loadings.
    """
    X_work = X.copy().astype(np.float64)
    mask = np.isnan(X_work)

    # Initialise missing entries with column means.
    col_means = np.nanmean(X_work, axis=0)
    # Guard against all-NaN columns.
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for j in range(X_work.shape[1]):
        X_work[mask[:, j], j] = col_means[j]

    n_missing = int(mask.sum())
    n_components = min(n_components, X_work.shape[0], X_work.shape[1])

    if n_missing == 0:
        # No missing data -- standard PCA.
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(X_work)
        return scores, pca.components_

    prev_imputed = X_work[mask].copy()

    for it in range(max_iter):
        # Centre.
        means = X_work.mean(axis=0)
        X_centred = X_work - means

        # Truncated SVD.
        U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)
        U_k = U[:, :n_components]
        S_k = S[:n_components]
        Vt_k = Vt[:n_components, :]

        # Reconstruct and fill missing positions only.
        recon = U_k @ np.diag(S_k) @ Vt_k + means
        X_work[mask] = recon[mask]

        # Check convergence on imputed values.
        cur_imputed = X_work[mask]
        change = np.linalg.norm(cur_imputed - prev_imputed)
        scale = np.linalg.norm(cur_imputed) + 1e-10
        rel_change = change / scale
        prev_imputed = cur_imputed.copy()

        if rel_change < tol:
            logger.debug(
                "sparse_pca_em converged at iteration %d (rel_change=%.2e)",
                it + 1, rel_change,
            )
            break
    else:
        logger.debug(
            "sparse_pca_em did not converge after %d iterations "
            "(rel_change=%.2e)", max_iter, rel_change,
        )

    scores = U_k * S_k[np.newaxis, :]
    components = Vt_k

    return scores, components


# ---------------------------------------------------------------------------
# E_contact: depth-normalised linked-read embedding
# ---------------------------------------------------------------------------


def build_contact_embedding(
    pair_counts_stacked: pd.DataFrame,
    panel: pd.DataFrame,
    n_pca: int = 20,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    feature_norm: str = "zscore",
) -> dict:
    """Build E_contact embedding from linked-read counts.

    Per-cell feature per pair is the depth-normalised linked-read count::

        u_ij^c = log(1 + n_linked) - log(lambda_c)

    where ``lambda_c = sum(n_linked)`` across all pairs for cell *c*.
    Features are normalised per pair across cells (``feature_norm``),
    then PCA and UMAP are applied.

    Parameters
    ----------
    pair_counts_stacked : DataFrame
        All cells stacked; must contain columns ``cell_name``, ``pair_id``,
        and ``n_linked``.
    panel : DataFrame
        Pair panel (used only for ordering; must have ``pair_id``).
    n_pca : int
        Number of principal components.
    n_neighbors, min_dist : float
        UMAP parameters.
    random_state : int
        Random seed for reproducibility.
    feature_norm : "zscore" | "quantile"
        Per-pair normalisation across cells.  ``"zscore"`` (default) for
        the classical pipeline; ``"quantile"`` for rank-based QN that
        matches the user's bulk / scATAC-seq style 250 kb-bin recipe
        (robust to outliers).

    Returns
    -------
    dict with keys ``'matrix'``, ``'pca'``, ``'umap'``, ``'cell_names'``.
    """
    import umap

    required = {"cell_name", "pair_id", "n_linked"}
    missing = required - set(pair_counts_stacked.columns)
    if missing:
        raise ValueError(f"pair_counts_stacked missing columns: {sorted(missing)}")

    # Pivot to cells x pairs matrix.
    pivot = pair_counts_stacked.pivot_table(
        index="cell_name",
        columns="pair_id",
        values="n_linked",
        aggfunc="sum",
        fill_value=0,
    )

    cell_names = list(pivot.index)
    n_linked = pivot.values.astype(np.float64)

    if len(cell_names) < 2:
        raise ValueError(
            f"Need at least 2 cells for embedding, got {len(cell_names)}"
        )

    # Compute depth-normalised features.
    lambda_c = n_linked.sum(axis=1)
    # Guard against zero-depth cells.
    lambda_c = np.where(lambda_c > 0, lambda_c, 1.0)

    matrix = np.log1p(n_linked) - np.log(lambda_c)[:, np.newaxis]

    # Per-pair normalisation across cells.
    matrix_z = _normalize_features(matrix, method=feature_norm)
    logger.info(f"E_contact feature normalisation: {feature_norm}")

    # PCA.
    n_comp = min(n_pca, matrix_z.shape[0] - 1, matrix_z.shape[1])
    if n_comp < 1:
        n_comp = 1
    pca = PCA(n_components=n_comp, random_state=random_state)
    pca_scores = pca.fit_transform(matrix_z)

    logger.info(
        "E_contact PCA: %d cells x %d pairs -> %d PCs (%.1f%% variance)",
        matrix_z.shape[0], matrix_z.shape[1], n_comp,
        pca.explained_variance_ratio_.sum() * 100,
    )

    # UMAP.
    n_nbrs = min(n_neighbors, len(cell_names) - 1)
    if n_nbrs < 2:
        n_nbrs = 2
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_nbrs,
        min_dist=min_dist,
        random_state=random_state,
    )
    umap_coords = reducer.fit_transform(pca_scores)

    return {
        "matrix": matrix_z,
        "pca": pca_scores,
        "umap": umap_coords,
        "cell_names": cell_names,
    }


# ---------------------------------------------------------------------------
# E_coord: precision-weighted delta-z embedding
# ---------------------------------------------------------------------------


def build_coord_embedding(
    delta_matrix: pd.DataFrame,
    n_pca: int = 20,
    pca_method: str = "masked",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    feature_norm: str = "zscore",
) -> dict:
    """Build E_coord embedding from delta-z values.

    Input ``delta_matrix`` is cells x pairs of ``delta_z`` values (with NaN
    where data is insufficient). Missing values are handled either by
    masked-loss PCA (iterative SVD) or simple mean imputation.

    Parameters
    ----------
    delta_matrix : DataFrame
        Cells (rows) x pairs (columns), with NaN for missing.
    n_pca : int
        Number of principal components.
    pca_method : ``"masked"`` or ``"mean_impute"``
        How to handle NaN during PCA.
    n_neighbors, min_dist : float
        UMAP parameters.
    random_state : int
        Random seed for reproducibility.
    feature_norm : "zscore" | "quantile"
        Per-pair normalisation across cells.  ``"zscore"`` (default) for
        the classical pipeline.  ``"quantile"`` performs rank-based QN
        (within observed values per pair) — robust to outliers; matches
        the user's bulk / scATAC-seq style preprocessing recipe.

    Returns
    -------
    dict with keys ``'matrix'``, ``'pca'``, ``'umap'``, ``'cell_names'``.
    """
    import umap

    if pca_method not in ("masked", "mean_impute"):
        raise ValueError(
            f"pca_method must be 'masked' or 'mean_impute', got {pca_method!r}"
        )

    cell_names = list(delta_matrix.index)
    if len(cell_names) < 2:
        raise ValueError(
            f"Need at least 2 cells for embedding, got {len(cell_names)}"
        )

    X = delta_matrix.values.astype(np.float64)

    # Drop columns that are entirely NaN.
    col_valid = ~np.all(np.isnan(X), axis=0)
    X = X[:, col_valid]

    if X.shape[1] == 0:
        raise ValueError("All pairs are NaN across all cells; cannot embed.")

    # Per-pair normalisation across cells.
    if feature_norm == "zscore":
        col_means = np.nanmean(X, axis=0)
        col_stds = np.nanstd(X, axis=0)
        col_stds = np.where(col_stds < 1e-10, 1.0, col_stds)
        X_std = (X - col_means[np.newaxis, :]) / col_stds[np.newaxis, :]
    elif feature_norm == "quantile":
        mask = ~np.isnan(X)
        X_std = _quantile_normalize_per_feature(X, mask=mask) - 0.5
    else:
        raise ValueError(f"Unknown feature_norm: {feature_norm!r}")
    logger.info(f"E_coord feature normalisation: {feature_norm}")

    # PCA.
    n_comp = min(n_pca, X_std.shape[0] - 1, X_std.shape[1])
    if n_comp < 1:
        n_comp = 1

    if pca_method == "masked":
        pca_scores, components = sparse_pca_em(
            X_std, n_components=n_comp, max_iter=50, tol=1e-4,
        )
    else:
        # Mean imputation: fill NaN with 0 (since already standardised).
        X_filled = np.nan_to_num(X_std, nan=0.0)
        pca = PCA(n_components=n_comp, random_state=random_state)
        pca_scores = pca.fit_transform(X_filled)

    nan_frac = np.isnan(X).sum() / X.size if X.size > 0 else 0.0
    logger.info(
        "E_coord PCA (%s): %d cells x %d pairs -> %d PCs "
        "(%.1f%% missing values)",
        pca_method, X.shape[0], X.shape[1], n_comp, nan_frac * 100,
    )

    # UMAP.
    n_nbrs = min(n_neighbors, len(cell_names) - 1)
    if n_nbrs < 2:
        n_nbrs = 2
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_nbrs,
        min_dist=min_dist,
        random_state=random_state,
    )
    umap_coords = reducer.fit_transform(pca_scores)

    return {
        "matrix": X_std,
        "pca": pca_scores,
        "umap": umap_coords,
        "cell_names": cell_names,
    }


# ---------------------------------------------------------------------------
# E_full: integrated contact + coordination embedding
# ---------------------------------------------------------------------------


def build_full_embedding(
    contact_features: np.ndarray,
    coord_features: np.ndarray,
    cell_names: list[str],
    n_pca: int = 20,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> dict:
    """Build E_full by concatenating standardised contact and coord features.

    Both input matrices should already be standardised. Any remaining NaN in
    ``coord_features`` is imputed with 0 before concatenation.

    Parameters
    ----------
    contact_features : ndarray, shape (n_cells, n_contact_features)
        Standardised contact matrix (from E_contact).
    coord_features : ndarray, shape (n_cells, n_coord_features)
        Standardised coordination matrix (from E_coord), NaN allowed.
    cell_names : list of str
        Cell identifiers, length must match n_cells.
    n_pca : int
        Number of principal components.
    n_neighbors, min_dist : float
        UMAP parameters.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``'pca'``, ``'umap'``, ``'cell_names'``.
    """
    import umap

    n_cells = contact_features.shape[0]
    if coord_features.shape[0] != n_cells:
        raise ValueError(
            f"Row count mismatch: contact has {n_cells}, "
            f"coord has {coord_features.shape[0]}"
        )
    if len(cell_names) != n_cells:
        raise ValueError(
            f"cell_names length ({len(cell_names)}) != n_cells ({n_cells})"
        )

    if n_cells < 2:
        raise ValueError(
            f"Need at least 2 cells for embedding, got {n_cells}"
        )

    # Impute any remaining NaN in coord features.
    coord_clean = np.nan_to_num(coord_features, nan=0.0)

    # Concatenate.
    combined = np.hstack([contact_features, coord_clean])

    # PCA.
    n_comp = min(n_pca, combined.shape[0] - 1, combined.shape[1])
    if n_comp < 1:
        n_comp = 1
    pca = PCA(n_components=n_comp, random_state=random_state)
    pca_scores = pca.fit_transform(combined)

    logger.info(
        "E_full PCA: %d cells x %d features -> %d PCs (%.1f%% variance)",
        combined.shape[0], combined.shape[1], n_comp,
        pca.explained_variance_ratio_.sum() * 100,
    )

    # UMAP.
    n_nbrs = min(n_neighbors, n_cells - 1)
    if n_nbrs < 2:
        n_nbrs = 2
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_nbrs,
        min_dist=min_dist,
        random_state=random_state,
    )
    umap_coords = reducer.fit_transform(pca_scores)

    return {
        "pca": pca_scores,
        "umap": umap_coords,
        "cell_names": cell_names,
    }
