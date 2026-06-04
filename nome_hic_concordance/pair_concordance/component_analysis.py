"""Per-category AUC analysis and marginal contribution tests.

Evaluates how well each embedding separates cell types, stratified by pair
category and distance band, and tests whether coordination information
(E_coord) adds discriminative power beyond contact information (E_contact).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as _roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AUC per category / distance band
# ---------------------------------------------------------------------------


def auc_per_category(
    embedding: dict,
    cell_labels: pd.Series,
    panel: pd.DataFrame,
    delta_matrix: pd.DataFrame | None = None,
    pair_counts: pd.DataFrame | None = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Per-(category, distance_band) cell-type classification AUC.

    For each subset of pairs defined by ``category`` and ``distance_band``
    columns in *panel*, re-runs PCA on the subset features, then evaluates
    cell-type separability via 5-fold cross-validated logistic regression.

    Parameters
    ----------
    embedding : dict
        Output from one of the ``build_*_embedding`` functions; must contain
        ``'pca'`` (ndarray) and ``'cell_names'`` (list).
    cell_labels : Series
        Mapping ``cell_name -> cell_type``.
    panel : DataFrame
        Pair panel with ``pair_id``, ``category``, and optionally
        ``distance_band`` columns.
    delta_matrix : DataFrame, optional
        Cells x pairs delta-z matrix (for E_coord subset analysis). If
        provided, subset features come from this matrix.
    pair_counts : DataFrame, optional
        Stacked pair counts (for E_contact subset analysis). If provided
        and ``delta_matrix`` is not, subset features come from pair_counts.
    n_folds : int
        Number of cross-validation folds.
    random_state : int
        Random seed.

    Returns
    -------
    DataFrame with columns: ``category``, ``distance_band``, ``auc_mean``,
    ``auc_std``, ``n_pairs``, ``n_cells``.
    """
    cell_names = embedding["cell_names"]

    # Align cell labels to embedding order.
    labels_aligned = cell_labels.reindex(cell_names)
    valid_mask = labels_aligned.notna()
    if valid_mask.sum() < 2:
        logger.warning("Fewer than 2 labelled cells; cannot compute AUC.")
        return _empty_auc_df()

    labels_valid = labels_aligned[valid_mask]
    unique_types = labels_valid.unique()
    if len(unique_types) < 2:
        logger.warning(
            "Only one cell type present (%s); cannot compute AUC.",
            unique_types[0],
        )
        return _empty_auc_df()

    valid_indices = np.where(valid_mask.values)[0]

    # Determine grouping columns.
    if "category" not in panel.columns:
        raise ValueError("panel must have a 'category' column")

    has_dist_band = "distance_band" in panel.columns
    if has_dist_band:
        groups = panel.groupby(["category", "distance_band"])
    else:
        groups = panel.groupby("category")

    rows: list[dict] = []

    for group_key, group_panel in groups:
        if has_dist_band:
            category, dist_band = group_key
        else:
            category = group_key
            dist_band = "all"

        pair_ids = set(group_panel["pair_id"])
        n_pairs = len(pair_ids)

        if n_pairs == 0:
            continue

        # Build feature matrix for this subset.
        X_subset = _extract_subset_features(
            pair_ids, cell_names, delta_matrix, pair_counts, valid_indices,
        )

        if X_subset is None or X_subset.shape[1] == 0:
            rows.append({
                "category": category,
                "distance_band": dist_band,
                "auc_mean": np.nan,
                "auc_std": np.nan,
                "n_pairs": n_pairs,
                "n_cells": int(valid_mask.sum()),
            })
            continue

        auc_mean, auc_std = _cv_auc(
            X_subset, labels_valid.values, n_folds, random_state,
        )
        rows.append({
            "category": category,
            "distance_band": dist_band,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "n_pairs": n_pairs,
            "n_cells": int(valid_mask.sum()),
        })

    if not rows:
        return _empty_auc_df()

    return pd.DataFrame(rows)


def _empty_auc_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected schema."""
    return pd.DataFrame(
        columns=["category", "distance_band", "auc_mean", "auc_std",
                 "n_pairs", "n_cells"]
    )


def _extract_subset_features(
    pair_ids: set,
    cell_names: list[str],
    delta_matrix: pd.DataFrame | None,
    pair_counts: pd.DataFrame | None,
    valid_indices: np.ndarray,
) -> np.ndarray | None:
    """Extract and PCA-reduce feature matrix for a subset of pairs."""
    X = None

    if delta_matrix is not None:
        # Select columns matching pair_ids.
        cols = [c for c in delta_matrix.columns if c in pair_ids]
        if cols:
            sub = delta_matrix.iloc[:, [delta_matrix.columns.get_loc(c) for c in cols]]
            X = sub.values[valid_indices].astype(np.float64)
            # Fill NaN with 0 for PCA.
            X = np.nan_to_num(X, nan=0.0)

    elif pair_counts is not None and "pair_id" in pair_counts.columns:
        # Pivot subset.
        sub_counts = pair_counts[pair_counts["pair_id"].isin(pair_ids)]
        if not sub_counts.empty and "cell_name" in sub_counts.columns:
            pivot = sub_counts.pivot_table(
                index="cell_name", columns="pair_id", values="n_linked",
                aggfunc="sum", fill_value=0,
            )
            # Reindex to match cell_names ordering.
            pivot = pivot.reindex(
                [cell_names[i] for i in valid_indices], fill_value=0,
            )
            X = pivot.values.astype(np.float64)

    if X is None or X.shape[1] == 0:
        return None

    # Standardise.
    scaler = StandardScaler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA to top 20 (or fewer).
    n_comp = min(20, X_scaled.shape[0] - 1, X_scaled.shape[1])
    if n_comp < 1:
        return None

    pca = PCA(n_components=n_comp)
    return pca.fit_transform(X_scaled)


def _cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    random_state: int,
) -> tuple[float, float]:
    """Cross-validated AUC for cell-type classification.

    Uses one-vs-rest logistic regression. Returns (mean, std) AUC across
    folds. Handles edge cases (too few samples, single class in fold).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    if n_classes < 2:
        return np.nan, np.nan

    # Ensure enough samples per class for stratified CV.
    min_class_count = min(np.bincount(y_enc))
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < 2:
        return np.nan, np.nan

    skf = StratifiedKFold(
        n_splits=actual_folds, shuffle=True, random_state=random_state,
    )

    aucs: list[float] = []
    for train_idx, test_idx in skf.split(X, y_enc):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        # Check that both train and test have >= 2 classes.
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # OvR via OneVsRestClassifier (the old LogisticRegression(multi_class=
        # "ovr") was deprecated in sklearn 1.5); same K one-vs-rest fits, so the
        # OvR-macro AUC below is unchanged.
        clf = OneVsRestClassifier(LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=random_state))
        try:
            clf.fit(X_train, y_train)
            if n_classes == 2:
                y_prob = clf.predict_proba(X_test)[:, 1]
                auc = _roc_auc_score(y_test, y_prob)
            else:
                y_prob = clf.predict_proba(X_test)
                auc = _roc_auc_score(
                    y_test, y_prob, multi_class="ovr", average="macro",
                )
            aucs.append(auc)
        except Exception:
            continue

    if not aucs:
        return np.nan, np.nan

    return float(np.mean(aucs)), float(np.std(aucs))


# ---------------------------------------------------------------------------
# Marginal contribution: does E_coord add info beyond E_contact?
# ---------------------------------------------------------------------------


def marginal_contribution(
    contact_pca: np.ndarray,
    coord_pca: np.ndarray,
    cell_labels: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Test whether E_coord adds discriminative power beyond E_contact.

    Fits two logistic regression models:

    - **base**: E_contact PCs only.
    - **augmented**: E_contact + E_coord PCs concatenated.

    Computes the likelihood ratio test statistic
    ``LR = 2 * (LL_augmented - LL_base)`` and tests against chi-squared with
    degrees of freedom = number of added E_coord PCs.

    Parameters
    ----------
    contact_pca : ndarray, shape (n_cells, n_contact_pcs)
        PCA scores from E_contact.
    coord_pca : ndarray, shape (n_cells, n_coord_pcs)
        PCA scores from E_coord.
    cell_labels : Series
        Mapping ``cell_name -> cell_type``, indexed consistently with the
        row order of both PCA arrays.
    n_folds : int
        Number of cross-validation folds (for AUC comparison).
    random_state : int
        Random seed.

    Returns
    -------
    dict with keys ``'auc_base'``, ``'auc_augmented'``, ``'delta_auc'``,
    ``'lr_statistic'``, ``'lr_pvalue'``.
    """
    if contact_pca.shape[0] != coord_pca.shape[0]:
        raise ValueError(
            f"Row count mismatch: contact has {contact_pca.shape[0]}, "
            f"coord has {coord_pca.shape[0]}"
        )

    y = cell_labels.values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    if n_classes < 2:
        logger.warning("Fewer than 2 cell types; marginal contribution undefined.")
        return {
            "auc_base": np.nan,
            "auc_augmented": np.nan,
            "delta_auc": np.nan,
            "lr_statistic": np.nan,
            "lr_pvalue": np.nan,
        }

    X_base = contact_pca.copy()
    X_aug = np.hstack([contact_pca, coord_pca])

    # Cross-validated AUCs.
    auc_base, _ = _cv_auc(X_base, y, n_folds, random_state)
    auc_aug, _ = _cv_auc(X_aug, y, n_folds, random_state)

    # Likelihood ratio test on full data.
    lr_stat, lr_pval = _likelihood_ratio_test(
        X_base, X_aug, y_enc, n_classes, random_state,
    )

    return {
        "auc_base": auc_base,
        "auc_augmented": auc_aug,
        "delta_auc": (auc_aug - auc_base) if np.isfinite(auc_base) and np.isfinite(auc_aug) else np.nan,
        "lr_statistic": lr_stat,
        "lr_pvalue": lr_pval,
    }


def _likelihood_ratio_test(
    X_base: np.ndarray,
    X_aug: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    random_state: int,
) -> tuple[float, float]:
    """Compute the likelihood ratio statistic and p-value.

    ``LR = 2 * (LL_augmented - LL_base)`` ~ chi2(df), where
    ``df = n_aug_features - n_base_features``.
    """
    try:
        clf_base = LogisticRegression(
            max_iter=2000, solver="lbfgs", random_state=random_state,
        )
        clf_base.fit(X_base, y)
        ll_base = _log_likelihood(clf_base, X_base, y)

        clf_aug = LogisticRegression(
            max_iter=2000, solver="lbfgs", random_state=random_state,
        )
        clf_aug.fit(X_aug, y)
        ll_aug = _log_likelihood(clf_aug, X_aug, y)
    except Exception as exc:
        logger.warning("Likelihood ratio test failed: %s", exc)
        return np.nan, np.nan

    lr_stat = 2.0 * (ll_aug - ll_base)
    # Ensure non-negative (numerical issues can produce small negatives).
    lr_stat = max(lr_stat, 0.0)

    df = X_aug.shape[1] - X_base.shape[1]
    if df <= 0:
        return np.nan, np.nan

    lr_pval = float(stats.chi2.sf(lr_stat, df))
    return float(lr_stat), lr_pval


def _log_likelihood(clf: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    """Sum of log-probabilities of the true class labels."""
    proba = clf.predict_proba(X)
    # Clip to avoid log(0).
    proba = np.clip(proba, 1e-15, 1.0 - 1e-15)
    ll = 0.0
    for i, yi in enumerate(y):
        ll += np.log(proba[i, yi])
    return ll


# ---------------------------------------------------------------------------
# Matched contact comparison
# ---------------------------------------------------------------------------


def matched_contact_comparison(
    contact_features: np.ndarray,
    coord_features: np.ndarray,
    cell_labels: pd.Series,
    n_subsample: int = 100_000,
    random_state: int = 42,
) -> dict:
    """Within the top quartile of contact similarity, test coordination.

    Among cell pairs with the highest contact similarity (top 25%), tests
    whether same-type pairs have higher coordination similarity than
    different-type pairs, controlling for contact similarity via logistic
    regression.

    Parameters
    ----------
    contact_features : ndarray, shape (n_cells, n_features)
        Contact feature matrix.
    coord_features : ndarray, shape (n_cells, n_features)
        Coordination feature matrix (NaN imputed with 0).
    cell_labels : Series
        Cell type labels, aligned to row order.
    n_subsample : int
        Maximum number of cell pairs to evaluate.
    random_state : int
        Random seed.

    Returns
    -------
    dict with regression coefficients and p-values:
        ``'coef_contact'``, ``'coef_coord'``, ``'coef_intercept'``,
        ``'pvalue_contact'``, ``'pvalue_coord'``,
        ``'n_pairs_evaluated'``, ``'n_pairs_top_quartile'``.
    """
    rng = np.random.RandomState(random_state)

    n_cells = contact_features.shape[0]
    if n_cells < 4:
        logger.warning("Too few cells (%d) for matched comparison.", n_cells)
        return _empty_matched_result()

    # Impute NaN in coord features.
    coord_clean = np.nan_to_num(coord_features, nan=0.0)

    y = cell_labels.values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    if len(le.classes_) < 2:
        logger.warning("Single cell type; matched comparison not meaningful.")
        return _empty_matched_result()

    # Generate all cell pairs (or subsample).
    n_total_pairs = n_cells * (n_cells - 1) // 2
    if n_total_pairs <= n_subsample:
        # Use all pairs.
        ii, jj = np.triu_indices(n_cells, k=1)
    else:
        # Subsample pairs.
        ii_list, jj_list = [], []
        sampled = set()
        while len(sampled) < n_subsample:
            batch = rng.randint(0, n_cells, size=(n_subsample * 2, 2))
            for a, b in batch:
                if a == b:
                    continue
                key = (min(a, b), max(a, b))
                if key not in sampled:
                    sampled.add(key)
                    if len(sampled) >= n_subsample:
                        break
            if len(batch) == 0:
                break
        pairs = np.array(sorted(sampled))
        ii, jj = pairs[:, 0], pairs[:, 1]

    # Compute pairwise similarities (cosine on standardised features).
    contact_norm = _row_normalise(contact_features)
    coord_norm = _row_normalise(coord_clean)

    sim_contact = np.sum(contact_norm[ii] * contact_norm[jj], axis=1)
    sim_coord = np.sum(coord_norm[ii] * coord_norm[jj], axis=1)
    same_type = (y_enc[ii] == y_enc[jj]).astype(np.float64)

    # Filter to top quartile of contact similarity.
    q75 = np.percentile(sim_contact, 75)
    top_mask = sim_contact >= q75
    n_top = int(top_mask.sum())

    if n_top < 10:
        logger.warning(
            "Too few pairs in top quartile (%d); matched comparison unreliable.",
            n_top,
        )
        return _empty_matched_result()

    sim_contact_top = sim_contact[top_mask]
    sim_coord_top = sim_coord[top_mask]
    same_type_top = same_type[top_mask]

    # Check that both classes exist in the top quartile.
    if len(np.unique(same_type_top)) < 2:
        logger.warning(
            "Only one class in top-quartile pairs; cannot fit logistic model."
        )
        return _empty_matched_result()

    # Logistic regression: same_type ~ sim_contact + sim_coord.
    X_reg = np.column_stack([sim_contact_top, sim_coord_top])
    # Standardise predictors.
    scaler = StandardScaler()
    X_reg = scaler.fit_transform(X_reg)

    try:
        clf = LogisticRegression(
            max_iter=2000, solver="lbfgs", random_state=random_state,
        )
        clf.fit(X_reg, same_type_top)

        # Approximate p-values via Wald test.
        pvalues = _wald_pvalues(clf, X_reg, same_type_top)
    except Exception as exc:
        logger.warning("Matched comparison regression failed: %s", exc)
        return _empty_matched_result()

    return {
        "coef_contact": float(clf.coef_[0, 0]),
        "coef_coord": float(clf.coef_[0, 1]),
        "coef_intercept": float(clf.intercept_[0]),
        "pvalue_contact": float(pvalues[0]),
        "pvalue_coord": float(pvalues[1]),
        "n_pairs_evaluated": int(len(ii)),
        "n_pairs_top_quartile": n_top,
    }


def _empty_matched_result() -> dict:
    """Return a result dict with NaN placeholders."""
    return {
        "coef_contact": np.nan,
        "coef_coord": np.nan,
        "coef_intercept": np.nan,
        "pvalue_contact": np.nan,
        "pvalue_coord": np.nan,
        "n_pairs_evaluated": 0,
        "n_pairs_top_quartile": 0,
    }


def _row_normalise(X: np.ndarray) -> np.ndarray:
    """L2-normalise each row (for cosine similarity via dot product)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return X / norms


def _wald_pvalues(
    clf: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Approximate Wald-test p-values for logistic regression coefficients.

    Computes the Hessian-based standard errors and derives two-sided p-values
    from the normal distribution.
    """
    proba = clf.predict_proba(X)[:, 1]
    W = np.diag(proba * (1 - proba))

    # Add intercept column.
    X_int = np.column_stack([X, np.ones(X.shape[0])])

    try:
        # Fisher information matrix.
        fisher = X_int.T @ W @ X_int
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan)

    # Standard errors for the feature coefficients (exclude intercept).
    se = np.sqrt(np.diag(cov)[:X.shape[1]])
    se = np.where(se < 1e-15, np.inf, se)

    coefs = clf.coef_.ravel()
    z = coefs / se
    pvalues = 2.0 * stats.norm.sf(np.abs(z))

    return pvalues
