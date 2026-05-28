"""Batch correction for the mean-accessibility matrix.

Applied **before** the k-NN graph is built for RWR imputation, so that
batch structure doesn't leak into cell-cell similarities and propagate
into Δ_signed.

Four methods, chosen for different complexity / capability trade-offs:

- ``"center"`` — Per-batch column-mean subtraction.  Simplest.  Removes
  per-batch location shifts only.
- ``"zscore"`` — Per-batch column mean + std normalization.  Removes
  both location and scale per-batch effects.
- ``"combat"`` — Empirical-Bayes batch correction (Johnson et al. 2007).
  Field standard in bulk methylation and RNA-seq.  Requires the optional
  ``pycombat`` / ``inmoose`` dependency.
- ``"harmony"`` — Iterative cell-batch correction in PC space
  (Korsunsky et al. 2019).  Field standard in scRNA-seq.  Requires the
  optional ``harmonypy`` dependency.

Also provides :func:`batch_anova_f_per_bin` for diagnostic
"how much batch effect is left" measurements.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALID_METHODS = ("none", "center", "zscore", "combat", "harmony")


def apply_batch_correction(
    matrix: pd.DataFrame,
    batch_labels: pd.Series,
    method: str = "combat",
) -> pd.DataFrame:
    """Apply batch correction to a cells × bins matrix.

    Parameters
    ----------
    matrix : DataFrame
        Cells × bins (e.g., mean accessibility per (cell, bin)).
        NaNs are allowed and propagated through the simpler methods;
        ComBat / Harmony require NaN-free inputs (caller should impute).
    batch_labels : Series
        ``cell_name -> batch_label``.  Cells without a label are grouped
        into a synthetic ``"__unknown_batch__"`` batch with a warning.
    method : str
        One of :data:`VALID_METHODS`.

    Returns
    -------
    DataFrame, same shape and index/columns as ``matrix``, with the
    per-batch effect removed.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {VALID_METHODS}; got {method!r}"
        )
    if method == "none":
        return matrix

    # Align batch labels to matrix index.
    batch_labels = pd.Series(batch_labels).reindex(matrix.index)
    n_na = int(batch_labels.isna().sum())
    if n_na > 0:
        logger.warning(
            f"{n_na} of {len(matrix)} cells have no batch label; "
            f"grouping them as '__unknown_batch__'."
        )
        batch_labels = batch_labels.fillna("__unknown_batch__")

    n_batches = batch_labels.nunique()
    if n_batches < 2:
        logger.warning(
            f"Only {n_batches} unique batch label(s) found "
            f"({batch_labels.unique().tolist()}); batch correction is a no-op."
        )
        return matrix

    logger.info(
        f"Applying batch correction (method='{method}', {n_batches} batches, "
        f"{len(matrix)} cells, {matrix.shape[1]:,} bins)"
    )

    if method == "center":
        return _per_batch_center(matrix, batch_labels)
    if method == "zscore":
        return _per_batch_zscore(matrix, batch_labels)
    if method == "combat":
        return _combat_correction(matrix, batch_labels)
    if method == "harmony":
        return _harmony_correction(matrix, batch_labels)
    raise ValueError(f"unreachable: method={method!r}")


# ---------------------------------------------------------------------------
# Method 1 — per-batch mean centering
# ---------------------------------------------------------------------------


def _per_batch_center(
    matrix: pd.DataFrame, batch_labels: pd.Series,
) -> pd.DataFrame:
    """Subtract per-batch column means.

    Each batch ends up with mean 0 per bin.  Only location shifts
    are removed; per-batch variance differences are preserved.
    """
    out = matrix.copy()
    for batch, cells in batch_labels.groupby(batch_labels).groups.items():
        sub = out.loc[cells]
        out.loc[cells] = sub.sub(sub.mean(axis=0), axis=1)
    return out


# ---------------------------------------------------------------------------
# Method 2 — per-batch z-score
# ---------------------------------------------------------------------------


def _per_batch_zscore(
    matrix: pd.DataFrame, batch_labels: pd.Series,
) -> pd.DataFrame:
    """Per-batch column mean + std normalization.

    Each batch ends up with mean 0 AND std 1 per bin (where std is well-
    defined).  Removes both location and scale per-batch effects.
    Zero-std bins are left at zero (no division).
    """
    out = matrix.copy()
    for batch, cells in batch_labels.groupby(batch_labels).groups.items():
        sub = out.loc[cells]
        mean = sub.mean(axis=0)
        std = sub.std(axis=0).replace(0.0, 1.0)
        out.loc[cells] = (sub - mean).div(std, axis=1)
    return out


# ---------------------------------------------------------------------------
# Method 3 — ComBat empirical Bayes
# ---------------------------------------------------------------------------


def _combat_correction(
    matrix: pd.DataFrame, batch_labels: pd.Series,
) -> pd.DataFrame:
    """ComBat empirical-Bayes batch correction (Johnson et al. 2007).

    Requires the optional `pycombat` (or `inmoose`) dependency.
    Falls back with a clear error message if neither is available.

    ComBat expects features × samples, so the matrix is transposed
    before being passed in and the result is transposed back.
    """
    # NaN-impute (ComBat doesn't tolerate NaNs).  Use per-bin mean.
    if matrix.isna().any().any():
        logger.info(
            "  ComBat: NaN-imputing the accessibility matrix with per-bin means "
            "(ComBat doesn't tolerate NaNs)."
        )
        col_means = matrix.mean(axis=0)
        matrix = matrix.fillna(col_means)

    pycombat = _try_import_pycombat()
    # Most pycombat APIs accept (data: features × samples DataFrame, batch: list[str]).
    data_t = matrix.T  # bins × cells
    batch_list = batch_labels.tolist()
    try:
        corrected_t = pycombat(data_t, batch_list)
    except TypeError:
        # Some versions take a numpy array; try that as fallback.
        corrected_t = pycombat(data_t.to_numpy(), batch_list)
        corrected_t = pd.DataFrame(corrected_t, index=data_t.index, columns=data_t.columns)

    return corrected_t.T


def _try_import_pycombat():
    """Return a callable ComBat implementation, trying multiple package names."""
    try:
        from combat.pycombat import pycombat
        return pycombat
    except ImportError:
        pass
    try:
        from inmoose.pycombat import pycombat_norm
        return pycombat_norm
    except ImportError:
        pass
    raise ImportError(
        "ComBat correction requires either the 'combat' or 'inmoose' "
        "Python package, neither of which is installed.\n"
        "Install with one of:\n"
        "  pip install combat\n"
        "  pip install inmoose\n"
        "Or use --batch-correction zscore for a no-dependency alternative."
    )


# ---------------------------------------------------------------------------
# Method 4 — Harmony
# ---------------------------------------------------------------------------


def _harmony_correction(
    matrix: pd.DataFrame, batch_labels: pd.Series,
    n_components: int = 50,
) -> pd.DataFrame:
    """Harmony iterative batch correction (Korsunsky et al. 2019).

    Requires the optional `harmonypy` dependency.  Operates in PC space:

    1. PCA → top ``n_components`` components of ``matrix``.
    2. Harmony iteratively soft-clusters cells across batches and
       computes per-cluster batch offsets, correcting each cell.
    3. Reconstruct cells × bins from the corrected PC scores.

    Note: reconstruction is approximate (lossy in the principal
    components beyond ``n_components``).  Use ``n_components`` as high
    as practical (default 50, capped at the matrix rank).
    """
    try:
        import harmonypy as hm
    except ImportError:
        raise ImportError(
            "Harmony correction requires the 'harmonypy' Python package, "
            "which is not installed.\n"
            "Install with:  pip install harmonypy\n"
            "Or use --batch-correction zscore (no dependencies) or "
            "--batch-correction combat instead."
        )

    if matrix.isna().any().any():
        logger.info(
            "  Harmony: NaN-imputing the accessibility matrix with per-bin means."
        )
        matrix = matrix.fillna(matrix.mean(axis=0))

    from sklearn.decomposition import PCA
    n_comp = max(1, min(n_components, matrix.shape[0] - 1, matrix.shape[1] - 1))
    pca = PCA(n_components=n_comp, random_state=42)
    pcs = pca.fit_transform(matrix.to_numpy(dtype=np.float64))

    meta = pd.DataFrame({"batch": batch_labels.values}, index=matrix.index)
    harmony_out = hm.run_harmony(pcs, meta, "batch")
    # harmonypy stores corrected coordinates in ho.Z_corr (shape: components × cells).
    corrected_pcs = harmony_out.Z_corr.T  # cells × n_comp

    # Reconstruct cells × bins.
    reconstructed = corrected_pcs @ pca.components_ + pca.mean_
    return pd.DataFrame(
        reconstructed, index=matrix.index, columns=matrix.columns,
    )


# ---------------------------------------------------------------------------
# Diagnostic — per-bin batch ANOVA F-statistic
# ---------------------------------------------------------------------------


def batch_anova_f_per_bin(
    matrix: pd.DataFrame, batch_labels: pd.Series,
) -> pd.Series:
    """Per-bin one-way ANOVA F-statistic across batches.

    High F values indicate strong residual batch effect at that bin;
    near-1 values indicate negligible batch effect.

    Returns
    -------
    Series of F-statistics, one per bin.
    """
    import scipy.stats as ss

    batch_labels = pd.Series(batch_labels).reindex(matrix.index)
    f_stats = np.full(matrix.shape[1], np.nan)
    cols = matrix.to_numpy()
    batch_values = batch_labels.values
    unique_batches = [b for b in batch_labels.dropna().unique()
                      if (batch_labels == b).sum() > 1]
    if len(unique_batches) < 2:
        return pd.Series(f_stats, index=matrix.columns, name="batch_anova_F")
    for j in range(matrix.shape[1]):
        col = cols[:, j]
        if np.isnan(col).any():
            continue
        groups = [col[batch_values == b] for b in unique_batches]
        # Skip bins where any group has zero variance (ANOVA undefined).
        if any(np.std(g) < 1e-10 for g in groups):
            continue
        f, _ = ss.f_oneway(*groups)
        f_stats[j] = float(f)
    return pd.Series(f_stats, index=matrix.columns, name="batch_anova_F")
