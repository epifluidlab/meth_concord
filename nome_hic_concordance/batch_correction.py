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


_UNKNOWN = "__unknown_batch__"


def _impute_for_dense_solver(matrix: pd.DataFrame, who: str) -> pd.DataFrame:
    """Fill NaNs so a solver that rejects them (PCA, ComBat) can run.

    Per-bin mean is the first pass, but a bin that NO cell covers has a NaN
    mean, so that pass leaves it NaN and the solver still rejects the whole
    matrix.  This is routine for the compartment baseline: a 250 kb bin at a
    chromosome end or low-contact region can be empty for every cell.  Such a
    column carries no signal, so zero-fill it -- PCA centres per feature and
    ComBat standardises per feature, so a constant column is inert either way.

    The caller re-masks the corrected matrix with the ORIGINAL NaN pattern
    (``corrected.where(raw.notna())``), so these filled positions never reach
    the regression; the fill exists only to let the dense solver run.
    """
    if not matrix.isna().any().any():
        return matrix
    logger.info(f"  {who}: NaN-imputing with per-bin means.")
    filled = matrix.fillna(matrix.mean(axis=0))
    dead = filled.columns[filled.isna().any(axis=0)]
    if len(dead):
        logger.info(
            f"  {who}: {len(dead)} bin(s) empty for every cell — zero-filled "
            f"for the solver (inert; re-masked to NaN afterward)."
        )
        filled = filled.fillna(0.0)
    return filled


def _coerce_batch_df(batch_labels, index: pd.Index) -> pd.DataFrame:
    """Normalize batch labels to a cells × batch-variables DataFrame.

    Accepts a Series (one batch variable) or a DataFrame (several batch
    variables, e.g. cell-culture / library-prep / sequencing batch).
    Reindexes to ``index``, fills missing labels with ``__unknown_batch__``
    per column, and casts to str.
    """
    if isinstance(batch_labels, pd.Series):
        df = batch_labels.to_frame(name=(batch_labels.name or "batch"))
    else:
        df = pd.DataFrame(batch_labels)
    df = df.reindex(index)
    for c in df.columns:
        n_na = int(df[c].isna().sum())
        if n_na:
            logger.warning(
                f"{n_na} of {len(index)} cells have no '{c}' label; "
                f"grouping them as '{_UNKNOWN}'.")
        df[c] = df[c].fillna(_UNKNOWN).astype(str)
    return df


def _composite_label(batch_df: pd.DataFrame) -> pd.Series:
    """Cartesian-product label across batch variables (for single-factor
    methods center/zscore/combat, which model one categorical batch)."""
    if batch_df.shape[1] == 1:
        return batch_df.iloc[:, 0]
    comp = batch_df.astype(str).agg("|".join, axis=1)
    n_groups = comp.nunique()
    small = int((comp.value_counts() < 3).sum())
    logger.info(
        f"Combined {list(batch_df.columns)} into {n_groups} composite batch "
        f"groups for single-factor correction "
        f"({small} group(s) have < 3 cells — consider harmony, which models "
        f"the batch variables jointly rather than as a product).")
    return comp


def apply_batch_correction(
    matrix: pd.DataFrame,
    batch_labels,
    method: str = "combat",
    random_state: int = 42,
) -> pd.DataFrame:
    """Apply batch correction to a cells × bins matrix.

    Parameters
    ----------
    matrix : DataFrame
        Cells × bins (e.g., mean accessibility per (cell, bin)).
        NaNs are allowed and propagated through the simpler methods;
        ComBat / Harmony require NaN-free inputs (caller should impute).
    batch_labels : Series | DataFrame
        ``cell_name -> batch_label`` (one batch variable), OR a cells ×
        batch-variables DataFrame for MULTIPLE batch sources (e.g.
        cell-culture batch + library-prep batch + sequencing batch).
        Harmony harmonizes over all variables jointly (its native
        ``vars_use`` list); center/zscore/combat collapse them to a
        single composite (cartesian-product) batch.  Cells without a
        label are pooled into ``"__unknown_batch__"`` with a warning.
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

    batch_df = _coerce_batch_df(batch_labels, matrix.index)
    levels = {c: int(batch_df[c].nunique()) for c in batch_df.columns}
    if all(v < 2 for v in levels.values()):
        logger.warning(
            f"No batch variable has >= 2 levels ({levels}); "
            f"batch correction is a no-op.")
        return matrix

    logger.info(
        f"Applying batch correction (method='{method}', "
        f"variables={list(batch_df.columns)}, levels={levels}, "
        f"{len(matrix)} cells, {matrix.shape[1]:,} bins)"
    )

    if method == "harmony":
        # Harmony models the batch variables jointly.
        return _harmony_correction(matrix, batch_df, random_state=random_state)

    # Single-factor methods: collapse to one composite batch.
    composite = _composite_label(batch_df)
    if method == "center":
        return _per_batch_center(matrix, composite)
    if method == "zscore":
        return _per_batch_zscore(matrix, composite)
    if method == "combat":
        return _combat_correction(matrix, composite)
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
    # NaN-impute (ComBat doesn't tolerate NaNs).
    matrix = _impute_for_dense_solver(matrix, "ComBat")

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


def harmonize_embedding(
    scores, batch_labels, random_state: int = 42, theta=None, n_threads=None,
):
    """Harmony on an existing low-dim embedding (the SVD/PCA scores).

    ``n_threads`` parallelises harmony: it is passed to harmonypy's ``ncores``
    (its own clustering parallelism) AND used to raise the numpy/BLAS thread
    count for the correction matrix algebra (via threadpoolctl).  This matters
    most when harmony runs on the FULL bin matrix (--n-svd 0), where there are
    thousands of features instead of a handful of SVD components.

    ``theta`` is harmony's diversity-clustering penalty — the batch-correction
    STRENGTH.  Higher theta mixes batches more aggressively (default in
    harmonypy is 2.0); lower theta is gentler and preserves more of the
    original structure.  A scalar is broadcast to one value per batch variable;
    ``None`` uses harmonypy's default.

    This is the STANDARD harmony workflow (scanpy's ``harmony_integrate``):
    correct the components that feed UMAP, AFTER all normalisation.

    Why it must be here and not on the raw matrix: applying harmony to the raw
    values and THEN normalising (per-cell rank / quantile) UNDOES the
    correction -- those per-cell transforms discard harmony's cross-cell value
    alignment and keep only the within-cell rank structure, which still carries
    the batch.  Harmony also has to operate on the SAME representation that
    feeds UMAP; correcting its own internal PCA and then running a separate SVD
    corrects the wrong space.  Both were happening, which is why a
    harmony-'corrected' UMAP still split every cell type by batch.

    ``scores`` is (n_cells, n_components) and is returned corrected, same shape.
    """
    scores = np.ascontiguousarray(np.asarray(scores, dtype=np.float64))
    n_cells, n_comp = scores.shape
    batch_df = _coerce_batch_df(batch_labels, pd.RangeIndex(n_cells))
    # Nothing to align if every cell shares one batch level in every variable.
    if all(batch_df[c].nunique() < 2 for c in batch_df.columns):
        return scores
    try:
        import harmonypy as hm
    except ImportError:
        raise ImportError(
            "Harmony correction requires the 'harmonypy' package "
            "(pip install harmonypy)."
        )
    meta = batch_df.reset_index(drop=True)
    vars_use = list(batch_df.columns)
    kw = {}
    if theta is not None:                     # broadcast the strength per var
        kw["theta"] = [float(theta)] * len(vars_use)
    if n_threads and int(n_threads) > 0:
        kw["ncores"] = int(n_threads)         # harmonypy's own parallelism
    # Also let the numpy/BLAS matrix algebra (the bulk of harmony) use n_threads.
    from contextlib import nullcontext
    try:
        from threadpoolctl import threadpool_limits
        tp = (threadpool_limits(int(n_threads))
              if n_threads and int(n_threads) > 0 else nullcontext())
    except Exception:
        tp = nullcontext()
    with tp:
        try:
            out = hm.run_harmony(scores, meta, vars_use,
                                 random_state=random_state, **kw)
        except TypeError:
            logger.warning("harmonypy.run_harmony rejected random_state / "
                           "ncores (old version); harmony may be "
                           "single-threaded and non-reproducible.")
            kw.pop("ncores", None)
            out = hm.run_harmony(scores, meta, vars_use, **kw)
    Z = np.asarray(out.Z_corr)
    if Z.shape == (n_comp, n_cells):
        Z = Z.T
    elif Z.shape != (n_cells, n_comp):
        raise ValueError(f"Unexpected harmonypy Z_corr shape {Z.shape}; "
                         f"expected ({n_cells}, {n_comp}) or transpose.")
    return np.ascontiguousarray(Z, dtype=np.float64)


def _harmony_correction(
    matrix: pd.DataFrame, batch_df: pd.DataFrame,
    n_components: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Harmony iterative batch correction (Korsunsky et al. 2019).

    Requires the optional `harmonypy` dependency.  Operates in PC space:

    1. PCA → top ``n_components`` components of ``matrix``.
    2. Harmony iteratively soft-clusters cells across batches and
       computes per-cluster batch offsets, correcting each cell.  When
       ``batch_df`` has multiple columns (e.g. cell-culture + library-prep
       + sequencing batch), all are passed to Harmony's ``vars_use`` and
       harmonized jointly.
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

    matrix = _impute_for_dense_solver(matrix, "Harmony")

    from sklearn.decomposition import PCA
    n_comp = max(1, min(n_components, matrix.shape[0] - 1, matrix.shape[1] - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    pcs = pca.fit_transform(matrix.to_numpy(dtype=np.float64))

    # Harmony aligns meta rows positionally with the PCA rows (matrix order).
    meta = batch_df.reset_index(drop=True)
    vars_use = list(batch_df.columns)
    # Pass the seed explicitly: harmonypy's k-means init is stochastic, and
    # relying on its internal default couples our reproducibility to their
    # version (and makes a --random-seed sweep leave harmony frozen).
    try:
        harmony_out = hm.run_harmony(pcs, meta, vars_use,
                                     random_state=random_state)
    except TypeError:                       # harmonypy without random_state
        logger.warning(
            "harmonypy.run_harmony has no random_state parameter "
            "(old version); harmony output may not be reproducible.")
        harmony_out = hm.run_harmony(pcs, meta, vars_use)
    # harmonypy's Z_corr orientation varies across versions: older releases
    # return (n_components, n_cells), newer ones (n_cells, n_components).
    # Orient robustly to (n_cells, n_comp) before reconstructing.
    n_cells = matrix.shape[0]
    Z = np.asarray(harmony_out.Z_corr)
    if Z.shape == (n_comp, n_cells):
        Z = Z.T
    elif Z.shape != (n_cells, n_comp):
        raise ValueError(
            f"Unexpected harmonypy Z_corr shape {Z.shape}; "
            f"expected ({n_cells}, {n_comp}) or ({n_comp}, {n_cells})."
        )
    corrected_pcs = Z  # (n_cells, n_comp)

    # Reconstruct cells × bins.
    reconstructed = corrected_pcs @ pca.components_ + pca.mean_
    return pd.DataFrame(
        reconstructed, index=matrix.index, columns=matrix.columns,
    )


# ---------------------------------------------------------------------------
# Diagnostic — per-bin batch ANOVA F-statistic
# ---------------------------------------------------------------------------


def batch_anova_f_per_bin(
    matrix: pd.DataFrame, batch_labels,
) -> pd.Series:
    """Per-bin one-way ANOVA F-statistic across batches.

    High F values indicate strong residual batch effect at that bin;
    near-1 values indicate negligible batch effect.  Accepts a Series
    (one batch variable) or a DataFrame (several); a DataFrame is
    collapsed to a composite (cartesian-product) batch for the one-way
    test.

    Returns
    -------
    Series of F-statistics, one per bin.
    """
    import scipy.stats as ss

    batch_df = _coerce_batch_df(batch_labels, matrix.index)
    batch_labels = _composite_label(batch_df)
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
