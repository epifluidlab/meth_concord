"""Pre-Layer-2 RWR imputation on the soft-count matrices.

Builds a k-NN graph across cells from a passed-in feature matrix
(typically QN(mean_accessibility) for the initial pass, QN(Δ_signed) for
optional refinement passes), then propagates the raw soft-count matrices
through that graph via random-walk-with-restart.  The resulting imputed
counts go into Layer 2 in place of the raw ones.

This is the key piece that lets per-(cell, bin) Δ carry signal beyond
the global prior in the sparse-per-bin regime (median ``n_linked = 1``
per (cell, bin) at 250 kb).  Without it, Layer 2 sees ``n = 1`` and the
posterior is prior-dominated by ~91% (decision-doc §1.3).

Public entry points:

- :func:`quantile_normalize_for_knn` — convenience wrapper around the
  shared QN helper that NaN-imputes to the centered median (0) so the
  result is safe for cosine similarity.
- :func:`build_knn_graph` — cosine k-NN graph from any cells × features
  matrix; returns symmetric adjacency ``W`` and row-normalized
  transition matrix ``P``.
- :func:`rwr_impute_matrices` — apply RWR to one or more cells × bins
  matrices using a shared transition matrix.
- :func:`run_rwr_pipeline` — full orchestrated pipeline with optional
  iterative refinement passes (the function that ``concordance.py``
  calls).  Takes a Layer-2 callback for refinement-pass Δ recomputation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .matrix_ops import _quantile_normalize_per_bin
from .utils import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QN helper (NaN-imputing wrapper around matrix_ops._quantile_normalize_per_bin)
# ---------------------------------------------------------------------------


def quantile_normalize_for_knn(matrix: np.ndarray) -> np.ndarray:
    """Per-bin quantile normalize, then center and NaN-impute to 0.

    Result is a NaN-free, zero-centred matrix where each bin has an
    identical uniform-in-(-0.5, 0.5) distribution across cells (modulo
    cells with NaN at that bin, which contribute the centered median 0).

    Suitable as input to cosine similarity for k-NN graph construction.
    """
    mask = ~np.isnan(matrix)
    qn = _quantile_normalize_per_bin(matrix, mask=mask)
    return np.nan_to_num(qn - 0.5, nan=0.0)


def post_norm_for_knn(matrix: np.ndarray) -> np.ndarray:
    """Step-7-style normalization applied to a Δ_signed matrix for k-NN.

    Mirrors :func:`matrix_ops.process_concordance_matrix` with
    ``per_cell_norm='rank'``, ``matrix_norm='quantile'`` and
    ``post_norm_recenter=True``:

        1. per-cell rank  — each row → rank/(n_bins+1) − 0.5
        2. per-bin QN     — each col → rank/(n_cells+1) − 0.5
        3. row recenter   — subtract per-row mean

    Used as the refinement-pass k-NN basis when
    ``refinement_basis='postnorm'``.  Removing the per-cell shift from the
    basis stops accessibility-level / depth community structure from being
    re-injected into the refined graph.  NaN entries are imputed to 0
    before ranking (matches the Step-7 default).
    """
    from scipy.stats import rankdata

    X = np.nan_to_num(matrix.astype(np.float64), nan=0.0)
    n_cells, n_bins = X.shape
    # 1. per-cell rank.
    pc = np.empty_like(X)
    for i in range(n_cells):
        pc[i] = rankdata(X[i], method="average") / (n_bins + 1) - 0.5
    # 2. per-bin QN (rank to uniform, centered).
    qn = _quantile_normalize_per_bin(pc, mask=np.ones_like(pc, dtype=bool)) - 0.5
    # 3. row recenter.
    return qn - qn.mean(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# k-NN graph construction
# ---------------------------------------------------------------------------


def build_knn_graph(
    features: np.ndarray,
    k: int = 15,
    metric: str = "cosine",
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build a symmetric k-NN graph + row-normalized transition matrix.

    Parameters
    ----------
    features : ndarray, cells × features.
        NaN-free.  Use :func:`quantile_normalize_for_knn` to prepare a
        raw mean-accessibility (or concordance) matrix.
    k : int
        Number of nearest neighbours per cell (excluding self).  Will be
        clamped to ``n_cells - 1``.
    metric : str
        Distance metric for sklearn's ``NearestNeighbors``.  Default
        ``"cosine"`` matches scHiCluster and is recommended for
        QN'd inputs.

    Returns
    -------
    W : csr_matrix (n_cells × n_cells)
        Symmetric adjacency matrix.  Entries are 1 - distance (i.e.
        similarity); higher values are closer neighbours.
    P : csr_matrix (n_cells × n_cells)
        Row-normalized transition matrix.  ``P[i, j]`` is the
        probability of stepping from cell ``i`` to cell ``j``.

    Notes
    -----
    Implementation: ``sklearn.neighbors.NearestNeighbors`` queries top
    ``k + 1`` neighbours (including self), then drops self.  For
    moderate ``n_cells`` (≤ a few thousand) this is fast.  For very
    large datasets, sklearn's ball-tree / kd-tree fallback is still
    reasonable since cosine distance has good index support.
    """
    from sklearn.neighbors import NearestNeighbors

    n_cells = features.shape[0]
    if n_cells < 2:
        raise ValueError(
            f"build_knn_graph requires >= 2 cells; got {n_cells}."
        )
    k_eff = min(k, n_cells - 1)
    if k_eff < k:
        logger.warning(
            f"Requested k={k} > n_cells-1={n_cells - 1}; clamped to {k_eff}."
        )

    # Query top k+1 (to include self) then drop self.
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, algorithm="auto")
    nn.fit(features)
    distances, indices = nn.kneighbors(features, return_distance=True)
    # First column is self (distance 0); skip it.
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Build adjacency: similarities = 1 - distance for cosine.
    # For other metrics this is still a monotone transform of distance.
    sim = 1.0 - distances
    sim = np.clip(sim, 0.0, None)  # numerical safety

    rows = np.repeat(np.arange(n_cells), k_eff)
    cols = indices.ravel()
    data = sim.ravel()
    W = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

    # Symmetrise.  If j is in i's top-k but i is not in j's top-k,
    # the (j, i) entry will be 0 in the upper-half; averaging gives sim/2.
    # That asymmetry is preserved by W = (W + W^T) / 2 so the resulting
    # weight reflects a "soft" neighbourhood.
    W = (W + W.T) * 0.5

    # Row-normalise → transition matrix.  Rows that are all-zero (no
    # neighbours: shouldn't happen post-symmetrize unless isolated) stay
    # all-zero, which means RWR returns the original value for that cell.
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    D_inv = sp.diags(1.0 / row_sums)
    P = D_inv @ W

    return W.tocsr(), P.tocsr()


# ---------------------------------------------------------------------------
# RWR solvers
# ---------------------------------------------------------------------------


def _rwr_closed_form(
    F: np.ndarray,
    P: sp.csr_matrix,
    restart_prob: float,
) -> np.ndarray:
    """Closed-form RWR: ``F_imputed = r · (I − (1 − r) · P)^(−1) · F``.

    Only suitable for small N (the matrix inverse is dense N × N).
    """
    n = P.shape[0]
    # (I - (1-r)*P) is sparse; its inverse is dense.  Solve linear
    # system rather than form the inverse explicitly.
    A = sp.eye(n) - (1.0 - restart_prob) * P
    # scipy.sparse.linalg.spsolve doesn't handle multiple RHS cleanly
    # for dense matrices; np.linalg.solve on A.toarray() is simpler at
    # this size.
    return float(restart_prob) * np.linalg.solve(A.toarray(), F)


def _rwr_iterative(
    F: np.ndarray,
    P: sp.csr_matrix,
    restart_prob: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int, bool]:
    """Iterative RWR: ``F_{t+1} = (1 − r) · P · F_t + r · F_0``.

    Returns ``(F_imputed, n_iter, converged)``.
    """
    F_out = F.copy()
    decay = 1.0 - restart_prob
    converged = False
    n_iter = max_iter
    for it in range(1, max_iter + 1):
        F_new = decay * (P @ F_out) + restart_prob * F
        diff = float(np.max(np.abs(F_new - F_out)))
        F_out = F_new
        if diff < tol:
            n_iter = it
            converged = True
            break
    return F_out, n_iter, converged


def rwr_impute_matrices(
    matrices: dict[str, np.ndarray],
    P: sp.csr_matrix,
    restart_prob: float = 0.5,
    solver: str = "auto",
    max_iter: int = 30,
    tol: float = 1e-4,
    n_cells_direct_threshold: int = 5000,
) -> tuple[dict[str, np.ndarray], dict]:
    """Apply RWR to one or more cells × bins matrices using a shared ``P``.

    Parameters
    ----------
    matrices : dict[name, ndarray]
        Each ndarray is ``(n_cells, n_bins)``.  All matrices share the
        same number of cells (rows) and the same row order (= row order
        of ``P``).
    P : csr_matrix
        Transition matrix from :func:`build_knn_graph`.
    restart_prob : float
        ``r`` in the RWR formulation.  Default 0.5 (scHiCluster's choice).
    solver : str
        ``"auto"`` (direct if n_cells ≤ threshold), ``"direct"``, or
        ``"iterative"``.
    max_iter, tol : iterative-solver parameters.
    n_cells_direct_threshold : int
        Under ``solver="auto"``, switch from direct (closed-form) to
        iterative when ``n_cells`` exceeds this.  Default 5000.

    Returns
    -------
    (imputed, diagnostics) :
        ``imputed`` has the same keys + shapes as ``matrices``.
        ``diagnostics`` records solver mode, iterations used, etc.
    """
    n_cells = P.shape[0]
    if not matrices:
        return {}, {"solver_used": None, "n_iter": 0, "converged": True}

    # Pick solver.
    if solver == "auto":
        use_direct = n_cells <= n_cells_direct_threshold
    elif solver == "direct":
        use_direct = True
    elif solver == "iterative":
        use_direct = False
    else:
        raise ValueError(
            f"solver must be 'auto', 'direct', or 'iterative'; got {solver!r}."
        )

    imputed: dict[str, np.ndarray] = {}
    diagnostics = {
        "solver_used": "direct" if use_direct else "iterative",
        "restart_prob": float(restart_prob),
        "n_cells": int(n_cells),
        "n_matrices": len(matrices),
        "matrices": {},
    }

    if use_direct:
        # Compute the propagation factor once and reuse for every matrix.
        # Factor = r · (I − (1 − r) · P)^(−1).
        # We solve A · F = r · F_raw with the same A for every matrix.
        A = sp.eye(n_cells) - (1.0 - restart_prob) * P
        A_dense = A.toarray()
        # LU factor once.
        from scipy.linalg import lu_factor, lu_solve
        lu = lu_factor(A_dense)
        for name, F in matrices.items():
            F = np.asarray(F, dtype=np.float64)
            F_imp = float(restart_prob) * lu_solve(lu, F)
            imputed[name] = F_imp
            diagnostics["matrices"][name] = {"shape": list(F.shape)}
    else:
        for name, F in matrices.items():
            F = np.asarray(F, dtype=np.float64)
            F_imp, n_iter, converged = _rwr_iterative(
                F, P, restart_prob, max_iter, tol,
            )
            imputed[name] = F_imp
            diagnostics["matrices"][name] = {
                "shape": list(F.shape),
                "n_iter": int(n_iter),
                "converged": bool(converged),
            }

    return imputed, diagnostics


# ---------------------------------------------------------------------------
# Full pipeline with optional refinement passes
# ---------------------------------------------------------------------------


def _top_eigengap(M: np.ndarray, top_k: int = 10,
                  random_state: int = 0) -> float:
    """Ratio of the top two SVD eigenvalue shares of ``M``.

    Used as the label-free criterion for auto-selecting the number of
    refinement passes: among *refined* passes it peaks when the single
    cell-type axis is most dominant, before over-smoothing spreads
    variance back out.  (Pass 0 is excluded by the caller because its
    dominant axis is the accessibility/batch confound, not cell type.)
    """
    from sklearn.decomposition import TruncatedSVD

    k = min(top_k, M.shape[0] - 1, M.shape[1])
    if k < 2:
        return float("inf")
    svd = TruncatedSVD(n_components=k, random_state=random_state).fit(M)
    vr = svd.explained_variance_ratio_
    return float(vr[0] / vr[1]) if vr[1] > 0 else float("inf")


def run_rwr_pipeline(
    soft_count_matrices: dict[str, np.ndarray],
    initial_feature_matrix: pd.DataFrame,
    layer2_fn: Callable[[dict[str, np.ndarray]], pd.DataFrame],
    n_refine_passes: int = 0,
    k: int = 15,
    restart_prob: float = 0.5,
    solver: str = "auto",
    max_iter: int = 30,
    tol: float = 1e-4,
    output_dir: str | None = None,
    refinement_basis: str = "raw",
    auto_select_passes: bool = False,
    max_auto_passes: int = 5,
    random_state: int = 0,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    """Orchestrate the full RWR + (optional) refinement loop.

    Parameters
    ----------
    soft_count_matrices : dict[name, ndarray]
        The raw soft-count matrices to be imputed.  All shape
        ``(n_cells, n_bins)``, same row order.  Typical keys:
        ``"k_open"``, ``"k_closed"``, ``"k_mixed"``, ``"n_linked"``,
        ``"k_u_open"``, ``"k_u_closed"``, ``"k_u_mixed"``, ``"n_unlinked"``.
    initial_feature_matrix : DataFrame
        Cells × bins matrix used to build the **pass-0 k-NN graph**.
        Typically the QN(mean_accessibility) matrix.  Row order must
        match the soft-count matrices.
    layer2_fn : callable
        ``f(imputed_dict) -> Δ_signed_df`` — given the imputed soft
        counts, produce a cells × bins Δ_signed DataFrame.  Called
        once per pass.  Row + column order of the returned DataFrame
        must match ``initial_feature_matrix``.
    n_refine_passes : int
        ``0`` (default): one-shot RWR using QN(mean_accessibility) k-NN,
        then one Layer-2 fit.
        ``>= 1``: after pass 0, rebuild k-NN from QN(Δ_signed) of the
        previous pass and re-impute (from original raw soft counts);
        Layer 2 is called once per pass.
    k, restart_prob, solver, max_iter, tol :
        RWR parameters (see :func:`rwr_impute_matrices`).
    output_dir : str, optional
        If provided, saves the k-NN graph and diagnostics to disk.
    refinement_basis : {"raw", "postnorm"}
        How to derive the refinement-pass k-NN basis from the previous
        pass's Δ_signed.  "raw" (default): per-column-rank(Δ) only.
        "postnorm": apply Step-7 normalization (per-cell rank → per-bin
        QN → row recenter) via :func:`post_norm_for_knn` first, so the
        per-cell shift is stripped before the graph is built.  See
        :func:`nome_hic_concordance.concordance.compute_signed_concordance_matrix`
        for when to use each.
    auto_select_passes : bool
        If True, ignore ``n_refine_passes`` as a fixed count and instead
        run up to ``max_auto_passes`` refinement passes, returning the one
        whose post-norm Δ has the largest top SVD eigengap (label-free;
        validated to track cell-type separation among refined passes).
        Stops early once the eigengap drops (patience 1).  Pass 0 is never
        selected (its dominant axis is the accessibility/batch confound).
    max_auto_passes : int
        Upper bound on refinement passes when ``auto_select_passes`` is True.

    Returns
    -------
    (imputed_counts_final, delta_signed_final, diagnostics) :
        ``imputed_counts_final`` is the dict of imputed soft-count
        matrices from the final pass.
        ``delta_signed_final`` is the Layer-2 output of the final pass.
        ``diagnostics`` records per-pass info (k-NN strength, convergence,
        Δ between consecutive passes for QC).
    """
    n_cells = initial_feature_matrix.shape[0]
    if n_cells < 2:
        raise ValueError(
            f"run_rwr_pipeline needs >= 2 cells; got {n_cells}."
        )
    if refinement_basis not in ("raw", "postnorm"):
        raise ValueError(
            f"refinement_basis must be 'raw' or 'postnorm'; got {refinement_basis!r}."
        )

    all_diagnostics: dict = {
        "n_cells": int(n_cells),
        "k": int(k),
        "restart_prob": float(restart_prob),
        "n_refine_passes": int(n_refine_passes),
        "refinement_basis": refinement_basis,
        "solver_requested": solver,
        "passes": [],
    }

    # Pass 0: k-NN from QN(initial_feature_matrix).
    qn_init = quantile_normalize_for_knn(initial_feature_matrix.to_numpy(dtype=np.float64))
    logger.info(
        f"RWR pass 0: building k-NN graph from QN(initial features) "
        f"(shape={qn_init.shape}, k={k})"
    )
    W, P = build_knn_graph(qn_init, k=k)
    imputed, rwr_diag = rwr_impute_matrices(
        soft_count_matrices, P,
        restart_prob=restart_prob, solver=solver,
        max_iter=max_iter, tol=tol,
    )
    delta_df = layer2_fn(imputed)
    all_diagnostics["passes"].append({
        "pass": 0,
        "knn_basis": "QN(initial_feature_matrix)",
        "rwr": rwr_diag,
    })

    if output_dir is not None:
        out = ensure_dir(output_dir)
        sp.save_npz(out / "cell_knn_graph_pass0.npz", W)

    # Auto-stop: run up to ``max_auto_passes`` refinement passes and keep the
    # one whose post-norm Δ has the largest top eigengap (the label-free
    # proxy validated to track cell-type AUC among *refined* passes).  Pass 0
    # is never a candidate — its dominant axis is the accessibility/batch
    # confound, which inflates the eigengap without carrying cell type.
    total_passes = max_auto_passes if auto_select_passes else n_refine_passes
    all_diagnostics["auto_select_passes"] = bool(auto_select_passes)

    best_eg = -np.inf
    best_imputed: dict[str, np.ndarray] | None = None
    best_delta: pd.DataFrame | None = None
    best_pass = 0
    no_improve = 0
    patience = 1  # stop after this many consecutive non-improving passes

    prev_delta = delta_df.copy()
    for pass_idx in range(1, total_passes + 1):
        # Rebuild k-NN from the previous pass's Δ_signed.  Two bases:
        #   "raw"      — per-column-rank(Δ) only (legacy).  Keeps any
        #                per-cell shift that survived into Δ, which under
        #                MLE re-injects depth/accessibility community
        #                structure into the refined graph.
        #   "postnorm" — apply the Step-7 normalization (per-cell rank →
        #                per-bin QN → row recenter) first, stripping the
        #                per-cell shift before the graph is built.
        prev_arr = prev_delta.to_numpy(dtype=np.float64)
        if refinement_basis == "postnorm":
            qn_delta = quantile_normalize_for_knn(post_norm_for_knn(prev_arr))
            basis_label = f"postnorm(Δ_signed_pass_{pass_idx - 1})"
        else:
            qn_delta = quantile_normalize_for_knn(prev_arr)
            basis_label = f"QN(Δ_signed_pass_{pass_idx - 1})"
        logger.info(
            f"RWR pass {pass_idx}: rebuilding k-NN graph from {basis_label}"
        )
        W, P = build_knn_graph(qn_delta, k=k)
        # IMPORTANT: re-impute from the ORIGINAL raw soft counts, not
        # the previous pass's imputed values.  Only the k-NN graph
        # evolves between passes (see design-doc §6 #5b).
        imputed, rwr_diag = rwr_impute_matrices(
            soft_count_matrices, P,
            restart_prob=restart_prob, solver=solver,
            max_iter=max_iter, tol=tol,
        )
        delta_df = layer2_fn(imputed)
        # Per-pass convergence diagnostic: ‖Δ_p − Δ_{p-1}‖.
        delta_change = float(
            np.nanmean(np.abs(delta_df.to_numpy() - prev_delta.to_numpy()))
        )
        pass_diag = {
            "pass": pass_idx,
            "knn_basis": basis_label,
            "mean_abs_delta_change": delta_change,
            "rwr": rwr_diag,
        }

        if auto_select_passes:
            eg = _top_eigengap(
                post_norm_for_knn(delta_df.to_numpy(dtype=np.float64)),
                random_state=random_state,
            )
            pass_diag["eigengap"] = float(eg)
            improved = eg > best_eg
            if improved:
                best_eg = eg
                best_imputed = {k_: v_.copy() for k_, v_ in imputed.items()}
                best_delta = delta_df.copy()
                best_pass = pass_idx
                no_improve = 0
            else:
                no_improve += 1
            logger.info(
                f"  pass {pass_idx} eigengap={eg:.3f} "
                f"(best={best_eg:.3f} at pass {best_pass})"
            )

        all_diagnostics["passes"].append(pass_diag)
        if output_dir is not None:
            sp.save_npz(
                ensure_dir(output_dir) / f"cell_knn_graph_pass{pass_idx}.npz", W,
            )
        prev_delta = delta_df.copy()

        if auto_select_passes and no_improve >= patience:
            logger.info(
                f"RWR auto-stop: eigengap peaked at pass {best_pass} "
                f"(={best_eg:.3f}); stopping after pass {pass_idx}."
            )
            break

    if auto_select_passes and best_imputed is not None:
        all_diagnostics["auto_selected_pass"] = int(best_pass)
        all_diagnostics["auto_selected_eigengap"] = float(best_eg)
        logger.info(f"RWR auto-selected refinement pass {best_pass}.")
        imputed, delta_df = best_imputed, best_delta

    if output_dir is not None:
        with open(ensure_dir(output_dir) / "rwr_diagnostics.json", "w") as fh:
            json.dump(all_diagnostics, fh, indent=2, default=_json_safe)

    return imputed, delta_df, all_diagnostics


def _json_safe(obj):
    """Coerce numpy scalars / dtypes / arrays to JSON-friendly types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Unserializable: {type(obj).__name__}")
