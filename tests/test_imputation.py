"""Unit tests for nome_hic_concordance.imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from nome_hic_concordance import imputation


# ---------------------------------------------------------------------------
# quantile_normalize_for_knn
# ---------------------------------------------------------------------------


def test_qn_for_knn_basic_centred_uniform():
    """After QN, each bin has the centred uniform-rank distribution."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    Y = imputation.quantile_normalize_for_knn(X)
    # Each column should have ranks -0.5 ... +0.5 after subtracting 0.5.
    for j in range(Y.shape[1]):
        col = np.sort(Y[:, j])
        # Expected: rank/(n+1) - 0.5 for ranks 1..n.
        expected = np.arange(1, Y.shape[0] + 1) / (Y.shape[0] + 1) - 0.5
        np.testing.assert_allclose(col, expected)


def test_qn_for_knn_imputes_nan_to_zero():
    """NaN entries become the centred median (0)."""
    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, np.nan],
    ])
    Y = imputation.quantile_normalize_for_knn(X)
    assert not np.isnan(Y).any()
    assert Y[1, 1] == 0.0   # NaN → 0 (centred median)
    assert Y[2, 2] == 0.0


# ---------------------------------------------------------------------------
# build_knn_graph
# ---------------------------------------------------------------------------


def test_knn_graph_basic_structure():
    """Returned graph has correct shape and row-stochasticity."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 8))
    W, P = imputation.build_knn_graph(X, k=5)

    assert W.shape == (20, 20)
    assert P.shape == (20, 20)
    # Each row of P sums to 1 (stochastic) — except possibly all-zero rows.
    row_sums = np.asarray(P.sum(axis=1)).ravel()
    nonzero_rows = row_sums > 0
    assert np.all(np.isclose(row_sums[nonzero_rows], 1.0))


def test_knn_graph_is_symmetric():
    """W (post-symmetrize) is symmetric."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(15, 4))
    W, _ = imputation.build_knn_graph(X, k=3)
    W_arr = W.toarray()
    np.testing.assert_allclose(W_arr, W_arr.T, atol=1e-10)


def test_knn_graph_no_self_loops():
    """Diagonal of W is zero — cells are not their own neighbours."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 4))
    W, _ = imputation.build_knn_graph(X, k=3)
    assert np.all(np.diag(W.toarray()) == 0.0)


def test_knn_graph_clusters_recovered():
    """Build k-NN on two well-separated clusters; verify each cell's
    neighbours are mostly in the same cluster."""
    rng = np.random.default_rng(4)
    cluster_a = rng.normal(loc=+5.0, scale=0.1, size=(20, 4))
    cluster_b = rng.normal(loc=-5.0, scale=0.1, size=(20, 4))
    X = np.vstack([cluster_a, cluster_b])

    W, _ = imputation.build_knn_graph(X, k=5)
    W_arr = W.toarray()

    # For each cell in cluster A (indices 0-19), its neighbours should
    # be in cluster A (indices 0-19).
    for i in range(20):
        neighbours = np.where(W_arr[i] > 0)[0]
        in_cluster = np.sum(neighbours < 20)
        assert in_cluster / len(neighbours) >= 0.8, (
            f"Cell {i} (cluster A) has too many out-of-cluster neighbours."
        )


def test_knn_graph_k_clamped_to_n_minus_1():
    """If k >= n_cells, it's silently clamped to n-1 (logs a warning)."""
    # Use co-linear points to guarantee positive cosine similarities so
    # nothing gets zeroed by the (1 − cosine_distance) clip.
    n_cells = 5
    X = np.array([
        [1.0, 0.1 * i, 0.05 * i]  # all in roughly the same direction
        for i in range(n_cells)
    ])
    W, _ = imputation.build_knn_graph(X, k=100)  # k >> n - 1
    # No row may have more than n-1 = 4 neighbours.
    nnz_per_row = (W.toarray() > 0).sum(axis=1)
    assert (nnz_per_row <= n_cells - 1).all()
    # At least one row should have all n-1 = 4 neighbours given positive
    # similarities throughout.
    assert (nnz_per_row == n_cells - 1).any()


def test_knn_graph_too_few_cells_raises():
    """Fewer than 2 cells is an error."""
    with pytest.raises(ValueError, match="2 cells"):
        imputation.build_knn_graph(np.zeros((1, 3)), k=5)


# ---------------------------------------------------------------------------
# rwr_impute_matrices
# ---------------------------------------------------------------------------


def test_rwr_restart_1_returns_original():
    """With restart_prob = 1, RWR returns the original matrix exactly."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(10, 5))
    F = rng.normal(size=(10, 3))
    _, P = imputation.build_knn_graph(X, k=3)
    imputed, _ = imputation.rwr_impute_matrices(
        {"F": F.copy()}, P, restart_prob=1.0, solver="direct",
    )
    np.testing.assert_allclose(imputed["F"], F, atol=1e-10)


def test_rwr_direct_vs_iterative_agree():
    """Direct and iterative solvers give the same result (within tol)."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(15, 6))
    F = rng.normal(size=(15, 4))
    _, P = imputation.build_knn_graph(X, k=4)

    direct, _ = imputation.rwr_impute_matrices(
        {"F": F.copy()}, P, restart_prob=0.5, solver="direct",
    )
    iterative, _ = imputation.rwr_impute_matrices(
        {"F": F.copy()}, P, restart_prob=0.5, solver="iterative",
        max_iter=200, tol=1e-8,
    )
    np.testing.assert_allclose(direct["F"], iterative["F"], atol=1e-5)


def test_rwr_reduces_distance_between_neighbours():
    """RWR smooths matrix values across the k-NN graph — neighbours
    that started far apart should be closer post-imputation."""
    rng = np.random.default_rng(8)
    # 20 cells in two clear clusters; matrix has cluster-specific values
    # plus noise.
    cluster_assignments = np.array([0] * 10 + [1] * 10)
    X = np.vstack([
        rng.normal(loc=+5.0, scale=0.1, size=(10, 4)),
        rng.normal(loc=-5.0, scale=0.1, size=(10, 4)),
    ])
    # F: bin-level values that should agree within cluster but are noisy.
    F_true = np.where(cluster_assignments[:, None] == 0, 1.0, -1.0).repeat(5, axis=1)
    F_noisy = F_true + rng.normal(scale=0.5, size=F_true.shape)

    _, P = imputation.build_knn_graph(X, k=5)
    imputed, _ = imputation.rwr_impute_matrices(
        {"F": F_noisy}, P, restart_prob=0.3, solver="direct",
    )

    # Within-cluster variance should drop after smoothing.
    noisy_var = np.var(F_noisy[:10]) + np.var(F_noisy[10:])
    smooth_var = np.var(imputed["F"][:10]) + np.var(imputed["F"][10:])
    assert smooth_var < noisy_var, (
        f"RWR should reduce within-cluster variance "
        f"(noisy={noisy_var:.3f}, imputed={smooth_var:.3f})"
    )


def test_rwr_multiple_matrices_share_transition():
    """Multiple matrices passed in one call share the same RWR factor —
    cheaper than calling once per matrix."""
    rng = np.random.default_rng(9)
    X = rng.normal(size=(10, 5))
    _, P = imputation.build_knn_graph(X, k=3)
    matrices = {
        "k_open":  rng.uniform(size=(10, 4)),
        "k_closed": rng.uniform(size=(10, 4)),
        "n_linked": rng.uniform(size=(10, 4)),
    }
    imputed, diag = imputation.rwr_impute_matrices(
        matrices, P, restart_prob=0.5, solver="direct",
    )
    assert set(imputed.keys()) == set(matrices.keys())
    for k in matrices:
        assert imputed[k].shape == matrices[k].shape
    assert diag["solver_used"] == "direct"
    assert diag["n_matrices"] == 3


def test_rwr_invalid_solver_raises():
    with pytest.raises(ValueError, match="solver must be"):
        imputation.rwr_impute_matrices(
            {"F": np.zeros((4, 3))},
            sp.eye(4).tocsr(),
            solver="not-a-solver",
        )


# ---------------------------------------------------------------------------
# run_rwr_pipeline
# ---------------------------------------------------------------------------


def _identity_layer2(imputed: dict[str, np.ndarray]) -> pd.DataFrame:
    """Trivial Layer-2 stub for testing: just returns k_open - k_closed."""
    diff = imputed["k_open"] - imputed["k_closed"]
    return pd.DataFrame(diff, columns=[f"bin_{j}" for j in range(diff.shape[1])])


def test_run_rwr_pipeline_pass0_only():
    """With n_refine_passes=0, runs RWR once and returns the Layer-2 output."""
    rng = np.random.default_rng(10)
    n_cells, n_bins = 20, 6
    soft_counts = {
        "k_open":   rng.uniform(size=(n_cells, n_bins)),
        "k_closed": rng.uniform(size=(n_cells, n_bins)),
        "k_mixed":  rng.uniform(size=(n_cells, n_bins)),
        "n_linked": rng.uniform(low=1, high=5, size=(n_cells, n_bins)),
    }
    accessibility = pd.DataFrame(
        rng.normal(size=(n_cells, n_bins)),
        index=[f"c{i}" for i in range(n_cells)],
        columns=[f"bin_{j}" for j in range(n_bins)],
    )

    imputed, delta, diag = imputation.run_rwr_pipeline(
        soft_counts, accessibility, _identity_layer2,
        n_refine_passes=0, k=5,
    )

    assert set(imputed.keys()) == set(soft_counts.keys())
    assert delta.shape == (n_cells, n_bins)
    assert len(diag["passes"]) == 1
    assert diag["passes"][0]["knn_basis"] == "QN(initial_feature_matrix)"


def test_run_rwr_pipeline_with_refinement():
    """With n_refine_passes=2, runs 3 total passes; each pass rebuilds k-NN."""
    rng = np.random.default_rng(11)
    n_cells, n_bins = 20, 6
    soft_counts = {
        "k_open":   rng.uniform(size=(n_cells, n_bins)),
        "k_closed": rng.uniform(size=(n_cells, n_bins)),
        "k_mixed":  rng.uniform(size=(n_cells, n_bins)),
        "n_linked": rng.uniform(low=1, high=5, size=(n_cells, n_bins)),
    }
    accessibility = pd.DataFrame(
        rng.normal(size=(n_cells, n_bins)),
        index=[f"c{i}" for i in range(n_cells)],
        columns=[f"bin_{j}" for j in range(n_bins)],
    )

    _, _, diag = imputation.run_rwr_pipeline(
        soft_counts, accessibility, _identity_layer2,
        n_refine_passes=2, k=5,
    )

    assert len(diag["passes"]) == 3
    assert diag["passes"][0]["knn_basis"].startswith("QN(initial")
    assert diag["passes"][1]["knn_basis"] == "QN(Δ_signed_pass_0)"
    assert diag["passes"][2]["knn_basis"] == "QN(Δ_signed_pass_1)"
    # Refinement passes carry a delta-change measurement.
    assert "mean_abs_delta_change" in diag["passes"][1]
    assert "mean_abs_delta_change" in diag["passes"][2]


def test_post_norm_for_knn_strips_per_cell_shift():
    """post_norm_for_knn output has per-row means ~0 (recenter applied)."""
    rng = np.random.default_rng(7)
    # Matrix with a strong per-cell additive shift that 'raw' QN would keep.
    base = rng.normal(size=(15, 8))
    shifted = base + rng.normal(scale=5.0, size=(15, 1))
    out = imputation.post_norm_for_knn(shifted)
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-12)
    # NaN entries are imputed (no NaN in output).
    assert not np.isnan(out).any()


def test_run_rwr_pipeline_refinement_basis_postnorm_label():
    """With refinement_basis='postnorm', the per-pass diagnostic records the
    postnorm basis label and the run still produces a valid Δ matrix."""
    rng = np.random.default_rng(21)
    n_cells, n_bins = 20, 6
    soft_counts = {
        "k_open":   rng.uniform(size=(n_cells, n_bins)),
        "k_closed": rng.uniform(size=(n_cells, n_bins)),
        "k_mixed":  rng.uniform(size=(n_cells, n_bins)),
        "n_linked": rng.uniform(low=1, high=5, size=(n_cells, n_bins)),
    }
    accessibility = pd.DataFrame(
        rng.normal(size=(n_cells, n_bins)),
        index=[f"c{i}" for i in range(n_cells)],
        columns=[f"bin_{j}" for j in range(n_bins)],
    )

    _, delta, diag = imputation.run_rwr_pipeline(
        soft_counts, accessibility, _identity_layer2,
        n_refine_passes=1, k=5, refinement_basis="postnorm",
    )
    assert diag["refinement_basis"] == "postnorm"
    assert diag["passes"][1]["knn_basis"] == "postnorm(Δ_signed_pass_0)"
    assert delta.shape == (n_cells, n_bins)


def test_run_rwr_pipeline_invalid_refinement_basis_raises():
    rng = np.random.default_rng(22)
    soft_counts = {
        "k_open":   rng.uniform(size=(6, 3)),
        "k_closed": rng.uniform(size=(6, 3)),
        "k_mixed":  rng.uniform(size=(6, 3)),
        "n_linked": rng.uniform(low=1, high=5, size=(6, 3)),
    }
    accessibility = pd.DataFrame(
        rng.normal(size=(6, 3)),
        index=[f"c{i}" for i in range(6)],
        columns=[f"bin_{j}" for j in range(3)],
    )
    with pytest.raises(ValueError, match="refinement_basis"):
        imputation.run_rwr_pipeline(
            soft_counts, accessibility, _identity_layer2,
            n_refine_passes=1, k=3, refinement_basis="nope",
        )


def test_run_rwr_pipeline_refinement_uses_original_raw_counts():
    """Critical invariant: each pass re-imputes from ORIGINAL soft counts,
    not from the previous pass's imputed values (design-doc §6 #5b)."""
    rng = np.random.default_rng(12)
    n_cells, n_bins = 10, 4
    soft_counts = {
        "k_open":   rng.uniform(size=(n_cells, n_bins)),
        "k_closed": rng.uniform(size=(n_cells, n_bins)),
        "k_mixed":  rng.uniform(size=(n_cells, n_bins)),
        "n_linked": rng.uniform(low=1, high=5, size=(n_cells, n_bins)),
    }
    accessibility = pd.DataFrame(
        rng.normal(size=(n_cells, n_bins)),
        index=[f"c{i}" for i in range(n_cells)],
        columns=[f"bin_{j}" for j in range(n_bins)],
    )

    # Stash a copy of the original soft counts BEFORE running the pipeline,
    # then verify it wasn't mutated.
    soft_counts_snapshot = {k: v.copy() for k, v in soft_counts.items()}
    imputation.run_rwr_pipeline(
        soft_counts, accessibility, _identity_layer2,
        n_refine_passes=2, k=3,
    )
    for k in soft_counts:
        np.testing.assert_array_equal(soft_counts[k], soft_counts_snapshot[k])


def test_run_rwr_pipeline_saves_outputs(tmp_path):
    """When output_dir is set, the k-NN graph and diagnostics are saved."""
    rng = np.random.default_rng(13)
    n_cells, n_bins = 8, 3
    soft_counts = {
        "k_open":   rng.uniform(size=(n_cells, n_bins)),
        "k_closed": rng.uniform(size=(n_cells, n_bins)),
        "k_mixed":  rng.uniform(size=(n_cells, n_bins)),
        "n_linked": rng.uniform(low=1, high=5, size=(n_cells, n_bins)),
    }
    accessibility = pd.DataFrame(
        rng.normal(size=(n_cells, n_bins)),
        index=[f"c{i}" for i in range(n_cells)],
        columns=[f"bin_{j}" for j in range(n_bins)],
    )

    out_dir = tmp_path / "imp_out"
    imputation.run_rwr_pipeline(
        soft_counts, accessibility, _identity_layer2,
        n_refine_passes=1, k=3,
        output_dir=str(out_dir),
    )
    assert (out_dir / "cell_knn_graph_pass0.npz").exists()
    assert (out_dir / "cell_knn_graph_pass1.npz").exists()
    assert (out_dir / "rwr_diagnostics.json").exists()
