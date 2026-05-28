"""Unit tests for matrix_ops.process_concordance_matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import matrix_ops


def _make_matrix(values):
    cells = [f"cell_{i}" for i in range(len(values))]
    bins = [f"chr1:{i * 5_000_000}" for i in range(len(values[0]))]
    return pd.DataFrame(values, index=cells, columns=bins)


def test_bin_filter_exact_threshold(tmp_path):
    # 5 cells, 3 bins; bin 0 has 4/5 non-NA, bin 1 has 3/5, bin 2 has 5/5
    mat = _make_matrix([
        [0.1, 0.2, 0.3],
        [0.2, np.nan, 0.4],
        [0.3, 0.4, 0.5],
        [0.4, np.nan, 0.6],
        [np.nan, 0.5, 0.7],
    ])
    # coverage_frac = [0.8, 0.6, 1.0]
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.8, output_dir=str(tmp_path),
    )
    assert result["retained_bins"] == ["chr1:0", "chr1:10000000"]


def test_imputation_fills_with_zero(tmp_path):
    mat = _make_matrix([
        [0.1, np.nan, 0.3],
        [0.2, 0.3, 0.4],
        [np.nan, 0.4, 0.5],
        [0.4, 0.5, 0.6],
        [0.5, 0.6, 0.7],
    ])
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path),
    )
    assert not result["imputed"].isna().any().any()
    # The entries that were NaN must now be 0.
    assert result["imputed"].iloc[0, 1] == 0.0
    assert result["imputed"].iloc[2, 0] == 0.0


def test_zscore_has_zero_mean_unit_std(tmp_path):
    rng = np.random.default_rng(0)
    mat = _make_matrix(rng.normal(0, 0.3, size=(30, 15)).tolist())
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path),
    )
    z = result["zscore"]
    means = z.mean(axis=0).to_numpy()
    stds = z.std(axis=0, ddof=1).to_numpy()
    np.testing.assert_allclose(means, 0.0, atol=1e-10)
    # After imputation+zscore each column has sample std 1 when the
    # underlying column is non-constant.
    np.testing.assert_allclose(stds, 1.0, atol=1e-6)


def test_raises_on_empty(tmp_path):
    empty = pd.DataFrame()
    with pytest.raises(ValueError):
        matrix_ops.process_concordance_matrix(empty, output_dir=str(tmp_path))


def test_per_cell_centering_removes_per_cell_shift(tmp_path):
    """With per_cell_center=True, each cell's row sums to ~0 after centering."""
    # 4 cells × 3 bins with HUGE per-cell shifts: cell_0 +10, cell_1 +20, etc.
    mat = _make_matrix([
        [10.0, 11.0, 12.0],   # cell_0: mean = 11
        [20.0, 21.0, 22.0],   # cell_1: mean = 21
        [30.0, 31.0, 32.0],   # cell_2: mean = 31
        [40.0, 41.0, 42.0],   # cell_3: mean = 41
    ])
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path),
        per_cell_center=True,
    )
    # imputed matrix is post-centering, so each row should sum to 0.
    row_sums = result["imputed"].sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)
    # And the column structure (each row goes [−1, 0, +1] after centering) is preserved.
    expected = np.array([
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(result["imputed"].to_numpy(), expected, atol=1e-10)


def test_per_cell_centering_off_preserves_raw_values(tmp_path):
    """With per_cell_center=False (default), values are not shifted."""
    mat = _make_matrix([
        [10.0, 11.0, 12.0],
        [20.0, 21.0, 22.0],
    ])
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path),
        per_cell_center=False,
    )
    np.testing.assert_allclose(result["imputed"].to_numpy(), mat.to_numpy())


def test_per_cell_zscore_removes_mean_and_scale(tmp_path):
    """per_cell_norm='zscore' should give each row mean=0 and std=1."""
    # 3 cells, each with a different mean AND different scale.
    mat = _make_matrix([
        [10.0, 11.0, 12.0, 13.0, 14.0],   # mean=12, std=√2.5
        [100.0, 110.0, 120.0, 130.0, 140.0],  # mean=120, std=10*√2.5
        [-5.0, -2.5, 0.0, 2.5, 5.0],      # mean=0, std=√6.25
    ])
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path),
        per_cell_norm="zscore",
    )
    z = result["imputed"]
    means = z.mean(axis=1).to_numpy()
    stds = z.std(axis=1).to_numpy()
    np.testing.assert_allclose(means, 0.0, atol=1e-10)
    np.testing.assert_allclose(stds, 1.0, atol=1e-10)


def test_per_cell_rank_normalisation(tmp_path):
    """per_cell_norm='rank' replaces each row with rank/(n+1) - 0.5."""
    mat = _make_matrix([
        [10.0, 20.0, 30.0, 40.0, 50.0],  # already sorted ascending
        [50.0, 40.0, 30.0, 20.0, 10.0],  # descending
    ])
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path),
        per_cell_norm="rank",
    )
    # For row 0 (ascending), ranks should be [1, 2, 3, 4, 5] / 6 - 0.5
    expected_ascending = np.array([1, 2, 3, 4, 5]) / 6 - 0.5
    np.testing.assert_allclose(
        result["imputed"].iloc[0].to_numpy(), expected_ascending,
    )
    # Row 1 (descending), ranks should be [5, 4, 3, 2, 1] / 6 - 0.5.
    expected_descending = np.array([5, 4, 3, 2, 1]) / 6 - 0.5
    np.testing.assert_allclose(
        result["imputed"].iloc[1].to_numpy(), expected_descending,
    )


def test_per_cell_norm_invalid_raises(tmp_path):
    mat = _make_matrix([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="per_cell_norm"):
        matrix_ops.process_concordance_matrix(
            mat, output_dir=str(tmp_path), per_cell_norm="nope",
        )


def test_post_norm_recenter_removes_residual_per_cell_shift(tmp_path):
    """post_norm_recenter=True should make per-row means in the normalised
    matrix exactly 0, even after per-cell rank + per-bin QN."""
    rng = np.random.default_rng(0)
    # Cells × bins matrix with per-cell shifts that won't fully wash out
    # under per-cell rank + per-bin QN alone.
    raw = rng.normal(size=(40, 30))
    # Inject a strong per-cell offset.
    raw += rng.normal(scale=5.0, size=(40, 1))
    mat = _make_matrix(raw.tolist())

    # Without post-norm recenter: per-row means in the normalised matrix
    # are NOT exactly 0 (this is the bug we're fixing).
    result_no_recenter = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path / "a"),
        matrix_norm="quantile", per_cell_norm="rank",
        post_norm_recenter=False,
    )
    no_recenter_row_means = result_no_recenter["zscore"].mean(axis=1).to_numpy()
    # The biggest absolute row mean should be nonzero — that's the artifact.
    assert np.max(np.abs(no_recenter_row_means)) > 1e-3

    # With post-norm recenter: per-row means are exactly 0.
    result_recenter = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path / "b"),
        matrix_norm="quantile", per_cell_norm="rank",
        post_norm_recenter=True,
    )
    recenter_row_means = result_recenter["zscore"].mean(axis=1).to_numpy()
    np.testing.assert_allclose(recenter_row_means, 0.0, atol=1e-10)


def test_post_norm_recenter_preserves_per_bin_centering_approximately(tmp_path):
    """After post_norm_recenter, per-bin (column) means should still be
    near 0 — since col means were already ≈ 0 after per-bin QN, and we
    only subtracted small row-mean shifts."""
    rng = np.random.default_rng(1)
    mat = _make_matrix(rng.normal(size=(40, 30)).tolist())
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path),
        matrix_norm="quantile", per_cell_norm="rank",
        post_norm_recenter=True,
    )
    col_means = result["zscore"].mean(axis=0).to_numpy()
    # Per-bin means should still be very near 0 (small perturbation only).
    assert float(np.max(np.abs(col_means))) < 0.05


def test_post_norm_recenter_off_by_default(tmp_path):
    """post_norm_recenter defaults to off for backward compatibility."""
    mat = _make_matrix([[1.0, 2.0], [3.0, 4.0]])
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path),
        matrix_norm="quantile",  # no per_cell_norm, no post_norm_recenter
    )
    # Just verify the call works — no per-row mean adjustment was applied.
    assert "imputed" in result


def test_per_cell_center_legacy_alias(tmp_path):
    """Legacy per_cell_center=True is equivalent to per_cell_norm='center'."""
    mat = _make_matrix([
        [10.0, 12.0],
        [20.0, 22.0],
    ])
    via_legacy = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path / "a"),
        per_cell_center=True,
    )
    via_new = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path / "b"),
        per_cell_norm="center",
    )
    np.testing.assert_allclose(
        via_legacy["imputed"].to_numpy(),
        via_new["imputed"].to_numpy(),
    )


def test_zero_variance_column_zscore_is_zero(tmp_path):
    mat = _make_matrix([
        [0.5, 0.1],
        [0.5, 0.2],
        [0.5, 0.3],
        [0.5, 0.4],
    ])
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path),
    )
    # First bin has zero variance -> z-score stays at 0.
    z = result["zscore"]
    np.testing.assert_array_equal(z["chr1:0"].to_numpy(), np.zeros(4))


