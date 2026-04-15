"""Unit tests for shrinkage.apply_shrinkage and estimate_shrinkage_n0."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nome_hic_concordance import shrinkage


def _make_matrix(values: np.ndarray, counts: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = [f"cell_{i}" for i in range(values.shape[0])]
    bins = [f"chr1:{i}" for i in range(values.shape[1])]
    return (
        pd.DataFrame(values, index=cells, columns=bins),
        pd.DataFrame(counts, index=cells, columns=bins),
    )


def test_apply_shrinkage_to_zero_when_n0_large():
    values = np.array([[0.6, 0.8], [-0.3, 0.5]])
    counts = np.array([[5, 10], [3, 2]])
    raw, cnt = _make_matrix(values, counts)

    shrunk = shrinkage.apply_shrinkage(raw, cnt, n0=10_000.0)
    # Extreme shrinkage brings values close to zero.
    assert np.all(np.abs(shrunk.to_numpy()) < 0.01)


def test_apply_shrinkage_identity_when_n0_small():
    values = np.array([[0.6, 0.8], [-0.3, 0.5]])
    counts = np.array([[5000, 10000], [3000, 2000]])
    raw, cnt = _make_matrix(values, counts)

    shrunk = shrinkage.apply_shrinkage(raw, cnt, n0=1.0)
    # Tiny pseudocount relative to coverage -> nearly identity.
    np.testing.assert_allclose(shrunk.to_numpy(), values, atol=1e-3)


def test_apply_shrinkage_preserves_nan():
    values = np.array([[0.6, np.nan], [-0.3, 0.5]])
    counts = np.array([[5, 10], [3, 2]])
    raw, cnt = _make_matrix(values, counts)

    shrunk = shrinkage.apply_shrinkage(raw, cnt, n0=30.0)
    assert np.isnan(shrunk.iloc[0, 1])
    assert not np.isnan(shrunk.iloc[0, 0])


def test_apply_shrinkage_never_increases_magnitude():
    """Shrinkage with non-negative n0 should only pull magnitude toward zero."""
    rng = np.random.default_rng(123)
    values = rng.normal(0, 0.3, size=(20, 50))
    counts = rng.integers(1, 100, size=(20, 50))
    raw, cnt = _make_matrix(values, counts)

    shrunk = shrinkage.apply_shrinkage(raw, cnt, n0=25.0)
    # |r_shrunk| <= |r_raw| entry-wise
    assert np.all(np.abs(shrunk.to_numpy()) <= np.abs(raw.to_numpy()) + 1e-12)


def test_estimate_shrinkage_n0_fallback(tmp_path):
    """With only a handful of valid entries, should fall back to default n0=30."""
    values = np.array([[0.1, np.nan], [0.2, 0.3]])
    counts = np.array([[5, 0], [5, 5]])
    raw, cnt = _make_matrix(values, counts)

    result = shrinkage.estimate_shrinkage_n0(
        raw, cnt,
        n_min_for_estimation=50,
        output_dir=str(tmp_path),
    )
    assert result["n0"] == 30.0


def test_estimate_shrinkage_n0_from_high_coverage(tmp_path):
    """With many high-coverage entries, the estimator should produce a finite n0."""
    rng = np.random.default_rng(7)
    # Simulate r = r_star + noise/sqrt(n), with r_star ~ N(0, 0.2)
    n_cells, n_bins = 50, 200
    n_pairs = rng.integers(80, 150, size=(n_cells, n_bins))
    r_star = rng.normal(0, np.sqrt(0.2), size=(n_cells, n_bins))
    noise = rng.normal(0, 1.0 / np.sqrt(np.maximum(n_pairs - 3, 1)))
    r_obs = np.clip(r_star + noise, -1.0, 1.0)

    raw, cnt = _make_matrix(r_obs, n_pairs)
    result = shrinkage.estimate_shrinkage_n0(
        raw, cnt,
        n_min_for_estimation=50,
        output_dir=str(tmp_path),
    )

    assert 0 < result["sigma2_prior"]
    # True sigma^2_prior = 0.2 -> n0 ~= 5. Loose tolerance because of finite sample.
    assert 2 < result["n0"] < 30
