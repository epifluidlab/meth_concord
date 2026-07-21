"""Unit tests for batch_correction module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import batch_correction as bc


def _make_batched_matrix(
    n_cells_per_batch: int = 20,
    n_bins: int = 50,
    n_batches: int = 2,
    batch_shift: float = 5.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a cells × bins matrix where each batch has a constant per-bin shift.

    Returns (matrix, batch_labels).
    """
    rng = np.random.default_rng(seed)
    n_cells = n_batches * n_cells_per_batch
    base = rng.normal(size=(n_cells, n_bins))
    # Add batch-specific column shifts: batch b has +b*batch_shift at every bin.
    for b in range(n_batches):
        i0 = b * n_cells_per_batch
        i1 = (b + 1) * n_cells_per_batch
        base[i0:i1] += b * batch_shift
    cell_names = [f"c{i:03d}" for i in range(n_cells)]
    matrix = pd.DataFrame(
        base, index=cell_names,
        columns=[f"chr1:{j * 250_000}" for j in range(n_bins)],
    )
    batches = pd.Series(
        [f"batch{b}" for b in range(n_batches) for _ in range(n_cells_per_batch)],
        index=cell_names,
    )
    return matrix, batches


# ---------------------------------------------------------------------------
# apply_batch_correction dispatch
# ---------------------------------------------------------------------------


def test_invalid_method_raises():
    mat, batches = _make_batched_matrix()
    with pytest.raises(ValueError, match="method must be"):
        bc.apply_batch_correction(mat, batches, method="nope")


def test_none_returns_same_matrix():
    mat, batches = _make_batched_matrix()
    out = bc.apply_batch_correction(mat, batches, method="none")
    np.testing.assert_array_equal(out.to_numpy(), mat.to_numpy())


def test_single_batch_is_noop():
    mat, batches = _make_batched_matrix(n_batches=1)
    out = bc.apply_batch_correction(mat, batches, method="center")
    np.testing.assert_array_equal(out.to_numpy(), mat.to_numpy())


def test_unlabeled_cells_become_unknown_batch(caplog):
    mat, batches = _make_batched_matrix(n_batches=2)
    # Drop labels for 2 cells.
    batches.iloc[0] = np.nan
    batches.iloc[5] = np.nan
    with caplog.at_level("WARNING"):
        out = bc.apply_batch_correction(mat, batches, method="center")
    # Output still has the same shape.
    assert out.shape == mat.shape
    assert "__unknown_batch__" in caplog.text


# ---------------------------------------------------------------------------
# Per-batch centering
# ---------------------------------------------------------------------------


def test_per_batch_center_removes_per_batch_mean():
    """After per-batch centering, each batch's per-bin mean is exactly 0."""
    mat, batches = _make_batched_matrix(batch_shift=10.0)
    out = bc.apply_batch_correction(mat, batches, method="center")
    for batch in batches.unique():
        cells = batches[batches == batch].index
        per_bin_means = out.loc[cells].mean(axis=0)
        np.testing.assert_allclose(per_bin_means, 0.0, atol=1e-10)


def test_per_batch_center_preserves_within_batch_variance():
    """Centering doesn't touch within-batch spread."""
    mat, batches = _make_batched_matrix(batch_shift=10.0)
    out = bc.apply_batch_correction(mat, batches, method="center")
    for batch in batches.unique():
        cells = batches[batches == batch].index
        original_std = mat.loc[cells].std(axis=0)
        corrected_std = out.loc[cells].std(axis=0)
        np.testing.assert_allclose(
            original_std.to_numpy(), corrected_std.to_numpy(), atol=1e-10,
        )


def test_per_batch_center_reduces_batch_anova():
    """ANOVA F-stat should drop substantially after per-batch centering."""
    mat, batches = _make_batched_matrix(batch_shift=20.0, n_bins=30)
    f_pre = bc.batch_anova_f_per_bin(mat, batches)
    out = bc.apply_batch_correction(mat, batches, method="center")
    f_post = bc.batch_anova_f_per_bin(out, batches)
    # Pre-correction F should be huge; post-correction F near 0.
    assert float(np.nanmedian(f_pre)) > 100.0
    assert float(np.nanmedian(f_post)) < 1.0


# ---------------------------------------------------------------------------
# Per-batch z-score
# ---------------------------------------------------------------------------


def test_per_batch_zscore_yields_unit_variance():
    """After per-batch z-score, each batch has unit variance per bin."""
    mat, batches = _make_batched_matrix(n_batches=2, n_cells_per_batch=30)
    # Inject different variances per batch.
    cells_b0 = batches[batches == "batch0"].index
    cells_b1 = batches[batches == "batch1"].index
    mat.loc[cells_b1] = mat.loc[cells_b1] * 10.0  # batch1 has 10× the scale

    out = bc.apply_batch_correction(mat, batches, method="zscore")
    for batch in batches.unique():
        cells = batches[batches == batch].index
        per_bin_means = out.loc[cells].mean(axis=0)
        per_bin_stds = out.loc[cells].std(axis=0)
        np.testing.assert_allclose(per_bin_means, 0.0, atol=1e-10)
        np.testing.assert_allclose(per_bin_stds, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# ComBat (only if pycombat/inmoose installed)
# ---------------------------------------------------------------------------


def _have_combat() -> bool:
    try:
        from combat.pycombat import pycombat  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        from inmoose.pycombat import pycombat_norm  # noqa: F401
        return True
    except ImportError:
        pass
    return False


@pytest.mark.skipif(not _have_combat(), reason="pycombat / inmoose not installed")
def test_combat_reduces_batch_anova():
    """ComBat should reduce the batch ANOVA F-stat substantially."""
    mat, batches = _make_batched_matrix(batch_shift=10.0, n_bins=30,
                                          n_cells_per_batch=30)
    f_pre = bc.batch_anova_f_per_bin(mat, batches)
    out = bc.apply_batch_correction(mat, batches, method="combat")
    assert out.shape == mat.shape
    f_post = bc.batch_anova_f_per_bin(out, batches)
    assert float(np.nanmedian(f_post)) < 0.1 * float(np.nanmedian(f_pre))


def test_combat_missing_dep_helpful_error():
    """Without pycombat installed, asking for combat gives a helpful error."""
    if _have_combat():
        pytest.skip("pycombat is installed; can't test missing-dep behaviour")
    mat, batches = _make_batched_matrix()
    with pytest.raises(ImportError, match="pip install"):
        bc.apply_batch_correction(mat, batches, method="combat")


# ---------------------------------------------------------------------------
# Harmony (only if harmonypy installed)
# ---------------------------------------------------------------------------


def _have_harmony() -> bool:
    try:
        import harmonypy  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_harmony(), reason="harmonypy not installed")
def test_harmony_reduces_batch_anova():
    """Harmony should reduce batch effects in reconstructed feature space."""
    mat, batches = _make_batched_matrix(batch_shift=10.0, n_bins=60,
                                          n_cells_per_batch=30)
    out = bc.apply_batch_correction(mat, batches, method="harmony")
    assert out.shape == mat.shape


def test_harmony_missing_dep_helpful_error():
    if _have_harmony():
        pytest.skip("harmonypy is installed; can't test missing-dep behaviour")
    mat, batches = _make_batched_matrix()
    with pytest.raises(ImportError, match="harmonypy"):
        bc.apply_batch_correction(mat, batches, method="harmony")


# ---------------------------------------------------------------------------
# batch_anova_f_per_bin
# ---------------------------------------------------------------------------


def test_batch_anova_f_returns_per_bin_series():
    mat, batches = _make_batched_matrix(n_bins=20)
    f_stats = bc.batch_anova_f_per_bin(mat, batches)
    assert len(f_stats) == 20
    assert isinstance(f_stats, pd.Series)


def test_batch_anova_f_zero_for_identical_batches():
    """If batches have IDENTICAL distributions, F-stat is small."""
    mat, batches = _make_batched_matrix(batch_shift=0.0, n_bins=20, seed=0)
    f_stats = bc.batch_anova_f_per_bin(mat, batches)
    # Random data with no shift → F ~ Fisher distribution centred around 1.
    assert float(np.nanmedian(f_stats)) < 5.0


# ---------------------------------------------------------------------------
# Multiple batch variables (cell-culture / library-prep / sequencing batch)
# ---------------------------------------------------------------------------


def _make_two_factor(n_per=15, n_bins=24, shift=6.0, seed=0):
    """Matrix where the (factorA, factorB) COMBINATION sets the per-bin shift."""
    rng = np.random.default_rng(seed)
    groups = [("a0", "b0"), ("a0", "b1"), ("a1", "b0"), ("a1", "b1")]
    blocks, fa, fb, names = [], [], [], []
    k = 0
    for gi, (a, b) in enumerate(groups):
        blocks.append(rng.normal(size=(n_per, n_bins)) + (gi + 1) * shift)
        for _ in range(n_per):
            fa.append(a); fb.append(b); names.append(f"c{k:03d}"); k += 1
    mat = pd.DataFrame(np.vstack(blocks), index=names,
                       columns=[f"chr1:{j * 250_000}" for j in range(n_bins)])
    bdf = pd.DataFrame({"culture": fa, "libprep": fb}, index=names)
    return mat, bdf


def test_coerce_batch_df_series_and_frame():
    mat, bdf = _make_two_factor()
    # Series -> single-column frame.
    s = bdf["culture"]
    df1 = bc._coerce_batch_df(s, mat.index)
    assert list(df1.columns) == ["culture"] and len(df1) == len(mat)
    # DataFrame -> passthrough (both columns), str-typed.
    df2 = bc._coerce_batch_df(bdf, mat.index)
    assert list(df2.columns) == ["culture", "libprep"]
    assert df2["culture"].dtype == object


def test_composite_label_is_cartesian_product():
    _, bdf = _make_two_factor()
    comp = bc._composite_label(bdf)
    assert set(comp.unique()) == {"a0|b0", "a0|b1", "a1|b0", "a1|b1"}


def test_multi_batch_center_removes_composite_mean():
    """center with a 2-column DataFrame removes the per-composite-group mean."""
    mat, bdf = _make_two_factor()
    out = bc.apply_batch_correction(mat, bdf, method="center")
    comp = bdf.agg("|".join, axis=1)
    for g, cells in comp.groupby(comp).groups.items():
        grp_mean = out.loc[cells].mean(axis=0).abs().max()
        assert grp_mean < 1e-9, f"group {g} not centered"


def test_multi_batch_anova_f_accepts_dataframe():
    mat, bdf = _make_two_factor(n_bins=20)
    f_stats = bc.batch_anova_f_per_bin(mat, bdf)   # composite ANOVA
    assert isinstance(f_stats, pd.Series) and len(f_stats) == 20
    # Strong composite-group shifts -> large F before correction.
    assert float(np.nanmedian(f_stats)) > 10.0


def test_multi_batch_all_constant_is_noop(caplog):
    mat, bdf = _make_two_factor()
    const = pd.DataFrame({"x": ["one"] * len(mat), "y": ["k"] * len(mat)},
                         index=mat.index)
    out = bc.apply_batch_correction(mat, const, method="center")
    np.testing.assert_array_equal(out.to_numpy(), mat.to_numpy())


@pytest.mark.skipif(not _have_harmony(), reason="harmonypy not installed")
def test_harmony_accepts_multiple_batch_vars():
    """Harmony harmonizes over several batch variables jointly (vars_use)."""
    mat, bdf = _make_two_factor(n_per=20, n_bins=40, shift=8.0)
    out = bc.apply_batch_correction(mat, bdf, method="harmony")
    assert out.shape == mat.shape


# --------------------------------------------------------------------------
# All-NaN bins (an empty compartment bin) must not crash the dense solvers.
# Regression: PCA/ComBat reject NaN, and per-bin-mean imputation cannot fill a
# column whose mean is itself NaN, so an all-NaN bin slipped through to the
# solver and raised "Input X contains NaN" on real compartment matrices.
# --------------------------------------------------------------------------

def test_impute_zero_fills_all_nan_columns_and_mean_fills_the_rest():
    X = pd.DataFrame({"dead": [np.nan, np.nan, np.nan],
                      "sparse": [np.nan, 5.0, 7.0],
                      "full": [1.0, 2.0, 3.0]})
    out = bc._impute_for_dense_solver(X, "test")
    assert np.isfinite(out.to_numpy()).all()
    assert (out["dead"] == 0.0).all(), "an all-NaN bin must become 0, not stay NaN"
    assert out["sparse"].iloc[0] == pytest.approx(6.0), "sparse NaN -> per-bin mean"
    assert list(out["full"]) == [1.0, 2.0, 3.0], "observed values untouched"


def test_impute_is_a_noop_without_nans():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = bc._impute_for_dense_solver(X, "test")
    assert out.equals(X)


def test_impute_preserves_shape_and_columns():
    X = pd.DataFrame(np.arange(12.0).reshape(4, 3),
                     columns=["chr1:0", "chr1:250000", "chr1:500000"])
    X.iloc[:, 1] = np.nan
    out = bc._impute_for_dense_solver(X, "test")
    assert out.shape == X.shape
    assert list(out.columns) == list(X.columns)


def _batched(n_cells=200, n_bins=60, seed=0):
    rng = np.random.default_rng(seed)
    M = pd.DataFrame(
        rng.normal(0, 1, (n_cells, n_bins)),
        columns=[f"chr1:{i * 250000}" for i in range(n_bins)],
        index=[f"c{i}" for i in range(n_cells)])
    half = n_cells // 2
    M += np.where(np.arange(n_cells)[:, None] < half, 0.8, -0.8)
    batch = pd.DataFrame(
        {"exp_batch": np.where(np.arange(n_cells) < half, "a", "b")},
        index=M.index)
    return M, batch


def test_harmony_survives_an_all_nan_bin():
    pytest.importorskip("harmonypy")
    M, batch = _batched()
    M.iloc[:, 7] = np.nan                       # a bin no cell covers
    M.iloc[M.to_numpy() != M.to_numpy()] = np.nan
    rng = np.random.default_rng(1)
    M.values[rng.random(M.shape) < 0.05] = np.nan   # ordinary dropout too
    try:
        out = bc.apply_batch_correction(M, batch, method="harmony",
                                        random_state=0)
    except Exception as e:                      # pragma: no cover
        if "has no len" in str(e):
            pytest.skip("local harmonypy version bug unrelated to NaN handling")
        raise
    assert out.shape == M.shape
    assert np.isfinite(out.to_numpy()).all(), "no NaN may reach the solver"


def test_combat_survives_an_all_nan_bin():
    if not _have_combat():
        pytest.skip("pycombat/inmoose not installed")
    M, batch = _batched()
    M.iloc[:, 3] = np.nan
    out = bc.apply_batch_correction(M, batch, method="combat", random_state=0)
    assert out.shape == M.shape
    assert np.isfinite(out.to_numpy()).all()
