"""Tests for the shared coordination-evaluation helpers (confound ladder etc.)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import coordination_eval as ce


# --------------------------------------------------------------------------
# standardize / zcols / svd_reduce
# --------------------------------------------------------------------------

def test_standardize_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    M = pd.DataFrame(rng.normal(5, 3, size=(40, 6)))
    X = ce.standardize(M)
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(X.std(axis=0), 1.0, atol=1e-9)


def test_standardize_imputes_nan_to_column_mean():
    M = pd.DataFrame({"a": [1.0, 3.0, np.nan], "b": [0.0, 0.0, 0.0]})
    X = ce.standardize(M)
    assert np.isfinite(X).all()
    # constant column 'b' has std 0 -> becomes all zeros (no influence)
    assert np.allclose(X[:, 1], 0.0)


def test_zcols_constant_column_is_zero():
    X = np.array([[1.0, 7.0], [1.0, 9.0], [1.0, 11.0]])
    Z = ce.zcols(X)
    assert np.allclose(Z[:, 0], 0.0)
    assert not np.allclose(Z[:, 1], 0.0)


def test_svd_reduce_shapes():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 100))
    assert ce.svd_reduce(X, 0).shape == (30, 100)          # disabled
    assert ce.svd_reduce(X, 10).shape == (30, 10)          # reduced
    small = rng.normal(size=(30, 5))
    assert ce.svd_reduce(small, 10).shape == (30, 5)       # fewer feats than k


def test_prep_only_svd_for_svd_lr():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 80))
    assert ce.prep(X, "lr", 10).shape == (30, 80)
    assert ce.prep(X, "rf", 10).shape == (30, 80)
    assert ce.prep(X, "svd-lr", 10).shape == (30, 10)


# --------------------------------------------------------------------------
# residualization — the confound ladder
# --------------------------------------------------------------------------

def test_residualize_removes_linear_marginal():
    rng = np.random.default_rng(3)
    M = rng.normal(size=(50, 4))
    # Δ is a pure linear function of the marginal in each column.
    D = 2.0 * M + 1.0
    R = ce.residualize(D, M)
    # residual must be ~uncorrelated with the marginal it was regressed on
    Rz, Mz = ce.zcols(R), ce.zcols(M)
    corr = (Rz * Mz).mean(axis=0)
    assert np.allclose(corr, 0.0, atol=1e-9)
    assert np.allclose(R, 0.0, atol=1e-9)  # nothing left after removing M


def test_residualize_constant_marginal_passes_through():
    rng = np.random.default_rng(4)
    D = rng.normal(size=(40, 3))
    M = np.ones((40, 3))            # constant marginal -> b = 0
    R = ce.residualize(D, M)
    assert np.allclose(R, ce.zcols(D))


def test_residualize_keeps_orthogonal_signal():
    rng = np.random.default_rng(5)
    M = rng.normal(size=(60, 5))
    signal = rng.normal(size=(60, 5))
    # remove the part of signal correlated with M so it is exactly orthogonal
    Mz = ce.zcols(M)
    b = (ce.zcols(signal) * Mz).mean(axis=0)
    ortho = ce.zcols(signal) - b * Mz
    D = ortho + 3.0 * Mz           # orthogonal signal + strong marginal
    R = ce.residualize(D, M)
    # the orthogonal signal survives (correlation with ortho stays high)
    corr = (ce.zcols(R) * ce.zcols(ortho)).mean(axis=0)
    assert (corr > 0.9).all()


def test_residualize_multi_orthogonal_to_all_covariates():
    rng = np.random.default_rng(6)
    M = rng.normal(size=(70, 4))
    Cov = rng.normal(size=(70, 4))
    D = 1.5 * M - 0.7 * (M * M) + 0.3 * Cov + rng.normal(scale=0.01, size=(70, 4))
    R = ce.residualize_multi(D, [M, M * M, Cov])
    for C in (M, M * M, Cov):
        corr = (ce.zcols(R) * ce.zcols(C)).mean(axis=0)
        assert np.allclose(corr, 0.0, atol=1e-2)


def test_residualize_against_removes_batch_projection():
    rng = np.random.default_rng(7)
    R = rng.normal(size=(40, 6))
    # batch one-hot: two batches of 20 cells
    B = np.zeros((40, 1)); B[:20, 0] = 1.0
    # inject a per-batch offset
    Rb = R.copy(); Rb[:20] += 5.0
    out = ce.residualize_against(Rb, B)
    # after removing the batch projection, the two halves have equal column means
    assert np.allclose(out[:20].mean(axis=0), out[20:].mean(axis=0), atol=1e-9)


def test_residualize_against_empty_design_is_identity():
    R = np.arange(12.0).reshape(4, 3)
    out = ce.residualize_against(R, np.zeros((4, 0)))
    assert np.allclose(out, R)


# --------------------------------------------------------------------------
# scoring + permutation null
# --------------------------------------------------------------------------

def _separable(n_per=20, n_feat=8, sep=2.5, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (n_per, n_feat)),
                   rng.normal(sep, 1, (n_per, n_feat))])
    y = np.array(["A"] * n_per + ["B"] * n_per)
    return X, y


@pytest.mark.parametrize("method", ["lr", "rf"])
def test_score_high_on_separable(method):
    X, y = _separable()
    mean, std = ce.score(X, y, method, n_folds=5, seed=0)
    assert mean > 0.9


def test_auc_perm_significant_on_separable():
    X, y = _separable(seed=1)
    mean, std, p = ce.auc_perm(X, y, "lr", n_folds=5, seed=0,
                               n_perm=30, n_threads=1)
    assert mean > 0.9
    assert p <= 0.05


def test_auc_perm_not_significant_on_noise():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 8))
    y = np.array(["A"] * 20 + ["B"] * 20)
    mean, std, p = ce.auc_perm(X, y, "lr", n_folds=5, seed=0,
                               n_perm=40, n_threads=1)
    # pure noise: chance AUC and a non-tiny permutation p-value
    assert p > 0.05


def test_cv_predict_binary_returns_finite_arrays():
    X, y = _separable(seed=3)
    out = ce.cv_predict(X, y, "lr", n_folds=5, seed=0)
    assert out is not None
    ytrue, yscore = out
    assert np.isfinite(ytrue).all() and np.isfinite(yscore).all()
    assert set(np.unique(ytrue)).issubset({0.0, 1.0})


def test_cv_predict_multiclass_flattened_ovr():
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(c * 3, 1, (15, 6)) for c in range(3)])
    y = np.array(["A"] * 15 + ["B"] * 15 + ["C"] * 15)
    out = ce.cv_predict(X, y, "lr", n_folds=5, seed=0)
    assert out is not None
    ytrue, yscore = out
    assert ytrue.shape == yscore.shape
    assert set(np.unique(ytrue)).issubset({0.0, 1.0})


# --------------------------------------------------------------------------
# plotting helpers
# --------------------------------------------------------------------------

def test_decade_fmt_labels():
    assert ce.decade_fmt(1000) == "3 (1kb)"
    assert ce.decade_fmt(100000) == "5 (100kb)"
    assert ce.decade_fmt(1_000_000) == "6 (1Mb)"
