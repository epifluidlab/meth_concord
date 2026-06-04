"""Tests for the J3 per-channel embedding core (design §6)."""

from __future__ import annotations

import numpy as np
import pytest

from nome_hic_concordance.joint_embedding import (
    build_embedding, channel_auc, normalize_channel, per_cell_rank,
    per_channel_svd,
)


# --------------------------------------------------------------------------
# per_cell_rank / normalize_channel
# --------------------------------------------------------------------------

def test_per_cell_rank_in_unit_interval_preserves_nan():
    M = np.array([[3.0, 1.0, 2.0, np.nan],
                  [np.nan, np.nan, np.nan, np.nan],
                  [10.0, 20.0, 30.0, 40.0]])
    R = per_cell_rank(M)
    # NaN preserved
    assert np.isnan(R[0, 3]) and np.isnan(R[1]).all()
    # row 0 observed ranks in (0,1), monotone with values (1<2<3)
    assert R[0, 1] < R[0, 2] < R[0, 0]
    assert ((R[0, :3] > 0) & (R[0, :3] < 1)).all()


def test_normalize_channel_nan_free_and_centered():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(30, 12))
    M[rng.random((30, 12)) < 0.2] = np.nan        # 20% missing
    X = normalize_channel(M)
    assert np.isfinite(X).all()                   # NaN-free after impute
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-9)   # per-bin centered


def test_normalize_channel_removes_per_cell_scale():
    # Two cells with the SAME ranking pattern but very different magnitude
    # should map to identical normalized rows (per-cell-rank removes scale).
    base = np.arange(1, 11, dtype=float)
    M = np.vstack([base, 1000 * base, base[::-1]])
    X = normalize_channel(M, do_rank=True)
    assert np.allclose(X[0], X[1], atol=1e-9)     # scale removed
    assert not np.allclose(X[0], X[2])            # different ordering kept


# --------------------------------------------------------------------------
# per_channel_svd
# --------------------------------------------------------------------------

def test_per_channel_svd_concat_width():
    rng = np.random.default_rng(1)
    chans = [rng.normal(size=(40, 50)) for _ in range(4)]
    scores, blocks = per_channel_svd(chans, k_per_channel=8)
    assert len(blocks) == 4
    assert all(b.shape == (40, 8) for b in blocks)
    assert scores.shape == (40, 32)               # 4 channels × 8


def test_per_channel_svd_handles_thin_channel():
    rng = np.random.default_rng(2)
    # one channel has only 3 regions (< k): padded/standardized, not crashing
    chans = [rng.normal(size=(20, 30)), rng.normal(size=(20, 3))]
    scores, blocks = per_channel_svd(chans, k_per_channel=10)
    assert blocks[0].shape[0] == 20 and blocks[1].shape[0] == 20
    assert scores.shape[0] == 20


# --------------------------------------------------------------------------
# channel_auc — the §6 "which co-state carries identity" diagnostic
# --------------------------------------------------------------------------

def test_channel_auc_identifies_informative_channel():
    rng = np.random.default_rng(3)
    n = 40
    y = np.array(["A"] * 20 + ["B"] * 20)
    # channel 0 carries a REGIONAL pattern that differs by group (high on the
    # first half of regions for A, on the second half for B) — this survives the
    # per-cell-rank / per-bin-QN normalization (a global per-cell shift would
    # not, by design). Channels 1-3 are pure noise.
    half = 15
    A = np.hstack([rng.normal(1, 0.3, (20, half)),
                   rng.normal(-1, 0.3, (20, 30 - half))])
    B = np.hstack([rng.normal(-1, 0.3, (20, half)),
                   rng.normal(1, 0.3, (20, 30 - half))])
    sep = np.vstack([A, B])
    noise = [rng.normal(0, 1, (n, 30)) for _ in range(3)]
    chans = [sep] + noise
    names = ["open", "closed", "poised", "active_methylated"]
    _, blocks, _ = build_embedding(chans, names, k_per_channel=6, seed=0)
    aucs = channel_auc(blocks, names, y, method="lr", n_folds=4, seed=0)
    assert aucs["open"][0] > 0.85                 # informative channel
    for nm in ("closed", "poised", "active_methylated"):
        assert aucs[nm][0] < 0.75                 # noise channels ~ chance


def test_build_embedding_shapes_and_normed():
    rng = np.random.default_rng(4)
    chans = [rng.normal(size=(25, 40)) for _ in range(4)]
    for c in chans:
        c[rng.random((25, 40)) < 0.1] = np.nan
    names = ["open", "closed", "poised", "active_methylated"]
    scores, blocks, normed = build_embedding(chans, names, k_per_channel=5)
    assert scores.shape == (25, 20)               # 4 × 5
    assert len(normed) == 4 and all(np.isfinite(N).all() for N in normed)
