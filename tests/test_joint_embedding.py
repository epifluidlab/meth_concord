"""Tests for the J3 per-channel embedding core (design §6)."""

from __future__ import annotations

import numpy as np
import pytest

from nome_hic_concordance.joint_embedding import (
    build_embedding, channel_auc, normalize_channel, per_cell_rank,
    per_channel_svd, per_component_auc, rwr_embed, tensor_embed,
)


def _two_cluster_channels(n_per=15, n_reg=30, noise=0.6, seed=0):
    """4 channels carrying a 2-cluster REGIONAL pattern (+ noise)."""
    rng = np.random.default_rng(seed)
    half = n_reg // 2

    def block(hi_first):
        sign = 1.0 if hi_first else -1.0
        base = sign * np.hstack([np.ones((n_per, half)),
                                 -np.ones((n_per, n_reg - half))])
        return base + rng.normal(0, noise, (n_per, n_reg))
    chans = [np.vstack([block(True), block(False)]) for _ in range(4)]
    y = np.array(["A"] * n_per + ["B"] * n_per)
    return chans, y


def _within_cluster_dist(M, y):
    from scipy.spatial.distance import pdist
    tot, n = 0.0, 0
    for g in np.unique(y):
        d = pdist(M[y == g])
        tot += d.sum(); n += len(d)
    return tot / max(n, 1)


def _cluster_basis(y, n_feat=8, seed=0):
    """A marginal-like (mean accessibility ⊕ methylation) basis that separates
    the clusters — stands in for the per-cell marginal the package RWR uses."""
    rng = np.random.default_rng(seed)
    classes = {g: i for i, g in enumerate(sorted(set(y)))}
    centers = rng.normal(0, 3, (len(classes), n_feat))
    return np.vstack([centers[classes[g]] + rng.normal(0, 0.3, n_feat) for g in y])


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


# --------------------------------------------------------------------------
# rwr_embed — standard (package) RWR densification on the marginal graph
# --------------------------------------------------------------------------

_NAMES = ["open", "closed", "poised", "active_methylated"]


def test_rwr_shapes_and_marginal_basis():
    chans, y = _two_cluster_channels(seed=1)
    normed = [normalize_channel(M) for M in chans]
    basis = _cluster_basis(y, seed=1)
    scores, blocks, smoothed, info = rwr_embed(
        normed, _NAMES, basis, k_per_channel=4, n_neighbors=5, seed=0)
    assert scores.shape == (30, 16)               # 4 channels × 4
    assert len(smoothed) == 4
    # graph built from the marginal (mean accessibility ⊕ methylation) — the
    # same basis as the package RWR
    assert "accessibility" in info["graph_basis"]
    assert "methylation" in info["graph_basis"]


def test_rwr_tightens_within_cluster():
    # RWR on a marginal graph that correctly groups same-cluster cells should
    # pull their channel rows closer together (the denoising property).
    chans, y = _two_cluster_channels(noise=0.6, seed=2)
    normed = [normalize_channel(M) for M in chans]
    basis = _cluster_basis(y, seed=2)
    _, _, smoothed, _ = rwr_embed(normed, _NAMES, basis, k_per_channel=4,
                                  n_neighbors=5, restart_prob=0.5, seed=0)
    before = _within_cluster_dist(normed[0], y)
    after = _within_cluster_dist(smoothed[0], y)
    assert after < before


def test_rwr_separates_clusters():
    chans, y = _two_cluster_channels(noise=0.6, seed=3)
    normed = [normalize_channel(M) for M in chans]
    basis = _cluster_basis(y, seed=3)
    _, blocks, _, _ = rwr_embed(normed, _NAMES, basis, k_per_channel=4,
                                n_neighbors=5, seed=0)
    aucs = channel_auc(blocks, _NAMES, y, method="lr", n_folds=4, seed=0)
    # every channel carries the 2-cluster pattern → all separate well
    assert all(aucs[n][0] > 0.8 for n in _NAMES)


def test_rwr_deterministic():
    chans, y = _two_cluster_channels(seed=4)
    normed = [normalize_channel(M) for M in chans]
    basis = _cluster_basis(y, seed=4)
    s1, *_ = rwr_embed(normed, _NAMES, basis, k_per_channel=4, n_neighbors=5,
                       seed=0)
    s2, *_ = rwr_embed(normed, _NAMES, basis, k_per_channel=4, n_neighbors=5,
                       seed=0)
    assert np.allclose(s1, s2)


def test_rwr_fixed_refine_passes():
    chans, y = _two_cluster_channels(seed=5)
    normed = [normalize_channel(M) for M in chans]
    basis = _cluster_basis(y, seed=5)
    _, _, _, info = rwr_embed(normed, _NAMES, basis, k_per_channel=4,
                              n_neighbors=5, n_refine_passes=2, seed=0)
    assert info["selected_pass"] == 2
    assert len(info["passes"]) == 3                      # pass 0, 1, 2
    assert info["passes"][0]["basis"].startswith("QN(mean")
    assert "concordance" in info["passes"][1]["basis"]   # refine from concordance


def test_rwr_auto_select_never_pass0():
    chans, y = _two_cluster_channels(seed=6)
    normed = [normalize_channel(M) for M in chans]
    basis = _cluster_basis(y, seed=6)
    _, _, _, info = rwr_embed(normed, _NAMES, basis, k_per_channel=4,
                              n_neighbors=5, auto_select=True, max_passes=3,
                              seed=0)
    assert info["auto_select"] is True
    assert info["selected_pass"] >= 1                    # pass 0 never selected
    # every refinement pass recorded an eigengap
    assert all("eigengap" in p for p in info["passes"][1:])


# --------------------------------------------------------------------------
# per_component_auc + per-channel skip-top
# --------------------------------------------------------------------------

def test_per_component_auc_flags_informative_component():
    rng = np.random.default_rng(7)
    y = np.array(["A"] * 15 + ["B"] * 15)
    # channel "open": component 0 separates the groups, the rest are noise
    sep = np.concatenate([rng.normal(2, 0.3, 15), rng.normal(-2, 0.3, 15)])
    block_open = np.column_stack([sep, rng.normal(0, 1, (30, 4))])
    block_noise = rng.normal(0, 1, (30, 5))
    pc = per_component_auc([block_open, block_noise], ["open", "closed"], y,
                           max_comp=5)
    assert pc["open"][0] > 0.9         # leading component carries cell type
    assert max(pc["open"][1:]) < 0.8   # remaining components ~ chance
    assert all(a < 0.8 for a in pc["closed"])   # pure-noise channel


def test_per_channel_svd_skip_top_changes_embedding():
    chans, _ = _two_cluster_channels(seed=8)
    normed = [normalize_channel(M) for M in chans]
    s0, _ = per_channel_svd(normed, k_per_channel=3, skip_top=0)
    s1, b1 = per_channel_svd(normed, k_per_channel=3, skip_top=1)
    assert s0.shape == s1.shape == (30, 12)   # 4 channels × 3, shape preserved
    assert not np.allclose(s0, s1)            # dropping the top component changes it
    assert all(b.shape == (30, 3) for b in b1)


# --------------------------------------------------------------------------
# tensor_embed (needs tensorly — skipped where absent)
# --------------------------------------------------------------------------

def test_tensor_embed_tucker_cell_and_channel_factors():
    pytest.importorskip("tensorly")
    chans, y = _two_cluster_channels(seed=9, n_reg=30)
    normed = [normalize_channel(M) for M in chans]
    cell, chan, info = tensor_embed(normed, _NAMES, method="tucker",
                                    rank=(6, 8, 4), seed=0)
    assert cell.shape == (30, 6)              # cells × r_cell = the embedding
    assert chan.shape == (4, 4)              # K channels × r_channel
    assert info["method"] == "tucker" and info["cell_factor_dim"] == 6


def test_tensor_embed_cp_shapes():
    pytest.importorskip("tensorly")
    chans, _ = _two_cluster_channels(seed=10, n_reg=30)
    normed = [normalize_channel(M) for M in chans]
    cell, chan, info = tensor_embed(normed, _NAMES, method="cp", rank=(5,), seed=0)
    assert cell.shape == (30, 5)             # cells × R
    assert chan.shape == (4, 5)              # K channels × R
    assert info["method"] == "cp"


def test_tensor_embed_separates_clusters():
    pytest.importorskip("tensorly")
    chans, y = _two_cluster_channels(noise=0.6, seed=11, n_reg=30)
    normed = [normalize_channel(M) for M in chans]
    cell, _, _ = tensor_embed(normed, _NAMES, method="tucker", rank=(6, 8, 4),
                              seed=0)
    from nome_hic_concordance import coordination_eval as ce
    auc, _ = ce.score(cell, y, "lr", 4, 0)
    assert auc > 0.8                         # the cell factor carries cell type
