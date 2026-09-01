"""Two- and three-stage marginal correction.

The fixture plants a marginal-driven replicate effect of the kind measured on
the real data: latent factors of p_bar leaking into the concordance with
BIN-SPECIFIC weights.  A uniform per-cell shift would not do -- the pipeline's
per-cell rank normalisation already removes those, which is why global depth
corrected nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import marginal_correction as mc


def _purity(S, ct, bt, k=25):
    """Cell-type purity with neighbours restricted to the OTHER replicate.

    Ordinary purity cannot see this failure: a cell whose neighbours are all
    its own type from its OWN replicate scores 1.0, which IS the failure.
    """
    from sklearn.neighbors import NearestNeighbors
    ct, bt = np.asarray(ct, str), np.asarray(bt, str)
    within, cross = np.full(len(ct), np.nan), np.full(len(ct), np.nan)
    for lv in sorted(pd.unique(bt)):
        own, oth = np.flatnonzero(bt == lv), np.flatnonzero(bt != lv)
        for dst, out in ((own, within), (oth, cross)):
            kk = min(k, len(dst) - (1 if dst is own else 0))
            nn = NearestNeighbors(n_neighbors=min(kk + 1, len(dst))).fit(S[dst])
            _, idx = nn.kneighbors(S[own])
            for r, i in enumerate(own):
                nb = dst[idx[r]]
                nb = nb[nb != i][:kk]
                out[i] = float(np.mean(ct[nb] == ct[i])) if len(nb) else np.nan
    w, c = float(np.nanmean(within)), float(np.nanmean(cross))
    return w, c, (c / w if w else np.nan)


def _scores(M, n=15, seed=0):
    """Per-cell rank then SVD -- the space the pipeline embeds."""
    from scipy.stats import rankdata
    from sklearn.decomposition import TruncatedSVD
    X = M.to_numpy(float)
    R = np.full_like(X, np.nan)
    for i in range(X.shape[0]):
        ok = ~np.isnan(X[i])
        if ok.sum():
            R[i, ok] = rankdata(X[i, ok]) / (ok.sum() + 1) - 0.5
    R = np.nan_to_num(R, nan=0.0)
    n = max(1, min(n, min(R.shape) - 1))
    return TruncatedSVD(n_components=n, random_state=seed).fit_transform(R)


@pytest.fixture(scope="module")
def planted():
    """Shared biology, plus p_bar factors leaking into the concordance."""
    rng = np.random.default_rng(0)
    n_t, per, nb, kt = 5, 40, 300, 3
    ct = np.repeat([f"T{t}" for t in range(n_t)], per * 2)
    bt = np.tile(np.repeat(["A", "B"], per), n_t)
    n = len(ct)
    cells = [f"c{i}" for i in range(n)]
    cols = [f"chr1:{j}" for j in range(nb)]
    cent = {t: rng.normal(0, 1.4, nb) for t in set(ct)}
    bio = np.vstack([cent[t] for t in ct]) + rng.normal(0, 0.35, (n, nb))
    # marginal: latent factors, the first REPLICATE-LINKED with little
    # per-cell jitter.  Heavy per-cell jitter would wreck within-replicate
    # type structure too, giving two equally-broken numbers and a ratio near
    # 1 -- the opposite of the signature being reproduced.
    Up = rng.normal(0, 0.25, (n, kt))
    Up[:, 0] += np.where(bt == "A", 1.0, -1.0)
    Vp = rng.normal(0, 1, (kt, nb))
    P = np.clip(0.35 + 0.05 * (Up @ Vp) / np.abs(Up @ Vp).std(), 0.02, 0.98)
    # they leak into the concordance with BIN-SPECIFIC weights
    M = bio + Up @ rng.normal(0, 3.0, (kt, nb)) + rng.normal(0, 0.2, (n, nb))
    f = dict(index=cells, columns=cols)
    return {
        "M": pd.DataFrame(M, **f),
        "P": pd.DataFrame(P, **f),
        "base": pd.DataFrame(bio + rng.normal(0, 0.4, (n, nb)), **f),
        "batch": pd.Series(bt, index=cells),
        "ct": ct, "bt": bt,
    }


# ------------------------------------------------------------- the claims --

def test_the_planted_effect_actually_breaks_cross_replicate_purity(planted):
    """A fixture everything passes tests nothing."""
    w, c, r = _purity(_scores(planted["M"]), planted["ct"], planted["bt"])
    assert w > 0.8, ("types must be recovered WITHIN replicate", w)
    assert r < 0.75, ("and broken ACROSS it", w, c, r)


def test_two_stage_recovers_cross_replicate_alignment(planted):
    out = mc.two_stage(planted["M"], planted["P"], planted["batch"],
                       baseline=planted["base"], k=3)
    w0, c0, r0 = _purity(_scores(planted["M"]), planted["ct"], planted["bt"])
    w1, c1, r1 = _purity(_scores(out), planted["ct"], planted["bt"])
    assert r1 > r0 + 0.1, (r0, r1)
    # and it must not buy the ratio by flattening cell state
    assert w1 > 0.9 * w0, (w0, w1)


def test_protection_is_wired_in_and_changes_the_fit(planted):
    """The protected and unprotected fits must not coincide.

    WHICH IS BETTER IS AN EMPIRICAL QUESTION, NOT A UNIT TEST.  On the real
    data unprotected stage 1 drove cross-replicate purity DOWN (0.142 ->
    0.083) while protected drove it up (-> 0.332), because p_bar carries
    real accessibility biology alongside the technical drift.  That result
    lives in the module docstring with its numbers; what is checked here is
    only that the baseline actually enters the design, so the two paths
    cannot silently be the same code.
    """
    prot = mc.protect_design(planted["base"], planted["M"].index, 10)
    assert prot is not None and prot.shape[1] == 10
    with_p = mc.stage1_marginal(planted["M"], planted["P"], prot, k=3)
    without = mc.stage1_marginal(planted["M"], planted["P"], None, k=3)
    d = float(np.nanmax(np.abs(with_p.to_numpy(float)
                               - without.to_numpy(float))))
    assert d > 1e-6, ("protection had no effect on the fit", d)


def test_stage_1_removes_the_marginal_factors_it_claims_to(planted):
    out = mc.stage1_marginal(planted["M"], planted["P"], None, k=3)
    from sklearn.decomposition import TruncatedSVD
    X = mc.marginal_transform(planted["P"].to_numpy(float), "d")
    U = TruncatedSVD(n_components=3, random_state=0).fit_transform(
        X - X.mean(0, keepdims=True))
    S = _scores(out, n=8)
    r = max(abs(np.corrcoef(S[:, i], U[:, j])[0, 1])
            for i in range(S.shape[1]) for j in range(U.shape[1]))
    assert r < 0.2, r


def test_stage_2_needs_a_reference_that_matches_the_target(planted):
    """Anchored on accessibility the local matching reached 0.290; anchored
    on the stage-1 output, 0.724.  The anchor space is not incidental."""
    prot = mc.protect_design(planted["base"], planted["M"].index, 10)
    s1 = mc.stage1_marginal(planted["M"], planted["P"], prot, k=3)
    good = mc.stage2_local(planted["M"], s1, planted["batch"], k=20)
    poor = mc.stage2_local(planted["M"], planted["M"], planted["batch"], k=20)
    _, _, r_good = _purity(_scores(good), planted["ct"], planted["bt"])
    _, _, r_poor = _purity(_scores(poor), planted["ct"], planted["bt"])
    assert r_good > r_poor, (r_good, r_poor)


def test_stage_3_residual_is_orthogonal_to_its_baselines(planted):
    out = mc.stage3_residual(planted["M"], [planted["base"]])
    B = planted["base"].to_numpy(float)
    B = B - B.mean(0, keepdims=True)
    R = out.to_numpy(float)
    R = R - R.mean(0, keepdims=True)
    # per bin, the residual must carry no remaining baseline slope
    num = float(np.abs((B * R).sum(0)).max())
    assert num < 1e-6 * max(1.0, float(np.abs(R).sum())), num


def test_three_stage_runs_and_keeps_the_frame(planted):
    out = mc.three_stage(planted["M"], planted["P"], planted["batch"],
                         baseline=planted["base"], k=3)
    assert out.shape == planted["M"].shape
    assert list(out.index) == list(planted["M"].index)
    assert list(out.columns) == list(planted["M"].columns)


# ------------------------------------------------------------- mechanics --

def test_nan_pattern_is_preserved(planted):
    M = planted["M"].copy()
    M.iloc[3, 5] = np.nan
    M.iloc[10, :4] = np.nan
    for out in (mc.stage1_marginal(M, planted["P"], None, k=2),
                mc.stage2_local(M, planted["base"], planted["batch"], k=10),
                mc.stage3_residual(M, [planted["base"]])):
        assert out.isna().equals(M.isna())


def test_transform_d_is_the_identitys_own_term():
    P = np.array([[0.1, 0.5, 0.9]])
    assert np.allclose(mc.marginal_transform(P, "d"), [[-0.8, 0.0, 0.8]])
    assert np.allclose(mc.marginal_transform(P, "raw"), P)
    lg = mc.marginal_transform(P, "logit")
    assert lg[0, 0] < 0 < lg[0, 2] and abs(lg[0, 1]) < 1e-9
    with pytest.raises(ValueError, match="transform must be"):
        mc.marginal_transform(P, "nope")


def test_one_replicate_level_is_a_no_op_not_a_crash(planted):
    one = pd.Series("only", index=planted["M"].index)
    out = mc.stage2_local(planted["M"], planted["base"], one, k=10)
    assert np.allclose(out.to_numpy(float), planted["M"].to_numpy(float))


def test_protect_design_is_optional_and_shaped_per_cell(planted):
    assert mc.protect_design(None, planted["M"].index) is None
    Z = mc.protect_design(planted["base"], planted["M"].index, 7)
    assert Z.shape == (len(planted["M"]), 7)
    assert np.allclose(Z.mean(0), 0, atol=1e-8)


def test_inputs_are_realigned_not_assumed_to_match(planted):
    """A caller passing the marginal in a different cell order must not get
    a silently scrambled correction."""
    shuffled = planted["P"].iloc[::-1]
    a = mc.stage1_marginal(planted["M"], planted["P"], None, k=2)
    b = mc.stage1_marginal(planted["M"], shuffled, None, k=2)
    assert np.allclose(a.to_numpy(float), b.to_numpy(float))
