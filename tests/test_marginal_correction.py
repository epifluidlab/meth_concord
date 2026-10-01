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


def test_stage_3_residual_still_has_something_left_in_it():
    """ORTHOGONALITY ALONE IS NOT ENOUGH, and this is not hypothetical.

    An all-zero matrix is orthogonal to everything, so the check above
    passed while the implementation regressed each bin on the WHOLE baseline
    matrix -- a design with more columns than cells, which lstsq fits
    exactly.  The residual came back at 1e-15 and within-replicate purity
    read 0.123 against a chance level of 0.115.  The shape here is the real
    regime, bins far outnumbering cells, where that degeneracy bites.
    """
    rng = np.random.default_rng(11)
    n, nb = 120, 400                      # bins >> cells, as in the data
    idx = [f"c{i}" for i in range(n)]
    cols = [f"b{j}" for j in range(nb)]
    shared = rng.normal(0, 1, (n, nb))
    base = pd.DataFrame(shared + rng.normal(0, 1, (n, nb)),
                        index=idx, columns=cols)
    M = pd.DataFrame(shared + rng.normal(0, 1, (n, nb)),
                     index=idx, columns=cols)
    out = mc.stage3_residual(M, [base])
    kept = float(np.nanstd(out.to_numpy(float))
                 / np.nanstd(M.to_numpy(float)))
    assert kept > 0.5, ("the residual was annihilated, not computed", kept)


def test_stage_3_matches_a_plain_per_bin_regression():
    """Pinned against the model comparison.variance_partition fits:
    y_j = a_j + b_j * base_j, one regression per bin."""
    rng = np.random.default_rng(12)
    n, nb = 90, 30
    idx = [f"c{i}" for i in range(n)]
    cols = [f"b{j}" for j in range(nb)]
    base = pd.DataFrame(rng.normal(0, 1, (n, nb)), index=idx, columns=cols)
    M = pd.DataFrame(rng.normal(0, 1, (n, nb)), index=idx, columns=cols)
    got = mc.stage3_residual(M, [base]).to_numpy(float)
    B, Y = base.to_numpy(float), M.to_numpy(float)
    for j in range(nb):
        x = np.column_stack([np.ones(n), B[:, j]])
        beta, *_ = np.linalg.lstsq(x, Y[:, j], rcond=None)
        assert np.allclose(got[:, j], Y[:, j] - x @ beta, atol=1e-8), j


def test_stage_3_handles_a_constant_baseline_bin():
    """A bin whose baseline never varies gives a singular system; the right
    answer is a zero slope, not an exception."""
    rng = np.random.default_rng(13)
    n, nb = 50, 12
    idx = [f"c{i}" for i in range(n)]
    cols = [f"b{j}" for j in range(nb)]
    B = rng.normal(0, 1, (n, nb))
    B[:, 3] = 2.0                          # constant across cells
    out = mc.stage3_residual(
        pd.DataFrame(rng.normal(0, 1, (n, nb)), index=idx, columns=cols),
        [pd.DataFrame(B, index=idx, columns=cols)])
    assert np.isfinite(out.to_numpy(float)).all()
    assert np.nanstd(out.to_numpy(float)[:, 3]) > 0.5


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


# --------------------------------------------- the Step-5 --marginal-correct --

def _step5(tmp_path, batch_labels=None, **kw):
    """Run Step 5 on the shared focal-active fixture."""
    from tests.test_signed_concordance import _focal_active_paths
    from nome_hic_concordance import concordance as cc
    src, out = tmp_path / "src", tmp_path / "05"
    src.mkdir(parents=True, exist_ok=True)
    cc.compute_signed_concordance_matrix(
        _focal_active_paths(src, n_cells=8), bin_size=1000, min_distance=100,
        min_gch_sites=2, baseline="distance-pooled", dm_prior="shared",
        output_dir=str(out), save_outputs=True, use_rwr=False,
        batch_labels=batch_labels, **kw)
    return out


def test_step5_marginal_correct_is_off_by_default(tmp_path):
    out = _step5(tmp_path)
    assert list(out.glob("*margreg*")) == []


def test_step5_marginal_correct_writes_both_stages(tmp_path):
    cells = [f"cell{i}" for i in range(8)]
    bl = pd.Series(["A"] * 4 + ["B"] * 4, index=cells, name="exp_batch")
    out = _step5(tmp_path, batch_labels=bl, marginal_correct=True,
                 marginal_correct_k=2)
    for stat in ("coordination", "direction_logor"):
        stem = f"concordance_{stat}.margreg_k2.method1_local"
        assert (out / f"{stem}.parquet").exists(), stat
        assert (out / f"{stem}.resid.parquet").exists(), stat
    # and the uncorrected originals must SURVIVE, not be replaced
    assert (out / "concordance_coordination.parquet").exists()


def test_step5_marginal_correct_needs_batch_labels(tmp_path, caplog):
    """Without them there is no replicate to align, and silently writing an
    unchanged 'corrected' matrix would be worse than skipping."""
    import logging
    with caplog.at_level(logging.WARNING):
        out = _step5(tmp_path, batch_labels=None, marginal_correct=True)
    assert list(out.glob("*margreg*")) == []
    assert "needs --batch-column" in caplog.text


def test_a_single_bin_matrix_is_returned_unchanged_with_a_warning(caplog):
    """Sparse contexts really do reduce to one usable bin after coverage
    filtering, and TruncatedSVD raises on a single feature.  Returning the
    matrix untouched and SAYING SO is the honest response -- a silent no-op
    is indistinguishable from a correction that worked."""
    import logging
    idx = [f"c{i}" for i in range(6)]
    M = pd.DataFrame(np.arange(6.0).reshape(6, 1), index=idx, columns=["b0"])
    P = pd.DataFrame(np.full((6, 1), 0.4), index=idx, columns=["b0"])
    batch = pd.Series(["A", "A", "A", "B", "B", "B"], index=idx)
    with caplog.at_level(logging.WARNING):
        s1 = mc.stage1_marginal(M, P, None, k=2)
        s2 = mc.stage2_local(M, M, batch, k=2)
        assert mc.protect_design(M, M.index, 5) is None
    assert np.allclose(s1.to_numpy(float), M.to_numpy(float))
    assert np.allclose(s2.to_numpy(float), M.to_numpy(float))
    assert "at least 2 cells and 2 bins" in caplog.text


# ------------------------------------- nome_hic_compare --marginal-correct --

def _compare_inputs(tmp_path, n=40, nb=60):
    rng = np.random.default_rng(3)
    cells = [f"c{i}" for i in range(n)]
    cols = [f"chr1:{j}" for j in range(nb)]
    bt = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    bio = rng.normal(0, 1, (n, nb))
    f = dict(index=cells, columns=cols)
    d = tmp_path
    pd.DataFrame(bio + rng.normal(0, .2, (n, nb)), **f).to_parquet(d / "base.parquet")
    pd.DataFrame(bio + rng.normal(0, .5, (n, nb)), **f).to_parquet(d / "conc.parquet")
    pd.DataFrame(np.clip(0.35 + .05 * bio
                         + np.where(bt == "A", .05, -.05)[:, None], .02, .98),
                 **f).to_parquet(d / "marg.parquet")
    sp = d / "s.tsv"
    pd.DataFrame({"cell_name": cells, "exp_batch": bt,
                  "group": np.resize(["T1", "T2"], n)}).to_csv(
        sp, sep="\t", index=False)
    return d, sp


def test_compare_marginal_correct_records_what_it_did(tmp_path):
    """The summary has to say the correction ran, with its k -- otherwise a
    corrected and an uncorrected run are indistinguishable afterwards."""
    import json
    import subprocess
    import sys as _sys
    d, sp = _compare_inputs(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [_sys.executable, "-m", "nome_hic_concordance.comparison_cli",
         "--baseline-matrix", str(d / "base.parquet"),
         "--concordance-matrix", f"gch={d / 'conc.parquet'}",
         "--samples", str(sp), "--output-dir", str(out),
         "--marginal-matrix", str(d / "marg.parquet"),
         "--marginal-correct", "--marginal-correct-k", "2",
         "--batch-column", "exp_batch", "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.load(open(out / "summary.json")).get("marginal_correct")
    assert got is not None, "summary.json did not record the correction"
    assert got["k"] == 2 and got["matrices"] == ["gch"]


def test_compare_marginal_correct_refuses_without_a_baseline(tmp_path):
    """Unprotected, stage 1 inverts (0.142 -> 0.083 measured).  Silently
    running it without a baseline would make things worse while looking
    like it worked."""
    import subprocess
    import sys as _sys
    d, sp = _compare_inputs(tmp_path)
    out = tmp_path / "out2"
    r = subprocess.run(
        [_sys.executable, "-m", "nome_hic_concordance.comparison_cli",
         "--concordance-matrix", f"gch={d / 'conc.parquet'}",
         "--samples", str(sp), "--output-dir", str(out),
         "--marginal-matrix", str(d / "marg.parquet"),
         "--marginal-correct", "--batch-column", "exp_batch",
         "--plot-format", "png"],
        capture_output=True, text=True)
    assert "no baseline matrix to protect" in r.stderr


def test_step5_marginal_correct_also_corrects_the_drawn_null(tmp_path):
    """--emit-null must give the corrected matrices a corrected null.

    Without it null_calibration_qq has nothing valid to plot the corrected
    statistic against, and plotting it against the UNCORRECTED null would
    put the correction itself on the Q-Q.
    """
    cells = [f"cell{i}" for i in range(8)]
    bl = pd.Series(["A"] * 4 + ["B"] * 4, index=cells, name="exp_batch")
    out = _step5(tmp_path, batch_labels=bl, marginal_correct=True,
                 marginal_correct_k=2, emit_null=True)
    for stat in ("coordination", "direction_logor"):
        stem = f"concordance_{stat}.margreg_k2.method1_local"
        assert (out / f"{stem}.parquet").exists(), stat
        assert (out / f"{stem}_null.parquet").exists(), f"{stat}: null missing"
        assert (out / f"{stem}.resid_null.parquet").exists(), stat
    # the plain drawn null must still be written too
    assert (out / "concordance_direction_logor_null.parquet").exists()
    assert (out / "concordance_marginal_null.parquet").exists()


# ------------------------------------ the Step-5 residual, the HCG path --

_STATS = ("coordination", "direction_logor")


def _script_main():
    import importlib.util
    from pathlib import Path
    path = (Path(__file__).resolve().parent.parent / "scripts"
            / "marginal_correct.py")
    spec = importlib.util.spec_from_file_location("marginal_correct_cli", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.main


def _batches(out):
    cells = pd.read_parquet(out / "concordance_coordination.parquet").index
    half = len(cells) // 2
    return pd.Series(["A"] * half + ["B"] * (len(cells) - half),
                     index=cells, name="exp_batch")


def test_step5_writes_the_hcg_residual_routinely(tmp_path):
    """A run without --marginal-correct (the HCG run) used to write no
    residual at all; it took a second pass with the script, and a run that
    skipped the pass had no concordance_coordination.resid.parquet."""
    out = _step5(tmp_path, emit_null=True)
    for stat in _STATS:
        assert (out / f"concordance_{stat}.resid.parquet").exists(), stat
        assert (out / f"concordance_{stat}.resid_null.parquet").exists(), stat
    assert list(out.glob("*margreg*")) == []
    # the uncorrected matrices are still there, untouched
    assert (out / "concordance_coordination.parquet").exists()


def test_step5_residual_is_the_file_the_script_writes(tmp_path):
    """Same names, same values as marginal_correct.py --residual-only, so a
    directory from an older run patched with the script and a new run agree.

    Run WITH batch correction on: Step 5 corrects the accessibility it builds
    the k-NN graph from, and the residual must not pick up that copy.  The
    script -- and every measurement behind the residual -- regresses on the
    raw mean in accessibility_mean.parquet.
    """
    src = tmp_path / "first"
    probe = _step5(src)
    bl = _batches(probe)
    out = _step5(tmp_path, batch_labels=bl, batch_correction="center",
                 emit_null=True)
    sp = tmp_path / "samples.tsv"
    pd.DataFrame({"cell_name": bl.index, "exp_batch": bl.values}).to_csv(
        sp, sep="\t", index=False)
    ref = tmp_path / "script"
    assert _script_main()([str(out), str(sp), "--batch-col", "exp_batch",
                           "--residual-only", "--output-dir", str(ref)]) == 0
    for stat in _STATS:
        for sfx in ("", "_null"):
            name = f"concordance_{stat}.resid{sfx}.parquet"
            a, b = pd.read_parquet(out / name), pd.read_parquet(ref / name)
            assert list(a.index) == list(b.index), name
            assert list(a.columns) == list(b.columns), name
            assert np.array_equal(a.to_numpy(float), b.to_numpy(float),
                                  equal_nan=True), name



def test_step5_residual_regresses_on_the_raw_accessibility(tmp_path,
                                                            monkeypatch):
    """Not the batch-corrected copy the k-NN graph is built from.

    Checked on the baseline actually handed to the regression: the shared
    fixture's accessibility is one value in every cell, so the fitted slope
    is zero against any baseline and the residual files alone could not
    tell the two apart.
    """
    seen = []
    real = mc.stage3_residual

    def spy(matrix, baselines):
        seen.append([b.copy() for b in baselines])
        return real(matrix, baselines)

    monkeypatch.setattr(mc, "stage3_residual", spy)
    probe = _step5(tmp_path / "first")
    bl = _batches(probe)
    seen.clear()
    from tests.test_signed_concordance import _focal_active_paths
    from nome_hic_concordance import concordance as cc
    src, out = tmp_path / "src", tmp_path / "05"
    src.mkdir()
    res = cc.compute_signed_concordance_matrix(
        _focal_active_paths(src, n_cells=8), bin_size=1000, min_distance=100,
        min_gch_sites=2, baseline="distance-pooled", dm_prior="shared",
        output_dir=str(out), save_outputs=True, use_rwr=False,
        batch_labels=bl, batch_correction="center")
    raw = pd.read_parquet(out / "accessibility_mean.parquet")
    assert seen, "stage3_residual was never called"
    for (base,) in seen:
        b = base.reindex(index=raw.index, columns=raw.columns)
        assert np.allclose(b.to_numpy(float), raw.to_numpy(float),
                           equal_nan=True)
    # the correction really did change the copy the graph used, so the
    # check above is not vacuous
    corrected = res["accessibility"].reindex(index=raw.index,
                                             columns=raw.columns)
    assert not np.allclose(corrected.to_numpy(float), raw.to_numpy(float),
                           equal_nan=True)


def test_step5_no_residual_writes_none(tmp_path):
    out = _step5(tmp_path, emit_null=True, residual=False)
    assert list(out.glob("*.resid*")) == []


def test_step5_with_marginal_correct_writes_no_plain_residual(tmp_path):
    """GCH keeps its own residual -- the .resid sibling of the corrected
    matrix -- and does not also get a plain one beside it."""
    cells = [f"cell{i}" for i in range(8)]
    bl = pd.Series(["A"] * 4 + ["B"] * 4, index=cells, name="exp_batch")
    out = _step5(tmp_path, batch_labels=bl, marginal_correct=True,
                 marginal_correct_k=2)
    assert (out / "concordance_coordination.margreg_k2.method1_local"
                  ".resid.parquet").exists()
    assert not (out / "concordance_coordination.resid.parquet").exists()


def test_step5_no_residual_also_skips_the_corrected_residual(tmp_path):
    """--no-residual means what the script's flag of the same name means."""
    cells = [f"cell{i}" for i in range(8)]
    bl = pd.Series(["A"] * 4 + ["B"] * 4, index=cells, name="exp_batch")
    out = _step5(tmp_path, batch_labels=bl, marginal_correct=True,
                 marginal_correct_k=2, residual=False)
    assert (out / "concordance_coordination.margreg_k2.method1_local"
                  ".parquet").exists()
    assert list(out.glob("*.resid*")) == []


def test_cli_no_residual_flag():
    from nome_hic_concordance.cli import parse_args
    base = ["--samples", "s.tsv", "--output-dir", "o"]
    assert parse_args(base).no_residual is False
    assert parse_args(base + ["--no-residual"]).no_residual is True


def test_step5_says_so_when_there_is_no_null_to_correct(tmp_path, caplog):
    import logging
    cells = [f"cell{i}" for i in range(8)]
    bl = pd.Series(["A"] * 4 + ["B"] * 4, index=cells, name="exp_batch")
    with caplog.at_level(logging.INFO):
        out = _step5(tmp_path, batch_labels=bl, marginal_correct=True,
                     marginal_correct_k=2)
    assert not list(out.glob("*margreg*_null.parquet"))
    assert "no drawn null to correct" in caplog.text
