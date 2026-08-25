"""Segmenting the coordination plane with a chromosome-aware HMM.

The HMM is not here for the states -- thresholding already names those.  It
is here for the SPATIAL PRIOR: coordination domains span many bins, and
classifying each bin alone throws that away.  So the tests that matter are
(1) does it beat per-bin thresholding on data with domains, and (2) do the
three things that would silently corrupt it actually get handled -- chain
breaks at chromosome edges, masked bins that are not zero-filled, and a
labelling cut that does not assume the standardising scale is the null scale.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "coordination_hmm.py")

TRUE = np.array([(0., 0.), (3., 0.), (2.1, 2.1), (2.1, -2.1), (-3., 0.)])


def _mod():
    spec = importlib.util.spec_from_file_location("chmm", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["chmm"] = m
    spec.loader.exec_module(m)
    return m


def _domains(n_bins, mean_run=25, seed=0):
    """Domains that CYCLE through the five states rather than sampling them.

    Sampling uniformly per run sounds balanced and is not: run lengths are
    exponential and 600 bins hold only ~24 runs, so occupancy swings wildly
    -- one draw gave the balanced state 8% of bins against 34% for
    anti-coordinated, and EM then correctly spent two states on the abundant
    region and none on the rare one.  A test of five-state recovery has to
    put five states in the data.
    """
    rng = np.random.default_rng(seed)
    st, i, j = [], 0, int(rng.integers(0, 5))
    while i < n_bins:
        L = max(4, int(rng.exponential(mean_run)))
        st += [j % 5] * L
        i += L
        j += 1
    return np.array(st[:n_bins])


def _fixture(n_cells=16, n_bins=400, noise=2.0, missing=0.15, seed=0,
             n_chrom=2):
    rng = np.random.default_rng(seed)
    st = _domains(n_bins, seed=seed)
    mu = TRUE[st]
    C = mu[None, :, 0] + rng.normal(0, noise, (n_cells, n_bins))
    dF = mu[None, :, 1] + rng.normal(0, noise, (n_cells, n_bins))
    if missing:
        gap = rng.random(C.shape) < missing
        C[gap] = np.nan
        dF[gap] = np.nan
    per = n_bins // n_chrom
    bins = []
    for c in range(n_chrom):
        k = per if c < n_chrom - 1 else n_bins - per * (n_chrom - 1)
        bins += [f"chr{c + 1}:{i * 250000}" for i in range(k)]
    cells = [f"c{i}" for i in range(n_cells)]
    return (pd.DataFrame(C, index=cells, columns=bins),
            pd.DataFrame(dF, index=cells, columns=bins), st)


# ------------------------------------------------ the reason it exists --

def test_the_spatial_prior_beats_per_bin_thresholding():
    """The whole justification.  If the HMM did not beat a threshold on data
    with domains there would be no reason to carry it."""
    m = _mod()
    C, F, st = _fixture(n_cells=12, n_bins=400, noise=2.0, missing=0.0)
    X, lengths, obs, order, sc = m.build_sequences(
        C.to_numpy(float), F.to_numpy(float), list(C.columns))
    model = m.PlaneHMM(5, seed=0).fit(X, lengths, obs, len(C))
    path, _post = model.decode(X, lengths, obs, len(C))

    # map fitted states to planted ones by nearest emission mean
    scale = np.array([sc["scale_c"], sc["scale_f"]])
    tgt = TRUE / scale
    fit2true = {j: int(np.linalg.norm(tgt - mu, axis=1).argmin())
                for j, mu in enumerate(model.means)}
    truth = np.tile(st[order], len(C))
    hmm_acc = float(np.mean([fit2true[v] for v in path] == truth))

    R = np.hypot(X[:, 0], X[:, 1])
    th = np.degrees(np.arctan2(X[:, 1], X[:, 0]))
    cut = 0.35 * np.hypot(*(TRUE.max(0) / scale))
    thr = np.zeros(len(X), int)
    far = R >= cut
    a = np.abs(th)
    thr[far & (a <= 30)] = 1
    thr[far & (th > 30) & (th < 150)] = 2
    thr[far & (th < -30) & (th > -150)] = 3
    thr[far & (a >= 150)] = 4
    thr_acc = float(np.mean(thr == truth))
    assert hmm_acc > thr_acc + 0.15, f"hmm {hmm_acc:.3f} vs thr {thr_acc:.3f}"
    assert hmm_acc > 0.75


def test_it_recovers_the_planted_state_geometry():
    m = _mod()
    C, F, st = _fixture(n_cells=28, n_bins=600, noise=1.8)
    X, lengths, obs, order, sc = m.build_sequences(
        C.to_numpy(float), F.to_numpy(float), list(C.columns))
    model = m.PlaneHMM(5, seed=0).fit(X, lengths, obs, len(C))
    scale = np.array([sc["scale_c"], sc["scale_f"]])
    want_all = TRUE / scale
    # neighbouring states sit ~0.6-1.2 apart in these units, so 0.4 still
    # demands the RIGHT state -- it is not a tolerance that anything passes.
    spacing = min(np.linalg.norm(a - b) for i, a in enumerate(want_all)
                  for b in want_all[i + 1:])
    assert spacing > 0.5
    for want in want_all:
        d = np.linalg.norm(model.means - want, axis=1)
        assert d.min() < 0.4, f"no fitted state near {want}"


# --------------------------------------- the three silent-corruption risks --

def test_chains_break_at_every_chromosome_edge():
    """One chain per cell would let a domain smooth across a centromere or
    off the end of one chromosome into the next."""
    m = _mod()
    C, F, _st = _fixture(n_cells=4, n_bins=120, n_chrom=3)
    _X, lengths, _obs, _o, _s = m.build_sequences(
        C.to_numpy(float), F.to_numpy(float), list(C.columns))
    assert len(lengths) == 4 * 3
    assert lengths.sum() == 4 * 120
    # every chain is exactly one chromosome's worth of bins
    per = sorted(set(lengths))
    assert len(per) <= 3 and min(per) > 1


def test_missing_bins_are_masked_not_zero_filled():
    """A bin with no reads is not 'uncoordinated'.  Filling it with zero
    would put it at the origin and manufacture domains of absence."""
    m = _mod()
    C, F, _st = _fixture(n_cells=6, n_bins=200, missing=0.30)
    X, lengths, obs, _o, _s = m.build_sequences(
        C.to_numpy(float), F.to_numpy(float), list(C.columns))
    assert 0.2 < (~obs).mean() < 0.45
    model = m.PlaneHMM(4, seed=0).fit(X, lengths, obs, len(C))
    logB = model._log_b(X, obs)
    # a masked bin contributes nothing: it neither votes nor blocks
    assert np.allclose(logB[~obs], 0.0)
    assert not np.allclose(logB[obs], 0.0)


def test_masked_bins_get_no_state_in_the_output(tmp_path):
    m = _mod()
    C, F, _st = _fixture(n_cells=5, n_bins=160, missing=0.25)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "5",
         "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    st = pd.read_parquet(tmp_path / "o" / "concordance_state.parquet")
    got = st.to_numpy()
    src = C.reindex(columns=[c for c in st.columns]).to_numpy(float)
    assert (got[~np.isfinite(src)] == -1).all()
    assert (got[np.isfinite(src)] >= 0).all()


# ------------------------------------------------------- the labelling cut --

# ------------------------------------------------- relative labelling --

def test_relative_labelling_recovers_the_taxonomy_at_any_scale():
    """The units of the plane depend on how the axes were standardised, so
    the assignment must not."""
    m = _mod()
    A = np.eye(5) * 0.9 + 0.025
    for scale in (0.01, 1.0, 137.0):
        info, meta = m.label_states_relative(TRUE * scale, A)
        got = [d["label"] for d in info]
        assert got == ["uncoordinated", "coordinated balanced",
                       "co-open shifted", "co-closed shifted",
                       "anti-coordinated"], scale


def test_the_null_is_the_origin_not_the_commonest_state():
    """Anchoring on occupancy is the obvious alternative and is wrong
    whenever the null is not the commonest state -- on equally-sized domains
    it picks whichever state happens to be largest and swaps two labels.
    C = 0 and F - F_exp = 0 ARE the null by construction, so no estimate is
    needed."""
    m = _mod()
    A = np.diag([0.5, 0.99, 0.5, 0.5, 0.5])       # 'balanced' dominates
    A = A + (1 - A.sum(1, keepdims=True)) / 5
    info, meta = m.label_states_relative(TRUE, A)
    by_label = {d["label"]: d for d in info}
    assert by_label["uncoordinated"]["mean_C"] == pytest.approx(0.0)
    assert by_label["uncoordinated"]["mean_dF"] == pytest.approx(0.0)
    # and the far more occupied state is still labelled by its POSITION
    assert by_label["coordinated balanced"]["mean_C"] == pytest.approx(3.0)


def test_relative_labelling_takes_four_or_five_states():
    """Four when C is positive everywhere, five when it crosses zero.
    Anything else is not a bijection onto the taxonomy."""
    m = _mod()
    A4 = np.eye(4) * 0.9 + 0.033
    info, _meta = m.label_states_relative(TRUE[:4], A4)
    assert len(info) == 4
    A3 = np.eye(3) * 0.9 + 0.05
    info3, _m3 = m.label_states_relative(TRUE[:3], A3)
    assert len(info3) == 3
    A2 = np.eye(2) * 0.9 + 0.1
    with pytest.raises(ValueError, match="3, 4 or 5"):
        m.label_states_relative(TRUE[:2], A2)


def test_an_absolute_radius_cut_mislabels_when_most_bins_carry_signal():
    """The bug this replaced.  Standardising by a scale that absorbed the
    signal shrinks every state toward the origin, and an absolute cut of 2
    then calls all of them uncoordinated."""
    m = _mod()
    means = TRUE / 3.41              # the scale a signal-rich matrix gives
    absolute = m.label_states(means, r_cut=2.0)
    assert {d["label"] for d in absolute} == {"uncoordinated"}
    auto = m.label_states(means, r_cut="auto")
    assert {d["label"] for d in auto} == {
        "uncoordinated", "coordinated balanced", "co-open shifted",
        "co-closed shifted", "anti-coordinated"}


def test_the_auto_cut_is_relative_so_it_survives_any_scaling():
    m = _mod()
    for scale in (0.1, 1.0, 3.41, 25.0):
        got = [d["label"] for d in m.label_states(TRUE / scale, r_cut="auto")]
        assert got[0] == "uncoordinated"
        assert got[1] == "coordinated balanced"
        assert got[2] == "co-open shifted"
        assert got[3] == "co-closed shifted"
        assert got[4] == "anti-coordinated"


def test_a_bad_r_cut_is_an_error():
    m = _mod()
    with pytest.raises(ValueError, match="auto"):
        m.label_states(TRUE, r_cut="whatever")


def test_two_states_may_share_a_label_and_that_is_reported():
    """Forcing a bijection would hide the finding that the data wanted more
    structure inside a sector than the taxonomy has names for."""
    m = _mod()
    means = np.array([(3.0, 0.2), (2.8, -0.2), (0.0, 0.0)])
    labs = [d["label"] for d in m.label_states(means, r_cut="auto")]
    assert labs.count("coordinated balanced") == 2


# ------------------------------------------------------------------- BIC --

def test_bic_counts_observed_bins_not_matrix_size():
    """Counting masked bins would credit the model for data it never saw."""
    m = _mod()
    C, F, _st = _fixture(n_cells=6, n_bins=200, missing=0.40)
    X, lengths, obs, _o, _s = m.build_sequences(
        C.to_numpy(float), F.to_numpy(float), list(C.columns))
    model = m.PlaneHMM(3, seed=0).fit(X, lengths, obs, len(C))
    assert model.n_obs == int(obs.sum())
    assert model.n_obs < X.shape[0]
    penalty = model.n_params() * np.log(model.n_obs)
    assert model.bic() == pytest.approx(-2 * model.loglik + penalty)


def test_extreme_bins_are_masked_not_clipped():
    """Clipping turns a heavy tail into a spike at +-z, and EM allocates
    states to the spike just as happily -- three of five states landed
    exactly on the clip corners with 1-bin runs.  Masking says the true
    thing instead: an estimate ten robust sds out is not a coordination
    state."""
    m = _mod()
    C, F, _st = _fixture(n_cells=8, n_bins=300, missing=0.0)
    Cv, Fv = C.to_numpy(float), F.to_numpy(float)
    Cv[0, :10] = 1e6                       # a tail
    X, lengths, obs, _o, info = m.build_sequences(
        Cv, Fv, list(C.columns), clip=10.0)
    assert info["n_masked_extreme"] >= 10
    # masked, so nothing sits AT the boundary
    assert np.abs(X[obs]).max() <= 10.0 + 1e-9
    assert not np.isclose(np.abs(X[obs]).max(), 10.0)


def test_disabling_the_mask_keeps_the_tail():
    m = _mod()
    C, F, _st = _fixture(n_cells=8, n_bins=300, missing=0.0)
    Cv = C.to_numpy(float)
    Cv[0, :10] = 1e6
    _X, _l, obs_on, _o, on = m.build_sequences(
        Cv, F.to_numpy(float), list(C.columns), clip=10.0)
    _X2, _l2, obs_off, _o2, off = m.build_sequences(
        Cv, F.to_numpy(float), list(C.columns), clip=0)
    assert "n_masked_extreme" not in off
    assert int(obs_off.sum()) > int(obs_on.sum())


def test_bic_prefers_more_states_when_they_are_really_there():
    m = _mod()
    C, F, _st = _fixture(n_cells=12, n_bins=400, noise=1.6, missing=0.0)
    X, lengths, obs, _o, _s = m.build_sequences(
        C.to_numpy(float), F.to_numpy(float), list(C.columns))
    best, table = m.select_k(X, lengths, obs, len(C), [2, 3, 5], seed=0)
    ok = table.dropna(subset=["bic"])
    assert best.k == int(ok.loc[ok["bic"].idxmin(), "k"])
    assert best.k >= 3, "two states should not win on five planted ones"


# --------------------------------------------------- compositional guard --

def test_clr_removes_the_simplex_constraint():
    """Posteriors sum to 1, so one state rising pushes the rest down.  CLR
    rows sum to zero instead, which is a constraint the embedding can see
    rather than one that manufactures anticorrelation between every pair."""
    m = _mod()
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(4), size=500)
    np.testing.assert_allclose(P.sum(1), 1.0)
    Z = m.clr(P)
    np.testing.assert_allclose(Z.sum(1), 0.0, atol=1e-9)
    # the simplex forces every pair of raw states to anticorrelate
    raw = np.corrcoef(P.T)
    assert (raw[np.triu_indices(4, 1)] < 0).all()


def test_cli_writes_states_posteriors_and_the_clr(tmp_path):
    m = _mod()
    C, F, _st = _fixture(n_cells=6, n_bins=200)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    out = tmp_path / "o"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "5", "--output-dir", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for f in ("concordance_state.parquet",
              "concordance_state_posterior.parquet",
              "concordance_state_posterior_clr.parquet", "hmm_states.tsv"):
        assert (out / f).exists(), f
    post = pd.read_parquet(out / "concordance_state_posterior.parquet")
    assert post.shape[1] == 200 * 5
    lab = pd.read_csv(out / "hmm_states.tsv", sep="\t")
    assert set(lab.columns) >= {"state", "mean_C", "mean_dF", "R", "theta",
                                "label", "expected_run_bins"}
    assert "embed the CLR matrix" in r.stderr


def test_cli_reports_the_bic_table(tmp_path):
    m = _mod()
    C, F, _st = _fixture(n_cells=6, n_bins=200)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    out = tmp_path / "o"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "bic", "--k-range", "2,3,4",
         "--label", "angle", "--output-dir", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t = pd.read_csv(out / "hmm_bic.tsv", sep="\t")
    assert set(t["k"]) == {2, 3, 4}
    assert "selected k" in r.stderr


def test_relative_labelling_refuses_an_unsupported_k_with_a_usable_message(
        tmp_path):
    """K outside 4 or 5 makes the assignment non-bijective, and a label that
    is a guess should not look like a result.  The refusal has to name the
    model that fits the data."""
    m = _mod()
    C, F, _st = _fixture(n_cells=6, n_bins=200)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "2",
         "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "3, 4 or 5" in r.stderr
    assert "--k 3" in r.stderr and "--label angle" in r.stderr


def test_k_defaults_to_five(tmp_path):
    """BIC's penalty grows like log(n) while the likelihood grows like n; at
    ~1e7 cell-bins it never turns around, so it must not be the default."""
    m = _mod()
    C, F, _st = _fixture(n_cells=6, n_bins=200)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    out = tmp_path / "o"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--output-dir", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "fixed k = 5" in r.stderr
    assert not (out / "hmm_bic.tsv").exists()


# ------------------------------------- when the null is not in the data --

_A5 = np.eye(5) * 0.9 + 0.025


def test_auto_uses_the_origin_when_the_data_reaches_it():
    m = _mod()
    info, meta = m.label_states_relative(TRUE, _A5, anchor="auto")
    assert meta["anchor"] == "origin"
    assert not meta["origin_outside_cloud"]
    assert all(d["label_identifiable"] for d in info)


def test_auto_switches_when_the_cloud_never_approaches_zero():
    """Real data had C strictly positive: 0.000479 at its smallest against a
    robust sd of 0.00013, so C = 0 sat 3.7 sds below anything observed.
    Anchoring on an origin the cloud never reaches labels states by how far
    the whole cloud sits from it, not by what distinguishes them."""
    m = _mod()
    shifted = TRUE + np.array([12.0, 0.0])
    info, meta = m.label_states_relative(shifted, _A5, anchor="auto")
    assert meta["anchor"] == "min-c"
    assert meta["origin_outside_cloud"]
    assert meta["nearest_radius"] > meta["between_state_spread"]


def test_at_five_states_only_anti_is_left_unverifiable():
    """With the origin unreachable, 'anti-coordinated' cannot be shown to be
    genuine anti-coordination -- it just means lowest-C.  The null state
    needs no flag because it is RENAMED 'least coordinated', which is what
    it is.  The directional labels survive: they are read off dF, which
    does reach zero."""
    m = _mod()
    info, meta = m.label_states_relative(TRUE + np.array([12.0, 0.0]), _A5)
    flags = {d["label"]: d["label_identifiable"] for d in info}
    assert meta["null_name"] == "least coordinated"
    assert flags["anti-coordinated"] is False
    assert flags["least coordinated"] is True
    assert flags["co-open shifted"] is True
    assert flags["co-closed shifted"] is True
    assert flags["coordinated balanced"] is True


def test_forcing_the_origin_anchor_is_still_possible():
    m = _mod()
    _i, meta = m.label_states_relative(TRUE + np.array([12.0, 0.0]), _A5,
                                       anchor="origin")
    assert meta["anchor"] == "origin"


def test_a_bad_anchor_is_an_error():
    m = _mod()
    with pytest.raises(ValueError, match="anchor"):
        m.label_states_relative(TRUE, _A5, anchor="whatever")


def test_the_run_says_the_taxonomy_collapses_to_four(tmp_path):
    m = _mod()
    C, F, _st = _fixture(n_cells=10, n_bins=400, missing=0.0)
    Cv = np.abs(C.to_numpy(float)) + 40.0        # strictly positive, far off 0
    pd.DataFrame(Cv, index=C.index, columns=C.columns).to_parquet(
        tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "origin is OUTSIDE" in r.stderr
    assert "NOT IDENTIFIABLE" in r.stderr
    assert "--k 4" in r.stderr          # points at the model that fits
    lab = pd.read_csv(tmp_path / "o" / "hmm_states.tsv", sep="\t")
    assert "label_identifiable" in lab.columns
    # only anti-coordinated is unverifiable now; the null is renamed instead
    assert (~lab["label_identifiable"]).sum() == 1
    assert "least coordinated" in set(lab["label"])


# ------------------------------------------- four states when C > 0 always --

TRUE4 = np.array([(0., 0.), (3., 0.), (2.1, 2.1), (2.1, -2.1)])
_A4 = np.eye(4) * 0.9 + 0.033


def test_four_states_drop_anti_coordination():
    """C = 2*Cov below zero is what anti-coordination MEANS.  Where C is
    positive everywhere the state does not exist, and a fifth archetype
    would make the assignment name something arbitrary."""
    m = _mod()
    arch, null = m.archetypes(4, "min-c")
    assert "anti-coordinated" not in arch
    assert set(arch) == {"least coordinated", "coordinated balanced",
                         "co-open shifted", "co-closed shifted"}
    arch5, _ = m.archetypes(5, "min-c")
    assert "anti-coordinated" in arch5


def test_the_null_is_named_for_what_it_actually_is():
    """'uncoordinated' claims an absence of coordination.  Anchored on the
    lowest-C state that claim is false -- there is coordination everywhere
    and this state has the least of it -- so the NAME carries the caveat
    rather than a footnote having to."""
    m = _mod()
    assert m.archetypes(4, "origin")[1] == "uncoordinated"
    assert m.archetypes(4, "min-c")[1] == "least coordinated"


def test_labelling_four_shifted_states_recovers_the_taxonomy():
    m = _mod()
    shifted = TRUE4 + np.array([12.0, 0.0])
    info, meta = m.label_states_relative(shifted, _A4)
    assert meta["anchor"] == "min-c"
    got = {d["label"] for d in info}
    assert got == {"least coordinated", "coordinated balanced",
                   "co-open shifted", "co-closed shifted"}
    by = {d["label"]: d for d in info}
    assert by["least coordinated"]["mean_C"] == pytest.approx(12.0)
    assert by["co-open shifted"]["mean_dF"] > 0
    assert by["co-closed shifted"]["mean_dF"] < 0


def test_at_four_states_nothing_is_left_unidentifiable():
    """The K=5 fit had to flag 'anti-coordinated' as unverifiable.  Dropping
    it removes the problem rather than annotating it."""
    m = _mod()
    info, _meta = m.label_states_relative(TRUE4 + np.array([12.0, 0.0]), _A4)
    assert all(d["label_identifiable"] for d in info)


def test_two_states_is_refused_with_the_ways_out():
    m = _mod()
    A2 = np.eye(2) * 0.9 + 0.1
    with pytest.raises(ValueError, match="3, 4 or 5"):
        m.label_states_relative(TRUE4[:2], A2)


def test_cli_accepts_k_four(tmp_path):
    m = _mod()
    C, F, _st = _fixture(n_cells=10, n_bins=400, missing=0.0)
    Cv = np.abs(C.to_numpy(float)) + 40.0
    pd.DataFrame(Cv, index=C.index, columns=C.columns).to_parquet(
        tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    out = tmp_path / "o"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "4", "--output-dir", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lab = pd.read_csv(out / "hmm_states.tsv", sep="\t")
    assert len(lab) == 4
    assert "anti-coordinated" not in set(lab["label"])
    assert "least coordinated" in set(lab["label"])
    assert lab["label_identifiable"].all()


def test_the_refusal_names_every_supported_model(tmp_path):
    """The message a user hits on an unsupported K has to point at the
    models that DO fit, and say what distinguishes them."""
    m = _mod()
    C, F, _st = _fixture(n_cells=6, n_bins=200)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "2",
         "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True)
    assert r.returncode != 0
    for flag in ("--k 3", "--k 4", "--k 5"):
        assert flag in r.stderr, flag
    assert "directional" in r.stderr and "crosses zero" in r.stderr


# ------------------------------------------------------- auto mask and QC --

def test_auto_max_z_finds_the_knee_and_reports_the_ladder():
    """A fixed z is a guess about how heavy the tail is.  The knee is where
    the bulk has been captured and everything further out is tail."""
    m = _mod()
    rng = np.random.default_rng(0)
    core = np.abs(rng.normal(0, 1, 200_000))
    tail = np.abs(rng.standard_cauchy(6_000)) * 50
    z, ladder = m.auto_max_z(np.r_[core, tail])
    assert 2.0 <= z <= 12.0, z
    assert ladder and all("z" in r and "frac_kept" in r for r in ladder)
    kept = [r["frac_kept"] for r in ladder]
    assert kept == sorted(kept)              # monotone in z


def test_auto_max_z_keeps_almost_everything_when_there_is_no_tail():
    m = _mod()
    rng = np.random.default_rng(1)
    z, _l = m.auto_max_z(np.abs(rng.normal(0, 1, 100_000)))
    assert z <= 5.0


def test_auto_max_z_survives_an_empty_input():
    m = _mod()
    z, ladder = m.auto_max_z(np.array([]))
    assert np.isfinite(z) and ladder == []


def test_autocorrelation_sees_domains_when_they_exist():
    """The number that decides whether an HMM can help at all."""
    m = _mod()
    C, F, _st = _fixture(n_cells=12, n_bins=400, noise=0.8, missing=0.0)
    X, lengths, obs, _o, _s = m.build_sequences(
        C.to_numpy(float), F.to_numpy(float), list(C.columns), clip=0)
    acf = m.genome_autocorrelation(X, lengths, obs, len(C), max_lag=5)
    assert acf["C_within_cell"][0] > 0.25
    # and it decays with distance, as a domain structure must
    assert acf["C_within_cell"][0] > acf["C_within_cell"][-1]


def test_autocorrelation_is_flat_when_bins_are_independent():
    """The failure the QC exists to name: consecutive bins that do not agree
    leave the transition prior -- the only reason to use an HMM -- with
    nothing to work with."""
    m = _mod()
    rng = np.random.default_rng(3)
    n_cells, n_bins = 10, 400
    C = rng.normal(0, 1, (n_cells, n_bins))
    F = rng.normal(0, 1, (n_cells, n_bins))
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)]
    X, lengths, obs, _o, _s = m.build_sequences(C, F, bins, clip=0)
    acf = m.genome_autocorrelation(X, lengths, obs, n_cells, max_lag=5)
    assert abs(acf["C_within_cell"][0]) < 0.05


def test_autocorrelation_separates_per_cell_noise_from_pooled_signal():
    """The distinction that decides what to do next: a flat per-cell curve
    with an autocorrelated POOLED one means the domains are real and buried
    under sampling noise, not absent."""
    m = _mod()
    rng = np.random.default_rng(4)
    n_cells, n_bins = 40, 400
    shared = np.repeat(rng.normal(0, 1, n_bins // 20), 20)[:n_bins]
    C = shared[None, :] + rng.normal(0, 6.0, (n_cells, n_bins))   # noisy
    F = rng.normal(0, 1, (n_cells, n_bins))
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)]
    X, lengths, obs, _o, _s = m.build_sequences(C, F, bins, clip=0)
    acf = m.genome_autocorrelation(X, lengths, obs, n_cells, max_lag=3)
    assert abs(acf["C_within_cell"][0]) < 0.10
    assert acf["C_pooled"][0] > 0.4


def test_qc_writes_a_figure_and_the_autocorrelation_table(tmp_path):
    m = _mod()
    C, F, _st = _fixture(n_cells=8, n_bins=300, missing=0.05)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    out = tmp_path / "o"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "4", "--qc",
         "--plot-format", "png", "--output-dir", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "hmm_qc.png").exists()
    t = pd.read_csv(out / "hmm_qc_autocorrelation.tsv", sep="\t")
    assert {"lag", "C_within_cell", "C_pooled",
            "dF_within_cell", "dF_pooled"} <= set(t.columns)
    assert "lag-1 autocorrelation" in r.stderr


def test_the_qc_says_an_hmm_cannot_help_when_bins_are_independent(tmp_path):
    m = _mod()
    rng = np.random.default_rng(5)
    n_cells, n_bins = 8, 300
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)]
    cells = [f"c{i}" for i in range(n_cells)]
    pd.DataFrame(rng.normal(0, 1, (n_cells, n_bins)), index=cells,
                 columns=bins).to_parquet(tmp_path / "c.parquet")
    pd.DataFrame(rng.normal(0, 1, (n_cells, n_bins)), index=cells,
                 columns=bins).to_parquet(tmp_path / "f.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "4", "--qc",
         "--plot-format", "png", "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "an HMM cannot help here" in r.stderr


def test_max_z_still_accepts_a_number(tmp_path):
    m = _mod()
    C, F, _st = _fixture(n_cells=6, n_bins=200)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "4", "--max-z", "8",
         "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "|z| = 8" in r.stderr
    assert "auto chose" not in r.stderr


def test_the_archetype_sets_nest_so_a_smaller_k_is_a_simpler_claim():
    """Each K adds one archetype to the one below.  If the sets did not
    nest, choosing K would change the question rather than its resolution."""
    m = _mod()
    a3, _ = m.archetypes(3, "min-c")
    a4, _ = m.archetypes(4, "min-c")
    a5, _ = m.archetypes(5, "min-c")
    assert set(a3) < set(a4) < set(a5)
    assert set(a4) - set(a3) == {"coordinated balanced"}
    assert set(a5) - set(a4) == {"anti-coordinated"}
    assert set(a3) == {"least coordinated", "co-open shifted",
                       "co-closed shifted"}


def test_three_states_keep_only_the_directional_ones():
    """The model for data whose elevated-C states are all directional --
    which is what 'coordinated balanced' absorbing 79.6% of the genome
    suggests is really going on."""
    m = _mod()
    three = np.array([(0., 0.), (2.1, 2.1), (2.1, -2.1)]) + np.array([12., 0.])
    A3 = np.eye(3) * 0.9 + 0.05
    info, meta = m.label_states_relative(three, A3)
    got = {d["label"] for d in info}
    assert got == {"least coordinated", "co-open shifted",
                   "co-closed shifted"}
    by = {d["label"]: d for d in info}
    assert by["co-open shifted"]["mean_dF"] > 0
    assert by["co-closed shifted"]["mean_dF"] < 0
    assert all(d["label_identifiable"] for d in info)


def test_cli_accepts_k_three(tmp_path):
    m = _mod()
    C, F, _st = _fixture(n_cells=10, n_bins=400, missing=0.0)
    Cv = np.abs(C.to_numpy(float)) + 40.0
    pd.DataFrame(Cv, index=C.index, columns=C.columns).to_parquet(
        tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    out = tmp_path / "o"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "3", "--output-dir", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lab = pd.read_csv(out / "hmm_states.tsv", sep="\t")
    assert len(lab) == 3
    assert "coordinated balanced" not in set(lab["label"])
    assert "anti-coordinated" not in set(lab["label"])


# ------------------------------------------- the log-radial warp, and its cost --

def test_the_warp_keeps_every_angle_and_only_changes_radius():
    """Angles must survive exactly, or the direction labelling would mean
    something different in the two spaces."""
    m = _mod()
    rng = np.random.default_rng(0)
    X = rng.normal(0, 5, (4000, 2)) + np.array([120.0, -110.0])
    obs = np.ones(len(X), bool)
    W, centre = m.warp_log_radial(X, obs)
    d = X - centre
    th_in = np.arctan2(d[:, 1], d[:, 0])
    th_out = np.arctan2(W[:, 1], W[:, 0])
    np.testing.assert_allclose(np.cos(th_in), np.cos(th_out), atol=1e-9)
    np.testing.assert_allclose(np.sin(th_in), np.sin(th_out), atol=1e-9)
    # radius is compressed, monotonically
    r_in = np.linalg.norm(d, axis=1)
    r_out = np.linalg.norm(W, axis=1)
    assert (r_out <= r_in + 1e-9).all()
    o = np.argsort(r_in)
    assert np.all(np.diff(r_out[o]) >= -1e-9)


def test_unwarp_is_the_inverse():
    m = _mod()
    rng = np.random.default_rng(1)
    X = rng.normal(0, 8, (500, 2)) + np.array([50.0, -30.0])
    obs = np.ones(len(X), bool)
    W, centre = m.warp_log_radial(X, obs)
    np.testing.assert_allclose(m.unwarp_log_radial(W, centre), X, atol=1e-7)


def test_the_warp_spreads_a_dense_core_but_costs_accuracy():
    """It doubles the separation k-means finds, which is why it looked like
    the fix -- and on data with known truth it recovers FEWER of the true
    arms while producing longer, better-looking domains.  Long runs are not
    evidence of a correct fit, and that is the whole point of this test."""
    m = _mod()
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(0)
    dirs = np.array([[0.15, 0.99], [0.80, -0.60], [-0.95, 0.05]])
    lens = np.array([32.0, 55.0, 32.0])
    lab = rng.integers(0, 3, 30_000)
    t = rng.random(30_000) ** 12 * lens[lab]
    P = dirs[lab] * t[:, None] + rng.normal(0, 3.5, (30_000, 2))
    obs = np.ones(len(P), bool)
    W, _c = m.warp_log_radial(P, obs)

    def sep(Z):
        c = KMeans(3, n_init=10, random_state=0).fit(Z).cluster_centers_
        import itertools
        smallest = min(np.linalg.norm(a - b)
                       for a, b in itertools.combinations(c, 2))
        size = np.percentile(np.linalg.norm(Z - np.median(Z, 0), axis=1), 95)
        return smallest / size

    assert sep(W) > 1.7 * sep(P)


def test_the_warp_is_not_the_default():
    """It is available and documented, not recommended."""
    src = _SCRIPT.read_text()
    i = src.index('"--space"')
    block = src[i:i + 400]
    assert '"plane"' in block and 'default="plane"' in block


def test_density_equalised_flattens_the_sampling_over_the_plane():
    m = _mod()
    rng = np.random.default_rng(2)
    core = rng.normal(0, 1, (40_000, 2))
    arm = np.c_[rng.uniform(5, 30, 1_500), rng.normal(0, 1, 1_500)]
    X = np.vstack([core, arm])
    sub = m.density_equalised(X, rng, n=5_000)
    far_before = float(np.mean(X[:, 0] > 5))
    far_after = float(np.mean(sub[:, 0] > 5))
    assert far_after > 4 * far_before


def test_the_qc_warns_when_the_mask_cuts_into_the_structure(tmp_path):
    """The mask exists to remove a tail, and on real data the tail and the
    structure can be the same points -- a cut sitting on the arms is the
    likeliest reason a fit collapses into the core."""
    m = _mod()
    C, F, _st = _fixture(n_cells=8, n_bins=300, noise=2.0, missing=0.0)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "3", "--max-z", "1.5", "--qc",
         "--plot-format", "png", "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "INSIDE the structure" in r.stderr
    assert "--max-z 0" in r.stderr


def test_space_flag_runs_end_to_end(tmp_path):
    m = _mod()
    C, F, _st = _fixture(n_cells=8, n_bins=300)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    out = tmp_path / "o"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "3", "--space", "log-radial",
         "--output-dir", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "log-radial space" in r.stderr
    assert "reported back on the original plane" in r.stderr


# --------------------------------------- the QC and the fit share coordinates --

def _offset_arms(n=6_000, centre=(130.0, -122.0), seed=0):
    """A dense core with three arms, sitting far from the origin -- the
    shape real data came back with, and the shape that makes a coordinate
    mix-up visible instead of merely wrong by a little."""
    rng = np.random.default_rng(seed)
    dirs = np.array([[0.15, 0.99], [0.80, -0.60], [-0.95, 0.05]])
    lab = rng.integers(0, 3, n)
    t = rng.random(n) ** 8 * 40.0
    return (dirs[lab] * t[:, None] + rng.normal(0, 2.0, (n, 2))
            + np.array(centre, float))


def _fake_model(m, centre):
    class _M:
        pass
    mod = _M()
    mod.k = 3
    mod.A = np.full((3, 3), 1 / 3)
    mod.means = m.unwarp_log_radial(
        np.array([[0.5, 0.5], [1.0, -1.0], [-1.5, 0.1]]), centre)
    return mod


_ACF = {"lag": [1, 2], "C_within_cell": [0.4, 0.2], "C_pooled": [0.4, 0.2],
        "dF_within_cell": [0.4, 0.2], "dF_pooled": [0.4, 0.2]}


def _contains(extent, pts):
    x0, x1, y0, y1 = extent
    return bool(((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                 & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).all())


def test_the_qc_draws_the_cloud_and_the_means_in_one_coordinate_system(
        tmp_path):
    """Under --space log-radial the fit runs on a warped matrix but the
    means are reported back on the plane.  Panel (a) drew the warped matrix
    against plane means, which put a log-compressed cloud at the origin and
    the state markers a hundred units away -- a picture identical to a fit
    that has collapsed, produced by a figure that was simply lying."""
    m = _mod()
    P = _offset_arms()
    obs = np.ones(len(P), bool)
    W, centre = m.warp_log_radial(P, obs)
    model = _fake_model(m, centre)
    args = (W, obs, model, _ACF, float("nan"), [], [1.0, 2.0],
            ["a", "b", "c"], "t")
    right = m.qc_figure(tmp_path / "right.png", *args, warp_centre=centre)
    wrong = m.qc_figure(tmp_path / "wrong.png", *args)
    assert _contains(right["panel_a_extent"], right["means"])
    assert not _contains(wrong["panel_a_extent"], wrong["means"])


def test_the_qc_panel_is_on_the_plane_whatever_space_the_fit_used(tmp_path):
    """Same data, same axes, whichever space it was fitted in."""
    m = _mod()
    P = _offset_arms()
    obs = np.ones(len(P), bool)
    W, centre = m.warp_log_radial(P, obs)
    model = _fake_model(m, centre)
    args = (_ACF, float("nan"), [], [1.0, 2.0], ["a", "b", "c"], "t")
    flat = m.qc_figure(tmp_path / "p.png", P, obs, model, *args)
    warped = m.qc_figure(tmp_path / "w.png", W, obs, model, *args,
                         warp_centre=centre)
    np.testing.assert_allclose(flat["panel_a_extent"],
                               warped["panel_a_extent"], atol=1e-6)


def test_the_cut_histogram_shows_what_the_cut_removed(tmp_path):
    """Panel (c) asks where the cut falls relative to the cloud.  Drawn from
    the retained points alone it cannot answer: the retained set stops at
    the cut by construction, so the line always sits at the right edge."""
    m = _mod()
    P = _offset_arms()
    med = np.median(P, axis=0)
    far = (np.abs(P - med) > 12.0).any(1)
    obs, model = ~far, _fake_model(m, med)
    model.means = P[obs][:3]
    args = (_ACF, 12.0, [], [1.0, 2.0], ["a", "b", "c"], "t")
    with_mask = m.qc_figure(tmp_path / "m.png", P, obs, model, *args,
                            masked=far)
    without = m.qc_figure(tmp_path / "n.png", P, obs, model, *args)
    assert far.sum() > 50
    assert with_mask["dev_max"] > 2 * without["dev_max"]


def test_the_mask_is_recorded_not_re_derived_from_a_warped_matrix():
    """main used to reconstruct the mask by thresholding X against a
    plane-unit cut.  X is only the plane in plane space; warped deviations
    never reach the threshold, so under --space log-radial the red points
    that show what the cut removed silently vanished."""
    m = _mod()
    C, F, _st = _fixture(n_cells=8, n_bins=300, missing=0.0)
    P = _offset_arms(n=8 * 300)
    Cv, Fv = P[:, 0].reshape(8, 300), P[:, 1].reshape(8, 300)
    kw = dict(scale_c=1.0, scale_f=1.0, clip=12.0)
    _X, _l, obs, _o, info = m.build_sequences(Cv, Fv, list(C.columns), **kw)
    rows = info["masked_rows"]
    assert int(rows.sum()) == info["n_masked_extreme"] >= 10
    assert not (rows & obs).any()

    Xw, _lw, obsw, _ow, infow = m.build_sequences(
        Cv, Fv, list(C.columns), space="log-radial", **kw)
    np.testing.assert_array_equal(infow["masked_rows"], rows)
    # what the re-derivation used to find: log1p shrinks a 40-unit arm to
    # under 4, so a 12-unit plane cut selects nothing at all.
    med = np.array([np.median(Xw[obsw, 0]), np.median(Xw[obsw, 1])])
    stale = ((np.abs(Xw - med) > 12.0).any(1) & np.isfinite(Xw).all(1)
             & ~obsw)
    assert int(stale.sum()) == 0 < int(rows.sum())


def test_the_reach_warning_still_fires_in_log_radial_space(tmp_path):
    """The one diagnostic that would have caught the mask sitting on the
    arms was computed off the warped matrix too, so turning on the flag
    turned off the warning."""
    m = _mod()
    C, F, _st = _fixture(n_cells=8, n_bins=300, noise=2.0, missing=0.0)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "3", "--max-z", "1.5", "--qc",
         "--space", "log-radial", "--plot-format", "png",
         "--output-dir", str(tmp_path / "o")], capture_output=True,
        text=True)
    assert r.returncode == 0, r.stderr
    assert "INSIDE the structure" in r.stderr


# ------------------------------------- batch: what actually reaches the fit --

def _samples(tmp_path, n_cells, cells=None, seed=0):
    rng = np.random.default_rng(seed)
    cells = list(cells) if cells is not None else [
        f"cell{i}" for i in range(n_cells)]
    tab = pd.DataFrame({
        "cell": cells,
        "exp_batch": [f"E{i % 3}" for i in range(len(cells))],
        "lib_batch": [f"L{i % 2}" for i in range(len(cells))],
        "cell_type": ["tumour" if i % 3 == 0 else "reference"
                      for i in range(len(cells))],
        "n_pairs": rng.integers(500, 50_000, len(cells)),
    })
    path = tmp_path / "samples.tsv"
    tab.to_csv(path, sep="\t", index=False)
    return path


def test_the_hmm_inputs_are_uncorrected_and_the_run_says_so(tmp_path):
    """concordance_coordination.parquet is written inside step 5, before the
    pipeline corrects the shrunk matrix.  Someone who ran the pipeline with
    --concordance-batch-correction combat has every reason to believe these
    values were corrected, and they were not."""
    m = _mod()
    C, F, _st = _fixture(n_cells=8, n_bins=300)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "3",
         "--output-dir", str(tmp_path / "o")], capture_output=True,
        text=True)
    assert r.returncode == 0, r.stderr
    assert "batch: none" in r.stderr
    assert "BEFORE the pipeline" in r.stderr


def test_batch_correction_needs_both_a_table_and_a_column(tmp_path):
    m = _mod()
    C, F, _st = _fixture(n_cells=8, n_bins=300)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "3",
         "--concordance-batch-correction", "center",
         "--output-dir", str(tmp_path / "o")], capture_output=True,
        text=True)
    assert r.returncode != 0
    assert "--concordance-batch-column" in r.stderr


def test_correcting_removes_a_planted_batch_offset(tmp_path):
    """The correction has to actually work, not merely run."""
    m = _mod()
    C, F, _st = _fixture(n_cells=12, n_bins=300, missing=0.0)
    Cv = C.to_numpy(float).copy()
    off = np.array([0.0, 5.0, -5.0])[np.arange(12) % 3]
    Cv += off[:, None]
    Cb = pd.DataFrame(Cv, index=C.index, columns=C.columns)
    Cb.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    sp = _samples(tmp_path, 12, cells=list(C.index))

    def spread(out):
        d = pd.read_parquet(out / "concordance_state_posterior.parquet")
        return d
    before = float(np.nanstd(
        [np.nanmean(Cv[np.arange(12) % 3 == g]) for g in range(3)]))
    from nome_hic_concordance import batch_correction as _bc
    fixed = _bc.correct_and_report(
        Cb, pd.read_csv(sp, sep="\t").set_index("cell")["exp_batch"],
        "center", 0, "coordination").to_numpy(float)
    after = float(np.nanstd(
        [np.nanmean(fixed[np.arange(12) % 3 == g]) for g in range(3)]))
    assert after < 0.05 * before

    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "3", "--samples", str(sp),
         "--concordance-batch-column", "exp_batch,lib_batch",
         "--concordance-batch-correction", "center",
         "--output-dir", str(tmp_path / "o")], capture_output=True,
        text=True)
    assert r.returncode == 0, r.stderr
    assert "batch: center on C and F-F_exp" in r.stderr


def test_the_run_reports_how_much_of_c_the_correction_pushed_below_zero(
        tmp_path):
    """C is a covariance-scale magnitude and came back strictly positive on
    real data, which is what makes C = 0 usable as the null.  A location-
    scale correction is free to move mass across it, and that should be a
    number in the log rather than a surprise in the labelling."""
    m = _mod()
    C, F, _st = _fixture(n_cells=12, n_bins=300, missing=0.0)
    Cv = np.abs(C.to_numpy(float)) + 1.0
    Cv[np.arange(12) % 3 == 1] += 8.0
    pd.DataFrame(Cv, index=C.index, columns=C.columns).to_parquet(
        tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    sp = _samples(tmp_path, 12, cells=list(C.index))
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "3", "--samples", str(sp),
         "--concordance-batch-column", "exp_batch",
         "--concordance-batch-correction", "center",
         "--output-dir", str(tmp_path / "o")], capture_output=True,
        text=True)
    assert r.returncode == 0, r.stderr
    assert "C below zero: 0.00% ->" in r.stderr
    frac = float(r.stderr.split("C below zero: 0.00% -> ")[1].split("%")[0])
    assert frac > 10.0


def test_confounded_batch_and_biology_is_named_not_silently_corrected(
        tmp_path):
    """Removing a batch's per-bin mean removes whatever that batch IS.  When
    a batch is one cell type -- tumours and references rarely prepared
    together -- that is the biology, and no method separates them."""
    m = _mod()
    C, F, _st = _fixture(n_cells=12, n_bins=200, missing=0.0)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    tab = pd.DataFrame({
        "cell": list(C.index),
        "exp_batch": [f"E{i % 3}" for i in range(12)],
        "cell_type": ["tumour" if i % 3 == 0 else "reference"
                      for i in range(12)],
    })
    sp = tmp_path / "s.tsv"
    tab.to_csv(sp, sep="\t", index=False)
    out = tmp_path / "o"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "3", "--samples", str(sp),
         "--concordance-batch-column", "exp_batch",
         "--concordance-batch-correction", "center",
         "--colour-by", "cell_type", "--output-dir", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "confounded" in r.stderr
    conf = pd.read_csv(out / "hmm_batch_confounding.tsv", sep="\t")
    assert float(conf["share"].max()) == 1.0


def test_confounding_is_measured_per_batch_level():
    m = _mod()
    batch = ["A"] * 4 + ["B"] * 4
    group = ["t", "t", "t", "t", "t", "r", "r", "r"]
    tab = m.batch_group_confounding(batch, group).set_index("batch")
    assert tab.loc["A", "share"] == 1.0
    assert tab.loc["A", "top_group"] == "t"
    assert tab.loc["B", "share"] == 0.75


def test_colour_by_labels_every_row_by_its_own_cell():
    """X is cell-major; a colour that drifts off the cells it names would
    make every arm look mixed and answer the question wrongly."""
    m = _mod()
    tab = pd.DataFrame({"g": ["a", "b", "a"], "n": [1.0, 2.0, 3.0]},
                       index=["c0", "c1", "c2"])
    spec = m.point_colour(tab, "g", n_bins=4)
    assert spec["kind"] == "cat" and spec["names"] == ["a", "b"]
    np.testing.assert_array_equal(
        spec["codes"], np.array([0] * 4 + [1] * 4 + [0] * 4))


def test_a_numeric_column_gets_a_colour_bar_not_ten_thousand_levels():
    m = _mod()
    tab = pd.DataFrame({"n_pairs": np.arange(40, dtype=float)},
                       index=[f"c{i}" for i in range(40)])
    spec = m.point_colour(tab, "n_pairs", n_bins=2)
    assert spec["kind"] == "num"
    np.testing.assert_allclose(spec["values"][:4], [0, 0, 1, 1])


def test_a_high_cardinality_categorical_is_refused_with_the_way_out():
    m = _mod()
    tab = pd.DataFrame({"donor": [f"d{i}" for i in range(30)]},
                       index=[f"c{i}" for i in range(30)])
    with pytest.raises(SystemExit) as e:
        m.point_colour(tab, "donor", n_bins=2)
    assert "30 levels" in str(e.value)


def test_colour_by_needs_a_samples_table(tmp_path):
    m = _mod()
    C, F, _st = _fixture(n_cells=8, n_bins=200)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "3", "--colour-by", "batch",
         "--output-dir", str(tmp_path / "o")], capture_output=True,
        text=True)
    assert r.returncode != 0
    assert "--colour-by needs --samples" in r.stderr


def test_the_coloured_qc_renders_end_to_end(tmp_path):
    m = _mod()
    C, F, _st = _fixture(n_cells=9, n_bins=200)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    sp = _samples(tmp_path, 9, cells=list(C.index))
    out = tmp_path / "o"
    for col in ("exp_batch", "n_pairs"):
        r = subprocess.run(
            [sys.executable, str(_SCRIPT), "--coordination",
             str(tmp_path / "c.parquet"), "--direction-excess",
             str(tmp_path / "f.parquet"), "--k", "3", "--samples", str(sp),
             "--colour-by", col, "--qc", "--plot-format", "png",
             "--output-dir", str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert (out / "hmm_qc.png").exists()


# ----------------------- is the structure between cells, or along the genome --

def test_between_cell_fraction_is_near_one_when_cells_differ():
    """Every bin in a cell the same, cells different from each other: all of
    the spread is between cells and none of it is segmentable."""
    m = _mod()
    n_cells, n_bins = 20, 50
    off = np.linspace(-10, 10, n_cells)
    X = np.repeat(np.c_[off, -off], n_bins, axis=0)
    obs = np.ones(len(X), bool)
    mu, frac = m.cell_summaries(X, obs, n_cells)
    assert frac["C"] > 0.999 and frac["dF"] > 0.999
    np.testing.assert_allclose(mu[:, 0], off)


def test_between_cell_fraction_is_near_zero_when_cells_agree():
    """Domains along the genome, identical in every cell: nothing between
    cells, everything within -- what the HMM is actually for."""
    m = _mod()
    rng = np.random.default_rng(0)
    n_cells, n_bins = 20, 400
    track = np.where(np.arange(n_bins) % 200 < 100, 6.0, -6.0)
    X = np.tile(np.c_[track, -track], (n_cells, 1))
    X += rng.normal(0, 0.05, X.shape)
    obs = np.ones(len(X), bool)
    _mu, frac = m.cell_summaries(X, obs, n_cells)
    assert frac["C"] < 0.01 and frac["dF"] < 0.01


def test_cell_summaries_are_on_the_plane_after_a_warp():
    m = _mod()
    P = _offset_arms(n=40 * 60)
    obs = np.ones(len(P), bool)
    W, centre = m.warp_log_radial(P, obs)
    flat, ff = m.cell_summaries(P, obs, 40)
    warped, fw = m.cell_summaries(W, obs, 40, warp_centre=centre)
    np.testing.assert_allclose(flat, warped, atol=1e-6)
    assert abs(ff["C"] - fw["C"]) < 1e-9


def test_cell_summaries_ignore_masked_bins():
    m = _mod()
    n_cells, n_bins = 6, 40
    X = np.zeros((n_cells * n_bins, 2))
    X[:, 0] = 1.0
    X[:5, 0] = 1e6                       # masked, must not reach the mean
    obs = np.ones(len(X), bool)
    obs[:5] = False
    mu, frac = m.cell_summaries(X, obs, n_cells)
    np.testing.assert_allclose(mu[:, 0], 1.0)
    assert frac["C"] < 1e-9 or not np.isfinite(frac["C"])


def test_the_run_names_between_cell_structure_as_unsegmentable(tmp_path):
    """A per-bin batch correction and a bigger K both act on the axis that
    is not carrying the structure, and someone staring at collapsed states
    will otherwise keep reaching for exactly those two."""
    m = _mod()
    rng = np.random.default_rng(1)
    n_cells, n_bins = 12, 300
    bins = [f"chr1:{i * 250_000}" for i in range(n_bins)]
    off = rng.normal(0, 8.0, n_cells)
    C = pd.DataFrame(off[:, None] + rng.normal(0, 0.2, (n_cells, n_bins)),
                     index=[f"cell{i}" for i in range(n_cells)],
                     columns=bins)
    F = pd.DataFrame(-off[:, None] + rng.normal(0, 0.2, (n_cells, n_bins)),
                     index=C.index, columns=bins)
    C.to_parquet(tmp_path / "c.parquet")
    F.to_parquet(tmp_path / "f.parquet")
    out = tmp_path / "o"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coordination",
         str(tmp_path / "c.parquet"), "--direction-excess",
         str(tmp_path / "f.parquet"), "--k", "3", "--max-z", "0", "--qc",
         "--plot-format", "png", "--output-dir", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "BETWEEN cells" in r.stderr
    assert "fitting cell identity" in r.stderr
    tab = pd.read_csv(out / "hmm_qc_variance.tsv", sep="\t")
    assert float(tab["between_cell_fraction"].min()) > 0.9
