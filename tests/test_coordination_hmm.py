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
    model = m.PlaneHMM(5, seed=0).fit(X, lengths, obs)
    path, _post = model.decode(X, lengths, obs)

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
    model = m.PlaneHMM(5, seed=0).fit(X, lengths, obs)
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
    model = m.PlaneHMM(4, seed=0).fit(X, lengths, obs)
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
         str(tmp_path / "f.parquet"), "--k", "4",
         "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    st = pd.read_parquet(tmp_path / "o" / "concordance_state.parquet")
    got = st.to_numpy()
    src = C.reindex(columns=[c for c in st.columns]).to_numpy(float)
    assert (got[~np.isfinite(src)] == -1).all()
    assert (got[np.isfinite(src)] >= 0).all()


# ------------------------------------------------------- the labelling cut --

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
    model = m.PlaneHMM(3, seed=0).fit(X, lengths, obs)
    assert model.n_obs == int(obs.sum())
    assert model.n_obs < X.shape[0]
    penalty = model.n_params() * np.log(model.n_obs)
    assert model.bic() == pytest.approx(-2 * model.loglik + penalty)


def test_bic_prefers_more_states_when_they_are_really_there():
    m = _mod()
    C, F, _st = _fixture(n_cells=12, n_bins=400, noise=1.6, missing=0.0)
    X, lengths, obs, _o, _s = m.build_sequences(
        C.to_numpy(float), F.to_numpy(float), list(C.columns))
    best, table = m.select_k(X, lengths, obs, [2, 3, 5], seed=0)
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
         str(tmp_path / "f.parquet"), "--k", "4", "--output-dir", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for f in ("concordance_state.parquet",
              "concordance_state_posterior.parquet",
              "concordance_state_posterior_clr.parquet", "hmm_states.tsv"):
        assert (out / f).exists(), f
    post = pd.read_parquet(out / "concordance_state_posterior.parquet")
    assert post.shape[1] == 200 * 4
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
         "--output-dir", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t = pd.read_csv(out / "hmm_bic.tsv", sep="\t")
    assert set(t["k"]) == {2, 3, 4}
    assert "selected k" in r.stderr
