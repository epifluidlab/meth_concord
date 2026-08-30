"""Measuring power and FPR on the dataset instead of on a fixture.

Every rate quoted for these tests came from a simulation whose depth,
sparsity and dispersion someone chose.  A real run has its own, and they
are what decide whether a p-value means anything.  This script measures
both on the counts at hand: false positives by relabelling which units are
the contrast arm, power by tilting a known effect into the real counts.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / \
    "contrast_evaluation.py"


def _mod():
    spec = importlib.util.spec_from_file_location("_ce", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _dataset(tmp_path, seed=0, n_bins=600, real=150, cov=0.20, ppc=4.0,
             effect=0.8, unit_sd=0.30):
    """Their cell-type split, a realistic sparsity, and real unit-level
    dispersion, with a known set of differential bins."""
    rng = np.random.default_rng(seed)
    ct = np.array(["tumor1"] * 40 + ["tumor2"] * 40 + ["GM12878"] * 50
                  + ["IMR90"] * 50 + ["monocyte"] * 50)
    n = len(ct)
    units = list(dict.fromkeys(ct))
    ui = np.array([units.index(c) for c in ct])
    tum = np.isin(ct, ["tumor1", "tumor2"])
    N = np.where(rng.random((n, n_bins)) < cov,
                 rng.poisson(ppc, (n, n_bins)), 0).astype(float)
    sd_bin = unit_sd * np.exp(rng.normal(0, .5, n_bins))
    lo = (-0.4 + (rng.normal(0, 1, (len(units), n_bins)) * sd_bin)[ui]
          + rng.normal(0, .25, (n, 1)))
    lo[:, :real] += np.where(tum[:, None], effect, 0.0)
    K = rng.binomial(N.astype(int), 1 / (1 + np.exp(-lo))).astype(float)
    e = 1 / (1 + np.exp(0.4))
    cells = [f"c{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)]
    cd = tmp_path / "counts"
    cd.mkdir(parents=True, exist_ok=True)
    for k, v in {"k_open": K, "k_closed": N - K, "k_mixed": N * .4,
                 "n_linked": N, "k_dm_open": N * e, "k_dm_closed": N * (1 - e),
                 "k_dm_mixed": N * .4, "n_dm": N}.items():
        pd.DataFrame(v, index=cells, columns=bins).to_parquet(
            cd / f"{k}.parquet")
    s = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n,
                  "cell_type": ct}).to_csv(s, sep="\t", index=False)
    return tmp_path, s, real


def test_it_evaluates_the_shipped_code_not_a_copy():
    """Re-implementing the methods here would measure the re-implementation.
    Every method scored is the function celltype_marker_bins.py calls."""
    mb = _mod()._marker_bins()
    for fn in ("glm_contrast", "glm_effect", "contrast_stats", "GLM_MODELS",
               "GLM_COUNT_KEYS", "moderated_t"):
        assert hasattr(mb, fn), fn
    src = _SCRIPT.read_text()
    assert "def glm_contrast" not in src
    assert "def moderated_t" not in src


def test_the_null_relabellings_are_balanced_and_exclude_the_truth():
    """Power and variance both depend on the arm sizes, so a 1-vs-4 split
    would not be the test that was actually run."""
    ce = _mod()
    units = ["tumor1", "tumor2", "GM12878", "IMR90", "monocyte"]
    got = ce.balanced_splits(units, 2, ["tumor1", "tumor2"])
    assert len(got) == 9                      # C(5,2) - 1
    assert all(len(g) == 2 for g in got)
    assert frozenset(["tumor1", "tumor2"]) not in got
    assert len(set(got)) == 9


def test_the_spike_shifts_the_log_odds_by_exactly_delta_and_keeps_depth():
    """Re-drawing the counts would replace the dataset's own over-dispersion
    with binomial noise and overstate power.  Tilting keeps the depth, the
    sparsity and each cell's deviation from its arm."""
    ce = _mod()
    mb = ce._marker_bins()
    rng = np.random.default_rng(0)
    n, nb = 30, 8
    s = rng.gamma(3, 4, (n, nb))
    f = rng.gamma(3, 4, (n, nb))
    counts = {"k_open": s, "k_closed": f, "k_mixed": np.zeros((n, nb)),
              "n_linked": s + f, "k_dm_open": np.ones((n, nb)),
              "k_dm_closed": np.ones((n, nb)),
              "k_dm_mixed": np.zeros((n, nb)), "n_dm": s + f}
    rows = np.arange(10)
    bins = np.array([1, 3, 5])
    out = ce.spike(counts, rows, bins, 0.7, "direction_logor", mb.GLM_MODELS)
    lo0 = np.log(s[np.ix_(rows, bins)] / f[np.ix_(rows, bins)])
    lo1 = np.log(out["k_open"][np.ix_(rows, bins)]
                 / out["k_closed"][np.ix_(rows, bins)])
    np.testing.assert_allclose(lo1 - lo0, 0.7, atol=1e-9)
    # Depth is preserved exactly, per cell and per bin.
    np.testing.assert_allclose(
        out["k_open"][np.ix_(rows, bins)] + out["k_closed"][np.ix_(rows, bins)],
        (s + f)[np.ix_(rows, bins)], rtol=1e-12)
    # Untouched rows and bins are untouched.
    np.testing.assert_array_equal(out["k_open"][10:], s[10:])
    np.testing.assert_array_equal(out["k_open"][:, 0], s[:, 0])


def test_unit_sd_gives_the_planted_effect_between_unit_variation():
    """At 0 every cell gets the same shift, which is the best possible case
    for a test that treats cells as replicates and not what a real tumour
    effect looks like."""
    ce = _mod()
    mb = ce._marker_bins()
    rng = np.random.default_rng(1)
    n, nb = 40, 6
    s = np.full((n, nb), 8.0)
    f = np.full((n, nb), 8.0)
    counts = {"k_open": s, "k_closed": f, "k_mixed": np.zeros((n, nb)),
              "n_linked": s + f, "k_dm_open": np.ones((n, nb)),
              "k_dm_closed": np.ones((n, nb)),
              "k_dm_mixed": np.zeros((n, nb)), "n_dm": s + f}
    uid = np.array(["A"] * 20 + ["B"] * 20)
    rows = np.arange(n)
    bins = np.arange(nb)
    flat = ce.spike(counts, rows, bins, 0.5, "direction_logor",
                    mb.GLM_MODELS, unit_ids=uid, unit_sd=0.0)
    vary = ce.spike(counts, rows, bins, 0.5, "direction_logor",
                    mb.GLM_MODELS, unit_ids=uid, unit_sd=0.4,
                    rng=np.random.default_rng(2))

    def unit_lo(o):
        lo = np.log(o["k_open"] / o["k_closed"])
        return lo[:20].mean(0), lo[20:].mean(0)

    fa, fb = unit_lo(flat)
    va, vb = unit_lo(vary)
    np.testing.assert_allclose(fa, fb, atol=1e-9)      # identical units
    assert float(np.mean(np.abs(va - vb))) > 0.1       # units now differ
    # Cells within a unit still share their unit's shift exactly.
    lo = np.log(vary["k_open"] / vary["k_closed"])
    assert float(np.ptp(lo[:20], axis=0).max()) < 1e-9


def test_calibration_reads_uniform_p_as_nominal():
    ce = _mod()
    rng = np.random.default_rng(3)
    c = ce.calibration(rng.random(20_000))
    assert abs(c["fpr_05"] - 0.05) < 0.01
    assert abs(c["fpr_01"] - 0.01) < 0.006
    assert c["ks"] < 0.02
    hot = ce.calibration(rng.random(20_000) ** 3)      # crowded at zero
    assert hot["fpr_05"] > 0.2 and hot["ks"] > 0.1


def test_an_arm_with_one_unit_cannot_be_evaluated(tmp_path):
    """No variance between units to estimate, so neither the test nor its
    evaluation means anything -- say so rather than returning numbers."""
    d, s, _ = _dataset(tmp_path, n_bins=60, real=10)
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--from-counts", str(d),
         "--samples", str(s), "--cell-type-column", "cell_type",
         "--contrast", "one=tumor1", "--methods", "glm",
         "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "fewer than 2 replication units" in r.stderr


def test_it_separates_a_calibrated_method_from_an_inflated_one(tmp_path):
    """The point of the whole script: on THIS data, which tests may be read
    as p-values at all."""
    d, s, real = _dataset(tmp_path, n_bins=400, real=100)
    out = tmp_path / "eval"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--from-counts", str(d),
         "--samples", str(s), "--cell-type-column", "cell_type",
         "--contrast", "mening=tumor1,tumor2",
         "--methods", "glm", "glm-cell", "--spike", "0.8",
         "--n-spike-bins", "100", "--threads", "2",
         "--output-dir", str(out), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2500:]
    t = pd.read_csv(out / "contrast_evaluation_direction_logor.tsv", sep="\t")
    g = t.set_index("method")
    # 9 relabellings x 400 bins of null p-values behind each rate.
    assert (g["n_permutations"] == 9).all()
    assert g.loc["glm", "n_null_p"] > 2_000
    # The unit-level p is near nominal on this data; the cell-level one is
    # not, and that is measured here rather than asserted from a fixture.
    assert g.loc["glm", "fpr_05"] < 0.10
    assert g.loc["glm-cell", "fpr_05"] > 2 * g.loc["glm", "fpr_05"]
    # Power is scored at each method's OWN threshold for a true 5% rate,
    # so an inflated test cannot buy power by calling more of everything.
    assert 0.0 <= g.loc["glm", "power_calibrated_delta_0.8"] <= 1.0
    assert g.loc["glm-cell", "threshold_for_true_5pct"] < 0.05
    assert (out / "contrast_evaluation_direction_logor.png").exists()
    assert "permuted FPR" in r.stderr and "UPPER bound" in r.stderr
