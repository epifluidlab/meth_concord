"""The null a Q-Q plot of these statistics is allowed to use.

Every sweep setting changes E and leaves O alone, so they are all claims
about where zero is -- and comparing them on the observed statistic cannot
settle it, because a setting whose E is too small makes C larger everywhere
and larger is indistinguishable from better without a reference.

The reference cannot be parametric.  The counts are DM posterior soft
counts, RWR has already pooled evidence across the cell graph, and the
imputed n is fractional, so N(0, 1) would assert a sampling theory the
pipeline does not obey.  It is drawn instead, and these tests pin what the
drawn null has to be and what may be read off it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "null_calibration_qq.py"


def _mod():
    spec = importlib.util.spec_from_file_location("_qq", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_a_centred_null_scores_zero_bias():
    m = _mod()
    rng = np.random.default_rng(0)
    null = rng.normal(0, 1, 200_000)
    obs = rng.normal(0, 1.6, 200_000)
    c = m.calibration(obs, null)
    assert abs(c["null_bias_z"]) < 0.05
    assert c["lambda"] == pytest.approx(1.6, abs=0.05)


def test_a_displaced_null_is_what_the_number_catches():
    """The prior gap was exactly this: no coordination in, 0.0113 out.  A
    model whose own null does not return zero hands that offset to every
    bin, and no amount of looking at the observed statistic reveals it."""
    m = _mod()
    rng = np.random.default_rng(1)
    null = rng.normal(0, 1, 200_000) + 4.0
    obs = rng.normal(0, 1, 200_000) + 4.0
    c = m.calibration(obs, null)
    assert c["null_bias_z"] == pytest.approx(4.0, abs=0.1)
    # and lambda looks perfect while the model is wrong, which is why the
    # script refuses to read it until the bias is small
    assert c["lambda"] == pytest.approx(1.0, abs=0.05)


def test_signal_shows_up_as_tail_excess_not_as_bias():
    m = _mod()
    rng = np.random.default_rng(2)
    null = rng.normal(0, 1, 200_000)
    obs = np.r_[rng.normal(0, 1, 190_000), rng.normal(6, 1, 10_000)]
    c = m.calibration(obs, null)
    assert abs(c["null_bias_z"]) < 0.05
    assert c["tail_excess"] > 0.03


def test_qq_points_compare_distributions_not_ranks():
    """The two samples need not be the same size; interpolating one onto the
    other's ranks would put the sample-size difference on the axis."""
    m = _mod()
    rng = np.random.default_rng(3)
    x, y = m.qq_points(rng.normal(0, 1, 5_000), rng.normal(0, 1, 500_000))
    assert len(x) == len(y) == 400
    assert np.corrcoef(x, y)[0, 1] > 0.99


def test_a_run_without_the_drawn_null_is_refused_with_the_way_out(tmp_path):
    m = _mod()
    d = tmp_path / "05"
    d.mkdir()
    pd.DataFrame(np.zeros((2, 3))).to_parquet(
        d / "concordance_coordination.parquet")
    with pytest.raises(SystemExit) as e:
        m.load_pair(d, "coordination")
    assert "--emit-null" in str(e.value)


def test_the_pipeline_emits_a_null_that_carries_no_coordination(tmp_path):
    """Drawn from non-contact pairs, so whatever comes back is the model's
    own offset rather than biology."""
    from tests.test_signed_concordance import _focal_active_paths
    from nome_hic_concordance import concordance as cc
    src, out = tmp_path / "src", tmp_path / "o"
    src.mkdir(parents=True, exist_ok=True)
    cc.compute_signed_concordance_matrix(
        _focal_active_paths(src, n_cells=3), bin_size=1000, min_distance=100,
        min_gch_sites=2, baseline="distance-pooled", dm_prior="shared",
        emit_null=True, output_dir=str(out), save_outputs=True,
        use_rwr=False)
    for stat in ("coordination", "direction_logor"):
        assert (out / f"concordance_{stat}_null.parquet").exists()
    real = pd.read_parquet(
        out / "concordance_coordination.parquet").to_numpy(float)
    null = pd.read_parquet(
        out / "concordance_coordination_null.parquet").to_numpy(float)
    assert real.shape == null.shape
    # the null must be closer to zero than the observed, or it is not a null
    assert abs(np.nanmedian(null)) < abs(np.nanmedian(real))


def test_the_null_is_off_by_default():
    """It splits the end pool in half, so the expected is noisier than a
    production run's.  A diagnostic that quietly degraded every run would
    be worse than not having it."""
    from nome_hic_concordance import concordance as cc
    import inspect
    sig = inspect.signature(cc.compute_signed_concordance_matrix)
    assert sig.parameters["emit_null"].default is False


def test_the_null_and_the_expected_share_no_read_end():
    """Sharing them makes the two agree more than chance, and the null comes
    back too narrow -- which reads as a calibrated model."""
    src = (Path(__file__).resolve().parents[1] / "nome_hic_concordance"
           / "concordance.py").read_text()
    i = src.index("if null_split:")
    block = src[i:i + 900]
    assert "permutation" in block
    assert "ep_df.iloc[_a]" in block and "[_b]" in block


def test_cli_runs_end_to_end(tmp_path):
    from tests.test_signed_concordance import _focal_active_paths
    from nome_hic_concordance import concordance as cc
    src, five = tmp_path / "src", tmp_path / "05"
    src.mkdir(parents=True, exist_ok=True)
    cc.compute_signed_concordance_matrix(
        _focal_active_paths(src, n_cells=3), bin_size=1000, min_distance=100,
        min_gch_sites=2, baseline="distance-pooled", dm_prior="shared",
        emit_null=True, output_dir=str(five), save_outputs=True,
        use_rwr=False)
    out = tmp_path / "qq"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--run", f"pooled={five}",
         "--statistic", "coordination", "--plot-format", "png",
         "--output-dir", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "null_qq_coordination.png").exists()
    tab = pd.read_csv(out / "null_calibration.tsv", sep="\t")
    assert "null_bias_z" in tab.columns and len(tab) == 1
    assert "null_bias_z" in r.stderr


def test_panel_a_is_standardised_on_each_run_s_own_null():
    """Under --dm-prior separate the whole distribution is a needle on the
    prior floor -- observed and drawn null both at about -0.19, with a
    spread a small fraction of that.  Plotted raw, an axis wide enough to
    reach zero leaves the data as a sliver and the departure from the
    diagonal, which is the entire question, invisible; and two runs at
    different offsets cannot share a panel at all."""
    m = _mod()
    rng = np.random.default_rng(0)
    needle_null = -0.19 + rng.normal(0, 0.002, 50_000)
    needle_obs = -0.19 + rng.normal(0, 0.0032, 50_000)
    qx, qy = m.qq_points(needle_obs, needle_null)
    nm, ns = float(np.median(qx)), m.robust_sd(qx)
    zx, zy = (qx - nm) / ns, (qy - nm) / ns
    # the standardised spread is order 1, not order 1e-3
    assert np.ptp(zx) > 3.0
    # and the slope recovers lambda, so the diagonal means what it says
    assert np.polyfit(zx, zy, 1)[0] == pytest.approx(1.6, abs=0.15)


def test_the_centring_keeps_the_excess_it_is_meant_to_show():
    """Centred on the NULL, not on the observed: an observed shifted above
    its null must still read as shifted."""
    m = _mod()
    rng = np.random.default_rng(1)
    null = rng.normal(0, 1, 100_000)
    obs = rng.normal(0.8, 1, 100_000)
    qx, qy = m.qq_points(obs, null)
    nm, ns = float(np.median(qx)), m.robust_sd(qx)
    assert float(np.median((qy - nm) / ns)) == pytest.approx(0.8, abs=0.06)


def test_a_spike_shaped_null_is_detected_not_reported_as_a_scale():
    """A sparse context leaves a large share of (cell, bin) entries with no
    reads, so the statistic collapses to one value for all of them.  MAD
    then measures the width of that spike and every entry carrying data
    lands thousands of nominal sds out -- an HCG run came back on a +-4000
    axis for exactly that reason, and nothing about it was large."""
    m = _mod()
    rng = np.random.default_rng(0)
    spike = np.full(80_000, 0.0) + rng.normal(0, 1e-6, 80_000)
    body = rng.normal(0, 1.0, 20_000)
    null = np.r_[spike, body]
    c = m.calibration(rng.normal(0, 1.0, 100_000), null)
    assert c["spike_ratio"] > 5.0
    assert c["null_bodysd"] > 100 * c["null_madsd"]
    # the nominal z it would put on the axis, which is the symptom
    assert abs(np.percentile(null, 99.9) / c["null_madsd"]) > 1_000


def test_a_well_behaved_null_is_not_flagged():
    m = _mod()
    rng = np.random.default_rng(1)
    null = rng.normal(0, 1, 200_000)
    c = m.calibration(rng.normal(0, 1.2, 200_000), null)
    assert c["spike_ratio"] == pytest.approx(1.0, abs=0.05)


def test_body_sd_matches_the_robust_sd_on_a_normal_sample():
    """Same quantity for a Gaussian, so the two are comparable and their
    ratio is a shape statistic rather than a units mismatch."""
    m = _mod()
    rng = np.random.default_rng(2)
    x = rng.normal(0, 3.0, 400_000)
    assert m.body_sd(x) == pytest.approx(3.0, rel=0.02)
    assert m.robust_sd(x) == pytest.approx(3.0, rel=0.02)
