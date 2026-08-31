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
    assert c["spike_ratio"] > 1_000
    # the nominal z it would put on the axis, which is the symptom
    assert abs(np.percentile(null, 99.9) / c["null_madsd"]) > 1_000


def test_an_exactly_degenerate_middle_is_not_rescued_by_the_fallback():
    """robust_sd falls back to the plain sd when MAD is exactly 0, which
    keeps the arithmetic finite and hides the spike: it hands back the sd
    of the TAILS as though it were the width of the middle.  With more than
    half the mass at one value -- an absence-filled matrix -- that is
    exactly the situation, so the diagnostic reads the raw MAD."""
    m = _mod()
    rng = np.random.default_rng(5)
    null = np.r_[np.zeros(80_000), rng.normal(0, 1.0, 20_000)]
    c = m.calibration(null, null)
    assert not np.isfinite(c["spike_ratio"])
    # and the fallback is what would have hidden it
    assert m.robust_sd(null) > 0.4 and m._mad_sd_raw(null) == 0.0


def test_a_well_behaved_null_is_not_flagged():
    m = _mod()
    rng = np.random.default_rng(1)
    null = rng.normal(0, 1, 200_000)
    c = m.calibration(rng.normal(0, 1.2, 200_000), null)
    assert c["spike_ratio"] == pytest.approx(1.0, abs=0.05)


def test_tail_sd_matches_the_robust_sd_on_a_normal_sample():
    """Same quantity for a Gaussian, so the two are comparable and their
    ratio is a shape statistic rather than a units mismatch."""
    m = _mod()
    rng = np.random.default_rng(2)
    x = rng.normal(0, 3.0, 400_000)
    assert m.tail_sd(x) == pytest.approx(3.0, rel=0.02)
    assert m.robust_sd(x) == pytest.approx(3.0, rel=0.02)


def _spiky_run(tmp_path, n_cells=40, n_bins=500, frac_empty=0.8, seed=0):
    """A sparse context: most (cell, bin) entries measured nothing, so the
    statistic is exactly 0 there on BOTH sides."""
    rng = np.random.default_rng(seed)
    d = tmp_path / "05"
    d.mkdir(parents=True, exist_ok=True)
    cells = [f"c{i}" for i in range(n_cells)]
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)]
    counts = (rng.random((n_cells, n_bins)) > frac_empty).astype(int) * 7
    obs = np.where(counts > 0, rng.normal(0.1, 1.0, counts.shape), 0.0)
    null = np.where(counts > 0, rng.normal(0.0, 1.0, counts.shape), 0.0)
    pd.DataFrame(obs, index=cells, columns=bins).to_parquet(
        d / "concordance_coordination.parquet")
    pd.DataFrame(null, index=cells, columns=bins).to_parquet(
        d / "concordance_coordination_null.parquet")
    pd.DataFrame(counts, index=cells, columns=bins).to_parquet(
        d / "pair_counts_per_bin.parquet")
    return d


def test_structural_zeros_make_the_null_a_spike(tmp_path):
    """The absences and the null draws are the same value, so the scale is
    read off the absences."""
    m = _mod()
    d = _spiky_run(tmp_path)
    obs, null, info = m.load_pair(d, "coordination")
    c = m.calibration(obs, null)
    assert info["frac_kept"] == 1.0
    assert c["spike_ratio"] > 5.0


def test_dropping_them_restores_a_readable_scale(tmp_path):
    """Masked on the RAW pair counts, from both samples together so the
    comparison stays like-for-like."""
    m = _mod()
    d = _spiky_run(tmp_path)
    obs, null, info = m.load_pair(d, "coordination", min_pairs=1)
    c = m.calibration(obs, null)
    assert 0.15 < info["frac_kept"] < 0.25          # the 20% that measured
    assert c["spike_ratio"] == pytest.approx(1.0, abs=0.1)
    assert c["null_madsd"] == pytest.approx(1.0, rel=0.1)
    # and the signal that was buried is now readable
    assert c["null_bias_z"] == pytest.approx(0.0, abs=0.05)


def test_the_mask_is_applied_to_both_samples_not_just_the_observed():
    """Masking one side would compare a filtered observed against an
    unfiltered null, which is a different distribution for a reason that
    has nothing to do with the model."""
    src = _SCRIPT.read_text()
    i = src.index("keep = counts.to_numpy(float)")
    block = src[i:i + 400]
    assert "obs_df = obs_df.where(keep)" in block
    assert "null_df = null_df.where(" in block


def test_min_pairs_without_the_counts_file_says_where_it_lives(tmp_path):
    m = _mod()
    d = _spiky_run(tmp_path)
    (d / "pair_counts_per_bin.parquet").unlink()
    with pytest.raises(SystemExit) as e:
        m.load_pair(d, "coordination", min_pairs=1)
    assert "pair_counts_per_bin.parquet" in str(e.value)


# ---------------------------------------------------------------------------
# The 95% band
# ---------------------------------------------------------------------------


def test_the_band_is_the_exact_order_statistic_interval():
    """The k-th smallest of n Uniform(0,1) draws is exactly Beta(k, n-k+1),
    so the band needs no simulation.  Check it against one."""
    qq = _mod()
    n = 400
    levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    lo, hi = qq.order_statistic_ci(levels, n)
    assert np.all(lo < levels) and np.all(levels < hi)
    assert np.all(np.diff(lo) > 0) and np.all(np.diff(hi) > 0)

    rng = np.random.default_rng(0)
    draws = np.sort(rng.random((4000, n)), axis=1)
    k = np.ceil(levels * n).astype(int)
    emp = draws[:, k - 1]
    cover = np.mean((emp >= lo) & (emp <= hi), axis=0)
    assert np.all(np.abs(cover - 0.95) < 0.02), cover


def test_the_band_narrows_as_the_sample_grows():
    qq = _mod()
    lv = np.array([0.1, 0.5, 0.9])
    w = [np.ptp(np.c_[qq.order_statistic_ci(lv, n)], axis=1).mean()
         for n in (100, 1_000, 10_000)]
    assert w[0] > w[1] > w[2]


def test_the_two_sample_band_brackets_a_null_drawn_from_the_null():
    """If the observed really is the null, its quantiles land inside."""
    qq = _mod()
    rng = np.random.default_rng(1)
    null = rng.normal(0, 1, 200_000)
    obs = rng.normal(0, 1, 5_000)
    qx, qy = qq.qq_points(obs, null)
    lo, hi = qq.qq_band(obs, null)
    assert float(np.mean((qy >= lo) & (qy <= hi))) > 0.9
    # A shifted observed leaves it.
    _, qy2 = qq.qq_points(obs + 0.4, null)
    assert float(np.mean((qy2 >= lo) & (qy2 <= hi))) < 0.5


def test_a_hairline_band_is_now_drawn_rather_than_omitted(tmp_path):
    """This used to be omitted, and the reasoning has moved.

    An i.i.d. band a thousandth of an sd wide would read as 'every departure
    is significant', which the independence assumption cannot support after
    RWR -- so the old code hid it.  Hiding it left the figure with no band at
    all at the real scale.  The fix is to widen the band honestly (resample
    whole cells, ~37x wider) and always draw it, not to suppress the
    symptom."""
    qq = _mod()
    rng = np.random.default_rng(2)
    null = rng.normal(0, 1, 400_000)
    obs = rng.normal(0, 1, 400_000)
    curves = {"r": qq.qq_points(obs, null)}
    tab = pd.DataFrame([{"run": "r", "null_bias_z": 0.0, "lambda": 1.0,
                         "spike_ratio": 1.0}])
    out = tmp_path / "f.png"
    qq.figure(out, tab, curves, "coordination", "t",
              bands={"r": qq.qq_band(obs, null)})
    assert out.exists() and out.stat().st_size > 1_000


def test_the_band_is_drawn_when_it_is_wide_enough_to_see(tmp_path):
    qq = _mod()
    rng = np.random.default_rng(3)
    null = rng.normal(0, 1, 50_000)
    obs = rng.normal(0, 1, 300)
    lo, hi = qq.qq_band(obs, null)
    qx, _ = qq.qq_points(obs, null)
    assert float(np.max(hi - lo)) > 0.02 * float(np.ptp(qx))


# --------------------------------------- the cell-resampled band and style --

def _matrix_run(tmp_path, n_cells=120, n_bins=400, excess=0.0, seed=5):
    """A run whose entries carry WITHIN-CELL correlation, as RWR leaves them."""
    rng = np.random.default_rng(seed)
    d = tmp_path
    d.mkdir(parents=True, exist_ok=True)
    cell = rng.normal(0, 0.8, (n_cells, 1))
    binf = rng.normal(0, 0.4, (1, n_bins))
    null = cell + binf + rng.normal(0, 1.0, (n_cells, n_bins))
    obs = null + excess * rng.standard_t(3, (n_cells, n_bins))
    idx = [f"c{i}" for i in range(n_cells)]
    col = [f"b{j}" for j in range(n_bins)]
    pd.DataFrame(obs, index=idx, columns=col).to_parquet(
        d / "concordance_coordination.parquet")
    pd.DataFrame(null, index=idx, columns=col).to_parquet(
        d / "concordance_coordination_null.parquet")
    return d, null


def test_the_cell_band_is_much_wider_than_the_iid_one():
    """The entire reason the band changed.

    Order statistics assume independent entries.  RWR pools evidence across
    the cell graph, so a cell's bins move together and the effective sample
    size is far below n.  Resampling whole cells has to come out wider -- if
    it ever does not, the band has stopped modelling the dependence and is
    back to declaring every wobble significant.
    """
    qq = _mod()
    rng = np.random.default_rng(11)
    mat = rng.normal(0, 0.8, (150, 1)) + rng.normal(0, 1.0, (150, 500))
    flat = mat.ravel()
    lo_i, hi_i = qq.order_statistic_ci(qq.QQ_LEVELS, flat.size)
    w_iid = np.median(np.quantile(flat, hi_i) - np.quantile(flat, lo_i))
    lo_c, hi_c = qq.qq_band_cells(mat, n_boot=120, seed=0)
    w_cell = float(np.median(hi_c - lo_c))
    assert w_cell > 5 * w_iid, (w_cell, w_iid)


def test_the_cell_band_brackets_the_nulls_own_quantiles():
    """A band that did not contain the thing it was built from would be
    miscalibrated in the direction that hides real departures."""
    qq = _mod()
    rng = np.random.default_rng(12)
    mat = rng.normal(0, 0.7, (200, 1)) + rng.normal(0, 1.0, (200, 600))
    lo, hi = qq.qq_band_cells(mat, n_boot=200, seed=1)
    q = np.quantile(mat.ravel(), qq.QQ_LEVELS)
    inside = float(np.mean((q >= lo) & (q <= hi)))
    assert inside > 0.9, inside
    assert np.all(np.diff(lo) >= -1e-9) and np.all(np.diff(hi) >= -1e-9)


def test_the_cell_band_needs_more_than_a_couple_of_cells():
    qq = _mod()
    assert qq.qq_band_cells(np.zeros((2, 10))) is None
    assert qq.qq_band_cells(np.zeros(10)) is None          # not a matrix


def test_return_matrix_keeps_whole_cells_as_rows(tmp_path):
    """The cap must fall on bins, not on scattered entries.

    Dropping entries at random would break the very within-cell correlation
    the band exists to measure and hand back the i.i.d. width.
    """
    qq = _mod()
    d, null = _matrix_run(tmp_path / "r", n_cells=40, n_bins=300)
    obs, nl, info, mat = qq.load_pair(d, "coordination", return_matrix=True)
    assert mat.shape[0] == 40                     # every cell kept
    assert mat.shape[1] == 300
    obs2, nl2, info2, small = qq.load_pair(d, "coordination", cap=40 * 10,
                                           return_matrix=True)
    assert small.shape[0] == 40                   # rows still intact
    assert small.shape[1] < 300                   # bins are what got dropped


def test_band_choices_all_run_and_none_omits_it(tmp_path):
    qq = _mod()
    d, _ = _matrix_run(tmp_path / "r", n_cells=60, n_bins=200, excess=0.5)
    for kind in ("cell", "iid", "none"):
        out = tmp_path / f"o_{kind}"
        r = subprocess.run(
            [sys.executable, str(_SCRIPT), "--run", f"x={d}",
             "--statistic", "coordination", "--band", kind,
             "--band-boot", "40", "--plot-format", "png",
             "--output-dir", str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        assert (out / "null_qq_coordination.png").exists()


def test_the_pdf_carries_no_type_3_fonts(tmp_path):
    """Type 3 text is not editable in Illustrator and several journals
    reject it outright; pdf.fonttype 42 is what avoids it."""
    import zlib
    import re as _re
    qq = _mod()
    d, _ = _matrix_run(tmp_path / "r", n_cells=40, n_bins=150)
    out = tmp_path / "o"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--run", f"x={d}",
         "--statistic", "coordination", "--band-boot", "30",
         "--plot-format", "pdf", "--output-dir", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    raw = (out / "null_qq_coordination.pdf").read_bytes()
    blobs = [raw]
    for mo in _re.finditer(rb"stream\r?\n", raw):
        s0 = mo.end()
        e0 = raw.find(b"endstream", s0)
        try:
            blobs.append(zlib.decompress(raw[s0:e0]))
        except Exception:
            pass
    joined = b"".join(blobs)
    assert joined.count(b"/Type3") == 0
    assert joined.count(b"/FontFile2") > 0        # embedded TrueType
