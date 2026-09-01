"""The marginal-correction CLI.

Writes matrices every downstream tool already accepts by path, so the tests
check the CONTRACT: names, shape, index, and that the failure modes say what
to do rather than raising a traceback from deep inside pandas.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "marginal_correct.py")


@pytest.fixture
def run_dir(tmp_path):
    """A minimal Step-5 directory: two statistics, a marginal, a baseline."""
    rng = np.random.default_rng(0)
    n, nb = 80, 120
    cells = [f"c{i}" for i in range(n)]
    cols = [f"chr1:{j}" for j in range(nb)]
    bt = np.array(["A"] * 40 + ["B"] * 40)
    bio = rng.normal(0, 1, (n, nb))
    d = tmp_path / "05_concordance"
    d.mkdir()
    frame = dict(index=cells, columns=cols)
    for stat in ("coordination", "direction_logor"):
        pd.DataFrame(bio + rng.normal(0, 0.3, (n, nb)), **frame).to_parquet(
            d / f"concordance_{stat}.parquet")
    pd.DataFrame(np.clip(0.35 + 0.05 * bio
                         + np.where(bt == "A", 0.04, -0.04)[:, None],
                         0.02, 0.98), **frame).to_parquet(
        d / "concordance_marginal.parquet")
    pd.DataFrame(bio + rng.normal(0, 0.2, (n, nb)), **frame).to_parquet(
        d / "accessibility_mean.parquet")
    sp = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "exp_batch": bt}).to_csv(
        sp, sep="\t", index=False)
    return d, sp, cells, cols


def _run(args, expect_ok=True):
    r = subprocess.run([sys.executable, str(_SCRIPT)] + args,
                       capture_output=True, text=True)
    if expect_ok:
        assert r.returncode == 0, r.stderr[-3000:]
    return r


def test_writes_both_stages_with_the_documented_names(run_dir):
    d, sp, cells, cols = run_dir
    _run([str(d), str(sp), "--batch-col", "exp_batch", "-k", "3"])
    for stat in ("coordination", "direction_logor"):
        stem = f"concordance_{stat}.margreg_k3.method1_local"
        two, three = d / f"{stem}.parquet", d / f"{stem}.resid.parquet"
        assert two.exists() and three.exists()
        for p in (two, three):
            M = pd.read_parquet(p)
            assert list(M.index) == cells
            assert list(M.columns) == cols
            assert np.isfinite(M.to_numpy(float)).all()


def test_k_appears_in_the_name_so_settings_are_not_lost(run_dir):
    d, sp, _, _ = run_dir
    _run([str(d), str(sp), "--batch-col", "exp_batch", "-k", "2"])
    assert (d / "concordance_coordination.margreg_k2.method1_local.parquet"
            ).exists()
    assert not (d / "concordance_coordination.margreg_k3.method1_local.parquet"
                ).exists()


def test_no_residual_skips_stage_three(run_dir):
    d, sp, _, _ = run_dir
    _run([str(d), str(sp), "--batch-col", "exp_batch", "-k", "3",
          "--no-residual"])
    stem = "concordance_coordination.margreg_k3.method1_local"
    assert (d / f"{stem}.parquet").exists()
    assert not (d / f"{stem}.resid.parquet").exists()


def test_the_two_stages_differ(run_dir):
    """Stage 3 removes the baseline-predicted part; if the files were
    identical the residual step silently did nothing."""
    d, sp, _, _ = run_dir
    _run([str(d), str(sp), "--batch-col", "exp_batch", "-k", "3"])
    stem = "concordance_coordination.margreg_k3.method1_local"
    a = pd.read_parquet(d / f"{stem}.parquet").to_numpy(float)
    b = pd.read_parquet(d / f"{stem}.resid.parquet").to_numpy(float)
    assert np.abs(a - b).max() > 1e-6


def test_output_dir_is_respected(run_dir, tmp_path):
    d, sp, _, _ = run_dir
    out = tmp_path / "elsewhere"
    _run([str(d), str(sp), "--batch-col", "exp_batch", "-k", "3",
          "--output-dir", str(out)])
    assert (out / "concordance_coordination.margreg_k3.method1_local.parquet"
            ).exists()
    assert not (d / "concordance_coordination.margreg_k3.method1_local.parquet"
                ).exists()


def test_a_missing_batch_column_says_what_is_available(run_dir):
    d, sp, _, _ = run_dir
    r = _run([str(d), str(sp), "--batch-col", "nope"], expect_ok=False)
    assert r.returncode != 0
    assert "nope" in r.stderr and "exp_batch" in r.stderr


def test_a_missing_marginal_names_the_file(run_dir):
    d, sp, _, _ = run_dir
    (d / "concordance_marginal.parquet").unlink()
    r = _run([str(d), str(sp), "--batch-col", "exp_batch"], expect_ok=False)
    assert r.returncode != 0
    assert "marginal matrix not found" in r.stderr


def test_an_unknown_statistic_is_a_warning_not_a_crash(run_dir):
    d, sp, _, _ = run_dir
    r = _run([str(d), str(sp), "--batch-col", "exp_batch", "-k", "3",
              "--stats", "coordination,not_a_stat"])
    assert "not_a_stat" in r.stderr
    assert (d / "concordance_coordination.margreg_k3.method1_local.parquet"
            ).exists()


def test_no_usable_statistic_fails_loudly(run_dir):
    d, sp, _, _ = run_dir
    r = _run([str(d), str(sp), "--batch-col", "exp_batch",
              "--stats", "not_a_stat"], expect_ok=False)
    assert r.returncode != 0 and "nothing written" in r.stderr


def test_one_replicate_level_warns_rather_than_failing(run_dir, tmp_path):
    """A single-level batch column makes stage 2 a no-op; that must be said,
    not silently produce an unchanged matrix that looks corrected."""
    d, _, cells, _ = run_dir
    sp = tmp_path / "one.tsv"
    pd.DataFrame({"cell_name": cells, "exp_batch": ["only"] * len(cells)}
                 ).to_csv(sp, sep="\t", index=False)
    r = _run([str(d), str(sp), "--batch-col", "exp_batch", "-k", "3"])
    assert "one replicate level" in r.stderr.lower()
