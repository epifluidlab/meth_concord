"""chrM as an enzyme-efficiency covariate.

chrM is not chromatinised, so its GpC methylation is close to a pure readout
of M.CviPI efficiency -- an INDEPENDENT technical covariate, unlike the
pair-derived marginal, which is computed from the same reads as the target.
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
           / "chrm_regress.py")


def _mod():
    spec = importlib.util.spec_from_file_location("chrm", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["chrm"] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------- finding the bins --

@pytest.mark.parametrize("col", [
    "chrM:0-250000", "chrM_0_250000", "chrMT:0-1000", "MT:0-1000",
    "M:0-1000", "chrm:0-250000",
])
def test_chrm_bins_are_found_under_the_naming_conventions_in_use(col):
    m = _mod()
    src = pd.DataFrame({"chr1:0-250000": [1.0, 2.0], col: [0.3, 0.4]})
    v, cols = m.chrm_covariate(src)
    assert cols == [col]
    assert np.allclose(v, [0.3, 0.4])


@pytest.mark.parametrize("col", [
    "chr1:0-250000", "chr10:0-250000", "chr11:250000-500000", "chrX:0-1000",
])
def test_autosomes_are_not_mistaken_for_chrm(col):
    """'M' alone is in the pattern, so a false positive would silently make
    an autosome the enzyme covariate."""
    m = _mod()
    v, cols = m.chrm_covariate(pd.DataFrame({col: [1.0, 2.0]}))
    assert cols == [] and v is None


def test_absent_chrm_returns_none_rather_than_an_empty_mean():
    m = _mod()
    v, cols = m.chrm_covariate(pd.DataFrame({"chr1:0-1": [1.0]}))
    assert v is None and cols == []


def test_several_chrm_bins_are_averaged():
    m = _mod()
    src = pd.DataFrame({"chrM:0-1000": [0.2, 0.4], "chrM:1000-2000": [0.4, 0.8],
                        "chr1:0-1000": [9.0, 9.0]})
    v, cols = m.chrm_covariate(src)
    assert len(cols) == 2 and np.allclose(v, [0.3, 0.6])


def test_a_cell_with_no_chrm_signal_is_nan_not_zero():
    """Zero would read as 'no enzyme activity' rather than 'not measured',
    and would drag that cell's correction the wrong way."""
    m = _mod()
    src = pd.DataFrame({"chrM:0-1000": [0.3, np.nan]})
    v, _ = m.chrm_covariate(src)
    assert v[0] == pytest.approx(0.3) and np.isnan(v[1])


# ---------------------------------------------------------- the regression --

def test_the_covariate_is_actually_removed():
    m = _mod()
    rng = np.random.default_rng(0)
    n, nb = 60, 40
    v = rng.normal(0, 1, n)
    Y = rng.normal(0, 1, (n, nb)) + np.outer(v, rng.normal(0, 2, nb))
    out = m.regress_per_bin(Y, v)
    resid = np.array([abs(np.corrcoef(v, out[:, j])[0, 1])
                      for j in range(nb)])
    assert resid.max() < 1e-8, resid.max()


def test_protection_changes_the_fit():
    m = _mod()
    rng = np.random.default_rng(1)
    n, nb = 60, 40
    v = rng.normal(0, 1, n)
    Y = rng.normal(0, 1, (n, nb))
    prot = rng.normal(0, 1, (n, 5))
    assert not np.allclose(m.regress_per_bin(Y, v),
                           m.regress_per_bin(Y, v, prot))


# ------------------------------------------------------------------- CLI --

@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(2)
    n, nb = 80, 60
    cells = [f"c{i}" for i in range(n)]
    cols = [f"chr1:{j}" for j in range(nb)]
    rp = np.array(["A"] * 40 + ["B"] * 40)
    ct = np.resize(["T1", "T2", "T3"], n)
    bio = rng.normal(0, 1, (n, nb))
    eff = np.where(rp == "A", 0.62, 0.38) + rng.normal(0, 0.03, n)
    src = pd.DataFrame(np.column_stack([bio, eff]), index=cells,
                       columns=cols + ["chrM:0-250000"])
    src.to_parquet(tmp_path / "src.parquet")
    tgt = bio + np.outer(eff - eff.mean(), rng.normal(0, 3, nb))
    pd.DataFrame(tgt, index=cells, columns=cols).to_parquet(
        tmp_path / "tgt.parquet")
    pd.DataFrame(bio + rng.normal(0, .2, (n, nb)), index=cells,
                 columns=cols).to_parquet(tmp_path / "base.parquet")
    sp = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "group": ct, "exp_batch": rp}).to_csv(
        sp, sep="\t", index=False)
    return tmp_path, sp


def _run(args, ok=True):
    r = subprocess.run([sys.executable, str(_SCRIPT)] + args,
                       capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stderr[-2000:]
    return r


def test_cli_reports_the_efficiency_readout_and_writes_output(inputs):
    d, sp = inputs
    out = d / "corrected.parquet"
    r = _run([str(d / "src.parquet"), str(d / "tgt.parquet"), str(sp),
              "--replicate-col", "exp_batch", "--celltype-col", "group",
              "--baseline", str(d / "base.parquet"), "--n-svd", "5",
              "--out", str(out)])
    assert "chrM bins matched: 1" in r.stdout
    assert "enzyme-efficiency readout" in r.stdout
    assert "AUC" in r.stdout
    assert out.exists()
    got = pd.read_parquet(out)
    assert got.shape == pd.read_parquet(d / "tgt.parquet").shape


def test_cli_says_what_to_do_when_chrm_was_excluded(inputs):
    """chrM is routinely dropped by --exclude-chrom; the covariate is then
    simply unavailable, and that has to be said, not raised from pandas."""
    d, sp = inputs
    r = _run([str(d / "base.parquet"), str(d / "tgt.parquet"), str(sp),
              "--replicate-col", "exp_batch", "--celltype-col", "group"],
             ok=False)
    assert r.returncode != 0
    assert "no chrM bins matched" in r.stderr
    assert "rebuild without --exclude-chrom" in r.stderr


def test_cli_lists_available_columns_when_one_is_wrong(inputs):
    d, sp = inputs
    r = _run([str(d / "src.parquet"), str(d / "tgt.parquet"), str(sp),
              "--replicate-col", "nope", "--celltype-col", "group"], ok=False)
    assert "nope" in r.stderr and "exp_batch" in r.stderr
