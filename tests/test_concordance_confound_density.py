"""Per-bin confound-density script.

Pins the statistics (per-column correlation, OLS residualization) and the one
property that determines how the figure must be read: under
``--residual-mode both`` the residual correlations are zero BY CONSTRUCTION,
whereas under ``other`` they are not.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "concordance_confound_density.py")


def _mod():
    spec = importlib.util.spec_from_file_location("ccd", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ccd"] = m
    spec.loader.exec_module(m)
    return m


def _frame(X, cells, bins):
    return pd.DataFrame(X, index=cells, columns=bins)


def _synthetic(n_cells=60, n_bins=25, seed=0):
    """C tracks 1D, D tracks 3D, both with independent noise."""
    rng = np.random.default_rng(seed)
    cells = [f"c{i}" for i in range(n_cells)]
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)]
    b1 = rng.normal(0, 1, (n_cells, n_bins))
    b3 = rng.normal(0, 1, (n_cells, n_bins))
    C = 0.8 * b1 + 0.3 * b3 + rng.normal(0, 0.5, (n_cells, n_bins))
    D = 0.7 * b3 + rng.normal(0, 0.5, (n_cells, n_bins))
    return (_frame(C, cells, bins), _frame(D, cells, bins),
            _frame(b1, cells, bins), _frame(b3, cells, bins))


# ------------------------------------------------------------- pure stats --

def test_per_column_corr_recovers_a_planted_correlation():
    m = _mod()
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (200, 4))
    Y = X * 1.0                                     # r = +1 exactly
    assert m.per_column_corr(Y, X) == pytest.approx(np.ones(4))
    assert m.per_column_corr(-Y, X) == pytest.approx(-np.ones(4))


def test_per_column_corr_nan_for_thin_or_constant_columns():
    m = _mod()
    Y = np.random.default_rng(2).normal(0, 1, (30, 3))
    X = Y.copy()
    X[:, 1] = 5.0                                   # constant -> undefined
    Y[5:, 2] = np.nan                               # only 5 obs -> below min
    out = m.per_column_corr(Y, X, min_obs=20)
    assert np.isfinite(out[0])
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_residual_is_orthogonal_to_its_own_regressors():
    """The property that makes --residual-mode both a QC, not a result."""
    m = _mod()
    C, D, b1, b3 = _synthetic()
    Cv, b1v, b3v = C.to_numpy(), b1.to_numpy(), b3.to_numpy()
    res = m.residualize(Cv, [b1v, b3v])
    for base in (b1v, b3v):
        r = m.per_column_corr(res, base)
        assert np.nanmax(np.abs(r)) < 1e-8, "OLS residual must be orthogonal"


def test_residualizing_on_the_other_baseline_stays_informative():
    m = _mod()
    C, D, b1, b3 = _synthetic()
    Cv, b1v, b3v = C.to_numpy(), b1.to_numpy(), b3.to_numpy()
    raw = np.nanmedian(np.abs(m.per_column_corr(Cv, b1v)))
    # C was built to depend on 1D, so removing only 3D must NOT remove it.
    other = np.nanmedian(np.abs(
        m.per_column_corr(m.residualize(Cv, [b3v]), b1v)))
    assert raw > 0.5
    assert other > 0.5, "regressing out 3D should leave the 1D dependence"


def test_residualize_skips_columns_with_too_few_observations():
    """Thin columns must come back NaN, not silently un-residualized."""
    m = _mod()
    C, _, b1, b3 = _synthetic()
    Cv, b1v, b3v = C.to_numpy(), b1.to_numpy(), b3.to_numpy()
    Cv[6:, 0] = np.nan                              # 6 obs, below min_obs
    res = m.residualize(Cv, [b1v, b3v], min_obs=20)
    assert np.all(np.isnan(res[:, 0]))
    assert np.isfinite(res[:, 1]).any()


def test_align_intersects_and_rejects_disjoint_inputs():
    m = _mod()
    C, D, b1, b3 = _synthetic(n_cells=10, n_bins=6)
    out = m._align({"C": C, "D": D, "1D": b1.iloc[:8], "3D": b3})
    assert all(v.shape == (8, 6) for v in out.values())
    with pytest.raises(SystemExit, match="No overlap"):
        m._align({"C": C, "3D": b3.rename(columns=lambda c: c + "_x")})


# ------------------------------------------------------------- end to end --

def _run(tmp_path, extra=None):
    m = _mod()
    C, D, b1, b3 = _synthetic()
    paths = {}
    for name, df in (("c", C), ("d", D), ("b1", b1), ("b3", b3)):
        p = tmp_path / f"{name}.parquet"
        df.to_parquet(p)
        paths[name] = str(p)
    argv = ["--concordance-c", paths["c"], "--concordance-d", paths["d"],
            "--baseline-1d", paths["b1"], "--baseline-3d", paths["b3"],
            "--context", "GCH", "--output-dir", str(tmp_path / "out"),
            "--plot-format", "png"] + (extra or [])
    assert m.main(argv) == 0
    return tmp_path / "out"


def test_end_to_end_emits_figure_and_table(tmp_path):
    out = _run(tmp_path)
    assert (out / "GCH_confound_density.png").exists()
    tsv = pd.read_csv(out / "GCH_confound_density.tsv", sep="\t")
    assert len(tsv) == 25                                   # one row per bin
    raw_c = [c for c in tsv.columns if c.startswith("C  vs")][0]
    res_c = [c for c in tsv.columns if c.startswith("C residual")][0]
    # Planted: C tracks 1D strongly; the both-baseline residual must not.
    assert tsv[raw_c].abs().median() > 0.5
    assert tsv[res_c].abs().max() < 1e-8


def test_residual_mode_other_changes_the_residual_curves(tmp_path):
    out = _run(tmp_path, ["--residual-mode", "other", "--stem", "oth"])
    tsv = pd.read_csv(out / "oth.tsv", sep="\t")
    res_c = [c for c in tsv.columns if c.startswith("C residual")][0]
    assert tsv[res_c].abs().median() > 0.4, "must NOT collapse to zero"


def test_axis_cell_switches_the_unit(tmp_path):
    out = _run(tmp_path, ["--axis", "cell", "--stem", "bycell"])
    tsv = pd.read_csv(out / "bycell.tsv", sep="\t")
    assert len(tsv) == 60 and tsv.columns[0] == "cell"


def test_spearman_runs_and_stays_bounded(tmp_path):
    out = _run(tmp_path, ["--method", "spearman", "--stem", "sp"])
    tsv = pd.read_csv(out / "sp.tsv", sep="\t")
    num = tsv.select_dtypes("number")
    assert num.abs().max().max() <= 1.0 + 1e-9
