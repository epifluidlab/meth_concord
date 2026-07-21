"""ICC of per-region values: where cell-type information lives.

The reason this exists: a genome-wide AVERAGE is blind to cell-type structure,
so the concordance-vs-distance curve can be identical across cell types while
the per-region pattern separates them perfectly.  ICC measures the thing the
average throws away.

The traps being pinned are the ones that would make a paper figure wrong:
ICC must be ~0 on pure noise, must not be inflated by unequal group sizes, must
not treat an unmeasurable region as a zero, and must be checked against a
PERMUTATION null because ICC is positively biased with many groups.
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
           / "celltype_variance_vs_distance.py")


def _mod():
    spec = importlib.util.spec_from_file_location("ctv", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ctv"] = m
    spec.loader.exec_module(m)
    return m


class _Args:
    min_cells_per_group = 3
    min_groups = 3


def _structured(rng, n_cells=90, n_reg=60, n_groups=3, effect=1.0, noise=0.3):
    """Each group has its own per-region profile: real cell-type structure."""
    gid = np.repeat(np.arange(n_groups), n_cells // n_groups)
    prof = rng.normal(0, effect, (n_groups, n_reg))
    X = prof[gid] + rng.normal(0, noise, (len(gid), n_reg))
    return X, gid


# ------------------------------------------------------------- the extremes --

def test_icc_is_near_zero_on_pure_noise():
    """No cell-type structure -> ICC must not manufacture any."""
    m = _mod()
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (90, 60))
    gid = np.repeat(np.arange(3), 30)
    icc, usable = m.icc_per_region(X, gid, 3)
    assert usable.all()
    assert np.nanmedian(icc) < 0.05, f"noise gave ICC={np.nanmedian(icc)}"


def test_icc_is_high_when_groups_really_differ():
    m = _mod()
    rng = np.random.default_rng(1)
    X, gid = _structured(rng, effect=1.0, noise=0.2)
    icc, _ = m.icc_per_region(X, gid, 3)
    assert np.nanmedian(icc) > 0.8, f"strong structure gave {np.nanmedian(icc)}"


def test_icc_tracks_the_effect_size_monotonically():
    m = _mod()
    rng = np.random.default_rng(2)
    got = []
    for effect in (0.0, 0.25, 0.5, 1.0, 2.0):
        X, gid = _structured(rng, effect=effect, noise=0.5)
        got.append(float(np.nanmedian(m.icc_per_region(X, gid, 3)[0])))
    assert all(b >= a - 0.02 for a, b in zip(got, got[1:])), got
    assert got[0] < 0.1 and got[-1] > 0.85, got


def test_negative_variance_component_is_clamped_not_negative():
    """MS_between < MS_within happens by chance; ICC must not go below 0."""
    m = _mod()
    rng = np.random.default_rng(3)
    for _ in range(20):
        X = rng.normal(0, 1, (60, 40))
        gid = np.repeat(np.arange(3), 20)
        icc, _ = m.icc_per_region(X, gid, 3)
        fin = icc[np.isfinite(icc)]
        assert (fin >= 0).all(), fin.min()
        assert (fin <= 1).all(), fin.max()


# ----------------------------------------------------------- the edge cases --

def test_unequal_group_sizes_do_not_inflate_icc():
    """n0 corrects for unbalanced groups; without it ICC drifts upward."""
    m = _mod()
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (120, 80))
    gid = np.r_[np.zeros(5), np.ones(15), np.full(100, 2)].astype(int)
    icc, _ = m.icc_per_region(X, gid, 3)
    assert np.nanmedian(icc) < 0.10, (
        f"unbalanced noise gave ICC={np.nanmedian(icc)}")


def test_region_with_too_few_cells_is_nan_not_zero():
    """Unmeasurable and unstructured are different claims."""
    m = _mod()
    rng = np.random.default_rng(5)
    X, gid = _structured(rng, n_cells=90, n_reg=4)
    X[:, 0] = np.nan                               # region 0 unobserved
    X[3:, 1] = np.nan                              # region 1: 3 cells, 1 group
    icc, usable = m.icc_per_region(X, gid, 3)
    assert not usable[0] and np.isnan(icc[0])
    assert not usable[1] and np.isnan(icc[1])
    assert usable[2] and np.isfinite(icc[2])


def test_nan_cells_are_ignored_not_treated_as_values():
    m = _mod()
    rng = np.random.default_rng(6)
    X, gid = _structured(rng, effect=1.0, noise=0.2)
    clean = float(np.nanmedian(m.icc_per_region(X, gid, 3)[0]))
    holed = X.copy()
    holed[rng.random(holed.shape) < 0.25] = np.nan   # 25% dropout
    got = float(np.nanmedian(m.icc_per_region(holed, gid, 3)[0]))
    assert abs(got - clean) < 0.15, (clean, got)


def test_min_groups_threshold_is_respected():
    m = _mod()
    rng = np.random.default_rng(7)
    X, gid = _structured(rng, n_groups=3, n_reg=10)
    _icc, usable = m.icc_per_region(X, gid, 3, min_groups=4)
    assert not usable.any(), "3 groups must not satisfy min_groups=4"


# ---------------------------------------------------------------- the null --

def test_permutation_null_brackets_noise_and_not_real_structure():
    """The null is the reference, because ICC is positively biased."""
    m = _mod()
    rng = np.random.default_rng(8)

    Xn = rng.normal(0, 1, (90, 60))
    gid = np.repeat(np.arange(3), 30)
    noise = m._summarise(Xn, gid, 3, _Args(), np.random.default_rng(0), 60)
    assert noise["p"] > 0.05, f"noise called significant: p={noise['p']}"
    assert noise["null_lo"] <= noise["icc"] <= noise["null_hi"] + 1e-9

    Xs, gids = _structured(rng, effect=1.0, noise=0.2)
    real = m._summarise(Xs, gids, 3, _Args(), np.random.default_rng(0), 60)
    assert real["p"] < 0.05, f"real structure missed: p={real['p']}"
    assert real["icc"] > real["null_hi"]


def test_permutation_p_is_never_zero():
    m = _mod()
    rng = np.random.default_rng(9)
    X, gid = _structured(rng, effect=3.0, noise=0.1)
    res = m._summarise(X, gid, 3, _Args(), np.random.default_rng(0), 50)
    assert res["p"] > 0, "a finite permutation cannot justify p = 0"
    assert res["p"] == pytest.approx(1.0 / 51, rel=0.3)


# --------------------------------------------------------------- the parser --

def test_matrix_spec_parsing():
    m = _mod()
    assert m._parse_matrix_spec("gch:35000=/a/b.parquet") == \
        ("gch", 35000.0, "/a/b.parquet")
    # a path containing '=' or ':' must still resolve
    assert m._parse_matrix_spec("hcg:1e5=/a=b/c.parquet")[2] == "/a=b/c.parquet"
    assert m._parse_matrix_spec("gch_mean:2000=/x/y.parquet")[0] == "gch_mean"


@pytest.mark.parametrize("bad", ["nodist=/a.parquet", "gch:35000",
                                 "gch:notanumber=/a.parquet"])
def test_matrix_spec_rejects_malformed(bad):
    m = _mod()
    with pytest.raises(SystemExit):
        m._parse_matrix_spec(bad)


# ------------------------------------------------------------- end to end ---

def test_end_to_end(tmp_path):
    m = _mod()
    rng = np.random.default_rng(11)
    samples = tmp_path / "s.tsv"
    n_per, groups = 12, ["A", "B", "C"]
    names = [f"{g}_{i}" for g in groups for i in range(n_per)]
    pd.DataFrame({"cell_name": names,
                  "bam_path": ["x.bam"] * len(names),
                  "group": [n.split("_")[0] for n in names]}
                 ).to_csv(samples, sep="\t", index=False)

    specs = []
    for dist, effect in ((20_000, 0.05), (100_000, 1.0)):
        X, _gid = _structured(rng, n_cells=len(names), n_reg=40,
                              effect=effect, noise=0.4)
        p = tmp_path / f"m{dist}.parquet"
        pd.DataFrame(X, index=names,
                     columns=[f"chr1:{i}" for i in range(40)]).to_parquet(p)
        specs += ["--matrix", f"gch:{dist}={p}"]

    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(_SCRIPT), "--samples", str(samples),
                        "--output-dir", str(out), "--n-permutations", "40",
                        "--plot-format", "png", *specs],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert (out / "celltype_variance_vs_distance.png").exists()
    df = pd.read_csv(out / "celltype_variance_vs_distance.tsv", sep="\t")
    assert len(df) == 2
    weak = df[df.distance == 20_000].iloc[0]
    strong = df[df.distance == 100_000].iloc[0]
    assert strong["icc"] > weak["icc"], "the structured band must score higher"
    assert strong["p"] < 0.05
