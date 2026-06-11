"""Tests for scripts/joint_distance_coordination.py.

Unit-tests the within-cell correlation accumulators and the distance-matched
partner shuffle, plus an end-to-end CLI smoke test on synthetic posteriors with
a KNOWN coordination structure (ends share a region latent → positive
GCH–GCH/HCG–HCG; accessibility = 1−methylation → negative cross), where the
distance-matched shuffle must collapse every correlation toward zero.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / \
    "joint_distance_coordination.py"


def _mod():
    spec = importlib.util.spec_from_file_location("jdc", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------
# Unit: correlation accumulators reproduce numpy's Pearson r
# --------------------------------------------------------------------------

def test_center_sums_match_pearson_single_bin():
    m = _mod()
    rng = np.random.default_rng(0)
    x = rng.random(500); y = 0.6 * x + 0.4 * rng.random(500)
    bins = np.zeros(500, dtype=np.int64)
    mask = np.ones(500, dtype=bool)
    acc = m._center(m._sums(bins, x, y, mask, 1))
    r, n = m._corr(acc, min_pairs=1)
    assert n[0] == 500
    assert abs(r[0] - np.corrcoef(x, y)[0, 1]) < 1e-9


def test_within_cell_pooling_removes_cell_mean_offset():
    """Two 'cells' with opposite global offsets but no within-cell coupling →
    a naive pooled r would be high; within-cell pooling must be ~0."""
    m = _mod()
    rng = np.random.default_rng(1)
    nb = 1
    acc = np.zeros((4, nb))
    for off in (-0.4, 0.4):                       # two cells, opposite offsets
        x = np.clip(0.5 + off + 0.05 * rng.standard_normal(400), 0, 1)
        y = np.clip(0.5 + off + 0.05 * rng.standard_normal(400), 0, 1)  # indep noise
        acc += m._center(m._sums(np.zeros(400, np.int64), x, y,
                                 np.ones(400, bool), nb))
    r, _ = m._corr(acc, min_pairs=1)
    assert abs(r[0]) < 0.2                        # cell-mean offset did not leak


# --------------------------------------------------------------------------
# Unit: the shuffle stays within distance bin and is a permutation
# --------------------------------------------------------------------------

def test_within_bin_partner_stays_in_bin_and_permutes():
    m = _mod()
    rng = np.random.default_rng(2)
    bins = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    p = m._within_bin_partner(bins, rng)
    assert np.array_equal(np.sort(p), np.arange(bins.size))   # a permutation
    assert np.array_equal(bins[p], bins)                      # partner same bin


def test_shuffle_kills_correlation():
    m = _mod()
    rng = np.random.default_rng(3)
    n = 2000
    z = rng.random(n)
    x = np.clip(z + 0.03 * rng.standard_normal(n), 0, 1)
    y = np.clip(z + 0.03 * rng.standard_normal(n), 0, 1)
    bins = np.zeros(n, dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    r_real, _ = m._corr(m._center(m._sums(bins, x, y, mask, 1)), 1)
    p = m._within_bin_partner(bins, np.random.default_rng(4))
    r_shuf, _ = m._corr(m._center(m._sums(bins, x, y[p], mask, 1)), 1)
    assert r_real[0] > 0.8
    assert abs(r_shuf[0]) < 0.1


def test_build_edges_spans_range():
    m = _mod()
    e = m._build_edges(1, 1_000_000, 8)
    assert e[0] == 1 and abs(e[-1] - 1_000_000) < 1e-6
    assert len(e) == 6 * 8 + 1


# --------------------------------------------------------------------------
# Integration: CLI on synthetic posteriors with a known structure
# --------------------------------------------------------------------------

def _write_cell(gdir, hdir, cell, seed, n=600, dist=50_000):
    """Pairs whose two ends share a region latent z (→ correlated a, correlated
    m) with accessibility = 1 − methylation (→ anti-correlated cross)."""
    rng = np.random.default_rng(seed)
    z = rng.random(n)
    sd = 0.05
    a1 = np.clip(z + sd * rng.standard_normal(n), 0.01, 0.99)
    a2 = np.clip(z + sd * rng.standard_normal(n), 0.01, 0.99)
    m1 = np.clip(1.0 - z + sd * rng.standard_normal(n), 0.01, 0.99)
    m2 = np.clip(1.0 - z + sd * rng.standard_normal(n), 0.01, 0.99)
    p1 = 1_000_000 + 1_000 * np.arange(n)
    p2 = p1 + dist
    cols = ["chrom", "pos1", "pos2", "n1", "n2", "posterior1", "posterior2"]
    pd.DataFrame({"chrom": "chr1", "pos1": p1, "pos2": p2, "n1": 5, "n2": 5,
                  "posterior1": a1, "posterior2": a2})[cols].to_parquet(
        gdir / f"{cell}.posteriors.parquet", index=False)
    pd.DataFrame({"chrom": "chr1", "pos1": p1, "pos2": p2, "n1": 5, "n2": 5,
                  "posterior1": m1, "posterior2": m2})[cols].to_parquet(
        hdir / f"{cell}.posteriors.parquet", index=False)


def test_cli_end_to_end(tmp_path):
    gdir = tmp_path / "gch"; hdir = tmp_path / "hcg"
    gdir.mkdir(); hdir.mkdir()
    for i in range(3):
        _write_cell(gdir, hdir, f"C{i}", seed=10 + i)
    out = tmp_path / "fig.pdf"
    cmd = [sys.executable, str(_SCRIPT),
           "--gch-dir", str(gdir), "--hcg-dir", str(hdir),
           "--out", str(out),
           "--min-distance", "10000", "--max-distance", "200000",
           "--bins-per-decade", "2", "--min-pairs", "20",
           "--min-gpc-sites", "2", "--min-cpg-sites", "2", "--n-threads", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert out.exists()
    tsv = out.with_suffix(".tsv")
    assert tsv.exists()
    df = pd.read_csv(tsv, sep="\t")
    # the populated bin (~50 kb) must show the planted structure
    pop = df[np.isfinite(df["gg_r_contact"]) & (df["gg_n"] >= 20)]
    assert len(pop) >= 1
    row = pop.iloc[(pop["center_bp"] - 50_000).abs().argmin()]
    assert row["gg_r_contact"] > 0.6           # ends share accessibility
    assert row["hh_r_contact"] > 0.6           # ends share methylation
    assert row["gh_r_contact"] < -0.3          # acc_i ↔ meth_j anti-correlated
    # distance-matched shuffle collapses every correlation toward 0
    assert abs(row["gg_r_shuffle"]) < 0.2
    assert abs(row["hh_r_shuffle"]) < 0.2
    assert abs(row["gh_r_shuffle"]) < 0.2
