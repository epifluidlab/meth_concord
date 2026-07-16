"""Tests for scripts/read_pair_stats.py.

Uses hand-built read-pair counts so the 0/1/2-sides split per threshold is
exactly predictable, plus a CLI smoke test for the PDF/TSV and the distance
filter.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "read_pair_stats.py"
_COLS = ["chrom", "pos1", "n1", "pos2", "n2", "distance"]


def _mod():
    spec = importlib.util.spec_from_file_location("rps", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(gdir, hdir, cell, gpc, cpg, dists=None):
    """gpc/cpg: lists of (n1, n2) site counts per pair."""
    n = len(gpc)
    p1 = 1_000_000 + 10_000 * np.arange(n)
    d = np.asarray(dists if dists is not None else [50_000] * n)
    p2 = p1 + d
    pd.DataFrame({"chrom": "chr1", "pos1": p1, "n1": [x[0] for x in gpc],
                  "pos2": p2, "n2": [x[1] for x in gpc],
                  "distance": d})[_COLS].to_parquet(
        gdir / f"{cell}.read_pairs.parquet", index=False)
    pd.DataFrame({"chrom": "chr1", "pos1": p1, "n1": [x[0] for x in cpg],
                  "pos2": p2, "n2": [x[1] for x in cpg],
                  "distance": d})[_COLS].to_parquet(
        hdir / f"{cell}.read_pairs.parquet", index=False)


def test_split_counts_sides():
    m = _mod()
    ok1 = np.array([False, True, True, False])
    ok2 = np.array([False, False, True, True])
    # sides qualifying: 0, 1, 2, 1  -> [neither=1, one=2, both=1]
    assert list(m._split(ok1, ok2)) == [1, 2, 1]


def test_process_cell_exact_counts(tmp_path):
    m = _mod()
    g = tmp_path / "g"; h = tmp_path / "h"; g.mkdir(); h.mkdir()
    # 4 pairs, GpC per side: (0,0) (3,0) (3,3) (1,1)
    gpc = [(0, 0), (3, 0), (3, 3), (1, 1)]
    # CpG per side: (0,0) (2,2) (2,0) (0,0)
    cpg = [(0, 0), (2, 2), (2, 0), (0, 0)]
    _write(g, h, "C0", gpc, cpg)
    total, gch, hcg, joint = m.process_cell(
        g / "C0.read_pairs.parquet", h / "C0.read_pairs.parquet",
        [1, 3], [1, 2], [1, 2], 0, 0)
    assert total == 4
    # GpC >= 1 -> sides: 0, 1, 2, 2  => neither=1, one=1, both=2
    assert list(gch[0]) == [1, 1, 2]
    # GpC >= 3 -> sides: 0, 1, 2, 0  => neither=2, one=1, both=1
    assert list(gch[1]) == [2, 1, 1]
    # CpG >= 1 -> sides: 0, 2, 1, 0  => neither=2, one=1, both=1
    assert list(hcg[0]) == [2, 1, 1]
    # CpG >= 2 -> same (all CpG counts are 0 or 2)
    assert list(hcg[1]) == [2, 1, 1]
    # JOINT >= 1 (side needs GpC>=1 AND CpG>=1): sides 0, 1, 1, 0
    #   pair2 (3,0)/(2,2): side1 ok, side2 no  -> 1
    #   pair3 (3,3)/(2,0): side1 ok, side2 no  -> 1
    assert list(joint[0]) == [2, 2, 0]
    # every row sums to the pair total
    for arr in (gch, hcg, joint):
        assert (arr.sum(axis=1) == 4).all()


def test_distance_filter(tmp_path):
    m = _mod()
    g = tmp_path / "g"; h = tmp_path / "h"; g.mkdir(); h.mkdir()
    gpc = [(5, 5)] * 4
    cpg = [(5, 5)] * 4
    _write(g, h, "C0", gpc, cpg, dists=[100, 100, 50_000, 50_000])
    total, gch, _, _ = m.process_cell(
        g / "C0.read_pairs.parquet", h / "C0.read_pairs.parquet",
        [1], [1], [1], 20_000, 0)
    assert total == 2                      # only the two 50 kb pairs
    assert list(gch[0]) == [0, 0, 2]


def test_subsample_modes():
    m = _mod()
    cells = [f"C{i:02d}" for i in range(20)]
    assert m.subsample(cells, 0, 0) == cells
    assert m.subsample(cells, 999, 0) == cells
    s1 = m.subsample(cells, 5, seed=1)
    assert len(s1) == 5 and s1 == sorted(s1)
    assert m.subsample(cells, 5, seed=1) == s1
    assert m.subsample(cells, 5, seed=2) != s1


def test_cli_end_to_end(tmp_path):
    g = tmp_path / "g"; h = tmp_path / "h"; g.mkdir(); h.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        n = 300
        gpc = list(zip(rng.integers(0, 8, n), rng.integers(0, 8, n)))
        cpg = list(zip(rng.integers(0, 5, n), rng.integers(0, 5, n)))
        _write(g, h, f"C{i}", gpc, cpg)
    out = tmp_path / "stats.pdf"
    cmd = [sys.executable, str(_SCRIPT),
           "--gch-read-pairs-dir", str(g), "--hcg-read-pairs-dir", str(h),
           "--out", str(out), "--n-threads", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert out.exists()
    tsv = out.with_suffix(".tsv")
    assert tsv.exists()
    # the QC provenance note is recorded in the log
    assert "not PCR duplicate, mapQ>30 on both sides" in r.stdout
    df = pd.read_csv(tsv, sep="\t")
    # 5 GpC + 3 CpG + 3 joint thresholds, x 3 categories
    assert len(df) == (5 + 3 + 3) * 3
    # every (panel, threshold) sums to 100%
    s = df.groupby(["panel", "min_sites_per_side"])["pct"].sum()
    assert np.allclose(s.to_numpy(), 100.0, atol=1e-6)
    # and to the pair total
    c = df.groupby(["panel", "min_sites_per_side"])["n_pairs"].sum().unique()
    assert len(c) == 1 and c[0] == 900


def test_cli_min_distance_label(tmp_path):
    g = tmp_path / "g"; h = tmp_path / "h"; g.mkdir(); h.mkdir()
    _write(g, h, "C0", [(3, 3)] * 50, [(2, 2)] * 50)
    out = tmp_path / "s2.pdf"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--gch-read-pairs-dir", str(g),
         "--hcg-read-pairs-dir", str(h), "--out", str(out),
         "--min-distance", "20000", "--n-threads", "1"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "linked 20,000-inf bp" in r.stdout
    assert "high-quality linked pairs" in r.stdout
