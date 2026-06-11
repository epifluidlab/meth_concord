"""Tests for scripts/joint_observability_check.py.

Covers the two upgrades: independent per-modality site thresholds
(--min-gpc-sites / --min-cpg-sites) and multithreaded aggregation across cells
(--n-threads), the latter must be bit-for-bit thread-invariant since cells are
summed/concatenated.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / \
    "joint_observability_check.py"


def _mod():
    spec = importlib.util.spec_from_file_location("joc", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_pairs(gdir, hdir, cell, seed, n=400):
    """Read-pair counts with random per-end GpC/CpG site depths so changing a
    modality's threshold demonstrably shifts the both/only/neither split."""
    rng = np.random.default_rng(seed)
    p1 = 1_000_000 + 1_000 * np.arange(n)
    p2 = p1 + rng.integers(5_000, 200_000, n)        # spread across the linked band
    cols = ["chrom", "pos1", "n1", "pos2", "n2", "distance"]
    pd.DataFrame({"chrom": "chr1", "pos1": p1, "n1": rng.integers(0, 8, n),
                  "pos2": p2, "n2": rng.integers(0, 8, n),
                  "distance": p2 - p1})[cols].to_parquet(
        gdir / f"{cell}.read_pairs.parquet", index=False)
    pd.DataFrame({"chrom": "chr1", "pos1": p1, "n1": rng.integers(0, 8, n),
                  "pos2": p2, "n2": rng.integers(0, 8, n),
                  "distance": p2 - p1})[cols].to_parquet(
        hdir / f"{cell}.read_pairs.parquet", index=False)


# --------------------------------------------------------------------------
# Unit: per-modality thresholds
# --------------------------------------------------------------------------

def test_process_cell_independent_thresholds(tmp_path):
    m = _mod()
    g = tmp_path / "g"; h = tmp_path / "h"; g.mkdir(); h.mkdir()
    _write_pairs(g, h, "C0", seed=0)
    gp = g / "C0.read_pairs.parquet"; hp = h / "C0.read_pairs.parquet"

    lo = m.process_cell(gp, hp, min_gpc=2, min_cpg=2, mind=20_000, bsz=250_000)
    # CpG made impossible → no end can be CpG-informative
    hard_cpg = m.process_cell(gp, hp, min_gpc=2, min_cpg=99, mind=20_000,
                              bsz=250_000)
    # GpC made impossible → no end can be GpC-informative
    hard_gpc = m.process_cell(gp, hp, min_gpc=99, min_cpg=2, mind=20_000,
                              bsz=250_000)

    tot = sum(lo["end_counts"].values())
    assert tot == sum(hard_cpg["end_counts"].values()) == sum(
        hard_gpc["end_counts"].values())                      # conservation
    # raising CpG threshold can only move both→gch_only, cpg_only→neither
    assert hard_cpg["end_counts"]["both"] <= lo["end_counts"]["both"]
    assert hard_cpg["end_counts"]["gch_only"] >= lo["end_counts"]["gch_only"]
    assert hard_cpg["end_counts"]["both"] == 0
    assert hard_cpg["end_counts"]["cpg_only"] == 0
    # symmetric for GpC
    assert hard_gpc["end_counts"]["both"] == 0
    assert hard_gpc["end_counts"]["gch_only"] == 0
    # the threshold split is not a no-op on this data
    assert lo["end_counts"]["both"] > 0


# --------------------------------------------------------------------------
# Unit: cell subsampling (0 = all, reproducible random, cap >= n = all)
# --------------------------------------------------------------------------

def test_subsample_modes():
    m = _mod()
    cells = [f"C{i:02d}" for i in range(20)]
    assert m.subsample(cells, 0, 0) == cells          # 0 = all
    assert m.subsample(cells, -1, 0) == cells          # negative = all
    assert m.subsample(cells, 999, 0) == cells         # cap >= n = all
    s1 = m.subsample(cells, 5, seed=1)
    assert len(s1) == 5 and s1 == sorted(s1) and set(s1) <= set(cells)
    assert m.subsample(cells, 5, seed=1) == s1         # reproducible
    assert m.subsample(cells, 5, seed=2) != s1         # seed changes the draw


# --------------------------------------------------------------------------
# Integration: thread-invariance + flag wiring
# --------------------------------------------------------------------------

def _run(gdir, hdir, *extra):
    cmd = [sys.executable, str(_SCRIPT),
           "--gch-read-pairs-dir", str(gdir),
           "--hcg-read-pairs-dir", str(hdir),
           "--min-distance", "20000", "--bin-size", "250000",
           *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    return r.stdout


def _report_body(stdout):
    """Drop the header line (carries the thread count) so the comparison is of
    the computed tallies only."""
    return stdout.split("=== per-end modality", 1)[1]


def test_cli_thread_invariant(tmp_path):
    g = tmp_path / "g"; h = tmp_path / "h"; g.mkdir(); h.mkdir()
    for i in range(6):
        _write_pairs(g, h, f"C{i}", seed=100 + i)
    out1 = _run(g, h, "--min-gpc-sites", "2", "--min-cpg-sites", "2",
                "--n-threads", "1")
    out3 = _run(g, h, "--min-gpc-sites", "2", "--min-cpg-sites", "2",
                "--n-threads", "3")
    assert _report_body(out1) == _report_body(out3)
    assert "3 threads" in out3 and "1 threads" in out1


def test_cli_min_sites_fallback(tmp_path):
    """--min-sites still drives both modalities when the specific flags are unset."""
    g = tmp_path / "g"; h = tmp_path / "h"; g.mkdir(); h.mkdir()
    for i in range(3):
        _write_pairs(g, h, f"C{i}", seed=200 + i)
    out = _run(g, h, "--min-sites", "3", "--n-threads", "1")
    assert "GpC>=3 CpG>=3" in out


def test_cli_max_cells_zero_uses_all(tmp_path):
    g = tmp_path / "g"; h = tmp_path / "h"; g.mkdir(); h.mkdir()
    for i in range(7):
        _write_pairs(g, h, f"C{i}", seed=300 + i)
    out = _run(g, h, "--max-cells", "0", "--n-threads", "1")
    assert "7 shared cells; using 7 (all)" in out


def test_cli_random_sample_reproducible(tmp_path):
    """A seeded subsample of the cells is byte-identical run-to-run."""
    g = tmp_path / "g"; h = tmp_path / "h"; g.mkdir(); h.mkdir()
    for i in range(10):
        _write_pairs(g, h, f"C{i}", seed=400 + i)
    a = _run(g, h, "--max-cells", "4", "--seed", "5", "--n-threads", "1")
    b = _run(g, h, "--max-cells", "4", "--seed", "5", "--n-threads", "1")
    assert "using 4 (random, seed 5)" in a
    assert _report_body(a) == _report_body(b)          # same seed → same tallies
