"""Export the 250 kb-bin signals to bigWig / bedGraph tracks.

Covers the pure logic (bin-grid parse, per-group + merged means, step vs smooth
intervals with contig clamping) and an end-to-end bedGraph run; the bigWig write
is checked only where pyBigWig is installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "concordance_bigwig.py")


def _mod():
    spec = importlib.util.spec_from_file_location("cbw", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["cbw"] = m
    spec.loader.exec_module(m)
    return m


def _matrix():
    cells = [f"c{i}" for i in range(6)]
    bins = [f"chr1:{i * 250000}" for i in range(4)]
    X = np.array([[1.0] * 4] * 3 + [[3.0] * 4] * 3)      # A -> 1, B -> 3
    return pd.DataFrame(X, index=cells, columns=bins)


def _groups():
    return pd.Series(["A", "A", "A", "B", "B", "B"],
                     index=[f"c{i}" for i in range(6)])


# ----------------------------------------------------------------- pure logic -

def test_bin_grid_parses_and_infers_width():
    m = _mod()
    g = m._bin_grid(["chr1:0", "chr1:250000", "chr2:500000"])
    assert g["chrom"].tolist() == ["chr1", "chr1", "chr2"]
    assert g["start"].tolist() == [0, 250000, 500000]
    assert (g["width"] == 250000).all()


def test_group_means_merged_and_per_group():
    m = _mod()
    tr = m._group_means(_matrix(), _groups(), "both", min_cells=1)
    assert set(tr) == {"all", "A", "B"}
    assert tr["A"].tolist() == [1, 1, 1, 1]
    assert tr["B"].tolist() == [3, 3, 3, 3]
    assert tr["all"].tolist() == [2, 2, 2, 2]              # (1*3+3*3)/6
    # min_cells filters small groups; 'merged' skips per-group entirely
    assert set(m._group_means(_matrix(), _groups(), "per-group", 4)) == set()
    assert set(m._group_means(_matrix(), _groups(), "merged", 1)) == {"all"}


def test_intervals_step_clamps_last_bin_and_drops_nan():
    m = _mod()
    grid = m._bin_grid([f"chr1:{i*250000}" for i in range(4)])
    vals = pd.Series([1.0, np.nan, 3.0, 4.0])
    iv = m._intervals(vals, grid, {"1": 900000}, smooth=False, bin_size=1000)
    assert iv == [("chr1", 0, 250000, 1.0),               # NaN bin dropped
                  ("chr1", 500000, 750000, 3.0),
                  ("chr1", 750000, 900000, 4.0)]          # last bin clamped


def test_intervals_smooth_interpolates_between_bin_centres():
    m = _mod()
    grid = m._bin_grid(["chr1:0", "chr1:250000"])          # centres 125k, 375k
    vals = pd.Series([0.0, 10.0])
    iv = m._intervals(vals, grid, {"1": 500000}, smooth=True, bin_size=1000)
    v = [r[3] for r in iv]
    assert all(np.diff(v) >= -1e-9)                        # monotone increasing
    assert v[0] == pytest.approx(0.0, abs=0.1)             # at/near first centre
    assert max(v) == pytest.approx(10.0, abs=0.2)          # reaches second centre


def test_read_chrom_sizes(tmp_path):
    m = _mod()
    p = tmp_path / "g.sizes"
    p.write_text("chr1\t195471971\nchr2\t182113224\n")
    assert m._read_chrom_sizes(str(p)) == {"1": 195471971, "2": 182113224}


# ------------------------------------------------------------------ end to end -

def _run(tmp_path, extra=None):
    m = _mod()
    d = tmp_path
    _matrix().to_parquet(d / "gch.parquet")
    pd.DataFrame({"cell_name": [f"c{i}" for i in range(6)],
                  "bam_path": ["x"] * 6,
                  "group": ["A", "A", "A", "B", "B", "B"]}
                 ).to_csv(d / "s.tsv", sep="\t", index=False)
    (d / "chrom.sizes").write_text("chr1\t900000\n")
    argv = ["--signal", f"gch={d/'gch.parquet'}", "--samples", str(d / "s.tsv"),
            "--cell-type-column", "group", "--chrom-sizes", str(d / "chrom.sizes"),
            "--output-dir", str(d / "out")] + (extra or [])
    m.main(argv)
    return d / "out"


def test_end_to_end_bedgraph_distinct_files_and_values(tmp_path):
    """One bedGraph per group + merged, with the right values and DISTINCT names
    (regression: 'gch.all' must not collapse to 'gch.bedgraph')."""
    out = _run(tmp_path)
    for grp, val in [("all", "2"), ("A", "1"), ("B", "3")]:
        f = out / f"gch.{grp}.bedgraph"
        assert f.exists(), f
        rows = [ln.split("\t") for ln in f.read_text().splitlines()]
        assert all(r[3] == val for r in rows)             # constant group mean
        assert rows[-1] == ["chr1", "750000", "900000", val]   # last bin clamped


def test_bigwig_written_when_pybigwig_present(tmp_path):
    pytest.importorskip("pyBigWig")
    import pyBigWig
    out = _run(tmp_path)
    bw = pyBigWig.open(str(out / "gch.B.bw"))
    assert bw.chroms() == {"chr1": 900000}
    assert bw.stats("chr1", 0, 900000)[0] == pytest.approx(3.0)
    bw.close()
