"""Signed-distance density of bin sets to a domain boundary.

Sign convention here is NEGATIVE INSIDE / POSITIVE OUTSIDE -- the opposite of
scripts/boundary_profile.py, which makes inside positive.  Getting that
backwards would invert every biological reading, so it is asserted directly
against bins placed at known positions rather than inferred from a plot.
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
           / "bin_set_boundary_distance.py")


def _mod():
    spec = importlib.util.spec_from_file_location("bsbd", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["bsbd"] = m
    spec.loader.exec_module(m)
    return m


def _bed(path, rows):
    pd.DataFrame(rows, columns=["chrom", "start", "end"]).to_csv(
        path, sep="\t", header=False, index=False)
    return str(path)


# ------------------------------------------------------- the sign convention --

def test_inside_is_negative_and_outside_positive(tmp_path):
    """The whole point of the script's orientation."""
    m = _mod()
    bp = m._boundary_mod()
    dom = pd.DataFrame([("chr1", 1_000_000, 2_000_000)],
                       columns=["chrom", "start", "end"])
    bins = pd.DataFrame(
        [("chr1", 1_400_000, 1_600_000),      # centre 1.5 Mb: INSIDE
         ("chr1", 100_000, 300_000),          # centre 0.2 Mb: outside, left
         ("chr1", 2_400_000, 2_600_000)],     # centre 2.5 Mb: outside, right
        columns=["chrom", "start", "end"])
    v = -bp.signed_distance(bins, dom)
    assert v[0] < 0, "a bin inside the domain must be negative"
    assert v[1] > 0 and v[2] > 0, "bins outside must be positive"
    # magnitudes: nearest edge, not the domain centre
    assert v[0] == pytest.approx(-500_000)    # 1.5 Mb is 0.5 Mb from either edge
    assert v[1] == pytest.approx(800_000)     # 0.2 Mb -> start at 1.0 Mb
    assert v[2] == pytest.approx(500_000)     # 2.5 Mb -> end at 2.0 Mb


# ------------------------------------------------- late-RT from the bigWigs --

def _write_bw(path, chrom, length, values, span):
    pyBigWig = pytest.importorskip("pyBigWig")
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(chrom, length)])
    starts = list(range(0, length, span))
    bw.addEntries([chrom] * len(starts), starts,
                  ends=[s + span for s in starts],
                  values=[float(v) for v in values])
    bw.close()
    return str(path)


def test_late_rt_domains_are_where_late_beats_early(tmp_path):
    """Derived from the data, not from somebody's threshold in a file."""
    m = _mod()
    n, span = 40, 50_000
    length = n * span
    # late wins only over bins 10..19 -> a single 500 kb domain
    early = [1.0] * n
    late = [0.0] * n
    for i in range(10, 20):
        late[i] = 2.0
    e = _write_bw(tmp_path / "e.bw", "chr1", length, early, span)
    l = _write_bw(tmp_path / "l.bw", "chr1", length, late, span)
    got = m.late_rt_domains(e, l, bin_size=span, min_width=100_000)
    assert len(got) == 1
    assert int(got.start.iloc[0]) == 10 * span
    assert int(got.end.iloc[0]) == 20 * span


def test_narrow_late_runs_are_dropped(tmp_path):
    """A 50 kb speck has two edges and would contribute distances as readily
    as a 5 Mb domain."""
    m = _mod()
    n, span = 40, 50_000
    early = [1.0] * n
    late = [0.0] * n
    late[5] = 2.0                                   # one bin only
    for i in range(20, 30):
        late[i] = 2.0                               # 500 kb
    e = _write_bw(tmp_path / "e.bw", "chr1", n * span, early, span)
    l = _write_bw(tmp_path / "l.bw", "chr1", n * span, late, span)
    got = m.late_rt_domains(e, l, bin_size=span, min_width=200_000)
    assert len(got) == 1 and int(got.start.iloc[0]) == 20 * span


def test_no_shared_contig_is_a_clear_error(tmp_path):
    m = _mod()
    e = _write_bw(tmp_path / "e.bw", "chr1", 400_000, [1.0] * 8, 50_000)
    l = _write_bw(tmp_path / "l.bw", "chr2", 400_000, [2.0] * 8, 50_000)
    with pytest.raises(SystemExit, match="no shared contigs"):
        m.late_rt_domains(e, l, bin_size=50_000, min_width=0)


# ------------------------------------------------------------------- CLI --

@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(0)
    allb = [("chr1", i * 250_000, (i + 1) * 250_000) for i in range(60)]
    _bed(tmp_path / "all.bed", allb)
    # "variable" bins sit inside the domain, "invariant" far outside
    _bed(tmp_path / "sel.bed", allb[20:30])
    _bed(tmp_path / "stab.bed", allb[50:60])
    _bed(tmp_path / "lad.bed", [("chr1", 5_000_000, 7_500_000)])
    return tmp_path


def _run(args, ok=True):
    r = subprocess.run([sys.executable, str(_SCRIPT)] + args,
                       capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stderr[-2500:]
    return r


def test_cli_writes_the_table_and_figure(inputs):
    d = inputs
    out = d / "out"
    _run(["--background", f"GCH={d / 'all.bed'}",
          "--bin-set", f"GCH:variable={d / 'sel.bed'}",
          "--bin-set", f"GCH:invariant={d / 'stab.bed'}",
          "--domain", f"cLAD={d / 'lad.bed'}",
          "--plot-format", "png", "--output-dir", str(out)])
    tab = pd.read_csv(out / "boundary_distance.tsv", sep="\t")
    assert (out / "boundary_distance.png").exists()
    assert set(tab["set"]) == {"GCH (background)", "GCH:variable",
                               "GCH:invariant"}
    # the set placed inside the LAD must have a lower median distance
    var = tab.loc[tab["set"] == "GCH:variable", "median_distance"].iloc[0]
    inv = tab.loc[tab["set"] == "GCH:invariant", "median_distance"].iloc[0]
    assert var < inv, (var, inv)


def test_a_bin_set_without_its_background_is_refused(inputs):
    d = inputs
    r = _run(["--background", f"GCH={d / 'all.bed'}",
              "--bin-set", f"HCG:variable={d / 'sel.bed'}",
              "--domain", f"cLAD={d / 'lad.bed'}",
              "--output-dir", str(d / "o")], ok=False)
    assert r.returncode != 0 and "no matching --background" in r.stderr


def test_unpaired_rt_tracks_are_refused(inputs):
    d = inputs
    r = _run(["--background", f"GCH={d / 'all.bed'}",
              "--bin-set", f"GCH:variable={d / 'sel.bed'}",
              "--rt-early", f"NPC={d / 'all.bed'}",
              "--output-dir", str(d / "o")], ok=False)
    assert "must be paired by NAME" in r.stderr


def test_no_domain_at_all_is_refused(inputs):
    d = inputs
    r = _run(["--background", f"GCH={d / 'all.bed'}",
              "--bin-set", f"GCH:variable={d / 'sel.bed'}",
              "--output-dir", str(d / "o")], ok=False)
    assert "nothing to measure against" in r.stderr
