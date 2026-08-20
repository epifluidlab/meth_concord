"""Per-bin repeat density from the UCSC RepeatMasker table.

Network-free: the download is separated from the aggregation, and only the
aggregation is tested here.
"""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "fetch_repeat_density.py")


def _mod():
    spec = importlib.util.spec_from_file_location("frd", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["frd"] = m
    spec.loader.exec_module(m)
    return m


def _rmsk(path, rows):
    """rows: (chrom, start, end, name, cls, family) -> a gzipped rmsk table."""
    with gzip.open(path, "wt") as fh:
        for chrom, start, end, name, cls, fam in rows:
            f = [""] * 13
            f[5], f[6], f[7] = chrom, str(start), str(end)
            f[10], f[11], f[12] = name, cls, fam
            fh.write("\t".join(f) + "\n")
    return path


def test_density_is_the_covered_fraction_of_the_bin(tmp_path):
    m = _mod()
    p = _rmsk(tmp_path / "r.txt.gz",
              [("chr1", 0, 25000, "L1A", "LINE", "L1")])      # 25 kb of 100 kb
    acc = m.accumulate(p, {"chr1": 100_000}, 100_000, {"LINE"}, set(), set())
    df = m.to_bedgraph(acc["LINE"], {"chr1": 100_000}, 100_000,
                       tmp_path / "o.bedGraph")
    assert df.value.iloc[0] == pytest.approx(0.25)


def test_an_element_spanning_a_boundary_is_split_between_the_bins(tmp_path):
    """Assigning it wholly to the bin it starts in would move density across
    a boundary, and satellite arrays are long enough for that to matter."""
    m = _mod()
    p = _rmsk(tmp_path / "r.txt.gz",
              [("chr1", 90_000, 110_000, "SAT", "Satellite", "centr")])
    acc = m.accumulate(p, {"chr1": 200_000}, 100_000, {"Satellite"}, set(),
                       set())
    assert acc["Satellite"][("chr1", 0)] == pytest.approx(10_000)
    assert acc["Satellite"][("chr1", 1)] == pytest.approx(10_000)


def test_bins_with_no_element_are_written_as_zero_not_dropped(tmp_path):
    """A bin with no LINE has density 0, which is a measurement.  Dropping it
    would let the enrichment treat those bins as missing instead."""
    m = _mod()
    p = _rmsk(tmp_path / "r.txt.gz", [("chr1", 0, 50_000, "L1", "LINE", "L1")])
    acc = m.accumulate(p, {"chr1": 300_000}, 100_000, {"LINE"}, set(), set())
    df = m.to_bedgraph(acc["LINE"], {"chr1": 300_000}, 100_000,
                       tmp_path / "o.bedGraph")
    assert len(df) == 3
    assert list(df.value) == pytest.approx([0.5, 0.0, 0.0])


def test_the_last_bin_is_measured_against_the_real_contig_length(tmp_path):
    """A short final bin divided by the full grid width would understate its
    density; the contig length is what it is actually out of."""
    m = _mod()
    p = _rmsk(tmp_path / "r.txt.gz", [("chr1", 100_000, 125_000, "L1",
                                       "LINE", "L1")])
    sizes = {"chr1": 150_000}                       # last bin is 50 kb
    acc = m.accumulate(p, sizes, 100_000, {"LINE"}, set(), set())
    df = m.to_bedgraph(acc["LINE"], sizes, 100_000, tmp_path / "o.bedGraph")
    assert df.end.iloc[-1] == 150_000
    assert df.value.iloc[-1] == pytest.approx(25_000 / 50_000)


def test_class_family_and_name_each_get_their_own_track(tmp_path):
    m = _mod()
    p = _rmsk(tmp_path / "r.txt.gz",
              [("chr1", 0, 10_000, "L1HS", "LINE", "L1"),
               ("chr1", 10_000, 20_000, "L1MA", "LINE", "L1"),
               ("chr1", 20_000, 30_000, "AluY", "SINE", "Alu")])
    acc = m.accumulate(p, {"chr1": 100_000}, 100_000,
                       {"LINE"}, {"L1"}, {"L1HS"})
    assert set(acc) == {"LINE", "fam_L1", "rep_L1HS"}
    assert acc["LINE"][("chr1", 0)] == 20_000        # both L1 elements
    assert acc["fam_L1"][("chr1", 0)] == 20_000
    assert acc["rep_L1HS"][("chr1", 0)] == 10_000    # the named one only
    assert "SINE" not in acc, "SINE was not requested"


def test_a_contig_absent_from_chrom_sizes_is_skipped(tmp_path):
    """rmsk carries alts and randoms; without a length there is no grid."""
    m = _mod()
    p = _rmsk(tmp_path / "r.txt.gz",
              [("chr1_KI270713v1_random", 0, 10_000, "L1", "LINE", "L1"),
               ("chr1", 0, 10_000, "L1", "LINE", "L1")])
    acc = m.accumulate(p, {"chr1": 100_000}, 100_000, {"LINE"}, set(), set())
    assert set(acc["LINE"]) == {("chr1", 0)}


def test_density_never_exceeds_one(tmp_path):
    m = _mod()
    p = _rmsk(tmp_path / "r.txt.gz",
              [("chr1", 0, 100_000, "SAT", "Satellite", "centr")])
    acc = m.accumulate(p, {"chr1": 100_000}, 100_000, {"Satellite"}, set(),
                       set())
    df = m.to_bedgraph(acc["Satellite"], {"chr1": 100_000}, 100_000,
                       tmp_path / "o.bedGraph")
    assert df.value.max() == pytest.approx(1.0)


def test_chroms_can_be_restricted(tmp_path):
    m = _mod()
    p = _rmsk(tmp_path / "r.txt.gz",
              [("chr1", 0, 10_000, "L1", "LINE", "L1"),
               ("chr2", 0, 10_000, "L1", "LINE", "L1")])
    acc = m.accumulate(p, {"chr1": 100_000, "chr2": 100_000}, 100_000,
                       {"LINE"}, set(), set(), chroms={"chr1"})
    assert set(acc["LINE"]) == {("chr1", 0)}
