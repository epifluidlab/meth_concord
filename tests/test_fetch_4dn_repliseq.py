"""4DN 2-stage Repli-seq fetcher.

Network-free: the portal query is separated from the grouping and the ratio,
and only the latter two are tested here.
"""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "fetch_4dn_repliseq.py")


def _mod():
    spec = importlib.util.spec_from_file_location("f4dnrs", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["f4dnrs"] = m
    spec.loader.exec_module(m)
    return m


def _rec(cell, condition, acc="4DNF", assembly="GRCh38"):
    return {"accession": acc, "genome_assembly": assembly,
            "track_and_facet_info": {"biosource_name": cell,
                                     "condition": condition,
                                     "experiment_type": "2-stage Repli-seq"}}


def _bg(path, rows, gz=True):
    text = "".join(f"{c}\t{s}\t{e}\t{v}\n" for c, s, e, v in rows)
    if gz:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        Path(path).write_text(text)
    return path


# ---------------------------------------------------------------- grouping ---

def test_early_and_late_are_paired_per_cell_type():
    m = _mod()
    got = m.group([_rec("K562 (Tier 2)", "early", "E1"),
                   _rec("K562 (Tier 2)", "late", "L1")])
    assert list(got) == ["K562"]
    assert [r["accession"] for r in got["K562"]["early"]] == ["E1"]


def test_a_cell_type_with_only_one_fraction_is_dropped():
    """One fraction cannot make a ratio, and carrying it forward would fail
    later with something less obvious than 'not available'."""
    m = _mod()
    assert m.group([_rec("K562 (Tier 2)", "early")]) == {}


def test_the_differentiated_state_stays_a_separate_track():
    """4DN writes derived states as a suffix after the tier.  Truncating there
    averages endoderm-differentiated H1 into plain H1 with nothing in the
    output to show it."""
    m = _mod()
    got = m.group([
        _rec("H1-hESC (Tier 1)", "early", "A"),
        _rec("H1-hESC (Tier 1)", "late", "B"),
        _rec("H1-hESC (Tier 1) differentiated to definitive endoderm",
             "early", "C"),
        _rec("H1-hESC (Tier 1) differentiated to definitive endoderm",
             "late", "D")])
    assert set(got) == {"H1-hESC",
                        "H1-hESC_differentiated_to_definitive_endoderm"}
    assert [r["accession"] for r in got["H1-hESC"]["early"]] == ["A"]


def test_a_genotype_in_the_condition_stays_in_the_label():
    """The engineered iPSC line ships three genotypes under one biosource
    name; merging them would average edited and unedited cells."""
    m = _mod()
    got = m.group([
        _rec("7889SA3.5", "IZ boundary deletion early fraction"),
        _rec("7889SA3.5", "IZ boundary deletion late fraction"),
        _rec("7889SA3.5", "unedited cells early fraction"),
        _rec("7889SA3.5", "unedited cells late fraction")])
    assert set(got) == {"7889SA3.5_IZ_boundary_deletion",
                        "7889SA3.5_unedited_cells"}


def test_precomputed_ratio_records_are_skipped():
    """Those name neither fraction; this script builds the ratio itself."""
    m = _mod()
    assert m.group([_rec("7889SA3.5", "unedited cells E/L fractions")]) == {}


def test_the_assembly_is_honoured():
    m = _mod()
    recs = [_rec("K562 (Tier 2)", "early", assembly="GRCm38"),
            _rec("K562 (Tier 2)", "late", assembly="GRCm38")]
    assert m.group(recs, assembly="GRCh38") == {}
    assert list(m.group(recs, assembly="GRCm38")) == ["K562"]


# ------------------------------------------------------------------ ratio ---

def test_totals_are_summed_into_the_target_bin(tmp_path):
    m = _mod()
    p = _bg(tmp_path / "a.bedGraph.gz",
            [("chr1", 0, 5000, 1.0), ("chr1", 5000, 10000, 3.0),
             ("chr1", 10000, 15000, 5.0)])
    got = m.binned_totals(p, 10_000)
    assert got[("chr1", 0)] == pytest.approx(4.0)
    assert got[("chr1", 1)] == pytest.approx(5.0)


def test_a_grid_that_splits_the_native_bin_is_refused(tmp_path):
    """Integer division would silently put a straddling interval wholly in one
    bin.  Better to refuse than to apportion invisibly."""
    m = _mod()
    p = _bg(tmp_path / "a.bedGraph.gz", [("chr1", 0, 5000, 1.0),
                                         ("chr1", 5000, 10000, 1.0)])
    with pytest.raises(SystemExit, match="not a multiple"):
        m.binned_totals(p, 7_000)


def test_the_ratio_is_of_totals_not_a_mean_of_ratios(tmp_path):
    """The whole point of the ordering.  One 5 kb bin carries almost all the
    early signal and the other almost none; the ratio of the summed totals is
    what was measured, while averaging the two per-bin log ratios gives a
    different number entirely -- and at native resolution, where a tenth of
    bins are zero, that second number is mostly pseudocount."""
    m = _mod()
    e = _bg(tmp_path / "e.bedGraph.gz", [("chr1", 0, 5000, 9.0),
                                         ("chr1", 5000, 10000, 1.0)])
    l = _bg(tmp_path / "l.bedGraph.gz", [("chr1", 0, 5000, 1.0),
                                         ("chr1", 5000, 10000, 1.0)])
    df, dropped = m.timing([e], [l], 10_000)
    assert dropped == 0
    ratio_of_totals = np.log2(10 / 2)                       # 2.3219
    mean_of_ratios = (np.log2(9 / 1) + np.log2(1 / 1)) / 2  # 1.5850
    assert df.value.iloc[0] == pytest.approx(ratio_of_totals)
    assert abs(ratio_of_totals - mean_of_ratios) > 0.5, "pick a sharper case"


def test_replicates_are_averaged_within_a_fraction(tmp_path):
    m = _mod()
    e1 = _bg(tmp_path / "e1.bedGraph.gz", [("chr1", 0, 5000, 2.0)])
    e2 = _bg(tmp_path / "e2.bedGraph.gz", [("chr1", 0, 5000, 6.0)])
    l = _bg(tmp_path / "l.bedGraph.gz", [("chr1", 0, 5000, 4.0)])
    df, _ = m.timing([e1, e2], [l], 5_000)
    assert df.value.iloc[0] == pytest.approx(0.0)     # mean(2,6)=4, 4/4 -> 0


def test_an_empty_fraction_drops_the_bin_rather_than_pseudocounting(tmp_path):
    """At 250 kb an all-zero fraction is an unmappable region, not a real
    extreme; a pseudocount would turn it into a confident large value."""
    m = _mod()
    e = _bg(tmp_path / "e.bedGraph.gz", [("chr1", 0, 5000, 4.0),
                                         ("chr1", 5000, 10000, 0.0)])
    l = _bg(tmp_path / "l.bedGraph.gz", [("chr1", 0, 5000, 2.0),
                                         ("chr1", 5000, 10000, 0.0)])
    df, dropped = m.timing([e], [l], 5_000)
    assert dropped == 1
    assert list(df.start) == [0]
    assert np.isfinite(df.value).all()


def test_chroms_can_be_restricted(tmp_path):
    m = _mod()
    e = _bg(tmp_path / "e.bedGraph.gz", [("chr1", 0, 5000, 4.0),
                                         ("chr2", 0, 5000, 4.0)])
    l = _bg(tmp_path / "l.bedGraph.gz", [("chr1", 0, 5000, 2.0),
                                         ("chr2", 0, 5000, 8.0)])
    df, _ = m.timing([e], [l], 5_000, chroms=["chr1"])
    assert list(df.chrom) == ["chr1"]


def test_domains_split_at_the_sign(tmp_path):
    m = _mod()
    df = pd.DataFrame({"chrom": ["chr1"] * 3, "start": [0, 100, 200],
                       "end": [100, 200, 300], "value": [1.5, -0.5, 0.0]})
    assert m.write_domains(df, tmp_path / "e.bed", early=True) == 1
    assert m.write_domains(df, tmp_path / "l.bed", early=False) == 1
    got = pd.read_csv(tmp_path / "e.bed", sep="\t", header=None)
    assert got.shape == (1, 3) and got[1].iloc[0] == 0


def test_the_bedgraph_is_four_columns_no_header(tmp_path):
    m = _mod()
    df = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [250000],
                       "value": [-1.25]})
    m.write_bedgraph(df, tmp_path / "x.bedGraph")
    got = pd.read_csv(tmp_path / "x.bedGraph", sep="\t", header=None)
    assert got.shape == (1, 4)
    assert got[3].iloc[0] == pytest.approx(-1.25)
