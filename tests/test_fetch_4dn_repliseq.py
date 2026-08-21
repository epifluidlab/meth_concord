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
    late = _bg(tmp_path / "l.bedGraph.gz", [("chr1", 0, 5000, 1.0),
                                            ("chr1", 5000, 10000, 1.0)])
    df, dropped = m.timing([e], [late], 10_000)
    assert dropped == 0
    ratio_of_totals = np.log2(10 / 2)                       # 2.3219
    mean_of_ratios = (np.log2(9 / 1) + np.log2(1 / 1)) / 2  # 1.5850
    assert df.value.iloc[0] == pytest.approx(ratio_of_totals)
    assert abs(ratio_of_totals - mean_of_ratios) > 0.5, "pick a sharper case"


def test_replicates_are_averaged_within_a_fraction(tmp_path):
    m = _mod()
    e1 = _bg(tmp_path / "e1.bedGraph.gz", [("chr1", 0, 5000, 2.0)])
    e2 = _bg(tmp_path / "e2.bedGraph.gz", [("chr1", 0, 5000, 6.0)])
    late = _bg(tmp_path / "l.bedGraph.gz", [("chr1", 0, 5000, 4.0)])
    df, _ = m.timing([e1, e2], [late], 5_000)
    assert df.value.iloc[0] == pytest.approx(0.0)     # mean(2,6)=4, 4/4 -> 0


def test_an_empty_fraction_drops_the_bin_rather_than_pseudocounting(tmp_path):
    """At 250 kb an all-zero fraction is an unmappable region, not a real
    extreme; a pseudocount would turn it into a confident large value."""
    m = _mod()
    e = _bg(tmp_path / "e.bedGraph.gz", [("chr1", 0, 5000, 4.0),
                                         ("chr1", 5000, 10000, 0.0)])
    late = _bg(tmp_path / "l.bedGraph.gz", [("chr1", 0, 5000, 2.0),
                                            ("chr1", 5000, 10000, 0.0)])
    df, dropped = m.timing([e], [late], 5_000)
    assert dropped == 1
    assert list(df.start) == [0]
    assert np.isfinite(df.value).all()


def test_chroms_can_be_restricted(tmp_path):
    m = _mod()
    e = _bg(tmp_path / "e.bedGraph.gz", [("chr1", 0, 5000, 4.0),
                                         ("chr2", 0, 5000, 4.0)])
    late = _bg(tmp_path / "l.bedGraph.gz", [("chr1", 0, 5000, 2.0),
                                            ("chr2", 0, 5000, 8.0)])
    df, _ = m.timing([e], [late], 5_000, chroms=["chr1"])
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


# -------------------------------------------------------------- partition ---

def _frame(values, chrom="chr1", grid=250_000):
    return pd.DataFrame({"chrom": [chrom] * len(values),
                         "start": np.arange(len(values)) * grid,
                         "end": (np.arange(len(values)) + 1) * grid,
                         "value": np.asarray(values, float)})


def test_partition_splits_constitutive_late_early_and_variable():
    m = _mod()
    part, counts, incomplete = m.rt_partition([
        _frame([-1.0, +1.0, -1.0, +2.0]),
        _frame([-2.0, +2.0, +1.0, +1.0])])
    assert list(part["class"]) == ["RT_constitutive_late",
                                   "RT_constitutive_early",
                                   "RT_variable",
                                   "RT_constitutive_early"]
    assert incomplete == 0


def test_a_sign_cut_makes_the_three_classes_exhaustive():
    m = _mod()
    rng = np.random.default_rng(0)
    frames = [_frame(rng.normal(size=200)) for _ in range(4)]
    part, _, _ = m.rt_partition(frames, threshold=0.0)
    assert not (part["class"] == "RT_unclassified").any()


def test_a_bin_missing_in_one_cell_type_is_not_called_constitutive():
    """Coverage differs between cell types.  Scoring a bin 'late in all' when
    only some cell types measured it would make the class largest exactly
    where the data are thinnest."""
    m = _mod()
    a = _frame([-1.0, -1.0, -1.0])
    b = _frame([-1.0, -1.0])                      # third bin never measured
    part, _, incomplete = m.rt_partition([a, b])
    assert len(part) == 2, "the unmeasured bin was classified anyway"
    assert incomplete == 1
    assert set(part["class"]) == {"RT_constitutive_late"}


def test_the_dead_zone_keeps_near_zero_bins_out_of_variable():
    """At a bare sign cut, -0.02 in one cell type and +0.02 in another reads as
    variable; that is noise, not biology."""
    m = _mod()
    frames = [_frame([-0.02, -1.5]), _frame([+0.02, -1.4])]
    bare, _, _ = m.rt_partition(frames, threshold=0.0)
    assert bare["class"].iloc[0] == "RT_variable"
    damped, counts, _ = m.rt_partition(frames, threshold=0.25)
    assert damped["class"].iloc[0] == "RT_unclassified"
    assert damped["class"].iloc[1] == "RT_constitutive_late"


def test_the_dead_zone_makes_the_classes_non_exhaustive_visibly():
    """With t > 0 the three sets stop partitioning; the count has to be
    reported rather than left implied."""
    m = _mod()
    part, counts, _ = m.rt_partition([_frame([0.0]), _frame([0.0])],
                                     threshold=0.5)
    assert counts.get("RT_unclassified") == 1
    assert sum(v for k, v in counts.items() if k != "RT_unclassified") == 0


def test_partition_is_the_lad_partition_with_late_in_place_of_lad():
    """Same shape, so the two stay consistent: constitutive = late everywhere,
    the third class = late nowhere, variable = the rest."""
    m = _mod()
    spec = importlib.util.spec_from_file_location(
        "damid", Path(__file__).resolve().parent.parent
        / "scripts" / "fetch_4dn_damid.py")
    d = importlib.util.module_from_spec(spec)
    sys.modules["damid"] = d
    spec.loader.exec_module(d)

    vals = [[-1.0, +1.0, -1.0], [-1.0, +1.0, +1.0]]
    part, _, _ = m.rt_partition([_frame(v) for v in vals])
    con, var, never = d.lad_partition(
        {str(i): np.array(v) < 0 for i, v in enumerate(vals)})
    assert list(con) == [c == "RT_constitutive_late" for c in part["class"]]
    assert list(var) == [c == "RT_variable" for c in part["class"]]
    assert list(never) == [c == "RT_constitutive_early" for c in part["class"]]


# --------------------------------------------------------- consensus -------

def test_consensus_is_the_mean_across_cell_types():
    """A boundary from one cell type is that cell type's boundary, and half
    the genome changes timing between them."""
    m = _mod()
    a = _frame([1.0, -1.0, 0.5])
    b = _frame([1.0, -1.0, -1.5])
    got = m.consensus_timing([a, b])
    np.testing.assert_allclose(got.value.to_numpy(), [1.0, -1.0, -0.5])


def test_consensus_uses_only_bins_every_cell_type_measured():
    """A mean over whichever cell types happen to cover a bin is strongest
    where the data is thinnest."""
    m = _mod()
    a = _frame([1.0, 1.0, 1.0])
    b = _frame([1.0, 1.0])                       # third bin never measured
    got = m.consensus_timing([a, b])
    assert len(got) == 2


def test_consensus_returns_none_when_nothing_is_shared():
    m = _mod()
    a = _frame([1.0])
    b = _frame([1.0])
    b["start"] = b["start"] + 7                  # a different grid entirely
    assert m.consensus_timing([a, b]) is None


def test_the_consensus_sign_splits_early_from_late():
    """The edges of the positive set are the transition the cell types agree
    on -- which is what a boundary profile should be measured against."""
    m = _mod()
    frames = [_frame([2.0, 2.0, -2.0, -2.0]), _frame([1.0, 3.0, -1.0, -3.0])]
    cons = m.consensus_timing(frames)
    early = cons.value > 0
    assert list(early) == [True, True, False, False]
