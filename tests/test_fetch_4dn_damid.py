"""4DN LaminB1 DamID fetcher.

Network-free: the portal query and the download are separated from the
selection and partition logic, and only the latter is tested here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "fetch_4dn_damid.py")


def _mod():
    spec = importlib.util.spec_from_file_location("f4dn", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["f4dn"] = m
    spec.loader.exec_module(m)
    return m


def _rec(cell, condition="LaminB1-Dam", assembly="GRCh38",
         desc="250kb bin dam-only normalized track - mean of replicates X",
         acc="4DNFTEST"):
    return {"accession": acc, "genome_assembly": assembly,
            "description": desc, "open_data_url": f"https://s3/{acc}.bw",
            "md5sum": "abc", "file_size": 1,
            "track_and_facet_info": {"condition": condition,
                                     "biosource_name": cell,
                                     "experiment_type": "DamID-seq"}}


def test_only_laminb1_is_selected():
    """The same 4DN collection carries nucleolar-targeting DamID, which gives
    NADs -- overlapping LADs substantially, but not the same thing."""
    m = _mod()
    got = m.select([
        _rec("HFFc6 (Tier 1)", acc="4DNLAM"),
        _rec("HFFc6 (Tier 1)", condition="Nucleolar Targeting Signal (4X-AP3)-Dam",
             acc="4DNNAD"),
        _rec("HFFc6 (Tier 1)", condition="Dam-only Control", acc="4DNCTL")])
    assert list(got) == ["HFFc6"]
    assert got["HFFc6"]["accession"] == "4DNLAM"


def test_the_bin_size_and_assembly_are_honoured():
    m = _mod()
    recs = [
        _rec("K562 (Tier 2)", acc="4DN250"),
        _rec("K562 (Tier 2)", acc="4DN100",
             desc="100kb bin dam-only normalized track - mean of replicates X"),
        _rec("K562 (Tier 2)", acc="4DNMM", assembly="GRCm38"),
    ]
    assert m.select(recs)["K562"]["accession"] == "4DN250"
    assert m.select(recs, bin_size="100kb")["K562"]["accession"] == "4DN100"
    assert m.select(recs, assembly="GRCm38")["K562"]["accession"] == "4DNMM"


def test_replicate_means_are_preferred_and_raw_counts_excluded():
    m = _mod()
    per_rep = _rec("U2OS", acc="4DNREP",
                   desc="250kb bin dam-only normalized track - U2OS_r1_LMNB1")
    assert m.select([per_rep]) == {}, "a single replicate is not the mean"
    assert m.select([per_rep], replicate_mean=False)["U2OS"]["accession"] \
        == "4DNREP"
    raw = _rec("U2OS", acc="4DNRAW", desc="250kb bin raw counts")
    assert m.select([raw]) == {}, "raw counts are not Dam-normalised"


def test_the_tier_suffix_is_stripped_from_the_cell_name():
    m = _mod()
    assert list(m.select([_rec("RPE-hTERT (Tier 2)")])) == ["RPE-hTERT"]


def test_a_record_without_an_open_mirror_says_so(tmp_path):
    """The portal's own download URL 403s without an account; open_data_url is
    what makes this work unauthenticated."""
    m = _mod()
    rec = _rec("K562")
    del rec["open_data_url"]
    with pytest.raises(SystemExit, match="no open_data_url"):
        m.download(rec, tmp_path / "x.bw")


# ------------------------------------------------------------- partition ----

def test_lad_partition_is_constitutive_facultative_inter():
    m = _mod()
    calls = {                      # 5 bins x 3 cell types
        "a": np.array([1, 1, 0, 0, 1], bool),
        "b": np.array([1, 0, 1, 0, 1], bool),
        "c": np.array([1, 0, 0, 0, 1], bool)}
    con, fac, inter = m.lad_partition(calls)
    assert list(con) == [True, False, False, False, True]
    assert list(fac) == [False, True, True, False, False]
    assert list(inter) == [False, False, False, True, False]
    # A partition: every bin in exactly one class.
    assert (con.astype(int) + fac + inter == 1).all()


def test_facultative_is_the_union_minus_constitutive():
    m = _mod()
    calls = {"a": np.array([1, 0], bool), "b": np.array([1, 1], bool)}
    con, fac, inter = m.lad_partition(calls)
    union = calls["a"] | calls["b"]
    np.testing.assert_array_equal(fac, union & ~con)
    assert not inter.any(), "every bin is a LAD somewhere"


def test_one_cell_type_makes_everything_constitutive():
    """cLAD shrinks as cell types are added, so the sets are not comparable
    between runs with different --cell-type lists."""
    m = _mod()
    calls = {"only": np.array([1, 0, 1], bool)}
    con, fac, inter = m.lad_partition(calls)
    assert list(con) == [True, False, True]
    assert not fac.any()


def test_write_bed_emits_three_columns_and_counts(tmp_path):
    m = _mod()
    bins = pd.DataFrame({"chrom": ["chr1"] * 3,
                         "start": [0, 250000, 500000],
                         "end": [250000, 500000, 750000]})
    n = m.write_bed(bins, np.array([True, False, True]), tmp_path / "x.bed")
    assert n == 2
    got = pd.read_csv(tmp_path / "x.bed", sep="\t", header=None)
    assert got.shape == (2, 3)
    assert list(got[1]) == [0, 500000]
