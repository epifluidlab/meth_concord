"""The duplicate flag means something different on chrM.

mtDNA is present in hundreds to thousands of copies per cell, so on a 16.6 kb
sequence two reads sharing start and end coordinates collide by CHANCE, not
by PCR.  Honouring the marker's calls there costs more than depth:
deduplication keeps one read per coordinate, which collapses molecules that
differ in methylation into a single observation -- destroying exactly the
between-molecule variance (var_A) that a reliability reference measures.

The other half of the requirement is that nothing else changes.  Every
nuclear contig must come out byte-identical to a run without the option, and
keep the fast C++ path.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

pysam = pytest.importorskip("pysam")

from nome_hic_concordance import bam_parser as bp
from nome_hic_concordance.cli import _keep_dup_chroms

_READ_LEN = 60


def _write_bam(path, spec, lens=(("chr1", 100_000), ("chrM", 16_569))):
    """spec: list of (chrom, p1, p2, is_duplicate)."""
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"},
           "SQ": [{"SN": c, "LN": n} for c, n in lens]}
    names = [c for c, _ in lens]
    recs = []
    for k, (chrom, p1, p2, dup) in enumerate(spec):
        tid = names.index(chrom)
        for pos, mpos, base in ((p1, p2, 99), (p2, p1, 147)):
            a = pysam.AlignedSegment()
            a.query_name = f"r{k}"
            a.flag = base | (0x400 if dup else 0)
            a.reference_id = tid
            a.reference_start = pos
            a.mapping_quality = 60
            a.cigarstring = f"{_READ_LEN}M"
            a.query_sequence = "T" + "C" * (_READ_LEN - 1)
            a.query_qualities = pysam.qualitystring_to_array("I" * _READ_LEN)
            a.next_reference_id = tid
            a.next_reference_start = mpos
            a.set_tag("MD", f"0C{_READ_LEN - 1}")
            recs.append(a)
    with pysam.AlignmentFile(str(path), "wb", header=hdr) as out:
        for a in sorted(recs, key=lambda x: (x.reference_id,
                                             x.reference_start)):
            out.write(a)
    pysam.index(str(path))
    return str(path)


def _ctx(lens=(("chr1", 100_000), ("chrM", 16_569))):
    out = {}
    for c, n in lens:
        pos = np.arange(0, n, 2, dtype=np.int64)
        out[c] = np.zeros(len(pos), dtype=[("pos", "<i8"), ("ref_base", "u1")])
        out[c]["pos"] = pos
        out[c]["ref_base"] = ord("C")
    return out


def _spec():
    """Four pairs per contig, half of them flagged duplicate."""
    s = []
    for chrom, base in (("chr1", 1_000), ("chrM", 1_000)):
        for i, dup in enumerate([False, False, True, True]):
            s.append((chrom, base + i * 300, base + i * 300 + 2_000, dup))
    return s


def _counts(df):
    return df["chrom"].value_counts().to_dict() if len(df) else {}


# ------------------------------------------------------------ the exemption --

def test_without_the_option_duplicates_are_dropped_everywhere(tmp_path):
    bam = _write_bam(tmp_path / "a.bam", _spec())
    df = bp.parse_cell_bam(bam, _ctx(), trim_ends=0, mapq_min=0)
    c = _counts(df)
    assert c.get("chr1") == 2 and c.get("chrM") == 2


def test_the_exemption_keeps_chrM_duplicates(tmp_path):
    bam = _write_bam(tmp_path / "a.bam", _spec())
    df = bp.parse_cell_bam(bam, _ctx(), trim_ends=0, mapq_min=0,
                           keep_duplicates_chroms={"M"})
    c = _counts(df)
    assert c.get("chrM") == 4, "chrM duplicates should have been kept"
    assert c.get("chr1") == 2, "chr1 must be unaffected"


def test_every_other_contig_is_bit_for_bit_unchanged(tmp_path):
    """The stated requirement: different criteria for chrM, the same for all
    other chromosomes.  Splitting the parse rather than relaxing the flag
    mask globally is what buys this."""
    bam = _write_bam(tmp_path / "a.bam", _spec())
    plain = bp.parse_cell_bam(bam, _ctx(), trim_ends=0, mapq_min=0)
    split = bp.parse_cell_bam(bam, _ctx(), trim_ends=0, mapq_min=0,
                              keep_duplicates_chroms={"M"})
    a = plain[plain["chrom"] == "chr1"].reset_index(drop=True)
    b = split[split["chrom"] == "chr1"].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_chrom_naming_variants_all_select_the_same_contig(tmp_path):
    bam = _write_bam(tmp_path / "a.bam", _spec())
    from nome_hic_concordance.utils import chrom_key
    for name in ("chrM", "M", "MT"):
        df = bp.parse_cell_bam(bam, _ctx(), trim_ends=0, mapq_min=0,
                               keep_duplicates_chroms={chrom_key(name)})
        assert _counts(df).get("chrM") == 4, f"{name} did not match"


def test_an_exemption_naming_no_present_contig_changes_nothing(tmp_path):
    bam = _write_bam(tmp_path / "a.bam", _spec())
    plain = bp.parse_cell_bam(bam, _ctx(), trim_ends=0, mapq_min=0)
    other = bp.parse_cell_bam(bam, _ctx(), trim_ends=0, mapq_min=0,
                              keep_duplicates_chroms={"22"})
    pd.testing.assert_frame_equal(plain.reset_index(drop=True),
                                  other.reset_index(drop=True))


def test_the_python_backend_honours_the_flag_when_not_exempt(tmp_path):
    """Guard against the exemption leaking into the shared code path."""
    bam = _write_bam(tmp_path / "a.bam", _spec())
    kept = bp._parse_cell_bam_python(bam, _ctx(), trim_ends=0, mapq_min=0,
                                     keep_duplicates=True)
    dropped = bp._parse_cell_bam_python(bam, _ctx(), trim_ends=0, mapq_min=0,
                                        keep_duplicates=False)
    assert len(kept) == 8 and len(dropped) == 4


# ------------------------------------------------------------- the defaults --

def _args(**kw):
    kw.setdefault("include_chrom", None)
    kw.setdefault("keep_duplicates_chrom", None)
    return argparse.Namespace(**kw)


def test_including_chrM_keeps_its_duplicates_without_a_second_flag():
    """A deduplicated reference is not merely thinner, it is wrong -- so the
    quiet failure must not be the easy one to hit."""
    assert _keep_dup_chroms(_args(include_chrom=["chrM"])) == {"M"}


def test_a_run_that_adds_no_contig_behaves_exactly_as_before():
    assert _keep_dup_chroms(_args()) == set()


def test_opting_out_is_possible_and_explicit():
    assert _keep_dup_chroms(
        _args(include_chrom=["chrM"], keep_duplicates_chrom=["none"])) == set()


def test_an_explicit_list_overrides_the_default():
    assert _keep_dup_chroms(
        _args(include_chrom=["chrM"],
              keep_duplicates_chrom=["chrY"])) == {"Y"}
    assert _keep_dup_chroms(
        _args(keep_duplicates_chrom=["chrM,chrY"])) == {"M", "Y"}
