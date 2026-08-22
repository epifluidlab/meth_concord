"""Getting chrM through context extraction.

chrM is excluded from the default contig set, and rightly so -- it is 16.6 kb
and nobody wants bins there.  But it is the one sequence carrying no
nucleosomes, which makes it the reference for per-cell assay reliability that
does NOT assume the tumour left its chromatin alone the way a cLAD or
replication-timing reference does.  So it needs a way in that does not force
the caller to re-specify every other contig.
"""

from __future__ import annotations

import pytest

from nome_hic_concordance.reference import select_chroms
from nome_hic_concordance.utils import parse_include_chroms

ALL = ["chr1", "chr2", "chrX", "chrY", "chrM", "chr1_KI270706v1_random"]


def test_the_default_set_is_unchanged_when_nothing_is_added():
    assert select_chroms(ALL) == ["chr1", "chr2", "chrX", "chrY"]


def test_include_chrom_adds_rather_than_replaces():
    """The whole point: asking for chrM must not cost you the autosomes."""
    assert select_chroms(ALL, extra_chroms=["chrM"]) == [
        "chr1", "chr2", "chrX", "chrY", "chrM"]


def test_an_explicit_chroms_list_still_wins():
    assert select_chroms(ALL, chroms=["chr1", "chrM"]) == ["chr1", "chrM"]
    # extra_chroms is redundant once chroms names everything wanted
    assert select_chroms(ALL, chroms=["chr1"], extra_chroms=["chrM"]) == ["chr1"]


def test_a_name_the_reference_does_not_use_is_an_error():
    """'MT' and 'chrM' are the same sequence to a BAM header but not to a
    2bit.  Accepting the wrong spelling silently would extract nothing and
    only surface hours later, at the reference step, far from the cause."""
    with pytest.raises(ValueError, match="not in the reference"):
        select_chroms(ALL, extra_chroms=["MT"])


def test_asking_twice_does_not_duplicate_the_work():
    assert select_chroms(ALL, extra_chroms=["chrM", "chrM"]).count("chrM") == 1
    assert select_chroms(ALL, extra_chroms=["chr1"]) == select_chroms(ALL)


def test_alt_contigs_stay_out():
    assert not any("_" in c for c in select_chroms(ALL, extra_chroms=["chrM"]))


@pytest.mark.parametrize("given,want", [
    (None, []),
    ("chrM", ["chrM"]),
    (["chrM,chrX", "chrY"], ["chrM", "chrX", "chrY"]),
    (["chrM", "chrM"], ["chrM"]),
    (["  chrM  "], ["chrM"]),
])
def test_the_flag_parses_repeated_and_comma_separated(given, want):
    assert parse_include_chroms(given) == want


def test_order_is_preserved_so_the_reference_reads_as_written():
    assert parse_include_chroms(["chrY,chrM"]) == ["chrY", "chrM"]
