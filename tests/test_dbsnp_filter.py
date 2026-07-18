"""dbSNP filtering must never silently no-op.

The original loader required a tabix index and returned an EMPTY set on any
failure, so an unindexed .vcf or a 'chr1' vs '1' naming mismatch disabled the
filter without a word — an unfiltered run then looks identical to a filtered
one.  These tests pin the three cases that must work and the one that must be
loud.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "site_distance_correlation.py")


def _mod():
    spec = importlib.util.spec_from_file_location("sdc_snp", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sdc_snp"] = m
    spec.loader.exec_module(m)
    return m


_HEADER = ("##fileformat=VCFv4.2\n"
           "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")


def _vcf(path: Path, chrom: str, positions_1based) -> Path:
    path.write_text(_HEADER + "".join(
        f"{chrom}\t{p}\trs{p}\tC\tT\t.\t.\t.\n" for p in positions_1based))
    return path


def _ctx(positions):
    arr = np.zeros(len(positions), dtype=[("pos", "<i8"), ("ref_base", "u1")])
    arr["pos"] = positions
    arr["ref_base"] = 67
    return {"GCH": {"chr1": arr.copy()}, "HCG": {"chr1": arr.copy()}}


def test_unindexed_vcf_still_filters(tmp_path):
    """The reported failure: a plain .vcf has no index, so fetch() raises."""
    m = _mod()
    vcf = _vcf(tmp_path / "plain.vcf", "chr1", [1001, 1011, 1021])
    out = m._drop_snp_sites(_ctx([1000, 1010, 1020, 1030]), str(vcf))
    assert list(out["GCH"]["chr1"]["pos"]) == [1030]
    assert list(out["HCG"]["chr1"]["pos"]) == [1030]


def test_indexed_vcf_filters(tmp_path):
    pysam = pytest.importorskip("pysam")
    m = _mod()
    plain = _vcf(tmp_path / "v.vcf", "chr1", [1001, 1011, 1021])
    gz = tmp_path / "v.vcf.gz"
    pysam.tabix_compress(str(plain), str(gz), force=True)
    pysam.tabix_index(str(gz), preset="vcf", force=True)
    out = m._drop_snp_sites(_ctx([1000, 1010, 1020, 1030]), str(gz))
    assert list(out["GCH"]["chr1"]["pos"]) == [1030]


def test_chrom_naming_mismatch_is_reconciled(tmp_path):
    """VCF says '1', the context BED says 'chr1' — very common for mouse."""
    m = _mod()
    vcf = _vcf(tmp_path / "nochr.vcf", "1", [1001, 1011])
    out = m._drop_snp_sites(_ctx([1000, 1010, 1020, 1030]), str(vcf))
    assert list(out["GCH"]["chr1"]["pos"]) == [1020, 1030]


def test_vcf_is_one_based_context_is_zero_based(tmp_path):
    """VCF POS 1001 must map to the 0-based context site 1000, not 1001.

    Uses motif_flank=0 so this isolates the coordinate convention; with the
    default 3-bp motif span the neighbour 1001 is (correctly) dropped too.
    """
    m = _mod()
    vcf = _vcf(tmp_path / "off.vcf", "chr1", [1001])
    out = m._drop_snp_sites(_ctx([1000, 1001]), str(vcf), motif_flank=0)
    assert list(out["GCH"]["chr1"]["pos"]) == [1001]


def test_snp_anywhere_in_the_motif_drops_the_site(tmp_path):
    """GCH/HCG are 3-bp motifs: a SNP at the flanking G or H invalidates the
    site just as much as one at the cytosine (a GCH whose G is polymorphic may
    really be an HCH, so its accessibility call is meaningless)."""
    m = _mod()
    # cytosines at 0-based 1000/2000/3000/4000; SNPs (1-based) hit:
    #   1000 -> 0-based  999 = the 5' flank of site 1000
    #   2001 -> 0-based 2000 = the cytosine of site 2000
    #   3002 -> 0-based 3001 = the 3' flank of site 3000
    vcf = _vcf(tmp_path / "motif.vcf", "chr1", [1000, 2001, 3002])
    out = m._drop_snp_sites(_ctx([1000, 2000, 3000, 4000]), str(vcf))
    assert list(out["GCH"]["chr1"]["pos"]) == [4000]


def test_cytosine_only_flank_would_be_missed(tmp_path):
    """Pins WHY the motif span matters: flank SNPs survive a centre-only check."""
    m = _mod()
    vcf = _vcf(tmp_path / "flank.vcf", "chr1", [1000, 3002])   # both flanks
    centre_only = m._drop_snp_sites(_ctx([1000, 3000, 4000]), str(vcf),
                                    motif_flank=0)
    assert list(centre_only["GCH"]["chr1"]["pos"]) == [1000, 3000, 4000]
    full = m._drop_snp_sites(_ctx([1000, 3000, 4000]), str(vcf))
    assert list(full["GCH"]["chr1"]["pos"]) == [4000]


def test_no_matching_chromosome_exits_loudly(tmp_path):
    """Zero usable SNPs must abort, not quietly run unfiltered."""
    m = _mod()
    vcf = _vcf(tmp_path / "other.vcf", "chrZZ", [5, 6])
    with pytest.raises(SystemExit) as e:
        m._drop_snp_sites(_ctx([1000, 1010]), str(vcf))
    assert "no SNP positions matched" in str(e.value)


def test_reference_loader_warns_instead_of_silent_empty(tmp_path, caplog):
    """The main pipeline's loader keeps its empty-set contract but now warns."""
    pytest.importorskip("pysam")
    from nome_hic_concordance.reference import _load_snp_positions

    vcf = _vcf(tmp_path / "plain.vcf", "chr1", [1001])
    with caplog.at_level("WARNING"):
        got = _load_snp_positions(str(vcf), "chr1")
    assert got == set()                       # unchanged behaviour
    assert "NO SNPs will be filtered" in caplog.text
