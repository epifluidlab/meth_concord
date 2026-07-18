"""Tests for the per-site cytosine extractor and the distance-correlation math.

The extractor must agree EXACTLY with bam_parser's aggregate counter — a read
that contributes (k, n) upstream must yield n sites here of which k are
methylated — otherwise the site-level curves describe different data than the
rest of the pipeline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from nome_hic_concordance.bam_parser import _count_context_in_read
from nome_hic_concordance.site_calls import site_calls_in_read

from .test_bam_parser import FakeRead

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "site_distance_correlation.py")


def _mod():
    spec = importlib.util.spec_from_file_location("sdc", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sdc"] = m
    spec.loader.exec_module(m)
    return m


def _read(ref_seq: str, query_seq: str, start: int = 0):
    pairs = [(i, start + i) for i in range(len(query_seq))]
    return FakeRead(reference_start=start, query_sequence=query_seq,
                    aligned_pairs=pairs, reference_sequence=ref_seq)


# --------------------------------------------------------------------------
# The extractor agrees with the aggregate counter
# --------------------------------------------------------------------------

def test_site_calls_match_aggregate_counts():
    #                0123456789
    ref = "AACGAACGAA"
    # C→T read: pos2 C stays methylated, pos6 C converted to T
    qry = "AACGAATGAA"
    r = _read(ref, qry)
    ctx_pos = np.array([2, 6], dtype=np.int64)
    ctx_ref = np.array([67, 67], dtype=np.uint8)      # both ref=C

    k, n = _count_context_in_read(r, ctx_pos, ctx_ref, trim_ends=0)
    pos, st = site_calls_in_read(r, ctx_pos, ctx_ref, trim_ends=0)
    assert n == pos.size, "site count must equal the aggregate n"
    assert k == int(st.sum()), "methylated count must equal the aggregate k"
    assert list(pos) == [2, 6]
    assert list(st) == [1, 0]                          # meth, unmeth


def test_site_calls_respect_trim_ends():
    ref = "AACGAACGAA"
    qry = "AACGAATGAA"          # C→T at pos 6 — also fixes the strand call
    r = _read(ref, qry)
    ctx_pos = np.array([2, 6], dtype=np.int64)
    ctx_ref = np.array([67, 67], dtype=np.uint8)
    pos_all, _ = site_calls_in_read(r, ctx_pos, ctx_ref, trim_ends=0)
    pos_trim, _ = site_calls_in_read(r, ctx_pos, ctx_ref, trim_ends=3)
    assert list(pos_all) == [2, 6]
    assert list(pos_trim) == [6]      # pos 2 falls inside the trimmed margin


def test_site_calls_skip_wrong_strand_sites():
    """A C→T read must not score ref=G sites (they'd look falsely methylated)."""
    ref = "AACGAACGAA"
    qry = "AATGAACGAA"                                 # a C→T conversion present
    r = _read(ref, qry)
    ctx_pos = np.array([2, 3], dtype=np.int64)
    ctx_ref = np.array([67, 71], dtype=np.uint8)       # one ref=C, one ref=G
    pos, _ = site_calls_in_read(r, ctx_pos, ctx_ref, trim_ends=0)
    assert list(pos) == [2], "only the ref=C site is informative for a C→T read"


def test_site_calls_empty_when_strand_undeterminable():
    ref = "AACGAACGAA"
    r = _read(ref, ref)                                # no mismatches at all
    ctx_pos = np.array([2, 6], dtype=np.int64)
    ctx_ref = np.array([67, 67], dtype=np.uint8)
    k, n = _count_context_in_read(r, ctx_pos, ctx_ref, trim_ends=0)
    pos, st = site_calls_in_read(r, ctx_pos, ctx_ref, trim_ends=0)
    assert (k, n) == (0, 0) and pos.size == 0 and st.size == 0


# --------------------------------------------------------------------------
# Distance binning / pairing / correlation math
# --------------------------------------------------------------------------

def test_pair_sites_within_read_is_upper_triangle():
    m = _mod()
    pos = np.array([0, 10, 100], np.int64)
    st = np.array([1, 0, 1], np.uint8)
    edges = m._build_edges(1, 1000, 3)
    rng = np.random.default_rng(0)
    bins, x, y = m._pair_sites(pos, st, pos, st, edges, True, rng, 0)
    assert bins.size == 3            # 3 choose 2 — never a site with itself
    # Full cross product is 9, but the 3 diagonal (i==j) entries have distance 0,
    # which is below --min-distance and so is filtered out: a site is never
    # correlated against itself even on the cross-product path.
    b2, _, _ = m._pair_sites(pos, st, pos, st, edges, False, rng, 0)
    assert b2.size == 6


def test_pair_sites_cap_bounds_the_blowup():
    m = _mod()
    pos = np.arange(40, dtype=np.int64) * 3
    st = (np.arange(40) % 2).astype(np.uint8)
    edges = m._build_edges(1, 1000, 3)
    rng = np.random.default_rng(0)
    bins, _, _ = m._pair_sites(pos, st, pos, st, edges, True, rng, cap=25)
    assert bins.size <= 25


def test_corr_matches_numpy_pearson():
    m = _mod()
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, 400).astype(np.uint8)
    y = ((x + rng.integers(0, 2, 400)) % 2).astype(np.uint8)
    acc = m._new_acc(1)
    m._accumulate(acc, np.zeros(400, np.int64), x, y)
    r, n = m._corr(acc, min_pairs=1)
    assert n[0] == 400
    assert abs(r[0] - np.corrcoef(x, y)[0, 1]) < 1e-9


def test_corr_respects_min_pairs():
    m = _mod()
    acc = m._new_acc(1)
    x = np.array([1, 0, 1, 0], np.uint8)
    m._accumulate(acc, np.zeros(4, np.int64), x, x)
    assert np.isnan(m._corr(acc, min_pairs=100)[0][0])   # too few → NaN
    assert m._corr(acc, min_pairs=1)[0][0] == pytest.approx(1.0)


def test_downsampling_is_genome_wide_not_truncated():
    """--max-pairs-per-cell must sample uniformly across each chromosome.

    Truncating (stop once the quota fills) would only ever return the START of
    the chromosome — a positional bias in a genome-wide statistic.
    """
    pysam = pytest.importorskip("pysam")
    bam_path = Path("/tmp/sdc/cell1.sorted.bam")
    if not bam_path.exists():
        pytest.skip("synthetic fixture BAM not present")
    from nome_hic_concordance.cli import _parse_context_bed
    from nome_hic_concordance.reference import load_context_from_bed
    from nome_hic_concordance.site_calls import iter_cell_read_pairs

    cp = load_context_from_bed(
        _parse_context_bed("GCH:/tmp/sdc/GCH.bed.gz,HCG:/tmp/sdc/HCG.bed.gz"),
        n_threads=1, c_offset=0)
    full = [p[1] for p in iter_cell_read_pairs(str(bam_path), cp)]
    sub = [p[1] for p in iter_cell_read_pairs(
        str(bam_path), cp, max_pairs_per_cell=len(full) // 4,
        rng=np.random.default_rng(0))]
    assert 0 < len(sub) < len(full)
    # the subsample must span nearly the same positional range as the full set,
    # not cluster at the chromosome start
    assert max(sub) > 0.8 * max(full), "sample looks truncated to the start"
    assert np.median(sub) > 0.5 * np.median(full)


def test_htslib_backend_matches_pysam_exactly():
    """The C++ fast path must be bit-identical to the validated pysam path.

    Uncapped (cap=0) both enumerate every site pair, so the LINKED accumulators
    are deterministic and must match exactly; the null draws a random partner
    and so is compared only for shape.
    """
    pytest.importorskip("pysam")
    import pysam  # noqa: F401  — loads the bundled libhts the ext resolves against
    try:
        from nome_hic_concordance import _site_dist_core as cpp
    except Exception:
        pytest.skip("htslib extension not built")
    bam = Path("/tmp/sdc/cell1.sorted.bam")
    if not bam.exists():
        pytest.skip("synthetic fixture BAM not present")
    from nome_hic_concordance.cli import _parse_context_bed
    from nome_hic_concordance.reference import load_context_from_bed

    m = _mod()
    cp = load_context_from_bed(
        _parse_context_bed("GCH:/tmp/sdc/GCH.bed.gz,HCG:/tmp/sdc/HCG.bed.gz"),
        n_threads=1, c_offset=0)
    edges = m._build_edges(1, 100_000, 3)
    ctx_g = {c: (a["pos"].astype(np.int64), a["ref_base"].astype(np.uint8))
             for c, a in cp["GCH"].items()}
    ctx_h = {c: (a["pos"].astype(np.int64), a["ref_base"].astype(np.uint8))
             for c, a in cp["HCG"].items()}

    res = cpp.parse_cell_site_distance(str(bam), ctx_g, ctx_h,
                                       [float(e) for e in edges],
                                       5, 30, 100, 0, 42, False)
    accs_py, n_py, elig_py = m.process_cell(str(bam), cp, edges, 5, 30, 0, 0,
                                            100, 42)
    assert int(res["n_pairs"]) == n_py
    assert int(res["n_null_eligible"]) == elig_py
    for key in ("gg", "hh", "gh"):
        a = np.asarray(res[f"{key}_linked"])
        b = accs_py[(key, "linked")]
        assert a.shape == b.shape
        assert np.allclose(a, b), f"{key}_linked differs between backends"
        assert a[0].sum() > 0, f"{key}: no site pairs accumulated"
        assert np.asarray(res[f"{key}_null"]).shape == b.shape
        # 'matched' is the true partner on the null-eligible subset, so with no
        # site-pair cap it involves no RNG and must agree exactly.
        assert np.allclose(np.asarray(res[f"{key}_matched"]),
                           accs_py[(key, "matched")]), \
            f"{key}_matched differs between backends"


def _qual_bam(tmp_path):
    """BAM where basecall quality varies INDEPENDENTLY of methylation state, so
    a baseq filter removes sites without removing one state entirely."""
    pysam = pytest.importorskip("pysam")
    chrom, clen = "chr1", 5000
    sites = list(range(1000, 1200, 10)) + list(range(2000, 2200, 10))
    sset = set(sites)
    ref = bytearray(b"A" * clen)
    for p in sites:
        ref[p] = ord("C")
    ref = ref.decode()
    fa = tmp_path / "ref.fa"
    fa.write_text(f">{chrom}\n" + "\n".join(ref[i:i + 60]
                                            for i in range(0, clen, 60)) + "\n")
    pysam.faidx(str(fa))
    hdr = {"HD": {"VN": "1.6"}, "SQ": [{"SN": chrom, "LN": clen}]}
    raw = tmp_path / "raw.bam"
    with pysam.AlignmentFile(raw, "wb", header=hdr) as bam:
        for i in range(50):
            for st, is_r1 in ((1000, True), (2000, False)):
                a = pysam.AlignedSegment()
                a.query_name = f"r{i}"; a.reference_id = 0
                a.reference_start = st; a.mapping_quality = 60
                a.cigarstring = "200M"
                seq, quals = [], []
                for k in range(200):
                    p = st + k
                    if p in sset:
                        seq.append("C" if (p // 10) % 2 == 0 else "T")
                        quals.append(10 if (p % 20) == 0 else 40)
                    else:
                        seq.append(ref[p]); quals.append(40)
                a.query_sequence = "".join(seq)
                a.query_qualities = pysam.qualitystring_to_array(
                    "".join(chr(33 + q) for q in quals))
                a.flag = (0x1 | 0x40) if is_r1 else (0x1 | 0x80)
                a.next_reference_id = 0
                a.next_reference_start = 2000 if is_r1 else 1000
                bam.write(a)
    srt = tmp_path / "s.bam"
    pysam.sort("-o", str(srt), str(raw))
    out = tmp_path / "q.bam"
    with open(out, "wb") as fh:                       # calmd adds the MD tag
        fh.write(pysam.calmd("-b", str(srt), str(fa), catch_stdout=True))
    pysam.index(str(out))
    return out, np.array(sites, np.int64)


def test_baseq_min_drops_low_quality_sites(tmp_path):
    pytest.importorskip("pysam")
    import pysam  # noqa: F401 — loads bundled libhts for the extension
    try:
        from nome_hic_concordance import _site_dist_core as cpp
    except Exception:
        pytest.skip("htslib extension not built")
    bam, sites = _qual_bam(tmp_path)
    ctx = {"chr1": (sites, np.full(len(sites), 67, np.uint8))}
    edges = [1.0, 10.0, 100.0, 1000.0, 10000.0]

    def n_pairs(bq):
        r = cpp.parse_cell_site_distance(str(bam), ctx, {}, edges, 0, 30, 100,
                                         0, 42, False, bq, {})
        return float(np.asarray(r["gg_linked"])[0].sum())

    off, on = n_pairs(0), n_pairs(30)
    assert off > 0 and on > 0
    assert on < off, "--baseq-min did not drop any low-quality basecall"


def test_blacklist_drops_pairs(tmp_path):
    """A blacklist covering one end must drop the whole pair."""
    pytest.importorskip("pysam")
    import pysam  # noqa: F401
    try:
        from nome_hic_concordance import _site_dist_core as cpp
    except Exception:
        pytest.skip("htslib extension not built")
    bam, sites = _qual_bam(tmp_path)
    ctx = {"chr1": (sites, np.full(len(sites), 67, np.uint8))}
    edges = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    base = cpp.parse_cell_site_distance(str(bam), ctx, {}, edges, 0, 30, 100,
                                        0, 42, False, 0, {})
    bl = {"chr1": (np.array([1900], np.int64), np.array([2300], np.int64))}
    masked = cpp.parse_cell_site_distance(str(bam), ctx, {}, edges, 0, 30, 100,
                                          0, 42, False, 0, bl)
    assert int(base["n_pairs"]) > 0
    assert int(masked["n_pairs"]) == 0, "blacklist did not drop the pairs"


def test_build_edges_exact_bounds():
    m = _mod()
    e = m._build_edges(1, 1_000_000, 8)
    assert e[0] == 1 and e[-1] == 1_000_000
    assert len(e) == 6 * 8 + 1


def test_perfectly_correlated_sites_give_r_one():
    """End-to-end on the accumulator: identical states → phi = 1."""
    m = _mod()
    edges = m._build_edges(1, 1000, 3)
    rng = np.random.default_rng(0)
    accs = {("gg", "linked"): m._new_acc(len(edges) - 1)}
    pos = np.array([0, 50], np.int64)
    for s in (0, 1, 0, 1, 1, 0):                       # varying but identical
        st = np.array([s, s], np.uint8)
        b, x, y = m._pair_sites(pos, st, pos, st, edges, True, rng, 0)
        m._accumulate(accs[("gg", "linked")], b, x, y)
    r, n = m._corr(accs[("gg", "linked")], min_pairs=1)
    assert n.sum() == 6
    assert np.nanmax(r) == pytest.approx(1.0)
