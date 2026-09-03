"""Part A extraction: both marks on one read end, and HCH without a site table.

The point of the joint extractor is the pairing WITHIN a read end -- GCH and
HCG counts that came off the same molecule at the same locus.  Two calls to the
established single-context parser cannot produce that, so the tests here check
two things that matter more than the plumbing:

  * the new counts agree with the established extractor mark by mark, so
    nothing about context assignment changed while the record shape did; and
  * HCH derived from the read's own MD-reconstructed reference agrees, site for
    site, with HCH passed in as an ordinary site table -- because the site
    table is what the design assumed and the reconstruction is what replaces it.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

pysam = pytest.importorskip("pysam")


def _cpp():
    try:
        from nome_hic_concordance import _joint_state_core as j
    except Exception:                                    # pragma: no cover
        pytest.skip("htslib extension not built")
    return j


# --------------------------------------------------------------- fixture ---

REF_NAME = "chrT"


def _make_reference(length=3000, seed=7):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def _sites(ref, pattern):
    """Independent context scan: (positions, ref_base) for one 3-mer pattern.

    Deliberately not importing reference.py -- a check against the code that
    produced the arrays would prove nothing about either.

    The minus strand reads 3'->5' along the plus-strand coordinates, so the
    minus-strand 3-mer centred on the C complementary to ref[i]=='G' is
    ``comp(ref[i+1]) C comp(ref[i-1])``.  An H in the FIRST position of the
    pattern therefore constrains the base to ref[i]'s RIGHT, and it excludes
    'C' (whose complement is G), not 'G'.  Getting that backwards silently
    makes HCH overlap GCH, which is exactly what a conversion control must
    never do; test_context_scan_agrees_with_reference_module guards it.
    """
    pos, base = [], []
    for i in range(1, len(ref) - 1):
        prev, c, nxt = ref[i - 1], ref[i], ref[i + 1]
        if pattern == "GCH":
            plus = (c == "C" and prev == "G" and nxt != "G")
            minus = (c == "G" and nxt == "C" and prev != "C")   # DGC
        elif pattern == "HCG":
            plus = (c == "C" and prev != "G" and nxt == "G")
            minus = (c == "G" and prev == "C" and nxt != "C")   # CGD
        elif pattern == "HCH":
            plus = (c == "C" and prev != "G" and nxt != "G")
            minus = (c == "G" and prev != "C" and nxt != "C")   # DGD
        else:
            raise ValueError(pattern)
        if plus:
            pos.append(i); base.append(ord("C"))
        if minus:
            pos.append(i); base.append(ord("G"))
    order = np.argsort(np.array(pos, np.int64), kind="stable")
    return (np.array(pos, np.int64)[order], np.array(base, np.uint8)[order])


def _md(ref_seg, read_seq):
    """MD tag for an ungapped alignment."""
    out, run = [], 0
    for r, q in zip(ref_seg, read_seq):
        if r == q:
            run += 1
        else:
            out.append(str(run)); out.append(r); run = 0
    out.append(str(run))
    return "".join(out)


def _convert(ref_seg, strand, meth_frac, rng):
    """Bisulfite-convert a reference segment; returns (read_seq, n_meth)."""
    src, dst = ("C", "T") if strand == "CT" else ("G", "A")
    out, n_meth = [], 0
    for b in ref_seg:
        if b == src and rng.random() >= meth_frac:
            out.append(dst)
        else:
            if b == src:
                n_meth += 1
            out.append(b)
    return "".join(out), n_meth


def _write_bam(path, ref, pairs, read_len=100):
    """pairs: list of (start1, start2, strand1, strand2, meth1, meth2)."""
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"},
           "SQ": [{"SN": REF_NAME, "LN": len(ref)}]}
    rng = random.Random(11)
    recs = []
    for i, (s1, s2, st1, st2, m1, m2) in enumerate(pairs):
        for end, (start, strand, meth) in enumerate(
                ((s1, st1, m1), (s2, st2, m2))):
            seg = ref[start:start + read_len]
            seq, _ = _convert(seg, strand, meth, rng)
            a = pysam.AlignedSegment()
            a.query_name = f"r{i}"
            a.query_sequence = seq
            a.flag = (0x1 | 0x2 | (0x40 if end == 0 else 0x80))
            a.reference_id = 0
            a.reference_start = start
            a.mapping_quality = 60
            a.cigarstring = f"{read_len}M"
            a.next_reference_id = 0
            a.next_reference_start = s2 if end == 0 else s1
            a.template_length = (s2 - s1) if end == 0 else (s1 - s2)
            a.query_qualities = pysam.qualitystring_to_array("I" * read_len)
            a.set_tag("MD", _md(seg, seq), "Z")
            recs.append(a)
    recs.sort(key=lambda r: r.reference_start)
    with pysam.AlignmentFile(str(path), "wb", header=hdr) as out:
        for r in recs:
            out.write(r)
    pysam.index(str(path))
    return path


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    ref = _make_reference()
    d = tmp_path_factory.mktemp("jsc")
    rng = random.Random(3)
    pairs = []
    for _ in range(60):
        s1 = rng.randrange(0, 1200)
        s2 = rng.randrange(1600, 2800)
        pairs.append((s1, s2,
                      rng.choice(["CT", "GA"]), rng.choice(["CT", "GA"]),
                      rng.choice([0.0, 0.3, 0.8, 1.0]),
                      rng.choice([0.0, 0.3, 0.8, 1.0])))
    bam = _write_bam(d / "cell.bam", ref, pairs)
    ctx = {p: _sites(ref, p) for p in ("GCH", "HCG", "HCH")}
    return {"ref": ref, "bam": str(bam), "ctx": ctx}


def _run(fixture, **kw):
    j = _cpp()
    ctx = fixture["ctx"]
    args = dict(contexts=[{REF_NAME: ctx["GCH"]}, {REF_NAME: ctx["HCG"]}],
                trim_ends=5, mapq_min=30, min_distance=0)
    args.update(kw)
    return j.parse_cell_joint_ends(fixture["bam"], **args)


# ------------------------------------------------- reference reconstruction ---

def test_md_reconstruction_matches_pysam(fixture):
    """The reconstruction is the only source of HCH context, so pin it against
    an independent implementation on every read, not on one."""
    j = _cpp()
    with pysam.AlignmentFile(fixture["bam"]) as bam:
        reads = [r for r in bam if not (r.flag & 0x900)]
    assert len(reads) == 120
    for i, read in enumerate(reads):
        span, r2q = j.reference_span(fixture["bam"], i)
        assert span is not None, "MD present, reconstruction must succeed"
        assert span == read.get_reference_sequence().upper(), (
            f"read {i}: reconstructed reference differs from pysam's")
        assert len(r2q) == read.reference_length


def test_reconstruction_is_the_true_reference_not_the_read(fixture):
    """A bisulfite read differs from the reference at every converted C; if the
    reconstruction returned the read back, HCH would be scored on converted
    bases and the conversion control would read 100% by construction."""
    j = _cpp()
    with pysam.AlignmentFile(fixture["bam"]) as bam:
        reads = [r for r in bam if not (r.flag & 0x900)]
    ref = fixture["ref"]
    n_differ = 0
    for i, read in enumerate(reads):
        span, _ = j.reference_span(fixture["bam"], i)
        true = ref[read.reference_start:read.reference_end]
        assert span == true
        if span != read.query_sequence:
            n_differ += 1
    assert n_differ > len(reads) // 2, (
        "most bisulfite reads must differ from the reference they came from")


# --------------------------------------------------- agreement with existing ---

def test_per_mark_counts_match_the_established_extractor(fixture):
    """Same reads, same contexts, same k/n -- the record shape changed, the
    context assignment did not."""
    from nome_hic_concordance.bam_parser import parse_cell_bam
    from nome_hic_concordance.reference import CONTEXT_DTYPE

    out = _run(fixture)
    for ci, mark in enumerate(("GCH", "HCG")):
        pos, base = fixture["ctx"][mark]
        arr = np.empty(len(pos), dtype=CONTEXT_DTYPE)
        arr["pos"] = pos
        arr["ref_base"] = base
        old = parse_cell_bam(fixture["bam"], {REF_NAME: arr},
                             trim_ends=5, mapq_min=30)
        old = old.sort_values(["pos1", "pos2"]).reset_index(drop=True)

        k = np.asarray(out["k"][ci]); n = np.asarray(out["n"][ci])
        mid = (np.asarray(out["pos_5p"]), np.asarray(out["pos_3p"]))
        mid = (np.minimum(*mid) + np.maximum(*mid) + 1) // 2
        e1 = np.asarray(out["end_idx"]) == 1
        new = np.rec.fromarrays(
            [mid[e1], mid[~e1], k[e1], n[e1], k[~e1], n[~e1]],
            names="pos1,pos2,k1,n1,k2,n2")
        new = new[np.lexsort((new.pos2, new.pos1))]

        assert len(new) == len(old), f"{mark}: pair count differs"
        for col in ("pos1", "pos2", "k1", "n1", "k2", "n2"):
            assert np.array_equal(new[col], old[col].to_numpy()), \
                f"{mark}: {col} differs from the established extractor"


def test_conversion_strand_agrees_with_the_other_implementation(fixture):
    """bam_common.hpp is a third copy of infer_conversion_strand; pin it.

    test_per_mark_counts_match_the_established_extractor is the exact check --
    a strand disagreement moves k and n -- but it reports as a count mismatch.
    This one says which of the two it was.
    """
    from nome_hic_concordance import _bam_parser_core as bpc
    rows = bpc.debug_strand_inference(fixture["bam"], REF_NAME, max_reads=10_000)
    want = {"CT": 0, "GA": 0, "unknown": 0}
    for r in rows:
        want[r["strand"]] += 1

    out = _run(fixture)
    conv = np.asarray(out["conv_strand"])
    got = {"unknown": int((conv == 0).sum()),
           "CT": int((conv == 1).sum()), "GA": int((conv == 2).sum())}
    assert got == want, f"strand calls differ: {got} vs {want}"
    assert want["unknown"] > 0, (
        "a fully methylated read has no conversion mismatch and must land in "
        "unknown; without one the agreement above is untested on that branch")


def test_marks_are_mutually_exclusive_as_the_reference_module_defines_them(fixture):
    """Same rule reference.py applies: reverse-complement the pattern and match
    the plus-strand trinucleotide against it."""
    from nome_hic_concordance.reference import (
        _matches_context, _reverse_complement_context)
    ref = fixture["ref"]
    for mark in ("GCH", "HCG", "HCH"):
        minus = _reverse_complement_context(mark)
        want = []
        for i in range(len(ref) - 2):
            tri = ref[i:i + 3]
            if tri in ("GCG", "CGC"):
                continue
            if _matches_context(tri, mark):
                want.append((i + 1, ord("C")))
            if _matches_context(tri, minus):
                want.append((i + 1, ord("G")))
        want.sort()
        pos, base = fixture["ctx"][mark]
        got = sorted(zip(pos.tolist(), base.tolist()))
        assert got == want, f"{mark}: fixture site table disagrees with reference.py"


# ------------------------------------------------------------------- HCH ---

def test_hch_from_reference_equals_hch_from_a_site_table(fixture):
    """The design assumed a genome-wide HCH site table; this replaces it with a
    per-read derivation.  They must agree exactly, or the conversion control is
    measuring something else."""
    j = _cpp()
    ctx = fixture["ctx"]
    as_table = j.parse_cell_joint_ends(
        fixture["bam"],
        contexts=[{REF_NAME: ctx["GCH"]}, {REF_NAME: ctx["HCG"]},
                  {REF_NAME: ctx["HCH"]}],
        trim_ends=5, mapq_min=30, min_distance=0, count_hch=True)
    k_tab = np.asarray(as_table["k"][2])
    n_tab = np.asarray(as_table["n"][2])
    k_ref = np.asarray(as_table["k_hch"])
    n_ref = np.asarray(as_table["n_hch"])
    assert n_ref.sum() > 0, "the fixture must actually contain HCH sites"
    assert np.array_equal(n_tab, n_ref), "HCH site counts differ"
    assert np.array_equal(k_tab, k_ref), "HCH methylated counts differ"


def test_hch_measures_non_conversion(fixture):
    """k_hch/n_hch must track the simulated non-conversion rate, not the
    methylation of the CpG sites."""
    j = _cpp()
    ref = fixture["ref"]
    # Two cells: one fully converted, one 50% unconverted.
    for frac, lo, hi in ((0.0, 0.0, 0.02), (0.5, 0.4, 0.6)):
        rng = random.Random(5)
        pairs = [(rng.randrange(0, 1000), rng.randrange(1500, 2800),
                  "CT", "CT", frac, frac) for _ in range(40)]
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            bam = _write_bam(Path(d) / "c.bam", ref, pairs)
            out = j.parse_cell_joint_ends(
                str(bam), contexts=[{REF_NAME: fixture["ctx"]["GCH"]}],
                trim_ends=5, mapq_min=30, min_distance=0)
        k = np.asarray(out["k_hch"]).sum()
        n = np.asarray(out["n_hch"]).sum()
        assert n > 0
        assert lo <= k / n <= hi, (
            f"simulated non-conversion {frac}: measured {k / n:.3f}")


def test_hch_excludes_gch_and_hcg_sites(fixture):
    """HCH is the conversion control precisely because it is not a NOMe target.
    Any overlap with GCH or HCG contaminates it with real methylation."""
    ctx = fixture["ctx"]
    for other in ("GCH", "HCG"):
        a = set(zip(ctx["HCH"][0].tolist(), ctx["HCH"][1].tolist()))
        b = set(zip(ctx[other][0].tolist(), ctx[other][1].tolist()))
        assert not (a & b), f"HCH overlaps {other}"


def test_gcg_is_in_neither_mark(fixture):
    """GCG is ambiguous between the two marks and must be dropped from both,
    not assigned to whichever is being processed."""
    ref = fixture["ref"]
    ctx = fixture["ctx"]
    gcg = {i + 1 for i in range(len(ref) - 2) if ref[i:i + 3] == "GCG"}
    assert gcg, "fixture must contain GCG"
    for mark in ("GCH", "HCG", "HCH"):
        pos = set(ctx[mark][0].tolist())
        assert not (gcg & pos), f"{mark} contains a GCG cytosine"


# ------------------------------------------------------- the record itself ---

def test_both_marks_land_on_one_row(fixture):
    """The whole reason this extractor exists."""
    out = _run(fixture)
    n_g = np.asarray(out["n"][0])
    n_h = np.asarray(out["n"][1])
    assert len(n_g) == len(n_h) == len(np.asarray(out["pair_idx"]))
    assert ((n_g > 0) & (n_h > 0)).sum() > 0, (
        "no read end carries both marks; there is nothing to correlate")


def test_ends_with_one_mark_only_are_still_emitted(fixture):
    """Design A.1 emits permissively -- re-extraction is expensive, filtering
    downstream is free.  Restrict HCG to the right half of the reference so
    the left-hand end of every pair genuinely has one mark and not the other.
    """
    ctx = fixture["ctx"]
    hpos, hbase = ctx["HCG"]
    keep = hpos >= 1500
    out = _run(fixture,
               contexts=[{REF_NAME: ctx["GCH"]},
                         {REF_NAME: (hpos[keep], hbase[keep])}])
    n_g = np.asarray(out["n"][0])
    n_h = np.asarray(out["n"][1])
    one_only = ((n_g == 0) & (n_h > 0)) | ((n_g > 0) & (n_h == 0))
    assert one_only.sum() > 0, "the restriction must produce single-mark ends"
    assert (n_h[np.asarray(out["end_idx"]) == 1] == 0).all()


def test_ends_with_no_sites_at_all_are_still_emitted(fixture):
    """The permissive rule is about the EXTRACTOR doing no filtering; an end
    whose strand could not be called has zero of both marks and must survive,
    or Part B cannot count what it is missing."""
    out = _run(fixture)
    n_g = np.asarray(out["n"][0])
    n_h = np.asarray(out["n"][1])
    assert ((n_g == 0) & (n_h == 0)).sum() > 0


def test_single_site_ends_exist_and_are_kept(fixture):
    """--min-hcg-sites 1 is the default, so n == 1 ends must survive
    extraction; A.3.2's variance trap only bites if they do."""
    out = _run(fixture)
    assert (np.asarray(out["n"][1]) == 1).sum() > 0


def test_pairs_are_two_consecutive_rows_in_canonical_order(fixture):
    out = _run(fixture)
    pair = np.asarray(out["pair_idx"]); end = np.asarray(out["end_idx"])
    assert np.array_equal(pair, np.repeat(np.arange(len(pair) // 2), 2))
    assert np.array_equal(end, np.tile(np.array([1, 2], np.int8), len(pair) // 2))
    p5, p3 = np.asarray(out["pos_5p"]), np.asarray(out["pos_3p"])
    lo = np.minimum(p5, p3)
    assert (lo[0::2] <= lo[1::2]).all(), "end 1 must be the leftmost end"


def test_strand_survives_in_pos_5p(fixture):
    """A reverse read's 5' end is its higher coordinate; collapsing to
    (start, end) would lose the strand the design asks for."""
    out = _run(fixture)
    rev = np.asarray(out["is_reverse"]).astype(bool)
    p5, p3 = np.asarray(out["pos_5p"]), np.asarray(out["pos_3p"])
    assert (p5[~rev] < p3[~rev]).all()
    if rev.any():
        assert (p5[rev] > p3[rev]).all()


def test_min_distance_filters_pairs_not_ends(fixture):
    out0 = _run(fixture, min_distance=0)
    out1 = _run(fixture, min_distance=1500)
    n0, n1 = len(np.asarray(out0["pair_idx"])), len(np.asarray(out1["pair_idx"]))
    assert n1 < n0 and n1 % 2 == 0, "a filtered pair must lose both its ends"
    d = np.asarray(out1["contact_dist"])
    assert (d >= 1500).all()


def test_read_id_identifies_the_molecule(fixture):
    """Per-molecule conversion (A.3.5) needs both ends of one molecule
    identifiable; the established extractor discards the qname."""
    out = _run(fixture)
    rid = list(out["read_id"])
    assert len(rid) == len(np.asarray(out["pair_idx"]))
    assert rid[0::2] == rid[1::2], "the two ends of a pair share a read id"
    assert len(set(rid)) == len(rid) // 2


def test_output_is_deterministic(fixture):
    a, b = _run(fixture), _run(fixture)
    for key in ("pair_idx", "pos_5p", "k_hch", "n_hch", "contact_dist"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key]))
    for ci in range(2):
        assert np.array_equal(np.asarray(a["k"][ci]), np.asarray(b["k"][ci]))


def test_unpaired_reads_are_counted_not_silently_dropped(fixture):
    out = _run(fixture)
    st = out["stats"]
    assert int(st["n_pairs"]) * 2 == len(np.asarray(out["pair_idx"]))
    assert int(st["n_reads_kept"]) == 2 * int(st["n_pairs"]) + int(st["n_unpaired"])
    assert int(st["peak_buffered_reads"]) > 0


def test_snp_positions_are_counted_per_end(fixture):
    j = _cpp()
    ctx = fixture["ctx"]
    snp = np.arange(0, 3000, 7, dtype=np.int64)
    out = j.parse_cell_joint_ends(
        fixture["bam"], contexts=[{REF_NAME: ctx["GCH"]}],
        snp_positions={REF_NAME: snp}, trim_ends=5, mapq_min=30,
        min_distance=0)
    n = np.asarray(out["n_masked_snp"])
    assert n.sum() > 0
    # 100 bp read, 5 bp trimmed each end -> 90 bp span, one SNP every 7 bp.
    assert set(np.unique(n)).issubset({12, 13, 14})


# ----------------------------------------------------- the A.1 record shape ---

def _record(fixture, **kw):
    from nome_hic_concordance.joint_state_extract import extract_cell
    from nome_hic_concordance.reference import CONTEXT_DTYPE
    cp = {}
    for mark in ("GCH", "HCG"):
        pos, base = fixture["ctx"][mark]
        arr = np.empty(len(pos), dtype=CONTEXT_DTYPE)
        arr["pos"] = pos
        arr["ref_base"] = base
        cp[mark] = {REF_NAME: arr}
    args = dict(cell_id="cellA", min_distance=0)
    args.update(kw)
    return extract_cell(fixture["bam"], cp, **args)


def test_record_has_the_design_a1_columns(fixture):
    df = _record(fixture)
    want = {"read_id", "cell_id", "end_idx", "chrom", "pos_5p", "pos_3p",
            "strand", "mapq", "n_gch", "k_gch", "n_hcg", "k_hcg",
            "n_hch", "k_hch", "n_masked_snp", "frag_len", "contact_dist",
            "is_cis"}
    assert want <= set(df.columns), f"missing: {sorted(want - set(df.columns))}"
    assert (df["k_gch"] <= df["n_gch"]).all()
    assert (df["k_hcg"] <= df["n_hcg"]).all()
    assert (df["k_hch"] <= df["n_hch"]).all()
    assert set(df["end_idx"]) == {1, 2}
    assert (df["cell_id"] == "cellA").all()


def test_a_missing_mark_names_itself(fixture):
    from nome_hic_concordance.joint_state_extract import extract_cell
    with pytest.raises(KeyError, match="HCG"):
        extract_cell(fixture["bam"], {"GCH": {}}, cell_id="c")


def test_conversion_rate_is_one_minus_hch_methylation(fixture):
    from nome_hic_concordance.joint_state_extract import conversion_rate
    df = _record(fixture)
    per_end = conversion_rate(df, level="end")
    obs = df["n_hch"] > 0
    assert np.allclose(per_end[obs],
                       1.0 - df.loc[obs, "k_hch"] / df.loc[obs, "n_hch"])
    assert per_end[~obs].isna().all(), (
        "an end with no HCH site has no control; NaN, not a perfect score")


def test_molecule_conversion_pools_both_ends(fixture):
    """The artifact is molecule-level, so the control has to be too."""
    from nome_hic_concordance.joint_state_extract import conversion_rate
    df = _record(fixture)
    per_mol = conversion_rate(df, level="molecule")
    assert len(per_mol) == len(df) // 2
    g = df.groupby("pair_idx")[["k_hch", "n_hch"]].sum()
    obs = g["n_hch"] > 0
    assert np.allclose(per_mol[obs], 1.0 - g.loc[obs, "k_hch"] / g.loc[obs, "n_hch"])
