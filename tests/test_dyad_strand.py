"""Distance-1 HCG–HCG is the two strands of ONE CpG dyad — a selected subset.

A read end only calls sites whose reference base matches its own inferred
conversion strand: a C→T read reads ref-'C' positions, a G→A read reads ref-'G'.
A CpG dyad is a ref-'C' at d and a ref-'G' at d+1, so a distance-1 HCG–HCG pair
can ONLY form when the two MATES of a pair disagree on strand.  In a directional
library they should agree, which makes that bin a selected subset rather than a
population estimate — and it is why a real run showed linked r = +0.043 there
while the non-linked null, which draws its partner from an unselected read, gave
+0.389.

These tests pin the three configurations, and the evidence threshold that
decides how readily a read is assigned a strand at all.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "site_distance_correlation.py")
CHROM, CLEN, RLEN = "chr1", 60_000, 100
DYADS = list(range(2_000, 40_000, 400))


def _mod():
    spec = importlib.util.spec_from_file_location("sdc_dyad", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sdc_dyad"] = m
    spec.loader.exec_module(m)
    return m


def _cpp():
    pytest.importorskip("pysam")
    import pysam  # noqa: F401  — loads the libhts the extension links against
    try:
        from nome_hic_concordance import _site_dist_core as cpp
    except Exception:
        pytest.skip("htslib extension not built")
    return cpp


def _dyad_bam(tmp_path, mode, reps=8, seed=0):
    """Read pairs over CpG dyads with PERFECTLY symmetric methylation.

    mode 'opposite'    — R2 is a genuine G→A (bottom-strand) read
    mode 'directional' — R2 is the same conversion strand as R1
    """
    pysam = pytest.importorskip("pysam")
    rng = np.random.default_rng(seed)
    ref = bytearray(b"A" * CLEN)
    for d in DYADS:
        ref[d] = ord("C"); ref[d + 1] = ord("G")    # HCG on top (A-C-G)
        ref[d - 30] = ord("C")                      # R1's C→T strand marker
        ref[d + 30] = ord("G")                      # R2's G→A strand marker
    ref = ref.decode()
    fa = tmp_path / "ref.fa"
    fa.write_text(f">{CHROM}\n" + "\n".join(ref[i:i + 60]
                                            for i in range(0, CLEN, 60)) + "\n")
    pysam.faidx(str(fa))
    hdr = {"HD": {"VN": "1.6"}, "SQ": [{"SN": CHROM, "LN": CLEN}]}
    raw = tmp_path / "raw.bam"
    with pysam.AlignmentFile(raw, "wb", header=hdr) as bam:
        for rep in range(reps):
            for di, d in enumerate(DYADS):
                meth = int(rng.integers(0, 2))      # SYMMETRIC: strands agree
                for is_r1, start in ((True, d - 60), (False, d - 40)):
                    a = pysam.AlignedSegment()
                    a.query_name = f"p{rep}_{di}"
                    a.reference_id = 0; a.reference_start = start
                    a.mapping_quality = 60; a.cigarstring = f"{RLEN}M"
                    seq = list(ref[start:start + RLEN])
                    if is_r1 or mode == "directional":
                        seq[d - 30 - start] = "T"                  # force C→T
                        seq[d - start] = "C" if meth else "T"
                    else:
                        seq[d + 30 - start] = "A"                  # force G→A
                        seq[d + 1 - start] = "G" if meth else "A"
                    a.query_sequence = "".join(seq)
                    a.query_qualities = pysam.qualitystring_to_array(
                        chr(33 + 40) * RLEN)
                    a.flag = (0x1 | 0x40) if is_r1 else (0x1 | 0x80)
                    a.next_reference_id = 0
                    a.next_reference_start = (d - 40) if is_r1 else (d - 60)
                    bam.write(a)
    srt = tmp_path / "s.bam"
    pysam.sort("-o", str(srt), str(raw))
    out = tmp_path / "d.bam"
    with open(out, "wb") as fh:                      # calmd adds the MD tag
        fh.write(pysam.calmd("-b", str(srt), str(fa), catch_stdout=True))
    pysam.index(str(out))
    pos = np.array(sorted(DYADS + [d + 1 for d in DYADS]), np.int64)
    dset = set(DYADS)
    rb = np.array([ord("C") if p in dset else ord("G") for p in pos], np.uint8)
    return out, {CHROM: (pos, rb)}


def _run(cpp, m, bam, ctx, strand_min_evidence=1):
    edges = m._build_edges(1, 1_000_000, 8)
    res = cpp.parse_cell_site_distance(
        str(bam), {}, ctx, [float(e) for e in edges],
        0, 30, 100, 0, 42, False, 0, {}, 0, 0, 0, 1000, strand_min_evidence)
    r, n = m._corr(np.asarray(res["hh_linked"]), 1)
    return res, r, n


def test_symmetric_dyad_read_by_opposite_strands_gives_r_one(tmp_path):
    """The estimator itself is sound: a symmetric dyad must come out at r = 1.

    This is the control for the real-data anomaly — if this were not 1.0 the
    fault would be in strand handling or state polarity, and it is not.
    """
    cpp, m = _cpp(), _mod()
    bam, ctx = _dyad_bam(tmp_path, "opposite")
    _res, r, n = _run(cpp, m, bam, ctx)
    assert n[0] > 0, "no distance-1 pairs were formed at all"
    assert r[0] == pytest.approx(1.0, abs=1e-9), (
        f"symmetric dyad should be perfectly correlated, got {r[0]}")


def test_same_strand_mates_cannot_form_a_distance_one_pair(tmp_path):
    """The selection effect, stated directly.

    When both mates carry the same conversion strand — the normal case for a
    directional library — they call the same ref-base class, so the dyad's two
    strands are never both observed and the bin is EMPTY.  Everything that does
    land there therefore came from a mate pair that disagreed.
    """
    cpp, m = _cpp(), _mod()
    bam, ctx = _dyad_bam(tmp_path, "directional")
    res, _r, n = _run(cpp, m, bam, ctx)
    assert n[0] == 0, (
        f"same-strand mates must not produce distance-1 pairs, got {int(n[0])}")
    assert int(res["n_strand_disagree"]) == 0
    assert int(res["n_strand_known"]) > 0


def test_mate_strand_disagreement_is_counted(tmp_path):
    """The diagnostic that tells you how much of that bin to believe."""
    cpp, m = _cpp(), _mod()
    bam, ctx = _dyad_bam(tmp_path, "opposite")
    res, _r, _n = _run(cpp, m, bam, ctx)
    known, dis = int(res["n_strand_known"]), int(res["n_strand_disagree"])
    assert known > 0 and dis == known, (
        "every pair here is opposite-strand by construction")


def test_strand_evidence_threshold_drops_single_mismatch_calls(tmp_path):
    """A one-mismatch strand call is a coin flip on a SNP or a sequencing error.

    Each read here carries exactly one informative mismatch when methylated, so
    raising the required margin must strand-unknown those reads and drop their
    sites rather than trusting them.
    """
    cpp, m = _cpp(), _mod()
    bam, ctx = _dyad_bam(tmp_path, "opposite")
    _res, _r, n_lo = _run(cpp, m, bam, ctx, strand_min_evidence=1)
    _res, _r, n_hi = _run(cpp, m, bam, ctx, strand_min_evidence=3)
    assert n_lo[0] > 0
    assert n_hi[0] < n_lo[0], (
        "a stricter evidence margin must retain fewer sites, "
        f"got {int(n_hi[0])} vs {int(n_lo[0])}")


def test_strand_evidence_below_one_is_clamped(tmp_path):
    """0 or negative must not mean 'accept a zero-margin tie'."""
    cpp, m = _cpp(), _mod()
    bam, ctx = _dyad_bam(tmp_path, "opposite")
    _res, _r, n1 = _run(cpp, m, bam, ctx, strand_min_evidence=1)
    _res, _r, n0 = _run(cpp, m, bam, ctx, strand_min_evidence=0)
    assert int(n0[0]) == int(n1[0])
