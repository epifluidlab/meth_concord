"""What happens to a pair with NO non-linked end within ±window.

The null swaps the true partner for another end whose midpoint lies within
±window.  A pair with no such neighbour cannot be given one, so it is EXCLUDED
from the null -- it is not carried as NaN.  At single-cell coverage that is the
majority of pairs, and the ones that survive sit in read-dense loci, so the null
must NOT be read against the all-pairs linked curve.  The 'matched' arm exists
for that: the true partner, restricted to exactly the null-eligible pairs.

These tests pin the exclusion rule, the matched/null pairing, and the regression
that eligibility must not depend on the RNG.
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
    spec = importlib.util.spec_from_file_location("sdc_null", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sdc_null"] = m
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


def _spaced_bam(tmp_path, second_starts, clen=300_000, rlen=100):
    """One read pair per entry in ``second_starts``.

    End 1 is pinned at 1000 for every pair, so the caller controls the spacing
    of the END-2 midpoints -- which is what the ±window neighbour search
    indexes.  Sites alternate C/T so the read carries a real C→T mismatch and
    the conversion strand stays inferable.
    """
    pysam = pytest.importorskip("pysam")
    chrom, first = "chr1", 1000
    pos = []
    for st in [first] + list(second_starts):
        pos.extend(range(st + 10, st + rlen - 10, 10))
    pos = sorted(set(pos))
    sset = set(pos)

    ref = bytearray(b"A" * clen)
    for p in pos:
        ref[p] = ord("C")
    ref = ref.decode()
    fa = tmp_path / "ref.fa"
    fa.write_text(f">{chrom}\n" + "\n".join(ref[i:i + 60]
                                            for i in range(0, clen, 60)) + "\n")
    pysam.faidx(str(fa))

    hdr = {"HD": {"VN": "1.6"}, "SQ": [{"SN": chrom, "LN": clen}]}
    raw = tmp_path / "raw.bam"
    with pysam.AlignmentFile(raw, "wb", header=hdr) as bam:
        for i, st2 in enumerate(second_starts):
            for st, is_r1 in ((first, True), (st2, False)):
                a = pysam.AlignedSegment()
                a.query_name = f"r{i}"
                a.reference_id = 0
                a.reference_start = st
                a.mapping_quality = 60
                a.cigarstring = f"{rlen}M"
                seq = []
                for k in range(rlen):
                    p = st + k
                    seq.append(("C" if (p // 10) % 2 == 0 else "T")
                               if p in sset else ref[p])
                a.query_sequence = "".join(seq)
                a.query_qualities = pysam.qualitystring_to_array(
                    chr(33 + 40) * rlen)
                a.flag = (0x1 | 0x40) if is_r1 else (0x1 | 0x80)
                a.next_reference_id = 0
                a.next_reference_start = st2 if is_r1 else first
                bam.write(a)

    srt = tmp_path / "s.bam"
    pysam.sort("-o", str(srt), str(raw))
    out = tmp_path / "q.bam"
    with open(out, "wb") as fh:                     # calmd adds the MD tag
        fh.write(pysam.calmd("-b", str(srt), str(fa), catch_stdout=True))
    pysam.index(str(out))
    ctx = {chrom: (np.array(pos, np.int64),
                   np.full(len(pos), ord("C"), np.uint8))}
    return out, ctx


def _run(cpp, bam, ctx, seed=42, window=100):
    m = _mod()
    edges = m._build_edges(1, 1_000_000, 3)
    return cpp.parse_cell_site_distance(str(bam), ctx, {},
                                        [float(e) for e in edges],
                                        5, 30, window, 0, seed, False)


# 3 clusters of 2: every window holds EXACTLY 2 ends, the case the old
# "4 random attempts, else give up" retry loop failed on with p=1/16.
_CLUSTERED = [50_000, 50_040, 120_000, 120_040, 200_000, 200_040]
# all >> 100bp apart, so no pair has a neighbour
_ISOLATED = [50_000, 120_000, 200_000]


def test_isolated_pairs_are_excluded_from_the_null(tmp_path):
    """A long-range pair with nothing within ±100bp contributes to LINKED only.

    This is the documented behaviour: excluded, not NaN.
    """
    cpp = _cpp()
    bam, ctx = _spaced_bam(tmp_path, _ISOLATED)
    res = _run(cpp, bam, ctx)

    assert int(res["n_pairs"]) == len(_ISOLATED)
    assert int(res["n_null_eligible"]) == 0
    # linked saw them; the null and its matched baseline are empty
    assert np.asarray(res["gg_linked"]).sum() > 0
    assert np.asarray(res["gg_null"]).sum() == 0
    assert np.asarray(res["gg_matched"]).sum() == 0


def test_null_eligible_count_is_seed_independent(tmp_path):
    """Eligibility is pure geometry, so it must not move with the RNG.

    The old retry loop gave up after 4 self-draws, silently dropping eligible
    pairs at a seed-dependent rate -- so this count used to vary, and even
    disagreed between the pysam and htslib backends.
    """
    cpp = _cpp()
    bam, ctx = _spaced_bam(tmp_path, _CLUSTERED)
    counts = {int(_run(cpp, bam, ctx, seed=s)["n_null_eligible"])
              for s in range(1, 9)}
    assert counts == {len(_CLUSTERED)}, (
        f"eligible count varied with seed: {sorted(counts)}")


def test_matched_arm_tracks_the_null_eligible_pairs(tmp_path):
    """'matched' is non-empty exactly when the null is, and never exceeds linked."""
    cpp = _cpp()
    bam, ctx = _spaced_bam(tmp_path, _CLUSTERED)
    res = _run(cpp, bam, ctx)

    assert int(res["n_null_eligible"]) == len(_CLUSTERED)
    assert np.asarray(res["gg_matched"]).sum() > 0
    assert np.asarray(res["gg_null"]).sum() > 0
    # matched is a SUBSET of linked pairs and drops the within-read arm, so its
    # site-pair count can never exceed linked's in any bin.
    n_linked = np.asarray(res["gg_linked"])[0]
    n_matched = np.asarray(res["gg_matched"])[0]
    assert (n_matched <= n_linked + 1e-9).all()


def test_eligible_never_exceeds_total_pairs(tmp_path):
    cpp = _cpp()
    for starts in (_CLUSTERED, _ISOLATED):
        sub = tmp_path / f"w{len(starts)}"
        sub.mkdir()
        bam, ctx = _spaced_bam(sub, starts)
        res = _run(cpp, bam, ctx)
        assert 0 <= int(res["n_null_eligible"]) <= int(res["n_pairs"])


def test_widening_the_window_admits_more_pairs(tmp_path):
    """The exclusion is a window effect, so a wider window recovers pairs.

    Documents the knob a user reaches for when the eligible fraction is low.
    """
    cpp = _cpp()
    bam, ctx = _spaced_bam(tmp_path, _ISOLATED)
    tight = int(_run(cpp, bam, ctx, window=100)["n_null_eligible"])
    wide = int(_run(cpp, bam, ctx, window=200_000)["n_null_eligible"])
    assert tight == 0
    assert wide == len(_ISOLATED)
