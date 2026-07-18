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


class _Args:
    """Minimal stand-in for the argparse namespace _pooled_null reads."""
    null_window = 100
    max_site_pairs_per_read = 0
    seed = 42


def _parse_with_pools(cpp, bam, ctx, donor_cap, anchor_cap, seed=42):
    m = _mod()
    edges = m._build_edges(1, 1_000_000, 3)
    res = cpp.parse_cell_site_distance(str(bam), ctx, {},
                                       [float(e) for e in edges],
                                       5, 30, 100, 0, seed, False, 0, {},
                                       donor_cap, anchor_cap)
    return res, edges


def test_pooled_null_rescues_pairs_that_are_isolated_within_their_own_cell(tmp_path):
    """The point of --null-scope pooled.

    Two cells whose ends are far apart WITHIN each cell but within ±100bp of
    each other ACROSS cells.  The within-cell null finds nothing; pooling the
    donor ends over the group finds a partner for every anchor.
    """
    cpp = _cpp()
    m = _mod()
    a_dir = tmp_path / "a"; a_dir.mkdir()
    b_dir = tmp_path / "b"; b_dir.mkdir()
    bam_a, ctx_a = _spaced_bam(a_dir, [50_000, 120_000, 200_000])
    bam_b, ctx_b = _spaced_bam(b_dir, [50_040, 120_040, 200_040])

    # a shared context covering both cells' sites, so chrom ids line up
    pos = np.union1d(ctx_a["chr1"][0], ctx_b["chr1"][0]).astype(np.int64)
    ctx = {"chr1": (pos, np.full(len(pos), ord("C"), np.uint8))}

    ra, edges = _parse_with_pools(cpp, bam_a, ctx, 10_000, 10_000)
    rb, _ = _parse_with_pools(cpp, bam_b, ctx, 10_000, 10_000, seed=43)

    # within-cell: nothing is eligible in either cell
    assert int(ra["n_null_eligible"]) == 0
    assert int(rb["n_null_eligible"]) == 0

    res = [({}, int(ra["n_pairs"]), 0, (dict(ra["donors"]), dict(ra["anchors"]))),
           ({}, int(rb["n_pairs"]), 0, (dict(rb["donors"]), dict(rb["anchors"])))]
    pooled = m._pooled_null(res, [0, 1], edges, _Args(), len(edges) - 1)

    assert pooled is not None
    assert pooled["n_anchors"] == 6                  # 3 pairs from each cell
    assert pooled["n_eligible"] == 6, (
        "pooling across cells should find a donor for every anchor, got "
        f"{pooled['n_eligible']}")
    # and it actually accumulated something
    assert sum(a.sum() for a in pooled["null_per_cell"]["gg"]) > 0


def test_pooled_null_keeps_per_cell_accumulators_for_the_bootstrap(tmp_path):
    cpp = _cpp()
    m = _mod()
    a_dir = tmp_path / "a"; a_dir.mkdir()
    b_dir = tmp_path / "b"; b_dir.mkdir()
    bam_a, ctx_a = _spaced_bam(a_dir, [50_000, 120_000, 200_000])
    bam_b, ctx_b = _spaced_bam(b_dir, [50_040, 120_040, 200_040])
    pos = np.union1d(ctx_a["chr1"][0], ctx_b["chr1"][0]).astype(np.int64)
    ctx = {"chr1": (pos, np.full(len(pos), ord("C"), np.uint8))}
    ra, edges = _parse_with_pools(cpp, bam_a, ctx, 10_000, 10_000)
    rb, _ = _parse_with_pools(cpp, bam_b, ctx, 10_000, 10_000, seed=43)
    res = [({}, 0, 0, (dict(ra["donors"]), dict(ra["anchors"]))),
           ({}, 0, 0, (dict(rb["donors"]), dict(rb["anchors"])))]
    pooled = m._pooled_null(res, [0, 1], edges, _Args(), len(edges) - 1)

    for key in ("gg", "hh", "gh"):
        assert len(pooled["null_per_cell"][key]) == 2      # one per cell
        assert len(pooled["matched_per_cell"][key]) == 2
    # the pooled total is the sum of the per-cell parts
    tot = sum(a for a in pooled["null_per_cell"]["gg"])
    assert np.isfinite(tot).all()


def test_long_range_anchors_are_sampled_separately(tmp_path):
    """Stratified anchor sampling.

    Only ~1% of real pairs are long-range, so a single uniform reservoir hands
    the long-range bins ~1% of an already small sample.  For a context as sparse
    as HCG (~1 site per read end) those bins then fall under --min-pairs and go
    NaN -- which is exactly what a 1577-cell run showed.  Long-range pairs get
    their own reservoir so a tight short cap cannot starve them.
    """
    cpp = _cpp()
    m = _mod()
    short = [1_200 + 20 * i for i in range(20)]      # span 200..580 -> short
    long_ = [50_000, 120_000, 200_000]               # span ~49kb+   -> long
    bam, ctx = _spaced_bam(tmp_path, short + long_)
    edges = m._build_edges(1, 1_000_000, 3)

    def run(anchor_cap, anchor_long_cap):
        return cpp.parse_cell_site_distance(
            str(bam), ctx, {}, [float(e) for e in edges],
            5, 30, 100, 0, 42, False, 0, {},
            0, anchor_cap, anchor_long_cap, 1_000)

    # no stratification (long cap 0) -> ONE reservoir, and a tight cap there
    # keeps only 2 anchors out of 23, long-range ones almost certainly lost.
    r = run(2, 0)
    assert int(r["n_anchor_long"]) == 0        # nothing routed to the long arm
    assert int(r["n_anchor_short"]) == 2       # ...but the pairs are not deleted

    # the same tight short cap, but long-range gets its own generous reservoir
    r = run(2, 100)
    assert int(r["n_anchor_short"]) == 2, "short stratum should stay capped"
    assert int(r["n_anchor_long"]) == 3, (
        "every long-range pair should survive its own reservoir, got "
        f"{int(r['n_anchor_long'])}")
    assert int(r["n_pairs_long_seen"]) == 3


def test_stratification_threshold_is_respected(tmp_path):
    """A pair lands in the long stratum only when its SPAN crosses the cut."""
    cpp = _cpp()
    m = _mod()
    bam, ctx = _spaced_bam(tmp_path, [3_000, 60_000])   # spans ~2kb and ~59kb
    edges = m._build_edges(1, 1_000_000, 3)

    def n_long(threshold):
        r = cpp.parse_cell_site_distance(
            str(bam), ctx, {}, [float(e) for e in edges],
            5, 30, 100, 0, 42, False, 0, {}, 0, 100, 100, threshold)
        return int(r["n_anchor_long"])

    assert n_long(1_000) == 2        # both spans exceed 1kb
    assert n_long(10_000) == 1       # only the 59kb pair
    assert n_long(100_000) == 0      # neither


def test_csr_concat_and_permute_round_trip():
    """The CSR plumbing must not scramble which sites belong to which end."""
    m = _mod()
    # two "cells", rows of differing width
    p1 = {"x_pos": np.array([10, 11, 20], np.int32),
          "x_st": np.array([1, 0, 1], np.int8),
          "x_ptr": np.array([0, 2, 3], np.int32)}          # rows: [10,11], [20]
    p2 = {"x_pos": np.array([30, 31, 32], np.int32),
          "x_st": np.array([0, 1, 0], np.int8),
          "x_ptr": np.array([0, 0, 3], np.int32)}          # rows: [], [30,31,32]
    pos, st, ptr = m._concat_csr([p1, p2], "x")
    rows = [list(pos[ptr[i]:ptr[i + 1]]) for i in range(len(ptr) - 1)]
    assert rows == [[10, 11], [20], [], [30, 31, 32]]

    order = np.array([3, 0, 2, 1])
    pos2, st2, ptr2 = m._permute_csr(pos, st, ptr, order)
    rows2 = [list(pos2[ptr2[i]:ptr2[i + 1]]) for i in range(len(ptr2) - 1)]
    assert rows2 == [[30, 31, 32], [10, 11], [], [20]]
    # states travel with their positions
    st_rows = [list(st2[ptr2[i]:ptr2[i + 1]]) for i in range(len(ptr2) - 1)]
    assert st_rows == [[0, 1, 0], [1, 0], [], [1]]


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
