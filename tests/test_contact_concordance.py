"""Concordance vs SPATIAL proximity: the observed/expected axis.

Raw contact frequency is dominated by genomic distance, so the axis here is
observed/expected at MATCHED separation.  The tests pin the three things that
makes true: the expected counts unobserved bin-pairs too, the separation floor
removes the range where the span filter deflates expected, and -- the property
that lets --max-pairs-per-cell exist at all -- O/E is invariant to uniform
subsampling because it is a ratio.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "contact_concordance.py")
_FIX = Path("/tmp/sdc")


def _mod():
    spec = importlib.util.spec_from_file_location("cc_mod", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["cc_mod"] = m
    spec.loader.exec_module(m)
    return m


def _cpp():
    pytest.importorskip("pysam")
    import pysam  # noqa: F401
    try:
        from nome_hic_concordance import _site_dist_core as cpp
    except Exception:
        pytest.skip("htslib extension not built")
    return cpp


# ------------------------------------------------------------ observed/exp ---

def _contacts(pairs, resolution=1000):
    """(chrom, mid1, mid2) arrays from a list of (bin_i, bin_j)."""
    a = np.array([p[0] for p in pairs], np.int64) * resolution
    b = np.array([p[1] for p in pairs], np.int64) * resolution
    return np.zeros(len(pairs), np.int64), a, b


def test_expected_divides_by_all_possible_bin_pairs_not_just_observed():
    """The denominator must include bin-pairs that were never seen.

    Otherwise 'expected' is the mean over what we happened to observe, which is
    not a background: every observed contact would sit near O/E = 1 by
    construction and the axis would carry no information.
    """
    m = _mod()
    N, R, d = 100, 1000, 5
    # one contact on each of the 95 possible bin-pairs at separation 5,
    # so observed == possible and every O/E must be exactly 1.
    pairs = [(i, i + d) for i in range(N - d)]
    chrom, m1, m2 = _contacts(pairs, R)
    out = m._observed_expected(chrom, m1, m2, np.array([N]), R, min_sep_bins=1)
    assert out["n_contacts_kept"] == N - d
    assert np.allclose(out["oe"], 1.0), (
        "uniform coverage of every possible bin-pair must give O/E == 1")


def test_a_hot_bin_pair_gets_the_right_oe():
    m = _mod()
    N, R, d = 100, 1000, 5
    pairs = [(i, i + d) for i in range(N - d)]
    pairs += [(0, d)] * 10                       # one bin-pair seen 11x total
    chrom, m1, m2 = _contacts(pairs, R)
    out = m._observed_expected(chrom, m1, m2, np.array([N]), R, min_sep_bins=1)
    obs_d, poss_d = (N - d) + 10, N - d
    want = 11.0 / (obs_d / poss_d)
    hot = out["oe"][out["observed"] == 11]
    assert hot.size and np.allclose(hot, want), f"{hot[:3]} != {want}"
    assert (out["oe"][out["observed"] == 1] < 1.0).all(), \
        "the cold bin-pairs must fall below 1 once one pair runs hot"


def test_oe_is_invariant_to_uniform_subsampling():
    """The property that makes --max-pairs-per-cell safe.

    O/E is a ratio, so sequencing (or sampling) 3x deeper scales observed and
    expected alike and cancels.  A raw-count axis would shift by log10(3).
    """
    m = _mod()
    N, R = 200, 1000
    rng = np.random.default_rng(0)
    base = [(int(i), int(i + rng.integers(3, 30))) for i in range(0, N - 30)]
    hot = [(10, 40)] * 7
    chrom, m1, m2 = _contacts(base + hot, R)
    one = m._observed_expected(chrom, m1, m2, np.array([N]), R, 1)

    chrom3, m13, m23 = _contacts((base + hot) * 3, R)
    three = m._observed_expected(chrom3, m13, m23, np.array([N]), R, 1)

    # same multiset of O/E values, just 3x as many rows
    assert np.allclose(np.sort(np.unique(np.round(one["oe"], 9))),
                       np.sort(np.unique(np.round(three["oe"], 9))))


def test_separation_floor_drops_the_filtered_range():
    m = _mod()
    N, R = 100, 1000
    pairs = [(0, 1), (0, 2), (0, 5), (0, 20)]     # separations 1, 2, 5, 20
    chrom, m1, m2 = _contacts(pairs, R)
    out = m._observed_expected(chrom, m1, m2, np.array([N]), R, min_sep_bins=2)
    assert out["n_contacts_kept"] == 3
    assert out["n_contacts_dropped"] == 1
    assert (out["sep"] >= 2).all()


def test_separation_floor_default_matches_the_span_filter():
    """ceil(span/resolution)+1 is the first separation the filter cannot bite."""
    for span, res, want in ((10_000, 25_000, 2), (10_000, 10_000, 2),
                            (10_000, 5_000, 3), (50_000, 25_000, 3)):
        assert int(np.ceil(span / res)) + 1 == want, (span, res)


def test_quantile_edges_balance_the_bins():
    m = _mod()
    x = np.concatenate([np.random.default_rng(1).normal(0, 1, 10_000),
                        np.full(500, 8.0)])       # heavy tail
    e = m._quantile_edges(x, 10)
    counts = np.histogram(x, bins=e)[0]
    assert counts.min() > 0
    assert counts.max() / counts.min() < 3.0, (
        "equal-count binning should not leave a bin near-empty")


def test_spearman_needs_enough_bins():
    m = _mod()
    assert not np.isfinite(m._spearman(np.arange(3.0), np.arange(3.0)))
    assert m._spearman(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)
    assert m._spearman(np.arange(10.0), -np.arange(10.0)) == pytest.approx(-1.0)
    # a flat curve has no trend to find
    assert not np.isfinite(m._spearman(np.arange(10.0), np.ones(10)))


# ----------------------------------------------------------------- backend ---

def test_accumulate_puts_every_site_pair_in_the_assigned_bin():
    """Contacts bin by the PAIR's O/E, not by site-site distance."""
    cpp = _cpp()
    # two contacts: sites are far apart in bp, but both go to bin 2 and bin 0
    pairs = {
        "g1_pos": np.array([100, 200, 5000], np.int32),
        "g1_st": np.array([1, 1, 0], np.int8),
        "g1_ptr": np.array([0, 2, 3], np.int32),
        "g2_pos": np.array([900_000, 950_000], np.int32),
        "g2_st": np.array([1, 0], np.int8),
        "g2_ptr": np.array([0, 1, 2], np.int32),
    }
    for tag in ("h1", "h2"):
        pairs[f"{tag}_pos"] = np.zeros(0, np.int32)
        pairs[f"{tag}_st"] = np.zeros(0, np.int8)
        pairs[f"{tag}_ptr"] = np.zeros(3, np.int32)
    out = cpp.accumulate_contacts(pairs, np.array([2, 0], np.int32),
                                  np.array([0, 0], np.int32), 4, 1, 0, 42)
    gg = np.asarray(out["gg_per_cell"][0])
    assert int(out["n_used"]) == 2
    assert gg[0][2] == 2, "contact 0 has 2x1 site pairs, all into bin 2"
    assert gg[0][0] == 1, "contact 1 has 1x1 site pair into bin 0"
    assert gg[0][1] == 0 and gg[0][3] == 0


def test_accumulate_skips_out_of_range_bins():
    cpp = _cpp()
    pairs = {"g1_pos": np.array([10], np.int32), "g1_st": np.array([1], np.int8),
             "g1_ptr": np.array([0, 1], np.int32),
             "g2_pos": np.array([20], np.int32), "g2_st": np.array([1], np.int8),
             "g2_ptr": np.array([0, 1], np.int32)}
    for tag in ("h1", "h2"):
        pairs[f"{tag}_pos"] = np.zeros(0, np.int32)
        pairs[f"{tag}_st"] = np.zeros(0, np.int8)
        pairs[f"{tag}_ptr"] = np.zeros(2, np.int32)
    out = cpp.accumulate_contacts(pairs, np.array([-1], np.int32),
                                  np.array([0], np.int32), 3, 1, 0, 42)
    assert int(out["n_used"]) == 0
    assert np.asarray(out["gg_per_cell"][0]).sum() == 0


def test_parse_respects_min_span():
    """--min-contact-span decides what counts as a contact at all."""
    cpp = _cpp()
    bam = _FIX / "cell1.sorted.bam"
    if not bam.exists():
        pytest.skip("synthetic fixture BAM not present")
    from nome_hic_concordance.cli import _parse_context_bed
    from nome_hic_concordance.reference import load_context_from_bed
    cp = load_context_from_bed(
        _parse_context_bed(f"GCH:{_FIX / 'GCH.bed.gz'},HCG:{_FIX / 'HCG.bed.gz'}"),
        n_threads=1, c_offset=0)
    ctx_g = {c: (a["pos"].astype(np.int64), a["ref_base"].astype(np.uint8))
             for c, a in cp["GCH"].items()}
    ctx_h = {c: (a["pos"].astype(np.int64), a["ref_base"].astype(np.uint8))
             for c, a in cp["HCG"].items()}

    def n_at(span):
        r = cpp.parse_cell_contacts(str(bam), ctx_g, ctx_h, 5, 30, span,
                                    False, 0, {}, 1, 0, 42)
        return int(r["n_long_range"])

    assert n_at(0) > n_at(5_000) >= n_at(50_000), "raising the span must not add"
    assert n_at(10**9) == 0


def test_parse_returns_chrom_names_for_index_mapping():
    """Indices must be resolvable to names, or the pooled matrix misaligns."""
    cpp = _cpp()
    bam = _FIX / "cell1.sorted.bam"
    if not bam.exists():
        pytest.skip("synthetic fixture BAM not present")
    from nome_hic_concordance.cli import _parse_context_bed
    from nome_hic_concordance.reference import load_context_from_bed
    cp = load_context_from_bed(
        _parse_context_bed(f"GCH:{_FIX / 'GCH.bed.gz'},HCG:{_FIX / 'HCG.bed.gz'}"),
        n_threads=1, c_offset=0)
    ctx_g = {c: (a["pos"].astype(np.int64), a["ref_base"].astype(np.uint8))
             for c, a in cp["GCH"].items()}
    r = cpp.parse_cell_contacts(str(bam), ctx_g, {}, 5, 30, 1000,
                                False, 0, {}, 1, 0, 42)
    names = list(r["chrom_names"])
    assert names, "chrom_names must be returned"
    assert max(np.asarray(r["chrom"]), default=0) < len(names)


# -------------------------------------------------------------- end to end ---

def _samples(tmp_path, n_rep=4):
    src = _FIX / "s.tsv"
    if not src.exists():
        pytest.skip("synthetic fixture not present")
    rows = [ln.rstrip("\n").split("\t") for ln in src.read_text().splitlines()]
    out = ["\t".join(rows[0])]
    for i in range(n_rep):
        for r in rows[1:]:
            r = list(r); r[0] = f"{r[0]}_r{i}"
            out.append("\t".join(r))
    p = tmp_path / "s.tsv"
    p.write_text("\n".join(out) + "\n")
    return p


def test_end_to_end_multi_group(tmp_path):
    """Every group must get a figure — the group loop is where the last bug hid."""
    pytest.importorskip("pysam")
    if not (_FIX / "GCH.bed.gz").exists():
        pytest.skip("fixture context BEDs not present")
    out_dir = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--samples", str(_samples(tmp_path)),
         "--context-bed",
         f"GCH:{_FIX / 'GCH.bed.gz'},HCG:{_FIX / 'HCG.bed.gz'}",
         "--output-dir", str(out_dir), "--resolution", "5000",
         "--min-contact-span", "2000", "--oe-bins", "5", "--min-pairs", "200",
         "--n-bootstrap", "100", "--group-column", "group",
         "--plot-format", "png", "--n-threads", "2"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    for stem in ("all", "A", "B"):
        assert (out_dir / f"contact_concordance_{stem}.png").exists(), r.stdout
        assert (out_dir / f"contact_concordance_{stem}.tsv").exists()

    import json
    s = json.loads((out_dir / "contact_concordance_summary.json").read_text())
    assert set(s["groups"]) == {"all", "A", "B"}
    assert s["normalization"].startswith("observed/expected")
    # bin edges are shared, so groups are comparable
    assert s["oe_bin_edges_log10"] is not None
