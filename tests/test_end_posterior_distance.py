"""End-posterior distance correlation.

The site-level script correlates single 0/1 calls, which at one cytosine on one
molecule is mostly Bernoulli noise.  This one correlates the per-END posteriors,
each already an average over that end's informative sites.

The fixture plants a known correlation through a shared latent per pair, so the
tests check that it is recovered, that the nulls destroy it, and that all three
measures are compared against the SAME synthetic pairs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "end_posterior_distance_correlation.py")


def _mod():
    spec = importlib.util.spec_from_file_location("epd", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["epd"] = m
    spec.loader.exec_module(m)
    return m


def _fixture(tmp_path, n_cells=8, rho_g=0.9, rho_h=0.4, seed=0):
    """Both contexts read off the SAME read pairs, as the pipeline emits them."""
    rng = np.random.default_rng(seed)
    for ctx in ("GCH", "HCG"):
        (tmp_path / ctx).mkdir(parents=True, exist_ok=True)
    cells = [f"c{i}" for i in range(n_cells)]
    for c in cells:
        rows = []
        for b in range(40):
            base = b * 20000
            for k in range(15):
                d = int(rng.integers(500, 15000))
                rows.append((base + k * 40, base + k * 40 + d, d))
        recs = {"GCH": [], "HCG": []}
        for (p1, p2, d) in rows:
            shared = rng.normal(0, 1)
            for ctx, rho in (("GCH", rho_g), ("HCG", rho_h)):
                f = lambda: float(1 / (1 + np.exp(                     # noqa
                    -(shared * rho + rng.normal(0, 0.6)))))
                recs[ctx].append({"chrom": "chr1", "pos1": p1, "pos2": p2,
                                  "n1": 6, "n2": 6, "distance": d,
                                  "posterior1": f(), "posterior2": f()})
        for ctx in ("GCH", "HCG"):
            pd.DataFrame(recs[ctx]).to_parquet(
                tmp_path / ctx / f"{c}.posteriors.parquet")
    sp = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells,
                  "group": np.repeat(["A", "B"], n_cells // 2),
                  "exp_batch": ["b1", "b2"] * (n_cells // 2),
                  "plate_batch": ["p1", "p2"] * (n_cells // 2)}).to_csv(
        sp, sep="\t", index=False)
    return str(sp)


def _run(tmp_path, extra=None):
    m = _mod()
    sp = _fixture(tmp_path)
    out = tmp_path / "out"
    argv = ["--posteriors", f"GCH={tmp_path / 'GCH'}",
            "--posteriors", f"HCG={tmp_path / 'HCG'}",
            "--samples", sp, "--min-pairs", "50", "--plot-format", "png",
            "--output-dir", str(out)] + (extra or [])
    assert m.main(argv) == 0
    return pd.read_csv(out / "end_posterior_dist_all.tsv", sep="\t"), out


# ------------------------------------------------------------- correctness --

def test_recovers_the_planted_correlation_and_its_ordering(tmp_path):
    res, _ = _run(tmp_path)
    ok = res[res["gg_linked_n"] > 0]
    gg = ok["gg_linked_r"].median()
    hh = ok["hh_linked_r"].median()
    assert gg > 0.5, "the strongly coupled context must show a clear r"
    assert hh > 0.15
    assert gg > hh, "rho_g > rho_h was planted, so gg must exceed hh"


def test_the_null_destroys_the_pairing_for_every_measure(tmp_path):
    res, _ = _run(tmp_path)
    ok = res[res["gg_null_n"] > 0]
    for m in ("gg", "hh", "gh"):
        assert abs(ok[f"{m}_null_r"].median()) < 0.1, m
        assert (ok[f"{m}_linked_r"].median()
                - ok[f"{m}_null_r"].median()) > 0.1, m


def test_all_three_measures_get_a_null(tmp_path):
    """Regression: an earlier version built per-context nulls only, so the
    cross-context panel had no null and its Δ was undefined."""
    res, _ = _run(tmp_path)
    for m in ("gg", "hh", "gh"):
        assert (res[f"{m}_null_n"] > 0).any(), f"{m} has no null at any distance"


def test_cross_context_needs_both_contexts_on_the_same_pair(tmp_path):
    m = _mod()
    sp = _fixture(tmp_path)
    g = pd.read_parquet(tmp_path / "GCH" / "c0.posteriors.parquet")
    h = pd.read_parquet(tmp_path / "HCG" / "c0.posteriors.parquet")
    ren = {"posterior1": "p1", "posterior2": "p2"}
    j = m.joined_pairs({"GCH": g.rename(columns=ren), "HCG": h.rename(columns=ren)})
    assert j is not None and len(j) == len(g)
    for c in ("p1_g", "p2_g", "p1_h", "p2_h"):
        assert c in j.columns
    # Only one context available -> no cross-context frame at all.
    assert m.joined_pairs({"GCH": g.rename(columns=ren)}) is None


def test_end_table_carries_both_contexts_sorted_by_position(tmp_path):
    m = _mod()
    df = pd.DataFrame({"pos1": [300, 100], "pos2": [400, 200],
                       "p1_g": [0.1, 0.2], "p2_g": [0.3, 0.4],
                       "p1_h": [0.5, 0.6], "p2_h": [0.7, 0.8]})
    pos, vals, pi = m.end_table(df, {"g": ("p1_g", "p2_g"),
                                     "h": ("p1_h", "p2_h")})
    assert list(pos) == [100, 200, 300, 400], "sorted for binary search"
    assert set(vals) == {"g", "h"} and len(vals["g"]) == 4
    # The two ends of one pair keep the same pair index, so a null can reject
    # re-creating a real ligation.
    assert pi[list(pos).index(100)] == pi[list(pos).index(200)]
    assert pi[list(pos).index(100)] != pi[list(pos).index(300)]


def test_pair_ids_are_unique_across_cells(tmp_path):
    """A pooled donor pool spans cells, so per-cell indices would collide and
    the ligation check would fire on unrelated pairs — or, if disabled, let
    real ligations into the null."""
    m = _mod()
    df = pd.DataFrame({"pos1": [100], "pos2": [200], "p1_g": [0.1],
                       "p2_g": [0.2], "p1_h": [0.3], "p2_h": [0.4]})
    cols = {"g": ("p1_g", "p2_g"), "h": ("p1_h", "p2_h")}
    a = m.end_table(df, cols, m._pair_base(0))[2]
    b = m.end_table(df, cols, m._pair_base(1))[2]
    assert not set(a) & set(b)
    # Offsetting must not disturb a within-cell comparison.
    assert (a == a[0]).all() and (b == b[0]).all()


def test_pooled_null_refuses_a_real_ligation_from_any_cell(tmp_path):
    """Two donor ends that are the two ends of ONE read pair are a ligation,
    whichever cell contributed them, and must never enter the null."""
    m = _mod()
    rng = np.random.default_rng(0)
    n = 4000
    # Ends laid out as adjacent pairs, so the partner is always the nearest
    # end and a matched separation lands on it constantly unless refused.
    starts = np.sort(rng.integers(0, 2_000_000, n)).astype(np.int64)
    pos = np.sort(np.concatenate([starts, starts + 5000]))
    pi = np.concatenate([m._pair_base(c) + np.arange(n // 2, dtype=np.int64)
                         for c in range(2)])
    pi = np.concatenate([pi, pi])[np.argsort(np.concatenate(
        [starts, starts + 5000]), kind="stable")]
    edges = m.build_edges(100, 1_000_000, 8)
    got = m._null_indices_distance_matched(
        pos[::2][:n], np.full(n, 5000.0), pos, pi, edges, rng,
        window=1000, anchor_window=10_000_000)
    assert got is not None
    ka, kb, _ = got
    assert (pi[ka] != pi[kb]).all(), "a real ligation reached the null"


# ------------------------------------------------------------------ options --

def test_group_split_emits_a_table_per_group(tmp_path):
    _, out = _run(tmp_path, ["--group-column", "group"])
    for label in ("all", "A", "B"):
        assert (out / f"end_posterior_dist_{label}.tsv").exists(), label
        assert (out / f"end_posterior_dist_{label}.png").exists(), label


def test_pooled_donor_pool_keeps_identifiable_pair_ids(tmp_path):
    """The pool used to stamp -1 on every end, which made the ligation check
    unusable and let real ligations from any cell into the null."""
    import argparse
    m = _mod()
    sp = _fixture(tmp_path)
    cells = pd.read_csv(sp, sep="\t")["cell_name"].astype(str).tolist()
    args = argparse.Namespace(min_sites=2)
    pools = m._build_pools(
        cells, {"GCH": tmp_path / "GCH", "HCG": tmp_path / "HCG"}, args,
        None, None, ())
    # Minimal args mean --pair-set joint: one pool, keyed by its pair set.
    (pos, vals, pi), = pools["joint"].values()
    assert (pi >= 0).all(), "no sentinel ids left"
    # Every end appears twice, once per member of its pair, and no two cells
    # share an id.
    counts = np.unique(pi, return_counts=True)[1]
    assert (counts == 2).all()
    assert np.unique(pi).size == pos.size // 2
    assert np.unique(pi // m._PAIR_STRIDE).size == len(cells)


def test_pooled_null_with_batch_matching_runs(tmp_path):
    res, _ = _run(tmp_path, ["--null-scope", "pooled",
                             "--null-match-column", "exp_batch,plate_batch"])
    ok = res[res["gg_null_n"] > 0]
    assert len(ok) > 0
    assert abs(ok["gg_null_r"].median()) < 0.15


def test_locus_matched_null_runs(tmp_path):
    res, _ = _run(tmp_path, ["--null-type", "locus-matched",
                             "--null-window", "50000"])
    assert (res["gg_null_n"] > 0).any()


def test_unknown_match_column_fails_loudly(tmp_path):
    m = _mod()
    sp = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="--null-match-column"):
        m.main(["--posteriors", f"GCH={tmp_path / 'GCH'}",
                "--posteriors", f"HCG={tmp_path / 'HCG'}",
                "--samples", sp, "--null-match-column", "nope",
                "--output-dir", str(tmp_path / "o2")])


def test_posteriors_spec_must_be_ctx_equals_dir(tmp_path):
    m = _mod()
    sp = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="CTX=DIR"):
        m.main(["--posteriors", "justapath", "--samples", sp,
                "--output-dir", str(tmp_path / "o3")])


# ------------------------------------------------------- shared stats module --

def test_pearson_helpers_handle_continuous_and_binary_alike():
    from nome_hic_concordance import distance_stats as ds
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    y = 0.8 * x + rng.normal(0, 0.4, 500)
    acc = ds.new_acc(1)
    ds.accumulate(acc, np.zeros(500, int), x, y)
    r, n = ds.corr_from_acc(acc, min_pairs=10)
    assert n[0] == 500
    assert r[0] == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-9)


def test_constant_bin_is_masked_not_reported_as_zero():
    from nome_hic_concordance import distance_stats as ds
    acc = ds.new_acc(1)
    ds.accumulate(acc, np.zeros(50, int), np.ones(50), np.arange(50.0))
    r, _ = ds.corr_from_acc(acc, min_pairs=10)
    assert np.isnan(r[0]), "no variance means undefined, not uncorrelated"


# ------------------------------------------------- threading and exclusion --

def test_result_does_not_depend_on_thread_count(tmp_path):
    """Each cell gets its own seeded RNG, so the answer must not depend on how
    many threads ran or the order they finished in."""
    one, _ = _run(tmp_path, ["--n-threads", "1"])
    many, _ = _run(tmp_path, ["--n-threads", "4"])
    assert np.allclose(one.fillna(-9).to_numpy(), many.fillna(-9).to_numpy())


def test_exclude_bed_drops_pairs_with_either_end_inside(tmp_path):
    m = _mod()
    sp = _fixture(tmp_path)
    bed = tmp_path / "x.bed"
    bed.write_text("chr1\t0\t100000\nchr1\t200000\t300000\n")
    base, _ = _run(tmp_path)
    excl, _ = _run(tmp_path, ["--exclude-bed", str(bed)])
    assert excl["gg_linked_n"].sum() < base["gg_linked_n"].sum()
    # Independent count of what should survive.
    g = pd.read_parquet(tmp_path / "GCH" / "c0.posteriors.parquet")
    bad = ((g.pos1 < 100000) | ((g.pos1 >= 200000) & (g.pos1 < 300000))
           | (g.pos2 < 100000) | ((g.pos2 >= 200000) & (g.pos2 < 300000)))
    merged, ec = m.load_exclusions([str(bed)])
    keep = m._keep_mask(g, merged, ec)
    assert keep.sum() == int((~bad).sum())


def test_exclusion_is_pair_level_not_end_level(tmp_path):
    """One end inside an excluded region removes the whole pair."""
    m = _mod()
    bed = tmp_path / "y.bed"
    bed.write_text("chr1\t1000\t2000\n")
    merged, ec = m.load_exclusions([str(bed)])
    df = pd.DataFrame({"chrom": ["chr1"] * 3,
                       "pos1": [1500, 5000, 500],     # in, out, out
                       "pos2": [9000, 1200, 800]})    # out, in,  out
    assert list(m._keep_mask(df, merged, ec)) == [False, False, True]


def test_overlapping_intervals_are_merged():
    m = _mod()
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as fh:
        fh.write("chr1\t100\t200\nchr1\t150\t300\nchr1\t500\t600\n")
        path = fh.name
    merged, _ = m.load_exclusions([path])
    starts, ends = merged["1"]
    assert list(starts) == [100, 500] and list(ends) == [300, 600]


def test_chrom_naming_mismatch_still_matches():
    """A BED using 'chr1' must still mask a table recording '1'."""
    m = _mod()
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as fh:
        fh.write("chr1\t0\t1000\n")
        path = fh.name
    merged, ec = m.load_exclusions([path])
    df = pd.DataFrame({"chrom": ["1", "1"], "pos1": [500, 5000],
                       "pos2": [5000, 6000]})
    assert list(m._keep_mask(df, merged, ec)) == [False, True]


def test_exclude_chrom_removes_whole_chromosomes():
    m = _mod()
    merged, ec = m.load_exclusions(None, ["chrM", "chrX"])
    df = pd.DataFrame({"chrom": ["chr1", "chrX", "M"],
                       "pos1": [1, 1, 1], "pos2": [2, 2, 2]})
    assert list(m._keep_mask(df, merged, ec)) == [True, False, False]


def test_per_cell_work_is_deterministic_given_its_index(tmp_path):
    """The property that makes ANY backend equivalent: a cell's contribution
    depends only on (cell, its index), never on execution order.

    Tested here rather than by running the process backend, which cannot work
    under a module loaded by importlib: loky re-imports the module in the
    child and this one has no importable name.  The cross-backend equivalence
    itself was checked by running the script directly.
    """
    m = _mod()
    sp = _fixture(tmp_path)
    args = m.main.__wrapped__ if hasattr(m.main, "__wrapped__") else None
    import argparse
    a = argparse.Namespace(
        min_sites=2, seed=42, null_type="distance-matched",
        null_scope="cell", null_window=1000,
        null_anchor_window=10_000_000)
    paths = {"GCH": tmp_path / "GCH", "HCG": tmp_path / "HCG"}
    edges = m.build_edges(100, 10_000_000, 8)
    first = m._accumulate_cell("c0", 3, paths, edges, a, {}, None, None, ())
    second = m._accumulate_cell("c0", 3, paths, edges, a, {}, None, None, ())
    assert first[1] == second[1] and first[2] == second[2]
    for k in first[0]:
        assert np.allclose(first[0][k], second[0][k]), k
    # A different index must give a different draw, or the per-cell seeding
    # would be a no-op and every cell would share one null stream.
    other = m._accumulate_cell("c0", 4, paths, edges, a, {}, None, None, ())
    assert not np.allclose(first[0][("gg", "null")], other[0][("gg", "null")])


def test_vectorised_null_places_most_anchors(tmp_path):
    """The retry is a bounded outer loop over vectorised passes; it must still
    place essentially every anchor, not quietly give up on them."""
    m = _mod()
    rng = np.random.default_rng(0)
    n = 5000
    pos = np.sort(rng.integers(0, 5_000_000, 2 * n)).astype(np.int64)
    pi = np.arange(2 * n) // 2
    edges = m.build_edges(100, 1_000_000, 8)
    anchor_pos = pos[::2][:n]
    anchor_dist = rng.integers(1000, 100_000, n).astype(float)
    got = m._null_indices_distance_matched(
        anchor_pos, anchor_dist, pos, pi, edges, rng,
        window=1000, anchor_window=10_000_000)
    assert got is not None
    ka, kb, d = got
    assert ka.size > 0.9 * n, f"only placed {ka.size}/{n}"
    assert (ka != kb).all()
    assert (pi[ka] != pi[kb]).all(), "never re-creates a real ligation"
    assert (d > 0).all()


def test_keep_mask_cost_is_independent_of_row_count(tmp_path):
    """Regression: the mask used to call _canon once per ROW (1.6 M calls for
    one cell) and np.unique over the string column, which profiling showed was
    the largest single cost in the script.  Canonicalisation must now touch
    only the handful of category labels."""
    m = _mod()
    bed = tmp_path / "z.bed"
    bed.write_text("chr1\t0\t1000\nchr2\t0\t1000\n")
    merged, ec = m.load_exclusions([str(bed)], ["chrM"])

    calls = {"n": 0}
    real = m._canon

    def counting(c):
        calls["n"] += 1
        return real(c)

    m._canon = counting
    try:
        n = 20000
        df = pd.DataFrame({
            "chrom": np.resize(np.array(["chr1", "chr2", "chr3", "chrM"]), n),
            "pos1": np.resize(np.array([500, 5000]), n),
            "pos2": np.resize(np.array([9000, 8000]), n)})
        keep = m._keep_mask(df, merged, ec)
    finally:
        m._canon = real
    assert calls["n"] <= 20, (
        f"_canon ran {calls['n']} times for {n} rows — it must scale with the "
        f"number of distinct chromosomes, not rows")
    # And it must still be correct: chrM dropped, chr1/chr2 hits dropped.
    assert not keep[df["chrom"].to_numpy() == "chrM"].any()
    assert keep[(df["chrom"] == "chr3").to_numpy()].all()


def test_pooled_null_is_unaffected_by_the_backend_plumbing(tmp_path):
    """The shared worker pool is built before any group's donor pool exists,
    so a pooled-scope run must not end up with an empty donor set."""
    res, _ = _run(tmp_path, ["--null-scope", "pooled", "--n-threads", "4",
                             "--null-match-column", "exp_batch"])
    assert (res["gg_null_n"] > 0).any(), "pooled null produced nothing"
    ok = res[res["gg_null_n"] > 0]
    assert abs(ok["gg_null_r"].median()) < 0.2


# ------------------------------------------------ significance and grouping --

def test_delta_gets_a_bootstrap_ci_and_fdr_adjusted_p(tmp_path):
    res, _ = _run(tmp_path, ["--n-bootstrap", "200"])
    for m in ("gg", "hh", "gh"):
        for col in ("_delta_lo", "_delta_hi", "_delta_p", "_delta_q"):
            assert f"{m}{col}" in res.columns, m + col
    ok = res[res["gg_delta_p"].notna()]
    assert len(ok) > 0
    # q is an adjustment of p, so it can only be larger, and both stay in [0,1].
    assert (ok["gg_delta_q"] >= ok["gg_delta_p"] - 1e-12).all()
    assert ok["gg_delta_p"].between(0, 1).all()
    # The bootstrap floors p at 2/(B+1); it can never be exactly zero.
    assert (ok["gg_delta_p"] > 0).all()


def test_ci_brackets_the_point_estimate_where_it_is_defined(tmp_path):
    res, _ = _run(tmp_path, ["--n-bootstrap", "200"])
    d = res["gg_linked_r"] - res["gg_null_r"]
    ok = res["gg_delta_lo"].notna() & d.notna()
    assert ok.any()
    inside = ((d[ok] >= res.loc[ok, "gg_delta_lo"] - 0.15)
              & (d[ok] <= res.loc[ok, "gg_delta_hi"] + 0.15))
    assert inside.mean() > 0.8, "the CI should sit around the estimate"


def test_bootstrap_can_be_switched_off(tmp_path):
    res, _ = _run(tmp_path, ["--n-bootstrap", "0"])
    assert "gg_delta_p" not in res.columns


def test_by_group_overlay_written_only_with_several_groups(tmp_path):
    _, out = _run(tmp_path / "grouped", ["--group-column", "group"])
    assert (out / "end_posterior_dist_by_group.png").exists()
    # Separate directory: reusing one would leave the first run's overlay in
    # place and the check below would pass for the wrong reason.
    _, out2 = _run(tmp_path / "ungrouped")
    assert not (out2 / "end_posterior_dist_by_group.png").exists()


def test_group_overlay_needs_at_least_two_groups(tmp_path):
    m = _mod()
    curves = {(k, c): (np.zeros(3), np.zeros(3))
              for k, _ in m._MEASURES for c, _ in m._CONDS}
    one = {"all": (curves, np.array([1.0, 2.0, 3.0])),
           "A": (curves, np.array([1.0, 2.0, 3.0]))}
    import argparse
    a = argparse.Namespace(plot_format="png")
    assert m._figure_by_group(one, tmp_path / "x.png", a) is False, (
        "'all' plus ONE group is not a comparison")


# --------------------------------------------------------------- Fisher z --

def _null_arms(k=30, nb=60, rho_a=0.0, rho_b=0.0, n=80, seed=0):
    """Per-cell accumulators for two arms with known true correlations."""
    from nome_hic_concordance import distance_stats as ds
    rng = np.random.default_rng(seed)
    A, B = [], []
    for _ in range(k):
        accs = []
        for rho in (rho_a, rho_b):
            acc = ds.new_acc(nb)
            bins = np.repeat(np.arange(nb), n)
            x = rng.normal(size=nb * n)
            y = rho * x + np.sqrt(1 - rho ** 2) * rng.normal(size=nb * n)
            ds.accumulate(acc, bins, x, y)
            accs.append(acc)
        A.append(accs[0])
        B.append(accs[1])
    return A, B


def test_cell_level_fisher_z_is_calibrated_when_the_arms_agree():
    """Both arms rho=0: the test must not manufacture significance."""
    from nome_hic_concordance import distance_stats as ds
    A, B = _null_arms(seed=1)
    dz, lo, hi, p, k = ds.fisher_z_cell(A, B, min_pairs=30)
    ok = np.isfinite(p)
    assert ok.sum() > 50
    assert np.mean(p[ok] < 0.05) < 0.15, "false positives near the 5% nominal"
    assert (k[ok] == 30).all(), "every cell should clear 30 pairs per bin"


def test_cell_level_fisher_z_finds_a_real_difference():
    from nome_hic_concordance import distance_stats as ds
    A, B = _null_arms(rho_a=0.5, rho_b=0.0, seed=2)
    dz, lo, hi, p, _ = ds.fisher_z_cell(A, B, min_pairs=30)
    ok = np.isfinite(p)
    assert (dz[ok] > 0).all(), "arm A is the more correlated one"
    assert np.mean(p[ok] < 0.05) > 0.9
    assert ((lo[ok] <= dz[ok]) & (dz[ok] <= hi[ok])).all()


def test_the_bias_correction_shrinks_z_and_fades_with_n():
    """Unequal n between the arms would otherwise leave a difference behind
    when the two true correlations are identical."""
    from nome_hic_concordance import distance_stats as ds
    small = ds._fisher_z(0.6, 10)
    large = ds._fisher_z(0.6, 100_000)
    assert small < large < np.arctanh(0.6)
    assert abs(large - np.arctanh(0.6)) < 1e-5


def test_pooled_fisher_z_is_wildly_anticonservative_next_to_the_cell_test():
    """The whole reason 'cell' is the default: pooling pretends every end pair
    is an independent observation, so its p collapses."""
    from nome_hic_concordance import distance_stats as ds
    A, B = _null_arms(rho_a=0.15, rho_b=0.0, seed=3)
    _, _, _, p_cell, _ = ds.fisher_z_cell(A, B, min_pairs=30)
    _, _, _, p_pool, _ = ds.fisher_z_pooled(sum(A), sum(B), min_pairs=30)
    ok = np.isfinite(p_cell) & np.isfinite(p_pool)
    assert ok.any()
    assert np.median(p_pool[ok]) < np.median(p_cell[ok]) / 100


def test_pooled_fisher_z_invents_significance_when_cells_disagree():
    """The failure mode that decides the default.

    Each cell has its own nonzero difference between the arms — which is what
    a cell-to-cell difference in how well the null matches looks like — but
    the MEAN difference across cells is zero, so the honest answer is "not
    significant".  Pooling the pairs asks a different and uninteresting
    question, "is the pooled difference exactly zero", and answers it with a
    standard error that knows nothing about the spread across cells.
    """
    from nome_hic_concordance import distance_stats as ds
    rng = np.random.default_rng(11)
    nb, n = 25, 100
    A, B = [], []
    for _ in range(40):
        # One draw per (cell, bin), so the bins are independent replicates of
        # the situation rather than 25 copies of a single draw.
        e = rng.normal(0, 0.15, size=nb)        # cancels across cells
        arms = [ds.new_acc(nb), ds.new_acc(nb)]
        for b in range(nb):
            for acc, rho in zip(arms, (0.3 + e[b], 0.3 - e[b])):
                x = rng.normal(size=n)
                y = rho * x + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
                ds.accumulate(acc, np.full(n, b), x, y)
        A.append(arms[0])
        B.append(arms[1])
    _, _, _, p_cell, _ = ds.fisher_z_cell(A, B, 30)
    _, _, _, p_pool, _ = ds.fisher_z_pooled(sum(A), sum(B), 30)
    rej_cell = np.mean(p_cell[np.isfinite(p_cell)] < 0.05)
    rej_pool = np.mean(p_pool[np.isfinite(p_pool)] < 0.05)
    # Over ten seeds this averages 0.05 for the cell test and 0.45 for the
    # pooled one; the bounds are loose enough to survive a reseed.
    assert rej_cell < 0.2, "the cell-level test should sit near its nominal 5%"
    assert rej_pool > 0.25, "the pooled test should fire on many bins"


def test_fisher_z_figure_and_columns_are_written(tmp_path):
    res, out = _run(tmp_path, ["--fisher-min-cell-pairs", "15"])
    assert (out / "end_posterior_dist_all_fisherz.png").exists()
    assert (out / "end_posterior_dist_all.png").exists(), (
        "the bootstrap figure must still be written alongside")
    for m in ("gg", "hh", "gh"):
        for col in ("_fz_dz", "_fz_lo", "_fz_hi", "_fz_p", "_fz_q", "_fz_n"):
            assert f"{m}{col}" in res.columns, m + col
    ok = res[res["gg_fz_p"].notna()]
    assert len(ok) > 0
    assert (ok["gg_fz_q"] >= ok["gg_fz_p"] - 1e-12).all()
    assert ok["gg_fz_p"].between(0, 1).all()


def test_fisher_z_and_the_bootstrap_agree_on_the_planted_signal(tmp_path):
    res, _ = _run(tmp_path, ["--fisher-min-cell-pairs", "15",
                             "--n-bootstrap", "200"])
    ok = res[res["gg_fz_dz"].notna() & res["gg_delta_lo"].notna()]
    assert len(ok) > 0
    d = ok["gg_linked_r"] - ok["gg_null_r"]
    assert (np.sign(ok["gg_fz_dz"]) == np.sign(d)).all(), (
        "z is monotone in r, so the two contrasts cannot disagree in sign")
    assert (ok["gg_fz_q"] < 0.05).any(), "the planted coupling is strong"


def test_fisher_z_can_be_switched_off(tmp_path):
    res, out = _run(tmp_path, ["--fisher-z", "none"])
    assert not (out / "end_posterior_dist_all_fisherz.png").exists()
    assert "gg_fz_p" not in res.columns


def test_pooled_mode_runs_end_to_end(tmp_path):
    res, out = _run(tmp_path, ["--fisher-z", "pooled"])
    assert (out / "end_posterior_dist_all_fisherz.png").exists()
    ok = res[res["gg_fz_p"].notna()]
    assert len(ok) > 0
    # n is the pair count here, not the 8 cells — that is exactly the flaw.
    assert (ok["gg_fz_n"] >= 50).all(), "--min-pairs is the floor"
    assert ok["gg_fz_n"].max() > 500


def test_fdr_is_applied_across_the_whole_figure(tmp_path):
    """One test per bin per measure, so the adjustment must span all three
    measures rather than being redone per panel."""
    from nome_hic_concordance import distance_stats as ds
    p = np.array([0.001, 0.02, 0.5, np.nan])
    q = ds.fdr(p)
    assert np.isnan(q[3])
    assert (q[np.isfinite(q)] >= p[np.isfinite(p)] - 1e-12).all()
    assert ds.stars(0.0005) == "***" and ds.stars(0.2) == ""


# ------------------------------------------- parallel scaling (regression) --

def _args(m, **kw):
    """The args these helpers actually read, defaulted the way main() does."""
    import argparse
    a = argparse.Namespace(
        min_sites=2, seed=42, null_type="distance-matched",
        null_scope="cell", null_window=1000,
        null_anchor_window=10_000_000, n_threads=1)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_load_cell_returns_categorical_chrom_with_shared_categories(tmp_path):
    """The join key must not be an object column.

    Factorising object dtype inside the merge hashes one Python string per row
    while holding the GIL, which is what stopped --n-threads from scaling
    (1.63x on 8 threads, against 2.48x once categorical).  The two contexts
    must also share ONE category set: pandas quietly falls back to object when
    merging categoricals whose categories differ, which would hand the whole
    cost straight back with every test still passing.
    """
    m = _mod()
    _fixture(tmp_path, n_cells=2)
    per_ctx = m._load_cell({"GCH": tmp_path / "GCH", "HCG": tmp_path / "HCG"},
                           "c0", 2)
    assert set(per_ctx) == {"GCH", "HCG"}
    for ctx, df in per_ctx.items():
        assert isinstance(df["chrom"].dtype, pd.CategoricalDtype), ctx
    a, b = per_ctx["GCH"]["chrom"], per_ctx["HCG"]["chrom"]
    assert list(a.cat.categories) == list(b.cat.categories)
    # and the join still works, on the same rows as before
    j = m.joined_pairs(per_ctx)
    assert j is not None and len(j) == len(per_ctx["GCH"])


def test_single_context_cell_still_loads(tmp_path):
    """The category-unifying step must not fire on a one-context cell."""
    m = _mod()
    _fixture(tmp_path, n_cells=2)
    per_ctx = m._load_cell({"GCH": tmp_path / "GCH"}, "c0", 2)
    assert set(per_ctx) == {"GCH"}
    assert len(per_ctx["GCH"]) > 0
    assert m.joined_pairs(per_ctx) is None


def test_drain_folds_identically_to_materialising_every_cell(tmp_path):
    """_drain replaces list(executor.map(...)); it must not change the sum."""
    m = _mod()
    _fixture(tmp_path, n_cells=4)
    paths = {"GCH": tmp_path / "GCH", "HCG": tmp_path / "HCG"}
    edges = m.build_edges(100, 100_000, 4)
    args = _args(m)
    jobs = [(f"c{i}", i) for i in range(4)]
    raw = [m._accumulate_cell(c, i, paths, edges, args, {}, {}, None, ())
           for c, i in jobs]

    want = {k: v.copy() for k, v in
            {(mm, cc): m.new_acc(len(edges) - 1)
             for mm, _ in m._MEASURES for cc, _ in m._CONDS}.items()}
    wp = wn = 0
    for a, np_, nn_ in raw:
        for k in want:
            want[k] += a[k]
        wp += np_
        wn += nn_

    got = {(mm, cc): m.new_acc(len(edges) - 1)
           for mm, _ in m._MEASURES for cc, _ in m._CONDS}
    per_cell = []
    gp, gn = m._drain(iter(raw), got, per_cell, len(jobs), "t")
    assert (gp, gn) == (wp, wn)
    assert len(per_cell) == 4          # the bootstrap still gets every cell
    for k in want:
        assert np.allclose(got[k], want[k])


def test_build_pools_parallel_matches_serial(tmp_path):
    """Parallelising the donor pass must not move a single donor end."""
    m = _mod()
    _fixture(tmp_path, n_cells=6)
    paths = {"GCH": tmp_path / "GCH", "HCG": tmp_path / "HCG"}
    cells = [f"c{i}" for i in range(6)]
    a1, a8 = _args(m), _args(m)
    a1.n_threads, a8.n_threads = 1, 8
    ser = m._build_pools(cells, paths, a1, None, None, ())
    par = m._build_pools(cells, paths, a8, None, None, ())
    assert set(ser) == set(par)
    for sk in ser:
        assert set(ser[sk]) == set(par[sk])
        for k in ser[sk]:
            pos_s, vals_s, pi_s = ser[sk][k]
            pos_p, vals_p, pi_p = par[sk][k]
            assert np.array_equal(pos_s, pos_p)
            assert np.array_equal(pi_s, pi_p)
            for name in vals_s:
                assert np.allclose(vals_s[name], vals_p[name])


def test_pin_inner_threads_stops_pyarrow_fanning_out():
    """--n-threads 10 must not mean 10 workers x a full pyarrow pool each."""
    import pyarrow as pa
    m = _mod()
    before = pa.cpu_count()
    try:
        pa.set_cpu_count(max(2, before))
        m._pin_inner_threads()
        assert pa.cpu_count() == 1
        assert pa.io_thread_count() == 1
    finally:
        pa.set_cpu_count(before)


def _run_cli(tmp_path, extra):
    """Run the script as a REAL subprocess.

    The process backend spawns children that re-import the main module from
    its path.  The in-test loader registers this script under a synthetic name
    that a child cannot import, so exercising --backend process through
    main() would only ever test the loader.  This is also the shape the user
    runs it in.
    """
    import subprocess
    sp = _fixture(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT),
         "--posteriors", f"GCH={tmp_path / 'GCH'}",
         "--posteriors", f"HCG={tmp_path / 'HCG'}",
         "--samples", sp, "--min-pairs", "50", "--plot-format", "png",
         "--output-dir", str(out)] + extra,
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return pd.read_csv(out / "end_posterior_dist_all.tsv", sep="\t"), r.stderr


def test_backends_agree_and_auto_picks_by_cell_count(tmp_path):
    """Switching the default backend must not move a single number.

    Each cell is seeded from its own index, not from a shared generator, so
    thread and process pools have to give bit-identical curves.  If they ever
    diverge, the parallelism is leaking state and the backend choice becomes a
    scientific decision rather than a performance one.
    """
    auto, err = _run_cli(tmp_path / "a", ["--n-threads", "2"])
    assert "--backend auto -> thread" in err            # 8 cells < 96
    thr, _ = _run_cli(tmp_path / "t",
                      ["--n-threads", "2", "--backend", "thread"])
    prc, _ = _run_cli(tmp_path / "p",
                      ["--n-threads", "2", "--backend", "process"])
    for name, other in (("thread", thr), ("process", prc)):
        assert list(other.columns) == list(auto.columns), name
        for c in auto.columns:
            assert np.allclose(auto[c], other[c], equal_nan=True), (name, c)


def test_process_backend_agrees_under_pooled_donors(tmp_path):
    """--null-scope pooled builds its donor pool per GROUP.

    That is the path where a process pool cannot ride on the run-wide
    initializer, so it takes a separate branch; check that branch lands on the
    same numbers as threads rather than on an empty or shared donor set.
    """
    kw = ["--n-threads", "2", "--null-scope", "pooled"]
    thr, _ = _run_cli(tmp_path / "t", kw + ["--backend", "thread"])
    prc, _ = _run_cli(tmp_path / "p", kw + ["--backend", "process"])
    for c in thr.columns:
        assert np.allclose(thr[c], prc[c], equal_nan=True), c


def test_auto_switches_to_processes_once_past_the_threshold(tmp_path):
    """The threshold is the whole point of 'auto'; pin its direction."""
    m = _mod()
    assert m._PROCESS_FROM_CELLS > 8      # the fixture must land below it
    _, err = _run_cli(tmp_path, ["--n-threads", "2", "--max-cells", "8"])
    assert "--backend auto -> thread" in err


# ------------------------------------------------- nulls stay in cis (fix) ---

def _multi_chrom_pairs(seed=0, n_chrom=19, n=4000):
    """Ends on many chromosomes over the SAME coordinate range, as a real
    genome has them -- the case a single-chromosome fixture cannot see."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(1, n_chrom + 1):
        p1 = rng.integers(0, 80_000_000, n)
        d = (10 ** rng.uniform(3, 7, n)).astype(np.int64)
        rows.append(pd.DataFrame({
            "chrom": f"chr{c}", "pos1": p1, "pos2": p1 + d,
            "distance": d.astype(float),
            "p1_h": rng.random(n), "p2_h": rng.random(n),
            "p1_g": rng.random(n), "p2_g": rng.random(n)}))
    j = pd.concat(rows, ignore_index=True)
    j["chrom"] = j["chrom"].astype("category")
    return j


def _chrom_of(m, gpos):
    return np.asarray(gpos) // m._GENOME_STRIDE


def test_distance_matched_null_never_crosses_chromosomes():
    """"Two loci d apart" is a cis statement.  With ends sorted by raw position
    a partner 10 kb away was usually the same coordinate on another
    chromosome; on 19 chromosomes 9 in 10 null pairs were trans."""
    m = _mod()
    j = _multi_chrom_pairs()
    j = j.assign(gpos1=m.genome_coords(j["chrom"], j["pos1"]),
                 gpos2=m.genome_coords(j["chrom"], j["pos2"]))
    cols = {"g": ("p1_g", "p2_g"), "h": ("p1_h", "p2_h")}
    d_pos, _, d_pi = m.end_table(j, cols, 0)
    edges = m.build_edges(1000, 10_000_000, 8)
    ka, kb, dn = m._null_indices_distance_matched(
        j["gpos1"].to_numpy(), j["distance"].to_numpy(), d_pos, d_pi, edges,
        np.random.default_rng(1), 1000, 10_000_000)
    assert ka.size > 0.9 * len(j)
    assert (_chrom_of(m, d_pos[ka]) == _chrom_of(m, d_pos[kb])).all(), (
        "a distance-matched null pair crossed chromosomes")
    assert (dn <= 10_000_000 + 1).all()


def test_the_raw_position_version_did_cross_chromosomes():
    """Pins what the fix fixed, so a regression back to raw positions fails
    here rather than quietly re-flattening panel a's long-range null."""
    m = _mod()
    j = _multi_chrom_pairs(seed=2)
    pos = np.concatenate([j["pos1"].to_numpy(), j["pos2"].to_numpy()])
    chrom = np.concatenate([j["chrom"].to_numpy(), j["chrom"].to_numpy()])
    pi = np.concatenate([np.arange(len(j))] * 2)
    o = np.argsort(pos, kind="stable")
    edges = m.build_edges(1000, 10_000_000, 8)
    ka, kb, _ = m._null_indices_distance_matched(
        j["pos1"].to_numpy(), j["distance"].to_numpy(), pos[o], pi[o], edges,
        np.random.default_rng(1), 1000, 10_000_000)
    assert (chrom[o][ka] != chrom[o][kb]).mean() > 0.8


def test_locus_matched_null_never_crosses_chromosomes():
    m = _mod()
    j = _multi_chrom_pairs(seed=3)
    j = j.assign(gpos1=m.genome_coords(j["chrom"], j["pos1"]),
                 gpos2=m.genome_coords(j["chrom"], j["pos2"]))
    d_pos, _, d_pi = m.end_table(j, {"g": ("p1_g", "p2_g")}, 0)
    ai, kb, _ = m._null_indices_locus_matched(
        j["gpos2"].to_numpy(), j["gpos1"].to_numpy(),
        np.arange(len(j), dtype=np.int64), d_pos, d_pi,
        np.random.default_rng(4), 1000)
    anchor_chrom = _chrom_of(m, j["gpos1"].to_numpy()[ai])
    assert (anchor_chrom == _chrom_of(m, d_pos[kb])).all()


def test_chrom_codes_are_stable_and_distinct():
    m = _mod()
    names = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY", "chrM"]
    codes = [m._chrom_code(c) for c in names]
    assert len(set(codes)) == len(codes)
    assert m._chrom_code("chr7") == m._chrom_code("7")
    assert m._chrom_code("chrM") == m._chrom_code("MT")
    assert m._chrom_code("chrUn_JH584304") == m._chrom_code("chrUn_JH584304")


def test_one_chromosome_draws_are_unchanged_by_the_shift():
    """Genome coordinates only add a constant on one chromosome, so every
    existing single-chromosome result is bit-identical."""
    m = _mod()
    rng = np.random.default_rng(0)
    pos = np.sort(rng.integers(0, 5_000_000, 4000)).astype(np.int64)
    pi = np.arange(4000) // 2
    anchors, dist = pos[::2], rng.integers(1000, 100_000, 2000).astype(float)
    edges = m.build_edges(100, 1_000_000, 8)
    shift = m._chrom_code("chr1") * m._GENOME_STRIDE
    a = m._null_indices_distance_matched(anchors, dist, pos, pi, edges,
                                         np.random.default_rng(5), 1000, 10**7)
    b = m._null_indices_distance_matched(anchors + shift, dist, pos + shift,
                                         pi, edges, np.random.default_rng(5),
                                         1000, 10**7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# ------------------------------------------------------- per-cell (Fig. 5c) ---

_PERCELL = ["--per-cell-block", "50000", "--per-cell-min-distance", "500",
            "--per-cell-min-pairs", "50", "--per-cell-bootstrap", "100",
            "--group-column", "group"]


def test_per_cell_table_and_panel_are_written(tmp_path):
    _, out = _run(tmp_path, _PERCELL)
    t = pd.read_csv(out / "end_posterior_percell.tsv", sep="\t")
    for c in ("cell_id", "group", "mark", "n_pairs", "n_blocks", "r_linked",
              "r_null", "delta", "se", "z", "r_null2", "delta_null",
              "se_null", "z_null", "q", "q_null", "min_distance"):
        assert c in t.columns, c
    assert set(t["mark"]) == {"GCH", "HCG", "HCG-GCH"}
    assert len(t) == 8 * 3, "one row per cell per mark"
    assert set(t["group"]) == {"A", "B"}
    for suffix in ("", "_main", "_ED"):
        assert (out / f"end_posterior_percell{suffix}.png").exists(), suffix


def test_per_cell_finds_the_planted_concordance(tmp_path):
    """The fixture plants a shared latent per pair (strong for GCH)."""
    _, out = _run(tmp_path, _PERCELL)
    t = pd.read_csv(out / "end_posterior_percell.tsv", sep="\t")
    g = t[t["mark"] == "GCH"]
    assert (g["delta"] > 0).all()
    assert (g["z"] > 3).mean() > 0.7
    assert (g["se"] > 0).all()


def test_panel_a_is_identical_with_or_without_per_cell(tmp_path):
    """The control draw runs on its own generator, so switching the per-cell
    output on must not move a single panel a number."""
    on, _ = _run(tmp_path / "on", _PERCELL)
    off, _ = _run(tmp_path / "off", ["--per-cell", "off"])
    pd.testing.assert_frame_equal(on, off)


def test_per_cell_off_writes_nothing_extra(tmp_path):
    _, out = _run(tmp_path, ["--per-cell", "off"])
    assert not (out / "end_posterior_percell.tsv").exists()


def test_negative_control_is_calibrated():
    """Two independent null re-pairings, put through the same per-cell test,
    must look like N(0,1) -- otherwise the panel's headline count is not
    interpretable."""
    from nome_hic_concordance import percell_existence as pce
    rng = np.random.default_rng(10)
    zs = []
    for cell in range(300):
        k, per = 40, 60
        blocks = np.repeat(np.arange(k), per)
        def arm():
            x = rng.normal(size=k * per)
            y = rng.normal(size=k * per)
            return pce.block_sums(blocks, x, y)
        st = pce.percell_stats(arm(), arm(), arm(), n_boot=150, seed=cell)
        zs.append(st["z_null"])
    zs = np.array(zs)
    assert abs(zs.mean()) < 0.25, zs.mean()
    assert 0.75 < zs.std() < 1.3, zs.std()


def test_block_bootstrap_widens_the_se_when_pairs_share_a_block():
    """Pairs in one domain share state; a 1/sqrt(n) SE treats them as
    independent and calls every cell significant.  The block bootstrap must
    see the dependence."""
    from nome_hic_concordance import percell_existence as pce
    rng = np.random.default_rng(11)
    k, per = 40, 60
    blocks = np.repeat(np.arange(k), per)

    def arm(shared):
        u = np.repeat(rng.normal(size=k), per) if shared else 0.0
        x = u + rng.normal(size=k * per)
        y = u + rng.normal(size=k * per)
        return pce.block_sums(blocks, x, y)

    indep = pce.percell_stats(arm(False), arm(False), n_boot=300, seed=1)
    clustered = pce.percell_stats(arm(True), arm(True), n_boot=300, seed=1)
    assert clustered["se"] > 1.5 * indep["se"]


def test_a_thin_cell_reports_nan_not_zero():
    from nome_hic_concordance import percell_existence as pce
    few = pce.block_sums(np.zeros(10, int), np.arange(10.0), np.arange(10.0))
    st = pce.percell_stats(few, few, min_pairs=100)
    assert np.isnan(st["delta"]) and np.isnan(st["z"])
    assert st["n_pairs"] == 10


# -------------------------------------- per-mark pair sets + matched arm ---

def _drop_hcg(tmp_path, cells):
    for c in cells:
        (tmp_path / "HCG" / f"{c}.posteriors.parquet").unlink()


def test_a_cell_without_hcg_still_counts_for_gch(tmp_path):
    """Under the old joint pair set a cell with no HCG output vanished from
    GCH-GCH too -- 858 of 1,577 cells in the real run."""
    m = _mod()
    sp = _fixture(tmp_path)
    _drop_hcg(tmp_path, ["c0", "c1", "c2"])
    out = {}
    for mode in ("per-mark", "joint"):
        o = tmp_path / mode
        assert m.main(["--posteriors", f"GCH={tmp_path / 'GCH'}",
                       "--posteriors", f"HCG={tmp_path / 'HCG'}",
                       "--samples", sp, "--min-pairs", "50",
                       "--plot-format", "png", "--output-dir", str(o),
                       "--pair-set", mode,
                       # follow --pair-set, so this compares the two SETS;
                       # by default the per-cell test is per-mark either way
                       "--per-cell-pair-set", "same"] + _PERCELL) == 0
        out[mode] = pd.read_csv(o / "end_posterior_percell.tsv", sep="\t")
    g_pm = set(out["per-mark"].query("mark == 'GCH'")["cell_id"])
    g_jt = set(out["joint"].query("mark == 'GCH'")["cell_id"])
    assert {"c0", "c1", "c2"} <= g_pm
    assert not ({"c0", "c1", "c2"} & g_jt)


def test_per_mark_uses_more_gch_pairs_than_joint(tmp_path):
    m = _mod()
    sp = _fixture(tmp_path)
    _drop_hcg(tmp_path, ["c0"])
    n = {}
    for mode in ("per-mark", "joint"):
        o = tmp_path / mode
        m.main(["--posteriors", f"GCH={tmp_path / 'GCH'}",
                "--posteriors", f"HCG={tmp_path / 'HCG'}", "--samples", sp,
                "--min-pairs", "50", "--plot-format", "png",
                "--output-dir", str(o), "--pair-set", mode, "--per-cell", "off"])
        n[mode] = pd.read_csv(o / "end_posterior_dist_all.tsv",
                              sep="\t")["gg_linked_n"].sum()
    assert n["per-mark"] > n["joint"]


def test_joint_mode_puts_every_measure_on_one_pair_set(tmp_path):
    res, _ = _run(tmp_path, ["--pair-set", "joint", "--per-cell", "off"])
    assert (res["gg_linked_n"] == res["hh_linked_n"]).all()


def test_matched_arm_is_reported_and_drives_delta_by_default(tmp_path):
    res, _ = _run(tmp_path, ["--per-cell", "off"])
    for mm in ("gg", "hh", "gh"):
        assert f"{mm}_matched_r" in res.columns
    # matched is a subset of linked, never larger
    assert (res["gg_matched_n"] <= res["gg_linked_n"]).all()


def _selective_eligibility_fixture(tmp_path, n_cells=12, seed=7):
    """Recreates the negative HCG Δ under the pooled locus-matched null.

    DENSE loci sit at the same positions in every cell and carry a shared
    per-locus state, so another cell's end there predicts the anchor as well
    as the real partner does.  SPARSE loci are unique to each cell, carry no
    shared state, and have no donor within the window, so the null can never
    place them.  Linked then mixes both; the null sees only the dense ones.
    """
    rng = np.random.default_rng(seed)
    for ctx in ("GCH", "HCG"):
        (tmp_path / ctx).mkdir(parents=True, exist_ok=True)
    dense_pos = np.arange(60) * 400_000 + 50_000
    locus_state = rng.normal(0, 1.5, dense_pos.size)
    cells = [f"c{i}" for i in range(n_cells)]
    for ci, c in enumerate(cells):
        recs = []
        for k, p1 in enumerate(dense_pos):
            s = locus_state[k]
            recs.append((p1, p1 + 5000, s + rng.normal(0, 0.5),
                         s + rng.normal(0, 0.5)))
        for _ in range(60):
            p1 = int(rng.integers(0, 2_000_000)) * 13 + ci * 7 + 30_000_000
            recs.append((p1, p1 + 5000, rng.normal(0, 1.5),
                         rng.normal(0, 1.5)))
        for ctx in ("GCH", "HCG"):
            sig = lambda v: 1 / (1 + np.exp(-v))                     # noqa
            pd.DataFrame([{"chrom": "chr1", "pos1": a, "pos2": b, "n1": 6,
                           "n2": 6, "distance": 5000,
                           "posterior1": sig(x), "posterior2": sig(y)}
                          for a, b, x, y in recs]).to_parquet(
                tmp_path / ctx / f"{c}.posteriors.parquet")
    sp = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells}).to_csv(sp, sep="\t", index=False)
    return str(sp)


def test_matched_arm_removes_the_selective_eligibility_artefact(tmp_path):
    """linked − null comes out NEGATIVE here although no contact effect was
    planted at all; matched − null, scored on the same loci, does not."""
    m = _mod()
    sp = _selective_eligibility_fixture(tmp_path)
    out = tmp_path / "out"
    assert m.main(["--posteriors", f"GCH={tmp_path / 'GCH'}",
                   "--posteriors", f"HCG={tmp_path / 'HCG'}",
                   "--samples", sp, "--null-type", "locus-matched",
                   "--null-scope", "pooled", "--null-window", "100",
                   "--min-pairs", "50", "--min-distance", "1000",
                   "--max-distance", "100000", "--plot-format", "png",
                   "--per-cell", "off", "--n-bootstrap", "0",
                   "--fisher-z", "none", "--output-dir", str(out)]) == 0
    r = pd.read_csv(out / "end_posterior_dist_all.tsv", sep="\t")
    row = r.loc[r["gg_null_n"].idxmax()]
    linked_minus_null = row["gg_linked_r"] - row["gg_null_r"]
    matched_minus_null = row["gg_matched_r"] - row["gg_null_r"]
    assert linked_minus_null < -0.15, linked_minus_null
    assert abs(matched_minus_null) < 0.08, matched_minus_null


def test_hcg_floor_of_one_keeps_more_hcg_pairs_than_two(tmp_path):
    res1, _ = _run(tmp_path / "a", ["--per-cell", "off", "--min-sites-hcg", "1"])
    res2, _ = _run(tmp_path / "b", ["--per-cell", "off", "--min-sites-hcg", "2"])
    assert res1["hh_linked_n"].sum() >= res2["hh_linked_n"].sum()


def test_explicit_min_sites_still_applies_to_both_marks():
    m = _mod()
    import argparse
    a = argparse.Namespace(min_sites=3, min_sites_gch=2, min_sites_hcg=1)
    assert m._min_sites(a) == {"GCH": 3, "HCG": 3}
    b = argparse.Namespace(min_sites=None, min_sites_gch=2, min_sites_hcg=1)
    assert m._min_sites(b) == {"GCH": 2, "HCG": 1}


# ------------------------------- --null-chrom any reproduces the original ---

_REPRO = ["--pair-set", "joint", "--min-sites", "2", "--null-chrom", "any",
          "--delta-arm", "linked"]


def _multi_chrom_fixture(root, n_cells=8, seed=3):
    rng = np.random.default_rng(seed)
    for ctx in ("GCH", "HCG"):
        (root / ctx).mkdir(parents=True, exist_ok=True)
    cells = [f"c{i}" for i in range(n_cells)]
    for c in cells:
        rows = []
        for ch in range(1, 7):
            for _ in range(100):
                p1 = int(rng.integers(0, 30_000_000))
                d = int(10 ** rng.uniform(2, 6.8))
                rows.append((f"chr{ch}", p1, p1 + d, d))
        for ctx in ("GCH", "HCG"):
            recs = []
            for chrom, p1, p2, d in rows:
                u = rng.normal()
                f = lambda: float(1 / (1 + np.exp(                    # noqa
                    -(u * 0.8 + rng.normal(0, 0.7)))))
                recs.append({"chrom": chrom, "pos1": p1, "pos2": p2,
                             "n1": 4, "n2": 4, "distance": d,
                             "posterior1": f(), "posterior2": f()})
            pd.DataFrame(recs).to_parquet(root / ctx / f"{c}.posteriors.parquet")
    sp = root / "s.tsv"
    pd.DataFrame({"cell_name": cells,
                  "group": np.repeat(["A", "B"], n_cells // 2)}).to_csv(
        sp, sep="\t", index=False)
    return str(sp)


def _main_on(m, root, sp, out, extra):
    return m.main(["--posteriors", f"GCH={root / 'GCH'}",
                   "--posteriors", f"HCG={root / 'HCG'}", "--samples", sp,
                   "--min-pairs", "20", "--plot-format", "png",
                   "--n-threads", "1", "--n-bootstrap", "100",
                   "--output-dir", str(out)] + extra)


def test_reproduction_flags_match_the_original_script_exactly(tmp_path):
    """--pair-set joint --min-sites 2 --null-chrom any --delta-arm linked must
    give the ORIGINAL panel a, column for column, on several chromosomes --
    the case where the old and new nulls actually differ."""
    import subprocess
    root = Path(__file__).resolve().parent.parent
    probe = subprocess.run(["git", "-C", str(root), "show",
                            "d5d3cf1:scripts/end_posterior_distance_correlation.py"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("original revision not available in this checkout")
    orig_path = tmp_path / "orig_epd.py"
    orig_path.write_text(probe.stdout)
    spec = importlib.util.spec_from_file_location("orig_epd", orig_path)
    orig = importlib.util.module_from_spec(spec)
    sys.modules["orig_epd"] = orig
    spec.loader.exec_module(orig)

    sp = _multi_chrom_fixture(tmp_path)
    assert _main_on(orig, tmp_path, sp, tmp_path / "old", []) == 0
    m = _mod()
    assert _main_on(m, tmp_path, sp, tmp_path / "new",
                    _REPRO + ["--per-cell", "off"]) == 0
    a = pd.read_csv(tmp_path / "old" / "end_posterior_dist_all.tsv", sep="\t")
    b = pd.read_csv(tmp_path / "new" / "end_posterior_dist_all.tsv", sep="\t")
    shared = [c for c in a.columns if c in b.columns]
    assert len(shared) > 30
    # The cross-context NULL is the one deliberate departure: the original
    # swapped HCG and GCH in its second order (see _null_xy).  The FDR runs
    # over the whole figure, so every q moves with gh's p-values; the raw
    # p-values of gg and hh do not.  Everything else -- gg, hh, the linked
    # and matched gh arms, every count -- must still reproduce exactly.
    fixed = {c for c in shared
             if c == "gh_null_r" or c.startswith(("gh_delta_", "gh_fz_"))
             or c.endswith("_q")}
    for c in shared:
        if c not in fixed:
            assert a[c].equals(b[c]), c
    assert not a["gh_null_r"].equals(b["gh_null_r"])


def test_panel_a_is_untouched_by_a_separate_per_cell_pass(tmp_path):
    """Panel c runs on per-mark pairs while panel a reproduces the joint
    original; its draws use their own generators, so switching it on must not
    move panel a."""
    m = _mod()
    sp = _multi_chrom_fixture(tmp_path)
    assert _main_on(m, tmp_path, sp, tmp_path / "on", _REPRO + _PERCELL) == 0
    assert _main_on(m, tmp_path, sp, tmp_path / "off",
                    _REPRO + ["--per-cell", "off"]) == 0
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "on" / "end_posterior_dist_all.tsv", sep="\t"),
        pd.read_csv(tmp_path / "off" / "end_posterior_dist_all.tsv", sep="\t"))


def test_per_cell_keeps_per_mark_pairs_when_panel_a_is_joint(tmp_path):
    m = _mod()
    sp = _fixture(tmp_path)
    _drop_hcg(tmp_path, ["c0", "c1"])
    out = tmp_path / "out"
    assert m.main(["--posteriors", f"GCH={tmp_path / 'GCH'}",
                   "--posteriors", f"HCG={tmp_path / 'HCG'}", "--samples", sp,
                   "--min-pairs", "50", "--plot-format", "png",
                   "--output-dir", str(out)] + _REPRO + _PERCELL) == 0
    t = pd.read_csv(out / "end_posterior_percell.tsv", sep="\t")
    assert {"c0", "c1"} <= set(t.query("mark == 'GCH'")["cell_id"])


def test_null_chrom_any_crosses_chromosomes_and_same_does_not():
    import argparse
    m = _mod()
    j = _multi_chrom_pairs(seed=9)
    j = j.assign(gpos1=m.genome_coords(j["chrom"], j["pos1"]),
                 gpos2=m.genome_coords(j["chrom"], j["pos2"]))
    cols = {"g": ("p1_g", "p2_g"), "h": ("p1_h", "p2_h")}
    edges = m.build_edges(1000, 10_000_000, 8)
    d = j["distance"].to_numpy()
    frac = {}
    for mode in ("same", "any"):
        a = argparse.Namespace(null_type="distance-matched", null_window=1000,
                               null_anchor_window=10_000_000, null_chrom=mode)
        d_pos, d_vals, d_pi = m.end_table(j, cols, 0, m._coords(a))
        xa, xb, dn, anchor, aidx = m._draw_null(
            j, d, cols, d_pos, d_vals, d_pi, 0, edges, a,
            np.random.default_rng(2))
        # chromosome of each null end, read off the genome coordinate
        frac[mode] = float(((xa["_gpos"] // m._GENOME_STRIDE)
                            != (xb["_gpos"] // m._GENOME_STRIDE)).mean())
    assert frac["same"] == 0.0
    assert frac["any"] > 0.8


# ------------------------------------------------------------- link-shuffle --

def _link_frames(m, codes, n=60, shift=None, seed=0):
    """Identical links in every cell; both ends' values encode the cell.

    ``shift`` moves every cell after the first by that many bp, so none of
    their ends can fall in the anchor's bins.
    """
    rng = np.random.default_rng(seed)
    pos1 = 1_000_000 + np.arange(n) * 500
    dist = np.exp(rng.uniform(np.log(2_000), np.log(150_000), n)).astype(int)
    frames = {}
    for ci, code in enumerate(codes):
        p1 = pos1 + (shift if (shift and ci > 0) else 0)
        f = pd.DataFrame({"chrom": "chr1", "pos1": p1, "pos2": p1 + dist,
                          "distance": dist, "p1": float(code),
                          "p2": float(code)})
        frames[ci] = f.assign(gpos1=m.genome_coords(f["chrom"], f["pos1"]),
                              gpos2=m.genome_coords(f["chrom"], f["pos2"]))
    return frames


def _contact_pool(m, frames, cols, edges):
    return m._merge_end_tables([
        m.end_table(m._contact_ends(f, edges), cols, m._pair_base(ci),
                    "genome") for ci, f in frames.items()])


def test_link_shuffle_takes_end_2_from_another_cell_in_bin_y():
    m = _mod()
    edges = m.build_edges(1000, 10_000_000, 8)
    cols = {"g": ("p1", "p2")}
    frames = _link_frames(m, codes=[0.0, 0.1, 0.2])
    pool = _contact_pool(m, frames, cols, edges)
    a = _args(m, null_type="link-shuffle", null_scope="pooled",
              link_shuffle_bin=250_000)
    j = frames[0]
    d = j["distance"].to_numpy()
    xa, xb, dn, anchor, ai = m._draw_link_shuffle(
        j, d, cols, pool, m._pair_base(0), edges, a, np.random.default_rng(1))
    assert len(ai) > 0.9 * len(j), "identical links in other cells: most place"
    assert set(np.unique(xb["g"])) <= {0.1, 0.2}, "never the anchor's own cell"
    assert (np.digitize(dn, edges) == np.digitize(d[ai], edges)).all(), \
        "the new pair stays in its distance bin"
    local = lambda g: np.asarray(g) % m._GENOME_STRIDE      # noqa: E731
    g2 = j["gpos2"].to_numpy()[ai]
    assert (local(anchor + dn) // 250_000 == local(g2) // 250_000).all(), \
        "the stand-in end 2 sits in the real end 2's bin"


def test_link_shuffle_has_no_donor_outside_bin_y():
    m = _mod()
    edges = m.build_edges(1000, 10_000_000, 8)
    cols = {"g": ("p1", "p2")}
    frames = _link_frames(m, codes=[0.0, 0.9], shift=5_000_000)
    pool = _contact_pool(m, frames, cols, edges)
    a = _args(m, null_type="link-shuffle", null_scope="pooled",
              link_shuffle_bin=250_000)
    j = frames[0]
    got = m._draw_link_shuffle(j, j["distance"].to_numpy(), cols, pool,
                               m._pair_base(0), edges, a,
                               np.random.default_rng(1))
    assert got is None, "same distances elsewhere are not the same place"


def test_a_long_range_pair_keeps_a_donor_whose_partner_is_elsewhere():
    """The point of not requiring the donor's whole link to match: past the
    bin size, another cell rarely has a contact joining the same two bins,
    but it often has SOME contact end in bin y."""
    m = _mod()
    edges = m.build_edges(1000, 10_000_000, 8)
    cols = {"g": ("p1", "p2")}

    def frame(p1, p2, code):
        f = pd.DataFrame({"chrom": "chr1", "pos1": [p1], "pos2": [p2],
                          "distance": [abs(p2 - p1)], "p1": [code],
                          "p2": [code]})
        return f.assign(gpos1=m.genome_coords(f["chrom"], f["pos1"]),
                        gpos2=m.genome_coords(f["chrom"], f["pos2"]))

    anchor = frame(1_000_000, 1_600_000, 0.0)      # 600 kb, end 2 in bin y
    donor = frame(1_620_000, 9_000_000, 0.7)       # partner 7 Mb away
    pool = _contact_pool(m, {0: anchor, 1: donor}, cols, edges)
    a = _args(m, null_type="link-shuffle", null_scope="pooled",
              link_shuffle_bin=250_000)
    got = m._draw_link_shuffle(anchor, anchor["distance"].to_numpy(), cols,
                               pool, m._pair_base(0), edges, a,
                               np.random.default_rng(0))
    assert got is not None
    xa, xb, dn, _anc, ai = got
    assert xb["g"][0] == 0.7 and dn[0] == 620_000


def test_link_shuffle_bins_follow_the_per_chromosome_grid():
    """Bin y is the pipeline's chr:k*250000 bin, not a bin of the genome-wide
    coordinate, whose boundaries sit 184,592 bp off on chr1."""
    m = _mod()
    edges = m.build_edges(1000, 10_000_000, 8)
    cols = {"g": ("p1", "p2")}

    def frame(p1, p2, code):
        f = pd.DataFrame({"chrom": "chr1", "pos1": [p1], "pos2": [p2],
                          "distance": [abs(p2 - p1)], "p1": [code],
                          "p2": [code]})
        return f.assign(gpos1=m.genome_coords(f["chrom"], f["pos1"]),
                        gpos2=m.genome_coords(f["chrom"], f["pos2"]))

    a = _args(m, null_type="link-shuffle", null_scope="pooled",
              link_shuffle_bin=250_000)
    anchor = frame(400_000, 1_240_000, 0.0)        # end 2 in [1.00, 1.25) Mb
    for donor_end, expect in ((1_260_000, None), (1_215_000, 0.5)):
        donor = frame(9_000_000, donor_end, 0.5)   # partner far away
        pool = _contact_pool(m, {0: anchor, 1: donor}, cols, edges)
        got = m._draw_link_shuffle(anchor, anchor["distance"].to_numpy(),
                                   cols, pool, m._pair_base(0), edges, a,
                                   np.random.default_rng(0))
        if expect is None:
            assert got is None, "1.26 Mb is the next bin on chr1's grid"
        else:
            assert got is not None and got[1]["g"][0] == expect


def test_link_shuffle_null_pairs_never_cross_chromosomes():
    m = _mod()
    edges = m.build_edges(1000, 10_000_000, 8)
    cols = {"g": ("p1_g", "p2_g"), "h": ("p1_h", "p2_h")}
    j = _multi_chrom_pairs(seed=4)
    j = j.assign(gpos1=m.genome_coords(j["chrom"], j["pos1"]),
                 gpos2=m.genome_coords(j["chrom"], j["pos2"]))
    half = len(j) // 2
    cells = {0: j.iloc[:half].reset_index(drop=True),
             1: j.iloc[half:].reset_index(drop=True)}
    pool = _contact_pool(m, cells, cols, edges)
    a = _args(m, null_type="link-shuffle", null_scope="pooled",
              link_shuffle_bin=250_000)
    xa, xb, dn, anchor, ai = m._draw_link_shuffle(
        cells[0], cells[0]["distance"].to_numpy(), cols, pool,
        m._pair_base(0), edges, a, np.random.default_rng(5))
    g1 = cells[0]["gpos1"].to_numpy()[ai]
    g2 = cells[0]["gpos2"].to_numpy()[ai]
    donor = np.where(g2 >= g1, g1 + dn, g1 - dn)
    assert len(ai) > 0
    assert (_chrom_of(m, donor) == _chrom_of(m, g1)).all()


def test_link_shuffle_strata_are_cell_type_by_batch():
    import argparse
    m = _mod()
    s = pd.DataFrame({"cell_name": ["a", "b", "c", "d"],
                      "group": ["T1", "T1", "T2", "T2"],
                      "exp_batch": ["e1", "e2", "e1", "e1"]})
    a = argparse.Namespace(cell_column="cell_name", group_column="group")
    batch_of = {"a": "e1", "b": "e2", "c": "e1", "d": "e1"}
    st = m._link_strata(s, a, batch_of)
    assert st == {"a": "T1|e1", "b": "T1|e2", "c": "T2|e1", "d": "T2|e1"}


def test_link_shuffle_same_cell_runs_and_says_so(tmp_path):
    """Under --null-scope cell link-shuffle is the same-cell, same-bin null."""
    t, out = _run_link(tmp_path, ["--null-scope", "cell"], rho=1.0)
    assert (t["gg_null_n"] > 0).any()


def test_centring_is_on_by_default_only_across_cells():
    m = _mod()
    pooled = _args(m, null_type="link-shuffle", null_scope="pooled")
    cell = _args(m, null_type="link-shuffle", null_scope="cell")
    assert m._center_on(pooled) and not m._center_on(cell)
    assert m._center_on(_args(m, null_type="link-shuffle", null_scope="cell",
                              cell_center="on"))


def test_same_cell_link_shuffle_never_takes_the_real_partner():
    m = _mod()
    edges = m.build_edges(1000, 10_000_000, 8)
    cols = {"g": ("p1", "p2")}
    # the anchor (row 0): end 1 at 1.0 Mb, real partner at 1.6 Mb (0.9);
    # rows 1-5: other contacts of the SAME cell with an end in that bin at
    # ~600 kb from end 1 (0.3) -- their own partners far away
    p1 = [1_000_000] + [9_000_000 + k for k in range(5)]
    p2 = [1_600_000] + [1_560_000 + 20_000 * k for k in range(5)]
    f = pd.DataFrame({"chrom": "chr1", "pos1": p1, "pos2": p2})
    f["distance"] = (f["pos2"] - f["pos1"]).abs()
    f["p1"] = [0.1] + [0.3] * 5
    f["p2"] = [0.9] + [0.3] * 5
    f = f.assign(gpos1=m.genome_coords(f["chrom"], f["pos1"]),
                 gpos2=m.genome_coords(f["chrom"], f["pos2"]))
    base = m._pair_base(0)
    pool = m._own_contact_pool(f, cols, base, edges)
    a = _args(m, null_type="link-shuffle", null_scope="cell",
              link_shuffle_bin=250_000)
    seen = set()
    for seed in range(40):
        got = m._draw_link_shuffle(f.iloc[:1], f["distance"].to_numpy()[:1],
                                   cols, pool, base, edges, a,
                                   np.random.default_rng(seed))
        assert got is not None
        xa, xb, dn, anchor, ai = got
        seen.add(float(xb["g"][0]))
        assert np.digitize(dn[0], edges) == np.digitize(600_000, edges)
        local = (anchor[0] + dn[0]) % m._GENOME_STRIDE
        assert 1_500_000 <= local < 1_750_000
    assert seen == {0.3}, "the real partner (0.9) must never stand in"


def test_same_cell_null_needs_no_centring_for_a_cell_offset(tmp_path):
    """Same nucleus in both arms: a per-cell level cannot open a gap."""
    t, _ = _run_link(tmp_path, ["--null-scope", "cell"], offset_sd=1.0)
    assert abs(_gap(t)) < 0.05


def test_same_cell_null_finds_planted_coordination_past_the_bin(tmp_path):
    t, _ = _run_link(tmp_path, ["--null-scope", "cell"], rho=1.5)
    assert _gap(t) > 0.3
    far = t[t["bin_lo"] >= 250_000]
    assert (far["gg_null_n"] > 0).any()


def _link_fixture(tmp_path, n_cells=16, offset_sd=0.0, rho=0.0, seed=3,
                  rho_of=None):
    """The same links in every cell, so every stratum has donors.

    ``offset_sd`` gives each cell its own level (logit scale) -- shared by
    both ends of its pairs, which is all a cross-cell null lacks; ``rho`` a
    real per-pair coordination.
    """
    rng = np.random.default_rng(seed)
    n = 600
    pos1 = 100_000 + np.sort(rng.integers(0, 2_400_000, n))
    dist = np.exp(rng.uniform(np.log(2_000), np.log(800_000), n)).astype(int)
    cells = [f"c{i}" for i in range(n_cells)]
    for ctx in ("GCH", "HCG"):
        (tmp_path / ctx).mkdir(parents=True, exist_ok=True)
    for ci, c in enumerate(cells):
        off = rng.normal(0, offset_sd)
        shared = rng.normal(0, 1, n)
        r_c = rho if rho_of is None else rho_of(ci)
        for ctx in ("GCH", "HCG"):
            sig = lambda: 1 / (1 + np.exp(-(off + r_c * shared         # noqa
                                            + rng.normal(0, 0.6, n))))
            pd.DataFrame({"chrom": "chr1", "pos1": pos1,
                          "pos2": pos1 + dist, "n1": 6, "n2": 6,
                          "distance": dist, "posterior1": sig(),
                          "posterior2": sig()}).to_parquet(
                tmp_path / ctx / f"{c}.posteriors.parquet")
    sp = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells,
                  "group": np.repeat(["A", "B"], n_cells // 2),
                  "exp_batch": np.tile(["e1", "e1", "e2", "e2"], n_cells // 4),
                  "lib_batch": np.tile(["l1", "l2"], n_cells // 2)}).to_csv(
        sp, sep="\t", index=False)
    return str(sp)


def _run_link(tmp_path, extra=(), **fx):
    m = _mod()
    sp = _link_fixture(tmp_path, **fx)
    out = tmp_path / "out"
    argv = ["--posteriors", f"GCH={tmp_path / 'GCH'}",
            "--posteriors", f"HCG={tmp_path / 'HCG'}",
            "--samples", sp, "--output-dir", str(out), "--plot-format", "png",
            "--null-type", "link-shuffle", "--null-scope", "pooled",
            "--group-column", "group",
            "--null-match-column", "exp_batch,lib_batch",
            "--min-distance", "1000", "--max-distance", "1000000",
            "--bins-per-decade", "4", "--min-pairs", "50",
            "--n-bootstrap", "0", "--fisher-z", "none"] + list(extra)
    assert m.main(argv) == 0
    return pd.read_csv(out / "end_posterior_dist_all.tsv", sep="\t"), out


def _gap(t, m="gg"):
    ok = t[f"{m}_matched_r"].notna() & t[f"{m}_null_r"].notna()
    assert ok.any()
    return float((t.loc[ok, f"{m}_matched_r"] - t.loc[ok, f"{m}_null_r"]).mean())


def test_cell_centring_stops_a_cell_level_offset_reading_as_concordance(
        tmp_path):
    """Two ends of one nucleus share its level; a cross-cell null does not."""
    off, _ = _run_link(tmp_path / "off", ["--cell-center", "off"],
                       offset_sd=1.0)
    on, _ = _run_link(tmp_path / "on", offset_sd=1.0)      # auto = on here
    assert _gap(off) > 0.2, "uncentred: the cell's level reads as concordance"
    assert abs(_gap(on)) < 0.05, "centred: no coordination, no gap"


def test_link_shuffle_finds_planted_coordination(tmp_path):
    t, _ = _run_link(tmp_path, rho=1.5)
    assert _gap(t) > 0.3
    assert (t["gg_null_n"] > 0).sum() >= 3


def test_link_shuffle_curve_continues_past_the_bin_size(tmp_path):
    t, _ = _run_link(tmp_path, rho=1.5)
    far = t[t["bin_lo"] >= 250_000]
    assert (far["gg_null_n"] > 0).any(), "null pairs beyond 250 kb"
    assert far["gg_null_r"].notna().any()


def test_link_shuffle_feeds_the_per_cell_test(tmp_path):
    t, out = _run_link(tmp_path, ["--per-cell", "on",
                                  "--per-cell-min-pairs", "50"], rho=1.5)
    pc = pd.read_csv(out / "end_posterior_percell.tsv", sep="\t")
    g = pc[pc["mark"] == "GCH"]
    assert (g["delta"] > 0).mean() > 0.8


def test_link_shuffle_never_builds_a_run_wide_pool(tmp_path, monkeypatch):
    """The run-wide pool is what ran out of memory on 1,577 cells."""
    m = _mod()
    sp = _link_fixture(tmp_path, rho=1.0)

    def boom(*a, **k):
        raise AssertionError("run-wide donor pool built under link-shuffle")

    monkeypatch.setattr(m, "_build_pools", boom)
    assert m.main(["--posteriors", f"GCH={tmp_path / 'GCH'}",
                   "--samples", sp, "--output-dir", str(tmp_path / "o"),
                   "--plot-format", "png", "--null-type", "link-shuffle",
                   "--null-scope", "pooled", "--group-column", "group",
                   "--null-match-column", "exp_batch,lib_batch",
                   "--min-distance", "1000", "--max-distance", "1000000",
                   "--bins-per-decade", "4", "--min-pairs", "50",
                   "--n-bootstrap", "0", "--fisher-z", "none",
                   "--per-cell", "off"]) == 0


def test_link_shuffle_per_cell_rows_stay_with_their_cells(tmp_path):
    """Strata finish out of order; per_cell[i] must still be cells[i]."""
    planted = {1, 4, 6, 11, 13}
    t, out = _run_link(tmp_path, ["--per-cell", "on",
                                  "--per-cell-min-pairs", "50",
                                  "--n-threads", "3", "--backend", "thread"],
                       rho_of=lambda ci: 1.5 if ci in planted else 0.0)
    pc = pd.read_csv(out / "end_posterior_percell.tsv", sep="\t")
    g = pc[pc["mark"] == "GCH"].set_index("cell_id")["delta"]
    hi = {f"c{i}" for i in planted}
    assert g[list(hi)].min() > g.drop(list(hi)).max()


def test_link_shuffle_backends_agree(tmp_path):
    """Serial, thread and process runs of the stratum path: same numbers."""
    import subprocess
    sp = _link_fixture(tmp_path, rho=1.0)
    common = ["--posteriors", f"GCH={tmp_path / 'GCH'}",
              "--posteriors", f"HCG={tmp_path / 'HCG'}",
              "--samples", sp, "--plot-format", "png",
              "--null-type", "link-shuffle", "--null-scope", "pooled",
              "--group-column", "group",
              "--null-match-column", "exp_batch,lib_batch",
              "--min-distance", "1000", "--max-distance", "1000000",
              "--bins-per-decade", "4", "--min-pairs", "50",
              "--n-bootstrap", "0", "--fisher-z", "none"]
    got = {}
    for name, extra in (("serial", ["--n-threads", "1"]),
                        ("thread", ["--n-threads", "3", "--backend",
                                    "thread"]),
                        ("process", ["--n-threads", "3", "--backend",
                                     "process"])):
        out = tmp_path / name
        r = subprocess.run([sys.executable, str(_SCRIPT)] + common
                           + ["--output-dir", str(out)] + extra,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        got[name] = pd.read_csv(out / "end_posterior_dist_all.tsv", sep="\t")
    for name in ("thread", "process"):
        for c in got["serial"].columns:
            assert np.allclose(got["serial"][c], got[name][c],
                               equal_nan=True), (name, c)


# ----------------------------------------------------- panel a: two lines --

def _drawn(fig, row_axes=3):
    """Labelled lines of the top row: {label: ydata}."""
    out = []
    for ax in fig.axes[:row_axes]:
        out.append({ln.get_label(): np.asarray(ln.get_ydata(), float)
                    for ln in ax.get_lines()
                    if not ln.get_label().startswith("_")})
    return out


@pytest.mark.parametrize("arm,expect", [("matched", 0.3), ("linked", 0.5)])
def test_panel_a_draws_the_delta_arm_and_the_null_only(tmp_path, arm, expect):
    """Row 1 shows the pairs row 2's Δ is taken on, and nothing else."""
    import argparse
    m = _mod()
    centers = np.array([1e3, 1e4, 1e5])
    n = np.full(3, 500)
    level = {"linked": 0.5, "matched": 0.3, "null": 0.1}
    curves = {(k, c): (np.full(3, v), n)
              for k, _ in m._MEASURES for c, v in level.items()}
    a = argparse.Namespace(delta_arm=arm, fisher_z="none")
    fig = m._figure(curves, centers, "all", tmp_path / "a.png", a)
    for lines in _drawn(fig):
        assert len(lines) == 2, lines.keys()
        obs = lines[m._OBS_LABEL[arm]]
        assert np.allclose(obs, expect)
        assert np.allclose(lines["non-linked"], 0.1)
    # every bin is marked on BOTH lines, so the gap can be read point by point
    for ax in fig.axes[:3]:
        for ln in ax.get_lines():
            if not ln.get_label().startswith("_"):
                assert ln.get_marker() not in ("", "None", None), \
                    ln.get_label()



def test_cross_context_null_keeps_hcg_on_x_and_gch_on_y():
    """The bug: the null's second order was (GCH, HCG) while the linked
    arm's is (HCG, GCH).  Pooled into one correlation, the marks' different
    means -- methylation ~0.65, accessibility ~0.3 -- made an independent
    null read about -0.2 at every distance."""
    m = _mod()
    rng = np.random.default_rng(0)
    n = 20_000
    xa = {"h": rng.normal(0.65, 0.1, n), "g": rng.normal(0.3, 0.1, n)}
    xb = {"h": rng.normal(0.65, 0.1, n), "g": rng.normal(0.3, 0.1, n)}
    pairs = m._null_xy("gh", xa, xb)
    x = np.concatenate([p[0] for p in pairs])
    y = np.concatenate([p[1] for p in pairs])
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.03, "independent ends: r ~ 0"
    assert abs(x.mean() - 0.65) < 0.01 and abs(y.mean() - 0.3) < 0.01, \
        "x is HCG and y is GCH in both orders"
    # and the linked arm is built the same way
    f = pd.DataFrame({"p1_h": xa["h"], "p2_h": xb["h"],
                      "p1_g": xa["g"], "p2_g": xb["g"]})
    cols = {"g": ("p1_g", "p2_g"), "h": ("p1_h", "p2_h")}
    lx = np.concatenate([p[0] for p in m._linked_xy("gh", f, cols)])
    assert abs(lx.mean() - 0.65) < 0.01


# --------------------------------------------------------------- per-region --

def _region_fixture(tmp_path, pattern, n_cells=10, n_regions=40,
                    region=1_000_000, seed=5):
    """Coordination strength set per REGION, for each mark.

    'shared': both marks coordinated strongly in the same regions;
    'independent': each mark's strong regions drawn on its own.
    """
    rng = np.random.default_rng(seed)
    rho_g = rng.uniform(0.0, 1.5, n_regions)
    rho_h = (rho_g.copy() if pattern == "shared"
             else rng.uniform(0.0, 1.5, n_regions))
    for ctx in ("GCH", "HCG"):
        (tmp_path / ctx).mkdir(parents=True, exist_ok=True)
    cells = [f"c{i}" for i in range(n_cells)]
    per = 60
    for c in cells:
        recs = {"GCH": [], "HCG": []}
        for k in range(n_regions):
            p1 = k * region + rng.integers(0, region - 320_000, per)
            dist = np.exp(rng.uniform(np.log(2_000), np.log(250_000),
                                      per)).astype(int)
            for ctx, rho in (("GCH", rho_g[k]), ("HCG", rho_h[k])):
                s = rng.normal(0, 1, per)          # one latent per pair
                f = lambda: 1 / (1 + np.exp(-(rho * s                 # noqa
                                              + rng.normal(0, 0.6, per))))
                recs[ctx].append(pd.DataFrame({
                    "chrom": "chr1", "pos1": p1, "pos2": p1 + dist,
                    "n1": 6, "n2": 6, "distance": dist,
                    "posterior1": f(), "posterior2": f()}))
        for ctx in ("GCH", "HCG"):
            pd.concat(recs[ctx]).to_parquet(
                tmp_path / ctx / f"{c}.posteriors.parquet")
    sp = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "group": "A"}).to_csv(
        sp, sep="\t", index=False)
    return str(sp)


def _run_regions(tmp_path, pattern, extra=()):
    m = _mod()
    sp = _region_fixture(tmp_path, pattern)
    out = tmp_path / "out"
    argv = ["--posteriors", f"GCH={tmp_path / 'GCH'}",
            "--posteriors", f"HCG={tmp_path / 'HCG'}",
            "--samples", sp, "--output-dir", str(out), "--plot-format", "png",
            "--min-distance", "1000", "--max-distance", "300000",
            "--bins-per-decade", "4", "--min-pairs", "20",
            "--n-bootstrap", "0", "--fisher-z", "none",
            "--null-anchor-window", "200000", "--per-cell-min-pairs", "20",
            "--region-size", "1000000", "--region-min-pairs", "30"] + list(extra)
    assert m.main(argv) == 0
    return out


def test_regions_coordinated_in_the_same_places_correlate(tmp_path):
    out = _run_regions(tmp_path, "shared")
    s = pd.read_csv(out / "end_posterior_regions_summary.tsv", sep="\t")
    assert len(s) == 1
    row = s.iloc[0]
    assert row["pearson_r"] > 0.6 and row["spearman_rho"] > 0.5
    assert row["rotation_p"] < 0.05
    assert abs(row["control_r"]) < 0.4, "noise alone does not correlate"
    assert (out / "end_posterior_regions.png").exists()


def test_independently_placed_coordination_does_not(tmp_path):
    out = _run_regions(tmp_path, "independent")
    s = pd.read_csv(out / "end_posterior_regions_summary.tsv", sep="\t")
    assert abs(s.iloc[0]["pearson_r"]) < 0.45


def test_regions_sit_on_the_per_chromosome_grid(tmp_path):
    out = _run_regions(tmp_path, "shared")
    r = pd.read_csv(out / "end_posterior_regions.tsv", sep="\t")
    assert set(r["chrom"]) == {"chr1"}
    assert (r["start"] % 1_000_000 == 0).all()
    assert (r["end"] - r["start"] == 1_000_000).all()
    assert {"GCH", "HCG"} <= set(r["mark"])


def test_region_bands_split_the_distance_range(tmp_path):
    out = _run_regions(tmp_path, "shared",
                       ["--region-bands", "1000,50000,300000"])
    r = pd.read_csv(out / "end_posterior_regions.tsv", sep="\t")
    assert sorted(set(zip(r["band_lo"], r["band_hi"]))) == [
        (1000.0, 50000.0), (50000.0, 300000.0)]


def test_per_region_off_writes_nothing(tmp_path):
    out = _run_regions(tmp_path, "shared", ["--per-region", "off"])
    assert not (out / "end_posterior_regions.tsv").exists()
