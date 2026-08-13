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
    (pos, vals, pi), = pools.values()
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
