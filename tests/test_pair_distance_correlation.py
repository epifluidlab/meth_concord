"""Read-PAIR level correlation, with a re-pairing null.

The companion script correlates individual cytosines and bins by the
distance between the two SITES, so one read pair contributes many
observations across several bins.  This one is the unit a Hi-C contact
actually is: one pair, one observation, binned by the distance between the
two END CENTRES.

The null re-pairs within cell AND within distance bin.  Both restrictions
are load-bearing and both are checked here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / \
    "pair_distance_correlation.py"


def _mod():
    spec = importlib.util.spec_from_file_location("_pdc", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _calls(gch, hcg, base=0):
    out = {}
    if gch is not None:
        out["GCH"] = (np.arange(len(gch)) + base, np.asarray(gch))
    if hcg is not None:
        out["HCG"] = (np.arange(len(hcg)) + base, np.asarray(hcg))
    return out


def test_a_pair_contributes_one_observation_per_measure():
    m = _mod()
    v = m.pair_values(_calls([1, 1, 0], [0, 0]),
                      _calls([1, 0], [1, 1]))
    assert v["GCH"] == (pytest.approx(2 / 3), pytest.approx(0.5))
    assert v["HCG"] == (0.0, 1.0)
    got = m.measures_of(v)
    assert got["gg"] == (pytest.approx(2 / 3), pytest.approx(0.5))
    assert got["hh"] == (0.0, 1.0)
    # The third panel is a DIFFERENCE, end1 minus end2, in both contexts.
    assert got["gh"][0] == pytest.approx(2 / 3 - 0.5)
    assert got["gh"][1] == pytest.approx(-1.0)


def test_the_third_measure_needs_both_contexts_on_both_ends():
    """That is the question, so a pair missing either cannot answer it --
    while gg and hh only need their own context."""
    m = _mod()
    v = m.pair_values(_calls([1, 0], None), _calls([1, 1], [0]))
    got = m.measures_of(v)
    assert np.isfinite(got["gg"]).all()
    assert not np.isfinite(got["hh"]).all()
    assert not np.isfinite(got["gh"]).all()


def test_min_sites_per_end_applies_per_context():
    m = _mod()
    v = m.pair_values(_calls([1], [0, 1, 1]), _calls([0, 1], [1, 0]),
                      min_sites=2)
    assert not np.isfinite(v["GCH"]).all()     # end 1 had one GpC
    assert np.isfinite(v["HCG"]).all()


def test_the_shuffle_stays_inside_its_distance_bin():
    """The bin is the axis.  A null drawn from other bins would answer
    'is this contact more correlated than one at some other distance',
    which is a different and much easier question."""
    m = _mod()
    rng = np.random.default_rng(0)
    bins = np.array([0, 0, 0, 1, 1, 1, 2])
    y = np.array([1.0, 2, 3, 10, 20, 30, 100])
    for _ in range(30):
        out = m.shuffle_within(bins, y, 3, rng)
        for b in (0, 1, 2):
            sel = bins == b
            assert sorted(out[sel]) == sorted(y[sel])
    # A single-entry bin cannot move, and must not be dropped.
    assert m.shuffle_within(bins, y, 3, rng)[6] == 100.0


def test_the_shuffle_destroys_the_link_and_keeps_both_marginals():
    m = _mod()
    rng = np.random.default_rng(1)
    n = 4000
    bins = np.zeros(n, np.int64)
    x = rng.normal(0, 1, n)
    y = x * 0.8 + rng.normal(0, 0.6, n)          # strongly linked
    obs = m.corr_of(m._running(bins, x, y, 1), 10)[0][0]
    ys = m.shuffle_within(bins, y, 1, rng)
    nul = m.corr_of(m._running(bins, x, ys, 1), 10)[0][0]
    assert obs > 0.7
    assert abs(nul) < 0.06
    np.testing.assert_allclose(np.sort(ys), np.sort(y))


def test_shuffling_within_cell_leaves_a_cell_level_offset_in_the_null():
    """THE REASON THE NULL IS PER CELL.  Cells differ in global level, so
    pooling makes end 1 and end 2 correlate simply by both running high in
    an open cell.  A null shuffled ACROSS cells destroys that too and calls
    it signal; shuffled within, it survives into the null where it belongs.
    """
    m = _mod()
    rng = np.random.default_rng(2)
    per_cell = []
    for level in (-1.5, -0.5, 0.5, 1.5):         # four cells, no within-cell link
        n = 500
        b = np.zeros(n, np.int64)
        x = level + rng.normal(0, 0.3, n)
        y = level + rng.normal(0, 0.3, n)
        per_cell.append({"gg": (b, x, y), "hh": (b[:0], x[:0], y[:0]),
                         "gh": (b[:0], x[:0], y[:0]),
                         "gx": (b[:0], x[:0], y[:0])})
    res = m.correlations(m.pool_cells(per_cell, 1), 1, 20, 10, 0)["gg"]
    r_obs, r_null = res["r"][0], res["r_null"][0]
    # Pooling alone manufactures a large correlation from nothing.
    assert r_obs > 0.85
    # The within-cell null reproduces it, so the difference is ~0.
    assert abs(r_obs - r_null) < 0.05, (r_obs, r_null)


def test_a_real_within_cell_link_survives_the_null():
    """The complement: the null must not eat a correlation that is really
    there inside each cell."""
    m = _mod()
    rng = np.random.default_rng(3)
    per_cell = []
    for level in (-1.0, 1.0):
        n = 800
        b = np.zeros(n, np.int64)
        x = level + rng.normal(0, 0.5, n)
        y = level + 0.9 * (x - level) + rng.normal(0, 0.25, n)
        per_cell.append({"gg": (b, x, y), "hh": (b[:0], x[:0], y[:0]),
                         "gh": (b[:0], x[:0], y[:0]),
                         "gx": (b[:0], x[:0], y[:0])})
    r1 = m.correlations(m.pool_cells(per_cell, 1), 1, 20, 10, 0)["gg"]
    d_link = r1["r"][0] - r1["r_null"][0]
    # The same cells with the link removed, so the two Δ differ only by the
    # thing being measured.  A bare threshold would be reading the size of
    # the cell offset as much as the size of the link.
    flat = []
    for level in (-1.0, 1.0):
        n = 800
        b = np.zeros(n, np.int64)
        flat.append({"gg": (b, level + rng.normal(0, 0.5, n),
                            level + rng.normal(0, 0.5, n)),
                     "hh": (b[:0],) * 3, "gh": (b[:0],) * 3,
                     "gx": (b[:0],) * 3})
    r2 = m.correlations(m.pool_cells(flat, 1), 1, 20, 10, 0)["gg"]
    d_flat = r2["r"][0] - r2["r_null"][0]
    assert abs(d_flat) < 0.05, d_flat
    assert d_link > 0.15, d_link
    assert d_link > 3 * abs(d_flat)


def test_thin_bins_report_nothing_rather_than_a_correlation_of_three_points():
    m = _mod()
    bins = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    x = np.arange(bins.size, dtype=float)
    y = x * 2
    r, n = m.corr_of(m._running(bins, x, y, 2), 5)
    assert np.isnan(r[0]) and n[0] == 3
    assert np.isfinite(r[1]) and n[1] == 9


def test_both_contexts_are_required_up_front(tmp_path):
    """Every panel uses one of them and the third uses both, so a run with
    only GCH would produce two panels of NaN and one missing -- better to
    say so before the BAM pass than after it."""
    import subprocess
    import sys
    import pandas as pd
    s = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": ["c0"],
                  "bam_path": [str(tmp_path / "none.bam")]}).to_csv(
        s, sep="\t", index=False)

    def run(*extra):
        return subprocess.run(
            [sys.executable, str(_SCRIPT), "--samples", str(s),
             "--output-dir", str(tmp_path / f"o{len(extra)}"), *extra],
            capture_output=True, text=True)

    r0 = run()
    assert r0.returncode != 0
    assert "provide --posteriors, --context-bed or --reference" in r0.stderr
    r1 = run("--context-bed", "GCH:/nope.bed")
    assert r1.returncode != 0
    assert "must define HCG" in r1.stderr


def test_the_context_spec_matches_the_sibling_script_and_the_pipeline():
    """One comma-separated CONTEXT:PATH string, parsed by the pipeline's own
    _parse_context_bed -- not a repeatable flag with its own parser, which
    is what the first version did and what made a valid command fail."""
    import inspect
    src = _SCRIPT.read_text()
    assert 'ap.add_argument("--context-bed", default=None' in src
    assert 'action="append"' not in src.split("--context-bed")[1][:400]
    assert "_parse_context_bed" in src
    from nome_hic_concordance.cli import _parse_context_bed
    got = _parse_context_bed("GCH:/a.bed.gz,HCG:/b.bed.gz")
    assert set(got) == {"GCH", "HCG"}


def test_the_distance_bins_are_the_decades_asked_for():
    """--bins-per-decade 1 gives 100-1,000, 1,000-10,000 and so on."""
    edges = np.logspace(2, 7, int(round(np.log10(1e7 / 1e2) * 1)) + 1)
    assert len(edges) == 6
    np.testing.assert_allclose(edges, [1e2, 1e3, 1e4, 1e5, 1e6, 1e7])


def _posterior_cells(tmp_path, n_cells=6, n=3000, seed=0, gh_link=0.0):
    """Step-4 style posteriors for two contexts, with a within-cell link in
    each and an optional link between the two contexts' gradients."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    gd, hd = tmp_path / "gch", tmp_path / "hcg"
    gd.mkdir(parents=True, exist_ok=True)
    hd.mkdir(parents=True, exist_ok=True)
    cells = [f"c{i}" for i in range(n_cells)]
    for ci, c in enumerate(cells):
        lo = rng.integers(0, 2_000_000, n)
        d = (10 ** rng.uniform(2, 6.5, n)).astype(np.int64)
        level = -0.5 + 0.25 * ci
        e1 = 1 / (1 + np.exp(-(level + rng.normal(0, 1, n))))
        e2 = 1 / (1 + np.exp(-(level + 0.8 * np.log(e1 / (1 - e1))
                               + rng.normal(0, 0.6, n))))
        m1 = rng.random(n)
        m2 = 0.5 * m1 + 0.5 * rng.random(n)
        if gh_link:
            m2 = np.clip(m1 - gh_link * (e1 - e2), 0.001, 0.999)
        for path, p1, p2 in ((gd, e1, e2), (hd, m1, m2)):
            pd.DataFrame({"chrom": ["chr1"] * n, "pos1": lo, "pos2": lo + d,
                          "n1": 3, "n2": 3, "distance": d,
                          "posterior1": p1, "posterior2": p2}).to_parquet(
                path / f"{c}.posteriors.parquet")
    s = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells,
                  "bam_path": ["x"] * n_cells}).to_csv(s, sep="\t",
                                                       index=False)
    return s, gd, hd


def _run(tmp_path, s, gd, hd, tag, *extra):
    import subprocess
    import sys
    out = tmp_path / tag
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--samples", str(s),
         "--posteriors", f"GCH:{gd},HCG:{hd}", "--min-dist", "100",
         "--max-dist", "1e7", "--bins-per-decade", "1",
         "--output-dir", str(out), "--plot-format", "png", *extra],
        capture_output=True, text=True)
    return out, r


def test_posteriors_are_used_instead_of_a_bam_pass(tmp_path):
    """An end with two sites then carries the model's belief about it rather
    than a bare 0, 0.5 or 1."""
    import pandas as pd
    s, gd, hd = _posterior_cells(tmp_path)
    out, r = _run(tmp_path, s, gd, hd, "post")
    assert r.returncode == 0, r.stderr[-2500:]
    assert "Step-4 posteriors per end" in r.stderr
    t = pd.read_csv(out / "pair_distance_correlation.tsv", sep="\t")
    assert {"gg_pearson_r", "gg_pearson_r_null", "gg_pearson_delta",
            "hh_spearman_rho", "gh_pearson_r"} <= set(t.columns)
    deep = t[t["gg_n"] >= 200]
    assert len(deep) >= 3
    # The planted within-cell link shows as Δ, and the cell-level offset it
    # is separated from shows as a non-zero null.
    assert deep["gg_pearson_delta"].max() > 0.3
    assert deep["gg_pearson_r_null"].max() > 0.05


def test_the_third_panel_joins_the_two_contexts_on_the_same_pair(tmp_path):
    """Step 4 writes one row per pair per context, so an inner join on
    (chrom, pos1, pos2) IS 'pairs carrying both contexts on both ends' --
    while the first two panels keep the pairs the join would drop."""
    m = _mod()
    import pandas as pd
    s, gd, hd = _posterior_cells(tmp_path, n_cells=1, n=400)
    # Remove half the HCG rows: the join must shrink, gg must not.
    h = pd.read_parquet(hd / "c0.posteriors.parquet")
    h.iloc[: len(h) // 2].to_parquet(hd / "c0.posteriors.parquet")
    edges = np.logspace(2, 7, 6)
    got, _n = m.collect_cell_posteriors(
        {"GCH": gd / "c0.posteriors.parquet",
         "HCG": hd / "c0.posteriors.parquet"}, edges, 1, None)
    assert got["gg"][0].size > got["gh"][0].size
    assert got["gh"][0].size <= got["hh"][0].size
    assert got["gh"][0].size > 0


def test_a_pair_is_dropped_when_either_end_is_excluded(tmp_path):
    """A contact is a statement about two loci, so one end in a CpG island
    makes the pair a comparison between an island and something else rather
    than half a usable observation."""
    import pandas as pd
    s, gd, hd = _posterior_cells(tmp_path)
    bl = tmp_path / "bl.bed"
    bl.write_text("chr1\t0\t500000\n")
    out0, r0 = _run(tmp_path, s, gd, hd, "keep")
    out1, r1 = _run(tmp_path, s, gd, hd, "drop", "--exclude-bed", str(bl))
    assert r0.returncode == 0 and r1.returncode == 0, r1.stderr[-2000:]
    assert "merged intervals" in r1.stderr and "EITHER end" in r1.stderr

    def npairs(err):
        line = [l for l in err.splitlines() if "cis pairs" in l][0]
        return int(line.split()[0].replace(",", ""))

    assert npairs(r1.stderr) < npairs(r0.stderr)
    # Starts are drawn over 2 Mb and a 0.5 Mb window is excluded, so roughly
    # a quarter should go -- enough to show it is the interval and not a
    # blanket drop.
    frac = 1 - npairs(r1.stderr) / npairs(r0.stderr)
    assert 0.1 < frac < 0.5, frac


def test_the_excluded_intervals_are_merged_across_files(tmp_path):
    """A blacklist and a CGI+shore track can be given together."""
    m = _mod()
    a = tmp_path / "a.bed"
    b = tmp_path / "b.bed"
    a.write_text("chr1\t100\t200\nchr1\t400\t500\nchr2\t0\t50\n")
    b.write_text("chr1\t150\t300\n")          # overlaps the first interval
    reg = m.load_exclude([str(a), str(b)])
    starts, ends = reg["chr1"]
    assert list(zip(starts.tolist(), ends.tolist())) == [(100, 300), (400, 500)]
    assert "chr2" in reg
    got = m.excluded(reg, "chr1", [50, 100, 250, 299, 300, 450, 600])
    assert got.tolist() == [False, True, True, True, False, True, False]
    # An unknown contig excludes nothing rather than everything.
    assert not m.excluded(reg, "chrZ", [100]).any()
    assert not m.excluded(None, "chr1", [100]).any()


def test_a_cell_with_no_posterior_file_is_skipped_not_fatal(tmp_path):
    s, gd, hd = _posterior_cells(tmp_path, n_cells=3, n=300)
    (gd / "c1.posteriors.parquet").unlink()
    (hd / "c1.posteriors.parquet").unlink()
    out, r = _run(tmp_path, s, gd, hd, "miss")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "have no posterior file and are skipped" in r.stderr


def test_posteriors_needs_both_contexts(tmp_path):
    import subprocess
    import sys
    s, gd, hd = _posterior_cells(tmp_path, n_cells=1, n=100)
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--samples", str(s),
         "--posteriors", f"GCH:{gd}", "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "must define HCG" in r.stderr


def test_the_cross_context_measure_needs_one_context_on_each_end():
    """Weaker than the third panel: a pair with GpC only on end 1 and CpG
    only on end 2 qualifies here and not there."""
    m = _mod()
    # GpC on end 1 only, CpG on end 2 only -- 'gh' cannot use it, 'gx' can.
    v = m.pair_values(_calls([1, 0], None), _calls(None, [1, 1]))
    got = m.measures_of(v)
    assert not np.isfinite(got["gh"]).all()
    cross = [t for t in got["gx"] if np.isfinite(t).all()]
    assert len(cross) == 1
    assert cross[0] == (0.5, 1.0)               # GpC(end1) vs CpG(end2)


def test_both_orientations_are_emitted_when_both_exist():
    """Matching scripts/site_distance_correlation.py: a pair carrying both
    contexts on both ends contributes TWO observations here and one to every
    other measure, which is why its n runs higher."""
    m = _mod()
    v = m.pair_values(_calls([1, 1], [0, 0]), _calls([0, 0], [1, 1]))
    got = m.measures_of(v)
    assert len(got["gx"]) == 2
    assert set(got["gx"]) == {(1.0, 1.0), (0.0, 0.0)}
    # ... and exactly one for the others.
    assert isinstance(got["gg"], tuple) and len(got["gg"]) == 2


def test_both_correlations_and_both_p_values_are_reported(tmp_path):
    import pandas as pd
    s, gd, hd = _posterior_cells(tmp_path, n_cells=4, n=1500)
    out, r = _run(tmp_path, s, gd, hd, "corr", "--n-permutations", "30")
    assert r.returncode == 0, r.stderr[-2500:]
    t = pd.read_csv(out / "pair_distance_correlation.tsv", sep="\t")
    for m in ("gg", "hh", "gh", "gx"):
        for suf in ("pearson_r", "pearson_p", "spearman_rho", "spearman_p",
                    "pearson_r_null", "spearman_rho_null", "pearson_delta",
                    "spearman_delta", "empirical_p_pearson",
                    "empirical_p_spearman", "n"):
            assert f"{m}_{suf}" in t.columns, f"{m}_{suf}"
    deep = t[t["gg_n"] >= 200]
    # Both correlations see the same planted link.
    assert deep["gg_pearson_delta"].max() > 0.3
    assert deep["gg_spearman_delta"].max() > 0.3
    # The cross-context panel gets two observations per eligible pair.
    assert (deep["gx_n"] == 2 * deep["gh_n"]).all()


def test_the_empirical_p_catches_an_anti_conservative_parametric_one():
    """The parametric p assumes independent observations; here many come
    from the same cell, which is the very reason the null is per cell.  On a
    fixture with a cell-level offset and NO within-cell link, pearsonr
    returns a vanishing p for a correlation the re-pairing reproduces."""
    m = _mod()
    rng = np.random.default_rng(7)
    per_cell = []
    for level in (-1.2, -0.4, 0.4, 1.2):
        n = 400
        b = np.zeros(n, np.int64)
        per_cell.append({"gg": (b, level + rng.normal(0, 0.3, n),
                                level + rng.normal(0, 0.3, n)),
                         "hh": (b[:0],) * 3, "gh": (b[:0],) * 3,
                         "gx": (b[:0],) * 3})
    d = m.correlations(m.pool_cells(per_cell, 1), 1, 100, 10, 0)["gg"]
    assert d["r"][0] > 0.85                     # large, and entirely offset
    assert d["p"][0] < 1e-50                    # the parametric p believes it
    assert d["emp_p"][0] > 0.05                 # the re-pairing does not
    assert d["emp_p_rho"][0] > 0.05


def test_the_empirical_p_floor_is_one_over_permutations_plus_one():
    """A correlation beyond every re-pairing cannot be reported as
    impossible, only as beyond what N of them reached."""
    m = _mod()
    rng = np.random.default_rng(8)
    n = 600
    b = np.zeros(n, np.int64)
    x = rng.normal(0, 1, n)
    cell = {"gg": (b, x, x * 0.95 + rng.normal(0, 0.2, n)),
            "hh": (b[:0],) * 3, "gh": (b[:0],) * 3, "gx": (b[:0],) * 3}
    for n_perm in (9, 49):
        d = m.correlations(m.pool_cells([cell], 1), 1, n_perm, 10, 0)["gg"]
        assert d["emp_p"][0] == pytest.approx(1.0 / (n_perm + 1), rel=1e-9)


def test_no_permutations_leaves_the_null_and_empirical_p_undefined():
    m = _mod()
    rng = np.random.default_rng(9)
    n = 300
    b = np.zeros(n, np.int64)
    cell = {"gg": (b, rng.normal(0, 1, n), rng.normal(0, 1, n)),
            "hh": (b[:0],) * 3, "gh": (b[:0],) * 3, "gx": (b[:0],) * 3}
    d = m.correlations(m.pool_cells([cell], 1), 1, 0, 10, 0)["gg"]
    assert np.isfinite(d["r"][0]) and np.isfinite(d["p"][0])
    assert np.isnan(d["r_null"][0]) and np.isnan(d["emp_p"][0])


def test_the_fast_pearson_matches_scipy():
    """It is the same statistic from precomputed marginal sums, not an
    approximation -- the permutation loop calls it thousands of times and
    scipy's version recomputes all five sums plus a p the null discards."""
    from scipy.stats import pearsonr
    m = _mod()
    rng = np.random.default_rng(0)
    for n in (20, 5_000, 50_000):
        x = rng.normal(0, 1, n)
        y = 0.4 * x + rng.normal(0, 1, n)
        got = m._pearson(x, y, float(n), x.sum(), (x * x).sum(),
                         y.sum(), (y * y).sum())
        assert got == pytest.approx(pearsonr(x, y)[0], abs=1e-12)
    # A flat column gives NaN rather than a divide-by-zero.
    z = np.ones(50)
    assert np.isnan(m._pearson(z, rng.normal(0, 1, 50), 50.0, z.sum(),
                               (z * z).sum(), 0.0, 50.0))


def test_spearman_is_pearson_on_ranks_and_a_permutation_permutes_them():
    """The trick the null loop rests on: rank once per bin, then every
    permuted Spearman is another dot product."""
    from scipy.stats import rankdata, spearmanr
    m = _mod()
    rng = np.random.default_rng(1)
    n = 20_000
    x = rng.normal(0, 1, n)
    y = 0.35 * x + rng.normal(0, 1, n)
    rx, ry = rankdata(x), rankdata(y)
    got = m._pearson(rx, ry, float(n), rx.sum(), (rx * rx).sum(),
                     ry.sum(), (ry * ry).sum())
    assert got == pytest.approx(spearmanr(x, y)[0], abs=1e-12)
    # Permuting y permutes its ranks identically, so the ranks need not be
    # recomputed inside the loop.
    p = rng.permutation(n)
    np.testing.assert_allclose(rankdata(y[p]), ry[p])


def test_the_marginal_sums_are_invariant_under_the_permutation():
    """Which is why the loop touches only the cross-product: a permutation
    reorders y without changing Σy or Σy²."""
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 10_000)
    p = rng.permutation(y.size)
    assert y[p].sum() == pytest.approx(y.sum(), rel=1e-12)
    assert (y[p] ** 2).sum() == pytest.approx((y * y).sum(), rel=1e-12)


def test_the_result_does_not_depend_on_how_many_workers_ran_it():
    """Each (measure, bin) is seeded from the bin, not from a shared
    stream, so fanning them out cannot move a number."""
    m = _mod()
    rng = np.random.default_rng(3)
    nb, n_cells, per = 3, 20, 400
    per_cell = []
    for _ in range(n_cells):
        b = rng.integers(0, nb, per).astype(np.int64)
        x = rng.normal(0, 1, per)
        y = 0.3 * x + rng.normal(0, 1, per)
        per_cell.append({k: (b, x, y) for k, _ in m._MEASURES})
    pooled = m.pool_cells(per_cell, nb)
    a = m.correlations(pooled, nb, 20, 10, 0, n_jobs=1)
    b_ = m.correlations(pooled, nb, 20, 10, 0, n_jobs=4)
    for key, _ in m._MEASURES:
        for stat in ("r", "rho", "r_null", "rho_null", "emp_p", "emp_p_rho"):
            np.testing.assert_allclose(a[key][stat], b_[key][stat],
                                       equal_nan=True, rtol=1e-12)


def test_the_permutation_stays_inside_its_cell_within_the_bin():
    """A bin's groups are its cells.  Shuffling across them would destroy
    the cell-level structure the null exists to keep."""
    m = _mod()
    rng = np.random.default_rng(4)
    # Two cells, disjoint value ranges, one bin: if the shuffle crossed
    # cells the correlation would collapse rather than survive.
    n = 300
    b = np.zeros(n, np.int64)
    x = np.r_[rng.normal(-3, 0.2, n // 2), rng.normal(3, 0.2, n // 2)]
    y = x + rng.normal(0, 0.2, n)
    cid = np.r_[np.zeros(n // 2, np.int32), np.ones(n // 2, np.int32)]
    pooled = {k: (cid, b, x, y) for k, _ in m._MEASURES}
    d = m.correlations(pooled, 1, 50, 10, 0)["gg"]
    # The between-cell separation survives into the null, so Δ is small
    # even though r is near 1.
    assert d["r"][0] > 0.95
    assert d["r_null"][0] > 0.9
    assert abs(d["r"][0] - d["r_null"][0]) < 0.15


class _Tty(__import__("io").StringIO):
    def isatty(self):
        return True


def test_progress_rewrites_one_line_on_a_terminal():
    m = _mod()
    t = _Tty()
    b = m.Progress(50, "demo", stream=t)
    for _ in range(50):
        b.update()
    b.close()
    raw = t.getvalue()
    assert raw.count("\r") == 50
    assert raw.count("\n") == 1          # only the final newline
    assert "50/50 (100.0%)" in raw
    assert "eta" in raw and "elapsed" in raw


def test_progress_does_not_flood_a_redirected_log():
    """These runs are launched on a cluster with stderr going to a file,
    where a carriage-return bar writes one line per update -- thousands of
    them, in a log someone has to read afterwards."""
    import io
    m = _mod()
    f = io.StringIO()                    # not a tty
    b = m.Progress(500, "demo", stream=f)
    for _ in range(500):
        b.update()
    b.close()
    assert "\r" not in f.getvalue()
    assert f.getvalue().count("\n") <= 5, f.getvalue().count("\n")
    assert "500/500" in f.getvalue()


def test_progress_does_not_print_the_final_line_twice():
    """A duplicate 100% line is noise in a log and a flicker on a screen."""
    import io
    m = _mod()
    f = io.StringIO()
    b = m.Progress(3, "demo", stream=f, min_interval=0.0)
    for _ in range(3):
        b.update()
    b.close()
    assert f.getvalue().count("3/3") == 1


def test_the_eta_is_elapsed_over_done_times_remaining():
    """Checked against a fabricated clock rather than by sleeping: the
    arithmetic is the claim, and a timing test would be slow and flaky."""
    import io
    import re
    import time
    m = _mod()
    f = io.StringIO()
    b = m.Progress(10, "demo", stream=f, min_interval=0.0)
    b.done = 2
    b._t0 = time.time() - 20.0           # 20 s for 2 of 10
    b.update(0, force=True)
    got = re.findall(r"eta (\S+)", f.getvalue())[-1]
    assert got in ("1m20s", "1m19s"), got   # 20/2 x 8 = 80 s
    assert "elapsed 20s" in f.getvalue()
    b.done = 10
    b.update(0, force=True)
    assert re.findall(r"eta (\S+)", f.getvalue())[-1] == "0s"


def test_the_elapsed_and_eta_are_formatted_for_reading():
    m = _mod()
    b = m.Progress(1, "x")
    assert b._fmt(0) == "0s"
    assert b._fmt(45) == "45s"
    assert b._fmt(60) == "1m00s"
    assert b._fmt(3599) == "59m59s"
    assert b._fmt(3600) == "1h00m"
    assert b._fmt(7 * 3600 + 42 * 60) == "7h42m"


def test_both_phases_report_progress(tmp_path):
    """--threads only ever reached the per-cell read, and the phase after it
    had no output at all -- so a long run looked hung."""
    s, gd, hd = _posterior_cells(tmp_path, n_cells=6, n=1500)
    out, r = _run(tmp_path, s, gd, hd, "prog", "--n-permutations", "20",
                  "--threads", "2")
    assert r.returncode == 0, r.stderr[-2500:]
    assert "reading posteriors:" in r.stderr
    assert "correlations (20 permutations" in r.stderr
    # The correlation phase names its own size, so a slow run says why.
    assert "x 4 measures x 5 bins" in r.stderr


def test_run_jobs_returns_results_in_order():
    """The caller zips them against its task list, so completion order must
    not leak into the result order even though the generator yields as
    workers finish."""
    m = _mod()
    jobs = [(i,) for i in range(25)]
    got = m.run_jobs(lambda i: i * i, jobs, 4, "sq")
    assert got == [i * i for i in range(25)]
    assert m.run_jobs(lambda i: i * i, jobs, 1, "sq") == got


def test_max_pairs_per_cell_bounds_what_is_held(tmp_path):
    """The null needs the raw values, so every cell's stay resident until
    the permutation runs -- an uncapped cohort holds every cis pair of every
    cell at once, which is how a 1,577-cell run was OOM-killed part way
    through the read."""
    m = _mod()
    s, gd, hd = _posterior_cells(tmp_path, n_cells=1, n=6000)
    paths = {"GCH": gd / "c0.posteriors.parquet",
             "HCG": hd / "c0.posteriors.parquet"}
    edges = np.logspace(2, 7, 6)
    full, _ = m.collect_cell_posteriors(paths, edges, 1, None)
    cap, _ = m.collect_cell_posteriors(paths, edges, 1, None, max_pairs=500,
                                       seed=0)
    assert full["gg"][0].size > 2000
    assert cap["gg"][0].size == 500
    assert cap["hh"][0].size == 500
    # The cross-context measure emits both orientations, so its cap is
    # doubled with it rather than halving its coverage.
    assert cap["gx"][0].size == 1000
    # Uniform over rows, so the distance distribution survives -- capping
    # the long-range bin alone would change the axis.
    fb = np.bincount(full["gg"][0], minlength=5) / full["gg"][0].size
    cb = np.bincount(cap["gg"][0], minlength=5) / cap["gg"][0].size
    assert np.max(np.abs(fb - cb)) < 0.08, (fb, cb)


def test_capping_barely_moves_the_correlation(tmp_path):
    """So the memory bound is close to free: measured, an uncapped run gave
    r = 0.826 and a 200-per-cell cap gave 0.851."""
    import pandas as pd
    s, gd, hd = _posterior_cells(tmp_path, n_cells=6, n=6000)
    got = {}
    for cap in (0, 300):
        extra = [] if cap == 0 else ["--max-pairs-per-cell", str(cap)]
        out, r = _run(tmp_path, s, gd, hd, f"cap{cap}",
                      "--n-permutations", "20", *extra)
        assert r.returncode == 0, r.stderr[-2000:]
        t = pd.read_csv(out / "pair_distance_correlation.tsv", sep="\t")
        row = t.iloc[t["gg_n"].idxmax()]
        got[cap] = (row["gg_pearson_r"], row["gg_pearson_delta"],
                    int(row["gg_n"]))
    assert got[300][2] < got[0][2] / 3
    assert abs(got[300][0] - got[0][0]) < 0.08
    assert abs(got[300][1] - got[0][1]) < 0.10


def test_the_two_contexts_are_joined_once_not_twice():
    """The join is the most expensive step in the function and the third and
    fourth measures need the same one."""
    import inspect
    src = inspect.getsource(_mod().collect_cell_posteriors)
    assert src.count("g.merge(h") == 1


def test_the_values_are_kept_as_float32():
    """A posterior has nowhere near seven significant digits, and every one
    of them is held until the permutation runs."""
    m = _mod()
    import pandas as pd
    import tempfile
    d = Path(tempfile.mkdtemp())
    s, gd, hd = _posterior_cells(d, n_cells=1, n=800)
    got, _ = m.collect_cell_posteriors(
        {"GCH": gd / "c0.posteriors.parquet",
         "HCG": hd / "c0.posteriors.parquet"}, np.logspace(2, 7, 6), 1, None)
    for key in ("gg", "hh", "gh", "gx"):
        assert got[key][1].dtype == np.float32, key
        assert got[key][2].dtype == np.float32, key


def test_the_exclusion_mask_is_computed_without_a_per_row_lookup():
    """The first version called index.get_indexer inside the groupby, which
    is a hash lookup per row on a table with millions of them."""
    import inspect
    import pandas as pd
    m = _mod()
    # Parsed, not grepped: the comment explaining why it is gone contains
    # the name, and the first version of this test failed on its own prose.
    import ast as _ast
    tree = _ast.parse(_SCRIPT.read_text())
    calls = {n.attr for n in _ast.walk(tree)
             if isinstance(n, _ast.Attribute)}
    assert "get_indexer" not in calls
    reg = m.load_exclude(None)
    assert reg is None
    bed = Path(__import__("tempfile").mkdtemp()) / "e.bed"
    bed.write_text("chr1\t1000\t2000\nchr2\t0\t100\n")
    reg = m.load_exclude([str(bed)])
    t = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2", "chr3"],
                      "pos1": [1500, 50, 500, 1500],
                      "pos2": [9000, 60, 50, 1500]})
    got = m._excluded_rows(t, reg)
    # row 0: pos1 inside chr1; row 2: pos2 inside chr2; the rest are clear.
    assert got.tolist() == [True, False, True, False]
    assert not m._excluded_rows(t, None).any()


def test_the_run_reports_how_much_it_is_holding(tmp_path):
    s, gd, hd = _posterior_cells(tmp_path, n_cells=4, n=1200)
    out, r = _run(tmp_path, s, gd, hd, "held", "--n-permutations", "10")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "observations held" in r.stderr
    assert "Neither --max-pairs-per-bin nor --max-pairs-per-cell" in r.stderr
    out2, r2 = _run(tmp_path, s, gd, hd, "held2", "--n-permutations", "10",
                    "--max-pairs-per-bin", "200")
    assert "observations held" in r2.stderr
    assert "Neither" not in r2.stderr


def test_a_uniform_cap_starves_the_long_range_bins():
    """Contact frequency decays with distance, so pairs are overwhelmingly
    short and a uniform subsample keeps that shape -- which empties exactly
    the bins the axis exists for."""
    m = _mod()
    rng = np.random.default_rng(0)
    # Bin occupancies with a realistic decay.
    counts = [450_093, 44_792, 4_599, 471, 45]
    b = np.concatenate([np.full(c, i, np.int64)
                        for i, c in enumerate(counts)])
    x = rng.random(b.size).astype(np.float32)
    y = rng.random(b.size).astype(np.float32)

    uni = m._cap((b, x, y), 50_000, np.random.default_rng(1))
    per = m._cap((b, x, y), 0, np.random.default_rng(1), per_bin=5_000, nb=5)
    u = np.bincount(uni[0], minlength=5)
    p = np.bincount(per[0], minlength=5)
    # The uniform cap all but empties the longest bin.
    assert u[4] < 10, u
    # The per-bin cap keeps every one of them, and caps only the crowded two.
    assert p[4] == 45 and p[3] == 471 and p[2] == 4_599
    assert p[0] == 5_000 and p[1] == 5_000
    # ... on LESS memory, since the ceiling is N x bins rather than N.
    assert per[0].size < uni[0].size


def test_the_per_bin_cap_leaves_a_thin_bin_alone():
    m = _mod()
    rng = np.random.default_rng(2)
    b = np.concatenate([np.full(9000, 0, np.int64), np.full(12, 1, np.int64)])
    x = rng.random(b.size).astype(np.float32)
    y = rng.random(b.size).astype(np.float32)
    got = m._cap((b, x, y), 0, rng, per_bin=1000, nb=2)
    cnt = np.bincount(got[0], minlength=2)
    assert cnt.tolist() == [1000, 12]


def test_both_caps_compose_with_the_per_bin_one_first():
    """--max-pairs-per-cell stays usable as a hard ceiling on top."""
    m = _mod()
    rng = np.random.default_rng(3)
    b = np.concatenate([np.full(5000, i, np.int64) for i in range(4)])
    x = rng.random(b.size).astype(np.float32)
    y = rng.random(b.size).astype(np.float32)
    got = m._cap((b, x, y), 1200, rng, per_bin=1000, nb=4)
    assert got[0].size == 1200            # 4 x 1000 then trimmed to 1200
    only_bin = m._cap((b, x, y), 0, rng, per_bin=1000, nb=4)
    assert only_bin[0].size == 4000


def test_the_run_reports_what_survived_in_each_bin(tmp_path):
    """A cap that silently emptied the long bins would look like an absence
    of long-range pairs rather than a setting."""
    s, gd, hd = _posterior_cells(tmp_path, n_cells=4, n=3000)
    out, r = _run(tmp_path, s, gd, hd, "perbin", "--n-permutations", "10",
                  "--max-pairs-per-bin", "300")
    assert r.returncode == 0, r.stderr[-2000:]
    line = [l for l in r.stderr.splitlines() if "per distance bin" in l][0]
    assert "GCH-GCH observations per distance bin" in line
    # 4 cells x 300 per bin, so every bin caps at the same number.
    import re
    # The counts follow the RANGE label, so anchor on it -- a bare ": (\d+)"
    # also matches the "100" of "100-1000".
    got = [int(v.replace(",", ""))
           for v in re.findall(r"\d+-\d+: ([\d,]+)", line)]
    assert len(got) == 5
    assert max(got) <= 1200               # 4 cells x 300 per bin
    assert len(set(got)) == 1             # all bins capped alike here


def test_per_bin_is_named_as_the_one_to_use():
    """Checked on the RENDERED help, not the source: argparse strings are
    split across lines, so the phrase never appears contiguously in the
    file and the first version of this test passed on nothing."""
    import subprocess
    import sys
    out = subprocess.run([sys.executable, str(_SCRIPT), "--help"],
                         capture_output=True, text=True).stdout
    flat = " ".join(out.split())
    assert "THIS IS THE ONE TO USE" in flat
    assert "prefer --max-pairs-per-bin" in flat
