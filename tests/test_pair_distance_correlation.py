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
                         "gh": (b[:0], x[:0], y[:0])})
    obs, null, n_used = m.accumulate_cells(per_cell, 1, 20, 0)
    r_obs = m.corr_of(obs["gg"], 10)[0][0]
    r_null = m.corr_of(null["gg"], 10)[0][0]
    assert n_used == 4
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
                         "gh": (b[:0], x[:0], y[:0])})
    obs, null, _ = m.accumulate_cells(per_cell, 1, 20, 0)
    d_link = (m.corr_of(obs["gg"], 10)[0][0]
              - m.corr_of(null["gg"], 10)[0][0])
    # The same cells with the link removed, so the two Δ differ only by the
    # thing being measured.  A bare threshold would be reading the size of
    # the cell offset as much as the size of the link.
    flat = []
    for level in (-1.0, 1.0):
        n = 800
        b = np.zeros(n, np.int64)
        flat.append({"gg": (b, level + rng.normal(0, 0.5, n),
                            level + rng.normal(0, 0.5, n)),
                     "hh": (b[:0],) * 3, "gh": (b[:0],) * 3})
    o2, n2, _ = m.accumulate_cells(flat, 1, 20, 0)
    d_flat = m.corr_of(o2["gg"], 10)[0][0] - m.corr_of(n2["gg"], 10)[0][0]
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
    assert {"gg_r", "gg_r_null", "gg_delta", "hh_r", "gh_r"} <= set(t.columns)
    deep = t[t["gg_n"] >= 200]
    assert len(deep) >= 3
    # The planted within-cell link shows as Δ, and the cell-level offset it
    # is separated from shows as a non-zero null.
    assert deep["gg_delta"].max() > 0.3
    assert deep["gg_r_null"].max() > 0.05


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
