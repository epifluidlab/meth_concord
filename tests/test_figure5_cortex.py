"""figure5_cortex.py: the cortex concordance figure from the runs' tables.

Small synthetic copies of each table, as the scripts write them; the figure
must draw from them, take panel b's regions as the run's summary did, read
a class pair in either order, test what the hot bins are against genome
rotations, and stop, naming them, without its tables."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "figure5_cortex",
    Path(__file__).resolve().parents[1] / "scripts" / "figure5_cortex.py")
f5 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(f5)

SYSTEMS = {"ab": ("A", "B"), "rt": ("early", "late"), "pmd": ("HMD", "PMD"),
           "chromhmm": f5.STATES}
BANDS = (("1kb-10kb", 1e3, 1e4), (f5.ALL_BAND, 1e3, 1e7))


def _distance(d: Path, rng):
    edges = np.logspace(2, 7, 41)
    t = pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:],
                      "center_bp": np.sqrt(edges[:-1] * edges[1:])})
    # the null matches few of the short bins' linked pairs, as it does
    share = np.clip((np.arange(40) - 10) / 20, 0, 1)
    for k, base in (("gg", 0.1), ("hh", 0.3), ("gh", -0.02)):
        t[f"{k}_linked_n"] = 100_000
        t[f"{k}_matched_n"] = (100_000 * share).astype(int)
        t[f"{k}_matched_r"] = np.where(share > 0, base + 0.03, np.nan)
        t[f"{k}_null_r"] = np.where(share > 0, base, np.nan)
        t[f"{k}_delta_lo"], t[f"{k}_delta_hi"] = 0.02, 0.04
        t[f"{k}_delta_q"] = 0.01
    t.to_csv(d / "end_posterior_dist_all.tsv", sep="\t", index=False)
    rows = []
    for i in range(120):
        g = list(f5.CELL_TYPES)[i % 10]
        for mark, mu in (("GCH", 0.05), ("HCG", 0.02)):
            rows.append((f"c{i}", g, mark, rng.normal(mu, 0.02),
                         rng.normal(0, 0.02)))
    pd.DataFrame(rows, columns=["cell_id", "group", "mark", "delta",
                                "delta_null"]).to_csv(
        d / "end_posterior_percell.tsv", sep="\t", index=False)
    reg = []
    for c in range(1, 4):
        for s in range(0, 200_000_000, 10_000_000):
            x = rng.normal(0.08, 0.03)
            for mark, v, n in (("GCH", x, 5000),
                               ("HCG", 0.5 * x + rng.normal(0, 0.01),
                                # a thin HCG region, left out as the run did
                                100 if s == 0 else 900)):
                reg.append((f"chr{c}", s, s + 10_000_000, 1e3, 1e7, mark, n,
                            n, v, rng.normal(0, 0.01), c))
    pd.DataFrame(reg, columns=["chrom", "start", "end", "band_lo", "band_hi",
                               "mark", "n_obs", "n_null", "delta",
                               "delta_ctl", "chrom_code"]).to_csv(
        d / "end_posterior_regions.tsv", sep="\t", index=False)
    pd.DataFrame([{"band_lo": 1e3, "band_hi": 1e7, "n_regions": 57,
                   "pearson_r": 0.5, "spearman_rho": 0.4, "rotation_p": 0.01,
                   "control_r": 0.0}]).to_csv(
        d / "end_posterior_regions_summary.tsv", sep="\t", index=False)


def _classes(c: Path, rng):
    rows = []
    for ctx in ("GCH", "HCG"):
        for sy, cls in SYSTEMS.items():
            for i, a_ in enumerate(cls):
                # one orientation per unordered pair, as the run stores them
                for b_ in cls[i:]:
                    for band, lo, hi in BANDS:
                        v = rng.uniform(0.005, 0.02)
                        w = rng.uniform(-0.002, 0.002)
                        rows.append({
                            "context": ctx, "system": sy, "anchor": b_,
                            "partner": a_, "band": band, "band_lo": lo,
                            "band_hi": hi, "delta": v,
                            "delta_lo": v - 0.002, "delta_hi": v + 0.002,
                            "control": 0.0, "control_lo": -0.001,
                            "control_hi": 0.001, "signed_rel": w,
                            "signed_rel_lo": w - 0.001,
                            "signed_rel_hi": w + 0.001,
                            "signed_rel_control": 0.0,
                            "signed_rel_control_lo": -0.0005,
                            "signed_rel_control_hi": 0.0005,
                            "sparse": False})
    pd.DataFrame(rows).to_csv(c / "summary.tsv", sep="\t", index=False)
    hot = []
    for ctx in ("GCH", "HCG"):
        for chrom in ("chr1", "chr2", "chr10"):
            for s in range(0, 50_000_000, 1_000_000):
                v = rng.normal(0.014, 0.004)
                row = {"context": ctx, "chrom": chrom, "start": s,
                       "end": s + 1_000_000, "band": f5.ALL_BAND,
                       "delta": v, "control": rng.normal(0, 0.003),
                       "se_control": 0.003, "tested": True,
                       "hot": v > 0.022, "cold": False}
                # the bins' tracks, as the run writes them beside each bin,
                # a --hotspot-annotation BED among them
                row.update({t: rng.uniform() for t, _ in f5.G_TRACKS})
                row["lamina (BED)"] = rng.uniform()
                row[f5.TSS_TRACK] = rng.poisson(20)
                hot.append(row)
    pd.DataFrame(hot).to_csv(c / "locus_hotspots.tsv.gz", sep="\t",
                             index=False)


def _tables(root: Path):
    rng = np.random.default_rng(0)
    d, c = root / "dist", root / "classes"
    d.mkdir(parents=True)
    c.mkdir(parents=True)
    _distance(d, rng)
    _classes(c, rng)
    return ["--distance", str(d), "--classes", str(c), "--output",
            str(root / "fig" / "Figure5.pdf")]


def test_the_figure_draws_from_the_tables(tmp_path):
    argv = _tables(tmp_path)
    assert f5.main(argv) == 0
    out = tmp_path / "fig" / "Figure5.pdf"
    assert out.exists() and out.stat().st_size > 20_000


def test_panel_a_leaves_out_the_bins_the_null_matched_few_of(tmp_path):
    _tables(tmp_path)
    dist = pd.read_csv(tmp_path / "dist" / "end_posterior_dist_all.tsv",
                       sep="\t")
    keep = f5.matched_bins(dist, "gg")
    share = dist["gg_matched_n"] / dist["gg_linked_n"]
    # a bin with a correlation but a twentieth matched is not drawn
    assert share[11] == pytest.approx(0.05)
    assert np.isfinite(dist.loc[11, "gg_matched_r"]) and not keep[11]
    assert not keep[:12].any() and keep[12:].all()
    # nor one with no linked pairs at all
    dist.loc[20, ["gg_linked_n", "gg_matched_n"]] = 0
    assert not f5.matched_bins(dist, "gg")[20]


def test_panel_b_takes_the_regions_the_summary_took(tmp_path):
    _tables(tmp_path)
    reg = pd.read_csv(tmp_path / "dist" / "end_posterior_regions.tsv",
                      sep="\t")
    s = pd.read_csv(tmp_path / "dist" / "end_posterior_regions_summary.tsv",
                    sep="\t").iloc[0]
    t = f5.region_table(reg, s)
    # 3 chromosomes x 20 regions, less the thin HCG one of each
    assert len(t) == 57
    assert not (t.index.get_level_values("start") == 0).any()
    # another band's regions stay out
    other = reg.assign(band_lo=1e4)
    assert f5.region_table(pd.concat([reg, other]), s).shape == t.shape


def test_a_class_pair_is_read_in_either_order(tmp_path):
    _tables(tmp_path)
    summ = pd.read_csv(tmp_path / "classes" / "summary.tsv", sep="\t")
    t = f5._pairs(summ, "GCH", f5.E_PAIRS)
    assert len(t) == len(f5.E_PAIRS) and t["delta"].notna().all()
    v, spans = f5.chromhmm_matrix(summ, "HCG")
    n = len(f5.STATES)
    assert np.isfinite(v[np.tril_indices(n)]).all()
    assert np.isnan(v[np.triu_indices(n, 1)]).all()
    # a sparse pair is left blank
    summ.loc[(summ["system"] == "chromhmm") & (summ["anchor"] == "quiescent")
             & (summ["partner"] == "promoter"), "sparse"] = True
    v2, _ = f5.chromhmm_matrix(summ, "GCH")
    assert np.isnan(v2[n - 1, 0])


def test_one_far_pair_does_not_set_the_colour_scale():
    v = np.array([[-1.55, np.nan], [0.49, 0.3]])
    assert f5._limit(v) == pytest.approx(0.49 * 1.1)
    assert f5._limit(np.array([[0.69, 0.4]])) == pytest.approx(0.69)


def _drift(rng, n, phi=0.9):
    """A series that drifts along the genome, as compartments and
    concordance do: AR(1)."""
    from scipy.signal import lfilter
    return lfilter([1], [1, -phi], rng.normal(size=n + 200))[200:]


def _bins(rng, n=600):
    """``n`` bins in genome order, each track drifting along it, and ΔC
    drifting too, independent of them all: its top 6% are the hot bins,
    in runs as the real ones are."""
    dc = _drift(rng, n) + 0.5 * rng.normal(size=n)
    t = pd.DataFrame({"delta": dc, "hot": dc > np.quantile(dc, 0.94),
                      "control": rng.normal(0, 1, n), "se_control": 1.0})
    for name, _ in f5.G_TRACKS:
        t[name] = np.exp(_drift(rng, n))
    return t


def test_hot_bins_on_a_track_are_starred_and_the_rest_are_not():
    rng = np.random.default_rng(3)
    # bins enough that one track alone can reach q < 0.05 among 17: the
    # smallest p is two over the shifts
    t = _bins(rng, n=1200)
    b = np.log(t["compartment: B"].to_numpy())
    t["hot"] = b + 0.5 * rng.normal(size=b.size) > np.quantile(b, 0.9)
    e = f5.hot_enrichment(t, f5.tested_tracks(t)).set_index("track")
    assert e.loc["compartment: B", "fold"] > 1.5
    assert e.loc["compartment: B", "q"] < 0.05
    assert e.loc["compartment: B", "p"] == e["p"].min()
    # neighbouring shifts are near copies, so the farthest tail is thinner
    # than it should be: now and then one of 18 unrelated tracks is called
    assert (e.drop(index="compartment: B")["q"] < 0.05).sum() <= 1
    # the control's most extreme bins, as many, sit on it as any bins do
    assert 0.6 < e.loc["compartment: B", "control_fold"] < 1.6


def test_neighbouring_bins_alike_are_not_a_finding():
    """Hot bins in runs and tracks in stretches, independent of each other:
    a test that takes the bins as independent (Mann-Whitney, hot against
    the rest) calls over a third of them; genome rotations about one in
    twenty."""
    from scipy.stats import mannwhitneyu
    naive, rot = [], []
    for seed in range(40):
        t = _bins(np.random.default_rng(seed))
        h = t["hot"].to_numpy()
        for name in f5.tested_tracks(t):
            x = t[name].to_numpy()
            naive.append(mannwhitneyu(x[h], x[~h]).pvalue)
        rot += list(f5.hot_enrichment(t, f5.tested_tracks(t))["p"])
    assert np.mean(np.array(naive) < 0.05) > 0.3
    assert np.mean(np.array(rot) < 0.05) < 0.09


def test_every_track_the_run_tested_is_drawn(tmp_path):
    _tables(tmp_path)
    hot = pd.read_csv(tmp_path / "classes" / "locus_hotspots.tsv.gz",
                      sep="\t")
    tracks = f5.tested_tracks(hot)
    assert not set(tracks) & f5.HOT_FIELDS
    rows, breaks = f5.g_rows(tracks)
    # the known ones in their groups, a BED after them under its own name
    assert [t for t, _ in rows] == [t for t, _ in f5.G_TRACKS] + [
        "lamina (BED)"]
    assert rows[-1] == ("lamina (BED)", "lamina (BED)")
    # a break after each group, the BED's own row last
    assert breaks == list(np.cumsum([len(g) for g in f5.G_GROUPS]))
    # a group the run did not test leaves no empty rows
    rows2, breaks2 = f5.g_rows([t for t in tracks if "chromHMM" not in t])
    assert len(rows2) == len(rows) - len(f5.STATES)
    no_states = [len(g) for g in f5.G_GROUPS if g[0][1] not in f5.STATES]
    assert breaks2 == list(np.cumsum(no_states))
    # every q is over all of them
    e = f5.hot_enrichment(hot[hot["context"] == "GCH"], tracks)
    assert len(e) == len(tracks) == len(f5.G_TRACKS) + 1


def test_the_figure_stops_naming_missing_tables(tmp_path):
    argv = _tables(tmp_path)
    (tmp_path / "classes" / "locus_hotspots.tsv.gz").unlink()
    with pytest.raises(SystemExit, match="locus_hotspots"):
        f5.main(argv)


def test_a_and_b_can_come_from_a_run_of_their_own(tmp_path):
    """--distance-ab: a and b from a run whose null keeps every stand-in,
    down to 100 bp; c and d still from --distance; and --supplementary
    draws --distance's own a and b."""
    argv = _tables(tmp_path)
    ab = tmp_path / "dist_ab"
    ab.mkdir()
    _distance(ab, np.random.default_rng(1))
    t = pd.read_csv(ab / "end_posterior_dist_all.tsv", sep="\t")
    # every bin matched, the shortest too
    for k in ("gg", "hh", "gh"):
        t[f"{k}_matched_n"] = t[f"{k}_linked_n"] // 2
        t[f"{k}_matched_r"] = t[f"{k}_matched_r"].fillna(0.3)
        t[f"{k}_null_r"] = t[f"{k}_null_r"].fillna(0.28)
    t.to_csv(ab / "end_posterior_dist_all.tsv", sep="\t", index=False)
    s = pd.read_csv(ab / "end_posterior_regions_summary.tsv", sep="\t")
    s.assign(pearson_r=0.7).to_csv(ab / "end_posterior_regions_summary.tsv",
                                   sep="\t", index=False)
    sup = tmp_path / "fig" / "FigureS5.pdf"
    argv += ["--distance-ab", str(ab), "--supplementary", str(sup)]
    args = f5.parse_args(argv)
    d = f5.read_tables(args)
    assert d["regions_summary"]["pearson_r"] == 0.7
    assert d["regions_summary_supp"]["pearson_r"] == 0.5
    assert f5.matched_bins(d["dist"], "gg").all()
    assert not f5.matched_bins(d["dist_supp"], "gg")[:12].any()
    fig = f5.figure_from_args(args, d)
    titles = [a.get_title(loc="left") for a in fig.axes]
    assert any(tt.startswith("r = +0.70") for tt in titles)
    # a reaches below 1 kb: the one-molecule range shaded and named
    texts = [tx.get_text() for a in fig.axes for tx in a.texts]
    assert "one\nmolecule" in texts
    assert f5.main(argv) == 0
    assert sup.exists() and sup.stat().st_size > 10_000


def test_a_supplementary_figure_needs_a_run_for_a_and_b(tmp_path):
    argv = _tables(tmp_path) + ["--supplementary",
                                str(tmp_path / "fig" / "S.pdf")]
    with pytest.raises(SystemExit, match="--distance-ab"):
        f5.main(argv)


def test_panel_f_draws_its_cells_as_squares_in_the_scale_s_colours():
    """No image a reader could flatten: one square per finite cell, in the
    colour bar's colour, a cell beyond the scale in its end colour."""
    import matplotlib.pyplot as plt
    rows = []
    for ctx, big in (("GCH", 0.4), ("HCG", -0.6)):
        for i, a_ in enumerate(f5.STATES):
            for b_ in f5.STATES[:i + 1]:
                w = big if (a_, b_) == ("enhancer", "promoter") else 0.001 * i
                rows.append({"context": ctx, "system": "chromhmm",
                             "anchor": a_, "partner": b_, "band": f5.ALL_BAND,
                             "signed_rel": w / 100,
                             "signed_rel_lo": w / 100 - 1e-4,
                             "signed_rel_hi": w / 100 + 1e-4,
                             "sparse": a_ == b_ == "promoter"})
    summ = pd.DataFrame(rows)
    f5._style()
    fig, axes = plt.subplots(1, 2)
    f5.panel_f(axes, summ)
    for ax, (_, mark, _) in zip(axes, f5.MARKS):
        assert not ax.images
        v, _ = f5.chromhmm_matrix(summ, mark)
        assert len(ax.patches) == int(np.isfinite(v).sum())
    # HCG's enhancer-promoter lies beyond its scale: the bar's bluest end
    sq = [p for p in axes[1].patches if p.get_xy() == (-0.5, 0.5)][0]
    cmap = plt.get_cmap("RdBu_r")
    assert np.allclose(sq.get_facecolor(), cmap(0.0))
    plt.close(fig)


def test_a_pdf_keeps_a_few_colour_image_at_8_bits(tmp_path):
    import re
    import matplotlib.pyplot as plt
    f5._style()
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(np.ma.masked_invalid([[0.0, 1.0, np.nan], [1.0, 2.0, 0.0]]),
              cmap="RdBu_r")
    fig.savefig(tmp_path / "f.pdf")
    plt.close(fig)
    bits = set(re.findall(rb"/BitsPerComponent (\d+)",
                          (tmp_path / "f.pdf").read_bytes()))
    assert bits == {b"8"}


def test_panel_d_says_whether_the_cell_types_differ(tmp_path):
    """Each mark's Kruskal-Wallis p across the types, from the cells drawn."""
    import matplotlib.pyplot as plt
    from scipy.stats import kruskal
    _tables(tmp_path)
    pc = pd.read_csv(tmp_path / "dist" / "end_posterior_percell.tsv",
                     sep="\t")
    order = list(f5.CELL_TYPES)
    f5._style()
    fig, axes = plt.subplots(2, 1)
    f5.panel_d(axes, pc, order, f5.CELL_TYPES)
    for ax, (_, mark, _) in zip(axes, f5.MARKS):
        d = pc[pc["mark"] == mark]
        want = kruskal(*[d.loc[d["group"] == g, "delta"] for g in order
                         if (d["group"] == g).any()]).pvalue
        txt = [t.get_text() for t in ax.texts if "Kruskal" in t.get_text()]
        assert txt == [f"across types: Kruskal–Wallis {f5._p_short(want)}"]
    plt.close(fig)
    assert np.isnan(f5.types_p([np.array([0.1, 0.2])]))


def test_the_shares_of_cells_print_one_decimal():
    """c's and d's shares of cells with Δ > 0 to one decimal, so a few
    cells below zero do not round up to 100% (1,574 of 1,577 is 99.8%);
    100% only when it is all of them."""
    import matplotlib.pyplot as plt
    rows = []
    for g, n in (("IT-L2/3", 202), ("IT-L4", 180), ("IT-L5/6", 246),
                 ("ASC", 949)):
        rows += [(f"{g}{i}", g, "GCH",
                  -0.01 if i == 0 and g != "ASC" else 0.05,
                  0.01 if i % 2 else -0.01) for i in range(n)]
    rows += [(f"h{i}", "ASC", "HCG", 0.02 if i < 524 else -0.02,
              0.01 if i % 2 else -0.01) for i in range(719)]
    pc = pd.DataFrame(rows, columns=["cell_id", "group", "mark", "delta",
                                     "delta_null"])
    f5._style()
    fig, axes = plt.subplots(2, 2)
    f5.panel_c(axes[:, 0], pc)
    f5.panel_d(axes[:, 1], pc, list(f5.CELL_TYPES), f5.CELL_TYPES)
    texts = [[t.get_text() for t in ax.texts] for ax in axes.ravel()]
    plt.close(fig)
    assert "99.8% of 1,577 cells Δ > 0" in texts[0]
    assert "72.9% of 719 cells Δ > 0" in texts[2]
    assert any(t.endswith("control (grey) 50.0%") for t in texts[0])
    shares = [t for t in texts[1] if t.endswith("%") and "control" not in t]
    assert shares == ["99.5%", "99.4%", "99.6%", "100%"]
    # all of them is an exact 100%; short of all, never 100.0%
    assert f5.cell_share(1.0) == "100%" and f5.cell_share(2499 / 2500) == "99.9%"


def test_the_supplementary_figure_takes_e_and_f_from_their_own_run(tmp_path):
    argv = _tables(tmp_path)
    cs = tmp_path / "classes_kept"
    cs.mkdir()
    _classes(cs, np.random.default_rng(7))
    sup = tmp_path / "fig" / "S.pdf"
    args = f5.parse_args(argv + ["--classes-supp", str(cs),
                                 "--supplementary", str(sup)])
    d = f5.read_tables(args)
    assert "summary_supp" in d and "dist_supp" not in d
    fig = f5.build_supplementary(d)
    # e's two marks and f's two heatmaps, f's cells as squares
    assert sum(bool(a.patches) for a in fig.axes) >= 2
    assert [t.get_text() for t in fig.texts] == ["a", "b"]
    assert f5.main(argv + ["--classes-supp", str(cs), "--supplementary",
                           str(sup)]) == 0 and sup.exists()
    # alone, --classes-supp has nowhere to go
    with pytest.raises(SystemExit, match="--supplementary"):
        f5.main(argv + ["--classes-supp", str(cs)])


def test_a_colour_bar_takes_a_round_tick_within_its_range():
    assert f5._nice_tick(0.046) == pytest.approx(0.04)
    assert f5._nice_tick(0.52) == pytest.approx(0.5)
    # a hair short of 0.2 still takes it, the scale widened to meet it
    assert f5._nice_tick(0.198) == pytest.approx(0.2)
    assert f5._nice_tick(0.163) == pytest.approx(0.1)


# ------------------------------------------------------------- panel h --


def _variability(c: Path, stat=f5.H_STAT, variable=(5, 15, 25)):
    """locus_variability.tsv.gz beside the hot bins: GCH's bins tested
    across the types, ``variable`` (bin numbers on each chromosome) the
    run's variable set, ΔC alone for HCG."""
    rng = np.random.default_rng(3)
    rows = []
    for ctx, st in (("GCH", stat), ("GCH", "delta"), ("HCG", "delta")):
        for chrom in ("chr1", "chr2", "chr10"):
            for b in range(50):
                v = ctx == "GCH" and st == stat and b in variable
                row = {"context": ctx, "system": "ab", "chrom": chrom,
                       "start": b * 1_000_000, "end": (b + 1) * 1_000_000,
                       "statistic": st, "set": "variable" if v else "",
                       "pooled": rng.normal(0.01, 0.002),
                       "between_type_sd": 0.004 if v else 0.0005,
                       "het_q": 0.001 if v else 0.6}
                for i, t in enumerate(f5.CELL_TYPES):
                    row[f"type:{t}"] = (0.01 * (-1) ** i if v
                                        else rng.normal(0.01, 0.001))
                rows.append(row)
    pd.DataFrame(rows).to_csv(c / "locus_variability.tsv.gz", sep="\t",
                              index=False)


def test_h_draws_the_variable_bins_and_writes_their_table(tmp_path):
    """With the statistic in the class run: h below g, a to g their size,
    one row per variable bin, its genes named and in the table."""
    argv = _tables(tmp_path)
    _variability(tmp_path / "classes")
    pd.DataFrame({0: ["chr1", "chr2"], 1: [5_100_000, 15_200_000],
                  2: [5_200_000, 15_300_000], 3: ["Olfr1", "Zfp2"]}).to_csv(
        tmp_path / "genes.bed", sep="\t", header=False, index=False)
    tab = tmp_path / "fig" / "h.tsv"
    argv += ["--genes", str(tmp_path / "genes.bed"), "--h-table", str(tab)]
    args = f5.parse_args(argv)
    fig = f5.figure_from_args(args)
    assert fig.get_figheight() == pytest.approx(10.9 + f5.H_EXTRA)
    assert [t.get_text() for t in fig.texts] == list("abcdefgh")
    heat = [a for a in fig.axes
            if a.get_title(loc="left").startswith("long-range GCH–GCH:")]
    # each bin's squares, and its strip
    assert len(heat) == 1 and len(heat[0].patches) == 9 * (
        len(f5.CELL_TYPES) + 1)
    texts = [t.get_text() for t in heat[0].texts]
    assert "Olfr1" in texts and "Zfp2" in texts
    # the places are in the table, not on the heatmap
    assert not [x.get_text() for x in heat[0].get_yticklabels()]
    assert f5.main(argv) == 0
    t = pd.read_csv(tab, sep="\t")
    assert len(t) == 9 and set(t["chrom"]) == {"chr1", "chr2", "chr10"}
    assert list(t["rank"]) == list(range(1, 10))
    assert t.loc[(t["chrom"] == "chr1") & (t["start"] == 5_000_000),
                 "genes"].item() == "Olfr1"
    assert t["IT-L2/3_pp"].abs().max() == pytest.approx(1.0)


def test_without_its_statistic_the_figure_is_as_before(tmp_path):
    argv = _tables(tmp_path)
    _variability(tmp_path / "classes", stat="signed_sgn")
    fig = f5.figure_from_args(f5.parse_args(argv))
    assert fig.get_figheight() == pytest.approx(10.9)
    assert [t.get_text() for t in fig.texts] == list("abcdefg")


def test_a_bins_genes_in_a_few_words():
    assert f5.gene_label(["Olfr1", "Abc1", "Olfr2", "Gm123", "Olfr3",
                          "1700001A01Rik", "BC147527", "H2-K1", "H2-D1",
                          "H2-Q1", "Def1", "Ghi2", "Jkl3"]) \
        == "Olfr ×3, H2 ×3, Abc1, Def1, Ghi2 +1"
    assert f5.gene_label(["Pcdha1", "Pcdhb2", "Pcdhgc3"]) == "Pcdh ×3"
    assert f5.gene_label(["Gm1", "Gm2"]) == f5.gene_label([]) == "—"


def test_what_the_variable_bins_are_is_tested_against_rotations():
    """A track high in clustered variable bins is found, at q < 0.05
    against genome rotations; one that is not is not."""
    rng = np.random.default_rng(4)
    n = 1000
    var = np.zeros(n, bool)
    for s in range(50, n, 120):
        var[s:s + 5] = True
    t = pd.DataFrame({"variable": var,
                      "planted": np.where(var, 3.0, 1.0) + rng.uniform(0, 1, n),
                      "other": rng.uniform(0, 1, n)})
    r = f5.variable_enrichment(t, ["planted", "other"]).set_index("track")
    assert r.loc["planted", "fold"] > 1.5 and r.loc["planted", "q"] < 0.05
    assert r.loc["other", "q"] > 0.05


def test_h_puts_each_bin_with_the_type_it_is_most_co_open_in():
    """Bins in blocks by type, in the types' order -- each with the type
    of its largest value, a type without contacts enough passed over --
    each block by that value, largest first."""
    nan = np.nan
    H = np.array([[0.1, 0.1, -0.9, 0.2],     # type 3
                  [0.3, 0.0, 0.1, nan],      # type 0
                  [nan, 1.0, 0.0, 0.0],      # type 1
                  [0.9, 0.1, 0.0, 0.1],      # type 0, more open
                  [0.0, 0.1, 0.8, nan]])     # type 2
    assert list(f5.most_open(H)) == [3, 0, 1, 0, 2]
    assert list(f5.block_order(H)) == [3, 1, 2, 4, 0]


def _chromhmm(d: Path):
    """A full-stack style segmentation and its state annotation, the two
    numbering their states differently, as the release does."""
    pd.DataFrame({"state": ["40", "41", "89", "86", "15"],
                  "comments": ["Strongly enriched over TSS",
                               "Promoter, flanking TSS state",
                               "Poised promoters/proximal enhancers",
                               "Bivalent promoter state",
                               "Weak enhancer"],
                  "group": ["TSS", "Prom", "Prom", "BivProm", "WkEnh"],
                  "mneumonics": ["mTSS1", "mPromF2", "mPromF7", "mBivProm1",
                                 "mEnhWk1"]}).to_csv(d / "states.tsv",
                                                     sep="\t", index=False)
    pd.DataFrame([("chr1", 0, 200_000, "1_mTSS1"),
                  ("chr1", 500_000, 1_500_000, "2_mPromF2"),
                  ("chr1", 2_000_000, 2_100_000, "3_mPromF7"),
                  ("chr1", 3_000_000, 3_300_000, "4_mBivProm1"),
                  ("chr1", 4_000_000, 5_000_000, "15_mEnhWk1")]).to_csv(
        d / "seg.bed.gz", sep="\t", header=False, index=False)
    return d / "seg.bed.gz", d / "states.tsv"


def test_h_s_chromhmm_rows_from_the_full_stack_states(tmp_path):
    """Matched by mnemonic, not by number: a segment across two bins split
    between them; the promoter class in its types, the rest in the run's
    groups."""
    bed, st = _chromhmm(tmp_path)
    p = f5.chromhmm_types(bed, st).set_index(["chrom", "start"])
    assert list(p.columns) == [c for c, _, _ in f5.H_CHROMHMM]
    tss, flank, poised, biv = (c for c, _, _ in f5.H_CHROMHMM[:4])
    assert p.loc[("chr1", 0), [tss, flank]].tolist() == pytest.approx(
        [0.2, 0.5])
    assert p.loc[("chr1", 1_000_000), flank] == pytest.approx(0.5)
    assert p.loc[("chr1", 2_000_000), poised] == pytest.approx(0.1)
    assert p.loc[("chr1", 3_000_000), biv] == pytest.approx(0.3)
    assert p.loc[("chr1", 4_000_000), "chromHMM: enhancer"] == pytest.approx(
        1.0)
    assert p.loc[("chr1", 3_000_000), "chromHMM: polycomb"] == 0


def test_h_draws_chromhmm_tss_and_cpg_islands_whatever_their_q(tmp_path):
    """With --chromhmm: the promoter class in its four types and the other
    states, then TSS density and CpG islands -- all drawn, whatever their
    q; the axes no taller than their rows."""
    argv = _tables(tmp_path)
    _variability(tmp_path / "classes")
    bed, st = _chromhmm(tmp_path)
    fig = f5.figure_from_args(f5.parse_args(
        argv + ["--chromhmm", str(bed), "--chromhmm-states", str(st)]))
    enr = [a for a in fig.axes
           if a.get_xlabel().startswith("annotation in these")]
    assert len(enr) == 1
    rows = [x.get_text() for x in enr[0].get_yticklabels()]
    assert rows[:len(f5.H_CHROMHMM)] == [lab for _, lab, _ in f5.H_CHROMHMM]
    assert {"TSS density", "CpG islands"} <= set(rows)
    assert "promoter" not in rows
    assert enr[0].get_position().height * fig.get_figheight() \
        == pytest.approx(len(rows) * f5.H_ROW_IN)
    with pytest.raises(SystemExit, match="go together"):
        f5.main(argv + ["--chromhmm", str(bed)])
