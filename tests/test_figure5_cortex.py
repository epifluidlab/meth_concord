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
    assert breaks == [7, 13, 15, 19]
    # a group the run did not test leaves no empty rows
    rows2, breaks2 = f5.g_rows([t for t in tracks if "chromHMM" not in t])
    assert len(rows2) == len(rows) - len(f5.STATES)
    assert breaks2 == [7, 9, 13]
    # every q is over all of them
    e = f5.hot_enrichment(hot[hot["context"] == "GCH"], tracks)
    assert len(e) == len(tracks) == len(f5.G_TRACKS) + 1


def test_the_figure_stops_naming_missing_tables(tmp_path):
    argv = _tables(tmp_path)
    (tmp_path / "classes" / "locus_hotspots.tsv.gz").unlink()
    with pytest.raises(SystemExit, match="locus_hotspots"):
        f5.main(argv)
