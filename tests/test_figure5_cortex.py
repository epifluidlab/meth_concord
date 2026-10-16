"""figure5_cortex.py: the cortex concordance figure from the runs' tables.

Small synthetic copies of each table, as the scripts write them; the figure
must draw from them, take panel b's regions as the run's summary did, read
a class pair in either order, test a track against genome rotations, and
stop, naming them, without its tables."""

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
                       "tested": True, "hot": v > 0.022, "cold": False}
                # the bins' tracks, as the run writes them beside each bin
                row.update({t: rng.uniform() for t, _ in f5.G_TRACKS})
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


def _bins(rng, n=400):
    """``n`` bins in genome order, each track drifting along it, ΔC
    independent of them all."""
    t = pd.DataFrame({"delta": rng.normal(0, 0.004, n),
                      "control": rng.normal(0, 0.004, n)})
    for name, _ in f5.G_TRACKS:
        t[name] = _drift(rng, n)
    return t


def test_a_track_the_bins_follow_is_starred_and_the_rest_are_not():
    rng = np.random.default_rng(3)
    t = _bins(rng)
    b = t["compartment: B"].to_numpy()
    t["delta"] += 0.004 * (b - b.mean()) / b.std()
    e = f5.track_rhos(t).set_index("track")
    assert e.loc["compartment: B", "rho"] > 0.5
    assert e.loc["compartment: B", "q"] < 0.05
    assert (e.drop(index="compartment: B")["q"] > 0.05).all()


def test_neighbouring_bins_alike_are_not_a_finding():
    """ΔC that drifts along the genome, independent of tracks that drift
    too: the plain Spearman p, which counts the bins as independent, calls
    a third of them; against genome rotations, about one in twenty."""
    from scipy.stats import spearmanr
    plain, rot = [], []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        t = _bins(rng, n=600)
        t["delta"] += 0.002 * _drift(rng, 600)
        plain += [spearmanr(t["delta"], t[n])[1] for n, _ in f5.G_TRACKS]
        rot += list(f5.track_rhos(t)["p"])
    assert np.mean(np.array(plain) < 0.05) > 0.3
    assert np.mean(np.array(rot) < 0.05) < 0.09


def test_the_figure_stops_naming_missing_tables(tmp_path):
    argv = _tables(tmp_path)
    (tmp_path / "classes" / "locus_hotspots.tsv.gz").unlink()
    with pytest.raises(SystemExit, match="locus_hotspots"):
        f5.main(argv)
