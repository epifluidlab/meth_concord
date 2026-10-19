"""figure6_xci.py: the X-inactivation figure from the runs' tables.

Small synthetic copies of each table, as the scripts write them; the figure
must draw from them, take the mosaic line's calls from its table or from
--mosaic-calls, and stop when it has neither."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "figure6_xci",
    Path(__file__).resolve().parents[1] / "scripts" / "figure6_xci.py")
f6 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(f6)
#: chrX as haplotype_contacts.py's locus spans it on hg38 (PAR1 aside).
X_LO, X_HI = 2_781_479, 156_028_938


def _tables(root: Path, with_calls=True):
    c, cm, un = root / "contacts", root / "cme", root / "units"
    for d in (c, cm, un):
        d.mkdir(parents=True)
    rows = []
    for sname, eff in (("chrX", 0.3), ("autosomes", 0.0)):
        for lo in np.round(np.arange(4.0, 8.0, 0.1), 1):
            v = eff * np.sign(lo - 6)
            rows.append((sname, lo, lo + 0.1, 100, 100, v, v - 0.1, v + 0.1,
                         "GM"))
    pd.DataFrame(rows, columns=["set", "log10_lo", "log10_hi", "contacts_Xi",
                                "contacts_Xa", "log2_ratio", "ci_lo",
                                "ci_hi", "group"]).to_csv(
        c / "haplotype_contacts_ps_ratio.tsv", sep="\t", index=False)
    pos = np.arange(5e6, 150e6, 1e6)
    xi = -0.3 * np.exp(-((pos / 1e6 - 115.8) / 3) ** 2)
    pd.DataFrame({"pos": pos, "log2_obs_exp_Xi": xi, "log2_obs_exp_Xa": 0.0,
                  "difference": xi, "group": "GM"}).to_csv(
        c / "haplotype_contacts_profile.tsv", sep="\t", index=False)
    # all of X in 10 Mb bins, the last short: the tracks and the maps
    e = np.r_[np.arange(X_LO, X_HI, 10_000_000), X_HI]
    n = e.size - 1
    lt = pd.DataFrame({"start": e[:-1], "end": e[1:]})
    for lab, hcg in (("Xi", 0.25), ("Xa", 0.38)):
        lt[f"partners_compartment_{lab}"] = -0.15 + 0.1 * np.cos(np.arange(n))
        lt[f"HCG_{lab}"] = hcg
        lt[f"GCH_{lab}"] = 0.66
    lt["compartment_bulk"] = np.sin(np.arange(n))
    lt.to_csv(c / "haplotype_locus_GM_chrX.tsv", sep="\t", index=False)
    i, j = np.triu_indices(n)
    pd.DataFrame({"start_1": e[i], "start_2": e[j],
                  "Xi": 0.02 / (1 + j - i), "Xa": 0.02 / (1 + j - i) ** 1.4,
                  "log2_oe_ratio": 0.0}).to_csv(
        c / "haplotype_locus_GM_chrX_map.tsv.gz", sep="\t", index=False)
    st = []
    for g, xi_, scale in (("GM", "1", 1.0), ("IMR", "cells", 1.0),
                          ("IMR", "haplotypes (bulk view)", -0.2)):
        for m, v in (("P(s): share ≥ 1 Mb", 0.09),
                     ("P(s): share ≥ 10 Mb", 0.07),
                     ("A/B compartments: ends' correlation", -0.1),
                     ("crossing DXZ4: log2 obs/exp", -0.27)):
            for sname in ("chrX", "autosomes", "chrX escape genes",
                          "chrX inactivated genes"):
                d = v * scale if sname != "autosomes" else 0.0
                st.append((g, xi_, m, sname, 0.3, 0.3 - d, d, d - 0.02,
                           d + 0.02))
    pd.DataFrame(st, columns=["group", "xi", "measure", "set", "copy1",
                              "copy2", "difference", "ci_lo", "ci_hi"]).to_csv(
        c / "haplotype_contacts.tsv", sep="\t", index=False)
    summ = [("GM", f"{m}{suf}", "chrX rest", v, v - 0.02, v + 0.02)
            for m in ("HCG", "GCH")
            for suf, v in (("", -0.1), ("_island", 0.23), ("_promoter", 0.4),
                           ("_enhancer", 0.3), ("_ctcf", 0.0))]
    pd.DataFrame(summ, columns=["group", "context", "set", "xi_minus_xa",
                                "xi_ci_lo", "xi_ci_hi"]).to_csv(
        root / "allelic_summary.tsv", sep="\t", index=False)
    pd.DataFrame([(ctx, cls, v, v - 0.03, v + 0.03)
                  for ctx in ("HCG_island", "HCG_promoter", "GCH_promoter")
                  for cls, v in (("escape genes", 0.0),
                                 ("inactivated genes", 0.4))],
                 columns=["context", "class", "difference", "ci_lo", "ci_hi"]
                 ).to_csv(c / "haplotype_genes_GM.tsv", sep="\t", index=False)
    rng = np.random.default_rng(0)
    pd.DataFrame({"cell": [f"c{i}" for i in range(80)],
                  "strength": rng.normal(0.25, 0.15, 80),
                  "noise_sd": 0.06}).to_csv(
        c / "haplotype_xci_cells_GM.tsv", sep="\t", index=False)
    pd.DataFrame([(k, 0.1 * k, m, sname, d, d - 0.02, d + 0.02)
                  for k in range(1, 5)
                  for m in ("P(s): share ≥ 1 Mb",
                            "A/B compartments: ends' correlation")
                  for sname, d in (("chrX", 0.1), ("autosomes", 0.0))],
                 columns=["stratum", "strength_mean", "measure", "set",
                          "difference", "ci_lo", "ci_hi"]).to_csv(
        c / "haplotype_xci_strata_GM.tsv", sep="\t", index=False)
    pd.DataFrame([("P(s): share ≥ 1 Mb", "chrX", 0.9),
                  ("A/B compartments: ends' correlation", "chrX", 0.5)],
                 columns=["measure", "set", "p_perm"]).to_csv(
        c / "haplotype_xci_trend_GM.tsv", sep="\t", index=False)
    if with_calls:
        pd.DataFrame({"cell": range(30), "xi_hap": [1] * 10 + [2] * 20}
                     ).to_csv(c / "haplotype_xi_calls_IMR.tsv", sep="\t",
                              index=False)
    bands = [(1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 1e7)]
    pd.DataFrame([(ctx, g, a, b, v, v - 0.01, v + 0.01, e, e - 0.004,
                   e + 0.004)
                  for ctx, v, e in (("HCG", 0.03, 0.015), ("GCH", 0.0, 0.0))
                  for g in ("GM", "IMR") for a, b in bands],
                 columns=["context", "group", "dist_lo", "dist_hi",
                          "observed", "observed_lo", "observed_hi",
                          "predicted", "predicted_lo", "predicted_hi"]
                 ).to_csv(cm / "copy_mixing_excess.tsv", sep="\t",
                          index=False)
    lo = 10 ** np.arange(3.0, 7.0, 0.1)
    pd.DataFrame([(ctx, a, a * 10 ** 0.1, v, v - 0.005, v + 0.005)
                  for ctx, v in (("HCG", 0.015), ("GCH", 0.0)) for a in lo],
                 columns=["context", "dist_lo", "dist_hi", "predicted",
                          "predicted_lo", "predicted_hi"]).to_csv(
        cm / "copy_mixing_excess_fine.tsv", sep="\t", index=False)
    pd.DataFrame([("gg", cmp_, a, b, v, v - 0.03, v + 0.03, 0.5)
                  for cmp_, v in (("chrX Xi", 0.05), ("chrX Xa", 0.05),
                                  ("autosomes same copy", 0.05),
                                  ("chrX − autosomes, same copy", 0.0),
                                  ("chrX Xi − chrX Xa", 0.0))
                  for a, b in bands],
                 columns=["measure", "comparison", "band_lo", "band_hi",
                          "value", "lo", "hi", "p"]).to_csv(
        un / "end_posterior_units_GM.tsv", sep="\t", index=False)
    return c, cm, un


def _argv(root, c, cm, un, extra=()):
    return ["--contacts", str(c), "--allelic-summary",
            str(root / "allelic_summary.tsv"), "--copy-mixing", str(cm),
            "--units", str(un), "--clonal", "GM", "--mosaic", "IMR",
            "--output", str(root / "fig" / "Figure6.pdf"), *extra]


def test_the_figure_draws_from_the_tables(tmp_path):
    c, cm, un = _tables(tmp_path)
    assert f6.main(_argv(tmp_path, c, cm, un)) == 0
    out = tmp_path / "fig" / "Figure6.pdf"
    assert out.exists() and out.stat().st_size > 20_000


def test_the_overview_puts_each_position_at_one_x(tmp_path):
    # b's maps and tracks: one width, so DXZ4's dotted line runs straight
    # down them all
    import matplotlib.pyplot as plt
    c, cm, un = _tables(tmp_path)
    fig = f6.figure_from_args(f6.parse_args(_argv(tmp_path, c, cm, un)))
    fig.canvas.draw()
    along = [ax for ax in fig.axes
             if np.allclose(ax.get_xlim(), (X_LO / 1e6, X_HI / 1e6))]
    assert len(along) == 6                       # two maps, four tracks
    dxz4 = sum(f6.XI_ELEMENTS["hg38"]["DXZ4"]) / 2e6
    at = [ax.transData.transform((dxz4, 0))[0] for ax in along]
    assert np.ptp(at) < 0.5                      # pixels
    plt.close(fig)


def test_the_partners_lines_sit_about_their_own_mean(tmp_path):
    # each copy's partners average over X's B-leaning track: drawn about
    # their mean, so 0 is X's average and the bars' 0 the A/B boundary
    import matplotlib.pyplot as plt
    c, cm, un = _tables(tmp_path)
    fig = f6.figure_from_args(f6.parse_args(_argv(tmp_path, c, cm, un)))
    lab = dict(f6.B_TRACKS)["partners_compartment"]
    ax = next(a for a in fig.axes if a.get_ylabel() == lab)
    lines = [ln.get_ydata() for ln in ax.get_lines()
             if len(ln.get_ydata()) > 2]
    assert len(lines) == 2
    for y in lines:
        assert abs(np.nanmean(y)) < 1e-9 and np.nanstd(y) > 0.05
    plt.close(fig)


def test_the_overview_needs_its_locus_tables(tmp_path):
    c, cm, un = _tables(tmp_path)
    (c / "haplotype_locus_GM_chrX_map.tsv.gz").unlink()
    with pytest.raises(SystemExit, match="--locus"):
        f6.main(_argv(tmp_path, c, cm, un))


def test_the_mosaic_calls_come_from_the_table_or_the_option(tmp_path):
    c, cm, un = _tables(tmp_path, with_calls=False)
    with pytest.raises(SystemExit, match="mosaic-calls"):
        f6.main(_argv(tmp_path, c, cm, un))
    assert f6.main(_argv(tmp_path, c, cm, un,
                         ("--mosaic-calls", "hap1=10,hap2=20"))) == 0


def test_observed_against_expected_is_tested_on_both_intervals():
    cme = pd.DataFrame({"observed": [0.02, 0.05, 0.02],
                        "observed_lo": [0.01, 0.045, 0.0],
                        "observed_hi": [0.03, 0.055, 0.04],
                        "predicted": [0.02, 0.02, 0.01],
                        "predicted_lo": [0.015, 0.015, 0.005],
                        "predicted_hi": [0.025, 0.025, 0.015]})
    p = f6.vs_expected(cme)
    assert p[0] == pytest.approx(1.0)
    assert p[1] < 1e-6
    # the expected interval widens the test: with it alone narrower, the
    # same gap is less sure
    wide = cme.assign(predicted_lo=cme["predicted"] - 0.02,
                      predicted_hi=cme["predicted"] + 0.02)
    assert f6.vs_expected(wide)[2] > p[2]


def test_e_says_its_counts_and_g_names_its_line(tmp_path):
    c, cm, un = _tables(tmp_path)
    fig = f6.figure_from_args(f6.parse_args(_argv(tmp_path, c, cm, un)))
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    # e: the mosaic line's calls in words, no bar chart of two numbers
    assert "IMR: hap1 silenced in 10 cells, hap2 in 20" in texts
    assert not any([x.get_text() for x in ax.get_xticklabels()]
                   == ["hap1", "hap2"] for ax in fig.axes)
    # g: the clonal line, its cells with a strength of all of them
    cells = pd.read_csv(c / "haplotype_xci_cells_GM.tsv", sep="\t")
    n = int(np.isfinite(cells["strength"]).sum())
    assert f"GM, {n} of {len(cells)} cells" in titles

