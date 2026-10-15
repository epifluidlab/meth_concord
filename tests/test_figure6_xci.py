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
    pd.DataFrame([(ctx, g, a, b, v, v - 0.01, v + 0.01)
                  for ctx, v in (("HCG", 0.03), ("GCH", 0.0))
                  for g in ("GM", "IMR") for a, b in bands],
                 columns=["context", "group", "dist_lo", "dist_hi",
                          "observed", "observed_lo", "observed_hi"]).to_csv(
        cm / "copy_mixing_excess.tsv", sep="\t", index=False)
    lo = 10 ** np.arange(3.0, 7.0, 0.1)
    pd.DataFrame([(ctx, a, a * 10 ** 0.1, v, v - 0.005, v + 0.005)
                  for ctx, v in (("HCG", 0.015), ("GCH", 0.0)) for a in lo],
                 columns=["context", "dist_lo", "dist_hi", "predicted",
                          "predicted_lo", "predicted_hi"]).to_csv(
        cm / "copy_mixing_excess_fine.tsv", sep="\t", index=False)
    pd.DataFrame([("gg", cmp_, a, b, v, v - 0.03, v + 0.03)
                  for cmp_, v in (("chrX Xi", 0.05), ("chrX Xa", 0.05),
                                  ("autosomes same copy", 0.05),
                                  ("chrX − autosomes, same copy", 0.0))
                  for a, b in bands],
                 columns=["measure", "comparison", "band_lo", "band_hi",
                          "value", "lo", "hi"]).to_csv(
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


def test_the_mosaic_calls_come_from_the_table_or_the_option(tmp_path):
    c, cm, un = _tables(tmp_path, with_calls=False)
    with pytest.raises(SystemExit, match="mosaic-calls"):
        f6.main(_argv(tmp_path, c, cm, un))
    assert f6.main(_argv(tmp_path, c, cm, un,
                         ("--mosaic-calls", "hap1=10,hap2=20"))) == 0
