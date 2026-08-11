"""Per-bin GCH-vs-HCG concordance scatter.

Pins the aggregation contract (cells intersected before averaging, thin bins
dropped), the reported statistics, and an end-to-end run emitting one figure
per statistic pair.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "gch_hcg_concordance_scatter.py")


def _mod():
    spec = importlib.util.spec_from_file_location("ghs", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ghs"] = m
    spec.loader.exec_module(m)
    return m


def _pair(n_cells=40, n_bins=30, rho=0.8, seed=0):
    """GCH/HCG matrices whose PER-BIN means are correlated by ~rho."""
    rng = np.random.default_rng(seed)
    cells = [f"c{i}" for i in range(n_cells)]
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)]
    bin_g = rng.normal(0, 1, n_bins)
    bin_h = rho * bin_g + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n_bins)
    g = pd.DataFrame(bin_g + rng.normal(0, 0.05, (n_cells, n_bins)),
                     index=cells, columns=bins)
    h = pd.DataFrame(bin_h + rng.normal(0, 0.05, (n_cells, n_bins)),
                     index=cells, columns=bins)
    return g, h


# --------------------------------------------------------------- unit tests --

def test_bin_summary_averages_over_the_shared_cells_only():
    m = _mod()
    g, h = _pair()
    h2 = h.iloc[:25]                       # HCG saw fewer cells
    out = m.bin_summary(g, h2, "mean", min_cells=5)
    assert len(out) == 30
    # gch column must be the mean over the 25 SHARED cells, not all 40.
    expected = g.iloc[:25, 0].mean()
    assert out.loc[out["bin"] == g.columns[0], "gch"].iloc[0] == \
        pytest.approx(expected)


def test_bin_summary_drops_bins_with_too_few_cells():
    m = _mod()
    g, h = _pair()
    g.iloc[3:, 0] = np.nan                 # bin 0 keeps only 3 cells
    out = m.bin_summary(g, h, "mean", min_cells=10)
    assert g.columns[0] not in set(out["bin"])
    assert len(out) == 29


def test_bin_summary_rejects_disjoint_inputs():
    m = _mod()
    g, h = _pair()
    with pytest.raises(SystemExit, match="No overlap"):
        m.bin_summary(g, h.rename(columns=lambda c: c + "_x"), "mean", 1)


def test_correlations_recover_a_planted_relationship():
    m = _mod()
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 400)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 400)
    s = m.correlations(x, y)
    assert s["pearson_r"] > 0.98 and s["spearman_rho"] > 0.98
    assert s["slope"] == pytest.approx(2.0, abs=0.05)
    assert s["intercept"] == pytest.approx(1.0, abs=0.05)
    assert s["pearson_p"] < 1e-50 and s["n"] == 400


def test_correlations_detect_a_monotone_but_nonlinear_relationship():
    """Spearman should clearly beat Pearson on a monotone curve — the reason
    both are reported."""
    m = _mod()
    x = np.linspace(0.01, 1, 300)
    s = m.correlations(x, np.exp(6 * x))
    assert s["spearman_rho"] > 0.999
    assert s["pearson_r"] < s["spearman_rho"] - 0.1


def test_pvalue_formatting_handles_underflow_and_plain_values():
    m = _mod()
    assert m._fmt_p(0.0) == "P < 1e-308"
    assert m._fmt_p(0.5) == "P = 0.5000"
    assert "e-" in m._fmt_p(1e-9)
    assert m._fmt_p(float("nan")) == "P = n/a"


def test_caption_names_both_tests_and_n():
    m = _mod()
    g, h = _pair()
    df = m.bin_summary(g, h, "mean", 5)
    cap = m.caption(m.correlations(df["gch"].to_numpy(), df["hcg"].to_numpy()))
    for token in ("Pearson", "Spearman", "P =", "bins"):
        assert token in cap, token


# --------------------------------------------------------------- end to end --

def test_end_to_end_emits_a_figure_per_pair(tmp_path):
    m = _mod()
    g, h = _pair(rho=0.85)
    paths = {}
    for name, df in (("gd", g), ("hd", h), ("gc", g * 2), ("hc", h * 2)):
        p = tmp_path / f"{name}.parquet"
        df.to_parquet(p)
        paths[name] = str(p)
    out = tmp_path / "out"
    assert m.main(["--gch-d", paths["gd"], "--hcg-d", paths["hd"],
                   "--gch-c", paths["gc"], "--hcg-c", paths["hc"],
                   "--min-cells", "5", "--plot-format", "png",
                   "--output-dir", str(out)]) == 0
    for key in ("D", "C"):
        assert (out / f"{key}_gch_vs_hcg_scatter.png").exists(), key
        assert (out / f"{key}_gch_vs_hcg_per_bin.tsv").exists(), key
        st = pd.read_csv(out / f"{key}_gch_vs_hcg_stats.tsv", sep="\t")
        assert st.loc[0, "pearson_r"] > 0.6
        assert st.loc[0, "n"] == 30


def test_requires_a_complete_pair(tmp_path):
    m = _mod()
    g, _ = _pair()
    p = tmp_path / "g.parquet"
    g.to_parquet(p)
    with pytest.raises(SystemExit, match="complete pair"):
        m.main(["--gch-d", str(p), "--output-dir", str(tmp_path / "o")])


def test_exclude_chrom_filters_bins(tmp_path):
    m = _mod()
    g, h = _pair(n_bins=10)
    g = g.rename(columns={g.columns[0]: "chrX:0"})
    h = h.rename(columns={h.columns[0]: "chrX:0"})
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp), h.to_parquet(hp)
    out = tmp_path / "out"
    m.main(["--gch-d", str(gp), "--hcg-d", str(hp), "--exclude-chrom", "chrX",
            "--min-cells", "5", "--plot-format", "png",
            "--output-dir", str(out)])
    df = pd.read_csv(out / "D_gch_vs_hcg_per_bin.tsv", sep="\t")
    assert len(df) == 9 and "chrX:0" not in set(df["bin"])
