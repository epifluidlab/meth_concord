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


def _with_batch(tmp_path, n_cells=40, n_bins=400, batch_bin_effect=0.0):
    """Two matrices plus a sample sheet; optionally add a batch x BIN effect
    shared by both contexts (the class that does NOT cancel under averaging).

    Uses many bins on purpose: with only a few dozen, the realized variance of
    the planted effect drifts far enough from its population value that the
    inflation it causes is not reliably measurable.
    """
    g, h = _pair(n_cells=n_cells, n_bins=n_bins, rho=0.3, seed=5)
    rng = np.random.default_rng(7)
    batch = np.array(["b1", "b2"] * (n_cells // 2))
    if batch_bin_effect:
        shared = rng.normal(0, batch_bin_effect, n_bins)   # same for GCH+HCG
        g.loc[batch == "b2"] += shared
        h.loc[batch == "b2"] += shared
    sp = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": g.index, "exp_batch": batch}).to_csv(
        sp, sep="\t", index=False)
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp), h.to_parquet(hp)
    return str(gp), str(hp), str(sp)


def test_per_cell_shift_leaves_the_correlation_untouched():
    """The 'same cells' intuition, made precise: a per-cell global offset is a
    constant after averaging, and a constant cannot move a correlation."""
    m = _mod()
    g, h = _pair()
    rng = np.random.default_rng(11)
    shift = pd.Series(rng.normal(0, 3, len(g.index)), index=g.index)
    base = m.correlations(*[m.bin_summary(g, h, "mean", 5)[c].to_numpy()
                            for c in ("gch", "hcg")])
    shifted = m.bin_summary(g.add(shift, axis=0), h.add(shift, axis=0),
                            "mean", 5)
    after = m.correlations(shifted["gch"].to_numpy(), shifted["hcg"].to_numpy())
    assert after["pearson_r"] == pytest.approx(base["pearson_r"], abs=1e-9)


def test_shared_batch_by_bin_effect_inflates_r_and_centering_removes_it(tmp_path):
    """The class of batch effect that DOES matter, and the fix for it."""
    m = _mod()
    gp, hp, sp = _with_batch(tmp_path, batch_bin_effect=1.5)
    out = tmp_path / "out"
    common = ["--gch-d", gp, "--hcg-d", hp, "--min-cells", "5",
              "--plot-format", "png", "--samples", sp,
              "--batch-column", "exp_batch"]
    m.main(common + ["--output-dir", str(out)])
    raw = pd.read_csv(out / "D_gch_vs_hcg_stats.tsv", sep="\t").loc[0, "pearson_r"]
    out2 = tmp_path / "out2"
    m.main(common + ["--batch-correction", "center", "--output-dir", str(out2)])
    corrected = pd.read_csv(out2 / "D_gch_vs_hcg_stats.tsv",
                            sep="\t").loc[0, "pearson_r"]
    # Base correlation is 0.3; the planted shared effect lifts it to ~0.55,
    # and centering should return it to roughly the base.
    assert raw > 0.45, "a shared batch x bin effect should inflate r"
    assert corrected < 0.40, "per-bin centering must remove the inflation"
    assert raw - corrected > 0.10


def test_per_batch_check_reports_one_row_per_batch(tmp_path):
    m = _mod()
    gp, hp, sp = _with_batch(tmp_path)
    out = tmp_path / "out"
    m.main(["--gch-d", gp, "--hcg-d", hp, "--min-cells", "5",
            "--plot-format", "png", "--samples", sp,
            "--batch-column", "exp_batch", "--per-batch-check",
            "--output-dir", str(out)])
    pb = pd.read_csv(out / "D_gch_vs_hcg_per_batch.tsv", sep="\t")
    assert set(pb["batch"]) == {"b1", "b2"}
    assert (pb["n_cells"] == 20).all()


def test_batch_options_require_samples_and_column(tmp_path):
    m = _mod()
    gp, hp, _ = _with_batch(tmp_path)
    with pytest.raises(SystemExit, match="--samples and --batch-column"):
        m.main(["--gch-d", gp, "--hcg-d", hp, "--per-batch-check",
                "--output-dir", str(tmp_path / "o")])


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


# ------------------------------------------- per-bin across-cell correlation --

def _within_bin_pair(n_cells=60, n_bins=40, within_rho=0.8, seed=3,
                     bin_rho=0.0):
    """Matrices whose WITHIN-BIN, across-cell variation is correlated by
    ~within_rho, while the per-bin MEANS are related by bin_rho.

    Separating the two is the whole point: with bin_rho = 0 the scatter of bin
    means sees nothing, and only the across-cell view finds the relationship.
    """
    rng = np.random.default_rng(seed)
    cells = [f"c{i}" for i in range(n_cells)]
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)]
    mg = rng.normal(0, 1, n_bins)
    mh = bin_rho * mg + np.sqrt(1 - bin_rho ** 2) * rng.normal(0, 1, n_bins)
    z = rng.normal(0, 1, (n_cells, n_bins))               # shared per cell/bin
    eg = rng.normal(0, 1, (n_cells, n_bins))
    eh = rng.normal(0, 1, (n_cells, n_bins))
    a = np.sqrt(within_rho)
    b = np.sqrt(1 - within_rho)
    g = pd.DataFrame(mg + a * z + b * eg, index=cells, columns=bins)
    h = pd.DataFrame(mh + a * z + b * eh, index=cells, columns=bins)
    return g, h


def test_per_bin_correlation_finds_what_the_mean_scatter_cannot():
    m = _mod()
    g, h = _within_bin_pair(within_rho=0.8, bin_rho=0.0)
    means = m.bin_summary(g, h, "mean", min_cells=5)
    assert abs(m.correlations(means["gch"].to_numpy(),
                              means["hcg"].to_numpy())["pearson_r"]) < 0.35, \
        "the bin means carry no relationship by construction"
    tab, _, _ = per_bin(m, g, h)
    assert tab["pearson_r"].median() > 0.6, "but the cells within bins do"
    assert (tab["pearson_q"] < 0.05).mean() > 0.9


def per_bin(m, g, h, **kw):
    return m.per_bin_correlations(g, h, kw.pop("min_cells", 5), **kw)


def test_per_bin_correlation_is_centred_on_zero_when_there_is_none():
    m = _mod()
    g, h = _within_bin_pair(within_rho=0.0, bin_rho=0.9)
    tab, _, _ = per_bin(m, g, h)
    assert abs(tab["pearson_r"].median()) < 0.15
    # A shared per-bin mean does NOT create across-cell correlation, which is
    # the separation this view exists to make.
    assert (tab["pearson_q"] < 0.05).mean() < 0.2


def test_spearman_uses_the_same_cells_as_pearson_and_survives_a_monotone_map():
    m = _mod()
    g, h = _within_bin_pair(within_rho=0.8, bin_rho=0.0)
    tab, _, _ = per_bin(m, g, h)
    warped = np.exp(h)                       # monotone, strongly nonlinear
    tab2, _, _ = per_bin(m, g, pd.DataFrame(warped, index=h.index,
                                            columns=h.columns))
    np.testing.assert_allclose(tab["spearman_rho"], tab2["spearman_rho"],
                               atol=1e-9)
    assert (tab2["pearson_r"] < tab2["spearman_rho"]).mean() > 0.7, \
        "Pearson should lose more than Spearman under the warp"
    assert (tab["n_cells"] == 60).all()


def test_missing_values_are_dropped_pairwise_not_globally():
    m = _mod()
    g, h = _within_bin_pair(n_cells=30, n_bins=6, within_rho=0.8)
    g.iloc[0, 0] = np.nan                 # only bin 0 loses this cell
    h.iloc[1, 1] = np.nan
    tab, _, _ = per_bin(m, g, h)
    assert list(tab["n_cells"]) == [29, 29, 30, 30, 30, 30]


def test_the_null_absorbs_a_shared_batch_by_bin_effect():
    """A batch x bin effect shared by the two contexts correlates them across
    cells on its own.  Permuting WITHIN batch keeps it, so the null moves with
    the observed curve instead of pretending the effect is signal."""
    m = _mod()
    g, h = _within_bin_pair(n_cells=60, n_bins=60, within_rho=0.0, seed=9)
    batch = pd.Series(["b1", "b2"] * 30, index=g.index, name="exp_batch")
    shared = np.random.default_rng(1).normal(0, 5.0, g.shape[1])
    g.loc[batch == "b2"] += shared
    h.loc[batch == "b2"] += shared
    obs, nr_batch, _ = per_bin(m, g, h, batch=batch, n_null=2)
    _, nr_global, _ = per_bin(m, g, h, batch=None, n_null=2)
    assert obs["pearson_r"].median() > 0.4, "the shared effect correlates them"
    assert abs(np.nanmedian(nr_batch) - obs["pearson_r"].median()) < 0.25, \
        "a within-batch null must reproduce it"
    assert abs(np.nanmedian(nr_global)) < 0.2, \
        "a global null would call the same effect signal"


# ------------------------------------------------------------- state flow ----

class _SArgs:
    min_cells = 5
    state_rule = "fdr"
    state_fdr = 0.05
    state_threshold = 0.0


def _state_matrix(n_cells=40, per_state=10, seed=0):
    """Bins that are consistently positive, consistently negative, or noise."""
    rng = np.random.default_rng(seed)
    cells = [f"c{i}" for i in range(n_cells)]
    blocks = [np.full((n_cells, per_state), 1.0),
              np.full((n_cells, per_state), -1.0),
              np.zeros((n_cells, per_state))]
    X = np.hstack(blocks) + rng.normal(0, 0.3, (n_cells, 3 * per_state))
    bins = [f"chr1:{i * 250000}" for i in range(3 * per_state)]
    return pd.DataFrame(X, index=cells, columns=bins)


def test_states_are_called_only_where_the_cells_agree():
    m = _mod()
    M = _state_matrix()
    lab, mean, q = m.bin_states(M, ("co-open", "co-closed"), _SArgs())
    assert list(lab[:10]) == ["co-open"] * 10
    assert list(lab[10:20]) == ["co-closed"] * 10
    assert (lab[20:] == "mixed").mean() > 0.8, lab[20:]


def test_threshold_rule_is_available_and_differs_from_fdr():
    m = _mod()
    M = _state_matrix()

    class T(_SArgs):
        state_rule = "threshold"
        state_threshold = 0.5

    lab, _, _ = m.bin_states(M, ("co-open", "co-closed"), T())
    assert list(lab[:10]) == ["co-open"] * 10
    assert (lab[20:] == "mixed").all(), "noise bins sit under the threshold"


def test_flow_table_is_a_full_three_by_three_even_when_a_state_is_empty():
    m = _mod()
    g = np.array(["co-open"] * 5 + ["co-closed"] * 5)
    h = np.array(["co-unmethylated"] * 5 + ["co-methylated"] * 5)
    tab = m.flow_table(g, h)
    assert list(tab.index) == list(m._GCH_STATES)
    assert list(tab.columns) == list(m._HCG_STATES)
    assert tab.loc["co-open", "co-unmethylated"] == 5
    assert tab.loc["mixed"].sum() == 0
    assert int(tab.to_numpy().sum()) == 10


def test_flow_table_drops_unusable_bins():
    m = _mod()
    g = np.array(["co-open", "unusable", "co-closed"])
    h = np.array(["co-unmethylated", "co-methylated", "unusable"])
    assert int(m.flow_table(g, h).to_numpy().sum()) == 1


def test_end_to_end_emits_the_density_and_the_flow(tmp_path):
    m = _mod()
    g, h = _within_bin_pair(n_cells=60, n_bins=60, within_rho=0.7,
                            bin_rho=0.5, seed=2)
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp)
    h.to_parquet(hp)
    out = tmp_path / "out"
    assert m.main(["--gch-d", str(gp), "--hcg-d", str(hp),
                   "--gch-c", str(gp), "--hcg-c", str(hp),
                   "--min-cells", "5", "--plot-format", "png",
                   "--output-dir", str(out)]) == 0
    assert (out / "D_per_bin_across_cell_density.png").exists()
    assert (out / "C_per_bin_across_cell_density.png").exists()
    tab = pd.read_csv(out / "D_per_bin_across_cell_corr.tsv", sep="\t")
    assert len(tab) == 60
    assert {"pearson_r", "pearson_q", "spearman_rho", "spearman_q"} \
        <= set(tab.columns)
    assert tab["pearson_r"].median() > 0.5
    # The flow is directional-only unless asked for on C.
    assert (out / "D_state_flow.png").exists()
    assert not (out / "C_state_flow.png").exists()
    counts = pd.read_csv(out / "D_state_flow_counts.tsv", sep="\t", index_col=0)
    assert list(counts.index) == list(m._GCH_STATES)
    assert int(counts.to_numpy().sum()) == 60
    per_bin_states = pd.read_csv(out / "D_state_flow_per_bin.tsv", sep="\t")
    assert len(per_bin_states) == 60
    assert set(per_bin_states["gch_state"]) <= set(m._GCH_STATES) | {"unusable"}


def test_the_new_views_can_be_switched_off(tmp_path):
    m = _mod()
    g, h = _within_bin_pair(n_cells=30, n_bins=20)
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp)
    h.to_parquet(hp)
    out = tmp_path / "out"
    assert m.main(["--gch-d", str(gp), "--hcg-d", str(hp),
                   "--min-cells", "5", "--plot-format", "png",
                   "--per-bin-correlation", "off", "--state-flow", "off",
                   "--output-dir", str(out)]) == 0
    assert (out / "D_gch_vs_hcg_scatter.png").exists()
    assert not (out / "D_per_bin_across_cell_density.png").exists()
    assert not (out / "D_state_flow.png").exists()


def test_flow_can_be_requested_for_c_too(tmp_path):
    m = _mod()
    g, h = _within_bin_pair(n_cells=30, n_bins=20)
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp)
    h.to_parquet(hp)
    out = tmp_path / "out"
    assert m.main(["--gch-c", str(gp), "--hcg-c", str(hp),
                   "--min-cells", "5", "--plot-format", "png",
                   "--state-flow-c", "on", "--output-dir", str(out)]) == 0
    assert (out / "C_state_flow.png").exists()
