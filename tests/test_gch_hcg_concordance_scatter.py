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
    state_reference = "zero"
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
    # The 3x3 table is the default view; directional-only unless asked on C.
    assert (out / "D_state_table.png").exists()
    assert (out / "D_state_table_enrichment.tsv").exists()
    assert not (out / "D_state_flow.png").exists(), "alluvial is opt-in now"
    assert not (out / "C_state_table.png").exists()
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
    assert not (out / "D_state_table.png").exists()


def test_flow_can_be_requested_for_c_too(tmp_path):
    m = _mod()
    g, h = _within_bin_pair(n_cells=30, n_bins=20)
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp)
    h.to_parquet(hp)
    out = tmp_path / "out"
    assert m.main(["--gch-c", str(gp), "--hcg-c", str(hp),
                   "--min-cells", "5", "--plot-format", "png",
                   "--state-flow-c", "on", "--state-flow-style", "all",
                   "--output-dir", str(out)]) == 0
    assert (out / "C_state_table.png").exists()
    assert (out / "C_state_flow.png").exists()


def test_empty_result_names_the_likely_cause_instead_of_crashing(tmp_path):
    """The failure it replaces was a numpy 'zero-size array to reduction
    operation minimum' from inside the plotting code, which says nothing about
    the batch correction that produced the empty frame."""
    m = _mod()
    g, h = _pair(n_cells=20, n_bins=10)
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp)
    h.to_parquet(hp)
    with pytest.raises(SystemExit, match="no bin survived"):
        m.main(["--gch-d", str(gp), "--hcg-d", str(hp),
                "--min-cells", "999", "--plot-format", "png",
                "--output-dir", str(tmp_path / "o")])


def test_blacklist_bed_drops_bins_from_both_contexts(tmp_path):
    m = _mod()
    g, h = _pair(n_cells=30, n_bins=20)
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp)
    h.to_parquet(hp)
    bed = tmp_path / "bl.bed"
    bed.write_text("chr1\t0\t1000000\n")          # covers bins 0-3
    out = tmp_path / "out"
    assert m.main(["--gch-d", str(gp), "--hcg-d", str(hp), "--min-cells", "5",
                   "--blacklist-bed", str(bed), "--plot-format", "png",
                   "--per-bin-correlation", "off", "--state-flow", "off",
                   "--output-dir", str(out)]) == 0
    per_bin = pd.read_csv(out / "D_gch_vs_hcg_per_bin.tsv", sep="\t")
    assert len(per_bin) == 16
    assert not any(str(b).endswith((":0", ":250000", ":500000", ":750000"))
                   for b in per_bin["bin"])


def test_blacklist_applies_before_the_batch_correction(tmp_path):
    """Order matters: a blacklisted bin left in place would still shape the
    correction, and ComBat pools across bins to build its priors."""
    m = _mod()
    calls = []
    from nome_hic_concordance import batch_correction as bc
    orig = bc.correct_and_report
    bc.correct_and_report = lambda df, *a, **k: (calls.append(df.shape[1])
                                                 or orig(df, *a, **k))
    try:
        g, h = _pair(n_cells=30, n_bins=20)
        gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
        g.to_parquet(gp)
        h.to_parquet(hp)
        sp = tmp_path / "s.tsv"
        pd.DataFrame({"cell_name": g.index,
                      "exp_batch": ["b1", "b2"] * 15}).to_csv(
            sp, sep="\t", index=False)
        bed = tmp_path / "bl.bed"
        bed.write_text("chr1\t0\t1000000\n")
        assert m.main(["--gch-d", str(gp), "--hcg-d", str(hp),
                       "--min-cells", "5", "--blacklist-bed", str(bed),
                       "--samples", str(sp), "--batch-column", "exp_batch",
                       "--batch-correction", "center", "--plot-format", "png",
                       "--per-bin-correlation", "off", "--state-flow", "off",
                       "--output-dir", str(tmp_path / "o")]) == 0
    finally:
        bc.correct_and_report = orig
    assert calls == [16, 16], calls


class _SArgsRef(_SArgs):
    state_reference = "zero"


class _SArgsMed(_SArgs):
    state_reference = "median"


def test_state_reference_zero_is_one_sided_for_a_mark_off_0_5():
    """sign(2F−1) = sign(mean level − 0.5): a mark whose genome-wide mean is
    not near 0.5 puts every bin on one side, and 'co-closed everywhere' is the
    base rate restated rather than a regional finding."""
    m = _mod()
    rng = np.random.default_rng(0)
    nb = 60
    level = rng.normal(-1.0, 0.4, nb)
    X = pd.DataFrame(level + rng.normal(0, 0.15, (60, nb)),
                     index=[f"c{i}" for i in range(60)],
                     columns=[f"chr1:{i * 250000}" for i in range(nb)])
    zero, _, _ = m.bin_states(X, ("co-open", "co-closed"), _SArgsRef())
    assert (zero == "co-closed").mean() > 0.95
    med, _, _ = m.bin_states(X, ("co-open", "co-closed"), _SArgsMed())
    assert 0.2 < (med == "co-open").mean() < 0.8, med


def test_flow_has_something_to_flow_once_the_reference_is_relative(tmp_path):
    m = _mod()
    rng = np.random.default_rng(1)
    n, nb = 80, 120
    cells = [f"c{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(nb)]
    # Both marks sit far off 0.5 -- GCH mostly closed, HCG mostly methylated --
    # exactly the real situation, with a genuine anti-correlation between them.
    lvl = rng.normal(0, 1.0, nb)
    g = pd.DataFrame(-1.5 + lvl + rng.normal(0, .2, (n, nb)),
                     index=cells, columns=bins)
    h = pd.DataFrame(1.5 - lvl + rng.normal(0, .2, (n, nb)),
                     index=cells, columns=bins)
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp)
    h.to_parquet(hp)

    def counts(extra, sub):
        assert m.main(["--gch-d", str(gp), "--hcg-d", str(hp),
                       "--min-cells", "5", "--plot-format", "png",
                       "--per-bin-correlation", "off",
                       "--output-dir", str(tmp_path / sub)] + extra) == 0
        return pd.read_csv(tmp_path / sub / "D_state_flow_counts.tsv",
                           sep="\t", index_col=0)

    def concentration(t):
        a = t.to_numpy()
        return a.max() / a.sum()

    zero = counts([], "zero")
    assert concentration(zero) > 0.9, "essentially one ribbon, no information"
    med = counts(["--state-reference", "median"], "med")
    assert concentration(med) < 0.75, "a real table to read"
    # Anti-correlated by construction: co-open goes to co-unmethylated.
    assert med.loc["co-open", "co-unmethylated"] > med.loc["co-open",
                                                           "co-methylated"]


class _SArgsExp(_SArgs):
    state_reference = "expected"


def test_state_expected_reference_separates_excess_from_level():
    """Both bins co-closed in absolute terms; one is co-closed by more than
    its own level predicts and the other by less.  Zero calls them the same."""
    m = _mod()
    rng = np.random.default_rng(0)
    p = np.array([0.2, 0.35])
    C = np.array([1.0, 1.0])
    ref = m.expected_directional(p, C)
    n = 60
    X = pd.DataFrame(
        np.vstack([np.full(n, ref[0] + 0.25), np.full(n, ref[1] - 0.25)]).T
        + rng.normal(0, 0.02, (n, 2)),
        index=[f"c{i}" for i in range(n)], columns=["chr1:0", "chr1:250000"])
    assert (X.to_numpy() < 0).all()
    zero, _, _ = m.bin_states(X, ("co-open", "co-closed"), _SArgs())
    assert list(zero) == ["co-closed", "co-closed"]
    exp, _, _ = m.bin_states(X, ("co-open", "co-closed"), _SArgsExp(),
                             np.tile(ref, (n, 1)))
    assert list(exp) == ["co-open", "co-closed"]


def test_state_expected_needs_the_components(tmp_path):
    m = _mod()
    X = pd.DataFrame(np.zeros((30, 3)), index=[f"c{i}" for i in range(30)],
                     columns=[f"chr1:{i}" for i in range(3)])
    with pytest.raises(SystemExit, match="marginal and coordination"):
        m.bin_states(X, ("co-open", "co-closed"), _SArgsExp())


def test_state_expected_end_to_end_uses_the_sibling_parquets(tmp_path):
    m = _mod()
    rng = np.random.default_rng(2)
    n, nb = 60, 60
    cells = [f"c{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(nb)]
    for ctx, lo, hi in (("GCH", 0.1, 0.4), ("HCG", 0.6, 0.9)):
        d = tmp_path / ctx
        d.mkdir()
        p = rng.uniform(lo, hi, nb)
        C = np.abs(rng.normal(1.0, 0.2, nb))
        ref = m.expected_directional(p, C)
        D = ref + rng.normal(0, 0.06, nb) + rng.normal(0, 0.01, (n, nb))
        pd.DataFrame(D, index=cells, columns=bins).to_parquet(
            d / "concordance_directional.parquet")
        pd.DataFrame(np.tile(p, (n, 1)), index=cells, columns=bins).to_parquet(
            d / "concordance_marginal.parquet")
        pd.DataFrame(np.tile(C, (n, 1)), index=cells, columns=bins).to_parquet(
            d / "concordance_coordination.parquet")
    args = ["--gch-d", str(tmp_path / "GCH" / "concordance_directional.parquet"),
            "--hcg-d", str(tmp_path / "HCG" / "concordance_directional.parquet"),
            "--min-cells", "5", "--plot-format", "png",
            "--per-bin-correlation", "off"]
    assert m.main(args + ["--output-dir", str(tmp_path / "z")]) == 0
    assert m.main(args + ["--state-reference", "expected",
                          "--output-dir", str(tmp_path / "e")]) == 0
    z = pd.read_csv(tmp_path / "z" / "D_state_flow_counts.tsv", sep="\t",
                    index_col=0).to_numpy()
    e = pd.read_csv(tmp_path / "e" / "D_state_flow_counts.tsv", sep="\t",
                    index_col=0).to_numpy()
    assert z.max() / z.sum() > 0.9, "zero: everything in one cell"
    assert e.max() / e.sum() < 0.75, "expected: a table with content"


# ------------------------------------------------------------ 3x3 state table --

def _tab(counts):
    m = _mod()
    return pd.DataFrame(counts, index=list(m._GCH_STATES),
                        columns=list(m._HCG_STATES))


def test_enrichment_is_against_independence_not_the_raw_count():
    """The point of the colour: with both margins lopsided, the biggest cell is
    biggest under NO relationship, so counts cannot show association."""
    m = _mod()
    # Perfectly independent table: every cell's observed equals its expected.
    row = np.array([600, 100, 300])[:, None]
    col = np.array([500, 100, 400])[None, :]
    indep = (row * col / 1000).astype(int)
    st = m.flow_enrichment(_tab(indep))
    assert np.allclose(st["log2_ratio"], 0.0, atol=1e-9)
    assert np.nanmax(np.abs(st["residual"])) < 1e-6
    # ...yet one cell holds 30% of everything.
    assert indep.max() / indep.sum() > 0.28


def test_enrichment_flags_the_canonical_diagonal():
    m = _mod()
    st = m.flow_enrichment(_tab([[500, 20, 100], [20, 10, 20],
                                 [100, 20, 500]]))
    i_open, i_closed = 0, 2
    j_un, j_me = 0, 2
    assert st["log2_ratio"][i_open, j_un] > 0.3
    assert st["log2_ratio"][i_open, j_me] < -0.3
    assert st["q"][i_open, j_un] < 0.001 and st["q"][i_open, j_me] < 0.001


def test_cells_with_a_tiny_expected_count_are_not_tested():
    """A handful of bins can be '5x enriched' with q<0.001 under the normal
    approximation and would otherwise set the colour scale for the figure."""
    m = _mod()
    st = m.flow_enrichment(_tab([[1000, 2, 400], [2, 4, 2], [400, 2, 1000]]))
    assert not st["reliable"][1, 1], "expected count here is under 5"
    assert np.isnan(st["q"][1, 1]) and np.isnan(st["residual"][1, 1])
    assert st["reliable"][0, 0] and np.isfinite(st["q"][0, 0])
    # The ratio is still reported, just not tested.
    assert np.isfinite(st["log2_ratio"][1, 1])


def test_state_table_is_the_default_view_and_writes_its_numbers(tmp_path):
    m = _mod()
    g, h = _within_bin_pair(n_cells=60, n_bins=120, within_rho=0.5, bin_rho=0.6)
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp)
    h.to_parquet(hp)
    out = tmp_path / "out"
    assert m.main(["--gch-d", str(gp), "--hcg-d", str(hp), "--min-cells", "5",
                   "--plot-format", "png", "--per-bin-correlation", "off",
                   "--state-reference", "median",
                   "--output-dir", str(out)]) == 0
    assert (out / "D_state_table.png").exists()
    assert not (out / "D_state_flow.png").exists()
    enr = pd.read_csv(out / "D_state_table_enrichment.tsv", sep="\t")
    assert len(enr) == 9
    assert {"n", "expected", "pct_total", "pct_of_gch_row", "pct_of_hcg_col",
            "log2_obs_over_exp", "adjusted_residual", "p", "q",
            "testable"} <= set(enr.columns)
    assert enr["n"].sum() == 120
    # Row percentages sum to 100 within each GCH state that has any bins.
    for _, grp in enr.groupby("gch_state"):
        if grp["n"].sum():
            assert grp["pct_of_gch_row"].sum() == pytest.approx(100.0)


def test_both_styles_can_be_emitted(tmp_path):
    m = _mod()
    g, h = _within_bin_pair(n_cells=40, n_bins=60)
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp)
    h.to_parquet(hp)
    out = tmp_path / "out"
    assert m.main(["--gch-d", str(gp), "--hcg-d", str(hp), "--min-cells", "5",
                   "--plot-format", "png", "--per-bin-correlation", "off",
                   "--state-flow-style", "all", "--state-reference", "median",
                   "--output-dir", str(out)]) == 0
    assert (out / "D_state_table.png").exists()
    assert (out / "D_state_flow.png").exists()


class _SArgsQuant(_SArgs):
    state_rule = "quantile"
    state_quantile = 1.0 / 3.0


def test_quantile_rule_is_unmoved_by_a_reference_that_shifts_everything():
    """The failure it answers: with the reference off in one direction, an
    absolute rule calls nearly every bin one state.  A share cannot."""
    m = _mod()
    rng = np.random.default_rng(0)
    nb, n = 300, 60
    strength = np.linspace(-1, 1, nb)
    X = pd.DataFrame(strength + rng.normal(0, 0.02, (n, nb)),
                     index=[f"c{i}" for i in range(n)],
                     columns=[f"chr1:{i * 250000}" for i in range(nb)])
    lab = m.bin_states(X, ("co-open", "co-closed"), _SArgsQuant())[0]
    assert (lab == "co-open").sum() == 100
    assert (lab == "co-closed").sum() == 100
    assert (lab == "mixed").sum() == 100

    shifted = m.bin_states(X + 5.0, ("co-open", "co-closed"),
                           _SArgsQuant())[0]
    np.testing.assert_array_equal(lab, shifted), "a common shift changes nothing"
    # ...whereas the fdr rule collapses entirely under the same shift.
    fdr_shift = m.bin_states(X + 5.0, ("co-open", "co-closed"), _SArgs())[0]
    assert (fdr_shift == "co-open").mean() > 0.95


def test_quantile_rule_gives_the_table_balanced_margins(tmp_path):
    m = _mod()
    rng = np.random.default_rng(2)
    n, nb = 60, 300
    cells = [f"c{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(nb)]
    z = rng.normal(0, 1, nb)
    g = pd.DataFrame(-3.0 + z + rng.normal(0, .05, (n, nb)),
                     index=cells, columns=bins)
    h = pd.DataFrame(2.0 - 0.8 * z + rng.normal(0, .05, (n, nb)),
                     index=cells, columns=bins)
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp)
    h.to_parquet(hp)
    out = tmp_path / "out"
    assert m.main(["--gch-d", str(gp), "--hcg-d", str(hp), "--min-cells", "5",
                   "--plot-format", "png", "--per-bin-correlation", "off",
                   "--state-rule", "quantile",
                   "--output-dir", str(out)]) == 0
    tab = pd.read_csv(out / "D_state_flow_counts.tsv", sep="\t", index_col=0)
    assert tab.sum(axis=1).tolist() == [100, 100, 100], "GCH margins fixed"
    assert tab.sum(axis=0).tolist() == [100, 100, 100], "HCG margins fixed"
    # Anti-coupled by construction, and with the margins fixed that shows up
    # purely as enrichment on the canonical diagonal.
    enr = pd.read_csv(out / "D_state_table_enrichment.tsv", sep="\t")
    canon = enr[((enr["gch_state"] == "co-open")
                 & (enr["hcg_state"] == "co-unmethylated"))
                | ((enr["gch_state"] == "co-closed")
                   & (enr["hcg_state"] == "co-methylated"))]
    assert (canon["log2_obs_over_exp"] > 0.5).all()
    assert (canon["q"] < 0.001).all()


# ---------------------------------------------------------------- slope view --

def test_zscore_is_immune_to_a_shift_or_a_rescale():
    """Why the slope view needs no reference: a z-score is already measured
    against the genome, so the offset that broke every threshold is gone."""
    m = _mod()
    v = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    z = m._zscore(v)
    np.testing.assert_allclose(m._zscore(v + 100.0), z, atol=1e-12)
    np.testing.assert_allclose(m._zscore(v * 7.0), z, atol=1e-12)
    assert abs(z.mean()) < 1e-12 and abs(z.std() - 1) < 1e-12
    assert np.isnan(m._zscore(np.zeros(5))).all(), "no spread, no z"


def test_slope_figure_reports_where_each_state_lands(tmp_path):
    m = _mod()
    rng = np.random.default_rng(0)
    nb = 300
    zg = np.linspace(-2, 2, nb)
    # Anti-coupled: the most co-open bins are the least co-methylated.
    h = -0.8 * zg + rng.normal(0, 0.3, nb)
    lab = np.where(zg > 0.7, "co-open",
                   np.where(zg < -0.7, "co-closed", "mixed"))
    out = tmp_path / "s.png"
    info = m.slope_figure(zg, h, lab, "D", out, _SlopeArgs())
    assert out.exists()
    assert info["n_bins"] == nb
    assert info["pearson_r"] < -0.8
    by = {r["gch_state"]: r for r in info["by_state"]}
    # co-open ends up LOW on the HCG axis and co-closed high — the canonical
    # pairing, read off a continuous scale instead of a 3x3 table.
    assert by["co-open"]["mean_hcg_z"] < by["mixed"]["mean_hcg_z"]
    assert by["mixed"]["mean_hcg_z"] < by["co-closed"]["mean_hcg_z"]


class _SlopeArgs:
    slope_max_lines = 0
    random_seed = 42


def test_slope_line_cap_is_drawing_only(tmp_path):
    """Subsampling may thin the hairball but must not change a number."""
    m = _mod()
    rng = np.random.default_rng(1)
    nb = 500
    zg = rng.normal(0, 1, nb)
    h = 0.5 * zg + rng.normal(0, 0.5, nb)
    lab = np.full(nb, "mixed", dtype=object)

    class Capped(_SlopeArgs):
        slope_max_lines = 50

    full = m.slope_figure(zg, h, lab, "D", tmp_path / "a.png", _SlopeArgs())
    cut = m.slope_figure(zg, h, lab, "D", tmp_path / "b.png", Capped())
    assert full["n_bins"] == cut["n_bins"] == nb
    assert full["pearson_r"] == pytest.approx(cut["pearson_r"])
    assert (full["by_state"][0]["mean_hcg_z"]
            == pytest.approx(cut["by_state"][0]["mean_hcg_z"]))


def test_slope_is_emitted_by_default_with_its_z_table(tmp_path):
    m = _mod()
    g, h = _within_bin_pair(n_cells=40, n_bins=120, within_rho=0.4,
                            bin_rho=0.7)
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp)
    h.to_parquet(hp)
    out = tmp_path / "out"
    assert m.main(["--gch-d", str(gp), "--hcg-d", str(hp), "--min-cells", "5",
                   "--plot-format", "png", "--per-bin-correlation", "off",
                   "--output-dir", str(out)]) == 0
    assert (out / "D_state_slope.png").exists()
    assert (out / "D_state_table.png").exists(), "heatmap is still default too"
    z = pd.read_csv(out / "D_state_slope_z.tsv", sep="\t")
    assert len(z) == 120
    assert {"bin", "gch_state", "hcg_state", "gch_z", "hcg_z",
            "delta_z"} == set(z.columns)
    ok = z["gch_z"].notna()
    assert abs(z.loc[ok, "gch_z"].mean()) < 1e-9
    np.testing.assert_allclose(z.loc[ok, "delta_z"],
                               z.loc[ok, "hcg_z"] - z.loc[ok, "gch_z"],
                               atol=1e-9)


def test_unknown_style_is_rejected(tmp_path):
    m = _mod()
    g, h = _within_bin_pair(n_cells=30, n_bins=40)
    gp, hp = tmp_path / "g.parquet", tmp_path / "h.parquet"
    g.to_parquet(gp)
    h.to_parquet(hp)
    with pytest.raises(SystemExit, match="unknown"):
        m.main(["--gch-d", str(gp), "--hcg-d", str(hp), "--min-cells", "5",
                "--plot-format", "png", "--per-bin-correlation", "off",
                "--state-flow-style", "heatmap,sankey",
                "--output-dir", str(tmp_path / "o")])
