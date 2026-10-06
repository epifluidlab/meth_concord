"""Cell-type metric bar plots across the concordance arms.

Pins the random control's defining property (per-bin permutation preserves the
distribution and the NaN pattern, destroys only the cell association), the
chromosome filter, and an end-to-end run producing one PDF per metric plus a
table whose ordering reflects the planted signal.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "celltype_metric_barplots.py")


def _mod():
    spec = importlib.util.spec_from_file_location("cmb", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["cmb"] = m
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------- unit tests --

def test_shuffle_preserves_distribution_and_nan_pattern():
    m = _mod()
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (30, 5))
    X[::4, 2] = np.nan
    df = pd.DataFrame(X, index=[f"c{i}" for i in range(30)],
                      columns=[f"chr1:{i}" for i in range(5)])
    sh = m._shuffle_within_bins(df, seed=1)
    assert (sh.isna() == df.isna()).all().all(), "NaN pattern must be identical"
    for col in df.columns:
        a = np.sort(df[col].dropna().to_numpy())
        b = np.sort(sh[col].dropna().to_numpy())
        assert np.allclose(a, b), "each bin keeps its exact value multiset"
    # ...but the cell association is gone (at least one column reordered).
    assert not np.allclose(df.to_numpy()[~df.isna().to_numpy()],
                           sh.to_numpy()[~sh.isna().to_numpy()])


def test_shuffle_destroys_cell_type_structure():
    """The control must actually collapse a planted signal."""
    m = _mod()
    from nome_hic_concordance import metrics
    rng = np.random.default_rng(2)
    n = 60
    lab = pd.Series(["A"] * (n // 2) + ["B"] * (n // 2),
                    index=[f"c{i}" for i in range(n)])
    X = rng.normal(0, 0.3, (n, 12))
    X[n // 2:] += 4.0                                   # strong group offset
    df = pd.DataFrame(X, index=lab.index,
                      columns=[f"chr1:{i}" for i in range(12)])
    real = metrics.label_agreement(df.to_numpy(), lab, n_perm=0)
    ctrl = metrics.label_agreement(
        m._shuffle_within_bins(df, seed=3).to_numpy(), lab, n_perm=0)
    assert real["knn_purity"] > 0.95
    assert ctrl["knn_purity"] < 0.75, "control must lose the structure"
    assert ctrl["ari"] < real["ari"]


def test_drop_chroms_filters_by_bin_prefix():
    m = _mod()
    df = pd.DataFrame(np.zeros((2, 4)), index=["a", "b"],
                      columns=["chr1:0", "chrX:0", "chrM:0", "chr2:0"])
    out = m._drop_chroms(df, {"chrM", "chrX"})
    assert list(out.columns) == ["chr1:0", "chr2:0"]
    assert m._drop_chroms(df, set()).shape[1] == 4


def test_arm_style_covers_every_requested_arm():
    m = _mod()
    keys = [k for k, *_ in m.ARM_STYLE]
    assert keys == ["baseline_1d", "baseline_3d", "D", "D_linear",
                    "D_quadratic", "C", "C_linear", "C_quadratic", "C_null",
                    "random"]


# --------------------------------------------------------------- end to end --

def _inputs(tmp_path, n_cells=60, n_bins=30, seed=0):
    """1D carries the cell type; C/D carry it plus independent noise.

    Each cell type has its own PROFILE across bins.  A per-cell offset shared
    by every bin -- what this fixture planted before -- is exactly what the
    default per-cell rank removes, which left C at chance and every "beats
    the random control" assertion passing on a lucky seed.
    """
    rng = np.random.default_rng(seed)
    cells = [f"c{i}" for i in range(n_cells)]
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)] + ["chrX:0"]
    g = np.repeat([0, 1, 2], n_cells // 3)
    profile = rng.normal(0, 1, (3, len(bins)))
    def mat(scale):
        base = profile[g] * scale + rng.normal(0, 0.4, (n_cells, len(bins)))
        return pd.DataFrame(base, index=cells, columns=bins)
    paths = {}
    for name, scale in (("c", 1.2), ("d", 1.0), ("b1", 1.5), ("b3", 0.8),
                        ("cnull", 0.0)):
        p = tmp_path / f"{name}.parquet"
        mat(scale).to_parquet(p)
        paths[name] = str(p)
    sp = tmp_path / "samples.tsv"
    pd.DataFrame({
        "cell_name": cells,
        "group": np.repeat(["A", "B", "C"], n_cells // 3),
        "exp_batch": ["b1", "b2"] * (n_cells // 2),
    }).to_csv(sp, sep="\t", index=False)
    paths["samples"] = str(sp)
    return paths


def _run(tmp_path, extra=None):
    m = _mod()
    p = _inputs(tmp_path)
    out = tmp_path / "out"
    argv = ["--concordance-c", p["c"], "--concordance-d", p["d"],
            "--baseline-1d", p["b1"], "--baseline-3d", p["b3"],
            "--samples", p["samples"], "--context", "GCH",
            "--n-svd", "5", "--n-perm", "0", "--exclude-chrom", "chrX",
            "--plot-format", "png", "--output-dir", str(out)] + (extra or [])
    assert m.main(argv) == 0
    return out


def test_end_to_end_emits_one_pdf_per_metric_and_a_table(tmp_path):
    out = _run(tmp_path)
    for metric in ("knn_purity", "ari", "nmi"):
        assert (out / f"GCH_{metric}_barplot.png").exists(), metric
    tab = pd.read_csv(out / "GCH_metrics.tsv", sep="\t").set_index("arm")
    # every arm the inputs allow: both baselines, D and C with residuals,
    # the random control -- no C_null without --concordance-c-null
    assert list(tab.index) == ["baseline_1d", "baseline_3d", "D", "D_linear",
                               "D_quadratic", "C", "C_linear", "C_quadratic",
                               "random"]
    # The planted structure must beat the shuffled control on every metric.
    for metric in ("knn_purity", "ari", "nmi"):
        assert tab.loc["baseline_1d", metric] > tab.loc["random", metric], metric
        assert tab.loc["C", metric] > tab.loc["random", metric], metric
    # chrX was excluded, so no arm may have used it.
    assert tab["n_bins"].max() <= 30


def test_residual_arms_are_scored_and_differ_from_the_raw_arm(tmp_path):
    out = _run(tmp_path)
    tab = pd.read_csv(out / "GCH_metrics.tsv", sep="\t").set_index("arm")
    for arm in ("C_linear", "C_quadratic", "D_linear", "D_quadratic"):
        assert np.isfinite(tab.loc[arm, "knn_purity"]), arm
    # Regressing out a baseline that carries the signal must cost something.
    assert tab.loc["C_linear", "knn_purity"] < tab.loc["C", "knn_purity"]


def test_extra_metrics_can_be_requested(tmp_path):
    out = _run(tmp_path, ["--metrics", "purity_index,silhouette"])
    assert (out / "GCH_purity_index_barplot.png").exists()
    assert (out / "GCH_silhouette_barplot.png").exists()


def test_missing_batch_column_fails_loudly(tmp_path):
    m = _mod()
    p = _inputs(tmp_path)
    with pytest.raises(SystemExit, match="--batch-column"):
        m.main(["--concordance-c", p["c"], "--concordance-d", p["d"],
                "--baseline-1d", p["b1"], "--baseline-3d", p["b3"],
                "--samples", p["samples"], "--batch-column", "nope",
                "--output-dir", str(tmp_path / "o2")])


# ----------------------------------------- concordance-stage batch correction --

def _batched_inputs(tmp_path, n_cells=60, n_bins=30, offset=3.0, seed=1):
    """As _inputs, but C and D carry a batch effect and the baselines do not,
    so a correction that leaks onto the baselines shows up in their bars.

    Both the cell-type signal and the batch effect are PER-BIN patterns, not
    per-cell offsets.  A per-cell offset would prove nothing here: the default
    --per-cell-norm rank already removes it, so there would be nothing left for
    the correction to do.  Per-bin is also the real shape of the problem, and
    the reason this stage exists — per-bin QN turns a per-bin batch offset into
    a batch-ordered ranking and preserves it.
    """
    rng = np.random.default_rng(seed)
    cells = [f"c{i}" for i in range(n_cells)]
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)]
    g = np.repeat([0, 1, 2], n_cells // 3)
    batch = np.array(["b1", "b2"] * (n_cells // 2))
    bi = (batch == "b2").astype(int)
    ct_pattern = rng.normal(0, 1.0, (3, n_bins))
    batch_pattern = rng.normal(0, offset, (2, n_bins))
    paths = {}
    for name, scale, batchy in (("c", 1.2, True), ("d", 1.0, True),
                                ("b1", 1.5, False), ("b3", 0.8, False)):
        base = ct_pattern[g] * scale + rng.normal(0, 0.4, (n_cells, n_bins))
        if batchy:
            base = base + batch_pattern[bi]
        p = tmp_path / f"{name}.parquet"
        pd.DataFrame(base, index=cells, columns=bins).to_parquet(p)
        paths[name] = str(p)
    sp = tmp_path / "samples.tsv"
    pd.DataFrame({"cell_name": cells,
                  "group": np.repeat(["A", "B", "C"], n_cells // 3),
                  "exp_batch": batch,
                  "lib_batch": np.tile(["L1", "L1", "L2", "L2"],
                                       n_cells // 4)}).to_csv(
        sp, sep="\t", index=False)
    paths["samples"] = str(sp)
    return paths


def _run_batched(tmp_path, sub, extra):
    m = _mod()
    p = _batched_inputs(tmp_path)
    out = tmp_path / sub
    argv = ["--concordance-c", p["c"], "--concordance-d", p["d"],
            "--baseline-1d", p["b1"], "--baseline-3d", p["b3"],
            "--samples", p["samples"], "--context", "GCH",
            "--n-svd", "5", "--n-perm", "0", "--plot-format", "png",
            "--output-dir", str(out)] + extra
    assert m.main(argv) == 0
    return pd.read_csv(out / "GCH_metrics.tsv", sep="\t").set_index("arm")


def test_concordance_stage_correction_lifts_c_and_d_only(tmp_path):
    """It must reach the concordance arms and leave the baselines they are
    measured against exactly where they were."""
    off = _run_batched(tmp_path, "off", ["--batch-column", "exp_batch"])
    on = _run_batched(tmp_path, "on",
                      ["--batch-column", "exp_batch",
                       "--concordance-batch-correction", "center"])
    assert on.loc["C", "knn_purity"] > off.loc["C", "knn_purity"]
    assert on.loc["D", "knn_purity"] > off.loc["D", "knn_purity"]
    assert (on.loc["baseline_1d", "knn_purity"]
            == off.loc["baseline_1d", "knn_purity"]), "baseline must not move"


def test_the_random_floor_also_goes_through_the_correction(tmp_path):
    """A floor that skipped the correction could not absorb structure the
    correction itself introduces."""
    m = _mod()
    p = _batched_inputs(tmp_path)
    seen = []
    from nome_hic_concordance import batch_correction as bc
    orig = bc.correct_and_report
    bc.correct_and_report = lambda df, bl, meth, rs, nm: (seen.append(nm)
                                                          or orig(df, bl, meth,
                                                                  rs, nm))
    try:
        out = tmp_path / "o"
        assert m.main(["--concordance-c", p["c"], "--concordance-d", p["d"],
                       "--baseline-1d", p["b1"], "--baseline-3d", p["b3"],
                       "--samples", p["samples"], "--n-svd", "5",
                       "--n-perm", "0", "--plot-format", "png",
                       "--batch-column", "exp_batch",
                       "--concordance-batch-correction", "center",
                       "--output-dir", str(out)]) == 0
    finally:
        bc.correct_and_report = orig
    assert seen == ["C", "D", "random"], seen


def test_concordance_batch_column_may_differ_from_batch_column(tmp_path):
    """The two corrections act at different stages, so a sweep can land on
    different variables for each."""
    a = _run_batched(tmp_path, "a",
                     ["--batch-column", "lib_batch",
                      "--concordance-batch-column", "exp_batch",
                      "--concordance-batch-correction", "center"])
    b = _run_batched(tmp_path, "b",
                     ["--batch-column", "lib_batch",
                      "--concordance-batch-correction", "center"])
    # exp_batch is the variable carrying the planted offset; lib_batch is not,
    # so pointing the correction at it does strictly less.
    assert a.loc["C", "knn_purity"] > b.loc["C", "knn_purity"]


def test_correction_without_a_batch_column_fails_loudly(tmp_path):
    m = _mod()
    p = _batched_inputs(tmp_path)
    with pytest.raises(SystemExit, match="concordance-batch-correction"):
        m.main(["--concordance-c", p["c"], "--concordance-d", p["d"],
                "--baseline-1d", p["b1"], "--baseline-3d", p["b3"],
                "--samples", p["samples"],
                "--concordance-batch-correction", "center",
                "--output-dir", str(tmp_path / "x")])


def test_unknown_concordance_batch_column_fails_loudly(tmp_path):
    m = _mod()
    p = _batched_inputs(tmp_path)
    with pytest.raises(SystemExit, match="--concordance-batch-column"):
        m.main(["--concordance-c", p["c"], "--concordance-d", p["d"],
                "--baseline-1d", p["b1"], "--baseline-3d", p["b3"],
                "--samples", p["samples"],
                "--concordance-batch-column", "nope",
                "--concordance-batch-correction", "center",
                "--output-dir", str(tmp_path / "y")])


def test_a_degenerate_arm_is_called_out_not_reported_as_chance(tmp_path, caplog):
    """Chance-level metrics have two causes — a feature with no cell-type
    information, and an arm that broke upstream — and they must be tellable
    apart in the log."""
    import logging
    m = _mod()
    p = _batched_inputs(tmp_path)
    dead = pd.read_parquet(p["c"])
    dead.iloc[:, :] = np.nan
    dead.to_parquet(tmp_path / "dead.parquet")
    out = tmp_path / "o"
    with caplog.at_level(logging.ERROR):
        assert m.main(["--concordance-c", str(tmp_path / "dead.parquet"),
                       "--concordance-d", p["d"],
                       "--baseline-1d", p["b1"], "--baseline-3d", p["b3"],
                       "--samples", p["samples"], "--n-svd", "5",
                       "--n-perm", "0", "--plot-format", "png",
                       "--output-dir", str(out)]) == 0
    assert "no variance" in caplog.text
    tab = pd.read_csv(out / "GCH_metrics.tsv", sep="\t").set_index("arm")
    assert tab.loc["C", "score_sd"] == 0.0
    assert tab.loc["D", "score_sd"] > 0.0


def test_a_correction_that_kills_the_matrix_stops_the_run(tmp_path,
                                                          monkeypatch):
    """Better a failed run than a bar sitting at chance for the wrong
    reason."""
    m = _mod()
    from nome_hic_concordance import batch_correction as bc
    p = _batched_inputs(tmp_path)
    monkeypatch.setattr(
        bc, "apply_batch_correction",
        lambda mat, b, method, random_state: pd.DataFrame(
            np.nan, index=mat.index, columns=mat.columns))
    with pytest.raises(ValueError, match="returned NaN for"):
        m.main(["--concordance-c", p["c"], "--concordance-d", p["d"],
                "--baseline-1d", p["b1"], "--baseline-3d", p["b3"],
                "--samples", p["samples"], "--n-svd", "5", "--n-perm", "0",
                "--plot-format", "png", "--batch-column", "exp_batch",
                "--concordance-batch-correction", "combat",
                "--output-dir", str(tmp_path / "x")])



# ------------------------------------------------- choosing the bars (--arms) --

def _run_min(tmp_path, extra, with_d=False):
    m = _mod()
    p = _inputs(tmp_path)
    out = tmp_path / "out"
    argv = ["--concordance-c", p["c"],
            "--baseline-1d", p["b1"], "--baseline-3d", p["b3"],
            "--samples", p["samples"], "--context", "GCH",
            "--n-svd", "5", "--n-perm", "0", "--exclude-chrom", "chrX",
            "--plot-format", "png", "--output-dir", str(out)]
    if with_d:
        argv += ["--concordance-d", p["d"]]
    assert m.main(argv + [x.format(**p) for x in extra]) == 0
    return pd.read_csv(out / "GCH_metrics.tsv", sep="\t"), out


def test_concordance_d_is_optional(tmp_path):
    tab, _ = _run_min(tmp_path, [])
    assert list(tab["arm"]) == ["baseline_1d", "baseline_3d", "C",
                                "C_linear", "C_quadratic", "random"]
    # with no D the random control shuffles C
    t = tab.set_index("arm")
    assert t.loc["C", "knn_purity"] > t.loc["random", "knn_purity"]


def test_main_figure_set_with_both_floors(tmp_path):
    """Two baselines, the provided C, C's drawn null and the random control
    -- nothing else, and no residual-of-a-residual arms."""
    tab, out = _run_min(tmp_path, [
        "--arms", "baseline_1d,baseline_3d,C,C_null,random",
        "--concordance-c-null", "{cnull}", "--c-label", "C residual"])
    assert list(tab["arm"]) == ["baseline_1d", "baseline_3d", "C", "C_null",
                                "random"]
    assert list(tab["label"]) == ["1D baseline", "3D baseline", "C residual",
                                  "C residual, drawn null", "random control"]
    t = tab.set_index("arm")
    for arm in ("C_null", "random"):
        assert np.isfinite(t.loc[arm, "knn_purity"]), arm
        # the planted C beats both floors
        assert t.loc["C", "knn_purity"] > t.loc[arm, "knn_purity"], arm
    assert (out / "GCH_knn_purity_barplot.png").exists()


def test_d_arms_without_d_are_refused(tmp_path):
    m = _mod()
    p = _inputs(tmp_path)
    with pytest.raises(SystemExit, match="--concordance-d"):
        m.main(["--concordance-c", p["c"], "--baseline-1d", p["b1"],
                "--baseline-3d", p["b3"], "--samples", p["samples"],
                "--arms", "C,D", "--output-dir", str(tmp_path / "o")])


def test_a_null_bar_without_the_null_is_refused(tmp_path):
    m = _mod()
    p = _inputs(tmp_path)
    with pytest.raises(SystemExit, match="--concordance-c-null"):
        m.main(["--concordance-c", p["c"], "--baseline-1d", p["b1"],
                "--baseline-3d", p["b3"], "--samples", p["samples"],
                "--arms", "C,C_null", "--output-dir", str(tmp_path / "o")])


def test_random_from_d_without_d_is_refused(tmp_path):
    m = _mod()
    p = _inputs(tmp_path)
    with pytest.raises(SystemExit, match="--random-source d"):
        m.main(["--concordance-c", p["c"], "--baseline-1d", p["b1"],
                "--baseline-3d", p["b3"], "--samples", p["samples"],
                "--random-source", "d", "--output-dir", str(tmp_path / "o")])


def test_unknown_arm_is_refused(tmp_path):
    m = _mod()
    p = _inputs(tmp_path)
    with pytest.raises(SystemExit, match="unknown"):
        m.main(["--concordance-c", p["c"], "--baseline-1d", p["b1"],
                "--baseline-3d", p["b3"], "--samples", p["samples"],
                "--arms", "C,E", "--output-dir", str(tmp_path / "o")])



def test_matrix_norm_none_is_accepted_as_in_nome_hic_compare(tmp_path):
    """HCG runs use --matrix-norm none there; the bar plots must follow."""
    tab, out = _run_min(tmp_path, ["--matrix-norm", "none"])
    t = tab.set_index("arm")
    assert np.isfinite(t.loc["C", "knn_purity"])
    assert t.loc["C", "knn_purity"] > t.loc["random", "knn_purity"]
    # and it is a different normalization, not quantile under another name
    q, _ = _run_min(tmp_path / "q", [])
    assert not np.isclose(t.loc["C", "score_sd"],
                          q.set_index("arm").loc["C", "score_sd"])
