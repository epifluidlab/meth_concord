"""Tests for the comparison harness (Phase 3).

Synthetic matrices with KNOWN relationships:
  - a 'redundant' concordance matrix == accessibility  → high R², no AUC gain
  - a 'unique' concordance matrix carrying an orthogonal cell-type signal
    → low R², positive AUC gain over the baselines
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import comparison


def _cells(n):
    return [f"c{i}" for i in range(n)]


def _bins(n, size, chrom="chr1"):
    return [f"{chrom}:{i * size}" for i in range(n)]


def test_align_to_grid_broadcasts_coarse_to_fine():
    cells = _cells(3)
    fine = _bins(8, 250_000)          # chr1:0 .. chr1:1750000
    coarse = _bins(2, 1_000_000)      # chr1:0, chr1:1000000
    pred = pd.DataFrame(
        [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]],
        index=cells, columns=coarse,
    )
    aligned = comparison.align_to_grid(pred, fine)
    assert list(aligned.columns) == fine
    # First four 250kb bins inherit the chr1:0 (=col 0) value; next four inherit chr1:1000000.
    np.testing.assert_array_equal(aligned.iloc[0, :4].to_numpy(), [10.0] * 4)
    np.testing.assert_array_equal(aligned.iloc[0, 4:].to_numpy(), [20.0] * 4)
    np.testing.assert_array_equal(aligned.iloc[2, :4].to_numpy(), [12.0] * 4)


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    n = 40
    cells = _cells(n)
    sig = np.r_[np.ones(n // 2), -np.ones(n // 2)]  # cell-type contrast
    bins8 = _bins(8, 250_000)

    # Weak accessibility signal (low amplitude vs noise).
    acc = rng.normal(0, 1.0, size=(n, 8))
    acc[:, 0] += 0.3 * sig
    acc[:, 1] += 0.3 * sig
    accessibility = pd.DataFrame(acc, index=cells, columns=bins8)

    # Redundant concordance == accessibility (+ negligible noise).
    redundant = accessibility + rng.normal(0, 1e-3, size=(n, 8))
    redundant = pd.DataFrame(redundant.to_numpy(), index=cells, columns=bins8)

    # Unique concordance: STRONG orthogonal cell-type signal.
    uniq = rng.normal(0, 0.3, size=(n, 8))
    uniq[:, 2] += 3.0 * sig
    uniq[:, 3] += 3.0 * sig
    unique = pd.DataFrame(uniq, index=cells, columns=bins8)

    labels = pd.Series(["A"] * (n // 2) + ["B"] * (n // 2), index=cells)
    return accessibility, redundant, unique, labels


def test_variance_partition_redundant_high_unique_low(tmp_path):
    accessibility, redundant, unique, _ = _synthetic()
    # Regress each on the (normalized) accessibility baseline.
    base = comparison._prep_matrix(accessibility, "acc", str(tmp_path),
                                   "quantile", 0.5)
    red_n = comparison._prep_matrix(redundant, "red", str(tmp_path),
                                    "quantile", 0.5)
    uniq_n = comparison._prep_matrix(unique, "uniq", str(tmp_path),
                                     "quantile", 0.5)
    red_resid = comparison.variance_partition(
        red_n, {"accessibility": base}, "red", str(tmp_path))
    uniq_resid = comparison.variance_partition(
        uniq_n, {"accessibility": base}, "uniq", str(tmp_path))

    red_r2 = pd.read_parquet(tmp_path / "regression_R2_per_bin_red.parquet")["R2"]
    uniq_r2 = pd.read_parquet(tmp_path / "regression_R2_per_bin_uniq.parquet")["R2"]
    # Redundant (== accessibility) is well explained; unique is not.
    assert np.nanmedian(red_r2) > 0.9
    assert np.nanmedian(uniq_r2) < 0.3
    # Residual of the redundant matrix is near-zero; unique residual retains signal.
    assert float(np.abs(red_resid.to_numpy()).mean()) < \
        float(np.abs(uniq_resid.to_numpy()).mean())


def test_classifier_unique_adds_auc_redundant_does_not(tmp_path):
    accessibility, redundant, unique, labels = _synthetic()
    matrices = {"accessibility": accessibility,
                "redundant": redundant, "unique": unique}
    tab = comparison.classifier_comparison(
        matrices, labels,
        baseline_names=["accessibility"],
        concordance_names=["unique", "redundant"],
        output_dir=str(tmp_path), matrix_norm="quantile", n_svd=5,
        min_cell_coverage=0.5, n_folds=5, random_state=0,
    )
    by = tab.set_index("feature_set")["auc_mean"]
    base = by["baselines"]
    # Unique signal lifts AUC over the baseline; redundant does not.
    assert by["baselines+unique"] > base + 0.05
    assert by["unique"] > by["accessibility"]
    gain = pd.read_csv(tmp_path / "classifier_auc_gain.tsv", sep="\t") \
        .set_index("concordance")["auc_gain_over_baselines"]
    assert gain["unique"] > gain["redundant"]


def test_run_comparison_end_to_end_smoke(tmp_path):
    accessibility, redundant, unique, labels = _synthetic()
    summary = comparison.run_comparison(
        {"accessibility": accessibility, "gch": unique, "hcg": redundant},
        labels,
        baseline_names=["accessibility"],
        concordance_names=["gch", "hcg"],
        output_dir=str(tmp_path),
        matrix_norm="quantile", n_svd=5, min_cell_coverage=0.5,
        n_folds=5, random_state=0,
    )
    assert (tmp_path / "summary.json").exists()
    assert "classifier" in summary and "residual" in summary
    # The unique ('gch') matrix should have lower baseline-explained R² →
    # a more cell-type-separated residual than the redundant ('hcg') one.
    assert "gch" in summary["residual"] and "hcg" in summary["residual"]
    with open(tmp_path / "summary.json") as fh:
        loaded = json.load(fh)
    assert loaded["n_cells"] == 40


def test_quadratic_level_control_removes_nonlinear_dependence(tmp_path):
    """When concordance is a NONLINEAR (quadratic) function of the level,
    linear control under-removes it (residual retains structure, R²<1) but
    quadratic control removes it (R²≈1, residual≈0)."""
    rng = np.random.default_rng(5)
    n, nb = 50, 12
    cells = _cells(n)
    bins = _bins(nb, 250_000)
    level = rng.normal(size=(n, nb))
    # concordance = pure quadratic function of the level (+ tiny noise).
    target_raw = level ** 2 + rng.normal(0, 1e-3, size=(n, nb))
    level_df = pd.DataFrame(level, index=cells, columns=bins)
    target_df = pd.DataFrame(target_raw, index=cells, columns=bins)
    base = comparison._prep_matrix(level_df, "lvl", str(tmp_path), "zscore", 0.5)
    tgt = comparison._prep_matrix(target_df, "tgt", str(tmp_path), "zscore", 0.5)

    comparison.variance_partition(tgt, {"lvl": base}, "lin", str(tmp_path),
                                  level_control="linear")
    comparison.variance_partition(tgt, {"lvl": base}, "quad", str(tmp_path),
                                  level_control="quadratic")
    r2_lin = pd.read_parquet(tmp_path / "regression_R2_per_bin_lin.parquet")["R2"]
    r2_quad = pd.read_parquet(tmp_path / "regression_R2_per_bin_quad.parquet")["R2"]
    # Quadratic explains the nonlinear level dependence much better.
    assert np.nanmedian(r2_quad) > np.nanmedian(r2_lin) + 0.2
    assert np.nanmedian(r2_quad) > 0.9


def test_design_matrix_linear_matches_plain_ols():
    """'linear' design (standardized) gives the same fit as raw-predictor OLS."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = 2.0 + 1.5 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(0, 0.1, 40)
    D = comparison._design_matrix(X, "linear")
    Draw = np.column_stack([np.ones(40), X])
    yhat_std = D @ np.linalg.lstsq(D, y, rcond=None)[0]
    yhat_raw = Draw @ np.linalg.lstsq(Draw, y, rcond=None)[0]
    np.testing.assert_allclose(yhat_std, yhat_raw, atol=1e-8)


def test_design_matrix_invalid_level_control_raises():
    with pytest.raises(ValueError, match="level_control"):
        comparison._design_matrix(np.zeros((5, 2)), "cubic")


def test_run_comparison_requires_shared_cells(tmp_path):
    a = pd.DataFrame(np.zeros((2, 4)), index=["x", "y"], columns=_bins(4, 250_000))
    b = pd.DataFrame(np.zeros((2, 4)), index=["p", "q"], columns=_bins(4, 250_000))
    with pytest.raises(ValueError, match="shared cells"):
        comparison.run_comparison(
            {"a": a, "b": b}, None, baseline_names=["a"],
            concordance_names=["b"], output_dir=str(tmp_path))


def test_knn_purity_stats_separated_vs_random():
    """Null-aware kNN purity: separated clusters beat chance (small perm_p);
    random embedding sits at chance (large perm_p)."""
    rng = np.random.default_rng(0)
    n = 60
    labels = pd.Series(["A"] * (n // 2) + ["B"] * (n // 2),
                       index=[f"c{i}" for i in range(n)])
    # Two clearly separated Gaussian blobs.
    sep = np.vstack([rng.normal(-5, 0.3, size=(n // 2, 4)),
                     rng.normal(+5, 0.3, size=(n // 2, 4))])
    s = comparison._knn_purity_stats(sep, labels, k=10, n_perm=200, seed=1)
    assert s["purity"] > 0.95
    assert abs(s["chance"] - 0.5) < 1e-9          # balanced 2-class
    assert s["perm_p"] < 0.05 and s["z"] > 3

    # Random embedding: purity ~ chance, not significant.
    rnd = rng.normal(0, 1, size=(n, 4))
    s2 = comparison._knn_purity_stats(rnd, labels, k=10, n_perm=200, seed=1)
    assert abs(s2["purity"] - s2["chance"]) < 0.12
    assert s2["perm_p"] > 0.05


def test_knn_purity_chance_reflects_imbalance():
    """Analytic chance = Σ p_g² rises with class imbalance (not fixed at 0.5)."""
    idx = [f"c{i}" for i in range(50)]
    bal = pd.Series(["A"] * 25 + ["B"] * 25, index=idx)
    imb = pd.Series(["A"] * 40 + ["B"] * 10, index=idx)
    scores = np.random.default_rng(0).normal(0, 1, size=(50, 3))
    cb = comparison._knn_purity_stats(scores, bal, n_perm=50)["chance"]
    ci = comparison._knn_purity_stats(scores, imb, n_perm=50)["chance"]
    assert abs(cb - 0.5) < 1e-9
    assert abs(ci - (0.8 ** 2 + 0.2 ** 2)) < 1e-9    # 0.68
    assert ci > cb


def _matrix(cells, bins, fn, seed):
    rng = np.random.default_rng(seed)
    data = np.array([[fn(c, b, rng) for b in range(bins)] for c in range(len(cells))])
    return pd.DataFrame(data, index=cells, columns=[f"chr1:{b*1000}" for b in range(bins)])


def test_pairwise_finds_headroom_where_aggregate_ceilings():
    """3 classes: baselines separate A/B/C-vs-rest at ceiling EXCEPT the A|B
    pair, which only concordance resolves → aggregate gain ~0 but pairwise
    shows a positive gain on the confusable A|B pair."""
    n = 60
    cells = [f"c{i}" for i in range(n)]
    lab = pd.Series((["A"] * 20 + ["B"] * 20 + ["C"] * 20), index=cells)
    grp = np.array([0] * 20 + [1] * 20 + [2] * 20)
    # Baseline: separates C from {A,B} strongly, but A and B are identical
    # (baseline cannot tell them apart → A|B AUC ~ chance).
    def base_fn(c, b, rng):
        center = 6.0 if grp[c] == 2 else 0.0
        return center + rng.normal(0, 0.5)
    # Concordance: cleanly separates A from B (orthogonal axis).
    def conc_fn(c, b, rng):
        center = 5.0 if grp[c] == 0 else (-5.0 if grp[c] == 1 else 0.0)
        return center + rng.normal(0, 0.5)
    base = _matrix(cells, 12, base_fn, 1)
    conc = _matrix(cells, 12, conc_fn, 2)
    pw = comparison.classifier_comparison_pairwise(
        {"accessibility": base, "gch": conc}, lab,
        baseline_names=["accessibility"], concordance_names=["gch"],
        output_dir=None if False else _tmp(), matrix_norm="zscore",
        n_svd=5, min_cell_coverage=0.0, n_folds=3, random_state=0,
        min_class_cells=5)
    ab = pw[pw["pair"] == "A|B"].iloc[0]
    assert ab["baseline_auc"] < 0.9            # baselines confuse A vs B
    assert ab["gain_gch"] > 0.05               # concordance rescues it


import tempfile
def _tmp():
    return tempfile.mkdtemp()


def test_variance_partition_regresses_out_batch():
    """The residual must be 'not explained by baselines AND batch'.

    Without batch as a covariate, the residual regresses concordance on the
    RAW batch-contaminated baseline and the baseline's batch leaks back in --
    which is why a residual UMAP could be MORE batch-split than its input.
    """
    import numpy as np, pandas as pd, tempfile
    from nome_hic_concordance import comparison as C
    rng = np.random.default_rng(1)
    n, nb = 200, 40
    cells = [f"c{i}" for i in range(n)]
    batch = np.where(np.arange(n) % 2 == 0, "a", "b")
    ct = rng.integers(0, 3, n)
    sig = ct[:, None] * rng.normal(0, 1, (3, nb))[ct]
    conc = sig + np.where(batch == "a", 1.5, -1.5)[:, None] + rng.normal(0, .3, (n, nb))
    base = 0.8 * sig + np.where(batch == "a", 1.0, -1.0)[:, None] + rng.normal(0, .3, (n, nb))
    cols = [f"chr1:{i}" for i in range(nb)]
    tdf = pd.DataFrame(conc, index=cells, columns=cols)
    bdf = pd.DataFrame(base, index=cells, columns=cols)
    blab = pd.Series(batch, index=cells)

    def batch_r2(resid):
        X = resid.to_numpy()
        tot = ((X - X.mean(0)) ** 2).sum()
        w = sum(((X[batch == g] - X[batch == g].mean(0)) ** 2).sum()
                for g in ("a", "b"))
        return 1 - w / tot

    with tempfile.TemporaryDirectory() as d:
        no = C.variance_partition(tdf, {"base": bdf}, "t", d, plot_format="png")
        bt = C.variance_partition(tdf, {"base": bdf}, "t", d, plot_format="png",
                                  batch_labels=blab)
    assert batch_r2(bt) < batch_r2(no) - 0.05, (batch_r2(no), batch_r2(bt))
    assert batch_r2(bt) < 0.05


def test_batch_design_shapes_and_centering():
    import numpy as np, pandas as pd
    from nome_hic_concordance import comparison as C
    cells = pd.Index([f"c{i}" for i in range(6)])
    # two batch variables, 2 and 3 levels -> (2-1)+(3-1) = 3 dummy columns
    bl = pd.DataFrame({"exp": ["a", "a", "b", "b", "a", "b"],
                       "plate": ["p1", "p2", "p3", "p1", "p2", "p3"]}, index=cells)
    B = C._batch_design(bl, cells)
    assert B.shape == (6, 3)
    assert np.allclose(B.mean(axis=0), 0.0), "columns must be mean-centred"
    assert C._batch_design(None, cells) is None
    # a single-level batch contributes no columns -> None
    one = pd.Series(["a"] * 6, index=cells)
    assert C._batch_design(one, cells) is None


# ------------------------------ concordance-stage batch correction (compare) --

def _batched(seed=0, offset=3.0):
    """Concordance and baseline matrices sharing a hard per-bin batch offset.

    The offset sits on the raw values, which is where it lands in the real
    pipeline: per-cell coverage drives the DM shrinkage and what RWR imputes,
    so batch is mixed into the bins before any embedding exists.
    """
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    n = 40
    cells = _cells(n)
    bins8 = _bins(8, 250_000)
    sig = np.r_[np.ones(n // 2), -np.ones(n // 2)]
    batch = pd.Series(["b1" if i % 2 == 0 else "b2" for i in range(n)],
                      index=cells)
    shift = np.where(batch.to_numpy() == "b1", offset, -offset)[:, None]
    base = pd.DataFrame(rng.normal(0, 1, (n, 8)) + 0.4 * sig[:, None],
                        index=cells, columns=bins8)
    conc = pd.DataFrame(rng.normal(0, 1, (n, 8)) + 0.4 * sig[:, None] + shift,
                        index=cells, columns=bins8)
    labels = pd.Series(["A"] * (n // 2) + ["B"] * (n // 2), index=cells)
    return base, conc, labels, batch


def test_concordance_stage_correction_hits_concordance_not_baselines(tmp_path):
    """The point of a separate flag: baselines come off a different stage and
    must be left as they are.  Checked downstream, on the analysis the
    correction is supposed to change, rather than on the bookkeeping."""
    import numpy as np
    from nome_hic_concordance import batch_correction as bc, comparison as C
    base, conc, labels, batch = _batched()
    assert float(np.nanmedian(bc.batch_anova_f_per_bin(conc, batch))) > 10.0, (
        "fixture must be batchy or the test proves nothing")

    def run(where, method):
        return C.run_comparison(
            {"accessibility": base, "gch": conc}, labels,
            baseline_names=["accessibility"], concordance_names=["gch"],
            output_dir=str(tmp_path / where), matrix_norm="quantile", n_svd=5,
            min_cell_coverage=0.5, n_folds=3, random_state=0,
            concordance_batch_labels=batch,
            concordance_batch_correction=method)

    off, on = run("off", "none"), run("on", "center")
    rec = on["concordance_batch_correction"]
    assert rec["method"] == "center" and rec["matrices"] == ["gch"]
    assert rec["applied_on"] == "raw_matrix_per_bin"
    assert "concordance_batch_correction" not in off
    assert on["marginal"]["gch"] != off["marginal"]["gch"], (
        "the corrected matrix must reach the analysis")
    assert on["marginal"]["accessibility"] == off["marginal"]["accessibility"], (
        "the baseline must be bit-identical either way")


def test_concordance_stage_correction_removes_the_offset(tmp_path):
    import numpy as np
    from nome_hic_concordance import batch_correction as bc
    base, conc, labels, batch = _batched()
    pre = float(np.nanmedian(bc.batch_anova_f_per_bin(conc, batch)))
    post = float(np.nanmedian(bc.batch_anova_f_per_bin(
        bc.correct_and_report(conc, batch, "center", 0, "gch"), batch)))
    assert post < pre / 10, (pre, post)


def test_compare_and_concordance_apply_the_same_transform():
    """Both CLIs must route through one implementation, or a setting chosen in
    one tool would not reproduce in the other."""
    import numpy as np
    from nome_hic_concordance import batch_correction as bc, cli
    _, conc, _, batch = _batched()
    a = cli._correct_concordance_batch(conc, batch, "center", 42, "x")
    b = bc.correct_and_report(conc, batch, "center", 42, "x")
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_compare_cli_defaults_the_column_to_batch_column(tmp_path):
    import pandas as pd
    from nome_hic_concordance import comparison_cli as cc
    base, conc, labels, batch = _batched()
    sheet = pd.DataFrame({"cell_name": labels.index, "group": labels.to_numpy(),
                          "exp_batch": batch.to_numpy(),
                          "lib_batch": ["L1", "L2"] * (len(labels) // 2)})
    sp = tmp_path / "s.tsv"
    sheet.to_csv(sp, sep="\t", index=False)
    base.to_parquet(tmp_path / "base.parquet")
    conc.to_parquet(tmp_path / "conc.parquet")
    argv = ["--accessibility-matrix", str(tmp_path / "base.parquet"),
            "--concordance-matrix", f"gch={tmp_path / 'conc.parquet'}",
            "--samples", str(sp), "--batch-column", "exp_batch",
            "--concordance-batch-correction", "center",
            "--n-svd", "5", "--n-folds", "3",
            "--output-dir", str(tmp_path / "out")]
    assert cc.main(argv) == 0
    import json
    with open(tmp_path / "out" / "summary.json") as fh:
        rec = json.load(fh)["concordance_batch_correction"]
    assert rec["method"] == "center" and rec["matrices"] == ["gch"]


def test_compare_cli_can_use_a_different_column_than_batch_column(tmp_path):
    """The two corrections act on different matrices at different stages, so a
    sweep can land on different variables for each."""
    import pandas as pd
    from nome_hic_concordance import comparison_cli as cc
    base, conc, labels, batch = _batched()
    sheet = pd.DataFrame({"cell_name": labels.index, "group": labels.to_numpy(),
                          "exp_batch": ["E1", "E2"] * (len(labels) // 2),
                          "lib_batch": batch.to_numpy()})
    sp = tmp_path / "s.tsv"
    sheet.to_csv(sp, sep="\t", index=False)
    base.to_parquet(tmp_path / "base.parquet")
    conc.to_parquet(tmp_path / "conc.parquet")
    args = cc.parse_args(
        ["--concordance-matrix", f"gch={tmp_path / 'conc.parquet'}",
         "--samples", str(sp), "--batch-column", "exp_batch",
         "--concordance-batch-column", "lib_batch",
         "--concordance-batch-correction", "center",
         "--output-dir", str(tmp_path / "out")])
    assert args.concordance_batch_column == "lib_batch"
    assert args.batch_column == "exp_batch"


def test_compare_cli_rejects_correction_without_a_column(tmp_path):
    import pandas as pd, pytest
    from nome_hic_concordance import comparison_cli as cc
    base, conc, labels, batch = _batched()
    sheet = pd.DataFrame({"cell_name": labels.index, "group": labels.to_numpy()})
    sp = tmp_path / "s.tsv"
    sheet.to_csv(sp, sep="\t", index=False)
    conc.to_parquet(tmp_path / "conc.parquet")
    with pytest.raises(ValueError, match="concordance-batch-correction"):
        cc.main(["--concordance-matrix", f"gch={tmp_path / 'conc.parquet'}",
                 "--samples", str(sp),
                 "--concordance-batch-correction", "center",
                 "--output-dir", str(tmp_path / "out")])


def test_compare_cli_rejects_an_unknown_concordance_batch_column(tmp_path):
    import pandas as pd, pytest
    from nome_hic_concordance import comparison_cli as cc
    base, conc, labels, batch = _batched()
    sheet = pd.DataFrame({"cell_name": labels.index, "group": labels.to_numpy(),
                          "exp_batch": batch.to_numpy()})
    sp = tmp_path / "s.tsv"
    sheet.to_csv(sp, sep="\t", index=False)
    conc.to_parquet(tmp_path / "conc.parquet")
    with pytest.raises(ValueError, match="concordance-batch-column"):
        cc.main(["--concordance-matrix", f"gch={tmp_path / 'conc.parquet'}",
                 "--samples", str(sp), "--concordance-batch-column", "nope",
                 "--concordance-batch-correction", "center",
                 "--output-dir", str(tmp_path / "out")])


def test_compare_cli_leaves_matrices_alone_when_off(tmp_path):
    import json
    import pandas as pd
    from nome_hic_concordance import comparison_cli as cc
    base, conc, labels, batch = _batched()
    sheet = pd.DataFrame({"cell_name": labels.index, "group": labels.to_numpy(),
                          "exp_batch": batch.to_numpy()})
    sp = tmp_path / "s.tsv"
    sheet.to_csv(sp, sep="\t", index=False)
    base.to_parquet(tmp_path / "base.parquet")
    conc.to_parquet(tmp_path / "conc.parquet")
    assert cc.main(["--accessibility-matrix", str(tmp_path / "base.parquet"),
                    "--concordance-matrix", f"gch={tmp_path / 'conc.parquet'}",
                    "--samples", str(sp), "--batch-column", "exp_batch",
                    "--n-svd", "5", "--n-folds", "3",
                    "--output-dir", str(tmp_path / "out")]) == 0
    with open(tmp_path / "out" / "summary.json") as fh:
        assert "concordance_batch_correction" not in json.load(fh)


def test_named_spec_on_a_bare_path_flag_says_which_flag_to_use(tmp_path):
    """The failure it replaces was a FileNotFoundError quoting
    'hcg_mean=output/.../accessibility_mean.parquet', which reads like a
    corrupt path rather than the wrong flag."""
    import pandas as pd, pytest
    from nome_hic_concordance import comparison_cli as cc
    base, conc, labels, batch = _batched()
    p = tmp_path / "acc.parquet"
    base.to_parquet(p)
    sheet = pd.DataFrame({"cell_name": labels.index, "group": labels.to_numpy()})
    sp = tmp_path / "s.tsv"
    sheet.to_csv(sp, sep="\t", index=False)
    conc.to_parquet(tmp_path / "conc.parquet")
    with pytest.raises(ValueError, match="--baseline-matrix hcg_mean="):
        cc.main(["--accessibility-matrix", f"hcg_mean={p}",
                 "--concordance-matrix", f"gch={tmp_path / 'conc.parquet'}",
                 "--samples", str(sp), "--output-dir", str(tmp_path / "o")])


def test_a_real_path_containing_an_equals_sign_still_loads(tmp_path):
    """The guard keys on the file not existing, so it cannot hijack a path
    that genuinely has '=' in it."""
    from nome_hic_concordance import comparison_cli as cc
    base, _, _, _ = _batched()
    odd = tmp_path / "run=1.parquet"
    base.to_parquet(odd)
    assert cc._plain_path(str(odd), "--accessibility-matrix") == str(odd)


def test_unnamed_baseline_is_not_called_concordance(tmp_path, caplog):
    """It used to be: a bare --baseline-matrix inherited the concordance
    default, so a BASELINE was labelled as the thing it is the control for in
    every figure, table and UMAP filename."""
    from nome_hic_concordance import comparison_cli as cc
    name, path = cc._parse_named("/x/acc.parquet", "baseline",
                                 "--baseline-matrix")
    assert (name, path) == ("baseline", "/x/acc.parquet")
    assert cc._parse_named("/x/c.parquet")[0] == "concordance"
    assert cc._parse_named("gch_mean=/x/acc.parquet")[0] == "gch_mean"


def test_two_bare_paths_collide_loudly_instead_of_overwriting(tmp_path):
    """The silent version put one matrix in BOTH baseline_names and
    concordance_names, so the variance partition regressed it on itself."""
    import pandas as pd, pytest
    from nome_hic_concordance import comparison_cli as cc
    base, conc, labels, _ = _batched()
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    base.to_parquet(a)
    conc.to_parquet(b)
    sheet = pd.DataFrame({"cell_name": labels.index, "group": labels.to_numpy()})
    sp = tmp_path / "s.tsv"
    sheet.to_csv(sp, sep="\t", index=False)
    with pytest.raises(ValueError, match="already taken"):
        cc.main(["--concordance-matrix", str(a),
                 "--concordance-matrix", str(b),
                 "--samples", str(sp), "--output-dir", str(tmp_path / "o")])


def test_a_baseline_may_not_shadow_a_fixed_matrix_name(tmp_path):
    import pandas as pd, pytest
    from nome_hic_concordance import comparison_cli as cc
    base, conc, labels, _ = _batched()
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    base.to_parquet(a)
    conc.to_parquet(b)
    sheet = pd.DataFrame({"cell_name": labels.index, "group": labels.to_numpy()})
    sp = tmp_path / "s.tsv"
    sheet.to_csv(sp, sep="\t", index=False)
    with pytest.raises(ValueError, match="already taken by a baseline"):
        cc.main(["--accessibility-matrix", str(a),
                 "--baseline-matrix", f"accessibility={a}",
                 "--concordance-matrix", f"gch={b}",
                 "--samples", str(sp), "--output-dir", str(tmp_path / "o")])
