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


def test_run_comparison_requires_shared_cells(tmp_path):
    a = pd.DataFrame(np.zeros((2, 4)), index=["x", "y"], columns=_bins(4, 250_000))
    b = pd.DataFrame(np.zeros((2, 4)), index=["p", "q"], columns=_bins(4, 250_000))
    with pytest.raises(ValueError, match="shared cells"):
        comparison.run_comparison(
            {"a": a, "b": b}, None, baseline_names=["a"],
            concordance_names=["b"], output_dir=str(tmp_path))
