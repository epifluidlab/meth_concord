"""Unit tests for matrix_ops.process_concordance_matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import matrix_ops


def _make_matrix(values):
    cells = [f"cell_{i}" for i in range(len(values))]
    bins = [f"chr1:{i * 5_000_000}" for i in range(len(values[0]))]
    return pd.DataFrame(values, index=cells, columns=bins)


def test_bin_filter_exact_threshold(tmp_path):
    # 5 cells, 3 bins; bin 0 has 4/5 non-NA, bin 1 has 3/5, bin 2 has 5/5
    mat = _make_matrix([
        [0.1, 0.2, 0.3],
        [0.2, np.nan, 0.4],
        [0.3, 0.4, 0.5],
        [0.4, np.nan, 0.6],
        [np.nan, 0.5, 0.7],
    ])
    # coverage_frac = [0.8, 0.6, 1.0]
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.8, output_dir=str(tmp_path),
    )
    assert result["retained_bins"] == ["chr1:0", "chr1:10000000"]


def test_imputation_fills_with_zero(tmp_path):
    mat = _make_matrix([
        [0.1, np.nan, 0.3],
        [0.2, 0.3, 0.4],
        [np.nan, 0.4, 0.5],
        [0.4, 0.5, 0.6],
        [0.5, 0.6, 0.7],
    ])
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path),
    )
    assert not result["imputed"].isna().any().any()
    # The entries that were NaN must now be 0.
    assert result["imputed"].iloc[0, 1] == 0.0
    assert result["imputed"].iloc[2, 0] == 0.0


def test_zscore_has_zero_mean_unit_std(tmp_path):
    rng = np.random.default_rng(0)
    mat = _make_matrix(rng.normal(0, 0.3, size=(30, 15)).tolist())
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path),
    )
    z = result["zscore"]
    means = z.mean(axis=0).to_numpy()
    stds = z.std(axis=0, ddof=1).to_numpy()
    np.testing.assert_allclose(means, 0.0, atol=1e-10)
    # After imputation+zscore each column has sample std 1 when the
    # underlying column is non-constant.
    np.testing.assert_allclose(stds, 1.0, atol=1e-6)


def test_raises_on_empty(tmp_path):
    empty = pd.DataFrame()
    with pytest.raises(ValueError):
        matrix_ops.process_concordance_matrix(empty, output_dir=str(tmp_path))


def test_zero_variance_column_zscore_is_zero(tmp_path):
    mat = _make_matrix([
        [0.5, 0.1],
        [0.5, 0.2],
        [0.5, 0.3],
        [0.5, 0.4],
    ])
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5, output_dir=str(tmp_path),
    )
    # First bin has zero variance -> z-score stays at 0.
    z = result["zscore"]
    np.testing.assert_array_equal(z["chr1:0"].to_numpy(), np.zeros(4))


def test_blacklist_drops_overlapping_bins(tmp_path):
    """Bins overlapping any blacklist interval are dropped, regardless of coverage."""
    # 4 cells x 4 bins (chr1, 1kb bins): 0-1000, 1000-2000, 2000-3000, 3000-4000.
    cells = [f"cell_{i}" for i in range(4)]
    bins = [f"chr1:{i*1000}" for i in range(4)]
    mat = pd.DataFrame(np.full((4, 4), 0.5), index=cells, columns=bins)

    # Blacklist: 500-1500 → overlaps bin0 [0,1000) and bin1 [1000,2000).
    # 3500-3700 → overlaps bin3 [3000,4000).  bin2 is clean.
    bed = tmp_path / "blacklist.bed"
    bed.write_text("chr1\t500\t1500\nchr1\t3500\t3700\n")

    out_dir = tmp_path / "out"
    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5,
        output_dir=str(out_dir),
        blacklist_bed=str(bed),
        bin_size=1000,
    )
    assert result["retained_bins"] == ["chr1:2000"]


def test_blacklist_other_chromosomes_unaffected(tmp_path):
    """Bins on chromosomes absent from the blacklist are untouched."""
    cells = [f"cell_{i}" for i in range(3)]
    bins = ["chr1:0", "chr2:0", "chr2:1000"]
    mat = pd.DataFrame(np.full((3, 3), 0.5), index=cells, columns=bins)

    bed = tmp_path / "blacklist.bed"
    bed.write_text("chr1\t0\t1000\n")  # only chr1

    result = matrix_ops.process_concordance_matrix(
        mat, min_cell_coverage=0.5,
        output_dir=str(tmp_path / "out"),
        blacklist_bed=str(bed),
        bin_size=1000,
    )
    assert "chr1:0" not in result["retained_bins"]
    assert "chr2:0" in result["retained_bins"]
    assert "chr2:1000" in result["retained_bins"]


def test_blacklist_requires_bin_size(tmp_path):
    cells = [f"cell_{i}" for i in range(2)]
    mat = pd.DataFrame(np.full((2, 1), 0.5), index=cells, columns=["chr1:0"])
    bed = tmp_path / "blacklist.bed"
    bed.write_text("chr1\t0\t1000\n")
    with pytest.raises(ValueError, match="bin_size"):
        matrix_ops.process_concordance_matrix(
            mat, output_dir=str(tmp_path / "out"),
            blacklist_bed=str(bed),  # no bin_size
        )


def test_blacklist_merges_overlapping_intervals(tmp_path):
    """Overlapping BED intervals on the same chrom should be merged."""
    bed = tmp_path / "blacklist.bed"
    # Two overlapping intervals that together cover 0-2500.
    bed.write_text("chr1\t0\t1500\nchr1\t1000\t2500\n")
    bl = matrix_ops._load_blacklist_bed(str(bed))
    assert bl["chr1"].tolist() == [[0, 2500]]
