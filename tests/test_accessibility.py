"""Unit tests for nome_hic_concordance.accessibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import accessibility


def _make_posteriors(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in ("pos1", "pos2"):
        df[col] = df[col].astype(np.int64)
    for col in ("n1", "n2"):
        df[col] = df[col].astype(np.int32)
    for col in ("posterior1", "posterior2"):
        df[col] = df[col].astype(float)
    return df


def test_accessibility_per_bin_mean_basic(tmp_path):
    """Hand-computed per-(cell, bin) mean across both ends."""
    # bin_size = 1000, min_gch_sites = 2.
    # Pair 1: chr1, pos1=100 (bin 0, n=3, p=0.9), pos2=900 (bin 0, n=2, p=0.7)
    #         → both ends to bin 0:  mean = (0.9 + 0.7) / 2 = 0.8
    # Pair 2: chr1, pos1=1200 (bin 1000, n=4, p=0.3), pos2=1900 (bin 1000, n=1, p=0.2)
    #         → only end1 (n>=2) → bin 1000 gets 0.3 / 1 = 0.3
    # Pair 3: chr2, pos1=50 (bin 0, n=5, p=1.0), pos2=950 (bin 0, n=5, p=0.5)
    #         → both ends to chr2:0:  mean = (1.0 + 0.5) / 2 = 0.75
    records = [
        {"chrom": "chr1", "pos1": 100,  "pos2": 900,  "n1": 3, "n2": 2,
         "posterior1": 0.9, "posterior2": 0.7},
        {"chrom": "chr1", "pos1": 1200, "pos2": 1900, "n1": 4, "n2": 1,
         "posterior1": 0.3, "posterior2": 0.2},
        {"chrom": "chr2", "pos1": 50,   "pos2": 950,  "n1": 5, "n2": 5,
         "posterior1": 1.0, "posterior2": 0.5},
    ]
    p = tmp_path / "cellA.posteriors.parquet"
    _make_posteriors(records).to_parquet(p)

    acc_df, n_ends_df = accessibility.compute_accessibility_matrix(
        {"cellA": str(p)},
        bin_size=1000, min_gch_sites=2, n_threads=1,
        output_dir=str(tmp_path / "out"),
    )

    assert acc_df.loc["cellA", "chr1:0"] == pytest.approx(0.8)
    assert acc_df.loc["cellA", "chr1:1000"] == pytest.approx(0.3)
    assert acc_df.loc["cellA", "chr2:0"] == pytest.approx(0.75)

    assert n_ends_df.loc["cellA", "chr1:0"] == 2
    assert n_ends_df.loc["cellA", "chr1:1000"] == 1
    assert n_ends_df.loc["cellA", "chr2:0"] == 2


def test_accessibility_skips_uninformative_ends(tmp_path):
    """Ends with n < min_gch_sites or NaN posterior are skipped."""
    records = [
        # Both ends in bin 0 of chr1.  Only end1 is informative.
        {"chrom": "chr1", "pos1": 100, "pos2": 500, "n1": 5, "n2": 1,
         "posterior1": 0.6, "posterior2": np.nan},
        # Both ends informative — both contribute.
        {"chrom": "chr1", "pos1": 200, "pos2": 700, "n1": 3, "n2": 4,
         "posterior1": 0.8, "posterior2": 0.4},
    ]
    p = tmp_path / "cellB.posteriors.parquet"
    _make_posteriors(records).to_parquet(p)

    acc_df, n_ends_df = accessibility.compute_accessibility_matrix(
        {"cellB": str(p)},
        bin_size=1000, min_gch_sites=2, n_threads=1,
        output_dir=str(tmp_path / "out"),
    )

    # bin chr1:0 gets: end1 of pair 1 (0.6), end1 + end2 of pair 2 (0.8, 0.4).
    # Mean = (0.6 + 0.8 + 0.4) / 3 = 0.6
    assert acc_df.loc["cellB", "chr1:0"] == pytest.approx(0.6)
    assert n_ends_df.loc["cellB", "chr1:0"] == 3


def test_accessibility_multi_cell_alignment(tmp_path):
    """Two cells with different bin coverage produce aligned columns + NaN fills."""
    records_a = [
        {"chrom": "chr1", "pos1": 100, "pos2": 200, "n1": 3, "n2": 3,
         "posterior1": 0.9, "posterior2": 0.8},  # → bin chr1:0
    ]
    records_b = [
        {"chrom": "chr1", "pos1": 1500, "pos2": 1600, "n1": 3, "n2": 3,
         "posterior1": 0.2, "posterior2": 0.3},  # → bin chr1:1000
    ]
    pa = tmp_path / "A.posteriors.parquet"
    pb = tmp_path / "B.posteriors.parquet"
    _make_posteriors(records_a).to_parquet(pa)
    _make_posteriors(records_b).to_parquet(pb)

    acc_df, _ = accessibility.compute_accessibility_matrix(
        {"A": str(pa), "B": str(pb)},
        bin_size=1000, min_gch_sites=2, n_threads=1,
        output_dir=str(tmp_path / "out"),
    )

    # Both bins should appear as columns; the unobserved bin in each cell is NaN.
    assert set(acc_df.columns) == {"chr1:0", "chr1:1000"}
    assert acc_df.loc["A", "chr1:0"] == pytest.approx(0.85)
    assert np.isnan(acc_df.loc["A", "chr1:1000"])
    assert np.isnan(acc_df.loc["B", "chr1:0"])
    assert acc_df.loc["B", "chr1:1000"] == pytest.approx(0.25)


def test_accessibility_empty_cell_returns_no_bins(tmp_path):
    """A cell with zero rows produces an empty (cell, bin) contribution."""
    # Mock a real empty Step-4 parquet (zero rows, correct schema).
    df = pd.DataFrame({
        "chrom": pd.Series([], dtype=str),
        "pos1": pd.Series([], dtype=np.int64),
        "pos2": pd.Series([], dtype=np.int64),
        "n1": pd.Series([], dtype=np.int32),
        "n2": pd.Series([], dtype=np.int32),
        "posterior1": pd.Series([], dtype=float),
        "posterior2": pd.Series([], dtype=float),
    })
    p = tmp_path / "empty.posteriors.parquet"
    df.to_parquet(p)

    acc_df, n_ends_df = accessibility.compute_accessibility_matrix(
        {"empty": str(p)},
        bin_size=1000, min_gch_sites=2, n_threads=1,
        output_dir=str(tmp_path / "out"), save_outputs=False,
    )
    assert acc_df.shape == (1, 0)
    assert n_ends_df.shape == (1, 0)


def test_accessibility_outputs_saved(tmp_path):
    """save_outputs=True writes parquet + tsv files."""
    records = [
        {"chrom": "chr1", "pos1": 100, "pos2": 200, "n1": 3, "n2": 3,
         "posterior1": 0.5, "posterior2": 0.5},
    ]
    p = tmp_path / "X.posteriors.parquet"
    _make_posteriors(records).to_parquet(p)

    out_dir = tmp_path / "out"
    accessibility.compute_accessibility_matrix(
        {"X": str(p)},
        bin_size=1000, min_gch_sites=2, n_threads=1,
        output_dir=str(out_dir), save_outputs=True,
    )
    assert (out_dir / "accessibility_mean.parquet").exists()
    assert (out_dir / "accessibility_n_ends.parquet").exists()
    assert (out_dir / "accessibility_mean.tsv").exists()


def test_accessibility_no_save_when_save_outputs_false(tmp_path):
    """save_outputs=False skips disk I/O — useful for tests / in-memory use."""
    records = [
        {"chrom": "chr1", "pos1": 100, "pos2": 200, "n1": 3, "n2": 3,
         "posterior1": 0.5, "posterior2": 0.5},
    ]
    p = tmp_path / "X.posteriors.parquet"
    _make_posteriors(records).to_parquet(p)

    out_dir = tmp_path / "out_noskip"
    accessibility.compute_accessibility_matrix(
        {"X": str(p)},
        bin_size=1000, min_gch_sites=2, n_threads=1,
        output_dir=str(out_dir), save_outputs=False,
    )
    assert not out_dir.exists()


def test_sort_bins_chromosome_ordering():
    """_sort_bins orders chr1 < chr2 < ... < chr22 < chrX < chrY."""
    bins = {"chr10:0", "chr1:500000", "chr1:0", "chr2:0", "chrX:0", "chr9:1000000"}
    out = accessibility._sort_bins(bins, chrom_sizes=None)
    assert out == [
        "chr1:0", "chr1:500000", "chr2:0", "chr9:1000000", "chr10:0", "chrX:0",
    ]


def test_sort_bins_uses_chrom_sizes_ordering(tmp_path):
    """When chrom_sizes is provided, its key order wins."""
    bins = {"chrM:0", "chr1:0", "chr2:0"}
    chrom_sizes = {"chr2": 1, "chr1": 1, "chrM": 1}
    out = accessibility._sort_bins(bins, chrom_sizes=chrom_sizes)
    assert out == ["chr2:0", "chr1:0", "chrM:0"]
