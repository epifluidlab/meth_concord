"""Unit tests for concordance._compute_cell_concordance and friends."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nome_hic_concordance.concordance import (
    _compute_cell_concordance,
    _sort_bins,
    compute_concordance_matrix,
)


def _make_posteriors(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    # Cast columns to expected types.
    for col in ("pos1", "pos2", "distance"):
        df[col] = df[col].astype(np.int64)
    for col in ("k1", "n1", "k2", "n2"):
        df[col] = df[col].astype(np.int32)
    for col in ("posterior1", "posterior2"):
        df[col] = df[col].astype(float)
    return df


def test_perfect_concordance(tmp_path):
    """If all pairs have identical posteriors, r = NaN due to zero variance.
    If all pairs have linearly related posteriors, r = 1.
    """
    # Build 10 pairs in chr1 bin 0 where posterior1 == posterior2
    records = []
    for i in range(10):
        p = 0.1 * i
        records.append({
            "chrom": "chr1", "pos1": i * 100, "k1": 1, "n1": 5,
            "pos2": i * 100 + 1000, "k2": 1, "n2": 5,
            "distance": 25_000,  # -> passes min_distance=20kb
            "posterior1": p, "posterior2": p,
        })
    df = _make_posteriors(records)

    parquet_path = tmp_path / "cell_A.posteriors.parquet"
    df.to_parquet(parquet_path)

    cell_name, conc, counts = _compute_cell_concordance(
        str(parquet_path), "cell_A",
        bin_size=5_000_000, min_distance=20_000,
        min_gch_sites=3, min_pairs_per_bin=5,
    )
    assert "chr1:0" in conc
    r = conc["chr1:0"]
    assert np.isclose(r, 1.0, atol=1e-9)
    assert counts["chr1:0"] == 10


def test_anti_correlated_pairs(tmp_path):
    records = []
    for i in range(10):
        records.append({
            "chrom": "chr2", "pos1": i * 100, "k1": 2, "n1": 4,
            "pos2": i * 100 + 50_000, "k2": 2, "n2": 4,
            "distance": 50_000,
            "posterior1": 0.1 * i,
            "posterior2": 1.0 - 0.1 * i,  # perfectly anticorrelated
        })
    df = _make_posteriors(records)
    parquet_path = tmp_path / "cell_B.posteriors.parquet"
    df.to_parquet(parquet_path)

    _, conc, _ = _compute_cell_concordance(
        str(parquet_path), "cell_B",
        bin_size=5_000_000, min_distance=20_000,
        min_gch_sites=3, min_pairs_per_bin=5,
    )
    assert np.isclose(conc["chr2:0"], -1.0, atol=1e-9)


def test_nan_for_insufficient_pairs(tmp_path):
    records = [
        {"chrom": "chr1", "pos1": 100, "k1": 1, "n1": 5,
         "pos2": 30_000, "k2": 1, "n2": 5, "distance": 29_900,
         "posterior1": 0.8, "posterior2": 0.7},
        {"chrom": "chr1", "pos1": 200, "k1": 1, "n1": 5,
         "pos2": 30_100, "k2": 1, "n2": 5, "distance": 29_900,
         "posterior1": 0.3, "posterior2": 0.4},
    ]
    df = _make_posteriors(records)
    parquet_path = tmp_path / "cell_C.posteriors.parquet"
    df.to_parquet(parquet_path)

    _, conc, counts = _compute_cell_concordance(
        str(parquet_path), "cell_C",
        bin_size=5_000_000, min_distance=20_000,
        min_gch_sites=3, min_pairs_per_bin=5,
    )
    assert np.isnan(conc["chr1:0"])
    assert counts["chr1:0"] == 2


def test_pairs_below_min_distance_excluded(tmp_path):
    records = []
    for i in range(8):
        records.append({
            "chrom": "chr1", "pos1": i * 100, "k1": 1, "n1": 5,
            "pos2": i * 100 + 500, "k2": 1, "n2": 5,  # short-range
            "distance": 500,
            "posterior1": 0.5, "posterior2": 0.5,
        })
    df = _make_posteriors(records)
    parquet_path = tmp_path / "cell_D.posteriors.parquet"
    df.to_parquet(parquet_path)

    _, conc, counts = _compute_cell_concordance(
        str(parquet_path), "cell_D",
        bin_size=5_000_000, min_distance=20_000,
        min_gch_sites=3, min_pairs_per_bin=5,
    )
    assert conc == {}
    assert counts == {}


def test_bin_assignment_uses_midpoint(tmp_path):
    """A pair with pos1=1M, pos2=9M should land in bin 5M (midpoint=5M)."""
    records = [{
        "chrom": "chr1",
        "pos1": 1_000_000, "k1": 3, "n1": 5,
        "pos2": 9_000_000, "k2": 3, "n2": 5,
        "distance": 8_000_000,
        "posterior1": 0.5, "posterior2": 0.5,
    } for _ in range(6)]

    # Different posteriors so variance isn't zero
    for i, r in enumerate(records):
        r["posterior1"] = 0.1 * i
        r["posterior2"] = 0.1 * i + 0.05

    df = _make_posteriors(records)
    parquet_path = tmp_path / "cell_E.posteriors.parquet"
    df.to_parquet(parquet_path)

    _, conc, _ = _compute_cell_concordance(
        str(parquet_path), "cell_E",
        bin_size=5_000_000, min_distance=20_000,
        min_gch_sites=3, min_pairs_per_bin=5,
    )
    # Midpoint 5_000_000 // 5_000_000 = 1 -> bin_start = 5_000_000
    assert "chr1:5000000" in conc


def test_sort_bins_orders_chromosome_and_start():
    bins = {"chr2:0", "chr1:5000000", "chr1:0", "chr10:0"}
    ordered = _sort_bins(bins, chrom_sizes=None)
    assert ordered == ["chr1:0", "chr1:5000000", "chr2:0", "chr10:0"]


def test_compute_concordance_matrix_end_to_end(tmp_path):
    """Assemble two cells' posteriors and verify the matrix shape + contents."""
    cell_paths = {}

    # Cell A has a high-r bin, cell B does not.
    records_a = []
    for i in range(10):
        records_a.append({
            "chrom": "chr1", "pos1": i * 100, "k1": 1, "n1": 5,
            "pos2": i * 100 + 50_000, "k2": 1, "n2": 5, "distance": 50_000,
            "posterior1": 0.1 * i, "posterior2": 0.1 * i + 0.02,
        })
    df_a = _make_posteriors(records_a)
    p_a = tmp_path / "cell_A.parquet"
    df_a.to_parquet(p_a)
    cell_paths["A"] = str(p_a)

    records_b = []
    for i in range(10):
        records_b.append({
            "chrom": "chr1", "pos1": i * 100, "k1": 1, "n1": 5,
            "pos2": i * 100 + 50_000, "k2": 1, "n2": 5, "distance": 50_000,
            "posterior1": 0.1 * (i % 2), "posterior2": 0.9 * (1 - i % 2),
        })
    df_b = _make_posteriors(records_b)
    p_b = tmp_path / "cell_B.parquet"
    df_b.to_parquet(p_b)
    cell_paths["B"] = str(p_b)

    conc_df, cnt_df = compute_concordance_matrix(
        cell_paths,
        chrom_sizes={"chr1": 250_000_000},
        bin_size=5_000_000,
        min_distance=20_000,
        min_gch_sites=3,
        min_pairs_per_bin=5,
        n_threads=1,
        output_dir=str(tmp_path / "output"),
    )

    assert list(conc_df.index) == ["A", "B"]
    assert "chr1:0" in conc_df.columns
    assert conc_df.loc["A", "chr1:0"] > 0.9
    assert conc_df.loc["B", "chr1:0"] < -0.9
    assert (cnt_df.loc["A", "chr1:0"]) == 10
