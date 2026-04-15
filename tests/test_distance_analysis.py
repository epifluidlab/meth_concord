"""Tests for distance_analysis._process_cell."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nome_hic_concordance import distance_analysis


def _make_posteriors(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in ("pos1", "pos2", "distance"):
        df[col] = df[col].astype(np.int64)
    for col in ("k1", "n1", "k2", "n2"):
        df[col] = df[col].astype(np.int32)
    for col in ("posterior1", "posterior2"):
        df[col] = df[col].astype(float)
    return df


def test_process_cell_distance_bins(tmp_path):
    # Create pairs across multiple distance bands.
    records = []
    # Short-range (1-5 kb): 6 high-r pairs
    for i in range(6):
        records.append({
            "chrom": "chr1", "pos1": 100 * i, "k1": 3, "n1": 5,
            "pos2": 100 * i + 2_000, "k2": 3, "n2": 5, "distance": 2_000,
            "posterior1": 0.1 * i, "posterior2": 0.1 * i + 0.02,
        })
    # 100kb-500kb: 6 pairs with weak signal
    for i in range(6):
        records.append({
            "chrom": "chr1", "pos1": 1_000_000 + 100 * i, "k1": 3, "n1": 5,
            "pos2": 1_000_000 + 100 * i + 200_000, "k2": 3, "n2": 5,
            "distance": 200_000,
            "posterior1": 0.5 + 0.05 * i, "posterior2": 0.5 - 0.05 * i,
        })
    df = _make_posteriors(records)
    p = tmp_path / "cell_X.parquet"
    df.to_parquet(p)

    rows = distance_analysis._process_cell(
        "cell_X", str(p),
        distance_bins=[(1_000, 5_000), (100_000, 500_000), (5_000_000, 20_000_000)],
        min_gch_sites=3,
    )

    by_bin = {(r["dist_bin_start"], r["dist_bin_end"]): r for r in rows}
    assert by_bin[(1_000, 5_000)]["n_pairs"] == 6
    assert by_bin[(1_000, 5_000)]["r"] > 0.99
    assert by_bin[(100_000, 500_000)]["n_pairs"] == 6
    assert by_bin[(100_000, 500_000)]["r"] < -0.99
    assert by_bin[(5_000_000, 20_000_000)]["n_pairs"] == 0
    assert np.isnan(by_bin[(5_000_000, 20_000_000)]["r"])


def test_dist_label_formatting():
    assert distance_analysis._format_dist_label(1_000, 5_000) == "1kb-5kb"
    assert distance_analysis._format_dist_label(1_000_000, 5_000_000) == "1Mb-5Mb"
    assert distance_analysis._format_dist_label(1_500, 3_500) == "1500-3500"
