"""Unit tests for the Beta-Binomial bin-level concordance pipeline.

Covers:
  - Linked/unlinked classification per cell per bin.
  - EB prior method-of-moments fit.
  - End-to-end Δ matrix construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nome_hic_concordance.concordance import (
    _compute_cell_betabinom_counts,
    _fit_betabinom_eb_prior,
    compute_betabinom_concordance_matrix,
)


def _make_posteriors(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in ("pos1", "pos2", "distance"):
        df[col] = df[col].astype(np.int64)
    for col in ("n1", "n2"):
        df[col] = df[col].astype(np.int32)
    for col in ("posterior1", "posterior2"):
        df[col] = df[col].astype(float)
    return df


def test_betabinom_counts_linked_and_unlinked(tmp_path):
    """Linked: both ends in bin AND distance >= min_distance.
    Unlinked: any informative end not part of a linked pair.
    """
    # bin_size=1000, min_distance=100, min_gch_sites=3
    # bin 0 = [0, 1000), bin 1000 = [1000, 2000)
    records = [
        # Pair 1: both ends in bin 0, distance=200 (>= 100), both ends informative.
        # → linked; both posteriors > 0.5 → active concordant.
        {"chrom": "chr1", "pos1": 100, "pos2": 300, "n1": 5, "n2": 5,
         "posterior1": 0.9, "posterior2": 0.9, "distance": 200},
        # Pair 2: both ends in bin 0, distance=50 (< 100, too short for "long-range").
        # → NOT linked.  Both ends informative → each becomes unlinked obs in bin 0.
        {"chrom": "chr1", "pos1": 400, "pos2": 450, "n1": 5, "n2": 5,
         "posterior1": 0.8, "posterior2": 0.3, "distance": 50},
        # Pair 3: pos1 in bin 0, pos2 in bin 1000.  Different bins → not linked.
        # Each end → unlinked in its own bin.
        {"chrom": "chr1", "pos1": 500, "pos2": 1500, "n1": 5, "n2": 5,
         "posterior1": 0.7, "posterior2": 0.2, "distance": 1000},
    ]
    df = _make_posteriors(records)
    p = tmp_path / "cell_A.parquet"
    df.to_parquet(p)

    cell, counts = _compute_cell_betabinom_counts(
        str(p), "cell_A",
        bin_size=1000, min_distance=100, min_gch_sites=3,
        posterior_threshold=0.5,
    )
    assert cell == "cell_A"

    # bin 0 should have: 1 linked pair (active-concordant), 2 unlinked ends from
    # pair 2 (one open, one closed), and 1 unlinked end from pair 3 (open).
    bin0 = counts["chr1:0"]
    assert bin0["n_linked"] == 1
    assert bin0["k_linked_active"] == 1
    assert bin0["n_unlinked"] == 3
    assert bin0["k_unlinked_open"] == 2

    # bin 1000 should have: 1 unlinked end from pair 3 (closed).
    bin1k = counts["chr1:1000"]
    assert bin1k["n_linked"] == 0
    assert bin1k["k_linked_active"] == 0
    assert bin1k["n_unlinked"] == 1
    assert bin1k["k_unlinked_open"] == 0


def test_betabinom_counts_uninformative_ends_excluded(tmp_path):
    """Ends with n < min_gch_sites should contribute to neither pool."""
    records = [
        # Pair 1: pos1 informative (n=5), pos2 not (n=1).  Both ends in bin 0.
        # → NOT linked (need both informative).  Only end1 → unlinked.
        {"chrom": "chr1", "pos1": 100, "pos2": 500, "n1": 5, "n2": 1,
         "posterior1": 0.9, "posterior2": 0.5, "distance": 400},
    ]
    df = _make_posteriors(records)
    p = tmp_path / "cell_B.parquet"
    df.to_parquet(p)

    _, counts = _compute_cell_betabinom_counts(
        str(p), "cell_B", bin_size=1000, min_distance=100,
        min_gch_sites=3, posterior_threshold=0.5,
    )
    assert counts["chr1:0"]["n_linked"] == 0
    assert counts["chr1:0"]["n_unlinked"] == 1
    assert counts["chr1:0"]["k_unlinked_open"] == 1


def test_betabinom_eb_prior_mom():
    """MoM EB prior recovers reasonable parameters from simulated data."""
    rng = np.random.default_rng(0)
    # Simulate 200 (cell, bin) observations from Beta(8, 4) → mean ≈ 0.667.
    alpha_true, beta_true = 8.0, 4.0
    n_obs = 200
    theta = rng.beta(alpha_true, beta_true, size=n_obs)
    n = rng.integers(5, 50, size=n_obs)
    k = rng.binomial(n, theta)

    counts_df = pd.DataFrame({"n_linked": n, "k_linked_active": k})
    alpha_hat, beta_hat, info = _fit_betabinom_eb_prior(counts_df, min_obs=50)

    mu_true = alpha_true / (alpha_true + beta_true)
    mu_hat = alpha_hat / (alpha_hat + beta_hat)
    assert abs(mu_hat - mu_true) < 0.05, f"mu_hat={mu_hat}, true={mu_true}"
    assert not info["fallback"]


def test_betabinom_eb_prior_fallback_when_sparse():
    """With too few observations, MoM falls back to Beta(1, 1)."""
    counts_df = pd.DataFrame({"n_linked": [5, 8], "k_linked_active": [3, 6]})
    alpha, beta, info = _fit_betabinom_eb_prior(counts_df, min_obs=50)
    assert alpha == 1.0 and beta == 1.0
    assert info["fallback"]


def test_betabinom_concordance_matrix_end_to_end(tmp_path):
    """Full pipeline: two cells, one bin per cell, expect non-NaN Δ."""
    # Cell A: many linked pairs all both-open in bin 0; few unlinked.
    rng = np.random.default_rng(1)
    rec_A = []
    for i in range(20):
        rec_A.append({
            "chrom": "chr1", "pos1": 100 + i, "pos2": 500 + i,
            "n1": 5, "n2": 5, "posterior1": 0.9, "posterior2": 0.9,
            "distance": 400,
        })
    # Some short-range unlinked observations.
    for i in range(20):
        rec_A.append({
            "chrom": "chr1", "pos1": 700 + i, "pos2": 720 + i,
            "n1": 5, "n2": 5, "posterior1": 0.8, "posterior2": 0.2,
            "distance": 20,  # short-range
        })
    df_A = _make_posteriors(rec_A)
    pA = tmp_path / "cell_A.parquet"
    df_A.to_parquet(pA)

    # Cell B: same number of pairs but half-half open/closed (low coordination).
    rec_B = []
    for i in range(20):
        p1 = 0.9 if i % 2 == 0 else 0.2
        p2 = 0.9 if (i + 1) % 2 == 0 else 0.2  # anti-correlated
        rec_B.append({
            "chrom": "chr1", "pos1": 100 + i, "pos2": 500 + i,
            "n1": 5, "n2": 5, "posterior1": p1, "posterior2": p2,
            "distance": 400,
        })
    for i in range(20):
        rec_B.append({
            "chrom": "chr1", "pos1": 700 + i, "pos2": 720 + i,
            "n1": 5, "n2": 5, "posterior1": 0.5, "posterior2": 0.5,
            "distance": 20,
        })
    df_B = _make_posteriors(rec_B)
    pB = tmp_path / "cell_B.parquet"
    df_B.to_parquet(pB)

    delta_df, pair_counts_df = compute_betabinom_concordance_matrix(
        {"A": str(pA), "B": str(pB)},
        bin_size=1000, min_distance=100, min_gch_sites=3,
        posterior_threshold=0.5, min_pairs_per_bin=5,
        eb_min_obs=2,  # tiny test → no fallback
        n_threads=1,
        output_dir=str(tmp_path / "output"),
    )

    # Both cells should have a non-NaN Δ at chr1:0.
    assert "chr1:0" in delta_df.columns
    assert delta_df.loc["A", "chr1:0"] > delta_df.loc["B", "chr1:0"], (
        "cell A (high coordination) should have higher Δ than cell B "
        "(anti-correlated)"
    )

    # pair_counts should reflect the linked counts.
    assert pair_counts_df.loc["A", "chr1:0"] == 20
    assert pair_counts_df.loc["B", "chr1:0"] == 20


def test_betabinom_no_pairs_in_bin_gives_nan(tmp_path):
    """If a cell has fewer than min_pairs_per_bin linked pairs in a bin, Δ=NaN."""
    rec = [
        # Just 3 linked pairs (< min_pairs_per_bin=5).
        {"chrom": "chr1", "pos1": 100 + i, "pos2": 500 + i,
         "n1": 5, "n2": 5, "posterior1": 0.9, "posterior2": 0.9, "distance": 400}
        for i in range(3)
    ]
    df = _make_posteriors(rec)
    p = tmp_path / "cell.parquet"
    df.to_parquet(p)

    delta_df, _ = compute_betabinom_concordance_matrix(
        {"X": str(p)},
        bin_size=1000, min_distance=100, min_gch_sites=3,
        posterior_threshold=0.5, min_pairs_per_bin=5,
        eb_min_obs=1, n_threads=1,
        output_dir=str(tmp_path / "output"),
    )
    # The bin should exist but be NaN (n_linked < min_pairs_per_bin).
    if "chr1:0" in delta_df.columns:
        assert np.isnan(delta_df.loc["X", "chr1:0"])
