"""Unit tests for the signed (Dirichlet-Multinomial) concordance pipeline.

Covers U-3 soft-count aggregation, DM EB MoM fit, the Δ_signed posterior
math from the design-doc §2.3 sanity-check table, and end-to-end
production of the cells × bins Δ_signed matrix (with and without RWR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import concordance


def _make_posteriors(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in ("pos1", "pos2", "distance"):
        df[col] = df[col].astype(np.int64)
    for col in ("n1", "n2"):
        df[col] = df[col].astype(np.int32)
    for col in ("posterior1", "posterior2"):
        df[col] = df[col].astype(float)
    return df


# ---------------------------------------------------------------------------
# U-3 soft-count aggregation
# ---------------------------------------------------------------------------


def test_soft_counts_linked_and_unlinked_split(tmp_path):
    """Pairs are split into linked (>= min_distance) and unlinked (< min_distance)
    soft-count pools, with U-3 contributions to all three categories per pair."""
    # bin_size=1000, min_distance=100, min_gch_sites=2
    records = [
        # Pair 1: same bin, distance=200 (linked).  Both ends informative.
        # p_l=0.9, p_r=0.8 → k_open=0.72, k_closed=0.02, k_mixed=0.26
        {"chrom": "chr1", "pos1": 100, "pos2": 300, "n1": 3, "n2": 3,
         "posterior1": 0.9, "posterior2": 0.8, "distance": 200},
        # Pair 2: same bin, distance=50 (unlinked).  Both ends informative.
        # p_l=0.6, p_r=0.4 → k_u_open=0.24, k_u_closed=0.24, k_u_mixed=0.52
        {"chrom": "chr1", "pos1": 400, "pos2": 450, "n1": 3, "n2": 3,
         "posterior1": 0.6, "posterior2": 0.4, "distance": 50},
        # Pair 3: same bin, distance=300, but end 2 has n=1 (not informative).
        # → contributes to NEITHER pool.
        {"chrom": "chr1", "pos1": 600, "pos2": 900, "n1": 3, "n2": 1,
         "posterior1": 0.5, "posterior2": 0.7, "distance": 300},
        # Pair 4: cross-bin (chr1:0 vs chr1:1000).  → excluded entirely.
        {"chrom": "chr1", "pos1": 500, "pos2": 1500, "n1": 5, "n2": 5,
         "posterior1": 0.5, "posterior2": 0.5, "distance": 1000},
    ]
    p = tmp_path / "cellA.posteriors.parquet"
    _make_posteriors(records).to_parquet(p)

    cell, counts = concordance._compute_cell_soft_counts(
        str(p), "cellA", bin_size=1000, min_distance=100, min_gch_sites=2,
    )
    assert cell == "cellA"
    # All counts go into chr1:0 (the only bin with qualifying pairs).
    rec = counts["chr1:0"]

    # Linked side from pair 1.
    assert rec["n_linked"] == pytest.approx(1.0)
    assert rec["k_open"] == pytest.approx(0.9 * 0.8)
    assert rec["k_closed"] == pytest.approx(0.1 * 0.2)
    assert rec["k_mixed"] == pytest.approx(0.9 * 0.2 + 0.1 * 0.8)

    # Unlinked side from pair 2.
    assert rec["n_unlinked"] == pytest.approx(1.0)
    assert rec["k_u_open"] == pytest.approx(0.6 * 0.4)
    assert rec["k_u_closed"] == pytest.approx(0.4 * 0.6)
    assert rec["k_u_mixed"] == pytest.approx(0.6 * 0.6 + 0.4 * 0.4)

    # Pairs 3 and 4 contributed nothing — only one bin in the output.
    assert set(counts.keys()) == {"chr1:0"}


def test_soft_counts_per_pair_sum_to_one(tmp_path):
    """U-3 invariant: every qualifying pair contributes 1 to n and the three
    soft-count categories sum to 1 per pair."""
    records = [
        # Multiple pairs with various posteriors.
        {"chrom": "chr1", "pos1": 100, "pos2": 200, "n1": 3, "n2": 3,
         "posterior1": 0.3, "posterior2": 0.7, "distance": 100},
        {"chrom": "chr1", "pos1": 300, "pos2": 400, "n1": 3, "n2": 3,
         "posterior1": 0.95, "posterior2": 0.95, "distance": 100},
        {"chrom": "chr1", "pos1": 500, "pos2": 600, "n1": 3, "n2": 3,
         "posterior1": 0.05, "posterior2": 0.05, "distance": 100},
        {"chrom": "chr1", "pos1": 700, "pos2": 800, "n1": 3, "n2": 3,
         "posterior1": 0.5, "posterior2": 0.5, "distance": 100},
    ]
    p = tmp_path / "cell.posteriors.parquet"
    _make_posteriors(records).to_parquet(p)

    _, counts = concordance._compute_cell_soft_counts(
        str(p), "c", bin_size=1000, min_distance=50, min_gch_sites=2,
    )
    rec = counts["chr1:0"]
    assert rec["n_linked"] == pytest.approx(4.0)
    # k_open + k_closed + k_mixed should equal n_linked.
    total = rec["k_open"] + rec["k_closed"] + rec["k_mixed"]
    assert total == pytest.approx(rec["n_linked"])


def test_soft_counts_empty_cell(tmp_path):
    """A cell with no qualifying pairs returns an empty dict."""
    df = pd.DataFrame({
        "chrom": pd.Series([], dtype=str),
        "pos1": pd.Series([], dtype=np.int64),
        "pos2": pd.Series([], dtype=np.int64),
        "n1": pd.Series([], dtype=np.int32),
        "n2": pd.Series([], dtype=np.int32),
        "distance": pd.Series([], dtype=np.int64),
        "posterior1": pd.Series([], dtype=float),
        "posterior2": pd.Series([], dtype=float),
    })
    p = tmp_path / "empty.posteriors.parquet"
    df.to_parquet(p)

    cell, counts = concordance._compute_cell_soft_counts(
        str(p), "empty", bin_size=1000, min_distance=100, min_gch_sites=2,
    )
    assert cell == "empty"
    assert counts == {}


# ---------------------------------------------------------------------------
# Dirichlet-Multinomial EB MoM fit
# ---------------------------------------------------------------------------


def test_dm_eb_prior_recovers_population_means():
    """MoM on simulated DM data recovers the true mu_open, mu_closed, mu_mixed."""
    rng = np.random.default_rng(0)
    # Simulate 200 (cell, bin) observations from Dirichlet(8, 4, 3) → mu = (8, 4, 3) / 15.
    alpha_true = np.array([8.0, 4.0, 3.0])
    n_obs = 200
    pis = rng.dirichlet(alpha_true, size=n_obs)
    ns = rng.integers(5, 50, size=n_obs)
    ks = np.array([
        rng.multinomial(n_, pi)
        for n_, pi in zip(ns, pis)
    ])

    counts_df = pd.DataFrame({
        "k_open": ks[:, 0].astype(float),
        "k_closed": ks[:, 1].astype(float),
        "k_mixed": ks[:, 2].astype(float),
        "n_linked": ns.astype(float),
    })
    alpha_hat, info = concordance._fit_dirichlet_multinomial_eb_prior(
        counts_df, min_obs=50,
    )
    mu_true = alpha_true / alpha_true.sum()
    mu_hat = np.array([
        info["mu_hat"]["open"], info["mu_hat"]["closed"], info["mu_hat"]["mixed"],
    ])
    np.testing.assert_allclose(mu_hat, mu_true, atol=0.05)
    assert not info["fallback"]
    # Sanity: alpha should sum to nu_hat.
    assert (alpha_hat["open"] + alpha_hat["closed"] + alpha_hat["mixed"]
            == pytest.approx(info["nu_hat"]))


def test_dm_eb_prior_fallback_when_sparse():
    """Below min_obs, falls back to uniform Dirichlet(1, 1, 1)."""
    counts_df = pd.DataFrame({
        "k_open":   [1.0, 2.0],
        "k_closed": [1.0, 1.0],
        "k_mixed":  [1.0, 0.0],
        "n_linked": [3.0, 3.0],
    })
    alpha, info = concordance._fit_dirichlet_multinomial_eb_prior(
        counts_df, min_obs=50,
    )
    assert alpha == {"open": 1.0, "closed": 1.0, "mixed": 1.0}
    assert info["fallback"]


# ---------------------------------------------------------------------------
# Sanity-check table from design-doc §2.3
# ---------------------------------------------------------------------------


def test_delta_signed_sanity_table_focal_active(tmp_path):
    """Bin with focal active accessibility (p̂=0.5, but pairs cluster open).

    From the design-doc §2.3 sanity table — the row we cared most about,
    where mean accessibility is moderate but 3D-contact pairs are
    disproportionately co-open.  Expected Δ_signed ≈ +0.4.
    """
    # 20 cells x 1 bin.  Each cell has many linked pairs all both-open
    # (focal active) and many unlinked pairs of mixed open/closed
    # (giving p̂ = 0.5 from per-end marginal).
    rng = np.random.default_rng(42)
    paths = {}
    for c in range(10):
        records = []
        # 20 linked pairs (long-range), both ends p=0.95 → mostly co-open.
        for i in range(20):
            records.append({
                "chrom": "chr1",
                "pos1": 100 + i * 2,
                "pos2": 500 + i * 2,
                "n1": 5, "n2": 5,
                "posterior1": 0.95, "posterior2": 0.95,
                "distance": 400,
            })
        # 20 unlinked short-range pairs with one end open and one closed
        # → soft per-end marginal p̂ ≈ 0.5.
        for i in range(20):
            records.append({
                "chrom": "chr1",
                "pos1": 700 + i * 2,
                "pos2": 720 + i * 2,
                "n1": 5, "n2": 5,
                "posterior1": 0.95, "posterior2": 0.05,
                "distance": 20,
            })
        path = tmp_path / f"c{c}.posteriors.parquet"
        _make_posteriors(records).to_parquet(path)
        paths[f"c{c}"] = str(path)

    result = concordance.compute_signed_concordance_matrix(
        paths,
        bin_size=1000, min_distance=100, min_gch_sites=2,
        min_pairs_per_bin=1,
        use_rwr=False,  # disable RWR for this synthetic test
        eb_min_obs=2,
        n_threads=1,
        output_dir=str(tmp_path / "out"),
        save_outputs=False,
    )
    delta = result["delta_signed"]
    assert "chr1:0" in delta.columns
    # All cells have the same pattern, so Δ_signed should be similar
    # across cells. Expected ≈ +0.7 (linked S = 1, baseline S_u = 0,
    # plus EB shrinkage pulling toward population mean which is also
    # high).
    cell_deltas = delta["chr1:0"].dropna()
    assert (cell_deltas > 0.3).all(), (
        f"Focal active should give strongly positive Δ_signed; "
        f"got: {cell_deltas.describe()}"
    )


def _focal_active_paths(tmp_path, n_cells=10):
    """Shared fixture: focal-active bin (linked co-open, unlinked p̂≈0.5)."""
    paths = {}
    for c in range(n_cells):
        records = []
        for i in range(20):
            records.append({
                "chrom": "chr1", "pos1": 100 + i * 2, "pos2": 500 + i * 2,
                "n1": 5, "n2": 5,
                "posterior1": 0.95, "posterior2": 0.95, "distance": 400,
            })
        for i in range(20):
            records.append({
                "chrom": "chr1", "pos1": 700 + i * 2, "pos2": 720 + i * 2,
                "n1": 5, "n2": 5,
                "posterior1": 0.95, "posterior2": 0.05, "distance": 20,
            })
        path = tmp_path / f"c{c}.posteriors.parquet"
        _make_posteriors(records).to_parquet(path)
        paths[f"c{c}"] = str(path)
    return paths


def test_concordance_formula_mle_matches_unshrunk_rate(tmp_path):
    """MLE Δ_signed on the focal-active bin equals the plug-in rate (≈0.9),
    with no shrinkage toward the population mean.

    Linked: 20 pairs both p=0.95 → S_l = (k_open−k_closed)/n ≈ 0.9.
    Unlinked: 20 pairs p=(0.95, 0.05) → S_u ≈ 0.  Δ ≈ 0.9.
    """
    paths = _focal_active_paths(tmp_path)
    result = concordance.compute_signed_concordance_matrix(
        paths, bin_size=1000, min_distance=100, min_gch_sites=2,
        min_pairs_per_bin=1, concordance_formula="mle",
        use_rwr=False, eb_min_obs=2, n_threads=1,
        output_dir=str(tmp_path / "out"), save_outputs=False,
    )
    cell_deltas = result["delta_signed"]["chr1:0"].dropna()
    # soft_open = 0.95^2 = 0.9025, soft_closed = 0.05^2 = 0.0025 per linked
    # pair → S_l = (0.9025 − 0.0025) = 0.9 exactly; S_u = 0.
    np.testing.assert_allclose(cell_deltas.to_numpy(), 0.9, atol=1e-9)


def test_concordance_formula_mle_more_extreme_than_dm(tmp_path):
    """MLE (no prior) should give a Δ_signed at least as extreme as DM
    (which shrinks toward the population mean) on the same input."""
    paths = _focal_active_paths(tmp_path)
    common = dict(
        bin_size=1000, min_distance=100, min_gch_sites=2,
        min_pairs_per_bin=1, use_rwr=False, eb_min_obs=2, n_threads=1,
        save_outputs=False,
    )
    dm = concordance.compute_signed_concordance_matrix(
        paths, concordance_formula="dm",
        output_dir=str(tmp_path / "dm"), **common,
    )["delta_signed"]["chr1:0"].dropna()
    mle = concordance.compute_signed_concordance_matrix(
        paths, concordance_formula="mle",
        output_dir=str(tmp_path / "mle"), **common,
    )["delta_signed"]["chr1:0"].dropna()
    # Both strongly positive; MLE not shrunk, so >= DM here.
    assert (mle >= dm - 1e-9).all()
    assert mle.mean() >= dm.mean()


def test_concordance_formula_invalid_raises(tmp_path):
    paths = _focal_active_paths(tmp_path, n_cells=3)
    with pytest.raises(ValueError, match="concordance_formula"):
        concordance.compute_signed_concordance_matrix(
            paths, bin_size=1000, min_distance=100, min_gch_sites=2,
            concordance_formula="nope", use_rwr=False, eb_min_obs=2,
            n_threads=1, output_dir=str(tmp_path / "x"), save_outputs=False,
        )


def test_dm_expected_for_bin_known_values():
    """Distance-matched expected = mean soft co-state of end-pairs at the
    contact's distance bucket."""
    epos = np.array([0, 1000, 2000], dtype=np.int64)
    ep = np.array([0.9, 0.9, 0.1], dtype=np.float64)
    # End-pairs: (0,1000) d=1000 po=0.81; (1000,2000) d=1000 po=0.09;
    # (0,2000) d=2000.  Bucket d=1000 → mean co-open = (0.81+0.09)/2 = 0.45.
    k_o, k_c, k_m, n, n_fb = concordance._dm_expected_for_bin(
        epos, ep, np.array([1000]), dist_bin=1000, max_ends=400, seed=0)
    assert n == 1
    assert n_fb == 0                       # the bucket was populated
    assert abs(k_o - 0.45) < 1e-9
    # co-closed at d=1000: ((1-.9)(1-.9)=0.01 + (1-.9)(1-.1)=0.09)/2 = 0.05
    assert abs(k_c - 0.05) < 1e-9


def test_compute_cell_dm_expected_returns_counts(tmp_path):
    paths = _focal_active_paths(tmp_path, n_cells=1)
    cell, counts = concordance._compute_cell_dm_expected(
        paths["c0"], "c0", bin_size=1000, min_distance=100, min_gch_sites=2,
        dm_window_bins=0, dm_dist_bin=100)
    assert "chr1:0" in counts
    rec = counts["chr1:0"]
    assert set(rec) == set(concordance._DM_KEYS) | {"n_dm_fallback"}
    assert rec["n_dm"] > 0 and rec["k_dm_open"] >= 0


def test_signed_distance_matched_baseline_log2ratio(tmp_path):
    """End-to-end: baseline='distance-matched' + log2-ratio produces a finite
    Δ matrix and fits a dm-expected prior."""
    paths = _focal_active_paths(tmp_path, n_cells=8)
    result = concordance.compute_signed_concordance_matrix(
        paths, bin_size=1000, min_distance=100, min_gch_sites=2,
        baseline="distance-matched", concordance_combine="log2-ratio",
        concordance_formula="dm", use_rwr=False, eb_min_obs=2, n_threads=1,
        output_dir=str(tmp_path / "out"), save_outputs=False,
    )
    delta = result["delta_signed"]
    assert delta.shape[0] == 8 and "chr1:0" in delta.columns
    assert np.isfinite(delta["chr1:0"].to_numpy()).all()
    assert "dm_expected" in result["priors"]


def test_pooled_expected_for_bin_known_values():
    """Distance-pooled expected = mean soft co-state of ALL null end-pairs
    with sep >= min_distance, excluding the observed contacts, scaled by the
    contact count."""
    epos = np.array([0, 1000, 2000, 3000], dtype=np.int64)
    ep = np.array([0.9, 0.8, 0.1, 0.2], dtype=np.float64)
    # Pairs with |Δ| >= 1500: (0,2000), (0,3000), (1000,3000).  Exclude the
    # observed contact (0,2000).  Remaining: (0,3000) [po=.18,pc=.08,pm=.74]
    # and (1000,3000) [po=.16,pc=.16,pm=.68].  Means: .17/.12/.71, ×2 contacts.
    k_o, k_c, k_m, n = concordance._pooled_expected_for_bin(
        epos, ep, contact_pairs=[(0, 2000)], n_contacts=2,
        min_distance=1500, max_ends=400, seed=0)
    assert n == 2
    assert abs(k_o - 0.34) < 1e-9
    assert abs(k_c - 0.24) < 1e-9
    assert abs(k_m - 1.42) < 1e-9
    # Rate (k/n) sums to 1 across the three co-states.
    assert abs((k_o + k_c + k_m) / n - 1.0) < 1e-9


def test_pooled_expected_for_bin_excludes_short_and_contacts():
    """No pair survives the min_distance filter → empty expected."""
    epos = np.array([0, 100, 200], dtype=np.int64)
    ep = np.array([0.9, 0.8, 0.1], dtype=np.float64)
    k_o, k_c, k_m, n = concordance._pooled_expected_for_bin(
        epos, ep, contact_pairs=[], n_contacts=3,
        min_distance=1000, max_ends=400, seed=0)
    assert (k_o, k_c, k_m, n) == (0.0, 0.0, 0.0, 0)


def test_compute_cell_dm_expected_pooled_mode(tmp_path):
    paths = _focal_active_paths(tmp_path, n_cells=1)
    cell, counts = concordance._compute_cell_dm_expected(
        paths["c0"], "c0", bin_size=1000, min_distance=100, min_gch_sites=2,
        dm_window_bins=0, dm_dist_bin=100, dm_max_ends=400, mode="pooled")
    assert "chr1:0" in counts
    rec = counts["chr1:0"]
    assert set(rec) == set(concordance._DM_KEYS) | {"n_dm_fallback"}
    assert rec["n_dm_fallback"] == 0       # pooled mode never falls back
    # 20 long-range contacts in bin chr1:0 → n_dm scaled to the contact count.
    assert rec["n_dm"] == 20
    assert rec["k_dm_open"] >= 0


def test_signed_distance_pooled_baseline_log2ratio(tmp_path):
    """End-to-end: baseline='distance-pooled' + log2-ratio produces a finite
    Δ matrix and fits an expected-pool prior."""
    paths = _focal_active_paths(tmp_path, n_cells=8)
    result = concordance.compute_signed_concordance_matrix(
        paths, bin_size=1000, min_distance=100, min_gch_sites=2,
        baseline="distance-pooled", concordance_combine="log2-ratio",
        concordance_formula="dm", use_rwr=False, eb_min_obs=2, n_threads=1,
        output_dir=str(tmp_path / "out"), save_outputs=False,
    )
    delta = result["delta_signed"]
    assert delta.shape[0] == 8 and "chr1:0" in delta.columns
    assert np.isfinite(delta["chr1:0"].to_numpy()).all()
    assert "dm_expected" in result["priors"]


def test_baseline_unlinked_alias_equals_short_range(tmp_path):
    """'unlinked' is a deprecated alias for 'short-range' — same result."""
    paths = _focal_active_paths(tmp_path, n_cells=5)
    common = dict(bin_size=1000, min_distance=100, min_gch_sites=2,
                  concordance_formula="mle", use_rwr=False, eb_min_obs=2,
                  n_threads=1, save_outputs=False)
    a = concordance.compute_signed_concordance_matrix(
        paths, baseline="short-range", output_dir=str(tmp_path / "sr"),
        **common)["delta_signed"]
    b = concordance.compute_signed_concordance_matrix(
        paths, baseline="unlinked", output_dir=str(tmp_path / "ul"),
        **common)["delta_signed"]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def _both_open_paths(tmp_path, n_cells=8):
    """Fixture where BOTH the linked and the short-range pools are co-open, so
    the baseline (E) carries the same signal as the observed (O)."""
    paths = {}
    for c in range(n_cells):
        recs = []
        for i in range(20):        # linked long-range: both ends open
            recs.append({"chrom": "chr1", "pos1": 100 + i * 2,
                         "pos2": 500 + i * 2, "n1": 5, "n2": 5,
                         "posterior1": 0.95, "posterior2": 0.95,
                         "distance": 400})
        for i in range(20):        # short-range baseline: ALSO both ends open
            recs.append({"chrom": "chr1", "pos1": 700 + i * 2,
                         "pos2": 720 + i * 2, "n1": 5, "n2": 5,
                         "posterior1": 0.95, "posterior2": 0.95,
                         "distance": 20})
        p = tmp_path / f"c{c}.parquet"
        _make_posteriors(recs).to_parquet(p)
        paths[f"c{c}"] = str(p)
    return paths


def test_combine_linked_only_skips_the_baseline(tmp_path):
    """'linked-only' uses O_open-O_closed with NO baseline subtracted.

    With the short-range pool ALSO co-open, 'signed-diff' cancels to ~0 (the
    baseline removes the marginal), while 'linked-only' keeps the raw co-open
    signal — the diagnostic that shows how much the baseline is doing.
    """
    paths = _both_open_paths(tmp_path)
    common = dict(bin_size=1000, min_distance=100, min_gch_sites=2,
                  min_pairs_per_bin=1, use_rwr=False, eb_min_obs=2,
                  n_threads=1, save_outputs=False)
    sd = concordance.compute_signed_concordance_matrix(
        paths, concordance_combine="signed-diff",
        output_dir=str(tmp_path / "sd"), **common)["delta_signed"]["chr1:0"]
    lo = concordance.compute_signed_concordance_matrix(
        paths, concordance_combine="linked-only",
        output_dir=str(tmp_path / "lo"), **common)["delta_signed"]["chr1:0"]
    assert sd.abs().mean() < 0.15, "baseline should cancel the co-open marginal"
    assert lo.mean() > 0.7, "linked-only keeps the raw co-open signal"
    assert (lo > sd + 0.5).all(), "linked-only must exceed the baselined Δ"


def test_combine_concordant_diff_sees_coordination_signed_diff_misses(tmp_path):
    """concordant-diff captures balanced-marginal concordance; signed-diff can't.

    Linked pairs are strongly CONCORDANT (half co-open, half co-closed) at a
    ~50% mean level, while the short-range baseline is all MIXED.  The
    (open − closed) contrast of 'signed-diff' cancels to ~0 (blind to the
    coordination), whereas 'concordant-diff' = (open + closed) excess is
    strongly positive.
    """
    paths = {}
    for c in range(8):
        recs = []
        for i in range(20):        # linked, both-open
            recs.append({"chrom": "chr1", "pos1": 100 + i * 2,
                         "pos2": 500 + i * 2, "n1": 5, "n2": 5,
                         "posterior1": 0.95, "posterior2": 0.95,
                         "distance": 400})
        for i in range(20):        # linked, both-closed  -> mean level ~0.5
            recs.append({"chrom": "chr1", "pos1": 141 + i * 2,
                         "pos2": 541 + i * 2, "n1": 5, "n2": 5,
                         "posterior1": 0.05, "posterior2": 0.05,
                         "distance": 400})
        for i in range(40):        # short-range baseline, all MIXED
            recs.append({"chrom": "chr1", "pos1": 700 + i * 2,
                         "pos2": 720 + i * 2, "n1": 5, "n2": 5,
                         "posterior1": 0.95, "posterior2": 0.05,
                         "distance": 20})
        p = tmp_path / f"c{c}.parquet"
        _make_posteriors(recs).to_parquet(p)
        paths[f"c{c}"] = str(p)
    common = dict(bin_size=1000, min_distance=100, min_gch_sites=2,
                  min_pairs_per_bin=1, baseline="short-range", use_rwr=False,
                  eb_min_obs=2, n_threads=1, save_outputs=False)
    sd = concordance.compute_signed_concordance_matrix(
        paths, concordance_combine="signed-diff",
        output_dir=str(tmp_path / "sd"), **common)["delta_signed"]["chr1:0"]
    cd = concordance.compute_signed_concordance_matrix(
        paths, concordance_combine="concordant-diff",
        output_dir=str(tmp_path / "cd"), **common)["delta_signed"]["chr1:0"]
    assert sd.abs().mean() < 0.2, "signed-diff is blind to balanced concordance"
    assert cd.mean() > 0.4, "concordant-diff must detect the concordance"


_HI, _LO = 0.95, 0.05
_MARGINAL = {"open": 0.75, "closed": 0.25, "balanced": 0.50}


def _state_block(m: float, coordinated: bool) -> list[tuple[float, float]]:
    """16 (p1, p2) end-states with per-end marginal ``m``.

    ``coordinated``  -> all mass on the concordant cells (16m co-open,
                        16(1−m) co-closed): maximal association.
    ``uncoordinated``-> the INDEPENDENT product (m², (1−m)², 2m(1−m)):
                        identical marginal, zero association.

    Holding the marginal fixed across the two is what makes the pair a clean
    coordination contrast — the marginal is the other degree of freedom.
    """
    n = 16
    if coordinated:
        n_oo, n_cc, n_mix = round(n * m), round(n * (1 - m)), 0
    else:
        n_oo, n_cc = round(n * m * m), round(n * (1 - m) * (1 - m))
        n_mix = (n - n_oo - n_cc) // 2
    return ([(_HI, _HI)] * n_oo + [(_LO, _LO)] * n_cc
            + [(_HI, _LO)] * n_mix + [(_LO, _HI)] * n_mix)


def _directional_paths(tmp_path, level: str, coordinated: bool, n_cells=8):
    """Linked pairs at openness ``level``, coordinated or not, against a
    short-range baseline that is the INDEPENDENT pairing at the SAME marginal.

    Using the independent product as the baseline (rather than, say, an
    all-mixed pool, which is itself strongly anti-concordant) means E carries
    the marginal but no association — so C isolates the coordination and an
    uncoordinated linked pool correctly scores ~0.
    """
    m = _MARGINAL[level]
    linked_states = _state_block(m, coordinated)
    base_states = _state_block(m, coordinated=False)
    paths = {}
    for c in range(n_cells):
        recs = []
        for i in range(32):
            p1, p2 = linked_states[i % len(linked_states)]
            recs.append({"chrom": "chr1", "pos1": 100 + i * 2,
                         "pos2": 500 + i * 2, "n1": 5, "n2": 5,
                         "posterior1": p1, "posterior2": p2, "distance": 400})
        for i in range(32):
            p1, p2 = base_states[i % len(base_states)]
            recs.append({"chrom": "chr1", "pos1": 700 + i * 2,
                         "pos2": 720 + i * 2, "n1": 5, "n2": 5,
                         "posterior1": p1, "posterior2": p2, "distance": 20})
        p = tmp_path / f"{level}_{coordinated}_c{c}.parquet"
        _make_posteriors(recs).to_parquet(p)
        paths[f"c{c}"] = str(p)
    return paths


def _run_directional(tmp_path, level, coordinated, tag):
    return concordance.compute_signed_concordance_matrix(
        _directional_paths(tmp_path, level, coordinated),
        bin_size=1000, min_distance=100, min_gch_sites=2, min_pairs_per_bin=1,
        baseline="short-range", concordance_combine="concordant-signed",
        use_rwr=False, eb_min_obs=2, n_threads=1, save_outputs=False,
        output_dir=str(tmp_path / tag),
    )


def test_concordant_signed_orients_co_open_positive_co_closed_negative(tmp_path):
    """D = C·(2F−1) gives the co-closed → mixed → co-open scale.

    Coordinated-open must be POSITIVE, coordinated-closed NEGATIVE, and an
    uncoordinated bin at the same marginal ~0 — the property neither
    'signed-diff' (marginal only) nor 'concordant-diff' (direction-blind) has.
    """
    d_open = _run_directional(tmp_path, "open", True, "o")["delta_signed"]["chr1:0"]
    d_closed = _run_directional(tmp_path, "closed", True, "c")["delta_signed"]["chr1:0"]
    d_none = _run_directional(tmp_path, "open", False, "n")["delta_signed"]["chr1:0"]

    # Reference values for this fixture: D = +0.151 / -0.151 / 0.000.
    assert d_open.mean() > 0.08, "coordinated co-open must be clearly positive"
    assert d_closed.mean() < -0.08, "coordinated co-closed must be clearly negative"
    assert abs(d_none.mean()) < 0.02, "uncoordinated bin must sit near zero"


def test_concordant_signed_magnitude_is_direction_symmetric(tmp_path):
    """|D| for co-open and co-closed coordination of equal strength matches,
    and both carry the SAME coordination magnitude C — the direction lives in
    F, not in the excess (which is identical for open and closed)."""
    r_open = _run_directional(tmp_path, "open", True, "o")
    r_closed = _run_directional(tmp_path, "closed", True, "c")
    d_o = r_open["delta_signed"]["chr1:0"].mean()
    d_c = r_closed["delta_signed"]["chr1:0"].mean()
    c_o = r_open["coordination"]["chr1:0"].mean()
    c_c = r_closed["coordination"]["chr1:0"].mean()
    assert d_o == pytest.approx(-d_c, abs=0.05)
    assert c_o == pytest.approx(c_c, abs=0.05), "C must be direction-blind"
    assert c_o > 0.15, "both must register real coordination (reference 0.304)"


def test_components_separate_balanced_coordination_from_mixed(tmp_path):
    """The documented D≈0 degeneracy is resolvable by reading C alongside D.

    A coordinated-but-BALANCED bin (half co-open, half co-closed) and a truly
    mixed bin both give D≈0, but only the balanced one has large C.
    """
    bal = _run_directional(tmp_path, "balanced", True, "b")
    mix = _run_directional(tmp_path, "balanced", False, "m")
    assert abs(bal["delta_signed"]["chr1:0"].mean()) < 0.05
    assert abs(mix["delta_signed"]["chr1:0"].mean()) < 0.05
    assert bal["coordination"]["chr1:0"].mean() > 0.25, \
        "balanced: C is large (reference 0.405)"
    assert abs(mix["coordination"]["chr1:0"].mean()) < 0.05, "mixed: C ~ 0"


def test_components_emitted_for_every_combine_mode(tmp_path):
    """C / F / D / p̄ are companion outputs, not tied to the headline combine,
    and the marginal tracks openness while C stays blind to it."""
    res = concordance.compute_signed_concordance_matrix(
        _directional_paths(tmp_path, "open", True),
        bin_size=1000, min_distance=100, min_gch_sites=2, min_pairs_per_bin=1,
        baseline="short-range", concordance_combine="signed-diff",
        use_rwr=False, eb_min_obs=2, n_threads=1, save_outputs=False,
        output_dir=str(tmp_path / "sd"),
    )
    for key in ("coordination", "direction", "directional", "marginal"):
        assert key in res, key
        assert res[key].shape == res["delta_signed"].shape
    # D == C·(2F−1) by construction.
    c = res["coordination"]["chr1:0"]
    f = res["direction"]["chr1:0"]
    d = res["directional"]["chr1:0"]
    assert d.to_numpy() == pytest.approx((c * (2 * f - 1)).to_numpy(), abs=1e-12)
    # 3:1 both-open:both-closed -> marginal ~0.75, direction F ~0.75.
    assert res["marginal"]["chr1:0"].mean() == pytest.approx(0.75, abs=0.08)
    assert f.mean() == pytest.approx(0.75, abs=0.08)


def test_components_written_to_disk(tmp_path):
    out = tmp_path / "out"
    concordance.compute_signed_concordance_matrix(
        _directional_paths(tmp_path, "open", True, n_cells=4),
        bin_size=1000, min_distance=100, min_gch_sites=2, min_pairs_per_bin=1,
        baseline="short-range", concordance_combine="concordant-signed",
        use_rwr=False, eb_min_obs=2, n_threads=1, save_outputs=True,
        output_dir=str(out),
    )
    for name in ("coordination", "direction", "directional", "marginal"):
        assert (out / f"concordance_{name}.parquet").exists(), name
        assert (out / f"concordance_{name}.tsv").exists(), name


def test_baseline_and_combine_invalid_raise(tmp_path):
    paths = _focal_active_paths(tmp_path, n_cells=3)
    with pytest.raises(ValueError, match="baseline"):
        concordance.compute_signed_concordance_matrix(
            paths, bin_size=1000, min_distance=100, min_gch_sites=2,
            baseline="nope", use_rwr=False, eb_min_obs=2, n_threads=1,
            output_dir=str(tmp_path / "a"), save_outputs=False)
    with pytest.raises(ValueError, match="concordance_combine"):
        concordance.compute_signed_concordance_matrix(
            paths, bin_size=1000, min_distance=100, min_gch_sites=2,
            concordance_combine="nope", use_rwr=False, eb_min_obs=2,
            n_threads=1, output_dir=str(tmp_path / "b"), save_outputs=False)


def test_min_cell_pairs_drops_sparse_cells(tmp_path):
    """Pre-imputation cell QC drops cells below the informative-pair floor."""
    paths = dict(_focal_active_paths(tmp_path, n_cells=4))  # 40 pairs each
    # One near-empty cell with a single informative pair.
    sparse = [{
        "chrom": "chr1", "pos1": 100, "pos2": 500, "n1": 5, "n2": 5,
        "posterior1": 0.9, "posterior2": 0.9, "distance": 400,
    }]
    sp = tmp_path / "sparse.posteriors.parquet"
    _make_posteriors(sparse).to_parquet(sp)
    paths["sparse"] = str(sp)

    result = concordance.compute_signed_concordance_matrix(
        paths, bin_size=1000, min_distance=100, min_gch_sites=2,
        min_cell_pairs=5, concordance_formula="mle",
        use_rwr=False, eb_min_obs=2, n_threads=1,
        output_dir=str(tmp_path / "out"), save_outputs=False,
    )
    cells = list(result["delta_signed"].index)
    assert "sparse" not in cells
    assert len(cells) == 4


def test_min_bin_coverage_drops_sparse_bins(tmp_path):
    """Pre-imputation bin QC drops bins seen in too few cells."""
    paths = {}
    for c in range(4):
        records = []
        # All 4 cells share a dense bin around chr1:0.
        for i in range(20):
            records.append({
                "chrom": "chr1", "pos1": 100 + i * 2, "pos2": 500 + i * 2,
                "n1": 5, "n2": 5, "posterior1": 0.9, "posterior2": 0.9,
                "distance": 400,
            })
        # Only cell 0 has any pair in a far bin (chr1:1,000,000).
        if c == 0:
            records.append({
                "chrom": "chr1", "pos1": 1_000_100, "pos2": 1_000_600,
                "n1": 5, "n2": 5, "posterior1": 0.9, "posterior2": 0.9,
                "distance": 500,
            })
        p = tmp_path / f"c{c}.posteriors.parquet"
        _make_posteriors(records).to_parquet(p)
        paths[f"c{c}"] = str(p)

    result = concordance.compute_signed_concordance_matrix(
        paths, bin_size=1000, min_distance=100, min_gch_sites=2,
        min_bin_coverage=0.5, concordance_formula="mle",
        use_rwr=False, eb_min_obs=2, n_threads=1,
        output_dir=str(tmp_path / "out"), save_outputs=False,
    )
    # The far bin (present in 1/4 = 25% < 50%) is dropped; the shared bin stays.
    bins = list(result["delta_signed"].columns)
    assert "chr1:1000000" not in bins
    assert "chr1:0" in bins


def test_delta_signed_silenced_bin_gives_negative(tmp_path):
    """Bin with most pairs co-closed → Δ_signed < 0."""
    paths = {}
    for c in range(10):
        records = []
        # 20 linked pairs, both ends p=0.05 → mostly co-closed.
        for i in range(20):
            records.append({
                "chrom": "chr1",
                "pos1": 100 + i * 2,
                "pos2": 500 + i * 2,
                "n1": 5, "n2": 5,
                "posterior1": 0.05, "posterior2": 0.05,
                "distance": 400,
            })
        # 20 unlinked short-range, mixed.
        for i in range(20):
            records.append({
                "chrom": "chr1",
                "pos1": 700 + i * 2,
                "pos2": 720 + i * 2,
                "n1": 5, "n2": 5,
                "posterior1": 0.5, "posterior2": 0.5,
                "distance": 20,
            })
        path = tmp_path / f"c{c}.posteriors.parquet"
        _make_posteriors(records).to_parquet(path)
        paths[f"c{c}"] = str(path)

    result = concordance.compute_signed_concordance_matrix(
        paths,
        bin_size=1000, min_distance=100, min_gch_sites=2,
        min_pairs_per_bin=1,
        use_rwr=False,
        eb_min_obs=2,
        n_threads=1,
        output_dir=str(tmp_path / "out"),
        save_outputs=False,
    )
    delta = result["delta_signed"]
    cell_deltas = delta["chr1:0"].dropna()
    assert (cell_deltas < -0.3).all(), (
        f"Silenced bin should give strongly negative Δ_signed; "
        f"got: {cell_deltas.describe()}"
    )


def test_delta_signed_chaotic_bin_near_zero(tmp_path):
    """Bin where linked + unlinked pools both have S ≈ 0 → Δ_signed ≈ 0."""
    paths = {}
    for c in range(10):
        records = []
        # 20 linked pairs: random mixing — half (0.95, 0.05), half (0.05, 0.95).
        # → S_linked ≈ 0.
        for i in range(20):
            if i % 2 == 0:
                p1, p2 = 0.95, 0.05
            else:
                p1, p2 = 0.05, 0.95
            records.append({
                "chrom": "chr1",
                "pos1": 100 + i * 2,
                "pos2": 500 + i * 2,
                "n1": 5, "n2": 5,
                "posterior1": p1, "posterior2": p2,
                "distance": 400,
            })
        # 20 unlinked: same pattern (also mixed).
        for i in range(20):
            if i % 2 == 0:
                p1, p2 = 0.95, 0.05
            else:
                p1, p2 = 0.05, 0.95
            records.append({
                "chrom": "chr1",
                "pos1": 700 + i * 2,
                "pos2": 720 + i * 2,
                "n1": 5, "n2": 5,
                "posterior1": p1, "posterior2": p2,
                "distance": 20,
            })
        path = tmp_path / f"c{c}.posteriors.parquet"
        _make_posteriors(records).to_parquet(path)
        paths[f"c{c}"] = str(path)

    result = concordance.compute_signed_concordance_matrix(
        paths,
        bin_size=1000, min_distance=100, min_gch_sites=2,
        min_pairs_per_bin=1,
        use_rwr=False,
        eb_min_obs=2,
        n_threads=1,
        output_dir=str(tmp_path / "out"),
        save_outputs=False,
    )
    delta = result["delta_signed"]
    cell_deltas = delta["chr1:0"].dropna()
    assert cell_deltas.abs().max() < 0.2, (
        f"Chaotic bin should give Δ_signed near 0; "
        f"got: {cell_deltas.describe()}"
    )


# ---------------------------------------------------------------------------
# End-to-end with RWR
# ---------------------------------------------------------------------------


def test_end_to_end_with_rwr(tmp_path):
    """Two cell types with different bin coordination — Δ_signed
    differentiates them after RWR."""
    rng = np.random.default_rng(7)
    paths = {}

    # Cell type A: 10 cells where chr1:0 is co-open coordinated.
    for c in range(10):
        records = []
        for i in range(10):  # 10 linked pairs per cell
            records.append({
                "chrom": "chr1",
                "pos1": 100 + i * 5,
                "pos2": 500 + i * 5,
                "n1": 5, "n2": 5,
                "posterior1": 0.95, "posterior2": 0.95,
                "distance": 400,
            })
        # 10 unlinked short-range mixed.
        for i in range(10):
            records.append({
                "chrom": "chr1",
                "pos1": 700 + i * 5,
                "pos2": 720 + i * 5,
                "n1": 5, "n2": 5,
                "posterior1": 0.5, "posterior2": 0.5,
                "distance": 20,
            })
        path = tmp_path / f"typeA_c{c}.posteriors.parquet"
        _make_posteriors(records).to_parquet(path)
        paths[f"typeA_c{c}"] = str(path)

    # Cell type B: 10 cells where chr1:0 is co-closed coordinated.
    for c in range(10):
        records = []
        for i in range(10):
            records.append({
                "chrom": "chr1",
                "pos1": 100 + i * 5,
                "pos2": 500 + i * 5,
                "n1": 5, "n2": 5,
                "posterior1": 0.05, "posterior2": 0.05,
                "distance": 400,
            })
        for i in range(10):
            records.append({
                "chrom": "chr1",
                "pos1": 700 + i * 5,
                "pos2": 720 + i * 5,
                "n1": 5, "n2": 5,
                "posterior1": 0.5, "posterior2": 0.5,
                "distance": 20,
            })
        path = tmp_path / f"typeB_c{c}.posteriors.parquet"
        _make_posteriors(records).to_parquet(path)
        paths[f"typeB_c{c}"] = str(path)

    result = concordance.compute_signed_concordance_matrix(
        paths,
        bin_size=1000, min_distance=100, min_gch_sites=2,
        min_pairs_per_bin=1,
        use_rwr=True, rwr_k=5,
        eb_min_obs=5,
        n_threads=1,
        output_dir=str(tmp_path / "out"),
        save_outputs=True,
    )
    delta = result["delta_signed"]
    type_a_cells = [c for c in delta.index if c.startswith("typeA_")]
    type_b_cells = [c for c in delta.index if c.startswith("typeB_")]
    type_a_mean = float(delta.loc[type_a_cells, "chr1:0"].mean())
    type_b_mean = float(delta.loc[type_b_cells, "chr1:0"].mean())

    assert type_a_mean > type_b_mean, (
        f"Type A (co-open) should have higher Δ_signed than type B (co-closed); "
        f"got A={type_a_mean:.3f}, B={type_b_mean:.3f}"
    )
    assert type_a_mean > 0, f"Type A mean should be positive; got {type_a_mean}"
    assert type_b_mean < 0, f"Type B mean should be negative; got {type_b_mean}"

    # Output files saved.
    out_dir = tmp_path / "out"
    assert (out_dir / "concordance_signed.parquet").exists()
    assert (out_dir / "pair_counts_per_bin.parquet").exists()
    assert (out_dir / "accessibility_mean.parquet").exists()
    assert (out_dir / "dirichlet_multinomial_prior.json").exists()
    # RWR artifacts.
    assert (out_dir / "cell_knn_graph_pass0.npz").exists()
    assert (out_dir / "rwr_diagnostics.json").exists()


def test_both_statistics_in_one_run(tmp_path):
    """statistic='both' produces both delta_signed and delta_open matrices."""
    rng = np.random.default_rng(8)
    paths = {}
    for c in range(5):
        records = []
        for i in range(10):
            records.append({
                "chrom": "chr1",
                "pos1": 100 + i * 5,
                "pos2": 500 + i * 5,
                "n1": 5, "n2": 5,
                "posterior1": 0.9, "posterior2": 0.9,
                "distance": 400,
            })
        for i in range(10):
            records.append({
                "chrom": "chr1",
                "pos1": 700 + i * 5,
                "pos2": 720 + i * 5,
                "n1": 5, "n2": 5,
                "posterior1": 0.5, "posterior2": 0.5,
                "distance": 20,
            })
        path = tmp_path / f"c{c}.posteriors.parquet"
        _make_posteriors(records).to_parquet(path)
        paths[f"c{c}"] = str(path)

    result = concordance.compute_signed_concordance_matrix(
        paths,
        bin_size=1000, min_distance=100, min_gch_sites=2,
        min_pairs_per_bin=1, statistic="both",
        use_rwr=False, eb_min_obs=2,
        n_threads=1,
        output_dir=str(tmp_path / "out"),
        save_outputs=True,
    )
    assert result["delta_signed"] is not None
    assert result["delta_open"] is not None
    # Both should peak at chr1:0.
    assert "chr1:0" in result["delta_signed"].columns
    assert "chr1:0" in result["delta_open"].columns
    # Files for both should exist.
    out_dir = tmp_path / "out"
    assert (out_dir / "concordance_signed.parquet").exists()
    assert (out_dir / "concordance_open.parquet").exists()


def test_invalid_statistic_raises(tmp_path):
    """statistic must be one of signed / open / both."""
    df = pd.DataFrame({
        "chrom": pd.Series([], dtype=str), "pos1": pd.Series([], dtype=np.int64),
        "pos2": pd.Series([], dtype=np.int64), "n1": pd.Series([], dtype=np.int32),
        "n2": pd.Series([], dtype=np.int32), "distance": pd.Series([], dtype=np.int64),
        "posterior1": pd.Series([], dtype=float), "posterior2": pd.Series([], dtype=float),
    })
    p = tmp_path / "empty.posteriors.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError, match="statistic"):
        concordance.compute_signed_concordance_matrix(
            {"x": str(p)}, statistic="invalid",
            save_outputs=False, output_dir=str(tmp_path / "out"),
        )


def test_max_distance_caps_linked_pool(tmp_path):
    """--max-distance restricts the linked pool to a band; short-range pool
    (< min_distance) is unaffected; pairs beyond max fall into NEITHER."""
    recs = []
    for i in range(3):                       # in-band linked (d=400)
        recs.append(dict(chrom="chr1", pos1=100 + i, pos2=500 + i, n1=5, n2=5,
                         posterior1=0.9, posterior2=0.9, distance=400))
    for i in range(2):                       # beyond-band (d=800)
        recs.append(dict(chrom="chr1", pos1=100 + i, pos2=900 + i, n1=5, n2=5,
                         posterior1=0.9, posterior2=0.9, distance=800))
    for i in range(2):                       # short-range (d=50)
        recs.append(dict(chrom="chr1", pos1=100 + i, pos2=150 + i, n1=5, n2=5,
                         posterior1=0.9, posterior2=0.1, distance=50))
    p = tmp_path / "c.posteriors.parquet"
    _make_posteriors(recs).to_parquet(p)
    _, c0 = concordance._compute_cell_soft_counts(str(p), "c", 1000, 100, 2, 0)
    assert c0["chr1:0"]["n_linked"] == 5     # no cap: 3 + 2
    assert c0["chr1:0"]["n_unlinked"] == 2
    _, c1 = concordance._compute_cell_soft_counts(str(p), "c", 1000, 100, 2, 500)
    assert c1["chr1:0"]["n_linked"] == 3     # cap 500: d=800 dropped
    assert c1["chr1:0"]["n_unlinked"] == 2   # short-range unchanged


def test_an_empty_distance_bucket_is_counted_as_a_fallback():
    """A contact whose bucket holds no re-paired ends is scored against the
    distance-POOLED mean -- the baseline distance matching was chosen
    instead of.  A run that does that for most of its long contacts is
    distance-matched in name only, so the count has to be visible."""
    epos = np.array([0, 1000, 2000], dtype=np.int64)
    ep = np.array([0.9, 0.9, 0.1], dtype=np.float64)
    # Contacts at 1000 (a populated bucket) and at 500_000 (empty).
    k_o, _k_c, _k_m, n, n_fb = concordance._dm_expected_for_bin(
        epos, ep, np.array([1000, 500_000]), dist_bin=1000, max_ends=400,
        seed=0)
    assert n == 2 and n_fb == 1
    # the fallen-back contact got the pooled mean over all three end-pairs
    pooled = (0.81 + 0.09 + 0.09) / 3
    assert abs(k_o - (0.45 + pooled)) < 1e-9


def test_the_fallback_count_never_reaches_the_imputed_matrices():
    """It is a diagnostic, not a count.  Sending it through RWR would make
    it a smoothed quantity and put a meaningless extra channel in the
    imputation."""
    assert "n_dm_fallback" not in concordance._DM_KEYS
    assert "n_dm_fallback" not in concordance._SOFT_COUNT_KEYS


def test_the_run_reports_how_much_of_the_matching_actually_happened(
        tmp_path, caplog):
    import logging
    paths = _focal_active_paths(tmp_path, n_cells=2)
    with caplog.at_level(logging.INFO,
                         logger="nome_hic_concordance.concordance"):
        concordance.compute_signed_concordance_matrix(
            paths, bin_size=1000, min_distance=100, min_gch_sites=2,
            baseline="distance-matched", dm_dist_bin=100,
            save_outputs=False, use_rwr=False,
        )
    assert any("Distance-matched null:" in r.message and "fell back"
               in r.message for r in caplog.records)
