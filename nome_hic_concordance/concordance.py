"""Per-bin concordance: Pearson r of paired read-end posteriors.

For each (cell, bin) with at least ``min_pairs_per_bin`` qualifying pairs,
we compute the Pearson correlation of the two posterior columns. Bins are
assigned using the midpoint of the two read ends. Pairs below the distance
or GCH-site thresholds are discarded.
"""

from __future__ import annotations

import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def _compute_cell_concordance(
    posteriors_path: str,
    cell_name: str,
    bin_size: int,
    min_distance: int,
    min_gch_sites: int,
    min_pairs_per_bin: int,
) -> tuple[str, dict[str, float], dict[str, int]]:
    """Compute per-bin concordance and pair counts for one cell."""
    df = pd.read_parquet(posteriors_path)

    # Required columns:
    required = {"chrom", "pos1", "pos2", "n1", "n2", "distance", "posterior1", "posterior2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Posteriors file for {cell_name} is missing columns {sorted(missing)}"
        )

    df = df[
        (df["distance"] >= min_distance)
        & (df["n1"] >= min_gch_sites)
        & (df["n2"] >= min_gch_sites)
        & df["posterior1"].notna()
        & df["posterior2"].notna()
    ]

    if len(df) == 0:
        return cell_name, {}, {}

    midpoint = (df["pos1"].astype(np.int64) + df["pos2"].astype(np.int64)) // 2
    bin_start = (midpoint // int(bin_size)) * int(bin_size)
    bin_id = df["chrom"].astype(str) + ":" + bin_start.astype(str)

    df = df.assign(bin_id=bin_id.values)

    concordance: dict[str, float] = {}
    pair_counts: dict[str, int] = {}

    for bid, group in df.groupby("bin_id", sort=False):
        n_pairs = len(group)
        pair_counts[bid] = int(n_pairs)

        if n_pairs < min_pairs_per_bin:
            concordance[bid] = np.nan
            continue

        p1 = group["posterior1"].to_numpy(dtype=float)
        p2 = group["posterior2"].to_numpy(dtype=float)

        if np.std(p1) < 1e-10 or np.std(p2) < 1e-10:
            concordance[bid] = np.nan
            continue

        r = float(np.corrcoef(p1, p2)[0, 1])
        concordance[bid] = r

    return cell_name, concordance, pair_counts


def compute_concordance_matrix(
    cell_posterior_paths: dict[str, str],
    chrom_sizes: dict[str, int] | None = None,
    bin_size: int = 5_000_000,
    min_distance: int = 20_000,
    min_gch_sites: int = 3,
    min_pairs_per_bin: int = 5,
    n_threads: int = 4,
    output_dir: str = "output/05_concordance",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute a (cells x bins) matrix of Pearson concordance values.

    Parameters
    ----------
    cell_posterior_paths : dict
        Mapping ``cell_name -> posteriors.parquet``.
    chrom_sizes : dict, optional
        Mapping ``chrom -> length`` used for sorting bin columns. If None,
        bins are sorted lexicographically by chromosome, then by start.
    bin_size : int
        Genomic bin size in bp.
    min_distance, min_gch_sites, min_pairs_per_bin : int
        Filtering thresholds (see design doc).
    n_threads : int
    output_dir : str

    Returns
    -------
    (concordance_df, pair_counts_df) : both DataFrames indexed by cell,
        columns are bin identifiers like ``"chr1:0"``.
    """
    from joblib import Parallel, delayed

    out_path = ensure_dir(output_dir)

    items = list(cell_posterior_paths.items())
    if n_threads > 1 and len(items) > 1:
        results = Parallel(n_jobs=n_threads)(
            delayed(_compute_cell_concordance)(
                path, cell_name, bin_size, min_distance, min_gch_sites, min_pairs_per_bin,
            )
            for cell_name, path in items
        )
    else:
        results = [
            _compute_cell_concordance(
                path, cell_name, bin_size, min_distance, min_gch_sites, min_pairs_per_bin,
            )
            for cell_name, path in items
        ]

    all_bins: set[str] = set()
    for _, conc, _ in results:
        all_bins.update(conc.keys())

    sorted_bins = _sort_bins(all_bins, chrom_sizes)

    concordance_data: dict[str, dict[str, float]] = {}
    pair_count_data: dict[str, dict[str, int]] = {}
    for cell_name, conc, counts in results:
        concordance_data[cell_name] = {b: conc.get(b, np.nan) for b in sorted_bins}
        pair_count_data[cell_name] = {b: counts.get(b, 0) for b in sorted_bins}

    concordance_df = pd.DataFrame.from_dict(concordance_data, orient="index")
    pair_counts_df = pd.DataFrame.from_dict(pair_count_data, orient="index")

    # Preserve column order
    concordance_df = concordance_df.reindex(columns=sorted_bins)
    pair_counts_df = pair_counts_df.reindex(columns=sorted_bins).fillna(0).astype(np.int64)

    concordance_df.columns.name = "bin"
    concordance_df.index.name = "cell"
    pair_counts_df.columns.name = "bin"
    pair_counts_df.index.name = "cell"

    concordance_df.to_parquet(out_path / "concordance_raw.parquet")
    pair_counts_df.to_parquet(out_path / "pair_counts_per_bin.parquet")
    concordance_df.to_csv(out_path / "concordance_raw.tsv", sep="\t")
    pair_counts_df.to_csv(out_path / "pair_counts_per_bin.tsv", sep="\t")

    logger.info(
        f"Concordance matrix: {concordance_df.shape[0]} cells x {concordance_df.shape[1]} bins"
    )
    return concordance_df, pair_counts_df


# ---------------------------------------------------------------------------
# Beta-Binomial bin-level concordance
# ---------------------------------------------------------------------------
#
# Alternative to the Pearson r approach.  For each (cell, bin):
#
#   LINKED   : read pairs with both ends in the bin AND distance >= min_distance
#              AND both ends informative (n >= min_gch_sites).  Binarize each
#              end at posterior_threshold; count pairs where BOTH ends are open.
#   UNLINKED : any informative read end in the bin that is NOT part of a linked
#              pair.  Used to estimate the marginal accessibility p̂.
#
# Model:
#   k_linked_active | n_linked, θ_L ~ Binomial(n_linked, θ_L)
#   θ_L ~ Beta(α_L, β_L)             (EB prior pooled across (cell, bin) obs)
#   θ_L | data ~ Beta(α_L + k, β_L + n - k)
#
#   p̂ = k_unlinked_open / n_unlinked
#   θ_U = p̂² (active concordance independence baseline)
#   Δ = E[θ_L | data] - θ_U
#
# Variance:
#   σ²(θ_L) = α_post · β_post / ((α+β+n)² · (α+β+n+1))
#   Var(p̂²) = 4 p̂² (1-p̂) / n_unlinked       (delta method on g(p)=p²)
#   σ²(Δ) ≈ σ²(θ_L) + Var(θ_U)            (independence of disjoint read pools)


def _compute_cell_betabinom_counts(
    posteriors_path: str,
    cell_name: str,
    bin_size: int,
    min_distance: int,
    min_gch_sites: int,
    posterior_threshold: float,
) -> tuple[str, dict[str, dict[str, int]]]:
    """For one cell, aggregate per-bin linked/unlinked counts.

    Returns
    -------
    (cell_name, {bin_id: {"n_linked": ..., "k_linked_active": ...,
                          "n_unlinked": ..., "k_unlinked_open": ...}})
    """
    df = pd.read_parquet(posteriors_path)
    required = {"chrom", "pos1", "pos2", "n1", "n2", "distance",
                "posterior1", "posterior2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Posteriors file for {cell_name} is missing columns {sorted(missing)}"
        )

    if len(df) == 0:
        return cell_name, {}

    chrom = df["chrom"].astype(str).values
    pos1 = df["pos1"].astype(np.int64).values
    pos2 = df["pos2"].astype(np.int64).values
    n1 = df["n1"].astype(np.int32).values
    n2 = df["n2"].astype(np.int32).values
    p1 = df["posterior1"].astype(float).values
    p2 = df["posterior2"].astype(float).values
    distance = df["distance"].astype(np.int64).values

    bsz = int(bin_size)
    bin1 = (pos1 // bsz) * bsz
    bin2 = (pos2 // bsz) * bsz

    # An end is informative if it has enough GCH sites AND a valid posterior.
    inf1 = (n1 >= min_gch_sites) & np.isfinite(p1)
    inf2 = (n2 >= min_gch_sites) & np.isfinite(p2)
    open1 = inf1 & (p1 > posterior_threshold)
    open2 = inf2 & (p2 > posterior_threshold)

    # Linked pair: both ends in same bin (same chrom + same bin start),
    # distance >= min_distance, both ends informative.
    same_bin = (bin1 == bin2)
    linked = same_bin & (distance >= min_distance) & inf1 & inf2
    both_open = linked & open1 & open2

    counts: dict[str, dict[str, int]] = {}

    def _bin_id(c: str, b: int) -> str:
        return f"{c}:{int(b)}"

    # Linked aggregation (per bin).
    linked_idx = np.where(linked)[0]
    for i in linked_idx:
        bid = _bin_id(chrom[i], int(bin1[i]))
        rec = counts.setdefault(bid, {
            "n_linked": 0, "k_linked_active": 0,
            "n_unlinked": 0, "k_unlinked_open": 0,
        })
        rec["n_linked"] += 1
        if both_open[i]:
            rec["k_linked_active"] += 1

    # Unlinked aggregation: any informative end that is NOT part of a linked pair.
    not_linked = ~linked
    # End 1 contributes if informative AND its pair is not a linked observation.
    e1_unlinked = inf1 & not_linked
    e1_idx = np.where(e1_unlinked)[0]
    for i in e1_idx:
        bid = _bin_id(chrom[i], int(bin1[i]))
        rec = counts.setdefault(bid, {
            "n_linked": 0, "k_linked_active": 0,
            "n_unlinked": 0, "k_unlinked_open": 0,
        })
        rec["n_unlinked"] += 1
        if open1[i]:
            rec["k_unlinked_open"] += 1

    # End 2 contributes if informative AND its pair is not a linked observation.
    e2_unlinked = inf2 & not_linked
    e2_idx = np.where(e2_unlinked)[0]
    for i in e2_idx:
        bid = _bin_id(chrom[i], int(bin2[i]))
        rec = counts.setdefault(bid, {
            "n_linked": 0, "k_linked_active": 0,
            "n_unlinked": 0, "k_unlinked_open": 0,
        })
        rec["n_unlinked"] += 1
        if open2[i]:
            rec["k_unlinked_open"] += 1

    return cell_name, counts


def _fit_betabinom_eb_prior(
    counts_df: pd.DataFrame,
    min_obs: int = 50,
    nu_floor: float = 2.0,
) -> tuple[float, float, dict]:
    """Method-of-moments EB prior on θ_L pooled across all (cell, bin) obs.

    Parameters
    ----------
    counts_df : DataFrame
        Long-format DataFrame with columns ``n_linked``, ``k_linked_active``.
        Only rows with ``n_linked >= 1`` are used.
    min_obs : int
        Minimum number of (cell, bin) observations needed to fit; otherwise
        fall back to a weakly informative Beta(1, 1).
    nu_floor : float
        Minimum concentration parameter ν = α + β.

    Returns
    -------
    (alpha, beta, info)
        info = {"mu_hat", "nu_hat", "n_obs", "fallback"}
    """
    df = counts_df.loc[counts_df["n_linked"] >= 1].copy()
    n_obs = len(df)
    if n_obs < min_obs:
        logger.warning(
            f"Only {n_obs} (cell, bin) observations with n_linked >= 1 "
            f"(< {min_obs}); using Beta(1, 1) fallback prior."
        )
        return 1.0, 1.0, {"mu_hat": 0.5, "nu_hat": 2.0, "n_obs": n_obs, "fallback": True}

    k = df["k_linked_active"].values.astype(float)
    n = df["n_linked"].values.astype(float)
    rate = k / n

    # Weighted mean (by n).
    mu_hat = float(k.sum() / n.sum())
    # Weighted variance of the rate.
    w = n / n.sum()
    s2 = float(np.sum(w * (rate - mu_hat) ** 2))
    # Mean n (harmonic-ish — for the binomial variance term).
    n_bar = float(np.sum(w * n))

    # Method-of-moments: solve for ν = α + β.
    # Var(rate) = mu(1-mu)/n + (1 - 1/n) · mu(1-mu)/(ν+1)
    # Simplify: ν = (mu(1-mu) - s2) / (s2 - mu(1-mu)/n_bar)
    mu1mu = mu_hat * (1.0 - mu_hat)
    numer = mu1mu - s2
    denom = s2 - mu1mu / max(n_bar, 1.0)
    if denom <= 0:
        # Empirical variance is at most the binomial sampling noise → no
        # detectable over-dispersion → tight prior is appropriate.
        nu_hat = 100.0
    elif numer <= 0:
        # Empirical variance exceeds the max binomial-mixture variance for
        # this μ → heterogeneity is "super-binomial".  A beta-binomial cannot
        # match this exactly; the closest fit is a very wide prior (small ν)
        # that lets the data dominate.
        nu_hat = nu_floor
    else:
        nu_hat = max(float(numer / denom), nu_floor)

    alpha = mu_hat * nu_hat
    beta = (1.0 - mu_hat) * nu_hat
    return alpha, beta, {
        "mu_hat": mu_hat, "nu_hat": nu_hat, "n_obs": n_obs, "fallback": False,
    }


def compute_betabinom_concordance_matrix(
    cell_posterior_paths: dict[str, str],
    chrom_sizes: dict[str, int] | None = None,
    bin_size: int = 250_000,
    min_distance: int = 1_000,
    min_gch_sites: int = 3,
    posterior_threshold: float = 0.5,
    min_pairs_per_bin: int = 5,
    eb_min_obs: int = 50,
    n_threads: int = 4,
    output_dir: str = "output/05_concordance",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute a (cells x bins) matrix of Beta-Binomial coordination excess Δ.

    For each (cell, bin):
      - Aggregate linked counts (long-range pairs, both ends in bin).
      - Aggregate unlinked counts (any other informative ends in bin).
      - Compute Δ = E[θ_L | data] - p̂² with EB prior on θ_L.

    Returns ``(delta_df, pair_counts_df)``:
      - delta_df: cells × bins, NaN where ``n_linked < min_pairs_per_bin``.
      - pair_counts_df: cells × bins, holds n_linked for downstream filtering.

    Also writes a long-format per-(cell, bin) counts table, EB prior, and
    Δ-variance matrix to ``output_dir``.
    """
    from joblib import Parallel, delayed

    out_path = ensure_dir(output_dir)
    items = list(cell_posterior_paths.items())

    # ---- Step 1: aggregate counts per (cell, bin) in parallel. ----
    logger.info(
        f"Aggregating per-(cell, bin) Beta-Binomial counts for "
        f"{len(items)} cells (bin_size={bin_size:,}, "
        f"min_distance={min_distance:,}, threshold={posterior_threshold})"
    )

    if n_threads > 1 and len(items) > 1:
        results = Parallel(n_jobs=n_threads)(
            delayed(_compute_cell_betabinom_counts)(
                path, cell_name, bin_size, min_distance,
                min_gch_sites, posterior_threshold,
            )
            for cell_name, path in items
        )
    else:
        results = [
            _compute_cell_betabinom_counts(
                path, cell_name, bin_size, min_distance,
                min_gch_sites, posterior_threshold,
            )
            for cell_name, path in items
        ]

    # ---- Step 2: flatten into a long DataFrame for EB fitting. ----
    rows = []
    for cell_name, cell_counts in results:
        for bid, rec in cell_counts.items():
            rows.append({
                "cell_name": cell_name,
                "bin_id": bid,
                "n_linked": rec["n_linked"],
                "k_linked_active": rec["k_linked_active"],
                "n_unlinked": rec["n_unlinked"],
                "k_unlinked_open": rec["k_unlinked_open"],
            })
    counts_long = pd.DataFrame(rows)
    counts_long.to_parquet(out_path / "betabinom_counts.parquet", index=False)
    logger.info(f"Aggregated {len(counts_long):,} (cell, bin) records")

    # ---- Step 3: fit EB prior on θ_L. ----
    alpha_L, beta_L, prior_info = _fit_betabinom_eb_prior(
        counts_long, min_obs=eb_min_obs,
    )
    logger.info(
        f"EB prior θ_L ~ Beta({alpha_L:.3f}, {beta_L:.3f})  "
        f"[μ={prior_info['mu_hat']:.4f}, ν={prior_info['nu_hat']:.1f}, "
        f"n_obs={prior_info['n_obs']}, fallback={prior_info['fallback']}]"
    )
    import json
    with open(out_path / "betabinom_prior.json", "w") as f:
        json.dump({
            "alpha_L": float(alpha_L), "beta_L": float(beta_L),
            **prior_info,
        }, f, indent=2)

    # ---- Step 4: compute Δ per (cell, bin) and pivot to matrix. ----
    n_l = counts_long["n_linked"].values.astype(float)
    k_l = counts_long["k_linked_active"].values.astype(float)
    n_u = counts_long["n_unlinked"].values.astype(float)
    k_u = counts_long["k_unlinked_open"].values.astype(float)

    # Posterior on θ_L.
    alpha_post = alpha_L + k_l
    beta_post = beta_L + (n_l - k_l)
    nu_post = alpha_post + beta_post
    theta_L = alpha_post / nu_post
    theta_L_var = (alpha_post * beta_post) / (nu_post ** 2 * (nu_post + 1.0))

    # Independence baseline.
    p_hat = np.where(n_u > 0, k_u / n_u, np.nan)
    theta_U = p_hat ** 2
    # Delta method: Var(p²) = (2p)² · p(1-p)/n = 4 p² (1-p) / n.
    var_p = np.where(n_u > 0, p_hat * (1.0 - p_hat) / np.maximum(n_u, 1.0), np.nan)
    theta_U_var = 4.0 * (p_hat ** 2) * var_p

    # Δ.
    delta = theta_L - theta_U
    delta_var = theta_L_var + theta_U_var

    # When n_linked < min_pairs_per_bin OR n_unlinked == 0 → NaN.
    mask_invalid = (n_l < min_pairs_per_bin) | (n_u == 0) | ~np.isfinite(delta)
    delta = np.where(mask_invalid, np.nan, delta)
    delta_var = np.where(mask_invalid | ~np.isfinite(delta_var), np.nan, delta_var)

    counts_long["theta_L"] = theta_L
    counts_long["theta_U"] = theta_U
    counts_long["p_hat"] = p_hat
    counts_long["delta"] = delta
    counts_long["delta_var"] = delta_var
    counts_long.to_parquet(out_path / "betabinom_counts.parquet", index=False)

    # ---- Step 5: pivot to cells × bins matrices. ----
    sorted_bins = _sort_bins(set(counts_long["bin_id"]), chrom_sizes)
    cell_names = sorted({c for c, _ in items})

    delta_df = (
        counts_long.pivot_table(
            index="cell_name", columns="bin_id", values="delta", aggfunc="first",
        )
        .reindex(index=sorted(cell_names), columns=sorted_bins)
    )
    delta_var_df = (
        counts_long.pivot_table(
            index="cell_name", columns="bin_id", values="delta_var", aggfunc="first",
        )
        .reindex(index=sorted(cell_names), columns=sorted_bins)
    )
    pair_counts_df = (
        counts_long.pivot_table(
            index="cell_name", columns="bin_id", values="n_linked",
            aggfunc="first", fill_value=0,
        )
        .reindex(index=sorted(cell_names), columns=sorted_bins, fill_value=0)
        .astype(np.int64)
    )

    delta_df.columns.name = "bin"
    delta_df.index.name = "cell"
    delta_var_df.columns.name = "bin"
    delta_var_df.index.name = "cell"
    pair_counts_df.columns.name = "bin"
    pair_counts_df.index.name = "cell"

    # Save: concordance_raw.parquet is the Δ matrix (replaces Pearson r).
    delta_df.to_parquet(out_path / "concordance_raw.parquet")
    delta_var_df.to_parquet(out_path / "concordance_delta_var.parquet")
    pair_counts_df.to_parquet(out_path / "pair_counts_per_bin.parquet")
    delta_df.to_csv(out_path / "concordance_raw.tsv", sep="\t")
    pair_counts_df.to_csv(out_path / "pair_counts_per_bin.tsv", sep="\t")

    logger.info(
        f"Beta-Binomial concordance matrix: {delta_df.shape[0]} cells x "
        f"{delta_df.shape[1]} bins "
        f"({delta_df.notna().sum().sum():,} non-NA entries)"
    )

    return delta_df, pair_counts_df


def _sort_bins(bins: set[str], chrom_sizes: dict[str, int] | None) -> list[str]:
    """Sort bin identifiers like ``'chr1:0'`` by chromosome and start."""
    parsed: list[tuple[str, int, str]] = []
    for b in bins:
        chrom, start_str = b.split(":", 1)
        try:
            start = int(start_str)
        except ValueError:
            start = 0
        parsed.append((chrom, start, b))

    if chrom_sizes is not None:
        order = {c: i for i, c in enumerate(chrom_sizes.keys())}
        parsed.sort(key=lambda t: (order.get(t[0], 1_000_000), t[1]))
    else:
        # Natural sort on chromosome name, then start.
        def _chrom_key(chrom: str) -> tuple[int, int | str]:
            base = chrom[3:] if chrom.startswith("chr") else chrom
            if base.isdigit():
                return (0, int(base))
            return (1, base)

        parsed.sort(key=lambda t: (_chrom_key(t[0]), t[1]))

    return [t[2] for t in parsed]


# ---------------------------------------------------------------------------
# Signed concordance with U-3 soft counts + Dirichlet-Multinomial Layer 2
# ---------------------------------------------------------------------------
#
# Per design-doc v0.3 §2:
#
#   Layer 1 (per-end posterior, already in Step 4): p_i = P(end is open | data)
#
#   Layer 2 (per-(cell, bin), this section):
#     For each PAIR (i, j) with posteriors (p_i, p_j):
#       contribution to k_open    = p_i · p_j
#       contribution to k_closed  = (1 − p_i)(1 − p_j)
#       contribution to k_mixed   = p_i(1 − p_j) + (1 − p_i) p_j
#
#     LINKED   = same-bin pairs with distance >= min_distance, both informative
#     UNLINKED = same-bin pairs with distance <  min_distance, both informative
#                (short-range pairs in the same bin — the 3-state baseline)
#
#     Dirichlet-Multinomial EB prior fit by MoM separately on linked & unlinked.
#     Δ_signed = (E[π_l_open] − E[π_l_closed]) − (E[π_u_open] − E[π_u_closed])
#
#     Pre-Layer-2 RWR (default on) imputes the soft-count matrices using a
#     k-NN graph built from QN(mean_accessibility); see imputation.py.


_SOFT_COUNT_KEYS = (
    "k_open", "k_closed", "k_mixed", "n_linked",
    "k_u_open", "k_u_closed", "k_u_mixed", "n_unlinked",
)

# Distance-matched expected (O/E baseline) soft counts — emitted only when
# baseline='distance-matched'.  For each observed long-range contact we add
# the *expected* co-state of random, distance-matched end-pairs at the same
# distance, so n_dm == n_linked and the per-(cell,bin) expected RATE is
# distance-matched to the observed contacts.
_DM_KEYS = ("k_dm_open", "k_dm_closed", "k_dm_mixed", "n_dm")


def _compute_cell_soft_counts(
    posteriors_path: str,
    cell_name: str,
    bin_size: int,
    min_distance: int,
    min_gch_sites: int,
    max_distance: int = 0,
) -> tuple[str, dict[str, dict[str, float]]]:
    """For one cell, aggregate per-bin U-3 soft expected counts.

    Each pair contributes its full posterior probabilities (no thresholding)
    to the three joint outcomes (both-open, both-closed, mixed) for either
    the linked pool (long-range same-bin pairs) or the unlinked pool
    (short-range same-bin pairs).

    Returns
    -------
    (cell_name, {bin_id: {key: float}}) where ``key`` is one of
    ``_SOFT_COUNT_KEYS``.  Bins with no contributions are omitted.
    """
    df = pd.read_parquet(
        posteriors_path,
        columns=["chrom", "pos1", "pos2", "n1", "n2", "distance",
                 "posterior1", "posterior2"],
    )

    if len(df) == 0:
        return cell_name, {}

    chrom = df["chrom"].astype(str).values
    pos1 = df["pos1"].to_numpy(dtype=np.int64)
    pos2 = df["pos2"].to_numpy(dtype=np.int64)
    n1 = df["n1"].to_numpy(dtype=np.int32)
    n2 = df["n2"].to_numpy(dtype=np.int32)
    p1 = df["posterior1"].to_numpy(dtype=np.float64, na_value=np.nan)
    p2 = df["posterior2"].to_numpy(dtype=np.float64, na_value=np.nan)
    distance = df["distance"].to_numpy(dtype=np.int64)

    bsz = int(bin_size)
    bin1 = (pos1 // bsz) * bsz
    bin2 = (pos2 // bsz) * bsz

    # Pair must be informative (both ends pass min_gch_sites AND have a
    # finite posterior) and same-bin.  Cross-bin pairs are excluded from
    # both pools (per design-doc §2.1).
    inf1 = (n1 >= min_gch_sites) & np.isfinite(p1)
    inf2 = (n2 >= min_gch_sites) & np.isfinite(p2)
    informative_pair = inf1 & inf2
    same_bin = bin1 == bin2

    qualify = same_bin & informative_pair
    if not qualify.any():
        return cell_name, {}

    # Soft per-pair contributions (U-3) — only computed for qualifying pairs.
    pq = p1[qualify]
    qq = p2[qualify]
    soft_open = pq * qq
    soft_closed = (1.0 - pq) * (1.0 - qq)
    soft_mixed = pq * (1.0 - qq) + (1.0 - pq) * qq

    # Linked = long-range (>= min_distance), optionally capped at max_distance
    # (>0) to focus a distance band, e.g. the 20–200 kb E–P regulatory scale.
    # Short-range (< min_distance) is the baseline pool.  Pairs beyond
    # max_distance fall into NEITHER pool.
    dq = distance[qualify]
    is_short = dq < int(min_distance)
    is_linked = dq >= int(min_distance)
    if max_distance and int(max_distance) > 0:
        is_linked = is_linked & (dq <= int(max_distance))
    chrom_q = chrom[qualify]
    bin_q = bin1[qualify]

    counts: dict[str, dict[str, float]] = {}

    # Vectorised aggregation per (chrom, bin) for linked + unlinked sides.
    pair_df = pd.DataFrame({
        "chrom": chrom_q,
        "bin": bin_q,
        "is_linked": is_linked,
        "is_short": is_short,
        "s_open": soft_open,
        "s_closed": soft_closed,
        "s_mixed": soft_mixed,
    })

    # Linked aggregation.
    linked_g = pair_df.loc[pair_df["is_linked"]].groupby(
        ["chrom", "bin"], sort=False,
    ).agg(
        n_linked=("s_open", "size"),
        k_open=("s_open", "sum"),
        k_closed=("s_closed", "sum"),
        k_mixed=("s_mixed", "sum"),
    )
    # Unlinked (short-range) aggregation.
    unlinked_g = pair_df.loc[pair_df["is_short"]].groupby(
        ["chrom", "bin"], sort=False,
    ).agg(
        n_unlinked=("s_open", "size"),
        k_u_open=("s_open", "sum"),
        k_u_closed=("s_closed", "sum"),
        k_u_mixed=("s_mixed", "sum"),
    )

    def _bid(c: str, b: int) -> str:
        return f"{c}:{int(b)}"

    def _empty_entry() -> dict[str, float]:
        return {k: 0.0 for k in _SOFT_COUNT_KEYS}

    for (c, b), row in linked_g.iterrows():
        bid = _bid(c, int(b))
        rec = counts.setdefault(bid, _empty_entry())
        rec["n_linked"] = float(row["n_linked"])
        rec["k_open"] = float(row["k_open"])
        rec["k_closed"] = float(row["k_closed"])
        rec["k_mixed"] = float(row["k_mixed"])

    for (c, b), row in unlinked_g.iterrows():
        bid = _bid(c, int(b))
        rec = counts.setdefault(bid, _empty_entry())
        rec["n_unlinked"] = float(row["n_unlinked"])
        rec["k_u_open"] = float(row["k_u_open"])
        rec["k_u_closed"] = float(row["k_u_closed"])
        rec["k_u_mixed"] = float(row["k_u_mixed"])

    return cell_name, counts


def _dm_expected_for_bin(
    epos: np.ndarray, ep: np.ndarray, contact_dists: np.ndarray,
    dist_bin: int, max_ends: int, seed: int,
) -> tuple[float, float, float, int]:
    """Distance-matched expected co-state for one bin's contacts.

    ``epos``/``ep`` are the positions and open-posteriors of the bin's read
    ends (the null pool).  Forms all (or a capped sample of) end-pairs,
    bins their separation into ``dist_bin``-bp buckets, and for each
    observed contact looks up the mean soft co-state of random end-pairs at
    the contact's distance bucket (falling back to the pooled mean when a
    bucket is empty).  Returns summed expected (k_dm_open, k_dm_closed,
    k_dm_mixed, n_dm) over the contacts, and how many of them took the
    fallback.

    THE FALLBACK IS NOT NEUTRAL.  A contact whose bucket is empty is scored
    against the DISTANCE-POOLED mean -- the very baseline distance matching
    was chosen instead of.  Empty buckets are not spread evenly: the end
    pool is capped at ``max_ends`` and a bin's ends thin out with
    separation, so the buckets that come up empty are the LONG ones, which
    is where the analysis lives.  A run can therefore be distance-matched
    for its short contacts and silently distance-pooled for its long ones,
    and since contacts skew short within any bucket while re-paired ends do
    not, that reintroduces a positive offset in exactly the bins the
    long-range question is about.  The count is returned so the run can say
    how much of itself was actually matched.
    """
    m = epos.shape[0]
    if m < 2 or contact_dists.size == 0:
        return 0.0, 0.0, 0.0, 0, 0
    if m > max_ends:                      # subsample the end pool for speed
        rng = np.random.default_rng(seed)
        idx = rng.choice(m, size=max_ends, replace=False)
        epos, ep = epos[idx], ep[idx]
        m = max_ends
    iu, ju = np.triu_indices(m, k=1)      # all unordered end-pairs (i<j)
    d = np.abs(epos[iu] - epos[ju])
    pa, pb = ep[iu], ep[ju]
    po = pa * pb
    pc = (1.0 - pa) * (1.0 - pb)
    pm = pa * (1.0 - pb) + (1.0 - pa) * pb
    bucket = (d // dist_bin).astype(np.int64)
    nb = int(bucket.max()) + 1
    cnt = np.bincount(bucket, minlength=nb).astype(np.float64)
    sum_o = np.bincount(bucket, weights=po, minlength=nb)
    sum_c = np.bincount(bucket, weights=pc, minlength=nb)
    sum_m = np.bincount(bucket, weights=pm, minlength=nb)
    safe = np.where(cnt > 0, cnt, 1.0)
    exp_o, exp_c, exp_m = sum_o / safe, sum_c / safe, sum_m / safe
    pooled_o = po.mean(); pooled_c = pc.mean(); pooled_m = pm.mean()

    cb = (contact_dists // dist_bin).astype(np.int64)
    k_o = k_c = k_m = 0.0
    n_fb = 0
    for b in cb:
        if b < nb and cnt[b] > 0:
            k_o += exp_o[b]; k_c += exp_c[b]; k_m += exp_m[b]
        else:                              # empty bucket → pooled fallback
            k_o += pooled_o; k_c += pooled_c; k_m += pooled_m
            n_fb += 1
    return (float(k_o), float(k_c), float(k_m), int(contact_dists.size),
            int(n_fb))


def _pooled_expected_for_bin(
    epos: np.ndarray, ep: np.ndarray,
    contact_pairs: list[tuple[int, int]], n_contacts: int,
    min_distance: int, max_ends: int, seed: int, max_distance: int = 0,
) -> tuple[float, float, float, int]:
    """Distance-POOLED expected co-state for one bin's contacts.

    Like ``_dm_expected_for_bin`` it re-pairs the bin's read ends into a
    non-contact null, but instead of matching each observed contact's
    distance it pools **all** synthetic end-pairs with separation
    >= ``min_distance`` into a single mean expected rate (excluding any
    pair that coincides with an observed contact).  This trades the
    distance control of the distance-matched null for many more pairs, so
    the per-bin expected is far less noisy ("more reads for the
    estimation").

    To stay a drop-in sibling of the distance-matched expected, the result
    is scaled so that ``n_dm == n_contacts`` and the per-state rate
    ``k_*/n_dm`` equals the pooled mean soft co-state — identical
    downstream DM / log2(O/E) handling.  The pooling benefit is realised
    purely through a less-noisy mean (the reported n, and hence the DM
    shrinkage, matches the distance-matched convention).
    """
    m = epos.shape[0]
    if m < 2 or n_contacts == 0:
        return 0.0, 0.0, 0.0, 0
    if m > max_ends:                       # subsample the end pool for speed
        rng = np.random.default_rng(seed)
        idx = rng.choice(m, size=max_ends, replace=False)
        epos, ep = epos[idx], ep[idx]
        m = max_ends
    iu, ju = np.triu_indices(m, k=1)       # all unordered end-pairs (i<j)
    d = np.abs(epos[iu] - epos[ju])
    keep = d >= int(min_distance)          # long-range pairs only
    if max_distance and int(max_distance) > 0:   # cap to the distance band
        keep &= d <= int(max_distance)
    if contact_pairs:                      # exclude the observed (linked) pairs
        lo = np.minimum(epos[iu], epos[ju])
        hi = np.maximum(epos[iu], epos[ju])
        for c_lo, c_hi in contact_pairs:
            keep &= ~((lo == int(c_lo)) & (hi == int(c_hi)))
    if not keep.any():
        return 0.0, 0.0, 0.0, 0
    pa, pb = ep[iu][keep], ep[ju][keep]
    mean_o = float((pa * pb).mean())
    mean_c = float(((1.0 - pa) * (1.0 - pb)).mean())
    mean_m = float((pa * (1.0 - pb) + (1.0 - pa) * pb).mean())
    nc = int(n_contacts)
    return mean_o * nc, mean_c * nc, mean_m * nc, nc


def _compute_cell_dm_expected(
    posteriors_path: str,
    cell_name: str,
    bin_size: int,
    min_distance: int,
    min_gch_sites: int,
    dm_window_bins: int = 0,
    dm_dist_bin: int = 10_000,
    dm_max_ends: int = 400,
    mode: str = "matched",
    max_distance: int = 0,
    random_state: int = 42,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Per-(cell, bin) expected soft counts (``_DM_KEYS``) from a non-contact
    end-pair null.

    The null pool for a bin is its own informative read ends (those of any
    pair, regardless of partner location); with ``dm_window_bins > 0`` it is
    pooled over ±that many neighbouring bins (more end-pairs at long
    distances where a single bin's pairs thin out).

    ``mode`` selects how the expected rate is formed from that pool:

    - ``"matched"`` (distance-matched): for each observed contact, the mean
      soft co-state of null end-pairs at the contact's distance bucket
      (``_dm_expected_for_bin``).  Controls distance + bin-marginal.
    - ``"pooled"`` (distance-pooled): the mean soft co-state of *all* null
      end-pairs with separation >= ``min_distance`` (excluding the observed
      contacts; ``_pooled_expected_for_bin``).  Drops distance control for
      many more pairs → a less-noisy expected.
    """
    df = pd.read_parquet(
        posteriors_path,
        columns=["chrom", "pos1", "pos2", "n1", "n2", "distance",
                 "posterior1", "posterior2"],
    )
    if len(df) == 0:
        return cell_name, {}
    chrom = df["chrom"].astype(str).values
    pos1 = df["pos1"].to_numpy(dtype=np.int64)
    pos2 = df["pos2"].to_numpy(dtype=np.int64)
    n1 = df["n1"].to_numpy(dtype=np.int32)
    n2 = df["n2"].to_numpy(dtype=np.int32)
    p1 = df["posterior1"].to_numpy(dtype=np.float64, na_value=np.nan)
    p2 = df["posterior2"].to_numpy(dtype=np.float64, na_value=np.nan)
    distance = df["distance"].to_numpy(dtype=np.int64)
    bsz = int(bin_size)
    inf1 = (n1 >= min_gch_sites) & np.isfinite(p1)
    inf2 = (n2 >= min_gch_sites) & np.isfinite(p2)

    # End pool: every informative end, by (chrom, bin).
    e_chrom = np.concatenate([chrom[inf1], chrom[inf2]])
    e_pos = np.concatenate([pos1[inf1], pos2[inf2]])
    e_p = np.concatenate([p1[inf1], p2[inf2]])
    e_bin = (e_pos // bsz)
    ends = pd.DataFrame({"chrom": e_chrom, "bin": e_bin, "pos": e_pos, "p": e_p})
    ends_by_bin = {key: g for key, g in ends.groupby(["chrom", "bin"], sort=False)}

    # Observed long-range contacts (same-bin, informative, linked); optionally
    # capped at max_distance to focus a distance band (e.g. 20–200 kb E–P).
    same_bin = (pos1 // bsz) == (pos2 // bsz)
    linked = same_bin & inf1 & inf2 & (distance >= int(min_distance))
    if max_distance and int(max_distance) > 0:
        linked = linked & (distance <= int(max_distance))
    if not linked.any():
        return cell_name, {}
    obs = pd.DataFrame({
        "chrom": chrom[linked], "bin": (pos1[linked] // bsz),
        "distance": distance[linked],
        "pos1": pos1[linked], "pos2": pos2[linked],
    })

    counts: dict[str, dict[str, float]] = {}
    # zlib.crc32 (not the builtin hash(): Python salts string hashes per
    # process via PYTHONHASHSEED, which made this end-pool subsample — and so
    # the O/E baseline — differ between otherwise identical runs).  Offsetting
    # by random_state keeps the per-cell decorrelation while letting a
    # --random-seed sweep actually perturb the subsample.
    seed_base = (zlib.crc32(cell_name.encode("utf-8")) + int(random_state)) % (2 ** 31)
    for (c, b), og in obs.groupby(["chrom", "bin"], sort=False):
        # Assemble the (optionally windowed) end pool for this bin.
        if dm_window_bins > 0:
            pool = [ends_by_bin[(c, bb)] for bb in range(int(b) - dm_window_bins,
                    int(b) + dm_window_bins + 1) if (c, bb) in ends_by_bin]
            ep_df = pd.concat(pool) if pool else None
        else:
            ep_df = ends_by_bin.get((c, int(b)))
        if ep_df is None or len(ep_df) < 2:
            continue
        if mode == "pooled":
            p1c = og["pos1"].to_numpy(dtype=np.int64)
            p2c = og["pos2"].to_numpy(dtype=np.int64)
            contact_pairs = list(zip(np.minimum(p1c, p2c).tolist(),
                                     np.maximum(p1c, p2c).tolist()))
            n_fb = 0
            k_o, k_c, k_m, n_dm = _pooled_expected_for_bin(
                ep_df["pos"].to_numpy(dtype=np.int64),
                ep_df["p"].to_numpy(dtype=np.float64),
                contact_pairs, n_contacts=len(og),
                min_distance=int(min_distance), max_ends=int(dm_max_ends),
                seed=seed_base + int(b), max_distance=int(max_distance),
            )
        else:
            k_o, k_c, k_m, n_dm, n_fb = _dm_expected_for_bin(
                ep_df["pos"].to_numpy(dtype=np.int64),
                ep_df["p"].to_numpy(dtype=np.float64),
                og["distance"].to_numpy(dtype=np.int64),
                dist_bin=int(dm_dist_bin), max_ends=int(dm_max_ends),
                seed=seed_base + int(b),
            )
        counts[f"{c}:{int(b) * bsz}"] = {
            "k_dm_open": k_o, "k_dm_closed": k_c, "k_dm_mixed": k_m,
            "n_dm": float(n_dm),
            # Deliberately NOT in _DM_KEYS: it is a diagnostic, not a count
            # to impute.  The matrix assembly takes only matrix_keys, so
            # extra entries here ride along and are dropped there.
            "n_dm_fallback": float(n_fb),
        }
    return cell_name, counts


def _fit_dirichlet_multinomial_eb_prior(
    counts_df: pd.DataFrame,
    open_col: str = "k_open",
    closed_col: str = "k_closed",
    mixed_col: str = "k_mixed",
    n_col: str = "n_linked",
    min_obs: int = 50,
    nu_floor: float = 2.0,
) -> tuple[dict[str, float], dict]:
    """Method-of-moments EB prior for the Dirichlet-Multinomial on three categories.

    Pools (cell, bin) observations and fits α = (α_open, α_closed, α_mixed)
    such that the population-level distribution of soft rates
    ``(k_i / n)`` matches the DM theoretical variance for some
    concentration ν = α_total.

    Returns
    -------
    (alpha_dict, info) :
        alpha_dict has keys 'open', 'closed', 'mixed' (floats);
        info has 'mu_hat' (dict), 'nu_hat' (float), 'n_obs' (int),
        'fallback' (bool).
    """
    df = counts_df.loc[counts_df[n_col] >= 1].copy()
    n_obs = int(len(df))
    if n_obs < min_obs:
        logger.warning(
            f"Only {n_obs} (cell, bin) observations with {n_col} >= 1 "
            f"(< {min_obs}); using uniform Dirichlet(1, 1, 1) fallback prior."
        )
        return ({"open": 1.0, "closed": 1.0, "mixed": 1.0},
                {"mu_hat": {"open": 1/3, "closed": 1/3, "mixed": 1/3},
                 "nu_hat": 3.0, "n_obs": n_obs, "fallback": True})

    n_arr = df[n_col].to_numpy(dtype=np.float64)
    n_sum = float(n_arr.sum())
    # Population-level marginal means (across pooled (cell, bin)).
    mus = {
        "open":   float(df[open_col].sum()   / n_sum),
        "closed": float(df[closed_col].sum() / n_sum),
        "mixed":  float(df[mixed_col].sum()  / n_sum),
    }

    # Method-of-moments: fit α_total on the category with the largest mean
    # (most data → tightest variance estimate).  The same α_total applies
    # to all three categories by Dirichlet symmetry.
    best_cat, best_mu = max(mus.items(), key=lambda kv: kv[1])
    best_col = {"open": open_col, "closed": closed_col, "mixed": mixed_col}[best_cat]
    k_best = df[best_col].to_numpy(dtype=np.float64)
    rate = k_best / n_arr
    weights = n_arr / n_sum
    s2 = float(np.sum(weights * (rate - best_mu) ** 2))
    n_bar = float(np.sum(weights * n_arr))

    mu1mu = best_mu * (1.0 - best_mu)
    numer = mu1mu - s2
    denom = s2 - mu1mu / max(n_bar, 1.0)
    if denom <= 0:
        # No detectable over-dispersion → tight prior.
        nu_hat = 100.0
    elif numer <= 0:
        # Empirical variance > max binomial-mixture variance → very wide prior.
        nu_hat = nu_floor
    else:
        nu_hat = max(float(numer / denom), nu_floor)

    alpha = {cat: mu * nu_hat for cat, mu in mus.items()}
    return alpha, {
        "mu_hat": mus, "nu_hat": float(nu_hat), "n_obs": n_obs,
        "fallback": False, "best_cat_for_mom": best_cat,
    }


def _dm_posterior_means(
    counts: dict[str, np.ndarray],
    alpha: dict[str, float],
    k_keys: tuple[str, str, str],
    n_key: str,
) -> dict[str, np.ndarray]:
    """Element-wise Dirichlet-Multinomial conjugate posterior means.

    Per design-doc §2.3 (latent-variable derivation):
        E[π_i | data] = (α_i + soft_k_i) / (α_total + n)
    For soft counts, this is the correct posterior mean by linearity of
    expectation under the latent-state interpretation.

    Parameters
    ----------
    counts : dict
        Maps each of ``k_keys`` and ``n_key`` to a cells × bins matrix.
    alpha : dict
        Maps 'open', 'closed', 'mixed' → α_i.
    k_keys : tuple
        Length-3 tuple naming the (open, closed, mixed) matrices in
        ``counts``.
    n_key : str
        Name of the n matrix in ``counts``.

    Returns
    -------
    dict mapping 'open', 'closed', 'mixed' → cells × bins matrix of
    posterior means.  NaN where ``n_key`` matrix has zero counts and the
    prior would require us to divide by zero.
    """
    open_k, closed_k, mixed_k = k_keys
    alpha_total = alpha["open"] + alpha["closed"] + alpha["mixed"]
    n = counts[n_key]
    denom = alpha_total + n
    # When n == 0 AND alpha_total > 0, the posterior is the prior mean.
    safe_denom = np.where(denom > 0, denom, np.nan)
    return {
        "open":   (alpha["open"]   + counts[open_k])   / safe_denom,
        "closed": (alpha["closed"] + counts[closed_k]) / safe_denom,
        "mixed":  (alpha["mixed"]  + counts[mixed_k])  / safe_denom,
    }


def _dm_posterior_variance_S(
    posterior_means: dict[str, np.ndarray],
    alpha: dict[str, float],
    n: np.ndarray,
) -> np.ndarray:
    """Variance of S = π_open − π_closed under the DM posterior.

    Var(S) = [μ_open + μ_closed − (μ_open − μ_closed)²] / (α_total + n + 1)

    Parameters
    ----------
    posterior_means : dict
        Output of :func:`_dm_posterior_means`.
    alpha : dict
        Prior parameters.
    n : ndarray
        Cells × bins n_linked (or n_unlinked).
    """
    mu_o = posterior_means["open"]
    mu_c = posterior_means["closed"]
    alpha_total = alpha["open"] + alpha["closed"] + alpha["mixed"]
    return (mu_o + mu_c - (mu_o - mu_c) ** 2) / (alpha_total + n + 1.0)


def decompose_components(
    o_open: np.ndarray, o_closed: np.ndarray,
    e_open: np.ndarray, e_closed: np.ndarray,
) -> dict[str, np.ndarray]:
    """Directional decomposition of the co-state distribution.

    The 3-state (open / closed / mixed) distribution has exactly TWO
    degrees of freedom: the marginal level p̄ and the association Cov.
    "Co-open vs co-closed" is NOT a third axis — it *is* the marginal:

      π_open − π_closed  ==  p̄_1 + p̄_2 − 1   is an exact per-pair
        identity (the p_i·p_j terms cancel), so 'signed-diff' carries
        zero coordination signal against a marginal-matched O/E null.

      (O_open − E_open) == (O_closed − E_closed) == Cov, identically —
        coordination moves mass out of `mixed` into BOTH concordant
        cells by the same amount, so the *excess* cannot be attributed
        to a direction either (a "co-open excess" map and a "co-closed
        excess" map would be the same map).

    Direction therefore has to come from where the concordant mass
    actually SITS rather than from the excess:

      C = (O_open + O_closed) − (E_open + E_closed)  coordination (=2·Cov)
      F = O_open / (O_open + O_closed)               realized direction
      D = C · (2F − 1)                               directional coordination

    D > 0 → coordinated, realized co-OPEN; D < 0 → coordinated
    co-CLOSED; D ≈ 0 → uncoordinated (mixed dominates).  D is *also* ≈0
    for a coordinated-but-balanced bin (F ≈ 0.5), which is
    direction-ambiguous rather than mixed — so C is emitted alongside D
    to separate the two cases (mixed → C ≈ 0; balanced-coordinated →
    C large with D ≈ 0).

    WHERE THE ZERO OF D BELONGS.  ``2F − 1`` puts "no directional
    preference" at F = 0.5, i.e. half the concordant mass co-open.  That
    is the right zero only when the marginal is p̄ = 0.5.  Under
    independence at marginal p̄ the concordant mass already splits

        F_indep = p̄² / (p̄² + (1 − p̄)²)

    which is 0.31 at p̄ = 0.40 and 0.012 at p̄ = 0.10.  So on sparse
    accessibility data a bin with NO directional preference scores
    2F − 1 ≈ −0.38, and D reports −0.38·C: every coordinated bin looks
    co-closed, and D collapses to a scaled −C.  That is the marginal
    contamination this docstring rejects for 'signed-diff' walking back
    in through F.  Measured on one meningioma GCH run: p̄ = 0.4009,
    F = 0.3230, corr(D, −C) = 0.970.

    Referencing F to the SAME expected pool removes it:

      F_exp = E_open / (E_open + E_closed)      direction under the null
      D' = C · (F − F_exp)                      marginal-free direction

    D' = 0 means the concordant mass sits where the marginal-matched
    null puts it — the honest "balanced" — and its sign says co-open or
    co-closed relative to that.  On the same run F − F_exp came out
    POSITIVE where 2F − 1 was negative, so the two disagree on direction,
    not merely on scale.

    Both are emitted.  D is kept unchanged so existing outputs do not
    move; D' ("directional_corrected") is the one to interpret.

    DIRECTION ON THE LOG-ODDS SCALE.  F − F_exp is a difference of two
    proportions, so its sampling variance is F(1−F)/n: the same excess
    means a different amount of evidence at F = 0.5 and at F = 0.03,
    and bins are not comparable across depths or across marginals.
    The log odds ratio referred to the same null is

      L = logit(F) − logit(F_exp) = log[(O_open/O_closed)
                                        / (E_open/E_closed)]

    which is the natural parameter of the 2×2 table.  It is what a
    logistic regression of co-open-vs-co-closed on group estimates, it
    is additive so effects compose by adding rather than by a
    marginal-dependent amount, and its variance is 1/O_open + 1/O_closed
    — free of F, which is what makes bins comparable.

    L IS NOT A SECOND STATISTIC ALONGSIDE F − F_exp.  Both are monotone
    in F at fixed F_exp, and to first order

      L ≈ (F − F_exp) / [F_exp (1 − F_exp)]

    so their difference carries information about F_exp — the marginal —
    and nothing else.  Feeding both to an embedding supplies one axis
    twice.  Use L, or use F − F_exp and ``marginal``; not L and F − F_exp.

    L SAYS NOTHING ABOUT HOW MUCH COORDINATION THERE IS.  Scaling the
    excess concordant mass up while holding the co-open:co-closed ratio
    fixed leaves L unchanged and raises C.  L is direction; C is
    magnitude; a claim that coordination INCREASED is a claim about C.
    """
    coord = (o_open + o_closed) - (e_open + e_closed)
    mass = o_open + o_closed
    # No concordant mass → no direction → F = 0.5 → D = 0.
    safe_mass = np.where(mass > 1e-12, mass, 1.0)
    frac = np.where(mass > 1e-12, o_open / safe_mass, 0.5)
    # The same split under the marginal-matched null.
    e_mass = e_open + e_closed
    safe_e = np.where(e_mass > 1e-12, e_mass, 1.0)
    frac_exp = np.where(e_mass > 1e-12, e_open / safe_e, 0.5)
    # Excess is 0, not 0.5 - F_exp, when either pool is empty: with no
    # observed concordant mass there is no direction to report, and a
    # non-zero excess there would be an artefact of the F = 0.5 default
    # rather than a measurement.
    ok = (mass > 1e-12) & (e_mass > 1e-12)
    excess = np.where(ok, frac - frac_exp, 0.0)
    # Same direction on the log-odds scale.  logit(F) = log(O_open /
    # O_closed) exactly, so the concordant mass cancels and only the
    # two ratios remain.  Clipped at the same eps 'log2-ratio' uses:
    # DM rates are strictly inside (0, 1) but can be small enough that
    # an unclipped log dominates every other bin.
    _eps = 1e-6
    _lr = lambda a, b: (np.log(np.clip(a, _eps, None))
                        - np.log(np.clip(b, _eps, None)))
    logor = np.where(ok, _lr(o_open, o_closed) - _lr(e_open, e_closed),
                     0.0)
    # Mean per-end openness, from the exact identity above.
    marginal = (o_open - o_closed + 1.0) / 2.0
    return {
        "coordination": coord,
        "direction": frac,
        "direction_expected": frac_exp,
        "direction_excess": excess,
        "direction_logor": logor,
        "directional": coord * (2.0 * frac - 1.0),
        "directional_corrected": coord * excess,
        "marginal": marginal,
    }


def compute_signed_concordance_matrix(
    cell_posterior_paths: dict[str, str],
    chrom_sizes: dict[str, int] | None = None,
    bin_size: int = 250_000,
    min_distance: int = 1_000,
    max_distance: int = 0,
    min_gch_sites: int = 2,
    min_pairs_per_bin: int = 1,
    min_cell_pairs: int = 0,
    min_bin_coverage: float = 0.0,
    statistic: str = "signed",
    concordance_formula: str = "dm",
    baseline: str = "short-range",
    concordance_combine: str = "signed-diff",
    dm_window_mb: float = 0.0,
    dm_dist_bin: int = 10_000,
    use_rwr: bool = True,
    rwr_k: int = 15,
    rwr_restart_prob: float = 0.5,
    rwr_n_refine_passes: int = 0,
    rwr_refinement_basis: str = "raw",
    rwr_auto_select_passes: bool = False,
    rwr_max_auto_passes: int = 5,
    rwr_solver: str = "auto",
    rwr_max_iter: int = 30,
    rwr_tol: float = 1e-4,
    eb_min_obs: int = 50,
    dm_prior: str = "separate",
    batch_labels: pd.Series | pd.DataFrame | None = None,
    batch_correction: str = "none",
    n_threads: int = 4,
    output_dir: str = "output/05_concordance",
    save_outputs: bool = True,
    random_state: int = 42,
    precomputed: dict | None = None,
) -> dict[str, object]:
    """Compute the signed (and optionally one-sided) concordance matrix.

    Implements the full Phase-1 pipeline:

    1.  Aggregate per-(cell, bin) U-3 soft expected counts (linked and
        short-range unlinked) across cells in parallel.
    2.  Compute the dense mean-accessibility matrix (needed for the
        RWR k-NN graph).
    3.  If ``use_rwr``: run pre-Layer-2 RWR imputation on the eight
        soft-count matrices, using a k-NN graph built from
        QN(mean_accessibility) and (optionally) refining via QN(Δ_signed)
        on subsequent passes.
    4.  Fit a Dirichlet-Multinomial EB prior on each pool (linked and
        unlinked) using MoM on the imputed counts.
    5.  Compute Δ_signed (and optionally Δ_open) per (cell, bin) from
        the conjugate posterior plug-in formula.

    Parameters
    ----------
    cell_posterior_paths : dict
        ``cell_name -> *.posteriors.parquet`` from Step 4.
    statistic : {"signed", "open", "both"}
        Which output matrix to produce as the headline.  "open" still
        uses U-3 soft counts but reports Δ_open = E[π_l_open] − p̂²
        (the one-sided variant).  "both" returns both matrices.
    concordance_formula : {"dm", "mle"}
        Estimator for the per-(cell, bin) rates feeding Δ.  "dm" (default)
        uses the Dirichlet-Multinomial posterior mean
        ``(k_i + α_i) / (n + α_total)`` — calibrated but, on POST-RWR
        counts, double-shrinks (RWR already pooled the cell graph, and the
        MoM concentration is inflated by RWR's variance reduction).  "mle"
        uses the plug-in ``k_i / n``, preserving RWR's cell-type-aware
        smoothing; empirically lifts the cell-type axis onto the dominant
        SVD component so ``--skip-top-svd`` is unnecessary.
    rwr_refinement_basis : {"raw", "postnorm"}
        For refinement passes (``rwr_n_refine_passes >= 1``): which form of
        the previous pass's Δ_signed to build the next k-NN graph from.
        "raw" (default) uses per-column-rank(Δ); "postnorm" first applies
        the Step-7 normalization (per-cell rank → per-bin QN → row
        recenter), removing per-cell shifts from the basis.  "postnorm"
        helps under ``concordance_formula="mle"``; it is a no-op gain (or
        slight loss) under "dm".
    use_rwr : bool
        Whether to run pre-Layer-2 RWR imputation.  Default True;
        decision-doc §6 #16 — without RWR, per-(cell, bin) Δ is
        prior-dominated for the typical NOMe-HiC data sparsity.
    rwr_k, rwr_restart_prob, rwr_n_refine_passes, rwr_solver,
    rwr_max_iter, rwr_tol :
        RWR parameters; see :mod:`nome_hic_concordance.imputation`.
    eb_min_obs : int
        Min (cell, bin) observations for DM MoM fit; below this, a
        uniform Dirichlet fallback is used.

    Returns
    -------
    dict with keys:
        - 'delta_signed' : DataFrame (cells × bins) or None
        - 'delta_open'   : DataFrame (cells × bins) or None
        - 'pair_counts'  : DataFrame (cells × bins) of n_linked
        - 'accessibility': DataFrame (cells × bins) of mean accessibility
        - 'imputed_counts' : dict of 8 cells × bins arrays (post-RWR
                             if use_rwr; raw soft counts otherwise)
        - 'priors'       : dict of fitted DM prior parameters
        - 'rwr_diagnostics' : dict (or None if use_rwr=False)
    """
    from joblib import Parallel, delayed

    from . import accessibility as _accessibility_mod
    from . import imputation as _imputation_mod

    if statistic not in ("signed", "open", "both"):
        raise ValueError(
            f"statistic must be 'signed', 'open', or 'both'; got {statistic!r}."
        )
    if concordance_formula not in ("dm", "mle"):
        raise ValueError(
            f"concordance_formula must be 'dm' or 'mle'; got {concordance_formula!r}."
        )
    # 'unlinked' is a deprecated alias for 'short-range' — the baseline pool
    # is short-distance (< min_distance) same-bin read pairs, which are real
    # short-range *contacts*, not non-contacts (the old name was a misnomer).
    if baseline == "unlinked":
        baseline = "short-range"
    if dm_prior not in ("separate", "shared"):
        raise ValueError(
            f"dm_prior must be 'separate' or 'shared'; got {dm_prior!r}.")
    if baseline not in ("short-range", "distance-matched", "distance-pooled"):
        raise ValueError(
            f"baseline must be 'short-range' (alias 'unlinked'), "
            f"'distance-matched', or 'distance-pooled'; got {baseline!r}.")
    # Both non-contact end-pair nulls populate the expected (_DM_KEYS) pool
    # and share the downstream O/E handling; they differ only in how each
    # bin's expected RATE is formed (see _compute_cell_dm_expected mode).
    uses_expected = baseline in ("distance-matched", "distance-pooled")
    if concordance_combine not in ("signed-diff", "log2-ratio", "linked-only",
                                   "concordant-diff", "concordant-signed",
                                   "direction-logor"):
        raise ValueError(
            f"concordance_combine must be 'signed-diff', 'log2-ratio', "
            f"'linked-only', 'concordant-diff', 'concordant-signed', or "
            f"'direction-logor'; got {concordance_combine!r}.")

    out_path = ensure_dir(output_dir) if save_outputs else None
    items = list(cell_posterior_paths.items())

    # Steps 1-3 -- reading every cell's posteriors, aggregating the soft
    # counts and building the accessibility matrix -- do NOT depend on the
    # batch correction, which first acts at step 3.5.  `precomputed` lets a
    # caller sweeping batch settings pay for them once instead of once per
    # setting; the returned dict carries exactly what to pass back.
    if precomputed is not None:
        raw_counts = precomputed["raw_counts"]
        accessibility_df = precomputed["accessibility_raw"].copy()
        cell_names = list(precomputed["cell_names"])
        sorted_bins = list(precomputed["sorted_bins"])
        uses_expected = bool(precomputed.get("uses_expected", uses_expected))
        n_cells, n_bins = len(cell_names), len(sorted_bins)
        logger.info(
            f"Reusing precomputed soft counts + accessibility "
            f"({n_cells} cells x {n_bins} bins); skipping steps 1-3.")
    else:
        # ---- Step 1: aggregate soft counts per cell -------------------------
        logger.info(
            f"Aggregating per-(cell, bin) U-3 soft counts for {len(items)} cells "
            f"(bin_size={bin_size:,}, min_distance={min_distance:,}, "
            f"min_gch_sites={min_gch_sites})"
        )
        if n_threads > 1 and len(items) > 1:
            per_cell = Parallel(n_jobs=n_threads)(
                delayed(_compute_cell_soft_counts)(
                    path, cell_name, bin_size, min_distance, min_gch_sites,
                    max_distance,
                )
                for cell_name, path in items
            )
        else:
            per_cell = [
                _compute_cell_soft_counts(
                    path, cell_name, bin_size, min_distance, min_gch_sites,
                    max_distance,
                )
                for cell_name, path in items
            ]

        # ---- Step 1b: non-contact end-pair expected (O/E baseline), optional --
        if uses_expected:
            dm_mode = "pooled" if baseline == "distance-pooled" else "matched"
            dm_window_bins = int(round((dm_window_mb * 1_000_000) / bin_size))
            if dm_mode == "pooled":
                logger.info(
                    f"Distance-pooled expected baseline: window=±{dm_window_bins} "
                    f"bins ({dm_window_mb} Mb), all null pairs with sep "
                    f">= {min_distance:,} bp pooled per bin"
                )
            else:
                logger.info(
                    f"Distance-matched expected baseline: window=±{dm_window_bins} "
                    f"bins ({dm_window_mb} Mb), dist-bin={dm_dist_bin:,} bp"
                )
            if n_threads > 1 and len(items) > 1:
                per_cell_dm = Parallel(n_jobs=n_threads)(
                    delayed(_compute_cell_dm_expected)(
                        path, cell_name, bin_size, min_distance, min_gch_sites,
                        dm_window_bins, dm_dist_bin, 400, dm_mode, max_distance,
                        random_state,
                    )
                    for cell_name, path in items
                )
            else:
                per_cell_dm = [
                    _compute_cell_dm_expected(
                        path, cell_name, bin_size, min_distance, min_gch_sites,
                        dm_window_bins, dm_dist_bin, 400, dm_mode, max_distance,
                        random_state,
                    )
                    for cell_name, path in items
                ]
            # Merge dm entries into the soft-count entries (same bin ids).
            dm_by_cell = {cn: d for cn, d in per_cell_dm}
            for cn, c in per_cell:
                for bid, dm_entry in dm_by_cell.get(cn, {}).items():
                    c.setdefault(bid, {k: 0.0 for k in _SOFT_COUNT_KEYS}).update(dm_entry)
            # How much of a distance-MATCHED run was actually matched.  A
            # contact whose distance bucket held no re-paired ends is scored
            # against the pooled mean instead, and empty buckets are the long
            # ones -- so this fraction says whether the baseline the run
            # claims is the baseline the long-range bins received.
            if baseline == "distance-matched":
                _fb = sum(e.get("n_dm_fallback", 0.0)
                          for d in dm_by_cell.values() for e in d.values())
                _nd = sum(e.get("n_dm", 0.0)
                          for d in dm_by_cell.values() for e in d.values())
                _frac = _fb / _nd if _nd > 0 else float("nan")
                logger.info(
                    f"Distance-matched null: {_fb:,.0f} of {_nd:,.0f} "
                    f"contacts ({_frac:.1%}) had an empty distance bucket "
                    f"and fell back to the distance-POOLED mean."
                )
                if _frac > 0.2:
                    logger.warning(
                        f"{_frac:.0%} of contacts were scored against the "
                        f"pooled mean, so this run is only partly distance-"
                        f"matched -- and the fallback concentrates at LONG "
                        f"separations, where a bin's re-paired ends thin "
                        f"out. Contacts skew short within a bucket while "
                        f"re-paired ends do not, so those bins keep the "
                        f"positive offset distance matching was meant to "
                        f"remove. Raise --dm-dist-bin-bp (wider buckets "
                        f"fill) or --dm-window-mb (pools ends from "
                        f"neighbouring bins, which is what thin bins need)."
                    )

        # ---- Step 2: assemble cells × bins soft-count matrices --------------
        all_bins: set[str] = set()
        for _, c in per_cell:
            all_bins.update(c.keys())
        sorted_bins = _sort_bins(all_bins, chrom_sizes)

        cell_names = [c for c, _ in per_cell]
        n_cells = len(cell_names)
        n_bins = len(sorted_bins)
        bin_to_col = {b: i for i, b in enumerate(sorted_bins)}

        matrix_keys = list(_SOFT_COUNT_KEYS) + (
            list(_DM_KEYS) if uses_expected else [])
        raw_counts: dict[str, np.ndarray] = {
            k: np.zeros((n_cells, n_bins), dtype=np.float64) for k in matrix_keys
        }
        for row_idx, (_, c) in enumerate(per_cell):
            for bid, entry in c.items():
                j = bin_to_col[bid]
                for k in matrix_keys:
                    raw_counts[k][row_idx, j] = entry.get(k, 0.0)

        logger.info(
            f"Soft-count matrices: {n_cells} cells × {n_bins} bins. "
            f"Total linked pairs across all cells: {raw_counts['n_linked'].sum():.0f}. "
            f"Total unlinked (short-range) pairs: {raw_counts['n_unlinked'].sum():.0f}."
        )

        # ---- Step 2.5: pre-imputation sparsity QC ---------------------------
        # Drop very sparse cells and bins BEFORE building the RWR graph.  A
        # near-empty cell has almost no raw pairs, so its Δ is dominated by
        # imputation; if its (noisy) accessibility profile links it to the
        # wrong neighbours it lands as a spurious island and injects noise
        # into the lower SVD components that fragment the UMAP.  Sparse bins
        # likewise add noise dimensions.  Both filters default OFF (0); set
        # via --min-cell-pairs / --min-bin-coverage.  Filtered here so the
        # accessibility matrix, batch labels, and Δ outputs all inherit the
        # same cell/bin set (accessibility is reindexed to cell_names below).
        if min_cell_pairs and min_cell_pairs > 0:
            total_pairs = (raw_counts["n_linked"] + raw_counts["n_unlinked"]).sum(axis=1)
            keep_cell = total_pairs >= float(min_cell_pairs)
            n_drop = int((~keep_cell).sum())
            if n_drop:
                logger.info(
                    f"Pre-imputation cell QC: dropping {n_drop}/{n_cells} cells with "
                    f"< {min_cell_pairs} informative pairs "
                    f"(kept median = {float(np.median(total_pairs[keep_cell])):.0f}, "
                    f"dropped max = {float(total_pairs[~keep_cell].max()) if n_drop else 0:.0f})."
                )
                cell_names = [c for c, keep in zip(cell_names, keep_cell) if keep]
                raw_counts = {k: v[keep_cell] for k, v in raw_counts.items()}
                n_cells = len(cell_names)

        if min_bin_coverage and min_bin_coverage > 0.0:
            has_pair = (raw_counts["n_linked"] + raw_counts["n_unlinked"]) > 0
            bin_cov = has_pair.mean(axis=0)
            keep_bin = bin_cov >= float(min_bin_coverage)
            n_drop = int((~keep_bin).sum())
            if n_drop:
                logger.info(
                    f"Pre-imputation bin QC: dropping {n_drop}/{n_bins} bins with "
                    f"raw coverage < {min_bin_coverage:.0%} of cells."
                )
                sorted_bins = [b for b, keep in zip(sorted_bins, keep_bin) if keep]
                raw_counts = {k: v[:, keep_bin] for k, v in raw_counts.items()}
                n_bins = len(sorted_bins)

        if n_cells < 2:
            raise ValueError(
                f"After sparsity QC only {n_cells} cell(s) remain; "
                f"lower --min-cell-pairs (was {min_cell_pairs})."
            )
        if n_bins == 0:
            raise ValueError(
                f"After sparsity QC no bins remain; lower --min-bin-coverage "
                f"(was {min_bin_coverage})."
            )

        # ---- Step 3: mean accessibility for the RWR k-NN basis --------------
        accessibility_df, _ = _accessibility_mod.compute_accessibility_matrix(
            cell_posterior_paths, chrom_sizes=chrom_sizes, bin_size=bin_size,
            min_gch_sites=min_gch_sites, n_threads=n_threads,
            output_dir=output_dir, save_outputs=save_outputs,
        )
        # Align accessibility row + column order to the soft-count matrices so
        # the RWR k-NN graph is built in the same cell order.
        accessibility_df = accessibility_df.reindex(
            index=cell_names, columns=sorted_bins,
        )

    accessibility_raw = accessibility_df.copy()

    # ---- Step 3.5: optional batch correction on accessibility ----------
    # Applied before QN + k-NN so batch structure doesn't leak into the
    # cell-cell similarity matrix.  Diagnostic batch-ANOVA F-stats are
    # logged (median per bin) before and after correction so the user
    # can see whether the chosen method actually neutralised batch.
    if batch_correction != "none" and batch_labels is not None:
        from . import batch_correction as _bc_mod
        # Pre-correction diagnostic.
        f_pre = _bc_mod.batch_anova_f_per_bin(accessibility_df, batch_labels)
        logger.info(
            f"Pre-batch-correction ANOVA F over batches: "
            f"median = {float(np.nanmedian(f_pre)):.2f}, "
            f"90th pct = {float(np.nanpercentile(f_pre, 90)):.2f}"
        )
        accessibility_df = _bc_mod.apply_batch_correction(
            accessibility_df, batch_labels, method=batch_correction,
            random_state=random_state,
        )
        # Post-correction diagnostic.
        f_post = _bc_mod.batch_anova_f_per_bin(accessibility_df, batch_labels)
        logger.info(
            f"Post-batch-correction ANOVA F over batches: "
            f"median = {float(np.nanmedian(f_post)):.2f}, "
            f"90th pct = {float(np.nanpercentile(f_post, 90)):.2f}"
        )
        if save_outputs and out_path is not None:
            pd.DataFrame({
                "bin": accessibility_df.columns,
                "F_pre":  f_pre.values,
                "F_post": f_post.values,
            }).to_csv(out_path / "batch_correction_anova_f.tsv",
                      sep="\t", index=False)

    # ---- Step 4 + 5: RWR (optional) + Layer 2 ----------------------------
    fitted_priors: dict[str, dict] = {}

    def _fit_and_rates(
        imputed: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit the DM priors on ``imputed`` and return the four co-state rates
        ``(o_open, o_closed, e_open, e_closed)``.

        Side effect: (re)populates ``fitted_priors`` for the counts passed in.
        Split out of :func:`_layer2` so the post-pipeline component matrices
        can be derived from the *returned* imputed counts — under
        ``rwr_auto_select_passes`` the pipeline may hand back an earlier
        pass, so capturing them as a side effect of the last ``_layer2``
        call would silently pair components from one pass with a Δ from
        another.
        """
        # Build long-format DataFrames for the prior fits.
        n_l_flat = imputed["n_linked"].ravel()
        long_linked = pd.DataFrame({
            "k_open":   imputed["k_open"].ravel(),
            "k_closed": imputed["k_closed"].ravel(),
            "k_mixed":  imputed["k_mixed"].ravel(),
            "n_linked": n_l_flat,
        })
        alpha_l, info_l = _fit_dirichlet_multinomial_eb_prior(
            long_linked,
            open_col="k_open", closed_col="k_closed",
            mixed_col="k_mixed", n_col="n_linked",
            min_obs=eb_min_obs,
        )

        n_u_flat = imputed["n_unlinked"].ravel()
        long_unlinked = pd.DataFrame({
            "k_open":   imputed["k_u_open"].ravel(),
            "k_closed": imputed["k_u_closed"].ravel(),
            "k_mixed":  imputed["k_u_mixed"].ravel(),
            "n_linked": n_u_flat,    # column name is just for the helper
        })
        alpha_u, info_u = _fit_dirichlet_multinomial_eb_prior(
            long_unlinked,
            open_col="k_open", closed_col="k_closed",
            mixed_col="k_mixed", n_col="n_linked",
            min_obs=eb_min_obs,
        )

        fitted_priors["linked"] = {"alpha": alpha_l, "info": info_l}
        fitted_priors["unlinked"] = {"alpha": alpha_u, "info": info_u}

        # Non-contact expected pool (distance-matched or distance-pooled).
        alpha_dm = None
        long_dm = None
        if uses_expected and "k_dm_open" in imputed:
            long_dm = pd.DataFrame({
                "k_open":   imputed["k_dm_open"].ravel(),
                "k_closed": imputed["k_dm_closed"].ravel(),
                "k_mixed":  imputed["k_dm_mixed"].ravel(),
                "n_linked": imputed["n_dm"].ravel(),
            })
            alpha_dm, info_dm = _fit_dirichlet_multinomial_eb_prior(
                long_dm, open_col="k_open", closed_col="k_closed",
                mixed_col="k_mixed", n_col="n_linked", min_obs=eb_min_obs,
            )
            fitted_priors["dm_expected"] = {"alpha": alpha_dm, "info": info_dm}

        # THE PRIOR GAP, AND WHY IT IS A FLOOR UNDER C.
        #
        # O and E are both Dirichlet-Multinomial posterior means,
        # (alpha_i + k_i) / (alpha_0 + n), and n_dm == n_linked by
        # construction, so the two share a denominator and
        #
        #   C = (O_open + O_closed) - (E_open + E_closed)
        #     = [ (a_l_conc - a_dm_conc) + (k_conc - k_dm_conc) ] / (a_0 + n)
        #
        # The first term does not depend on the data at all.  Fitting a
        # separate EB prior per pool sets each pool's shrinkage TARGET to
        # its own global mean, and those means differ by exactly the
        # genome-wide average coordination -- so a bin with no information
        # is handed the global average instead of zero, and C cannot reach
        # zero however good the null is.  Post-RWR the imputed n is
        # typically < 1, which makes the posterior ~99% prior for nearly
        # every entry, so this term dominates: on a real run the gap was
        # 0.0113 and the median C was 0.011293.
        #
        # dm_prior="shared" fits ONE prior on both pools.  The gap becomes
        # exactly 0, the shared denominator cancels it at every n, and
        # C = 0 means what it should: the two pools agree.
        if alpha_dm is not None:
            _cl = ((alpha_l["open"] + alpha_l["closed"]) / sum(alpha_l.values()))
            _cd = ((alpha_dm["open"] + alpha_dm["closed"])
                   / sum(alpha_dm.values()))
            # L = log(O_open/O_closed) - log(E_open/E_closed) cancels the
            # shared DENOMINATOR rather than sharing it, so the prior enters
            # as the log-odds ratio of the two priors' own open:closed
            # splits.  Same defect, same n=0 limit, same cure: with one
            # prior it is exactly 0.
            _llor = float(np.log(alpha_l["open"] / alpha_l["closed"])
                          - np.log(alpha_dm["open"] / alpha_dm["closed"]))
            fitted_priors["prior_gap"] = {
                "concordant_linked": float(_cl),
                "concordant_dm_expected": float(_cd),
                "gap": float(_cl - _cd),
                "implied_C_floor_at_n0": float(_cl - _cd),
                "implied_logor_floor_at_n0": _llor,
                "dm_prior": str(dm_prior),
            }
            if dm_prior == "shared":
                alpha_sh, info_sh = _fit_dirichlet_multinomial_eb_prior(
                    pd.concat([long_linked, long_dm], ignore_index=True),
                    open_col="k_open", closed_col="k_closed",
                    mixed_col="k_mixed", n_col="n_linked",
                    min_obs=eb_min_obs,
                )
                fitted_priors["shared"] = {"alpha": alpha_sh,
                                           "info": info_sh}
                alpha_l = alpha_dm = alpha_sh
                logger.info(
                    f"DM prior: SHARED between the linked and expected "
                    f"pools (nu={sum(alpha_sh.values()):.1f}). The "
                    f"separate fits differed by {_cl - _cd:+.6f} in "
                    f"concordant mass, which was an additive floor under C."
                )
            elif abs(_cl - _cd) > 0:
                logger.warning(
                    f"DM prior: the linked and expected pools were fitted "
                    f"SEPARATELY and their concordant masses differ by "
                    f"{_cl - _cd:+.6f} ({_cl:.4f} vs {_cd:.4f}). That "
                    f"difference is an additive floor under C at every "
                    f"bin, and the whole of C where the imputed n is small "
                    f"-- compare it against median(C) before reading C as "
                    f"coordination. Use --dm-prior shared to remove it."
                )

        # Compute the per-(cell, bin) Δ_signed.  Two estimators:
        #
        #   "dm"  — Dirichlet-Multinomial posterior plug-in:
        #             E[π_i] = (k_i + α_i) / (n + α_total).
        #           Shrinks each rate toward the global prior mean.  This
        #           is the classical, calibrated estimator — but on
        #           POST-RWR counts it double-shrinks: RWR already pooled
        #           evidence across the cell graph (cell-type-aware), and
        #           the MoM-fit concentration (α_total ~ 100) is inflated
        #           because RWR reduced the very cross-cell variance MoM
        #           reads.  With fractional imputed n (often < 1), the
        #           posterior is ~99% prior, crushing cell-type signal
        #           onto a thin PC.
        #
        #   "mle" — maximum-likelihood plug-in: E[π_i] = k_i / n.
        #           Keeps RWR's cell-type-aware smoothing intact instead
        #           of layering a cell-type-blind global shrinkage on top.
        #           Empirically lifts the cell-type axis onto the dominant
        #           SVD component (no --skip-top-svd needed).  Robust to
        #           the small-n variance blow-up via the downstream
        #           per-cell rank normalization in Step 7.  Entries with
        #           n == 0 contribute 0 to their side (matches the
        #           independence-baseline convention).
        # Per-pool (open, closed) rates under the chosen estimator.
        def _side_rates(ko, kc, km, n, alpha):
            if concordance_formula == "mle":
                safe = np.where(n > 0, n, np.nan)
                return (np.nan_to_num(ko / safe, nan=0.0),
                        np.nan_to_num(kc / safe, nan=0.0))
            pm = _dm_posterior_means(
                {"k_open": ko, "k_closed": kc, "k_mixed": km, "n": n},
                alpha, ("k_open", "k_closed", "k_mixed"), "n")
            return pm["open"], pm["closed"]

        # Observed = the long-range (linked) contacts.
        o_open, o_closed = _side_rates(
            imputed["k_open"], imputed["k_closed"], imputed["k_mixed"],
            imputed["n_linked"], alpha_l)

        # Baseline (E): short-range pool, or non-contact expected
        # (distance-matched / distance-pooled) pool.
        if uses_expected:
            e_open, e_closed = _side_rates(
                imputed["k_dm_open"], imputed["k_dm_closed"],
                imputed["k_dm_mixed"], imputed["n_dm"], alpha_dm)
        else:
            e_open, e_closed = _side_rates(
                imputed["k_u_open"], imputed["k_u_closed"],
                imputed["k_u_mixed"], imputed["n_unlinked"], alpha_u)

        return o_open, o_closed, e_open, e_closed


    def _layer2(imputed: dict[str, np.ndarray]) -> pd.DataFrame:
        """Layer-2 callback used by run_rwr_pipeline (or called directly when
        use_rwr=False): fit the DM priors, then combine O and E into the
        headline Δ matrix."""
        o_open, o_closed, e_open, e_closed = _fit_and_rates(imputed)

        # Combine observed (O) and expected (E).
        if concordance_combine == "linked-only":
            # Long-range (linked) contacts ONLY — the baseline E is NOT
            # subtracted.  A diagnostic to see whether the concordance UMAP is
            # driven by the short-range baseline: this is the raw marginal-laden
            # (O_open − O_closed), which is expected to correlate MORE strongly
            # with the 1D accessibility baseline, not less.
            delta_signed = o_open - o_closed
        elif concordance_combine == "log2-ratio":
            # log2(O/E) per co-state, then the open-vs-closed contrast.  DM
            # keeps rates strictly in (0,1); for MLE we floor with a tiny eps.
            eps = 1e-6
            log2 = lambda x: np.log2(np.clip(x, eps, None))
            delta_signed = (log2(o_open) - log2(e_open)) \
                - (log2(o_closed) - log2(e_closed))
        elif concordance_combine == "concordant-diff":
            # Excess CONCORDANT mass (co-open + co-closed) over the null =
            # the mixed DEFICIT (E_mixed − O_mixed) = 2·Cov(p_i, p_j): the
            # marginal-free coordination MAGNITUDE (sign = concordant vs
            # anti-concordant).  Unlike 'signed-diff', the (open − closed)
            # contrast does NOT cancel the coordination here; unlike
            # 'log2-ratio' there is no marginal-dependent scaling, so it stays
            # sensitive at ~50% level where log2-ratio vanishes.  This is the
            # theoretically cleanest coordination statistic against an O/E null.
            delta_signed = (o_open + o_closed) - (e_open + e_closed)
        elif concordance_combine == "direction-logor":
            # L = log(O_open/O_closed) - log(E_open/E_closed), the natural
            # parameter of the 2x2 table.  Identical to 'log2-ratio' up to
            # the base (log2-ratio = L / ln 2) with ONE difference that
            # matters on sparse data: where a bin has no observed concordant
            # mass, log2-ratio still returns log2(E_closed/E_open) from the
            # clip -- a confident direction read off the null, in a bin with
            # no data.  decompose_components returns 0 there instead.
            delta_signed = decompose_components(
                o_open, o_closed, e_open, e_closed)["direction_logor"]
        elif concordance_combine == "concordant-signed":
            # D = C·(2F−1): the coordination magnitude of 'concordant-diff',
            # SIGNED by which co-state realizes it.  Gives the single
            # co-closed (−) → mixed (0) → co-open (+) scale.  See _decompose
            # for why the sign cannot come from the excess itself.
            delta_signed = decompose_components(o_open, o_closed, e_open, e_closed)[
                "directional"]
        else:  # 'signed-diff' (legacy): (O_open−O_closed) − (E_open−E_closed)
            delta_signed = (o_open - o_closed) - (e_open - e_closed)

        # The min_pairs_per_bin filter only makes sense WITHOUT RWR.
        # With RWR on, every (cell, bin) has an imputed value blended
        # from neighbours via the QN(accessibility) k-NN graph; those
        # values carry real cell-type signal even when the cell itself
        # had raw n_linked = 0.  That's exactly what RWR is for.  And
        # imputed n_linked is a *fractional* weighted aggregate that
        # is typically < 1, so using it as the threshold would discard
        # almost all of the entries RWR was meant to fill in.  Hence:
        # filter only when RWR is off.  (When use_rwr=False, the
        # imputed dict is just the raw counts, so the threshold has
        # its original n>=N meaning.)
        if not use_rwr:
            invalid = imputed["n_linked"] < float(min_pairs_per_bin)
            delta_signed = np.where(invalid, np.nan, delta_signed)

        return pd.DataFrame(
            delta_signed, index=cell_names, columns=sorted_bins,
        )

    rwr_diagnostics = None
    if use_rwr and n_cells >= 2:
        logger.info(
            f"Running pre-Layer-2 RWR imputation "
            f"(k={rwr_k}, restart={rwr_restart_prob}, "
            f"refine_passes={rwr_n_refine_passes})"
        )
        imputed_counts, delta_signed_df, rwr_diagnostics = (
            _imputation_mod.run_rwr_pipeline(
                soft_count_matrices=raw_counts,
                initial_feature_matrix=accessibility_df.fillna(np.nan),
                layer2_fn=_layer2,
                n_refine_passes=rwr_n_refine_passes,
                refinement_basis=rwr_refinement_basis,
                auto_select_passes=rwr_auto_select_passes,
                max_auto_passes=rwr_max_auto_passes,
                k=rwr_k, restart_prob=rwr_restart_prob,
                solver=rwr_solver, max_iter=rwr_max_iter, tol=rwr_tol,
                output_dir=output_dir if save_outputs else None,
                random_state=random_state,
            )
        )
    else:
        if use_rwr and n_cells < 2:
            logger.warning(
                f"Skipping RWR: only {n_cells} cell(s); RWR requires >= 2."
            )
        # Run Layer 2 directly on the raw soft counts.
        imputed_counts = raw_counts
        delta_signed_df = _layer2(imputed_counts)

    # ---- Component matrices (the interpretable 2 d.o.f. + the signed mix) --
    # Derived from the counts the pipeline actually returned (see
    # _fit_and_rates), so they always correspond to the returned Δ — and the
    # re-fit leaves `fitted_priors` describing those same counts.  Emitted for
    # every combine mode: they are the axes to visualise (coordination C and
    # marginal p̄ are the two independent d.o.f.; directional D is their
    # signed combination), independent of which one is the headline Δ.
    _o_o, _o_c, _e_o, _e_c = _fit_and_rates(imputed_counts)
    components = {
        name: pd.DataFrame(arr, index=cell_names, columns=sorted_bins)
        for name, arr in decompose_components(_o_o, _o_c, _e_o, _e_c).items()
    }
    _pol = coordination_polar(components["coordination"].to_numpy(float),
                              components["direction_excess"].to_numpy(float))
    polar_info = {k: v for k, v in _pol.items()
                  if k not in ("R", "theta", "p")}
    logger.info(
        "Coordination plane: scale(C) = %.4g, scale(F-F_exp) = %.4g, "
        "rho = %+.3f over %d cell-bins.  R is Mahalanobis in these units, so "
        "R^2 is chi-square(2) and polar_p is one test across both axes.  A "
        "large |rho| means the two axes are not measuring separate things in "
        "this run and the plane is nearly a line.",
        polar_info.get("scale_c", float("nan")),
        polar_info.get("scale_f", float("nan")),
        polar_info.get("rho", float("nan")),
        polar_info.get("n_used", 0))
    # theta is CIRCULAR, so it is also emitted as its sine and cosine.
    # Euclidean distance on degrees is wrong exactly where the interesting
    # states are: +179 and -179 are two degrees apart on the circle and 358
    # apart numerically, and that tear runs straight through the
    # anti-coordinated sector.  Anything that embeds or clusters direction
    # must use the pair, never the degrees.
    _th = np.radians(_pol["theta"])
    _plane, _plane_cols, _plane_info = coordination_plane_matrix(
        components["coordination"].to_numpy(float),
        components["direction_excess"].to_numpy(float), list(sorted_bins),
        scale_c=_pol.get("scale_c"), scale_f=_pol.get("scale_f"))
    if _plane is not None:
        components["plane"] = pd.DataFrame(_plane, index=cell_names,
                                           columns=_plane_cols)
    for _key, _arr in (("polar_R", _pol["R"]), ("polar_theta", _pol["theta"]),
                       ("polar_theta_sin", np.sin(_th)),
                       ("polar_theta_cos", np.cos(_th)),
                       ("polar_p", _pol["p"])):
        components[_key] = pd.DataFrame(_arr, index=cell_names,
                                        columns=sorted_bins)
    for _df in components.values():
        _df.columns.name = "bin"
        _df.index.name = "cell"

    # ---- Optionally compute the one-sided Δ_open -------------------------
    delta_open_df = None
    if statistic in ("open", "both"):
        # One-sided Δ_open uses per-end marginal p̂ for the independence
        # baseline (not the pair-based unlinked DM).  This matches the
        # existing one-sided convention so users can compare with the
        # legacy --concordance-method betabinom output.
        alpha_l = fitted_priors["linked"]["alpha"]
        if concordance_formula == "mle":
            n_l = imputed_counts["n_linked"]
            safe_n_l = np.where(n_l > 0, n_l, np.nan)
            pm_l = {"open": np.nan_to_num(
                imputed_counts["k_open"] / safe_n_l, nan=0.0)}
        else:
            pm_l = _dm_posterior_means(
                imputed_counts, alpha_l, ("k_open", "k_closed", "k_mixed"),
                "n_linked",
            )
        n_u = imputed_counts["n_unlinked"]
        # p̂ from the per-end-rate identity:
        # sum_pairs (p_l + p_r) = 2 · k_u_open + k_u_mixed
        # p̂ = sum_pairs (p_l + p_r) / (2 · n_unlinked)
        p_hat = np.where(
            n_u > 0,
            (2.0 * imputed_counts["k_u_open"] + imputed_counts["k_u_mixed"])
            / (2.0 * n_u),
            np.nan,
        )
        delta_open_arr = pm_l["open"] - p_hat ** 2
        # Same filter semantics as for delta_signed: only meaningful when
        # RWR is off (see comment in _layer2 above).
        if not use_rwr:
            invalid = imputed_counts["n_linked"] < float(min_pairs_per_bin)
            delta_open_arr = np.where(invalid, np.nan, delta_open_arr)
        delta_open_df = pd.DataFrame(
            delta_open_arr, index=cell_names, columns=sorted_bins,
        )

    # ---- pair_counts (n_linked from raw, integer) ------------------------
    pair_counts_df = pd.DataFrame(
        raw_counts["n_linked"].astype(np.int64),
        index=cell_names, columns=sorted_bins,
    )
    pair_counts_df.columns.name = "bin"
    pair_counts_df.index.name = "cell"

    # Final pretty-up.
    delta_signed_df.columns.name = "bin"
    delta_signed_df.index.name = "cell"
    if delta_open_df is not None:
        delta_open_df.columns.name = "bin"
        delta_open_df.index.name = "cell"

    # ---- Save outputs ----------------------------------------------------
    if save_outputs and out_path is not None:
        # Headline matrix (whichever statistic the user asked for).
        if statistic == "signed":
            delta_signed_df.to_parquet(out_path / "concordance_signed.parquet")
            delta_signed_df.to_csv(out_path / "concordance_signed.tsv", sep="\t")
        elif statistic == "open":
            assert delta_open_df is not None
            delta_open_df.to_parquet(out_path / "concordance_open.parquet")
            delta_open_df.to_csv(out_path / "concordance_open.tsv", sep="\t")
        else:  # both
            delta_signed_df.to_parquet(out_path / "concordance_signed.parquet")
            delta_open_df.to_parquet(out_path / "concordance_open.parquet")
            delta_signed_df.to_csv(out_path / "concordance_signed.tsv", sep="\t")
            delta_open_df.to_csv(out_path / "concordance_open.tsv", sep="\t")

        pair_counts_df.to_parquet(out_path / "pair_counts_per_bin.parquet")
        pair_counts_df.to_csv(out_path / "pair_counts_per_bin.tsv", sep="\t")

        # Component matrices — the axes for the companion UMAPs.
        for _name, _df in components.items():
            _df.to_parquet(out_path / f"concordance_{_name}.parquet")
            _df.to_csv(out_path / f"concordance_{_name}.tsv", sep="\t")

        # Soft counts + priors for diagnostics / reproducibility.
        soft_long = []
        for row_idx, cell in enumerate(cell_names):
            for j, bid in enumerate(sorted_bins):
                if raw_counts["n_linked"][row_idx, j] == 0 and \
                        raw_counts["n_unlinked"][row_idx, j] == 0:
                    continue
                soft_long.append({
                    "cell": cell, "bin": bid,
                    **{k: float(raw_counts[k][row_idx, j]) for k in _SOFT_COUNT_KEYS},
                })
        if soft_long:
            pd.DataFrame(soft_long).to_parquet(
                out_path / "soft_counts_raw.parquet", index=False,
            )

        import json
        with open(out_path / "dirichlet_multinomial_prior.json", "w") as fh:
            json.dump(fitted_priors, fh, indent=2, default=_json_safe)

    logger.info(
        f"Signed concordance matrix: {n_cells} cells × {n_bins} bins "
        f"({delta_signed_df.notna().sum().sum():,} non-NA entries)"
    )

    return {
        "delta_signed":   delta_signed_df,
        "delta_open":     delta_open_df,
        "pair_counts":    pair_counts_df,
        "accessibility":  accessibility_df,
        "imputed_counts": imputed_counts,
        "priors":         fitted_priors,
        "rwr_diagnostics": rwr_diagnostics,
        # Everything a batch-setting sweep needs to skip steps 1-3 next time.
        "raw_counts": raw_counts,
        "accessibility_raw": accessibility_raw,
        "cell_names": list(cell_names),
        "sorted_bins": list(sorted_bins),
        "uses_expected": uses_expected,
        # C (marginal-free coordination), F (realized direction),
        # D = C·(2F−1) (directional coordination), p̄ (mean per-end openness).
        # D' = C·(F−F_exp) is the one to interpret: D's zero sits at F = 0.5,
        # which is only "no preference" when p̄ = 0.5, so on sparse data D
        # collapses to a scaled −C (see _decompose).
        # The polar reading of the (C, dF) plane.  R separates
        # 'uncoordinated' from all three coordinated states -- including
        # 'coordinated balanced', which neither axis can do alone -- and
        # theta names which.  polar_p is a chi-square(2) test across both.
        "plane":                 components.get("plane"),
        "polar_R":               components["polar_R"],
        "polar_theta":           components["polar_theta"],
        "polar_theta_sin":       components["polar_theta_sin"],
        "polar_theta_cos":       components["polar_theta_cos"],
        "polar_p":               components["polar_p"],
        "coordination":          components["coordination"],
        "direction":             components["direction"],
        "direction_expected":    components["direction_expected"],
        "direction_excess":      components["direction_excess"],
        # Same direction as direction_excess, on the scale where the
        # variance does not depend on F.  Not an independent axis: see
        # _decompose.  Direction only -- magnitude is 'coordination'.
        "direction_logor":       components["direction_logor"],
        "directional":           components["directional"],
        "directional_corrected": components["directional_corrected"],
        "marginal":              components["marginal"],
    }


def _robust_sd(x):
    """MAD-based scale.  The (C, dF) plane has heavy tails from thin bins,
    and a plain sd would let those set the units for every other bin."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    mad = float(np.median(np.abs(x - np.median(x))))
    if mad > 0:
        return 1.4826 * mad
    sd = float(np.std(x))
    return sd if sd > 0 else np.nan


def coordination_polar(coordination, direction_excess, scale_c=None,
                       scale_f=None, rho=None):
    """Departure from the null in the (C, dF) plane: how far, and which way.

    C and dF answer different questions and each is blind where the other
    sees:

      C  alone cannot tell coordinated co-open from coordinated co-closed --
         Cov is unchanged by swapping open for closed, so both score the same.
      dF alone cannot tell coordinated co-open from UNCOORDINATED.  It is a
         ratio describing where the concordant mass sits and does not know
         whether that mass exceeds chance, so it assigns a direction to a bin
         with no coordination to have a direction about.

    Neither can be dropped and no monotone function of one recovers the
    other: the state is two-dimensional.  What CAN be done is to change
    coordinates.  Standardise both axes, then read the plane in polar form:

      R      distance from the null in either sense.  R ~ 0 only when BOTH
             axes are null, so it separates 'uncoordinated' from all three
             coordinated states INCLUDING 'coordinated balanced' -- which is
             what a single magnitude has to do to be worth having.
      theta  which kind of departure.  0 deg = pure magnitude, no directional
             preference (coordinated balanced); +90 = direction shifted
             co-open with no excess mass; 180 = anti-coordinated; -90 =
             shifted co-closed.  The four cells become four sectors, and
             everything between them is representable rather than forced
             into a corner.

    R is Mahalanobis, not Euclidean: C and dF are correlated in real data,
    and an unwhitened radius counts the shared part twice, calling a bin
    extreme for being ordinary along the correlated direction.  Whitened,
    R^2 is chi-square with 2 d.o.f. under the null, so ``p`` is a calibrated
    test of "this bin departs from the null in coordination space" -- one
    p-value across both axes, which is what a summary statistic owes you.

    Scales and rho are estimated ACROSS THE WHOLE MATRIX by default, so they
    are set by millions of entries dominated by noise and no single bin can
    move its own units.  Pass them explicitly to put two runs on one plane
    instead of letting each choose units and be silently incomparable.
    """
    C = np.asarray(coordination, float)
    dF = np.asarray(direction_excess, float)
    if C.shape != dF.shape:
        raise ValueError(f"C has shape {C.shape}, direction excess "
                         f"{dF.shape}")
    ok = np.isfinite(C) & np.isfinite(dF)
    sc = _robust_sd(C[ok]) if scale_c is None else float(scale_c)
    sf = _robust_sd(dF[ok]) if scale_f is None else float(scale_f)
    info = {"scale_c": sc, "scale_f": sf, "n_used": int(ok.sum())}
    if not np.isfinite(sc) or not np.isfinite(sf):
        nan = np.full(C.shape, np.nan)
        info["note"] = "one axis has no spread; polar form undefined"
        return {"R": nan, "theta": nan, "p": nan, **info}
    zc, zf = C / sc, dF / sf
    if rho is None:
        rho = (float(np.corrcoef(zc[ok], zf[ok])[0, 1])
               if int(ok.sum()) > 2 else 0.0)
    rho = float(np.clip(rho if np.isfinite(rho) else 0.0, -0.99, 0.99))
    info["rho"] = rho
    with np.errstate(invalid="ignore"):
        r2 = (zc ** 2 - 2.0 * rho * zc * zf + zf ** 2) / (1.0 - rho ** 2)
        r2 = np.where(np.isfinite(r2) & (r2 > 0), r2,
                      np.where(ok, 0.0, np.nan))
        R = np.sqrt(r2)
        theta = np.degrees(np.arctan2(zf, zc))
        pv = np.exp(-r2 / 2.0)          # chi-square(2) upper tail
    theta = np.where(ok, theta, np.nan)
    return {"R": R, "theta": theta, "p": np.where(ok, pv, np.nan), **info}


def coordination_plane_matrix(coordination, direction_excess, bins,
                              scale_c=None, scale_f=None):
    """The two axes side by side, as ONE matrix to embed.

    Two separate UMAPs answer two questions and leave the reader to join
    them by eye.  One embedding over both axes answers the question actually
    being asked -- do cells differ in WHERE they sit on the coordination
    plane -- and SVD before UMAP does the joining.

    Cartesian, not polar.  (R, theta) is the same information, but theta is
    circular and an SVD on degrees tears the axis at +-180: on a fixture
    where two cell types sit at +175 and -175 -- ten degrees apart -- the
    degree stack puts them TWICE as far apart as a type 175 degrees away
    (ratio 2.00), while the Cartesian stack gives 0.09.  Since
    R*cos(theta) and R*sin(theta) ARE the standardised C and F - F_exp, the
    faithful representation is just the two matrices concatenated, and no
    polar step is needed at all.

    Correlation between the axes needs no whitening here.  SVD finds its own
    basis, so a shared component becomes the leading singular vector -- an
    interpretable axis rather than something to divide out first.

    Columns are ``<bin>|C`` and ``<bin>|dF``.  This matrix is for EMBEDDING;
    it is twice as wide as the genome and its columns are not bin labels, so
    do not feed it to anything that expects to parse coordinates out of them.
    """
    C = np.asarray(coordination, float)
    dF = np.asarray(direction_excess, float)
    if C.shape != dF.shape:
        raise ValueError(f"C has shape {C.shape}, direction excess "
                         f"{dF.shape}")
    if C.shape[1] != len(bins):
        raise ValueError(f"{C.shape[1]} columns but {len(bins)} bin labels")
    ok = np.isfinite(C) & np.isfinite(dF)
    sc = _robust_sd(C[ok]) if scale_c is None else float(scale_c)
    sf = _robust_sd(dF[ok]) if scale_f is None else float(scale_f)
    if not np.isfinite(sc) or not np.isfinite(sf):
        return None, [], {"note": "one axis has no spread"}
    stacked = np.hstack([C / sc, dF / sf])
    names = [f"{b}|C" for b in bins] + [f"{b}|dF" for b in bins]
    return stacked, names, {"scale_c": sc, "scale_f": sf,
                            "n_bins": int(C.shape[1])}


COORD_CLASSES = ("uncoordinated", "coordinated balanced", "co-open shifted",
                 "co-closed shifted", "anti-coordinated")


def coordination_class(R, theta, r_cut=2.0, sector=30.0):
    """Name each bin's sector.  Returns an integer code into COORD_CLASSES.

    r_cut is a distance in standardised units, so 2.0 is roughly the 86th
    percentile of a chi-square(2): nearer the origin than that is not
    distinguishable from the null on either axis and is called uncoordinated
    whatever its angle.  Naming a sector for a bin at the origin is exactly
    how dF alone goes wrong.
    """
    R = np.asarray(R, float)
    th = np.asarray(theta, float)
    out = np.zeros(R.shape, dtype=np.int8)
    far = np.isfinite(R) & np.isfinite(th) & (R >= r_cut)
    a = np.abs(th)
    out[far & (a <= sector)] = 1                                # balanced
    out[far & (th > sector) & (th < 180.0 - sector)] = 2        # co-open
    out[far & (th < -sector) & (th > -180.0 + sector)] = 3      # co-closed
    out[far & (a >= 180.0 - sector)] = 4                        # anti-coord
    out[~np.isfinite(R) | ~np.isfinite(th)] = -1
    return out


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Unserializable: {type(obj).__name__}")
