"""Per-bin concordance: Pearson r of paired read-end posteriors.

For each (cell, bin) with at least ``min_pairs_per_bin`` qualifying pairs,
we compute the Pearson correlation of the two posterior columns. Bins are
assigned using the midpoint of the two read ends. Pairs below the distance
or GCH-site thresholds are discarded.
"""

from __future__ import annotations

import logging
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


def _compute_cell_soft_counts(
    posteriors_path: str,
    cell_name: str,
    bin_size: int,
    min_distance: int,
    min_gch_sites: int,
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

    is_linked = (distance[qualify] >= int(min_distance))
    chrom_q = chrom[qualify]
    bin_q = bin1[qualify]

    counts: dict[str, dict[str, float]] = {}

    # Vectorised aggregation per (chrom, bin) for linked + unlinked sides.
    pair_df = pd.DataFrame({
        "chrom": chrom_q,
        "bin": bin_q,
        "is_linked": is_linked,
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
    unlinked_g = pair_df.loc[~pair_df["is_linked"]].groupby(
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


def compute_signed_concordance_matrix(
    cell_posterior_paths: dict[str, str],
    chrom_sizes: dict[str, int] | None = None,
    bin_size: int = 250_000,
    min_distance: int = 1_000,
    min_gch_sites: int = 2,
    min_pairs_per_bin: int = 1,
    statistic: str = "signed",
    concordance_formula: str = "dm",
    use_rwr: bool = True,
    rwr_k: int = 15,
    rwr_restart_prob: float = 0.5,
    rwr_n_refine_passes: int = 0,
    rwr_refinement_basis: str = "raw",
    rwr_solver: str = "auto",
    rwr_max_iter: int = 30,
    rwr_tol: float = 1e-4,
    eb_min_obs: int = 50,
    batch_labels: pd.Series | None = None,
    batch_correction: str = "none",
    n_threads: int = 4,
    output_dir: str = "output/05_concordance",
    save_outputs: bool = True,
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

    out_path = ensure_dir(output_dir) if save_outputs else None
    items = list(cell_posterior_paths.items())

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
            )
            for cell_name, path in items
        )
    else:
        per_cell = [
            _compute_cell_soft_counts(
                path, cell_name, bin_size, min_distance, min_gch_sites,
            )
            for cell_name, path in items
        ]

    # ---- Step 2: assemble cells × bins soft-count matrices --------------
    all_bins: set[str] = set()
    for _, c in per_cell:
        all_bins.update(c.keys())
    sorted_bins = _sort_bins(all_bins, chrom_sizes)

    cell_names = [c for c, _ in per_cell]
    n_cells = len(cell_names)
    n_bins = len(sorted_bins)
    bin_to_col = {b: i for i, b in enumerate(sorted_bins)}

    raw_counts: dict[str, np.ndarray] = {
        k: np.zeros((n_cells, n_bins), dtype=np.float64) for k in _SOFT_COUNT_KEYS
    }
    for row_idx, (_, c) in enumerate(per_cell):
        for bid, entry in c.items():
            j = bin_to_col[bid]
            for k in _SOFT_COUNT_KEYS:
                raw_counts[k][row_idx, j] = entry[k]

    logger.info(
        f"Soft-count matrices: {n_cells} cells × {n_bins} bins. "
        f"Total linked pairs across all cells: {raw_counts['n_linked'].sum():.0f}. "
        f"Total unlinked (short-range) pairs: {raw_counts['n_unlinked'].sum():.0f}."
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

    def _layer2(imputed: dict[str, np.ndarray]) -> pd.DataFrame:
        """Layer-2 callback used by run_rwr_pipeline (or called directly
        when use_rwr=False).  Fits DM priors and computes the headline
        Δ_signed matrix.  Side effect: populates ``fitted_priors``.
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
        if concordance_formula == "mle":
            def _mle_signed(k_open, k_closed, n):
                safe_n = np.where(n > 0, n, np.nan)
                s = (k_open - k_closed) / safe_n
                return np.nan_to_num(s, nan=0.0)
            S_l = _mle_signed(imputed["k_open"], imputed["k_closed"],
                              imputed["n_linked"])
            S_u = _mle_signed(imputed["k_u_open"], imputed["k_u_closed"],
                              imputed["n_unlinked"])
            delta_signed = S_l - S_u
        else:
            pm_l = _dm_posterior_means(
                imputed, alpha_l, ("k_open", "k_closed", "k_mixed"), "n_linked",
            )
            pm_u = _dm_posterior_means(
                {"k_open": imputed["k_u_open"], "k_closed": imputed["k_u_closed"],
                 "k_mixed": imputed["k_u_mixed"], "n_unlinked": imputed["n_unlinked"]},
                alpha_u, ("k_open", "k_closed", "k_mixed"), "n_unlinked",
            )
            S_l = pm_l["open"] - pm_l["closed"]
            S_u = pm_u["open"] - pm_u["closed"]
            delta_signed = S_l - S_u

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
                k=rwr_k, restart_prob=rwr_restart_prob,
                solver=rwr_solver, max_iter=rwr_max_iter, tol=rwr_tol,
                output_dir=output_dir if save_outputs else None,
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
    }


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Unserializable: {type(obj).__name__}")
