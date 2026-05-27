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
