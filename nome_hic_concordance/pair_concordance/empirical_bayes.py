"""Pooled empirical Bayes priors per pair category.

For each pair category (e.g. "tf_anchored_NEUROD2", "co_folded",
"random_distance_matched"), estimate Beta priors on the linked
active-concordance rate by method of moments from the cross-cell,
cross-pair distribution.

The fitted priors feed into :mod:`.hierarchical` which computes
Beta-binomial posteriors and coordination excess (delta) per pair per cell.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_priors(
    pair_counts: pd.DataFrame,
    panel: pd.DataFrame,
    pool_by: str = "category",
    min_obs: int = 10,
    nu_floor: float = 2.0,
) -> pd.DataFrame:
    """Estimate Beta(alpha_L, beta_L) priors per category (or category x band).

    For each pooling group, the method-of-moments estimator computes:

        mu_hat  = sum(k) / sum(n)              (weighted mean)
        sigma2  = weighted variance of k/n
        nu      = mu*(1-mu)/sigma2 - 1, floored at nu_floor
        alpha_L = mu * nu
        beta_L  = (1 - mu) * nu

    Parameters
    ----------
    pair_counts : pd.DataFrame
        Stacked across cells.  Required columns: ``pair_id``, ``cell_name``,
        ``category``, ``n_linked``, ``k_linked_active``.  When *pool_by* is
        ``"category_band"``, the column ``distance_band`` must also be present.
    panel : pd.DataFrame
        Panel table used for category lookup.  Must contain ``pair_id`` and
        ``category`` columns (and ``distance_band`` when relevant).
    pool_by : str
        Grouping strategy: ``"category"`` pools all pairs in the same category;
        ``"category_band"`` pools by category x distance_band.
    min_obs : int
        Minimum number of observations (pair-cell combinations with n_linked > 0)
        required to fit a prior for a group.  Groups below this threshold fall
        back to the global prior.
    nu_floor : float
        Minimum concentration parameter nu = alpha + beta.  Prevents
        degenerate priors when the observed variance is very large.

    Returns
    -------
    pd.DataFrame
        One row per pooling group with columns: ``category``,
        ``distance_band`` (if *pool_by* is ``"category_band"``),
        ``alpha_L``, ``beta_L``, ``mu_hat``, ``nu_hat``,
        ``n_obs_pairs``, ``n_obs_cells``.
    """
    required_cols = {"pair_id", "cell_name", "category", "n_linked", "k_linked_active"}
    missing = required_cols - set(pair_counts.columns)
    if missing:
        raise ValueError(f"pair_counts is missing columns: {sorted(missing)}")

    if pool_by == "category_band" and "distance_band" not in pair_counts.columns:
        raise ValueError(
            "pool_by='category_band' requires a 'distance_band' column in pair_counts"
        )

    if pool_by not in ("category", "category_band"):
        raise ValueError(f"pool_by must be 'category' or 'category_band', got {pool_by!r}")

    # Only keep observations with at least one linked read.
    df = pair_counts.loc[pair_counts["n_linked"] > 0].copy()
    df["rate"] = df["k_linked_active"] / df["n_linked"]

    group_cols = ["category"]
    if pool_by == "category_band":
        group_cols.append("distance_band")

    # Compute global fallback prior from all data.
    global_prior = _mom_beta(df, nu_floor=nu_floor)
    logger.info(
        "Global fallback prior: alpha_L=%.3f, beta_L=%.3f (mu=%.4f, nu=%.1f, n=%d)",
        global_prior["alpha_L"],
        global_prior["beta_L"],
        global_prior["mu_hat"],
        global_prior["nu_hat"],
        global_prior["n_obs_pairs"],
    )

    rows: list[dict] = []
    for group_key, grp in df.groupby(group_cols, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        n_obs = len(grp)
        if n_obs < min_obs:
            logger.debug(
                "Group %s has only %d obs (< min_obs=%d); using global prior",
                group_key, n_obs, min_obs,
            )
            row = dict(global_prior)
            for col, val in zip(group_cols, group_key):
                row[col] = val
            row["fallback"] = True
            rows.append(row)
            continue

        prior = _mom_beta(grp, nu_floor=nu_floor)
        for col, val in zip(group_cols, group_key):
            prior[col] = val
        prior["fallback"] = False
        rows.append(prior)

    result = pd.DataFrame(rows)

    # Re-order columns for readability.
    lead = group_cols + ["alpha_L", "beta_L", "mu_hat", "nu_hat",
                         "n_obs_pairs", "n_obs_cells", "fallback"]
    lead = [c for c in lead if c in result.columns]
    rest = [c for c in result.columns if c not in lead]
    result = result[lead + rest]

    n_groups = len(result)
    n_fallback = int(result["fallback"].sum())
    logger.info(
        "Fitted priors for %d groups (%d used global fallback)",
        n_groups, n_fallback,
    )

    return result


def _mom_beta(
    df: pd.DataFrame,
    nu_floor: float = 2.0,
) -> dict:
    """Method-of-moments Beta fit from a subset of pair-cell observations.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``n_linked``, ``k_linked_active``, ``rate``,
        ``pair_id``, and ``cell_name``.
    nu_floor : float
        Floor on the concentration parameter.

    Returns
    -------
    dict with alpha_L, beta_L, mu_hat, nu_hat, n_obs_pairs, n_obs_cells.
    """
    n = df["n_linked"].values.astype(float)
    k = df["k_linked_active"].values.astype(float)
    rate = df["rate"].values.astype(float)

    # Weighted mean: mu_hat = sum(k) / sum(n).
    sum_n = float(np.sum(n))
    sum_k = float(np.sum(k))
    if sum_n == 0:
        mu_hat = 0.5
    else:
        mu_hat = sum_k / sum_n

    # Weighted variance of rate, with weights = n.
    if len(rate) < 2 or sum_n == 0:
        sigma2 = 0.25  # maximum-variance fallback
    else:
        weights = n / sum_n
        sigma2 = float(np.sum(weights * (rate - mu_hat) ** 2))

    # Prevent degenerate cases.
    mu_hat = float(np.clip(mu_hat, 1e-4, 1.0 - 1e-4))
    sigma2 = max(sigma2, 1e-8)

    # Method of moments: nu = mu*(1-mu)/sigma2 - 1.
    nu_raw = mu_hat * (1.0 - mu_hat) / sigma2 - 1.0
    nu = max(nu_raw, nu_floor)

    alpha_l = mu_hat * nu
    beta_l = (1.0 - mu_hat) * nu

    n_obs_pairs = int(df["pair_id"].nunique())
    n_obs_cells = int(df["cell_name"].nunique())

    return {
        "alpha_L": float(alpha_l),
        "beta_L": float(beta_l),
        "mu_hat": float(mu_hat),
        "nu_hat": float(nu),
        "n_obs_pairs": n_obs_pairs,
        "n_obs_cells": n_obs_cells,
    }
