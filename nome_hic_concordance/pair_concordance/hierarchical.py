"""Beta-binomial posterior on linked concordance and coordination excess.

For each (pair_id, cell_id), computes:

- **Linked concordance posterior** theta_L via Beta-binomial conjugacy with
  empirical Bayes priors from :mod:`.empirical_bayes`.
- **Independence baseline** theta_U from unlinked marginal rates.
- **Coordination excess** delta = E[theta_L] - theta_U with a z-score.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-cell delta computation
# ---------------------------------------------------------------------------


def compute_delta(
    pair_counts: pd.DataFrame,
    priors: pd.DataFrame,
    panel: pd.DataFrame,
    min_unlinked: int = 5,
) -> pd.DataFrame:
    """Compute coordination excess (delta) per pair for one cell.

    For each pair, the linked concordance posterior is:

        theta_L | data ~ Beta(alpha_cat + k, beta_cat + n - k)

    The independence baseline is:

        theta_U_active = p_i * p_j

    where p_i = k_unlinked_i_open / n_unlinked_i (and similarly for j).

    Parameters
    ----------
    pair_counts : pd.DataFrame
        One cell's pair-level counts.  Required columns: ``pair_id``,
        ``category``, ``n_linked``, ``k_linked_active``,
        ``n_unlinked_i``, ``k_unlinked_i_open``,
        ``n_unlinked_j``, ``k_unlinked_j_open``.
        If *priors* were fit with ``pool_by="category_band"``, the column
        ``distance_band`` must also be present.
    priors : pd.DataFrame
        Output of :func:`.empirical_bayes.fit_priors`.  Must contain
        ``category``, ``alpha_L``, ``beta_L`` (and ``distance_band`` if
        applicable).
    panel : pd.DataFrame
        Panel table with ``pair_id`` and ``category`` columns.
    min_unlinked : int
        Minimum unlinked reads per anchor for a reliable marginal rate.
        Anchors below this threshold use the category-pooled rate instead.

    Returns
    -------
    pd.DataFrame
        One row per pair with columns: ``pair_id``, ``n_linked``,
        ``k_linked_active``, ``theta_L_mean``, ``theta_L_var``,
        ``p_i``, ``p_j``, ``theta_U_active``, ``theta_U_var``,
        ``delta``, ``delta_var``, ``delta_z``.
    """
    # Merge category (and distance_band if needed) from the panel —
    # classify.py only stores per-cell counts, not panel metadata.
    panel_lookup_cols = ["pair_id"]
    if "category" not in pair_counts.columns:
        panel_lookup_cols.append("category")
    if "distance_band" in priors.columns and "distance_band" not in pair_counts.columns:
        if "distance_band" in panel.columns:
            panel_lookup_cols.append("distance_band")
    if len(panel_lookup_cols) > 1:
        pair_counts = pair_counts.merge(
            panel[panel_lookup_cols], on="pair_id", how="left",
        )

    required = {
        "pair_id", "category", "n_linked", "k_linked_active",
        "n_unlinked_i", "k_unlinked_i_open",
        "n_unlinked_j", "k_unlinked_j_open",
    }
    missing = required - set(pair_counts.columns)
    if missing:
        raise ValueError(f"pair_counts is missing columns: {sorted(missing)}")

    # Determine prior merge keys.
    prior_keys = ["category"]
    if "distance_band" in priors.columns and "distance_band" in pair_counts.columns:
        prior_keys.append("distance_band")

    # Merge priors onto pair_counts.
    prior_cols = prior_keys + ["alpha_L", "beta_L"]
    df = pair_counts.merge(
        priors[prior_cols],
        on=prior_keys,
        how="left",
    )

    # Warn about pairs with no matching prior (should be rare).
    no_prior = df["alpha_L"].isna()
    if no_prior.any():
        n_missing = int(no_prior.sum())
        logger.warning(
            "%d pairs have no matching prior; using uniform Beta(1, 1)", n_missing
        )
        df.loc[no_prior, "alpha_L"] = 1.0
        df.loc[no_prior, "beta_L"] = 1.0

    # ---- Linked concordance posterior (Beta-binomial conjugacy) ----
    n_linked = df["n_linked"].values.astype(float)
    k_linked = df["k_linked_active"].values.astype(float)
    alpha_prior = df["alpha_L"].values.astype(float)
    beta_prior = df["beta_L"].values.astype(float)

    alpha_post = alpha_prior + k_linked
    beta_post = beta_prior + (n_linked - k_linked)

    ab_sum = alpha_post + beta_post
    theta_L_mean = alpha_post / ab_sum
    theta_L_var = (alpha_post * beta_post) / (ab_sum ** 2 * (ab_sum + 1.0))

    # ---- Independence baseline (unlinked marginal rates) ----
    n_ui = df["n_unlinked_i"].values.astype(float)
    k_ui = df["k_unlinked_i_open"].values.astype(float)
    n_uj = df["n_unlinked_j"].values.astype(float)
    k_uj = df["k_unlinked_j_open"].values.astype(float)

    # Category-pooled fallback rates for anchors with too few unlinked reads.
    pooled_rates = _category_pooled_rates(pair_counts)

    low_i = n_ui < min_unlinked
    low_j = n_uj < min_unlinked
    if low_i.any() or low_j.any():
        logger.debug(
            "Falling back to pooled rates for %d anchor-i and %d anchor-j entries",
            int(low_i.sum()), int(low_j.sum()),
        )

    # Compute per-pair marginal rates, substituting pooled fallback when needed.
    p_i = np.where(n_ui > 0, k_ui / n_ui, 0.5)
    p_j = np.where(n_uj > 0, k_uj / n_uj, 0.5)

    categories = df["category"].values
    for cat, pooled_rate in pooled_rates.items():
        cat_mask = categories == cat
        p_i = np.where(low_i & cat_mask, pooled_rate, p_i)
        p_j = np.where(low_j & cat_mask, pooled_rate, p_j)

    theta_U = p_i * p_j

    # Unlinked variance via delta method:
    # Var(p_i*p_j) = p_j^2 * p_i*(1-p_i)/n_i + p_i^2 * p_j*(1-p_j)/n_j
    n_ui_safe = np.maximum(n_ui, 1.0)
    n_uj_safe = np.maximum(n_uj, 1.0)
    var_pi = p_i * (1.0 - p_i) / n_ui_safe
    var_pj = p_j * (1.0 - p_j) / n_uj_safe
    theta_U_var = (p_j ** 2) * var_pi + (p_i ** 2) * var_pj

    # ---- Coordination excess ----
    delta = theta_L_mean - theta_U
    delta_var = theta_L_var + theta_U_var
    delta_z = np.where(delta_var > 0, delta / np.sqrt(delta_var), 0.0)

    # Flag pairs where the category-pooled fallback was used for
    # the unlinked baseline (per Math v1.1 §5.2).
    used_fallback = (low_i | low_j).astype(int)

    result = pd.DataFrame({
        "pair_id": df["pair_id"].values,
        "n_linked": n_linked.astype(int),
        "k_linked_active": k_linked.astype(int),
        "theta_L_mean": theta_L_mean,
        "theta_L_var": theta_L_var,
        "p_i": p_i,
        "p_j": p_j,
        "theta_U_active": theta_U,
        "theta_U_var": theta_U_var,
        "used_fallback": used_fallback,
        "delta": delta,
        "delta_var": delta_var,
        "delta_z": delta_z,
    })

    return result


def _category_pooled_rates(pair_counts: pd.DataFrame) -> dict[str, float]:
    """Compute category-level pooled open rate from all unlinked data.

    For each category, pools across all pairs:

        pooled_rate = sum(k_unlinked_open) / sum(n_unlinked)

    using both anchor i and anchor j contributions.

    Parameters
    ----------
    pair_counts : pd.DataFrame
        Must contain ``category``, ``n_unlinked_i``, ``k_unlinked_i_open``,
        ``n_unlinked_j``, ``k_unlinked_j_open``.

    Returns
    -------
    dict mapping category name to its pooled open rate.
    """
    rates: dict[str, float] = {}
    for cat, grp in pair_counts.groupby("category", sort=False):
        total_n = (
            grp["n_unlinked_i"].sum() + grp["n_unlinked_j"].sum()
        )
        total_k = (
            grp["k_unlinked_i_open"].sum() + grp["k_unlinked_j_open"].sum()
        )
        if total_n > 0:
            rates[cat] = float(total_k) / float(total_n)
        else:
            rates[cat] = 0.5
    return rates


# ---------------------------------------------------------------------------
# Multi-cell orchestration
# ---------------------------------------------------------------------------


def compute_all_deltas(
    cell_pair_count_paths: dict[str, str],
    priors: pd.DataFrame,
    panel: pd.DataFrame,
    output_dir: str,
    n_threads: int = 4,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute delta for all cells and assemble cells x pairs matrices.

    Reads each cell's pair-count parquet, calls :func:`compute_delta`, writes
    per-cell results, then pivots into (cells x pairs) delta and delta_var
    matrices.

    Parameters
    ----------
    cell_pair_count_paths : dict
        Mapping ``cell_name -> parquet path`` for per-cell pair counts.
    priors : pd.DataFrame
        Output of :func:`.empirical_bayes.fit_priors`.
    panel : pd.DataFrame
        Panel table with ``pair_id`` and ``category`` columns.
    output_dir : str
        Directory for per-cell and matrix outputs.
    n_threads : int
        Number of parallel workers.
    **kwargs
        Forwarded to :func:`compute_delta` (e.g. ``min_unlinked``).

    Returns
    -------
    (delta_matrix, delta_var_matrix)
        Both are DataFrames indexed by cell_name with pair_id columns.
    """
    from joblib import Parallel, delayed

    out_path = ensure_dir(output_dir)
    per_cell_dir = ensure_dir(out_path / "per_cell")

    items = list(cell_pair_count_paths.items())
    logger.info("Computing deltas for %d cells (n_threads=%d)", len(items), n_threads)

    if n_threads > 1 and len(items) > 1:
        results = Parallel(n_jobs=n_threads)(
            delayed(_process_one_cell)(
                cell_name, path, priors, panel, per_cell_dir, kwargs,
            )
            for cell_name, path in items
        )
    else:
        results = [
            _process_one_cell(cell_name, path, priors, panel, per_cell_dir, kwargs)
            for cell_name, path in items
        ]

    # Assemble cells x pairs matrices.
    delta_rows: dict[str, dict[str, float]] = {}
    delta_var_rows: dict[str, dict[str, float]] = {}
    all_pairs: set[str] = set()

    for cell_name, cell_df in results:
        if cell_df is None or len(cell_df) == 0:
            continue
        d_map = dict(zip(cell_df["pair_id"], cell_df["delta"]))
        dv_map = dict(zip(cell_df["pair_id"], cell_df["delta_var"]))
        delta_rows[cell_name] = d_map
        delta_var_rows[cell_name] = dv_map
        all_pairs.update(d_map.keys())

    sorted_pairs = sorted(all_pairs)
    delta_matrix = pd.DataFrame.from_dict(delta_rows, orient="index")
    delta_var_matrix = pd.DataFrame.from_dict(delta_var_rows, orient="index")

    # Ensure consistent column order.
    delta_matrix = delta_matrix.reindex(columns=sorted_pairs)
    delta_var_matrix = delta_var_matrix.reindex(columns=sorted_pairs)

    delta_matrix.index.name = "cell"
    delta_matrix.columns.name = "pair_id"
    delta_var_matrix.index.name = "cell"
    delta_var_matrix.columns.name = "pair_id"

    delta_matrix.to_parquet(out_path / "delta_matrix.parquet")
    delta_var_matrix.to_parquet(out_path / "delta_var_matrix.parquet")

    logger.info(
        "Delta matrices: %d cells x %d pairs",
        delta_matrix.shape[0], delta_matrix.shape[1],
    )

    return delta_matrix, delta_var_matrix


def _process_one_cell(
    cell_name: str,
    parquet_path: str,
    priors: pd.DataFrame,
    panel: pd.DataFrame,
    per_cell_dir: Path,
    kwargs: dict,
) -> tuple[str, pd.DataFrame | None]:
    """Load one cell's pair counts, compute delta, and save results."""
    try:
        pair_counts = pd.read_parquet(parquet_path)
    except Exception as exc:
        logger.warning("Failed to read %s for cell %s: %s", parquet_path, cell_name, exc)
        return cell_name, None

    if len(pair_counts) == 0:
        logger.debug("Cell %s has no pair counts; skipping", cell_name)
        return cell_name, None

    try:
        cell_df = compute_delta(pair_counts, priors, panel, **kwargs)
    except Exception as exc:
        logger.warning("compute_delta failed for cell %s: %s", cell_name, exc)
        return cell_name, None

    out_file = per_cell_dir / f"{cell_name}.delta.parquet"
    cell_df.to_parquet(out_file, index=False)

    return cell_name, cell_df
