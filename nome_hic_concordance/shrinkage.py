"""Empirical Bayes shrinkage of per-bin Pearson concordance values.

Under the model ``r*_cb ~ Normal(0, sigma2_prior)`` with sampling variance
approximately ``1/(n-3)``, the Bayes-optimal shrinkage estimator is
``r_shrunk = r_raw * n / (n + n0)`` where ``n0 = 1 / sigma2_prior``.

``estimate_shrinkage_n0`` fits ``sigma2_prior`` empirically from high-coverage
bins; ``apply_shrinkage`` applies the per-entry multiplier.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def estimate_shrinkage_n0(
    concordance_raw: pd.DataFrame,
    pair_counts: pd.DataFrame,
    n_min_for_estimation: int = 50,
    output_dir: str = "output/05_concordance",
) -> dict:
    """Empirical Bayes estimate of the shrinkage pseudocount ``n0``.

    Returns a dict with ``n0``, ``sigma2_prior``, and diagnostic counts.
    Also writes ``shrinkage_n0.json`` and ``shrinkage_analysis.png``.
    """
    out_path = ensure_dir(output_dir)

    r_flat = concordance_raw.to_numpy(dtype=float).ravel()
    n_flat = pair_counts.to_numpy(dtype=float).ravel()

    valid = ~np.isnan(r_flat) & (n_flat >= 3)
    r_valid = r_flat[valid]
    n_valid = n_flat[valid]

    high_cov = n_valid >= n_min_for_estimation
    r_high = r_valid[high_cov]
    n_high = n_valid[high_cov]

    if len(r_high) < 20:
        logger.warning(
            "Too few high-coverage entries for reliable sigma2_prior estimation. "
            "Using default n0=30."
        )
        n0 = 30.0
        sigma2_prior = 1.0 / n0
    else:
        var_observed = float(np.var(r_high))
        var_sampling = float(np.mean(1.0 / (n_high - 3)))
        sigma2_prior = max(var_observed - var_sampling, 0.01)
        n0 = 1.0 / sigma2_prior

    logger.info("Shrinkage analysis:")
    logger.info(f"  High-coverage entries (n >= {n_min_for_estimation}): {len(r_high)}")
    logger.info(f"  sigma2_prior = {sigma2_prior:.4f}")
    logger.info(f"  Optimal n0 = {n0:.1f}")

    result = {
        "n0": float(n0),
        "sigma2_prior": float(sigma2_prior),
        "n_high_coverage_entries": int(len(r_high)),
        "n_min_for_estimation": int(n_min_for_estimation),
        "n_valid_entries": int(len(r_valid)),
    }

    with open(out_path / "shrinkage_n0.json", "w") as f:
        json.dump(result, f, indent=2)

    try:
        _plot_shrinkage_diagnostics(r_valid, n_valid, n0, sigma2_prior, out_path)
    except Exception as exc:
        logger.warning(f"Shrinkage diagnostic plot failed: {exc}")

    return result


def apply_shrinkage(
    concordance_raw: pd.DataFrame,
    pair_counts: pd.DataFrame,
    n0: float,
) -> pd.DataFrame:
    """Apply ``r_shrunk = r_raw * n / (n + n0)`` entry-wise.

    NaN entries in ``concordance_raw`` remain NaN.
    """
    n_arr = pair_counts.to_numpy(dtype=float)
    r_arr = concordance_raw.to_numpy(dtype=float)

    multiplier = np.where(n_arr + n0 > 0, n_arr / (n_arr + n0), 0.0)
    shrunk = r_arr * multiplier
    shrunk[np.isnan(r_arr)] = np.nan

    return pd.DataFrame(
        shrunk,
        index=concordance_raw.index,
        columns=concordance_raw.columns,
    )


def _plot_shrinkage_diagnostics(
    r_valid: np.ndarray,
    n_valid: np.ndarray,
    n0: float,
    sigma2_prior: float,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: empirical Var(r) vs n_pairs, with model.
    ax = axes[0]
    if len(n_valid) > 0:
        top = max(10.0, float(np.percentile(n_valid, 95)))
        bin_edges = np.arange(5, top + 1, 10)
    else:
        bin_edges = np.arange(5, 50, 10)

    bin_centers: list[float] = []
    bin_vars: list[float] = []
    for i in range(len(bin_edges) - 1):
        mask = (n_valid >= bin_edges[i]) & (n_valid < bin_edges[i + 1])
        if np.sum(mask) >= 5:
            bin_vars.append(float(np.var(r_valid[mask])))
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)

    if bin_centers:
        ax.scatter(bin_centers, bin_vars, s=20, alpha=0.7)
        x_theory = np.linspace(5, max(bin_centers), 200)
        ax.plot(
            x_theory,
            sigma2_prior + 1.0 / np.maximum(x_theory - 3, 1e-3),
            "r-",
            label="sigma2_prior + 1/(n-3)",
        )
    ax.axhline(sigma2_prior, color="gray", linestyle="--",
               label=f"sigma2_prior = {sigma2_prior:.4f}")
    ax.set_xlabel("Number of pairs in bin")
    ax.set_ylabel("Variance of r")
    ax.set_title("Variance vs. pair count")
    ax.legend(fontsize=8)

    # Panel 2: shrinkage multiplier curve.
    ax = axes[1]
    n_range = np.arange(1, max(int(n0 * 5), 200) + 1)
    shrink_factor = n_range / (n_range + n0)
    ax.plot(n_range, shrink_factor, "b-")
    ax.axvline(n0, color="r", linestyle="--", label=f"n0 = {n0:.0f}")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("Number of pairs")
    ax.set_ylabel("Shrinkage multiplier n/(n + n0)")
    ax.set_title("Shrinkage strength")
    ax.legend()

    # Panel 3: before/after distribution.
    ax = axes[2]
    r_shrunk = r_valid * n_valid / (n_valid + n0)
    bins = 50
    if len(r_valid) > 0:
        ax.hist(r_valid, bins=bins, alpha=0.5, label="Raw r", density=True)
        ax.hist(r_shrunk, bins=bins, alpha=0.5, label="Shrunk r", density=True)
    ax.set_xlabel("Concordance (r)")
    ax.set_ylabel("Density")
    ax.set_title("Raw vs. shrunk distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path / "shrinkage_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
