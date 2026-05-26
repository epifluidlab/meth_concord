"""Two-component beta-binomial mixture model fitted by EM.

Models methylation counts ``k_i | n_i`` per read end as a mixture of a
"closed" and "accessible" beta-binomial component. Fitted parameters are
used by :mod:`posterior` to compute ``P(accessible | k, n)`` for each read end.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core beta-binomial log-PMF
# ---------------------------------------------------------------------------


def betabinom_logpmf(
    k: np.ndarray | float,
    n: np.ndarray | float,
    alpha: float,
    beta_param: float,
) -> np.ndarray:
    """Vectorised log-PMF of the beta-binomial distribution.

    ``P(k | n, alpha, beta) = C(n, k) * B(k + alpha, n - k + beta) / B(alpha, beta)``

    Uses ``scipy.special`` for numerical stability.
    """
    from scipy.special import betaln, gammaln

    return (
        gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        + betaln(k + alpha, n - k + beta_param)
        - betaln(alpha, beta_param)
    )


# ---------------------------------------------------------------------------
# Subsampling helper (used before EM fitting to bound compute)
# ---------------------------------------------------------------------------


def subsample_for_em(
    k_all: np.ndarray,
    n_all: np.ndarray,
    max_samples: int = 10_000_000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample read ends for EM fitting, stratified by ``n``.

    Preserves the marginal distribution of coverage per read end so the
    mixture components are not biased toward low- or high-coverage ends.

    Pass ``max_samples=0`` to disable subsampling (use all read ends).
    """
    if max_samples <= 0 or len(k_all) <= max_samples:
        return k_all, n_all

    rng = np.random.RandomState(random_state)
    unique_n = np.unique(n_all)

    chosen_parts: list[np.ndarray] = []
    for n_val in unique_n:
        mask = n_all == n_val
        n_in_stratum = int(mask.sum())
        # Proportional allocation; always take at least 1.
        n_to_sample = max(1, int(round(max_samples * n_in_stratum / len(k_all))))
        stratum_indices = np.where(mask)[0]
        if n_to_sample < len(stratum_indices):
            chosen = rng.choice(stratum_indices, size=n_to_sample, replace=False)
        else:
            chosen = stratum_indices
        chosen_parts.append(chosen)

    indices = np.concatenate(chosen_parts)
    rng.shuffle(indices)

    logger.info(
        f"Subsampled {len(k_all):,} -> {len(indices):,} read ends for EM fitting"
    )
    return k_all[indices], n_all[indices]


# ---------------------------------------------------------------------------
# BetaBinomialMixture class
# ---------------------------------------------------------------------------


class BetaBinomialMixture:
    """Two-component beta-binomial mixture, fitted by EM."""

    def __init__(
        self,
        max_iter: int = 200,
        tol: float = 1e-6,
        n_restarts: int = 5,
        random_state: int = 42,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.random_state = random_state

        # Fitted parameters. Index 0 = closed, index 1 = accessible.
        self.pi_: float | None = None
        self.alpha_: np.ndarray | None = None
        self.beta_: np.ndarray | None = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.log_likelihood_: float | None = None

    # ----- Sufficient-statistics EM -----------------------------------------
    #
    # Key insight: betabinom_logpmf(k, n, α, β) only depends on the (k, n)
    # pair, not on which read it came from.  With 10M read ends there are
    # typically only ~500 unique (k, n) combinations.  We compute all
    # scipy special functions on ~500 elements instead of 10M — a ~20,000×
    # reduction in the inner-loop array sizes.

    @staticmethod
    def _compress_to_sufficient_stats(
        k: np.ndarray, n: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Collapse (k, n) arrays to unique pairs with counts.

        Returns (uk, un, counts, N_total).
        """
        from scipy.special import gammaln

        pairs = np.column_stack([k.astype(np.int32), n.astype(np.int32)])
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        uk = unique_pairs[:, 0].astype(float)
        un = unique_pairs[:, 1].astype(float)
        counts = counts.astype(float)
        return uk, un, counts, len(k)

    def _e_step_compressed(
        self,
        uk: np.ndarray,
        un: np.ndarray,
        counts: np.ndarray,
        N: int,
        log_cnk: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """E-step on compressed sufficient statistics (~500 elements)."""
        from scipy.special import betaln

        log_p0 = (log_cnk
                   + betaln(uk + self.alpha_[0], un - uk + self.beta_[0])
                   - betaln(self.alpha_[0], self.beta_[0]))
        log_p1 = (log_cnk
                   + betaln(uk + self.alpha_[1], un - uk + self.beta_[1])
                   - betaln(self.alpha_[1], self.beta_[1]))

        pi = float(np.clip(self.pi_, 1e-12, 1.0 - 1e-12))
        log_w0 = np.log(1 - pi) + log_p0
        log_w1 = np.log(pi) + log_p1

        log_total = np.logaddexp(log_w0, log_w1)
        gamma = np.exp(log_w1 - log_total)  # P(accessible) per unique (k,n)

        # Total log-likelihood, weighted by counts.
        ll = float(np.sum(counts * log_total))
        return gamma, ll

    def _m_step_component_compressed(
        self,
        uk: np.ndarray,
        un: np.ndarray,
        eff_weights: np.ndarray,
        alpha_init: float,
        beta_init: float,
        log_cnk: np.ndarray,
        unmk: np.ndarray,
    ) -> tuple[float, float]:
        """M-step on compressed data with analytical gradient (~500 elements)."""
        from scipy.optimize import minimize
        from scipy.special import betaln, digamma

        def neg_ll_and_grad(params: np.ndarray):
            a, b = params
            if a <= 0 or b <= 0:
                return 1e12, np.zeros(2)

            ka = uk + a
            nmkb = unmk + b
            ab = a + b
            nab = un + ab

            ll_terms = log_cnk + betaln(ka, nmkb) - betaln(a, b)
            neg_ll = -float(np.sum(eff_weights * ll_terms))

            psi_ka = digamma(ka)
            psi_nmkb = digamma(nmkb)
            psi_nab = digamma(nab)
            psi_ab = digamma(ab)
            psi_a = digamma(a)
            psi_b = digamma(b)

            dll_da = float(np.sum(eff_weights * (psi_ka - psi_nab + psi_ab - psi_a)))
            dll_db = float(np.sum(eff_weights * (psi_nmkb - psi_nab + psi_ab - psi_b)))

            return neg_ll, np.array([-dll_da, -dll_db])

        result = minimize(
            neg_ll_and_grad,
            x0=np.array([alpha_init, beta_init]),
            method="L-BFGS-B",
            jac=True,
            bounds=[(1e-4, 500.0), (1e-4, 500.0)],
        )
        return float(result.x[0]), float(result.x[1])

    def _fit_once(
        self,
        k: np.ndarray,
        n: np.ndarray,
        pi_init: float,
        alpha_init: np.ndarray,
        beta_init: np.ndarray,
    ) -> tuple[float, int, bool]:
        """Run EM from a single initialisation using sufficient statistics."""
        from scipy.special import gammaln

        self.pi_ = float(pi_init)
        self.alpha_ = alpha_init.copy().astype(float)
        self.beta_ = beta_init.copy().astype(float)

        # Compress 10M read ends → ~500 unique (k, n) pairs.
        uk, un, counts, N = self._compress_to_sufficient_stats(k, n)
        logger.info(
            f"    Compressed {N:,} read ends → {len(uk):,} unique (k,n) pairs"
        )

        # Pre-compute constants on the compressed arrays (~500 elements).
        log_cnk = gammaln(un + 1) - gammaln(uk + 1) - gammaln(un - uk + 1)
        unmk = un - uk

        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            gamma, ll = self._e_step_compressed(uk, un, counts, N, log_cnk)

            # Relative tolerance: |ΔLL| / max(|LL|, 1) < tol.  Absolute
            # tolerance is impractical for large datasets (LL ~ 10^7 means
            # 1e-6 absolute is unreachable by IEEE-754 floats).
            if np.isfinite(prev_ll):
                rel_change = abs(ll - prev_ll) / max(abs(ll), 1.0)
                if rel_change < self.tol:
                    return ll, iteration + 1, True
            prev_ll = ll

            # pi = weighted mean of gamma across all data points.
            self.pi_ = float(np.sum(counts * gamma) / N)

            # Effective weights per unique (k,n) for each component.
            w1 = counts * gamma        # accessible
            w0 = counts * (1 - gamma)  # closed

            self.alpha_[1], self.beta_[1] = self._m_step_component_compressed(
                uk, un, w1, self.alpha_[1], self.beta_[1], log_cnk, unmk,
            )
            self.alpha_[0], self.beta_[0] = self._m_step_component_compressed(
                uk, un, w0, self.alpha_[0], self.beta_[0], log_cnk, unmk,
            )

        _, final_ll = self._e_step_compressed(uk, un, counts, N, log_cnk)
        return final_ll, self.max_iter, False

    # ----- Public API -------------------------------------------------------

    def fit(self, k: np.ndarray, n: np.ndarray) -> "BetaBinomialMixture":
        """Fit mixture via EM with random restarts.

        Parameters
        ----------
        k : array of shape (N,)
            Methylated context counts per read end.
        n : array of shape (N,)
            Total context counts per read end. Must be >= 1 (entries with
            ``n == 0`` are filtered out).
        """
        k = np.asarray(k)
        n = np.asarray(n)

        mask = n >= 1
        k = k[mask].astype(float)
        n = n[mask].astype(float)

        if len(k) == 0:
            raise ValueError("No read ends with n >= 1 to fit mixture model.")

        logger.info(f"Fitting beta-binomial mixture on {len(k):,} read ends...")

        rng = np.random.RandomState(self.random_state)
        base_inits = {
            "pi": 0.3,
            "alpha": np.array([2.0, 8.0]),
            "beta": np.array([8.0, 2.0]),
        }

        best_ll = -np.inf
        best_params: dict | None = None

        for restart in range(self.n_restarts):
            if restart == 0:
                pi_init = base_inits["pi"]
                alpha_init = base_inits["alpha"].copy()
                beta_init = base_inits["beta"].copy()
            else:
                pi_init = float(np.clip(
                    base_inits["pi"] + rng.uniform(-0.15, 0.15), 0.05, 0.95
                ))
                alpha_init = base_inits["alpha"] * rng.uniform(0.5, 2.0, size=2)
                beta_init = base_inits["beta"] * rng.uniform(0.5, 2.0, size=2)

            ll, n_iter, converged = self._fit_once(k, n, pi_init, alpha_init, beta_init)
            logger.info(
                f"  Restart {restart}: LL={ll:.2f}, iters={n_iter}, converged={converged}"
            )

            if ll > best_ll:
                best_ll = ll
                best_params = {
                    "pi": self.pi_,
                    "alpha": self.alpha_.copy(),
                    "beta": self.beta_.copy(),
                    "n_iter": n_iter,
                    "converged": converged,
                }

        assert best_params is not None
        self.pi_ = best_params["pi"]
        self.alpha_ = best_params["alpha"]
        self.beta_ = best_params["beta"]
        self.n_iter_ = best_params["n_iter"]
        self.converged_ = best_params["converged"]
        self.log_likelihood_ = best_ll

        # Canonicalise: component 0 must have lower mean than component 1.
        mean0 = self.alpha_[0] / (self.alpha_[0] + self.beta_[0])
        mean1 = self.alpha_[1] / (self.alpha_[1] + self.beta_[1])
        if mean0 > mean1:
            self.alpha_ = self.alpha_[::-1].copy()
            self.beta_ = self.beta_[::-1].copy()
            self.pi_ = 1.0 - self.pi_
            logger.info("  Swapped components so component 1 = accessible.")

        self._log_summary()
        return self

    def predict_proba(self, k: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Compute ``P(accessible | k, n)`` for each read end.

        Read ends with ``n == 0`` have undefined posteriors and are returned
        as ``NaN``.
        """
        if self.pi_ is None or self.alpha_ is None or self.beta_ is None:
            raise RuntimeError("Model is not fitted.")

        k = np.asarray(k).astype(float)
        n = np.asarray(n).astype(float)

        from scipy.special import betaln

        out = np.full(len(k), np.nan, dtype=float)
        valid = n >= 1
        if valid.any():
            kv, nv = k[valid], n[valid]
            log_p0 = betabinom_logpmf(kv, nv, self.alpha_[0], self.beta_[0])
            log_p1 = betabinom_logpmf(kv, nv, self.alpha_[1], self.beta_[1])
            pi = float(np.clip(self.pi_, 1e-12, 1.0 - 1e-12))
            log_w0 = np.log(1 - pi) + log_p0
            log_w1 = np.log(pi) + log_p1
            gamma = np.exp(log_w1 - np.logaddexp(log_w0, log_w1))
            out[valid] = gamma
        return out

    # ----- Serialisation ----------------------------------------------------

    def _log_summary(self) -> None:
        for j, label in enumerate(["Closed", "Accessible"]):
            mean = self.alpha_[j] / (self.alpha_[j] + self.beta_[j])
            od = 1.0 / (self.alpha_[j] + self.beta_[j] + 1.0)
            logger.info(
                f"  {label}: alpha={self.alpha_[j]:.3f}, beta={self.beta_[j]:.3f}, "
                f"mean={mean:.3f}, overdispersion={od:.4f}"
            )
        logger.info(
            f"  pi(accessible)={self.pi_:.3f}, LL={self.log_likelihood_:.2f}"
        )

    def to_dict(self) -> dict:
        if self.pi_ is None or self.alpha_ is None or self.beta_ is None:
            raise RuntimeError("Model is not fitted.")
        return {
            "pi": float(self.pi_),
            "alpha_closed": float(self.alpha_[0]),
            "beta_closed": float(self.beta_[0]),
            "alpha_accessible": float(self.alpha_[1]),
            "beta_accessible": float(self.beta_[1]),
            "mean_closed": float(self.alpha_[0] / (self.alpha_[0] + self.beta_[0])),
            "mean_accessible": float(self.alpha_[1] / (self.alpha_[1] + self.beta_[1])),
            "overdispersion_closed": float(1.0 / (self.alpha_[0] + self.beta_[0] + 1.0)),
            "overdispersion_accessible": float(1.0 / (self.alpha_[1] + self.beta_[1] + 1.0)),
            "log_likelihood": float(self.log_likelihood_) if self.log_likelihood_ is not None else None,
            "n_iter": int(self.n_iter_),
            "converged": bool(self.converged_),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BetaBinomialMixture":
        obj = cls()
        obj.pi_ = float(d["pi"])
        obj.alpha_ = np.array([d["alpha_closed"], d["alpha_accessible"]], dtype=float)
        obj.beta_ = np.array([d["beta_closed"], d["beta_accessible"]], dtype=float)
        obj.log_likelihood_ = d.get("log_likelihood")
        obj.n_iter_ = int(d.get("n_iter", 0))
        obj.converged_ = bool(d.get("converged", False))
        return obj

    @classmethod
    def from_json(cls, path: str) -> "BetaBinomialMixture":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Streaming subsample — reads one cell at a time to avoid loading all parquets
# ---------------------------------------------------------------------------


def stream_subsample_from_parquets(
    cell_parquet_paths: dict[str, str],
    max_subsample: int = 10_000_000,
    short_range_max: int = 1_000,
    min_distance: int = 20_000,
    random_state: int = 42,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """Stream through per-cell parquets, drawing a proportional sample per cell.

    Only one cell's data is in memory at a time, so peak RAM is bounded by
    the size of a single parquet plus the accumulated subsample (~160 MB for
    10M read ends).

    The per-cell quota is set slightly above ``max_subsample / n_cells`` to
    allow the final n-stratified pass in :func:`subsample_for_em` to trim
    precisely.

    Parameters
    ----------
    cell_parquet_paths : dict
        Mapping ``cell_name -> parquet path``.
    max_subsample : int
        Target number of read ends per distance subset.
    short_range_max, min_distance : int
        Distance thresholds for the short/long subsets.
    random_state : int

    Returns
    -------
    (subsampled, subsample_log)

    subsampled : dict
        ``{subset_name: (k_array, n_array)}`` with <= ``max_subsample`` entries
        each, already n-stratified.
    subsample_log : dict
        Per-subset metadata (original and used counts).
    """
    n_cells = len(cell_parquet_paths)
    if n_cells == 0:
        empty = np.array([], dtype=float)
        return {"all": (empty, empty), "short": (empty, empty), "long": (empty, empty)}, {}

    no_limit = max_subsample <= 0  # 0 = use all read ends, no subsampling
    if no_limit:
        per_cell_quota = 2**63  # effectively unlimited
    else:
        # 20% headroom so the final n-stratified trim has enough material.
        per_cell_quota = int(np.ceil(max_subsample * 1.2 / n_cells))
    rng = np.random.RandomState(random_state)

    # Accumulators: lists of small arrays, concatenated at the end.
    collectors: dict[str, tuple[list, list]] = {
        "all": ([], []),
        "short": ([], []),
        "long": ([], []),
    }
    total_ends: dict[str, int] = {"all": 0, "short": 0, "long": 0}

    for cell_name, parquet_path in cell_parquet_paths.items():
        df = pd.read_parquet(
            parquet_path, columns=["k1", "n1", "k2", "n2", "distance"],
        )
        if len(df) == 0:
            continue

        # Stack both ends.  Each pair contributes two read ends.
        k_both = np.concatenate([df["k1"].values, df["k2"].values]).astype(float)
        n_both = np.concatenate([df["n1"].values, df["n2"].values]).astype(float)

        # The distance array must be duplicated to match (one copy per end).
        dist = np.concatenate([df["distance"].values, df["distance"].values]).astype(float)

        # Drop uninformative ends.
        valid = n_both >= 1
        k_both, n_both, dist = k_both[valid], n_both[valid], dist[valid]

        # Assign each read end to the applicable distance subsets.
        masks: dict[str, np.ndarray] = {
            "all": np.ones(len(k_both), dtype=bool),
            "short": dist < short_range_max,
            "long": dist >= min_distance,
        }

        for subset, mask in masks.items():
            k_sub = k_both[mask]
            n_sub = n_both[mask]
            total_ends[subset] += len(k_sub)

            if len(k_sub) == 0:
                continue

            # Draw at most per_cell_quota from this cell for this subset.
            if len(k_sub) > per_cell_quota:
                idx = rng.choice(len(k_sub), per_cell_quota, replace=False)
                k_sub = k_sub[idx]
                n_sub = n_sub[idx]

            collectors[subset][0].append(k_sub)
            collectors[subset][1].append(n_sub)

        del df, k_both, n_both, dist  # free eagerly

    # Concatenate per-subset pools, then apply final n-stratified subsample.
    subsampled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    subsample_log: dict[str, dict] = {}

    for subset in ("all", "short", "long"):
        k_parts, n_parts = collectors[subset]
        if not k_parts:
            subsampled[subset] = (np.array([], dtype=float), np.array([], dtype=float))
            subsample_log[subset] = {
                "original_read_ends": 0, "used_read_ends": 0, "subsampled": False,
            }
            continue

        k_pool = np.concatenate(k_parts)
        n_pool = np.concatenate(n_parts)
        del k_parts, n_parts  # free the lists

        pool_size = len(k_pool)
        k_pool, n_pool = subsample_for_em(k_pool, n_pool, max_subsample, random_state)

        subsampled[subset] = (k_pool, n_pool)
        subsample_log[subset] = {
            "original_read_ends": int(total_ends[subset]),
            "pool_before_stratify": int(pool_size),
            "used_read_ends": int(len(k_pool)),
            "subsampled": bool(total_ends[subset] > len(k_pool)),
        }

    return subsampled, subsample_log


# ---------------------------------------------------------------------------
# Three-subset fit + comparison diagnostic
# ---------------------------------------------------------------------------


def fit_and_compare_from_parquets(
    cell_parquet_paths: dict[str, str],
    short_range_max: int = 1_000,
    min_distance: int = 20_000,
    output_dir: str = "output/03_mixture_model",
    max_subsample: int = 10_000_000,
    random_state: int = 42,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_restarts: int = 5,
) -> dict:
    """Fit mixture on three distance subsets using streaming subsampling.

    Memory-efficient alternative to :func:`fit_and_compare`.  Instead of
    loading all parquets into a single DataFrame, this streams through each
    cell's parquet once (peak RAM ≈ one parquet + the subsample pool).

    Parameters
    ----------
    cell_parquet_paths : dict
        Mapping ``cell_name -> parquet path``.
    short_range_max, min_distance : int
        Thresholds for the short- and long-range subsets (bp).
    output_dir : str
    max_subsample, random_state, max_iter, tol, n_restarts :
        Passed through to EM fitting.

    Returns
    -------
    dict with keys ``models`` and ``comparison``.
    """
    out_path = ensure_dir(output_dir)

    logger.info(
        f"Stream-subsampling from {len(cell_parquet_paths)} cell parquets "
        f"(target {max_subsample:,} read ends per subset)..."
    )
    subsampled, subsample_log = stream_subsample_from_parquets(
        cell_parquet_paths,
        max_subsample=max_subsample,
        short_range_max=short_range_max,
        min_distance=min_distance,
        random_state=random_state,
    )

    with open(out_path / "subsample_log.json", "w") as f:
        json.dump(subsample_log, f, indent=2)

    results: dict[str, BetaBinomialMixture] = {}

    for name in ("all", "short", "long"):
        k_arr, n_arr = subsampled[name]
        if len(k_arr) == 0:
            logger.warning(f"Subset '{name}' is empty; skipping EM fit.")
            continue

        logger.info(f"\nFitting mixture on '{name}' subset ({len(k_arr):,} read ends)...")
        model = BetaBinomialMixture(
            max_iter=max_iter, tol=tol, n_restarts=n_restarts, random_state=random_state,
        )
        model.fit(k_arr, n_arr)
        results[name] = model

        with open(out_path / f"em_fit_{name}.json", "w") as f:
            json.dump(model.to_dict(), f, indent=2)

    comparison = {
        name: {
            "mean_closed": float(m.alpha_[0] / (m.alpha_[0] + m.beta_[0])),
            "mean_accessible": float(m.alpha_[1] / (m.alpha_[1] + m.beta_[1])),
            "od_closed": float(1.0 / (m.alpha_[0] + m.beta_[0] + 1.0)),
            "od_accessible": float(1.0 / (m.alpha_[1] + m.beta_[1] + 1.0)),
            "pi": float(m.pi_),
        }
        for name, m in results.items()
    }

    for param in ("mean_closed", "mean_accessible"):
        vals = [comparison[s][param] for s in comparison]
        if len(vals) >= 2:
            spread = max(vals) - min(vals)
            if spread > 0.05:
                logger.warning(
                    f"  WARNING: {param} differs by {spread:.3f} across subsets — "
                    f"investigate possible technical bias between short/long-range pairs."
                )

    with open(out_path / "mixture_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    try:
        _plot_mixture_comparison(comparison, out_path)
        # Use the "all" subsample for the density overlay — it's already the
        # right size for a histogram (~10M points).
        k_all, n_all = subsampled.get("all", (np.array([]), np.array([])))
        _plot_mixture_fit_density_from_arrays(results, k_all, n_all, out_path)
    except Exception as exc:
        logger.warning(f"Failed to generate mixture diagnostic plots: {exc}")

    return {"models": results, "comparison": comparison}


def fit_and_compare(
    all_read_ends: pd.DataFrame,
    short_range_max: int = 1_000,
    min_distance: int = 20_000,
    output_dir: str = "output/03_mixture_model",
    max_subsample: int = 10_000_000,
    random_state: int = 42,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_restarts: int = 5,
) -> dict:
    """Fit mixture on three subsets from an in-memory DataFrame.

    .. note::

       For large datasets (hundreds of cells), prefer
       :func:`fit_and_compare_from_parquets` which streams through cells
       one at a time and uses ~100× less RAM.
    """
    out_path = ensure_dir(output_dir)

    subsets = {
        "all": all_read_ends,
        "short": all_read_ends[all_read_ends["distance"] < short_range_max],
        "long": all_read_ends[all_read_ends["distance"] >= min_distance],
    }

    results: dict[str, BetaBinomialMixture] = {}
    subsample_log: dict[str, dict] = {}

    for name, df in subsets.items():
        if len(df) == 0:
            logger.warning(f"Subset '{name}' is empty; skipping EM fit.")
            continue

        k_all = np.concatenate([df["k1"].values, df["k2"].values]).astype(float)
        n_all = np.concatenate([df["n1"].values, df["n2"].values]).astype(float)
        mask = n_all >= 1
        k_all, n_all = k_all[mask], n_all[mask]

        if len(k_all) == 0:
            logger.warning(f"Subset '{name}' has no read ends with n >= 1.")
            continue

        orig_size = len(k_all)
        k_all, n_all = subsample_for_em(k_all, n_all, max_subsample, random_state)
        subsample_log[name] = {
            "original_read_ends": int(orig_size),
            "used_read_ends": int(len(k_all)),
            "subsampled": bool(orig_size > len(k_all)),
        }

        logger.info(f"\nFitting mixture on '{name}' subset ({len(k_all):,} read ends)...")
        model = BetaBinomialMixture(
            max_iter=max_iter, tol=tol, n_restarts=n_restarts, random_state=random_state,
        )
        model.fit(k_all, n_all)
        results[name] = model

        with open(out_path / f"em_fit_{name}.json", "w") as f:
            json.dump(model.to_dict(), f, indent=2)

    with open(out_path / "subsample_log.json", "w") as f:
        json.dump(subsample_log, f, indent=2)

    comparison = {
        name: {
            "mean_closed": float(m.alpha_[0] / (m.alpha_[0] + m.beta_[0])),
            "mean_accessible": float(m.alpha_[1] / (m.alpha_[1] + m.beta_[1])),
            "od_closed": float(1.0 / (m.alpha_[0] + m.beta_[0] + 1.0)),
            "od_accessible": float(1.0 / (m.alpha_[1] + m.beta_[1] + 1.0)),
            "pi": float(m.pi_),
        }
        for name, m in results.items()
    }

    for param in ("mean_closed", "mean_accessible"):
        vals = [comparison[s][param] for s in comparison]
        if len(vals) >= 2:
            spread = max(vals) - min(vals)
            if spread > 0.05:
                logger.warning(
                    f"  WARNING: {param} differs by {spread:.3f} across subsets — "
                    f"investigate possible technical bias between short/long-range pairs."
                )

    with open(out_path / "mixture_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    try:
        _plot_mixture_comparison(comparison, out_path)
        k_obs = np.concatenate([all_read_ends["k1"].values, all_read_ends["k2"].values])
        n_obs = np.concatenate([all_read_ends["n1"].values, all_read_ends["n2"].values])
        mask = n_obs >= 1
        _plot_mixture_fit_density_from_arrays(results, k_obs[mask], n_obs[mask], out_path)
    except Exception as exc:
        logger.warning(f"Failed to generate mixture diagnostic plots: {exc}")

    return {"models": results, "comparison": comparison}


# ---------------------------------------------------------------------------
# Per-cell diagnostic fitting
# ---------------------------------------------------------------------------


def fit_cell_specific_models(
    cell_parquet_paths: dict[str, str],
    pooled_model: BetaBinomialMixture,
    max_samples_per_cell: int = 1_000_000,
    n_threads: int = 4,
    output_dir: str = "output/03_mixture_model",
    group_labels: "pd.Series | None" = None,
) -> pd.DataFrame:
    """Fit mixture independently per cell for diagnostic comparison.

    The pipeline still uses the pooled model for posteriors — this function
    exists only to flag cells with divergent parameters.
    """
    from joblib import Parallel, delayed

    out_path = ensure_dir(output_dir)

    def _fit_one_cell(cell_name: str, parquet_path: str) -> dict:
        df = pd.read_parquet(parquet_path)
        k_all = np.concatenate([df["k1"].values, df["k2"].values]).astype(float)
        n_all = np.concatenate([df["n1"].values, df["n2"].values]).astype(float)
        mask = n_all >= 1
        k_all, n_all = k_all[mask], n_all[mask]

        if len(k_all) < 50:
            # Too few observations to fit a 5-parameter mixture.
            return {"cell_name": cell_name, "n_read_ends": int(len(k_all)),
                    "pi": np.nan,
                    "alpha_closed": np.nan, "beta_closed": np.nan,
                    "alpha_accessible": np.nan, "beta_accessible": np.nan,
                    "mean_closed": np.nan, "mean_accessible": np.nan,
                    "overdispersion_closed": np.nan, "overdispersion_accessible": np.nan,
                    "log_likelihood": np.nan, "n_iter": 0, "converged": False}

        if len(k_all) > max_samples_per_cell:
            rng = np.random.RandomState(hash(cell_name) % 2**31)
            idx = rng.choice(len(k_all), max_samples_per_cell, replace=False)
            k_all, n_all = k_all[idx], n_all[idx]

        model = BetaBinomialMixture(max_iter=100, tol=1e-5, n_restarts=3)
        try:
            model.fit(k_all, n_all)
        except Exception as exc:
            logger.warning(f"Cell {cell_name} fit failed: {exc}")
            return {"cell_name": cell_name, "n_read_ends": int(len(k_all)),
                    "pi": np.nan, "alpha_closed": np.nan, "beta_closed": np.nan,
                    "alpha_accessible": np.nan, "beta_accessible": np.nan,
                    "mean_closed": np.nan, "mean_accessible": np.nan,
                    "overdispersion_closed": np.nan, "overdispersion_accessible": np.nan,
                    "log_likelihood": np.nan, "n_iter": 0, "converged": False}

        return {"cell_name": cell_name, "n_read_ends": int(len(k_all)), **model.to_dict()}

    items = list(cell_parquet_paths.items())
    if n_threads > 1 and len(items) > 1:
        results = Parallel(n_jobs=n_threads)(
            delayed(_fit_one_cell)(name, path) for name, path in items
        )
    else:
        results = [_fit_one_cell(name, path) for name, path in items]

    cell_params = pd.DataFrame(results)
    if group_labels is not None:
        cell_params = cell_params.merge(
            group_labels.rename("group").reset_index().rename(columns={"index": "cell_name"}),
            on="cell_name", how="left",
        )

    cell_params.to_csv(out_path / "cell_specific_mixture_params.tsv", sep="\t", index=False)

    try:
        _plot_cell_vs_pooled(cell_params, pooled_model, out_path)
    except Exception as exc:
        logger.warning(f"Cell-vs-pooled diagnostic plot failed: {exc}")

    return cell_params


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def _plot_mixture_comparison(comparison: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not comparison:
        return

    names = list(comparison.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    mean_vals = {name: [comparison[name]["mean_closed"], comparison[name]["mean_accessible"]]
                 for name in names}
    od_vals = {name: [comparison[name]["od_closed"], comparison[name]["od_accessible"]]
               for name in names}
    pi_vals = [comparison[name]["pi"] for name in names]

    x = np.arange(2)
    width = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        axes[0].bar(x + i * width, mean_vals[name], width, label=name)
        axes[1].bar(x + i * width, od_vals[name], width, label=name)
    axes[0].set_xticks(x + width * (len(names) - 1) / 2)
    axes[0].set_xticklabels(["closed", "accessible"])
    axes[0].set_ylabel("component mean")
    axes[0].set_title("Component mean per subset")
    axes[0].legend(fontsize=8)

    axes[1].set_xticks(x + width * (len(names) - 1) / 2)
    axes[1].set_xticklabels(["closed", "accessible"])
    axes[1].set_ylabel("overdispersion 1/(a+b+1)")
    axes[1].set_title("Component overdispersion")
    axes[1].legend(fontsize=8)

    axes[2].bar(np.arange(len(names)), pi_vals)
    axes[2].set_xticks(np.arange(len(names)))
    axes[2].set_xticklabels(names)
    axes[2].set_ylabel("pi (accessible fraction)")
    axes[2].set_title("Mixing weight pi")

    plt.tight_layout()
    plt.savefig(out_path / "mixture_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_mixture_fit_density_from_arrays(
    models: dict[str, BetaBinomialMixture],
    k_all: np.ndarray,
    n_all: np.ndarray,
    out_path: Path,
) -> None:
    """Overlay fitted mixture densities on observed k/n histograms for small n.

    Accepts pre-stacked, pre-filtered ``(k, n)`` arrays (e.g. from the
    stream-subsampled pool) so no full DataFrame is needed in memory.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not models or len(k_all) == 0:
        return

    n_values = [3, 5, 10, 20]
    fig, axes = plt.subplots(1, len(n_values), figsize=(4 * len(n_values), 3.5), sharey=False)
    if len(n_values) == 1:
        axes = [axes]

    for ax, n_val in zip(axes, n_values):
        mask_n = n_all == n_val
        k_sub = k_all[mask_n]
        if len(k_sub) == 0:
            ax.set_title(f"n={n_val} (no data)")
            continue

        k_grid = np.arange(n_val + 1)
        obs_counts = np.bincount(k_sub.astype(int), minlength=n_val + 1)
        obs_density = obs_counts / obs_counts.sum()
        ax.bar(k_grid, obs_density, alpha=0.35, color="steelblue", label="observed")

        for name, model in models.items():
            n_arr = np.full_like(k_grid, n_val, dtype=float)
            log_p0 = betabinom_logpmf(k_grid.astype(float), n_arr, model.alpha_[0], model.beta_[0])
            log_p1 = betabinom_logpmf(k_grid.astype(float), n_arr, model.alpha_[1], model.beta_[1])
            pi = float(np.clip(model.pi_, 1e-12, 1.0 - 1e-12))
            p = (1 - pi) * np.exp(log_p0) + pi * np.exp(log_p1)
            ax.plot(k_grid, p, marker="o", label=name)

        ax.set_title(f"n = {n_val}")
        ax.set_xlabel("k (methylated)")
        ax.set_ylabel("density / probability")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path / "mixture_fit_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cell_vs_pooled(
    cell_params: pd.DataFrame,
    pooled_model: BetaBinomialMixture,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = cell_params.dropna(subset=["mean_closed", "mean_accessible"])
    if len(valid) == 0:
        return

    pooled_mean_closed = pooled_model.alpha_[0] / (pooled_model.alpha_[0] + pooled_model.beta_[0])
    pooled_mean_accessible = pooled_model.alpha_[1] / (pooled_model.alpha_[1] + pooled_model.beta_[1])

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.scatter(valid["mean_accessible"], valid["mean_closed"], alpha=0.7, s=20)
    ax.scatter([pooled_mean_accessible], [pooled_mean_closed],
               marker="x", s=150, color="red", label="pooled")
    ax.set_xlabel("mean_accessible (per cell)")
    ax.set_ylabel("mean_closed (per cell)")
    ax.set_title("Per-cell component means")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(valid["pi"], bins=30, color="steelblue", alpha=0.8)
    ax.axvline(float(pooled_model.pi_), color="red", linestyle="--", label=f"pooled pi={float(pooled_model.pi_):.3f}")
    ax.set_xlabel("pi (accessible fraction)")
    ax.set_ylabel("# cells")
    ax.set_title("Mixing weight across cells")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.scatter(
        valid["overdispersion_accessible"], valid["overdispersion_closed"], alpha=0.7, s=20
    )
    pooled_od_c = 1.0 / (pooled_model.alpha_[0] + pooled_model.beta_[0] + 1.0)
    pooled_od_a = 1.0 / (pooled_model.alpha_[1] + pooled_model.beta_[1] + 1.0)
    ax.scatter([pooled_od_a], [pooled_od_c], marker="x", s=150, color="red", label="pooled")
    ax.set_xlabel("overdispersion_accessible")
    ax.set_ylabel("overdispersion_closed")
    ax.set_title("Per-cell overdispersion")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if "group" in valid.columns and valid["group"].notna().any():
        groups = [g for g in valid["group"].dropna().unique()]
        data = [valid.loc[valid["group"] == g, "mean_accessible"].dropna().values for g in groups]
        if any(len(d) > 0 for d in data):
            ax.violinplot(data, showmeans=True)
            ax.set_xticks(np.arange(1, len(groups) + 1))
            ax.set_xticklabels(groups, rotation=30, ha="right")
            ax.set_ylabel("mean_accessible")
            ax.set_title("mean_accessible by group")
        else:
            ax.set_visible(False)
    else:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path / "cell_vs_pooled_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
