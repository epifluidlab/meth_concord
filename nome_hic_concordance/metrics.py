"""Label-agreement metrics for an embedding / feature space.

Used to put a NUMBER on "does this representation separate the cell types?"
so competing matrices (Δ, coordination C, direction F, marginal p̄,
accessibility) can be ranked instead of eyeballed.

Four complementary views, because each one alone is misleading:

* ``silhouette`` (ASW) — geometric: how much tighter a label is than its
  nearest neighbouring label.  Sensitive to cluster *shape*; a label split
  into two well-separated islands scores badly even though it is perfectly
  identifiable.
* ``knn_purity`` — local: the share of each cell's k nearest neighbours
  carrying its own label.  Immune to fragmentation (only the neighbourhood
  matters), so pairing it with ASW distinguishes "mixed" from "shattered":
  high purity + low ASW == clean but fragmented.
* ``ari`` / ``nmi`` — global partition agreement against a k-means
  clustering at k = #labels.  Answers "would unsupervised clustering
  recover these labels?"

``knn_chance`` (= Σ p_g², the expected purity under random neighbours) is
reported alongside because a raw purity is uninterpretable without it: with
imbalanced or few classes, chance alone can be high.

This lives in its own module rather than in ``comparison`` because
``comparison`` imports ``visualization``, which needs these too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_NAN = float("nan")
_KEYS = ("silhouette", "knn_purity", "knn_chance", "ari", "nmi",
         "n_cells", "n_groups")


def silhouette(scores: np.ndarray, labels: pd.Series | np.ndarray) -> float:
    """Mean silhouette width of ``labels`` in ``scores`` space."""
    from sklearn.metrics import silhouette_score
    y = np.asarray(labels)
    if len(np.unique(y)) < 2 or scores.shape[0] < 3:
        return _NAN
    try:
        return float(silhouette_score(scores, y))
    except Exception:
        return _NAN


def knn_purity(scores: np.ndarray, labels: pd.Series | np.ndarray,
               k: int = 15) -> float:
    """Fraction of each cell's k nearest neighbours sharing its label."""
    from sklearn.neighbors import NearestNeighbors
    y = np.asarray(labels)
    n = scores.shape[0]
    if n < 3:
        return _NAN
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(scores)
    _, idx = nn.kneighbors(scores)
    idx = idx[:, 1:]                                     # drop self
    return float(np.mean([(y[idx[i]] == y[i]).mean() for i in range(n)]))


def knn_chance(labels: pd.Series | np.ndarray) -> float:
    """Expected kNN purity under random neighbours: Σ_g p_g²."""
    y = pd.Series(np.asarray(labels))
    if len(y) == 0:
        return _NAN
    p = y.value_counts(normalize=True).to_numpy()
    return float((p ** 2).sum())


def cluster_agreement(scores: np.ndarray, labels: pd.Series | np.ndarray,
                      random_state: int = 42) -> tuple[float, float]:
    """(ARI, NMI) between ``labels`` and k-means at k = #labels.

    k-means (not Leiden) so the result is deterministic and has no
    resolution parameter to tune — the question is only whether an
    unsupervised partition of the same cardinality recovers the labels.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (adjusted_rand_score,
                                 normalized_mutual_info_score)
    y = np.asarray(labels)
    n_groups = len(np.unique(y))
    if n_groups < 2 or scores.shape[0] <= n_groups:
        return _NAN, _NAN
    try:
        km = KMeans(n_clusters=n_groups, random_state=random_state,
                    n_init=10).fit_predict(scores)
    except Exception:
        return _NAN, _NAN
    return (float(adjusted_rand_score(y, km)),
            float(normalized_mutual_info_score(y, km)))


def label_agreement(
    scores: np.ndarray,
    labels: pd.Series | np.ndarray | None,
    k: int = 15,
    random_state: int = 42,
) -> dict[str, float]:
    """All metrics at once; NaNs when ``labels`` is missing or degenerate.

    Cells whose label is null are dropped (an unlabelled cell has nothing to
    agree with) rather than treated as their own class, which would depress
    every metric.
    """
    empty = {key: _NAN for key in _KEYS}
    if labels is None:
        return empty
    y = pd.Series(np.asarray(labels))
    keep = y.notna().to_numpy()
    if keep.sum() < 3:
        return empty
    X = np.asarray(scores, dtype=float)[keep]
    y = y[keep].to_numpy()
    if len(np.unique(y)) < 2:
        return empty
    ari, nmi = cluster_agreement(X, y, random_state=random_state)
    return {
        "silhouette": silhouette(X, y),
        "knn_purity": knn_purity(X, y, k=k),
        "knn_chance": knn_chance(y),
        "ari": ari,
        "nmi": nmi,
        "n_cells": float(X.shape[0]),
        "n_groups": float(len(np.unique(y))),
    }


def format_metrics(m: dict[str, float], space: str = "") -> str:
    """One-line caption for a figure.  Empty string when nothing is finite."""
    if not m or not np.isfinite(m.get("knn_purity", _NAN)):
        return ""
    parts = [
        f"ASW {m['silhouette']:.2f}",
        f"kNN-purity {m['knn_purity']:.2f} (chance {m['knn_chance']:.2f})",
        f"ARI {m['ari']:.2f}",
        f"NMI {m['nmi']:.2f}",
    ]
    line = "  ·  ".join(parts)
    return f"{line}   [{space}]" if space else line
