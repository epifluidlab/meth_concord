"""Section C.3: the two RWR graphs, and why one of them is dangerous.

**Genomic-neighbour (default).**  Random walk with restart on the 1-D bin
adjacency graph, per cell per chromosome.  Coordination is spatially
autocorrelated along the genome -- compartments, TADs -- so smoothing along it
is the direct analogue of scHiCluster's convolution + RWR step, and it is
NON-CIRCULAR with respect to cell typing: no cell ever sees another cell.

**Cell-neighbour (off by default).**  Smoothing across a cell kNN graph.
Powerful, and circular the moment the graph is built from the coordination
features themselves -- the clusters then partly reflect the graph rather than
the data, and every downstream ARI is inflated by construction.  The design
says to enforce this in code, not documentation, so:

* the caller must name the feature set's ``provenance`` from a fixed allowlist;
* and :func:`assert_independent` compares the proposed features against the
  coordination matrix and refuses on identity, on equality, and on cell-cell
  geometry that is too similar to be independent.

A label alone would not be enforcement -- anyone can pass the wrong string.
The geometry check is what makes it one.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

#: Feature sets a cell kNN graph may legitimately be built from.  Each is
#: measured independently of the per-cell coordination statistic.
ALLOWED_PROVENANCE = ("marginal_hcg", "marginal_gch", "contact_only",
                      "external")

#: Spearman correlation between the two matrices' cell-cell distance geometries
#: above which the feature set is refused.  Not a p-value: the question is
#: whether the graph would encode the coordination structure, and by this point
#: it already does.
GEOMETRY_MAX = 0.9


class CircularGraphError(ValueError):
    """Raised when a cell kNN graph would be built from the coordination data."""


def genomic_neighbor_graph(bin_chrom, bin_start, bandwidth: int = 2,
                           decay: float = 0.5) -> sp.csr_matrix:
    """Row-stochastic transition matrix over BINS, within chromosome only.

    Weight falls off as ``decay ** |i - j|`` out to ``bandwidth`` bins.  Edges
    never cross a chromosome boundary: two bins that are adjacent in the array
    but on different contigs are not neighbours in the genome, and joining them
    would smooth the last bin of one chromosome into the first of the next.

    Bins must be given in genomic order within each chromosome.
    """
    chrom = np.asarray(bin_chrom)
    start = np.asarray(bin_start, dtype=np.int64)
    n = len(chrom)
    if n == 0:
        return sp.csr_matrix((0, 0))
    order = np.lexsort((start, chrom))
    if not np.array_equal(order, np.arange(n)):
        raise ValueError("bins must be sorted by (chrom, start); a graph built "
                         "on unsorted bins joins non-adjacent loci")

    rows, cols, vals = [], [], []
    for d in range(1, int(bandwidth) + 1):
        same = chrom[d:] == chrom[:-d]
        i = np.flatnonzero(same)
        w = float(decay) ** d
        rows.extend(i.tolist()); cols.extend((i + d).tolist()); vals.extend([w] * i.size)
        rows.extend((i + d).tolist()); cols.extend(i.tolist()); vals.extend([w] * i.size)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    # An isolated bin (a one-bin contig) restarts on itself rather than
    # producing a zero row, which would silently zero that bin's values.
    iso = deg == 0
    if iso.any():
        A = A + sp.csr_matrix((np.ones(iso.sum()),
                               (np.flatnonzero(iso), np.flatnonzero(iso))),
                              shape=(n, n))
        deg = np.asarray(A.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.maximum(deg, 1e-12)) @ A


def _distance_geometry(X: np.ndarray) -> np.ndarray:
    """Upper-triangle of the cell-cell Euclidean distance matrix."""
    X = np.asarray(X, float)
    X = np.where(np.isfinite(X), X, 0.0)
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    iu = np.triu_indices(d.shape[0], k=1)
    return d[iu]


def _spearman(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / denom) if denom > 0 else float("nan")


def assert_independent(features, coordination, provenance: str,
                       geometry_max: float = GEOMETRY_MAX) -> dict:
    """Refuse a cell kNN feature set that is the coordination matrix in disguise.

    Three checks, in increasing subtlety: the same object, the same numbers,
    and the same cell-cell geometry.  The third is the one that matters -- a
    caller who passes the coordination matrix's own SVD scores passes the first
    two.
    """
    if provenance not in ALLOWED_PROVENANCE:
        raise CircularGraphError(
            f"provenance {provenance!r} is not one of {ALLOWED_PROVENANCE}. "
            f"A cell kNN graph must come from a feature set measured "
            f"independently of the coordination statistic (C.3.2).")
    if coordination is None:
        return {"provenance": provenance, "geometry_spearman": float("nan"),
                "checked_against_coordination": False}

    F = np.asarray(features, float)
    C = np.asarray(coordination, float)
    if features is coordination:
        raise CircularGraphError(
            "the cell kNN graph was handed the coordination matrix itself; "
            "the clusters would then partly reflect the graph, not the data "
            "(C.3.2)")
    if F.shape == C.shape and np.allclose(np.nan_to_num(F), np.nan_to_num(C)):
        raise CircularGraphError(
            "the proposed features are numerically identical to the "
            "coordination matrix (C.3.2)")
    if F.shape[0] != C.shape[0]:
        raise CircularGraphError(
            f"features have {F.shape[0]} cells, the coordination matrix "
            f"{C.shape[0]}; they cannot be compared, so independence cannot "
            f"be established")
    rho = _spearman(_distance_geometry(F), _distance_geometry(C))
    if np.isfinite(rho) and rho > geometry_max:
        raise CircularGraphError(
            f"the proposed features reproduce the coordination matrix's "
            f"cell-cell geometry (Spearman {rho:.3f} > {geometry_max}).  "
            f"Declared provenance was {provenance!r}, but a graph built on "
            f"this would encode the very structure the embedding is meant to "
            f"discover (C.3.2).")
    return {"provenance": provenance, "geometry_spearman": float(rho),
            "checked_against_coordination": True}


def cell_neighbor_graph(features, coordination, provenance: str, k: int = 20,
                        geometry_max: float = GEOMETRY_MAX):
    """Row-stochastic kNN transition matrix over CELLS, after the guard.

    Returns ``(P, report)``.  The guard runs first and raises; there is no flag
    to skip it, because a flag to skip it is how it ends up skipped.
    """
    report = assert_independent(features, coordination, provenance, geometry_max)
    X = np.asarray(features, float)
    X = np.where(np.isfinite(X), X, 0.0)
    n = X.shape[0]
    k = int(min(max(k, 1), max(n - 1, 1)))
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    A = sp.csr_matrix((np.ones(n * k), (rows, nn.ravel())), shape=(n, n))
    A = A.maximum(A.T)                       # symmetrise: kNN is not mutual
    deg = np.asarray(A.sum(axis=1)).ravel()
    P = sp.diags(1.0 / np.maximum(deg, 1e-12)) @ A
    report.update(k=k, n_cells=n)
    return P, report
