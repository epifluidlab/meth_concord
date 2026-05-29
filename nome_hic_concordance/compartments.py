"""scHi-C compartment baseline (design doc Part C input / Phase 2).

Turns per-cell Hi-C contacts (the read pairs parsed in Step 2) into a
cells × bins A/B-compartment-score matrix — the scHi-C feature set that
`nome_hic_compare` pits long-range methylation concordance against.

Algorithm (the scHiCluster compartment recipe, implemented self-contained
so it has no fragile external-API dependency):

  per cell, per chromosome:
    1. bin the contacts into a symmetric bin × bin matrix at `resolution`;
    2. (optional) RWR-impute the sparse single-cell matrix on its own
       contact graph — scHiCluster's denoising step;
    3. observed/expected normalize by genomic-distance band;
    4. Pearson-correlate the O/E rows;
    5. compartment score = first eigenvector (PC1) of that correlation
       matrix, sign-oriented to a pseudobulk reference so the sign is
       consistent across cells.

Output: ``compartment_scores.parquet`` (cells × bins, bin labels
``"chr1:0"`` … at ``resolution``).

An optional ``--use-schicluster`` passthrough is provided for users who
want bit-identical scHiCluster output (requires the optional
``schicluster`` + ``cooler`` deps); the self-contained path is the default.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)

AUTOSOMES = [f"chr{i}" for i in range(1, 23)]


def _norm_chrom(c: str) -> str:
    c = str(c)
    return c if c.startswith("chr") else f"chr{c}"


# ---------------------------------------------------------------------------
# Core per-cell, per-chromosome compartment math
# ---------------------------------------------------------------------------

def _bin_matrix(pos1: np.ndarray, pos2: np.ndarray, resolution: int,
                n_bins: int) -> np.ndarray:
    """Symmetric bin × bin contact-count matrix for one chromosome."""
    b1 = np.clip(pos1 // resolution, 0, n_bins - 1)
    b2 = np.clip(pos2 // resolution, 0, n_bins - 1)
    M = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(M, (b1, b2), 1.0)
    np.add.at(M, (b2, b1), 1.0)
    # The two add.at double-count the diagonal (b1==b2); halve it back.
    diag = np.diag(M).copy() / 2.0
    np.fill_diagonal(M, diag)
    return M


def _rwr_impute(M: np.ndarray, restart: float = 0.5) -> np.ndarray:
    """Random-walk-with-restart smoothing on a cell's contact graph.

    ``F = r (I - (1-r) P)^{-1}`` with ``P`` the row-normalized contact
    matrix.  Denoises sparse single-cell maps (scHiCluster's key step).
    """
    n = M.shape[0]
    rs = M.sum(axis=1)
    rs = np.where(rs > 0, rs, 1.0)
    P = M / rs[:, None]
    A = np.eye(n) - (1.0 - restart) * P
    try:
        F = restart * np.linalg.solve(A, np.eye(n))
    except np.linalg.LinAlgError:
        return M
    return 0.5 * (F + F.T)  # resymmetrize


def _oe_normalize(M: np.ndarray) -> np.ndarray:
    """Observed/expected: divide each diagonal band by its mean."""
    n = M.shape[0]
    out = np.zeros_like(M)
    for d in range(n):
        idx = np.arange(n - d)
        band = M[idx, idx + d]
        mu = band.mean()
        if mu > 0:
            vals = band / mu
            out[idx, idx + d] = vals
            out[idx + d, idx] = vals
    return out


def _compartment_pc1(M: np.ndarray) -> np.ndarray:
    """First-eigenvector compartment profile of a contact matrix.

    O/E → Pearson correlation of rows → top eigenvector.  Zero-coverage
    bins are excluded and returned as NaN.
    """
    n = M.shape[0]
    score = np.full(n, np.nan)
    cov = M.sum(axis=1) > 0
    if cov.sum() < 3:
        return score
    oe = _oe_normalize(M[np.ix_(cov, cov)])
    oe = np.log1p(oe)  # stabilize
    C = np.corrcoef(oe)
    C = np.nan_to_num(C, nan=0.0)
    # Top eigenvector of the correlation matrix.
    vals, vecs = np.linalg.eigh(C)
    pc1 = vecs[:, -1]
    score[cov] = pc1
    return score


def _orient(score: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip the sign of ``score`` to positively correlate with ``reference``."""
    mask = np.isfinite(score) & np.isfinite(reference)
    if mask.sum() < 2:
        return score
    if np.corrcoef(score[mask], reference[mask])[0, 1] < 0:
        return -score
    return score


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _load_contacts(path: str, chroms: set[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["chrom", "pos1", "pos2"])
    df = df.copy()
    df["chrom"] = df["chrom"].map(_norm_chrom)
    return df[df["chrom"].isin(chroms)]


def _cell_compartments(
    path: str, chrom_bins: dict[str, int], resolution: int,
    impute: str, restart: float,
) -> dict[str, np.ndarray]:
    """Per-chromosome compartment PC1 (unoriented) for one cell."""
    chroms = set(chrom_bins)
    df = _load_contacts(path, chroms)
    out: dict[str, np.ndarray] = {}
    for chrom, nb in chrom_bins.items():
        sub = df[df["chrom"] == chrom]
        if len(sub) == 0:
            out[chrom] = np.full(nb, np.nan)
            continue
        M = _bin_matrix(sub["pos1"].to_numpy(), sub["pos2"].to_numpy(),
                        resolution, nb)
        if impute == "rwr":
            M = _rwr_impute(M, restart=restart)
        out[chrom] = _compartment_pc1(M)
    return out


def compute_compartment_matrix(
    cell_paths: dict[str, str],
    chrom_sizes: dict[str, int],
    resolution: int = 1_000_000,
    chroms: list[str] | None = None,
    impute: str = "rwr",
    restart: float = 0.5,
    n_threads: int = 4,
    output_dir: str | None = None,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """Cells × bins compartment-score matrix from per-cell contacts.

    Parameters
    ----------
    cell_paths : dict[cell_name -> read_pairs.parquet path]
    chrom_sizes : dict[chrom -> length]
    resolution : bin size in bp (default 1 Mb — the compartment scale).
    chroms : chromosomes to use (default 22 autosomes present in chrom_sizes).
    impute : 'rwr' (scHiCluster-style denoising) or 'none'.
    """
    if chroms is None:
        chroms = [c for c in AUTOSOMES if c in chrom_sizes
                  or _norm_chrom(c) in {_norm_chrom(k) for k in chrom_sizes}]
    # Normalize chrom_sizes keys.
    size_map = {_norm_chrom(k): v for k, v in chrom_sizes.items()}
    chrom_bins = {c: int(np.ceil(size_map[c] / resolution))
                  for c in chroms if c in size_map}
    if not chrom_bins:
        raise ValueError(
            f"No requested chromosomes found in chrom_sizes "
            f"(wanted {chroms[:3]}..., have {list(size_map)[:3]}...).")

    cells = sorted(cell_paths)
    logger.info(
        f"Compartments: {len(cells)} cells × {len(chrom_bins)} chroms "
        f"@ {resolution:,} bp (impute={impute}).")

    # Per-cell, per-chrom PC1 (unoriented).
    def _one(cell):
        return cell, _cell_compartments(
            cell_paths[cell], chrom_bins, resolution, impute, restart)

    if n_threads > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_threads)(delayed(_one)(c) for c in cells)
    else:
        results = [_one(c) for c in cells]
    per_cell = dict(results)

    # Pseudobulk reference per chrom = mean across cells (orient signs to it).
    reference: dict[str, np.ndarray] = {}
    for chrom, nb in chrom_bins.items():
        stack = np.vstack([per_cell[c][chrom] for c in cells])
        reference[chrom] = np.nanmean(stack, axis=0)

    # Assemble oriented cells × bins.
    col_labels: list[str] = []
    for chrom, nb in chrom_bins.items():
        col_labels += [f"{chrom}:{i * resolution}" for i in range(nb)]

    mat = np.full((len(cells), len(col_labels)), np.nan)
    for ri, cell in enumerate(cells):
        offset = 0
        for chrom, nb in chrom_bins.items():
            oriented = _orient(per_cell[cell][chrom], reference[chrom])
            mat[ri, offset:offset + nb] = oriented
            offset += nb

    comp_df = pd.DataFrame(mat, index=cells, columns=col_labels)
    comp_df.index.name = "cell"
    comp_df.columns.name = "bin"

    if save_outputs and output_dir is not None:
        out = ensure_dir(output_dir)
        comp_df.to_parquet(out / "compartment_scores.parquet")
        comp_df.to_csv(out / "compartment_scores.tsv", sep="\t")
        logger.info(f"Wrote {out / 'compartment_scores.parquet'}")
    return comp_df
