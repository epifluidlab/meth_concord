"""Tests for the scHi-C compartment baseline (Phase 2).

Synthetic per-cell contacts with a planted A/B block structure (contacts
enriched within each compartment) → the PC1 compartment score should
recover the blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import compartments


def test_bin_matrix_symmetric_counts():
    pos1 = np.array([0, 1_000_000, 0])
    pos2 = np.array([2_000_000, 1_000_000, 2_000_000])
    M = compartments._bin_matrix(pos1, pos2, 1_000_000, 3)
    assert np.allclose(M, M.T)              # symmetric
    assert M[0, 2] == 2.0 and M[2, 0] == 2.0  # two (0,2) contacts
    assert M[1, 1] == 1.0                    # one diagonal contact, not doubled


def test_compartment_pc1_recovers_planted_blocks():
    # Block O/E-like matrix: within-block high, across-block low.
    n = 20
    comp = np.array([0] * 10 + [1] * 10)
    M = np.ones((n, n)) * 1.0
    for i in range(n):
        for j in range(n):
            if comp[i] == comp[j]:
                M[i, j] += 5.0
    score = compartments._compartment_pc1(M)
    # PC1 should split the two blocks (opposite signs).
    r = np.corrcoef(score, comp)[0, 1]
    assert abs(r) > 0.9


def _make_cell(path, n_bins, comp, n_contacts, resolution, seed):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n_bins, n_contacts)
    same = rng.random(n_contacts) < 0.85           # 85% within-compartment
    j = np.empty(n_contacts, dtype=int)
    for k in range(n_contacts):
        if same[k]:
            pool = np.where(comp == comp[i[k]])[0]
        else:
            pool = np.arange(n_bins)
        j[k] = rng.choice(pool)
    pos1 = i * resolution + rng.integers(0, resolution, n_contacts)
    pos2 = j * resolution + rng.integers(0, resolution, n_contacts)
    pd.DataFrame({"chrom": "chr1", "pos1": pos1.astype("int64"),
                  "pos2": pos2.astype("int64")}).to_parquet(path)


def test_compute_compartment_matrix_recovers_compartments(tmp_path):
    n_bins, resolution = 20, 1_000_000
    comp = np.array([0] * 10 + [1] * 10)
    cell_paths = {}
    for c in range(6):
        p = tmp_path / f"cell{c}.read_pairs.parquet"
        _make_cell(p, n_bins, comp, n_contacts=4000, resolution=resolution, seed=c)
        cell_paths[f"cell{c}"] = str(p)

    comp_df = compartments.compute_compartment_matrix(
        cell_paths, {"chr1": n_bins * resolution},
        resolution=resolution, chroms=["chr1"], impute="none",
        n_threads=1, output_dir=str(tmp_path), save_outputs=True,
    )
    assert comp_df.shape == (6, n_bins)
    assert (tmp_path / "compartment_scores.parquet").exists()
    # Pseudobulk compartment profile recovers the planted blocks (sign of
    # the compartment axis is arbitrary, so compare |correlation|).
    colmean = comp_df.mean(axis=0).to_numpy()
    assert abs(np.corrcoef(colmean, comp)[0, 1]) > 0.8
    # Every cell recovers them strongly AND with a consistent sign (the
    # reference-orientation step makes the sign agree across cells).
    per_cell_r = np.array([np.corrcoef(comp_df.iloc[r].to_numpy(), comp)[0, 1]
                           for r in range(6)])
    assert np.all(np.abs(per_cell_r) > 0.7)
    assert np.all(per_cell_r > 0) or np.all(per_cell_r < 0)


def test_rwr_impute_preserves_shape_and_symmetry():
    rng = np.random.default_rng(0)
    M = rng.poisson(1.0, size=(12, 12)).astype(float)
    M = M + M.T
    F = compartments._rwr_impute(M, restart=0.5)
    assert F.shape == M.shape
    assert np.allclose(F, F.T, atol=1e-8)


def test_compute_requires_known_chrom(tmp_path):
    p = tmp_path / "c.read_pairs.parquet"
    pd.DataFrame({"chrom": ["chr1"], "pos1": [0], "pos2": [1_000_000]}).to_parquet(p)
    with pytest.raises(ValueError, match="chrom"):
        compartments.compute_compartment_matrix(
            {"c": str(p)}, {"chrZ": 5_000_000}, resolution=1_000_000,
            chroms=["chr5"], n_threads=1, save_outputs=False)
