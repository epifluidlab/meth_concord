"""Tests for the joint GCH+HCG per-cell coordination core (synthetic parquets)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import joint_coordination as jc
from nome_hic_concordance.joint_coordination import (
    _G_DELTA, _H_DELTA, _J_DELTA, _N_O, RVEC, build_matrices, process_cell,
    scan_priors,
)


# --------------------------------------------------------------------------
# Synthetic context-run writers
# --------------------------------------------------------------------------

def _write_ctx(path, pos1, pos2, post1, post2, n1, n2, chrom="chr1"):
    pd.DataFrame({
        "chrom": chrom,
        "pos1": np.asarray(pos1, np.int64),
        "pos2": np.asarray(pos2, np.int64),
        "n1": np.asarray(n1, np.int32),
        "n2": np.asarray(n2, np.int32),
        "posterior1": np.asarray(post1, np.float64),
        "posterior2": np.asarray(post2, np.float64),
    }).to_parquet(path, index=False)


def _ramp_contacts(n_loci=30, step=10_000, base=1_000_000, seed=0):
    """Chain contacts i->i+2 (20kb) and i->i+3 (30kb) along one chromosome,
    with a per-locus accessibility ``a`` in (0.05, 0.95)."""
    rng = np.random.default_rng(seed)
    a = 0.05 + 0.9 * rng.random(n_loci)
    p = base + step * np.arange(n_loci)
    pos1, pos2, a1, a2 = [], [], [], []
    for i in range(n_loci):
        for k in (2, 3):
            if i + k < n_loci:
                pos1.append(p[i]); pos2.append(p[i + k])
                a1.append(a[i]); a2.append(a[i + k])
    return (np.array(pos1), np.array(pos2), np.array(a1), np.array(a2), a, p)


# --------------------------------------------------------------------------
# Structural correctness
# --------------------------------------------------------------------------

def test_process_cell_structure(tmp_path):
    pos1, pos2, a1, a2, *_ = _ramp_contacts()
    n = np.full(len(pos1), 5)
    _write_ctx(tmp_path / "C1.posteriors.parquet", pos1, pos2, a1, a2, n, n)
    # HCG present and observed (m varies)
    m1 = 1.0 - a1; m2 = 1.0 - a2
    _write_ctx(tmp_path / "C1.hcg.parquet", pos1, pos2, m1, m2, n, n)

    cell, out, qc = process_cell(
        tmp_path / "C1.posteriors.parquet", tmp_path / "C1.hcg.parquet",
        ["chrom"], min_d=10_000, max_d=50_000, min_gpc=2, min_cpg=2,
        a_prior=0.5, m_prior=0.5, cap=10**9, seed=0)

    assert cell == "C1"
    assert "chrom" in out and "chr1" in out["chrom"]
    vec = out["chrom"]["chr1"]
    assert vec.shape == (RVEC,)
    assert vec[_N_O] > 0
    assert np.isfinite(vec[_J_DELTA]).all()
    assert np.isfinite(vec[_G_DELTA]) and np.isfinite(vec[_H_DELTA])
    # QC fields present
    assert 0.0 <= qc["frac_pairs_full"] <= 1.0
    assert qc["n_contacts_band"] > 0


def test_missing_hcg_identity(tmp_path):
    """With HCG fully missing (lenient → constant m prior), the methylation
    factor cancels in O/E, so the joint channels reduce to the accessibility
    ratios:  open ≈ active_methylated,  closed ≈ poised,  and the gch signed
    contrast = open − closed."""
    pos1, pos2, a1, a2, *_ = _ramp_contacts(seed=1)
    n = np.full(len(pos1), 5)
    zero = np.zeros(len(pos1), np.int32)
    nan = np.full(len(pos1), np.nan)
    _write_ctx(tmp_path / "C1.posteriors.parquet", pos1, pos2, a1, a2, n, n)
    # HCG run with the SAME keys but no CpG sites (n=0, posterior NaN) -> missing
    _write_ctx(tmp_path / "C1.hcg.parquet", pos1, pos2, nan, nan, zero, zero)

    _, out, _ = process_cell(
        tmp_path / "C1.posteriors.parquet", tmp_path / "C1.hcg.parquet",
        ["chrom"], min_d=10_000, max_d=50_000, min_gpc=2, min_cpg=1,
        a_prior=0.5, m_prior=0.37, cap=10**9, seed=0)
    vec = out["chrom"]["chr1"]
    jo, jc_, jp, jam = vec[_J_DELTA]
    # methylation cancels: open<->am and closed<->poised collapse
    assert jo == pytest.approx(jam, abs=1e-9)
    assert jc_ == pytest.approx(jp, abs=1e-9)
    # gch signed contrast = open - closed
    assert vec[_G_DELTA] == pytest.approx(jo - jc_, abs=1e-9)


def test_observed_methylation_breaks_identity(tmp_path):
    """When HCG is genuinely observed and anti-correlated with accessibility,
    the joint channels diverge from the accessibility-only identity."""
    pos1, pos2, a1, a2, *_ = _ramp_contacts(seed=2)
    n = np.full(len(pos1), 5)
    # methylation present and NOT a constant: distinct from accessibility
    m1 = np.clip(0.8 - 0.5 * a1, 0.05, 0.95)
    m2 = np.clip(0.8 - 0.5 * a2, 0.05, 0.95)
    _write_ctx(tmp_path / "C1.posteriors.parquet", pos1, pos2, a1, a2, n, n)
    _write_ctx(tmp_path / "C1.hcg.parquet", pos1, pos2, m1, m2, n, n)

    _, out, _ = process_cell(
        tmp_path / "C1.posteriors.parquet", tmp_path / "C1.hcg.parquet",
        ["chrom"], min_d=10_000, max_d=50_000, min_gpc=2, min_cpg=2,
        a_prior=0.5, m_prior=0.5, cap=10**9, seed=0)
    vec = out["chrom"]["chr1"]
    jo, jc_, jp, jam = vec[_J_DELTA]
    # methylation now carries information: open != am, closed != poised
    assert abs(jo - jam) > 1e-3
    assert abs(jc_ - jp) > 1e-3


# --------------------------------------------------------------------------
# build_matrices + scan_priors
# --------------------------------------------------------------------------

def test_build_matrices_shapes(tmp_path):
    pos1, pos2, a1, a2, *_ = _ramp_contacts(seed=3)
    n = np.full(len(pos1), 5)
    results = []
    for ci in range(4):
        gp = tmp_path / f"C{ci}.posteriors.parquet"
        hp = tmp_path / f"C{ci}.hcg.parquet"
        _write_ctx(gp, pos1, pos2, a1, a2, n, n)
        _write_ctx(hp, pos1, pos2, 1 - a1, 1 - a2, n, n)
        results.append(process_cell(gp, hp, ["chrom", 5], 10_000, 50_000,
                                    2, 2, 0.5, 0.5, 10**9, ci))
    mats = build_matrices(results, "chrom")
    R = len(mats["region_keys"])
    assert R >= 1
    Dj, Mj, Cj = mats["joint"]
    assert Dj.shape == (4, R * 4)        # 4 channels per region, region-major
    assert Mj.shape == Dj.shape and Cj.shape == Dj.shape
    Dg, Mg, Cg = mats["gch"]
    assert Dg.shape == (4, R)
    Dh, _, _ = mats["hcg"]
    assert Dh.shape == (4, R)


def test_scan_priors_recovers_mean(tmp_path):
    # a context run whose observed posteriors average ~0.3
    rng = np.random.default_rng(0)
    post = np.clip(rng.normal(0.3, 0.05, 200), 0, 1)
    pos = np.arange(200) * 1000 + 1_000_000
    n = np.full(200, 5)
    p = tmp_path / "C.posteriors.parquet"
    _write_ctx(p, pos, pos + 20_000, post, post, n, n)
    prior = scan_priors([p], min_sites=2, n_threads=1)
    assert prior == pytest.approx(0.3, abs=0.03)


def test_scan_priors_ignores_uninformative(tmp_path):
    pos = np.arange(50) * 1000 + 1_000_000
    n_lo = np.full(50, 0)          # below threshold -> ignored
    post = np.full(50, 0.9)
    p = tmp_path / "C.posteriors.parquet"
    _write_ctx(p, pos, pos + 20_000, post, post, n_lo, n_lo)
    # all uninformative -> falls back to 0.5
    assert scan_priors([p], min_sites=2, n_threads=1) == pytest.approx(0.5)
