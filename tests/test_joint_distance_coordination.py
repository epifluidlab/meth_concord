"""Tests for scripts/joint_distance_coordination.py.

Unit-tests the within-cell correlation accumulators and the distance-matched
partner shuffle, plus an end-to-end CLI smoke test on synthetic posteriors with
a KNOWN coordination structure (ends share a region latent → positive
GCH–GCH/HCG–HCG; accessibility = 1−methylation → negative cross), where the
distance-matched shuffle must collapse every correlation toward zero.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / \
    "joint_distance_coordination.py"


def _mod():
    spec = importlib.util.spec_from_file_location("jdc", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------
# Unit: correlation accumulators reproduce numpy's Pearson r
# --------------------------------------------------------------------------

def test_center_sums_match_pearson_single_bin():
    m = _mod()
    rng = np.random.default_rng(0)
    x = rng.random(500); y = 0.6 * x + 0.4 * rng.random(500)
    bins = np.zeros(500, dtype=np.int64)
    mask = np.ones(500, dtype=bool)
    acc = m._center(m._sums(bins, x, y, mask, 1))
    r, n = m._corr(acc, min_pairs=1)
    assert n[0] == 500
    assert abs(r[0] - np.corrcoef(x, y)[0, 1]) < 1e-9


def test_within_cell_pooling_removes_cell_mean_offset():
    """Two 'cells' with opposite global offsets but no within-cell coupling →
    a naive pooled r would be high; within-cell pooling must be ~0."""
    m = _mod()
    rng = np.random.default_rng(1)
    nb = 1
    acc = np.zeros((4, nb))
    for off in (-0.4, 0.4):                       # two cells, opposite offsets
        x = np.clip(0.5 + off + 0.05 * rng.standard_normal(400), 0, 1)
        y = np.clip(0.5 + off + 0.05 * rng.standard_normal(400), 0, 1)  # indep noise
        acc += m._center(m._sums(np.zeros(400, np.int64), x, y,
                                 np.ones(400, bool), nb))
    r, _ = m._corr(acc, min_pairs=1)
    assert abs(r[0]) < 0.2                        # cell-mean offset did not leak


# --------------------------------------------------------------------------
# Unit: the shuffle stays within distance bin and is a permutation
# --------------------------------------------------------------------------

def test_within_bin_partner_stays_in_bin_and_permutes():
    m = _mod()
    rng = np.random.default_rng(2)
    bins = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    p = m._within_bin_partner(bins, rng)
    assert np.array_equal(np.sort(p), np.arange(bins.size))   # a permutation
    assert np.array_equal(bins[p], bins)                      # partner same bin


def test_shuffle_kills_correlation():
    m = _mod()
    rng = np.random.default_rng(3)
    n = 2000
    z = rng.random(n)
    x = np.clip(z + 0.03 * rng.standard_normal(n), 0, 1)
    y = np.clip(z + 0.03 * rng.standard_normal(n), 0, 1)
    bins = np.zeros(n, dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    r_real, _ = m._corr(m._center(m._sums(bins, x, y, mask, 1)), 1)
    p = m._within_bin_partner(bins, np.random.default_rng(4))
    r_shuf, _ = m._corr(m._center(m._sums(bins, x, y[p], mask, 1)), 1)
    assert r_real[0] > 0.8
    assert abs(r_shuf[0]) < 0.1


def test_build_edges_spans_range():
    m = _mod()
    e = m._build_edges(1, 1_000_000, 8)
    assert e[0] == 1 and abs(e[-1] - 1_000_000) < 1e-6
    assert len(e) == 6 * 8 + 1


# --------------------------------------------------------------------------
# Unit: band-summary helpers
# --------------------------------------------------------------------------

def test_band_edges_and_human_dist():
    m = _mod()
    assert m._band_edges(1, 1_000_000) == [1, 1_000, 10_000, 100_000, 1_000_000]
    assert m._band_edges(10_000, 200_000) == [10_000, 100_000, 200_000]
    assert m._human_dist(7) == "7bp"
    assert m._human_dist(6_800) == "6.8kb"
    assert m._human_dist(1_000_000) == "1Mb"


def test_r_scalar_matches_pearson():
    m = _mod()
    rng = np.random.default_rng(1)
    x = rng.random(400); y = 0.6 * x + 0.4 * rng.random(400)
    acc = m._center(m._sums(np.zeros(400, np.int64), x, y,
                            np.ones(400, bool), 1))      # (4, 1)
    r, n = m._r_scalar(acc[:, 0])
    assert n == 400
    assert abs(r - np.corrcoef(x, y)[0, 1]) < 1e-9


def test_pool_band_selects_right_columns():
    m = _mod()
    acc = np.arange(4 * 5, dtype=float).reshape(4, 5)
    centers = np.array([5, 50, 500, 5_000, 50_000], float)
    # [1, 1000) → first three bins
    assert np.allclose(m._pool_band(acc, centers, 1, 1_000, last=False),
                       acc[:, :3].sum(axis=1))
    # last band is inclusive of the upper edge
    assert np.allclose(m._pool_band(acc, centers, 1_000, 50_000, last=True),
                       acc[:, 3:].sum(axis=1))


# --------------------------------------------------------------------------
# Integration: CLI on synthetic posteriors with a known structure
# --------------------------------------------------------------------------

def _write_cell(gdir, hdir, cell, seed, n=600, dist=50_000):
    """Pairs whose two ends share a region latent z (→ correlated a, correlated
    m) with accessibility = 1 − methylation (→ anti-correlated cross)."""
    rng = np.random.default_rng(seed)
    z = rng.random(n)
    sd = 0.05
    a1 = np.clip(z + sd * rng.standard_normal(n), 0.01, 0.99)
    a2 = np.clip(z + sd * rng.standard_normal(n), 0.01, 0.99)
    m1 = np.clip(1.0 - z + sd * rng.standard_normal(n), 0.01, 0.99)
    m2 = np.clip(1.0 - z + sd * rng.standard_normal(n), 0.01, 0.99)
    p1 = 1_000_000 + 1_000 * np.arange(n)
    p2 = p1 + dist
    cols = ["chrom", "pos1", "pos2", "n1", "n2", "posterior1", "posterior2"]
    pd.DataFrame({"chrom": "chr1", "pos1": p1, "pos2": p2, "n1": 5, "n2": 5,
                  "posterior1": a1, "posterior2": a2})[cols].to_parquet(
        gdir / f"{cell}.posteriors.parquet", index=False)
    pd.DataFrame({"chrom": "chr1", "pos1": p1, "pos2": p2, "n1": 5, "n2": 5,
                  "posterior1": m1, "posterior2": m2})[cols].to_parquet(
        hdir / f"{cell}.posteriors.parquet", index=False)


def test_cli_end_to_end(tmp_path):
    gdir = tmp_path / "gch"; hdir = tmp_path / "hcg"
    gdir.mkdir(); hdir.mkdir()
    for i in range(3):
        _write_cell(gdir, hdir, f"C{i}", seed=10 + i)
    out = tmp_path / "fig.pdf"
    cmd = [sys.executable, str(_SCRIPT),
           "--gch-dir", str(gdir), "--hcg-dir", str(hdir),
           "--out", str(out),
           "--min-distance", "10000", "--max-distance", "200000",
           "--bins-per-decade", "2", "--min-pairs", "20",
           "--min-gpc-sites", "2", "--min-cpg-sites", "2", "--n-threads", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert out.exists()
    tsv = out.with_suffix(".tsv")
    assert tsv.exists()
    # the band-level summary stats log is printed
    assert "concordance summary by distance band" in r.stdout
    assert "headline (within-cell r)" in r.stdout
    assert "10kb-100kb" in r.stdout                # the populated ~50 kb band
    df = pd.read_csv(tsv, sep="\t")
    # the populated bin (~50 kb) must show the planted structure
    pop = df[np.isfinite(df["gg_r_contact"]) & (df["gg_n"] >= 20)]
    assert len(pop) >= 1
    row = pop.iloc[(pop["center_bp"] - 50_000).abs().argmin()]
    assert row["gg_r_contact"] > 0.6           # ends share accessibility
    assert row["hh_r_contact"] > 0.6           # ends share methylation
    assert row["gh_r_contact"] < -0.3          # acc_i ↔ meth_j anti-correlated
    # distance-matched shuffle collapses every correlation toward 0
    assert abs(row["gg_r_shuffle"]) < 0.2
    assert abs(row["hh_r_shuffle"]) < 0.2
    assert abs(row["gh_r_shuffle"]) < 0.2


_N_REG = 40
_REG_SIZE = 10_000
# fixed compartment latent A_r per region (SAME across cells, so pooling many
# cells denoises the same latent); methylation = 1 − A_r.
_A = 0.2 + 0.6 * (np.arange(_N_REG) + 0.5) / _N_REG


def _write_region_cell(gdir, hdir, cell, seed, per_reg=2, dist=200, noise=0.7):
    """SPARSE per cell: only ``per_reg`` ends per region, so a single cell can't
    denoise (per-cell mean of ~2 noisy ends is still noisy) — only pooling the
    region across MANY cells recovers the planted −1 compartment coupling."""
    rng = np.random.default_rng(seed)
    cols = ["chrom", "pos1", "pos2", "n1", "n2", "posterior1", "posterior2"]
    g, h = [], []
    for r in range(_N_REG):
        A = _A[r]
        base = 1_000_000 + r * _REG_SIZE * 2         # regions well separated
        p1 = base + 10 * np.arange(per_reg)
        p2 = p1 + dist                               # both ends inside the region
        a1 = np.clip(A + noise * rng.standard_normal(per_reg), 0.01, 0.99)
        a2 = np.clip(A + noise * rng.standard_normal(per_reg), 0.01, 0.99)
        m1 = np.clip(1 - A + noise * rng.standard_normal(per_reg), 0.01, 0.99)
        m2 = np.clip(1 - A + noise * rng.standard_normal(per_reg), 0.01, 0.99)
        g.append(pd.DataFrame({"chrom": "chr1", "pos1": p1, "pos2": p2, "n1": 5,
                               "n2": 5, "posterior1": a1, "posterior2": a2}))
        h.append(pd.DataFrame({"chrom": "chr1", "pos1": p1, "pos2": p2, "n1": 5,
                               "n2": 5, "posterior1": m1, "posterior2": m2}))
    pd.concat(g)[cols].to_parquet(gdir / f"{cell}.posteriors.parquet", index=False)
    pd.concat(h)[cols].to_parquet(hdir / f"{cell}.posteriors.parquet", index=False)


def _run_gh(tmp_path, region_bin, n_cells=30):
    gdir = tmp_path / "g"; hdir = tmp_path / "h"
    gdir.mkdir(parents=True, exist_ok=True); hdir.mkdir(parents=True, exist_ok=True)
    for i in range(n_cells):
        _write_region_cell(gdir, hdir, f"R{i}", seed=70 + i)
    out = tmp_path / f"r{region_bin}.pdf"
    cmd = [sys.executable, str(_SCRIPT), "--gch-dir", str(gdir),
           "--hcg-dir", str(hdir), "--out", str(out),
           "--min-distance", "50", "--max-distance", "1000",
           "--bins-per-decade", "3", "--min-pairs", "30", "--n-threads", "1",
           "--region-bin-size", str(region_bin)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    df = pd.read_csv(out.with_suffix(".tsv"), sep="\t")
    pop = df[np.isfinite(df["gh_r_contact"]) & (df["gh_n"] >= 30)]
    row = pop.iloc[(pop["center_bp"] - 200).abs().argmin()]
    return row, r.stdout


def _write_offset_cell(gdir, hdir, cell, level, seed, n=400, dist=200):
    """A cell with a global offset `level` but NO within-cell end-to-end coupling
    (the two ends are independent draws around `level`).  Pooling cells of
    different levels makes r(a_i,a_j) high (cell-type covariation); within-cell
    centring removes it."""
    rng = np.random.default_rng(seed)
    cols = ["chrom", "pos1", "pos2", "n1", "n2", "posterior1", "posterior2"]
    def ind():
        return np.clip(level + 0.05 * rng.standard_normal(n), 0.01, 0.99)
    p1 = 1_000_000 + 1_000 * np.arange(n); p2 = p1 + dist
    pd.DataFrame({"chrom": "chr1", "pos1": p1, "pos2": p2, "n1": 5, "n2": 5,
                  "posterior1": ind(), "posterior2": ind()})[cols].to_parquet(
        gdir / f"{cell}.posteriors.parquet", index=False)
    pd.DataFrame({"chrom": "chr1", "pos1": p1, "pos2": p2, "n1": 5, "n2": 5,
                  "posterior1": ind(), "posterior2": ind()})[cols].to_parquet(
        hdir / f"{cell}.posteriors.parquet", index=False)


def test_pooled_includes_cell_offset_within_cell_removes_it(tmp_path):
    gdir = tmp_path / "g"; hdir = tmp_path / "h"; gdir.mkdir(); hdir.mkdir()
    _write_offset_cell(gdir, hdir, "A", level=0.8, seed=1)
    _write_offset_cell(gdir, hdir, "B", level=0.2, seed=2)

    def run(mode):
        out = tmp_path / f"{mode}.pdf"
        cmd = [sys.executable, str(_SCRIPT), "--gch-dir", str(gdir),
               "--hcg-dir", str(hdir), "--out", str(out),
               "--min-distance", "50", "--max-distance", "1000",
               "--bins-per-decade", "3", "--min-pairs", "30", "--n-threads", "1",
               "--cell-centering", mode]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        df = pd.read_csv(out.with_suffix(".tsv"), sep="\t")
        pop = df[np.isfinite(df["gg_r_contact"]) & (df["gg_n"] >= 30)]
        return pop.iloc[(pop["center_bp"] - 200).abs().argmin()]["gg_r_contact"], r.stdout

    pooled, out_p = run("pooled")
    within, _ = run("within-cell")
    assert pooled > 0.7                       # cell-type offset inflates pooled r
    assert abs(within) < 0.2                  # within-cell centring removes it
    assert "pooled across cells" in out_p


def test_region_pooling_across_cells_recovers_coupling(tmp_path):
    # raw single-end posteriors: cross coupling attenuated by per-end noise
    raw, _ = _run_gh(tmp_path / "raw", region_bin=0)
    # region means POOLED ACROSS 30 SPARSE CELLS recover the planted −1 coupling,
    # which no single (2-end) cell could denoise on its own.
    reg, out = _run_gh(tmp_path / "reg", region_bin=_REG_SIZE)
    assert abs(raw["gh_r_contact"]) < 0.45            # per-end: attenuated
    assert reg["gh_r_contact"] < -0.7                 # pooled region: strong
    assert reg["gh_r_contact"] < raw["gh_r_contact"] - 0.3
    # same-modality terms go ~1 below the bin size (both ends share region mean)
    assert reg["gg_r_contact"] > 0.9
    assert "pooled across all cells" in out           # pass-1 ran
