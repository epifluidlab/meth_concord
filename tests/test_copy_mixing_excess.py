"""copy_mixing_excess.py: X's excess concordance from its copies' levels.

A synthetic X whose copies differ in 100 kb domains -- Xi above Xa by 0.4
in a fifth of them, below by 0.2 in the rest -- read with noise at each
SNP; an autosome whose haplotypes do not differ.  Within a domain the
expected excess is ½ E[δ²] = 0.032, across distant domains ½ E[δ]² =
0.0032, and the noise must add nothing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "copy_mixing_excess",
    Path(__file__).resolve().parents[1] / "scripts" / "copy_mixing_excess.py")
cme = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cme)

L = 30_000_000
DOMAIN = 100_000


def _true_delta(pos, rng):
    dom = rng.random(L // DOMAIN + 1)
    return np.where(dom[pos // DOMAIN] < 0.2, 0.4, -0.2)


def test_lag_sums_is_every_pair():
    rng = np.random.default_rng(0)
    pos = np.sort(rng.choice(2_000_000, 400, replace=False))
    v = rng.normal(size=pos.size)
    edges = np.array([1e3, 1e4, 1e5, 1e6])
    blk, S, N = cme.lag_sums(pos[::-1], v[::-1], edges, block=500_000)
    for k in range(3):
        s = n = 0.0
        for i in range(pos.size):
            for j in range(i + 1, pos.size):
                if edges[k] <= pos[j] - pos[i] < edges[k + 1]:
                    s += v[i] * v[j]
                    n += 1
        assert np.isclose(S[k].sum(), s) and N[k].sum() == n
    # pairs counted in the block of their left point
    assert list(blk) == sorted(set(pos // 500_000))


def test_prediction_is_the_copies_difference_not_the_noise():
    rng = np.random.default_rng(1)
    pos = np.sort(rng.choice(L, 15_000, replace=False))
    truth = _true_delta(pos, rng)
    noisy = truth + rng.normal(0, 0.3, pos.size)
    edges = 10.0 ** cme.FINE
    _, S0, N0 = cme.lag_sums(pos, truth, edges, 5_000_000)
    _, S1, N1 = cme.lag_sums(pos, noisy, edges, 5_000_000)
    f0, f1 = cme._curve(S0, N0), cme._curve(S1, N1)
    w = np.ones(len(cme.FINE) - 1)
    near, far = (1e3, 2e4), (1e6, 1e7)
    b0 = cme.band_values(f0, w, [near, far])
    b1 = cme.band_values(f1, w, [near, far])
    assert abs(b0[0] - 0.032) < 0.004 and abs(b0[1] - 0.0032) < 0.002
    assert np.allclose(b1, b0, atol=0.004)
    # an autosome with no copy difference predicts nothing
    a_noise = rng.normal(0, 0.3, pos.size)
    _, SA, NA = cme.lag_sums(pos, a_noise, edges, 5_000_000)
    assert np.all(np.abs(cme.band_values(cme._curve(SA, NA), w,
                                         [near, far])) < 0.003)


def _inputs(tmp: Path):
    rng = np.random.default_rng(2)
    rows = []
    for grp in ("GM12878", "IMR90"):
        for chrom in ("chrX", "chr1"):
            pos = np.sort(rng.choice(L, 15_000, replace=False))
            hcg = (_true_delta(pos, rng) if chrom == "chrX" and
                   grp == "GM12878" else np.zeros(pos.size))
            for p, d in zip(pos, hcg):
                rows.append((grp, chrom, int(p),
                             d + rng.normal(0, 0.3), 8, 9,
                             rng.normal(0, 0.1), 20, 22))
    pd.DataFrame(rows, columns=[
        "group", "chrom", "pos", "HCG_hap1_minus_hap2", "HCG_reads_ref",
        "HCG_reads_alt", "GCH_hap1_minus_hap2", "GCH_reads_ref",
        "GCH_reads_alt"]).to_csv(tmp / "snps.tsv", sep="\t", index=False)
    conc = []
    for ctx, v in (("HCG", 0.03), ("GCH", 0.0)):
        for g, kind in (("all cells", "all"), ("GM12878", "cell type"),
                        ("IMR90", "cell type")):
            for band in ("1kb-10kb", "10kb-100kb", "100kb-1Mb", "1Mb-10Mb",
                         "all 1kb-10Mb"):
                conc.append((ctx, "chrX − autosomes", kind, g, band, "delta",
                             1000, v, v - 0.01, v + 0.01))
            conc.append((ctx, "chrX − autosomes", kind, g, "1kb-10kb", "w",
                         1000, 0.5, 0.4, 0.6))
    pd.DataFrame(conc, columns=[
        "context", "comparison", "group_kind", "group", "band", "statistic",
        "pairs_test", "diff", "diff_lo", "diff_hi"]).to_csv(
        tmp / "conc.tsv", sep="\t", index=False)
    ps = [("chrX", c, lo, round(lo + 0.1, 2), 100, 0.01, 0.1, "GM12878")
          for c in ("Xi", "Xa") for lo in np.round(np.arange(3, 7, 0.1), 2)]
    pd.DataFrame(ps, columns=["set", "copy", "log10_lo", "log10_hi",
                              "contacts", "share", "per_kb", "group"]).to_csv(
        tmp / "ps.tsv", sep="\t", index=False)
    # posterior units: Xa above Xi by 0.08, as δ's mean says on sites
    pd.DataFrame([("HCG", "chrX: hap2 − hap1", "cell type", "GM12878",
                   "all 1kb-10Mb", "level", 0.08, 0.04, 0.12)],
                 columns=["context", "comparison", "group_kind", "group",
                          "band", "statistic", "diff", "diff_lo",
                          "diff_hi"]).to_csv(tmp / "hap.tsv", sep="\t",
                                             index=False)


def test_cli_predicts_x_excess_for_hcg_and_none_for_gch(tmp_path):
    _inputs(tmp_path)
    o = tmp_path / "out"
    cme.main(["--allelic-snps", str(tmp_path / "snps.tsv"),
              "--concordance", str(tmp_path / "conc.tsv"),
              "--ps", str(tmp_path / "ps.tsv"),
              "--haplotype-concordance", str(tmp_path / "hap.tsv"),
              "--chrx-par", "none", "--chrx-escape", "none",
              "--n-boot", "50", "--plot-format", "png",
              "--output-dir", str(o)])
    t = pd.read_csv(o / "copy_mixing_excess.tsv", sep="\t")
    assert set(t["group"]) == {"GM12878", "IMR90"}
    h = t[(t["context"] == "HCG") & (t["group"] == "GM12878")].set_index(
        "band")
    assert 0.024 < h.loc["1kb-10kb", "predicted"] < 0.04
    assert h.loc["1Mb-10Mb", "predicted"] < 0.008
    assert (h["predicted_lo"] < h["predicted"]).all()
    assert abs(h["site_delta_xi_minus_xa"].iloc[0] + 0.08) < 0.02
    assert 0.7 < h["posterior_over_site_k"].iloc[0] < 1.3
    assert h["k_lo"].iloc[0] < h["k_hi"].iloc[0]
    g = t[t["context"] == "GCH"]
    assert (g["predicted"].abs() < 0.002).all()
    assert "posterior_over_site_k" not in g or g[
        "posterior_over_site_k"].isna().all()
    f = pd.read_csv(o / "copy_mixing_excess_fine.tsv", sep="\t")
    assert len(f) == 2 * (len(cme.FINE) - 1)
    assert (o / "copy_mixing_excess.png").exists()
