"""Functional-enrichment analysis of the marker-bin BED files.

Covers the interval engine (merge / coverage / gene overlap, all pure numpy),
the statistics (BH-FDR, hypergeometric ORA, Mann-Whitney coverage test incl. the
perfectly-separated edge case), and an end-to-end run that must recover a planted
signal and degrade gracefully when no annotations are given.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "marker_bin_enrichment.py")


def _mod():
    spec = importlib.util.spec_from_file_location("mbe", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["mbe"] = m
    spec.loader.exec_module(m)
    return m


def _bins(rows):
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


# ------------------------------------------------------------- interval ops --

def test_merge_unions_overlapping_intervals():
    m = _mod()
    out = m._merge(np.array([[100, 300], [200, 400], [500, 600]]))
    assert out.tolist() == [[100, 400], [500, 600]]
    assert m._merge(np.empty((0, 2))).shape == (0, 2)


def test_coverage_fraction():
    m = _mod()
    bins = _bins([("chr1", 0, 1000), ("chr1", 2000, 3000), ("chr2", 0, 1000)])
    annot = _bins([("chr1", 100, 300), ("chr1", 500, 600),
                   ("chr1", 200, 400),                        # overlaps -> merged
                   ("chr1", 2800, 3200)])                     # 0.2 of bin 2 (edge)
    cov = m._coverage(bins, annot)
    # bin 1: merge([100,300),[200,400)) = [100,400) (300) + [500,600) (100) = 0.4
    assert cov[0] == pytest.approx(0.4)
    assert cov[1] == pytest.approx(0.2)          # (2800..3000) = 200/1000
    assert cov[2] == 0.0                          # chr2 uncovered
    assert m._coverage(bins, bins.iloc[:0]).tolist() == [0, 0, 0]


def test_genes_over_bins_and_chr_prefix_agnostic():
    m = _mod()
    bins = _bins([("chr1", 0, 1000), ("chr1", 2000, 3000)])
    genes = pd.DataFrame({
        "chrom": ["1", "1", "1", "chr2"],            # note '1' vs bins' 'chr1'
        "start": [500, 2500, 1500, 500],
        "end":   [600, 2600, 1600, 600],
        "name":  ["G1", "G2", "G3_gap", "G4_chr2"]})
    got = m._genes_over_bins(bins, genes)
    assert got == {"G1", "G2"}                     # gap + other-chrom excluded


def test_canon_and_chrom_key():
    m = _mod()
    assert m._canon("chr1") == "1" and m._canon("1") == "1"
    assert m._canon("chrX") == "X"
    assert sorted(["chr10", "chr2", "chrX", "chr1"], key=m._chrom_key) == \
        ["chr1", "chr2", "chr10", "chrX"]


# -------------------------------------------------------------- statistics ---

def test_bh_fdr_monotone_and_nan_safe():
    m = _mod()
    p = np.array([0.001, 0.01, 0.5, np.nan])
    q = m._bh(p)
    assert np.isnan(q[3])
    fin = np.isfinite(p)
    assert np.all(q[fin] + 1e-12 >= p[fin])
    assert np.all((q[fin] >= 0) & (q[fin] <= 1))


def test_mwu_perfect_separation_is_significant_not_nan():
    """The fg=all-1 / bg=all-0 case must yield a small p, not NaN."""
    m = _mod()
    cov = np.r_[np.ones(30), np.zeros(170)]
    fg = np.r_[np.ones(30, bool), np.zeros(170, bool)]
    row = m._mwu_row("x", cov, fg)
    assert row["log2_fold_change"] > 10
    assert np.isfinite(row["pvalue"]) and row["pvalue"] < 1e-10
    # everything identical -> no difference -> not significant (p=1 or NaN)
    p_flat = m._mwu_row("y", np.ones(200), fg)["pvalue"]
    assert np.isnan(p_flat) or p_flat >= 0.05


def test_ora_hypergeometric_enrichment():
    m = _mod()
    universe = [f"G{i}" for i in range(10)]
    fg = ["G0", "G1", "G2"]
    sets = {"hit": {"G0", "G1", "G2", "G3"}, "miss": {"G7", "G8", "G9"}}
    df = m._ora(fg, universe, sets, min_size=1, max_size=100)
    hit = df[df["term"] == "hit"].iloc[0]
    assert hit["n_overlap"] == 3 and hit["fold_enrichment"] == pytest.approx(2.5)
    assert hit["pvalue"] < 0.1
    assert "miss" not in df["term"].values          # k==0 -> dropped


# ---------------------------------------------------------------- end to end -

def _planted(d: Path):
    """A marker all.bed with 30 fg bins on chr1[0:30], plus matching gene BED,
    ChromHMM (Enh over fg), a CpG annotation in fg, and a GMT gene set."""
    BS = 250_000
    rows = [(c, i * BS, (i + 1) * BS) for c in ("chr1", "chr2")
            for i in range(100)]
    bins = _bins(rows)
    sel = np.zeros(len(bins), bool); sel[:30] = True
    bins["celltype_icc"] = 0.5; bins["celltype_pvalue"] = 1e-3
    bins["celltype_fdr"] = 1e-3; bins["batch_icc"] = 0.01; bins["selected"] = sel
    bins.to_csv(d / "bins.all.bed", sep="\t", index=False)
    genes = [(c, i * BS + 50000, i * BS + 150000, f"G_{c}_{i}", ".", "+")
             for c in ("chr1", "chr2") for i in range(100)]
    pd.DataFrame(genes).to_csv(d / "genes.bed", sep="\t", index=False,
                               header=False)
    pd.DataFrame([("chr1", 0, 30 * BS, "Enh"), ("chr1", 30 * BS, 100 * BS, "Quies"),
                  ("chr2", 0, 100 * BS, "Quies")]).to_csv(
        d / "hmm.bed", sep="\t", index=False, header=False)
    pd.DataFrame([("chr1", i * BS + 10000, i * BS + 20000) for i in range(30)]
                 ).to_csv(d / "cpg.bed", sep="\t", index=False, header=False)
    with open(d / "sets.gmt", "w") as fh:
        fh.write("pathway_A\td\t" + "\t".join(f"G_chr1_{i}" for i in range(30)) + "\n")
        fh.write("pathway_B\td\t" + "\t".join(f"G_chr2_{i}" for i in range(50)) + "\n")


def test_end_to_end_recovers_planted_signal(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    m = _mod()
    _planted(tmp_path)
    out = tmp_path / "out"
    m.main(["--bed", f"gch={tmp_path/'bins.all.bed'}",
            "--gene-annotation", str(tmp_path / "genes.bed"),
            "--chromhmm", str(tmp_path / "hmm.bed"),
            "--annotation", f"CpG={tmp_path/'cpg.bed'}",
            "--gene-sets", f"MSig={tmp_path/'sets.gmt'}",
            "--output-dir", str(out)])
    assert (out / "gch_enrichment.pdf").exists()

    hmm = pd.read_csv(out / "gch_chromhmm_enrichment.tsv", sep="\t")
    enh = hmm[hmm["annotation"] == "state:Enh"].iloc[0]
    assert enh["log2_fold_change"] > 0 and enh["fdr"] < 0.05    # Enh enriched
    quies = hmm[hmm["annotation"] == "state:Quies"].iloc[0]
    assert quies["log2_fold_change"] < 0                        # Quies depleted

    ora = pd.read_csv(out / "gch_MSig_enrichment.tsv", sep="\t")
    assert (ora["term"] == "pathway_A").any()
    assert ora[ora["term"] == "pathway_A"].iloc[0]["fdr"] < 0.05
    assert "pathway_B" not in ora["term"].values               # no fg overlap

    chrom = pd.read_csv(out / "gch_chrom_enrichment.tsv", sep="\t")
    assert chrom[chrom["chrom"] == "chr1"].iloc[0]["frac_of_fg"] == 1.0
    genes = pd.read_csv(out / "gch_foreground_genes.tsv", sep="\t")
    assert len(genes) == 30 and set(genes["gene"]) == {f"G_chr1_{i}"
                                                       for i in range(30)}


def test_runs_without_any_annotation(tmp_path):
    """With only --bed the script still writes the chrom enrichment + a PDF."""
    pytest.importorskip("matplotlib")
    m = _mod()
    _planted(tmp_path)
    out = tmp_path / "out"
    m.main(["--bed", f"gch={tmp_path/'bins.all.bed'}", "--output-dir", str(out)])
    assert (out / "gch_enrichment.pdf").exists()
    assert (out / "gch_chrom_enrichment.tsv").exists()
    assert not (out / "gch_feature_enrichment.tsv").exists()    # no genes given


def test_fdr_foreground_selection(tmp_path):
    """--fdr defines the foreground from celltype_fdr instead of 'selected'."""
    pytest.importorskip("matplotlib")
    m = _mod()
    BS = 250_000
    bins = _bins([("chr1", i * BS, (i + 1) * BS) for i in range(20)])
    bins["celltype_icc"] = 0.5; bins["celltype_pvalue"] = 0.5
    bins["celltype_fdr"] = [0.001] * 5 + [0.5] * 15          # 5 significant
    bins["batch_icc"] = 0.01; bins["selected"] = False        # 'selected' all off
    bins.to_csv(tmp_path / "b.all.bed", sep="\t", index=False)
    out = tmp_path / "out"
    m.main(["--bed", f"gch={tmp_path/'b.all.bed'}", "--fdr", "0.05",
            "--output-dir", str(out)])
    s = __import__("json").loads((out / "enrichment_summary.json").read_text())
    assert s["gch"]["n_foreground"] == 5                      # by FDR, not 'selected'
