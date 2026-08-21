"""Functional-enrichment analysis of the marker-bin BED files.

Covers the interval engine (merge / coverage / gene overlap, all pure numpy),
the statistics (BH-FDR, hypergeometric ORA, Mann-Whitney coverage test incl. the
perfectly-separated edge case), and an end-to-end run that must recover a planted
signal and degrade gracefully when no annotations are given.
"""

from __future__ import annotations

import importlib.util
import io
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


# ------------------------------------------------ annotation readers + fetch --

def test_read_ucsc_bed_handles_leading_bin_column(tmp_path):
    """cpgIslandExt has a leading 'bin' column -> chrom/start/end at 1/2/3."""
    m = _mod()
    p = tmp_path / "cpg.txt"
    p.write_text("611\tchr1\t100\t200\tCpG\n613\tchr1\t500\t700\tCpG\n")
    df = m._read_ucsc_bed(str(p), cols=(1, 2, 3))
    assert df["chrom"].tolist() == ["chr1", "chr1"]
    assert df["start"].tolist() == [100, 500]
    assert df["end"].tolist() == [200, 700]


def test_read_cytoband_groups_by_giestain(tmp_path):
    m = _mod()
    p = tmp_path / "cb.txt"
    p.write_text("chr1\t0\t100\tqA1\tgpos100\nchr1\t100\t200\tqA2\tgneg\n"
                 "chr2\t0\t50\tqA1\tgpos100\n")
    bands = m._read_cytoband(str(p))
    assert set(bands) == {"band:gpos100", "band:gneg"}
    assert len(bands["band:gpos100"]) == 2                 # chr1 + chr2
    assert bands["band:gneg"]["chrom"].tolist() == ["chr1"]


def test_read_genes_streams_gtf_gene_lines(tmp_path):
    m = _mod()
    p = tmp_path / "g.gtf"
    p.write_text(
        '#c\n'
        'chr1\tHAVANA\tgene\t1000\t2000\t.\t+\t.\tgene_name "Sox2"; gene_id "X";\n'
        'chr1\tHAVANA\texon\t1000\t1100\t.\t+\t.\tgene_name "Sox2";\n'      # skip
        'chr2\tHAVANA\tgene\t5000\t6000\t.\t-\t.\tgene_id "ENSMUSG1";\n')   # id only
    g = m._read_genes(str(p))
    assert g["name"].tolist() == ["Sox2", "ENSMUSG1"]      # exon line skipped
    assert g.iloc[0][["chrom", "start", "end", "strand"]].tolist() == \
        ["chr1", 999, 2000, "+"]                            # GTF 1-based -> 0-based


def test_fetch_and_resolve_with_file_url(tmp_path):
    """The downloader works offline via file:// URLs (fetch, cache, reuse)."""
    m = _mod()
    src = tmp_path / "data.bed"
    src.write_text("chr1\t0\t100\n")
    cache = tmp_path / "cache"
    url = src.resolve().as_uri()                            # file:///.../data.bed
    got = Path(m._fetch(url, cache))
    assert got.read_text() == "chr1\t0\t100\n" and got.parent == cache
    assert m._fetch(url, cache) == got                      # second call = cached
    # _resolve: URL -> fetched path; local path -> as-is; bad URL -> None
    assert m._resolve(url, cache, "x") == str(got)
    assert m._resolve(str(src), cache, "x") == str(src)
    assert m._resolve("file:///no/such/file.bed", cache, "x") is None


def test_label_to_group_collapses_via_mnemonic(tmp_path):
    """Full-stack-style '<N>_<mnemonic>' labels collapse to groups via the
    mnemonic (auto-detected column), NOT the leading number."""
    m = _mod()
    ann = pd.DataFrame({
        "state": ["1", "2", "3"],                 # numbering differs from the BED
        "mneumonics": ["mHET9", "mTSS3", "mEnhWk1"],
        "group": ["HET", "TSS", "WkEnh"]})
    labels = ["15_mHET9", "100_mTSS3", "1_mEnhWk1"]
    mp, hit, tot, col = m._label_to_group(labels, ann, "group")
    assert col == "mneumonics" and hit == 3 and tot == 3
    assert mp == {"15_mHET9": "HET", "100_mTSS3": "TSS", "1_mEnhWk1": "WkEnh"}
    # the leading number must NOT be used (BED '1' maps by mnemonic, not state 1)
    assert mp["1_mEnhWk1"] == "WkEnh"              # state col says 1->HET; ignored
    # an unknown label falls through to 'unmapped'
    mp2, *_ = m._label_to_group(["99_mXYZ"], ann, "group")
    assert mp2["99_mXYZ"] == "unmapped"


def test_parse_pairs_label_vs_bare_url():
    m = _mod()
    d = m._parse_pairs(["GO=https://x.org/lib?libraryName=GO_BP", "local.gmt"], "x")
    assert d["GO"] == "https://x.org/lib?libraryName=GO_BP"  # '=' in query kept
    assert d["local"] == "local.gmt"                         # bare -> stem key


def test_genome_registry_presets_are_well_formed():
    m = _mod()
    assert {"mm10", "mm39", "hg38", "hg19"} <= set(m._GENOME)
    mm10 = m._GENOME["mm10"]
    assert "gencode.vM25" in mm10["gene_annotation"]
    assert mm10["chrom_sizes"].endswith("mm10.chrom.sizes")
    assert set(mm10["gene_sets"]) == {"GO_BP", "KEGG"}
    assert "KEGG_2019_Mouse" in mm10["gene_sets"]["KEGG"]
    for g in m._GENOME.values():                             # all http(s) URLs
        assert m._is_url(g["gene_annotation"]) and m._is_url(g["chrom_sizes"])


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


def test_chromhmm_group_collapse_end_to_end(tmp_path):
    """A fine-grained state BED ('<N>_<mnemonic>') + a groups TSV is enriched by
    GROUP -- redundant enhancer states collapse to one 'enhancers' row."""
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    m = _mod()
    BS = 250_000
    rows = [(c, i * BS, (i + 1) * BS) for c in ("chr1", "chr2")
            for i in range(100)]
    bins = _bins(rows)
    sel = np.zeros(len(bins), bool); sel[:30] = True
    bins["celltype_icc"] = 0.5; bins["celltype_pvalue"] = 1e-3
    bins["celltype_fdr"] = 1e-3; bins["batch_icc"] = 0.01; bins["selected"] = sel
    bins.to_csv(tmp_path / "bins.all.bed", sep="\t", index=False)
    # 3 redundant enhancer states over the fg (chr1[0:30]); quiescent elsewhere.
    seg = [("chr1", i * BS, (i + 1) * BS, f"{1 + i % 3}_mEnhA{1 + i % 3}")
           for i in range(30)]
    seg += [("chr1", i * BS, (i + 1) * BS, "9_mQuies1") for i in range(30, 100)]
    seg += [("chr2", i * BS, (i + 1) * BS, "9_mQuies1") for i in range(100)]
    pd.DataFrame(seg).to_csv(tmp_path / "seg.bed", sep="\t", index=False,
                             header=False)
    pd.DataFrame({"state": ["1", "2", "3", "9"],
                  "mneumonics": ["mEnhA1", "mEnhA2", "mEnhA3", "mQuies1"],
                  "group": ["enhancers", "enhancers", "enhancers", "quiescent"]}
                 ).to_csv(tmp_path / "ann.tsv", sep="\t", index=False)
    out = tmp_path / "out"
    m.main(["--bed", f"gch={tmp_path/'bins.all.bed'}",
            "--chromhmm", str(tmp_path / "seg.bed"),
            "--chromhmm-groups", str(tmp_path / "ann.tsv"),
            "--chromhmm-per-state", "--output-dir", str(out)])
    hmm = pd.read_csv(out / "gch_chromhmm_enrichment.tsv", sep="\t")
    assert set(hmm["annotation"]) == {"enhancers", "quiescent"}   # collapsed
    enh = hmm[hmm["annotation"] == "enhancers"].iloc[0]
    assert enh["log2_fold_change"] > 0 and enh["fdr"] < 0.05      # enriched in fg
    assert hmm[hmm["annotation"] == "quiescent"].iloc[0]["log2_fold_change"] < 0
    # --chromhmm-per-state also wrote the un-collapsed table (4 states)
    st = pd.read_csv(out / "gch_chromhmm_state_enrichment.tsv", sep="\t")
    assert len(st) == 4


# ------------------------------------------------ continuous signal tracks --

def test_binned_mean_overlap_weighted():
    m = _mod()
    bins = _bins([("chr1", 0, 1000), ("chr1", 1000, 2000), ("chr2", 0, 1000)])
    sig = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1"],
                        "start": [0, 400, 1000], "end": [400, 1000, 2000],
                        "value": [10.0, 20.0, 5.0]})
    mn = m._binned_mean(bins, sig)
    assert mn[0] == pytest.approx(16.0)          # (400*10 + 600*20)/1000
    assert mn[1] == pytest.approx(5.0)
    assert np.isnan(mn[2])                        # chr2 uncovered -> NaN


def test_signal_bin_means_reads_bedgraph(tmp_path):
    m = _mod()
    p = tmp_path / "s.bedgraph"
    p.write_text("chr1\t0\t250000\t2.0\nchr1\t250000\t500000\t0.5\n")
    bins = _bins([("chr1", 0, 250000), ("chr1", 250000, 500000)])
    assert m._signal_bin_means(bins, str(p)).tolist() == [2.0, 0.5]


def _rt_bedgraph(tmp_path):
    BS = 250_000
    seg = [(c, i * BS, (i + 1) * BS, 2.0 if (c == "chr1" and i < 30) else 0.2)
           for c in ("chr1", "chr2") for i in range(100)]
    pd.DataFrame(seg).to_csv(tmp_path / "rt.bedgraph", sep="\t", index=False,
                             header=False)


def test_signal_enrichment_end_to_end_bedgraph(tmp_path):
    """--signal-track: per-bin mean signal compared fg-vs-bg -> signed diff + p."""
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    m = _mod()
    _planted(tmp_path)                            # 30 fg bins on chr1[0:30]
    _rt_bedgraph(tmp_path)
    out = tmp_path / "out"
    m.main(["--bed", f"gch={tmp_path/'bins.all.bed'}",
            "--signal-track", f"RT={tmp_path/'rt.bedgraph'}",
            "--output-dir", str(out)])
    sig = pd.read_csv(out / "gch_signal_enrichment.tsv", sep="\t")
    row = sig[sig["annotation"] == "RT"].iloc[0]
    assert row["mean_signal_fg"] > row["mean_signal_bg"]   # fg is 'early'
    assert row["difference"] > 1.5 and row["fdr"] < 0.05
    # the publication figure + per-bin values are written for custom plotting
    assert (out / "gch_signal_figure.pdf").exists()
    vals = pd.read_csv(out / "gch_signal_bin_values.tsv.gz", sep="\t")
    assert {"foreground", "RT"} <= set(vals.columns) and len(vals) == 200


def test_signal_enrichment_bigwig(tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    m = _mod()
    _planted(tmp_path)
    BS = 250_000
    bw = pyBigWig.open(str(tmp_path / "rt.bw"), "w")
    bw.addHeader([("chr1", 25_000_000), ("chr2", 25_000_000)])
    for c in ("chr1", "chr2"):
        bw.addEntries([c] * 100, [i * BS for i in range(100)],
                      ends=[(i + 1) * BS for i in range(100)],
                      values=[2.0 if (c == "chr1" and i < 30) else 0.2
                              for i in range(100)])
    bw.close()
    out = tmp_path / "out"
    m.main(["--bed", f"gch={tmp_path/'bins.all.bed'}",
            "--signal-track", f"RT={tmp_path/'rt.bw'}", "--output-dir", str(out)])
    sig = pd.read_csv(out / "gch_signal_enrichment.tsv", sep="\t")
    assert sig[sig["annotation"] == "RT"].iloc[0]["difference"] > 1.5


# ------------------------------------------------- explicit background BEDs --

def test_background_bed_makes_a_foreground_only_file_usable(tmp_path):
    """The subset BEDs celltype_marker_bins writes drop their membership flags,
    so on their own they are the foreground compared against nothing."""
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    m = _mod()
    _planted(tmp_path)
    allb = pd.read_csv(tmp_path / "bins.all.bed", sep="\t")
    sel = allb[allb["selected"]].drop(columns="selected")
    sel.to_csv(tmp_path / "sel.bed", sep="\t", index=False)
    out = tmp_path / "out"
    m.main(["--bed", f"gch={tmp_path/'sel.bed'}",
            "--background-bed", f"gch={tmp_path/'bins.all.bed'}",
            "--chromhmm", str(tmp_path / "hmm.bed"),
            "--output-dir", str(out)])
    s = __import__("json").loads((out / "enrichment_summary.json").read_text())
    # The 30 selected bins are the foreground; the other 170 the background,
    # NOT all 200 (which would put the foreground on both sides).
    assert s["gch"]["n_foreground"] == 30
    assert s["gch"]["n_bins"] == 200
    hmm = pd.read_csv(out / "gch_chromhmm_enrichment.tsv", sep="\t")
    enh = hmm[hmm["annotation"] == "state:Enh"].iloc[0]
    assert enh["log2_fold_change"] > 0 and enh["fdr"] < 0.05


def test_background_containing_the_foreground_is_deduplicated(tmp_path):
    m = _mod()
    BS = 250_000
    fg = _bins([("chr1", i * BS, (i + 1) * BS) for i in range(10)])
    fg.to_csv(tmp_path / "fg.bed", sep="\t", index=False)
    allb = _bins([("chr1", i * BS, (i + 1) * BS) for i in range(40)])
    allb.to_csv(tmp_path / "all.bed", sep="\t", index=False)
    bins, mask, desc = m._split_with_background(
        str(tmp_path / "fg.bed"), str(tmp_path / "all.bed"), "gch")
    assert len(bins) == 40, "each bin appears exactly once"
    assert mask.sum() == 10 and (~mask).sum() == 30
    assert "10 fg / 30 bg" in desc
    # No coordinate is on both sides.
    key = bins[["chrom", "start", "end"]].astype(str).agg("|".join, axis=1)
    assert not (set(key[mask]) & set(key[~mask]))


def test_a_disjoint_background_is_kept_whole(tmp_path):
    m = _mod()
    BS = 250_000
    _bins([("chr1", i * BS, (i + 1) * BS) for i in range(5)]).to_csv(
        tmp_path / "fg.bed", sep="\t", index=False)
    _bins([("chr2", i * BS, (i + 1) * BS) for i in range(12)]).to_csv(
        tmp_path / "bg.bed", sep="\t", index=False)
    bins, mask, _ = m._split_with_background(
        str(tmp_path / "fg.bed"), str(tmp_path / "bg.bed"), "gch")
    assert mask.sum() == 5 and (~mask).sum() == 12


def test_a_plain_three_column_bed_works_as_the_background(tmp_path):
    """So the background can be any region set, not only our own output."""
    m = _mod()
    BS = 250_000
    _bins([("chr1", i * BS, (i + 1) * BS) for i in range(5)]).to_csv(
        tmp_path / "fg.bed", sep="\t", index=False)
    pd.DataFrame([("chr1", i * BS, (i + 1) * BS) for i in range(30)]).to_csv(
        tmp_path / "plain.bed", sep="\t", index=False, header=False)
    bins, mask, _ = m._split_with_background(
        str(tmp_path / "fg.bed"), str(tmp_path / "plain.bed"), "gch")
    assert mask.sum() == 5 and (~mask).sum() == 25


def test_a_background_that_is_only_the_foreground_stops_the_run(tmp_path):
    m = _mod()
    BS = 250_000
    b = _bins([("chr1", i * BS, (i + 1) * BS) for i in range(6)])
    b.to_csv(tmp_path / "fg.bed", sep="\t", index=False)
    b.to_csv(tmp_path / "bg.bed", sep="\t", index=False)
    with pytest.raises(SystemExit, match="no background bins left"):
        m._split_with_background(str(tmp_path / "fg.bed"),
                                 str(tmp_path / "bg.bed"), "gch")


def test_a_foreground_only_bed_without_a_background_says_what_to_do(tmp_path):
    """The failure it replaces was silent: fg = every bin, background empty,
    and every enrichment a comparison of the foreground with itself."""
    m = _mod()
    BS = 250_000
    _bins([("chr1", i * BS, (i + 1) * BS) for i in range(8)]).to_csv(
        tmp_path / "fg.bed", sep="\t", index=False)
    with pytest.raises(SystemExit, match="--background-bed"):
        m.main(["--bed", f"gch={tmp_path/'fg.bed'}",
                "--output-dir", str(tmp_path / "out")])


def test_background_name_must_match_a_bed_name(tmp_path):
    m = _mod()
    _planted(tmp_path)
    with pytest.raises(SystemExit, match="no matching --bed"):
        m.main(["--bed", f"gch={tmp_path/'bins.all.bed'}",
                "--background-bed", f"hcg={tmp_path/'bins.all.bed'}",
                "--output-dir", str(tmp_path / "out")])


# ------------------------------------------------ scale-free signal effect --

def test_auc_is_the_rank_effect_size_of_the_mann_whitney():
    m = _mod()
    assert m._auc([3, 4, 5], [0, 1, 2]) == 1.0        # complete separation
    assert m._auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert m._auc([1, 2, 3], [1, 2, 3]) == 0.5        # ties -> a half each
    assert np.isnan(m._auc([], [1, 2]))


def test_auc_is_invariant_to_the_track_scale_and_the_difference_is_not():
    """The reason the summary figure uses the AUC: two tracks on different
    scales cannot be compared by their difference of means."""
    m = _mod()
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 400), rng.normal(0, 1, 400)
    auc = m._auc(a, b)
    assert m._auc(a * 1000, b * 1000) == pytest.approx(auc)
    assert m._auc(a + 50, b + 50) == pytest.approx(auc)
    # ...whereas the difference of means scales with the units.
    assert (a * 1000).mean() - (b * 1000).mean() > 100 * (a.mean() - b.mean())


def test_block_ci_is_wider_than_a_per_bin_one_when_bins_are_correlated():
    """Signal is organised into domains, so neighbouring bins are not
    independent and a per-bin interval is too narrow."""
    m = _mod()
    rng = np.random.default_rng(1)
    n_blocks, per = 20, 40
    # One value per DOMAIN, repeated across its bins: 800 bins carrying only
    # 20 independent draws.
    domain = rng.normal(0, 1, n_blocks)
    val = np.repeat(domain, per)
    fg = np.zeros(val.size, bool)
    fg[np.repeat(np.arange(n_blocks) % 2 == 0, per)] = True
    blocks = np.repeat([f"chr1:{i}" for i in range(n_blocks)], per)
    lo, hi = m._auc_block_ci(val, fg, blocks, n_boot=200)
    # A bin-level bootstrap on the same data, for comparison.
    rng2 = np.random.default_rng(2)
    per_bin = [m._auc(val[i][fg[i]], val[i][~fg[i]])
               for i in (rng2.integers(0, val.size, val.size)
                         for _ in range(200))]
    assert (hi - lo) > 2 * (np.percentile(per_bin, 97.5)
                            - np.percentile(per_bin, 2.5))


def test_signal_enrichment_reports_the_effect_size_columns(tmp_path):
    pytest.importorskip("scipy")
    m = _mod()
    BS = 250_000
    bins = _bins([("chr1", i * BS, (i + 1) * BS) for i in range(60)])
    fg = np.zeros(60, bool)
    fg[:20] = True
    rng = np.random.default_rng(0)
    noisy = tmp_path / "t.bedGraph"
    with open(noisy, "w") as fh:
        for i in range(60):
            v = (2.0 if i < 20 else 0.0) + rng.normal(0, 0.1)
            fh.write(f"chr1\t{i * BS}\t{(i + 1) * BS}\t{v}\n")
    # 60 x 250 kb spans 15 Mb, so the default 10 Mb block gives two blocks and
    # no interval -- correct, but not what this test is about.
    df, _ = m._signal_enrichment(bins, fg, {"early": str(noisy)},
                                 block_bp=1_000_000, n_boot=60)
    r = df.iloc[0]
    assert {"auc", "auc_lo", "auc_hi", "cohens_d", "sd_signal_fg",
            "sd_signal_bg"} <= set(df.columns)
    assert r["auc"] == pytest.approx(1.0), "the foreground is entirely higher"
    assert r["cohens_d"] > 5 and r["difference"] == pytest.approx(2.0, abs=0.1)
    assert r["auc_lo"] <= r["auc"] <= r["auc_hi"] + 1e-12


def test_auc_survives_a_zero_variance_track_where_cohens_d_cannot(tmp_path):
    """Two constant groups have no pooled SD, so d is undefined -- while the
    rank effect size is still exactly 1.  Another reason the figure leads with
    the AUC."""
    pytest.importorskip("scipy")
    m = _mod()
    BS = 250_000
    bins = _bins([("chr1", i * BS, (i + 1) * BS) for i in range(60)])
    fg = np.zeros(60, bool)
    fg[:20] = True
    flat = tmp_path / "flat.bedGraph"
    with open(flat, "w") as fh:
        for i in range(60):
            fh.write(f"chr1\t{i * BS}\t{(i + 1) * BS}\t{2.0 if i < 20 else 0.0}\n")
    r = m._signal_enrichment(bins, fg, {"early": str(flat)}, n_boot=40)[0].iloc[0]
    assert np.isnan(r["cohens_d"])
    assert r["auc"] == pytest.approx(1.0)


def test_too_few_blocks_gives_no_interval_rather_than_a_fake_one(tmp_path):
    m = _mod()
    val = np.arange(40, dtype=float)
    fg = np.zeros(40, bool)
    fg[:20] = True
    blocks = np.repeat(["chr1:0", "chr1:1"], 20)      # only two blocks
    assert np.isnan(m._auc_block_ci(val, fg, blocks, n_boot=100)).all()


# ------------------------------------------------- the Zhou PMD/HMD reader --

def _pmd_file(path):
    """The published 6-column shape: chrom/start/end/score/class/label."""
    rows = [("chr1", 0, 100000, "0.22", "PMD", "commonPMD"),
            ("chr1", 100000, 200000, "0.32", "PMD", "commonPMD"),
            ("chr1", 200000, 300000, "0.08", "HMD", "commonHMD"),
            ("chr1", 300000, 400000, "NA", "NA", "Neither")]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, header=False)
    return str(path)


def test_pmd_file_splits_into_its_domain_classes(tmp_path):
    """Loading it whole would test PMDs, HMDs and the unclassified remainder as
    one set -- a union nobody means to ask about."""
    m = _mod()
    got = m._read_pmd(_pmd_file(tmp_path / "pmd.bed"))
    assert set(got) == {"PMD", "HMD"}, "'Neither' is the residue, not a class"
    assert len(got["PMD"]) == 2 and len(got["HMD"]) == 1
    assert list(got["PMD"]["start"]) == [0, 100000]
    assert list(got["HMD"]["end"]) == [300000]


def test_pmd_reader_rejects_a_file_of_the_wrong_shape(tmp_path):
    m = _mod()
    p = tmp_path / "plain.bed"
    pd.DataFrame([("chr1", 0, 100)]).to_csv(p, sep="\t", index=False,
                                            header=False)
    with pytest.raises(ValueError, match="6-column PMD file"):
        m._read_pmd(str(p))


def test_the_human_presets_point_at_the_domain_scale_sets():
    m = _mod()
    for build in ("hg38", "hg19"):
        pre = m._GENOME[build]
        assert pre["annotations"]["PMD"]["reader"] == "pmd"
        assert build in pre["annotations"]["PMD"]["url"]
        assert pre["chromhmm"].endswith(f"{build}_genome_100_segments.bed.gz")
        assert pre["chromhmm_groups"].endswith(".csv")
    # hg19 alone carries a LAD table; UCSC has none for hg38.
    assert "LAD" in m._GENOME["hg19"]["annotations"]
    assert "LAD" not in m._GENOME["hg38"]["annotations"]


def test_a_comma_separated_state_annotation_is_sniffed(tmp_path):
    """The full-stack annotation ships as TSV for mouse and CSV for human, and
    the group column is 'group' in one and 'Group' in the other."""
    m = _mod()
    p = tmp_path / "states.csv"
    p.write_text("state,mneumonics,Group\n1,Quies1,HET\n2,TSS1,TSS\n")
    ann = pd.read_csv(p, sep=None, engine="python", dtype=str)
    assert list(ann.columns) == ["state", "mneumonics", "Group"]
    gcol = [c for c in ann.columns if c.lower() == "group"][0]
    mp, hit, tot, col = m._label_to_group(["1_Quies1", "2_TSS1"], ann, gcol)
    assert hit == tot == 2
    assert mp["1_Quies1"] == "HET" and mp["2_TSS1"] == "TSS"


def test_annotation_manifest_reads_labels_and_skips_comments(tmp_path):
    m = _mod()
    p = tmp_path / "man.tsv"
    p.write_text("# a comment\n\nLAD\t/path/to/lad.bed\nPMD\t/path/to/pmd.bed\n")
    got = m._read_annotation_manifest(str(p))
    assert got == {"LAD": "/path/to/lad.bed", "PMD": "/path/to/pmd.bed"}


def test_a_malformed_manifest_line_names_its_line_number(tmp_path):
    m = _mod()
    p = tmp_path / "man.tsv"
    p.write_text("LAD\t/ok.bed\nbroken-line\n")
    with pytest.raises(SystemExit, match=":2:"):
        m._read_annotation_manifest(str(p))


# --------------------------------------------------------------- cache key ---

def test_the_cache_does_not_confuse_two_assemblies(tmp_path, monkeypatch):
    """UCSC serves cytoBand.txt.gz under every assembly at the same basename.
    Keyed by basename, an mm10 run's cytobands get handed to a later hg38 run
    in the same --cache-dir: mouse chr1-19 match human chr1-19 by name, so the
    wrong coordinates flow through to every band:* row with no error."""
    mbe = _mod()
    ucsc = "https://hgdownload.soe.ucsc.edu/goldenPath"
    served = {f"{ucsc}/mm10/database/cytoBand.txt.gz": b"MOUSE",
              f"{ucsc}/hg38/database/cytoBand.txt.gz": b"HUMAN"}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, **kw: _Resp(served[req.full_url]))

    mouse = mbe._fetch(f"{ucsc}/mm10/database/cytoBand.txt.gz", tmp_path)
    human = mbe._fetch(f"{ucsc}/hg38/database/cytoBand.txt.gz", tmp_path)

    assert mouse != human, "one cache slot for two assemblies"
    assert mouse.read_bytes() == b"MOUSE"
    assert human.read_bytes() == b"HUMAN", "hg38 was served the mm10 file"


def test_a_second_fetch_of_the_same_url_is_cached(tmp_path, monkeypatch):
    mbe = _mod()
    url = "https://example.org/a/thing.bed.gz"
    calls = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import urllib.request

    def fake(req, **kw):
        calls.append(req.full_url)
        return _Resp(b"payload")

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    first = mbe._fetch(url, tmp_path)
    second = mbe._fetch(url, tmp_path)
    assert first == second
    assert len(calls) == 1, "the cache did not hit on an identical url"


# ------------------------------------------------------- manifest merging ---

def _manifest(path, rows):
    Path(path).write_text("".join(f"{k}\t{v}\n" for k, v in rows))
    return str(path)


def test_several_annotation_manifests_are_merged(tmp_path):
    """--annotation-manifest repeats, so LADs and replication timing can come
    from separate files.  As a single-valued option argparse kept only the
    last, and the first source vanished from the report with nothing to say
    it had been asked for."""
    m = _mod()
    a = _manifest(tmp_path / "lad.tsv", [("cLAD", "/a/c.bed"),
                                         ("fLAD", "/a/f.bed")])
    b = _manifest(tmp_path / "rt.tsv", [("RT_variable", "/b/v.bed")])
    got = m._merge_manifests([a, b])
    assert got == {"cLAD": "/a/c.bed", "fLAD": "/a/f.bed",
                   "RT_variable": "/b/v.bed"}


def test_a_label_in_two_manifests_is_an_error(tmp_path):
    """Keeping one silently would drop an annotation from the report, and its
    absence is invisible -- nothing says it was meant to be there."""
    m = _mod()
    a = _manifest(tmp_path / "one.tsv", [("cLAD", "/a/c.bed")])
    b = _manifest(tmp_path / "two.tsv", [("cLAD", "/b/other.bed")])
    with pytest.raises(SystemExit, match="defined in both"):
        m._merge_manifests([a, b])


def test_the_same_label_pointing_at_the_same_file_is_fine(tmp_path):
    """Passing one manifest twice is harmless, not a conflict."""
    m = _mod()
    a = _manifest(tmp_path / "one.tsv", [("cLAD", "/a/c.bed")])
    assert m._merge_manifests([a, a]) == {"cLAD": "/a/c.bed"}


def test_no_manifest_gives_an_empty_mapping():
    m = _mod()
    assert m._merge_manifests([]) == {}


# ------------------------------------------------------------ region test ---

def test_a_grid_aligned_annotation_is_recognised_as_all_or_nothing():
    """A BED written per bin -- the LAD partition, the RT classes -- covers a
    bin fully or not at all, so its coverage has exactly two values."""
    m = _mod()
    assert m._is_binary_coverage(np.array([0.0, 1.0, 1.0, 0.0]))
    assert m._is_binary_coverage(np.array([0.0, 0.0]))
    assert not m._is_binary_coverage(np.array([0.0, 0.4, 1.0]))
    assert not m._is_binary_coverage(np.array([0.5]))


def test_binary_coverage_ignores_nan_and_tolerates_rounding():
    m = _mod()
    assert m._is_binary_coverage(np.array([0.0, 1.0, np.nan]))
    assert m._is_binary_coverage(np.array([1e-9, 1 - 1e-9]))
    assert not m._is_binary_coverage(np.array([np.nan, np.nan]))


def test_fisher_row_builds_the_right_two_by_two():
    m = _mod()
    #        covered  not
    #  fg       3      1
    #  bg       1      5
    cov = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fg = np.array([True] * 4 + [False] * 6)
    r = m._fisher_row("x", cov, fg)
    assert (r["n_fg_in"], r["n_fg_out"]) == (3, 1)
    assert (r["n_bg_in"], r["n_bg_out"]) == (1, 5)
    assert r["odds_ratio"] == pytest.approx((3 * 5) / (1 * 1))
    assert r["mean_frac_fg"] == pytest.approx(0.75)
    assert r["test"] == "fisher"


def test_fisher_is_exact_where_the_rank_test_only_approximates():
    """The reason to switch: with a small foreground the exact p and the
    normal-approximation rank p are different numbers."""
    m = _mod()
    cov = np.array([1.0] * 4 + [0.0] * 16)
    fg = np.array([True] * 4 + [False] * 16)
    fisher = m._fisher_row("x", cov, fg)["pvalue"]
    mwu = m._mwu_row("x", cov, fg)["pvalue"]
    assert fisher == pytest.approx(1 / 4845, rel=1e-6)   # 1 / C(20,4)
    assert abs(fisher - mwu) > 1e-6, "the two tests should not coincide here"


def test_every_region_annotation_gets_the_same_test_by_default():
    """One panel of LADs, PMDs and replication timing must carry one test and
    one effect measure, or no row can be read against its neighbour."""
    m = _mod()
    bins = _bins([("chr1", i * 100, (i + 1) * 100) for i in range(10)])
    fg = np.array([True] * 4 + [False] * 6)
    grid = pd.DataFrame({"chrom": ["chr1"] * 3,
                         "start": [0, 100, 200], "end": [100, 200, 300]})
    graded = pd.DataFrame({"chrom": ["chr1"] * 3,
                           "start": [0, 140, 250], "end": [60, 200, 360]})
    df, _ = m._region_enrichment(bins, fg, {"grid": grid, "graded": graded})
    assert set(df["test"]) == {"fisher"}
    assert df["odds_ratio"].notna().any()


def test_auto_still_splits_by_annotation_when_asked():
    m = _mod()
    bins = _bins([("chr1", i * 100, (i + 1) * 100) for i in range(10)])
    fg = np.array([True] * 4 + [False] * 6)
    grid = pd.DataFrame({"chrom": ["chr1"] * 3,
                         "start": [0, 100, 200], "end": [100, 200, 300]})
    graded = pd.DataFrame({"chrom": ["chr1"] * 3,
                           "start": [0, 140, 250], "end": [60, 200, 360]})
    df, _ = m._region_enrichment(bins, fg, {"grid": grid, "graded": graded},
                                 test="auto")
    got = dict(zip(df["annotation"], df["test"]))
    assert got["grid"] == "fisher" and got["graded"] == "mannwhitney"


def test_a_straddling_bin_goes_to_the_side_holding_most_of_it():
    """PMDs have their own coordinates, so 23% of 250 kb bins are partly
    covered; the majority rule decides them, and the threshold is a flag."""
    m = _mod()
    bins = _bins([("chr1", 0, 100), ("chr1", 100, 200)])
    fg = np.array([True, False])
    ann = pd.DataFrame({"chrom": ["chr1", "chr1"],
                        "start": [0, 100], "end": [70, 130]})   # 70% and 30%
    r = m._fisher_row("x", m._coverage(bins, ann), fg, min_frac=0.5)
    assert (r["n_fg_in"], r["n_bg_in"]) == (1, 0)
    loose = m._fisher_row("x", m._coverage(bins, ann), fg, min_frac=0.2)
    assert (loose["n_fg_in"], loose["n_bg_in"]) == (1, 1)


def test_region_test_can_be_forced():
    m = _mod()
    bins = _bins([("chr1", i * 100, (i + 1) * 100) for i in range(10)])
    fg = np.array([True] * 4 + [False] * 6)
    grid = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [300]})
    df, _ = m._region_enrichment(bins, fg, {"grid": grid}, test="mwu")
    assert df["test"].iloc[0] == "mannwhitney"
    df, _ = m._region_enrichment(bins, fg, {"grid": grid}, test="fisher")
    assert df["test"].iloc[0] == "fisher"


def test_a_peak_scale_annotation_is_flagged_not_silently_empty(capsys):
    """CpG islands are ~1 kb and never fill half a 250 kb bin, so the 2x2 has
    a zero column and the row reads p = 1 -- 'no enrichment' when it means
    'this test cannot see this annotation at this bin size'."""
    m = _mod()
    bins = _bins([("chr1", i * 250000, (i + 1) * 250000) for i in range(8)])
    fg = np.array([True] * 3 + [False] * 5)
    peaks = pd.DataFrame({"chrom": ["chr1"] * 8,
                          "start": [i * 250000 for i in range(8)],
                          "end": [i * 250000 + 1000 for i in range(8)]})
    df, _ = m._region_enrichment(bins, fg, {"CpG_island": peaks})
    assert df["n_fg_in"].iloc[0] == 0 and df["n_bg_in"].iloc[0] == 0
    err = capsys.readouterr().err
    assert "peak-scale" in err and "CpG_island" in err
    assert "--signal-track" in err


def test_an_annotation_absent_from_the_bins_is_not_flagged(capsys):
    """Zero coverage everywhere is a real absence, not a scale mismatch."""
    m = _mod()
    bins = _bins([("chr1", 0, 100), ("chr1", 100, 200)])
    fg = np.array([True, False])
    elsewhere = pd.DataFrame({"chrom": ["chr9"], "start": [0], "end": [100]})
    m._region_enrichment(bins, fg, {"band:stalk": elsewhere})
    assert "peak-scale" not in capsys.readouterr().err


# ------------------------------------------------- publication figures ------

def _ann_table():
    return pd.DataFrame({
        "annotation": ["cLAD", "ciLAD", "band:gpos25", "CpG_island", "PMD"],
        "test": ["fisher"] * 5,
        "n_fg_in": [0, 80, 44, 0, 29], "n_fg_out": [120, 40, 76, 120, 91],
        "n_bg_in": [152, 226, 25, 0, 266],
        "n_bg_out": [528, 454, 655, 680, 414],
        "odds_ratio": [0.0, 4.02, 15.2, np.nan, 0.50],
        "fdr": [1e-12, 4e-11, 1e-21, 1.0, 3e-3]})


def test_the_annotation_figure_drops_bands_and_cpg_islands(tmp_path):
    """A Giemsa band is a stain, not a functional annotation; a CpG island is
    ~1 kb so its Fisher row is empty rather than negative.  Neither belongs
    beside LADs and replication timing in a figure about domains."""
    m = _mod()
    out = tmp_path / "a.pdf"
    assert m._annotation_dotplot(_ann_table(), out, "gch") is not None
    assert out.stat().st_size > 0


def test_a_zero_odds_ratio_still_appears_in_the_figure(tmp_path):
    """An annotation the foreground never enters has odds ratio 0 and log
    -inf -- and it is the STRONGEST depletion in the table, so dropping it as
    unplottable removes the biggest result from the figure."""
    m = _mod()
    d = _ann_table()
    kept = d[~d["annotation"].str.contains(r"^band:|^CpG_island$", regex=True)]
    a = kept["n_fg_in"] + 0.5
    b = kept["n_fg_out"] + 0.5
    c = kept["n_bg_in"] + 0.5
    e = kept["n_bg_out"] + 0.5
    lor = np.log2((a * e) / (b * c))
    assert np.isfinite(lor).all()
    assert lor.iloc[0] < -5, "cLAD should be the most extreme point"
    out = tmp_path / "a.pdf"
    m._annotation_dotplot(d, out, "gch")
    assert out.stat().st_size > 0


def test_the_annotation_figure_keeps_everything_when_asked(tmp_path):
    m = _mod()
    out = tmp_path / "a.pdf"
    assert m._annotation_dotplot(_ann_table(), out, "gch", exclude="") \
        is not None


def test_a_table_without_the_two_by_two_counts_makes_no_figure(tmp_path):
    """Only Fisher rows carry the counts the Haldane correction needs."""
    m = _mod()
    d = pd.DataFrame({"annotation": ["x"], "test": ["mannwhitney"],
                      "fdr": [0.01]})
    assert m._annotation_dotplot(d, tmp_path / "a.pdf", "gch") is None


def test_geneset_dotplot_puts_the_gene_ratio_on_the_axis(tmp_path):
    m = _mod()
    ora = pd.DataFrame({"term": [f"path{i}" for i in range(6)],
                        "n_overlap": [15, 12, 9, 8, 6, 6],
                        "fdr": [0.001, 0.004, 0.01, 0.02, 0.03, 0.04]})
    out = tmp_path / "k.pdf"
    assert m._geneset_dotplot(ora, out, "hcg", "KEGG", 90) is not None
    assert out.stat().st_size > 0


def test_geneset_dotplot_on_an_empty_table_returns_none(tmp_path):
    m = _mod()
    empty = pd.DataFrame(columns=["term", "n_overlap", "fdr"])
    assert m._geneset_dotplot(empty, tmp_path / "k.pdf", "h", "KEGG", 10) \
        is None


def test_chromatin_states_use_mann_whitney_by_default():
    """A full-stack segment has a MEDIAN length of 200 bp -- 0.08% of a 250 kb
    bin -- so a majority-overlap Fisher test returns an empty table for every
    state, and the graded coverage is all there is to test."""
    m = _mod()
    bins = _bins([("chr1", i * 250000, (i + 1) * 250000) for i in range(10)])
    fg = np.array([True] * 4 + [False] * 6)
    # 200 bp segments scattered through each bin: real coverage, never 50%.
    rows = [("chr1", i * 250000 + j * 1000, i * 250000 + j * 1000 + 200)
            for i in range(10) for j in range(50)]
    tiny = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    fisher, _ = m._region_enrichment(bins, fg, {"state": tiny}, test="fisher")
    mwu, _ = m._region_enrichment(bins, fg, {"state": tiny}, test="mwu")
    assert fisher["n_fg_in"].iloc[0] == 0, "majority overlap sees nothing"
    assert np.isfinite(mwu["mean_frac_fg"].iloc[0])
    assert mwu["mean_frac_fg"].iloc[0] > 0
