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
