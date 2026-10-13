"""allelic_imbalance.py: methylation per allele at heterozygous SNPs.

Synthetic bisulfite reads (C->T strand) on chr1 and chrX with phased SNPs:
on chrX the inactive copy's CpGs are methylated and the active copy's are
not.  'GM' cells all silence haplotype 2 (clonal); 'IMR' cells silence
haplotype 1 in half of them and haplotype 2 in the rest (polyclonal).
chr1 has no allelic difference."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pysam = pytest.importorskip("pysam")

_SPEC = importlib.util.spec_from_file_location(
    "allelic_imbalance",
    Path(__file__).resolve().parents[1] / "scripts" / "allelic_imbalance.py")
ai = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ai)

L = 20_000
READ = 100


def _md(read_seq: str, ref_seq: str) -> str:
    """The MD tag of an ungapped alignment."""
    out, run = [], 0
    for q, r in zip(read_seq, ref_seq):
        if q == r:
            run += 1
        else:
            out.append(f"{run}{r}")
            run = 0
    out.append(str(run))
    return "".join(out)


def _bisulfite(ref_seq: str, sites: set, start: int, meth: float, rng):
    """A C->T read of ref_seq from ``start``: HCG site Cs kept with
    probability ``meth``, every other C converted."""
    out = []
    for i, b in enumerate(ref_seq):
        if b == "C":
            b = "C" if (start + i) in sites and rng.random() < meth else "T"
        out.append(b)
    return "".join(out)


def make_data(out: Path, n_cells=6, reads_per_hap=4, seed=0):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    ref = {c: "".join(rng.choice(list("ACGT"), L)) for c in ("chr1", "chrX")}
    # HCG sites: the C of a CG not preceded by G, plus strand
    sites = {c: {i for i in range(1, L - 1) if s[i] == "C"
                 and s[i + 1] == "G" and s[i - 1] != "G"}
             for c, s in ref.items()}
    bed = out / "HCG.bed"
    with open(bed, "w") as fh:
        for c in ("chr1", "chrX"):
            for i in sorted(sites[c]):
                fh.write(f"{c}\t{i}\t{i + 1}\t+\n")
    pysam.tabix_index(str(bed), preset="bed", force=True)
    # SNPs: A>T at A positions away from the sites, phased; on chr1 and X
    snps = []
    for c in ("chr1", "chrX"):
        cand = [i for i in range(500, L - 6000, 150) if ref[c][i] == "A"]
        for k, i in enumerate(cand[:20]):
            snps.append((c, i, "A", "T", "0|1" if k % 2 == 0 else "1|0"))
    vcf = out / "snps.vcf"
    with open(vcf, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 "S1\n")
        for c, i, r, a, gt in snps:
            fh.write(f"{c}\t{i + 1}\t.\t{r}\t{a}\t.\tPASS\t.\tGT\t{gt}\n")
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": c, "LN": L} for c in ("chr1", "chrX")]}
    rows = []
    for grp in ("GM", "IMR"):
        for ci in range(n_cells):
            # the haplotype that is the inactive X in this cell
            xi = 2 if grp == "GM" or ci >= n_cells // 2 else 1
            name = f"{grp}_c{ci}"
            bam_path = out / f"{name}.bam"
            segs = []
            truth = []      # each pair's two read midpoints and haplotype
            n = 0
            for c, i, r, a, gt in snps:
                tid = 0 if c == "chr1" else 1
                for hap in (1, 2):
                    # the allele on this haplotype
                    allele = (r if gt[0] == "0" else a) if hap == 1 else \
                        (r if gt[2] == "0" else a)
                    meth = (0.9 if hap == xi else 0.1) if c == "chrX" \
                        else 0.5
                    for _ in range(reads_per_hap):
                        s1 = i - int(rng.integers(10, READ - 10))
                        s2 = int(rng.integers(i + 5000, L - READ - 1))
                        q1 = list(_bisulfite(ref[c][s1:s1 + READ], sites[c],
                                             s1, meth, rng))
                        q1[i - s1] = allele
                        q1 = "".join(q1)
                        q2 = _bisulfite(ref[c][s2:s2 + READ], sites[c], s2,
                                        meth, rng)
                        qn = f"r{n}"
                        n += 1
                        truth.append((c, s1 + READ // 2, s2 + READ // 2,
                                      hap))
                        for seq, st, mst, first in ((q1, s1, s2, True),
                                                    (q2, s2, s1, False)):
                            seg = pysam.AlignedSegment()
                            seg.query_name = qn
                            seg.query_sequence = seq
                            seg.flag = 0x1 | (0x40 if first else 0x80)
                            seg.reference_id = tid
                            seg.reference_start = st
                            seg.mapping_quality = 60
                            seg.cigarstring = f"{READ}M"
                            seg.next_reference_id = tid
                            seg.next_reference_start = mst
                            seg.template_length = mst - st
                            seg.query_qualities = pysam.qualitystring_to_array(
                                "I" * READ)
                            seg.set_tag("MD", _md(seq, ref[c][st:st + READ]))
                            segs.append(seg)
            segs.sort(key=lambda x: (x.reference_id, x.reference_start))
            with pysam.AlignmentFile(str(bam_path), "wb",
                                     header=header) as fh:
                for seg in segs:
                    fh.write(seg)
            pysam.index(str(bam_path))
            pd.DataFrame(truth, columns=["chrom", "pos1", "pos2", "hap"]
                         ).to_csv(out / f"{name}.truth.tsv", sep="\t",
                                  index=False)
            rows.append((name, str(bam_path), grp))
    pd.DataFrame(rows, columns=["cell_name", "bam_path", "group"]).to_csv(
        out / "samples.tsv", sep="\t", index=False)
    return out


def test_allele_calls_know_the_conversion_strand():
    # C->T read: T fits both C (unmethylated) and T
    assert ai.call_allele("T", "C", "T", "CT") == -1
    assert ai.call_allele("C", "C", "T", "CT") == 0
    # G->A read: A fits both G and A
    assert ai.call_allele("A", "A", "G", "GA") == -1
    assert ai.call_allele("G", "A", "G", "GA") == 1
    assert ai.call_allele("T", "A", "T", "CT") == 1
    assert ai.call_allele("T", "A", "T", "GA") == 1
    assert ai.call_allele("A", "A", "T", "unknown") == -1
    assert ai.call_allele("G", "A", "T", "CT") == -1


def test_vcf_keeps_heterozygous_snvs_and_their_phase(tmp_path):
    v = tmp_path / "x.vcf"
    v.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n"
        "chr1\t10\t.\tA\tG\t.\t.\t.\tGT\t0|1\t1|1\n"
        "chr1\t20\t.\tC\tT\t.\t.\t.\tGT:DP\t1|0:9\t0|1:3\n"
        "chr1\t30\t.\tG\tA\t.\t.\t.\tGT\t0/1\t0/0\n"
        "chr1\t40\t.\tG\tGA\t.\t.\t.\tGT\t0|1\t0|1\n"
        "chr1\t50\t.\tG\tA,C\t.\t.\t.\tGT\t0|1\t0|1\n"
        "chr2\t60\t.\tT\tC\t.\t.\t.\tGT\t1/1\t0|1\n")
    t = ai.read_het_snps(v)
    assert list(t["pos"]) == [9, 19, 29]
    assert list(t["ref_hap"]) == [1, 2, 0]
    tb = ai.read_het_snps(v, sample="B", chroms=["chr2"])
    assert list(zip(tb["chrom"], tb["pos"], tb["ref_hap"])) == [
        ("chr2", 59, 1)]


def test_clonal_line_is_imbalanced_on_x_and_a_polyclonal_one_per_cell(
        tmp_path):
    d = make_data(tmp_path / "d")
    (d / "half.bed").write_text("chrX\t0\t10000\n")
    o = tmp_path / "out"
    ai.main(["--samples", str(d / "samples.tsv"),
             "--vcf", f"GM={d / 'snps.vcf'}", "--vcf", f"IMR={d / 'snps.vcf'}",
             "--context-bed", f"HCG:{d / 'HCG.bed.gz'}", "--mates",
             "--region", f"half={d / 'half.bed'}",
             "--min-reads", "3", "--n-threads", "2", "--plot-format", "png",
             "--output-dir", str(o)])
    snp = pd.read_csv(o / "allelic_snps.tsv", sep="\t")
    x = snp[snp["chrom"] == "chrX"].set_index("group")
    a = snp[snp["chrom"] == "chr1"].set_index("group")
    # GM: haplotype 2 is the methylated, inactive X at every SNP
    assert (x.loc["GM", "HCG_hap1_minus_hap2"] < -0.5).all()
    assert (x.loc["GM", "HCG_q"] < 0.05).mean() > 0.8
    # IMR: the cells disagree, so the pool cancels
    assert x.loc["IMR", "HCG_hap1_minus_hap2"].abs().median() < 0.2
    assert (x.loc["IMR", "HCG_q"] < 0.05).mean() < 0.3
    assert a["HCG_alt_minus_ref"].abs().median() < 0.2
    ch = pd.read_csv(o / "allelic_chromosomes.tsv", sep="\t")
    c = ch[ch["context"] == "HCG"].set_index(["group", "chrom"])
    assert c.loc[("GM", "chrX"), "share_imbalanced"] > 0.8
    assert c.loc[("GM", "chr1"), "share_imbalanced"] < 0.2
    cells = pd.read_csv(o / "allelic_cells.tsv", sep="\t")
    cx = cells[cells["chrom"] == "chrX"].set_index("cell")
    assert (cx.filter(like="GM_", axis=0)["HCG_hap1_minus_hap2"]
            < -0.5).all()
    imr = cx.filter(like="IMR_", axis=0)["HCG_hap1_minus_hap2"]
    assert (imr.loc[["IMR_c0", "IMR_c1", "IMR_c2"]] > 0.5).all()
    assert (imr.loc[["IMR_c3", "IMR_c4", "IMR_c5"]] < -0.5).all()
    # the mates' sites count too: more sites than the SNP reads alone
    assert (cx["HCG_sites_hap1"] > 0).all()
    assert (o / "allelic_HCG.png").exists()
    # the region's sites: a subset, with the same call
    assert (cx["HCG_half_sites_hap1"] < cx["HCG_sites_hap1"]).all()
    assert (cx["HCG_half_sites_hap1"] > 0).all()
    assert (cx.filter(like="GM_", axis=0)["HCG_half_hap1_minus_hap2"]
            < -0.5).all()
    assert (o / "allelic_HCG_half.png").exists()


def test_read_pairs_get_their_haplotype_keyed_as_the_pipeline_keys_them(
        tmp_path):
    """--haplotype-dir: every read pair with a phased SNP gets its
    haplotype, keyed on its reads' midpoints exactly as the pipeline's
    parser keys the pair -- so a posterior row finds it."""
    d = make_data(tmp_path / "d", n_cells=2, reads_per_hap=2)
    o = tmp_path / "out"
    ai.main(["--samples", str(d / "samples.tsv"),
             "--vcf", f"GM={d / 'snps.vcf'}",
             "--context-bed", f"HCG:{d / 'HCG.bed.gz'}",
             "--haplotype-dir", str(o / "haps"), "--min-reads", "1",
             "--n-threads", "1", "--plot-format", "png",
             "--output-dir", str(o)])
    sites = ai._ref.load_context_from_bed({"HCG": str(d / "HCG.bed.gz")},
                                          chroms=["chr1", "chrX"],
                                          n_threads=1)
    snps = ai.read_het_snps(d / "snps.vcf")
    n_x = {}
    both = {"chrX": [0, 0], "autosomes": [0, 0]}
    for cell in ("GM_c0", "GM_c1"):
        h = pd.read_parquet(o / "haps" / f"{cell}.haplotypes.parquet")
        truth = pd.read_csv(d / f"{cell}.truth.tsv", sep="\t")
        # a mate over another SNP shows its REF base: a pair whose mate
        # thereby reads as the other haplotype is no call
        over = [snps[(snps["chrom"] == c) & (snps["pos"] >= p2 - READ // 2)
                     & (snps["pos"] < p2 + READ // 2)]
                for c, p2 in zip(truth["chrom"], truth["pos2"])]
        mixed = [bool((o_["ref_hap"] != hp).any())
                 for o_, hp in zip(over, truth["hap"])]
        for c, o_, mx in zip(truth["chrom"], over, mixed):
            k = "chrX" if c == "chrX" else "autosomes"
            both[k][0] += len(o_) > 0
            both[k][1] += mx
        want = truth[~np.array(mixed)]
        m = h.merge(want, on=["chrom", "pos1", "pos2"],
                    suffixes=("", "_true"))
        assert len(m) == len(h) == len(want) > 0.9 * len(truth)
        assert (m["hap"] == m["hap_true"]).all()
        assert set(h["chrom"]) == {"chr1", "chrX"}
        rp = ai._bp.parse_cell_bam(str(d / f"{cell}.bam"), sites["HCG"])
        keys = set(zip(rp["chrom"], rp["pos1"], rp["pos2"]))
        assert set(zip(h["chrom"], h["pos1"], h["pos2"])) <= keys
        n_x[cell] = want[want["chrom"] == "chrX"].groupby("hap").size()
    pt = pd.read_csv(o / "haplotype_pairs.tsv", sep="\t")
    x = pt[pt["chrom"] == "chrX"].set_index("cell")
    for cell, n in n_x.items():
        assert x.loc[cell, "pairs_hap1"] == n[1]
        assert x.loc[cell, "pairs_hap2"] == n[2]
    # the phase check: pairs whose mate also calls a SNP, 5-15 kb apart
    ph = pd.read_csv(o / "phase_check.tsv", sep="\t").set_index(
        "chromosomes")
    for k, (nb, nd) in both.items():
        assert ph.loc[k, "distance"] == "1-100kb"
        assert ph.loc[k, "both_called"] == nb > 0
        assert ph.loc[k, "disagree"] == nd


def test_snp_sums_merge_like_a_dict():
    """Cells' sums merged a few rows at a time: the totals and the cells of
    every key, as a plain dict would have them."""
    rng = np.random.default_rng(3)
    s = ai.SnpSums(4, flush_rows=7)
    ref: dict = {}
    for _ in range(12):
        k = np.unique(rng.integers(0, 40, 9)).astype(np.int64)
        v = rng.random((k.size, 4)).astype(np.float32)
        s.add(k, v)
        for k_, v_ in zip(k, v):
            t = ref.setdefault(int(k_), np.zeros(5))
            t[:4] += v_
            t[4] += 1
    s.flush()
    assert list(s.keys) == sorted(ref)
    np.testing.assert_allclose(s.vals, np.array([ref[k] for k in s.keys]),
                               rtol=1e-6)


def test_cells_in_batches_give_the_same_tables(tmp_path, monkeypatch):
    """Without joblib's generators the cells go in batches: the same
    tables."""
    d = make_data(tmp_path / "d", n_cells=4, reads_per_hap=2)
    outs = []
    for modes in (ai.IMAP_MODES, ()):
        monkeypatch.setattr(ai, "IMAP_MODES", modes)
        o = tmp_path / f"out{len(modes)}"
        ai.main(["--samples", str(d / "samples.tsv"),
                 "--vcf", f"GM={d / 'snps.vcf'}",
                 "--vcf", f"IMR={d / 'snps.vcf'}",
                 "--context-bed", f"HCG:{d / 'HCG.bed.gz'}",
                 "--min-reads", "1", "--n-threads", "2",
                 "--plot-format", "png", "--output-dir", str(o)])
        outs.append(o)
    for name in ("allelic_snps.tsv", "allelic_cells.tsv"):
        a, b = (pd.read_csv(o / name, sep="\t") for o in outs)
        cols = list(a.columns[:3])
        pd.testing.assert_frame_equal(
            a.sort_values(cols).reset_index(drop=True),
            b.sort_values(cols).reset_index(drop=True))


def _cpg_snp_data(out: Path, n_cells=4, reads=60, seed=1):
    """chr1 with a CpG at 1000-1001 whose G is a SNP (G>A): on the A allele
    the C is a CpA C, never methylated; every other HCG site is methylated
    alike on both alleles (0.8)."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    L, p = 6000, 1000
    ref = list(rng.choice(list("ACGT"), L))
    ref[p - 1:p + 2] = list("ACG")
    ref = "".join(ref)
    sites = {i for i in range(1, L - 1) if ref[i] == "C"
             and ref[i + 1] == "G" and ref[i - 1] != "G"}
    assert p in sites
    with open(out / "HCG.bed", "w") as fh:
        for i in sorted(sites):
            fh.write(f"chr1\t{i}\t{i + 1}\t+\n")
    pysam.tabix_index(str(out / "HCG.bed"), preset="bed", force=True)
    (out / "snps.vcf").write_text(
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\t"
        f"INFO\tFORMAT\tS1\nchr1\t{p + 2}\t.\tG\tA\t.\tPASS\t.\tGT\t0|1\n")
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": "chr1", "LN": L}]}
    rows = []
    for ci in range(n_cells):
        segs, n = [], 0
        for allele in "GA" * reads:
            s1 = p + 1 - int(rng.integers(10, READ - 10))
            s2 = int(rng.integers(3000, L - READ - 1))
            q1 = list(_bisulfite(ref[s1:s1 + READ], sites, s1, 0.8, rng))
            q1[p + 1 - s1] = allele
            if allele == "A":
                q1[p - s1] = "T"            # the CpA C: unmethylated
            q1 = "".join(q1)
            q2 = _bisulfite(ref[s2:s2 + READ], sites, s2, 0.8, rng)
            for seq, st, mst, first in ((q1, s1, s2, True),
                                        (q2, s2, s1, False)):
                seg = pysam.AlignedSegment()
                seg.query_name = f"r{n}"
                seg.query_sequence = seq
                seg.flag = 0x1 | (0x40 if first else 0x80)
                seg.reference_id = 0
                seg.reference_start = st
                seg.mapping_quality = 60
                seg.cigarstring = f"{READ}M"
                seg.next_reference_id = 0
                seg.next_reference_start = mst
                seg.template_length = mst - st
                seg.query_qualities = pysam.qualitystring_to_array("I" * READ)
                seg.set_tag("MD", _md(seq, ref[st:st + READ]))
                segs.append(seg)
            n += 1
        segs.sort(key=lambda x: x.reference_start)
        bam = out / f"c{ci}.bam"
        with pysam.AlignmentFile(str(bam), "wb", header=header) as fh:
            for seg in segs:
                fh.write(seg)
        pysam.index(str(bam))
        rows.append((f"c{ci}", str(bam), "L"))
    pd.DataFrame(rows, columns=["cell_name", "bam_path", "group"]).to_csv(
        out / "samples.tsv", sep="\t", index=False)
    return out


@pytest.mark.parametrize("flank, artefact", [(0, True), (1, False)])
def test_a_snp_beside_a_site_is_not_an_imbalance(tmp_path, monkeypatch,
                                                flank, artefact):
    """A SNP that turns a CpG into CpA leaves an unmethylated C on its ALT
    allele: counted, that site alone makes the ALT reads look the less
    methylated; left out (within 1 bp of the SNP), the alleles agree."""
    d = _cpg_snp_data(tmp_path / "d")
    monkeypatch.setattr(ai, "SNP_FLANK", flank)
    o = tmp_path / "o"
    ai.main(["--samples", str(d / "samples.tsv"),
             "--vcf", f"L={d / 'snps.vcf'}",
             "--context-bed", f"HCG:{d / 'HCG.bed.gz'}",
             "--n-threads", "1", "--plot-format", "png",
             "--output-dir", str(o)])
    r = pd.read_csv(o / "allelic_snps.tsv", sep="\t").iloc[0]
    assert r["reads_ref"] == r["reads_alt"] == 240
    if artefact:
        assert r["HCG_alt_minus_ref"] < -0.08 and r["HCG_p"] < 1e-4
    else:
        assert abs(r["HCG_alt_minus_ref"]) < 0.04 and r["HCG_p"] > 0.01
