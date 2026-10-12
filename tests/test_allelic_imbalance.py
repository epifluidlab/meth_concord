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
    o = tmp_path / "out"
    ai.main(["--samples", str(d / "samples.tsv"),
             "--vcf", f"GM={d / 'snps.vcf'}", "--vcf", f"IMR={d / 'snps.vcf'}",
             "--context-bed", f"HCG:{d / 'HCG.bed.gz'}", "--mates",
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
    c = ch.set_index(["group", "chrom"])
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
