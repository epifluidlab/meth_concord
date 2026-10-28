#!/usr/bin/env python3
"""SNP-based allelic imbalance in methylation and accessibility, per cell line.

Every read that covers a heterozygous SNP is given the SNP's allele --
bisulfite-aware: on a C->T read a T may be a T or an unmethylated C, on a
G->A read an A may be an A or an unmethylated G, so a base that fits both
alleles calls neither, and a C/T SNP is read only on G->A reads, a G/A SNP
only on C->T reads: on its own converted strand a C stays C only when
methylated or protected, and its calls would pick the C allele's reads by
their methylation -- and its HCG / GCH sites are counted as the pipeline
counts them (nome_hic_concordance.bam_parser: the conversion strand from the
read's mismatches, the sites of that strand only) -- but not the sites
within SNP_FLANK bp of any SNP: at a site a C/T SNP reads as methylation,
and beside one a SNP can take its context away -- a CpG made CpA on one
allele leaves a C no methyltransferase marks, so that allele's reads look
unmethylated there, an imbalance of the sequence, not of the chromatin.

Per cell line (the samples' group) and SNP, pooled over the line's cells:
each allele's methylation (HCG) and accessibility (GCH), alt less ref, and a
Welch t-test over the reads' own levels (allelic_snps.tsv).  Per line and
chromosome: the share of tested SNPs with imbalance and the median |alt -
ref| (allelic_chromosomes.tsv) -- chrX against the autosomes.  A line whose
cells all silence the same X (clonal: GM12878) is imbalanced on X pooled over
cells; a line whose cells silence either X (polyclonal: IMR-90) cancels in
the pool, though every cell has an inactive X.

With phased SNPs (0|1, 1|0) each read also gets a haplotype, and with
--mates its Hi-C mate too -- a cis contact keeps to one copy of its
chromosome, so the mate's sites are on the read's haplotype without a SNP of
their own.  Per cell and chromosome each haplotype's level
(allelic_cells.tsv): on X the inactive copy is the more methylated at CpG
islands, the same haplotype in every cell of a clonal line, either one in a
polyclonal line.  Figures: allelic_<context>.pdf.

--region NAME=BED adds each context again with only its sites inside the
BED, as <context>_<NAME>: the inactive X is the more methylated at CpG
islands but the less in gene bodies, so over all sites the two partly
cancel, and HCG_island is the sharper call of which copy is silenced.

Per line, context and set of chromosomes (allelic_summary.tsv): hap1 less
hap2 with the sites pooled over the cells and a 95% interval from
resampling them -- on the autosomes, chrX, and chrX's parts: the escape
genes and PAR1, where a cell's two copies are alike (the controls), the
inactive X's own elements (XIST, FIRRE, ICCE, DXZ4) and the rest of X --
and the contrasts between them (the rest of X less each control, each
element less the rest).  The inactive X is the haplotype the more
methylated at CpG islands (--xi-context) when the line's cells agree
(clonal); the rows then give Xi less Xa.  allelic_exceptions.tsv: the loci
of X whose SNPs go against the rest of X (q < 0.05), named by --genes;
allelic_summary_<line>: the figure.  --summary-only makes these again from
the tables, no BAM read.

Phase check (phase_check.tsv): where both reads of a cis pair -- or two
SNPs on one read -- carry phased SNP calls, how often their haplotypes
disagree, by the reads' distance, on chrX and the autosomes.  Allele-call
errors and contacts between the two homologs give a floor; switch errors of
the phasing a share that climbs with distance; a phase that means nothing,
one half.

--haplotype-dir DIR writes, per cell, every cis read pair a phased SNP puts
on one haplotype ({cell}.haplotypes.parquet: chrom, pos1, pos2 -- each
read's midpoint, as nome_hic_concordance keys a read pair -- hap, and the
pair's reads with a SNP call), and per cell and chromosome how many
(haplotype_pairs.tsv): annotation_class_concordance.py --chrx --haplotypes
DIR reshuffles each contact's partners from its own copy of X.

The VCF holds the line's genotypes, heterozygous SNVs used.
phased_snps.py makes phased ones on hg38: NA12878 from the 1000 Genomes
high-coverage phased panel (1kgp), IMR-90 from its haplotype-resolved
assemblies, lifted from T2T-CHM13 (syri).

    python scripts/allelic_imbalance.py \\
        --samples gm12878_imr90.samples.tsv \\
        --vcf GM12878=NA12878.het.vcf.gz --vcf IMR90=IMR90.het.vcf.gz \\
        --context-bed GCH:GCH_motif.hg38.bed.gz,HCG:HCG_motif.hg38.bed.gz \\
        --region island=cpgIslands.bed --region promoter=PLS.bed \\
        --chroms chrX,chr7,chr11,chr15 --mates --n-threads 10 \\
        --genes ncbiRefSeqCurated.txt.gz --output-dir output/allelic_imbalance
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance import bam_parser as _bp  # noqa: E402
from nome_hic_concordance import reference as _ref  # noqa: E402
from nome_hic_concordance.utils import chrom_key, manifest_text  # noqa: E402
from nome_hic_concordance.xci import (  # noqa: E402
    parse_elements, parse_escape, parse_par, read_genes, x_part)

logger = logging.getLogger("allelic_imbalance")

#: The bases a read can show for each true base, by its conversion strand:
#: on a C->T read an unmethylated C reads as T, on a G->A read an
#: unmethylated G as A.
_SHOWS = {"CT": {"A": "A", "C": "CT", "G": "G", "T": "T"},
          "GA": {"A": "A", "C": "C", "G": "GA", "T": "T"}}
#: The phase check's distance classes: two SNPs on one read, then the two
#: reads of a pair by the distance between their starts.
PHASE_EDGES = (0, 1_000, 100_000, 1_000_000, 10_000_000)
PHASE_LABELS = ("same read", "<1kb", "1-100kb", "0.1-1Mb", "1-10Mb",
                ">10Mb")
#: Sites this close to a SNP are not counted: the SNP can be the site, its
#: CpG / GpC partner or its H.
SNP_FLANK = 1
#: chrX's parts, in the order allelic_summary.tsv lists them -- the rest of
#: X, and the controls where a cell's two copies are alike -- then the
#: inactive X's elements.
X_PARTS = ("rest", "escape", "PAR1")
#: A line is clonal -- its cells silence the same X -- when this share of
#: them (with --min-cell-sites on each haplotype) agrees with the line's
#: pooled call of the inactive X.
CLONAL_SHARE = 0.75


def call_allele(base: str, ref: str, alt: str, strand: str) -> int:
    """0 for the ref allele, 1 for alt, -1 when the read cannot tell them
    apart under its conversion strand: the base fits both or neither, or
    the two alleles can show the same base on that strand -- a C/T SNP on a
    C->T read, a G/A SNP on a G->A read.  There a C (G) stays itself only
    when methylated or protected, so calling it would pick that allele's
    reads by their methylation: the C allele would gain the converted
    strand's methylated and accessible reads, the T allele none of its."""
    s = _SHOWS.get(strand)
    if s is None:
        return -1
    rs, as_ = s.get(ref, ref), s.get(alt, alt)
    if set(rs) & set(as_):
        return -1
    r, a = base in rs, base in as_
    if r and not a:
        return 0
    if a and not r:
        return 1
    return -1


def read_het_snps(path, sample=None, chroms=None) -> pd.DataFrame:
    """The heterozygous biallelic SNVs of one sample of a VCF (``sample``,
    else the first): chrom, pos (0-based), ref, alt, and ref_hap -- the
    haplotype carrying REF, 1 for 0|1 and 2 for 1|0, 0 when unphased."""
    want = {chrom_key(c) for c in chroms} if chroms else None
    opener = gzip.open if str(path).endswith(".gz") else open
    rows, col = [], None
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                head = line.rstrip("\n").split("\t")
                col = 9 if sample is None else head.index(sample)
                continue
            c = line[:line.find("\t")]
            if want is not None and chrom_key(c) not in want:
                continue
            f = line.rstrip("\n").split("\t")
            ref, alt = f[3].upper(), f[4].upper()
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" \
                    or alt not in "ACGT" or col is None or len(f) <= col:
                continue
            fmt = f[8].split(":")
            if "GT" not in fmt:
                continue
            gt = f[col].split(":")[fmt.index("GT")]
            if gt in ("0|1", "1|0"):
                hap = 1 if gt == "0|1" else 2
            elif gt in ("0/1", "1/0"):
                hap = 0
            else:
                continue
            rows.append((c, int(f[1]) - 1, ref, alt, hap))
    return pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt",
                                       "ref_hap"])


def _base_at(read, pos: int, pairs):
    """The read's base and quality at reference ``pos``, or None; ``pairs``
    is the read's ref -> query map when it has indels."""
    if pairs is None:
        qp = pos - read.reference_start + read.query_alignment_start
    else:
        qp = pairs.get(pos)
    seq = read.query_sequence
    if qp is None or seq is None or qp < 0 or qp >= len(seq):
        return None
    q = read.query_qualities
    return seq[qp], (q[qp] if q is not None else 40)


def cell_alleles(bam_path, snps: dict, sites: dict, contexts, trim_ends: int,
                 mapq_min: int, min_bq: int, mates: bool, pairs_path=None,
                 xparts=None):
    """One cell: per (chrom, SNP, allele) its reads and, per context, the
    sites (k methylated of n) and the reads' own levels (count, sum, sum of
    squares) -- as arrays: keys (:func:`snp_key`, sorted) and their sums
    (float32), small enough to send back for every cell; per (chrom,
    haplotype, 'snp' | 'mate') the sites; with ``mates``, the mates of the
    haplotype-tagged reads too.  ``snps``: {chrom: (pos, ref, alt,
    ref_hap)}, ``sites``: {context: {chrom: (positions, reference
    bases)}}.

    With ``pairs_path``, the cis read pairs on one haplotype are written
    there: a pair whose reads both pass the pipeline's filters (flags,
    ``mapq_min``, the mate on the same chromosome), one of them at least
    with phased SNP calls of one haplotype, none of the other haplotype --
    keyed as nome_hic_concordance.bam_parser keys it, each read's midpoint
    (pos1 the read starting first).  Also returns {(chrom, hap): pairs},
    and the phase check: {(chrom, distance class): [read pairs -- or reads
    with two phased SNPs -- both called, of them disagreeing]}.

    With ``xparts`` -- (PAR1, escape regions, {element: (start, end)}) --
    a chrX read's sites also go to its part of X ('<chrom>|<part>', by the
    read's midpoint; :func:`nome_hic_concordance.xci.x_part`)."""
    import pysam
    pysam.set_verbosity(0)
    nc = len(contexts)
    by_snp: dict = {}
    by_hap: dict = {}
    want = mates or pairs_path is not None
    on_hap: list = []          # the pairs on one haplotype
    phase: dict = {}

    on_x = False

    def add_hap(chrom, hap, kind, kn, read):
        names = (chrom,)
        if on_x:
            mid = (read.reference_start + read.reference_end) // 2
            names = (chrom, f"{chrom}|{x_part([mid], *xparts)[0]}")
        for nm in names:
            v = by_hap.setdefault((nm, hap, kind), np.zeros(2 * nc + 1))
            v[0] += 1
            v[1:] += kn

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for chrom, (sp, sref, salt, shap) in snps.items():
            here = [sites[c].get(chrom) for c in contexts]
            on_x = xparts is not None and chrom_key(chrom) == "X"
            try:
                reads = bam.fetch(chrom)
            except ValueError:
                continue
            tagged: dict = {}
            mixed: set = set()
            span: dict = {}
            for read in reads:
                if read.flag & _bp._SKIP_FLAGS or \
                        read.mapping_quality < mapq_min:
                    continue
                s, e = read.reference_start, read.reference_end
                i0, i1 = np.searchsorted(sp, [s, e])
                if i0 >= i1:
                    continue
                strand = _bp._infer_conversion_strand(read)
                if strand == "unknown":
                    continue
                pairs = (None if read.reference_length
                         == read.query_alignment_length else
                         {r: q for q, r in read.get_aligned_pairs(
                             matches_only=True)})
                calls = []
                for j in range(i0, i1):
                    b = _base_at(read, int(sp[j]), pairs)
                    if b is None or b[1] < min_bq:
                        continue
                    al = call_allele(b[0], sref[j], salt[j], strand)
                    if al >= 0:
                        calls.append((j, al))
                if not calls:
                    continue
                kn = np.zeros(2 * nc)
                for ci, h in enumerate(here):
                    if h is not None:
                        kn[2 * ci:2 * ci + 2] = _bp._count_context_in_read(
                            read, h[0], h[1], trim_ends)
                for j, al in calls:
                    v = by_snp.setdefault((chrom, int(sp[j]), al),
                                          np.zeros(1 + 5 * nc))
                    v[0] += 1
                    for ci in range(nc):
                        k_, n_ = kn[2 * ci], kn[2 * ci + 1]
                        v[1 + 5 * ci:3 + 5 * ci] += (k_, n_)
                        if n_ > 0:
                            f = k_ / n_
                            v[3 + 5 * ci:6 + 5 * ci] += (1, f, f * f)
                haps = {(shap[j] if al == 0 else 3 - shap[j])
                        for j, al in calls if shap[j]}
                if sum(1 for j, _ in calls if shap[j]) > 1:
                    v = phase.setdefault((chrom, PHASE_LABELS[0]), [0, 0])
                    v[0] += 1
                    v[1] += len(haps) > 1
                if len(haps) == 1:
                    hap = haps.pop()
                    add_hap(chrom, hap, "snp", kn, read)
                    if read.next_reference_name == chrom:
                        tagged.setdefault(read.query_name, {})[
                            read.is_read1] = hap
                        span[read.query_name] = abs(
                            read.reference_start - read.next_reference_start)
                elif len(haps) > 1:
                    mixed.add(read.query_name)
            for qn, t_ in tagged.items():
                if len(t_) == 2:
                    k_ = (chrom, PHASE_LABELS[int(np.searchsorted(
                        PHASE_EDGES, span[qn], side="right"))])
                    v = phase.setdefault(k_, [0, 0])
                    v[0] += 1
                    v[1] += len(set(t_.values())) > 1
            if not (want and tagged):
                continue
            ends: dict = {}
            for read in bam.fetch(chrom):
                t_ = tagged.get(read.query_name)
                if t_ is None or read.flag & _bp._SKIP_FLAGS or \
                        read.mapping_quality < mapq_min:
                    continue
                if pairs_path is not None and \
                        read.next_reference_name == chrom:
                    ends.setdefault(read.query_name, []).append(
                        (read.reference_start, read.reference_end))
                # the mate only, and only once: a pair whose two reads both
                # carry a SNP is counted already
                if not mates or len(t_) > 1 or read.is_read1 in t_:
                    continue
                kn = np.zeros(2 * nc)
                for ci, h in enumerate(here):
                    if h is not None:
                        kn[2 * ci:2 * ci + 2] = _bp._count_context_in_read(
                            read, h[0], h[1], trim_ends)
                add_hap(chrom, next(iter(t_.values())), "mate", kn, read)
            for qn, e in ends.items():
                hs = set(tagged[qn].values())
                if len(e) != 2 or len(hs) != 1 or qn in mixed:
                    continue
                (s1, e1), (s2, e2) = sorted(e)
                on_hap.append((chrom, (s1 + e1) // 2, (s2 + e2) // 2,
                               hs.pop(), len(tagged[qn])))
    counts: dict = {"phase": phase}
    if pairs_path is not None:
        t = pd.DataFrame(on_hap, columns=["chrom", "pos1", "pos2", "hap",
                                          "snp_reads"]).astype(
            {"pos1": np.int64, "pos2": np.int64, "hap": np.int8,
             "snp_reads": np.int8})
        t.to_parquet(pairs_path, index=False)
        counts.update(t.groupby(["chrom", "hap"]).size().to_dict())
    ci_of = {c: i for i, c in enumerate(snps)}
    keys = np.fromiter((snp_key(ci_of[c], p_, a_) for c, p_, a_ in by_snp),
                       np.int64, len(by_snp))
    vals = (np.array(list(by_snp.values()), np.float32) if by_snp
            else np.zeros((0, 1 + 5 * nc), np.float32))
    o = np.argsort(keys)
    return keys[o], vals[o], by_hap, counts


def snp_key(chrom_index: int, pos: int, allele: int) -> int:
    """One int64 per (chromosome, position, allele): the chromosome's index
    in the cell's SNP table, the 0-based position, the allele (0 REF)."""
    return (int(chrom_index) << 33) | (int(pos) << 1) | int(allele)


class SnpSums:
    """Per (chromosome, SNP, allele) -- :func:`snp_key` -- the sums over
    cells and, last, the cells: merged a few million rows at a time,
    numpy only, so no cell's results are held longer than its merge."""

    def __init__(self, width: int, flush_rows: int = 2_000_000):
        self.keys = np.zeros(0, np.int64)
        self.vals = np.zeros((0, width + 1))
        self.flush_rows = flush_rows
        self._pend: list = []
        self._n = 0

    def add(self, keys, vals):
        if len(keys):
            self._pend.append((keys, vals))
            self._n += len(keys)
        if self._n >= self.flush_rows:
            self.flush()

    def flush(self):
        if not self._pend:
            return
        k = np.concatenate([self.keys] + [k_ for k_, _ in self._pend])
        v = np.concatenate(
            [self.vals] + [np.column_stack([v_.astype(np.float64),
                                            np.ones(len(v_))])
                           for _, v_ in self._pend])
        self._pend, self._n = [], 0
        o = np.argsort(k, kind="stable")
        k, v = k[o], v[o]
        start = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        self.keys = k[start]
        self.vals = np.add.reduceat(v, start, axis=0)


def _welch(n1, m1, v1, n2, m2, v2):
    """Two-sided Welch t-test p from two groups' size, mean and variance."""
    from scipy.stats import t as t_dist
    with np.errstate(invalid="ignore", divide="ignore"):
        se2 = v1 / n1 + v2 / n2
        t = (m2 - m1) / np.sqrt(se2)
        df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2
                         / (n2 - 1))
        p = 2 * t_dist.sf(np.abs(t), df)
    return np.where((n1 >= 2) & (n2 >= 2) & (se2 > 0), p, np.nan)


def fdr_bh(p) -> np.ndarray:
    """Benjamini-Hochberg q over the finite p-values; NaN stays NaN."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size:
        o = ok[np.argsort(p[ok])]
        m = o.size
        v = p[o] * m / np.arange(1, m + 1)
        q[o] = np.minimum(np.minimum.accumulate(v[::-1])[::-1], 1.0)
    return q


def snp_table(sums: SnpSums, chroms, snps: pd.DataFrame, contexts,
              min_reads: int) -> pd.DataFrame:
    """Per SNP: each allele's reads and cells, and per context its level
    (sites pooled), alt less ref, and the Welch p over the reads' own
    levels; with phase, hap1 less hap2 too.  ``sums``: merged, its keys'
    chromosome indices into ``chroms``."""
    sums.flush()
    if not len(sums.keys):
        return pd.DataFrame()
    w = 1 + 5 * len(contexts)
    site = sums.keys >> 1
    us, inv = np.unique(site, return_inverse=True)
    alt_ = (sums.keys & 1).astype(bool)
    R, A = np.zeros((us.size, w)), np.zeros((us.size, w))
    cr, ca = np.zeros(us.size, np.int64), np.zeros(us.size, np.int64)
    R[inv[~alt_]] = sums.vals[~alt_, :w]
    A[inv[alt_]] = sums.vals[alt_, :w]
    cr[inv[~alt_]] = sums.vals[~alt_, w]
    ca[inv[alt_]] = sums.vals[alt_, w]
    out = pd.DataFrame({"chrom": np.asarray(list(chroms), object)[us >> 32],
                        "pos": us & 0xFFFFFFFF})
    out = out.merge(snps.drop_duplicates(["chrom", "pos"]),
                    on=["chrom", "pos"], how="left")
    out["reads_ref"], out["reads_alt"] = R[:, 0], A[:, 0]
    out["cells_ref"], out["cells_alt"] = cr, ca
    sign = np.where(out["ref_hap"] == 1, -1.0,
                    np.where(out["ref_hap"] == 2, 1.0, np.nan))
    for ci, ctx in enumerate(contexts):
        b = 1 + 5 * ci
        with np.errstate(invalid="ignore", divide="ignore"):
            lr, la = R[:, b] / R[:, b + 1], A[:, b] / A[:, b + 1]
            mr, ma = R[:, b + 3] / R[:, b + 2], A[:, b + 3] / A[:, b + 2]
            vr = (R[:, b + 4] - R[:, b + 2] * mr ** 2) / (R[:, b + 2] - 1)
            va = (A[:, b + 4] - A[:, b + 2] * ma ** 2) / (A[:, b + 2] - 1)
        out[f"{ctx}_ref"], out[f"{ctx}_alt"] = lr, la
        out[f"{ctx}_alt_minus_ref"] = la - lr
        # hap1 less hap2: alt - ref when REF is on haplotype 2
        out[f"{ctx}_hap1_minus_hap2"] = sign * (la - lr)
        out[f"{ctx}_reads_ref"], out[f"{ctx}_reads_alt"] = R[:, b + 2], \
            A[:, b + 2]
        ok = (R[:, b + 2] >= min_reads) & (A[:, b + 2] >= min_reads)
        out[f"{ctx}_p"] = np.where(ok, _welch(R[:, b + 2], mr, vr,
                                              A[:, b + 2], ma, va), np.nan)
        out[f"{ctx}_q"] = fdr_bh(out[f"{ctx}_p"].to_numpy())
    return out


def chromosome_table(snp: pd.DataFrame, contexts, q_cut: float = 0.05):
    """Per chromosome and context: SNPs tested, the share imbalanced (q <
    q_cut), the median |alt - ref|, and with phase the median hap1 - hap2
    -- one sign on X in a clonal line."""
    rows = []
    for c, g in snp.groupby("chrom", sort=False):
        for ctx in contexts:
            t = g[g[f"{ctx}_p"].notna()]
            rows.append({
                "chrom": c, "is_X": chrom_key(c) == "X", "context": ctx,
                "snps_tested": len(t),
                "share_imbalanced": (float((t[f"{ctx}_q"] < q_cut).mean())
                                     if len(t) else np.nan),
                "median_abs_diff": (float(t[f"{ctx}_alt_minus_ref"].abs()
                                          .median()) if len(t) else np.nan),
                "median_hap1_minus_hap2": (
                    float(t[f"{ctx}_hap1_minus_hap2"].median())
                    if t[f"{ctx}_hap1_minus_hap2"].notna().any()
                    else np.nan)})
    return pd.DataFrame(rows)


def cell_table(per_cell: dict, contexts) -> pd.DataFrame:
    """Per cell and chromosome -- part 'all' -- and on X per part (rest,
    escape, PAR1, the inactive X's elements): each haplotype's level per
    context, SNP reads and (with --mates) their mates pooled, and hap1 less
    hap2."""
    rows = []
    for (cell, group), agg in per_cell.items():
        for name in sorted({k[0] for k in agg}):
            chrom, _, part = name.partition("|")
            r = {"cell": cell, "group": group, "chrom": chrom,
                 "part": part or "all"}
            for hap in (1, 2):
                v = sum((agg[k] for k in agg if k[0] == name and
                         k[1] == hap), np.zeros(2 * len(contexts) + 1))
                r[f"reads_hap{hap}"] = int(v[0])
                for ci, ctx in enumerate(contexts):
                    k_, n_ = v[1 + 2 * ci], v[2 + 2 * ci]
                    r[f"{ctx}_sites_hap{hap}"] = int(n_)
                    r[f"{ctx}_hap{hap}"] = k_ / n_ if n_ else np.nan
            for ctx in contexts:
                r[f"{ctx}_hap1_minus_hap2"] = (r[f"{ctx}_hap1"]
                                               - r[f"{ctx}_hap2"])
            rows.append(r)
    return pd.DataFrame(rows)


def plot_allelic(snp: pd.DataFrame, cells: pd.DataFrame, context: str,
                 path, q_cut: float = 0.05, min_sites: int = 20):
    """a: per line, |alt - ref| per tested SNP on chrX and the autosomes
    (the share with q < q_cut in the legend); b: per phased SNP on chrX,
    hap1 - hap2 along the chromosome -- one sign throughout where every
    cell silences the same X; c: per cell, chrX hap1 - hap2 (cells with
    ``min_sites`` sites on each haplotype)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if snp.empty or f"{context}_p" not in snp:
        return
    groups = list(dict.fromkeys(snp["group"]))
    cols = ("#d6604d", "#4393c3", "#1b9e77", "#7570b3")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4),
                             gridspec_kw={"wspace": 0.45})
    ax0, ax1, ax2 = axes
    ticks, labels, k = [], [], 0.0
    for gi, g in enumerate(groups):
        t = snp[(snp["group"] == g) & snp[f"{context}_p"].notna()]
        isx = t["chrom"].map(chrom_key).eq("X")
        for xi, m in enumerate((~isx, isx)):
            v = 100 * t.loc[m, f"{context}_alt_minus_ref"].abs().dropna()
            if len(v):
                ax0.boxplot(v, positions=[k], widths=0.6, showfliers=False,
                            patch_artist=True,
                            boxprops={"facecolor": cols[gi % 4],
                                      "alpha": 0.9 if xi else 0.35})
                sh = float((t.loc[m, f"{context}_q"] < q_cut).mean())
                ticks.append(k)
                labels.append(f"{g}\n{'X' if xi else 'auto'} {sh:.0%}")
            k += 1
        k += 0.6
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(labels, fontsize=4.5)
    ax0.set_ylabel(f"|alt − ref| {context}, pp")
    ax0.set_title(f"a  per SNP (share with q < {q_cut:g})", loc="left",
                  fontsize=6)
    for gi, g in enumerate(groups):
        t = snp[(snp["group"] == g)
                & snp["chrom"].map(chrom_key).eq("X")
                & snp[f"{context}_p"].notna()
                & snp[f"{context}_hap1_minus_hap2"].notna()]
        if len(t):
            sig = t[f"{context}_q"] < q_cut
            ax1.scatter(t["pos"] / 1e6, 100 * t[f"{context}_hap1_minus_hap2"],
                        s=np.where(sig, 5, 2), lw=0, color=cols[gi % 4],
                        alpha=0.8, label=g)
    ax1.axhline(0, color="0.6", lw=0.4)
    ax1.set_xlabel("chrX, Mb")
    ax1.set_ylabel(f"hap1 − hap2 {context}, pp")
    ax1.set_title("b  phased SNPs on X", loc="left", fontsize=6)
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(fontsize=5, frameon=False)
    if cells is not None and not cells.empty:
        if "part" in cells:
            cells = cells[cells["part"].eq("all")]
        x = cells[cells["chrom"].map(chrom_key).eq("X")
                  & (cells[f"{context}_sites_hap1"] >= min_sites)
                  & (cells[f"{context}_sites_hap2"] >= min_sites)]
        for gi, g in enumerate(groups):
            v = 100 * x.loc[x["group"] == g,
                            f"{context}_hap1_minus_hap2"].dropna()
            if len(v):
                ax2.hist(v, bins=30, histtype="step", color=cols[gi % 4],
                         label=f"{g} ({len(v)} cells; "
                               f"{np.mean(v > 0):.0%} > 0)")
        ax2.axvline(0, color="0.6", lw=0.4)
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend(fontsize=5, frameon=False)
    ax2.set_xlabel("chrX hap1 − hap2 per cell, pp")
    ax2.set_ylabel("cells")
    ax2.set_title("c  which X, cell by cell", loc="left", fontsize=6)
    fig.suptitle(f"{context}: allelic imbalance at heterozygous SNPs",
                 fontsize=7, y=1.04)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _set_masks(cells: pd.DataFrame) -> dict:
    """allelic_summary's sets of rows: the autosomes, chrX, and chrX's
    parts (X_PARTS, then the inactive X's elements)."""
    key = cells["chrom"].map(chrom_key)
    part = (cells["part"].astype(str) if "part" in cells
            else pd.Series("all", index=cells.index))
    is_x = key.eq("X")
    out = {"autosomes": ~is_x & ~key.isin(["Y", "M"]) & part.eq("all"),
           "chrX": is_x & part.eq("all")}
    have = set(part[is_x & part.ne("all")])
    for p_ in [p_ for p_ in X_PARTS if p_ in have] + sorted(have
                                                           - set(X_PARTS)):
        out[f"chrX {p_}"] = is_x & part.eq(p_)
    return out


def _contrasts(sets) -> list:
    """The differences between sets allelic_summary tests: the rest of X
    against each control, each element of the inactive X against the rest
    of X."""
    rest = "chrX rest" if "chrX rest" in sets else "chrX"
    out = [(rest, c) for c in ("chrX escape", "chrX PAR1", "autosomes")
           if c in sets and rest in sets]
    out += [(s_, rest) for s_ in sets if s_.startswith("chrX ")
            and s_ not in ("chrX rest", "chrX escape", "chrX PAR1")
            and rest in sets]
    return out


def allelic_summary(cells: pd.DataFrame, contexts, n_boot: int = 1000,
                    min_sites: int = 20, xi_context: str = "HCG_island",
                    seed: int = 0) -> pd.DataFrame:
    """Per line, context and set -- the autosomes, chrX, and chrX's parts:
    the rest, the escape genes and PAR1, where the copies are alike (the
    controls), the inactive X's elements -- hap1 less hap2, the sites
    pooled over the line's cells, its 95% interval from resampling the
    cells (``n_boot``), p (normal, from the resampled SE) and BH q over the
    table; and over the cells with ``min_sites`` on each haplotype, the
    median difference and the shares with hap1 below and above hap2.  Rows
    of kind 'contrast': one set's difference less another's (:func:
    `_contrasts`), on the same resampled cells.

    The inactive X: the haplotype more methylated at ``xi_context`` (CpG
    islands) on the rest of X, when the pooled interval is clear of 0 and
    CLONAL_SHARE of the cells agree -- a clonal line; its rows then give
    Xi less Xa too (xi_minus_xa).  A polyclonal line's cells silence either
    copy: no call (xi_hap 0)."""
    from scipy.stats import norm
    masks = _set_masks(cells)
    rng = np.random.default_rng(seed)
    rows = []
    for g, cg in cells.groupby("group", sort=False):
        ids = pd.Index(pd.unique(cg["cell"]))
        nc_ = len(ids)
        w = rng.multinomial(nc_, np.full(nc_, 1.0 / nc_),
                            size=n_boot).astype(float)
        est: dict = {}
        g_rows = []
        for sname, m in masks.items():
            sub = cg[m.loc[cg.index]]
            if sub.empty:
                continue
            for ctx in contexts:
                v = []
                for h in (1, 2):
                    n_ = sub[f"{ctx}_sites_hap{h}"].astype(float)
                    k_ = sub[f"{ctx}_hap{h}"].fillna(0).astype(float) * n_
                    v += [x.groupby(sub["cell"]).sum().reindex(
                        ids, fill_value=0).to_numpy() for x in (k_, n_)]
                k1, n1, k2, n2 = v
                if not (n1.sum() > 0 and n2.sum() > 0):
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    d = k1.sum() / n1.sum() - k2.sum() / n2.sum()
                    b = w @ k1 / (w @ n1) - w @ k2 / (w @ n2)
                    ok = (n1 >= min_sites) & (n2 >= min_sites)
                    dc = k1[ok] / n1[ok] - k2[ok] / n2[ok]
                est[(sname, ctx)] = (d, b)
                g_rows.append({
                    "group": g, "context": ctx, "set": sname, "kind": "set",
                    "cells": int(((n1 + n2) > 0).sum()),
                    "sites_hap1": int(n1.sum()), "sites_hap2": int(n2.sum()),
                    "hap1": k1.sum() / n1.sum(), "hap2": k2.sum() / n2.sum(),
                    "hap1_minus_hap2": d, "boot": b,
                    "cells_tested": int(ok.sum()),
                    "share_cells_hap1_lower": (float(np.mean(dc < 0))
                                               if ok.any() else np.nan),
                    "share_cells_hap1_higher": (float(np.mean(dc > 0))
                                                if ok.any() else np.nan),
                    "median_cell_diff": (float(np.median(dc)) if ok.any()
                                         else np.nan)})
        for a_, b_ in _contrasts(list(dict.fromkeys(s_ for s_, _ in est))):
            for ctx in contexts:
                if (a_, ctx) in est and (b_, ctx) in est:
                    (da, ba), (db, bb) = est[(a_, ctx)], est[(b_, ctx)]
                    g_rows.append({"group": g, "context": ctx,
                                   "set": f"{a_} − {b_}", "kind": "contrast",
                                   "hap1_minus_hap2": da - db,
                                   "boot": ba - bb})
        for r in g_rows:
            b = r.pop("boot")
            b = b[np.isfinite(b)]
            se = float(np.std(b, ddof=1)) if b.size > 1 else np.nan
            r["ci_lo"], r["ci_hi"] = (np.percentile(b, [2.5, 97.5])
                                      if b.size else (np.nan, np.nan))
            r["se"] = se
            r["p"] = (float(2 * norm.sf(abs(r["hap1_minus_hap2"]) / se))
                      if se > 0 else np.nan)
        # which haplotype is the inactive X
        call = {r["set"]: r for r in g_rows
                if r["kind"] == "set" and r["context"] == xi_context}
        c_ = call.get("chrX rest", call.get("chrX"))
        xi, agree = 0, np.nan
        if c_ is not None and (c_["ci_lo"] > 0 or c_["ci_hi"] < 0):
            h = 1 if c_["hap1_minus_hap2"] > 0 else 2
            agree = c_["share_cells_hap1_higher" if h == 1
                       else "share_cells_hap1_lower"]
            xi = h if agree >= CLONAL_SHARE else 0
        sgn = {1: 1.0, 2: -1.0}.get(xi, np.nan)
        for r in g_rows:
            r["xi_hap"], r["xi_cells_agree"] = xi, agree
            r["xi_minus_xa"] = sgn * r["hap1_minus_hap2"]
            lo, hi = sgn * r["ci_lo"], sgn * r["ci_hi"]
            r["xi_ci_lo"], r["xi_ci_hi"] = min(lo, hi), max(lo, hi)
            if r["kind"] == "set":
                r["share_cells_xi_lower"] = (
                    r["share_cells_hap1_lower"] if xi == 1 else
                    r["share_cells_hap1_higher"] if xi == 2 else np.nan)
        rows += g_rows
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = fdr_bh(out["p"].to_numpy())
    cols = ["group", "context", "set", "kind", "cells", "cells_tested",
            "sites_hap1", "sites_hap2", "hap1", "hap2", "hap1_minus_hap2",
            "ci_lo", "ci_hi", "se", "p", "q", "median_cell_diff",
            "share_cells_hap1_lower", "share_cells_hap1_higher", "xi_hap",
            "xi_cells_agree", "xi_minus_xa", "xi_ci_lo", "xi_ci_hi",
            "share_cells_xi_lower"]
    return out.reindex(columns=cols)


def allelic_exceptions(snp: pd.DataFrame, summary: pd.DataFrame, contexts,
                       xparts=None, genes=None, q_cut: float = 0.05,
                       merge: int = 50_000,
                       min_diff: float = 0.05) -> pd.DataFrame:
    """Per line and context, the SNPs on X -- outside PAR1 and the escape
    genes -- imbalanced (q < q_cut) against the rest of X, where the rest
    of X has a direction: its pooled difference ``min_diff`` or more, its
    interval clear of 0.  Merged into loci (``merge`` bp apart): their span,
    the SNPs against and -- in the span -- with the rest of X, the median
    difference of those against, the least q, the part of X (an element of
    the inactive X?) and, with ``genes``, the genes within 10 kb, the
    nearest first, else the nearest."""
    sm = summary[summary["kind"].eq("set")].set_index(
        ["group", "context", "set"])
    gx = None
    if genes is not None and len(genes):
        gx = genes[genes["chrom"].map(chrom_key).eq("X")].sort_values(
            "start")
    rows = []
    for g, t in snp.groupby("group", sort=False):
        tx = t[t["chrom"].map(chrom_key).eq("X")]
        if tx.empty:
            continue
        pos = tx["pos"].to_numpy(np.int64)
        part = (x_part(pos, *xparts) if xparts is not None
                else np.full(pos.size, "rest", dtype=object))
        for ctx in contexts:
            if f"{ctx}_q" not in tx:
                continue
            key = next((k for k in ((g, ctx, "chrX rest"), (g, ctx, "chrX"))
                        if k in sm.index), None)
            if key is None:
                continue
            r = sm.loc[key]
            if not (r["ci_lo"] > 0 or r["ci_hi"] < 0) or \
                    abs(r["hap1_minus_hap2"]) < min_diff:
                continue
            sgn = np.sign(r["hap1_minus_hap2"])
            d = tx[f"{ctx}_hap1_minus_hap2"].to_numpy(float)
            q = tx[f"{ctx}_q"].to_numpy(float)
            sig = (q < q_cut) & np.isfinite(d) & ~np.isin(part,
                                                         ["PAR1", "escape"])
            against = np.flatnonzero(sig & (np.sign(d) == -sgn))
            if not against.size:
                continue
            o = against[np.argsort(pos[against], kind="stable")]
            for grp_ in np.split(o, np.flatnonzero(np.diff(pos[o]) > merge)
                                 + 1):
                a_, b_ = int(pos[grp_].min()), int(pos[grp_].max()) + 1
                inside = sig & (pos >= a_) & (pos < b_) & (np.sign(d) == sgn)
                parts_ = sorted(set(part[grp_]) - {"rest"}) or ["rest"]
                near = ""
                if gx is not None and len(gx):
                    # the genes there, the nearest first
                    dist = np.maximum(np.maximum(gx["start"] - b_,
                                                 a_ - gx["end"]), 0)
                    o_ = np.argsort(dist.to_numpy(), kind="stable")
                    hit = gx["name"].to_numpy()[o_][
                        dist.to_numpy()[o_] < 10_000]
                    if len(hit):
                        near = ",".join(list(dict.fromkeys(hit))[:6])
                    else:
                        near = (f"{gx['name'].iloc[o_[0]]} "
                                f"({int(dist.iloc[o_[0]]) / 1e3:.0f} kb)")
                xi = r.get("xi_hap", 0)
                med = float(np.median(d[grp_]))
                rows.append({
                    "group": g, "context": ctx, "chrom": tx["chrom"].iloc[0],
                    "start": a_, "end": b_, "snps_against": int(grp_.size),
                    "snps_with": int(inside.sum()),
                    "median_hap1_minus_hap2": med,
                    "median_xi_minus_xa": ({1: med, 2: -med}.get(xi, np.nan)),
                    "rest_of_x_hap1_minus_hap2": float(r["hap1_minus_hap2"]),
                    "min_q": float(np.min(q[grp_])), "part": ",".join(parts_),
                    "genes": near})
    out = pd.DataFrame(rows, columns=[
        "group", "context", "chrom", "start", "end", "snps_against",
        "snps_with", "median_hap1_minus_hap2", "median_xi_minus_xa",
        "rest_of_x_hap1_minus_hap2", "min_q", "part", "genes"])
    return out.sort_values(["group", "context", "snps_against", "min_q"],
                           ascending=[True, True, False, True],
                           ignore_index=True)


#: allelic_summary_<line>: each set's colour, each element's marker.
_SET_STYLE = {"chrX rest": "#b2182b", "chrX": "#b2182b",
              "chrX escape": "#f4a582", "chrX PAR1": "#1b7837",
              "autosomes": "0.55"}
_EL_MARK = ("*", "s", "^", "D", "v", "P", "X", "h")
_EL_COLS = ("#762a83", "#e08214", "#2166ac", "#4d9221", "#c51b7d",
            "#8c510a", "#01665e", "#542788")


def _context_order(contexts) -> list:
    """The contexts grouped by modality, HCG first."""
    first = {m: i for i, m in enumerate(dict.fromkeys(
        c.split("_")[0] for c in contexts))}
    mods = sorted(first, key=lambda m: (m != "HCG", first[m]))
    return [c for m in mods for c in contexts if c.split("_")[0] == m]


def _along(exc: pd.DataFrame, order) -> str:
    """The context whose loci against the rest of X are the most in the
    inactive X's elements -- then the most SNPs against it."""
    if exc is None or not len(exc):
        return next((c for c in order if c.startswith("HCG")), order[0])
    el = ~exc["part"].isin(["rest"])
    t = pd.DataFrame({"in_el": exc["snps_against"].where(el, 0),
                      "n": exc["snps_against"], "context": exc["context"]}
                     ).groupby("context").sum()
    t["share"] = t["in_el"] / t["n"]
    return t.sort_values(["share", "n"], ascending=False).index[0]


def plot_allelic_summary(summary: pd.DataFrame, snp: pd.DataFrame,
                         exc: pd.DataFrame, group: str, contexts, path,
                         xparts=None, along=None, q_cut: float = 0.05):
    """One line: a, per context, the rest of X against its controls -- the
    escape genes, PAR1, the autosomes -- Xi less Xa (hap1 less hap2 when
    the line has no call), pooled over cells with its 95% interval; b, the
    inactive X's elements against the rest of X; c, along X, each tested
    SNP of the ``along`` context (default: :func:`_along`), the loci of
    allelic_exceptions.tsv named."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summary[(summary["group"] == group) & summary["kind"].eq("set")]
    if s.empty:
        return
    xi = int(s["xi_hap"].iloc[0])
    val, lo, hi = (("xi_minus_xa", "xi_ci_lo", "xi_ci_hi") if xi else
                   ("hap1_minus_hap2", "ci_lo", "ci_hi"))
    lab = "Xi − Xa" if xi else "hap1 − hap2"
    sgn = {1: 1.0, 2: -1.0}.get(xi, 1.0)
    order = [c for c in _context_order(contexts) if c in set(s["context"])]
    have = set(s["set"])
    rest = "chrX rest" if "chrX rest" in have else "chrX"
    sets = [x for x in (rest, "chrX escape", "chrX PAR1", "autosomes")
            if x in have]
    els = [x for x in dict.fromkeys(s["set"]) if x.startswith("chrX ")
           and x not in ("chrX rest", "chrX escape", "chrX PAR1")]
    # each element's colour and marker, in panels b and c alike
    names = list(dict.fromkeys(list((xparts or (None, None, {}))[2] or {})
                               + [x[5:] for x in els]))
    el_col = {nm: _EL_COLS[i % 8] for i, nm in enumerate(names)}
    el_mark = {nm: _EL_MARK[i % 8] for i, nm in enumerate(names)}
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.25, 1), hspace=0.45,
                          wspace=0.55)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    y = {c: -i for i, c in enumerate(order)}

    def dots(ax, names, styles):
        k = len(names)
        for j, (nm, (col, mk)) in enumerate(zip(names, styles)):
            off = ((k - 1) / 2 - j) * min(0.18, 0.7 / max(k, 1))
            t = s[s["set"] == nm].set_index("context")
            for c in order:
                if c not in t.index:
                    continue
                r = t.loc[c]
                ax.errorbar(100 * r[val], y[c] + off,
                            xerr=[[100 * (r[val] - r[lo])],
                                  [100 * (r[hi] - r[val])]],
                            fmt=mk, ms=3.2, color=col, mec=col, lw=0.7,
                            capsize=0, label=nm if c == order[0] or not
                            ax.get_legend_handles_labels()[1].count(nm)
                            else None)
        ax.axvline(0, color="0.6", lw=0.5)
        ax.set_yticks([y[c] for c in order])
        ax.set_yticklabels(order, fontsize=5.5)
        ax.set_xlabel(f"{lab}, pp (95% interval over cells)", fontsize=6)
        ax.tick_params(labelsize=5.5)
        h_, l_ = ax.get_legend_handles_labels()
        keep = dict(zip(l_, h_))
        ax.legend(keep.values(), keep.keys(), fontsize=5, frameon=False,
                  loc="best")

    dots(ax_a, sets, [(_SET_STYLE.get(x, "0.3"), "o") for x in sets])
    ax_a.set_title(f"a  {group}: the rest of X and its controls", loc="left",
                   fontsize=6.5)
    if els:
        dots(ax_b, [rest] + els,
             [(_SET_STYLE[rest], "o")] + [(el_col[x[5:]], el_mark[x[5:]])
                                          for x in els])
        ax_b.set_title("b  the inactive X's elements", loc="left",
                       fontsize=6.5)
    else:
        ax_b.axis("off")
    # c: along X
    e = exc[exc["group"] == group] if exc is not None and len(exc) else \
        pd.DataFrame(columns=["context", "snps_against"])
    if along is None or f"{along}_q" not in snp:
        along = _along(e, order)
    t = snp[(snp["group"] == group) & snp["chrom"].map(chrom_key).eq("X")]
    t = t[t[f"{along}_p"].notna() & t[f"{along}_hap1_minus_hap2"].notna()]
    if len(t):
        pos = t["pos"].to_numpy(np.int64)
        part = (x_part(pos, *xparts) if xparts is not None
                else np.full(pos.size, "rest", dtype=object))
        dv = 100 * sgn * t[f"{along}_hap1_minus_hap2"].to_numpy(float)
        sig = t[f"{along}_q"].to_numpy(float) < q_cut
        cols = {"rest": "0.35", "escape": _SET_STYLE["chrX escape"],
                "PAR1": _SET_STYLE["chrX PAR1"], **el_col}
        for p_ in dict.fromkeys(part):
            m = part == p_
            ax_c.scatter(pos[m] / 1e6, dv[m], s=np.where(sig[m], 6, 1.5),
                         lw=0, color=cols.get(p_, "0.35"),
                         alpha=0.85 if p_ != "rest" else 0.5,
                         label=p_ if p_ != "rest" else "rest of X")
        r_ = s[(s["set"] == rest) & (s["context"] == along)]
        if len(r_):
            ax_c.axhline(100 * r_[val].iloc[0], color=_SET_STYLE[rest],
                         lw=0.6, ls="--")
        if xparts is not None and xparts[0]:
            ax_c.axvspan(xparts[0][0] / 1e6, xparts[0][1] / 1e6,
                         color=_SET_STYLE["chrX PAR1"], alpha=0.08, lw=0)
        if xparts is not None:
            for i, (nm, (a_, b_)) in enumerate((xparts[2] or {}).items()):
                ax_c.axvline((a_ + b_) / 2e6, color=cols.get(nm, "0.3"),
                             lw=0.5, ls=":")
                ax_c.text((a_ + b_) / 2e6, 1.0, nm, fontsize=5,
                          color=cols.get(nm, "0.3"), ha="center",
                          va="bottom", transform=ax_c.get_xaxis_transform())
        for _, r in e[e["context"] == along].head(12).iterrows():
            lbl = str(r["genes"]).split(",")[0] or r["part"]
            ax_c.annotate(f"{lbl} ({int(r['snps_against'])})",
                          (r["start"] / 1e6, 100 * sgn
                           * r["median_hap1_minus_hap2"]),
                          fontsize=4.5, xytext=(3, 3),
                          textcoords="offset points")
        ax_c.axhline(0, color="0.6", lw=0.4)
        ax_c.legend(fontsize=5, frameon=False, markerscale=2, ncol=4,
                    loc="lower left")
    ax_c.set_xlabel("chrX, Mb", fontsize=6)
    ax_c.set_ylabel(f"{lab} {along}, pp", fontsize=6)
    ax_c.tick_params(labelsize=5.5)
    ax_c.set_title(f"c  each tested SNP (q < {q_cut:g} larger); dashed: the "
                   f"rest of X pooled; (n): SNPs against it", loc="left",
                   fontsize=6.5)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def summarize(out: Path, snp: pd.DataFrame, cells: pd.DataFrame, contexts,
              xparts, args) -> None:
    """allelic_summary.tsv, allelic_exceptions.tsv, allelic_summary_<line>;
    the calls and the rest of X against its controls in the log."""
    import re
    summary = allelic_summary(cells, contexts, args.n_boot,
                              args.min_cell_sites, args.xi_context, args.seed)
    summary.to_csv(out / "allelic_summary.tsv", sep="\t", index=False)
    if summary.empty:
        return
    genes = read_genes(args.genes) if args.genes else None
    exc = allelic_exceptions(snp, summary, contexts, xparts, genes,
                             min_diff=args.exception_min_diff)
    exc.to_csv(out / "allelic_exceptions.tsv", sep="\t", index=False)
    if args.xi_context not in contexts:
        logger.warning(f"--xi-context {args.xi_context} is not a context "
                       f"here ({', '.join(contexts)}): no call of the "
                       f"inactive X -- e.g. --region island=cpgIslands.bed")
    for g, s in summary.groupby("group", sort=False):
        xi, ag = int(s["xi_hap"].iloc[0]), s["xi_cells_agree"].iloc[0]
        logger.info(
            f"{g}: " + (f"the inactive X is hap{xi} -- the more methylated at "
                        f"{args.xi_context} in {ag:.0%} of cells" if xi else
                        "no line-wide inactive X (no call at "
                        f"{args.xi_context}, or under {CLONAL_SHARE:.0%} of "
                        "cells agree: polyclonal)"))
        v = ("xi_minus_xa", "xi_ci_lo", "xi_ci_hi") if xi else \
            ("hap1_minus_hap2", "ci_lo", "ci_hi")
        t = s[s["kind"].eq("set")].pivot(index="context", columns="set")
        have = list(t[v[0]].columns)
        sets = [x for x in ("chrX rest", "chrX escape", "chrX PAR1",
                            "autosomes", "chrX") if x in have]
        sets += [x for x in have if x not in sets]
        lines = []
        for c in _context_order(contexts):
            if c in t.index:
                lines.append(f"  {c:<16}" + "  ".join(
                    f"{x} {100 * t.loc[c, (v[0], x)]:+.1f} "
                    f"[{100 * t.loc[c, (v[1], x)]:+.1f}, "
                    f"{100 * t.loc[c, (v[2], x)]:+.1f}]"
                    for x in sets if np.isfinite(t.loc[c, (v[0], x)])))
        logger.info(f"{g}: {'Xi − Xa' if xi else 'hap1 − hap2'}, pp:\n"
                    + "\n".join(lines))
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(g))
        plot_allelic_summary(summary, snp, exc, g, contexts,
                             out / f"allelic_summary_{name}."
                                   f"{args.plot_format}",
                             xparts, args.along)
    if len(exc):
        lines = []
        for (g, c), t in exc.groupby(["group", "context"], sort=False):
            top = ", ".join(
                f"{str(r['genes']).split(',')[0] or r['part']}"
                f"{'' if r['part'] == 'rest' else ' [' + r['part'] + ']'} "
                f"({int(r['snps_against'])})"
                for _, r in t.head(5).iterrows())
            lines.append(f"  {g} {c}: {len(t)} loci, "
                         f"{int(t['snps_against'].sum())} SNPs; {top}")
        logger.info("loci of X against the rest of X (allelic_exceptions."
                    "tsv; in [] an element of the inactive X):\n"
                    + "\n".join(lines))


#: joblib's ways of handing back each call as it finishes, tried in turn
#: (1.4, 1.3); without either, the cells go in batches.
IMAP_MODES = ("generator_unordered", "generator")


def _imap(calls, n_jobs: int):
    """Each of ``calls`` (joblib delayed) run on ``n_jobs`` processes,
    handed back as it finishes -- never all held at once."""
    from joblib import Parallel
    for mode in IMAP_MODES:
        try:
            par = Parallel(n_jobs=n_jobs, return_as=mode)
        except (TypeError, ValueError):
            continue
        yield from par(calls)
        return
    calls = list(calls)
    step = 4 * max(int(n_jobs), 1)
    with Parallel(n_jobs=n_jobs) as par:
        for i in range(0, len(calls), step):
            yield from par(calls[i:i + step])


def _cell_task(cell, *args, **kw):
    return cell, cell_alleles(*args, **kw)


def _pairs(values) -> dict:
    out = {}
    for v in values or []:
        for tok in str(v).split(","):
            if tok.strip():
                k, _, p = tok.strip().partition("=")
                out[k.strip()] = p.strip()
    return out


def _summary_only(out: Path, x_parts, args) -> int:
    """--summary-only: the summary from the tables already in ``out``."""
    paths = [out / "allelic_snps.tsv", out / "allelic_cells.tsv"]
    for p_ in paths:
        if not p_.exists():
            sys.exit(f"--summary-only: no {p_}; run without it first")
    cells = pd.read_csv(paths[1], sep="\t", dtype={"cell": str,
                                                   "group": str})
    contexts = [c[:-len("_sites_hap1")] for c in cells.columns
                if c.endswith("_sites_hap1")]
    if "part" not in cells or not cells["part"].ne("all").any():
        logger.warning(f"{paths[1]} has no parts of chrX -- written before "
                       f"they were added, or without them: chrX and the "
                       f"autosomes only.  Run again without --summary-only "
                       f"for PAR1, the escape genes and the elements.")
    head = pd.read_csv(paths[0], sep="\t", nrows=0).columns
    keep = ["group", "chrom", "pos"] + [
        f"{c}_{x}" for c in contexts for x in ("hap1_minus_hap2", "p", "q")
        if f"{c}_{x}" in head]
    snp = pd.read_csv(paths[0], sep="\t", usecols=keep,
                      dtype={"group": str, "chrom": str})
    summarize(out, snp, cells, contexts, x_parts, args)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--samples",
                    help="Cell sheet (TSV) with the cell, its BAM and its "
                         "line.")
    ap.add_argument("--cell-column", default="cell_name")
    ap.add_argument("--bam-column", default="bam_path")
    ap.add_argument("--group-column", default="group")
    ap.add_argument("--vcf", action="append", metavar="GROUP=VCF",
                    help="A line's genotypes (repeatable; heterozygous SNVs "
                         "used, phased or not).")
    ap.add_argument("--vcf-sample", action="append", default=None,
                    metavar="GROUP=NAME",
                    help="The VCF column of the line (default: the first).")
    ap.add_argument("--context-bed",
                    help="CTX:PATH[,CTX:PATH] tabix-indexed site BEDs, as "
                         "nome_hic_concordance takes them.")
    ap.add_argument("--context-bed-offset", type=int, default=0,
                    help="As nome_hic_concordance's (0).")
    ap.add_argument("--region", action="append", default=None,
                    metavar="NAME=BED",
                    help="Also each context with only its sites inside this "
                         "BED (chrom, start, end), as <context>_<NAME> -- "
                         "e.g. island=cpgIslands.bed (repeatable).")
    ap.add_argument("--chroms", default=None,
                    help="Chromosomes to read (comma-separated; default: "
                         "every one in the VCFs).")
    ap.add_argument("--mates", action="store_true",
                    help="Also count the cis Hi-C mates of haplotype-tagged "
                         "reads (phased SNPs only); reads each chromosome "
                         "twice.")
    ap.add_argument("--haplotype-dir", default=None,
                    help="Write each cell's read pairs on one haplotype "
                         "({cell}.haplotypes.parquet; phased SNPs only) and "
                         "their counts (haplotype_pairs.tsv), for "
                         "annotation_class_concordance.py --chrx "
                         "--haplotypes.")
    ap.add_argument("--genome", default="hg38",
                    help="The assembly, for chrX's parts (hg38: PAR1, the "
                         "escape genes, the inactive X's elements; hg19: "
                         "PAR1 and the escape genes; none).")
    ap.add_argument("--chrx-par", default="auto", metavar="auto|none|S-E",
                    help="PAR1 (chrX START-END, 0-based): both copies "
                         "active, a control.  'auto': by --genome.")
    ap.add_argument("--chrx-escape", default="auto",
                    metavar="auto|none|BED",
                    help="Genes that escape X inactivation, outside PAR1 -- "
                         "a control.  'auto': Tukiainen et al. 2017's (XIST "
                         "left out), 10 kb flanks, by --genome; or a BED.")
    ap.add_argument("--xi-elements", default="auto",
                    metavar="auto|none|BED",
                    help="Elements of the inactive X, each a part of its "
                         "own.  'auto' (hg38): XIST, FIRRE, ICCE, DXZ4; or a "
                         "BED with names (chrom, start, end, name).")
    ap.add_argument("--xi-context", default="HCG_island",
                    help="The context that names the inactive X: the "
                         "haplotype the more methylated there on the rest of "
                         "X (HCG_island, from --region island=...).")
    ap.add_argument("--genes", default=None,
                    help="Genes to name the loci of allelic_exceptions.tsv "
                         "(GTF, UCSC genePred such as ncbiRefSeqCurated.txt"
                         ".gz, or BED with names).")
    ap.add_argument("--exception-min-diff", type=float, default=0.05,
                    help="allelic_exceptions.tsv: only contexts whose rest "
                         "of X differs this much (0.05: 5 pp) -- a "
                         "direction a locus can go against.")
    ap.add_argument("--min-cell-sites", type=int, default=20,
                    help="Sites a cell needs on each haplotype to count in "
                         "the per-cell shares of allelic_summary.tsv.")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="Resamplings of the cells for the intervals.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--along", default=None,
                    help="The context of allelic_summary_<line>'s SNPs "
                         "along X (default: the one with the most SNPs "
                         "against the rest of X).")
    ap.add_argument("--summary-only", action="store_true",
                    help="Only the summary, exceptions and their figures, "
                         "from allelic_snps.tsv and allelic_cells.tsv in "
                         "--output-dir; no BAM read.")
    ap.add_argument("--trim-ends", type=int, default=5)
    ap.add_argument("--mapq-min", type=int, default=30)
    ap.add_argument("--min-base-quality", type=int, default=20)
    ap.add_argument("--min-reads", type=int, default=3,
                    help="Reads each allele needs for a SNP to be tested.")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--plot-format", default="pdf")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    genome = None if str(args.genome).lower() == "none" else args.genome
    try:
        x_par = parse_par(args.chrx_par, genome)
    except ValueError as e:
        ap.error(f"--chrx-par {e}")
    x_esc = parse_escape(args.chrx_escape, genome, x_par)
    x_el = parse_elements(args.xi_elements, genome)
    x_parts = ((x_par, x_esc, x_el) if x_par or x_el or (
        x_esc is not None and len(x_esc)) else None)
    logger.info("chrX's parts: " + (", ".join(
        ([f"PAR1 {x_par[0]:,}-{x_par[1]:,}"] if x_par else [])
        + ([f"escape genes {len(x_esc)} regions, "
            f"{(x_esc[:, 1] - x_esc[:, 0]).sum() / 1e6:.2f} Mb"]
           if x_esc is not None and len(x_esc) else [])
        + ([f"the inactive X's {', '.join(x_el)}"] if x_el else []))
        if x_parts else "none (--genome hg38, or --chrx-par / --chrx-escape "
                        "/ --xi-elements)"))
    if args.summary_only:
        return _summary_only(out, x_parts, args)
    if not (args.samples and args.vcf and args.context_bed):
        ap.error("--samples, --vcf and --context-bed are required (but for "
                 "--summary-only)")
    from joblib import delayed

    samples = pd.read_csv(manifest_text(args.samples), sep="\t", dtype=str)
    vcfs, names = _pairs(args.vcf), _pairs(args.vcf_sample)
    chroms = ([c.strip() for c in args.chroms.split(",") if c.strip()]
              if args.chroms else None)
    beds = {}
    for tok in args.context_bed.split(","):
        ctx, _, p = tok.strip().partition(":")
        beds[ctx.strip().upper()] = p.strip()
    contexts = list(beds)
    snps = {}
    for g, v in vcfs.items():
        snps[g] = read_het_snps(v, names.get(g), chroms)
        logger.info(f"{g}: {len(snps[g]):,} heterozygous SNVs "
                    f"({int((snps[g]['ref_hap'] > 0).sum()):,} phased)")
    want = sorted({c for t in snps.values() for c in t["chrom"]})
    sites = _ref.load_context_from_bed(beds, chroms=want,
                                       n_threads=args.n_threads,
                                       c_offset=args.context_bed_offset)
    for nm, path in _pairs(args.region).items():
        reg = pd.read_csv(path, sep="\t", header=None, comment="#",
                          usecols=[0, 1, 2], dtype={0: str})
        for ctx in list(beds):
            sub = {}
            for c, a in sites[ctx].items():
                r = reg[reg[0] == c].sort_values(1)
                if r.empty:
                    continue
                st_, en_ = r[1].to_numpy(np.int64), r[2].to_numpy(np.int64)
                i = np.searchsorted(st_, a["pos"], side="right") - 1
                ok = (i >= 0) & (a["pos"] < en_[np.maximum(i, 0)])
                sub[c] = a[ok]
            sites[f"{ctx}_{nm}"] = sub
            contexts.append(f"{ctx}_{nm}")
            logger.info(f"{ctx}_{nm}: {sum(len(v) for v in sub.values()):,} "
                        f"sites inside {path}")
    snp_rows, cell_rows, pair_rows = [], {}, []
    phase_rows: dict = {}
    hp_dir = Path(args.haplotype_dir) if args.haplotype_dir else None
    if hp_dir is not None:
        hp_dir.mkdir(parents=True, exist_ok=True)
    # each line's SNPs, the alleles as fixed-width strings: shared with the
    # worker processes as they are, not copied for every cell
    pers = {}
    for g, tab in snps.items():
        tab = tab.sort_values(["chrom", "pos"])
        pers[g] = (tab, {c: (x["pos"].to_numpy(np.int64),
                             x["ref"].to_numpy(dtype="<U1"),
                             x["alt"].to_numpy(dtype="<U1"),
                             x["ref_hap"].to_numpy(np.int64))
                         for c, x in tab.groupby("chrom", sort=False)})
    # no site within SNP_FLANK of a SNP -- a C/T SNP at a site reads as
    # methylation, one beside it can change its context -- any line's
    # SNPs, so that one copy of the sites serves all
    drop: dict = {}
    for _, per in pers.values():
        for c, v in per.items():
            drop.setdefault(c, []).append(v[0])
    drop = {c: np.unique(np.concatenate(v)) for c, v in drop.items()}
    st = {}
    n_near = dict.fromkeys(contexts, 0)
    for ctx in contexts:
        st[ctx] = {}
        for c, a in sites[ctx].items():
            if c in drop:
                u, sp_ = drop[c], a["pos"]
                i = np.searchsorted(u, sp_)
                near = np.minimum(np.abs(sp_ - u[np.maximum(i - 1, 0)]),
                                  np.abs(u[np.minimum(i, u.size - 1)] - sp_))
                keep = near > SNP_FLANK
                n_near[ctx] += int((~keep).sum())
                st[ctx][c] = (sp_[keep], a["ref_base"][keep])
    logger.info(f"sites within {SNP_FLANK} bp of a SNP, left out: "
                + ", ".join(f"{k} {v:,}" for k, v in n_near.items()))
    del sites, drop
    for g, (tab, per) in pers.items():
        cells = samples[samples[args.group_column].astype(str) == g]
        if cells.empty or tab.empty:
            logger.warning(f"{g}: no cells or no SNPs; skipped")
            continue
        logger.info(f"{g}: {len(cells):,} cells")
        if hp_dir is not None and not (tab["ref_hap"] > 0).any():
            logger.warning(f"{g}: no phased SNP, so no read pair gets a "
                           f"haplotype")
        sums = SnpSums(1 + 5 * len(contexts))
        calls = (delayed(_cell_task)(
            c, b, per, st, contexts, args.trim_ends, args.mapq_min,
            args.min_base_quality, args.mates,
            None if hp_dir is None else hp_dir / f"{c}.haplotypes.parquet",
            x_parts)
            for c, b in zip(cells[args.cell_column], cells[args.bam_column]))
        for n_done, (cell, (sk, sv, by_hap, npair)) in enumerate(
                _imap(calls, args.n_threads), 1):
            sums.add(sk, sv)
            del sk, sv
            if n_done % 50 == 0 or n_done == len(cells):
                logger.info(f"{g}: {n_done:,} of {len(cells):,} cells")
            cell_rows[(cell, g)] = by_hap
            for (c_, dl), (nb, nd) in npair.pop("phase", {}).items():
                k_ = (g, "chrX" if chrom_key(c_) == "X" else "autosomes",
                      dl)
                v = phase_rows.setdefault(k_, [0, 0])
                v[0] += nb
                v[1] += nd
            for c_ in sorted({k[0] for k in npair}):
                pair_rows.append({"cell": cell, "group": g, "chrom": c_,
                                  "pairs_hap1": npair.get((c_, 1), 0),
                                  "pairs_hap2": npair.get((c_, 2), 0)})
        t = snp_table(sums, per, tab, contexts, args.min_reads)
        del sums
        if not t.empty:
            t.insert(0, "group", g)
            snp_rows.append(t)
            for ctx in contexts:
                x = t[t[f"{ctx}_p"].notna()]
                isx = x["chrom"].map(chrom_key).eq("X")
                imb = x[f"{ctx}_q"] < 0.05
                logger.info(
                    f"{g} {ctx}: {len(x):,} SNPs tested; imbalanced (q < "
                    f"0.05) on X {float(imb[isx].mean()):.1%} of "
                    f"{int(isx.sum()):,}, autosomes "
                    f"{float(imb[~isx].mean()):.1%} of "
                    f"{int((~isx).sum()):,}")
    if not snp_rows:
        sys.exit("no SNP covered by any read")
    snp = pd.concat(snp_rows, ignore_index=True)
    snp.to_csv(out / "allelic_snps.tsv", sep="\t", index=False)
    chrom_tab = pd.concat([chromosome_table(x, contexts).assign(group=g)
                           for g, x in snp.groupby("group", sort=False)],
                          ignore_index=True)
    chrom_tab.to_csv(out / "allelic_chromosomes.tsv", sep="\t", index=False)
    cells = cell_table(cell_rows, contexts)
    if not cells.empty:
        cells.to_csv(out / "allelic_cells.tsv", sep="\t", index=False)
    for ctx in contexts:
        plot_allelic(snp, cells, ctx,
                     out / f"allelic_{ctx}.{args.plot_format}")
    if not cells.empty:
        summarize(out, snp, cells, contexts, x_parts, args)
    if phase_rows:
        ph = pd.DataFrame([(g, c, dl, nb, nd) for (g, c, dl), (nb, nd)
                           in phase_rows.items()],
                          columns=["group", "chromosomes", "distance",
                                   "both_called", "disagree"])
        ph["distance"] = pd.Categorical(ph["distance"], PHASE_LABELS,
                                        ordered=True)
        ph = ph.sort_values(["group", "chromosomes", "distance"])
        ph["share_disagree"] = ph["disagree"] / ph["both_called"]
        ph.to_csv(out / "phase_check.tsv", sep="\t", index=False)
        logger.info("phase check -- calls on one molecule that disagree on "
                    "the haplotype (a floor of allele-call errors; switch "
                    "errors climb with distance; no phase: 0.5):\n"
                    + ph.to_string(index=False,
                                   float_format=lambda v: f"{v:.3f}"))
    if hp_dir is not None:
        pt = pd.DataFrame(pair_rows, columns=["cell", "group", "chrom",
                                              "pairs_hap1", "pairs_hap2"])
        pt.to_csv(out / "haplotype_pairs.tsv", sep="\t", index=False)
        if not pt.empty:
            pt["n"] = pt["pairs_hap1"] + pt["pairs_hap2"]
            pt["on"] = np.where(pt["chrom"].map(chrom_key).eq("X"), "chrX",
                                "autosomes")
            med = pt.groupby(["group", "on", "cell"])["n"].sum().groupby(
                ["group", "on"]).median()
            logger.info(f"read pairs on one haplotype per cell (median), "
                        f"in {hp_dir}:\n" + med.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
