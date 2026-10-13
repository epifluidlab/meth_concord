#!/usr/bin/env python3
"""SNP-based allelic imbalance in methylation and accessibility, per cell line.

Every read that covers a heterozygous SNP is given the SNP's allele --
bisulfite-aware: on a C->T read a T may be a T or an unmethylated C, on a
G->A read an A may be an A or an unmethylated G, so a base that fits both
alleles calls neither -- and its HCG / GCH sites are counted as the pipeline
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
        --chroms chrX,chr7,chr11,chr15 --mates --n-threads 10 \\
        --output-dir output/allelic_imbalance
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


def call_allele(base: str, ref: str, alt: str, strand: str) -> int:
    """0 for the ref allele, 1 for alt, -1 when the base fits both or
    neither under the read's conversion strand."""
    s = _SHOWS.get(strand)
    if s is None:
        return -1
    r, a = base in s.get(ref, ref), base in s.get(alt, alt)
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
                 mapq_min: int, min_bq: int, mates: bool, pairs_path=None):
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
    with two phased SNPs -- both called, of them disagreeing]}."""
    import pysam
    pysam.set_verbosity(0)
    nc = len(contexts)
    by_snp: dict = {}
    by_hap: dict = {}
    want = mates or pairs_path is not None
    on_hap: list = []          # the pairs on one haplotype
    phase: dict = {}

    def add_hap(chrom, hap, kind, kn):
        v = by_hap.setdefault((chrom, hap, kind), np.zeros(2 * nc + 1))
        v[0] += 1
        v[1:] += kn

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for chrom, (sp, sref, salt, shap) in snps.items():
            here = [sites[c].get(chrom) for c in contexts]
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
                    add_hap(chrom, hap, "snp", kn)
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
                add_hap(chrom, next(iter(t_.values())), "mate", kn)
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
    """Per cell and chromosome: each haplotype's level per context, SNP
    reads and (with --mates) their mates pooled, and hap1 less hap2."""
    rows = []
    for (cell, group), agg in per_cell.items():
        for chrom in sorted({k[0] for k in agg}):
            r = {"cell": cell, "group": group, "chrom": chrom}
            for hap in (1, 2):
                v = sum((agg[k] for k in agg if k[0] == chrom and
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--samples", required=True,
                    help="Cell sheet (TSV) with the cell, its BAM and its "
                         "line.")
    ap.add_argument("--cell-column", default="cell_name")
    ap.add_argument("--bam-column", default="bam_path")
    ap.add_argument("--group-column", default="group")
    ap.add_argument("--vcf", action="append", required=True,
                    metavar="GROUP=VCF",
                    help="A line's genotypes (repeatable; heterozygous SNVs "
                         "used, phased or not).")
    ap.add_argument("--vcf-sample", action="append", default=None,
                    metavar="GROUP=NAME",
                    help="The VCF column of the line (default: the first).")
    ap.add_argument("--context-bed", required=True,
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
            None if hp_dir is None else hp_dir / f"{c}.haplotypes.parquet")
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
