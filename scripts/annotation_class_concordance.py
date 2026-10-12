#!/usr/bin/env python3
"""Long-range concordance by genomic annotation class -- per cell, from the
read pairs themselves: no bins, no imputation.

Where are the long-range concordant contacts, and how much does their
concordance vary from cell to cell?  Each contact (a read pair at least
--min-distance long, both ends called) is classed by the annotation at its
two ends -- the LEFT end (the anchor) first, then the right one (the
partner) -- under up to four annotation systems:

  ab        A/B compartment, called per cell type from that type's pooled
            contacts (the PC1 of the O/E correlation matrix, oriented by
            replication timing or gene density -- never by GCH or HCG, which
            would make the classes partly the thing under test)
  rt        replication timing, log2(early/late) of Repli-seq tracks
  chromhmm  a chromHMM segmentation collapsed to six groups (promoter,
            enhancer, transcribed, polycomb, heterochromatin, quiescent)
  pmd       PMD / HMD domains

plus any --annotation NAME=BED (a labelled BED's labels, or a plain BED's
inside / outside).  A contact's class pair is UNORDERED -- which end lies on
the left means nothing, so A-B and B-A are one pair -- unless
--ordered-pairs asks for the split as a check.

THE OBSERVED, per cell c and class pair (K1, K2) in a distance band:

    C = 2 * mean[(a - mu_c,K1) * (p - mu_c,K2)]

over the contacts with an anchor in K1 and a partner in K2, where a and p
are the two ends' posteriors and mu_c,K the cell's mean posterior over ALL
its called ends in class K -- its base level (accessibility or methylation)
there, the marginal.  Centring on the cell's own marginal is what lets the
null come from other cells: a cell whose ends are all a little more open
would otherwise score as concordant against partners from cells that are
not, and that would turn up as cell-to-cell variation in concordance.  C is
twice a covariance, and so scales with the class's own variance -- a mostly
closed class has a small C at any correlation -- so r = C / (2 sd sd) is
reported beside it for comparing classes.

THE NULL reshuffles the same links, at BOTH ends.  Keeping the left end,
the right one is replaced by an end of ANOTHER cell's contact of the same
stratum (cell type, optionally x --null-match-column), in the SAME CLASS as
the end it replaces, at a distance in the contact's own log-distance bin,
inside that end's --position-bin bin; within a draw no donor goes to two
contacts (nome_hic_concordance.link_shuffle.assign_stratum).  Then the
mirror image: the right end kept, the left one replaced.  The null product
is the mean of the two, so neither end is privileged -- reshuffling the
right end only made early-late draw late donors and late-early early ones,
and the two orientations drifted apart.  Donors are centred on their OWN
cell's marginal.  A contact enters only if both ends have a donor.  Two
independent draws: Δ = C - C_null1 is the contrast; C_null2 - C_null1 is
the NEGATIVE CONTROL -- what Δ looks like with no concordance at all, at
the same depth, on the same contacts.  A class pair x distance band with
fewer than --min-band-pairs contacts is flagged 'sparse' and not drawn
(cross-class pairs barely exist at short range: two ends 1-30 kb apart are
in different replication-timing classes only at a domain boundary).

SIGNED BY CO-STATE.  C counts agreement whether it is co-open or
co-closed.  S = ΔC x (2F - 1), with F = co-open / (co-open + co-closed) of
the cell's real contacts in the class pair, carries the same concordance
signed by where its mass sits: + co-open, - co-closed, ~0 with no
concordance.  (A 2x2 table at fixed levels has a single association, so
the sign comes from the levels; S is ΔC and the levels read together.)
Its control is the same formula on reshuffled links alone -- draw 2 in
place of the real partners: (C_null2 - C_null1) x (2 F_null2 - 1).  For F,
a donor's posterior is put on the receiving cell's level in that class.
2F - 1 reads F against an even split; for GCH, closed over most of the
genome, nearly every class then reads co-closed, promoters included.  So
signed_rel reads it against the cell's OWN genome-wide co-open share F_g
(all its contacts annotated in that system):

    w = tanh(½[logit F - logit F_g]) = (F - F_g) / (F + F_g - 2 F F_g)

-- 0 at the cell's background, + more co-open than it, - more co-closed;
2F - 1 is the case F_g = 0.5.  For independent ends logit F ~ logit L1 +
logit L2, so w follows how much more open than the genome the two ends
are, and a cell's overall labelling efficiency cancels.  The control uses
the reshuffled draw's own F_g.  As w <= 1, ΔC x w shrinks classes near
the background; signed_sgn = ΔC x sign(w) keeps ΔC at full size and takes
only the direction from w.  Each of the three -- signed_rel (ΔC x w),
signed_sgn (ΔC x sign(w)) and signed (ΔC x (2F - 1)) -- gets its own
figures, fig_<context>_<system>_signed_{rel,sign,even} and
main_figure_signed_{rel,sign,even} (--signed-figures).

CELL-TO-CELL VARIATION is read against that control.  Per cell, Δ carries
sampling noise; the control carries the same noise and nothing else, so

    excess SD = sqrt(Var_cells(Δ) - Var_cells(control))

is the between-cell spread of real concordance.  Split-half reliability
(each cell's contacts split in two at random) checks it independently, and
an F test across cell types (for Δ and for the control) says how much of it
is cell type.  The marginal gets the same summary -- mean, between-cell
spread beyond its sampling noise, cell-type F -- as the reference.

WHAT EXPLAINS THE WITHIN-TYPE VARIATION.  Per cell, from its own read
pairs: % near (contacts 25 kb - 2 Mb among those >= 25 kb) and % mitotic
(2 - 12 Mb), the cell-cycle plane of Nagano et al. 2017, and the
repli-score (read ends in early- over early + late-replicating windows,
which rises in S phase); depth and the cell's genome-wide level beside
them.  Per class pair, each covariate's within-type Spearman rho, and all
together the share of the REAL within-type variation they explain -- real
meaning beyond the reshuffled control's noise (covariate_association.tsv,
covariates_<context>_<statistic>, cell_cycle_covariates).  Per distance
band too (covariates_<context>_<statistic>_by_band): ΔC changes with
distance, so a cell's all-distance value follows its own mix of contact
distances -- which is what % near and % mitotic measure; an association
that holds within a band is not that mix.

Beside the cell cycle: how well a cell's ends are called (% called, sites
per end, how decisive the posteriors are), % short pairs, batch
(--batch-column), and its compartment strength (excess A-A / B-B pairing
of its contacts >= 1 Mb).  Two checks first decide whether there is
anything about the CELLS to explain.  Locus reassignment
(locus_reassignment.tsv): a cell's ΔC averages the few hundred loci it
happened to sample, so cells differ where loci differ even if cells do
not; reassigning contacts among a type's cells at random, each keeping its
count, leaves noise plus that, and only the rest -- cell_share -- is a
property of cells (split for every statistic).  Cross-class correlation
(cross_class_correlation.tsv, cross_class_<context>): whether a cell's ΔC in
one state predicts it in another, corrected for noise by split-half
reliability -- one per-cell factor or region-specific variation.  And Δr,
ΔC over each cell's own class SDs (covariates_<context>_delta_r): C is a
covariance, so it grows with a cell's variance -- with its level -- even
when its coordination does not; what follows the level in ΔC but not in Δr
is that scale.

Further checks, each a component of the variance split.  Labeling
efficiency: a cell's GCH methylated at sites open in nearly every cell
(--open-sites; default CpG-island TSSs), which reads M.CviPI's reach into
that nucleus.  Doublets: Scrublet's score on the cells' regional profiles,
doublets simulated from halves of two cells of types the profiles tell
apart.  Bisulfite conversion: --context-counts HCY=... (C in no CpG, GpC or
CpA, so nothing methylates it): what reads methylated was not converted.
mCA: GCH counts GCA, and the C of GCA is a CpA, which adult neurons
methylate endogenously -- HCA less HCY methylated (--context-counts HCA=...
HCY=..., the pipeline run with --contexts HCA,HCY --steps 1,2), or GCA less
GCY within GCH.  Division history: HMD less PMD methylation (HCG), as
PMDs lose methylation with every division.  Substructure: within-type
principal components of the cells' regional profiles (subtypes, area or
layer gradients, animals).  And the scale term, the cell's two class SDs.
The bar of covariates_<context>_<statistic> splits the cells' part by group
-- Shapley shares of the covariates' R^2, so groups that overlap split what
they explain together (shap_<group> in covariate_association.tsv).  Where
the variation is about loci rather than cells, locus_hotspots_<context>
maps it: per 1 Mb bin, the concordance of every cell's contacts with an end
there, bins above or below the median bin at an FDR fitted to the control,
and what the concordant bins are -- each class, PMD borders, CpG islands,
TSSs, imprinted genes, the protocadherin cluster, olfactory-receptor and Zfp
genes, --hotspot-annotation BEDs (locus_hotspots.tsv.gz,
locus_hotspot_enrichment.tsv) -- each also with gene density accounted for
(rho_given_tss), as imprinted genes, CpG islands and promoters all sit
where genes are dense.

ALL REGIONS.  Beside the class pairs, every annotated contact of a system
together ("all regions"); signed, each contact carries its class pair's
weight, so the value is the contact-weighted mean of the class pairs'.

GROUPS OF CELLS.  --contrast-group NAME=TYPE[,TYPE...], two or more times
(e.g. tumor1=Tumor1, tumor2=Tumor2, monocyte=Monocyte), every pair compared
(or each against --contrast-reference): per region group and band, each
group's pooled concordance with 95% bootstrap intervals over its cells, the
difference with its interval and BH q, the difference beyond the pair's
all-regions ratio (a group more concordant everywhere -- labeling, depth, a
global state -- differs in every class; which classes differ beyond that),
and the difference adjusted for the technical covariates and batch
(contrast_classes.tsv, contrast_<ctx>_<stat>, contrast_overview_<ctx>_<stat>;
each group's batches in contrast_groups.tsv); and along the genome, each 1 Mb
bin's difference, the bins that differ -- also against the second group's map
rescaled to the first's -- and what they are, with p against genome rotations
(contrast_hotspots.tsv.gz, contrast_hotspot_enrichment.tsv,
contrast_hotspots_<ctx>).

ALLELIC STATES.  A contact's two ends lie on one copy of a chromosome, and
its reshuffled partners come from either copy, so where the two copies differ
-- an inactive X, an imprinted domain -- contacts are concordant beyond the
null without any SNP.  --chrx: each cell's sex from its X and Y coverage
(cell_sex.tsv, or --sex-column), and X-X against autosome-autosome
concordance per cell type, sex and group (chrx_concordance.tsv,
chrx_percell.tsv.gz, chrx_<ctx>); chrX enters nothing else, and in a male the
one X is the reference with no second copy.  --imprinting: concordance
inside imprinted domains (the gene annotation's imprinted genes merged, or
--imprinting-bed) against windows matched in length, chromatin (the
contacts' class pairs) and TSS count, per cell type and group, and each
pair's difference -- neurons against glia where imprinting is neuron-specific
(Ube3a, Grb10) (imprinting_domains.tsv, imprinting_concordance.tsv,
imprinting_contrast.tsv, imprinting_windows.tsv, imprinting_<ctx>).  --pcdh:
the same for the clustered protocadherins -- the whole cluster and its alpha,
beta and gamma parts; in a neuron each allele picks a few of their promoters,
which loop to the HS5-1 enhancer -- with each group's accessibility or
methylation level in each domain beside its concordance (pcdh_*).

VARIABLE AND STABLE LOCI.  Per 1 Mb bin, each cell type's concordance, and
how much it varies between types beyond sampling (a heterogeneity test
against the control): the most variable bins, the variable ones beyond each
type's genome-wide level (a type that is more concordant everywhere does not
count), and the most stable concordant ones -- with what each set is
enriched in (locus_variability.tsv.gz, locus_variability_enrichment.tsv,
locus_variability_<ctx>_<stat>) and as plain BEDs for
marker_bin_enrichment.py (locus_<set>_<ctx>_<stat>.bed, with
locus_tested_* as the background).

CELL TYPE FROM CONCORDANCE.  celltype_auc_<statistic> (celltype_auc.tsv,
celltype_auc_by_type.tsv): per context and both together, how well each
region group's per-cell concordance predicts cell type -- a random forest,
cross-validated, one-vs-rest macro AUC -- beside its control (a statistic
that predicts type from reshuffled links alone carries type through its
construction), the marginal level, and marginal + concordance, against which
the marginal alone says what the concordance adds.  A/B is called per cell
type, so compartment classes are type-specific by construction: read their
rows with that in mind.  In post-
mitotic neurons % near tracks chromatin compaction rather than the cycle,
and only cycling glia should have S-phase repli-scores.

UNITS.  C is a difference of probabilities (excess agreement), so the
figures show C, ΔC and the signed values in percentage points (x100); the
tables keep fractions.  |C| <= 0.5 (at most 2 L (1 - L) at level L), the
weight w lies in [-1, 1], so a signed value lies within ±|ΔC|; Δr is a
correlation, in [-1, 1], and is not rescaled.

Outputs (--output-dir): cell_covariates.tsv, covariate_association.tsv,
locus_reassignment.tsv, cross_class_correlation.tsv, percell.tsv.gz (one row per cell x system x class
pair x band), summary.tsv, marginal_summary.tsv, coverage.tsv,
compartments_ab.tsv + compartments_qc.tsv (with ab), and one figure per
context x system.

    python scripts/annotation_class_concordance.py \\
        --posteriors GCH=output/final_25mb/context_GCH/04_posteriors \\
        --posteriors HCG=output/final_25mb/context_HCG/04_posteriors \\
        --samples mouse_cortex_1577_cells.samples.with_batches.fixed.tsv \\
        --genome mm10 \\
        --rt-early F121_NPC.repliseq.early.bw --rt-late F121_NPC.repliseq.late.bw \\
        --pmd PMD_coordinates_mm10.bed.gz \\
        --n-threads 10 --output-dir output/annotation_class_concordance
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance import link_shuffle as _ls  # noqa: E402
from nome_hic_concordance.compartments import _oe_normalize  # noqa: E402
from nome_hic_concordance.utils import manifest_text  # noqa: E402

logger = logging.getLogger("annotation_class_concordance")

#: The full-stack chromHMM's 16 groups, collapsed so a cell keeps enough
#: contacts per class pair.  'artifacts' is left out.
CHROMHMM_GROUPS = {
    # mouse full-stack (mm10) group names
    "TSS": "promoter", "Prom": "promoter",
    "enhancers": "enhancer", "WkEnh": "enhancer", "TxEnh": "enhancer",
    "OpenC": "enhancer",
    "Tx": "transcribed", "TxEx": "transcribed", "TxWk": "transcribed",
    "ReprPC": "polycomb", "ReprPC and openC": "polycomb",
    "BivProm": "polycomb",
    "HET": "heterochromatin", "ZNF": "heterochromatin",
    "quiescent": "quiescent",
    # human full-stack (hg19 / hg38: Vu & Ernst 2022) group names -- its
    # "quescient" is spelled so in the release; 'others' (gaps, artifacts)
    # stays out
    "promoters": "promoter", "bivalent promoters": "polycomb",
    "weak enhancers": "enhancer", "transcribed and enhancer": "enhancer",
    "acetylations": "enhancer", "DNase": "enhancer",
    "transcription": "transcribed", "weak transcription": "transcribed",
    "exon": "transcribed", "polycomb repressed": "polycomb",
    "quescient": "quiescent",
}
_CHROMHMM_LC = {k.lower(): v for k, v in CHROMHMM_GROUPS.items()}
CHROMHMM_ORDER = ("promoter", "enhancer", "transcribed", "polycomb",
                  "heterochromatin", "quiescent")
SYSTEMS = ("ab", "rt", "chromhmm", "pmd")


def _canon(c) -> str:
    """Chromosome name without a leading 'chr', so chr1 == 1."""
    s = str(c)
    return s[3:] if s[:3].lower() == "chr" else s


#: Genome coordinate = chromosome code x STRIDE + position (link_shuffle's).
STRIDE = _ls.GENOME_STRIDE


def _chrom_code(name) -> int:
    """A stable integer per chromosome name -- the same in every thread and
    every annotation, so genome coordinates agree without a shared table:
    numbered chromosomes as their number, X / Y / M fixed, any other name
    hashed ('chr' ignored)."""
    c = _canon(name).upper()
    if c.isdigit():
        return int(c)
    special = {"X": 1001, "Y": 1002, "M": 1003, "MT": 1003}
    if c in special:
        return special[c]
    return 2000 + zlib.crc32(c.encode()) % 100_000


def _codes_of(chrom) -> np.ndarray:
    """:func:`_chrom_code` of every name, computed once per distinct name;
    integers are taken as codes already."""
    arr = np.asarray(chrom)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64)
    idx, uniq = pd.factorize(arr)
    return np.array([_chrom_code(c) for c in uniq], np.int64)[idx]


def _enrichment():
    """scripts/marker_bin_enrichment.py: its mm10 downloads (cached where
    that script caches them) and its BED / bigWig / chromHMM readers."""
    spec = importlib.util.spec_from_file_location(
        "marker_bin_enrichment",
        Path(__file__).with_name("marker_bin_enrichment.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- intervals --


class Intervals:
    """Non-overlapping intervals, each with a class code, on one genome
    coordinate (chromosome code x STRIDE + position): ``lookup_g`` gives a
    coordinate's code in a single sorted search, whatever the chromosome --
    -1 outside every interval, or ``outside`` on a chromosome the intervals
    are on."""

    def __init__(self, name, classes, chrom, start, end, code, outside=-1):
        self.name = name
        self.classes = tuple(classes)
        self.outside = int(outside)
        cc = _codes_of(chrom)
        code = np.asarray(code, np.int64)
        keep = code >= 0
        g0 = (cc * STRIDE + np.asarray(start, np.int64))[keep]
        g1 = (cc * STRIDE + np.asarray(end, np.int64))[keep]
        o = np.argsort(g0, kind="mergesort")
        self._g0, self._g1 = g0[o], g1[o]
        self._k = code[keep][o].astype(np.int8)
        self._chroms = np.unique(cc)

    def lookup_g(self, g) -> np.ndarray:
        g = np.asarray(g, np.int64)
        out = np.full(g.size, -1, np.int8)
        if self.outside >= 0:
            out[np.isin(g // STRIDE, self._chroms)] = self.outside
        i = np.searchsorted(self._g0, g, side="right") - 1
        ok = i >= 0
        ok[ok] = g[ok] < self._g1[i[ok]]
        out[ok] = self._k[i[ok]]
        return out

    def lookup(self, chrom, pos) -> np.ndarray:
        return self.lookup_g(_codes_of(chrom) * STRIDE
                             + np.asarray(pos, np.int64))


def _merged(df: pd.DataFrame) -> pd.DataFrame:
    """Overlapping or touching intervals merged, per chromosome."""
    out = []
    for c, g in df.groupby("chrom", sort=False):
        g = g.sort_values("start")
        st, en = g["start"].to_numpy(), g["end"].to_numpy()
        new = np.r_[True, st[1:] > np.maximum.accumulate(en)[:-1]]
        out.append(pd.DataFrame({
            "chrom": c, "start": st[new],
            "end": np.maximum.reduceat(en, np.flatnonzero(new))}))
    return pd.concat(out, ignore_index=True) if out else df


def _class_name(path) -> str:
    n = Path(str(path)).name
    for ext in (".gz", ".bed", ".txt"):
        n = n[: -len(ext)] if n.lower().endswith(ext) else n
    return n


def read_annotation(name, path, max_labels: int = 12) -> Intervals:
    """Any BED as an annotation system.  With a 4th column of at most
    ``max_labels`` distinct labels, its labels are the classes (the BED's
    intervals must not overlap); otherwise two classes, inside the
    (merged) intervals and outside them -- e.g. LAD / non-LAD.  Several
    BEDs (a list) are one class each, named by file -- e.g. cLAD.bed,
    fLAD.bed, ciLAD.bed, a partition; they must not overlap."""
    if isinstance(path, (list, tuple)):
        parts = []
        for pth in path:
            x = pd.read_csv(pth, sep="\t", header=None, comment="#",
                            compression="infer", dtype={0: str},
                            usecols=[0, 1, 2])
            x[3] = _class_name(pth)
            parts.append(x)
        df = pd.concat(parts, ignore_index=True)
        classes = tuple(_class_name(pth) for pth in path)
        iv = Intervals(name, classes, df[0], df[1], df[2],
                       df[3].map({c: i for i, c in enumerate(classes)}))
        return iv
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     compression="infer", dtype={0: str})
    if df.shape[1] >= 4 and df[3].astype(str).nunique() <= max_labels:
        lab = df[3].astype(str)
        classes = tuple(dict.fromkeys(sorted(lab.unique())))
        code = lab.map({c: i for i, c in enumerate(classes)})
        return Intervals(name, classes, df[0], df[1], df[2], code)
    iv = _merged(pd.DataFrame({"chrom": df[0].astype(str),
                               "start": df[1].astype(np.int64),
                               "end": df[2].astype(np.int64)}))
    return Intervals(name, (name, f"non-{name}"), iv["chrom"], iv["start"],
                     iv["end"], np.zeros(len(iv), int), outside=1)


def read_pmd(path, class_col: int = 5, common_only: bool = False) -> Intervals:
    """PMD / HMD from the Zhou-lab 6-column file (chrom, start, end, score,
    class, label).  The class is column 5 (PMD / HMD); with ``common_only``
    column 6 instead, keeping commonPMD / commonHMD -- the subset shared
    across their samples, which in the mouse file is a small minority."""
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     compression="infer", dtype={0: str})
    if common_only:
        lab = df[5].astype(str).map({"commonHMD": 0, "commonPMD": 1})
    else:
        lab = df[class_col - 1].astype(str).str.upper().map({"HMD": 0,
                                                             "PMD": 1})
    code = lab.fillna(-1).astype(int)
    return Intervals("pmd", ("HMD", "PMD"), df[0], df[1], df[2], code)


def read_chromhmm(bed, groups_tsv, enr) -> Intervals:
    """A chromHMM BED (column 4 = state) mapped to its groups through the
    state annotation (as marker_bin_enrichment.py does), then to
    CHROMHMM_ORDER."""
    seg = enr._read_bed(bed, want_name=True)
    ann = pd.read_csv(groups_tsv, sep=None, engine="python", dtype=str)
    gcol = next((c for c in ann.columns if c.lower() == "group"), None)
    if gcol is None:
        sys.exit(f"{groups_tsv}: no 'group' column ({list(ann.columns)})")
    mp, hit, tot, col = enr._label_to_group(seg["name"].unique(), ann, gcol)
    grp = seg["name"].map(mp).map(
        lambda g: _CHROMHMM_LC.get(str(g).strip().lower()))
    logger.info(f"chromHMM: {hit}/{tot} states matched via {col!r}; "
                f"{grp.notna().mean():.1%} of segments in "
                f"{len(CHROMHMM_ORDER)} groups")
    code = grp.map({g: i for i, g in enumerate(CHROMHMM_ORDER)})
    return Intervals("chromhmm", CHROMHMM_ORDER, seg["chrom"], seg["start"],
                     seg["end"], code.fillna(-1).astype(int))


_TRACKS: dict = {}


def _track(path) -> dict:
    """{chromosome code: (starts, ends, values)} of a bigWig -- all its
    intervals -- or a bedGraph, read once per run."""
    path = str(path)
    if path in _TRACKS:
        return _TRACKS[path]
    out = {}
    if path.lower().endswith((".bw", ".bigwig")):
        import pyBigWig
        bw = pyBigWig.open(path)
        for c in bw.chroms():
            iv = bw.intervals(c)
            if iv:
                a_ = np.asarray(iv, float)
                out[_chrom_code(c)] = (a_[:, 0].astype(np.int64),
                                       a_[:, 1].astype(np.int64), a_[:, 2])
        bw.close()
    else:
        df = pd.read_csv(path, sep="\t", header=None, comment="#",
                         compression="infer", dtype={0: str})
        for c, g in df.groupby(0):
            out[_chrom_code(c)] = (g[1].to_numpy(np.int64),
                                   g[2].to_numpy(np.int64),
                                   pd.to_numeric(g[3], errors="coerce")
                                   .to_numpy(float))
    for c, (st, en, v) in list(out.items()):
        ok = np.isfinite(v) & (en > st)
        o = np.argsort(st[ok], kind="mergesort")
        out[c] = (st[ok][o], en[ok][o], v[ok][o])
    _TRACKS[path] = out
    return out


def _window_means(track: dict, code, start, end) -> np.ndarray:
    """Coverage-weighted mean of a track over windows [start, end) of one
    chromosome, all at once from the track's running integral; NaN where
    nothing covers a window.  The track's intervals must not overlap."""
    start = np.asarray(start, np.int64)
    end = np.asarray(end, np.int64)
    t = track.get(int(code))
    if t is None or t[0].size == 0:
        return np.full(start.size, np.nan)
    st, en, v = t
    cum_i = np.concatenate([[0.0], np.cumsum(v * (en - st))])
    cum_l = np.concatenate([[0.0], np.cumsum((en - st).astype(float))])

    def upto(x):          # integral and covered length over [0, x)
        j = np.searchsorted(st, x, side="right") - 1
        jj = np.maximum(j, 0)
        part = np.where(j >= 0, np.clip(np.minimum(x, en[jj]) - st[jj], 0,
                                        None), 0)
        return (np.where(j >= 0, cum_i[jj] + v[jj] * part, 0.0),
                np.where(j >= 0, cum_l[jj] + part, 0.0))
    i1, l1 = upto(end)
    i0, l0 = upto(start)
    cov = l1 - l0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cov > 0, (i1 - i0) / cov, np.nan)


def rt_log2(early, late, chrom, start, end) -> np.ndarray:
    """log2(early / late) per window (chromosome, start, end), each side the
    mean of its tracks."""
    import warnings
    codes = _codes_of(chrom)
    start = np.asarray(start, np.int64)
    end = np.asarray(end, np.int64)

    def side(paths):
        vals = []
        for pth in paths:
            tr = _track(pth)
            v = np.full(codes.size, np.nan)
            for c in np.unique(codes):
                sel = codes == c
                v[sel] = _window_means(tr, c, start[sel], end[sel])
            vals.append(v)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # empty windows
            return np.nanmean(vals, 0)
    e, lt = side(early), side(late)
    pos = np.concatenate([e[np.isfinite(e) & (e > 0)],
                          lt[np.isfinite(lt) & (lt > 0)]])
    eps = 1e-3 * float(np.median(pos)) if pos.size else 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log2((e + eps) / (lt + eps))
    r[~(np.isfinite(e) & np.isfinite(lt)) | ((e <= 0) & (lt <= 0))] = np.nan
    return r


def rt_from_log2(tracks, chrom, start, end) -> np.ndarray:
    """log2(early / late) per window from tracks that already hold it (e.g.
    fetch_4dn_repliseq.py's <cell type>_RT_250kb.bedGraph): the mean of the
    tracks' window means."""
    import warnings
    codes = _codes_of(chrom)
    start = np.asarray(start, np.int64)
    end = np.asarray(end, np.int64)
    vals = []
    for pth in tracks:
        tr = _track(pth)
        v = np.full(codes.size, np.nan)
        for c in np.unique(codes):
            sel = codes == c
            v[sel] = _window_means(tr, c, start[sel], end[sel])
        vals.append(v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # empty windows
        return np.nanmean(vals, 0)


def _rt_path(x) -> str:
    """A --rt-log2 value: a path, or NAME=path as fetch_4dn_repliseq.py's
    signal_track_flags.txt writes them."""
    x = str(x)
    return x.split("=", 1)[1] if "=" in x and not Path(x).exists() else x


def read_rt(early, late, window: int, mid: float, exclude=(),
            log2=None) -> Intervals:
    """Early / late (and, with ``mid`` > 0, mid: |log2 E/L| <= mid) per
    window, from early and late fraction tracks -- or from ``log2``, tracks
    of log2(E/L) itself.  In post-mitotic neurons this is a chromatin-state
    proxy (early ~ A), not replication."""
    excl = {_chrom_code(c) for c in exclude}
    ext: dict = {}
    for pth in (list(log2) if log2 else list(early) + list(late)):
        for c, (st, en, _) in _track(pth).items():
            if c not in excl and en.size:
                ext[c] = max(ext.get(c, 0), int(en.max()))
    st = [np.arange(0, n, window, dtype=np.int64) for _, n in sorted(ext.items())]
    ch = np.concatenate([np.full(x.size, c, np.int64)
                         for (c, _), x in zip(sorted(ext.items()), st)])
    st = np.concatenate(st)
    r = (rt_from_log2(log2, ch, st, st + window) if log2
         else rt_log2(early, late, ch, st, st + window))
    code = np.full(st.size, -1)
    code[r > mid] = 0
    code[r < -mid] = 1
    classes = ("early", "late")
    if mid > 0:
        code[np.isfinite(r) & (np.abs(r) <= mid)] = 2
        classes = ("early", "late", "mid")
    logger.info(f"RT: {st.size:,} windows of {window:,} bp; "
                + ", ".join(f"{c} {np.mean(code == i):.1%}"
                            for i, c in enumerate(classes)))
    return Intervals("rt", classes, ch, st, st + window, code)


# ------------------------------------------------------- A/B per cell type --


def _categorical(path, columns):
    """A cell's parquet with its chromosome column as a category, the codes
    of the categories (:func:`_chrom_code`, -1 last, for a missing name) and
    whether each is kept, so per-row work is integer indexing."""
    import pyarrow.parquet as pq
    df = pq.read_table(str(path), columns=columns).to_pandas(
        categories=["chrom"])
    names = [str(c) for c in df["chrom"].cat.categories]
    ci = df["chrom"].cat.codes.to_numpy().astype(np.int64)  # -1: missing
    lut = np.array([_chrom_code(c) for c in names] + [-1], np.int64)
    return df, names, ci, lut


#: Contact-distance bands of the cell-cycle shares (Nagano et al. 2017):
#: among contacts >= 25 kb, "near" ones and the mitotic band.
NEAR = (25_000, 2_000_000)
MITOTIC = (2_000_000, 12_000_000)
#: The distances the per-cell P(s) slope is fitted over.
PS_RANGE = (50_000, 2_000_000)
#: Every annotated contact of a system, as one region group.
ALL_REGIONS = "all regions"


def _contact_pass(path, min_d: int, res: int, exclude, rt=None) -> tuple:
    """One cell's read pairs, once: their A/B bins -- (chromosome code, left
    bin, right bin) at ``res`` for pairs >= min_d, and {code: name} -- and
    its cell-cycle covariates:

      contacts_25kb  pairs at least 25 kb apart
      frac_near      their share at 25 kb - 2 Mb, which rises through S/G2
      frac_mitotic   their share at 2 - 12 Mb, the mitotic band
      repli_score    read ends in early- over early + late-replicating
                     windows (``rt``): early domains are copied first, so
                     it rises in S phase
    """
    df, names, ci, lut = _categorical(path, ["chrom", "pos1", "pos2"])
    ok_lut = np.array([_canon(c) not in exclude for c in names] + [False])
    keep = ok_lut[ci]
    cc = lut[ci][keep]
    p1 = df["pos1"].to_numpy(np.int64)[keep]
    p2 = df["pos2"].to_numpy(np.int64)[keep]
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    sep = hi - lo
    lr = sep >= min_d
    bins = (cc[lr], (lo[lr] // res).astype(np.int32),
            (hi[lr] // res).astype(np.int32),
            {int(lut[j]): _canon(n) for j, n in enumerate(names)})
    n25 = int((sep >= NEAR[0]).sum())
    cov = {"read_pairs": int(sep.size), "contacts_25kb": n25,
           "frac_short": (float((sep < min_d).sum()) / sep.size
                          if sep.size else np.nan),
           "frac_near": (float(((sep >= NEAR[0]) & (sep < NEAR[1])).sum())
                         / n25 if n25 else np.nan),
           "frac_mitotic": (float(((sep >= MITOTIC[0])
                                   & (sep < MITOTIC[1])).sum()) / n25
                            if n25 else np.nan),
           "repli_score": np.nan}
    if rt is not None:
        k = rt.lookup_g(np.concatenate([cc * STRIDE + p1, cc * STRIDE + p2]))
        e_, l_ = int((k == 0).sum()), int((k == 1).sum())
        cov["repli_score"] = e_ / (e_ + l_) if e_ + l_ else np.nan
    # how its contacts fall off with distance: the log-log slope of the
    # contact density between PS_RANGE's ends
    cov["ps_slope"] = np.nan
    m = (sep >= PS_RANGE[0]) & (sep <= PS_RANGE[1])
    if m.sum() >= 100:
        edges = np.logspace(np.log10(PS_RANGE[0]), np.log10(PS_RANGE[1]), 13)
        cnt = np.histogram(sep[m], edges)[0]
        ok_ = cnt > 0
        if ok_.sum() >= 4:
            mid = np.sqrt(edges[:-1] * edges[1:])
            cov["ps_slope"] = float(np.polyfit(
                np.log10(mid[ok_]), np.log10(cnt[ok_] / np.diff(edges)[ok_]),
                1)[0])
    return bins, cov


def _top_eigvecs(M: np.ndarray, k: int = 3) -> np.ndarray:
    """The top ``k`` eigenvectors (columns) of the correlation of the log
    O/E rows -- nome_hic_concordance.compartments' recipe, keeping more
    than PC1.  Bins with no contacts are NaN."""
    n = M.shape[0]
    out = np.full((n, k), np.nan)
    cov = M.sum(axis=1) > 0
    if cov.sum() <= k + 1:
        return out
    C = np.nan_to_num(np.corrcoef(np.log1p(_oe_normalize(M[np.ix_(cov, cov)]))),
                      nan=0.0)
    vals, vecs = np.linalg.eigh(C)
    out[cov] = vecs[:, ::-1][:, :k]
    return out


def compartment_track(M: np.ndarray, ref: np.ndarray, k: int = 3):
    """The A/B track of one chromosome: of the top ``k`` eigenvectors, the
    one that best follows the reference (|r|), signed so that it correlates
    positively.  PC1 alone is not reliable -- on a sparse or unevenly
    covered map it can follow coverage along the chromosome instead -- so,
    as standard tools do with GC content, the reference picks.  Returns
    (track, which eigenvector, r with the reference)."""
    ev = _top_eigvecs(M, k)
    best, best_r = None, 0.0
    for i in range(k):
        ok = np.isfinite(ev[:, i]) & np.isfinite(ref)
        if ok.sum() < 3 or np.std(ev[ok, i]) == 0 or np.std(ref[ok]) == 0:
            continue
        r = float(np.corrcoef(ev[ok, i], ref[ok])[0, 1])
        if best is None or abs(r) > abs(best_r):
            best, best_r = i, r
    if best is None:
        return np.full(M.shape[0], np.nan), -1, np.nan
    return ev[:, best] * np.sign(best_r), best + 1, abs(best_r)


def compartments_from_contacts(by_cell: list, cell_type: list, res: int,
                               orient, min_contacts: int):
    """A/B per cell type from its POOLED contacts.

    ``by_cell[i]``: the A/B bins of :func:`_contact_pass` for cell i.  Per type and
    chromosome: the bin x bin count matrix -- each type's contacts counted
    in one pass -- and its compartment track (:func:`compartment_track`:
    the eigenvector of the O/E correlation that best follows
    ``orient(code, name, n_bins)``, an A-like reference per bin, signed to
    correlate positively with it).  A type with fewer than ``min_contacts``
    contacts takes the all-cell call.  Returns ({type: Intervals}, the
    per-bin table, QC)."""
    name_of: dict = {}
    nb: dict = {}
    for cc, b1, b2, nm in by_cell:
        name_of.update(nm)
        for c in np.unique(cc):
            nb[int(c)] = max(nb.get(int(c), 0), int(b2[cc == c].max()) + 1)
    chroms = sorted(nb)
    types = sorted(set(cell_type))
    flats: dict = {}
    n_type = dict.fromkeys(types + ["__all__"], 0)
    for (cc, b1, b2, _), t in zip(by_cell, cell_type):
        for c in np.unique(cc):
            sel = cc == c
            flats.setdefault((t, int(c)), []).append(
                b1[sel].astype(np.int64) * nb[int(c)] + b2[sel])
            n_type[t] += int(sel.sum())
            n_type["__all__"] += int(sel.sum())
    counts = {t: {} for t in types + ["__all__"]}
    for c in chroms:
        tot = np.zeros(nb[c] * nb[c])
        for t in types:
            f = flats.pop((t, c), [])
            v = (np.bincount(np.concatenate(f), minlength=nb[c] * nb[c])
                 .astype(float) if f else np.zeros(nb[c] * nb[c]))
            counts[t][c] = v
            tot += v
        counts["__all__"][c] = tot
    scores, refs, picked = {}, {}, {}
    for t in types + ["__all__"]:
        scores[t], picked[t] = {}, []
        for c in chroms:
            U = counts[t][c].reshape(nb[c], nb[c])
            M = U + U.T - np.diag(np.diag(U))
            if c not in refs:
                refs[c] = np.asarray(orient(c, name_of.get(c, str(c)),
                                            nb[c]), float)
            scores[t][c], which, r = compartment_track(M, refs[c])
            picked[t].append(f"{name_of.get(c, c)}:PC{which}")
    rows, qc, maps = [], [], {}
    allv = np.concatenate([scores["__all__"][c] for c in chroms])
    for t in types + ["__all__"]:
        own = n_type[t] >= min_contacts or t == "__all__"
        src = t if own else "__all__"
        v = np.concatenate([scores[src][c] for c in chroms])
        ok = np.isfinite(v) & np.isfinite(allv)
        rv = np.concatenate([refs[c] for c in chroms])
        okr = np.isfinite(v) & np.isfinite(rv)
        qc.append({"cell_type": t, "contacts": n_type[t],
                   "called_from": "own" if own else "all cells",
                   "eigenvector": ",".join(picked[src]),
                   "frac_A": float(np.mean(v[np.isfinite(v)] > 0)),
                   "r_with_all_cells": (float(np.corrcoef(v[ok], allv[ok])[0, 1])
                                        if ok.sum() > 2 else np.nan),
                   "r_with_reference": (float(np.corrcoef(v[okr], rv[okr])[0, 1])
                                        if okr.sum() > 2 else np.nan)})
        ch, st, en, code = [], [], [], []
        for c in chroms:
            pc = scores[src][c]
            fin = np.isfinite(pc) & (pc != 0)
            b = np.flatnonzero(fin)
            ch += [c] * b.size
            st.append(b * res)
            en.append((b + 1) * res)
            code.append(np.where(pc[b] > 0, 0, 1))
            rows.append(pd.DataFrame({"cell_type": t,
                                      "chrom": name_of.get(c, str(c)),
                                      "start": np.arange(nb[c]) * res,
                                      "end": (np.arange(nb[c]) + 1) * res,
                                      "pc1": pc,
                                      "ab": np.where(np.isfinite(pc),
                                                     np.where(pc > 0, "A", "B"),
                                                     "")}))
        maps[t] = Intervals("ab", ("A", "B"), ch, np.concatenate(st),
                            np.concatenate(en), np.concatenate(code))
    return maps, pd.concat(rows, ignore_index=True), pd.DataFrame(qc)


# ----------------------------------------------------------------- one cell --


def _read_cell(path, min_sites: int, min_d: int, max_d: int, exclude,
               systems: dict, open_g=None, open_w: int = 200,
               prof_bin: int = 0, with_x: bool = False, loci=None) -> dict:
    """One cell's contacts -- both ends called, min_d <= d <= max_d -- as
    genome coordinates, with every system's class at both ends, and per
    system x class the count, sum and sum of squares of ALL its called
    ends' posteriors (the marginal: every read pair's ends, short ones
    included).  With ``open_g``, the sites methylated and the sites of its
    ends within ``open_w`` of an open site (the posterior instead when the
    file has no counts); with ``prof_bin``, its called ends' posterior sum
    and count per bin of that size.  With ``with_x``, its chrX contacts
    and the marginal of its X ends and of its autosomal ends, kept apart
    ("x") so that nothing but the chrX analysis sees them.  With ``loci``
    (genome intervals, n x 2), the count and posterior sum of its called
    ends in each.  Integer work only, so threads run in parallel."""
    import pyarrow.parquet as pq
    cols = ["chrom", "pos1", "pos2", "n1", "n2", "posterior1", "posterior2"]
    has_k = {"k1", "k2"} <= set(pq.read_schema(str(path)).names)
    df, names, ci, lut = _categorical(path, cols + (["k1", "k2"] if has_k
                                                    else []))
    ok_lut = np.array([_canon(c) not in exclude for c in names] + [False])
    keep = ok_lut[ci]
    base = lut[ci][keep] * STRIDE
    pos1 = df["pos1"].to_numpy(np.int64)[keep]
    pos2 = df["pos2"].to_numpy(np.int64)[keep]
    p1 = df["posterior1"].to_numpy(np.float64, na_value=np.nan)[keep]
    p2 = df["posterior2"].to_numpy(np.float64, na_value=np.nan)[keep]
    inf1 = (df["n1"].to_numpy()[keep] >= min_sites) & np.isfinite(p1)
    inf2 = (df["n2"].to_numpy()[keep] >= min_sites) & np.isfinite(p2)
    g1, g2 = base + pos1, base + pos2
    e_g = np.concatenate([g1[inf1], g2[inf2]])
    e_p = np.concatenate([p1[inf1], p2[inf2]])
    sep = np.abs(pos2 - pos1)
    cm = inf1 & inf2 & (sep >= min_d) & (sep <= max_d)
    left = pos1 <= pos2
    g_lo = np.where(left, g1, g2)[cm]
    g_hi = np.where(left, g2, g1)[cm]
    out = {"g_lo": g_lo, "g_hi": g_hi,
           "a": np.where(left, p1, p2)[cm].astype(np.float32),
           "p": np.where(left, p2, p1)[cm].astype(np.float32),
           "cls": {}, "marg": {},
           "level": (float(e_p.sum()), int(e_p.size))}   # genome-wide
    out["chroms"] = {int(lut[i]): names[i] for i in range(len(names))
                     if ok_lut[i]}
    # How well this cell's ends are called: the share called, the sites an
    # end carries and how decisive its posterior is.
    n1_ = df["n1"].to_numpy()[keep]
    n2_ = df["n2"].to_numpy()[keep]
    if open_g is not None and open_g.size:
        gg = np.concatenate([g1, g2])
        nn = np.concatenate([n1_, n2_]).astype(float)
        m = nn > 0
        near = np.zeros(gg.size, bool)
        near[m] = _nearest(open_g, gg[m]) <= open_w
        if has_k:
            kk = np.concatenate([df["k1"].to_numpy()[keep],
                                 df["k2"].to_numpy()[keep]]).astype(float)
            out["open"] = (float(kk[near].sum()), float(nn[near].sum()))
        else:
            called = np.concatenate([inf1, inf2]) & near
            out["open"] = (float(np.concatenate([p1, p2])[called].sum()),
                           float(called.sum()))
    if prof_bin:
        key = (e_g // STRIDE) * (1 << 20) + (e_g % STRIDE) // prof_bin
        uk, inv = np.unique(key, return_inverse=True)
        # and a random half of the ends: two halves make a doublet with one
        # cell's depth, so it is no less noisy than a real cell
        half = np.random.default_rng(zlib.crc32(str(path).encode())).random(
            e_p.size) < 0.5
        out["prof"] = (uk, np.bincount(inv, weights=e_p).astype(np.float32),
                       np.bincount(inv).astype(np.int32),
                       np.bincount(inv, weights=e_p * half,
                                   minlength=uk.size).astype(np.float32),
                       np.bincount(inv, weights=half.astype(float),
                                   minlength=uk.size).astype(np.int32))
    n_called = int(inf1.sum() + inf2.sum())
    out["quality"] = {
        "frac_called": n_called / (2 * pos1.size) if pos1.size else np.nan,
        "sites_mean": (float(n1_[inf1].sum() + n2_[inf2].sum()) / n_called
                       if n_called else np.nan),
        "decisive": (float(np.abs(2 * e_p - 1).mean()) if e_p.size
                     else np.nan)}
    for name, lk in systems.items():
        k = lk.lookup_g(e_g)
        ok = k >= 0
        n_k = len(lk.classes)
        out["marg"][name] = np.stack([
            np.bincount(k[ok], minlength=n_k).astype(float),
            np.bincount(k[ok], weights=e_p[ok], minlength=n_k),
            np.bincount(k[ok], weights=e_p[ok] ** 2, minlength=n_k)])
        out["cls"][name] = (lk.lookup_g(g_lo), lk.lookup_g(g_hi))
    if loci is not None:
        lc = np.zeros((len(loci), 2))
        for i, (l0, l1) in enumerate(loci):
            m = (e_g >= l0) & (e_g < l1)
            lc[i] = m.sum(), e_p[m].sum()
        out["loci"] = lc
    if with_x:
        xl = (lut[ci] == _chrom_code("X"))
        xb = lut[ci][xl] * STRIDE
        q1 = df["pos1"].to_numpy(np.int64)[xl]
        q2 = df["pos2"].to_numpy(np.int64)[xl]
        r1 = df["posterior1"].to_numpy(np.float64, na_value=np.nan)[xl]
        r2 = df["posterior2"].to_numpy(np.float64, na_value=np.nan)[xl]
        j1 = (df["n1"].to_numpy()[xl] >= min_sites) & np.isfinite(r1)
        j2 = (df["n2"].to_numpy()[xl] >= min_sites) & np.isfinite(r2)
        xs = np.abs(q2 - q1)
        xm = j1 & j2 & (xs >= min_d) & (xs <= max_d)
        lf = q1 <= q2
        h1, h2 = xb + q1, xb + q2
        xe = np.concatenate([r1[j1], r2[j2]])
        au = (e_g // STRIDE >= 1) & (e_g // STRIDE < 1000)
        out["x"] = {"g_lo": np.where(lf, h1, h2)[xm],
                    "g_hi": np.where(lf, h2, h1)[xm],
                    "a": np.where(lf, r1, r2)[xm].astype(np.float32),
                    "p": np.where(lf, r2, r1)[xm].astype(np.float32),
                    # count, sum, sum of squares: autosomes, then X
                    "marg": np.array([[au.sum(), xe.size],
                                      [e_p[au].sum(), xe.sum()],
                                      [(e_p[au] ** 2).sum(),
                                       (xe ** 2).sum()]], float)}
    return out


# --------------------------------------------------------------- the null --


def _pool(g, v, cell):
    o = np.argsort(g, kind="mergesort")
    v = np.asarray(v, float)[o]
    return (g[o], v, np.asarray(cell, np.int64)[o],
            np.concatenate([[0.0], np.cumsum(v)]))


def reshuffle(g_lo, g_hi, d, cell, x_cls, y_cls, a_c, p_c, stratum, n_cls,
              dbins, position_bin, max_widen, seed, names=None, n_threads=1):
    """Two null draws for BOTH ends of every contact.

    The right end first: keep the left end, and replace the right one by a
    donor -- an end of another cell's contact in the same stratum, in the
    right end's class, at a distance in the contact's own log bin, inside
    the right end's ``position_bin`` bin, never a donor another contact got
    in that draw (link_shuffle.assign_stratum).  Then the mirror image: keep
    the right end and replace the left one, from donors to ITS left, in the
    left end's class and bin.  Reshuffling only one end would make the null
    depend on which end sits on the left -- an early-late contact would
    always draw a late donor and a late-early one an early donor -- so
    orientations that should agree could not.  Donor values are centred on
    their own cell's class marginal (``a_c``/``p_c`` for every end).

    ``stratum``: an integer per contact (or labels), ``names`` the label of
    each integer, which seeds its draws.  The stratum x class x end
    assignments are independent and run on ``n_threads`` threads.

    Returns (ok, right 2 x n, left 2 x n, reused): ``ok`` -- both ends have
    a donor; ``right``/``left`` -- the centred donor of each end per draw."""
    n = g_lo.size
    right = np.full((2, n), np.nan)
    left = np.full((2, n), np.nan)
    lv_r = np.full(n, -1, np.int64)
    lv_l = np.full(n, -1, np.int64)
    usable = (x_cls >= 0) & (y_cls >= 0) & np.isfinite(a_c) & np.isfinite(p_c)
    pb = np.int64(position_bin)
    # Mirror: g -> M - g turns "to the left" into "to the right".
    M = np.int64(max(int(g_hi.max()) if n else 0, 0)
                 + int(np.ceil(dbins[-1])) + int(pb) + 2)
    stratum = np.asarray(stratum)
    if stratum.dtype.kind not in "iu":
        stratum, names = pd.factorize(stratum)
        names = list(names)
    stratum = stratum.astype(np.int64)
    n_s = int(stratum.max()) + 1 if n else 0
    order = np.argsort(stratum, kind="stable")
    bnd = np.searchsorted(stratum[order], np.arange(n_s + 1))
    tasks = []
    for si in range(n_s):
        idx = order[bnd[si]:bnd[si + 1]]
        if idx.size == 0:
            continue
        xs, ys, us = x_cls[idx], y_cls[idx], usable[idx]
        fa, fp = np.isfinite(a_c[idx]), np.isfinite(p_c[idx])
        key = zlib.crc32(str(names[si] if names is not None else si).encode())
        for k in range(n_cls):
            lo_i = idx[(xs == k) & fa]
            hi_i = idx[(ys == k) & fp]
            if lo_i.size + hi_i.size == 0:
                continue
            for side, ask in (("right", idx[us & (ys == k)]),
                              ("left", idx[us & (xs == k)])):
                if ask.size:
                    tasks.append((key, k, side, lo_i, hi_i, ask))

    def bin_start(g):
        """The start of g's position bin, counted from its chromosome's
        start: STRIDE is not a multiple of the bin, so g // pb would shift
        every chromosome but the first by its own offset."""
        return (g // STRIDE) * STRIDE + ((g % STRIDE) // pb) * pb

    def run(task):
        key, k, side, lo_i, hi_i, ask = task
        pg = np.concatenate([g_lo[lo_i], g_hi[hi_i]])
        pv = np.concatenate([a_c[lo_i], p_c[hi_i]])
        pc = np.concatenate([cell[lo_i], cell[hi_i]])
        rng = np.random.default_rng([seed, key, k,
                                     0 if side == "right" else 1])
        if side == "right":
            pool = _pool(pg, pv, pc)
            start = bin_start(g_hi[ask])
            lv, _, dp, ru = _ls.assign_stratum(
                g_lo[ask], d[ask], cell[ask], pool, pool, dbins, rng,
                n_draws=2, bounds=(start, start + pb), max_widen=max_widen)
        else:
            pool = _pool(M - pg, pv, pc)
            b0 = bin_start(g_lo[ask])
            lv, _, dp, ru = _ls.assign_stratum(
                M - g_hi[ask], d[ask], cell[ask], pool, pool, dbins, rng,
                n_draws=2, bounds=(M - b0 - pb + 1, M - b0 + 1),
                max_widen=max_widen)
        return side, ask, lv, dp, ru

    if n_threads > 1 and len(tasks) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=int(n_threads)) as ex:
            results = list(ex.map(run, tasks))
    else:
        results = [run(t) for t in tasks]
    reused = 0
    for side, ask, lv, dp, ru in results:
        if side == "right":
            lv_r[ask], right[:, ask] = lv, dp
        else:
            lv_l[ask], left[:, ask] = lv, dp
        reused += ru
    ok = usable & (lv_r >= 0) & (lv_l >= 0)
    return ok, right, left, reused


# -------------------------------------------------------------- statistics --


def _bands(edges):
    lab = [f"{_fmt_bp(a)}-{_fmt_bp(b)}" for a, b in zip(edges[:-1], edges[1:])]
    return lab + [f"all {_fmt_bp(edges[0])}-{_fmt_bp(edges[-1])}"]


def _fmt_bp(x) -> str:
    x = float(x)
    for unit, f in (("Mb", 1e6), ("kb", 1e3)):
        if x >= f:
            v = x / f
            return f"{v:g}{unit}"
    return f"{x:g}bp"


def accumulate(cell, kx, ky, band, half, weights: dict, n_cells, n_k,
               n_band):
    """Per (cell, class, class, band) sums of each of ``weights`` (and the
    count n) -- band n_band is every band together -- overall and in half
    0."""
    size = n_cells * n_k * n_k * (n_band + 1)
    base = (cell * n_k + kx) * n_k + ky
    out = {}
    for tag, b in (("band", band), ("all", np.full(band.size, n_band))):
        key = base * (n_band + 1) + b
        for name, w in [("n", None)] + list(weights.items()):
            out.setdefault(name, np.zeros(size))
            out[name] += np.bincount(key, weights=w, minlength=size)
            h = half == 0
            out.setdefault(name + "_h0", np.zeros(size))
            out[name + "_h0"] += np.bincount(
                key[h], weights=None if w is None else w[h], minlength=size)
    shape = (n_cells, n_k, n_k, n_band + 1)
    return {k: v.reshape(shape) for k, v in out.items()}


def _within_type_spearman(x, y, types) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    df = pd.DataFrame({"x": x[ok], "y": y[ok], "t": np.asarray(types)[ok]})
    # ranks with each type's mean rank removed
    rx = df["x"].rank()
    ry = df["y"].rank()
    rx = rx - rx.groupby(df["t"]).transform("mean")
    ry = ry - ry.groupby(df["t"]).transform("mean")
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _type_f(v, types) -> tuple:
    from scipy.stats import f_oneway
    groups = [v[(np.asarray(types) == t) & np.isfinite(v)]
              for t in np.unique(types)]
    groups = [g for g in groups if g.size >= 3]
    if len(groups) < 2:
        return np.nan, np.nan
    f, p = f_oneway(*groups)
    return float(f), float(p)


def _per_cell(val, ctrl, v0, v1, c0, c1, types, rng, n_boot, prefix=""):
    """Between-cell statistics of one per-cell value against its control:
    mean, share above 0, SDs, the excess SD (with a bootstrap CI over
    cells), split-half reliability (``v0``/``v1``: the two halves of each
    cell's contacts) and the cell-type F test -- each for the control too."""
    ok = np.isfinite(val) & np.isfinite(ctrl)
    val, ctrl, t = val[ok], ctrl[ok], np.asarray(types)[ok]
    out = {}
    if val.size < 5:
        return out
    out.update({f"{prefix}cell_mean": float(val.mean()),
                f"{prefix}cell_frac_pos": float(np.mean(val > 0)),
                f"{prefix}control_frac_pos": float(np.mean(ctrl > 0)),
                f"{prefix}cell_sd": float(val.std(ddof=1)),
                f"{prefix}control_sd": float(ctrl.std(ddof=1)),
                f"{prefix}excess_sd": float(np.sqrt(max(
                    val.var(ddof=1) - ctrl.var(ddof=1), 0)))})
    bi = rng.integers(0, val.size, (min(n_boot, 200), val.size))
    ex = np.sqrt(np.maximum(val[bi].var(1, ddof=1)
                            - ctrl[bi].var(1, ddof=1), 0))
    out[f"{prefix}excess_sd_lo"], out[f"{prefix}excess_sd_hi"] = (
        np.percentile(ex, [2.5, 97.5]))
    v0, v1, c0, c1 = v0[ok], v1[ok], c0[ok], c1[ok]
    h = np.isfinite(v0) & np.isfinite(v1) & np.isfinite(c0) & np.isfinite(c1)
    if h.sum() >= 5:
        out[f"{prefix}split_half_r"] = float(np.corrcoef(v0[h], v1[h])[0, 1])
        out[f"{prefix}control_split_half_r"] = float(
            np.corrcoef(c0[h], c1[h])[0, 1])
    out[f"{prefix}type_F"], out[f"{prefix}type_p"] = _type_f(val, t)
    out[f"{prefix}control_type_F"], out[f"{prefix}control_type_p"] = (
        _type_f(ctrl, t))
    return out


def _signed(dc, oo, oc):
    """ΔC x (2F - 1), F = oo / (oo + oc): the co-open share of the
    concordant mass."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return dc * (2 * oo / (oo + oc) - 1)


def _weight(F, Fg):
    """How much more co-open than the reference ``Fg`` a co-open share F is,
    on [-1, 1]: tanh(½[logit F - logit Fg]) = (F - Fg) / (F + Fg - 2 F Fg).
    0 at the reference, +1 all co-open, -1 all co-closed; 2F - 1 is the
    case Fg = 0.5."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (F - Fg) / (F + Fg - 2 * F * Fg)


def _signed_rel(dc, oo, oc, Fg):
    """ΔC x the weight of this class pair's co-open share against the
    cell's genome-wide one."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return dc * _weight(oo / (oo + oc), Fg)


def _signed_sgn(dc, oo, oc, Fg):
    """ΔC x sign(w): ΔC at full size, with only the direction taken from
    the co-open share against the cell's genome-wide one."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return dc * np.sign(_weight(oo / (oo + oc), Fg))


def summarize(acc, sd, mu, cells, types, classes, band_labels, edges,
              min_pairs, n_boot, seed, context, system):
    """One row per (class, class, band)."""
    rng = np.random.default_rng(seed)
    n_cells = len(cells)
    W = rng.multinomial(n_cells, np.full(n_cells, 1.0 / n_cells),
                        size=n_boot).astype(float)
    # Each cell's genome-wide co-open share -- all its contacts annotated
    # in this system, every class pair -- the reference signed_rel is read
    # against; the reshuffled draw 2's own, for the control.
    A = len(band_labels) - 1
    g = {k: acc[k][:, :, :, A].sum((1, 2)) for k in ("oo", "oc", "no2",
                                                     "nc2")}
    with np.errstate(invalid="ignore", divide="ignore"):
        Fg = g["oo"] / (g["oo"] + g["oc"])
        Fgn = g["no2"] / (g["no2"] + g["nc2"])
        bFg = (W @ g["oo"]) / (W @ g["oo"] + W @ g["oc"])
        bFgn = (W @ g["no2"]) / (W @ g["no2"] + W @ g["nc2"])
    Fg_pool = g["oo"].sum() / (g["oo"].sum() + g["oc"].sum())
    Fgn_pool = g["no2"].sum() / (g["no2"].sum() + g["nc2"].sum())
    rows = []
    for i, k1 in enumerate(classes):
        for j, k2 in enumerate(classes):
            for b, bl in enumerate(band_labels):
                n = acc["n"][:, i, j, b]
                if n.sum() == 0:
                    continue
                xy, xy1, xy2 = (acc[k][:, i, j, b] for k in ("xy", "xy1",
                                                              "xy2"))
                N = n.sum()
                d_pool = 2 * (xy.sum() - xy1.sum()) / N
                c_pool = 2 * (xy2.sum() - xy1.sum()) / N
                bn = W @ n
                with np.errstate(invalid="ignore", divide="ignore"):
                    bd = 2 * (W @ xy - W @ xy1) / bn
                    bc = 2 * (W @ xy2 - W @ xy1) / bn
                lo_, hi_ = np.nanpercentile(bd, [2.5, 97.5])
                p_boot = min(1.0, 2 * min(np.mean(bd <= 0), np.mean(bd >= 0))
                             + 1.0 / (n_boot + 1))
                last = b == len(band_labels) - 1
                row = {"context": context, "system": system,
                       "anchor": k1, "partner": k2, "band": bl,
                       "band_lo": float(edges[0] if last else edges[b]),
                       "band_hi": float(edges[-1] if last else edges[b + 1]),
                       "pairs": int(N), "cells_any": int((n > 0).sum()),
                       "C_obs": 2 * xy.sum() / N, "C_null": 2 * xy1.sum() / N,
                       "delta": d_pool, "delta_lo": lo_, "delta_hi": hi_,
                       "p_boot": p_boot, "control": c_pool,
                       "control_lo": np.nanpercentile(bc, 2.5),
                       "control_hi": np.nanpercentile(bc, 97.5),
                       "r_obs": np.nan, "r_null": np.nan}
                # Signed by where the concordant mass sits: ΔC x (2F - 1),
                # F the co-open share of it.  Its control is the same
                # formula on reshuffled links alone -- draw 2 standing in
                # for the real partners: (C2 - C1) x (2 F_2 - 1).
                oo, oc, no2, nc2 = (acc[k][:, i, j, b]
                                    for k in ("oo", "oc", "no2", "nc2"))
                row["F"] = float(oo.sum() / (oo.sum() + oc.sum()))
                row["F_null"] = float(no2.sum() / (no2.sum() + nc2.sum()))
                row["signed"] = d_pool * (2 * row["F"] - 1)
                row["signed_control"] = c_pool * (2 * row["F_null"] - 1)
                bs = _signed(bd, W @ oo, W @ oc)
                bsc = _signed(bc, W @ no2, W @ nc2)
                row["signed_lo"], row["signed_hi"] = np.nanpercentile(
                    bs, [2.5, 97.5])
                row["signed_p_boot"] = min(1.0, 2 * min(
                    np.mean(bs <= 0), np.mean(bs >= 0)) + 1.0 / (n_boot + 1))
                row["signed_control_lo"], row["signed_control_hi"] = (
                    np.nanpercentile(bsc, [2.5, 97.5]))
                # The same against the cell's genome-wide co-open share
                # instead of an even split: GCH is closed over most of the
                # genome, so against 1:1 every class reads co-closed.
                row["F_ref"], row["F_ref_null"] = Fg_pool, Fgn_pool
                row["signed_rel"] = d_pool * _weight(row["F"], Fg_pool)
                row["signed_rel_control"] = c_pool * _weight(row["F_null"],
                                                             Fgn_pool)
                with np.errstate(invalid="ignore", divide="ignore"):
                    br_ = bd * _weight((W @ oo) / (W @ oo + W @ oc), bFg)
                    brc = bc * _weight((W @ no2) / (W @ no2 + W @ nc2), bFgn)
                row["signed_rel_lo"], row["signed_rel_hi"] = (
                    np.nanpercentile(br_, [2.5, 97.5]))
                row["signed_rel_p_boot"] = min(1.0, 2 * min(
                    np.mean(br_ <= 0), np.mean(br_ >= 0)) + 1.0 / (n_boot + 1))
                row["signed_rel_control_lo"], row["signed_rel_control_hi"] = (
                    np.nanpercentile(brc, [2.5, 97.5]))
                # ... and with ΔC at full size: ΔC x sign(w)
                row["signed_sgn"] = d_pool * np.sign(
                    _weight(row["F"], Fg_pool))
                row["signed_sgn_control"] = c_pool * np.sign(
                    _weight(row["F_null"], Fgn_pool))
                with np.errstate(invalid="ignore", divide="ignore"):
                    bg_ = bd * np.sign(_weight((W @ oo) / (W @ oo + W @ oc),
                                               bFg))
                    bgc = bc * np.sign(_weight((W @ no2) / (W @ no2
                                                            + W @ nc2), bFgn))
                row["signed_sgn_lo"], row["signed_sgn_hi"] = (
                    np.nanpercentile(bg_, [2.5, 97.5]))
                row["signed_sgn_p_boot"] = min(1.0, 2 * min(
                    np.mean(bg_ <= 0), np.mean(bg_ >= 0)) + 1.0 / (n_boot + 1))
                row["signed_sgn_control_lo"], row["signed_sgn_control_hi"] = (
                    np.nanpercentile(bgc, [2.5, 97.5]))
                # Correlation scale: each cell's products over its own class
                # SDs, pooled -- comparable between classes of different
                # variance, as C is not.
                s12 = sd[:, i] * sd[:, j]
                okr = (n > 0) & (s12 > 0)
                if okr.any():
                    w_ = np.where(okr, 1.0 / np.where(okr, s12, 1.0), 0.0)
                    nr, rx, rx1 = n * okr, xy * w_, xy1 * w_
                    row["r_obs"] = float(rx.sum() / nr.sum())
                    row["r_null"] = float(rx1.sum() / nr.sum())
                    row["delta_r"] = row["r_obs"] - row["r_null"]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        br = (W @ rx - W @ rx1) / (W @ nr)
                    row["delta_r_lo"], row["delta_r_hi"] = np.nanpercentile(
                        br, [2.5, 97.5])
                # Per cell.
                m = n >= min_pairs
                row["cells"] = int(m.sum())
                if m.sum() >= 5:
                    dc = 2 * (xy[m] - xy1[m]) / n[m]
                    cc = 2 * (xy2[m] - xy1[m]) / n[m]
                    t_m = np.asarray(types)[m]
                    okc = s12[m] > 0
                    if okc.sum() >= 5:
                        dr = (dc / 2 / np.where(okc, s12[m], 1))[okc]
                        cr = (cc / 2 / np.where(okc, s12[m], 1))[okc]
                        row["cell_sd_r"] = float(dr.std(ddof=1))
                        row["control_sd_r"] = float(cr.std(ddof=1))
                        row["excess_sd_r"] = float(np.sqrt(max(
                            dr.var(ddof=1) - cr.var(ddof=1), 0)))
                        bi_ = rng.integers(0, dr.size,
                                           (min(n_boot, 200), dr.size))
                        ex_ = np.sqrt(np.maximum(dr[bi_].var(1, ddof=1)
                                                 - cr[bi_].var(1, ddof=1), 0))
                        row["excess_sd_r_lo"], row["excess_sd_r_hi"] = (
                            np.percentile(ex_, [2.5, 97.5]))
                    keys = ("n", "xy", "xy1", "xy2", "oo", "oc", "no2",
                            "nc2")
                    h0 = {k: acc[k + "_h0"][:, i, j, b][m] for k in keys}
                    h1 = {k: acc[k][:, i, j, b][m] - h0[k] for k in keys}
                    with np.errstate(invalid="ignore", divide="ignore"):
                        d0 = 2 * (h0["xy"] - h0["xy1"]) / h0["n"]
                        d1 = 2 * (h1["xy"] - h1["xy1"]) / h1["n"]
                        c0 = 2 * (h0["xy2"] - h0["xy1"]) / h0["n"]
                        c1 = 2 * (h1["xy2"] - h1["xy1"]) / h1["n"]
                    row.update(_per_cell(dc, cc, d0, d1, c0, c1, t_m, rng,
                                         n_boot))
                    # the same, signed by the co-open share of the mass
                    row.update(_per_cell(
                        _signed(dc, oo[m], oc[m]), _signed(cc, no2[m], nc2[m]),
                        _signed(d0, h0["oo"], h0["oc"]),
                        _signed(d1, h1["oo"], h1["oc"]),
                        _signed(c0, h0["no2"], h0["nc2"]),
                        _signed(c1, h1["no2"], h1["nc2"]),
                        t_m, rng, n_boot, prefix="signed_"))
                    fg, fgn = Fg[m], Fgn[m]
                    row.update(_per_cell(
                        _signed_rel(dc, oo[m], oc[m], fg),
                        _signed_rel(cc, no2[m], nc2[m], fgn),
                        _signed_rel(d0, h0["oo"], h0["oc"], fg),
                        _signed_rel(d1, h1["oo"], h1["oc"], fg),
                        _signed_rel(c0, h0["no2"], h0["nc2"], fgn),
                        _signed_rel(c1, h1["no2"], h1["nc2"], fgn),
                        t_m, rng, n_boot, prefix="signed_rel_"))
                    row.update(_per_cell(
                        _signed_sgn(dc, oo[m], oc[m], fg),
                        _signed_sgn(cc, no2[m], nc2[m], fgn),
                        _signed_sgn(d0, h0["oo"], h0["oc"], fg),
                        _signed_sgn(d1, h1["oo"], h1["oc"], fg),
                        _signed_sgn(c0, h0["no2"], h0["nc2"], fgn),
                        _signed_sgn(c1, h1["no2"], h1["nc2"], fgn),
                        t_m, rng, n_boot, prefix="signed_sgn_"))
                    row["rho_marginal_anchor"] = _within_type_spearman(
                        dc, mu[m, i], t_m)
                    row["rho_marginal_partner"] = _within_type_spearman(
                        dc, mu[m, j], t_m)
                    row["rho_pairs"] = _within_type_spearman(
                        dc, np.log(n[m]), t_m)
                rows.append(row)
    return rows


def marginal_summary(marg, cells, types, classes, min_ends, context, system):
    """The marginal's mean, between-cell spread beyond its sampling noise,
    and cell-type F, per class -- the reference for the concordance's."""
    rows = []
    n, s, s2 = marg[:, 0], marg[:, 1], marg[:, 2]
    for k, name in enumerate(classes):
        m = n[:, k] >= min_ends
        if m.sum() < 5:
            continue
        mu = s[m, k] / n[m, k]
        var = np.maximum(s2[m, k] / n[m, k] - mu ** 2, 0)
        noise = float(np.sqrt(np.mean(var / n[m, k])))
        f, p = _type_f(mu, np.asarray(types)[m])
        rows.append({"context": context, "system": system, "class": name,
                     "cells": int(m.sum()), "ends_median": float(np.median(n[m, k])),
                     "mean": float(mu.mean()), "cell_sd": float(mu.std(ddof=1)),
                     "noise_sd": noise,
                     "excess_sd": float(np.sqrt(max(mu.var(ddof=1)
                                                    - noise ** 2, 0))),
                     "type_F": f, "type_p": p})
    return rows


# ------------------------------------------------------- cell covariates --

#: Within-type principal components of the cells' regional profiles.
SUB_PCS = tuple(f"sub_pc{i}" for i in range(1, 11))
#: Per-cell covariates the within-type variation is set against, grouped:
#: each group is one component of the variance split (with batch).
COV_GROUPS = (
    ("technical", ("log10_contacts", "frac_short", "frac_called",
                   "sites_mean", "decisive")),
    ("doublets", ("doublet_score",)),
    ("bisulfite conversion", ("nonconversion",)),
    # the cell's 1D signal: its genome-wide level, its level in the pair's
    # classes, and the other context's level
    ("1D baseline", ("level", "level_other", "mu_pair_1", "mu_pair_2")),
    ("scale", ("scale",)),
    # the cell's contacts themselves: how its chromatin folds -- the
    # distance profile (% near / % mitotic are that too: in post-mitotic
    # neurons they read folding, not the cycle), compartments, and how
    # many of its contacts this class pair takes
    ("3D contacts", ("compartment_strength", "ps_slope", "frac_near",
                     "frac_mitotic", "pair_share")),
    ("cell cycle", ("repli_score",)),
    ("labeling efficiency", ("open_site_level",)),
    ("mCA", ("mca_index",)),
    ("division history", ("pmd_depth",)),
    ("substructure", SUB_PCS),
)
COVARIATES = tuple(c for _, grp in COV_GROUPS for c in grp)
#: Shown one by one in the heatmaps; the substructure PCs only together.
HEATMAP_COVARIATES = tuple(c for c in COVARIATES if c not in SUB_PCS)
COV_LABEL = {"log10_contacts": "depth", "frac_short": "% short pairs",
             "frac_called": "% ends called", "sites_mean": "sites / end",
             "decisive": "call decisiveness", "level": "level",
             "level_other": "other context's level",
             "mu_pair_1": "class level", "mu_pair_2": "partner class level",
             "ps_slope": "P(s) slope", "pair_share": "pair's share",
             "scale": "scale (class SDs)",
             "frac_near": "% near", "frac_mitotic": "% mitotic",
             "repli_score": "repli-score",
             "compartment_strength": "compartment strength",
             "open_site_level": "open-site GCH", "mca_index": "mCA index",
             "doublet_score": "doublet score",
             "nonconversion": "non-conversion (HCY)",
             "pmd_depth": "PMD depth",
             **{c: c.replace("sub_pc", "substructure PC") for c in SUB_PCS}}
#: The components of the variance split, in the order the bar shows them.
GROUP_COLOR = {"technical": "#08519c", "1D baseline": "#54278f",
               "scale": "#9e9ac8", "cell cycle": "#006d2c",
               "3D contacts": "#74c476",
               "batch": "#8c510a", "labeling efficiency": "#e6550d",
               "mCA": "#c51b7d", "division history": "#1b9e77",
               "substructure": "#cb181d", "doublets": "#bcbd22",
               "bisulfite conversion": "#17becf"}
#: How the cell was measured, not what it is: removed for the corrected
#: cell-type AUC, and grouped first in the split of the total variance.
TECHNICAL_GROUPS = ("technical", "batch", "labeling efficiency", "scale",
                    "doublets", "bisulfite conversion")
#: The total split's components in the order its bar shows them.
TOTAL_ORDER = ("technical", "batch", "labeling efficiency", "scale",
               "doublets", "bisulfite conversion",
               "cell type", "1D baseline", "3D contacts", "cell cycle",
               "mCA", "division history", "substructure")
TOTAL_COLOR = {**GROUP_COLOR, "cell type": "#000000"}
#: Covariates measured per context (from that context's calls).
CTX_COVARIATES = ("level", "sites_mean", "decisive", "frac_called")
_CONTROL_OF = {"delta": "control", "delta_r": "control_r",
               "signed_rel": "signed_rel_control",
               "signed_sgn": "signed_sgn_control",
               "signed": "signed_control"}
#: Statistics the within-type variation is examined for.
COV_STATS = ("delta", "delta_r", "signed_rel", "signed_sgn")
_STAT_LABEL = {"delta": "ΔC", "delta_r": "Δr (ΔC on the correlation scale)",
               "signed_rel": "ΔC × w", "signed_sgn": "ΔC × sign(w)",
               "signed": "ΔC × (2F − 1)"}


def _center(v, t):
    return v - pd.Series(v).groupby(t).transform("mean").to_numpy()


def _r2(v, Z, n_types):
    """(R^2, adjusted R^2, k, residual df) of a no-intercept OLS of
    type-centred ``v`` on type-centred columns ``Z``."""
    n = v.size
    if Z.ndim == 1:
        Z = Z[:, None]
    sd = Z.std(0) if Z.size else np.zeros(0)
    Z = Z[:, sd > 0] / sd[sd > 0] if Z.size else Z
    k = Z.shape[1]
    dfr = n - n_types - k - 1
    if k == 0 or dfr <= 0 or not (v ** 2).sum() > 0:
        return np.nan, np.nan, k, dfr
    beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
    r2 = 1 - ((v - Z @ beta) ** 2).sum() / (v ** 2).sum()
    return (float(r2), float(1 - (1 - r2) * (n - n_types - 1) / dfr), k,
            dfr)


def _center_obs(v, t):
    """Type-centred, each type's mean taken over its observed values; a
    missing value becomes its type's mean (0)."""
    s_ = pd.Series(np.asarray(v, float))
    out = (s_ - s_.groupby(np.asarray(t)).transform("mean")).to_numpy(float)
    return np.where(np.isfinite(out), out, 0.0)


def _type_impute(v, t) -> np.ndarray:
    """A missing value at its cell type's mean (the overall mean in a type
    with none measured)."""
    s_ = pd.Series(np.asarray(v, float))
    m = s_.groupby(np.asarray(t)).transform("mean")
    return np.nan_to_num(s_.fillna(m).fillna(s_.mean()).to_numpy(float))


def _total_split(y, t, Xo, avail, B, noise_v, perm_v, perm_sd,
                 nonlinear: bool = False) -> dict:
    """The variance of a per-cell statistic over ALL cells, split.

    Sampling noise (``noise_v``: the control's within-type variance); which
    loci were sampled (``perm_v``, the within-type variance left after
    reassigning contacts among a type's cells, less the noise); cell type
    and each covariate group, batch included (Shapley shares of the
    adjusted R^2 of one OLS on all of them -- batch effects are estimated
    with cell type in the model, so a batch that holds more of one type
    does not take that type's difference with it); and the rest: the
    variation between cells that none of them accounts for, the residual
    variance less ``perm_v`` -- with a one-sided p against the
    reassignment's spread ``perm_sd``.  Also cell type's R^2 alone and its
    unique part (with everything else in the model first).

    ``nonlinear``: how much of it any function of the same covariates
    could explain -- the cross-validated R^2 of gradient-boosted trees
    (curves, thresholds, interactions), beside a linear fit's on the same
    folds -- and what would be left then (tot_residual_nonlinear): an
    effect of labeling or depth that is not a straight line would
    otherwise pass for unexplained cell-to-cell variation."""
    from scipy.stats import norm
    n = y.size
    yc = y - y.mean()
    V = float(yc @ yc) / max(n - 1, 1)
    out = {"total_var": V}
    if not V > 0:
        return out
    Tm = pd.get_dummies(pd.Series(np.asarray(t).astype(str)),
                        drop_first=True).to_numpy(float)
    X = (np.column_stack([_type_impute(Xo[:, j], t)
                          for j in range(Xo.shape[1])])
         if Xo.shape[1] else np.empty((n, 0)))
    blocks = [m_ for m_ in (Tm, X, B) if m_.size]
    Z = np.column_stack(blocks) if blocks else np.empty((n, 0))
    if Z.size:
        Z = Z - Z.mean(0)
    groups, off = {}, Tm.shape[1]
    if Tm.shape[1]:
        groups["cell type"] = list(range(Tm.shape[1]))
    for gname, cols in COV_GROUPS:
        jj = [off + avail.index(c) for c in cols if c in avail]
        if jj:
            groups[gname] = jj
    off += X.shape[1]
    if B.size:
        groups["batch"] = list(range(off, off + B.shape[1]))
    r2a = _r2(yc, Z, 0)[1] if Z.size else 0.0
    r2a = max(r2a, 0.0) if np.isfinite(r2a) else 0.0
    if groups:
        phi, r2 = _shapley_r2(yc, Z, groups)
        f_ = r2a / r2 if np.isfinite(r2) and r2 > 0 else 0.0
        for g, v in phi.items():
            out[f"tot_{g}"] = v * f_ if np.isfinite(v) else np.nan
    if Tm.shape[1]:
        a_ = _r2(yc, Z[:, :Tm.shape[1]], 0)[1]
        rest = Z[:, Tm.shape[1]:]
        b_ = _r2(yc, rest, 0)[1] if rest.shape[1] else 0.0
        out["type_r2_alone"] = a_
        out["type_r2_unique"] = r2a - max(b_ if np.isfinite(b_) else 0, 0)
    out["tot_noise"] = min(noise_v / V, 1.0)
    base = noise_v
    if np.isfinite(perm_v):
        out["tot_loci"] = max(perm_v - noise_v, 0) / V
        base = max(perm_v, noise_v)
    resid = (1 - r2a) * V - base
    out["tot_residual"] = max(resid, 0) / V
    out["residual_var"] = resid
    if np.isfinite(perm_sd) and perm_sd > 0:
        out["residual_p"] = float(norm.sf(resid / perm_sd))
    if nonlinear and Z.size and n >= 50:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import KFold, cross_val_predict
        kf = KFold(5, shuffle=True, random_state=0)
        ss = float(yc @ yc)
        for nm, mdl in (("linear", LinearRegression()),
                        ("nonlinear", HistGradientBoostingRegressor(
                            max_iter=100, learning_rate=0.1,
                            min_samples_leaf=20, random_state=0))):
            pred = cross_val_predict(mdl, Z, yc, cv=kf)
            out[f"R2_cv_{nm}"] = float(1 - ((yc - pred) ** 2).sum() / ss)
        r2n = max(out["R2_cv_nonlinear"], out["R2_cv_linear"], r2a, 0.0)
        out["tot_residual_nonlinear"] = max((1 - r2n) * V - base, 0) / V
    return out


def _shapley_r2(y, Z, groups: dict):
    """The R^2 of an OLS of ``y`` on ``Z`` (both type-centred, no intercept)
    shared out among column groups by their Shapley values: each group's
    marginal R^2 averaged over every order the groups could enter in, so
    variance two groups explain together is split between them and the
    values add up to the full R^2.  Returns ({group: share}, full R^2)."""
    from math import factorial
    sd = Z.std(0) if Z.size else np.zeros(0)
    keep = np.flatnonzero(sd > 0)
    remap = {j: i for i, j in enumerate(keep)}
    Z = Z[:, keep] / sd[keep]
    cols = {g: np.array([remap[j] for j in jj if j in remap], np.int64)
            for g, jj in groups.items()}
    cols = {g: c for g, c in cols.items() if c.size}
    names = list(cols)
    m = len(names)
    yy = float(y @ y)
    if m == 0 or yy <= 0:
        return {g: np.nan for g in groups}, np.nan
    G, b = Z.T @ Z, Z.T @ y
    ridge = 1e-9 * max(float(np.trace(G)), 1.0) / G.shape[0]
    v = np.zeros(1 << m)
    for mask in range(1, 1 << m):
        idx = np.concatenate([cols[names[i]] for i in range(m)
                              if mask >> i & 1])
        Gs = G[np.ix_(idx, idx)] + ridge * np.eye(idx.size)
        try:
            beta = np.linalg.solve(Gs, b[idx])
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(Gs, b[idx], rcond=None)[0]
        v[mask] = float(b[idx] @ beta) / yy
    size = np.array([bin(x).count("1") for x in range(1 << m)])
    wt = np.array([factorial(k) * factorial(m - k - 1) / factorial(m)
                   for k in range(m)])
    masks = np.arange(1 << m)
    phi = {}
    for i, g in enumerate(names):
        bit = 1 << i
        wo = masks[(masks & bit) == 0]
        phi[g] = float(np.sum(wt[size[wo]] * (v[wo | bit] - v[wo])))
    return {g: phi.get(g, np.nan) for g in groups}, float(v[-1])


def _nearest(sites, g) -> np.ndarray:
    """Distance from each genome coordinate ``g`` to the nearest of the
    sorted genome coordinates ``sites`` (never across chromosomes: they are
    STRIDE apart)."""
    g = np.asarray(g, np.int64)
    if sites is None or sites.size == 0:
        return np.full(g.size, np.iinfo(np.int64).max, np.int64)
    i = np.searchsorted(sites, g)
    lo = sites[np.maximum(i - 1, 0)]
    hi = sites[np.minimum(i, sites.size - 1)]
    return np.minimum(np.abs(g - lo), np.abs(hi - g))


def open_site_coords(path=None, genes=None, cpg=None, exclude=()):
    """Sorted genome coordinates of sites open in (nearly) every cell, and
    what they are: the midpoints of the BED ``path`` (e.g. CTCF motifs in
    constitutive peaks), else the TSSs of ``genes`` that lie in a CpG
    island (``cpg``) -- CpG-island promoters are open and unmethylated in
    almost every cell type -- or every TSS without islands."""
    if path:
        df = pd.read_csv(path, sep="\t", header=None, comment="#",
                         compression="infer", dtype={0: str}, usecols=[0, 1, 2])
        chrom = df[0].astype(str).to_numpy()
        pos = ((df[1].to_numpy(np.int64) + df[2].to_numpy(np.int64)) // 2)
        label = Path(path).name
    elif genes is not None and len(genes):
        chrom = genes["chrom"].astype(str).to_numpy()
        pos = np.where(genes["strand"].to_numpy() == "-",
                       genes["end"].to_numpy(np.int64) - 1,
                       genes["start"].to_numpy(np.int64))
        label = "TSSs"
        if cpg is not None and len(cpg):
            iv = Intervals("cgi", ("cgi",), cpg["chrom"].astype(str),
                           cpg["start"], cpg["end"], np.zeros(len(cpg)))
            inside = iv.lookup(chrom, pos) >= 0
            chrom, pos = chrom[inside], pos[inside]
            label = "CpG-island TSSs"
    else:
        return None, None
    keep = np.array([_canon(c) not in exclude for c in chrom], bool)
    g = np.unique(_codes_of(chrom[keep]) * STRIDE + pos[keep])
    return g, label


def _count_pass(path, exclude, open_g=None, open_w: int = 200,
                distal_min: int = 5000) -> tuple:
    """One cell's raw methylation from a read-pairs file (k methylated of n
    sites per end): the sites methylated and the sites over all its ends --
    and, with ``open_g``, over its ends within ``open_w`` of an open site
    and its ends at least ``distal_min`` from every one (k, n, k_open,
    n_open, k_distal, n_distal)."""
    cols = ["chrom", "k1", "n1", "k2", "n2"]
    if open_g is not None:
        cols += ["pos1", "pos2"]
    df, names, ci, lut = _categorical(path, cols)
    ok = np.array([_canon(c) not in exclude for c in names] + [False])[ci]
    k = np.concatenate([df["k1"].to_numpy()[ok], df["k2"].to_numpy()[ok]]
                       ).astype(float)
    n = np.concatenate([df["n1"].to_numpy()[ok], df["n2"].to_numpy()[ok]]
                       ).astype(float)
    out = [float(k.sum()), float(n.sum())]
    if open_g is not None and open_g.size:
        base = lut[ci][ok] * STRIDE
        g = np.concatenate([base + df["pos1"].to_numpy(np.int64)[ok],
                            base + df["pos2"].to_numpy(np.int64)[ok]])
        dist = _nearest(open_g, g)
        for m in (dist <= open_w, dist >= distal_min):
            out += [float(k[m].sum()), float(n[m].sum())]
    return tuple(out)


def substructure_pcs(profiles: dict, types: dict, n_pcs: int = 5,
                     min_cells: int = 20, min_cover: float = 0.5
                     ) -> pd.DataFrame:
    """Within-type substructure: the cells' regional profiles, reduced.

    ``profiles``: {context: {cell: (bin keys, posterior sums, ends)}} -- per
    cell, its ends' mean posterior in each bin, less the cell's own mean (its
    level, a covariate of its own).  Within each cell type of at least
    ``min_cells`` cells: the bins observed in ``min_cover`` of its cells, of
    every context, each centred and scaled over the type's cells (a missing
    value at the type's mean), and the top ``n_pcs`` principal components,
    as z-scores within the type.  Subtypes, a cortical-area or layer
    gradient, or an animal, show as such components."""
    cells = sorted(set().union(*[set(p) for p in profiles.values()]))
    out = np.full((len(cells), n_pcs), np.nan)
    if not cells:
        return pd.DataFrame({"cell": cells})
    row = {c: i for i, c in enumerate(cells)}
    blocks = []
    for prof in profiles.values():
        if not prof:
            continue
        keys = np.unique(np.concatenate([v[0] for v in prof.values()]))
        M = np.full((len(cells), keys.size), np.nan, np.float32)
        for c, v in prof.items():
            k, sm, n = v[:3]
            if n.sum() == 0:
                continue
            M[row[c], np.searchsorted(keys, k)] = sm / n - sm.sum() / n.sum()
        blocks.append(M)
    M = np.concatenate(blocks, 1) if blocks else np.empty((len(cells), 0))
    t = np.array([types.get(c, "") for c in cells])
    for ty in np.unique(t):
        r = np.flatnonzero(t == ty)
        if r.size < min_cells:
            continue
        X = M[r]
        X = X[:, np.isfinite(X).mean(0) >= min_cover]
        if X.shape[1] <= n_pcs:
            continue
        with np.errstate(invalid="ignore"):
            mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
        X = np.nan_to_num((X - mu) / np.where(sd > 0, sd, 1))
        U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
        k = min(n_pcs, U.shape[1])
        sc = U[:, :k] * S[:k]
        sd_ = sc.std(0)
        out[r, :k] = (sc - sc.mean(0)) / np.where(sd_ > 0, sd_, 1)
    return pd.DataFrame({"cell": cells, **{f"sub_pc{i + 1}": out[:, i]
                                           for i in range(n_pcs)}})


def _profile_matrices(profiles: dict):
    """Cells x bins (every context side by side) of the profiles' posterior
    sums and end counts -- and of a random half of the ends, when the
    profiles carry one -- with the cells in row order."""
    cells = sorted(set().union(*[set(p) for p in profiles.values()]))
    row = {c: i for i, c in enumerate(cells)}
    blocks = []
    for prof in profiles.values():
        if not prof:
            continue
        keys = np.unique(np.concatenate([v[0] for v in prof.values()]))
        half = all(len(v) >= 5 for v in prof.values())
        mats = [np.zeros((len(cells), keys.size), np.float32)
                for _ in range(4 if half else 2)]
        for c, v in prof.items():
            j = np.searchsorted(keys, v[0])
            for m_, x in zip(mats, v[1:]):
                m_[row[c], j] = x
        blocks.append(mats if half else mats + [None, None])
    if not blocks:
        e = np.empty((len(cells), 0))
        return cells, e, e, None, None
    S = np.concatenate([b[0] for b in blocks], 1)
    N = np.concatenate([b[1] for b in blocks], 1)
    if all(b[2] is not None for b in blocks):
        return (cells, S, N, np.concatenate([b[2] for b in blocks], 1),
                np.concatenate([b[3] for b in blocks], 1))
    return cells, S, N, None, None


def doublet_scores(profiles: dict, types: dict = None, n_pcs: int = 30,
                   sim_ratio: float = 2.0, k: int = 0,
                   min_cover: float = 0.5, min_sep: float = 2.0,
                   seed: int = 0, sim_out: list = None) -> pd.DataFrame:
    """Scrublet's doublet score on the cells' regional profiles -- the way
    single-cell methylomes are screened (MethylScrublet, Liu et al. 2021).

    Doublets are simulated by merging a random half of one cell's ends with
    a random half of another's, bin by bin -- one cell's depth, so a
    simulated doublet is as noisy as a real cell (merged whole, it would be
    less noisy than any real cell and crowd every neighbourhood).  With
    ``types``, only from two types the profiles tell apart -- d' >=
    ``min_sep`` along the line between their centroids in the principal
    components of the real cells (the centroids' distance over the types'
    spread along that line; noise off the line does not count): a doublet
    of two types that look alike cannot be told from one cell, and
    simulating it would only crowd those types' neighbourhoods.  Real and simulated cells share the components (bins
    observed in ``min_cover`` of the real cells, each standardized over
    them); a cell's score is the share of simulated doublets among its
    ``k`` nearest neighbours (default sqrt(n) / 2).

    A doublet matters to concordance: centred on the pair's mean, each
    nucleus's contacts agree through the nuclei's difference in level, which
    the reshuffled partners do not, so a doublet of two levels reads as
    concordant."""
    from sklearn.neighbors import NearestNeighbors
    cells, S, N, Sh, Nh = _profile_matrices(profiles)
    n = len(cells)
    none = pd.DataFrame({"cell": cells, "doublet_score": np.nan})
    if n < 20 or S.shape[1] == 0:
        return none
    use = (N > 0).mean(0) >= min_cover
    S, N = S[:, use], N[:, use]
    if Sh is None:                                 # no halves: whole cells
        Sh, Nh = S, N
    else:
        Sh, Nh = Sh[:, use], Nh[:, use]
    with np.errstate(invalid="ignore", divide="ignore"):
        M = S / N
        mu, sd = np.nanmean(M, 0), np.nanstd(M, 0)
    sd = np.where(sd > 0, sd, 1)
    Z = np.nan_to_num((M - mu) / sd)
    c0 = Z.mean(0)
    _, _, Vt = np.linalg.svd(Z - c0, full_matrices=False)
    V = Vt[:min(n_pcs, Vt.shape[0])].T
    P = (Z - c0) @ V
    rng = np.random.default_rng(seed)
    n_sim = int(round(sim_ratio * n))
    if types is not None:
        t = np.array([str(types.get(c, "")) for c in cells])
        ut = np.unique(t)
        cen = np.array([P[t == x].mean(0) for x in ut])
        # a centroid's noise: two types that do not differ still lie
        # sum(var) / n apart, squared -- taken off before d' is formed
        cnoise = np.array([P[t == x].var(0).sum() / max((t == x).sum(), 1)
                           for x in ut])
        ok_pair = np.zeros((ut.size, ut.size), bool)
        for i in range(ut.size):
            for j in range(i + 1, ut.size):
                dv = cen[j] - cen[i]
                d2 = float(dv @ dv)
                if d2 == 0:
                    continue
                u = dv / np.sqrt(d2)
                vi = ((P[t == ut[i]] - cen[i]) @ u).var()
                vj = ((P[t == ut[j]] - cen[j]) @ u).var()
                dpr = (np.sqrt(max(d2 - cnoise[i] - cnoise[j], 0))
                       / np.sqrt(max(0.5 * (vi + vj), 1e-12)))
                ok_pair[i, j] = ok_pair[j, i] = dpr >= min_sep
        ti = np.searchsorted(ut, t)
        can = np.flatnonzero(ok_pair[ti].any(1))   # cells with a partner type
        if can.size == 0:
            logger.warning("doublet score: no two cell types are told apart "
                           "by their regional profiles; not scored")
            return none
        logger.info(f"doublet score: {int(ok_pair.sum() // 2)} of "
                    f"{ut.size * (ut.size - 1) // 2} type pairs told apart "
                    f"(d' >= {min_sep:g}) and simulated")
        a = can[rng.integers(0, can.size, n_sim)]
        b = np.empty(n_sim, np.int64)
        for x in np.unique(ti[a]):
            m = ti[a] == x
            other = np.flatnonzero(ok_pair[x][ti])
            b[m] = other[rng.integers(0, other.size, int(m.sum()))]
    else:
        a = rng.integers(0, n, n_sim)
        b = (a + rng.integers(1, n, n_sim)) % n   # never a cell with itself
    with np.errstate(invalid="ignore", divide="ignore"):
        Ms = (Sh[a] + Sh[b]) / (Nh[a] + Nh[b])
    Ps = (np.nan_to_num((Ms - mu) / sd) - c0) @ V
    k = k or max(5, int(round(0.5 * np.sqrt(n))))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(np.vstack([P, Ps]))
    _, ind = nn.kneighbors(P)
    q = (ind[:, 1:] >= n).mean(1)                     # the cell itself first
    if sim_out is not None:
        _, ind_s = nn.kneighbors(Ps)
        sim_out.append((ind_s[:, 1:] >= n).mean(1))
    return pd.DataFrame({"cell": cells, "doublet_score": q})


def locus_reassignment(cell, ctype, pair, band, parts: dict, n_band: int,
                       min_pairs: int, n_perm: int, rng, fg=None,
                       fgn=None) -> pd.DataFrame:
    """How much of the within-type variance of a per-cell statistic is a
    property of the CELLS, not of which loci each happened to sample?

    ``parts``: per contact, its contribution to its cell's ΔC and to the
    control (``d``, ``c``; a cell's value is their mean), optionally the
    same on the correlation scale (``r``, ``rc``: each over 2 sd sd of its
    own cell) and the co-open / co-closed mass of the real partners and of
    reshuffled draw 2 (``oo``, ``oc``, ``no2``, ``nc2``; with ``fg``/``fgn``,
    each cell's genome-wide co-open share, for ΔC x w and ΔC x sign(w)).
    Within each cell type, class pair and band, the contacts are reassigned
    to cells at random -- every cell keeping its number of contacts, each
    contact its parts -- ``n_perm`` times: the variance left is sampling
    noise plus locus sampling, with no cell differences (a reassigned cell
    is read against its type's mean co-open share).  Returned per statistic
    and (pair, band; band ``n_band`` = all), over the cells with >=
    ``min_pairs`` contacts: the within-type variance observed, of the
    control, and reassigned (mean, sd, permutation p), and the split
    noise_share + locus_share + cell_share = 1."""
    n_cells = ctype.size
    T = int(ctype.max()) + 1 if n_cells else 1
    ct = ctype[cell]
    stats = ["delta"] + (["delta_r"] if "r" in parts else [])
    # signed parts per contact (``sd``/``sc``, ``gd``/``gc``: a contact's
    # part of ΔC times its class pair's weight, and the control's): the
    # signed statistics are then means like ΔC
    direct = "sd" in parts and "sc" in parts
    signed = (not direct and fg is not None
              and all(k in parts for k in ("oo", "oc", "no2", "nc2")))
    if direct:
        stats += ["signed_rel"] + (["signed_sgn"] if "gd" in parts
                                   and "gc" in parts else [])
    if signed:
        stats += ["signed_rel", "signed_sgn"]
        fg, fgn = np.asarray(fg, float), np.asarray(fgn, float)
        fg_t = np.array([np.nanmean(fg[ctype == t])
                         if np.isfinite(fg[ctype == t]).any() else np.nan
                         for t in range(T)])
    moved = [k for k in ("d", "r", "oo", "oc", "sd", "gd") if k in parts]
    out = []
    for b_arr in (band, np.full(band.size, n_band)):
        grp = pair.astype(np.int64) * (n_band + 1) + b_arr
        g_ids, g = np.unique(grp, return_inverse=True)
        G = g_ids.size
        if G == 0:
            continue
        size = G * n_cells
        key = g * n_cells + cell
        N = np.bincount(key, minlength=size)
        el = np.flatnonzero(N >= min_pairs)
        if el.size == 0:
            continue
        gg, cc_ = el // n_cells, el % n_cells
        gt = gg * T + ctype[cc_]
        Ne = N[el].astype(float)

        def sums(w, key_=key):
            return np.bincount(key_, weights=w, minlength=size)[el]

        def values(S, ref, ctrl=False):
            m = S["c" if ctrl else "d"] / Ne
            v = {"delta": m}
            if "delta_r" in stats:
                v["delta_r"] = S["rc" if ctrl else "r"] / Ne
            if direct:
                v["signed_rel"] = S["sc" if ctrl else "sd"] / Ne
                if "signed_sgn" in stats:
                    v["signed_sgn"] = S["gc" if ctrl else "gd"] / Ne
            if signed:
                o_, c_ = ("no2", "nc2") if ctrl else ("oo", "oc")
                with np.errstate(invalid="ignore", divide="ignore"):
                    w = _weight(S[o_] / (S[o_] + S[c_]), ref)
                v["signed_rel"], v["signed_sgn"] = m * w, m * np.sign(w)
            return v

        def wvar(x):
            f = np.isfinite(x)
            x0 = np.where(f, x, 0.0)
            cnt = np.bincount(gt, weights=f.astype(float), minlength=G * T)
            mean = np.bincount(gt, weights=x0, minlength=G * T) / np.maximum(
                cnt, 1)
            nc = np.bincount(gg, weights=f.astype(float), minlength=G)
            nt = np.bincount(np.flatnonzero(cnt > 0) // T, minlength=G)
            ss = np.bincount(gg, weights=np.where(f, (x0 - mean[gt]) ** 2,
                                                  0.0), minlength=G)
            return ss / np.maximum(nc - nt, 1), nc

        S = {k: sums(w) for k, w in parts.items()}
        v_obs = values(S, fg[cc_] if signed else None)
        v_ctl = values(S, fgn[cc_] if signed else None, ctrl=True)
        var_o = {st: wvar(v_obs[st]) for st in stats}
        var_c = {st: wvar(v_ctl[st])[0] for st in stats}
        perm = {st: np.full((max(int(n_perm), 0), G), np.nan) for st in stats}
        if n_perm > 0:
            blk = g * T + ct
            o = np.argsort(blk, kind="stable")
            bs = blk[o]
            st_ = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
            en = np.r_[st_[1:], bs.size]
            multi = [(a_, e_) for a_, e_ in zip(st_, en) if e_ - a_ > 1]
            key_o = key[o]
            P = {k: parts[k][o] for k in moved}
            ref_p = fg_t[ctype[cc_]] if signed else None
            for b_ in range(int(n_perm)):
                pidx = np.arange(bs.size)
                for a_, e_ in multi:
                    pidx[a_:e_] = a_ + rng.permutation(e_ - a_)
                vp = values({k: sums(P[k][pidx], key_o) for k in moved},
                            ref_p)
                for st in stats:
                    perm[st][b_] = wvar(vp[st])[0]
        for st in stats:
            v_o, nc_ = var_o[st]
            for i in range(G):
                if nc_[i] < 3:
                    continue
                vo, vc = float(v_o[i]), float(var_c[st][i])
                pi = perm[st][:, i]
                vp = float(np.nanmean(pi)) if n_perm > 0 else np.nan
                row = {"statistic": st,
                       "pair": int(g_ids[i] // (n_band + 1)),
                       "band_index": int(g_ids[i] % (n_band + 1)),
                       "cells": int(nc_[i]), "var_obs": vo, "var_ctrl": vc,
                       "var_perm": vp,
                       "var_perm_sd": (float(np.nanstd(pi, ddof=1))
                                       if n_perm > 1 else np.nan),
                       "perm_p": ((1 + int((pi >= vo).sum())) / (n_perm + 1)
                                  if n_perm > 0 else np.nan)}
                if vo > 0 and np.isfinite(vp):
                    row["noise_share"] = min(vc / vo, 1.0)
                    row["locus_share"] = float(np.clip(
                        (vp - vc) / vo, 0, 1 - row["noise_share"]))
                    row["cell_share"] = float(np.clip(1 - vp / vo, 0, 1))
                out.append(row)
    return pd.DataFrame(out)


def compartment_strength(by_cell, cell_type, maps: dict, res: int,
                         min_dist: int, min_contacts: int = 20):
    """Per cell, how strongly its long-range contacts (>= ``min_dist``)
    join same-compartment loci: the share that are A-A or B-B (its type's
    A/B), minus the share expected if its A and B ends paired at random,
    f_A^2 + f_B^2.  NaN under ``min_contacts`` annotated contacts."""
    out = np.full(len(by_cell), np.nan)
    mb = max(1, int(np.ceil(min_dist / res)))
    half = res // 2
    for i, ((cc, b1, b2, _), t) in enumerate(zip(by_cell, cell_type)):
        sel = (b2 - b1) >= mb
        if sel.sum() < min_contacts:
            continue
        mp = maps.get(t, maps["__all__"])
        base = cc[sel] * STRIDE
        k1 = mp.lookup_g(base + b1[sel].astype(np.int64) * res + half)
        k2 = mp.lookup_g(base + b2[sel].astype(np.int64) * res + half)
        ok = (k1 >= 0) & (k2 >= 0)
        n = int(ok.sum())
        if n < min_contacts:
            continue
        k1, k2 = k1[ok], k2[ok]
        same = float(((k1 == k2)).sum()) / n
        fa = float((k1 == 0).sum() + (k2 == 0).sum()) / (2 * n)
        out[i] = same - (fa ** 2 + (1 - fa) ** 2)
    return out


def covariate_association(percell: pd.DataFrame, cov: pd.DataFrame,
                          ctx_cov: dict, locus: pd.DataFrame = None,
                          batch_cols=(), stats=COV_STATS,
                          min_cells: int = 30) -> pd.DataFrame:
    """What explains the WITHIN-cell-type variation of a class pair's
    per-cell concordance?

    Per context x system x class pair x distance band x statistic, over the
    cells with enough contacts:

    - each covariate's within-type Spearman rho (ranks with each type's
      mean rank removed, over the cells it is measured in; p from t on
      n - types - 1 df; BH q over the table's single covariates);
    - all covariates together, batch alone and both: the adjusted R^2 of a
      no-intercept OLS on type-centred values (R2_adj, R2_adj_batch,
      R2_adj_all; a cell missing a covariate is put at its type's mean),
      and per batch column an F test (batch_<col>_p);
    - the same full fit on the control, which has nothing to explain
      (control_R2_adj: well above 0 means a covariate tracks noise);
    - for same-state pairs over all distances, R2_adj_all shared out among
      the covariate groups (COV_GROUPS, and batch) by their Shapley values
      -- each group's added R^2 averaged over every order the groups could
      enter in, so what two groups explain together is split between them
      -- as shap_<group>, shares of the total within-type variance that add
      up to R2_adj_all.

    'scale' is the cell's two class SDs multiplied: C is a covariance, so a
    cell whose posteriors vary more in these classes has a larger ΔC at the
    same coordination.

    R^2 is of the TOTAL within-type variance.  bio_share = 1 -
    Var_w(control) / Var_w(value) is the part beyond sampling noise, and
    explained_share = R2_adj_all / bio_share the share of it the
    covariates (with batch) explain.  ``locus`` (:func:`locus_reassignment`)
    also takes out what merely follows which loci a cell sampled:
    cell_share is the part that is a property of the cells, and
    explained_cell_share = R2_adj_all / cell_share.

    Read the bands as well as 'all'.  ΔC changes with distance -- GCH's
    excess rises ~10x from 1 to 100 kb -- so a cell's all-distance value
    follows its own mix of contact distances, and % near and % mitotic ARE
    that mix: they can correlate with it with no cell cycle involved.
    Within one decade band the mix barely varies; a covariate that holds
    there is not the mix."""
    from scipy.stats import f as f_dist
    from scipy.stats import t as t_dist
    pc = percell[percell["enough_pairs"]]
    # the pair's share of the cell's contacts (of the system, in the band)
    tot_pairs = None
    if "pairs" in percell:
        pr = percell[percell["anchor"] != ALL_REGIONS]
        tot_pairs = pr.groupby(["context", "system", "band", "cell"])[
            "pairs"].sum()
    # the other context's genome-wide level
    other = {}
    for c_ in ctx_cov:
        o_ = [x for x in ctx_cov if x != c_]
        if o_:
            other[c_] = ctx_cov[o_[0]][["cell", "level"]].rename(
                columns={"level": "level_other"})
    lk = None
    if locus is not None and not locus.empty:
        if "statistic" not in locus:
            locus = locus.assign(statistic="delta")
        lk = locus.set_index(["context", "system", "anchor", "partner",
                              "band", "statistic"])
    rows = []
    for (ctx, sy, k1, k2, bl), g in pc.groupby(
            ["context", "system", "anchor", "partner", "band"], sort=False):
        d = g.merge(cov.drop(columns=["cell_type"], errors="ignore"),
                    on="cell", how="left")
        cx = ctx_cov.get(ctx)
        if cx is not None:
            d = d.merge(cx, on="cell", how="left")
        if ctx in other:
            d = d.merge(other[ctx], on="cell", how="left")
        if "sd_anchor" in d and "sd_partner" in d:
            d["scale"] = d["sd_anchor"] * d["sd_partner"]
        if "mu_anchor" in d:
            d["mu_pair_1"] = d["mu_anchor"]
            d["mu_pair_2"] = d["mu_partner"] if k1 != k2 else np.nan
        if tot_pairs is not None:
            try:
                tp = tot_pairs.loc[(ctx, sy, bl)]
                d["pair_share"] = (d["pairs"].to_numpy(float)
                                   / d["cell"].map(tp).to_numpy(float))
            except KeyError:
                pass
        avail = [c for c in COVARIATES if c in d
                 and d[c].notna().sum() >= min_cells and d[c].nunique() > 1]
        X0 = d[avail].to_numpy(float) if avail else np.empty((len(d), 0))
        bcols = [b for b in batch_cols if b in d and d[b].nunique() > 1]
        for stat in stats:
            if stat not in d:
                continue
            y0 = d[stat].to_numpy(float)
            c0 = d[_CONTROL_OF[stat]].to_numpy(float)
            ok = np.isfinite(y0) & np.isfinite(c0)
            if ok.sum() < min_cells:
                continue
            t = d["cell_type"].to_numpy()[ok]
            n, n_t = int(ok.sum()), len(set(t))
            y, c = _center(y0[ok], t), _center(c0[ok], t)
            Xo = X0[ok]
            X = (np.column_stack([_center_obs(Xo[:, j], t)
                                  for j in range(Xo.shape[1])])
                 if Xo.shape[1] else np.empty((n, 0)))
            Bs = {}
            for b in bcols:
                dm = pd.get_dummies(d.loc[ok, b].astype(str), drop_first=True)
                if dm.shape[1]:
                    Bs[b] = np.column_stack([_center(dm[col].to_numpy(float),
                                                     t)
                                             for col in dm.columns])
            Ball = (np.column_stack(list(Bs.values())) if Bs
                    else np.empty((n, 0)))
            row = {"context": ctx, "system": sy, "anchor": k1, "partner": k2,
                   "band": bl, "statistic": stat, "cells": n}
            for cname in COVARIATES:
                rho = p_ = np.nan
                if cname in avail:
                    xv = Xo[:, avail.index(cname)]
                    f_ = np.isfinite(xv)
                    if f_.sum() >= min_cells:
                        rx = _center(pd.Series(xv[f_]).rank().to_numpy(), t[f_])
                        ry = _center(pd.Series(y0[ok][f_]).rank().to_numpy(),
                                     t[f_])
                        if rx.std() > 0 and ry.std() > 0:
                            rho = float(np.corrcoef(rx, ry)[0, 1])
                            df_ = max(int(f_.sum()) - len(set(t[f_])) - 1, 1)
                            tt = rho * np.sqrt(df_ / max(1 - rho ** 2, 1e-12))
                            p_ = float(2 * t_dist.sf(abs(tt), df_))
                row[f"rho_{cname}"], row[f"p_{cname}"] = rho, p_
            row["R2_adj"] = _r2(y, X, n_t)[1]
            row["R2_adj_batch"] = _r2(y, Ball, n_t)[1]
            XB = np.column_stack([X, Ball]) if Ball.size else X
            row["R2_adj_all"] = _r2(y, XB, n_t)[1]
            row["control_R2_adj"] = _r2(c, XB, n_t)[1]
            for b, Bm in Bs.items():
                r2b, _, kb, dfr = _r2(y, Bm, n_t)
                row[f"batch_{b}_p"] = (
                    float(f_dist.sf((r2b / kb) / ((1 - r2b) / dfr), kb, dfr))
                    if np.isfinite(r2b) and r2b < 1 else np.nan)
            vy = (y ** 2).sum() / max(n - n_t, 1)
            vc = (c ** 2).sum() / max(n - n_t, 1)
            row["bio_share"] = float(1 - vc / vy) if vy > 0 else np.nan
            r2a = row["R2_adj_all"]
            if np.isfinite(r2a) and row["bio_share"] > 0:
                row["explained_share"] = float(np.clip(max(r2a, 0)
                                                       / row["bio_share"],
                                                       0, 1))
            if lk is not None and (ctx, sy, k1, k2, bl, stat) in lk.index:
                L = lk.loc[(ctx, sy, k1, k2, bl, stat)]
                for col in ("noise_share", "locus_share", "cell_share",
                            "perm_p", "var_obs", "var_perm", "var_perm_sd",
                            "var_ctrl"):
                    row[col] = float(L[col]) if col in L else np.nan
                cs = row.get("cell_share", np.nan)
                if np.isfinite(r2a) and np.isfinite(cs) and cs > 0:
                    row["explained_cell_share"] = float(
                        np.clip(max(r2a, 0) / cs, 0, 1))
            # the splits for the all-distance band, which the figures show
            # (2^groups regressions each)
            whole = k1 == k2 and str(bl).startswith("all")
            if whole and np.isfinite(r2a):
                groups = {}
                for gname, cols in COV_GROUPS:
                    jj = [avail.index(x) for x in cols if x in avail]
                    if jj:
                        groups[gname] = jj
                if Ball.size:
                    groups["batch"] = list(range(X.shape[1],
                                                 X.shape[1] + Ball.shape[1]))
                if groups:
                    phi, r2 = _shapley_r2(y, XB, groups)
                    f_ = max(r2a, 0) / r2 if np.isfinite(r2) and r2 > 0 else 0
                    for gname, v in phi.items():
                        row[f"shap_{gname}"] = (v * f_ if np.isfinite(v)
                                                else np.nan)
            if whole:
                # over all cells, cell type a component of its own
                Bt = [pd.get_dummies(d.loc[ok, b].astype(str),
                                     drop_first=True).to_numpy(float)
                      for b in bcols]
                Bt = (np.column_stack(Bt) if Bt and sum(x.shape[1] for x in Bt)
                      else np.empty((n, 0)))
                row.update(_total_split(
                    y0[ok], t, Xo, avail, Bt, vc,
                    row.get("var_perm", np.nan),
                    row.get("var_perm_sd", np.nan),
                    nonlinear=stat in ("delta", "signed_rel")))
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        pcols = [f"p_{c}" for c in HEATMAP_COVARIATES]
        q = fdr_bh(out[pcols].to_numpy(float).ravel()).reshape(
            len(out), len(pcols))
        for j, c in enumerate(HEATMAP_COVARIATES):
            out[f"q_{c}"] = q[:, j]
        if "residual_p" in out:
            out["residual_q"] = fdr_bh(out["residual_p"].to_numpy(float))
    return out


def cross_class_correlation(percell: pd.DataFrame, summary: pd.DataFrame,
                            min_cells: int = 30,
                            min_reliability: float = 0.2,
                            stat: str = "delta", cov: pd.DataFrame = None,
                            ctx_cov: dict = None,
                            batch_cols=()) -> pd.DataFrame:
    """Is the within-type variation one per-cell factor or region-specific?

    Per context and system, between DISJOINT same-state class pairs (A-A
    against B-B, early-early against late-late, the chromHMM states): the
    correlation across cells of their per-cell ΔC (all distances, each
    type's mean removed), and the same corrected for sampling noise with
    each pair's split-half reliability (Spearman-Brown, 2r / (1 + r)):
    r_disattenuated = r / sqrt(rel_1 rel_2).  Near 1: the classes move
    together -- one per-cell factor, technical or a nuclear state.  Near 0:
    each class varies on its own.  Not corrected (NaN) when a reliability
    is under ``min_reliability``: dividing by a near-zero reliability gives
    anything.  Classes of different systems share contacts, so they are not
    compared.

    With ``cov``, also r_resid: the same correlation of what is left in
    each class after cell type, every covariate and batch are regressed out
    (:func:`_residualize`).  If the classes still move together, what is
    left is one per-cell factor none of them measures -- a nucleus-wide
    state, or a technical factor not yet in the model; if not, what is left
    is particular to regions."""
    all_b = next((b for b in dict.fromkeys(percell["band"])
                  if str(b).startswith("all")), None)
    pc = percell[(percell["band"] == all_b) & percell["enough_pairs"]
                 & (percell["anchor"] == percell["partner"])
                 & (percell["anchor"] != ALL_REGIONS)]
    rel = {}
    s_all = summary[summary["band"] == all_b]
    hcol = "split_half_r" if stat == "delta" else f"{stat}_split_half_r"
    resid = (_residualize(pc, stat, cov, ctx_cov, batch_cols,
                          keep_type=False) if cov is not None else None)
    for _, r in s_all[s_all["anchor"] == s_all["partner"]].iterrows():
        h = r.get(hcol, np.nan)
        rel[(r["context"], r["system"], r["anchor"])] = (
            2 * h / (1 + h) if np.isfinite(h) and h > 0 else np.nan)
    rows = []
    for (ctx, sy), g in pc.groupby(["context", "system"], sort=False):
        W = g.pivot_table(index="cell", columns="anchor", values=stat)
        t = (g.drop_duplicates("cell").set_index("cell")["cell_type"]
             .reindex(W.index))
        Wc = W - W.groupby(t).transform("mean")
        states = list(dict.fromkeys(g["anchor"]))
        for i, a in enumerate(states):
            for b in states[i + 1:]:
                both = Wc[a].notna() & Wc[b].notna()
                if both.sum() < min_cells:
                    continue
                r_ = float(np.corrcoef(Wc.loc[both, a], Wc.loc[both, b])[0, 1])
                ra, rb = rel.get((ctx, sy, a), np.nan), rel.get((ctx, sy, b),
                                                                 np.nan)
                rr = np.nan
                if resid is not None and (ctx, sy, a) in resid and \
                        (ctx, sy, b) in resid:
                    xa, xb = resid[(ctx, sy, a)], resid[(ctx, sy, b)]
                    m_ = xa.notna() & xb.notna()
                    if m_.sum() >= min_cells and xa[m_].std() > 0 \
                            and xb[m_].std() > 0:
                        rr = float(np.corrcoef(xa[m_], xb[m_])[0, 1])
                rows.append({"context": ctx, "system": sy, "statistic": stat,
                             "state_1": a, "r_resid": rr,
                             "state_2": b, "cells": int(both.sum()), "r": r_,
                             "reliability_1": ra, "reliability_2": rb,
                             "r_disattenuated": (
                                 r_ / np.sqrt(ra * rb)
                                 if ra >= min_reliability
                                 and rb >= min_reliability else np.nan)})
    return pd.DataFrame(rows)


# ------------------------------------------------- cell-type prediction --

#: The feature sets each region group predicts cell type from.
AUC_VARIANTS = ("concordance", "concordance, technical removed", "control",
                "marginal", "marginal + concordance", "marginal + control")


def _auc(pos, score) -> float:
    """ROC AUC from ranks (Mann-Whitney U), ties shared."""
    from scipy.stats import rankdata
    pos = np.asarray(pos, bool)
    n1 = int(pos.sum())
    n0 = pos.size - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _cv_type_auc(X, y, n_folds: int, seed: int, n_boot: int = 200,
                 n_trees: int = 300):
    """Cell type from ``X``: out-of-fold class probabilities of a random
    forest (300 trees, max_features sqrt, as coordination_eval.cv_rf;
    leaves of >= 5 cells), stratified ``n_folds``-fold.  Returns the macro
    one-vs-rest AUC, its 95% bootstrap interval over cells, and every
    type's own AUC -- or None with under two types."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    labels, yy = np.unique(np.asarray(y).astype(str), return_inverse=True)
    if labels.size < 2:
        return None
    folds = min(n_folds, int(np.bincount(yy).min()))
    if folds < 2:
        return None
    P = np.zeros((yy.size, labels.size))
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=seed).split(X, yy):
        clf = RandomForestClassifier(n_estimators=n_trees,
                                     max_features="sqrt", min_samples_leaf=5,
                                     n_jobs=1, random_state=seed)
        clf.fit(X[tr], yy[tr])
        P[np.ix_(te, clf.classes_)] = clf.predict_proba(X[te])
    per = np.array([_auc(yy == j, P[:, j]) for j in range(labels.size)])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, yy.size, yy.size)
        boot.append(np.nanmean([_auc(yy[i] == j, P[i, j])
                                for j in range(labels.size)]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(per.mean()), float(lo), float(hi), dict(zip(labels, per))


def _residualize(pc: pd.DataFrame, stat: str, cov: pd.DataFrame,
                 ctx_cov: dict, batch_cols=(), groups=None,
                 keep_type: bool = True) -> pd.DataFrame:
    """Per region group (context, system, class), the per-cell statistic
    with covariate ``groups`` (all of COV_GROUPS by default) and batch
    regressed out of it, in one OLS with cell type in the model (a missing
    covariate at its type's mean).  ``keep_type``: only the covariates'
    part is taken out -- as ComBat does with cell type as a covariate, so
    a difference between types they do not account for stays; otherwise
    cell type's part goes too, leaving what nothing measured accounts
    for.  Cells x (context, system, class)."""
    want = [c for g, cols in COV_GROUPS
            if groups is None or g in groups for c in cols]
    use_batch = groups is None or "batch" in groups
    out = {}
    for key, g in pc.groupby(["context", "system", "anchor"], sort=False):
        d = g.merge(cov.drop(columns=["cell_type"], errors="ignore"),
                    on="cell", how="left")
        cx = (ctx_cov or {}).get(key[0])
        if cx is not None:
            d = d.merge(cx, on="cell", how="left")
        o_ = [x for x in (ctx_cov or {}) if x != key[0]]
        if o_:
            d = d.merge(ctx_cov[o_[0]][["cell", "level"]].rename(
                columns={"level": "level_other"}), on="cell", how="left")
        if "sd_anchor" in d and "sd_partner" in d:
            d["scale"] = d["sd_anchor"] * d["sd_partner"]
        if "mu_anchor" in d:
            d["mu_pair_1"] = d["mu_anchor"]
        y = d[stat].to_numpy(float)
        ok = np.isfinite(y)
        t = d["cell_type"].astype(str).to_numpy()
        res = np.full(y.size, np.nan)
        av = [c for c in want if c in d and d[c].notna().sum() >= 10
              and d[c].nunique() > 1]
        Z = [_type_impute(d[c].to_numpy(float)[ok], t[ok]) for c in av]
        if use_batch:
            for b in batch_cols:
                if b in d and d[b].nunique() > 1:
                    Z += list(pd.get_dummies(d.loc[ok, b].astype(str),
                                             drop_first=True).to_numpy(
                        float).T)
        yo = y[ok] - y[ok].mean()
        Tm = pd.get_dummies(pd.Series(t[ok]), drop_first=True).to_numpy(float)
        Tm = Tm - Tm.mean(0) if Tm.size else Tm
        Z = np.column_stack(Z) - np.column_stack(Z).mean(0) if Z else None
        if Z is not None and ok.sum() > Z.shape[1] + Tm.shape[1] + 10:
            M = np.column_stack([Tm, Z]) if Tm.size else Z
            beta = np.linalg.lstsq(M, yo, rcond=None)[0]
            fit = Z @ beta[Tm.shape[1]:] + (
                0 if keep_type or not Tm.size else Tm @ beta[:Tm.shape[1]])
            res[ok] = (y[ok] if keep_type else yo) - fit
        elif not keep_type and Tm.size:
            beta = np.linalg.lstsq(Tm, yo, rcond=None)[0]
            res[ok] = yo - Tm @ beta
        else:
            res[ok] = y[ok] if keep_type else yo
        out[key] = pd.Series(res, index=d["cell"].to_numpy())
    return pd.DataFrame(out)


def _remove_technical(pc: pd.DataFrame, stat: str, cov: pd.DataFrame,
                      ctx_cov: dict, batch_cols=()) -> pd.DataFrame:
    """The statistic with the technical groups (TECHNICAL_GROUPS: depth and
    call quality, batch, labeling efficiency, scale, doublets, bisulfite
    conversion) regressed out, cell type protected (:func:`_residualize`)."""
    return _residualize(pc, stat, cov, ctx_cov, batch_cols,
                        groups=TECHNICAL_GROUPS, keep_type=True)


def celltype_auc(percell: pd.DataFrame, stat: str = "signed_rel",
                 n_folds: int = 5, seed: int = 0, n_threads: int = 1,
                 min_type_cells: int = 10, n_trees: int = 300,
                 cov: pd.DataFrame = None, ctx_cov: dict = None,
                 batch_cols=()):
    """How well does a region group's per-cell concordance predict cell
    type?

    Per context -- and the contexts together -- over all distances, the
    same-state class pairs ("region groups": A-A, early-early, promoter-
    promoter, ...): each group alone, each system's groups together and all
    groups together, as features of a cross-validated random forest
    (:func:`_cv_type_auc`), five ways --

      concordance             the statistic ``stat`` itself
      concordance, technical  the same with the technical groups and batch
        removed               regressed out, cell type protected
                              (:func:`_remove_technical`; needs ``cov``)
      control                 the same on reshuffled links alone: an AUC
                              here is cell-type information the statistic
                              carries without any coordination
      marginal                the cell's mean posterior in the class
      marginal + concordance  against marginal alone: what the
                              concordance adds to the level
      marginal + control      the same for the control

    The same cells in all five: those with enough contacts in the group
    (with several groups, in any of them, a missing value at the column's
    median); types with fewer than ``min_type_cells`` of them are left out.
    Returns (macro AUC table, per-type AUC table)."""
    from joblib import Parallel, delayed
    all_b = next((b for b in dict.fromkeys(percell["band"])
                  if str(b).startswith("all")), None)
    pc = percell[(percell["band"] == all_b) & percell["enough_pairs"]
                 & (percell["anchor"] == percell["partner"])]
    ctrl = _CONTROL_OF[stat]
    if pc.empty or stat not in pc or ctrl not in pc:
        return pd.DataFrame(), pd.DataFrame()
    W = {m: pc.pivot_table(index="cell", values=m, aggfunc="first",
                           columns=["context", "system", "anchor"])
         for m in (stat, ctrl, "mu_anchor")}
    if cov is not None:
        W["corrected"] = _remove_technical(pc, stat, cov, ctx_cov, batch_cols)
    types = pc.drop_duplicates("cell").set_index("cell")["cell_type"].astype(
        str)
    contexts = list(dict.fromkeys(pc["context"]))
    ctx_sets = [(c, [c]) for c in contexts]
    if len(contexts) > 1:
        ctx_sets.append((" + ".join(contexts), contexts))
    feats = {"concordance": (stat,),
             "concordance, technical removed": ("corrected",),
             "control": (ctrl,),
             "marginal": ("mu_anchor",),
             "marginal + concordance": ("mu_anchor", stat),
             "marginal + control": ("mu_anchor", ctrl)}
    tasks = []
    for ctx_name, ctxs in ctx_sets:
        cols_ctx = [c for c in W[stat].columns if c[0] in ctxs]
        allr = [c for c in cols_ctx if c[2] == ALL_REGIONS]
        cols_ctx = [c for c in cols_ctx if c[2] != ALL_REGIONS]
        groups = list(dict.fromkeys((sy, k) for _, sy, k in cols_ctx))
        sets = []
        if allr:
            # all regions once: the system with the most cells
            best = max(allr, key=lambda c: int(W[stat][c].notna().sum()))
            sets.append(("all", ALL_REGIONS, [(best[1], ALL_REGIONS)]))
            cols_ctx = cols_ctx + [c for c in allr if c[1] == best[1]]
        sets += [(sy, k, [(sy, k)]) for sy, k in groups]
        systems_ = list(dict.fromkeys(sy for sy, _ in groups))
        for sy in systems_:
            gs = [g for g in groups if g[0] == sy]
            if len(gs) > 1:
                sets.append((sy, "all classes", gs))
        if len(systems_) > 1:
            sets.append(("all", "all classes", groups))
        for sy, name, gs in sets:
            cols = [c for c in cols_ctx if (c[1], c[2]) in gs
                    and (c[2] != ALL_REGIONS or name == ALL_REGIONS)]
            M = W[stat].reindex(columns=cols)
            cells = M.index[M.notna().any(axis=1).to_numpy()]
            y = types.reindex(cells)
            cnt = y.value_counts()
            cells = cells[y.isin(cnt.index[cnt >= min_type_cells]).to_numpy()]
            y = types.reindex(cells).to_numpy()
            if len(set(y)) < 2:
                continue
            for variant in AUC_VARIANTS:
                if any(m not in W for m in feats[variant]):
                    continue
                X = np.column_stack([W[m].reindex(index=cells, columns=cols)
                                     .to_numpy(float) for m in feats[variant]])
                with np.errstate(all="ignore"):
                    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), 0)
                X = np.where(np.isfinite(X), X, np.nan_to_num(med)[None, :])
                tasks.append(((ctx_name, sy, name, variant, len(cells),
                               len(set(y))), X, y))
    # processes: a forest's per-tree Python work holds the GIL
    res = Parallel(n_jobs=n_threads)(
        delayed(_cv_type_auc)(X, y, n_folds, seed, n_trees=n_trees)
        for _, X, y in tasks)
    rows, per_rows = [], []
    for (key, _, _), r in zip(tasks, res):
        if r is None:
            continue
        ctx_name, sy, name, variant, n, nt = key
        base = {"context": ctx_name, "system": sy, "group": name,
                "features": variant, "statistic": stat}
        rows.append({**base, "cells": n, "types": nt, "auc": r[0],
                     "auc_lo": r[1], "auc_hi": r[2]})
        per_rows += [{**base, "cell_type": t, "auc": a}
                     for t, a in r[3].items()]
    return pd.DataFrame(rows), pd.DataFrame(per_rows)


def _auc_rows(res: pd.DataFrame) -> list:
    """Row order of the AUC figures: system by system, its groups then all
    its classes; all groups last."""
    keys = list(dict.fromkeys(zip(res["system"], res["group"])))
    order = [x for x in SYSTEMS if any(k[0] == x for k in keys)] + [
        x for x in dict.fromkeys(k[0] for k in keys)
        if x not in SYSTEMS and x != "all"]
    out = [k for k in keys if k[1] == ALL_REGIONS]
    for sy in order:
        out += [k for k in keys if k[0] == sy and k[1] != "all classes"]
        out += [k for k in keys if k[0] == sy and k[1] == "all classes"]
    return out + [k for k in keys if k[0] == "all" and k[1] != ALL_REGIONS]


def plot_celltype_auc(res: pd.DataFrame, per: pd.DataFrame, stat: str, path):
    """a: per context, the cross-validated macro AUC of cell type from each
    region group's concordance (filled, 95% CI), its control (open), the
    marginal (grey square) and marginal + concordance (black diamond); b:
    the concordance's AUC for every cell type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    if res.empty:
        return
    contexts = list(dict.fromkeys(res["context"]))
    rows = _auc_rows(res)
    ny = len(rows)
    label = [("all regions (pooled contacts)" if g == ALL_REGIONS else
              f"{SYSTEM_LABEL.get(sy, sy)}: {g}" if sy != "all"
              else "all region groups") for sy, g in rows]
    types = sorted(per["cell_type"].astype(str).unique())
    style = {"concordance": dict(marker="o", color="#2b7bba", ms=3.5),
             "concordance, technical removed": dict(marker="^",
                                                    color="#e6550d", ms=3.5),
             "control": dict(marker="o", mfc="none", color="0.45", ms=3.5),
             "marginal": dict(marker="s", color="0.7", ms=3),
             "marginal + concordance": dict(marker="D", color="0.1", ms=2.8),
             "marginal + control": dict(marker="D", mfc="none", color="0.45",
                                        ms=2.8)}
    with plt.rc_context(figure_style.RC):
        fig = plt.figure(figsize=(figure_style.FIG_W_IN,
                                  1.2 + 0.32 * ny))
        gs = fig.add_gridspec(2, len(contexts), height_ratios=[1, 1],
                              hspace=0.18, wspace=0.08, top=0.94,
                              bottom=0.07)
        x_lo = min(0.4, float(np.nanmin(res["auc"])) - 0.03)
        hm_axes = []
        for j, ctx in enumerate(contexts):
            ax = fig.add_subplot(gs[0, j])
            r = res[res["context"] == ctx].set_index(["system", "group",
                                                      "features"])
            for k, v in enumerate(AUC_VARIANTS):
                dy = (k - 2.5) * 0.11
                for i, key in enumerate(rows):
                    if key + (v,) not in r.index:
                        continue
                    x = r.loc[key + (v,)]
                    if v in ("concordance", "concordance, technical removed",
                             "marginal + concordance"):
                        ax.plot([x["auc_lo"], x["auc_hi"]], [i + dy] * 2,
                                color=style[v]["color"], lw=0.6)
                    ax.plot([x["auc"]], [i + dy], ls="none", label=v,
                            **style[v])
            ax.axvline(0.5, color="0.5", lw=0.5, ls="--")
            ax.set_xlim(x_lo, 1.0)
            ax.set_xlabel("macro AUC (one-vs-rest)", fontsize=5.5)
            ax.set_ylim(ny - 0.5, -0.5)
            ax.set_yticks(range(ny))
            ax.set_yticklabels(label if j == 0 else [], fontsize=5)
            for i, (sy, g) in enumerate(rows):
                if g == "all classes" or sy == "all":
                    ax.axhspan(i - 0.5, i + 0.5, color="0.93", zorder=0)
            ax.set_title(f"{'a' if j == 0 else ''}  {ctx}", loc="left",
                         fontsize=6.5)
            if j == len(contexts) - 1:
                h, lab = ax.get_legend_handles_labels()
                uniq = dict(zip(lab, h))
                ax.legend(uniq.values(), uniq.keys(), fontsize=4.5,
                          loc="upper left", bbox_to_anchor=(1.02, 1.0),
                          borderaxespad=0)
            ax2 = fig.add_subplot(gs[1, j])
            p_ = per[(per["context"] == ctx)
                     & (per["features"] == "concordance")]
            H = (p_.set_index(["system", "group", "cell_type"])["auc"]
                 .unstack("cell_type").reindex(index=pd.MultiIndex.from_tuples(
                     rows), columns=types))
            im = ax2.imshow(H.to_numpy(float), cmap="magma_r", vmin=0.5,
                            vmax=1.0, aspect="auto")
            hm_axes.append(ax2)
            for i in range(ny):
                for k_, t in enumerate(types):
                    v_ = H.iat[i, k_]
                    if np.isfinite(v_):
                        ax2.text(k_, i, f"{v_:.2f}"[1:], ha="center",
                                 va="center", fontsize=3.5,
                                 color="w" if v_ > 0.8 else "k")
            ax2.set_yticks(range(ny))
            ax2.set_yticklabels(label if j == 0 else [], fontsize=5)
            ax2.set_xticks(range(len(types)))
            ax2.set_xticklabels(types, rotation=60, ha="right", fontsize=5)
            if j == 0:
                ax2.set_title("b  each cell type against the rest "
                              "(concordance)", loc="left", fontsize=6.5)
        cb = fig.colorbar(im, ax=hm_axes, fraction=0.03, pad=0.02)
        cb.ax.tick_params(labelsize=5)
        cb.set_label("AUC", fontsize=6)
        fig.suptitle(f"How well {_STAT_LABEL.get(stat, stat)} in each region "
                     f"group predicts cell type (random forest, cross-"
                     f"validated)", fontsize=7, y=0.985)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


# ------------------------------------------------------- groups of cells --

#: The statistics groups of cells are compared on, with their controls.
CONTRAST_STATS = {"delta": "control", "delta_r": "control_r",
                  "signed_rel": "signed_rel_control",
                  "signed_sgn": "signed_sgn_control"}
#: The all-regions statistic whose ratio between two groups is a
#: statistic's overall scale -- the signed ones scale with ΔC.
CONTRAST_SCALE = {"delta": "delta", "delta_r": "delta_r",
                  "signed_rel": "delta", "signed_sgn": "delta"}
#: The compared groups' colours, in the order they were given.
GROUP_PALETTE = ("#d6604d", "#4393c3", "#1b9e77", "#7570b3", "#e7298a",
                 "#66a61e", "#e6ab02", "#a6761d")


def contrast_pairs(names, reference=None) -> list:
    """The pairs of groups compared, each the first minus the second: every
    other group against ``reference`` when one is named, else every pair
    in the order the groups were given."""
    names = list(names)
    if reference is not None:
        return [(n, reference) for n in names if n != reference]
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]


def _boot_p(dif) -> float:
    """Two-sided bootstrap p that a difference is 0, from its draws."""
    fin = np.asarray(dif, float)
    fin = fin[np.isfinite(fin)]
    if not fin.size:
        return np.nan
    return float(min(1.0, 2 * min(np.mean(fin <= 0), np.mean(fin >= 0))
                     + 1.0 / (fin.size + 1)))


def group_contrast(acc, sd, tot, S_sg, groups: dict, classes, band_labels,
                   n_boot: int, seed: int, min_pairs: int = 50,
                   pairs=None) -> list:
    """Groups of cells compared, pooled as the summary pools all cells.

    ``groups``: {name: indices of its cells}; ``pairs``: [(A, B), ...],
    each compared as A minus B (default: the first two groups).  Per class
    pair x band, and all regions x band: each group's ΔC, Δr, ΔC x w and
    ΔC x sign(w) -- its cells' contacts pooled; the signed ones against
    the group's own genome-wide co-open share, all regions with each
    contact's class-pair weight (``S_sg``) -- with 95% bootstrap intervals
    over the group's cells; per pair the difference with its interval (the
    two groups resampled independently), two-sided bootstrap p, and the
    same difference of the controls.  And the difference beyond the pair's
    overall ratio, ``diff_scaled``: A less k B, k (``scale_ratio``) the two
    groups' ratio of all-regions ΔC (Δr for Δr; CONTRAST_SCALE) in the
    same system and band, refitted in every bootstrap draw -- a group more
    concordant everywhere (labeling, depth, a global state) differs in
    every class, and this asks which classes differ beyond that; given
    only where both groups' all-regions value is clearly above 0 (its
    bootstrap 2.5th percentile).  For the signed statistics it takes in
    any change of the weight w too -- the class's co-open share against the
    genome's, a 1D difference as much as a concordance one.  Rows where
    either group has fewer than ``min_pairs`` contacts are left out."""
    if pairs is None:
        pairs = [tuple(list(groups)[:2])]
    n_k, nb = len(classes), len(band_labels)
    A_ = nb - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        s12 = sd[:, :, None] * sd[:, None, :]
        inv = np.where(s12 > 0, 1.0 / np.where(s12 > 0, s12, 1), 0.0)
    r_all = ((acc["xy"] - acc["xy1"]) * inv[..., None]).sum((1, 2))
    rc_all = ((acc["xy2"] - acc["xy1"]) * inv[..., None]).sum((1, 2))

    def cell_parts(ix):
        f = (lambda a_: a_[ix].reshape(ix.size, -1))   # noqa: E731
        P = {"n": f(acc["n"]), "d": f(2 * (acc["xy"] - acc["xy1"])),
             "c": f(2 * (acc["xy2"] - acc["xy1"])),
             "r": f((acc["xy"] - acc["xy1"]) * inv[..., None]),
             "rc": f((acc["xy2"] - acc["xy1"]) * inv[..., None]),
             "oo": f(acc["oo"]), "oc": f(acc["oc"]),
             "no": f(acc["no2"]), "nc": f(acc["nc2"])}
        G = {k: acc[k][ix][..., A_].sum((1, 2)) for k in ("oo", "oc", "no2",
                                                          "nc2")}
        Aall = {"n": tot["n"][ix], "d": 2 * (tot["xy"][ix] - tot["xy1"][ix]),
                "c": 2 * (tot["xy2"][ix] - tot["xy1"][ix]),
                "r": r_all[ix], "rc": rc_all[ix],
                **{k: S_sg[k][ix] for k in ("sd", "sc", "gd", "gc")}}
        return P, G, Aall

    def pooled(Wt, P, G, Aall):
        S = {k: Wt @ v for k, v in P.items()}
        g = {k: Wt @ v for k, v in G.items()}
        Sa = {k: Wt @ v for k, v in Aall.items()}
        with np.errstate(invalid="ignore", divide="ignore"):
            n = S["n"]
            o = {"delta": S["d"] / n, "delta_r": S["r"] / n,
                 "control": S["c"] / n, "control_r": S["rc"] / n}
            w = _weight(S["oo"] / (S["oo"] + S["oc"]),
                        (g["oo"] / (g["oo"] + g["oc"]))[:, None])
            wn = _weight(S["no"] / (S["no"] + S["nc"]),
                         (g["no2"] / (g["no2"] + g["nc2"]))[:, None])
            o.update({"signed_rel": o["delta"] * w,
                      "signed_sgn": o["delta"] * np.sign(w),
                      "signed_rel_control": o["control"] * wn,
                      "signed_sgn_control": o["control"] * np.sign(wn)})
            m = Sa["n"]
            al = {"delta": Sa["d"] / m, "delta_r": Sa["r"] / m,
                  "control": Sa["c"] / m, "control_r": Sa["rc"] / m,
                  "signed_rel": Sa["sd"] / m, "signed_sgn": Sa["gd"] / m,
                  "signed_rel_control": Sa["sc"] / m,
                  "signed_sgn_control": Sa["gc"] / m}
        return o, al

    est, boot, cnt = {}, {}, {}
    for gi, nm in enumerate(dict.fromkeys(g for p in pairs for g in p)):
        rng = np.random.default_rng([seed, gi])
        ix = np.asarray(groups[nm], np.int64)
        P, G, Aall = cell_parts(ix)
        est[nm] = pooled(np.ones((1, ix.size)), P, G, Aall)
        W = rng.multinomial(ix.size, np.full(ix.size, 1.0 / ix.size),
                            size=n_boot).astype(float) if ix.size else \
            np.zeros((n_boot, 0))
        boot[nm] = pooled(W, P, G, Aall)
        cnt[nm] = (P["n"].sum(0), (P["n"] > 0).sum(0), Aall["n"].sum(0),
                   (Aall["n"] > 0).sum(0))
    rows = []

    def emit(na, nb_, anchor, partner, b, which, j):
        pa, pb = cnt[na][2 * which][j], cnt[nb_][2 * which][j]
        if pa < min_pairs or pb < min_pairs:
            return
        base = {"anchor": anchor, "partner": partner,
                "band": band_labels[b], "group_A": na, "group_B": nb_,
                "pairs_A": int(pa), "pairs_B": int(pb),
                "cells_A": int(cnt[na][2 * which + 1][j]),
                "cells_B": int(cnt[nb_][2 * which + 1][j])}
        for st, ct in CONTRAST_STATS.items():
            va, vb = est[na][which][st][0, j], est[nb_][which][st][0, j]
            ba, bb = boot[na][which][st][:, j], boot[nb_][which][st][:, j]
            dif = ba - bb
            sc = CONTRAST_SCALE[st]
            ka, kb = boot[na][1][sc][:, b], boot[nb_][1][sc][:, b]
            with np.errstate(invalid="ignore", divide="ignore"):
                lo_a, hi_a = np.nanpercentile(ba, [2.5, 97.5])
                lo_b, hi_b = np.nanpercentile(bb, [2.5, 97.5])
                dlo, dhi = np.nanpercentile(dif, [2.5, 97.5])
                k_pt = est[na][1][sc][0, b] / est[nb_][1][sc][0, b]
                sdif = ba - ka / kb * bb
                # a ratio of two overall levels both clearly above 0 only
                pos = bool(np.nanpercentile(ka, 2.5) > 0
                           and np.nanpercentile(kb, 2.5) > 0)
            # all regions against its own ratio is 0 by construction
            s_ok = (pos and not (which == 1 and sc == st)
                    and np.isfinite(sdif).sum() >= 10)
            s_lo, s_hi = (np.nanpercentile(sdif, [2.5, 97.5]) if s_ok
                          else (np.nan, np.nan))
            rows.append({**base, "statistic": st, "value_A": float(va),
                         "lo_A": float(lo_a), "hi_A": float(hi_a),
                         "value_B": float(vb), "lo_B": float(lo_b),
                         "hi_B": float(hi_b), "diff": float(va - vb),
                         "diff_lo": float(dlo), "diff_hi": float(dhi),
                         "p_diff": _boot_p(dif),
                         "control_diff": float(est[na][which][ct][0, j]
                                               - est[nb_][which][ct][0, j]),
                         "scale_ratio": float(k_pt) if pos else np.nan,
                         "diff_scaled": (float(va - k_pt * vb) if s_ok
                                         else np.nan),
                         "diff_scaled_lo": float(s_lo),
                         "diff_scaled_hi": float(s_hi),
                         "p_scaled": _boot_p(sdif) if s_ok else np.nan})

    for na, nb_ in pairs:
        for k1 in range(n_k):
            for k2 in range(n_k):
                for b in range(nb):
                    emit(na, nb_, classes[k1], classes[k2], b, 0,
                         (k1 * n_k + k2) * nb + b)
        for b in range(nb):
            emit(na, nb_, ALL_REGIONS, ALL_REGIONS, b, 1, b)
    return rows


def _map_scale(a, b, se_a, se_b) -> dict:
    """How one group's map of bins relates to another's: each map's
    variance beyond its noise (its variance less the mean squared SE; as a
    share of the variance, ``reliability``), their correlation with the
    noise taken out (``r``), and A as ``offset`` + k B, k the ratio of the
    maps' SDs beyond noise -- the symmetric choice (B on A gives 1/k).  A
    group more concordant everywhere has k > 1 and offset 0; one shifted
    everywhere, k = 1 and an offset."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {"k": np.nan, "offset": np.nan, "r": np.nan,
           "reliability_A": np.nan, "reliability_B": np.nan}
    if a.size < 3:
        return out
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sa = va - np.mean(np.asarray(se_a, float) ** 2)
    sb = vb - np.mean(np.asarray(se_b, float) ** 2)
    if va > 0:
        out["reliability_A"] = float(sa / va)
    if vb > 0:
        out["reliability_B"] = float(sb / vb)
    if sa > 0 and sb > 0:
        out["k"] = float(np.sqrt(sa / sb))
        out["r"] = float(np.cov(a, b)[0, 1] / np.sqrt(sa * sb))
        out["offset"] = float(a.mean() - out["k"] * b.mean())
    return out


def hotspot_contrast(grid, arrays, groups: dict, n_band: int,
                     min_contacts: int, fdr: float, min_cells: int = 10,
                     pairs=None) -> pd.DataFrame:
    """Where groups of cells differ in concordance along the genome.

    ``arrays``: (g_lo, cell, d, pair, g_hi, c, band, ...) of the contacts;
    ``groups``: {name: boolean over the cells}; ``pairs``: [(A, B), ...],
    each compared as A minus B (default: the first two groups).  Per bin of
    ``grid``, all distances: each group's ΔC (:func:`locus_hotspots`), the
    difference with its standard error, and calls at an FDR against a null
    fitted to the controls' same quantity (:func:`_empirical_q`): bins
    higher in one group than the other (higher_A / higher_B); bins whose
    difference departs from the median bin's (above_median /
    below_median); and -- a group more concordant everywhere differs most
    where both are most concordant, which is no difference in where --
    bins that depart from B's map rescaled to A's (above_scaled /
    below_scaled): ``resid``, A less offset + k B, the scale from the
    maps' SDs beyond noise (:func:`_map_scale`; scale_k, scale_offset,
    r_maps)."""
    if pairs is None:
        pairs = [tuple(list(groups)[:2])]
    g_lo, cell, d, _, g_hi, c, band = arrays[:7]
    per = {}
    for nm in dict.fromkeys(g for p in pairs for g in p):
        m = np.asarray(groups[nm], bool)[cell]
        if not m.any():
            continue
        hs = locus_hotspots(grid, g_lo[m], g_hi[m], cell[m], d[m], c[m],
                            band[m], n_band)
        per[nm] = hs[hs["band_index"] == n_band].set_index("grid_index")
    cols = ["contacts", "cells", "delta", "control", "se", "se_control"]
    out = []
    for na, nb_ in pairs:
        if na not in per or nb_ not in per:
            continue
        j = per[na][["code", "start", "end"] + cols].join(
            per[nb_][cols], lsuffix="_A", rsuffix="_B", how="inner")
        j = j[(j["contacts_A"] >= min_contacts)
              & (j["contacts_B"] >= min_contacts)
              & (j["cells_A"] >= min_cells)
              & (j["cells_B"] >= min_cells)].sort_index().copy()
        if j.empty:
            continue
        j["diff"] = j["delta_A"] - j["delta_B"]
        j["control_diff"] = j["control_A"] - j["control_B"]
        se = np.sqrt(j["se_A"] ** 2 + j["se_B"] ** 2)
        sec = np.sqrt(j["se_control_A"] ** 2 + j["se_control_B"] ** 2)
        j["median_diff"] = float(j["diff"].median())
        with np.errstate(invalid="ignore", divide="ignore"):
            za, za0 = j["diff"] / se, j["control_diff"] / sec
            zr = (j["diff"] - j["diff"].median()) / se
            zr0 = (j["control_diff"] - j["control_diff"].median()) / sec
        j["z_diff"] = za
        j["q_higher_A"] = _empirical_q(za, za0)
        j["q_higher_B"] = _empirical_q(-za, -za0)
        j["higher_A"] = (j["q_higher_A"] < fdr) & (j["diff"] > 0)
        j["higher_B"] = (j["q_higher_B"] < fdr) & (j["diff"] < 0)
        j["q_above_median"] = _empirical_q(zr, zr0)
        j["q_below_median"] = _empirical_q(-zr, -zr0)
        j["above_median"] = (j["q_above_median"] < fdr) & (zr > 0)
        j["below_median"] = (j["q_below_median"] < fdr) & (zr < 0)
        sc = _map_scale(j["delta_A"], j["delta_B"], j["se_A"], j["se_B"])
        k, off = sc["k"], sc["offset"]
        j["scale_k"], j["scale_offset"], j["r_maps"] = k, off, sc["r"]
        j["reliability_A"] = sc["reliability_A"]
        j["reliability_B"] = sc["reliability_B"]
        j["resid"] = j["delta_A"] - off - k * j["delta_B"]
        j["z_resid"] = np.nan
        j["q_above_scaled"] = j["q_below_scaled"] = np.nan
        if np.isfinite(k):
            ca, cb = j["control_A"], j["control_B"]
            r0 = (ca - ca.mean()) - k * (cb - cb.mean())
            with np.errstate(invalid="ignore", divide="ignore"):
                zs = j["resid"] / np.sqrt(j["se_A"] ** 2
                                          + k ** 2 * j["se_B"] ** 2)
                zs0 = r0 / np.sqrt(j["se_control_A"] ** 2
                                   + k ** 2 * j["se_control_B"] ** 2)
            j["z_resid"] = zs
            j["q_above_scaled"] = _empirical_q(zs, zs0)
            j["q_below_scaled"] = _empirical_q(-zs, -zs0)
        j["above_scaled"] = (j["q_above_scaled"] < fdr) & (j["z_resid"] > 0)
        j["below_scaled"] = (j["q_below_scaled"] < fdr) & (j["z_resid"] < 0)
        j["group_A"], j["group_B"] = na, nb_
        out.append(j.reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _rotation_p(x, y, min_shift: int = 50) -> float:
    """Two-sided p of the Spearman correlation of two series along the
    genome, against every circular shift of one by ``min_shift`` bins or
    more (a tenth of the series when that is fewer): neighbouring bins are
    not independent, and a shift keeps each series' own autocorrelation
    while breaking their alignment.  NaN when too few shifts remain to
    reach p < 0.05."""
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    ms = max(1, min(min_shift, n // 10))
    if n - 2 * ms + 1 < 19:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    # every circular shift at once: cc[s] = mean_i rx[i] ry[i + s]
    cc = np.fft.irfft(np.conj(np.fft.rfft(rx)) * np.fft.rfft(ry), n) / n
    null = cc[ms:n - ms + 1]
    return float((1 + np.sum(np.abs(null) >= abs(cc[0]) - 1e-12))
                 / (1 + null.size))


def hotspot_contrast_enrichment(hc: pd.DataFrame, tracks: dict) -> list:
    """Per pair of groups and annotation track, across the compared bins:
    Spearman rho of the groups' difference with the track, of the
    difference beyond B's rescaled map (``resid``), of the controls'
    difference, and of each group's own ΔC -- with p against genome
    rotations (:func:`_rotation_p`; p_diff is the plain Spearman p)."""
    from scipy.stats import spearmanr
    rows = []
    if hc.empty:
        return rows
    for (na, nb_), h in hc.groupby(["group_A", "group_B"], sort=False):
        h = h.sort_values("grid_index")
        gi = h["grid_index"].to_numpy()
        dif = h["diff"].to_numpy(float)
        res = h["resid"].to_numpy(float)
        fin = np.isfinite(res)
        for name, v in tracks.items():
            x = v[gi]
            if np.ptp(x) == 0:
                continue
            r, p_ = spearmanr(dif, x)
            row = {"group_A": na, "group_B": nb_, "track": name,
                   "bins": int(gi.size), "rho_diff": float(r),
                   "p_diff": float(p_), "p_rot_diff": _rotation_p(dif, x),
                   "rho_resid": (float(spearmanr(res[fin], x[fin])[0])
                                 if fin.sum() > 2 else np.nan),
                   "p_rot_resid": _rotation_p(res, x),
                   "rho_control_diff": float(spearmanr(h["control_diff"],
                                                       x)[0]),
                   "rho_A": float(spearmanr(h["delta_A"], x)[0]),
                   "rho_B": float(spearmanr(h["delta_B"], x)[0])}
            for k in ("higher_A", "higher_B", "above_median", "below_median",
                      "above_scaled", "below_scaled"):
                hk = h[k].to_numpy(bool)
                row[f"{k}_with"] = int((hk & (x > 0)).sum())
                row[f"{k}_total"] = int(hk.sum())
            rows.append(row)
    return rows


def adjusted_contrast(percell: pd.DataFrame, cov: pd.DataFrame, ctx_cov: dict,
                      group_of: dict, names, batch_cols=(),
                      min_cells: int = 20) -> pd.DataFrame:
    """Two groups' difference with how the cells were measured held fixed.

    ``group_of``: {cell: group}, the cells of ``names`` = (A, B).  Per
    region group (same-state class pairs and all regions) x band x
    statistic: a least-squares fit of the per-cell values, weighted by the
    cell's contacts, on the group, the technical covariates
    (TECHNICAL_GROUPS; a missing one at the cell type's mean) and batch --
    the group's coefficient is the adjusted difference.  A batch column
    whose every batch holds one group only is left out, as it would take
    the groups' difference with it (batch_adjusted then names none);
    diff_wls is the same fit on the group alone."""
    from scipy.stats import t as t_dist
    tech = [c for g, cols in COV_GROUPS if g in TECHNICAL_GROUPS
            for c in cols]
    group_of = {c: g for c, g in group_of.items() if g in names[:2]}
    pc = percell[percell["enough_pairs"]
                 & (percell["anchor"] == percell["partner"])
                 & percell["cell"].isin(group_of)]
    rows = []
    for (ctx, sy, k, bl), g in pc.groupby(["context", "system", "anchor",
                                           "band"], sort=False):
        d = g.merge(cov.drop(columns=["cell_type"], errors="ignore"),
                    on="cell", how="left")
        cx = (ctx_cov or {}).get(ctx)
        if cx is not None:
            d = d.merge(cx, on="cell", how="left")
        if "sd_anchor" in d:
            d["scale"] = d["sd_anchor"] * d["sd_partner"]
        grp = d["cell"].map(group_of)
        xg = (grp == names[0]).to_numpy(float)
        if xg.sum() < min_cells or (1 - xg).sum() < min_cells:
            continue
        t = d["cell_type"].astype(str).to_numpy()
        av = [c for c in tech if c in d and d[c].notna().sum() >= 10
              and d[c].nunique() > 1]
        bused = []
        for b in batch_cols:
            if b in d and d[b].nunique() > 1 and \
                    d.groupby(b)["cell"].apply(
                        lambda s_: grp[s_.index].nunique()).max() > 1:
                bused.append(b)
        for st in CONTRAST_STATS:
            if st not in d:
                continue
            y = d[st].to_numpy(float)
            w = d["pairs"].to_numpy(float)
            ok = np.isfinite(y) & (w > 0)
            if xg[ok].sum() < min_cells or (1 - xg[ok]).sum() < min_cells:
                continue
            Z = [_type_impute(d[c].to_numpy(float)[ok], t[ok]) for c in av]
            for b in bused:
                Z += list(pd.get_dummies(d.loc[ok, b].astype(str),
                                         drop_first=True).to_numpy(float).T)

            def wls(cols_):
                X = np.column_stack([np.ones(ok.sum()), xg[ok]] + cols_)
                sw = np.sqrt(w[ok])
                Xw, yw = X * sw[:, None], y[ok] * sw
                beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
                res = yw - Xw @ beta
                dfr = max(Xw.shape[0] - np.linalg.matrix_rank(Xw), 1)
                s2 = float(res @ res) / dfr
                cv = s2 * np.linalg.pinv(Xw.T @ Xw)
                return float(beta[1]), float(np.sqrt(max(cv[1, 1], 0))), dfr
            b0, _, _ = wls([])
            b1, se1, dfr = wls(Z)
            tt = b1 / se1 if se1 > 0 else np.nan
            rows.append({"context": ctx, "system": sy, "anchor": k,
                         "partner": k, "band": bl, "group_A": names[0],
                         "group_B": names[1], "statistic": st,
                         "diff_wls": b0, "diff_adjusted": b1,
                         "diff_adjusted_se": se1,
                         "p_adjusted": (float(2 * t_dist.sf(abs(tt), dfr))
                                        if np.isfinite(tt) else np.nan),
                         "batch_adjusted": ",".join(bused)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_adjusted"] = np.nan
        for _, g in out.groupby(["context", "statistic"], sort=False):
            out.loc[g.index, "q_adjusted"] = fdr_bh(g["p_adjusted"].to_numpy())
    return out


def _contrast_rows(a: pd.DataFrame) -> list:
    """The region groups a contrast figure shows, top to bottom: all
    regions (from the system with the most cells), then each system's
    same-state class pairs."""
    all_b = next((b for b in dict.fromkeys(a["band"])
                  if str(b).startswith("all")), None)
    pick = None
    al = a[(a["anchor"] == ALL_REGIONS) & (a["band"] == all_b)]
    if not al.empty:
        pick = al.assign(n_=al["cells_A"] + al["cells_B"]).sort_values(
            "n_").iloc[-1]["system"]
    return ([(pick, ALL_REGIONS)] if pick is not None else []) + [
        (sy, k) for sy in [s_ for s_ in SYSTEMS if s_ in set(a["system"])]
        + [s_ for s_ in dict.fromkeys(a["system"]) if s_ not in SYSTEMS]
        for k in dict.fromkeys(a.loc[(a["system"] == sy)
                                     & (a["anchor"] != ALL_REGIONS),
                                     "anchor"])]


def _legend_below(fig, handles, labels, ncol: int, fontsize: float = 5):
    """One legend under everything the figure has drawn."""
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y0 = min(ax.get_tightbbox(rnd).transformed(inv).y0 for ax in fig.axes)
    fig.legend(handles, labels, loc="upper center", ncol=ncol,
               bbox_to_anchor=(0.5, y0 - 0.01), fontsize=fontsize,
               frameon=False)


def _suptitle_above(fig, text: str, fontsize: float = 7):
    """The figure's title, above everything its axes have drawn."""
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    top = max(ax.get_tightbbox(rnd).y1 for ax in fig.axes)
    fig.suptitle(text, fontsize=fontsize, va="bottom",
                 y=fig.transFigure.inverted().transform((0, top))[1] + 0.01)


def _adjusted_label(a: pd.DataFrame) -> str:
    """What the adjusted difference held fixed."""
    b = (a["batch_adjusted"].dropna().astype(str) if "batch_adjusted" in a
         else pd.Series(dtype=str))
    if (b != "").any():
        return "adjusted for technical covariates + batch"
    return ("adjusted for technical covariates (batch not: each batch "
            "holds one group)")


def plot_contrast(tab: pd.DataFrame, context: str, stat: str, path,
                  q_cut: float = 0.05, pair=None):
    """a: each group's pooled concordance (pp, 95% CI), all distances, per
    region group; b: the first group less the second (filled: q < q_cut),
    beyond the pair's all-regions ratio (square), and adjusted for the
    technical covariates and batch (diamond); c: the difference by
    distance band (* q < q_cut)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from nome_hic_concordance import figure_style
    a = tab[(tab["context"] == context) & (tab["statistic"] == stat)
            & (tab["anchor"] == tab["partner"])]
    if pair is not None:
        a = a[(a["group_A"] == pair[0]) & (a["group_B"] == pair[1])]
    if a.empty:
        return
    na, nb_ = a["group_A"].iloc[0], a["group_B"].iloc[0]
    bands = list(dict.fromkeys(a["band"]))
    all_b = next((b for b in bands if str(b).startswith("all")), bands[-1])
    bands = [b for b in bands if b != all_b]
    keys = _contrast_rows(a)
    ix = a.set_index(["system", "anchor", "band"])
    ny = len(keys)
    PP = 100
    cA, cB = GROUP_PALETTE[0], GROUP_PALETTE[1]
    with plt.rc_context(figure_style.RC):
        fig = plt.figure(figsize=(figure_style.FIG_W_IN, 1.0 + 0.2 * ny))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.2], wspace=0.08)
        ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))
        yy = np.arange(ny)
        has_adj = has_sc = False
        for i, (sy, k) in enumerate(keys):
            if (sy, k, all_b) not in ix.index:
                continue
            r = ix.loc[(sy, k, all_b)]
            for dy, nm, col in ((-0.15, "A", cA), (0.15, "B", cB)):
                ax0.plot([PP * r[f"lo_{nm}"], PP * r[f"hi_{nm}"]],
                         [i + dy] * 2, color=col, lw=0.8)
                ax0.plot([PP * r[f"value_{nm}"]], [i + dy], "o", color=col,
                         ms=3)
            q = r.get("q_diff", np.nan)
            sig = np.isfinite(q) and q < q_cut
            ax1.plot([PP * r["diff_lo"], PP * r["diff_hi"]], [i] * 2,
                     color="0.2", lw=0.8)
            ax1.plot([PP * r["diff"]], [i], "o", ms=3.5,
                     mfc="0.1" if sig else "white", mec="0.1")
            if np.isfinite(r.get("diff_scaled", np.nan)):
                has_sc = True
                qs = r.get("q_scaled", np.nan)
                ax1.plot([PP * r["diff_scaled"]], [i + 0.22], "s", ms=2.6,
                         mec="#1b9e77", mew=0.8,
                         mfc=("#1b9e77" if np.isfinite(qs) and qs < q_cut
                              else "none"))
            if np.isfinite(r.get("diff_adjusted", np.nan)):
                has_adj = True
                qa = r.get("q_adjusted", np.nan)
                ax1.plot([PP * r["diff_adjusted"]], [i - 0.22], "D", ms=2.6,
                         mec="#e6550d", mew=0.8,
                         mfc=("#e6550d" if np.isfinite(qa) and qa < q_cut
                              else "none"))
        ax0.axvline(0, color="0.6", lw=0.4)
        ax1.axvline(0, color="0.6", lw=0.4)
        for ax in (ax0, ax1):
            ax.set_ylim(ny - 0.5, -0.5)
            ax.set_yticks(yy)
        ax0.set_yticklabels([_row_label(sy, k) for sy, k in keys],
                            fontsize=5.5)
        ax1.set_yticklabels([])
        ax0.set_xlabel(f"{_STAT_LABEL.get(stat, stat)}, pp, all distances")
        ax1.set_xlabel(f"{na} − {nb_}, pp")
        M = np.full((ny, len(bands)), np.nan)
        Q = np.full((ny, len(bands)), np.nan)
        for i, (sy, k) in enumerate(keys):
            for jb, bl in enumerate(bands):
                if (sy, k, bl) in ix.index:
                    M[i, jb] = PP * ix.loc[(sy, k, bl), "diff"]
                    Q[i, jb] = ix.loc[(sy, k, bl)].get("q_diff", np.nan)
        lim = (float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0)
        im = ax2.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(ny):
            for jb in range(len(bands)):
                if np.isfinite(Q[i, jb]) and Q[i, jb] < q_cut:
                    ax2.text(jb, i, "*", ha="center", va="center",
                             fontsize=7)
        ax2.set_yticks([])
        ax2.set_xticks(range(len(bands)))
        ax2.set_xticklabels(bands, rotation=40, ha="right", fontsize=5)
        cb = fig.colorbar(im, ax=ax2, fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=5)
        cb.set_label(f"{na} − {nb_}, pp", fontsize=6)
        for ax, lab in zip((ax0, ax1, ax2), "abc"):
            ax.set_title(lab, loc="left", fontweight="bold")
        _suptitle_above(fig, f"{context}: {na} vs {nb_} -- "
                             f"{_STAT_LABEL.get(stat, stat)} by region group "
                             f"(filled / *: q < {q_cut:g})")
        nA, nB = int(a["cells_A"].max()), int(a["cells_B"].max())
        hd = [Line2D([], [], color=cA, marker="o", ms=3, lw=0.8),
              Line2D([], [], color=cB, marker="o", ms=3, lw=0.8),
              Line2D([], [], color="0.1", marker="o", ms=3.5, mfc="white",
                     lw=0.8)]
        lb = [f"{na} ({nA:,} cells)", f"{nb_} ({nB:,} cells)",
              f"{na} − {nb_}"]
        if has_sc:
            hd.append(Line2D([], [], ls="none", marker="s", ms=2.6,
                             mec="#1b9e77", mfc="none", mew=0.8))
            lb.append(f"beyond the all-regions ratio ({na} − k {nb_})")
        if has_adj:
            hd.append(Line2D([], [], ls="none", marker="D", ms=2.6,
                             mec="#e6550d", mfc="none", mew=0.8))
            lb.append(_adjusted_label(a))
        _legend_below(fig, hd, lb, ncol=3)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_contrast_overview(tab: pd.DataFrame, context: str, stat: str,
                           path, q_cut: float = 0.05):
    """Every compared pair at once, all distances.  a: each group's pooled
    concordance (pp, 95% CI) per region group; b: per pair, the difference
    as is, beyond the pair's all-regions ratio, and adjusted for the
    technical covariates and batch (pp; * q < q_cut)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from nome_hic_concordance import figure_style
    a = tab[(tab["context"] == context) & (tab["statistic"] == stat)
            & (tab["anchor"] == tab["partner"])]
    all_b = next((b for b in dict.fromkeys(a["band"])
                  if str(b).startswith("all")), None)
    a = a[a["band"] == all_b]
    pairs = list(dict.fromkeys(zip(a["group_A"], a["group_B"])))
    if len(pairs) < 2:
        return
    groups = list(dict.fromkeys(g for p in pairs for g in p))
    keys = _contrast_rows(a)
    ny = len(keys)
    PP = 100
    val, ncell = {}, {}
    for side in ("A", "B"):
        for _, r in a.iterrows():
            g = r[f"group_{side}"]
            val.setdefault((g, r["system"], r["anchor"]),
                           (r[f"value_{side}"], r[f"lo_{side}"],
                            r[f"hi_{side}"]))
            ncell[g] = max(ncell.get(g, 0), int(r[f"cells_{side}"]))
    kinds = [("as is", "diff", "q_diff"),
             ("beyond ratio", "diff_scaled", "q_scaled"),
             ("adjusted", "diff_adjusted", "q_adjusted")]
    kinds = [kd for kd in kinds if kd[1] in a and a[kd[1]].notna().any()]
    ix = a.set_index(["group_A", "group_B", "system", "anchor"])
    M = np.full((ny, len(pairs) * len(kinds)), np.nan)
    Q = np.full(M.shape, np.nan)
    for i, (sy, k) in enumerate(keys):
        for pi, (ga, gb) in enumerate(pairs):
            if (ga, gb, sy, k) not in ix.index:
                continue
            r = ix.loc[(ga, gb, sy, k)]
            for ki, (_, col, qcol) in enumerate(kinds):
                M[i, pi * len(kinds) + ki] = PP * r[col]
                Q[i, pi * len(kinds) + ki] = r.get(qcol, np.nan)
    with plt.rc_context(figure_style.RC):
        fig = plt.figure(figsize=(figure_style.FIG_W_IN, 1.3 + 0.2 * ny))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.06)
        ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        ng = len(groups)
        off = (np.arange(ng) - (ng - 1) / 2) * min(0.6 / max(ng - 1, 1),
                                                   0.25)
        for gi, g in enumerate(groups):
            col = GROUP_PALETTE[gi % len(GROUP_PALETTE)]
            for i, (sy, k) in enumerate(keys):
                if (g, sy, k) not in val:
                    continue
                v, lo, hi = val[(g, sy, k)]
                ax0.plot([PP * lo, PP * hi], [i + off[gi]] * 2, color=col,
                         lw=0.8)
                ax0.plot([PP * v], [i + off[gi]], "o", color=col, ms=2.6)
        ax0.axvline(0, color="0.6", lw=0.4)
        ax0.set_ylim(ny - 0.5, -0.5)
        ax0.set_yticks(np.arange(ny))
        ax0.set_yticklabels([_row_label(sy, k) for sy, k in keys],
                            fontsize=5.5)
        ax0.set_xlabel(f"{_STAT_LABEL.get(stat, stat)}, pp, all distances")
        fin = np.abs(M[np.isfinite(M)])
        lim = float(np.percentile(fin, 95)) if fin.size else 1.0
        im = ax1.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(ny):
            for jc in range(M.shape[1]):
                if np.isfinite(Q[i, jc]) and Q[i, jc] < q_cut:
                    ax1.text(jc, i, "*", ha="center", va="center",
                             fontsize=7)
        for pi in range(1, len(pairs)):
            ax1.axvline(pi * len(kinds) - 0.5, color="white", lw=2)
        ax1.set_yticks([])
        ax1.set_xticks(range(M.shape[1]))
        ax1.set_xticklabels([kd[0] for _ in pairs for kd in kinds],
                            rotation=40, ha="right", fontsize=5)
        sec = ax1.secondary_xaxis("bottom")
        sec.set_xticks([pi * len(kinds) + (len(kinds) - 1) / 2
                        for pi in range(len(pairs))])
        sec.set_xticklabels([f"{ga} − {gb}" for ga, gb in pairs],
                            fontsize=5.5)
        sec.tick_params(length=0, pad=30)
        sec.spines["bottom"].set_visible(False)
        cb = fig.colorbar(im, ax=ax1, fraction=0.04, pad=0.02,
                          extend="both")
        cb.ax.tick_params(labelsize=5)
        cb.set_label("difference, pp", fontsize=6)
        ax0.set_title("a", loc="left", fontweight="bold")
        ax1.set_title("b", loc="left", fontweight="bold")
        _suptitle_above(fig, f"{context}: {_STAT_LABEL.get(stat, stat)} -- "
                             f"every group, and each pair's difference "
                             f"(* q < {q_cut:g})")
        hd = [Line2D([], [], color=GROUP_PALETTE[gi % len(GROUP_PALETTE)],
                     marker="o", ms=2.6, lw=0.8) for gi in range(ng)]
        lb = [f"{g} ({ncell.get(g, 0):,} cells)" for g in groups]
        _legend_below(fig, hd, lb, ncol=min(ng, 4))
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_hotspot_contrast(hc: pd.DataFrame, enr: pd.DataFrame, context: str,
                          path, fdr: float = 0.05, pair=None):
    """a: along the genome, per bin, the first group's ΔC against the
    second's rescaled to it (pp; :func:`_map_scale`) -- or, without a
    scale, the plain difference against the median bin's -- with the bins
    that depart called; b: per annotation track, Spearman rho with the
    track of the difference (bar), of the difference beyond the rescaled
    map (diamond, filled: q < 0.05), the controls' (circle), and each
    group's own ΔC (triangles)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    a = hc[hc["context"] == context] if not hc.empty else hc
    if pair is not None and not a.empty:
        a = a[(a["group_A"] == pair[0]) & (a["group_B"] == pair[1])]
    if a.empty:
        return
    na, nb_ = a["group_A"].iloc[0], a["group_B"].iloc[0]
    e = enr
    if not enr.empty:
        e = enr[(enr["context"] == context) & (enr["group_A"] == na)
                & (enr["group_B"] == nb_)].sort_values("rho_diff")
    codes = np.sort(a["code"].unique())
    span = a.groupby("code")["end"].max().reindex(codes).to_numpy(float)
    off = dict(zip(codes, np.r_[0, np.cumsum(span)[:-1]]))
    x = a["code"].map(off).to_numpy() + (a["start"] + a["end"]) / 2
    rank = a["code"].map({c: i for i, c in enumerate(codes)}).to_numpy()
    names = a.groupby("code")["chrom"].first().reindex(codes)
    k = float(a["scale_k"].iloc[0]) if "scale_k" in a else np.nan
    scaled = np.isfinite(k)
    with plt.rc_context(figure_style.RC):
        fig = plt.figure(figsize=(figure_style.FIG_W_IN,
                                  2.0 + 0.13 * max(len(e), 4)))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 0.13 * max(len(e), 4)],
                              hspace=0.6)
        ax = fig.add_subplot(gs[0])
        if scaled:
            y = 100 * a["resid"].to_numpy(float)
            calls = (("above_scaled", "#d6604d",
                      f"more concordant in {na} than its scale says"),
                     ("below_scaled", "#4393c3",
                      f"less concordant in {na} than its scale says"))
            ref = 0.0
            o_ = 100 * float(a["scale_offset"].iloc[0])
            ax.set_title(
                f"a  {context}: {na} against {nb_} rescaled ({na} = "
                f"{k:.2f} × {nb_} {'+' if o_ >= 0 else '−'} {abs(o_):.2f} "
                f"pp; maps r = {a['r_maps'].iloc[0]:.2f} beyond noise): "
                f"{int(a['above_scaled'].sum())} bins above, "
                f"{int(a['below_scaled'].sum())} below", loc="left",
                fontsize=6.5)
            ax.set_ylabel(f"ΔC {na} − rescaled {nb_} (pp)")
        else:
            y = 100 * a["diff"].to_numpy(float)
            calls = (("higher_A", "#d6604d", f"higher in {na}"),
                     ("higher_B", "#4393c3", f"higher in {nb_}"))
            ref = 100 * float(a["diff"].median())
            ax.set_title(f"a  {context}: where {na} and {nb_} differ "
                         f"(dotted: the median bin; "
                         f"{int(a['above_median'].sum())} bins above it, "
                         f"{int(a['below_median'].sum())} below)",
                         loc="left", fontsize=6.5)
            ax.set_ylabel(f"ΔC {na} − {nb_} (pp)")
        ax.scatter(x, y, s=1.5, lw=0, c=np.where(rank % 2, "0.2", "0.55"))
        for kc, c_, lab in calls:
            m = a[kc].to_numpy(bool)
            ax.scatter(x[m], y[m], s=3, color=c_, lw=0,
                       label=f"{lab} ({int(m.sum())} bins, FDR {fdr:g})")
        ax.axhline(0, color="0.4", lw=0.4)
        if ref:
            ax.axhline(ref, color="k", lw=0.4, ls=":")
        ax.set_xticks([off[c] + s_ / 2 for c, s_ in zip(codes, span)])
        ax.set_xticklabels([_canon(n) for n in names], fontsize=4.5)
        ax.set_xlim(0, float(np.sum(span)))
        ax.legend(fontsize=4.5, markerscale=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
        if not e.empty:
            ax2 = fig.add_subplot(gs[1])
            yy = np.arange(len(e))
            ax2.barh(yy, e["rho_diff"], height=0.7,
                     color=np.where(e["rho_diff"] > 0, "#d6604d", "#4393c3"))
            ax2.scatter(e["rho_control_diff"], yy, s=8, facecolors="none",
                        edgecolors="0.2", lw=0.5, zorder=3, label="control")
            ax2.scatter(e["rho_A"], yy, s=8, marker="^", color="#b2182b",
                        lw=0, zorder=4, label=f"{na} alone")
            ax2.scatter(e["rho_B"], yy, s=8, marker="v", color="#2166ac",
                        lw=0, zorder=4, label=f"{nb_} alone")
            if "rho_resid" in e and e["rho_resid"].notna().any():
                rr = e["rho_resid"].to_numpy(float)
                qr = (e["q_resid"] if "q_resid" in e else
                      e.get("p_rot_resid", pd.Series(np.nan, e.index)))
                sig = (qr.to_numpy(float) < 0.05)
                ax2.scatter(rr[sig], yy[sig], s=9, marker="D", color="k",
                            lw=0, zorder=5,
                            label=f"beyond rescaled {nb_} (q < 0.05)")
                ax2.scatter(rr[~sig], yy[~sig], s=9, marker="D",
                            facecolors="white", edgecolors="k", lw=0.5,
                            zorder=5, label=f"beyond rescaled {nb_}")
            for y_, (r_, q_) in enumerate(zip(e["rho_diff"],
                                              e.get("q_diff", e["p_diff"]))):
                if q_ < 0.05:
                    ax2.text(r_, y_, " *" if r_ >= 0 else "* ", va="center",
                             ha="left" if r_ >= 0 else "right", fontsize=6)
            ax2.axvline(0, color="0.4", lw=0.4)
            ax2.set_yticks(yy)
            ax2.set_yticklabels(e["track"], fontsize=5)
            ax2.set_xlabel(f"Spearman ρ across bins with the track: bar, "
                           f"{na} − {nb_} (* q < 0.05, genome rotations)")
            ax2.set_title("b  what the differing bins are", loc="left",
                          fontsize=6.5)
            _legend_below(fig, *ax2.get_legend_handles_labels(), ncol=3,
                          fontsize=4.5)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


# ---------------------------------------------- chrX and imprinted domains --

#: The chrX analysis' system: each end's chromosome, an autosome or X.
CHROM_SYSTEM = "chrX"
#: Imprinted genes whose imprinting differs between neurons and glia --
#: Ube3a maternal in neurons and from both alleles in glia; Grb10 paternal
#: from a neuron promoter, maternal elsewhere -- marked † in the figures.
NEURON_IMPRINTED = ("Ube3a", "Grb10")


class ChromosomeClasses:
    """Autosomes against chrX as an annotation system: an end on a numbered
    chromosome is 'autosomes', one on X 'chrX', any other -1.  A contact's
    two ends share its chromosome -- and one copy of it -- so where the two
    copies of X differ, one inactive, X-X contacts are concordant beyond
    their reshuffled partners, which come from either copy."""

    def __init__(self):
        self.name = CHROM_SYSTEM
        self.classes = ("autosomes", "chrX")
        self._x = _chrom_code("X")

    def lookup_g(self, g) -> np.ndarray:
        code = np.asarray(g, np.int64) // STRIDE
        out = np.full(code.size, -1, np.int8)
        out[(code >= 1) & (code < 1000)] = 0
        out[code == self._x] = 1
        return out


def _sex_counts(path) -> dict:
    """One cell's read pairs per chromosome and the furthest position they
    reach: {chromosome code: (pairs, max position)}."""
    df, names, ci, lut = _categorical(path, ["chrom", "pos1", "pos2"])
    code = lut[ci]
    hi = np.maximum(df["pos1"].to_numpy(np.int64),
                    df["pos2"].to_numpy(np.int64))
    out = {}
    for k in np.unique(code[code >= 0]):
        m = code == k
        out[int(k)] = (int(m.sum()), int(hi[m].max()))
    return out


def call_sex(counts: dict, min_pairs: int = 2000, x_female: float = 0.8,
             x_male: float = 0.7) -> pd.DataFrame:
    """Each cell's sex from its coverage (``counts``: {cell:
    :func:`_sex_counts`}).  X dosage: its read pairs per bp of X over those
    per bp of the autosomes, a chromosome's length the furthest position
    any cell reaches -- about 1 with two X, 0.5 with one.  Female at
    ``x_female`` or more, male at ``x_male`` or less, else unknown, and
    unknown with fewer than ``min_pairs`` autosomal pairs.  A 'female' with
    a quarter of the males' median share of pairs on Y -- or, without
    males, ten times the females' median -- is unknown too: a male-female
    doublet, or a male."""
    X, Y = _chrom_code("X"), _chrom_code("Y")
    length: dict = {}
    for c in counts.values():
        for k, (_, mx) in c.items():
            length[k] = max(length.get(k, 0), mx)
    auto = [k for k in length if 1 <= k < 1000]
    L_a, L_x = sum(length[k] for k in auto), length.get(X, 0)
    rows = []
    for cell, c in counts.items():
        n_a = sum(c.get(k, (0, 0))[0] for k in auto)
        n_x, n_y = c.get(X, (0, 0))[0], c.get(Y, (0, 0))[0]
        rows.append({"cell": cell, "pairs_autosomes": n_a, "pairs_X": n_x,
                     "pairs_Y": n_y,
                     "x_dosage": ((n_x / L_x) / (n_a / L_a)
                                  if n_a and L_x and L_a else np.nan),
                     "y_share": n_y / n_a if n_a else np.nan})
    t = pd.DataFrame(rows)
    if t.empty:
        return t.assign(sex_call=pd.Series(dtype=str))
    t["sex_call"] = np.where(
        t["x_dosage"] >= x_female, "female",
        np.where(t["x_dosage"] <= x_male, "male", "unknown"))
    t.loc[(t["pairs_autosomes"] < min_pairs) | t["x_dosage"].isna(),
          "sex_call"] = "unknown"
    fem, mal = t["sex_call"] == "female", t["sex_call"] == "male"
    if t["pairs_Y"].sum() > 0 and fem.any():
        y_cut = (0.25 * float(t.loc[mal, "y_share"].median()) if mal.any()
                 else 10 * max(float(t.loc[fem, "y_share"].median()), 1e-5)
                 if fem.sum() >= 10 else np.inf)
        t.loc[fem & (t["y_share"] > y_cut), "sex_call"] = "unknown"
    return t


def _given_sex(values) -> np.ndarray:
    """A sample sheet's sex column as female / male / unknown."""
    v = pd.Series(values).astype(str).str.strip().str.lower()
    return np.where(v.isin(["f", "female", "xx", "woman"]), "female",
                    np.where(v.isin(["m", "male", "xy", "man"]), "male",
                             "unknown"))


def class_pair_difference(acc, sd, groups: dict, classes, pair_a, pair_b,
                          band_labels, n_boot: int, seed: int) -> list:
    """Per group of cells (``groups``: {name: cell indices}): two class
    pairs' pooled ΔC and Δr by band -- each cell's contacts pooled, as the
    summary pools them -- and the second less the first, from the same
    bootstrap draws of the group's cells, so the interval knows what the
    two share."""
    rng = np.random.default_rng(seed)
    parts = []
    for i, j in ((classes.index(pair_a[0]), classes.index(pair_a[1])),
                 (classes.index(pair_b[0]), classes.index(pair_b[1]))):
        with np.errstate(invalid="ignore", divide="ignore"):
            s12 = sd[:, i] * sd[:, j]
            inv = np.where(s12 > 0, 1.0 / np.where(s12 > 0, s12, 1), 0.0)
        dd = acc["xy"][:, i, j, :] - acc["xy1"][:, i, j, :]
        parts.append({"n": acc["n"][:, i, j, :], "d": 2 * dd,
                      "r": dd * inv[:, None],
                      "c": 2 * (acc["xy2"][:, i, j, :]
                                - acc["xy1"][:, i, j, :])})

    def pooled(Wt, P, ix):
        n = Wt @ P["n"][ix]
        with np.errstate(invalid="ignore", divide="ignore"):
            return {"delta": (Wt @ P["d"][ix]) / n,
                    "delta_r": (Wt @ P["r"][ix]) / n,
                    "control": (Wt @ P["c"][ix]) / n}, n
    rows = []
    for g, ix in groups.items():
        ix = np.asarray(ix, np.int64)
        if not ix.size:
            continue
        W = rng.multinomial(ix.size, np.full(ix.size, 1.0 / ix.size),
                            size=n_boot).astype(float)
        one = np.ones((1, ix.size))
        (ea, na), (eb, nb_) = (pooled(one, P, ix) for P in parts)
        ba, bb = (pooled(W, P, ix)[0] for P in parts)
        for b, bl in enumerate(band_labels):
            for st in ("delta", "delta_r"):
                with np.errstate(invalid="ignore"):
                    lo_a, hi_a = np.nanpercentile(ba[st][:, b], [2.5, 97.5])
                    lo_b, hi_b = np.nanpercentile(bb[st][:, b], [2.5, 97.5])
                    dif = bb[st][:, b] - ba[st][:, b]
                    dlo, dhi = np.nanpercentile(dif, [2.5, 97.5])
                rows.append({
                    "group": g, "band": bl, "statistic": st,
                    "cells": int(ix.size), "pairs_A": int(na[0, b]),
                    "pairs_B": int(nb_[0, b]), "value_A": float(ea[st][0, b]),
                    "lo_A": float(lo_a), "hi_A": float(hi_a),
                    "value_B": float(eb[st][0, b]), "lo_B": float(lo_b),
                    "hi_B": float(hi_b),
                    "diff": float(eb[st][0, b] - ea[st][0, b]),
                    "diff_lo": float(dlo), "diff_hi": float(dhi),
                    "p_diff": _boot_p(dif),
                    "control_diff": float(eb["control"][0, b]
                                          - ea["control"][0, b])})
    return rows


def chromosome_concordance(cell, g_lo, g_hi, a, p, marg, strat, groups: dict,
                           edges, band_labels, dbins, position_bin: int,
                           max_widen: int, n_boot: int, seed: int,
                           n_threads: int = 1):
    """chrX against the autosomes (:class:`ChromosomeClasses`), as one
    system's class pairs: every contact's ends centred on its cell's mean
    on autosomes or on X (``marg``: cells x (count, sum, sum of squares) x
    (autosomes, X)), its partner reshuffled within ``strat`` -- the cell's
    type, and sex when known, so an X donor never comes from the other
    sex -- and per group of cells (:func:`class_pair_difference`) X-X less
    autosome-autosome ΔC and Δr by band.  Returns (rows, per-cell rows of
    every cell's own two values at all distances)."""
    lk = ChromosomeClasses()
    kx = lk.lookup_g(g_lo).astype(np.int64)
    ky = lk.lookup_g(g_hi).astype(np.int64)
    n_cells = marg.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = marg[:, 1] / marg[:, 0]
        sd = np.sqrt(np.maximum(marg[:, 2] / marg[:, 0] - mu ** 2, 0))
    a_c = np.where(kx >= 0, a - mu[cell, np.maximum(kx, 0)], np.nan)
    p_c = np.where(ky >= 0, p - mu[cell, np.maximum(ky, 0)], np.nan)
    dist = g_hi - g_lo
    band = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0,
                   len(edges) - 2)
    s_code, s_names = pd.factorize(np.asarray(strat))
    ok, right, left, _ = reshuffle(
        g_lo, g_hi, dist, cell, kx, ky, a_c, p_c, s_code[cell], 2, dbins,
        position_bin, max_widen, seed=seed, names=list(s_names),
        n_threads=n_threads)
    idx = np.flatnonzero(ok)
    x_, y_ = a_c[idx], p_c[idx]
    w = {"xy": x_ * y_,
         "xy1": 0.5 * (x_ * right[0, idx] + left[0, idx] * y_),
         "xy2": 0.5 * (x_ * right[1, idx] + left[1, idx] * y_)}
    acc = accumulate(cell[idx], np.minimum(kx, ky)[idx],
                     np.maximum(kx, ky)[idx], band[idx],
                     np.zeros(idx.size, np.int64), w, n_cells, 2,
                     len(edges) - 1)
    rows = class_pair_difference(acc, sd, groups, lk.classes,
                                 ("autosomes", "autosomes"),
                                 ("chrX", "chrX"), band_labels, n_boot, seed)
    A_ = len(band_labels) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        pc = {"pairs_autosomes": acc["n"][:, 0, 0, A_],
              "pairs_X": acc["n"][:, 1, 1, A_]}
        for tag, k in (("autosomes", 0), ("X", 1)):
            n = acc["n"][:, k, k, A_]
            dd = acc["xy"][:, k, k, A_] - acc["xy1"][:, k, k, A_]
            pc[f"delta_{tag}"] = 2 * dd / n
            pc[f"delta_r_{tag}"] = dd / n / (sd[:, k] ** 2)
    return rows, pd.DataFrame(pc)


def plot_chrx(tab: pd.DataFrame, pcell: pd.DataFrame, sex: pd.DataFrame,
              context: str, path, q_cut: float = 0.05):
    """a: ΔC by distance, autosomes (solid) and chrX (dashed), per group; b:
    chrX less autosomes, all distances, per group (filled: q < q_cut); c:
    each cell's own chrX less autosomes Δr; d: the cells' X dosage and the
    sex called from it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    t = tab[(tab["context"] == context) & (tab["statistic"] == "delta")]
    if t.empty:
        return
    bands = list(dict.fromkeys(t["band"]))
    all_b = next((b for b in bands if str(b).startswith("all")), bands[-1])
    bands = [b for b in bands if b != all_b]
    # the curves: each sex when there are two, else each cell type
    kind = ("sex" if (t["group_kind"] == "sex").any() else
            "cell type" if t.loc[t["group_kind"] == "cell type",
                                 "group"].nunique() <= 8 else "all")
    curves = list(dict.fromkeys(t.loc[t["group_kind"] == kind, "group"]))
    PP = 100
    with plt.rc_context(figure_style.RC):
        fig = plt.figure(figsize=(figure_style.FIG_W_IN, 2.4))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.1, 1.0, 1.0],
                              wspace=0.75)
        ax0, ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(4))
        xx = np.arange(len(bands))
        for gi, g in enumerate(curves):
            col = GROUP_PALETTE[gi % len(GROUP_PALETTE)]
            r = t[(t["group"] == g) & (t["group_kind"] == kind)
                  ].set_index("band").reindex(bands)
            ax0.plot(xx, PP * r["autosomes"], "-o", color=col, ms=2.5,
                     lw=0.8, label=f"{g}: autosomes")
            ax0.plot(xx, PP * r["chrX"], "--s", color=col, ms=2.5, lw=0.8,
                     label=f"{g}: chrX")
        ax0.axhline(0, color="0.6", lw=0.4)
        ax0.set_xticks(xx)
        ax0.set_xticklabels(bands, rotation=40, ha="right", fontsize=5)
        ax0.set_ylabel("ΔC, pp")
        f = t[t["band"] == all_b].reset_index(drop=True)
        yy = np.arange(len(f))
        for i, r in f.iterrows():
            q = r.get("q_diff", np.nan)
            ax1.plot([PP * r["diff_lo"], PP * r["diff_hi"]], [i, i],
                     color="0.2", lw=0.8)
            ax1.plot([PP * r["diff"]], [i], "o", ms=3, mec="0.1",
                     mfc="0.1" if np.isfinite(q) and q < q_cut else "white")
        ax1.axvline(0, color="0.6", lw=0.4)
        ax1.set_yticks(yy)
        ax1.set_yticklabels([f"{g} ({n:,})" for g, n in
                             zip(f["group"], f["cells"])], fontsize=5)
        ax1.set_ylim(len(f) - 0.5, -0.5)
        ax1.set_xlabel("chrX − autosomes, ΔC pp")
        pc = pcell[pcell["context"] == context]
        if not pc.empty:
            by = "sex" if kind == "sex" else "cell_type"
            for gi, (g, x) in enumerate(pc.groupby(by, sort=False)):
                v = PP * (x["delta_r_X"] - x["delta_r_autosomes"])
                v = v[np.isfinite(v)]
                if v.size:
                    ax2.hist(v, bins=30, histtype="step", lw=0.8,
                             color=GROUP_PALETTE[gi % len(GROUP_PALETTE)],
                             label=f"{g}: {np.mean(v > 0):.0%} > 0")
            ax2.axvline(0, color="0.6", lw=0.4)
            ax2.legend(fontsize=4.5, frameon=False, loc="upper left")
        ax2.set_xlabel("per cell:\nchrX − autosomes, Δr pp")
        if sex is not None and not sex.empty:
            for s_, col in (("female", "#c51b7d"), ("male", "#2166ac"),
                            ("unknown", "0.6")):
                x = sex[sex["sex_call"] == s_]
                if len(x):
                    ax3.scatter(x["x_dosage"], np.log10(x["y_share"] + 1e-6),
                                s=3, lw=0, color=col,
                                label=f"{s_} ({len(x):,})")
            ax3.legend(fontsize=4.5, frameon=False)
        ax3.set_xlabel("X dosage\n(X / autosome coverage)")
        ax3.set_ylabel("log10 share of pairs on Y")
        for ax, lab in zip((ax0, ax1, ax2, ax3), "abcd"):
            ax.set_title(lab, loc="left", fontweight="bold")
        _suptitle_above(fig, f"{context}: chrX against the autosomes -- "
                             f"contacts on one copy of X, partners "
                             f"reshuffled from either")
        from matplotlib.lines import Line2D
        hd = [Line2D([], [], color=GROUP_PALETTE[gi % len(GROUP_PALETTE)],
                     lw=0.8) for gi in range(len(curves))]
        hd += [Line2D([], [], color="0.3", lw=0.8, ls="-", marker="o",
                      ms=2.5),
               Line2D([], [], color="0.3", lw=0.8, ls="--", marker="s",
                      ms=2.5)]
        _legend_below(fig, hd, [str(g) for g in curves]
                      + ["autosomes (panel a)", "chrX (panel a)"],
                      ncol=min(len(curves) + 2, 6))
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def imprinted_domains(genes: pd.DataFrame, names=None,
                      merge: int = 500_000, flank: int = 50_000,
                      exclude=()) -> pd.DataFrame:
    """Imprinted domains from a gene annotation: the genes named in
    ``names`` (case-blind; IMPRINTED_GENES), within ``merge`` bp of each
    other on a chromosome merged into one domain, ``flank`` bp added on
    each side.  Each is named after its first and last imprinted gene."""
    want = {x.lower() for x in (IMPRINTED_GENES if names is None
                                else names)}
    g = genes[genes["name"].astype(str).str.lower().isin(want)].copy()
    g = g[~g["chrom"].map(_canon).isin(exclude)]
    rows = []
    for c, x in g.sort_values(["chrom", "start"]).groupby("chrom", sort=False):
        st, en = x["start"].to_numpy(), x["end"].to_numpy()
        nm = x["name"].astype(str).to_numpy()
        new = np.r_[True, st[1:] > np.maximum.accumulate(en)[:-1] + merge]
        for s_, e_ in zip(np.flatnonzero(new), np.r_[np.flatnonzero(new)[1:],
                                                      st.size]):
            gs = list(dict.fromkeys(nm[s_:e_]))
            rows.append({"chrom": c, "start": max(int(st[s_:e_].min())
                                                  - flank, 0),
                         "end": int(en[s_:e_].max()) + flank,
                         "name": gs[0] if len(gs) == 1 else
                         f"{gs[0]}–{gs[-1]}",
                         "genes": ",".join(gs)})
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "name",
                                       "genes"])


#: The clustered protocadherins' parts, by gene name (case-blind).
PCDH_PARTS = {"Pcdha": r"^pcdha", "Pcdhb": r"^pcdhb", "Pcdhg": r"^pcdhg"}


def pcdh_domains(genes: pd.DataFrame, flank: int = 50_000) -> pd.DataFrame:
    """The clustered protocadherins: the whole cluster and its α, β and γ
    parts (PCDH_PARTS: Pcdha*, Pcdhb*, Pcdhg*, or the human PCDHA* ...),
    each the span of its genes on the chromosome most of them are on, with
    ``flank`` bp on each side -- the α part's takes in the HS5-1 enhancer
    just past it.  The non-clustered ones (Pcdh9, Pcdh17 ...) are not
    matched."""
    nm = genes["name"].astype(str).str.lower()
    part = {k: genes[nm.str.match(rx)] for k, rx in PCDH_PARTS.items()}
    every = pd.concat(part.values())
    cols = ["chrom", "start", "end", "name", "genes"]
    if every.empty:
        return pd.DataFrame(columns=cols)
    chrom = every["chrom"].value_counts().index[0]
    rows = []
    for k, g in [("Pcdh cluster", every)] + list(part.items()):
        g = g[g["chrom"] == chrom].sort_values("start")
        if g.empty:
            continue
        rows.append({"chrom": chrom,
                     "start": max(int(g["start"].min()) - flank, 0),
                     "end": int(g["end"].max()) + flank, "name": k,
                     "genes": ",".join(dict.fromkeys(g["name"].astype(str)))})
    return pd.DataFrame(rows, columns=cols)


def domain_levels(lv, names, groups: dict, pairs=(), n_boot: int = 200,
                  seed: int = 0):
    """Each domain's 1D level -- the mean posterior of the called ends in
    it, every read pair's (``lv``: cells x domains x (count, sum)) -- per
    group of cells with a 95% bootstrap interval over the cells, and per
    pair of groups (A - B) the difference with its interval and bootstrap
    p.  Concordance can differ between groups because the domain is more
    open (or methylated) in one; this says whether it is.  Returns (rows,
    pair rows)."""
    rng = np.random.default_rng(seed)
    rows, crows, boot = [], [], {}
    for g, ix in groups.items():
        ix = np.asarray(ix, np.int64)
        if not ix.size:
            continue
        W = rng.multinomial(ix.size, np.full(ix.size, 1.0 / ix.size),
                            size=n_boot).astype(float)
        n_, s_ = lv[ix, :, 0], lv[ix, :, 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            v = s_.sum(0) / n_.sum(0)
            b = (W @ s_) / (W @ n_)
        boot[g] = (v, b)
        for r, nm in enumerate(names):
            with np.errstate(invalid="ignore"):
                lo, hi = np.nanpercentile(b[:, r], [2.5, 97.5])
            rows.append({"group": g, "domain": nm, "level": float(v[r]),
                         "level_lo": float(lo), "level_hi": float(hi),
                         "level_ends": int(n_[:, r].sum())})
    for a_, b_ in pairs:
        if a_ not in boot or b_ not in boot:
            continue
        (va, ba), (vb, bb) = boot[a_], boot[b_]
        for r, nm in enumerate(names):
            dif = ba[:, r] - bb[:, r]
            with np.errstate(invalid="ignore"):
                lo, hi = np.nanpercentile(dif, [2.5, 97.5])
            crows.append({"group_A": a_, "group_B": b_, "domain": nm,
                          "level_diff": float(va[r] - vb[r]),
                          "level_diff_lo": float(lo),
                          "level_diff_hi": float(hi),
                          "level_p": _boot_p(dif)})
    return rows, crows


def read_domains(path) -> pd.DataFrame:
    """A BED of domains: chrom, start, end, and a name (4th column; else
    chrom:start-end)."""
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     compression="infer", dtype={0: str})
    nm = (df[3].astype(str) if df.shape[1] > 3 else
          df[0].astype(str) + ":" + df[1].astype(str) + "-"
          + df[2].astype(str))
    return pd.DataFrame({"chrom": df[0].astype(str),
                         "start": df[1].astype(int),
                         "end": df[2].astype(int), "name": nm,
                         "genes": nm})


def _vs_windows(x, cv, se: float = 0.0, tested: bool = True) -> dict:
    """A domain's value against its matched windows' values ``cv``: their
    mean and SD, and z over both the windows' spread and the domain's own
    standard error ``se`` -- the windows have their minimum of contacts,
    a sparse domain far fewer, so their spread alone understates its noise
    -- with a normal two-sided p from z (the windows are too few for an
    empirical p to pass a correction over domains) and the empirical one.
    Not ``tested`` (too few contacts): no z or p."""
    from scipy.stats import norm
    cv = cv[np.isfinite(cv)]
    out = {"windows": int(cv.size), "windows_mean": np.nan,
           "windows_sd": np.nan, "z": np.nan, "p": np.nan,
           "p_empirical": np.nan}
    if cv.size < 5 or not np.isfinite(x):
        return out
    m, s = float(cv.mean()), float(np.std(cv, ddof=1))
    out.update({"windows_mean": m, "windows_sd": s})
    tot = float(np.sqrt(s ** 2 + (se if np.isfinite(se) else 0.0) ** 2))
    if tested and tot > 0:
        out["z"] = (x - m) / tot
        out["p"] = float(2 * norm.sf(abs(out["z"])))
        out["p_empirical"] = float(min(1.0, 2 * min(np.mean(cv >= x),
                                                    np.mean(cv <= x))
                                       + 1.0 / (cv.size + 1)))
    return out


def domain_concordance(domains: pd.DataFrame, g_lo, g_hi, cell, d, c,
                       n_cells: int, groups: dict, pairs=(),
                       tss_g=None, pair=None, n_controls: int = 100,
                       min_contacts: int = 30, n_boot: int = 200,
                       seed: int = 0, pooled: bool = True):
    """Concordance inside each domain against matched windows elsewhere.

    The contacts with both ends in a domain, from the per-contact parts of
    ΔC (``d``) and its control (``c``) -- the hotspots' (:func:`locus_
    hotspots`) -- pooled over each group's cells (``groups``: {name: cell
    indices}) with a 95% bootstrap interval over the cells.  Against
    ``n_controls`` windows of the domain's length, on the autosomes and
    away from every domain, with ``min_contacts`` contacts, most like the
    domain in the chromatin its contacts join -- the share of its contacts
    in each class pair of the system (``pair``), L1 distance -- and in TSS
    count (``tss_g``, genome coordinates; |log ratio|): the excess over the
    windows' mean, z and p (:func:`_vs_windows`; a domain with fewer than
    ``min_contacts`` contacts is not tested).  Also, unless not
    ``pooled`` (domains that overlap), every domain pooled against every
    window pooled ('all domains'), and per pair of groups
    (A - B) the same against the windows' differences.  A contact's two
    ends lie on one copy of the chromosome, so where the two parental
    copies differ -- an imprinted domain -- its ends agree beyond
    reshuffled partners, which come from either copy.  Returns (rows,
    contrast rows, the windows)."""
    rng = np.random.default_rng(seed)
    o = np.argsort(g_lo, kind="stable")
    gl, gh = np.asarray(g_lo, np.int64)[o], np.asarray(g_hi, np.int64)[o]
    ce = np.asarray(cell, np.int64)[o]
    dd, cc = np.asarray(d, float)[o], np.asarray(c, float)[o]
    pr = np.asarray(pair, np.int64)[o] if pair is not None else None
    n_pr = int(pr.max()) + 1 if pr is not None and pr.size else 1

    def mix(ix):
        """The share of these contacts in each class pair."""
        if pr is None or not ix.size:
            return np.zeros(n_pr)
        return np.bincount(pr[ix], minlength=n_pr) / ix.size

    def inside(s, e):
        i0, i1 = np.searchsorted(gl, [s, e])
        return i0 + np.flatnonzero(gh[i0:i1] < e)

    def per_cell(ix):
        return (np.bincount(ce[ix], weights=dd[ix], minlength=n_cells),
                np.bincount(ce[ix], minlength=n_cells).astype(float),
                np.bincount(ce[ix], weights=cc[ix], minlength=n_cells))
    dm = domains.reset_index(drop=True)
    code = _codes_of(dm["chrom"])
    g0 = code * STRIDE + dm["start"].to_numpy(np.int64)
    g1 = code * STRIDE + dm["end"].to_numpy(np.int64)
    # where windows may fall: the autosomes' extent, away from every domain
    ch = gl // STRIDE
    keys = np.array([k for k in np.unique(ch) if 1 <= k < 1000], np.int64)
    e0 = np.array([gl[ch == k].min() for k in keys], np.int64)
    e1 = np.array([gh[ch == k].max() for k in keys], np.int64)
    od = np.argsort(g0)
    d0, d1 = g0[od], np.maximum.accumulate(g1[od])
    tss = np.sort(np.asarray(tss_g, np.int64)) if tss_g is not None else None

    def n_tss(s, e):
        return (np.searchsorted(tss, e) - np.searchsorted(tss, s)
                if tss is not None else np.zeros(np.size(s)))
    S, N, T, owner, wins = [], [], [], [], []
    for r in range(len(dm)):
        here = inside(g0[r], g1[r])
        for x, v in zip((S, N, T), per_cell(here)):
            x.append(v)
        owner.append(-1)
        L = int(g1[r] - g0[r])
        room = np.maximum(e1 - e0 - L, 0).astype(float)
        if not keys.size or room.sum() <= 0:
            continue
        m = 30 * n_controls
        ki = rng.choice(keys.size, m, p=room / room.sum())
        st = e0[ki] + (rng.random(m) * room[ki]).astype(np.int64)
        en = st + L
        j = np.searchsorted(d0, en)
        free = ~((j > 0) & (d1[np.maximum(j - 1, 0)] > st))
        st, en = st[free], en[free]
        cand = [inside(a_, b_) for a_, b_ in zip(st, en)]
        good = np.flatnonzero(np.array([x.size for x in cand], int)
                              >= min_contacts)
        if not good.size:
            continue
        t_d = n_tss(g0[r:r + 1], g1[r:r + 1])[0]
        t_w = n_tss(st[good], en[good])
        m_d = mix(here)
        gap = (np.abs(np.log1p(t_w) - np.log1p(t_d))
               + np.array([np.abs(mix(cand[k]) - m_d).sum() for k in good]))
        for k in good[np.lexsort((rng.random(good.size), gap))[:n_controls]]:
            for x, v in zip((S, N, T), per_cell(cand[k])):
                x.append(v)
            owner.append(r)
            wins.append({"domain": dm.at[r, "name"],
                         "code": int(st[k] // STRIDE),
                         "start": int(st[k] % STRIDE),
                         "end": int(en[k] % STRIDE),
                         "contacts": int(cand[k].size),
                         "tss": int(n_tss(st[k:k + 1], en[k:k + 1])[0]),
                         "class_mix_distance": float(
                             np.abs(mix(cand[k]) - m_d).sum())})
    S, N, T, owner = np.array(S), np.array(N), np.array(T), np.array(owner)
    dom = np.flatnonzero(owner < 0)             # domain r -> its row
    ctl = owner >= 0
    rows, crows, vals = [], [], {}
    for gname, ix in groups.items():
        ix = np.asarray(ix, np.int64)
        if not ix.size:
            continue
        W = rng.multinomial(ix.size, np.full(ix.size, 1.0 / ix.size),
                            size=n_boot).astype(float)
        Sd, Nd = S[dom][:, ix], N[dom][:, ix]
        with np.errstate(invalid="ignore", divide="ignore"):
            v = S[:, ix].sum(1) / N[:, ix].sum(1)
            vc = T[dom][:, ix].sum(1) / Nd.sum(1)
            bd = (W @ Sd.T) / (W @ Nd.T)
            b_dom = (W @ Sd.sum(0)) / (W @ Nd.sum(0))
            b_win = (W @ S[ctl][:, ix].sum(0)) / (W @ N[ctl][:, ix].sum(0))
        vals[gname] = (v, bd, Nd.sum(1))
        for r in range(len(dm)):
            x = float(v[dom[r]])
            with np.errstate(invalid="ignore"):
                se_ = float(np.nanstd(bd[:, r], ddof=1))
            vw = _vs_windows(x, v[owner == r], se_,
                             Nd[r].sum() >= min_contacts)
            with np.errstate(invalid="ignore"):
                lo, hi = np.nanpercentile(bd[:, r], [2.5, 97.5])
            rows.append({
                "group": gname, "domain": dm.at[r, "name"],
                "chrom": dm.at[r, "chrom"], "start": int(dm.at[r, "start"]),
                "end": int(dm.at[r, "end"]), "genes": dm.at[r, "genes"],
                "contacts": int(Nd[r].sum()), "cells": int((Nd[r] > 0).sum()),
                "delta": x, "lo": float(lo), "hi": float(hi),
                "control": float(vc[r]), **vw,
                "excess": x - vw["windows_mean"],
                "excess_lo": lo - vw["windows_mean"],
                "excess_hi": hi - vw["windows_mean"]})
        if not pooled:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            x = float(Sd.sum() / Nd.sum())
            w_ = float(S[ctl][:, ix].sum() / N[ctl][:, ix].sum())
            lo, hi = np.nanpercentile(b_dom, [2.5, 97.5])
            elo, ehi = np.nanpercentile(b_dom - b_win, [2.5, 97.5])
        rows.append({"group": gname, "domain": "all domains", "chrom": "",
                     "start": -1, "end": -1, "genes": "",
                     "contacts": int(Nd.sum()),
                     "cells": int((Nd.sum(0) > 0).sum()), "delta": x,
                     "lo": float(lo), "hi": float(hi), "control": np.nan,
                     "windows": int(ctl.sum()), "windows_mean": w_,
                     "windows_sd": np.nan, "z": np.nan,
                     "p": _boot_p(b_dom - b_win), "p_empirical": np.nan,
                     "excess": x - w_, "excess_lo": float(elo),
                     "excess_hi": float(ehi)})
    for a_, b_ in pairs:
        if a_ not in vals or b_ not in vals:
            continue
        (va, bda, na_), (vb, bdb, nb_) = vals[a_], vals[b_]
        dv = va - vb
        for r in range(len(dm)):
            x = float(dv[dom[r]])
            with np.errstate(invalid="ignore"):
                se_ = float(np.nanstd(bda[:, r] - bdb[:, r], ddof=1))
            vw = _vs_windows(x, dv[owner == r], se_,
                             min(na_[r], nb_[r]) >= min_contacts)
            with np.errstate(invalid="ignore"):
                lo, hi = np.nanpercentile(bda[:, r] - bdb[:, r], [2.5, 97.5])
            crows.append({
                "group_A": a_, "group_B": b_, "domain": dm.at[r, "name"],
                "chrom": dm.at[r, "chrom"], "start": int(dm.at[r, "start"]),
                "end": int(dm.at[r, "end"]), "genes": dm.at[r, "genes"],
                "diff": x, "lo": float(lo), "hi": float(hi), **vw,
                "excess": x - vw["windows_mean"],
                "excess_lo": lo - vw["windows_mean"],
                "excess_hi": hi - vw["windows_mean"]})
    return rows, crows, pd.DataFrame(wins)


def plot_domains(tab: pd.DataFrame, ctab: pd.DataFrame, context: str,
                 path, what: str = "imprinted domains",
                 mark_neuron: bool = True, show_groups=(),
                 q_cut: float = 0.05):
    """a: per domain, ΔC of all cells (black) and of ``show_groups``, with
    95% intervals, against the matched windows (grey: their mean ± SD;
    filled: q < q_cut); b: per pair of groups, the domain's difference
    against the windows' differences; c, with levels in ``tab``: each
    group's mean posterior in the domain (accessibility or methylation).
    With ``mark_neuron``, † marks the domains with neuron-specific
    imprinting (NEURON_IMPRINTED)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from nome_hic_concordance import figure_style
    t = tab[tab["context"] == context]
    if t.empty:
        return
    doms = (["all domains"] if (t["domain"] == "all domains").any() else []
            ) + list(dict.fromkeys(t.loc[t["domain"] != "all domains",
                                         "domain"]))
    gen = t.drop_duplicates("domain").set_index("domain")["genes"]
    neuro = {x.lower() for x in NEURON_IMPRINTED}

    def label(dmn):
        gs = {x.lower() for x in str(gen.get(dmn, "")).split(",")}
        return dmn + (" †" if mark_neuron and gs & neuro else "")
    c_ = (ctab[ctab["context"] == context] if ctab is not None
          and not ctab.empty else pd.DataFrame())
    pairs = (list(dict.fromkeys(zip(c_["group_A"], c_["group_B"])))
             if not c_.empty else [])
    show = ["all cells"] + [g for g in show_groups if g != "all cells"
                            and g in set(t["group"])]
    has_lv = "level" in t and t["level"].notna().any()
    ny = len(doms)
    PP = 100
    with plt.rc_context(figure_style.RC):
        ncol = 1 + (1 if pairs else 0) + (1 if has_lv else 0)
        fig = plt.figure(figsize=(figure_style.FIG_W_IN, 1.0 + 0.2 * ny))
        gs = fig.add_gridspec(1, ncol, wspace=0.08)
        ax0 = fig.add_subplot(gs[0, 0])
        off = (np.arange(len(show)) - (len(show) - 1) / 2) * min(
            0.6 / max(len(show) - 1, 1), 0.22)
        hd, lb = [], []
        for gi, g in enumerate(show):
            col = "k" if g == "all cells" else GROUP_PALETTE[
                (gi - 1) % len(GROUP_PALETTE)]
            x = t[t["group"] == g].set_index("domain")
            for i, dmn in enumerate(doms):
                if dmn not in x.index:
                    continue
                r = x.loc[dmn]
                if g == "all cells" and np.isfinite(r["windows_mean"]):
                    m_, s_ = PP * r["windows_mean"], PP * r["windows_sd"]
                    ax0.add_patch(plt.Rectangle(
                        (m_ - s_, i - 0.4), 2 * s_ if np.isfinite(s_) else 0,
                        0.8, color="0.88", lw=0, zorder=0))
                    ax0.plot([m_, m_], [i - 0.4, i + 0.4], color="0.6",
                             lw=0.6, zorder=1)
                y = i + off[gi]
                q = r.get("q", np.nan)
                ax0.plot([PP * r["lo"], PP * r["hi"]], [y, y], color=col,
                         lw=0.7)
                ax0.plot([PP * r["delta"]], [y], "o", ms=2.8, mec=col,
                         mfc=col if np.isfinite(q) and q < q_cut else "white")
            n_ = int(t.loc[t["group"] == g, "cells"].max())
            hd.append(Line2D([], [], color=col, marker="o", ms=2.8, lw=0.7))
            lb.append(f"{g} ({n_:,} cells)")
        hd.append(plt.Rectangle((0, 0), 1, 1, color="0.88"))
        lb.append("matched windows, mean ± SD")
        ax0.axvline(0, color="0.6", lw=0.4)
        ax0.set_ylim(ny - 0.5, -0.5)
        ax0.set_yticks(np.arange(ny))
        ax0.set_yticklabels([label(x) for x in doms], fontsize=5.5)
        ax0.set_xlabel("ΔC within the domain, pp")
        ax0.set_title("a", loc="left", fontweight="bold")
        if pairs:
            ax1 = fig.add_subplot(gs[0, 1])
            offp = (np.arange(len(pairs)) - (len(pairs) - 1) / 2) * min(
                0.6 / max(len(pairs) - 1, 1), 0.22)
            for pi, (ga, gb) in enumerate(pairs):
                col = GROUP_PALETTE[pi % len(GROUP_PALETTE)]
                x = c_[(c_["group_A"] == ga) & (c_["group_B"] == gb)
                       ].set_index("domain")
                for i, dmn in enumerate(doms):
                    if dmn not in x.index:
                        continue
                    r = x.loc[dmn]
                    y = i + offp[pi]
                    if np.isfinite(r["windows_mean"]):
                        m_, s_ = PP * r["windows_mean"], PP * r["windows_sd"]
                        ax1.plot([m_ - s_, m_ + s_], [y, y], color="0.8",
                                 lw=2.2, solid_capstyle="butt", zorder=0)
                    q = r.get("q", np.nan)
                    ax1.plot([PP * r["lo"], PP * r["hi"]], [y, y], color=col,
                             lw=0.7)
                    ax1.plot([PP * r["diff"]], [y], "o", ms=2.8, mec=col,
                             mfc=col if np.isfinite(q) and q < q_cut
                             else "white")
                hd.append(Line2D([], [], color=col, marker="o", ms=2.8,
                                 lw=0.7))
                lb.append(f"{ga} − {gb} (grey: matched windows)")
            ax1.axvline(0, color="0.6", lw=0.4)
            ax1.set_ylim(ny - 0.5, -0.5)
            ax1.set_yticks(np.arange(ny))
            ax1.set_yticklabels([])
            ax1.set_xlabel("difference, pp")
            ax1.set_title("b", loc="left", fontweight="bold")
        if has_lv:
            ax2 = fig.add_subplot(gs[0, ncol - 1])
            for gi, g in enumerate(show):
                col = "k" if g == "all cells" else GROUP_PALETTE[
                    (gi - 1) % len(GROUP_PALETTE)]
                x = t[t["group"] == g].set_index("domain")
                for i, dmn in enumerate(doms):
                    if dmn not in x.index or not np.isfinite(
                            x.loc[dmn, "level"]):
                        continue
                    r = x.loc[dmn]
                    y = i + off[gi]
                    ax2.plot([PP * r["level_lo"], PP * r["level_hi"]], [y, y],
                             color=col, lw=0.7)
                    ax2.plot([PP * r["level"]], [y], "s", ms=2.6, color=col)
            ax2.set_ylim(ny - 0.5, -0.5)
            ax2.set_yticks(np.arange(ny))
            ax2.set_yticklabels([])
            ax2.set_xlabel(f"{context} level in the domain, %")
            ax2.set_title("b" if not pairs else "c", loc="left",
                          fontweight="bold")
        _suptitle_above(fig, f"{context}: concordance inside {what} "
                             f"(filled: q < {q_cut:g}"
                             + ("; † neuron-specific imprinting)"
                                if mark_neuron else ")"))
        _legend_below(fig, hd, lb, ncol=3)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


# ------------------------------------------- variable and stable loci --


def locus_variability(grid, g_lo, g_hi, ctype_contact, parts: dict,
                      type_names, min_contacts: int = 100, min_types: int = 3,
                      fraction: float = 0.1, fdr: float = 0.05,
                      min_z: float = 3.0) -> pd.DataFrame:
    """Which loci are concordant in one cell type and not another, and which
    the same in every one?

    Per bin of ``grid`` (all distances) and statistic (``parts``: {name:
    (each contact's part, the control's)}, a contact counted in both its
    ends' bins, once when they share one): each cell type's mean with its
    standard error (types with >= ``min_contacts`` contacts in the bin), the
    pooled value over every type, and the between-type variance beyond
    sampling -- DerSimonian-Laird tau^2 from Cochran's Q -- with a
    heterogeneity q against a null fitted to the controls' Q
    (:func:`_empirical_q`).  Bins with ``min_types`` types are tested.

      variable  heterogeneous (q < ``fdr``), the ``fraction`` of tested bins
                with the largest between-type SD
      stable    concordant (|pooled z| >= ``min_z``), the ``fraction`` of
                tested bins whose between-type SD is smallest relative to
                their concordance -- concordance the same in every type;
                bins without any would be 'stable' trivially

    A cell type whose concordance is higher everywhere -- labeling, depth,
    or a global state -- makes the most concordant bins the most 'variable'
    with the same order of types in each.  ``set_scaled`` is the variable
    set after each type is divided by its genome-wide level (its contacts'
    mean |value| over the tested bins, relative to the types' mean): bins
    whose pattern of types differs from the types' overall one."""
    T = len(type_names)
    i_lo, i_hi = grid.index_g(g_lo), grid.index_g(g_hi)
    sel_lo, sel_hi = i_lo >= 0, (i_hi >= 0) & (i_hi != i_lo)
    ct = np.asarray(ctype_contact, np.int64)
    out = []
    for stat, (v, vc) in parts.items():
        v, vc = np.asarray(v, float), np.asarray(vc, float)

        def acc_(w=None):
            o = np.zeros(grid.n * T)
            for idx, sel in ((i_lo, sel_lo), (i_hi, sel_hi)):
                o += np.bincount(idx[sel] * T + ct[sel],
                                 weights=None if w is None else w[sel],
                                 minlength=grid.n * T)
            return o.reshape(grid.n, T)
        n, s1, s2, c1, c2 = acc_(), acc_(v), acc_(v * v), acc_(vc), acc_(vc * vc)
        with np.errstate(invalid="ignore", divide="ignore"):
            x, xc = s1 / n, c1 / n
            se = np.sqrt(np.maximum(s2 / n - x ** 2, 0) / n)
            sec = np.sqrt(np.maximum(c2 / n - xc ** 2, 0) / n)
            ok = (n >= min_contacts) & (se > 0) & (sec > 0)
            k = ok.sum(1)
            test = k >= min_types

            def het(val, err):
                w = np.where(ok, 1.0 / np.where(ok, err, 1) ** 2, 0.0)
                sw = w.sum(1)
                mw = (w * np.where(ok, val, 0)).sum(1) / np.where(sw > 0, sw,
                                                                  1)
                Q = (w * np.where(ok, val - mw[:, None], 0) ** 2).sum(1)
                df = np.maximum(k - 1, 1)
                den = sw - (w ** 2).sum(1) / np.where(sw > 0, sw, 1)
                tau2 = np.maximum((Q - df) / np.where(den > 0, den, np.inf), 0)
                return Q, df, tau2, (Q - df) / np.sqrt(2 * df)
            Q, df, tau2, zq = het(x, se)
            _, _, _, zq0 = het(xc, sec)
            N = n.sum(1)
            pooled = s1.sum(1) / N
            pse = np.sqrt(np.maximum(s2.sum(1) / N - pooled ** 2, 0) / N)
            zp = pooled / pse
        b = np.flatnonzero(test)
        if b.size == 0:
            continue
        # each type's genome-wide level, for the scaled call
        with np.errstate(invalid="ignore", divide="ignore"):
            wt = np.where(ok[b], n[b], 0.0)
            lvl = (wt * np.abs(np.where(ok[b], x[b], 0))).sum(0) / wt.sum(0)
            sc = lvl / np.nanmean(lvl[lvl > 0]) if (lvl > 0).any() else lvl
            sc = np.where(np.isfinite(sc) & (sc > 0), sc, 1.0)
            Qs, _, tau2s, zqs = het(x / sc, se / sc)
            _, _, _, zqs0 = het(xc / sc, sec / sc)
        hqs = _empirical_q(zqs[b], zqs0[b])
        hq = _empirical_q(zq[b], zq0[b])
        tau = np.sqrt(tau2[b])
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = tau / np.abs(pooled[b])
        m = int(round(fraction * b.size))
        label = np.full(b.size, "", object)
        var_c = np.flatnonzero(hq < fdr)
        top = var_c[np.argsort(-tau[var_c], kind="mergesort")][:m]
        label[top] = "variable"
        st_c = np.flatnonzero((np.abs(zp[b]) >= min_z) & (label == ""))
        low = st_c[np.argsort(cv[st_c], kind="mergesort")][:m]
        label[low] = "stable"
        taus = np.sqrt(tau2s[b])
        label_s = np.full(b.size, "", object)
        vs_c = np.flatnonzero(hqs < fdr)
        label_s[vs_c[np.argsort(-taus[vs_c], kind="mergesort")][:m]] = \
            "variable"
        df_ = pd.DataFrame({
            "grid_index": b, "code": grid.codes[b // grid.per],
            "start": (b % grid.per) * grid.size,
            "end": (b % grid.per + 1) * grid.size, "statistic": stat,
            "types_tested": k[b], "contacts": N[b], "pooled": pooled[b],
            "z_pooled": zp[b], "between_type_sd": tau, "cv": cv,
            "Q": Q[b], "df": df[b], "het_q": hq, "set": label,
            "between_type_sd_scaled": taus, "het_q_scaled": hqs,
            "set_scaled": label_s})
        for j, nm in enumerate(type_names):
            df_[f"type:{nm}"] = np.where(ok[b, j], x[b, j], np.nan)
        out.append(df_)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


#: The variable set after each cell type's genome-wide level is divided out.
VARIABLE_SCALED = "variable beyond type level"


def variability_enrichment(lv: pd.DataFrame, tracks: dict) -> pd.DataFrame:
    """Per statistic, set (variable / stable) and annotation track: the
    track's mean over the set's bins against the rest of the tested bins
    (log2 fold), the share of bins it touches in each, and a two-sided
    Mann-Whitney p, BH q per statistic and set."""
    from scipy.stats import mannwhitneyu
    rows = []
    for (stat, ctx), g in lv.groupby(["statistic", "context"], sort=False):
        gi = g["grid_index"].to_numpy()
        sets = {"variable": (g["set"] == "variable").to_numpy(),
                "stable": (g["set"] == "stable").to_numpy()}
        if "set_scaled" in g:
            sets[VARIABLE_SCALED] = (g["set_scaled"] == "variable").to_numpy()
        for st, fg in sets.items():
            if fg.sum() < 3 or (~fg).sum() < 3:
                continue
            for name, v in tracks.items():
                x = v[gi]
                if np.ptp(x) == 0:
                    continue
                a_, b_ = x[fg], x[~fg]
                eps = 1e-6 + 1e-3 * float(np.mean(np.abs(x)))
                try:
                    p_ = float(mannwhitneyu(a_, b_,
                                            alternative="two-sided").pvalue)
                except ValueError:
                    p_ = np.nan
                rows.append({"context": ctx, "statistic": stat, "set": st,
                             "track": name, "bins": int(fg.sum()),
                             "mean_set": float(a_.mean()),
                             "mean_rest": float(b_.mean()),
                             "log2_fold": float(np.log2((a_.mean() + eps)
                                                        / (b_.mean() + eps))),
                             "share_set": float((a_ > 0).mean()),
                             "share_rest": float((b_ > 0).mean()), "p": p_})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = np.nan
        for _, g in out.groupby(["context", "statistic", "set"], sort=False):
            out.loc[g.index, "q"] = fdr_bh(g["p"].to_numpy())
    return out


def plot_locus_variability(lv: pd.DataFrame, en: pd.DataFrame, context: str,
                           stat: str, path, q_cut: float = 0.05,
                           n_heat: int = 60):
    """a: per bin, the pooled concordance against its between-type SD
    (pp), the variable and stable bins marked; b: the variable bins'
    concordance in each cell type; c: what each set is enriched in -- log2
    fold of every annotation track over the rest of the tested bins
    (* q < q_cut)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    a = lv[(lv["context"] == context) & (lv["statistic"] == stat)]
    if a.empty:
        return
    e = (en[(en["context"] == context) & (en["statistic"] == stat)]
         if not en.empty else en)
    tcols = [c for c in a.columns if c.startswith("type:")]
    PP = 100
    tr = (list(dict.fromkeys(e.sort_values("log2_fold")["track"]))
          if not e.empty else [])
    with plt.rc_context(figure_style.RC):
        fig = plt.figure(figsize=(figure_style.FIG_W_IN * 1.15,
                                  max(2.6, 0.9 + 0.13 * len(tr))))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.9, 1.2], wspace=0.9)
        ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))
        sv = a["set_scaled"] if "set_scaled" in a else pd.Series("", a.index)
        ax0.scatter(PP * a.loc[(sv == "variable") & (a["set"] != "variable"),
                               "pooled"],
                    PP * a.loc[(sv == "variable") & (a["set"] != "variable"),
                               "between_type_sd"], s=4, lw=0, color="#fc9272",
                    label=VARIABLE_SCALED + " only", zorder=2)
        for st, col, ms in (("", "0.75", 2), ("stable", "#2166ac", 4),
                            ("variable", "#b2182b", 4)):
            m = a["set"] == st
            ax0.scatter(PP * a.loc[m, "pooled"],
                        PP * a.loc[m, "between_type_sd"], s=ms, lw=0,
                        color=col, label=st or "other tested",
                        zorder=3 if st else 1)
        ax0.set_xlabel(f"{_STAT_LABEL.get(stat, stat)}, all types, pp")
        ax0.set_ylabel("between-type SD, pp")
        ax0.legend(fontsize=5, markerscale=2, loc="best")
        v = a[a["set"] == "variable"].sort_values("between_type_sd",
                                                  ascending=False).head(n_heat)
        if not v.empty and tcols:
            H = PP * v[tcols].to_numpy(float)
            o = np.lexsort((-np.nan_to_num(np.nanmax(H, 1)),
                            np.nanargmax(np.where(np.isfinite(H), H, -np.inf),
                                         1)))
            H = H[o]
            lim = float(np.nanmax(np.abs(H))) if np.isfinite(H).any() else 1
            im = ax1.imshow(H, cmap="RdBu_r", vmin=-lim, vmax=lim,
                            aspect="auto", interpolation="nearest")
            ax1.set_xticks(range(len(tcols)))
            ax1.set_xticklabels([c[5:] for c in tcols], rotation=60,
                                ha="right", fontsize=5)
            names_ = v["chrom"] if "chrom" in v else v["code"].astype(str)
            ax1.set_yticks(range(len(v)))
            ax1.set_yticklabels([f"{_canon(c)}:{s_ // 1_000_000}Mb" for c, s_
                                 in zip(np.asarray(names_)[o],
                                        v["start"].to_numpy()[o])],
                                fontsize=3.5)
            cb = fig.colorbar(im, ax=ax1, orientation="horizontal",
                              fraction=0.05, pad=0.22)
            cb.ax.tick_params(labelsize=5)
            cb.set_label("pp", fontsize=6)
        ax1.set_title(f"b  top {len(v)} variable bins, by cell type",
                      loc="left", fontsize=6.5)
        if tr:
            yy = np.arange(len(tr))
            cap = 4.0                         # a class absent from a set
            have = [x for x in (("variable", "#b2182b"),
                                (VARIABLE_SCALED, "#fc9272"),
                                ("stable", "#2166ac"))
                    if (e["set"] == x[0]).any()]
            hgt = 0.8 / max(len(have), 1)
            for i_, (st, col) in enumerate(have):
                dy = (i_ - (len(have) - 1) / 2) * hgt
                x_ = e[e["set"] == st].set_index("track").reindex(tr)
                f_all = x_["log2_fold"].to_numpy(float)
                ax2.barh(yy + dy, np.clip(f_all, -cap, cap), height=hgt,
                         color=col, label=st)
                for y_, (f_, q_) in enumerate(zip(f_all, x_["q"])):
                    if not np.isfinite(f_):
                        continue
                    fc = float(np.clip(f_, -cap, cap))
                    txt = ("<" if f_ < -cap else ">" if f_ > cap else "") + (
                        "*" if np.isfinite(q_) and q_ < q_cut else "")
                    if txt:
                        ax2.text(fc, y_ + dy, (" " + txt) if fc >= 0 else
                                 (txt + " "), va="center",
                                 ha="left" if fc >= 0 else "right",
                                 fontsize=5)
            ax2.set_xlim(-cap - 0.8, cap + 0.8)
            ax2.axvline(0, color="0.4", lw=0.4)
            ax2.set_yticks(yy)
            ax2.set_yticklabels(tr, fontsize=5)
            ax2.set_xlabel("log2 fold vs the other tested bins\n"
                           "(< >: beyond ±4)")
            ax2.legend(fontsize=5, loc="upper center",
                       bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
        ax0.set_title("a  concordance and how it varies", loc="left",
                      fontsize=6.5)
        ax2.set_title("c  what each set is enriched in", loc="left",
                      fontsize=6.5)
        fig.suptitle(f"{context} -- {_STAT_LABEL.get(stat, stat)}: loci that "
                     f"vary between cell types, and loci that do not "
                     f"(1 Mb bins)", fontsize=7)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


# --------------------------------------------------- cell x chromosome --


def cell_chromosome_test(g_lo, cell, ctype, d, n_perm: int, rng,
                         min_contacts: int = 50, pair=None) -> dict:
    """Does a cell's concordance differ between its own chromosomes beyond
    sampling -- a cell-specific state of whole territories?

    Per cell and chromosome with >= ``min_contacts`` contacts, the mean of
    the contacts' parts of ΔC (``d``).  An additive fit -- a cell effect
    plus a chromosome effect, least squares over the entries -- leaves each
    entry's cell x chromosome interaction plus sampling.  The null keeps
    every cell's own effect in each class pair (``pair``) -- a cell whose
    A-A contacts are more concordant is so on every chromosome, by however
    many A-A contacts each chromosome happens to give it: each contact's
    part, less its cell's mean in its class pair, is reassigned among the
    cells of a type within its chromosome and class pair, and the receiving
    cell's mean added back.  That leaves sampling (and how the types differ
    between chromosomes); the observed residual beyond it is the cell x
    chromosome part.  It is split in two, with at least five chromosomes:
    a cell effect scaled chromosome by chromosome (the leading rank-one
    term of the residual -- e.g. a cell's A-versus-B difference, carried
    by how pure each chromosome's A and B labels are), and what is beyond
    it, particular to a cell and a chromosome -- the part a territory or
    condensate state would show as.  Returns the entries' variance and its shares --
    chromosome, cell (the fitted effects), sampling, cell x chromosome --
    with a permutation p."""
    cell = np.asarray(cell, np.int64)
    ctype = np.asarray(ctype, np.int64)
    chrom = np.asarray(g_lo, np.int64) // STRIDE
    codes, ci = np.unique(chrom, return_inverse=True)
    K, C = codes.size, ctype.size
    pair = (np.zeros(cell.size, np.int64) if pair is None
            else np.unique(np.asarray(pair), return_inverse=True)[1])
    P_ = int(pair.max()) + 1 if pair.size else 1
    key = cell * K + ci
    N = np.bincount(key, minlength=C * K).reshape(C, K)
    ok = N >= min_contacts
    for _ in range(3):                      # cells and chromosomes enough
        ok &= (ok.sum(1) >= 2)[:, None] & (ok.sum(0) >= 10)[None, :]
    out = {"cells": int(ok.any(1).sum()), "chromosomes": int(ok.any(0).sum()),
           "entries": int(ok.sum())}
    if out["entries"] < 50:
        return out
    W = ok.astype(float)

    rank1 = int(ok.any(0).sum()) >= 5

    def fit(dv):
        S = np.bincount(key, weights=dv, minlength=C * K).reshape(C, K)
        Y = np.where(ok, S / np.maximum(N, 1), 0.0)
        a, b = np.zeros(C), np.zeros(K)
        for _ in range(50):
            b = (W * (Y - a[:, None])).sum(0) / np.maximum(W.sum(0), 1)
            a = (W * (Y - b[None, :])).sum(1) / np.maximum(W.sum(1), 1)
        R = (Y - a[:, None] - b[None, :]) * W
        r1 = np.nan
        if rank1:
            # the leading rank-one term of the residual, entries observed
            u = np.linalg.svd(R, full_matrices=False)[0][:, 0]
            for _ in range(30):
                v = (W * R * u[:, None]).sum(0) / np.maximum(
                    (W * u[:, None] ** 2).sum(0), 1e-12)
                u = (W * R * v[None, :]).sum(1) / np.maximum(
                    (W * v[None, :] ** 2).sum(1), 1e-12)
            R1 = (R - np.outer(u, v)) * W
            r1 = float((R1 ** 2).sum() / W.sum())
        return Y, a, b, float((R ** 2).sum() / W.sum()), r1
    d = np.asarray(d, float)
    Y, a, b, r_obs, r1_obs = fit(d)
    # each contact less its cell's mean in its class pair
    cp = cell * P_ + pair
    m_cp = (np.bincount(cp, weights=d, minlength=C * P_)
            / np.maximum(np.bincount(cp, minlength=C * P_), 1))[cp]
    dc = d - m_cp
    blk = (ctype[cell] * K + ci) * P_ + pair
    o = np.argsort(blk, kind="stable")
    bs = blk[o]
    st = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
    en = np.r_[st[1:], bs.size]
    multi = [(x, y) for x, y in zip(st, en) if y - x > 1]
    do = dc[o]
    null = []
    for _ in range(int(n_perm)):
        pidx = np.arange(bs.size)
        for x, y in multi:
            pidx[x:y] = x + rng.permutation(y - x)
        dp = np.empty_like(do)
        dp[o] = do[pidx]
        null.append(fit(dp + m_cp)[3:])
    null = np.asarray(null, float).reshape(-1, 2)
    null, null1 = null[:, 0], null[:, 1]
    yv = Y[ok]
    v = float(yv.var())
    out.update({"var_entries": v, "residual_var": r_obs})
    if v > 0:
        out["chromosome_share"] = float(np.broadcast_to(b[None, :], Y.shape)[
            ok].var() / v)
        out["cell_share"] = float(np.broadcast_to(a[:, None], Y.shape)[
            ok].var() / v)
    if null.size:
        out["null_residual_var"] = float(null.mean())
        out["null_residual_sd"] = float(null.std(ddof=1)) if null.size > 1 \
            else np.nan
        out["sampling_share"] = float(null.mean() / v) if v > 0 else np.nan
        out["cell_x_chromosome_share"] = (float(max(r_obs - null.mean(), 0)
                                                / v) if v > 0 else np.nan)
        out["perm_p"] = float((1 + (null >= r_obs).sum()) / (null.size + 1))
        if rank1 and v > 0 and np.isfinite(null1).all():
            idio = max(r1_obs - null1.mean(), 0) / v
            out["idiosyncratic_share"] = float(min(
                idio, out["cell_x_chromosome_share"]))
            out["scaled_share"] = float(out["cell_x_chromosome_share"]
                                        - out["idiosyncratic_share"])
            out["idiosyncratic_p"] = float(
                (1 + (null1 >= r1_obs).sum()) / (null1.size + 1))
    return out


def plot_cell_chromosome(tab: pd.DataFrame, path):
    """Per context, the variance of the cells' per-chromosome concordance
    split into chromosome, cell, sampling and cell x chromosome."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    if tab.empty or "sampling_share" not in tab:
        return
    tab = tab.copy()
    if "idiosyncratic_share" in tab:
        # where the split is known, the interaction in its two parts
        known = tab["idiosyncratic_share"].notna()
        tab.loc[known, "cell_x_chromosome_share"] = np.nan
    parts = (("chromosome_share", "#525252", "chromosome"),
             ("cell_share", "#08519c", "cell"),
             ("sampling_share", "0.85", "sampling (+ type x chromosome)"),
             ("cell_x_chromosome_share", "#cb181d", "cell x chromosome"),
             ("scaled_share", "#fc9272",
              "cell x chromosome: a cell effect, scaled"),
             ("idiosyncratic_share", "#99000d",
              "cell x chromosome: particular (territory?)"))
    parts = tuple(x for x in parts if x[0] in tab)
    with plt.rc_context(figure_style.RC):
        fig, ax = plt.subplots(figsize=(figure_style.FIG_W_IN * 0.6,
                                        0.8 + 0.3 * len(tab)))
        yy = np.arange(len(tab))
        W_ = np.column_stack([np.clip(np.nan_to_num(tab[c].to_numpy(float)),
                                      0, None) for c, _, _ in parts])
        W_ = W_ / np.maximum(W_.sum(1, keepdims=True), 1e-12)
        left = np.zeros(len(tab))
        for j, (_, c_, lab) in enumerate(parts):
            ax.barh(yy, W_[:, j], left=left, color=c_, label=lab, height=0.6)
            left += W_[:, j]
        for i, (_, r) in enumerate(tab.iterrows()):
            ax.text(1.01, i, f"p = {r.get('perm_p', np.nan):.3g}"
                    + (f"; particular p = {r['idiosyncratic_p']:.3g}"
                       if np.isfinite(r.get("idiosyncratic_p", np.nan))
                       else ""), va="center", fontsize=5)
        ax.set_yticks(yy)
        ax.set_yticklabels(tab["context"])
        ax.set_ylim(len(tab) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of the variance of per-cell, per-chromosome ΔC")
        ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1.3, 1.0),
                  borderaxespad=0)
        ax.set_title("Does a cell's concordance differ between its own "
                     "chromosomes?", loc="left", fontsize=6.5)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


# ---------------------------------------------------------- locus hotspots --

#: Mouse genes with established parent-of-origin (imprinted) expression and
#: an imprinting DMR at or near them.  One allele is methylated and the
#: other not, and a contact joins two loci on the same chromosome copy, so
#: allele-specific methylation reads as concordance -- in these loci first.
IMPRINTED_GENES = (
    "H19", "Igf2", "Igf2r", "Airn", "Kcnq1ot1", "Kcnq1", "Cdkn1c", "Phlda2",
    "Ascl2", "Peg3", "Zim1", "Zim2", "Snrpn", "Snurf", "Ube3a", "Ndn",
    "Magel2", "Mkrn3", "Peg12", "Mest", "Plagl1", "Grb10", "Meg3", "Rian",
    "Mirg", "Dlk1", "Rtl1", "Dio3", "Gnas", "Nnat", "Peg10", "Sgce", "Zrsr1",
    "Impact", "Rasgrf1", "Zdbf2", "Peg13", "Kcnk9", "Nap1l5", "Commd1",
    "Slc38a4", "Sfmbt2", "Mcts2", "H13", "Blcap",
    # imprinted in human, without a mouse name above
    "DIRAS3", "L3MBTL1", "ZNF597", "ZNF331", "HYMAI", "FAM50B", "MEG8",
    "NAA60", "SLC22A18", "GNAS-AS1", "PEG3-AS1", "SNHG14", "MAGEL2",
    "NAP1L5", "GPR1-AS1", "ZFP57")
#: Gene families tested as hotspot annotations, by gene name.
GENE_FAMILIES = {"protocadherin cluster": r"^pcdh[abg]",
                 "olfactory receptor genes": r"^(olfr\d|or\d+[a-z]+\d)",
                 "zinc-finger (Zfp / ZNF) genes": r"^(zfp\d|znf\d)"}


class BinGrid:
    """Genome bins of one size, as one dense index: chromosome (by code)
    x bin."""

    def __init__(self, codes, max_pos: int, size: int):
        self.codes = np.unique(np.asarray(list(codes), np.int64))
        self.size = int(size)
        self.per = int(max_pos // size) + 1
        self.n = self.codes.size * self.per

    def index_g(self, g) -> np.ndarray:
        """The bin of each genome coordinate; -1 off the grid."""
        g = np.asarray(g, np.int64)
        code = g // STRIDE
        if self.codes.size == 0:
            return np.full(g.size, -1, np.int64)
        r = np.minimum(np.searchsorted(self.codes, code), self.codes.size - 1)
        b = (g % STRIDE) // self.size
        ok = (self.codes[r] == code) & (b < self.per)
        return np.where(ok, r * self.per + b, -1)

    def coverage_g(self, g0, g1) -> np.ndarray:
        """Per bin, the share of it the intervals [g0, g1) cover (genome
        coordinates; overlaps are not merged here)."""
        g0, g1 = np.asarray(g0, np.int64), np.asarray(g1, np.int64)
        code = g0 // STRIDE
        out = np.zeros(self.n)
        if g0.size == 0 or self.codes.size == 0:
            return out
        r = np.minimum(np.searchsorted(self.codes, code), self.codes.size - 1)
        ok = (self.codes[r] == code) & (g1 > g0)
        st, en, r = g0[ok] % STRIDE, g1[ok] - code[ok] * STRIDE, r[ok]
        b0, b1 = st // self.size, (en - 1) // self.size
        span = b1 - b0 + 1
        rep = np.repeat(np.arange(st.size), span)
        b = b0[rep] + (np.arange(rep.size)
                       - np.repeat(np.cumsum(span) - span, span))
        bp = (np.minimum(en[rep], (b + 1) * self.size)
              - np.maximum(st[rep], b * self.size))
        keep = b < self.per
        np.add.at(out, r[rep][keep] * self.per + b[keep], bp[keep])
        return out / self.size

    def coverage(self, df: pd.DataFrame) -> np.ndarray:
        """:meth:`coverage_g` of a chrom / start / end table, merged."""
        if df is None or not len(df):
            return np.zeros(self.n)
        m = _merged(df.assign(chrom=df["chrom"].map(_canon)))
        c = _codes_of(m["chrom"].to_numpy()) * STRIDE
        return self.coverage_g(c + m["start"].to_numpy(np.int64),
                               c + m["end"].to_numpy(np.int64))

    def counts_g(self, g) -> np.ndarray:
        i = self.index_g(g)
        return np.bincount(i[i >= 0], minlength=self.n).astype(float)


def locus_hotspots(grid: BinGrid, g_lo, g_hi, cell, d, c, band,
                   n_band: int) -> pd.DataFrame:
    """Where concordance concentrates along the genome, pooled over cells.

    Per bin of ``grid`` and band (index ``n_band`` = all distances): the
    contacts with an end in the bin -- counted once when both ends are --
    their mean contribution to ΔC (``d``: the real partner's product less
    the reshuffled one's) and to the control (``c``: reshuffled draw 2 less
    draw 1), with per-contact standard errors and z-scores, and for 'all'
    the cells they come from."""
    i_lo, i_hi = grid.index_g(g_lo), grid.index_g(g_hi)
    nb = n_band + 1
    band = np.asarray(band, np.int64)
    sel_lo, sel_hi = i_lo >= 0, (i_hi >= 0) & (i_hi != i_lo)

    def acc(w=None):
        out = np.zeros(grid.n * nb)
        for idx, sel in ((i_lo, sel_lo), (i_hi, sel_hi)):
            out += np.bincount(idx[sel] * nb + band[sel],
                               weights=None if w is None else w[sel],
                               minlength=grid.n * nb)
        out = out.reshape(grid.n, nb)
        out[:, n_band] = out[:, :n_band].sum(1)
        return out
    n, s1, s2, t1, t2 = acc(), acc(d), acc(d * d), acc(c), acc(c * c)
    cell = np.asarray(cell, np.int64)
    o = (np.argsort(cell, kind="stable") if cell.size > 1
         and np.any(np.diff(cell) < 0) else np.arange(cell.size))
    cs, lo_, hi_ = cell[o], i_lo[o], i_hi[o]
    bounds = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1], True])
    n_cells = np.zeros(grid.n, np.int64)
    for a_, e_ in zip(bounds[:-1], bounds[1:]):
        u = np.unique(np.concatenate([lo_[a_:e_], hi_[a_:e_]]))
        n_cells[u[u >= 0]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        md, mc = s1 / n, t1 / n
        se = np.sqrt(np.maximum(s2 / n - md ** 2, 0) / n)
        sec = np.sqrt(np.maximum(t2 / n - mc ** 2, 0) / n)
    rows = []
    for b in range(nb):
        nz = np.flatnonzero(n[:, b] > 0)
        rows.append(pd.DataFrame({
            "grid_index": nz, "code": grid.codes[nz // grid.per],
            "start": (nz % grid.per) * grid.size,
            "end": (nz % grid.per + 1) * grid.size, "band_index": b,
            "contacts": n[nz, b].astype(np.int64),
            "cells": n_cells[nz] if b == n_band else -1,
            "delta": md[nz, b], "control": mc[nz, b], "se": se[nz, b],
            "se_control": sec[nz, b]}))
    out = pd.concat(rows, ignore_index=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["z"] = out["delta"] / out["se"]
        out["z_control"] = out["control"] / out["se_control"]
    return out


def _empirical_q(z, z0) -> np.ndarray:
    """BH q of each observed z (large = called) against an empirical null
    fitted to the control's z0: normal, centred at their median and as wide
    as their MAD says -- contacts of one cell and one locus are not
    independent, so the per-contact z is wider than N(0, 1), and the control
    shows by how much."""
    from scipy.stats import norm
    z = np.asarray(z, float)
    z0 = np.asarray(z0, float)
    z0 = z0[np.isfinite(z0)]
    q = np.full(z.size, np.nan)
    ok = np.isfinite(z)
    if not ok.any() or z0.size < 5:
        return q
    m0 = float(np.median(z0))
    s0 = max(float(1.4826 * np.median(np.abs(z0 - m0))), 1.0)
    q[ok] = fdr_bh(norm.sf((z[ok] - m0) / s0))
    return q


def call_hotspots(hs: pd.DataFrame, n_band: int, fdr: float,
                  min_contacts: int, min_cells: int = 10) -> pd.DataFrame:
    """Hot and cold bins of :func:`locus_hotspots` (per context, all
    distances): tested when they have ``min_contacts`` contacts from
    ``min_cells`` cells.  Concordance is positive over most of the genome,
    so a bin is hot or cold against the typical bin -- ``excess``, its ΔC
    less the median tested bin's, in z units ``z_excess`` -- called at an
    FDR against a null fitted to the control's same z (:func:`_empirical_q`):
    reshuffled draw 2 less draw 1 has the same per-contact dependence as
    the real partners, and no concordance."""
    hs = hs.copy()
    hs["tested"] = ((hs["band_index"] == n_band)
                    & (hs["contacts"] >= min_contacts)
                    & (hs["cells"] >= min_cells))
    for col in ("excess", "z_excess", "q_hot", "q_cold"):
        hs[col] = np.nan
    for _, g in hs[hs["tested"]].groupby("context", sort=False):
        ex = g["delta"] - g["delta"].median()
        with np.errstate(invalid="ignore", divide="ignore"):
            z = ex / g["se"]
            z0 = (g["control"] - g["control"].median()) / g["se_control"]
        hs.loc[g.index, "excess"] = ex
        hs.loc[g.index, "z_excess"] = z
        hs.loc[g.index, "q_hot"] = _empirical_q(z, z0)
        hs.loc[g.index, "q_cold"] = _empirical_q(-z, -z0)
    hs["hot"] = (hs["q_hot"] < fdr) & (hs["excess"] > 0)
    hs["cold"] = (hs["q_cold"] < fdr) & (hs["excess"] < 0)
    return hs


def hotspot_tracks(grid: BinGrid, systems: dict, ab_maps, genes, cpg,
                   user: dict) -> dict:
    """Per-bin annotation values the hotspot map is tested against: the
    share of the bin in each class of each system, PMD borders, CpG
    islands, TSSs, imprinted genes, the clustered gene families, and any
    --hotspot-annotation BED."""
    import re
    tracks = {}
    sy = dict(systems)
    if ab_maps:
        sy["ab"] = ab_maps["__all__"]
    for name, iv in sy.items():
        lab = SYSTEM_LABEL.get(name, name)
        for k, cls in enumerate(iv.classes):
            sel = iv._k == k
            tracks[f"{lab}: {cls}"] = grid.coverage_g(iv._g0[sel], iv._g1[sel])
        if name == "pmd" and "PMD" in iv.classes:
            sel = iv._k == iv.classes.index("PMD")
            tracks["PMD borders"] = grid.counts_g(
                np.concatenate([iv._g0[sel], iv._g1[sel] - 1]))
    if cpg is not None and len(cpg):
        tracks["CpG islands"] = grid.coverage(cpg)
    if genes is not None and len(genes):
        g = genes.assign(chrom=genes["chrom"].map(_canon))
        tss = np.where(g["strand"].to_numpy() == "-",
                       g["end"].to_numpy(np.int64) - 1,
                       g["start"].to_numpy(np.int64))
        tracks["TSSs (count)"] = grid.counts_g(
            _codes_of(g["chrom"].to_numpy()) * STRIDE + tss)
        # symbols compared without case: Peg3 (mouse) is PEG3 (human)
        names = g["name"].astype(str).str.lower()
        tracks["imprinted genes"] = grid.coverage(
            g[names.isin({x.lower() for x in IMPRINTED_GENES})])
        for fam, rx in GENE_FAMILIES.items():
            tracks[fam] = grid.coverage(
                g[names.map(lambda x: re.match(rx, x) is not None)])
    for name, df in (user or {}).items():
        tracks[name] = grid.coverage(df)
    return {k: v for k, v in tracks.items() if np.any(v > 0)}


def hotspot_enrichment(hs: pd.DataFrame, tracks: dict) -> pd.DataFrame:
    """Per context and annotation track, over the tested bins: Spearman rho
    between a bin's ΔC and the track (and the control's ΔC, for scale), and
    whether hot / cold bins fall where the track is more often than other
    bins (odds ratio and Fisher's exact p)."""
    from scipy.stats import fisher_exact, rankdata, spearmanr
    rows = []

    def _resid(r, z):
        z = np.column_stack([np.ones(z.size), z])
        return r - z @ np.linalg.lstsq(z, r, rcond=None)[0]
    for ctx, a in hs[hs["tested"]].groupby("context", sort=False):
        gi = a["grid_index"].to_numpy()
        tss = tracks.get("TSSs (count)")
        rt = rankdata(tss[gi]) if tss is not None else None
        for name, v in tracks.items():
            x = v[gi]
            if np.ptp(x) == 0:
                continue
            rho, p = spearmanr(a["delta"], x)
            rho0, _ = spearmanr(a["control"], x)
            row = {"context": ctx, "track": name, "bins": int(gi.size),
                   "bins_with": int((x > 0).sum()), "rho": float(rho),
                   "p_rho": float(p), "rho_control": float(rho0)}
            if rt is not None and name != "TSSs (count)":
                # gene density accounted for: imprinted genes, CpG islands
                # and promoters all sit where genes are dense
                ex = _resid(rankdata(a["delta"]), rt)
                xx = _resid(rankdata(x), rt)
                if ex.std() > 0 and xx.std() > 0:
                    row["rho_given_tss"] = float(np.corrcoef(ex, xx)[0, 1])
            for kind in ("hot", "cold"):
                h = a[kind].to_numpy(bool)
                tab = [[int((h & (x > 0)).sum()), int((h & (x == 0)).sum())],
                       [int((~h & (x > 0)).sum()),
                        int((~h & (x == 0)).sum())]]
                row[f"{kind}_with"] = tab[0][0]
                row[f"{kind}_total"] = int(h.sum())
                if h.any():
                    orr, pf = fisher_exact(tab)
                    row[f"{kind}_odds_ratio"] = float(orr)
                    row[f"{kind}_p"] = float(pf)
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_rho"] = np.nan
        for _, g in out.groupby("context", sort=False):
            out.loc[g.index, "q_rho"] = fdr_bh(g["p_rho"].to_numpy())
    return out


def plot_cross_class(cc: pd.DataFrame, context: str, path, stat: str = None):
    """Per system, the noise-corrected correlation across cells between its
    same-state class pairs' per-cell ΔC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    a = cc[cc["context"] == context] if not cc.empty else cc
    if stat is not None and "statistic" in a:
        a = a[a["statistic"] == stat]
    if a.empty:
        return
    systems = [x for x in SYSTEMS if x in set(a["system"])] + [
        x for x in dict.fromkeys(a["system"]) if x not in SYSTEMS]
    rows_ = [("r_disattenuated", "r", "noise-corrected")]
    if "r_resid" in a and a["r_resid"].notna().any():
        rows_.append(("r_resid", "r_resid", "after cell type, every "
                                            "covariate and batch"))
    states = {sy: list(dict.fromkeys(
        list(a.loc[a["system"] == sy, "state_1"])
        + list(a.loc[a["system"] == sy, "state_2"]))) for sy in systems}
    n_max = max(len(st) for st in states.values())
    with plt.rc_context(figure_style.RC):
        # panel widths follow the number of states, so that every cell has
        # the same size
        fig, axes = plt.subplots(
            len(rows_), len(systems), squeeze=False,
            figsize=(figure_style.FIG_W_IN,
                     (1.2 + 0.3 * n_max) * len(rows_)),
            gridspec_kw={"wspace": 0.6, "hspace": 0.9, "width_ratios":
                         [max(len(states[sy]), 2) for sy in systems]})
        for ri, (val, raw, what) in enumerate(rows_):
            for ax, sy in zip(axes[ri], systems):
                b = a[a["system"] == sy]
                st = states[sy]
                M = np.full((len(st), len(st)), np.nan)
                Mr = M.copy()
                np.fill_diagonal(M, 1.0)
                for _, r in b.iterrows():
                    i, j = st.index(r["state_1"]), st.index(r["state_2"])
                    M[i, j] = M[j, i] = r[val]
                    Mr[i, j] = Mr[j, i] = r[raw]
                # raw r, faint, where a class is too noisy to correct for
                ax.imshow(np.where(np.isfinite(M), np.nan, np.clip(Mr, -1, 1)),
                          cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.35)
                im = ax.imshow(np.clip(M, -1, 1), cmap="RdBu_r", vmin=-1,
                               vmax=1)
                for i in range(len(st)):
                    for j in range(len(st)):
                        if i == j:
                            continue
                        if np.isfinite(M[i, j]):
                            # noise correction can overshoot 1 when the
                            # reliabilities are themselves noisy
                            ax.text(j, i, ">1" if M[i, j] > 1 else
                                    f"{M[i, j]:.2f}".replace("0.", ".", 1),
                                    ha="center", va="center", fontsize=4.5)
                        elif np.isfinite(Mr[i, j]):
                            ax.text(j, i, f"({Mr[i, j]:.2f})".replace(
                                "0.", ".", 1), ha="center", va="center",
                                fontsize=4.5, color="0.4")
                ax.set_xticks(range(len(st)))
                ax.set_xticklabels(st, rotation=60, ha="right", fontsize=5)
                ax.set_yticks(range(len(st)))
                ax.set_yticklabels(st, fontsize=5)
                ax.set_title(SYSTEM_LABEL.get(sy, sy) + (
                    f"\n({what})" if sy == systems[0] else ""), fontsize=6)
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02,
                          pad=0.02)
        cb.ax.tick_params(labelsize=5)
        cb.set_label("r across cells, noise-corrected", fontsize=6)
        fig.suptitle(f"{context}: does a cell's "
                     f"{_STAT_LABEL.get(stat or 'delta', stat)} move together "
                     f"across states?\n(in brackets, faint: raw r -- a class "
                     f"too noisy across cells to correct for)", fontsize=7)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


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


def plot_cell_cycle(cov: pd.DataFrame, path):
    """The covariates themselves: % near against % mitotic (the cell-cycle
    plane of Nagano et al. 2017), repli-score by cell type, and repli-score
    against % near."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    types = sorted(cov["cell_type"].astype(str).unique())
    cmap = plt.get_cmap("tab10" if len(types) <= 10 else "tab20")
    col = {t: cmap(i % cmap.N) for i, t in enumerate(types)}
    with plt.rc_context(figure_style.RC):
        fig, ax = plt.subplots(1, 3, figsize=(figure_style.FIG_W_IN, 2.4))
        for t in types:
            d = cov[cov["cell_type"].astype(str) == t]
            ax[0].scatter(100 * d["frac_near"], 100 * d["frac_mitotic"], s=3,
                          color=col[t], label=t, alpha=0.7, lw=0)
            ax[2].scatter(100 * d["frac_near"], d["repli_score"], s=3,
                          color=col[t], alpha=0.7, lw=0)
        ax[0].set_xlabel("% near (25 kb–2 Mb)")
        ax[0].set_ylabel("% mitotic band (2–12 Mb)")
        ax[0].legend(fontsize=4.5, markerscale=2, ncol=2, loc="best")
        data = [cov.loc[cov["cell_type"].astype(str) == t,
                        "repli_score"].dropna().to_numpy() for t in types]
        bp = ax[1].boxplot(data, vert=True, widths=0.6, showfliers=True,
                           patch_artist=True, flierprops={"markersize": 1.5},
                           medianprops={"color": "0.1", "lw": 0.8})
        for b, t in zip(bp["boxes"], types):
            b.set_facecolor(col[t])
            b.set_linewidth(0.5)
        ax[1].set_xticks(range(1, len(types) + 1))
        ax[1].set_xticklabels(types, rotation=60, ha="right")
        ax[1].set_ylabel("repli-score (early / early + late)")
        ax[2].set_xlabel("% near (25 kb–2 Mb)")
        ax[2].set_ylabel("repli-score")
        for a, lab in zip(ax, "abc"):
            a.set_title(lab, loc="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_variance_total(assoc: pd.DataFrame, context: str, stat: str, path,
                        q_cut: float = 0.05):
    """Same-state class pairs (rows), all distances: the variance of the
    per-cell statistic over ALL cells, split into sampling noise, which loci
    were sampled, the technical groups (batch with them), cell type, the
    other covariate groups, and what is left -- variation between cells
    none of them accounts for (* q < q_cut that it is more than sampling)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    if assoc.empty or "tot_noise" not in assoc:
        return
    all_b = next((b for b in dict.fromkeys(assoc["band"])
                  if str(b).startswith("all")), None)
    a = _same_state(assoc, context, stat, all_b)
    a = a[a["tot_noise"].notna()] if not a.empty else a
    if a.empty:
        return
    ny = len(a)
    col = lambda c: (a[c].to_numpy(float) if c in a  # noqa: E731
                     else np.full(ny, np.nan))
    groups = [g for g in TOTAL_ORDER
              if f"tot_{g}" in a and a[f"tot_{g}"].notna().any()]
    parts = [(np.nan_to_num(col("tot_noise")), "0.85", "sampling noise"),
             (np.nan_to_num(col("tot_loci")), "#f4c58f",
              "which loci were sampled")]
    parts += [(np.clip(np.nan_to_num(col(f"tot_{g}")), 0, None),
               TOTAL_COLOR[g], g) for g in groups]
    resid = np.nan_to_num(col("tot_residual"))
    parts.append((resid, "#9ecae1", "between cells, unexplained"))
    tot = np.sum([w for w, _, _ in parts], axis=0)
    parts = [(w / np.where(tot > 0, tot, 1), c_, lab) for w, c_, lab in parts]
    q = col("residual_q")
    with plt.rc_context(figure_style.RC):
        fig, ax = plt.subplots(figsize=(figure_style.FIG_W_IN * 0.75,
                                        0.9 + 0.2 * ny))
        yy = np.arange(ny)
        left = np.zeros(ny)
        for w, c_, lab in parts:
            ax.barh(yy, w, left=left, color=c_, label=lab, height=0.72)
            left = left + w
        rn = col("tot_residual_nonlinear")
        for i in range(ny):
            r_ = parts[-1][0][i]
            ax.text(1.01, i, f"{100 * r_:.0f}%"
                    + ("*" if np.isfinite(q[i]) and q[i] < q_cut else "")
                    + (f" · {100 * rn[i]:.0f}% nonlin." if np.isfinite(rn[i])
                       else ""),
                    va="center", ha="left", fontsize=5)
            ct = col("tot_cell type")[i]
            if np.isfinite(ct):
                ax.text(-0.01, i, f"type {100 * ct:.0f}%", va="center",
                        ha="right", fontsize=4.5, color="0.3")
        ax.set_yticks(yy)
        ax.set_yticklabels([_row_label(sy, k)
                            for sy, k in zip(a["system"], a["anchor"])],
                           fontsize=5.5)
        ax.tick_params(axis="y", pad=28)
        ax.set_ylim(ny - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of the variance across all cells")
        ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1.17, 1.0),
                  borderaxespad=0)
        ax.set_title(f"{context} — {_STAT_LABEL.get(stat, stat)}: what varies "
                     f"between cells (right: unexplained, * more than "
                     f"sampling at q < {q_cut:g}; nonlin.: left even by a "
                     f"nonlinear fit of the covariates)", loc="left",
                     fontsize=6.5)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_extra_covariates(cov: pd.DataFrame, path):
    """The cell covariates of the later checks, by cell type: GCH at open
    sites (M.CviPI labeling efficiency), PMD depth (HMD less PMD
    methylation: division history), the mCA index (GCA less GCC + GCT
    methylated: endogenous CpA methylation read as accessibility), and the
    first two substructure PCs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    panels = [(c, lab) for c, lab in (
        ("open_site_level", "GCH methylated at open sites"),
        ("pmd_depth", "PMD depth (HMD − PMD, HCG)"),
        ("mca_index", "mCA index"),
        ("doublet_score", "doublet score"),
        ("nonconversion", "non-conversion (HCY, distal)"),
        ("hcy_open_excess", "HCY: open sites − distal"))
        if c in cov and cov[c].notna().sum() > 2]
    has_pc = all(c in cov and cov[c].notna().sum() > 2
                 for c in ("sub_pc1", "sub_pc2"))
    n = len(panels) + int(has_pc)
    if n == 0:
        return
    types = sorted(cov["cell_type"].astype(str).unique())
    cmap = plt.get_cmap("tab10" if len(types) <= 10 else "tab20")
    col = {t: cmap(i % cmap.N) for i, t in enumerate(types)}
    with plt.rc_context(figure_style.RC):
        fig, ax = plt.subplots(1, n, figsize=(min(figure_style.FIG_W_IN * 1.4,
                                                  2.0 * n + 0.6), 2.4),
                               squeeze=False)
        ax = ax[0]
        for a_, (c, lab) in zip(ax, panels):
            data = [cov.loc[cov["cell_type"].astype(str) == t, c].dropna()
                    .to_numpy() for t in types]
            keep = [i for i, x in enumerate(data) if x.size]
            bp = a_.boxplot([data[i] for i in keep], widths=0.6,
                            patch_artist=True, flierprops={"markersize": 1.5},
                            medianprops={"color": "0.1", "lw": 0.8})
            for b, i in zip(bp["boxes"], keep):
                b.set_facecolor(col[types[i]])
                b.set_linewidth(0.5)
            a_.set_xticks(range(1, len(keep) + 1))
            a_.set_xticklabels([types[i] for i in keep], rotation=60,
                               ha="right")
            a_.set_ylabel(lab)
            if c in ("mca_index", "hcy_open_excess"):
                a_.axhline(0, color="0.5", lw=0.5, ls="--")
        if has_pc:
            a_ = ax[-1]
            for t in types:
                d = cov[cov["cell_type"].astype(str) == t]
                a_.scatter(d["sub_pc1"], d["sub_pc2"], s=3, color=col[t],
                           label=t, alpha=0.7, lw=0)
            a_.set_xlabel("substructure PC1 (z, within type)")
            a_.set_ylabel("substructure PC2")
            a_.legend(fontsize=4.5, markerscale=2, ncol=2, loc="best")
        for a_, lab in zip(ax, "abcdefg"):
            a_.set_title(lab, loc="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_hotspots(hs: pd.DataFrame, enrich: pd.DataFrame, context: str,
                  path, note: str = "", fdr: float = 0.05):
    """a: along the genome, per bin, the concordance of the contacts with an
    end in it, pooled over cells (ΔC, pp; dotted: the median bin) -- bins
    above it red, below it blue, the control grey; b: per annotation track,
    Spearman rho across bins between ΔC and the track (bar), and the
    control's (open circle); * q < 0.05."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    a = hs[(hs["context"] == context) & hs["tested"]]
    if a.empty:
        return
    e = (enrich[enrich["context"] == context].sort_values("rho")
         if not enrich.empty else enrich)
    codes = np.sort(a["code"].unique())
    span = a.groupby("code")["end"].max().reindex(codes).to_numpy(float)
    off = dict(zip(codes, np.r_[0, np.cumsum(span)[:-1]]))
    x = a["code"].map(off).to_numpy() + (a["start"] + a["end"]) / 2
    rank = a["code"].map({c: i for i, c in enumerate(codes)}).to_numpy()
    names = a.groupby("code")["chrom"].first().reindex(codes)
    with plt.rc_context(figure_style.RC):
        fig = plt.figure(figsize=(figure_style.FIG_W_IN,
                                  1.9 + 0.13 * max(len(e), 4)))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 0.13 * max(len(e), 4)],
                              hspace=0.6)
        ax = fig.add_subplot(gs[0])
        ax.scatter(x, 100 * a["control"], s=1, color="0.8", lw=0,
                   label="control (reshuffled draw 2 − draw 1)")
        ax.scatter(x, 100 * a["delta"], s=1.5, lw=0,
                   c=np.where(rank % 2, "0.2", "0.55"))
        for k, c_, lab in (("hot", "#cb181d", "hot"),
                           ("cold", "#2171b5", "cold")):
            m = a[k].to_numpy(bool)
            ax.scatter(x[m], 100 * a["delta"][m], s=3, color=c_, lw=0,
                       label=f"{lab} ({int(m.sum())} bins, FDR {fdr:g})")
        ax.axhline(0, color="0.4", lw=0.4)
        ax.axhline(100 * float(a["delta"].median()), color="#cb181d", lw=0.4,
                   ls=":")
        mids = [off[c] + s_ / 2 for c, s_ in zip(codes, span)]
        ax.set_xticks(mids)
        ax.set_xticklabels([_canon(n) for n in names], fontsize=4.5)
        ax.set_xlim(0, float(np.sum(span)))
        ax.set_ylabel("ΔC (pp)")
        ax.legend(fontsize=4.5, markerscale=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
        ax.set_title(f"a  {context}: concordance along the genome, pooled "
                     f"over cells{note}", loc="left", fontsize=6.5)
        if not e.empty:
            ax2 = fig.add_subplot(gs[1])
            yy = np.arange(len(e))
            ax2.barh(yy, e["rho"], color=np.where(e["rho"] > 0, "#cb181d",
                                                  "#2171b5"), height=0.7)
            ax2.scatter(e["rho_control"], yy, s=8, facecolors="none",
                        edgecolors="0.2", lw=0.5, zorder=3, label="control")
            if "rho_given_tss" in e and e["rho_given_tss"].notna().any():
                ax2.scatter(e["rho_given_tss"], yy, s=9, marker="|",
                            color="k", lw=1.0, zorder=4,
                            label="given TSS density")
            for y_, (r_, q_) in enumerate(zip(e["rho"], e["q_rho"])):
                if q_ < 0.05:
                    ax2.text(r_, y_, " *" if r_ >= 0 else "* ", va="center",
                             ha="left" if r_ >= 0 else "right", fontsize=6)
            ax2.axvline(0, color="0.4", lw=0.4)
            ax2.set_yticks(yy)
            ax2.set_yticklabels(e["track"], fontsize=5)
            ax2.set_xlabel("Spearman ρ across bins: ΔC against the track")
            ax2.legend(fontsize=4.5, loc="lower right")
            ax2.set_title("b  what the concordant bins are", loc="left",
                          fontsize=6.5)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def _all_regions_system(assoc, context):
    """The system whose all-regions row the figures show: the one most
    cells enter over all distances."""
    a = assoc[(assoc["context"] == context)
              & (assoc["anchor"] == ALL_REGIONS)]
    if a.empty:
        return None
    all_b = a[a["band"].astype(str).str.startswith("all")]
    a = all_b if not all_b.empty else a
    return a.groupby("system")["cells"].max().idxmax()


def _same_state(assoc, context, stat, band):
    a = assoc[(assoc["context"] == context) & (assoc["statistic"] == stat)
              & (assoc["band"] == band) & (assoc["anchor"] == assoc["partner"])]
    allr = a[a["anchor"] == ALL_REGIONS]
    a = a[a["anchor"] != ALL_REGIONS]
    order = [x for x in SYSTEMS if x in set(a["system"])] + [
        x for x in dict.fromkeys(a["system"]) if x not in SYSTEMS]
    out = pd.concat([a[a["system"] == x] for x in order]) if order else a
    sy = _all_regions_system(assoc, context) if not allr.empty else None
    if sy is not None and (allr["system"] == sy).any():
        out = pd.concat([allr[allr["system"] == sy], out])
    return out


def _row_label(sy, k) -> str:
    return ("all regions" if k == ALL_REGIONS
            else f"{SYSTEM_LABEL.get(sy, sy)}: {k}–{k}")


def plot_covariates(assoc: pd.DataFrame, context: str, stat: str, path,
                    q_cut: float = 0.05):
    """Same-state class pairs (rows, by annotation system) x covariates, all
    distances: the within-type Spearman rho (colour; * q < q_cut), and on
    the right the within-type variance split into its components --
    sampling noise, which loci were sampled, and the cells' own part, that
    split again into what each covariate group explains (Shapley shares,
    so overlapping groups split what they explain together) and the rest."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    if assoc.empty:
        return
    all_b = next((b for b in dict.fromkeys(assoc["band"])
                  if str(b).startswith("all")), None)
    a = _same_state(assoc, context, stat, all_b)
    if a.empty:
        return
    covs = [c for c in HEATMAP_COVARIATES
            if f"rho_{c}" in a and a[f"rho_{c}"].notna().any()]
    R = a[[f"rho_{c}" for c in covs]].to_numpy(float)
    Q = a[[f"q_{c}" for c in covs]].to_numpy(float)
    ny = len(a)
    lim = max(0.1, float(np.nanmax(np.abs(R))) if np.isfinite(R).any()
              else 0.1)
    with plt.rc_context(figure_style.RC):
        fig = plt.figure(figsize=(figure_style.FIG_W_IN * 0.8
                                  + 0.12 * max(len(covs) - 6, 0),
                                  1.3 + 0.2 * ny))
        gs = fig.add_gridspec(1, 3, width_ratios=[2.2 + 0.2 * max(
            len(covs) - 6, 0), 0.08, 1.1], wspace=0.4)
        ax0, cax, ax1 = (fig.add_subplot(gs[0, i]) for i in range(3))
        im = ax0.imshow(R, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(ny):
            for j in range(R.shape[1]):
                if np.isfinite(R[i, j]):
                    ax0.text(j, i, f"{R[i, j]:+.2f}"
                             + ("*" if Q[i, j] < q_cut else ""),
                             ha="center", va="center", fontsize=5)
        ax0.set_xticks(range(len(covs)))
        ax0.set_xticklabels([COV_LABEL[c] for c in covs], rotation=40,
                            ha="right")
        ax0.set_yticks(range(ny))
        ax0.set_yticklabels([_row_label(sy, k)
                             for sy, k in zip(a["system"], a["anchor"])])
        ax0.set_title(f"within-type ρ, all distances  (* q < {q_cut:g})",
                      loc="left")
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=5)
        yy = np.arange(ny)
        col = lambda c: (a[c].to_numpy(float) if c in a  # noqa: E731
                         else np.full(ny, np.nan))
        noise = np.where(np.isfinite(col("noise_share")), col("noise_share"),
                         1 - np.clip(col("bio_share"), 0, 1))
        locus = np.nan_to_num(col("locus_share"))
        cell_ = np.where(np.isfinite(col("cell_share")), col("cell_share"),
                         np.clip(col("bio_share"), 0, 1))
        split = np.isfinite(col("locus_share")).any()
        rest = ("cells: unexplained" if split else
                "cells + loci (not split): unexplained")
        cell_ = np.clip(np.nan_to_num(cell_), 0, None)
        groups = [g for g in GROUP_COLOR
                  if f"shap_{g}" in a and a[f"shap_{g}"].notna().any()]
        parts = [(noise, "0.85", "sampling noise"),
                 (locus, "#f4c58f", "which loci were sampled")]
        if groups:
            # each group's Shapley share, within the cells' part
            S = np.column_stack([np.clip(np.nan_to_num(col(f"shap_{g}")),
                                         0, None) for g in groups])
            tot = S.sum(1)
            S = S * np.where(tot > cell_, cell_ / np.where(tot > 0, tot, 1),
                             1)[:, None]
            parts += [(S[:, i], GROUP_COLOR[g], f"cells: {g}")
                      for i, g in enumerate(groups)]
            parts.append((cell_ - S.sum(1), "#9ecae1", rest))
        else:
            expl = cell_ * np.nan_to_num(np.where(
                np.isfinite(col("explained_cell_share")),
                col("explained_cell_share"), col("explained_share")))
            parts += [(expl, "#08519c", "cells: explained by covariates"),
                      (cell_ - expl, "#9ecae1", rest)]
        left = np.zeros(ny)
        for w, c_, lab in parts:
            w = np.clip(np.nan_to_num(w), 0, None)
            ax1.barh(yy, w, left=left, color=c_, label=lab, height=0.7)
            left = left + w
        ax1.set_ylim(ny - 0.5, -0.5)
        ax1.set_yticks(yy)
        ax1.set_yticklabels([])
        ax1.set_xlim(0, 1)
        ax1.set_xlabel("share of within-type var.")
        leg = ax1.legend(fontsize=5, loc="upper left",
                         bbox_to_anchor=(1.03, 1.0), ncol=1, borderaxespad=0)
        fig.canvas.draw()
        rnd = fig.canvas.get_renderer()
        top = max(leg.get_window_extent(rnd).y1,
                  ax0.get_tightbbox(rnd).y1, ax1.get_tightbbox(rnd).y1)
        fig.suptitle(f"{context} — {_STAT_LABEL.get(stat, stat)}: what "
                     f"explains the within-type variation", fontsize=7,
                     va="bottom",
                     y=fig.transFigure.inverted().transform((0, top))[1]
                     + 0.01)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_covariates_by_band(assoc: pd.DataFrame, context: str, stat: str,
                            path, q_cut: float = 0.05):
    """The within-type rho of each covariate in each distance band (one
    heatmap per band, * q < q_cut): an association that is only there over
    all distances, and not within any band, follows the cell's mix of
    contact distances rather than the covariate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    if assoc.empty:
        return
    bands = list(dict.fromkeys(assoc["band"]))
    bands = [b for b in bands if not str(b).startswith("all")] + [
        b for b in bands if str(b).startswith("all")]
    blocks = [(b, _same_state(assoc, context, stat, b)) for b in bands]
    blocks = [(b, x) for b, x in blocks if not x.empty]
    if not blocks:
        return
    rows = list(dict.fromkeys(
        (sy, k) for _, x in blocks for sy, k in zip(x["system"], x["anchor"])))
    ny = len(rows)
    covs = [c for c in HEATMAP_COVARIATES
            if any(f"rho_{c}" in x and x[f"rho_{c}"].notna().any()
                   for _, x in blocks)]
    lim = 0.1
    mats = []
    for b, x in blocks:
        x = x.set_index(["system", "anchor"]).reindex(rows)
        R = x[[f"rho_{c}" for c in covs]].to_numpy(float)
        Q = x[[f"q_{c}" for c in covs]].to_numpy(float)
        mats.append((b, R, Q))
        if np.isfinite(R).any():
            lim = max(lim, float(np.nanmax(np.abs(R))))
    with plt.rc_context(figure_style.RC):
        fig, axes = plt.subplots(1, len(mats), sharey=True,
                                 figsize=(figure_style.FIG_W_IN,
                                          1.4 + 0.2 * ny), squeeze=False)
        for ax, (b, R, Q) in zip(axes[0], mats):
            im = ax.imshow(R, cmap="RdBu_r", vmin=-lim, vmax=lim,
                           aspect="auto")
            for i in range(ny):
                for j in range(R.shape[1]):
                    if np.isfinite(Q[i, j]) and Q[i, j] < q_cut:
                        ax.text(j, i, "*", ha="center", va="center",
                                fontsize=7)
            ax.set_xticks(range(len(covs)))
            ax.set_xticklabels([COV_LABEL[c] for c in covs],
                               rotation=60, ha="right", fontsize=5)
            ax.set_title(str(b), fontsize=6)
        axes[0][0].set_yticks(range(ny))
        axes[0][0].set_yticklabels([_row_label(sy, k) for sy, k in rows],
                                   fontsize=5)
        cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
        cb.ax.tick_params(labelsize=5)
        cb.set_label("within-type ρ", fontsize=6)
        fig.suptitle(f"{context} — {_STAT_LABEL.get(stat, stat)}: "
                     f"within-type ρ by distance band (* q < {q_cut:g})",
                     fontsize=7)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


# ------------------------------------------------------------------ figure --


#: Columns each statistic is drawn from.
_STAT = {
    "delta": {"point": "delta", "ctrl": "control", "pc": "delta",
              "pc_ctrl": "control", "prefix": "",
              "label": "ΔC, percentage points\ncontacts − reshuffled"},
    "signed": {"point": "signed", "ctrl": "signed_control", "pc": "signed",
               "pc_ctrl": "signed_control", "prefix": "signed_",
               "label": "ΔC × (2F − 1), pp\n+ co-open, − co-closed"},
    "signed_rel": {"point": "signed_rel", "ctrl": "signed_rel_control",
                   "pc": "signed_rel", "pc_ctrl": "signed_rel_control",
                   "prefix": "signed_rel_",
                   "label": "ΔC × w, pp\n+ more co-open, − more co-closed "
                            "than the cell's genome"},
    "signed_sgn": {"point": "signed_sgn", "ctrl": "signed_sgn_control",
                   "pc": "signed_sgn", "pc_ctrl": "signed_sgn_control",
                   "prefix": "signed_sgn_",
                   "label": "ΔC × sign(w), pp\n+ more co-open, − more "
                            "co-closed than the cell's genome"},
}
#: --signed-figures tag -> statistic, and how its figures are titled.
_SIGNED_STAT = {"rel": "signed_rel", "sign": "signed_sgn", "even": "signed"}
_SIGNED_TITLE = {"rel": "ΔC × w (F against the cell's genome)",
                 "sign": "ΔC × sign(w)",
                 "even": "ΔC × (2F − 1)"}
#: C is a difference of probabilities -- excess agreement -- so the figures
#: show it in percentage points.  The tables keep fractions.
PP = 100.0


def plot(summary: pd.DataFrame, percell: pd.DataFrame, path, title,
         stat: str = "delta"):
    """Per context x system: the pooled value per class pair and band (95%
    CI over cells, the control beside it), the between-cell SD against the
    control's, and the per-cell values against the control's."""
    S = _STAT[stat]
    pt, pf = S["point"], S["prefix"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    if summary.empty:
        return
    bands = list(dict.fromkeys(summary["band"]))
    all_band = bands[-1]
    pairs = list(dict.fromkeys(zip(summary["anchor"], summary["partner"])))
    labels = [f"{a}–{b}" for a, b in pairs]
    ny = len(pairs)
    with plt.rc_context(figure_style.RC):
        fig, ax = plt.subplots(1, 3, figsize=(figure_style.FIG_W_IN,
                                              1.2 + 0.22 * ny),
                               gridspec_kw={"width_ratios": [1.5, 1, 1.2]})
        a = ax[0]
        off = np.linspace(-0.3, 0.3, len(bands))
        for bi, b in enumerate(bands):
            t = summary[summary["band"] == b].set_index(["anchor", "partner"])
            y = [yi + off[bi] for yi, pr in enumerate(pairs) if pr in t.index]
            r = t.loc[[pr for pr in pairs if pr in t.index]]
            col = ("0.15" if b == all_band
                   else figure_style.PALETTE[bi % len(figure_style.PALETTE)])
            a.errorbar(PP * r[pt], y,
                       xerr=[PP * (r[pt] - r[pt + "_lo"]),
                             PP * (r[pt + "_hi"] - r[pt])],
                       fmt="o", ms=3, lw=0.8, color=col, label=b)
            if b == all_band:
                a.scatter(PP * r[S["ctrl"]], y, marker="d", s=10,
                          color="0.6", zorder=3, label="control (all)")
        a.axvline(0, color="0.4", lw=0.6)
        a.set_yticks(range(ny))
        a.set_yticklabels(labels)
        a.invert_yaxis()
        a.set_xlabel(S["label"])
        a.set_title("Where: pooled, 95% CI over cells", loc="left")
        a.legend(loc="best", fontsize=5)
        a = ax[1]
        t = summary[summary["band"] == all_band].set_index(["anchor",
                                                            "partner"])
        r = t.reindex(pairs)
        col_ = lambda c: r[c] if c in r else pd.Series(np.nan,  # noqa: E731
                                                         index=r.index)
        yy = np.arange(ny)
        a.barh(yy - 0.18, PP * col_(pf + "cell_sd"), height=0.34,
               color=figure_style.PALETTE[0], label="per cell")
        a.barh(yy + 0.18, PP * col_(pf + "control_sd"), height=0.34,
               color="0.7", label="control")
        for yi, v in zip(yy, col_(pf + "split_half_r")):
            if np.isfinite(v):
                a.annotate(f"r½ {v:.2f}", (0, yi), xytext=(2, 0),
                           textcoords="offset points", va="center",
                           fontsize=5, color="white")
        a.set_yticks(yy)
        a.set_yticklabels([])
        a.invert_yaxis()
        a.set_xlabel("between-cell SD, pp (all distances)")
        a.set_title("How variable", loc="left")
        a.legend(loc="lower right", fontsize=5)
        a = ax[2]
        pc = percell[percell["band"] == all_band]
        data, pos = [], []
        for yi, (k1, k2) in enumerate(pairs):
            s = pc[(pc["anchor"] == k1) & (pc["partner"] == k2)
                   & pc["enough_pairs"]]
            data += [PP * s[S["pc"]].dropna().to_numpy(),
                     PP * s[S["pc_ctrl"]].dropna().to_numpy()]
            pos += [yi - 0.18, yi + 0.18]
        bp = a.boxplot(data, positions=pos, vert=False, widths=0.3,
                       showfliers=False, patch_artist=True,
                       medianprops={"color": "0.1", "lw": 0.8})
        for i_, box in enumerate(bp["boxes"]):
            box.set_facecolor(figure_style.PALETTE[0] if i_ % 2 == 0
                              else "0.8")
            box.set_linewidth(0.5)
        a.axvline(0, color="0.4", lw=0.6)
        a.set_yticks(yy)
        a.set_yticklabels([])
        a.set_ylim(ny - 0.5, -0.5)
        a.set_xlabel("per cell, pp (all distances)")
        a.set_title("Per cell vs control", loc="left")
        fig.suptitle(title, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


#: Fig 5a/5c's colours, so the new panels read with them.
CONTEXT_COLOR = {"GCH": "#1f77b4", "HCG": "#d62728"}
SYSTEM_LABEL = {"ab": "compartment", "rt": "replication timing",
                "pmd": "PMD / HMD", "chromhmm": "chromHMM"}


def _variance_split(pc: pd.DataFrame, val="delta_r", ctrl="control_r"):
    """Shares of the between-cell variance of ``val`` that are sampling
    noise (the control's variance), cell type, and variation within cell
    types -- the biological parts each net of the same quantity on the
    control."""
    d = pc[val].to_numpy(float)
    c = pc[ctrl].to_numpy(float)
    t = pc["cell_type"].to_numpy()
    ok = np.isfinite(d) & np.isfinite(c)
    d, c, t = d[ok], c[ok], t[ok]
    if d.size < 5:
        return None

    def between(x):
        g = pd.Series(x).groupby(t)
        return float(((g.mean() - x.mean()) ** 2 * g.size()).sum()
                     / (x.size - 1))
    vn = c.var(ddof=1)
    vb = max(between(d) - between(c), 0.0)
    vw = max(d.var(ddof=1) - vn - vb, 0.0)
    tot = vn + vb + vw
    return (vn / tot, vb / tot, vw / tot) if tot > 0 else None


_MAIN = {
    "r": {"point": "delta_r", "excess": "excess_sd_r", "pc": "delta_r",
          "pc_ctrl": "control_r", "ylab": "Δr", "xlab": "Δr, all distances",
          "scale": 1.0, "xlab_f": "excess SD"},
    "signed": {"point": "signed", "excess": "signed_excess_sd",
               "pc": "signed", "pc_ctrl": "signed_control",
               "ylab": "ΔC×(2F−1), pp",
               "xlab": "ΔC × (2F − 1), pp, all distances",
               "scale": PP, "xlab_f": "excess SD, pp"},
    "signed_rel": {"point": "signed_rel", "excess": "signed_rel_excess_sd",
                   "pc": "signed_rel", "pc_ctrl": "signed_rel_control",
                   "ylab": "ΔC×w, pp",
                   "xlab": "ΔC × w, pp, all distances",
                   "scale": PP, "xlab_f": "excess SD, pp"},
    "signed_sgn": {"point": "signed_sgn", "excess": "signed_sgn_excess_sd",
                   "pc": "signed_sgn", "pc_ctrl": "signed_sgn_control",
                   "ylab": "ΔC×sign(w), pp",
                   "xlab": "ΔC × sign(w), pp, all distances",
                   "scale": PP, "xlab_f": "excess SD, pp"},
}


def plot_main(summary: pd.DataFrame, percell: pd.DataFrame, path,
              note: str | None = None, min_band_pairs: int = 500,
              stat: str = "r"):
    """A candidate main-figure row: d, Δr against distance by compartment
    class pair; e, pooled Δr of contacts with both ends in the same state,
    across the annotation systems; f, excess between-cell SD beyond the
    control; g, how much of the between-cell variance is noise, cell type,
    or variation within cell types.  A distance band with fewer than
    ``min_band_pairs`` contacts is not drawn in d (cross-compartment pairs
    barely exist at short range)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    S = _MAIN[stat]
    pt, ex, sc = S["point"], S["excess"], S["scale"]
    if summary.empty or pt not in summary:
        return
    ctxs = list(dict.fromkeys(summary["context"]))
    systems = ([x for x in SYSTEMS if x in set(summary["system"])]
               + [x for x in dict.fromkeys(summary["system"])
                  if x not in SYSTEMS])
    all_b = next(b for b in dict.fromkeys(summary["band"])
                 if str(b).startswith("all"))
    rows = []
    for sy in systems:
        ss = summary[(summary["system"] == sy) & (summary["band"] == all_b)]
        for k in dict.fromkeys(ss["anchor"]):
            if ((ss["anchor"] == k) & (ss["partner"] == k)).any():
                rows.append((sy, k))
    ny = len(rows)
    y = np.arange(ny)
    col_of = {c: CONTEXT_COLOR.get(c, figure_style.PALETTE[i % 8])
              for i, c in enumerate(ctxs)}
    off = (np.linspace(-0.17, 0.17, len(ctxs)) if len(ctxs) > 1
           else np.zeros(1))
    blocks = []
    for yi, (sy, _) in enumerate(rows):
        if not blocks or blocks[-1][0] != sy:
            blocks.append([sy, yi, yi])
        blocks[-1][2] = yi

    def shade(ax):
        for bi, (_, a0, a1) in enumerate(blocks):
            if bi % 2:
                ax.axhspan(a0 - 0.5, a1 + 0.5, color="0.94", lw=0, zorder=0)

    def at_all(ctx):
        return summary[(summary["context"] == ctx)
                       & (summary["band"] == all_b)].set_index(
            ["system", "anchor", "partner"]).reindex(
            [(sy, k, k) for sy, k in rows])

    with plt.rc_context(figure_style.RC):
        fig = plt.figure(figsize=(figure_style.FIG_W_IN,
                                  max(2.6, 1.0 + 0.15 * ny)))
        gs = fig.add_gridspec(len(ctxs), 4,
                              width_ratios=[1.25, 1.0, 0.75, 0.75],
                              wspace=0.42, hspace=0.45)
        fig_h = fig.get_size_inches()[1]
        below = -0.42 / (fig_h * 0.75)     # legend under an x label
        # d -- by distance
        dsys = next((x for x in ("ab", "rt", "pmd") if x in systems),
                    systems[0])
        for ci, ctx in enumerate(ctxs):
            ax = fig.add_subplot(gs[ci, 0])
            s = summary[(summary["context"] == ctx)
                        & (summary["system"] == dsys)
                        & (summary["band"] != all_b)
                        & (summary["pairs"] >= min_band_pairs)]
            for pi, ((k1, k2), g) in enumerate(s.groupby(
                    ["anchor", "partner"], sort=False)):
                g = g.sort_values("band_lo")
                x = np.sqrt(g["band_lo"] * g["band_hi"])
                cc = figure_style.PALETTE[pi % 8]
                ax.plot(x, sc * g[pt], "-" if k1 == k2 else "--",
                        marker="o", ms=2.2, lw=0.9, color=cc,
                        label=f"{k1}–{k2}")
                ax.fill_between(x, sc * g[pt + "_lo"], sc * g[pt + "_hi"],
                                color=cc, alpha=0.15, lw=0)
            ax.axhline(0, color="0.5", lw=0.6)
            ax.set_xscale("log")
            ax.set_ylabel(f"{ctx}   {S['ylab']}")
            if ci == len(ctxs) - 1:
                ax.set_xlabel("distance between the two ends (bp)")
            else:
                ax.tick_params(labelbottom=False)
            if ci == 0:
                lo_, hi_ = ax.get_ylim()
                ax.set_ylim(lo_, hi_ + 0.35 * (hi_ - lo_))   # room for key
                ax.set_title(f"d  By distance ({SYSTEM_LABEL.get(dsys, dsys)})",
                             loc="left")
                ax.legend(ncol=4, fontsize=5, handlelength=1.6,
                          columnspacing=0.8, loc="upper left")
        # e -- where
        ax_e = fig.add_subplot(gs[:, 1])
        shade(ax_e)
        for ci, ctx in enumerate(ctxs):
            v = at_all(ctx)
            ax_e.errorbar(sc * v[pt], y + off[ci],
                          xerr=[sc * (v[pt] - v[pt + "_lo"]),
                                sc * (v[pt + "_hi"] - v[pt])],
                          fmt="o", ms=2.5, lw=0.8, color=col_of[ctx],
                          label=ctx, zorder=3)
        ax_e.axvline(0, color="0.5", lw=0.6)
        ax_e.set_yticks(y)
        ax_e.set_yticklabels([k for _, k in rows])
        ax_e.set_ylim(ny - 0.5, -0.5)
        ax_e.set_xlabel(S["xlab"])
        ax_e.set_title("e  Both ends in one state", loc="left")
        ax_e.legend(loc="upper center", bbox_to_anchor=(0.5, below),
                    ncol=len(ctxs))
        # f -- how variable
        ax_f = fig.add_subplot(gs[:, 2], sharey=ax_e)
        shade(ax_f)
        for ci, ctx in enumerate(ctxs):
            v = at_all(ctx)
            if ex not in v:
                continue
            ax_f.errorbar(sc * v[ex], y + off[ci],
                          xerr=[sc * (v[ex] - v[ex + "_lo"]),
                                sc * (v[ex + "_hi"] - v[ex])],
                          fmt="s", ms=2.3, lw=0.8, color=col_of[ctx])
        ax_f.set_xlim(left=0)
        ax_f.tick_params(labelleft=False)
        ax_f.set_xlabel(S["xlab_f"])
        ax_f.set_title("f  Cell to cell", loc="left")
        # g -- what varies
        ax_g = fig.add_subplot(gs[:, 3], sharey=ax_e)
        for ci, ctx in enumerate(ctxs):
            shades = ("0.82", col_of[ctx],
                      matplotlib.colors.to_rgba(col_of[ctx], 0.35))
            for yi, (sy, k) in enumerate(rows):
                pc = percell[(percell["context"] == ctx)
                             & (percell["system"] == sy)
                             & (percell["anchor"] == k)
                             & (percell["partner"] == k)
                             & (percell["band"] == all_b)
                             & percell["enough_pairs"]]
                v = (_variance_split(pc, S["pc"], S["pc_ctrl"])
                     if len(pc) else None)
                if v is None:
                    continue
                left = 0.0
                for frac, cc in zip(v, shades):
                    ax_g.barh(yi + off[ci], frac, left=left,
                              height=0.3 if len(ctxs) > 1 else 0.6,
                              color=cc, lw=0)
                    left += frac
        ax_g.set_xlim(0, 1)
        ax_g.tick_params(labelleft=False)
        ax_g.set_xlabel("share of variance")
        ax_g.set_title("g  What varies", loc="left")
        ax_g.legend(handles=[mpatches.Patch(color="0.82", label="noise"),
                             mpatches.Patch(color="0.3", label="cell type"),
                             mpatches.Patch(color="0.3", alpha=0.35,
                                            label="within type")],
                    loc="upper center", bbox_to_anchor=(0.5, below),
                    ncol=3, columnspacing=0.8, handlelength=1.0)
        for sy, a0, a1 in blocks:            # annotation system brackets
            tr = ax_g.get_yaxis_transform()
            ax_g.plot([1.04, 1.04], [a0 - 0.35, a1 + 0.35], color="0.4",
                      lw=0.7, transform=tr, clip_on=False)
            ax_g.text(1.08, (a0 + a1) / 2, SYSTEM_LABEL.get(sy, sy),
                      transform=tr,
                      va="center", ha="left", fontsize=6, color="0.25",
                      rotation=90 if a1 - a0 >= 3 else 0)
        if note:
            fig.suptitle(note, fontsize=7, color="#b03a2e", y=1.03)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


# -------------------------------------------------------------------- main --


def _parse_pairs(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"expected NAME=PATH, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _min_sites(spec: str, context: str) -> int:
    """'2' for every context, or 'GCH=2,HCG=1'."""
    if "=" not in spec:
        return int(spec)
    d = dict(x.split("=") for x in spec.split(","))
    return int(d.get(context, d.get(context.upper(), 2)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posteriors", action="append", required=True,
                    metavar="NAME=DIR",
                    help="A context's 04_posteriors directory; repeatable "
                         "(GCH=..., HCG=...).  The first also supplies the "
                         "contacts the A/B call is made from.")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-column", default="cell_name")
    ap.add_argument("--group-column", default="group",
                    help="Cell type: the stratum reshuffled donors come from, "
                         "and what A/B is called per.")
    ap.add_argument("--null-match-column", default=None,
                    help="Comma-separated --samples columns that further "
                         "restrict donors (e.g. exp_batch).")
    ap.add_argument("--systems", default="ab,rt,chromhmm,pmd",
                    help="Annotation systems to run; each needs its input "
                         "(ab: contacts plus an orientation reference; rt: "
                         "--rt-early/--rt-late; chromhmm: --chromhmm or "
                         "--genome mm10; pmd: --pmd).")
    ap.add_argument("--min-sites", default="2",
                    help="Sites for an end to be called: '2', or per context "
                         "'GCH=2,HCG=2'.")
    ap.add_argument("--min-distance", type=int, default=1000,
                    help="Contacts at least this long (1 kb): below it most "
                         "read pairs are one unligated fragment.")
    ap.add_argument("--max-distance", type=int, default=10_000_000)
    ap.add_argument("--bands", default="1000,10000,100000,1000000,10000000",
                    help="Distance-band edges (bp): by default the decades "
                         "1-10 kb, 10-100 kb, 100 kb-1 Mb and 1-10 Mb.  A-B "
                         "pairs barely exist below ~100 kb and the GCH excess "
                         "itself rises ~10x from 1 to 100 kb, so classes are "
                         "compared within bands; an 'all' band is added.")
    ap.add_argument("--bins-per-decade", type=int, default=4,
                    help="Log-distance bins a donor is matched in (4).")
    ap.add_argument("--position-bin", type=int, default=250_000,
                    help="A donor lies in the same bin of this size as the "
                         "real partner (250 kb), as in "
                         "end_posterior_distance_correlation.py's "
                         "link-shuffle.")
    ap.add_argument("--max-widen", type=int, default=2)
    ap.add_argument("--min-pairs", type=int, default=20,
                    help="Contacts a cell needs in a class pair x band to "
                         "enter the per-cell statistics (20).  The pooled Δ "
                         "uses every contact.")
    ap.add_argument("--min-ends", type=int, default=20,
                    help="Called ends a cell needs in a class for its "
                         "marginal to be summarised (20).")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--ab-resolution", type=int, default=250_000)
    ap.add_argument("--ab-min-contacts", type=int, default=300_000,
                    help="A cell type with fewer pooled contacts takes the "
                         "all-cell A/B call (300,000).")
    ap.add_argument("--ab-orient", choices=["auto", "rt", "genes"],
                    default="auto",
                    help="What A is oriented to: replication timing (early = "
                         "A) or gene density.  auto: rt when --rt-early is "
                         "given, else genes.")
    ap.add_argument("--rt-early", action="append", default=None,
                    help="Early Repli-seq track (bigWig or bedGraph); "
                         "repeatable -- the tracks are averaged.")
    ap.add_argument("--rt-late", action="append", default=None)
    ap.add_argument("--rt-log2", action="append", default=None,
                    metavar="[NAME=]TRACK",
                    help="A track of log2(early/late) itself (bigWig or "
                         "bedGraph), instead of --rt-early/--rt-late; "
                         "repeatable -- averaged.  NAME=path is accepted, so "
                         "fetch_4dn_repliseq.py's flags file can be passed "
                         "as $(sed 's/--signal-track/--rt-log2/g' "
                         "annots/rt_4dn/signal_track_flags.txt).")
    ap.add_argument("--rt-window", type=int, default=50_000)
    ap.add_argument("--rt-mid", type=float, default=0.0,
                    help="|log2 early/late| at or below this is 'mid' (0: "
                         "two classes).")
    ap.add_argument("--chromhmm", default=None,
                    help="chromHMM BED (column 4 = state).  --genome mm10 "
                         "fills in the full-stack model.")
    ap.add_argument("--chromhmm-groups", default=None)
    ap.add_argument("--pmd", default=None,
                    help="PMD/HMD file or URL, e.g. the Zhou-lab "
                         "PMD_coordinates_mm10.bed.gz; --genome hg38 / hg19 "
                         "fill it.")
    ap.add_argument("--pmd-class-col", type=int, default=5,
                    help="1-based column holding PMD / HMD (5).")
    ap.add_argument("--pmd-common-only", action="store_true",
                    help="Use column 6's commonPMD / commonHMD instead.")
    ap.add_argument("--annotation", action="append", default=None,
                    metavar="NAME=BED",
                    help="Any further annotation, repeatable: a BED whose "
                         "4th column holds up to 12 labels is classed by "
                         "label; any other BED is inside vs outside its "
                         "intervals (e.g. LAD=lads.bed gives LAD / non-LAD).")
    ap.add_argument("--ordered-pairs", action="store_true",
                    help="Keep a contact's class pair ordered (left end "
                         "first) -- A-B apart from B-A.  Only as a check: "
                         "the orientation means nothing, and with the "
                         "two-ended null the two should agree.")
    ap.add_argument("--signed-figures", default="rel,sign,even",
                    help="Which signed figures to draw, each to its own "
                         "file fig_<context>_<system>_signed_<tag> and "
                         "main_figure_signed_<tag> (all three by default).  "
                         "rel: ΔC x w, w the co-open share F read against "
                         "the cell's own genome-wide one (+ more co-open "
                         "than the cell's contacts overall).  sign: "
                         "ΔC x sign(w), ΔC at full size with only that "
                         "direction.  even: ΔC x (2F - 1), F against 1:1 -- "
                         "for GCH, closed over most of the genome, nearly "
                         "every class then reads co-closed.  summary.tsv "
                         "carries all three (signed_rel, signed_sgn, "
                         "signed).")
    ap.add_argument("--no-covariates", action="store_true",
                    help="Skip the per-cell cell-cycle covariates (% near, "
                         "% mitotic, repli-score) and the analysis of what "
                         "explains the within-type variation.")
    ap.add_argument("--batch-column", default=None,
                    help="Comma-separated --samples columns tested as "
                         "batch effects on the within-type variation (e.g. "
                         "exp_batch,lib_batch).")
    ap.add_argument("--n-locus-perm", type=int, default=30,
                    help="Reassignments of contacts among a type's cells "
                         "for the locus-sampling control (30; 0 skips it).")
    ap.add_argument("--compartment-min-distance", type=int,
                    default=1_000_000,
                    help="Contacts at least this long make a cell's "
                         "compartment strength (1 Mb).")
    ap.add_argument("--covariate-min-cells", type=int, default=30,
                    help="Cells a class pair needs (with --min-pairs "
                         "contacts each) for its covariate analysis (30).")
    ap.add_argument("--open-sites", default=None,
                    help="BED of sites open in (nearly) every cell -- e.g. "
                         "CTCF motifs in constitutive peaks.  A cell's GCH "
                         "methylated at its ends near one reads its M.CviPI "
                         "labeling efficiency.  Default: the TSSs of "
                         "--gene-annotation (or --genome's) in a CpG "
                         "island.")
    ap.add_argument("--open-site-window", type=int, default=200,
                    help="An end this close to an open site counts as at it "
                         "(200 bp).")
    ap.add_argument("--cpg-islands", default=None,
                    help="CpG islands (the UCSC cpgIslandExt table or a "
                         "BED), for the default open sites and the hotspot "
                         "tracks; --genome fills it.")
    ap.add_argument("--context-counts", action="append", default=None,
                    metavar="NAME=DIR",
                    help="A further context's read-pair counts (its "
                         "02_read_pair_counts or 04_posteriors directory), "
                         "read for each cell's raw methylation only; "
                         "repeatable.  HCY=... is bisulfite non-conversion "
                         "(C in no CpG, GpC or CpA: nothing methylates it); "
                         "HCA=... with HCY=... the mCA index, endogenous CpA "
                         "methylation (high in neurons, read by GCH as "
                         "accessibility) less that background; GCA=... with "
                         "GCY=... an mCA index within GCH (the pipeline run "
                         "with --contexts HCA,HCY --steps 1,2).")
    ap.add_argument("--substructure-bin", type=int, default=1_000_000,
                    help="Bin of the cells' regional profiles, whose "
                         "within-type principal components are the "
                         "substructure covariates (1 Mb).")
    ap.add_argument("--n-substructure-pcs", type=int, default=5,
                    help="Substructure PCs per cell type (5; 0 skips them).")
    ap.add_argument("--hotspot-bin", type=int, default=1_000_000,
                    help="Bin of the locus-hotspot map (1 Mb).")
    ap.add_argument("--hotspot-system", default="auto",
                    help="The system whose class-matched null the hotspot "
                         "map uses (auto: the one with the most contacts "
                         "with a reshuffled partner).")
    ap.add_argument("--hotspot-annotation", action="append", default=None,
                    metavar="NAME=BED",
                    help="A further track the hotspot map is tested "
                         "against (e.g. imprinting control regions, IAP "
                         "elements); repeatable.")
    ap.add_argument("--variability-fraction", type=float, default=0.1,
                    help="Share of the tested 1 Mb bins called variable "
                         "across cell types, and stable (0.1 each).")
    ap.add_argument("--variability-min-contacts", type=int, default=100,
                    help="Contacts a cell type needs in a bin to enter its "
                         "variability (100).")
    ap.add_argument("--contrast-group", action="append", default=None,
                    metavar="NAME=TYPE[,TYPE...]",
                    help="Groups of cells to compare, given two or more "
                         "times, each a list of values of --contrast-column "
                         "(e.g. tumor1=Tumor1, tumor2=Tumor2, "
                         "monocyte=Monocyte): every pair, the earlier given "
                         "minus the later (or each against "
                         "--contrast-reference) -- per region group and "
                         "band, each group's pooled concordance and the "
                         "difference (contrast_classes.tsv, contrast_<ctx>_"
                         "<statistic>[_<A>_vs_<B>], contrast_overview_<ctx>_"
                         "<statistic>), and where along the genome they "
                         "differ (contrast_hotspots.tsv.gz, "
                         "contrast_hotspot_enrichment.tsv, "
                         "contrast_hotspots_<ctx>[_<A>_vs_<B>]).")
    ap.add_argument("--contrast-column", default=None,
                    help="--samples column the --contrast-group values are "
                         "in (default --group-column).")
    ap.add_argument("--contrast-reference", default=None, metavar="NAME",
                    help="Compare every other --contrast-group against this "
                         "one only, instead of every pair.")
    ap.add_argument("--chrx", action="store_true",
                    help="chrX against the autosomes: each cell's sex from "
                         "its X and Y coverage (cell_sex.tsv), and per cell "
                         "type, sex and --contrast-group the concordance of "
                         "X-X contacts less autosome-autosome ones "
                         "(chrx_concordance.tsv, chrx_percell.tsv.gz, "
                         "chrx_<ctx>).  chrX enters nothing else.")
    ap.add_argument("--sex-column", default=None,
                    help="--samples column with each cell's sex (female / "
                         "male, F / M), used instead of the call from "
                         "coverage.")
    ap.add_argument("--imprinting", action="store_true",
                    help="Concordance inside imprinted domains -- the "
                         "imprinted genes of the gene annotation merged "
                         "(--imprinting-merge), or --imprinting-bed -- "
                         "against matched windows, per cell type and "
                         "--contrast-group, and each pair's difference "
                         "(imprinting_domains.tsv, imprinting_concordance"
                         ".tsv, imprinting_contrast.tsv, imprinting_<ctx>).")
    ap.add_argument("--pcdh", action="store_true",
                    help="Concordance inside the clustered protocadherins "
                         "-- the whole cluster and its alpha, beta, gamma "
                         "parts, from the gene annotation -- against "
                         "matched windows, per cell type and --contrast-"
                         "group (e.g. neuron=..., glia=...), each pair's "
                         "difference, and each group's accessibility or "
                         "methylation level there (pcdh_domains.tsv, "
                         "pcdh_concordance.tsv, pcdh_contrast.tsv, "
                         "pcdh_windows.tsv, pcdh_<ctx>).  The matched-window "
                         "settings are the --imprinting-* ones.")
    ap.add_argument("--imprinting-bed", default=None,
                    help="BED of the domains (name in column 4) instead of "
                         "the gene annotation's imprinted genes.")
    ap.add_argument("--imprinting-merge", type=int, default=500_000,
                    help="Imprinted genes this close form one domain.")
    ap.add_argument("--imprinting-flank", type=int, default=50_000,
                    help="Added to each side of a domain.")
    ap.add_argument("--imprinting-windows", type=int, default=100,
                    help="Matched windows per domain (100).")
    ap.add_argument("--imprinting-min-contacts", type=int, default=30,
                    help="Contacts a matched window needs (30).")
    ap.add_argument("--celltype-auc-stats", default="signed_rel",
                    help="Per-cell statistics whose power to predict cell "
                         "type, per region group, is tested (comma-separated "
                         "from delta, delta_r, signed_rel, signed_sgn; "
                         "'none' skips): celltype_auc.tsv and figures.")
    ap.add_argument("--celltype-auc-folds", type=int, default=5)
    ap.add_argument("--celltype-auc-trees", type=int, default=300,
                    help="Trees per random forest (300, as "
                         "coordination_eval.cv_rf).")
    ap.add_argument("--hotspot-fdr", type=float, default=0.05)
    ap.add_argument("--hotspot-min-contacts", type=int, default=200,
                    help="Contacts a bin needs to be tested (200).")
    ap.add_argument("--no-hotspots", action="store_true")
    ap.add_argument("--min-band-pairs", type=int, default=500,
                    help="A class pair x distance band with fewer contacts "
                         "is flagged sparse and left out of the figures "
                         "(500).")
    ap.add_argument("--gene-annotation", default=None,
                    help="GTF/BED for --ab-orient genes; --genome fills it.")
    ap.add_argument("--genome", default=None,
                    help="Fill chromHMM and genes from "
                         "marker_bin_enrichment.py's downloads (mm10).")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--exclude-chrom", default="chrM,chrX,chrY")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot-format", default="pdf")
    ap.add_argument("--figure-note", default=None,
                    help="A line printed above main_figure (e.g. 'synthetic "
                         "data').")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exclude = {_canon(c) for c in args.exclude_chrom.split(",") if c.strip()}
    if args.chrx and "X" not in exclude:
        logger.info("--chrx: chrX is analysed against the autosomes only, "
                    "and kept out of everything else")
        exclude.add("X")
    want = [s.strip() for s in args.systems.split(",") if s.strip()]
    signed_figs = [t.strip() for t in args.signed_figures.split(",")
                   if t.strip()]
    bad = set(signed_figs) - set(_SIGNED_STAT)
    if bad:
        sys.exit(f"unknown --signed-figures {sorted(bad)}; choose from "
                 f"{sorted(_SIGNED_STAT)}")
    bad = set(want) - set(SYSTEMS)
    if bad:
        sys.exit(f"unknown --systems {sorted(bad)}; choose from {SYSTEMS}")
    bad = {x.strip() for x in args.celltype_auc_stats.split(",")
           if x.strip() and x.strip() != "none"} - set(COV_STATS)
    if bad:
        sys.exit(f"unknown --celltype-auc-stats {sorted(bad)}; choose from "
                 f"{COV_STATS}")
    edges = np.array([float(x) for x in args.bands.split(",")])
    band_labels = _bands(edges)
    contexts = _parse_pairs(args.posteriors)

    samples = pd.read_csv(manifest_text(args.samples), sep="\t", dtype=str)
    samples = samples.set_index(args.cell_column)
    strat_cols = [args.group_column] + [
        c.strip() for c in (args.null_match_column or "").split(",") if c.strip()]

    enr = _enrichment()
    preset = enr._GENOME.get(args.genome, {}) if args.genome else {}
    cache = Path(args.cache_dir) if args.cache_dir else (
        Path.home() / ".cache" / "marker_bin_enrichment"
        / (args.genome or "custom"))

    systems: dict = {}
    rt_log2_tracks = [_rt_path(x) for x in (args.rt_log2 or [])]
    have_rt = bool(rt_log2_tracks) or bool(args.rt_early and args.rt_late)
    if "rt" in want:
        if have_rt:
            systems["rt"] = read_rt(args.rt_early, args.rt_late,
                                    args.rt_window, args.rt_mid, exclude,
                                    log2=rt_log2_tracks)
        else:
            logger.warning("rt: needs --rt-early and --rt-late, or "
                           "--rt-log2; skipped")
    if "chromhmm" in want:
        bed = enr._resolve(args.chromhmm or preset.get("chromhmm"), cache,
                           "chromhmm")
        grp = enr._resolve(args.chromhmm_groups
                           or preset.get("chromhmm_groups"), cache,
                           "chromhmm-groups")
        if bed and grp:
            systems["chromhmm"] = read_chromhmm(bed, grp, enr)
        else:
            logger.warning("chromhmm: needs --chromhmm and --chromhmm-groups "
                           "(or --genome mm10); skipped")
    if "pmd" in want:
        src = args.pmd or (preset.get("annotations", {}).get("PMD") or {}
                           ).get("url")
        pth = enr._resolve(src, cache, "PMD") if src else None
        if pth:
            systems["pmd"] = read_pmd(pth, args.pmd_class_col,
                                      args.pmd_common_only)
        else:
            logger.warning("pmd: needs --pmd (or a --genome that has one: "
                           "hg38, hg19); skipped")
    for k, v in _parse_pairs(args.annotation).items():
        if k in SYSTEMS:
            sys.exit(f"--annotation {k}: that name is a built-in system")
        files = [enr._resolve(x.strip(), cache, f"annotation {k}")
                 for x in v.split(",") if x.strip()]
        if files and all(files):
            systems[k] = read_annotation(
                k, files if len(files) > 1 else files[0])
            logger.info(f"annotation {k}: classes {systems[k].classes}")

    # Genes and CpG islands: the default open sites (CpG-island TSSs) and
    # the hotspot map's tracks.
    genes = cpg = None
    if not (args.no_covariates and args.no_hotspots) or args.imprinting \
            or args.pcdh:
        ga = enr._resolve(args.gene_annotation
                          or preset.get("gene_annotation"), cache,
                          "gene-annotation")
        if ga:
            genes = enr._read_genes(ga)
        cp = enr._resolve(args.cpg_islands or preset.get("cpg_islands"),
                          cache, "CpG islands")
        if cp:
            cpg = (enr._read_ucsc_bed(cp, cols=(1, 2, 3))
                   if "cpgisland" in Path(cp).name.lower()
                   else enr._read_bed(cp))
    open_g = None
    if not args.no_covariates:
        open_g, open_label = open_site_coords(args.open_sites, genes, cpg,
                                              exclude)
        if open_g is None:
            logger.warning("labeling efficiency: needs --open-sites, "
                           "--gene-annotation or --genome; skipped")
        else:
            logger.info(f"labeling efficiency: GCH at {open_g.size:,} "
                        f"{open_label}")
    imp_domains, tss_g = pd.DataFrame(), None
    dsets: dict = {}               # {set name: its domains}
    if args.imprinting:
        if args.imprinting_bed:
            imp_domains = read_domains(enr._resolve(
                args.imprinting_bed, cache, "imprinting BED"))
        elif genes is not None:
            imp_domains = imprinted_domains(genes, merge=args.imprinting_merge,
                                            flank=args.imprinting_flank,
                                            exclude=exclude)
        if imp_domains.empty:
            logger.warning("imprinting: no domains -- needs --imprinting-"
                           "bed, or a gene annotation (--gene-annotation or "
                           "--genome) with imprinted genes; skipped")
        else:
            dsets["imprinting"] = imp_domains
    if args.pcdh:
        pc_ = (pcdh_domains(genes, flank=args.imprinting_flank)
               if genes is not None else pd.DataFrame())
        if pc_.empty:
            logger.warning("pcdh: no clustered protocadherin genes -- needs "
                           "a gene annotation (--gene-annotation or "
                           "--genome); skipped")
        else:
            dsets["pcdh"] = pc_
    for nm_, dm_ in dsets.items():
        dm_.to_csv(out / f"{nm_}_domains.tsv", sep="\t", index=False)
        logger.info(f"{nm_}: {len(dm_)} domains: " + ", ".join(
            f"{r_['name']} ({r_['chrom']}:{r_['start']:,}-{r_['end']:,})"
            for _, r_ in dm_.iterrows()))
    if dsets and genes is not None:
        tss_g = (_codes_of(genes["chrom"]) * STRIDE + np.where(
            genes["strand"].to_numpy() == "-",
            genes["end"].to_numpy(np.int64) - 1,
            genes["start"].to_numpy(np.int64)))
    # every set's domains, for each cell's level in them
    loci_g, loci_of = None, {}
    if dsets:
        parts_, at = [], 0
        for nm_, dm_ in dsets.items():
            cd_ = _codes_of(dm_["chrom"]) * STRIDE
            parts_.append(np.column_stack(
                [cd_ + dm_["start"].to_numpy(np.int64),
                 cd_ + dm_["end"].to_numpy(np.int64)]))
            loci_of[nm_] = slice(at, at + len(dm_))
            at += len(dm_)
        loci_g = np.concatenate(parts_)
    acc_ctx = next((c for c in contexts if c.upper() == "GCH"),
                   next((c for c in contexts if c.upper().startswith("GC")),
                        None))
    hcg_ctx = next((c for c in contexts if c.upper() == "HCG"),
                   next((c for c in contexts if c.upper().endswith("CG")),
                        None))
    hot_user = {k: enr._read_bed(enr._resolve(v, cache, f"hotspot {k}"))
                for k, v in _parse_pairs(args.hotspot_annotation).items()}

    # groups of cells to compare
    group_of: dict = {}
    contrast_names: list = []
    cpairs: list = []
    group_batches = pd.DataFrame()
    if args.contrast_group:
        cg = _parse_pairs(args.contrast_group)
        if len(cg) < 2:
            sys.exit("--contrast-group: give two or more groups, "
                     "NAME=TYPE[,TYPE...] each")
        if args.contrast_reference is not None and \
                args.contrast_reference not in cg:
            sys.exit(f"--contrast-reference {args.contrast_reference!r} is "
                     f"not a --contrast-group (have: {list(cg)})")
        ccol = args.contrast_column or args.group_column
        if ccol not in samples.columns:
            sys.exit(f"--contrast-column {ccol!r} not in --samples")
        lab = samples[ccol].astype(str).str.strip()
        for nm, v in cg.items():
            want = {x.strip().lower() for x in v.split(",") if x.strip()}
            hit = lab[lab.str.lower().isin(want)]
            if hit.empty:
                sys.exit(f"--contrast-group {nm}: no cell with {ccol} in "
                         f"{sorted(want)} (have: "
                         f"{sorted(lab.unique())[:20]})")
            for c in hit.index:
                if c in group_of:
                    sys.exit(f"--contrast-group: cell {c} is in two groups")
                group_of[c] = nm
            contrast_names.append(nm)
        cpairs = contrast_pairs(contrast_names, args.contrast_reference)
        logger.info("contrast groups: " + ", ".join(
            f"{nm} ({sum(1 for v in group_of.values() if v == nm):,} cells)"
            for nm in contrast_names) + "; compared: " + ", ".join(
            f"{a} − {b}" for a, b in cpairs))
        # which batches each group's cells come from: a group alone in its
        # batches can't be told from them
        gser = pd.Series(group_of, name="group")
        rows_ = [{"group": g_, "batch_column": "all", "batch": "all",
                  "cells": int(n_)}
                 for g_, n_ in gser.value_counts().reindex(
                     contrast_names).items()]
        for b in [x.strip() for x in (args.batch_column or "").split(",")
                  if x.strip() and x.strip() in samples.columns]:
            tab_ = pd.crosstab(gser, samples.loc[gser.index, b])
            for g_, r_ in tab_.iterrows():
                for lv, n_ in r_.items():
                    if n_:
                        rows_.append({"group": g_, "batch_column": b,
                                      "batch": lv, "cells": int(n_)})
            logger.info(f"contrast groups by {b}:\n{tab_.to_string()}")
        group_batches = pd.DataFrame(rows_)
    contrast_rows, chot_rows, cenr_rows = [], [], []
    var_rows, var_en = [], []
    chrx_rows, chrx_cells = [], []
    dom_rows = {k: [] for k in dsets}
    dom_crows = {k: [] for k in dsets}
    dom_wins = {k: [] for k in dsets}

    from joblib import Parallel, delayed
    first = next(iter(contexts.values()))
    cells_all = sorted(f.name[: -len(".posteriors.parquet")]
                       for f in Path(first).glob("*.posteriors.parquet"))
    cells_all = [c for c in cells_all if c in samples.index]
    if not cells_all:
        sys.exit(f"no cells of {args.samples} under {first}")
    sex_tab, sex_of = None, {}
    chrx_on = args.chrx
    if args.chrx:
        cnt = Parallel(n_jobs=args.n_threads, prefer="threads")(
            delayed(_sex_counts)(Path(first) / f"{c}.posteriors.parquet")
            for c in cells_all)
        sex_tab = call_sex(dict(zip(cells_all, cnt)))
        sex_tab.insert(1, "cell_type", [str(samples.at[c, args.group_column])
                                        for c in sex_tab["cell"]])
        sex_tab["sex"] = sex_tab["sex_call"]
        if args.sex_column:
            if args.sex_column not in samples.columns:
                sys.exit(f"--sex-column {args.sex_column!r} not in --samples")
            sex_tab["sex_given"] = _given_sex(
                samples.loc[sex_tab["cell"], args.sex_column].to_numpy())
            both = (sex_tab["sex_given"] != "unknown") & (
                sex_tab["sex_call"] != "unknown")
            agree = int((both & (sex_tab["sex_given"]
                                 == sex_tab["sex_call"])).sum())
            logger.info(f"sex: --sex-column agrees with the coverage call "
                        f"in {agree:,} of {int(both.sum()):,} cells")
            sex_tab["sex"] = sex_tab["sex_given"]
        sex_of = dict(zip(sex_tab["cell"], sex_tab["sex"]))
        sex_tab.to_csv(out / "cell_sex.tsv", sep="\t", index=False)
        logger.info("sex per cell type:\n" + pd.crosstab(
            sex_tab["cell_type"], sex_tab["sex"]).to_string())
        if not (sex_tab["pairs_X"] > 0).any():
            logger.warning(
                "--chrx: no cell has a read pair on chrX -- its posteriors "
                "were made without it (nome_hic_concordance drops chrX and "
                "chrY unless given --exclude-chrom none).  Rerun its steps "
                "2 and 4 with --exclude-chrom none into a new --output-dir, "
                "its 03_mixture_model/em_fit_all.json copied in so the "
                "autosomes' posteriors stay as they are, and point "
                "--posteriors there.  chrX skipped.")
            chrx_on = False

    ab_maps = None
    # One pass over every cell's read pairs: the A/B bins (with ab) and the
    # cell-cycle covariates (unless --no-covariates).  The repli-score needs
    # the replication-timing classes even when rt is not a system.
    rt_cov = systems.get("rt")
    if rt_cov is None and have_rt and not args.no_covariates:
        rt_cov = read_rt(args.rt_early, args.rt_late, args.rt_window, 0.0,
                         exclude, log2=rt_log2_tracks)
    cell_cov = None
    by_cell = None
    if "ab" in want or not args.no_covariates:
        logger.info(f"reading {len(cells_all):,} cells' read pairs for "
                    + " and ".join(x for x, on in (
                        ("A/B", "ab" in want),
                        ("cell-cycle covariates", not args.no_covariates))
                        if on))
        passed = Parallel(n_jobs=args.n_threads, prefer="threads")(
            delayed(_contact_pass)(Path(first) / f"{c}.posteriors.parquet",
                                   args.min_distance, args.ab_resolution,
                                   exclude, rt_cov)
            for c in cells_all)
        by_cell = [x[0] for x in passed]
        if not args.no_covariates:
            cell_cov = pd.DataFrame([dict(cell=c, cell_type=str(
                samples.at[c, args.group_column]), **x[1])
                for c, x in zip(cells_all, passed)])
            cell_cov["log10_contacts"] = np.log10(
                cell_cov["contacts_25kb"].clip(lower=1))
        del passed
    if "ab" in want:
        orient = args.ab_orient
        if orient == "auto":
            orient = "rt" if have_rt else "genes"
        res = args.ab_resolution
        if orient == "rt":
            if not have_rt:
                sys.exit("--ab-orient rt needs --rt-early and --rt-late, or "
                         "--rt-log2")

            def ref(code, name, nb):
                st = np.arange(nb, dtype=np.int64) * res
                if rt_log2_tracks:
                    return rt_from_log2(rt_log2_tracks,
                                        np.full(nb, code, np.int64), st,
                                        st + res)
                return rt_log2(args.rt_early, args.rt_late,
                               np.full(nb, code, np.int64), st, st + res)
        else:
            ga = enr._resolve(args.gene_annotation
                              or preset.get("gene_annotation"), cache,
                              "gene-annotation")
            if not ga:
                sys.exit("ab: orient by genes needs --gene-annotation or "
                         "--genome")
            genes = enr._read_genes(ga)
            tss = np.where(genes["strand"].to_numpy() == "-",
                           genes["end"].to_numpy() - 1,
                           genes["start"].to_numpy())
            gch = genes["chrom"].map(_canon).to_numpy()

            def ref(code, name, nb):
                return np.bincount(np.minimum(tss[gch == name] // res,
                                              nb - 1),
                                   minlength=nb).astype(float)
        logger.info(f"A/B: pooling contacts per cell type at {res:,} bp, "
                    f"oriented by {orient}")
        ctype = [str(samples.at[c, args.group_column]) for c in cells_all]
        ab_maps, ab_tab, ab_qc = compartments_from_contacts(
            by_cell, ctype, res, ref, args.ab_min_contacts)
        if cell_cov is not None:
            cell_cov["compartment_strength"] = compartment_strength(
                by_cell, ctype, ab_maps, res,
                args.compartment_min_distance)
        ab_tab.to_csv(out / "compartments_ab.tsv", sep="\t", index=False)
        ab_qc.to_csv(out / "compartments_qc.tsv", sep="\t", index=False)
        logger.info("A/B per cell type:\n" + ab_qc.to_string(index=False))
    del by_cell

    percell_rows, summary_rows, marg_rows, cov_rows = [], [], [], []
    ctx_cov: dict = {}
    locus_rows: list = []
    extra_cov: dict = {}           # cell-level covariates of the later checks
    profiles: dict = {}            # per context: the cells' regional profiles
    hot_rows, enrich_rows, chrom_rows = [], [], []
    prof_bin = args.substructure_bin if cell_cov is not None else 0
    for ci_ctx, (ctx, d) in enumerate(contexts.items()):
        cells = [c for c in cells_all
                 if (Path(d) / f"{c}.posteriors.parquet").exists()]
        types = np.array([str(samples.at[c, args.group_column])
                          for c in cells])
        stratum = np.array(["|".join(str(samples.at[c, k]) for k in strat_cols)
                            for c in cells])
        strat_code, strat_names = pd.factorize(stratum)
        ms = _min_sites(args.min_sites, ctx)

        def one(i, c):
            sysd = dict(systems)
            if ab_maps is not None:
                t = types[i]
                sysd["ab"] = ab_maps.get(t, ab_maps["__all__"])
            return _read_cell(Path(d) / f"{c}.posteriors.parquet", ms,
                              args.min_distance, args.max_distance, exclude,
                              sysd, open_g if ctx == acc_ctx else None,
                              args.open_site_window, prof_bin,
                              with_x=chrx_on, loci=loci_g)
        logger.info(f"{ctx}: reading {len(cells):,} cells")
        res_ = Parallel(n_jobs=args.n_threads, prefer="threads")(
            delayed(one)(i, c) for i, c in enumerate(cells))
        ctx_cov[ctx] = pd.DataFrame({
            "cell": cells,
            "level": [r["level"][0] / r["level"][1] if r["level"][1]
                      else np.nan for r in res_],
            **{k: [r["quality"][k] for r in res_]
               for k in ("sites_mean", "decisive", "frac_called")}})
        chrom_names = {}
        for r in res_:
            chrom_names.update(r["chroms"])
        if ctx == acc_ctx and open_g is not None:
            # GCH methylated at open sites: labeling efficiency
            extra_cov["open_site_level"] = {
                c: (r["open"][0] / r["open"][1]
                    if r.get("open") and r["open"][1] >= 30 else np.nan)
                for c, r in zip(cells, res_)}
        if prof_bin:
            profiles[ctx] = {c: r["prof"] for c, r in zip(cells, res_)
                             if r.get("prof") is not None}
        grid = BinGrid(chrom_names, 300_000_000, args.hotspot_bin)
        best_hot = best_arrays = None
        type_code = pd.factorize(types)[0]
        cgroups = ({nm: np.flatnonzero([group_of.get(c) == nm
                                        for c in cells])
                    for nm in contrast_names} if contrast_names else {})
        # the pairs whose two groups both have cells in this context
        cpairs_ctx = [(a_, b_) for a_, b_ in cpairs
                      if cgroups[a_].size and cgroups[b_].size]
        cell = np.concatenate([np.full(r["g_lo"].size, i, np.int64)
                               for i, r in enumerate(res_)])
        g_lo = np.concatenate([r["g_lo"] for r in res_])
        g_hi = np.concatenate([r["g_hi"] for r in res_])
        a = np.concatenate([r["a"] for r in res_]).astype(float)
        p = np.concatenate([r["p"] for r in res_]).astype(float)
        dist = g_hi - g_lo
        band = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0,
                       len(edges) - 2)
        half = np.random.default_rng([args.seed, ci_ctx]).integers(
            0, 2, g_lo.size)
        logger.info(f"{ctx}: {g_lo.size:,} contacts in {len(cells):,} cells")
        dbins = _ls.log_dbins(args.min_distance, args.max_distance,
                              args.bins_per_decade)
        order = ([x for x in SYSTEMS if x in systems
                  or (x == "ab" and ab_maps)]
                 + [x for x in systems if x not in SYSTEMS])
        for si, name in enumerate(order):
            classes = (ab_maps["__all__"].classes if name == "ab"
                       else systems[name].classes)
            n_k = len(classes)
            marg = np.stack([r["marg"][name] for r in res_])   # cells x 3 x K
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = marg[:, 1] / marg[:, 0]
                sd = np.sqrt(np.maximum(marg[:, 2] / marg[:, 0] - mu ** 2, 0))
            kx = np.concatenate([r["cls"][name][0] for r in res_]).astype(
                np.int64)
            ky = np.concatenate([r["cls"][name][1] for r in res_]).astype(
                np.int64)
            a_c = np.where(kx >= 0, a - mu[cell, np.maximum(kx, 0)], np.nan)
            p_c = np.where(ky >= 0, p - mu[cell, np.maximum(ky, 0)], np.nan)
            ok, right, left, reused = reshuffle(
                g_lo, g_hi, dist, cell, kx, ky, a_c, p_c, strat_code[cell],
                n_k, dbins, args.position_bin, args.max_widen,
                seed=args.seed + 7919 * ci_ctx + si,
                names=list(strat_names), n_threads=args.n_threads)
            # The pair's classes, unordered unless asked: which end of a
            # contact lies on the left carries no meaning.
            if args.ordered_pairs:
                k1s, k2s = kx, ky
            else:
                k1s, k2s = np.minimum(kx, ky), np.maximum(kx, ky)
            for k1 in range(n_k):
                for k2 in range(n_k):
                    s_ = (k1s == k1) & (k2s == k2) & (kx >= 0) & (ky >= 0)
                    if not s_.any():
                        continue
                    cov_rows.append({
                        "context": ctx, "system": name, "anchor": classes[k1],
                        "partner": classes[k2], "contacts": int(s_.sum()),
                        "with_donor": float(ok[s_].mean())})
            cov_rows.append({"context": ctx, "system": name,
                             "anchor": "any", "partner": "any",
                             "contacts": int(g_lo.size),
                             "annotated": float(np.mean((kx >= 0)
                                                        & (ky >= 0))),
                             "with_donor": float(ok.mean()) if ok.size
                             else np.nan,
                             "donors_shared": reused / max(2 * ok.sum(), 1)})
            idx = np.flatnonzero(ok)
            x_, y_ = a_c[idx], p_c[idx]
            # null product: the mean of reshuffling either end
            n1 = 0.5 * (x_ * right[0, idx] + left[0, idx] * y_)
            n2 = 0.5 * (x_ * right[1, idx] + left[1, idx] * y_)
            # each contact's part of ΔC, and of the control
            d_, c_ = 2 * (x_ * y_ - n1), 2 * (n2 - n1)
            if idx.size and (
                    args.hotspot_system == name
                    or (args.hotspot_system == "auto" and (
                        best_hot is None or idx.size > best_hot[0]))):
                best_hot = (idx.size, name, None if args.no_hotspots else
                            locus_hotspots(grid, g_lo[idx], g_hi[idx],
                                           cell[idx], d_, c_, band[idx],
                                           len(edges) - 1))
                best_arrays = (g_lo[idx], cell[idx], d_,
                               k1s[idx] * n_k + k2s[idx], g_hi[idx], c_,
                               band[idx])
            # Co-open / co-closed mass, for F.  A donor's posterior is put
            # on the RECEIVING cell's level in that class (its centred value
            # plus that cell's mean), as the centring does for C, so a
            # cell's own level does not pass for a direction.
            a_, p_ = a[idx], p[idx]
            mu_l = mu[cell[idx], kx[idx]]
            mu_r = mu[cell[idx], ky[idx]]
            w = {"xy": x_ * y_, "xy1": n1, "xy2": n2,
                 "oo": a_ * p_, "oc": (1 - a_) * (1 - p_)}
            for dr in (0, 1):
                qr = np.clip(right[dr, idx] + mu_r, 0, 1)
                ql = np.clip(left[dr, idx] + mu_l, 0, 1)
                w[f"no{dr + 1}"] = 0.5 * (a_ * qr + ql * p_)
                w[f"nc{dr + 1}"] = 0.5 * ((1 - a_) * (1 - qr)
                                          + (1 - ql) * (1 - p_))
            acc = accumulate(cell[idx], k1s[idx], k2s[idx], band[idx],
                             half[idx], w, len(cells), n_k, len(edges) - 1)
            # every annotated contact of the system together.  Its signed
            # value gives each contact its class pair's weight (the pair's
            # co-open share against the cell's genome-wide one, all
            # distances) and averages: the contact-weighted mean of the class
            # pairs' signed values -- over all classes at once the share is
            # the genome-wide one, and the weight 0 by definition
            A_ = len(edges) - 1
            tot = {k: acc[k].sum((1, 2)) for k in ("n", "xy", "xy1", "xy2",
                                                   "oo", "oc", "no2", "nc2")}
            ci_ = cell[idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                fg_c = tot["oo"][:, A_] / (tot["oo"][:, A_] + tot["oc"][:, A_])
                fgn_c = tot["no2"][:, A_] / (tot["no2"][:, A_]
                                              + tot["nc2"][:, A_])
                Fp = acc["oo"][..., A_] / (acc["oo"][..., A_]
                                           + acc["oc"][..., A_])
                Fpn = acc["no2"][..., A_] / (acc["no2"][..., A_]
                                             + acc["nc2"][..., A_])
                wk = np.nan_to_num(_weight(Fp[ci_, k1s[idx], k2s[idx]],
                                           fg_c[ci_]))
                wkn = np.nan_to_num(_weight(Fpn[ci_, k1s[idx], k2s[idx]],
                                            fgn_c[ci_]))
                ek = np.nan_to_num(2 * Fp[ci_, k1s[idx], k2s[idx]] - 1)
                ekn = np.nan_to_num(2 * Fpn[ci_, k1s[idx], k2s[idx]] - 1)
            sgn_parts = {"sd": d_ * wk, "sc": c_ * wkn,
                         "gd": d_ * np.sign(wk), "gc": c_ * np.sign(wkn),
                         "ed": d_ * ek, "ec": c_ * ekn}
            nb1 = len(edges)
            kb_ = ci_ * nb1 + band[idx]
            ka_ = ci_ * nb1 + A_
            S_sg = {k: (np.bincount(kb_, weights=v, minlength=len(cells) * nb1)
                        + np.bincount(ka_, weights=v,
                                      minlength=len(cells) * nb1)
                        ).reshape(len(cells), nb1)
                    for k, v in sgn_parts.items()}
            if best_hot is not None and best_hot[1] == name and \
                    best_arrays is not None:
                best_arrays = best_arrays[:7] + (sgn_parts["sd"],
                                                 sgn_parts["sc"])
            if cpairs_ctx:
                for r_ in group_contrast(
                        acc, sd, tot, S_sg, cgroups, classes, band_labels,
                        args.n_boot, args.seed + 97 * si + ci_ctx,
                        pairs=cpairs_ctx):
                    contrast_rows.append({"context": ctx, "system": name,
                                          **r_})
            if cell_cov is not None and args.n_locus_perm > 0 and idx.size:
                pr = k1s[idx] * n_k + k2s[idx]
                with np.errstate(invalid="ignore", divide="ignore"):
                    s12 = sd[ci_, k1s[idx]] * sd[ci_, k2s[idx]]
                    inv = np.where(s12 > 0, 0.5 / np.where(s12 > 0, s12, 1),
                                   0.0)
                lr = locus_reassignment(
                    ci_, type_code, pr, band[idx],
                    {"d": d_, "c": c_, "r": d_ * inv, "rc": c_ * inv,
                     "oo": w["oo"], "oc": w["oc"], "no2": w["no2"],
                     "nc2": w["nc2"]}, len(edges) - 1,
                    args.min_pairs, args.n_locus_perm,
                    np.random.default_rng([args.seed, ci_ctx, si, 4049]),
                    fg=fg_c, fgn=fgn_c)
                if not lr.empty:
                    lr.insert(0, "context", ctx)
                    lr.insert(1, "system", name)
                    lr.insert(2, "anchor",
                              np.asarray(classes)[lr["pair"] // n_k])
                    lr.insert(3, "partner",
                              np.asarray(classes)[lr["pair"] % n_k])
                    lr.insert(4, "band",
                              np.asarray(band_labels)[lr["band_index"]])
                    locus_rows.append(lr.drop(columns=["pair",
                                                       "band_index"]))
                lr = locus_reassignment(
                    ci_, type_code, np.zeros(idx.size, np.int64), band[idx],
                    {"d": d_, "c": c_, "r": d_ * inv, "rc": c_ * inv,
                     **{k: sgn_parts[k] for k in ("sd", "sc", "gd", "gc")}},
                    len(edges) - 1, args.min_pairs, args.n_locus_perm,
                    np.random.default_rng([args.seed, ci_ctx, si, 4051]))
                if not lr.empty:
                    lr.insert(0, "context", ctx)
                    lr.insert(1, "system", name)
                    lr.insert(2, "anchor", ALL_REGIONS)
                    lr.insert(3, "partner", ALL_REGIONS)
                    lr.insert(4, "band",
                              np.asarray(band_labels)[lr["band_index"]])
                    locus_rows.append(lr.drop(columns=["pair",
                                                       "band_index"]))
            summary_rows += summarize(acc, sd, mu, cells, types, classes,
                                      band_labels, edges, args.min_pairs,
                                      args.n_boot, args.seed + si, ctx, name)
            marg_rows += marginal_summary(marg, cells, types, classes,
                                          args.min_ends, ctx, name)
            nz = np.argwhere(acc["n"] > 0)
            if nz.size:
                c_, i_, j_, b_ = nz.T
                n_ = acc["n"][c_, i_, j_, b_]
                xy = acc["xy"][c_, i_, j_, b_]
                xy1 = acc["xy1"][c_, i_, j_, b_]
                xy2 = acc["xy2"][c_, i_, j_, b_]
                percell_rows.append(pd.DataFrame({
                    "context": ctx, "system": name,
                    "cell": np.asarray(cells)[c_], "cell_type": types[c_],
                    "anchor": np.asarray(classes)[i_],
                    "partner": np.asarray(classes)[j_],
                    "band": np.asarray(band_labels)[b_], "pairs": n_,
                    "enough_pairs": n_ >= args.min_pairs,
                    "C_obs": 2 * xy / n_, "C_null": 2 * xy1 / n_,
                    "delta": 2 * (xy - xy1) / n_,
                    "control": 2 * (xy2 - xy1) / n_,
                    "mu_anchor": mu[c_, i_], "mu_partner": mu[c_, j_],
                    "sd_anchor": sd[c_, i_], "sd_partner": sd[c_, j_]}))
                pcr = percell_rows[-1]
                with np.errstate(invalid="ignore", divide="ignore"):
                    s12_ = (pcr["sd_anchor"] * pcr["sd_partner"]).to_numpy()
                    s12_ = np.where(s12_ > 0, s12_, np.nan)
                    pcr["delta_r"] = pcr["delta"].to_numpy() / 2 / s12_
                    pcr["control_r"] = pcr["control"].to_numpy() / 2 / s12_
                    oo_, oc_, no2_, nc2_ = (acc[k][c_, i_, j_, b_]
                                            for k in ("oo", "oc", "no2",
                                                      "nc2"))
                    pcr["F"] = oo_ / (oo_ + oc_)
                    pcr["F_null"] = no2_ / (no2_ + nc2_)
                    pcr["signed"] = _signed(pcr["delta"].to_numpy(), oo_, oc_)
                    pcr["signed_control"] = _signed(
                        pcr["control"].to_numpy(), no2_, nc2_)
                    A_ = len(band_labels) - 1
                    gk = {k: acc[k][:, :, :, A_].sum((1, 2))
                          for k in ("oo", "oc", "no2", "nc2")}
                    fg_ = (gk["oo"] / (gk["oo"] + gk["oc"]))[c_]
                    fgn_ = (gk["no2"] / (gk["no2"] + gk["nc2"]))[c_]
                    pcr["F_ref"] = fg_
                    pcr["signed_rel"] = _signed_rel(
                        pcr["delta"].to_numpy(), oo_, oc_, fg_)
                    pcr["signed_rel_control"] = _signed_rel(
                        pcr["control"].to_numpy(), no2_, nc2_, fgn_)
                    pcr["signed_sgn"] = _signed_sgn(
                        pcr["delta"].to_numpy(), oo_, oc_, fg_)
                    pcr["signed_sgn_control"] = _signed_sgn(
                        pcr["control"].to_numpy(), no2_, nc2_, fgn_)
            ci_a, bi_a = np.nonzero(tot["n"] > 0)
            if ci_a.size:
                g_ = {k: v[ci_a, bi_a] for k, v in tot.items()}
                with np.errstate(invalid="ignore", divide="ignore"):
                    s12m = sd[:, :, None] * sd[:, None, :]
                    inv12 = np.where(s12m > 0, 1.0 / np.where(s12m > 0, s12m,
                                                               1), 0.0)
                    r_num = ((acc["xy"] - acc["xy1"])
                             * inv12[..., None]).sum((1, 2))[ci_a, bi_a]
                    rc_num = ((acc["xy2"] - acc["xy1"])
                              * inv12[..., None]).sum((1, 2))[ci_a, bi_a]
                    lev = marg[:, 1].sum(1) / marg[:, 0].sum(1)
                    lsd = np.sqrt(np.maximum(marg[:, 2].sum(1)
                                             / marg[:, 0].sum(1) - lev ** 2, 0))
                    dlt = 2 * (g_["xy"] - g_["xy1"]) / g_["n"]
                    ctl = 2 * (g_["xy2"] - g_["xy1"]) / g_["n"]
                    percell_rows.append(pd.DataFrame({
                        "context": ctx, "system": name,
                        "cell": np.asarray(cells)[ci_a],
                        "cell_type": types[ci_a], "anchor": ALL_REGIONS,
                        "partner": ALL_REGIONS,
                        "band": np.asarray(band_labels)[bi_a],
                        "pairs": g_["n"],
                        "enough_pairs": g_["n"] >= args.min_pairs,
                        "C_obs": 2 * g_["xy"] / g_["n"],
                        "C_null": 2 * g_["xy1"] / g_["n"],
                        "delta": dlt, "control": ctl,
                        "mu_anchor": lev[ci_a], "mu_partner": lev[ci_a],
                        "sd_anchor": lsd[ci_a], "sd_partner": lsd[ci_a],
                        "delta_r": r_num / g_["n"],
                        "control_r": rc_num / g_["n"],
                        "F": g_["oo"] / (g_["oo"] + g_["oc"]),
                        "F_null": g_["no2"] / (g_["no2"] + g_["nc2"]),
                        "F_ref": fg_c[ci_a],
                        **{col_: S_sg[k_][ci_a, bi_a] / g_["n"]
                           for col_, k_ in (
                               ("signed", "ed"), ("signed_control", "ec"),
                               ("signed_rel", "sd"),
                               ("signed_rel_control", "sc"),
                               ("signed_sgn", "gd"),
                               ("signed_sgn_control", "gc"))}}))
            logger.info(f"{ctx} {name}: {ok.mean():.1%} of contacts "
                        f"annotated at both ends with a reshuffled partner")
        if chrx_on:
            # chrX against the autosomes: every contact, X ones included,
            # as one system of two classes
            xs = [r["x"] for r in res_]
            sx = np.array([sex_of.get(c_, "unknown") for c_ in cells])
            grp_x = {"all cells": np.arange(len(cells))}
            kind_x = {"all cells": "all"}
            two_sexes = len({x for x in sx if x != "unknown"}) > 1
            for kind, lab in (("cell type", types),
                              ("sex", sx if two_sexes else None)):
                for g_ in (dict.fromkeys(lab) if lab is not None else ()):
                    if g_ not in grp_x and g_ != "unknown":
                        grp_x[g_] = np.flatnonzero(lab == g_)
                        kind_x[g_] = kind
            for g_, ix_ in cgroups.items():
                if g_ not in grp_x and ix_.size:
                    grp_x[g_], kind_x[g_] = ix_, "contrast group"
            rx, pcx = chromosome_concordance(
                np.concatenate([cell] + [np.full(x["g_lo"].size, i, np.int64)
                                         for i, x in enumerate(xs)]),
                np.concatenate([g_lo] + [x["g_lo"] for x in xs]),
                np.concatenate([g_hi] + [x["g_hi"] for x in xs]),
                np.concatenate([a] + [x["a"].astype(float) for x in xs]),
                np.concatenate([p] + [x["p"].astype(float) for x in xs]),
                np.stack([x["marg"] for x in xs]),
                np.array([f"{st_}|{sx_}" for st_, sx_ in zip(stratum, sx)]),
                grp_x, edges, band_labels, dbins, args.position_bin,
                args.max_widen, args.n_boot, args.seed + 6007 * ci_ctx,
                args.n_threads)
            for r_ in rx:
                r_.update(context=ctx, group_kind=kind_x[r_["group"]])
            chrx_rows += rx
            pcx.insert(0, "context", ctx)
            pcx.insert(1, "cell", cells)
            pcx.insert(2, "cell_type", types)
            pcx.insert(3, "sex", sx)
            chrx_cells.append(pcx)
            r0 = next((r_ for r_ in rx if r_["group"] == "all cells"
                       and r_["statistic"] == "delta"
                       and str(r_["band"]).startswith("all")), None)
            if r0 is not None:
                logger.info(f"{ctx} chrX: X-X {100 * r0['value_B']:.2f} pp "
                            f"vs autosomes {100 * r0['value_A']:.2f} pp, "
                            f"difference {100 * r0['diff']:+.2f} pp "
                            f"[{100 * r0['diff_lo']:+.2f}, "
                            f"{100 * r0['diff_hi']:+.2f}] "
                            f"({r0['pairs_B']:,} X contacts)")
            del xs
        if ctx == hcg_ctx and "pmd" in systems and cell_cov is not None:
            # HMD less PMD methylation: PMDs lose methylation with every
            # division, so their depth is a cell's division history
            mg = np.stack([r["marg"]["pmd"] for r in res_])
            ih = systems["pmd"].classes.index("HMD")
            ip = systems["pmd"].classes.index("PMD")
            with np.errstate(invalid="ignore", divide="ignore"):
                mu_ = mg[:, 1] / mg[:, 0]
            enough = (mg[:, 0, ih] >= 30) & (mg[:, 0, ip] >= 30)
            extra_cov["pmd_depth"] = dict(zip(cells, np.where(
                enough, mu_[:, ih] - mu_[:, ip], np.nan)))
        if best_arrays is not None and cell_cov is not None \
                and args.n_locus_perm > 0:
            # territories: a cell's concordance on each of its chromosomes
            ct = cell_chromosome_test(
                best_arrays[0], best_arrays[1], type_code, best_arrays[2],
                args.n_locus_perm,
                np.random.default_rng([args.seed, ci_ctx, 7331]),
                pair=best_arrays[3])
            ct.update({"context": ctx, "system": best_hot[1]})
            chrom_rows.append(ct)
            logger.info(f"{ctx} cell x chromosome ({best_hot[1]} null): "
                        + ", ".join(f"{k} {ct[k]:.3f}" for k in (
                            "chromosome_share", "cell_share",
                            "sampling_share", "cell_x_chromosome_share",
                            "perm_p") if k in ct))
        tracks = None
        if best_hot is not None and best_hot[2] is not None:
            hs = best_hot[2]
            hs.insert(0, "context", ctx)
            hs.insert(1, "system", best_hot[1])
            hs = call_hotspots(hs, len(edges) - 1, args.hotspot_fdr,
                               args.hotspot_min_contacts)
            hs["chrom"] = hs["code"].map(chrom_names)
            hs["band"] = np.asarray(band_labels)[hs["band_index"]]
            tracks = hotspot_tracks(grid, systems, ab_maps, genes, cpg,
                                    hot_user)
            enrich_rows.append(hotspot_enrichment(hs, tracks))
            for k, v in tracks.items():
                hs[k] = v[hs["grid_index"].to_numpy()]
            hot_rows.append(hs)
            t_ = hs[hs["tested"]]
            logger.info(f"{ctx} hotspots ({best_hot[1]} null, "
                        f"{args.hotspot_bin:,} bp bins): {int(t_['hot'].sum())}"
                        f" hot and {int(t_['cold'].sum())} cold of "
                        f"{len(t_):,} bins at FDR {args.hotspot_fdr:g}")
        if best_arrays is not None and len(best_arrays) >= 9 and \
                not args.no_hotspots:
            # loci concordant in some cell types and not others, and loci
            # concordant alike in all
            tnames = list(pd.factorize(types)[1])
            lv = locus_variability(
                grid, best_arrays[0], best_arrays[4],
                type_code[best_arrays[1]],
                {"delta": (best_arrays[2], best_arrays[5]),
                 "signed_rel": (best_arrays[7], best_arrays[8])},
                tnames, args.variability_min_contacts,
                min(3, len(tnames)), args.variability_fraction,
                args.hotspot_fdr)
            if not lv.empty:
                if tracks is None:
                    tracks = hotspot_tracks(grid, systems, ab_maps, genes,
                                            cpg, hot_user)
                lv.insert(0, "context", ctx)
                lv.insert(1, "system", best_hot[1])
                lv["chrom"] = lv["code"].map(chrom_names)
                var_en.append(variability_enrichment(lv, tracks))
                var_rows.append(lv)
                for st_, g_ in lv.groupby("statistic"):
                    nv = int((g_["set"] == "variable").sum())
                    nvs = int((g_["set_scaled"] == "variable").sum())
                    ns = int((g_["set"] == "stable").sum())
                    logger.info(f"{ctx} {st_}: {nv} variable ({nvs} beyond "
                                f"each type's level) and {ns} stable of "
                                f"{len(g_):,} bins tested across cell types")
        if cpairs_ctx and best_arrays is not None and not args.no_hotspots:
            # where along the genome the groups differ
            masks = {nm: np.isin(np.arange(len(cells)), ix)
                     for nm, ix in cgroups.items()}
            hc = hotspot_contrast(grid, best_arrays, masks, len(edges) - 1,
                                  args.hotspot_min_contacts, args.hotspot_fdr,
                                  pairs=cpairs_ctx)
            if not hc.empty:
                if tracks is None:
                    tracks = hotspot_tracks(grid, systems, ab_maps, genes,
                                            cpg, hot_user)
                hc.insert(0, "context", ctx)
                hc.insert(1, "system", best_hot[1])
                hc["chrom"] = hc["code"].map(chrom_names)
                en_ = pd.DataFrame(hotspot_contrast_enrichment(hc, tracks))
                if not en_.empty:
                    en_.insert(0, "context", ctx)
                    # genome-rotation p where there are bins enough for it
                    for col, p_, alt in (("q_diff", "p_rot_diff", "p_diff"),
                                         ("q_resid", "p_rot_resid", None)):
                        pv = en_[p_].to_numpy(float)
                        if alt is not None:
                            pv = np.where(np.isfinite(pv), pv,
                                          en_[alt].to_numpy(float))
                        en_[col] = np.nan
                        for _, g_ in en_.groupby(["group_A", "group_B"],
                                                 sort=False):
                            en_.loc[g_.index, col] = fdr_bh(
                                pv[en_.index.get_indexer(g_.index)])
                    cenr_rows.append(en_)
                for k, v in tracks.items():
                    hc[k] = v[hc["grid_index"].to_numpy()]
                chot_rows.append(hc)
                for (a_, b_), h_ in hc.groupby(["group_A", "group_B"],
                                               sort=False):
                    logger.info(
                        f"{ctx} {a_} vs {b_}: median bin difference "
                        f"{100 * h_['median_diff'].iloc[0]:+.2f} pp; "
                        f"{int(h_['higher_A'].sum())} bins higher in {a_}, "
                        f"{int(h_['higher_B'].sum())} in {b_}; {a_} = "
                        f"{h_['scale_k'].iloc[0]:.2f} x {b_} "
                        f"{100 * h_['scale_offset'].iloc[0]:+.2f} pp (maps r "
                        f"= {h_['r_maps'].iloc[0]:.2f} beyond noise), "
                        f"{int(h_['above_scaled'].sum())} bins above that, "
                        f"{int(h_['below_scaled'].sum())} below "
                        f"(of {len(h_):,})")
        if dsets and best_arrays is not None:
            # imprinted domains, the protocadherins: the hotspots'
            # per-contact parts inside each domain against matched windows
            grp_i = {"all cells": np.arange(len(cells))}
            kind_i = {"all cells": "all"}
            for g_ in dict.fromkeys(types):
                if g_ not in grp_i:
                    grp_i[g_], kind_i[g_] = np.flatnonzero(types == g_), \
                        "cell type"
            for g_, ix_ in cgroups.items():
                if g_ not in grp_i and ix_.size:
                    grp_i[g_], kind_i[g_] = ix_, "contrast group"
            lv_all = np.stack([r["loci"] for r in res_])
            for si_, (nm_, dm_) in enumerate(dsets.items()):
                ir, icr, iw = domain_concordance(
                    dm_, best_arrays[0], best_arrays[4], best_arrays[1],
                    best_arrays[2], best_arrays[5], len(cells), grp_i,
                    pairs=cpairs_ctx, tss_g=tss_g, pair=best_arrays[3],
                    n_controls=args.imprinting_windows,
                    min_contacts=args.imprinting_min_contacts,
                    n_boot=args.n_boot,
                    seed=args.seed + 3001 * ci_ctx + 17 * si_,
                    pooled=nm_ != "pcdh")
                lr, lcr = domain_levels(
                    lv_all[:, loci_of[nm_]], list(dm_["name"]), grp_i,
                    cpairs_ctx, args.n_boot,
                    args.seed + 3001 * ci_ctx + 17 * si_ + 5)
                lvd = {(r_["group"], r_["domain"]): r_ for r_ in lr}
                lcd = {(r_["group_A"], r_["group_B"], r_["domain"]): r_
                       for r_ in lcr}
                for r_ in ir:
                    r_.update(context=ctx, system=best_hot[1],
                              group_kind=kind_i[r_["group"]])
                    x_ = lvd.get((r_["group"], r_["domain"]))
                    if x_ is not None:
                        r_.update({k: x_[k] for k in (
                            "level", "level_lo", "level_hi", "level_ends")})
                for r_ in icr:
                    r_.update(context=ctx, system=best_hot[1])
                    x_ = lcd.get((r_["group_A"], r_["group_B"],
                                  r_["domain"]))
                    if x_ is not None:
                        r_.update({k: x_[k] for k in (
                            "level_diff", "level_diff_lo", "level_diff_hi",
                            "level_p")})
                dom_rows[nm_] += ir
                dom_crows[nm_] += icr
                if not iw.empty:
                    iw.insert(0, "context", ctx)
                    iw.insert(1, "system", best_hot[1])
                    iw.insert(3, "chrom", iw["code"].map(chrom_names))
                    dom_wins[nm_].append(iw.drop(columns="code"))
                head = "all domains" if nm_ != "pcdh" else "Pcdh cluster"
                r0 = next((r_ for r_ in ir if r_["group"] == "all cells"
                           and r_["domain"] == head), None)
                if r0 is not None:
                    logger.info(
                        f"{ctx} {nm_} -- {head} ({best_hot[1]} null): "
                        f"{100 * r0['delta']:.2f} pp vs matched windows "
                        f"{100 * r0['windows_mean']:.2f} pp, excess "
                        f"{100 * r0['excess']:+.2f} pp "
                        f"[{100 * r0['excess_lo']:+.2f}, "
                        f"{100 * r0['excess_hi']:+.2f}] "
                        f"({r0['contacts']:,} contacts)")
                for r_ in icr:
                    if r_["domain"] == head:
                        logger.info(
                            f"{ctx} {nm_} -- {head}: {r_['group_A']} - "
                            f"{r_['group_B']} {100 * r_['diff']:+.2f} pp "
                            f"[{100 * r_['lo']:+.2f}, {100 * r_['hi']:+.2f}]"
                            f" (matched windows "
                            f"{100 * r_['windows_mean']:+.2f} pp); level "
                            f"{100 * r_.get('level_diff', np.nan):+.2f} pp")
        best_arrays = None
        del res_

    percell = (pd.concat(percell_rows, ignore_index=True) if percell_rows
               else pd.DataFrame())
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["ordered"] = bool(args.ordered_pairs)
        summary["sparse"] = summary["pairs"] < args.min_band_pairs
    percell.to_csv(out / "percell.tsv.gz", sep="\t", index=False)
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    pd.DataFrame(marg_rows).to_csv(out / "marginal_summary.tsv", sep="\t",
                                   index=False)
    pd.DataFrame(cov_rows).to_csv(out / "coverage.tsv", sep="\t", index=False)
    for (ctx, name), s in summary.groupby(["context", "system"], sort=False):
        pc = percell[(percell["context"] == ctx) & (percell["system"] == name)]
        plot(s[~s["sparse"] | s["band"].str.startswith("all")], pc,
             out / f"fig_{ctx}_{name}.{args.plot_format}",
             f"{ctx} — {name}: long-range concordance by class pair")
    plot_main(summary, percell, out / f"main_figure.{args.plot_format}",
              note=args.figure_note, min_band_pairs=args.min_band_pairs)
    # The same, signed by where the concordant mass sits: ΔC x (2F - 1),
    # against the same formula on reshuffled links alone.
    for (ctx, name), s in summary.groupby(["context", "system"], sort=False):
        pc = percell[(percell["context"] == ctx) & (percell["system"] == name)]
        for tag in signed_figs:
            plot(s[~s["sparse"] | s["band"].str.startswith("all")], pc,
                 out / f"fig_{ctx}_{name}_signed_{tag}.{args.plot_format}",
                 f"{ctx} — {name}: {_SIGNED_TITLE[tag]} by class pair",
                 stat=_SIGNED_STAT[tag])
    for tag in signed_figs:
        plot_main(summary, percell,
                  out / f"main_figure_signed_{tag}.{args.plot_format}",
                  note=args.figure_note, min_band_pairs=args.min_band_pairs,
                  stat=_SIGNED_STAT[tag])
    # What explains the within-type variation: the cell-cycle covariates
    # (with depth and the cell's level beside them).
    if cell_cov is not None:
        for k, v in extra_cov.items():
            cell_cov[k] = cell_cov["cell"].map(v)
        if profiles and args.n_substructure_pcs > 0:
            sp = substructure_pcs(
                profiles, dict(zip(cell_cov["cell"], cell_cov["cell_type"])),
                args.n_substructure_pcs)
            cell_cov = cell_cov.merge(sp, on="cell", how="left")
            logger.info(f"substructure: {args.n_substructure_pcs} PCs within "
                        f"each cell type of {sp.iloc[:, 1].notna().sum():,} "
                        f"cells")
        if profiles:
            sim = []
            ds = doublet_scores(
                profiles, dict(zip(cell_cov["cell"], cell_cov["cell_type"])),
                seed=args.seed, sim_out=sim)
            cell_cov = cell_cov.merge(ds, on="cell", how="left")
            if sim and ds["doublet_score"].notna().any():
                thr = float(np.median(sim[0]))
                logger.info(
                    f"doublet score (share of simulated doublets among a "
                    f"cell's neighbours): median {ds['doublet_score'].median():.3f}"
                    f"; {int((ds['doublet_score'] >= thr).sum()):,} cells at or "
                    f"above the simulated doublets' median ({thr:.3f})")
        profiles.clear()
        meth = {}
        for nm, dirp in _parse_pairs(args.context_counts).items():
            files = {}
            for c in cells_all:
                f = next((f for f in (Path(dirp) / f"{c}.read_pairs.parquet",
                                      Path(dirp) / f"{c}.posteriors.parquet")
                          if f.exists()), None)
                if f is not None:
                    files[c] = f
            if not files:
                logger.warning(f"--context-counts {nm}: no cells' read pairs "
                               f"under {dirp}")
                continue
            kn = Parallel(n_jobs=args.n_threads, prefer="threads")(
                delayed(_count_pass)(f, exclude, open_g,
                                     args.open_site_window)
                for f in files.values())
            meth[nm] = {c: (x[0] / x[1] if x[1] >= 100 else np.nan)
                        for c, x in zip(files, kn)}
            cell_cov[f"meth_{nm}"] = cell_cov["cell"].map(meth[nm])
            if open_g is not None:
                # at open sites and away from them: what follows
                # accessibility is not what the chemistry left
                for j, part in ((2, "open"), (4, "distal")):
                    cell_cov[f"meth_{nm}_{part}"] = cell_cov["cell"].map({
                        c: (x[j] / x[j + 1] if len(x) > j + 1
                            and x[j + 1] >= 100 else np.nan)
                        for c, x in zip(files, kn)})
        up = {k.upper(): k for k in meth}

        def _m(nm, part=""):
            """meth_<name>, away from open sites when that is known: GpC
            methyltransferase off-target activity, if any, sits where the
            DNA is open; the chemistry does not care."""
            c_ = f"meth_{up[nm]}_distal"
            if part == "" and c_ in cell_cov and cell_cov[c_].notna().any():
                return cell_cov[c_]
            return cell_cov[f"meth_{up[nm]}{part}"]
        if "HCY" in up:
            # C neither in a CpG nor a GpC, nor before an A: nothing should
            # methylate it, so what reads methylated was not converted
            cell_cov["nonconversion"] = _m("HCY")
            if f"meth_{up['HCY']}_open" in cell_cov:
                cell_cov["hcy_open_excess"] = (
                    cell_cov[f"meth_{up['HCY']}_open"]
                    - cell_cov[f"meth_{up['HCY']}_distal"])
        if "GCA" in up and "GCY" in up:
            cell_cov["mca_index_gch"] = (cell_cov[f"meth_{up['GCA']}"]
                                         - cell_cov[f"meth_{up['GCY']}"])
        if "HCA" in up and "HCY" in up:
            # CpA outside GpC less the unconverted background: endogenous
            # mCA with no accessibility in it
            cell_cov["mca_index"] = _m("HCA") - _m("HCY")
        elif "mca_index_gch" in cell_cov:
            cell_cov["mca_index"] = cell_cov["mca_index_gch"]
        for col_, what in (("mca_index", "mCA index"),
                           ("nonconversion", "non-conversion (HCY, away "
                                             "from open sites)"),
                           ("hcy_open_excess", "HCY methylated at open "
                                               "sites less away from them")):
            if col_ in cell_cov:
                logger.info(f"{what} by cell type:\n" + cell_cov.groupby(
                    "cell_type")[col_].describe()[["count", "mean", "50%"]]
                    .to_string(float_format=lambda v: f"{v:.4f}"))
        if meth and "mca_index" not in cell_cov:
            logger.warning("the mCA index needs --context-counts HCA=... and "
                           "HCY=... (or GCA=... and GCY=...)")
    if chrom_rows:
        ct_ = pd.DataFrame(chrom_rows)
        lead = ["context", "system"]
        ct_[lead + [c for c in ct_.columns if c not in lead]].to_csv(
            out / "cell_chromosome.tsv", sep="\t", index=False)
        plot_cell_chromosome(ct_, out / f"cell_chromosome."
                                        f"{args.plot_format}")
    if hot_rows:
        hs = pd.concat(hot_rows, ignore_index=True)
        en = pd.concat(enrich_rows, ignore_index=True)
        lead = ["context", "system", "chrom", "start", "end", "band",
                "contacts", "cells", "delta", "control", "se", "se_control",
                "z", "z_control", "tested", "excess", "z_excess", "q_hot",
                "q_cold", "hot", "cold"]
        hs[lead + [c for c in hs.columns if c not in lead + [
            "grid_index", "code", "band_index"]]].to_csv(
            out / "locus_hotspots.tsv.gz", sep="\t", index=False)
        en.to_csv(out / "locus_hotspot_enrichment.tsv", sep="\t",
                  index=False)
        for ctx_ in contexts:
            sy_ = hs.loc[hs["context"] == ctx_, "system"]
            plot_hotspots(hs, en, ctx_, out / f"locus_hotspots_{ctx_}."
                                              f"{args.plot_format}",
                          note=(f" ({SYSTEM_LABEL.get(sy_.iloc[0], sy_.iloc[0])}"
                                f"-matched null, {_fmt_bp(args.hotspot_bin)}"
                                f" bins)" if len(sy_) else ""),
                          fdr=args.hotspot_fdr)
    batch_cols = []
    if cell_cov is not None and not percell.empty:
        batch_cols = [b.strip() for b in (args.batch_column or "").split(",")
                      if b.strip()]
        for b in batch_cols:
            if b in samples.columns:
                cell_cov[b] = cell_cov["cell"].map(samples[b])
            else:
                logger.warning(f"--batch-column {b!r} not in --samples")
        batch_cols = [b for b in batch_cols if b in cell_cov]
        wide = cell_cov.copy()
        for ctx_, cx in ctx_cov.items():
            wide = wide.merge(cx.rename(columns={
                k: f"{k}_{ctx_}" for k in CTX_COVARIATES}), on="cell",
                how="left")
        wide.to_csv(out / "cell_covariates.tsv", sep="\t", index=False)
        plot_cell_cycle(cell_cov, out / f"cell_cycle_covariates."
                                         f"{args.plot_format}")
        plot_extra_covariates(cell_cov, out / f"cell_covariates_extra."
                                              f"{args.plot_format}")
        locus = (pd.concat(locus_rows, ignore_index=True) if locus_rows
                 else pd.DataFrame())
        locus.to_csv(out / "locus_reassignment.tsv", sep="\t", index=False)
        assoc = covariate_association(percell, cell_cov, ctx_cov, locus,
                                      batch_cols,
                                      min_cells=args.covariate_min_cells)
        assoc.to_csv(out / "covariate_association.tsv", sep="\t",
                     index=False)
        for ctx_ in contexts:
            for st_ in ("delta", "delta_r", "signed_rel"):
                plot_covariates(assoc, ctx_, st_,
                                out / f"covariates_{ctx_}_{st_}."
                                      f"{args.plot_format}")
                plot_covariates_by_band(
                    assoc, ctx_, st_,
                    out / f"covariates_{ctx_}_{st_}_by_band."
                          f"{args.plot_format}")
                plot_variance_total(assoc, ctx_, st_,
                                    out / f"variance_total_{ctx_}_{st_}."
                                          f"{args.plot_format}")
        if not assoc.empty:
            top = assoc[(assoc["statistic"] == "delta")
                        & assoc["band"].astype(str).str.startswith("all")
                        ].sort_values("cells", ascending=False).head(10)
            cols_ = [c for c in ("context", "system", "anchor", "partner",
                                 "cells", "noise_share", "locus_share",
                                 "cell_share", "perm_p", "R2_adj_all",
                                 "explained_cell_share", "control_R2_adj")
                     if c in top] + [f"shap_{g}" for g in GROUP_COLOR
                                     if f"shap_{g}" in top
                                     and top[f"shap_{g}"].notna().any()]
            logger.info("within-type variation of ΔC (all distances): "
                        "noise | which loci | cells, and the share of the "
                        "cell part the covariates explain:\n"
                        + top[cols_].to_string(index=False, float_format=
                                               lambda v: f"{v:.3f}"))
        cc_ = pd.concat([cross_class_correlation(
            percell, summary, args.covariate_min_cells, stat=st_,
            cov=cell_cov, ctx_cov=ctx_cov, batch_cols=batch_cols)
            for st_ in ("delta", "signed_rel")], ignore_index=True)
        cc_.to_csv(out / "cross_class_correlation.tsv", sep="\t",
                   index=False)
        for ctx_ in contexts:
            plot_cross_class(cc_, ctx_, out / f"cross_class_{ctx_}."
                                              f"{args.plot_format}",
                             stat="delta")
            plot_cross_class(cc_, ctx_, out / f"cross_class_{ctx_}_"
                                              f"signed_rel.{args.plot_format}",
                             stat="signed_rel")
    # The groups of cells compared.
    def pair_tag(a_, b_):
        """File-name suffix of a pair; none when only one pair is compared."""
        import re
        if len(cpairs) < 2:
            return ""
        return "_" + "_vs_".join(re.sub(r"[^A-Za-z0-9.-]+", "_", str(x))
                                 for x in (a_, b_))
    if not group_batches.empty:
        group_batches.to_csv(out / "contrast_groups.tsv", sep="\t",
                             index=False)
    if contrast_rows:
        ct = pd.DataFrame(contrast_rows)
        ct["q_diff"] = ct["q_scaled"] = np.nan
        for _, g in ct.groupby(["context", "statistic", "group_A", "group_B"],
                               sort=False):
            ct.loc[g.index, "q_diff"] = fdr_bh(g["p_diff"].to_numpy())
            ct.loc[g.index, "q_scaled"] = fdr_bh(g["p_scaled"].to_numpy())
        if cell_cov is not None:
            adj = [adjusted_contrast(percell, cell_cov, ctx_cov, group_of,
                                     pr_, batch_cols) for pr_ in cpairs]
            adj = [x for x in adj if not x.empty]
            if adj:
                ct = ct.merge(pd.concat(adj, ignore_index=True),
                              on=["context", "system", "anchor", "partner",
                                  "band", "group_A", "group_B", "statistic"],
                              how="left")
        ct.to_csv(out / "contrast_classes.tsv", sep="\t", index=False)
        for ctx_ in contexts:
            for st_ in ("delta", "signed_rel"):
                for a_, b_ in cpairs:
                    plot_contrast(ct, ctx_, st_,
                                  out / f"contrast_{ctx_}_{st_}"
                                        f"{pair_tag(a_, b_)}."
                                        f"{args.plot_format}",
                                  pair=(a_, b_))
                if len(cpairs) > 1:
                    plot_contrast_overview(ct, ctx_, st_,
                                           out / f"contrast_overview_{ctx_}_"
                                                 f"{st_}.{args.plot_format}")
        top = ct[(ct["anchor"] == ct["partner"])
                 & ct["band"].astype(str).str.startswith("all")
                 & (ct["statistic"] == "signed_rel")]
        cols_ = [c for c in ("context", "system", "anchor", "value_A",
                             "value_B", "diff", "q_diff", "diff_scaled",
                             "q_scaled", "diff_adjusted", "q_adjusted")
                 if c in top]
        for (a_, b_), t_ in top.groupby(["group_A", "group_B"], sort=False):
            t_ = t_.reindex(t_["q_diff"].sort_values().index).head(12)
            logger.info(f"{a_} vs {b_}, ΔC x w, all distances (smallest "
                        f"q):\n" + t_[cols_].to_string(
                            index=False, float_format=lambda v: f"{v:.4f}"))
    if var_rows:
        lv_ = pd.concat(var_rows, ignore_index=True)
        en_ = pd.concat(var_en, ignore_index=True) if var_en else \
            pd.DataFrame()
        lead = ["context", "system", "chrom", "start", "end", "statistic",
                "set", "set_scaled"]
        lv_[lead + [c for c in lv_.columns if c not in lead + [
            "grid_index", "code"]]].to_csv(out / "locus_variability.tsv.gz",
                                           sep="\t", index=False)
        en_.to_csv(out / "locus_variability_enrichment.tsv", sep="\t",
                   index=False)
        for (ctx_, st_), g_ in lv_.groupby(["context", "statistic"],
                                           sort=False):
            # plain BEDs for marker_bin_enrichment.py: --bed NAME=set
            # --background-bed NAME=tested
            for nm, m in (("variable", g_["set"] == "variable"),
                          ("variable_beyond_type_level",
                           g_["set_scaled"] == "variable"),
                          ("stable", g_["set"] == "stable"),
                          ("tested", np.ones(len(g_), bool))):
                g_.loc[m, ["chrom", "start", "end"]].sort_values(
                    ["chrom", "start"]).to_csv(
                    out / f"locus_{nm}_{ctx_}_{st_}.bed", sep="\t",
                    header=False, index=False)
            plot_locus_variability(lv_, en_, ctx_, st_,
                                   out / f"locus_variability_{ctx_}_{st_}."
                                         f"{args.plot_format}")
    if chot_rows:
        hc_ = pd.concat(chot_rows, ignore_index=True)
        hc_.drop(columns=["grid_index", "code"]).to_csv(
            out / "contrast_hotspots.tsv.gz", sep="\t", index=False)
        en_ = (pd.concat(cenr_rows, ignore_index=True) if cenr_rows
               else pd.DataFrame())
        en_.to_csv(out / "contrast_hotspot_enrichment.tsv", sep="\t",
                   index=False)
        for ctx_ in contexts:
            for a_, b_ in cpairs:
                plot_hotspot_contrast(hc_, en_, ctx_,
                                      out / f"contrast_hotspots_{ctx_}"
                                            f"{pair_tag(a_, b_)}."
                                            f"{args.plot_format}",
                                      fdr=args.hotspot_fdr, pair=(a_, b_))
    # chrX against the autosomes
    if chrx_rows:
        cx = pd.DataFrame(chrx_rows).rename(columns={
            "pairs_A": "pairs_autosomes", "pairs_B": "pairs_X",
            "value_A": "autosomes", "lo_A": "autosomes_lo",
            "hi_A": "autosomes_hi", "value_B": "chrX", "lo_B": "chrX_lo",
            "hi_B": "chrX_hi"})
        cx["q_diff"] = np.nan
        for _, g in cx.groupby(["context", "statistic", "band"], sort=False):
            cx.loc[g.index, "q_diff"] = fdr_bh(g["p_diff"].to_numpy())
        lead = ["context", "group_kind", "group", "band", "statistic"]
        cx = cx[lead + [c for c in cx.columns if c not in lead]]
        cx.to_csv(out / "chrx_concordance.tsv", sep="\t", index=False)
        pcx = pd.concat(chrx_cells, ignore_index=True)
        pcx.to_csv(out / "chrx_percell.tsv.gz", sep="\t", index=False)
        for ctx_ in contexts:
            plot_chrx(cx, pcx, sex_tab, ctx_,
                      out / f"chrx_{ctx_}.{args.plot_format}")
    # concordance inside each set of domains: imprinted, protocadherin
    what_of = {"imprinting": "imprinted domains",
               "pcdh": "the clustered protocadherins"}
    for nm_ in dsets:
        if not dom_rows[nm_]:
            continue
        it = pd.DataFrame(dom_rows[nm_])
        it["q"] = np.nan
        for _, g in it[it["domain"] != "all domains"].groupby(
                ["context", "group"], sort=False):
            it.loc[g.index, "q"] = fdr_bh(g["p"].to_numpy())
        lead = ["context", "system", "group_kind", "group", "domain"]
        it = it[lead + [c for c in it.columns if c not in lead]]
        it.to_csv(out / f"{nm_}_concordance.tsv", sep="\t", index=False)
        if dom_wins[nm_]:
            pd.concat(dom_wins[nm_], ignore_index=True).to_csv(
                out / f"{nm_}_windows.tsv", sep="\t", index=False)
        ic = pd.DataFrame(dom_crows[nm_])
        if not ic.empty:
            ic["q"] = np.nan
            for _, g in ic.groupby(["context", "group_A", "group_B"],
                                   sort=False):
                ic.loc[g.index, "q"] = fdr_bh(g["p"].to_numpy())
            lead = ["context", "system", "group_A", "group_B", "domain"]
            ic = ic[lead + [c for c in ic.columns if c not in lead]]
            ic.to_csv(out / f"{nm_}_contrast.tsv", sep="\t", index=False)
        tnames = list(dict.fromkeys(it.loc[it["group_kind"] == "cell type",
                                           "group"]))
        show = contrast_names or (tnames if len(tnames) <= 6 else [])
        for ctx_ in contexts:
            plot_domains(it, ic, ctx_,
                         out / f"{nm_}_{ctx_}.{args.plot_format}",
                         what=what_of.get(nm_, nm_),
                         mark_neuron=nm_ == "imprinting", show_groups=show)
    # How well each region group's concordance predicts cell type.
    auc_stats = [x.strip() for x in args.celltype_auc_stats.split(",")
                 if x.strip() and x.strip() != "none"]
    bad = set(auc_stats) - set(COV_STATS)
    if bad:
        sys.exit(f"unknown --celltype-auc-stats {sorted(bad)}; choose from "
                 f"{COV_STATS}")
    if auc_stats and not percell.empty:
        aucs, pers = [], []
        for st_ in auc_stats:
            logger.info(f"cell-type AUC of {st_} per region group")
            r_, p_ = celltype_auc(percell, st_, args.celltype_auc_folds,
                                  args.seed, args.n_threads,
                                  n_trees=args.celltype_auc_trees,
                                  cov=cell_cov, ctx_cov=ctx_cov,
                                  batch_cols=batch_cols)
            if r_.empty:
                continue
            aucs.append(r_)
            pers.append(p_)
            plot_celltype_auc(r_, p_, st_, out / f"celltype_auc_{st_}."
                                                 f"{args.plot_format}")
            tab = r_.pivot_table(index=["context", "system", "group"],
                                 columns="features", values="auc").reindex(
                columns=list(AUC_VARIANTS))
            logger.info(f"cell-type AUC from {st_} (macro, one-vs-rest):\n"
                        + tab.to_string(float_format=lambda v: f"{v:.3f}"))
        if aucs:
            pd.concat(aucs).to_csv(out / "celltype_auc.tsv", sep="\t",
                                   index=False)
            pd.concat(pers).to_csv(out / "celltype_auc_by_type.tsv",
                                   sep="\t", index=False)
    logger.info(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
