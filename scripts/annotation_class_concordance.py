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
property of cells.  Cross-class correlation (cross_class_correlation.tsv,
cross_class_<context>): whether a cell's ΔC in one state predicts it in
another, corrected for noise by split-half reliability -- one per-cell
factor or region-specific variation.  In post-
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
    "TSS": "promoter", "Prom": "promoter",
    "enhancers": "enhancer", "WkEnh": "enhancer", "TxEnh": "enhancer",
    "OpenC": "enhancer",
    "Tx": "transcribed", "TxEx": "transcribed", "TxWk": "transcribed",
    "ReprPC": "polycomb", "ReprPC and openC": "polycomb",
    "BivProm": "polycomb",
    "HET": "heterochromatin", "ZNF": "heterochromatin",
    "quiescent": "quiescent",
}
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


def read_annotation(name, path, max_labels: int = 12) -> Intervals:
    """Any BED as an annotation system.  With a 4th column of at most
    ``max_labels`` distinct labels, its labels are the classes (the BED's
    intervals must not overlap); otherwise two classes, inside the
    (merged) intervals and outside them -- e.g. LAD / non-LAD."""
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
    grp = seg["name"].map(mp).map(CHROMHMM_GROUPS)
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


def read_rt(early, late, window: int, mid: float, exclude=()) -> Intervals:
    """Early / late (and, with ``mid`` > 0, mid: |log2 E/L| <= mid) per
    window.  In post-mitotic neurons this is a chromatin-state proxy (early
    ~ A), not replication."""
    excl = {_chrom_code(c) for c in exclude}
    ext: dict = {}
    for pth in list(early) + list(late):
        for c, (st, en, _) in _track(pth).items():
            if c not in excl and en.size:
                ext[c] = max(ext.get(c, 0), int(en.max()))
    st = [np.arange(0, n, window, dtype=np.int64) for _, n in sorted(ext.items())]
    ch = np.concatenate([np.full(x.size, c, np.int64)
                         for (c, _), x in zip(sorted(ext.items()), st)])
    st = np.concatenate(st)
    r = rt_log2(early, late, ch, st, st + window)
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
               systems: dict) -> dict:
    """One cell's contacts -- both ends called, min_d <= d <= max_d -- as
    genome coordinates, with every system's class at both ends, and per
    system x class the count, sum and sum of squares of ALL its called
    ends' posteriors (the marginal: every read pair's ends, short ones
    included).  Integer work only, so threads run in parallel."""
    df, names, ci, lut = _categorical(
        path, ["chrom", "pos1", "pos2", "n1", "n2", "posterior1",
               "posterior2"])
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
    # How well this cell's ends are called: the share called, the sites an
    # end carries and how decisive its posterior is.
    n1_ = df["n1"].to_numpy()[keep]
    n2_ = df["n2"].to_numpy()[keep]
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

#: Per-cell covariates the within-type variation is set against, grouped.
COV_GROUPS = (
    ("technical", ("log10_contacts", "frac_short", "frac_called",
                   "sites_mean", "decisive")),
    ("level", ("level",)),
    ("cell cycle", ("frac_near", "frac_mitotic", "repli_score")),
    ("3D", ("compartment_strength",)),
)
COVARIATES = tuple(c for _, grp in COV_GROUPS for c in grp)
COV_LABEL = {"log10_contacts": "depth", "frac_short": "% short pairs",
             "frac_called": "% ends called", "sites_mean": "sites / end",
             "decisive": "call decisiveness", "level": "level",
             "frac_near": "% near", "frac_mitotic": "% mitotic",
             "repli_score": "repli-score",
             "compartment_strength": "compartment strength"}
#: Covariates measured per context (from that context's calls).
CTX_COVARIATES = ("level", "sites_mean", "decisive", "frac_called")
_CONTROL_OF = {"delta": "control", "signed_rel": "signed_rel_control",
               "signed_sgn": "signed_sgn_control",
               "signed": "signed_control"}


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


def locus_reassignment(cell, ctype, pair, band, d, c, n_band: int,
                       min_pairs: int, n_perm: int, rng) -> pd.DataFrame:
    """How much of the within-type variance of per-cell ΔC is a property of
    the CELLS, not of which loci each happened to sample?

    ``d``/``c``: each contact's contribution to its cell's ΔC and to the
    control (a cell's value is their mean).  Within each cell type, class
    pair and band, the contributions are reassigned to cells at random --
    every cell keeping its number of contacts -- ``n_perm`` times: the
    variance left is sampling noise plus locus sampling, with no cell
    differences.  Returned per (pair, band; band ``n_band`` = all), over
    the cells with >= ``min_pairs`` contacts: the within-type variance
    observed, of the control, and reassigned (mean, sd, permutation p), and
    the split  noise_share + locus_share + cell_share = 1."""
    n_cells = ctype.size
    T = int(ctype.max()) + 1 if n_cells else 1
    ct = ctype[cell]
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
        cnt_gt = np.bincount(gt, minlength=G * T)
        ncell = np.bincount(gg, minlength=G)
        ntype = np.bincount(np.unique(gt) // T, minlength=G)
        den = np.maximum(ncell - ntype, 1)

        def wvar(w):
            m = np.bincount(key, weights=w, minlength=size)[el] / N[el]
            mean = np.bincount(gt, weights=m, minlength=G * T) / np.maximum(
                cnt_gt, 1)
            return np.bincount(gg, weights=(m - mean[gt]) ** 2,
                               minlength=G) / den
        v_obs, v_ctrl = wvar(d), wvar(c)
        perm = np.full((max(int(n_perm), 0), G), np.nan)
        if n_perm > 0:
            blk = g * T + ct
            o = np.argsort(blk, kind="stable")
            bs = blk[o]
            st = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
            en = np.r_[st[1:], bs.size]
            ds = d[o]
            for b_ in range(int(n_perm)):
                dp = ds.copy()
                for a_, e_ in zip(st, en):
                    if e_ - a_ > 1:
                        dp[a_:e_] = dp[a_:e_][rng.permutation(e_ - a_)]
                back = np.empty_like(d)
                back[o] = dp
                perm[b_] = wvar(back)
        for i in range(G):
            if ncell[i] < 3:
                continue
            vo, vc = float(v_obs[i]), float(v_ctrl[i])
            vp = float(np.nanmean(perm[:, i])) if n_perm > 0 else np.nan
            row = {"pair": int(g_ids[i] // (n_band + 1)),
                   "band_index": int(g_ids[i] % (n_band + 1)),
                   "cells": int(ncell[i]), "var_obs": vo, "var_ctrl": vc,
                   "var_perm": vp,
                   "var_perm_sd": (float(np.nanstd(perm[:, i], ddof=1))
                                   if n_perm > 1 else np.nan),
                   "perm_p": ((1 + int((perm[:, i] >= vo).sum()))
                              / (n_perm + 1) if n_perm > 0 else np.nan)}
            if vo > 0 and np.isfinite(vp):
                row["noise_share"] = min(vc / vo, 1.0)
                row["locus_share"] = float(np.clip((vp - vc) / vo, 0,
                                                   1 - row["noise_share"]))
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
                          batch_cols=(), stats=("delta", "signed_rel",
                                                "signed_sgn"),
                          min_cells: int = 30) -> pd.DataFrame:
    """What explains the WITHIN-cell-type variation of a class pair's
    per-cell concordance?

    Per context x system x class pair x distance band x statistic, over the
    cells with enough contacts:

    - each covariate's within-type Spearman rho (ranks with each type's
      mean rank removed; p from t on n - types - 1 df; BH q over the table);
    - all numeric covariates together, batch alone and both: the adjusted
      R^2 of a no-intercept OLS on type-centred values (R2_adj, R2_adj_batch,
      R2_adj_all), and per batch column an F test (batch_<col>_p);
    - the same full fit on the control, which has nothing to explain
      (control_R2_adj: well above 0 means a covariate tracks noise).

    R^2 is of the TOTAL within-type variance.  bio_share = 1 -
    Var_w(control) / Var_w(value) is the part beyond sampling noise, and
    explained_share = R2_adj_all / bio_share the share of it the
    covariates (with batch) explain.  For ΔC, ``locus``
    (:func:`locus_reassignment`) also takes out what merely follows which
    loci a cell sampled: cell_share is the part that is a property of the
    cells, and explained_cell_share = R2_adj_all / cell_share.

    Read the bands as well as 'all'.  ΔC changes with distance -- GCH's
    excess rises ~10x from 1 to 100 kb -- so a cell's all-distance value
    follows its own mix of contact distances, and % near and % mitotic ARE
    that mix: they can correlate with it with no cell cycle involved.
    Within one decade band the mix barely varies; a covariate that holds
    there is not the mix."""
    from scipy.stats import f as f_dist
    from scipy.stats import t as t_dist
    pc = percell[percell["enough_pairs"]]
    lk = None
    if locus is not None and not locus.empty:
        lk = locus.set_index(["context", "system", "anchor", "partner",
                              "band"])
    rows = []
    for (ctx, sy, k1, k2, bl), g in pc.groupby(
            ["context", "system", "anchor", "partner", "band"], sort=False):
        d = g.merge(cov.drop(columns=["cell_type"], errors="ignore"),
                    on="cell", how="left")
        cx = ctx_cov.get(ctx)
        if cx is not None:
            d = d.merge(cx, on="cell", how="left")
        avail = [c for c in COVARIATES if c in d
                 and d[c].notna().sum() >= min_cells and d[c].nunique() > 1]
        X0 = d[avail].to_numpy(float) if avail else np.empty((len(d), 0))
        bcols = [b for b in batch_cols if b in d and d[b].nunique() > 1]
        for stat in stats:
            if stat not in d:
                continue
            y0 = d[stat].to_numpy(float)
            c0 = d[_CONTROL_OF[stat]].to_numpy(float)
            ok = np.isfinite(y0) & np.isfinite(c0) & np.isfinite(X0).all(1)
            if ok.sum() < min_cells:
                continue
            t = d["cell_type"].to_numpy()[ok]
            n, n_t = int(ok.sum()), len(set(t))
            y, c = _center(y0[ok], t), _center(c0[ok], t)
            X = (np.column_stack([_center(X0[ok, j], t)
                                  for j in range(X0.shape[1])])
                 if X0.shape[1] else np.empty((n, 0)))
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
            ry = _center(pd.Series(y0[ok]).rank().to_numpy(), t)
            df_ = max(n - n_t - 1, 1)
            for cname in COVARIATES:
                rho = p_ = np.nan
                if cname in avail:
                    j = avail.index(cname)
                    rx = _center(pd.Series(X0[ok, j]).rank().to_numpy(), t)
                    if rx.std() > 0 and ry.std() > 0:
                        rho = float(np.corrcoef(rx, ry)[0, 1])
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
            if stat == "delta" and lk is not None and \
                    (ctx, sy, k1, k2, bl) in lk.index:
                L = lk.loc[(ctx, sy, k1, k2, bl)]
                for col in ("noise_share", "locus_share", "cell_share",
                            "perm_p", "var_obs", "var_perm"):
                    row[col] = float(L[col]) if col in L else np.nan
                cs = row.get("cell_share", np.nan)
                if np.isfinite(r2a) and np.isfinite(cs) and cs > 0:
                    row["explained_cell_share"] = float(
                        np.clip(max(r2a, 0) / cs, 0, 1))
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        pcols = [f"p_{c}" for c in COVARIATES]
        q = fdr_bh(out[pcols].to_numpy(float).ravel())
        for j, c in enumerate(COVARIATES):
            out[f"q_{c}"] = q.reshape(len(out), len(COVARIATES))[:, j]
    return out


def cross_class_correlation(percell: pd.DataFrame, summary: pd.DataFrame,
                            min_cells: int = 30) -> pd.DataFrame:
    """Is the within-type variation one per-cell factor or region-specific?

    Per context and system, between DISJOINT same-state class pairs (A-A
    against B-B, early-early against late-late, the chromHMM states): the
    correlation across cells of their per-cell ΔC (all distances, each
    type's mean removed), and the same corrected for sampling noise with
    each pair's split-half reliability (Spearman-Brown, 2r / (1 + r)):
    r_disattenuated = r / sqrt(rel_1 rel_2).  Near 1: the classes move
    together -- one per-cell factor, technical or a nuclear state.  Near 0:
    each class varies on its own.  Classes of different systems share
    contacts, so they are not compared."""
    all_b = next((b for b in dict.fromkeys(percell["band"])
                  if str(b).startswith("all")), None)
    pc = percell[(percell["band"] == all_b) & percell["enough_pairs"]
                 & (percell["anchor"] == percell["partner"])]
    rel = {}
    s_all = summary[summary["band"] == all_b]
    for _, r in s_all[s_all["anchor"] == s_all["partner"]].iterrows():
        h = r.get("split_half_r", np.nan)
        rel[(r["context"], r["system"], r["anchor"])] = (
            2 * h / (1 + h) if np.isfinite(h) and h > 0 else np.nan)
    rows = []
    for (ctx, sy), g in pc.groupby(["context", "system"], sort=False):
        W = g.pivot_table(index="cell", columns="anchor", values="delta")
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
                rows.append({"context": ctx, "system": sy, "state_1": a,
                             "state_2": b, "cells": int(both.sum()), "r": r_,
                             "reliability_1": ra, "reliability_2": rb,
                             "r_disattenuated": (r_ / np.sqrt(ra * rb)
                                                 if ra > 0 and rb > 0
                                                 else np.nan)})
    return pd.DataFrame(rows)


def plot_cross_class(cc: pd.DataFrame, context: str, path):
    """Per system, the noise-corrected correlation across cells between its
    same-state class pairs' per-cell ΔC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance import figure_style
    a = cc[cc["context"] == context] if not cc.empty else cc
    if a.empty:
        return
    systems = [x for x in SYSTEMS if x in set(a["system"])] + [
        x for x in dict.fromkeys(a["system"]) if x not in SYSTEMS]
    states = {sy: list(dict.fromkeys(
        list(a.loc[a["system"] == sy, "state_1"])
        + list(a.loc[a["system"] == sy, "state_2"]))) for sy in systems}
    n_max = max(len(st) for st in states.values())
    with plt.rc_context(figure_style.RC):
        # panel widths follow the number of states, so that every cell has
        # the same size
        fig, axes = plt.subplots(
            1, len(systems), squeeze=False,
            figsize=(figure_style.FIG_W_IN, 1.2 + 0.3 * n_max),
            gridspec_kw={"wspace": 0.6, "width_ratios":
                         [max(len(states[sy]), 2) for sy in systems]})
        for ax, sy in zip(axes[0], systems):
            b = a[a["system"] == sy]
            st = states[sy]
            M = np.full((len(st), len(st)), np.nan)
            np.fill_diagonal(M, 1.0)
            for _, r in b.iterrows():
                i, j = st.index(r["state_1"]), st.index(r["state_2"])
                M[i, j] = M[j, i] = r["r_disattenuated"]
            im = ax.imshow(np.clip(M, -1, 1), cmap="RdBu_r", vmin=-1, vmax=1)
            for i in range(len(st)):
                for j in range(len(st)):
                    if i != j and np.isfinite(M[i, j]):
                        # noise correction can overshoot 1 when the
                        # reliabilities are themselves noisy
                        ax.text(j, i, ">1" if M[i, j] > 1 else
                                f"{M[i, j]:.2f}".replace("0.", ".", 1),
                                ha="center", va="center", fontsize=4.5)
            ax.set_xticks(range(len(st)))
            ax.set_xticklabels(st, rotation=60, ha="right", fontsize=5)
            ax.set_yticks(range(len(st)))
            ax.set_yticklabels(st, fontsize=5)
            ax.set_title(SYSTEM_LABEL.get(sy, sy), fontsize=6)
        cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.02)
        cb.ax.tick_params(labelsize=5)
        cb.set_label("r across cells, noise-corrected", fontsize=6)
        fig.suptitle(f"{context}: does a cell's ΔC move together across "
                     f"states?", fontsize=7)
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


def _same_state(assoc, context, stat, band):
    a = assoc[(assoc["context"] == context) & (assoc["statistic"] == stat)
              & (assoc["band"] == band) & (assoc["anchor"] == assoc["partner"])]
    order = [x for x in SYSTEMS if x in set(a["system"])] + [
        x for x in dict.fromkeys(a["system"]) if x not in SYSTEMS]
    return pd.concat([a[a["system"] == x] for x in order]) if order else a


def plot_covariates(assoc: pd.DataFrame, context: str, stat: str, path,
                    q_cut: float = 0.05):
    """Same-state class pairs (rows, by annotation system) x covariates, all
    distances: the within-type Spearman rho (colour; * q < q_cut), and on
    the right the share of the within-type variance that is real (light)
    and the part of it the covariates explain together (dark)."""
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
    covs = [c for c in COVARIATES if a[f"rho_{c}"].notna().any()]
    R = a[[f"rho_{c}" for c in covs]].to_numpy(float)
    Q = a[[f"q_{c}" for c in covs]].to_numpy(float)
    ny = len(a)
    lim = max(0.1, float(np.nanmax(np.abs(R))) if np.isfinite(R).any()
              else 0.1)
    with plt.rc_context(figure_style.RC):
        fig = plt.figure(figsize=(figure_style.FIG_W_IN * 0.8,
                                  1.3 + 0.2 * ny))
        gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 0.08, 1.1],
                              wspace=0.4)
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
        ax0.set_yticklabels([f"{SYSTEM_LABEL.get(sy, sy)}: {k}–{k}"
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
        expl = cell_ * np.nan_to_num(np.where(
            np.isfinite(col("explained_cell_share")),
            col("explained_cell_share"), col("explained_share")))
        left = np.zeros(ny)
        for w, c_, lab in ((noise, "0.85", "sampling noise"),
                           (locus, "#f4c58f", "which loci were sampled"),
                           (expl, "#08519c", "cells: explained by covariates"),
                           (cell_ - expl, "#9ecae1", "cells: unexplained")):
            w = np.clip(np.nan_to_num(w), 0, None)
            ax1.barh(yy, w, left=left, color=c_, label=lab, height=0.7)
            left = left + w
        ax1.set_ylim(ny - 0.5, -0.5)
        ax1.set_yticks(yy)
        ax1.set_yticklabels([])
        ax1.set_xlim(0, 1)
        ax1.set_xlabel("share of within-type var.")
        leg = ax1.legend(fontsize=5, loc="lower left",
                         bbox_to_anchor=(0, 1.0), ncol=1, borderaxespad=0.2)
        fig.canvas.draw()
        top = max(leg.get_window_extent().y1,
                  ax0.title.get_window_extent().y1)
        fig.suptitle(f"{context} — {stat}: what explains the within-type "
                     f"variation", fontsize=7, va="bottom",
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
    covs = [c for c in COVARIATES
            if any(x[f"rho_{c}"].notna().any() for _, x in blocks)]
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
        axes[0][0].set_yticklabels([f"{SYSTEM_LABEL.get(sy, sy)}: {k}–{k}"
                                    for sy, k in rows], fontsize=5)
        cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
        cb.ax.tick_params(labelsize=5)
        cb.set_label("within-type ρ", fontsize=6)
        fig.suptitle(f"{context} — {stat}: within-type ρ by distance band "
                     f"(* q < {q_cut:g})", fontsize=7)
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
    ap.add_argument("--rt-window", type=int, default=50_000)
    ap.add_argument("--rt-mid", type=float, default=0.0,
                    help="|log2 early/late| at or below this is 'mid' (0: "
                         "two classes).")
    ap.add_argument("--chromhmm", default=None,
                    help="chromHMM BED (column 4 = state).  --genome mm10 "
                         "fills in the full-stack model.")
    ap.add_argument("--chromhmm-groups", default=None)
    ap.add_argument("--pmd", default=None,
                    help="PMD/HMD file, e.g. the Zhou-lab "
                         "PMD_coordinates_mm10.bed.gz.")
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
    if "rt" in want:
        if args.rt_early and args.rt_late:
            systems["rt"] = read_rt(args.rt_early, args.rt_late,
                                    args.rt_window, args.rt_mid, exclude)
        else:
            logger.warning("rt: needs --rt-early and --rt-late; skipped")
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
        if args.pmd:
            systems["pmd"] = read_pmd(args.pmd, args.pmd_class_col,
                                      args.pmd_common_only)
        else:
            logger.warning("pmd: needs --pmd; skipped")
    for k, v in _parse_pairs(args.annotation).items():
        if k in SYSTEMS:
            sys.exit(f"--annotation {k}: that name is a built-in system")
        rv = enr._resolve(v, cache, f"annotation {k}")
        if rv:
            systems[k] = read_annotation(k, rv)
            logger.info(f"annotation {k}: classes {systems[k].classes}")

    from joblib import Parallel, delayed
    first = next(iter(contexts.values()))
    cells_all = sorted(f.name[: -len(".posteriors.parquet")]
                       for f in Path(first).glob("*.posteriors.parquet"))
    cells_all = [c for c in cells_all if c in samples.index]
    if not cells_all:
        sys.exit(f"no cells of {args.samples} under {first}")

    ab_maps = None
    # One pass over every cell's read pairs: the A/B bins (with ab) and the
    # cell-cycle covariates (unless --no-covariates).  The repli-score needs
    # the replication-timing classes even when rt is not a system.
    rt_cov = systems.get("rt")
    if rt_cov is None and args.rt_early and args.rt_late and \
            not args.no_covariates:
        rt_cov = read_rt(args.rt_early, args.rt_late, args.rt_window, 0.0,
                         exclude)
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
            orient = "rt" if args.rt_early and args.rt_late else "genes"
        res = args.ab_resolution
        if orient == "rt":
            if not (args.rt_early and args.rt_late):
                sys.exit("--ab-orient rt needs --rt-early and --rt-late")

            def ref(code, name, nb):
                st = np.arange(nb, dtype=np.int64) * res
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
                              sysd)
        logger.info(f"{ctx}: reading {len(cells):,} cells")
        res_ = Parallel(n_jobs=args.n_threads, prefer="threads")(
            delayed(one)(i, c) for i, c in enumerate(cells))
        ctx_cov[ctx] = pd.DataFrame({
            "cell": cells,
            "level": [r["level"][0] / r["level"][1] if r["level"][1]
                      else np.nan for r in res_],
            **{k: [r["quality"][k] for r in res_]
               for k in ("sites_mean", "decisive", "frac_called")}})
        type_code = pd.factorize(types)[0]
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
            if cell_cov is not None and args.n_locus_perm > 0 and idx.size:
                pr = k1s[idx] * n_k + k2s[idx]
                lr = locus_reassignment(
                    cell[idx], type_code, pr, band[idx],
                    2 * (x_ * y_ - n1), 2 * (n2 - n1), len(edges) - 1,
                    args.min_pairs, args.n_locus_perm,
                    np.random.default_rng([args.seed, ci_ctx, si, 4049]))
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
            logger.info(f"{ctx} {name}: {ok.mean():.1%} of contacts "
                        f"annotated at both ends with a reshuffled partner")
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
        locus = (pd.concat(locus_rows, ignore_index=True) if locus_rows
                 else pd.DataFrame())
        locus.to_csv(out / "locus_reassignment.tsv", sep="\t", index=False)
        assoc = covariate_association(percell, cell_cov, ctx_cov, locus,
                                      batch_cols,
                                      min_cells=args.covariate_min_cells)
        assoc.to_csv(out / "covariate_association.tsv", sep="\t",
                     index=False)
        for ctx_ in contexts:
            for st_ in ("delta", "signed_rel"):
                plot_covariates(assoc, ctx_, st_,
                                out / f"covariates_{ctx_}_{st_}."
                                      f"{args.plot_format}")
                plot_covariates_by_band(
                    assoc, ctx_, st_,
                    out / f"covariates_{ctx_}_{st_}_by_band."
                          f"{args.plot_format}")
        if not assoc.empty:
            top = assoc[(assoc["statistic"] == "delta")
                        & assoc["band"].astype(str).str.startswith("all")
                        ].sort_values("cells", ascending=False).head(10)
            cols_ = [c for c in ("context", "system", "anchor", "partner",
                                 "cells", "noise_share", "locus_share",
                                 "cell_share", "perm_p", "R2_adj_all",
                                 "explained_cell_share", "control_R2_adj")
                     if c in top]
            logger.info("within-type variation of ΔC (all distances): "
                        "noise | which loci | cells, and the share of the "
                        "cell part the covariates explain:\n"
                        + top[cols_].to_string(index=False, float_format=
                                               lambda v: f"{v:.3f}"))
        cc_ = cross_class_correlation(percell, summary,
                                      args.covariate_min_cells)
        cc_.to_csv(out / "cross_class_correlation.tsv", sep="\t",
                   index=False)
        for ctx_ in contexts:
            plot_cross_class(cc_, ctx_, out / f"cross_class_{ctx_}."
                                              f"{args.plot_format}")
    logger.info(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
