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
the reshuffled draw's own F_g.  The signed figures
(fig_<context>_<system>_signed, main_figure_signed) draw signed_rel;
--signed-reference even draws ΔC x (2F - 1).

CELL-TO-CELL VARIATION is read against that control.  Per cell, Δ carries
sampling noise; the control carries the same noise and nothing else, so

    excess SD = sqrt(Var_cells(Δ) - Var_cells(control))

is the between-cell spread of real concordance.  Split-half reliability
(each cell's contacts split in two at random) checks it independently, and
an F test across cell types (for Δ and for the control) says how much of it
is cell type.  The marginal gets the same summary -- mean, between-cell
spread beyond its sampling noise, cell-type F -- as the reference.

UNITS.  C is a difference of probabilities (excess agreement), so the
figures show C, ΔC and the signed values in percentage points (x100); the
tables keep fractions.  |C| <= 0.5 (at most 2 L (1 - L) at level L), the
weight w lies in [-1, 1], so a signed value lies within ±|ΔC|; Δr is a
correlation, in [-1, 1], and is not rescaled.

Outputs (--output-dir): percell.tsv.gz (one row per cell x system x class
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
    """Non-overlapping intervals per chromosome, each with a class code;
    ``lookup`` gives a position's code, -1 outside every interval."""

    def __init__(self, name, classes, chrom, start, end, code, outside=-1):
        self.name = name
        self.classes = tuple(classes)
        self.outside = int(outside)    # code off the intervals (-1: none)
        df = pd.DataFrame({"chrom": [_canon(c) for c in chrom],
                           "start": np.asarray(start, np.int64),
                           "end": np.asarray(end, np.int64),
                           "code": np.asarray(code, np.int8)})
        df = df[df["code"] >= 0]
        self._by = {c: (g["start"].to_numpy(), g["end"].to_numpy(),
                        g["code"].to_numpy())
                    for c, g in df.sort_values(["chrom", "start"])
                    .groupby("chrom", sort=False)}

    def lookup(self, chrom, pos) -> np.ndarray:
        pos = np.asarray(pos, np.int64)
        out = np.full(pos.size, -1, np.int8)
        codes, uniq = pd.factorize(np.asarray(chrom))
        for j, c in enumerate(uniq):
            t = self._by.get(_canon(c))
            if t is None:
                continue
            sel = np.flatnonzero(codes == j)
            if self.outside >= 0:
                out[sel] = self.outside
            s, e, k = t
            i = np.searchsorted(s, pos[sel], side="right") - 1
            ok = i >= 0
            ok[ok] = pos[sel][ok] < e[i[ok]]
            out[sel[ok]] = k[i[ok]]
        return out


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


def _track_extents(path) -> dict:
    """{chrom: length} covered by a bigWig (its header) or bedGraph."""
    p = str(path).lower()
    if p.endswith((".bw", ".bigwig")):
        import pyBigWig
        bw = pyBigWig.open(str(path))
        out = dict(bw.chroms())
        bw.close()
        return out
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     compression="infer", dtype={0: str})
    return df.groupby(0)[2].max().astype(int).to_dict()


def rt_log2(early, late, windows: pd.DataFrame, enr) -> np.ndarray:
    """log2(early / late) per window, each side the mean of its tracks."""
    e = np.nanmean([enr._signal_bin_means(windows, t) for t in early], 0)
    lt = np.nanmean([enr._signal_bin_means(windows, t) for t in late], 0)
    pos = np.concatenate([e[np.isfinite(e) & (e > 0)],
                          lt[np.isfinite(lt) & (lt > 0)]])
    eps = 1e-3 * float(np.median(pos)) if pos.size else 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log2((e + eps) / (lt + eps))
    r[~(np.isfinite(e) & np.isfinite(lt)) | ((e <= 0) & (lt <= 0))] = np.nan
    return r


def _windows(extents: dict, size: int, exclude=()) -> pd.DataFrame:
    rows = [(c, s, min(s + size, n)) for c, n in extents.items()
            if _canon(c) not in exclude for s in range(0, int(n), size)]
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def read_rt(early, late, window: int, mid: float, enr,
            exclude=()) -> Intervals:
    """Early / late (and, with ``mid`` > 0, mid: |log2 E/L| <= mid) per
    window.  In post-mitotic neurons this is a chromatin-state proxy (early
    ~ A), not replication."""
    w = _windows(_track_extents(early[0]), window, exclude)
    r = rt_log2(early, late, w, enr)
    code = np.full(len(w), -1)
    code[r > mid] = 0
    code[r < -mid] = 1
    classes = ("early", "late")
    if mid > 0:
        code[np.isfinite(r) & (np.abs(r) <= mid)] = 2
        classes = ("early", "late", "mid")
    logger.info(f"RT: {len(w):,} windows of {window:,} bp; "
                + ", ".join(f"{c} {np.mean(code == i):.1%}"
                            for i, c in enumerate(classes)))
    return Intervals("rt", classes, w["chrom"], w["start"], w["end"], code)


# ------------------------------------------------------- A/B per cell type --


def _contact_bins(path, min_d: int, res: int, exclude) -> tuple:
    """One cell's read pairs >= min_d as (chromosome names, per-pair index
    into them, left bin, right bin) at ``res``."""
    df = pd.read_parquet(path, columns=["chrom", "pos1", "pos2"])
    lo = np.minimum(df["pos1"].to_numpy(np.int64), df["pos2"].to_numpy(np.int64))
    hi = np.maximum(df["pos1"].to_numpy(np.int64), df["pos2"].to_numpy(np.int64))
    idx, uniq = pd.factorize(df["chrom"].astype(str).to_numpy())
    uniq = np.array([_canon(c) for c in uniq])
    keep = ((hi - lo) >= min_d) & ~np.isin(uniq, list(exclude))[idx]
    return (uniq, idx[keep].astype(np.int16), (lo[keep] // res).astype(np.int32),
            (hi[keep] // res).astype(np.int32))


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

    ``by_cell[i]``: :func:`_contact_bins` of cell i.  Per type and
    chromosome: the bin x bin count matrix and its compartment track
    (:func:`compartment_track`: the eigenvector of the O/E correlation that
    best follows ``orient(chrom, n_bins)``, an A-like reference per bin,
    signed to correlate positively with it).  A type with fewer than
    ``min_contacts`` contacts takes the all-cell call.  Returns ({type:
    Intervals}, the per-bin table, QC)."""
    chroms = sorted({c for u, *_ in by_cell for c in u})
    nb = {c: 0 for c in chroms}
    for u, idx, b1, b2 in by_cell:
        for j, c in enumerate(u):
            s = idx == j
            if s.any():
                nb[c] = max(nb[c], int(b2[s].max()) + 1)
    types = sorted(set(cell_type))
    counts = {t: {c: np.zeros(nb[c] * nb[c]) for c in chroms}
              for t in types + ["__all__"]}
    n_type = dict.fromkeys(types + ["__all__"], 0)
    for (u, idx, b1, b2), t in zip(by_cell, cell_type):
        for j, c in enumerate(u):
            s = idx == j
            if not s.any():
                continue
            flat = np.bincount(b1[s].astype(np.int64) * nb[c] + b2[s],
                               minlength=nb[c] * nb[c])
            counts[t][c] += flat
            counts["__all__"][c] += flat
            n_type[t] += int(s.sum())
            n_type["__all__"] += int(s.sum())
    scores, refs, picked = {}, {}, {}
    for t in types + ["__all__"]:
        scores[t], picked[t] = {}, []
        for c in chroms:
            U = counts[t][c].reshape(nb[c], nb[c])
            M = U + U.T - np.diag(np.diag(U))
            if c not in refs:
                refs[c] = np.asarray(orient(c, nb[c]), float)
            scores[t][c], which, r = compartment_track(M, refs[c])
            picked[t].append(f"{c}:PC{which}")
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
            rows.append(pd.DataFrame({"cell_type": t, "chrom": c,
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
    """One cell's contacts -- both ends called, min_d <= d <= max_d -- with
    every system's class at both ends, and per system x class the count,
    sum and sum of squares of ALL its called ends' posteriors (the
    marginal: every read pair's ends, short ones included)."""
    df = pd.read_parquet(path, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                        "posterior1", "posterior2"])
    idx, uniq = pd.factorize(df["chrom"].astype(str).to_numpy())
    uniq = np.array([_canon(c) for c in uniq])
    keep = ~np.isin(uniq, list(exclude))[idx]
    chrom = uniq[idx[keep]]
    pos1 = df["pos1"].to_numpy(np.int64)[keep]
    pos2 = df["pos2"].to_numpy(np.int64)[keep]
    p1 = df["posterior1"].to_numpy(np.float64, na_value=np.nan)[keep]
    p2 = df["posterior2"].to_numpy(np.float64, na_value=np.nan)[keep]
    inf1 = (df["n1"].to_numpy()[keep] >= min_sites) & np.isfinite(p1)
    inf2 = (df["n2"].to_numpy()[keep] >= min_sites) & np.isfinite(p2)
    e_chr = np.concatenate([chrom[inf1], chrom[inf2]])
    e_pos = np.concatenate([pos1[inf1], pos2[inf2]])
    e_p = np.concatenate([p1[inf1], p2[inf2]])
    sep = np.abs(pos2 - pos1)
    cm = inf1 & inf2 & (sep >= min_d) & (sep <= max_d)
    left = pos1 <= pos2
    lo = np.where(left, pos1, pos2)[cm]
    hi = np.where(left, pos2, pos1)[cm]
    out = {"chrom": chrom[cm], "lo": lo, "hi": hi,
           "a": np.where(left, p1, p2)[cm].astype(np.float32),
           "p": np.where(left, p2, p1)[cm].astype(np.float32),
           "cls": {}, "marg": {}}
    for name, lk in systems.items():
        k = lk.lookup(e_chr, e_pos)
        ok = k >= 0
        n_k = len(lk.classes)
        out["marg"][name] = np.stack([
            np.bincount(k[ok], minlength=n_k).astype(float),
            np.bincount(k[ok], weights=e_p[ok], minlength=n_k),
            np.bincount(k[ok], weights=e_p[ok] ** 2, minlength=n_k)])
        out["cls"][name] = (lk.lookup(out["chrom"], lo),
                            lk.lookup(out["chrom"], hi))
    return out


# --------------------------------------------------------------- the null --


def _pool(g, v, cell):
    o = np.argsort(g, kind="mergesort")
    v = np.asarray(v, float)[o]
    return (g[o], v, np.asarray(cell, np.int64)[o],
            np.concatenate([[0.0], np.cumsum(v)]))


def reshuffle(g_lo, g_hi, d, cell, x_cls, y_cls, a_c, p_c, stratum, n_cls,
              dbins, position_bin, max_widen, seed):
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
    reused = 0
    for s in np.unique(stratum):
        in_s = stratum == s
        for k in range(n_cls):
            lo_k = in_s & (x_cls == k) & np.isfinite(a_c)
            hi_k = in_s & (y_cls == k) & np.isfinite(p_c)
            if not (lo_k.any() or hi_k.any()):
                continue
            pg = np.concatenate([g_lo[lo_k], g_hi[hi_k]])
            pv = np.concatenate([a_c[lo_k], p_c[hi_k]])
            pc = np.concatenate([cell[lo_k], cell[hi_k]])
            key = zlib.crc32(str(s).encode())
            for side, ask in (("right", np.flatnonzero(usable & in_s
                                                       & (y_cls == k))),
                              ("left", np.flatnonzero(usable & in_s
                                                      & (x_cls == k)))):
                if ask.size == 0:
                    continue
                rng = np.random.default_rng([seed, key, k,
                                             0 if side == "right" else 1])
                if side == "right":
                    pool = _pool(pg, pv, pc)
                    start = (g_hi[ask] // pb) * pb
                    lv, _, dp, ru = _ls.assign_stratum(
                        g_lo[ask], d[ask], cell[ask], pool, pool, dbins, rng,
                        n_draws=2, bounds=(start, start + pb),
                        max_widen=max_widen)
                    lv_r[ask], right[:, ask] = lv, dp
                else:
                    pool = _pool(M - pg, pv, pc)
                    b0 = (g_lo[ask] // pb) * pb
                    lv, _, dp, ru = _ls.assign_stratum(
                        M - g_hi[ask], d[ask], cell[ask], pool, pool, dbins,
                        rng, n_draws=2, bounds=(M - b0 - pb + 1, M - b0 + 1),
                        max_widen=max_widen)
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
}
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
          "scale": 1.0, "xlab_f": "excess between-cell SD"},
    "signed": {"point": "signed", "excess": "signed_excess_sd",
               "pc": "signed", "pc_ctrl": "signed_control",
               "ylab": "ΔC×(2F−1), pp",
               "xlab": "ΔC × (2F − 1), pp, all distances",
               "scale": PP, "xlab_f": "excess between-cell SD, pp"},
    "signed_rel": {"point": "signed_rel", "excess": "signed_rel_excess_sd",
                   "pc": "signed_rel", "pc_ctrl": "signed_rel_control",
                   "ylab": "ΔC×w, pp",
                   "xlab": "ΔC × w, pp, all distances",
                   "scale": PP, "xlab_f": "excess between-cell SD, pp"},
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
        ax_g.set_xlabel("share of between-cell variance")
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
    ap.add_argument("--signed-reference", choices=["genome", "even"],
                    default="genome",
                    help="What the signed figures read the co-open share F "
                         "against.  'genome' (default): the cell's own "
                         "genome-wide F -- a class pair is + when its "
                         "concordant contacts are more co-open than the "
                         "cell's contacts overall.  'even': 1:1, i.e. "
                         "ΔC x (2F - 1); for GCH, closed over most of the "
                         "genome, nearly every class then reads co-closed.  "
                         "summary.tsv carries both (signed, signed_rel).")
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
                                    args.rt_window, args.rt_mid, enr, exclude)
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
    if "ab" in want:
        orient = args.ab_orient
        if orient == "auto":
            orient = "rt" if args.rt_early and args.rt_late else "genes"
        res = args.ab_resolution
        if orient == "rt":
            if not (args.rt_early and args.rt_late):
                sys.exit("--ab-orient rt needs --rt-early and --rt-late")

            def ref(c, nb):
                w = pd.DataFrame({"chrom": "chr" + c,
                                  "start": np.arange(nb) * res,
                                  "end": (np.arange(nb) + 1) * res})
                return rt_log2(args.rt_early, args.rt_late, w, enr)
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

            def ref(c, nb):
                return np.bincount(np.minimum(tss[gch == c] // res, nb - 1),
                                   minlength=nb).astype(float)
        logger.info(f"A/B: pooling contacts per cell type at {res:,} bp, "
                    f"oriented by {orient}")
        by_cell = Parallel(n_jobs=args.n_threads, prefer="threads")(
            delayed(_contact_bins)(Path(first) / f"{c}.posteriors.parquet",
                                   args.min_distance, res, exclude)
            for c in cells_all)
        ctype = [str(samples.at[c, args.group_column]) for c in cells_all]
        ab_maps, ab_tab, ab_qc = compartments_from_contacts(
            by_cell, ctype, res, ref, args.ab_min_contacts)
        del by_cell
        ab_tab.to_csv(out / "compartments_ab.tsv", sep="\t", index=False)
        ab_qc.to_csv(out / "compartments_qc.tsv", sep="\t", index=False)
        logger.info("A/B per cell type:\n" + ab_qc.to_string(index=False))

    percell_rows, summary_rows, marg_rows, cov_rows = [], [], [], []
    for ci_ctx, (ctx, d) in enumerate(contexts.items()):
        cells = [c for c in cells_all
                 if (Path(d) / f"{c}.posteriors.parquet").exists()]
        types = np.array([str(samples.at[c, args.group_column])
                          for c in cells])
        stratum = np.array(["|".join(str(samples.at[c, k]) for k in strat_cols)
                            for c in cells])
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
        chrom_code = {c: i for i, c in enumerate(sorted(
            {x for r in res_ for x in np.unique(r["chrom"])}))}
        cell = np.concatenate([np.full(r["lo"].size, i, np.int64)
                               for i, r in enumerate(res_)])
        code = np.concatenate([pd.Series(r["chrom"], dtype=object)
                               .map(chrom_code).to_numpy(np.int64)
                               if r["lo"].size else np.empty(0, np.int64)
                               for r in res_])
        lo = np.concatenate([r["lo"] for r in res_])
        hi = np.concatenate([r["hi"] for r in res_])
        a = np.concatenate([r["a"] for r in res_]).astype(float)
        p = np.concatenate([r["p"] for r in res_]).astype(float)
        dist = hi - lo
        g_lo = code * _ls.GENOME_STRIDE + lo
        g_hi = code * _ls.GENOME_STRIDE + hi
        band = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0,
                       len(edges) - 2)
        half = np.random.default_rng([args.seed, ci_ctx]).integers(
            0, 2, lo.size)
        logger.info(f"{ctx}: {lo.size:,} contacts in {len(cells):,} cells")
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
                g_lo, g_hi, dist, cell, kx, ky, a_c, p_c, stratum[cell], n_k,
                dbins, args.position_bin, args.max_widen,
                seed=args.seed + 7919 * ci_ctx + si)
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
                             "contacts": int(lo.size),
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
        sk = "signed_rel" if args.signed_reference == "genome" else "signed"
        plot(s[~s["sparse"] | s["band"].str.startswith("all")], pc,
             out / f"fig_{ctx}_{name}_signed.{args.plot_format}",
             f"{ctx} — {name}: " + (
                 "ΔC × w(F against the cell's genome) by class pair"
                 if sk == "signed_rel" else "ΔC × (2F − 1) by class pair"),
             stat=sk)
    plot_main(summary, percell,
              out / f"main_figure_signed.{args.plot_format}",
              note=args.figure_note, min_band_pairs=args.min_band_pairs,
              stat="signed_rel" if args.signed_reference == "genome"
              else "signed")
    logger.info(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
