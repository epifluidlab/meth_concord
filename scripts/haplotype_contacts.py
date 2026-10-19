#!/usr/bin/env python3
"""The 3D genome of the inactive and the active X, from haplotype-tagged
contacts.

allelic_imbalance.py --haplotype-dir writes, per cell, the cis read pairs a
phased SNP puts on one copy of a chromosome ({cell}.haplotypes.parquet).  A
clonal line's cells silence the same X (GM12878: hap1, the paternal), so its
contacts on that haplotype are the inactive X's (Xi), the others the active
X's (Xa): --xi GM12878=1.  In a polyclonal line each cell's own methylation
names its inactive X: --xi IMR90=cells with --xi-cells allelic_cells.tsv --
the haplotype the more methylated at CpG islands (--xi-context).  The
autosomes' two copies, named by the same haplotypes, are the control: they
fold alike.  A line with no --xi: hap1 against hap2.

Four measures, per line, set of chromosomes and copy, each with a 95%
interval from resampling the cells, the copies' difference and, where it
applies, that difference on X less on the autosomes (haplotype_contacts.tsv):

  * P(s), the contacts' distances (haplotype_contacts_ps.tsv: per bin the
    share of a copy's contacts and, as P(s) is drawn, that share per kb of
    distance): the share 0.1-1 Mb apart (TAD scale), 1 Mb or more, and
    10 Mb or more, of those
    --min-distance or more; and per distance, log2 of one copy's P(s) over
    the other's with its interval (haplotype_contacts_ps_ratio.tsv) -- the
    inactive X, folded into two megadomains, trades contacts within TADs
    for long-range ones;
  * the megadomain boundary at DXZ4: at each position along X, the
    contacts --profile-min-distance or more apart that cross it, over those
    expected of contacts of their lengths placed at random on X outside
    PAR1 (or, with --profile-window, within that of the position)
    (haplotype_contacts_profile.tsv) -- the inactive X's contacts avoid
    crossing DXZ4; each copy's log2 there, and the rank of the copies'
    difference there among the positions --profile-exclude or more away;
  * TAD boundaries (--boundaries: positions, e.g. CTCF-anchored boundaries
    of bulk Hi-C): log2 of the share of contacts crossing a position
    --boundary-flank or more from a boundary over the share crossing the
    boundary, the boundaries pooled; the autosomes' copies show that the
    data see TADs;
  * superloops between the inactive X's elements (DXZ4, FIRRE, ICCE, XIST):
    the contacts linking two of their windows (--loop-window) over those
    linking the same windows slid together along X -- a handful of contacts
    each at single-cell depth, so in the table only.

With --compartments (a bulk compartment track, positive A), a fifth: the
correlation of the track at a long-range contact's two ends -- A with A, B
with B -- per copy: the inactive X's compartments are attenuated.

Contacts with an end in PAR1 -- both copies active there -- are not X's;
those within PAR1 are a control set of P(s).  Figures:
haplotype_contacts_<line>.<fmt>.

Loci (--locus; default all of X and DXZ4 +/- 10 Mb), drawn as a genome
browser would, the two copies against each other
(haplotype_locus_<line>_<locus>.<fmt>, and their tracks in .tsv): each
copy's contact map, at the finest resolution giving the locus at most 64
bins (single-cell depth: megabases, not TADs), the bulk TADs (--tads)
between them; per copy, insulation, the compartment of its long-range
partners against the bulk track, the level per context of its phased SNPs
(--allelic-snps, allelic_imbalance.py's allelic_snps.tsv) and its ΔC
(--concordance-tracks, annotation_class_concordance.py's
chrx_haplotype_tracks.tsv) -- the last two for a line with one inactive X
(--xi GROUP=1|2), whose haplotypes are its copies.

At kinds of element (--element NAME=BED: CpG islands, promoters, CTCF
sites...), the contacts with an end in one -- on X outside PAR1 and the
escape genes, as allelic_imbalance.py reads their levels, and on the
autosomes, the control -- per copy: their P(s) shares and their share of
all the copy's contacts (haplotype_contacts.tsv, sets 'chrX NAME',
'autosomes NAME' and 'NAME: chrX − autosomes').

Is a copy's level tied to its folding?  With --genes, X's genes in two
classes, those that escape X inactivation (--chrx-escape) and the others,
which it silences: each copy's P(s) shares over the contacts with an end in
each (haplotype_contacts.tsv, sets 'chrX escape genes' and 'chrX inactivated
genes', and 'escape − inactivated genes': the one's copies' difference less
the other's) and, with --allelic-snps, each copy's level at the phased SNPs
inside (haplotype_genes_<line>.tsv) -- the escape genes, active on both
copies, should look like the active X in both.  Along X (--windows), side by
side windows: per copy its contacts' shares 0.1-1 Mb and 1 Mb or more
apart, its partners' compartment, the contacts crossing the window and its
level, and across the windows each level difference against each 3D
difference, Spearman's rho with p from sliding one series along X against
the other (haplotype_windows_<line>.tsv, _correlation_<line>.tsv).  Figure:
haplotype_genes_windows_<line>.

What single cells add.  A line whose cells silence either X (--xi
GROUP=cells) is also measured as bulk data sees it: hap1 against hap2, each
the inactive X in about half the cells, so the copies' differences cancel
-- beside each cell's own inactive X (haplotype_contacts.tsv rows with xi
'haplotypes (bulk view)'; figure haplotype_bulk_vs_cells_<line>, with each
cell's hap1 less hap2 at X's islands; the calls in
haplotype_xi_calls_<line>.tsv).  A line silencing one haplotype
(--xi GROUP=1|2) with --xi-cells: its cells grouped by their own XCI
strength (Xi less Xa at --xi-context) into --xci-strata quantile strata --
per stratum the copies' measures, per measure the slope on the strata's
strength with a permutation p, and how much the cells' strengths spread
beyond their sites' noise (haplotype_xci_strata_, _trend_ and
_cells_<line>.tsv; figure haplotype_xci_strata_<line>): does a cell's
inactive X fold the more like one the more firmly it is methylated?

    python scripts/haplotype_contacts.py \\
        --haplotypes output/haplotypes --samples samples.tsv \\
        --group-column cell_type --xi GM12878=1 \\
        --xi IMR90=cells --xi-cells allelic_imbalance/allelic_cells.tsv \\
        --boundaries gm12878_tad_boundaries_ctcf.hg38.bed \\
        --tads gm12878_tad_boundaries.hg38.bed \\
        --compartments gm12878_compartments.hg38.bedGraph \\
        --allelic-snps allelic_imbalance/allelic_snps.tsv \\
        --concordance-tracks concordance/chrx_haplotype_tracks.tsv \\
        --output-dir output/haplotype_contacts
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance.utils import chrom_key, manifest_text  # noqa: E402
from nome_hic_concordance.xci import (  # noqa: E402
    GENE_CLASSES, cell_xi_table, gene_classes, in_intervals, parse_elements,
    parse_escape, parse_par, read_genes)

logger = logging.getLogger("haplotype_contacts")

#: P(s) bins, log10 of the distance: from 1 kb, a tenth of a decade each.
PS_EDGES = np.round(np.arange(3.0, 8.55, 0.1), 2)
#: P(s) is summarised by the share of contacts this far apart: (name,
#: from, to -- None: no end).
PS_CLASSES = (("0.1–1 Mb", 100_000, 1_000_000), ("≥ 1 Mb", 1_000_000, None),
              ("≥ 10 Mb", 10_000_000, None))
#: The sets of contacts: X outside PAR1, X within PAR1, the autosomes.
SETS = ("chrX", "chrX PAR1", "autosomes")
#: --element: where its contacts are counted, and the measure of how many
#: of a copy's contacts have an end in it.
ELEMENT_SETS = ("chrX", "autosomes")
ELEMENT_SHARE = "contacts with an end in it: share"
#: The boundary profile's offsets from the boundaries.
BOUNDARY_OFFSETS = np.arange(-1_000_000, 1_000_001, 100_000)
#: Without --profile-window, the crossing profile keeps this far from
#: either end of X (outside PAR1), where few contacts cross.
PROFILE_MARGIN = 5_000_000
#: Contacts this far apart or more show which compartments they join.
COMPARTMENT_MIN_DISTANCE = 2_000_000
#: The escape genes' copies' difference less the inactivated genes'.
GENE_CONTRAST = "escape − inactivated genes"


class Eigen:
    """A compartment track (bedGraph: chrom, start, end, value; positive A):
    its value at any position, NaN where it has none."""

    def __init__(self, path):
        t = pd.read_csv(path, sep="\t", header=None, comment="#",
                        usecols=[0, 1, 2, 3], dtype={0: str})
        t = t[~t[0].str.startswith(("track", "browser"))]
        self.by = {}
        for c, g in t.groupby(0, sort=False):
            g = g.sort_values(1)
            self.by[chrom_key(c)] = (g[1].to_numpy(np.int64),
                                     g[2].to_numpy(np.int64),
                                     g[3].to_numpy(float))

    def at(self, chrom, pos) -> np.ndarray:
        pos = np.asarray(pos, np.int64)
        out = np.full(pos.shape, np.nan)
        key = pd.Series(np.asarray(chrom)).map(chrom_key).to_numpy()
        for c in np.unique(key):
            if c not in self.by:
                continue
            st, en, v = self.by[c]
            m = key == c
            i = np.searchsorted(st, pos[m], "right") - 1
            ok = (i >= 0) & (pos[m] < en[np.maximum(i, 0)])
            out[np.flatnonzero(m)[ok]] = v[i[ok]]
        return out


class Geometry:
    """Where the measures look: the profile's positions along X, the
    boundaries and their offsets, the superloops' windows and their
    slides."""

    def __init__(self, x_len: int, par, elements: dict, boundaries: dict,
                 args, eigen: Eigen | None = None):
        self.par = par
        self.eig = eigen
        self.min_d = int(args.min_distance)
        self.x_lo = int(par[1]) if par else 0
        self.x_hi = int(x_len)
        self.w = int(args.profile_window)          # 0: the whole of X
        self.prof_min = int(args.profile_min_distance)
        self.prof_excl = int(args.profile_exclude)
        step = int(args.profile_step)
        # no window runs off the chromosome
        margin = self.w if self.w > 0 else PROFILE_MARGIN
        self.grid = np.arange(self.x_lo + margin, x_len - margin + 1, step,
                              dtype=np.int64)
        self.dxz4 = None
        if "DXZ4" in elements and self.grid.size:
            a_, b_ = elements["DXZ4"]
            self.dxz4 = int(np.argmin(np.abs(self.grid - (a_ + b_) // 2)))
        self.bw = int(args.boundary_window)
        self.b_min = int(args.boundary_min_distance)
        self.flank = int(args.boundary_flank)
        # the boundaries per set: X outside PAR1, the autosomes
        self.bnd = {"chrX": {}, "autosomes": {}}
        for c, p in (boundaries or {}).items():
            k = chrom_key(c)
            if k == "X":
                p = p[p >= self.x_lo] if par else p
                self.bnd["chrX"][c] = np.sort(p)
            elif k not in ("Y", "M"):
                self.bnd["autosomes"][c] = np.sort(p)
        # superloops: pairs of elements far enough apart, the shifts of
        # their windows along X
        self.r = int(args.loop_window)
        lstep = int(args.loop_step)
        mids = sorted(((a_ + b_) // 2, nm) for nm, (a_, b_) in
                      elements.items())
        self.loops = []
        for i, (a_, na) in enumerate(mids):
            for b_, nb in mids[i + 1:]:
                if b_ - a_ < 2 * self.r + 1_000_000:
                    continue
                t_lo = -((a_ - self.r - self.x_lo) // lstep) * lstep
                t_hi = ((x_len - b_ - self.r) // lstep) * lstep
                shifts = np.arange(t_lo, t_hi + 1, lstep, dtype=np.int64)
                if 0 in shifts:
                    self.loops.append((f"{na}–{nb}", a_, b_, shifts))
        self.loop_excl = int(args.loop_exclude)
        # X's genes by GENE_CLASSES (--genes): {class: (start, end) rows}
        self.gene_iv: dict = {}
        # --element: {name: {chromosome key: (start, end) rows}}; the escape
        # genes, which the elements' X leaves out
        self.elem_iv: dict = {}
        self.escape_iv = None


def read_elements(specs, flank: int) -> dict:
    """--element NAME=BED: per name and chromosome (chrom_key) the BED's
    intervals, each ``flank`` bp wider each side, sorted and merged."""
    out = {}
    for spec in specs or []:
        name, _, path = str(spec).partition("=")
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise SystemExit(f"--element {spec}: NAME=BED")
        b = pd.read_csv(path, sep="\t", header=None, comment="#",
                        usecols=[0, 1, 2], dtype={0: str})
        b = b[~b[0].str.startswith(("track", "browser"))]
        per = {}
        for c, g in b.groupby(b[0].map(chrom_key), sort=False):
            iv = np.column_stack([np.maximum(g[1].to_numpy(np.int64) - flank,
                                             0),
                                  g[2].to_numpy(np.int64) + flank])
            iv = iv[np.argsort(iv[:, 0], kind="stable")]
            new = np.r_[True, iv[1:, 0] > np.maximum.accumulate(iv[:, 1])[:-1]]
            per[c] = np.column_stack(
                [iv[new, 0], np.maximum.reduceat(iv[:, 1],
                                                 np.flatnonzero(new))])
        out[name] = per
    return out


def read_boundaries(path) -> dict:
    """A BED's boundary positions per chromosome: each interval's
    midpoint."""
    b = pd.read_csv(path, sep="\t", header=None, comment="#",
                    usecols=[0, 1, 2], dtype={0: str})
    b = b[~b[0].astype(str).str.startswith(("track", "browser"))]
    mid = (b[1].astype(np.int64) + b[2].astype(np.int64)) // 2
    return {c: np.unique(m.to_numpy(np.int64))
            for c, m in mid.groupby(b[0], sort=False)}


def read_pairs(path) -> pd.DataFrame | None:
    """A cell's read pairs on one haplotype: chrom, the two ends (lo <
    hi) and the haplotype; a pair of ends given both haplotypes, left out."""
    if not Path(path).exists():
        return None
    t = pd.read_parquet(path, columns=["chrom", "pos1", "pos2", "hap"])
    p1 = t["pos1"].to_numpy(np.int64)
    p2 = t["pos2"].to_numpy(np.int64)
    t = pd.DataFrame({"chrom": t["chrom"].astype(str).to_numpy(),
                      "lo": np.minimum(p1, p2), "hi": np.maximum(p1, p2),
                      "hap": t["hap"].to_numpy(np.int8)})
    g = t.groupby(["chrom", "lo", "hi"], sort=False)["hap"].agg(
        ["min", "max"]).reset_index()
    g = g[g["min"] == g["max"]]
    return pd.DataFrame({"chrom": g["chrom"], "lo": g["lo"], "hi": g["hi"],
                         "hap": g["min"].astype(np.int8)})


def _ranges(n: int, lo, hi) -> np.ndarray:
    """How many of the [lo, hi) index ranges hold each of 0..n-1."""
    ok = hi > lo
    d = np.zeros(n + 1)
    np.add.at(d, np.clip(lo[ok], 0, n), 1)
    np.add.at(d, np.clip(hi[ok], 0, n), -1)
    return np.cumsum(d)[:n]


def _span_counts(pos, lo, hi, w):
    """For sorted positions ``pos``: per contact (lo < hi) how many lie
    with both ends within ``w`` of them -- (hi - w, lo + w] -- and of those
    how many it crosses -- (lo, hi] too: as index ranges [a, b)."""
    a_n = np.searchsorted(pos, hi - w, "right")
    b_n = np.searchsorted(pos, lo + w, "right")
    a_c = np.searchsorted(pos, np.maximum(lo, hi - w), "right")
    b_c = np.searchsorted(pos, np.minimum(hi, lo + w), "right")
    return a_n, b_n, a_c, b_c


def _crossing(lo, hi, geo: Geometry):
    """Per contact (lo < hi) and position of the profile: whether it
    crosses the position, and the chance that a contact of its length placed
    at random would -- anywhere on X outside PAR1, or, with a window, with
    both ends within it of the position (the contact then counts only
    there)."""
    g = geo.grid[None, :].astype(float)
    lo_, hi_ = lo[:, None], hi[:, None]
    d = (hi - lo).astype(float)[:, None]
    cross = (lo_ < g) & (g <= hi_)
    if geo.w > 0:
        inw = (lo_ >= g - geo.w) & (hi_ < g + geo.w)
        e = np.where(inw, np.minimum(d / (2 * geo.w - d), 1.0), 0.0)
        return cross & inw, e
    a_, b_ = geo.x_lo, geo.x_hi
    span = b_ - a_ - d
    num = np.clip(np.minimum(g, b_ - d) - np.maximum(a_, g - d), 0, None)
    return cross, np.where(span > 0, num / np.maximum(span, 1.0), 0.0)


def cell_counts(t: pd.DataFrame, xi_hap: int, geo: Geometry) -> dict:
    """One cell's counts per copy -- 0 the haplotype ``xi_hap`` (the
    inactive X's, on every chromosome), 1 the other; hap1, hap2 when it is
    0."""
    n_ps = PS_EDGES.size - 1
    out = {"ps": np.zeros((len(SETS), 2, n_ps)),
           "share": np.zeros((len(SETS), 2, 1 + len(PS_CLASSES))),
           "prof_o": np.zeros((2, geo.grid.size)),
           "prof_e": np.zeros((2, geo.grid.size)),
           "bnd_n": np.zeros((2, 2, BOUNDARY_OFFSETS.size)),
           "bnd_c": np.zeros((2, 2, BOUNDARY_OFFSETS.size)),
           "comp": np.zeros((2, 2, 4))}
    for i, (_, _, _, sh) in enumerate(geo.loops):
        out[f"loop{i}"] = np.zeros((2, sh.size))
    if geo.gene_iv:
        out["gshare"] = np.zeros((len(geo.gene_iv), 2, 1 + len(PS_CLASSES)))
    if geo.elem_iv:
        # per element, set (ELEMENT_SETS) and copy: its contacts in all and
        # by PS_CLASSES; and per set and copy all of its contacts
        out["eshare"] = np.zeros((len(geo.elem_iv), len(ELEMENT_SETS), 2,
                                  1 + len(PS_CLASSES)))
        out["edenom"] = np.zeros((len(ELEMENT_SETS), 2))
    if t is None or t.empty:
        return out
    t = t[(t["hi"] - t["lo"]) >= 10 ** PS_EDGES[0]]
    copy = np.where(t["hap"].to_numpy() == (xi_hap or 1), 0, 1)
    key = t["chrom"].map(chrom_key).to_numpy()
    lo, hi = t["lo"].to_numpy(np.int64), t["hi"].to_numpy(np.int64)
    chrom = t["chrom"].to_numpy()
    s_ = np.full(len(t), -1)
    is_x = key == "X"
    if geo.par:
        in1 = (lo >= geo.par[0]) & (lo < geo.par[1])
        in2 = (hi >= geo.par[0]) & (hi < geo.par[1])
        s_[is_x & ~in1 & ~in2] = 0
        s_[is_x & in1 & in2] = 1
    else:
        s_[is_x] = 0
    s_[~is_x & ~np.isin(key, ["Y", "M"])] = 2
    d = hi - lo
    b = np.clip(np.searchsorted(PS_EDGES, np.log10(d), "right") - 1, 0,
                n_ps - 1)
    ok = s_ >= 0
    np.add.at(out["ps"], (s_[ok], copy[ok], b[ok]), 1)
    far = ok & (d >= geo.min_d)
    np.add.at(out["share"], (s_[far], copy[far], 0), 1)
    for j, (_, lo_d, hi_d) in enumerate(PS_CLASSES, 1):
        m = far & (d >= lo_d) & (d < (hi_d or np.inf))
        np.add.at(out["share"], (s_[m], copy[m], j), 1)
    # the same of X's contacts with an end in each class of genes
    for gi, iv in enumerate(geo.gene_iv.values()):
        inn = far & (s_ == 0) & (in_intervals(lo, iv) | in_intervals(hi, iv))
        np.add.at(out["gshare"], (gi, copy[inn], 0), 1)
        for j, (_, lo_d, hi_d) in enumerate(PS_CLASSES, 1):
            m = inn & (d >= lo_d) & (d < (hi_d or np.inf))
            np.add.at(out["gshare"], (gi, copy[m], j), 1)
    # the same of the contacts with an end in each element: on X outside
    # the escape genes too, as the levels there are read, and on the
    # autosomes
    if geo.elem_iv:
        sx = far & (s_ == 0)
        if geo.escape_iv is not None and len(geo.escape_iv):
            sx &= (~in_intervals(lo, geo.escape_iv)
                   & ~in_intervals(hi, geo.escape_iv))
        base = (sx, far & (s_ == 2))
        for si, m0 in enumerate(base):
            np.add.at(out["edenom"], (si, copy[m0]), 1)
        for ei, per in enumerate(geo.elem_iv.values()):
            hit = np.zeros(len(t), bool)
            for ck, iv in per.items():
                mc = key == ck
                if mc.any():
                    hit[mc] = in_intervals(lo[mc], iv) | in_intervals(hi[mc],
                                                                      iv)
            for si, m0 in enumerate(base):
                inn = m0 & hit
                np.add.at(out["eshare"], (ei, si, copy[inn], 0), 1)
                for j, (_, lo_d, hi_d) in enumerate(PS_CLASSES, 1):
                    mm = inn & (d >= lo_d) & (d < (hi_d or np.inf))
                    np.add.at(out["eshare"], (ei, si, copy[mm], j), 1)
    # the crossing profile along X
    m = (s_ == 0) & (d >= geo.prof_min)
    if geo.w > 0:
        m &= d < 2 * geo.w
    if m.any() and geo.grid.size:
        o_, e_ = _crossing(lo[m], hi[m], geo)
        for k in (0, 1):
            mk = copy[m] == k
            out["prof_o"][k] += o_[mk].sum(0)
            out["prof_e"][k] += e_[mk].sum(0)
    # the boundaries, pooled, at each offset
    for si, sname in enumerate(("chrX", "autosomes")):
        want = 0 if sname == "chrX" else 2
        m = (s_ == want) & (d >= geo.b_min) & (d < 2 * geo.bw)
        if not m.any():
            continue
        for c in np.unique(chrom[m]):
            pos = geo.bnd[sname].get(c)
            if pos is None or not pos.size:
                continue
            mc = m & (chrom == c)
            for oi, off in enumerate(BOUNDARY_OFFSETS):
                a_n, b_n, a_c, b_c = _span_counts(pos + off, lo[mc], hi[mc],
                                                  geo.bw)
                n_ = b_n - a_n
                c_ = np.maximum(b_c - a_c, 0)
                for k in (0, 1):
                    mk = copy[mc] == k
                    out["bnd_n"][si, k, oi] += n_[mk].sum()
                    out["bnd_c"][si, k, oi] += c_[mk].sum()
    # compartments: the track at a long-range contact's two ends -- each
    # order once, so x and y have one mean: n, sum x, sum x^2, sum xy
    if geo.eig is not None:
        for si, want in enumerate((0, 2)):
            m = (s_ == want) & (d >= COMPARTMENT_MIN_DISTANCE)
            if not m.any():
                continue
            e1 = geo.eig.at(chrom[m], lo[m])
            e2 = geo.eig.at(chrom[m], hi[m])
            ok = np.isfinite(e1) & np.isfinite(e2)
            for k in (0, 1):
                mk = ok & (copy[m] == k)
                out["comp"][si, k] += (2 * mk.sum(),
                                       (e1[mk] + e2[mk]).sum(),
                                       (e1[mk] ** 2 + e2[mk] ** 2).sum(),
                                       2 * (e1[mk] * e2[mk]).sum())
    # superloops: the windows slid along X together
    for i, (_, a_, b_, sh) in enumerate(geo.loops):
        m = (s_ == 0) & (np.abs(d - (b_ - a_)) < 2 * geo.r)
        if not m.any():
            continue
        u, v = lo[m] - a_, hi[m] - b_
        t_lo = np.maximum(u, v) - geo.r      # exclusive
        t_hi = np.minimum(u, v) + geo.r      # inclusive
        ia = np.searchsorted(sh, t_lo, "right")
        ib = np.searchsorted(sh, t_hi, "right")
        for k in (0, 1):
            mk = copy[m] == k
            out[f"loop{i}"][k] += _ranges(sh.size, ia[mk], ib[mk])
    return out


def contact_stats(a: dict, geo: Geometry) -> list:
    """The measures from summed counts ``a`` (leading axes: resamplings):
    [(measure, set, per-copy values [..., 2])]."""
    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for si, sname in enumerate(SETS):
            if sname == "chrX PAR1" and not geo.par:
                continue
            sh = a["share"][..., si, :, :]
            for j, (nm, lo_d, _) in enumerate(PS_CLASSES, 1):
                if sname == "chrX PAR1" and lo_d >= geo.par[1] - geo.par[0]:
                    continue
                out.append((f"P(s): share {nm}", sname,
                            sh[..., j] / sh[..., 0]))
        for gi, gname in enumerate(geo.gene_iv):
            sh = a["gshare"][..., gi, :, :]
            for j, (nm, _, _) in enumerate(PS_CLASSES, 1):
                out.append((f"P(s): share {nm}", f"chrX {gname}",
                            sh[..., j] / sh[..., 0]))
        for ei, ename in enumerate(geo.elem_iv):
            for si, sname in enumerate(ELEMENT_SETS):
                sh = a["eshare"][..., ei, si, :, :]
                for j, (nm, _, _) in enumerate(PS_CLASSES, 1):
                    out.append((f"P(s): share {nm}", f"{sname} {ename}",
                                sh[..., j] / sh[..., 0]))
                out.append((ELEMENT_SHARE, f"{sname} {ename}",
                            sh[..., 0] / a["edenom"][..., si, :]))
        if geo.dxz4 is not None:
            out.append(("crossing DXZ4: log2 obs/exp", "chrX", np.log2(
                a["prof_o"][..., geo.dxz4] / a["prof_e"][..., geo.dxz4])))
        if geo.eig is not None:
            for si, sname in enumerate(("chrX", "autosomes")):
                c = a["comp"][..., si, :, :]
                n_, mu = c[..., 0], c[..., 1] / c[..., 0]
                out.append(("A/B compartments: ends' correlation", sname,
                            (c[..., 3] / n_ - mu ** 2)
                            / (c[..., 2] / n_ - mu ** 2)))
        flank = np.abs(BOUNDARY_OFFSETS) >= geo.flank
        zero = int(np.flatnonzero(BOUNDARY_OFFSETS == 0)[0])
        for si, sname in enumerate(("chrX", "autosomes")):
            if not geo.bnd[sname]:
                continue
            n_, c_ = a["bnd_n"][..., si, :, :], a["bnd_c"][..., si, :, :]
            fl = c_[..., flank].sum(-1) / n_[..., flank].sum(-1)
            at = c_[..., zero] / n_[..., zero]
            out.append(("TAD boundary insulation", sname, np.log2(fl / at)))
        if geo.loops:
            o_s, e_s = 0, 0
            for i, (nm, _, _, sh) in enumerate(geo.loops):
                k = a[f"loop{i}"]
                bg = np.abs(sh) >= geo.loop_excl
                o_ = k[..., int(np.flatnonzero(sh == 0)[0])]
                e_ = k[..., bg].mean(-1)
                out.append((f"superloop {nm}: obs/exp", "chrX", o_ / e_))
                o_s, e_s = o_s + o_, e_s + e_
            out.append(("superloops pooled: obs/exp", "chrX", o_s / e_s))
    return out


def fdr_bh(p) -> np.ndarray:
    """Benjamini-Hochberg q over the finite p-values; NaN stays NaN."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size:
        o = ok[np.argsort(p[ok])]
        v = p[o] * o.size / np.arange(1, o.size + 1)
        q[o] = np.minimum(np.minimum.accumulate(v[::-1])[::-1], 1.0)
    return q


def _stack(counts: list) -> dict:
    return {k: np.stack([c[k] for c in counts]) for k in counts[0]}


def group_stats(a: dict, geo: Geometry, w, labels) -> pd.DataFrame:
    """Per measure and set each copy's value, their difference with its
    95% interval over resampled cells (``w``: the resamplings' weights of
    the cells of ``a``, stacked counts), p (normal, from the resampled SE);
    where both X and the autosomes have it, the difference on X less on the
    autosomes."""
    from scipy.stats import norm
    full = contact_stats({k: v.sum(0) for k, v in a.items()}, geo)
    boot = contact_stats({k: np.tensordot(w, v, axes=(1, 0))
                          for k, v in a.items()}, geo)
    rows, by = [], {}
    for (meas, sname, v), (_, _, vb) in zip(full, boot):
        d, db = v[0] - v[1], vb[:, 0] - vb[:, 1]
        by[(meas, sname)] = (d, db)
        rows.append(_row(meas, sname, v, d, db, norm, labels))
    ge, gi = (f"chrX {c}" for c in GENE_CLASSES)
    for meas in dict.fromkeys(m for m, _ in by):
        if (meas, "chrX") in by and (meas, "autosomes") in by:
            (dx, bx), (da, ba) = by[(meas, "chrX")], by[(meas, "autosomes")]
            rows.append(_row(meas, "chrX − autosomes", (np.nan, np.nan),
                             dx - da, bx - ba, norm, labels))
        # each element's copies' difference on X less on the autosomes
        for ename in geo.elem_iv:
            kx, ka = (meas, f"chrX {ename}"), (meas, f"autosomes {ename}")
            if kx in by and ka in by:
                (dx, bx), (da, ba) = by[kx], by[ka]
                rows.append(_row(meas, f"{ename}: chrX − autosomes",
                                 (np.nan, np.nan), dx - da, bx - ba, norm,
                                 labels))
        # the escape genes' copies' difference less the inactivated genes'
        if (meas, ge) in by and (meas, gi) in by:
            (d1, b1), (d2, b2) = by[(meas, ge)], by[(meas, gi)]
            rows.append(_row(meas, GENE_CONTRAST, (np.nan, np.nan), d1 - d2,
                             b1 - b2, norm, labels))
    return pd.DataFrame(rows)


def _sum_by(a: dict, groups: np.ndarray, k: int) -> dict:
    """The stacked counts ``a`` (cells first) summed per group 0..k-1:
    each array with the groups as its leading axis."""
    return {key: np.stack([v[groups == j].sum(0) for j in range(k)])
            for key, v in a.items()}


def _quantile_groups(vals: np.ndarray, k: int) -> np.ndarray:
    """Each value's quantile group 0..k-1, by rank (ties in order)."""
    r = np.argsort(np.argsort(vals, kind="stable"), kind="stable")
    return (r * k) // max(vals.size, 1)


def xci_strata(a: dict, strength: np.ndarray, geo: Geometry, labels,
               k: int, n_boot: int, n_perm: int, rng):
    """A line with one inactive X, its cells grouped by their own XCI
    strength (``strength``, per cell of ``a``; NaN without a call) into
    ``k`` quantile strata: per stratum each measure's copies and their
    difference with its 95% interval over the stratum's cells resampled
    (:func:`group_stats`); and per measure the slope of that difference on
    the strata's mean strength, its p from the strengths permuted among the
    cells (``n_perm``).  A cell's two copies share its depth and its
    preparation, so their difference is the cell's inactive X against its
    active one whatever the cell's quality; the autosomes' copies are the
    control.  Returns (strata rows, slope rows)."""
    ok = np.flatnonzero(np.isfinite(strength))
    if ok.size < 2 * k:
        return pd.DataFrame(), pd.DataFrame()
    s = strength[ok]
    sub = {key: v[ok] for key, v in a.items()}
    grp = _quantile_groups(s, k)
    rows = []
    for j in range(k):
        m = grp == j
        n = int(m.sum())
        w = rng.multinomial(n, np.full(n, 1.0 / n),
                            size=n_boot).astype(float)
        t = group_stats({key: v[m] for key, v in sub.items()}, geo, w,
                        labels)
        t.insert(0, "stratum", j + 1)
        t.insert(1, "cells", n)
        t.insert(2, "strength_mean", float(s[m].mean()))
        t.insert(3, "strength_min", float(s[m].min()))
        t.insert(4, "strength_max", float(s[m].max()))
        rows.append(t)

    def slopes(g_, vals):
        # the strata's mean strength, of the strengths that made them
        x = np.array([vals[g_ == j].mean() for j in range(k)])
        out = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for meas, sname, v in contact_stats(_sum_by(sub, g_, k), geo):
                d = v[:, 0] - v[:, 1]
                f = np.isfinite(d)
                if f.sum() >= 3 and np.ptp(x[f]) > 0:
                    out[(meas, sname)] = float(np.polyfit(x[f], d[f], 1)[0])
        return out
    obs = slopes(grp, s)
    null = {key: [] for key in obs}
    for _ in range(n_perm):
        sp = rng.permutation(s)
        res = slopes(_quantile_groups(sp, k), sp)
        for key in obs:
            null[key].append(res.get(key, np.nan))
    trend = []
    for (meas, sname), b in obs.items():
        nb_ = np.asarray(null[(meas, sname)], float)
        nb_ = np.abs(nb_[np.isfinite(nb_)])
        trend.append({"measure": meas, "set": sname, "strata": k,
                      "cells": int(ok.size), "slope": b,
                      "p_perm": (1 + int((nb_ >= abs(b) - 1e-12).sum()))
                      / (1 + nb_.size), "permutations": int(nb_.size)})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(trend)


def strength_summary(strength: np.ndarray, noise_sd: np.ndarray) -> dict:
    """How much of the cells' spread in XCI strength is more than their
    sites' binomial noise: the observed SD, the noise's (root mean square),
    the excess, the share of cells whose inactive X reads the less
    methylated, and that share expected of the noise alone."""
    from scipy.stats import norm
    ok = np.isfinite(strength) & np.isfinite(noise_sd) & (noise_sd > 0)
    s, e = strength[ok], noise_sd[ok]
    if s.size < 3:
        return {}
    var = float(np.var(s, ddof=1))
    nv = float(np.mean(e ** 2))
    return {"cells": int(s.size), "mean": float(s.mean()),
            "sd": var ** 0.5, "noise_sd": nv ** 0.5,
            "excess_sd": max(var - nv, 0.0) ** 0.5,
            "share_below_0": float((s < 0).mean()),
            "share_below_0_noise": float(np.mean(norm.cdf(-s.mean() / e)))}


def _row(meas, sname, v, d, db, norm, labels) -> dict:
    db = db[np.isfinite(db)]
    se = float(np.std(db, ddof=1)) if db.size > 1 else np.nan
    lo, hi = (np.percentile(db, [2.5, 97.5]) if db.size
              else (np.nan, np.nan))
    return {"measure": meas, "set": sname,
            labels[0]: float(v[0]), labels[1]: float(v[1]),
            "difference": float(d), "ci_lo": lo, "ci_hi": hi, "se": se,
            "p": (float(2 * norm.sf(abs(d) / se)) if se > 0 and np.isfinite(d)
                  else np.nan)}


def profile_table(a: dict, geo: Geometry, labels) -> pd.DataFrame:
    """Per position along X and copy: the contacts crossing it, those
    expected, log2 of their ratio; and the copies' difference."""
    o_, e_ = a["prof_o"].sum(0), a["prof_e"].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log2(o_ / e_)
    t = pd.DataFrame({"pos": geo.grid})
    for k, lab in enumerate(labels):
        t[f"crossing_{lab}"] = o_[k]
        t[f"expected_{lab}"] = e_[k]
        t[f"log2_obs_exp_{lab}"] = r[k]
    t["difference"] = r[0] - r[1]
    return t


def ps_ratio_table(a: dict, w, labels) -> pd.DataFrame:
    """Per set and distance bin: log2 of the first copy's P(s) over the
    second's, each a share of its own contacts, with its 95% interval over
    the resampled cells (``w``)."""
    ps = a["ps"]
    full, boot = ps.sum(0), np.tensordot(w, ps, axes=(1, 0))

    def ratio(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            f = x / x.sum(-1, keepdims=True)
            return np.log2(f[..., 0, :] / f[..., 1, :])
    r, rb = ratio(full), ratio(boot)
    # a bin without contacts on either copy: no ratio
    r = np.where((full[:, 0, :] > 0) & (full[:, 1, :] > 0), r, np.nan)
    rows = []
    for si, sname in enumerate(SETS):
        for b in range(PS_EDGES.size - 1):
            v = rb[:, si, b]
            v = v[np.isfinite(v)]
            lo, hi = (np.percentile(v, [2.5, 97.5]) if v.size > 1
                      else (np.nan, np.nan))
            rows.append({"set": sname, "log10_lo": PS_EDGES[b],
                         "log10_hi": PS_EDGES[b + 1],
                         f"contacts_{labels[0]}": int(full[si, 0, b]),
                         f"contacts_{labels[1]}": int(full[si, 1, b]),
                         "log2_ratio": r[si, b], "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def ps_table(a: dict, labels) -> pd.DataFrame:
    """Per set, copy and distance bin: the contacts, their share of the
    copy's contacts, and that share per kb of distance -- P(s) as Hi-C
    draws it (a share per log10 bin is s P(s), flat where P(s) ~ 1/s)."""
    ps = a["ps"].sum(0)
    kb = (10 ** PS_EDGES[1:] - 10 ** PS_EDGES[:-1]) / 1e3
    rows = []
    for si, sname in enumerate(SETS):
        for k, lab in enumerate(labels):
            tot = ps[si, k].sum()
            for b in range(PS_EDGES.size - 1):
                sh = ps[si, k, b] / tot if tot else np.nan
                rows.append({"set": sname, "copy": lab,
                             "log10_lo": PS_EDGES[b],
                             "log10_hi": PS_EDGES[b + 1],
                             "contacts": int(ps[si, k, b]), "share": sh,
                             "per_kb": sh / kb[b]})
    return pd.DataFrame(rows)


def boundary_table(a: dict, labels) -> pd.DataFrame:
    n_, c_ = a["bnd_n"].sum(0), a["bnd_c"].sum(0)
    rows = []
    for si, sname in enumerate(("chrX", "autosomes")):
        for k, lab in enumerate(labels):
            for oi, off in enumerate(BOUNDARY_OFFSETS):
                rows.append({"set": sname, "copy": lab, "offset": int(off),
                             "contacts": int(n_[si, k, oi]),
                             "crossing": int(c_[si, k, oi]),
                             "share_crossing": (c_[si, k, oi] / n_[si, k, oi]
                                                if n_[si, k, oi] else
                                                np.nan)})
    return pd.DataFrame(rows)


#: haplotype_contacts_<line>: the copies' colours (the inactive X red).
_COLS = ("#b2182b", "#2166ac")


def plot_contacts(stats: pd.DataFrame, ps: pd.DataFrame, ratio: pd.DataFrame,
                  prof: pd.DataFrame, bnd: pd.DataFrame, geo: Geometry,
                  elements: dict, labels, group: str, path):
    """a: P(s) of each copy, X and the autosomes; b: log2 of one copy's
    P(s) over the other's, with its 95% interval -- X and the autosomes;
    c: along X, the contacts crossing each position over those expected,
    each copy and their difference, the inactive X's elements marked; d:
    the share of contacts crossing a position by its offset from the TAD
    boundaries, over the flanks'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.35})
    (ax_a, ax_b), (ax_c, ax_d) = axes
    for sname, ls in (("chrX", "-"), ("autosomes", "--")):
        for k, lab in enumerate(labels):
            t = ps[(ps["set"] == sname) & (ps["copy"] == lab)
                   & (ps["log10_lo"] >= 4) & (ps["contacts"] > 0)]
            if len(t):
                ax_a.plot(10 ** ((t["log10_lo"] + t["log10_hi"]) / 2),
                          t["per_kb"], ls, color=_COLS[k], lw=0.9,
                          alpha=1 if sname == "chrX" else 0.5,
                          label=f"{sname} {lab}")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("genomic distance, bp", fontsize=6)
    ax_a.set_ylabel("contact frequency (share per kb)", fontsize=6)
    st = stats.set_index(["measure", "set"])
    ttl = "a  P(s)"
    k_ = (f"P(s): share {PS_CLASSES[1][0]}", "chrX")
    if k_ in st.index:
        r = st.loc[k_]
        ttl += (f": {PS_CLASSES[1][0]} {100 * r[labels[0]]:.0f}% vs "
                f"{100 * r[labels[1]]:.0f}% on X")
    ax_a.set_title(ttl, loc="left", fontsize=6.5)
    ax_a.legend(fontsize=5, frameon=False)
    for sname, col in (("chrX", _COLS[0]), ("autosomes", "0.45")):
        t = ratio[(ratio["set"] == sname) & (ratio["log10_lo"] >= 4)
                  & (ratio[[f"contacts_{lab}" for lab in labels]].min(axis=1)
                     >= 20)]
        if t.empty:
            continue
        x = 10 ** ((t["log10_lo"] + t["log10_hi"]) / 2)
        ax_b.fill_between(x, t["ci_lo"], t["ci_hi"], color=col, alpha=0.2,
                          lw=0)
        ax_b.plot(x, t["log2_ratio"], color=col, lw=0.9, label=sname)
    ax_b.axhline(0, color="0.6", lw=0.4)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("genomic distance, bp", fontsize=6)
    ax_b.set_ylabel(f"log2 P(s) {labels[0]} / {labels[1]}", fontsize=6)
    ax_b.set_title("b  where the copies' contacts differ (95% interval)",
                   loc="left", fontsize=6.5)
    if ax_b.get_legend_handles_labels()[0]:
        ax_b.legend(fontsize=5, frameon=False)
    if len(prof):
        x = prof["pos"] / 1e6
        for k, lab in enumerate(labels):
            ax_c.plot(x, prof[f"log2_obs_exp_{lab}"], color=_COLS[k], lw=0.8,
                      label=f"X {lab}")
        ax_c.plot(x, prof["difference"], color="black", lw=0.8,
                  label=f"{labels[0]} − {labels[1]}")
        for nm, (a_, b_) in elements.items():
            ax_c.axvline((a_ + b_) / 2e6, color="0.4", lw=0.5, ls=":")
            ax_c.text((a_ + b_) / 2e6, 1.0, nm, fontsize=5, ha="center",
                      va="bottom", transform=ax_c.get_xaxis_transform())
        ax_c.axhline(0, color="0.6", lw=0.4)
        ax_c.legend(fontsize=5, frameon=False, loc="lower left")
    ax_c.set_xlabel("chrX, Mb", fontsize=6)
    ax_c.set_ylabel("log2 contacts crossing, obs / exp", fontsize=6)
    scope = (f"within {geo.w / 1e6:g} Mb" if geo.w > 0 else "on all of X")
    ax_c.set_title(f"c  crossing each position (≥ {geo.prof_min / 1e6:g} "
                   f"Mb apart, {scope})", loc="left", fontsize=6.5)
    flank = np.abs(BOUNDARY_OFFSETS) >= geo.flank
    for sname, ls in (("chrX", "-"), ("autosomes", "--")):
        for k, lab in enumerate(labels):
            t = bnd[(bnd["set"] == sname) & (bnd["copy"] == lab)]
            if not t["contacts"].sum():
                continue
            base = (t["crossing"].to_numpy()[flank].sum()
                    / t["contacts"].to_numpy()[flank].sum())
            ax_d.plot(t["offset"] / 1e3, np.log2(t["share_crossing"] / base),
                      ls, marker="o", ms=1.8, color=_COLS[k], lw=0.8,
                      alpha=1 if sname == "chrX" else 0.5,
                      label=f"{sname} {lab}")
    ax_d.axhline(0, color="0.6", lw=0.4)
    ax_d.set_xlabel("offset from the TAD boundaries, kb", fontsize=6)
    ax_d.set_ylabel("log2 share crossing / flanks'", fontsize=6)
    ax_d.set_title("d  TAD boundaries", loc="left", fontsize=6.5)
    if ax_d.get_legend_handles_labels()[0]:
        ax_d.legend(fontsize=5, frameon=False)
    for ax in axes.ravel():
        ax.tick_params(labelsize=5.5)
    fig.suptitle(f"{group}: {labels[0]} against {labels[1]}", fontsize=7,
                 y=0.98)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


#: A locus's map resolution: the finest of these giving it at most
#: MAP_BINS bins.
MAP_RESOLUTIONS = (50_000, 100_000, 200_000, 250_000, 500_000, 1_000_000,
                   2_000_000, 2_500_000, 5_000_000, 10_000_000)
MAP_BINS = 64


def parse_loci(values, elements: dict, x_name: str, x_lo: int, x_len: int):
    """--locus NAME=CHROM:START-END (or NAME=CHROM, all of it): [(name,
    start, end)] on X.  By default all of X outside PAR1 and DXZ4 +/- 10
    Mb; 'none': no locus."""
    if values and any(str(v).strip().lower() == "none" for v in values):
        return []
    out = []
    for v in values or []:
        nm, _, reg = str(v).partition("=")
        c, _, se = reg.replace(",", "").partition(":")
        if chrom_key(c) != "X":
            raise ValueError(f"--locus {v}: on chrX only")
        if se:
            a_, _, b_ = se.partition("-")
            if not (a_.isdigit() and b_.isdigit() and int(a_) < int(b_)):
                raise ValueError(f"--locus {v}: NAME=chrX:START-END")
            out.append((nm.strip(), int(a_), min(int(b_), x_len)))
        else:
            out.append((nm.strip(), x_lo, x_len))
    if values:
        return out
    out = [(x_name, x_lo, x_len)]
    if "DXZ4" in elements:
        mid = sum(elements["DXZ4"]) // 2
        out.append(("DXZ4", max(mid - 10_000_000, x_lo),
                    min(mid + 10_000_000, x_len)))
    return out


def x_contacts(t: pd.DataFrame, xi_hap: int, geo: Geometry):
    """A cell's contacts on X outside PAR1 (1 kb or more apart): low end,
    high end and copy (0 the haplotype ``xi_hap``, as
    :func:`cell_counts`)."""
    if t is None or t.empty:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, np.int8),)
    m = t["chrom"].map(chrom_key).eq("X").to_numpy()
    lo, hi = t["lo"].to_numpy(np.int64)[m], t["hi"].to_numpy(np.int64)[m]
    ok = (hi - lo) >= 10 ** PS_EDGES[0]
    if geo.par:
        ok &= ~((lo >= geo.par[0]) & (lo < geo.par[1]))
        ok &= ~((hi >= geo.par[0]) & (hi < geo.par[1]))
    cp = np.where(t["hap"].to_numpy()[m] == (xi_hap or 1), 0, 1)
    return lo[ok], hi[ok], cp[ok].astype(np.int8)


def read_allelic(path, contexts) -> pd.DataFrame:
    """allelic_imbalance.py's allelic_snps.tsv on X, phased SNPs only: per
    line and SNP each haplotype's level per context and its reads."""
    head = pd.read_csv(path, sep="\t", nrows=0).columns
    need = ["group", "chrom", "pos", "ref_hap"]
    for c in contexts:
        need += [f"{c}_ref", f"{c}_alt", f"{c}_reads_ref", f"{c}_reads_alt"]
    miss = [c for c in need if c not in head]
    if miss:
        sys.exit(f"--allelic-snps {path}: no {', '.join(miss)} "
                 f"(--track-contexts)")
    parts = []
    for ch in pd.read_csv(path, sep="\t", usecols=need, chunksize=500_000,
                          dtype={"group": str, "chrom": str}):
        ch = ch[ch["chrom"].map(chrom_key).eq("X")
                & ch["ref_hap"].isin([1, 2])]
        if len(ch):
            parts.append(ch)
    t = pd.concat(parts, ignore_index=True) if parts else \
        pd.DataFrame(columns=need)
    out = t[["group", "chrom", "pos"]].copy()
    r1 = t["ref_hap"].eq(1).to_numpy()
    for c in contexts:
        for h, (mine, other) in ((1, ("ref", "alt")), (2, ("alt", "ref"))):
            out[f"{c}_hap{h}"] = np.where(r1, t[f"{c}_{mine}"],
                                          t[f"{c}_{other}"])
            out[f"{c}_reads_hap{h}"] = np.where(
                r1, t[f"{c}_reads_{mine}"], t[f"{c}_reads_{other}"])
    return out


def _binned(pos, val, w, edges) -> np.ndarray:
    """``val`` weighted by ``w``, averaged per bin of ``edges``."""
    ok = np.isfinite(val) & np.isfinite(w) & (w > 0) & (pos >= edges[0]) \
        & (pos < edges[-1])
    b = np.searchsorted(edges, pos[ok], "right") - 1
    n = np.bincount(b, w[ok], edges.size - 1)
    s_ = np.bincount(b, w[ok] * val[ok], edges.size - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s_ / n, np.nan)


def locus_tracks(lo, hi, cp, start: int, end: int, eig, x_name: str,
                 allelic, conc, contexts, xi_hap: int, smooth: float):
    """One locus's tracks per copy (0, 1): the contact map -- the copy's
    share of its X contacts per pair of bins, smoothed -- insulation (log2
    contacts crossing over those expected within 4 bins), the mean
    compartment signal of the bins' long-range partners, and, per context,
    the level (allelic_snps) and ΔC (chrx_haplotype_tracks) of each copy."""
    from scipy.ndimage import gaussian_filter
    width = end - start
    res = next((r for r in MAP_RESOLUTIONS if width / r <= MAP_BINS),
               MAP_RESOLUTIONS[-1])
    edges = np.arange(start, end + res, res, dtype=np.int64)
    edges[-1] = min(edges[-1], end)
    nb = edges.size - 1
    mid = (edges[:-1] + edges[1:]) // 2
    tot = np.array([(cp == k).sum() for k in (0, 1)], float)
    inside = (lo >= start) & (hi < end)
    maps = np.zeros((2, nb, nb))
    for k in (0, 1):
        m = inside & (cp == k)
        i = np.searchsorted(edges, lo[m], "right") - 1
        j = np.searchsorted(edges, hi[m], "right") - 1
        np.add.at(maps[k], (i, j), 1)
        maps[k] = maps[k] + maps[k].T - np.diag(np.diag(maps[k]))
        if smooth > 0:
            maps[k] = gaussian_filter(maps[k], smooth)
        maps[k] /= max(tot[k], 1)
    out = {"res": res, "edges": edges, "maps": maps, "tot": tot,
           "ratio": oe_ratio(maps)}
    # insulation within 4 bins, from contacts a bin or more apart
    from types import SimpleNamespace
    g_ = SimpleNamespace(grid=mid, w=4 * res, x_lo=start, x_hi=end)
    ins = np.full((2, nb), np.nan)
    for k in (0, 1):
        m = (cp == k) & ((hi - lo) >= res) & ((hi - lo) < 8 * res) \
            & (hi > start) & (lo < end)
        if m.any():
            o_, e_ = _crossing(lo[m], hi[m], g_)
            with np.errstate(invalid="ignore", divide="ignore"):
                ins[k] = np.log2(o_.sum(0) / e_.sum(0))
    out["insulation"] = ins
    # compartments: the bulk signal per bin, and each copy's partners'
    if eig is not None:
        out["eigen"] = _binned(mid, eig.at(np.full(nb, x_name), mid),
                               np.ones(nb), edges)
        part = np.full((2, nb), np.nan)
        far = (hi - lo) >= COMPARTMENT_MIN_DISTANCE
        for k in (0, 1):
            m = far & (cp == k)
            pos = np.concatenate([lo[m], hi[m]])
            oth = np.concatenate([hi[m], lo[m]])
            part[k] = _binned(pos, eig.at(np.full(pos.size, x_name), oth),
                              np.ones(pos.size), edges)
        out["partners"] = part
    # 1D: each copy's level per context, from the phased SNPs
    if allelic is not None and len(allelic):
        pos = allelic["pos"].to_numpy(np.int64)
        for c in [c for c in contexts if f"{c}_hap1" in allelic]:
            lv = np.full((2, nb), np.nan)
            for k in (0, 1):
                h = (xi_hap or 1) if k == 0 else 3 - (xi_hap or 1)
                lv[k] = _binned(pos, allelic[f"{c}_hap{h}"].to_numpy(float),
                                allelic[f"{c}_reads_hap{h}"].to_numpy(float),
                                edges)
            out[f"level_{c}"] = lv
    # concordance: each copy's ΔC in its context's own windows, and copy
    # 0's less copy 1's -- every context of --track-contexts, empty where
    # none was tagged
    if conc is not None:
        for c in contexts:
            t = conc[conc["context"].astype(str).str.upper() == c.upper()]
            mid_ = (t["bin_start"] + t["bin_end"]) / 2
            t = t.assign(center=t["center"].fillna(mid_) if "center" in t
                         else mid_)
            # every window over the locus: the lines run to its edges
            t = t[(t["bin_end"] > start) & (t["bin_start"] < end)]
            rows = {}
            for k in (0, 1):
                h = (xi_hap or 1) if k == 0 else 3 - (xi_hap or 1)
                rows[k] = t[t["haplotype"] == f"hap{h}"].sort_values(
                    "center")
            d = t[t["haplotype"] == "hap1-hap2"].sort_values("center")
            if (xi_hap or 1) == 2 and len(d):
                d = d.assign(delta=-d["delta"], delta_lo=-d["delta_hi"],
                             delta_hi=-d["delta_lo"])
            rows["diff"] = d
            out[f"delta_{c}"] = rows
    return out


def oe_ratio(maps) -> np.ndarray:
    """log2 of copy 0's contacts over copy 1's, each first over its own
    mean at that distance in the locus -- where the two fold differently
    apart from how fast their contacts fall with distance (P(s))."""
    n = maps.shape[1]
    oe = np.full(maps.shape, np.nan)
    for k in (0, 1):
        for off in range(n):
            i = np.arange(n - off)
            v = maps[k][i, i + off]
            mu = v.mean()
            if mu > 0:
                oe[k][i, i + off] = v / mu
                oe[k][i + off, i] = v / mu
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log2(oe[0] / oe[1])
    return np.where(np.isfinite(r), r, np.nan)


def map_table(tr: dict, labels) -> pd.DataFrame:
    """A locus's maps, pixel by pixel (the upper triangle): each copy's
    share of its contacts, smoothed, and log2 of the copies' observed over
    expected."""
    e = tr["edges"]
    i, j = np.triu_indices(e.size - 1)
    return pd.DataFrame({"start_1": e[i], "start_2": e[j],
                         labels[0]: tr["maps"][0][i, j],
                         labels[1]: tr["maps"][1][i, j],
                         "log2_oe_ratio": tr["ratio"][i, j]})


def locus_table(tr: dict, contexts, labels) -> pd.DataFrame:
    """A locus's binned tracks, per copy."""
    e = tr["edges"]
    t = pd.DataFrame({"start": e[:-1], "end": e[1:]})
    for k, lab in enumerate(labels):
        t[f"insulation_{lab}"] = tr["insulation"][k]
        if "partners" in tr:
            t[f"partners_compartment_{lab}"] = tr["partners"][k]
        for c in contexts:
            if f"level_{c}" in tr:
                t[f"{c}_{lab}"] = tr[f"level_{c}"][k]
    if "eigen" in tr:
        t["compartment_bulk"] = tr["eigen"]
    # each bin's ΔC: that of the context's window centred nearest its
    # middle, holding it
    mid = (e[:-1] + e[1:]) // 2
    for c in contexts:
        if f"delta_{c}" not in tr:
            continue
        for k, lab in ((0, labels[0]), (1, labels[1]),
                       ("diff", f"{labels[0]}_minus_{labels[1]}")):
            r = tr[f"delta_{c}"][k]
            v = np.full(mid.size, np.nan)
            if len(r):
                holds = (r["bin_start"].to_numpy()[None, :] <= mid[:, None]) \
                    & (mid[:, None] < r["bin_end"].to_numpy()[None, :])
                dist = np.where(holds, np.abs(r["center"].to_numpy()[None, :]
                                              - mid[:, None]), np.inf)
                i, ok = dist.argmin(1), holds.any(1)
                v[ok] = r["delta"].to_numpy()[i[ok]]
            t[f"{c}_dC_{lab}"] = v
    return t


def _triangle(ax, m, edges, up: bool, vmin, vmax, depth: float,
              log: bool = True, cmap: str = "YlOrRd"):
    """A contact map as a triangle along the genome, up or down, to
    ``depth`` of the distance (bp) between the bins; log10 of ``m``, or
    ``m`` itself."""
    x_, y_ = np.meshgrid(edges, edges)
    xs, ys = (x_ + y_) / 2e6, (x_ - y_) / 2e6
    if not up:
        ys = -ys
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = np.log10(m) if log else np.asarray(m, float)
    lm = np.ma.masked_where(np.tril(np.ones(m.shape, bool), -1)
                            | ~np.isfinite(lm), lm)
    qm = ax.pcolormesh(xs, ys, lm, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="flat", rasterized=True)
    ax.set_ylim((0, depth / 2e6) if up else (-depth / 2e6, 0))
    ax.set_yticks([])
    for sp in ("left", "right", "top", "bottom"):
        ax.spines[sp].set_visible(False)
    return qm


def _beside(fig, axes, width=0.012, pad=0.01):
    """Axes for a colour bar just right of ``axes``, as tall as they are
    together: beside them, not taken out of their width, so every panel
    keeps one x for one position."""
    b = [ax.get_position() for ax in axes]
    y0, y1 = min(p.y0 for p in b), max(p.y1 for p in b)
    return fig.add_axes([max(p.x1 for p in b) + pad, y0, width, y1 - y0])


def plot_locus(tr: dict, name: str, start: int, end: int, group: str,
               labels, contexts, elements: dict, tads, escape, path):
    """:func:`locus_figure`, saved to ``path``."""
    import matplotlib.pyplot as plt
    fig = locus_figure(tr, name, start, end, group, labels, contexts,
                       elements, tads, escape)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def locus_figure(tr: dict, name: str, start: int, end: int, group: str,
                 labels, contexts, elements: dict, tads, escape):
    """The two copies along one locus, as a genome browser shows them: each
    copy's contact map (copy 1 up, copy 2 down), the bulk TADs between them,
    log2 of copy 1's over copy 2's, each as observed over expected at its
    distance (red: copy 1's the more; blue: copy 2's),
    and per copy insulation, compartments (the bulk signal; each copy's
    long-range partners'), each context's level and ΔC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = [("insulation", "insulation\nlog2 obs/exp")]
    if "partners" in tr:
        lines.append(("partners", "A/B\ncompartment"))
    for c in contexts:
        if f"level_{c}" in tr:
            lines.append((f"level_{c}", f"{c} level"))
    for c in contexts:
        if f"delta_{c}" in tr:
            r = pd.concat([tr[f"delta_{c}"][0], tr[f"delta_{c}"][1]])
            w = (f"\n{(r['bin_end'] - r['bin_start']).max() / 1e6:g} Mb "
                 f"windows" if len(r) else "")
            lines.append((f"delta_{c}",
                          f"{dict(CONC_CONTEXTS).get(c, c)} ΔC{w}"))
    hr = [3.2, 0.35, 3.2, 2.6] + [1.0] * len(lines)
    fig = plt.figure(figsize=(7.2, 6.2 + 1.0 * len(lines)))
    gs = fig.add_gridspec(len(hr), 1, height_ratios=hr, hspace=0.18)
    lo_mb, hi_mb = start / 1e6, end / 1e6
    pos = tr["maps"][tr["maps"] > 0]
    vmax = np.log10(np.percentile(pos, 99.5)) if pos.size else 0
    vmin = vmax - 2.5
    depth = 0.5 * (end - start)
    ax_up = fig.add_subplot(gs[0])
    qm = _triangle(ax_up, tr["maps"][0], tr["edges"], True, vmin, vmax,
                   depth)
    ax_up.set_ylabel(labels[0], rotation=0, ha="right", va="center",
                     fontsize=7)
    ax_up.set_title(f"{group} {name}: chrX {lo_mb:.1f}-{hi_mb:.1f} Mb, "
                    f"{tr['res'] / 1e3:g} kb bins ({int(tr['tot'][0]):,} / "
                    f"{int(tr['tot'][1]):,} contacts on X)", fontsize=7,
                    loc="left")
    ax_t = fig.add_subplot(gs[1])
    if tads is not None:
        b = np.sort(tads[(tads >= start) & (tads < end)])
        for i, (a_, b_) in enumerate(zip(b[:-1], b[1:])):
            ax_t.plot([a_ / 1e6, b_ / 1e6], [i % 2, i % 2], color="#3182bd",
                      lw=2.2, solid_capstyle="butt")
    ax_t.set_ylim(-0.8, 1.8)
    ax_t.set_yticks([])
    ax_t.set_ylabel("TADs", rotation=0, ha="right", va="center", fontsize=6)
    ax_dn = fig.add_subplot(gs[2])
    _triangle(ax_dn, tr["maps"][1], tr["edges"], False, vmin, vmax, depth)
    ax_dn.set_ylabel(labels[1], rotation=0, ha="right", va="center",
                     fontsize=7)
    cb = fig.colorbar(qm, cax=_beside(fig, [ax_up, ax_dn]))
    cb.set_label("log10 share of the copy's contacts", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    # where the copies fold differently, their P(s) aside
    ax_r = fig.add_subplot(gs[3])
    qr = _triangle(ax_r, tr["ratio"], tr["edges"], True, -1, 1, depth,
                   log=False, cmap="RdBu_r")
    ax_r.set_ylabel(f"{labels[0]} / {labels[1]}", rotation=0, ha="right",
                    va="center", fontsize=7)
    cr = fig.colorbar(qr, cax=_beside(fig, [ax_r]))
    cr.set_label("log2 obs/exp,\n" + f"{labels[0]} over {labels[1]}",
                 fontsize=5.5)
    cr.ax.tick_params(labelsize=5)
    mid = (tr["edges"][:-1] + tr["edges"][1:]) / 2e6
    axes = []
    for r, (key, ylab) in enumerate(lines):
        ax = fig.add_subplot(gs[4 + r], sharex=axes[0] if axes else None)
        axes.append(ax)
        if key == "partners":
            e = tr["eigen"]
            ax.bar(mid, np.where(e > 0, e, 0), width=tr["res"] / 1e6,
                   color="#e6550d", alpha=0.35, lw=0, label="bulk A")
            ax.bar(mid, np.where(e < 0, e, 0), width=tr["res"] / 1e6,
                   color="#3182bd", alpha=0.35, lw=0, label="bulk B")
        if key.startswith("delta_"):
            if not (len(tr[key][0]) or len(tr[key][1])):
                ax.text(0.5, 0.5, f"no {key[6:]} contact tagged on a copy "
                                  f"here", transform=ax.transAxes,
                        ha="center", va="center", fontsize=6, color="0.4")
            # each copy's, and copy 0's less copy 1's (black), at the
            # windows' centres, each with its 95% interval
            for k, lab, col, lw in ((0, labels[0], _COLS[0], 0.8),
                                    (1, labels[1], _COLS[1], 0.8),
                                    ("diff", f"{labels[0]} − {labels[1]}",
                                     "black", 1.0)):
                t = tr[key][k]
                if not len(t):
                    continue
                x = t["center"].to_numpy() / 1e6
                ax.fill_between(x, t["delta_lo"], t["delta_hi"], color=col,
                                alpha=0.12 if k != "diff" else 0.18, lw=0)
                ax.plot(x, t["delta"], color=col, lw=lw, label=lab)
            ax.axhline(0, color="0.6", lw=0.4)
            if key == next(k_ for k_, _ in lines if k_.startswith("delta_")):
                ax.legend(fontsize=5, frameon=False, ncol=3,
                          loc="upper right")
        else:
            v = tr[key]
            for k, lab in enumerate(labels):
                ax.plot(mid, v[k], color=_COLS[k], lw=0.9, marker="o",
                        ms=1.5, label=lab)
            if key == "insulation":
                ax.axhline(0, color="0.6", lw=0.4)
        ax.set_ylabel(ylab, fontsize=6)
        ax.tick_params(labelsize=5.5)
        if r == 0:
            ax.legend(fontsize=5, frameon=False, ncol=2, loc="upper right")
        if key == "partners":
            ax.legend(fontsize=5, frameon=False, ncol=4, loc="upper right")
    for ax in [ax_up, ax_t, ax_dn, ax_r] + axes:
        ax.set_xlim(lo_mb, hi_mb)
        for nm, (a_, b_) in elements.items():
            if a_ < end and b_ > start:
                ax.axvline((a_ + b_) / 2e6, color="0.3", lw=0.5, ls=":")
        if escape is not None and ax in axes:
            for a_, b_ in escape:
                if a_ < end and b_ > start:
                    ax.axvspan(a_ / 1e6, b_ / 1e6, color="#fdae6b",
                               alpha=0.15, lw=0)
    for nm, (a_, b_) in elements.items():
        if a_ < end and b_ > start:
            ax_up.text((a_ + b_) / 2e6, 1.0, nm, fontsize=5.5, ha="center",
                       va="bottom", transform=ax_up.get_xaxis_transform())
    for ax in [ax_up, ax_t, ax_dn, ax_r] + axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(axis="x", length=0)
    if axes:
        axes[-1].set_xlabel("chrX, Mb", fontsize=6.5)
    return fig


#: The two six-panel figures' rows of concordance: (context in the tables,
#: label).
CONC_CONTEXTS = (("HCG", "HCG-HCG"), ("GCH", "GCH-GCH"),
                 ("HCG×GCH", "HCG-GCH"))


def _orient(t: pd.DataFrame, xi_hap: int, cols=("delta", "level")):
    """hap1-hap2 rows as Xi less Xa (``xi_hap`` 2: negated, the interval
    swapped)."""
    if (xi_hap or 1) != 2 or t.empty:
        return t
    t = t.copy()
    for c in cols:
        if c in t:
            lo, hi = t[f"{c}_lo"].copy(), t[f"{c}_hi"].copy()
            t[c], t[f"{c}_lo"], t[f"{c}_hi"] = -t[c], -hi, -lo
    return t


def level_track(allelic: pd.DataFrame, ctx: str, xi_hap: int, start: int,
                end: int, window: int, step: int, n_boot: int = 200,
                seed: int = 0) -> pd.DataFrame:
    """Along X, windows ``window`` wide slid ``step``: each copy's level of
    ``ctx`` at the phased SNPs inside (weighted by their reads) and log2 of
    Xi's over Xa's, with its 95% interval from resampling the SNPs."""
    t = allelic.sort_values("pos")
    pos = t["pos"].to_numpy(np.int64)
    h0 = xi_hap or 1
    l0 = t[f"{ctx}_hap{h0}"].to_numpy(float)
    l1 = t[f"{ctx}_hap{3 - h0}"].to_numpy(float)
    w0 = t[f"{ctx}_reads_hap{h0}"].to_numpy(float)
    w1 = t[f"{ctx}_reads_hap{3 - h0}"].to_numpy(float)
    ok = np.isfinite(l0) & np.isfinite(l1) & (w0 > 0) & (w1 > 0)
    pos, l0, l1, w0, w1 = pos[ok], l0[ok], l1[ok], w0[ok], w1[ok]
    rng = np.random.default_rng([seed, 31])
    rows = []
    for c in range(start + window // 2, end - window // 2 + 1, step):
        a_, b_ = np.searchsorted(pos, [c - window // 2, c + window // 2])
        m = b_ - a_
        if m < 3:
            continue
        s_ = slice(a_, b_)
        x0, x1 = (w0[s_] * l0[s_]).sum() / w0[s_].sum(), \
            (w1[s_] * l1[s_]).sum() / w1[s_].sum()
        W = rng.multinomial(m, np.full(m, 1.0 / m), size=n_boot)
        with np.errstate(invalid="ignore", divide="ignore"):
            b0 = (W @ (w0[s_] * l0[s_])) / (W @ w0[s_])
            b1 = (W @ (w1[s_] * l1[s_])) / (W @ w1[s_])
            r = np.log2(b0 / b1)
        r = r[np.isfinite(r)]
        lo, hi = (np.percentile(r, [2.5, 97.5]) if r.size else
                  (np.nan, np.nan))
        rows.append({"center": c, "snps": int(m), "xi": x0, "xa": x1,
                     "log2": np.log2(x0 / x1) if x0 > 0 and x1 > 0
                     else np.nan, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def _band(ax, x, y, lo, hi, col, label, lw=0.9, alpha=0.18):
    ax.fill_between(x, lo, hi, color=col, alpha=alpha, lw=0)
    ax.plot(x, y, color=col, lw=lw, label=label)


def plot_by_distance(ratio: pd.DataFrame, dist: pd.DataFrame | None,
                     labels, xi_hap: int, group: str, path):
    """Against the contacts' distance, the copies compared, X (red) against
    the autosomes' copies (grey), each with its 95% interval: a, log2 of
    their P(s); b, c, log2 of the ends' HCG and GCH level; d-f, ΔC less ΔC,
    HCG-HCG, GCH-GCH, HCG at one end and GCH at the other."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.6),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    ax = axes[0, 0]
    for sname, col in (("chrX", _COLS[0]), ("autosomes", "0.45")):
        t = ratio[(ratio["set"] == sname) & (ratio["log10_lo"] >= 4)
                  & (ratio[[f"contacts_{lab}" for lab in labels]]
                     .min(axis=1) >= 20)]
        if len(t):
            _band(ax, 10 ** ((t["log10_lo"] + t["log10_hi"]) / 2),
                  t["log2_ratio"], t["ci_lo"], t["ci_hi"], col, sname)
    ax.set_title(f"a  contacts, log2 P(s) {labels[0]} / {labels[1]}",
                 loc="left", fontsize=6.5)
    panels = [(axes[0, 1], "HCG", "level", "b  HCG level of the ends, log2"),
              (axes[1, 0], "GCH", "level", "c  GCH level of the ends, log2"),
              (axes[1, 1], "HCG", "delta", "d  HCG-HCG ΔC"),
              (axes[2, 0], "GCH", "delta", "e  GCH-GCH ΔC"),
              (axes[2, 1], "HCG×GCH", "delta", "f  HCG-GCH ΔC")]
    for ax, ctx, col_, ttl in panels:
        t = (dist[(dist["context"].astype(str) == ctx)
                  & (dist["haplotype"] == "hap1-hap2")]
             if dist is not None else pd.DataFrame())
        t = _orient(t, xi_hap)
        drew = False
        for sname, col in (("chrX", _COLS[0]), ("autosomes", "0.45")):
            u = t[t["set"] == sname].sort_values("dist_lo") if len(t) \
                else t
            u = u[u[col_].notna()] if len(u) else u
            if len(u):
                _band(ax, np.sqrt(u["dist_lo"] * u["dist_hi"]), u[col_],
                      u[f"{col_}_lo"], u[f"{col_}_hi"], col, sname)
                drew = True
        if not drew:
            ax.text(0.5, 0.5, "not in --concordance-distance",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=6, color="0.4")
        what = (f"{labels[0]} / {labels[1]}" if col_ == "level"
                else f"{labels[0]} − {labels[1]}")
        ax.set_title(f"{ttl}, {what}", loc="left", fontsize=6.5)
    # one distance axis for all six: P(s) and the concordance's ranges
    xs = [10 ** float(ratio["log10_lo"].min()), 10 ** float(
        ratio["log10_hi"].max())] if len(ratio) else [1e3, 3e8]
    if dist is not None and len(dist):
        xs = [min(xs[0], float(dist["dist_lo"].min())),
              max(xs[1], float(dist["dist_hi"].max()))]
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xlim(max(xs[0], 1e3), xs[1])
        ax.axhline(0, color="0.6", lw=0.4)
        ax.set_xlabel("distance between the contact's ends, bp", fontsize=6)
        ax.tick_params(labelsize=5.5)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=5, frameon=False)
    fig.suptitle(f"{group}: {labels[0]} against {labels[1]} by distance "
                 f"(95% intervals over cells)", fontsize=7, y=0.995)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_along_x(prof: pd.DataFrame, levels: dict, conc, labels,
                 xi_hap: int, elements: dict, group: str, path,
                 x_range=None):
    """Along X, the copies compared: a, contacts crossing each position,
    log2 observed over expected, each copy and their difference; b, c, log2
    of the copies' HCG and GCH level at the phased SNPs (``levels``,
    :func:`level_track`); d-f, each copy's ΔC and their difference,
    HCG-HCG, GCH-GCH, HCG at one end and GCH at the other (``conc``,
    chrx_haplotype_tracks); each with its 95% interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.6),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    ax = axes[0, 0]
    if len(prof):
        x = prof["pos"] / 1e6
        for k, lab in enumerate(labels):
            ax.plot(x, prof[f"log2_obs_exp_{lab}"], color=_COLS[k], lw=0.8,
                    label=lab)
        ax.plot(x, prof["difference"], color="black", lw=0.9,
                label=f"{labels[0]} − {labels[1]}")
    ax.set_title("a  contacts crossing each position, log2 obs/exp",
                 loc="left", fontsize=6.5)
    for ax, ctx, ttl in ((axes[0, 1], "HCG", "b  HCG level"),
                         (axes[1, 0], "GCH", "c  GCH level")):
        t = levels.get(ctx)
        if t is not None and len(t):
            _band(ax, t["center"] / 1e6, t["log2"], t["lo"], t["hi"],
                  "black", f"log2 {labels[0]} / {labels[1]}")
        else:
            ax.text(0.5, 0.5, "no --allelic-snps", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6, color="0.4")
        ax.set_title(f"{ttl}, log2 {labels[0]} / {labels[1]} (phased SNPs)",
                     loc="left", fontsize=6.5)
    for ax, (ctx, nm), ttl in zip((axes[1, 1], axes[2, 0], axes[2, 1]),
                                  CONC_CONTEXTS, ("d", "e", "f")):
        t = (conc[conc["context"].astype(str) == ctx] if conc is not None
             else pd.DataFrame())
        drew = False
        if len(t):
            mid_ = (t["bin_start"] + t["bin_end"]) / 2
            t = t.assign(center=t["center"].fillna(mid_) if "center" in t
                         else mid_)
            h0 = xi_hap or 1
            for hp, col, lab, lw in ((f"hap{h0}", _COLS[0], labels[0], 0.7),
                                     (f"hap{3 - h0}", _COLS[1], labels[1],
                                      0.7),
                                     ("hap1-hap2", "black",
                                      f"{labels[0]} − {labels[1]}", 0.9)):
                u = t[t["haplotype"] == hp].sort_values("center")
                if hp == "hap1-hap2":
                    u = _orient(u, xi_hap, cols=("delta",))
                if len(u):
                    _band(ax, u["center"] / 1e6, u["delta"], u["delta_lo"],
                          u["delta_hi"], col, lab, lw=lw,
                          alpha=0.12 if col != "black" else 0.2)
                    drew = True
            w = (t["bin_end"] - t["bin_start"]).max()
            ax.set_title(f"{ttl}  {nm} ΔC ({w / 1e6:g} Mb windows)",
                         loc="left", fontsize=6.5)
        if not drew:
            ax.text(0.5, 0.5, "not in --concordance-tracks",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=6, color="0.4")
            ax.set_title(f"{ttl}  {nm} ΔC", loc="left", fontsize=6.5)
    for ax in axes.ravel():
        ax.axhline(0, color="0.6", lw=0.4)
        if x_range:
            ax.set_xlim(x_range[0] / 1e6, x_range[1] / 1e6)
        for nm, (a_, b_) in elements.items():
            ax.axvline((a_ + b_) / 2e6, color="0.5", lw=0.4, ls=":")
            ax.text((a_ + b_) / 2e6, 1.0, nm, fontsize=4.5, ha="center",
                    va="bottom", transform=ax.get_xaxis_transform())
        ax.set_xlabel("chrX, Mb", fontsize=6)
        ax.tick_params(labelsize=5.5)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=4.8, frameon=False, loc="lower left")
    fig.suptitle(f"{group}: {labels[0]} against {labels[1]} along X "
                 f"(95% intervals)", fontsize=7, y=0.995)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def gene_level_table(allelic: pd.DataFrame, gene_iv: dict, contexts,
                     xi_hap: int, labels, n_boot: int = 1000,
                     seed: int = 0) -> pd.DataFrame:
    """Per context and class of genes (``gene_iv``) each copy's level at
    the phased SNPs inside, weighted by their reads, and the first copy
    less the second with its 95% interval from resampling the SNPs; and that
    difference in the escape genes less in the inactivated ones."""
    rng = np.random.default_rng([seed, 41])
    pos = allelic["pos"].to_numpy(np.int64)
    h0 = xi_hap or 1
    rows = []
    for ctx in contexts:
        if f"{ctx}_hap1" not in allelic:
            continue
        l0 = allelic[f"{ctx}_hap{h0}"].to_numpy(float)
        l1 = allelic[f"{ctx}_hap{3 - h0}"].to_numpy(float)
        w0 = allelic[f"{ctx}_reads_hap{h0}"].to_numpy(float)
        w1 = allelic[f"{ctx}_reads_hap{3 - h0}"].to_numpy(float)
        ok = np.isfinite(l0) & np.isfinite(l1) & (w0 > 0) & (w1 > 0)
        boot, point = {}, {}
        for gname, iv in gene_iv.items():
            m = ok & in_intervals(pos, iv)
            n = int(m.sum())
            r = {"context": ctx, "class": gname, "snps": n}
            if n >= 3:
                a0, a1 = w0[m] * l0[m], w1[m] * l1[m]
                x0, x1 = a0.sum() / w0[m].sum(), a1.sum() / w1[m].sum()
                W = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
                with np.errstate(invalid="ignore", divide="ignore"):
                    bd = (W @ a0) / (W @ w0[m]) - (W @ a1) / (W @ w1[m])
                boot[gname], point[gname] = bd, x0 - x1
                lo, hi = np.nanpercentile(bd, [2.5, 97.5])
                r.update({labels[0]: x0, labels[1]: x1, "difference": x0 - x1,
                          "ci_lo": lo, "ci_hi": hi})
            rows.append(r)
        if all(c in boot for c in GENE_CLASSES):
            bd = boot[GENE_CLASSES[0]] - boot[GENE_CLASSES[1]]
            lo, hi = np.nanpercentile(bd, [2.5, 97.5])
            rows.append({"context": ctx, "class": GENE_CONTRAST,
                         "difference": point[GENE_CLASSES[0]]
                         - point[GENE_CLASSES[1]], "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


#: The windows' 3D measures, each a copy less the other: (column, name).
WINDOW_3D = (("d_share_far", "share of contacts ≥ 1 Mb"),
             ("d_share_tad", "share 0.1–1 Mb"),
             ("d_partners", "partners' compartment"),
             ("d_crossing", "crossing, log2 obs/exp"))


def window_table(lo, hi, cp, geo: Geometry, x_end: int, size: int,
                 allelic, contexts, xi_hap: int, prof, x_name: str, escape,
                 labels, min_ends: int = 50) -> pd.DataFrame:
    """Along X outside PAR1, windows ``size`` wide side by side: per copy
    the ends inside of its contacts --min-distance or more apart, the
    share of them 0.1-1 Mb and 1 Mb or more apart (with ``min_ends`` or
    more), the compartment of their partners 2 Mb or more away
    (--compartments), the contacts crossing the window's middle (``prof``,
    the nearest position), and per context the level at the phased SNPs
    inside; each first copy less the second ('d_'), and the bp of escape
    genes inside."""
    edges = np.arange(geo.x_lo, x_end + 1, size, dtype=np.int64)
    nb = edges.size - 1
    if nb < 1:
        return pd.DataFrame()
    d = hi - lo
    keep = d >= geo.min_d
    ends = np.concatenate([lo[keep], hi[keep]])
    oth = np.concatenate([hi[keep], lo[keep]])
    dd = np.concatenate([d[keep], d[keep]])
    cc = np.concatenate([cp[keep], cp[keep]])
    b = np.searchsorted(edges, ends, "right") - 1
    inw = (b >= 0) & (b < nb)
    mid = (edges[:-1] + edges[1:]) // 2
    t = pd.DataFrame({"start": edges[:-1], "end": edges[1:]})
    for k, lab in enumerate(labels):
        m = inw & (cc == k)
        n_ = np.bincount(b[m], minlength=nb).astype(float)
        ok = n_ >= min_ends
        t[f"ends_{lab}"] = n_
        for col, a_, b_ in (("share_far", 1_000_000, np.inf),
                            ("share_tad", 100_000, 1_000_000)):
            c_ = np.bincount(b[m & (dd >= a_) & (dd < b_)], minlength=nb)
            with np.errstate(invalid="ignore", divide="ignore"):
                t[f"{col}_{lab}"] = np.where(ok, c_ / n_, np.nan)
        if geo.eig is not None:
            mm = m & (dd >= COMPARTMENT_MIN_DISTANCE)
            ev = geo.eig.at(np.full(int(mm.sum()), x_name), oth[mm])
            fin = np.isfinite(ev)
            s_ = np.bincount(b[mm][fin], ev[fin], nb)
            c_ = np.bincount(b[mm][fin], minlength=nb)
            with np.errstate(invalid="ignore", divide="ignore"):
                t[f"partners_{lab}"] = np.where(c_ >= max(10, min_ends // 5),
                                                s_ / c_, np.nan)
        if prof is not None and len(prof):
            gp = prof["pos"].to_numpy(np.int64)
            j = np.abs(gp[None, :] - mid[:, None]).argmin(1)
            near = np.abs(gp[j] - mid) <= size // 2
            t[f"crossing_{lab}"] = np.where(
                near, prof[f"log2_obs_exp_{lab}"].to_numpy(float)[j], np.nan)
        if allelic is not None and len(allelic):
            h = (xi_hap or 1) if k == 0 else 3 - (xi_hap or 1)
            pos = allelic["pos"].to_numpy(np.int64)
            for c in contexts:
                if f"{c}_hap{h}" in allelic:
                    t[f"{c}_{lab}"] = _binned(
                        pos, allelic[f"{c}_hap{h}"].to_numpy(float),
                        allelic[f"{c}_reads_hap{h}"].to_numpy(float), edges)
    for meas in ("share_far", "share_tad", "partners", "crossing", *contexts):
        a_, b_ = f"{meas}_{labels[0]}", f"{meas}_{labels[1]}"
        if a_ in t and b_ in t:
            t[f"d_{meas}"] = t[a_] - t[b_]
    esc = np.zeros(nb)
    if escape is not None and len(escape):
        for e0, e1 in escape:
            esc += np.clip(np.minimum(edges[1:], e1)
                           - np.maximum(edges[:-1], e0), 0, None)
    t["escape_bp"] = esc
    return t


def _shift_test(x, y, min_shift: int):
    """Spearman's rho of the windows where both series are finite, and its
    p against the series slid against each other along X, circularly, by
    each shift of ``min_shift`` windows or more -- which keeps either's
    autocorrelation: (rho, p, shifts, windows)."""
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    # a series without variation has no rank correlation
    if ok.sum() < 8 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan, np.nan, 0, int(ok.sum())
    rho = float(spearmanr(x[ok], y[ok])[0])
    null = []
    for s in range(min_shift, x.size - min_shift + 1):
        xs = np.roll(x, s)
        o2 = np.isfinite(xs) & np.isfinite(y)
        if o2.sum() >= 8:
            null.append(spearmanr(xs[o2], y[o2])[0])
    null = np.abs(np.asarray(null, float))
    p = (1 + int((null >= abs(rho) - 1e-12).sum())) / (1 + null.size)
    return rho, p, int(null.size), int(ok.sum())


def _rank_residual(v, z) -> np.ndarray:
    """``v``'s ranks less their least-squares line on ``z``'s, over the
    windows where both are finite (NaN elsewhere)."""
    from scipy.stats import rankdata
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v) & np.isfinite(z)
    if ok.sum() < 3:
        return out
    rv, rz = rankdata(v[ok]), rankdata(z[ok])
    X = np.column_stack([rz, np.ones(rz.size)])
    out[ok] = rv - X @ np.linalg.lstsq(X, rv, rcond=None)[0]
    return out


def window_correlations(t: pd.DataFrame, contexts, labels=None,
                        min_shift: int = 2) -> pd.DataFrame:
    """Across the windows, each context's level difference against each 3D
    difference (WINDOW_3D): Spearman's rho with its p from circular shifts
    (:func:`_shift_test`); and with ``labels``, the same given the second
    copy's own level of the context -- both series' ranks residualised on
    it before the test.  The copies' differences can follow the active X
    alone: where it is the more methylated (gene bodies) it is the more
    folded into TADs, with the inactive X alike everywhere."""
    rows = []
    for ctx in contexts:
        if f"d_{ctx}" not in t:
            continue
        x = t[f"d_{ctx}"].to_numpy(float)
        zc = f"{ctx}_{labels[1]}" if labels else None
        z = t[zc].to_numpy(float) if zc and zc in t else None
        for col, nm in WINDOW_3D:
            if col not in t:
                continue
            y = t[col].to_numpy(float)
            rho, p, ns, n = _shift_test(x, y, min_shift)
            if not np.isfinite(rho):
                continue
            r = {"context": ctx, "measure_3d": nm, "column": col,
                 "windows": n, "rho": rho, "p_shift": p, "shifts": ns}
            if z is not None:
                if np.ptp(z[np.isfinite(z)]) > 0:
                    rp, pp, _, _ = _shift_test(_rank_residual(x, z),
                                               _rank_residual(y, z),
                                               min_shift)
                else:
                    rp, pp = rho, p
                r[f"rho_given_{labels[1]}_level"] = rp
                r[f"p_shift_given_{labels[1]}_level"] = pp
            rows.append(r)
    return pd.DataFrame(rows)


_GENE_COLS = {GENE_CLASSES[0]: "#e08214", GENE_CLASSES[1]: "#5e3c99"}


def plot_genes_windows(stats: pd.DataFrame, glev: pd.DataFrame,
                       wt: pd.DataFrame, corr: pd.DataFrame, labels,
                       group: str, path):
    """The copies' differences and the genes and windows: a, each
    context's level at the escape and the inactivated genes; b, the share
    of their contacts 0.1-1 Mb and 1 Mb or more apart (95% intervals); c-f,
    along X, windows' level difference against their 3D differences
    (rho, p from sliding the series along X)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2),
                             gridspec_kw={"hspace": 0.6, "wspace": 0.4})
    what = f"{labels[0]} − {labels[1]}"

    def pts(ax, cats, get, ttl, ylab):
        for j, gname in enumerate(GENE_CLASSES):
            for i, cat in enumerate(cats):
                r = get(cat, gname)
                if r is None or not np.isfinite(r[0]):
                    continue
                x = i + (j - 0.5) * 0.3
                ax.errorbar(x, r[0], yerr=[[r[0] - r[1]], [r[2] - r[0]]],
                            fmt="o", ms=3.5, color=_GENE_COLS[gname],
                            capsize=2, lw=0.9,
                            label=gname if i == 0 else None)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=6)
        ax.axhline(0, color="0.6", lw=0.4)
        ax.set_title(ttl, loc="left", fontsize=6.5)
        ax.set_ylabel(ylab, fontsize=6)
        ax.tick_params(labelsize=5.5)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=5, frameon=False)

    def lev(cat, gname):
        if glev is None or not len(glev):
            return None
        r = glev[(glev["context"] == cat) & (glev["class"] == gname)]
        if not len(r) or "difference" not in r:
            return None
        r = r.iloc[0]
        return r["difference"], r["ci_lo"], r["ci_hi"]
    cats = list(dict.fromkeys(glev["context"])) if glev is not None and \
        len(glev) else []
    pts(axes[0, 0], cats, lev, f"a  level at the genes, {what}",
        "level difference")

    def ps(cat, gname):
        r = stats[(stats["measure"] == f"P(s): share {cat}")
                  & (stats["set"] == f"chrX {gname}")]
        if not len(r):
            return None
        r = r.iloc[0]
        return r["difference"], r["ci_lo"], r["ci_hi"]
    pts(axes[0, 1], [nm for nm, _, _ in PS_CLASSES[:2]], ps,
        f"b  their contacts, P(s) share, {what}", "share difference")
    panels = [(ctx, col, nm) for ctx in ("HCG", "GCH")
              for col, nm in WINDOW_3D if col in ("d_share_far", "d_partners")]
    for ax, (ctx, col, nm), ttl in zip(
            [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]], panels,
            "cdef"):
        if wt is None or not len(wt) or f"d_{ctx}" not in wt or col not in wt:
            ax.text(0.5, 0.5, "no windows", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6, color="0.4")
            ax.set_title(f"{ttl}  {ctx}: {nm}", loc="left", fontsize=6.5)
            continue
        esc = wt["escape_bp"] > 0
        for m, c_, lab in ((~esc, "0.3", "windows"),
                           (esc, _GENE_COLS[GENE_CLASSES[0]],
                            "with escape genes")):
            ax.scatter(wt.loc[m, f"d_{ctx}"], wt.loc[m, col], s=7, color=c_,
                       lw=0, label=lab)
        r = corr[(corr["context"] == ctx) & (corr["column"] == col)] if \
            corr is not None and len(corr) else pd.DataFrame()
        sub = (f"ρ = {r['rho'].iloc[0]:+.2f}, p = {r['p_shift'].iloc[0]:.2g}"
               if len(r) else "")
        gc = f"rho_given_{labels[1]}_level"
        if len(r) and gc in r and np.isfinite(r[gc].iloc[0]):
            sub += (f"; given {labels[1]}'s level ρ = {r[gc].iloc[0]:+.2f}, "
                    f"p = {r[f'p_shift_given_{labels[1]}_level'].iloc[0]:.2g}")
        ax.set_title(f"{ttl}  {ctx} level vs {nm}\n{sub}", loc="left",
                     fontsize=5.6)
        ax.set_xlabel(f"{ctx} level, {what}", fontsize=6)
        ax.set_ylabel(f"{nm}, {what}", fontsize=6)
        ax.axhline(0, color="0.7", lw=0.4)
        ax.axvline(0, color="0.7", lw=0.4)
        ax.tick_params(labelsize=5.5)
        ax.legend(fontsize=4.8, frameon=False)
    fig.suptitle(f"{group}: {what} at the escape and the inactivated genes, "
                 f"and in windows along X", fontsize=7, y=0.995)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


#: The measures the per-cell figures follow, in order, where present.
_KEY_MEASURES = ("P(s): share 0.1–1 Mb", "P(s): share ≥ 1 Mb",
                 "P(s): share ≥ 10 Mb", "A/B compartments: ends' correlation",
                 "crossing DXZ4: log2 obs/exp")


def plot_xci_strata(cells: pd.DataFrame, summ: dict, strata: pd.DataFrame,
                    trend: pd.DataFrame, labels, group: str, path):
    """a, the cells' XCI strength (their inactive X less their active one at
    the islands) against what their sites' noise alone would spread; b-d,
    per stratum of it the copies' difference in folding, X (red) and the
    autosomes' copies (grey), 95% intervals, with the slope's permutation
    p."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    meas = [m for m in _KEY_MEASURES[:4] if m in set(strata["measure"])][:3]
    fig, axes = plt.subplots(1, 1 + len(meas),
                             figsize=(2.6 * (1 + len(meas)), 2.5),
                             squeeze=False)
    ax = axes[0, 0]
    s = cells["strength"].to_numpy(float)
    s = s[np.isfinite(s)]
    bins = np.linspace(np.nanmin(s) - 0.05, np.nanmax(s) + 0.05, 31)
    ax.hist(s, bins=bins, color="0.6")
    if summ:
        xs = np.linspace(bins[0], bins[-1], 200)
        e = cells["noise_sd"].to_numpy(float)
        e = e[np.isfinite(e) & (e > 0)]
        dens = np.mean([norm.pdf(xs, summ["mean"], v) for v in e], axis=0)
        ax.plot(xs, dens * s.size * (bins[1] - bins[0]), color="black",
                lw=0.9, label="sites' noise alone")
        ax.legend(fontsize=5, frameon=False)
        ax.set_title(f"a  XCI strength per cell\nSD {summ['sd']:.3f}, noise "
                     f"{summ['noise_sd']:.3f}, excess "
                     f"{summ['excess_sd']:.3f}", loc="left", fontsize=6.2)
    ax.axvline(0, color="0.5", lw=0.5)
    ax.set_xlabel(f"{labels[0]} − {labels[1]} at islands", fontsize=6)
    ax.tick_params(labelsize=5.5)
    for ax, m, ttl in zip(axes[0, 1:], meas, "bcd"):
        sub_ = []
        for sname, col in (("chrX", _COLS[0]), ("autosomes", "0.45")):
            t = strata[(strata["measure"] == m) & (strata["set"] == sname)]
            if not len(t):
                continue
            ax.errorbar(t["strength_mean"], t["difference"],
                        yerr=[t["difference"] - t["ci_lo"],
                              t["ci_hi"] - t["difference"]],
                        fmt="o-", ms=3, lw=0.8, color=col, capsize=1.5,
                        label=sname)
            r = (trend[(trend["measure"] == m) & (trend["set"] == sname)]
                 if len(trend) else trend)
            if len(r):
                sub_.append(f"{sname} slope {r['slope'].iloc[0]:+.2f}, "
                            f"p {r['p_perm'].iloc[0]:.2g}")
        ax.axhline(0, color="0.6", lw=0.4)
        ax.set_title(f"{ttl}  {m}\n" + "; ".join(sub_), loc="left",
                     fontsize=5.4)
        ax.set_xlabel("stratum's mean XCI strength", fontsize=6)
        ax.set_ylabel(f"{labels[0]} − {labels[1]}", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.legend(fontsize=5, frameon=False)
    fig.suptitle(f"{group}: cells grouped by their own XCI strength",
                 fontsize=7, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_bulk_vs_cells(st_cells: pd.DataFrame, st_hap: pd.DataFrame,
                       calls: pd.DataFrame, group: str, path):
    """A line whose cells silence either X: a, each cell's hap1 less hap2
    at the islands -- one or the other haplotype silenced; then per
    measure the copies' difference on X as bulk data sees it (hap1 against
    hap2) and with each cell's own inactive X (95% intervals)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meas = [m for m in _KEY_MEASURES
            if ((st_cells["measure"] == m) & (st_cells["set"] == "chrX"))
            .any()]
    fig, axes = plt.subplots(1, 1 + len(meas),
                             figsize=(2.2 * (1 + len(meas)), 2.4),
                             squeeze=False)
    ax = axes[0, 0]
    v = calls["hap1_minus_hap2"].to_numpy(float) if len(calls) else []
    ax.hist(v, bins=30, color="0.6")
    ax.axvline(0, color="0.5", lw=0.5)
    n1 = int((np.asarray(v) > 0).sum())
    ax.set_title(f"a  each cell's X: hap1 − hap2\nhap1 silenced in {n1}, "
                 f"hap2 in {len(v) - n1}", loc="left", fontsize=6.2)
    ax.set_xlabel("methylation at X's islands", fontsize=6)
    ax.tick_params(labelsize=5.5)
    for ax, m in zip(axes[0, 1:], meas):
        for i, (t, lab, col) in enumerate(((st_hap, "hap1 − hap2\n(bulk)",
                                            "0.45"),
                                           (st_cells, "Xi − Xa\n(per cell)",
                                            _COLS[0]))):
            r = t[(t["measure"] == m) & (t["set"] == "chrX")]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.errorbar(i, r["difference"],
                        yerr=[[r["difference"] - r["ci_lo"]],
                              [r["ci_hi"] - r["difference"]]],
                        fmt="o", ms=4, color=col, capsize=2, lw=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["hap1 − hap2\n(bulk)", "Xi − Xa\n(per cell)"],
                           fontsize=5.5)
        ax.set_xlim(-0.6, 1.6)
        ax.axhline(0, color="0.6", lw=0.4)
        ax.set_title(m, loc="left", fontsize=5.8)
        ax.tick_params(labelsize=5.5)
    fig.suptitle(f"{group}: as bulk sees it, and cell by cell", fontsize=7,
                 y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


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
    ap.add_argument("--haplotypes", required=True,
                    help="allelic_imbalance.py --haplotype-dir: "
                         "{cell}.haplotypes.parquet.")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-column", default="cell_name")
    ap.add_argument("--group-column", default="group")
    ap.add_argument("--groups", default=None,
                    help="Lines to analyse (comma-separated; default all).")
    ap.add_argument("--xi", action="append", default=None,
                    metavar="GROUP=1|2|cells",
                    help="A line's inactive X: the haplotype, the same in "
                         "every cell (clonal), or 'cells' -- each cell's own "
                         "from --xi-cells (repeatable).  Without: hap1 "
                         "against hap2.")
    ap.add_argument("--xi-cells", default=None,
                    help="allelic_imbalance.py's allelic_cells.tsv, for "
                         "--xi GROUP=cells.")
    ap.add_argument("--xi-context", default="HCG_island",
                    help="The context of --xi-cells that names a cell's "
                         "inactive X: its more methylated haplotype.")
    ap.add_argument("--min-cell-sites", type=int, default=20,
                    help="Sites a cell needs on each haplotype for its "
                         "call.")
    ap.add_argument("--boundaries", default=None,
                    help="BED of TAD boundaries (their midpoints), e.g. "
                         "CTCF-anchored ones of bulk Hi-C.")
    ap.add_argument("--genome", default="hg38",
                    help="For PAR1 and the inactive X's elements (hg38).")
    ap.add_argument("--chrx-par", default="auto", metavar="auto|none|S-E")
    ap.add_argument("--xi-elements", default="auto",
                    metavar="auto|none|BED",
                    help="The inactive X's elements: DXZ4 (the megadomain "
                         "boundary) and the superloops' anchors.  'auto' "
                         "(hg38): XIST, FIRRE, ICCE, DXZ4.")
    ap.add_argument("--min-distance", type=int, default=20_000,
                    help="P(s)'s shares: of the contacts this far apart.")
    ap.add_argument("--profile-window", type=int, default=0,
                    help="The crossing profile: count a contact at a "
                         "position only with both ends within this of it "
                         "(0, default: anywhere on X).")
    ap.add_argument("--profile-step", type=int, default=1_000_000)
    ap.add_argument("--profile-min-distance", type=int, default=1_000_000,
                    help="The crossing profile: contacts this far apart.")
    ap.add_argument("--profile-exclude", type=int, default=20_000_000,
                    help="DXZ4's rank: against the positions this far from "
                         "it or more.")
    ap.add_argument("--boundary-window", type=int, default=300_000)
    ap.add_argument("--boundary-min-distance", type=int, default=20_000)
    ap.add_argument("--boundary-flank", type=int, default=500_000)
    ap.add_argument("--loop-window", type=int, default=250_000)
    ap.add_argument("--loop-step", type=int, default=250_000)
    ap.add_argument("--loop-exclude", type=int, default=2_000_000,
                    help="Slides this close to none are not the "
                         "superloops' background.")
    ap.add_argument("--compartments", default=None,
                    help="A bulk compartment track (bedGraph; positive A), "
                         "e.g. GM12878's eigenvector: the correlation of a "
                         "long-range contact's two ends per copy, and the "
                         "loci's compartment track.")
    ap.add_argument("--locus", action="append", default=None,
                    metavar="NAME=chrX:START-END",
                    help="A locus to draw as a genome browser would, the "
                         "two copies against each other "
                         "(haplotype_locus_<line>_<name>; repeatable).  "
                         "Default: all of X and DXZ4 +/- 10 Mb; 'none'.")
    ap.add_argument("--tads", default=None,
                    help="BED of the TAD boundaries the loci show (default: "
                         "--boundaries).")
    ap.add_argument("--chrx-escape", default="auto",
                    metavar="auto|none|BED",
                    help="The escape genes, shaded in the loci's tracks.")
    ap.add_argument("--allelic-snps", default=None,
                    help="allelic_imbalance.py's allelic_snps.tsv: each "
                         "copy's level per --track-contexts along the "
                         "loci.")
    ap.add_argument("--track-contexts", default="HCG,GCH",
                    help="The contexts of --allelic-snps to draw.")
    ap.add_argument("--concordance-tracks", default=None,
                    help="annotation_class_concordance.py's "
                         "chrx_haplotype_tracks.tsv: each copy's ΔC along "
                         "the loci.")
    ap.add_argument("--concordance-distance", default=None,
                    help="annotation_class_concordance.py's "
                         "chrx_haplotype_distance.tsv: each copy's ΔC and "
                         "level by the contacts' distance "
                         "(haplotype_by_distance_<line>).")
    ap.add_argument("--level-window", type=int, default=2_000_000,
                    help="haplotype_along_x_<line>: the windows of the "
                         "phased SNPs' levels.")
    ap.add_argument("--level-step", type=int, default=500_000)
    ap.add_argument("--map-smooth", type=float, default=1.0,
                    help="The loci's contact maps: Gaussian smoothing, in "
                         "bins (0: none).")
    ap.add_argument("--genes", default=None,
                    help="Genes (GTF, genePred -- e.g. ncbiRefSeqCurated -- "
                         "or BED): X's escape genes (--chrx-escape) against "
                         "its inactivated ones, in each copy's P(s) and, "
                         "with --allelic-snps, its level "
                         "(haplotype_genes_<line>.tsv).")
    ap.add_argument("--gene-flank", type=int, default=10_000,
                    help="--genes: each gene this much wider on each side.")
    ap.add_argument("--element", action="append", default=None,
                    metavar="NAME=BED",
                    help="Repeatable: the contacts with an end in the BED's "
                         "intervals, per copy -- their P(s) shares and their "
                         "share of the copy's contacts -- on X outside PAR1 "
                         "and the escape genes and on the autosomes, e.g. "
                         "island=cpgIslands.bed promoter=PLS.bed.")
    ap.add_argument("--element-flank", type=int, default=100,
                    help="--element: bp each side of an interval an end may "
                         "lie and count as in it (100, about half a read).")
    ap.add_argument("--gene-contexts",
                    default="HCG,GCH,HCG_island,GCH_island,HCG_promoter,"
                            "GCH_promoter",
                    help="--genes: the contexts of --allelic-snps compared "
                         "at the genes (those it has).")
    ap.add_argument("--windows", type=int, default=2_000_000,
                    help="Along X, windows this wide side by side: each "
                         "copy's 3D measures against its level at the "
                         "phased SNPs, for a line with one inactive X "
                         "(haplotype_windows_<line>.tsv); 0: none.")
    ap.add_argument("--xci-strata", type=int, default=4,
                    help="A line whose cells silence the same haplotype "
                         "(--xi GROUP=1|2), its cells grouped by their own "
                         "XCI strength -- Xi less Xa at --xi-context, from "
                         "--xi-cells -- into this many quantile strata: does "
                         "the copies' folding follow (haplotype_xci_strata_"
                         "<line>)?  0: none.  Not for --xi GROUP=cells, "
                         "whose calls come from the same signal and would "
                         "misorient the weakest cells.")
    ap.add_argument("--n-perm", type=int, default=200,
                    help="--xci-strata: permutations for the slopes' p.")
    ap.add_argument("--window-min-ends", type=int, default=50,
                    help="--windows: the contact ends a copy needs in a "
                         "window for its shares.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-format", default="pdf")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    genome = None if str(args.genome).lower() == "none" else args.genome
    try:
        par = parse_par(args.chrx_par, genome)
    except ValueError as e:
        ap.error(f"--chrx-par {e}")
    elements = parse_elements(args.xi_elements, genome)
    xi = _pairs(args.xi)
    for g, v in xi.items():
        if v not in ("1", "2", "cells"):
            ap.error(f"--xi {g}={v}: 1, 2 or cells")
        if v == "cells" and not args.xi_cells:
            ap.error(f"--xi {g}=cells needs --xi-cells")
    samples = pd.read_csv(manifest_text(args.samples), sep="\t", dtype=str)
    for col in (args.cell_column, args.group_column):
        if col not in samples:
            sys.exit(f"--samples has no column {col!r}")
    groups = ([g.strip() for g in args.groups.split(",") if g.strip()]
              if args.groups else list(dict.fromkeys(
                  samples[args.group_column].dropna())))
    hdir = Path(args.haplotypes)
    calls = pd.DataFrame()
    if args.xi_cells:
        try:
            calls = cell_xi_table(args.xi_cells, args.xi_context,
                                  args.min_cell_sites)
        except ValueError as e:
            sys.exit(f"--xi-cells {e} (--xi-context)")
    cells_xi = dict(zip(calls["cell"], calls["xi_hap"])) if len(calls) \
        else {}
    data: dict = {}
    x_len, x_name = 0, "chrX"
    for g in groups:
        cs = samples.loc[samples[args.group_column] == g, args.cell_column]
        got = {}
        for c in cs:
            t = read_pairs(hdir / f"{c}.haplotypes.parquet")
            if t is not None:
                got[c] = t
                onx = t["chrom"].map(chrom_key).eq("X")
                if onx.any():
                    x_len = max(x_len, int(t.loc[onx, "hi"].max()) + 1)
                    x_name = str(t.loc[onx, "chrom"].iloc[0])
        if not got:
            logger.warning(f"{g}: none of its {len(cs)} cells has "
                           f"{hdir}/<cell>.haplotypes.parquet; skipped")
            continue
        logger.info(f"{g}: {len(got)} of {len(cs)} cells with haplotype-"
                    f"tagged contacts")
        data[g] = got
    if not data:
        sys.exit(f"no cell of --samples has a file in {hdir} (written by "
                 f"allelic_imbalance.py --haplotype-dir)")
    bnd = read_boundaries(args.boundaries) if args.boundaries else {}
    eig = Eigen(args.compartments) if args.compartments else None
    geo = Geometry(x_len, par, elements, bnd, args, eigen=eig)
    try:
        loci = parse_loci(args.locus, elements, x_name, geo.x_lo, x_len)
    except ValueError as e:
        ap.error(str(e))
    tads = None
    tad_path = args.tads or args.boundaries
    if loci and tad_path:
        tb = read_boundaries(tad_path)
        tads = next((v for c, v in tb.items() if chrom_key(c) == "X"), None)
    escape = parse_escape(args.chrx_escape, genome, par)
    geo.escape_iv = escape
    geo.elem_iv = read_elements(args.element, int(args.element_flank))
    if geo.elem_iv:
        logger.info("--element: " + ", ".join(
            f"{k} {sum(len(v) for v in per.values()):,} intervals"
            for k, per in geo.elem_iv.items()))
    if args.genes:
        if escape is None or not len(escape):
            logger.warning("--genes but no escape genes (--chrx-escape): no "
                           "escape against inactivated genes")
        else:
            geo.gene_iv = gene_classes(read_genes(args.genes), escape, par,
                                       args.gene_flank)
            logger.info("X's genes: " + ", ".join(
                f"{k} {len(v):,} regions, {int((v[:, 1] - v[:, 0]).sum()):,} "
                f"bp" for k, v in geo.gene_iv.items()))
    contexts = [c.strip() for c in args.track_contexts.split(",")
                if c.strip()]
    # the genes' contexts: those of --gene-contexts the SNP table has
    gene_ctx = []
    if args.allelic_snps and geo.gene_iv:
        head = set(pd.read_csv(args.allelic_snps, sep="\t", nrows=0).columns)
        gene_ctx = [c for c in (s.strip() for s in
                                args.gene_contexts.split(","))
                    if c and {f"{c}_ref", f"{c}_reads_ref"} <= head]
    allelic = (read_allelic(args.allelic_snps,
                            list(dict.fromkeys(contexts + gene_ctx)))
               if args.allelic_snps else None)
    conc = (pd.read_csv(args.concordance_tracks, sep="\t",
                        dtype={"group": str})
            if args.concordance_tracks else None)
    cdist = (pd.read_csv(args.concordance_distance, sep="\t",
                         dtype={"group": str})
             if args.concordance_distance else None)
    if geo.dxz4 is None:
        logger.warning("no DXZ4 among --xi-elements: no megadomain boundary")
    if not args.boundaries:
        logger.warning("no --boundaries: no TAD boundary insulation")
    rng = np.random.default_rng(args.seed)
    all_stats, all_ps, all_prof, all_bnd, all_ratio = [], [], [], [], []
    for g, got in data.items():
        mode = xi.get(g)
        labels = ("Xi", "Xa") if mode else ("hap1", "hap2")
        counts, pooled, names, counts_h = [], [], [], []
        n_drop = 0
        for c, t in got.items():
            if mode == "cells":
                h = cells_xi.get(c)
                if h is None:
                    n_drop += 1
                    continue
                # as bulk data would see the same cells: hap1 against hap2
                counts_h.append(cell_counts(t, 0, geo))
            else:
                h = int(mode) if mode else 0
            counts.append(cell_counts(t, h, geo))
            names.append(c)
            if loci or args.windows > 0:
                pooled.append(x_contacts(t, h, geo))
        if mode == "cells":
            logger.info(f"{g}: each cell's inactive X from {args.xi_cells} "
                        f"({args.xi_context}); {n_drop} cells without a call "
                        f"left out")
        if not counts:
            continue
        a = _stack(counts)
        n = len(counts)
        w = rng.multinomial(n, np.full(n, 1.0 / n),
                            size=args.n_boot).astype(float)
        st = group_stats(a, geo, w, labels)
        st.insert(0, "group", g)
        st.insert(1, "xi", mode or "none")
        st.insert(2, "cells", len(counts))
        st = st.rename(columns={labels[0]: "copy1", labels[1]: "copy2"})
        st.insert(st.columns.get_loc("copy1"), "copies",
                  f"{labels[0]}|{labels[1]}")
        all_stats.append(st)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(g))
        # a mosaic line: hap1 against hap2, as bulk sees it, beside each
        # cell's own inactive X
        if counts_h:
            sh = group_stats(_stack(counts_h), geo, w, ("hap1", "hap2"))
            sh.insert(0, "group", g)
            sh.insert(1, "xi", "haplotypes (bulk view)")
            sh.insert(2, "cells", len(counts_h))
            sh = sh.rename(columns={"hap1": "copy1", "hap2": "copy2"})
            sh.insert(sh.columns.get_loc("copy1"), "copies", "hap1|hap2")
            all_stats.append(sh)
            cg = calls[calls["cell"].isin(set(names))]
            cg.assign(group=g).to_csv(out / f"haplotype_xi_calls_{name}.tsv",
                                      sep="\t", index=False)
            plot_bulk_vs_cells(st, sh, cg, g,
                               out / f"haplotype_bulk_vs_cells_{name}."
                                     f"{args.plot_format}")
        # a line silencing one haplotype: its cells by their own strength
        if mode in ("1", "2") and args.xci_strata > 0 and len(calls):
            sign = 1.0 if mode == "1" else -1.0
            cx = calls.set_index("cell").reindex(names)
            strength = sign * cx["hap1_minus_hap2"].to_numpy(float)
            strata, trend = xci_strata(a, strength, geo, labels,
                                       args.xci_strata, args.n_boot,
                                       args.n_perm, rng)
            if len(strata):
                summ = strength_summary(strength,
                                        cx["noise_sd"].to_numpy(float))
                per_cell = pd.DataFrame({"cell": names,
                                         "strength": strength,
                                         "noise_sd": cx["noise_sd"]
                                         .to_numpy(float)})
                for t_, f_ in ((strata, "strata"), (trend, "trend"),
                               (per_cell, "cells")):
                    t_.assign(group=g).to_csv(
                        out / f"haplotype_xci_{f_}_{name}.tsv", sep="\t",
                        index=False)
                plot_xci_strata(per_cell, summ, strata, trend, labels, g,
                                out / f"haplotype_xci_strata_{name}."
                                      f"{args.plot_format}")
                key = (trend[trend["set"].isin(["chrX", "autosomes"])]
                       if len(trend) else trend)
                logger.info(
                    f"{g}: XCI strength per cell"
                    + (f", mean {summ['mean']:+.3f}, SD {summ['sd']:.3f} "
                       f"against {summ['noise_sd']:.3f} from its sites "
                       f"alone (excess {summ['excess_sd']:.3f}); "
                       f"{summ['share_below_0']:.1%} of cells below 0 "
                       f"against {summ['share_below_0_noise']:.1%} expected "
                       f"of noise" if summ else "")
                    + ".  The copies' difference on the strata's strength:\n"
                    + (key.to_string(index=False,
                                     float_format=lambda v: f"{v:.3g}")
                       if len(key) else "no slope (one strength)"))
            else:
                logger.info(f"{g}: too few cells with an XCI strength for "
                            f"{args.xci_strata} strata")
        elif mode in ("1", "2") and args.xci_strata > 0:
            logger.info(f"{g}: no --xi-cells, so no XCI strata")
        ps = ps_table(a, labels).assign(group=g)
        ratio = ps_ratio_table(a, w, labels).assign(group=g)
        prof = profile_table(a, geo, labels).assign(group=g)
        bt = boundary_table(a, labels).assign(group=g)
        all_ps.append(ps)
        all_ratio.append(ratio)
        all_prof.append(prof)
        all_bnd.append(bt)
        if geo.dxz4 is not None:
            dif = prof["difference"].to_numpy()
            far = np.abs(geo.grid - geo.grid[geo.dxz4]) >= geo.prof_excl
            ok = far & np.isfinite(dif)
            if ok.any() and np.isfinite(dif[geo.dxz4]):
                rank = (1 + int((dif[ok] <= dif[geo.dxz4]).sum())) / (
                    1 + int(ok.sum()))
                logger.info(f"{g}: contacts crossing DXZ4, log2 obs/exp, "
                            f"{labels[0]} less {labels[1]}: "
                            f"{dif[geo.dxz4]:+.2f}; as low or lower at "
                            f"{rank:.1%} of X's {int(ok.sum())} positions "
                            f"{geo.prof_excl / 1e6:g} Mb or more away")
                st.loc[st["measure"].eq("crossing DXZ4: log2 obs/exp"),
                       "rank_along_x"] = rank
        plot_contacts(st.rename(columns={"copy1": labels[0],
                                         "copy2": labels[1]}),
                      ps, ratio, prof, bt, geo, elements, labels, g,
                      out / f"haplotype_contacts_{name}.{args.plot_format}")
        # the copies compared by distance and along X: contacts, levels,
        # concordance -- by haplotype only where the line has one inactive X
        one_x = mode not in ("cells",)
        xh = int(mode) if one_x and mode else 0
        plot_by_distance(
            ratio, (cdist[cdist["group"] == g] if one_x and cdist is not None
                    else None), labels, xh, g,
            out / f"haplotype_by_distance_{name}.{args.plot_format}")
        lv_ = {}
        if one_x and allelic is not None:
            al_g = allelic[allelic["group"] == g]
            for c_ in ("HCG", "GCH"):
                if len(al_g) and f"{c_}_hap1" in al_g:
                    lv_[c_] = level_track(al_g, c_, xh, geo.x_lo, x_len,
                                          args.level_window, args.level_step,
                                          seed=args.seed)
                    lv_[c_].assign(group=g, context=c_).to_csv(
                        out / f"haplotype_levels_{name}_{c_}.tsv", sep="\t",
                        index=False)
        plot_along_x(prof, lv_, (conc[conc["group"] == g] if one_x and conc
                                 is not None else None),
                     labels, xh, elements, g,
                     out / f"haplotype_along_x_{name}.{args.plot_format}",
                     x_range=(geo.x_lo, x_len))
        if pooled:
            lo_ = np.concatenate([x[0] for x in pooled])
            hi_ = np.concatenate([x[1] for x in pooled])
            cp_ = np.concatenate([x[2] for x in pooled])
        # the escape genes against the inactivated ones, and windows along
        # X: each copy's level against its folding -- per copy only where
        # the line has one inactive X
        if one_x and allelic is not None and (geo.gene_iv or
                                              args.windows > 0):
            al_g = allelic[allelic["group"] == g]
            gl = (gene_level_table(al_g, geo.gene_iv, gene_ctx or contexts,
                                   xh, labels, args.n_boot, args.seed)
                  if geo.gene_iv else pd.DataFrame())
            if len(gl):
                gl.assign(group=g).to_csv(out / f"haplotype_genes_{name}.tsv",
                                          sep="\t", index=False)
            wt, corr = pd.DataFrame(), pd.DataFrame()
            if args.windows > 0 and pooled:
                wt = window_table(lo_, hi_, cp_, geo, x_len, args.windows,
                                  al_g, contexts, xh, prof, x_name, escape,
                                  labels, args.window_min_ends)
                corr = window_correlations(wt, contexts, labels)
                wt.assign(group=g).to_csv(
                    out / f"haplotype_windows_{name}.tsv", sep="\t",
                    index=False)
                corr.assign(group=g).to_csv(
                    out / f"haplotype_windows_correlation_{name}.tsv",
                    sep="\t", index=False)
                if len(corr):
                    logger.info(f"{g}: windows along X, {labels[0]} less "
                                f"{labels[1]}, level against 3D:\n"
                                + corr.to_string(index=False, float_format=(
                                    lambda v: f"{v:.3g}")))
            plot_genes_windows(st.rename(columns={"copy1": labels[0],
                                                  "copy2": labels[1]}),
                               gl, wt, corr, labels, g,
                               out / f"haplotype_genes_windows_{name}."
                                     f"{args.plot_format}")
        if loci and pooled:
            # a polyclonal line's haplotypes are not its copies: the
            # SNPs' and the concordance's tracks are by haplotype
            one = mode not in ("cells",)
            al_ = (allelic[allelic["group"] == g] if one and allelic
                   is not None else None)
            co_ = (conc[conc["group"] == g] if one and conc is not None
                   else None)
            if not one and (allelic is not None or conc is not None):
                logger.info(f"{g}: each cell's own inactive X -- the levels "
                            f"and ΔC per haplotype are not per copy, so the "
                            f"loci show the contacts only")
            for nm, a_, b_ in loci:
                # HCG at one end, GCH at the other, where the tracks have it
                cx_ = contexts + [c for c, _ in CONC_CONTEXTS
                                  if c not in contexts and co_ is not None
                                  and (co_["context"] == c).any()]
                tr = locus_tracks(lo_, hi_, cp_, a_, b_, eig, x_name, al_,
                                  co_, cx_,
                                  int(mode) if one and mode else 0,
                                  args.map_smooth)
                ln = re.sub(r"[^A-Za-z0-9_.-]+", "_", nm)
                locus_table(tr, cx_, labels).to_csv(
                    out / f"haplotype_locus_{name}_{ln}.tsv", sep="\t",
                    index=False)
                map_table(tr, labels).to_csv(
                    out / f"haplotype_locus_{name}_{ln}_map.tsv.gz",
                    sep="\t", index=False)
                plot_locus(tr, nm, a_, b_, g, labels, cx_, elements,
                           tads, escape,
                           out / f"haplotype_locus_{name}_{ln}."
                                 f"{args.plot_format}")
    if not all_stats:
        sys.exit("no cell left to analyse")
    stats = pd.concat(all_stats, ignore_index=True)
    stats["q"] = fdr_bh(stats["p"].to_numpy())
    stats.to_csv(out / "haplotype_contacts.tsv", sep="\t", index=False)
    pd.concat(all_ps, ignore_index=True).to_csv(
        out / "haplotype_contacts_ps.tsv", sep="\t", index=False)
    pd.concat(all_ratio, ignore_index=True).to_csv(
        out / "haplotype_contacts_ps_ratio.tsv", sep="\t", index=False)
    pd.concat(all_prof, ignore_index=True).to_csv(
        out / "haplotype_contacts_profile.tsv", sep="\t", index=False)
    pd.concat(all_bnd, ignore_index=True).to_csv(
        out / "haplotype_contacts_boundaries.tsv", sep="\t", index=False)
    logger.info("copy 1 against copy 2 (haplotype_contacts.tsv):\n"
                + stats[["group", "measure", "set", "copies", "copy1",
                         "copy2", "difference", "ci_lo", "ci_hi", "q"]]
                .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
