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
    parse_elements, parse_escape, parse_par)

logger = logging.getLogger("haplotype_contacts")

#: P(s) bins, log10 of the distance: from 1 kb, a tenth of a decade each.
PS_EDGES = np.round(np.arange(3.0, 8.55, 0.1), 2)
#: P(s) is summarised by the share of contacts this far apart: (name,
#: from, to -- None: no end).
PS_CLASSES = (("0.1–1 Mb", 100_000, 1_000_000), ("≥ 1 Mb", 1_000_000, None),
              ("≥ 10 Mb", 10_000_000, None))
#: The sets of contacts: X outside PAR1, X within PAR1, the autosomes.
SETS = ("chrX", "chrX PAR1", "autosomes")
#: The boundary profile's offsets from the boundaries.
BOUNDARY_OFFSETS = np.arange(-1_000_000, 1_000_001, 100_000)
#: Without --profile-window, the crossing profile keeps this far from
#: either end of X (outside PAR1), where few contacts cross.
PROFILE_MARGIN = 5_000_000
#: Contacts this far apart or more show which compartments they join.
COMPARTMENT_MIN_DISTANCE = 2_000_000


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
    for meas in dict.fromkeys(m for m, _ in by):
        if (meas, "chrX") in by and (meas, "autosomes") in by:
            (dx, bx), (da, ba) = by[(meas, "chrX")], by[(meas, "autosomes")]
            rows.append(_row(meas, "chrX − autosomes", (np.nan, np.nan),
                             dx - da, bx - ba, norm, labels))
    return pd.DataFrame(rows)


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


def xi_of_cells(path, context: str, min_sites: int) -> dict:
    """Each cell's inactive X from allelic_cells.tsv: the haplotype the more
    methylated at ``context`` on X (outside PAR1 and the escape genes when
    the table has X's parts), ``min_sites`` on each; {cell: 1 | 2}."""
    c = pd.read_csv(path, sep="\t", dtype={"cell": str})
    x = c[c["chrom"].map(chrom_key).eq("X")]
    if "part" in x and x["part"].eq("rest").any():
        x = x[x["part"].eq("rest")]
    elif "part" in x:
        x = x[x["part"].eq("all")]
    need = [f"{context}_sites_hap1", f"{context}_sites_hap2",
            f"{context}_hap1_minus_hap2"]
    if not set(need) <= set(x.columns):
        sys.exit(f"--xi-cells {path}: no {context} columns (--xi-context)")
    x = x[(x[need[0]] >= min_sites) & (x[need[1]] >= min_sites)
          & x[need[2]].notna() & x[need[2]].ne(0)]
    return dict(zip(x["cell"], np.where(x[need[2]] > 0, 1, 2)))


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
    out = {"res": res, "edges": edges, "maps": maps, "tot": tot}
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
        for c in contexts:
            lv = np.full((2, nb), np.nan)
            for k in (0, 1):
                h = (xi_hap or 1) if k == 0 else 3 - (xi_hap or 1)
                lv[k] = _binned(pos, allelic[f"{c}_hap{h}"].to_numpy(float),
                                allelic[f"{c}_reads_hap{h}"].to_numpy(float),
                                edges)
            out[f"level_{c}"] = lv
    # concordance: each copy's ΔC in its context's own bins -- every
    # context of --track-contexts, empty where none was tagged
    if conc is not None:
        for c in contexts:
            t = conc[conc["context"].astype(str).str.upper() == c.upper()]
            t = t[(t["bin_end"] > start) & (t["bin_start"] < end)]
            rows = {}
            for k in (0, 1):
                h = (xi_hap or 1) if k == 0 else 3 - (xi_hap or 1)
                rows[k] = t[t["haplotype"] == f"hap{h}"].sort_values(
                    "bin_start")
            out[f"delta_{c}"] = rows
    return out


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
    # each bin's ΔC: that of the context's track bin holding its middle
    mid = (e[:-1] + e[1:]) // 2
    for c in contexts:
        if f"delta_{c}" not in tr:
            continue
        for k, lab in enumerate(labels):
            r = tr[f"delta_{c}"][k]
            v = np.full(mid.size, np.nan)
            if len(r):
                st_, en_ = r["bin_start"].to_numpy(), r["bin_end"].to_numpy()
                i = np.searchsorted(st_, mid, "right") - 1
                ok = (i >= 0) & (mid < en_[np.maximum(i, 0)])
                v[ok] = r["delta"].to_numpy()[i[ok]]
            t[f"{c}_dC_{lab}"] = v
    return t


def _triangle(ax, m, edges, up: bool, vmin, vmax, depth: float):
    """A contact map as a triangle along the genome, up or down, to
    ``depth`` of the distance (bp) between the bins."""
    x_, y_ = np.meshgrid(edges, edges)
    xs, ys = (x_ + y_) / 2e6, (x_ - y_) / 2e6
    if not up:
        ys = -ys
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = np.log10(m)
    lm = np.ma.masked_where(np.tril(np.ones(m.shape, bool), -1)
                            | ~np.isfinite(lm), lm)
    qm = ax.pcolormesh(xs, ys, lm, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                       shading="flat", rasterized=True)
    ax.set_ylim((0, depth / 2e6) if up else (-depth / 2e6, 0))
    ax.set_yticks([])
    for sp in ("left", "right", "top", "bottom"):
        ax.spines[sp].set_visible(False)
    return qm


def plot_locus(tr: dict, name: str, start: int, end: int, group: str,
               labels, contexts, elements: dict, tads, escape, path):
    """The two copies along one locus, as a genome browser shows them: each
    copy's contact map (copy 1 up, copy 2 down), the bulk TADs between them,
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
            w = (f", {(r['bin_end'] - r['bin_start']).iloc[0] / 1e6:g} Mb"
                 if len(r) else "")
            lines.append((f"delta_{c}", f"{c}-{c}\nΔC{w}"))
    hr = [3.2, 0.35, 3.2] + [1.0] * len(lines)
    fig = plt.figure(figsize=(7.2, 4.4 + 1.0 * len(lines)))
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
    cb = fig.colorbar(qm, ax=[ax_up, ax_dn], fraction=0.015, pad=0.01)
    cb.set_label("log10 share of the copy's contacts", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    mid = (tr["edges"][:-1] + tr["edges"][1:]) / 2e6
    axes = []
    for r, (key, ylab) in enumerate(lines):
        ax = fig.add_subplot(gs[3 + r], sharex=axes[0] if axes else None)
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
            for k, lab in enumerate(labels):
                t = tr[key][k]
                if not len(t):
                    continue
                x = np.r_[t["bin_start"].to_numpy(), t["bin_end"].iloc[-1]]
                ax.fill_between(x / 1e6, np.r_[t["delta_lo"],
                                               t["delta_lo"].iloc[-1]],
                                np.r_[t["delta_hi"], t["delta_hi"].iloc[-1]],
                                step="post", color=_COLS[k], alpha=0.15,
                                lw=0)
                ax.step(x / 1e6, np.r_[t["delta"], t["delta"].iloc[-1]],
                        where="post", color=_COLS[k], lw=0.9, label=lab)
            ax.axhline(0, color="0.6", lw=0.4)
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
    for ax in [ax_up, ax_t, ax_dn] + axes:
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
    for ax in [ax_up, ax_t, ax_dn] + axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(axis="x", length=0)
    if axes:
        axes[-1].set_xlabel("chrX, Mb", fontsize=6.5)
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
    ap.add_argument("--map-smooth", type=float, default=1.0,
                    help="The loci's contact maps: Gaussian smoothing, in "
                         "bins (0: none).")
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
    cells_xi = (xi_of_cells(args.xi_cells, args.xi_context,
                            args.min_cell_sites)
                if "cells" in xi.values() else {})
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
    escape = parse_escape(args.chrx_escape, genome, par) if loci else None
    contexts = [c.strip() for c in args.track_contexts.split(",")
                if c.strip()]
    allelic = (read_allelic(args.allelic_snps, contexts)
               if loci and args.allelic_snps else None)
    conc = (pd.read_csv(args.concordance_tracks, sep="\t",
                        dtype={"group": str})
            if loci and args.concordance_tracks else None)
    if geo.dxz4 is None:
        logger.warning("no DXZ4 among --xi-elements: no megadomain boundary")
    if not args.boundaries:
        logger.warning("no --boundaries: no TAD boundary insulation")
    rng = np.random.default_rng(args.seed)
    all_stats, all_ps, all_prof, all_bnd, all_ratio = [], [], [], [], []
    for g, got in data.items():
        mode = xi.get(g)
        labels = ("Xi", "Xa") if mode else ("hap1", "hap2")
        counts, pooled = [], []
        n_drop = 0
        for c, t in got.items():
            if mode == "cells":
                h = cells_xi.get(c)
                if h is None:
                    n_drop += 1
                    continue
            else:
                h = int(mode) if mode else 0
            counts.append(cell_counts(t, h, geo))
            if loci:
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
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(g))
        plot_contacts(st.rename(columns={"copy1": labels[0],
                                         "copy2": labels[1]}),
                      ps, ratio, prof, bt, geo, elements, labels, g,
                      out / f"haplotype_contacts_{name}.{args.plot_format}")
        if loci and pooled:
            lo_ = np.concatenate([x[0] for x in pooled])
            hi_ = np.concatenate([x[1] for x in pooled])
            cp_ = np.concatenate([x[2] for x in pooled])
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
                tr = locus_tracks(lo_, hi_, cp_, a_, b_, eig, x_name, al_,
                                  co_, contexts,
                                  int(mode) if one and mode else 0,
                                  args.map_smooth)
                ln = re.sub(r"[^A-Za-z0-9_.-]+", "_", nm)
                locus_table(tr, contexts, labels).to_csv(
                    out / f"haplotype_locus_{name}_{ln}.tsv", sep="\t",
                    index=False)
                plot_locus(tr, nm, a_, b_, g, labels, contexts, elements,
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
