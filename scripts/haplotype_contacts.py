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

Contacts with an end in PAR1 -- both copies active there -- are not X's;
those within PAR1 are a control set of P(s).  Figures:
haplotype_contacts_<line>.<fmt>.

    python scripts/haplotype_contacts.py \\
        --haplotypes output/haplotypes --samples samples.tsv \\
        --group-column cell_type --xi GM12878=1 \\
        --xi IMR90=cells --xi-cells allelic_imbalance/allelic_cells.tsv \\
        --boundaries gm12878_tad_boundaries_ctcf.hg38.bed \\
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
from nome_hic_concordance.xci import parse_elements, parse_par  # noqa: E402

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


class Geometry:
    """Where the measures look: the profile's positions along X, the
    boundaries and their offsets, the superloops' windows and their
    slides."""

    def __init__(self, x_len: int, par, elements: dict, boundaries: dict,
                 args):
        self.par = par
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
           "bnd_c": np.zeros((2, 2, BOUNDARY_OFFSETS.size))}
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
    x_len = 0
    for g in groups:
        cs = samples.loc[samples[args.group_column] == g, args.cell_column]
        got = {}
        for c in cs:
            t = read_pairs(hdir / f"{c}.haplotypes.parquet")
            if t is not None:
                got[c] = t
                xx = t.loc[t["chrom"].map(chrom_key).eq("X"), "hi"]
                if len(xx):
                    x_len = max(x_len, int(xx.max()) + 1)
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
    geo = Geometry(x_len, par, elements, bnd, args)
    if geo.dxz4 is None:
        logger.warning("no DXZ4 among --xi-elements: no megadomain boundary")
    if not args.boundaries:
        logger.warning("no --boundaries: no TAD boundary insulation")
    rng = np.random.default_rng(args.seed)
    all_stats, all_ps, all_prof, all_bnd, all_ratio = [], [], [], [], []
    for g, got in data.items():
        mode = xi.get(g)
        labels = ("Xi", "Xa") if mode else ("hap1", "hap2")
        counts = []
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
