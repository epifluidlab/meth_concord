#!/usr/bin/env python3
"""Site-level correlation vs genomic distance: GCH–GCH, HCG–HCG, HCG–GCH.

Correlates the methylation state of INDIVIDUAL cytosines (not read-end
averages) as a function of the genomic distance between them, and compares each
curve against a distance-matched, NON-LINKED null.

Only high-quality read pairs are used — uniquely mapped, not a PCR duplicate,
mapQ > 30 on BOTH ends, both ends on the same chromosome.  Those filters are
applied here exactly as ``bam_parser`` applies them upstream (shared skip-mask
and strand logic; see ``nome_hic_concordance/site_calls.py``).

Three site pairings, each a Pearson r on the 0/1 calls (the phi coefficient):

  * GCH–GCH : two GpC sites
  * HCG–HCG : two CpG sites
  * HCG–GCH : a CpG site vs a GpC site (cross-context; the shortest-distance
              bin is the "HCG and its next GCH" case)

Site pairs come from BOTH arms, giving one continuous curve:
  * WITHIN-read  — two sites on the same read  → the short range (≲ read length)
  * CROSS-end    — a site on end 1 × a site on end 2 → the long range (contact)

Two conditions are plotted:
  * LINKED  — the read pair's true partner end (the real ligation), ALL pairs
  * NULL    — the partner end is replaced by a DIFFERENT end within
              ±--null-window bp of it (same distance, different molecule), so a
              linked-minus-null gap is same-molecule coordination rather than
              genomic proximity.

The question the null answers is whether local methylation at ONE END
correlates with local methylation at the OTHER end, so Δ = LINKED − NULL is
taken against the full linked curve over every pair.

A third arm, MATCHED — the true partner restricted to the pairs the null could
actually use — is computed and written to the TSV but NOT plotted.  It exists
only as a check: a pair can enter the null solely if some other end lies within
±window of its partner, and if that subset were unrepresentative, matched would
diverge from linked.  It is built from subsampled anchors, so as a control it is
strictly noisier than linked, and with the pooled null (eligibility near 100%)
it is near-redundant as well.

--null-scope pooled (the default) is what makes eligibility a non-issue: the
substitute end is drawn from the read ends of EVERY cell in the group being
plotted -- the group's own cells for a group figure, all cells for the 'all'
figure -- which raises donor density by roughly the number of cells.  Each cell
contributes a bounded reservoir sample, so this costs one BAM pass and a
predictable amount of memory rather than holding every pair.  --null-scope cell
restores the old within-cell behaviour, where only a read-density-biased
minority of pairs is eligible.

Each panel is paired with Δ = linked − null, its 95% CI, and a p-value from a
CELL-level bootstrap: site pairs are not independent (tens per read, ~1e6 reads
per cell), so a test that treats them as independent would report significance
for everything.  One test is made per bin per measure -- ~144 for a single
group -- so the p-values are FDR-adjusted (--fdr, Benjamini-Hochberg by
default) across the whole figure and the rings mark q<0.05; the raw p is kept
in the TSV next to q.  Bins with fewer than --min-pairs site pairs on EITHER side are
masked, which is what keeps a thin, noisy null from putting a lone spike in an
otherwise smooth curve.  The figure starts at --plot-min-distance (10bp):
below that the pairs are adjacent-motif artefacts rather than a distance
relationship.

Because the null swaps the PARTNER END, it applies to the cross-end arm only.
Within-read pairs have no non-linked counterpart — both cytosines sit on the
same molecule by construction — so the null curve is undefined (NaN) below
roughly the read length, and those bins show the linked curve alone.

By default every cell in --samples is pooled.  With --group-column, a separate
figure + stats file is also written per group (plus the pooled "all").

Usage
-----
    python scripts/site_distance_correlation.py \\
        --samples samples.tsv \\
        --context-bed GCH:/ref/GCH.bed.gz,HCG:/ref/HCG.bed.gz \\
        --output-dir output/site_distance_corr \\
        --max-cells 50 --max-pairs-per-cell 20000 --n-threads 8
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("site_distance")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance.site_calls import iter_cell_read_pairs  # noqa: E402

# htslib backend: does the whole hot path (parse → sites → pair → bin →
# accumulate) in C++.  Essential here — only ~1% of pairs are long-range, so the
# long-range arm needs FULL BAMs, and the Python loop cannot carry ~1e6
# pairs/cell.  Verified bit-identical to the pysam path, ~100x faster.
try:                                                        # pysam first: its
    import pysam as _pysam                                  # bundled libhts
    _ = _pysam                                              # supplies symbols
    from nome_hic_concordance import _site_dist_core as _CPP
except Exception:                                           # pragma: no cover
    _CPP = None

_MEASURES = (("gg", "GCH–GCH  r(site_i, site_j)"),
             ("hh", "HCG–HCG  r(site_i, site_j)"),
             ("gh", "HCG–GCH  r(CpG, GpC)  [cross-context]"))
_CONDS = (("linked", "linked (true partner, all pairs)"),
          ("matched", "linked (true partner, null-eligible pairs)"),
          ("null", "non-linked (±window)"))
_CSTYLE = {"linked":  dict(ls="-", marker="o", ms=3.2, lw=1.6, alpha=1.00),
           "matched": dict(ls="-.", marker="s", ms=2.6, lw=1.3, alpha=0.85),
           "null":    dict(ls="--", marker="", lw=1.4, alpha=0.65)}
# 'matched' is still computed and written to the TSV (it says whether the
# null-eligible subset is representative) but it is NOT drawn: it is built from
# subsampled anchors, so as a control it is strictly noisier than linked, and
# with the pooled null -- eligibility near 100% -- it is near-redundant too.
_PLOT_CONDS = ("linked", "null")
_TLAB = {1: "1bp", 10: "10bp", 100: "100bp", 1_000: "1kb", 10_000: "10kb",
         100_000: "100kb", 1_000_000: "1Mb", 10_000_000: "10Mb"}
_QC = ("high-quality read pairs only: uniquely mapped, no PCR duplicate, "
       "mapQ>30 both ends, same chromosome")


# --------------------------------------------------------------------------
# Sufficient statistics
# --------------------------------------------------------------------------

def _new_acc(nb):
    """(6, nb) raw sums: n, Σx, Σy, Σx², Σy², Σxy."""
    return np.zeros((6, nb))


def _accumulate(acc, bins, x, y):
    """Add site pairs (already binned) into the (6, nb) sums."""
    if bins.size == 0:
        return
    nb = acc.shape[1]
    x = x.astype(np.float64); y = y.astype(np.float64)
    acc[0] += np.bincount(bins, minlength=nb)
    acc[1] += np.bincount(bins, weights=x, minlength=nb)
    acc[2] += np.bincount(bins, weights=y, minlength=nb)
    acc[3] += np.bincount(bins, weights=x * x, minlength=nb)
    acc[4] += np.bincount(bins, weights=y * y, minlength=nb)
    acc[5] += np.bincount(bins, weights=x * y, minlength=nb)


def _corr(acc, min_pairs):
    """(6, nb) sums → (r, n) per bin; NaN where n < min_pairs or no variance."""
    n, sx, sy, sxx, syy, sxy = acc
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = np.where(n > 0, n, 1.0)
        cov = sxy - sx * sy / nz
        vx = sxx - sx * sx / nz
        vy = syy - sy * sy / nz
        den = np.sqrt(vx * vy)
        r = np.where((n >= min_pairs) & (den > 0), cov / np.where(den > 0, den, 1.0),
                     np.nan)
    return r, n


_METRICS = {
    "phi": "Pearson r (phi) of 0/1 calls",
    "yule_q": "Yule's Q  (marginal-free)",
    "log_or": "log odds ratio  (marginal-free)",
    "phi_norm": "phi / phi$_{max}$  (marginal-normalised)",
}


def _table(acc):
    """2×2 counts (n11, n10, n01, n00) from the sufficient statistics.

    Nothing extra had to be stored: for 0/1 data the accumulator ALREADY is the
    table.  Σxy is n11, Σx and Σy are the row and column totals, and n closes
    it.  So every 2×2 measure below is available on existing runs without
    re-parsing a single BAM.
    """
    acc = np.asarray(acc, dtype=np.float64)
    take = (lambda i: acc[i]) if acc.ndim == 2 else (lambda i: acc[:, i])
    n, sx, sy, sxy = take(0), take(1), take(2), take(5)
    return sxy, sx - sxy, sy - sxy, n - sx - sy + sxy, n


def _phi_envelope(acc):
    """(phi_min, phi_max) attainable GIVEN the marginals.

    phi cannot reach ±1 unless the two marginals match.  With HCG ~0.78 and GCH
    ~0.25 the range is about [-0.92, +0.31] -- so a cross-context phi near zero
    may be a ceiling, not an absence of association.  Plotting this envelope is
    what makes that visible instead of implicit.
    """
    a, b, c, d, n = _table(acc)
    with np.errstate(invalid="ignore", divide="ignore"):
        p1 = np.where(n > 0, (a + b) / np.where(n > 0, n, 1), np.nan)
        p2 = np.where(n > 0, (a + c) / np.where(n > 0, n, 1), np.nan)
        lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
        pmax = np.sqrt(lo * (1 - hi) / ((1 - lo) * hi))
        pmin = np.where(p1 + p2 <= 1,
                        -np.sqrt(p1 * p2 / ((1 - p1) * (1 - p2))),
                        -np.sqrt((1 - p1) * (1 - p2) / (p1 * p2)))
    return pmin, pmax


def _metric(acc, min_pairs, metric="phi"):
    """(value, n) per bin for any supported 2×2 association measure.

    Works on a single (6, nb) accumulator and on a stacked (B, 6, nb) one, so
    the bootstrap uses exactly the same code path as the point estimate.

    phi is Pearson r on 0/1 data, which is what the pipeline has always
    reported -- but it is bounded by the marginals, so it is NOT comparable
    across measures with different methylation rates.  yule_q and log_or are
    functions of the odds ratio and are invariant to the marginals, which makes
    them the honest choice for comparing GCH-GCH against HCG-GCH.
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; pick from {list(_METRICS)}")
    a, b, c, d, n = _table(acc)
    with np.errstate(invalid="ignore", divide="ignore"):
        if metric in ("phi", "phi_norm"):
            den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
            v = np.where(den > 0, (a * d - b * c) / np.where(den > 0, den, 1.0),
                         np.nan)
            if metric == "phi_norm":
                pmin, pmax = _phi_envelope(acc)
                bound = np.where(v >= 0, pmax, np.abs(pmin))
                v = np.where(bound > 0, v / np.where(bound > 0, bound, 1.0),
                             np.nan)
        else:
            # Haldane-Anscombe: +0.5 per cell keeps a zero cell from sending
            # the odds ratio to 0 or infinity.
            A, B, C, D = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            v = ((A * D - B * C) / (A * D + B * C) if metric == "yule_q"
                 else np.log((A * D) / (B * C)))
        v = np.where(n >= min_pairs, v, np.nan)
    return v, n


def _corr_stack(acc, min_pairs):
    """:func:`_corr` vectorised over a leading (bootstrap) axis: (B, 6, nb) → r."""
    n, sx, sy, sxx, syy, sxy = (acc[:, i] for i in range(6))
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = np.where(n > 0, n, 1.0)
        cov = sxy - sx * sy / nz
        den = np.sqrt((sxx - sx * sx / nz) * (syy - sy * sy / nz))
        return np.where((n >= min_pairs) & (den > 0),
                        cov / np.where(den > 0, den, 1.0), np.nan)


def _bootstrap_delta(per_cell_a, per_cell_b, min_pairs, n_boot, seed,
                     metric="phi"):
    """Cell-level bootstrap of r(a) − r(b) per bin → (lo, hi, p).

    The unit of resampling is the CELL, not the site pair.  Site pairs are
    massively non-independent -- tens of them come off one read, ~1e6 reads off
    one cell -- so a Fisher z-test on the site-pair count would treat ~1e8
    correlated observations as independent and return p-values of order 1e-300
    for any difference at all.  Resampling cells is the honest unit and is what
    the sibling joint_distance_coordination.py already does for its CIs.
    """
    a = np.stack(per_cell_a).astype(np.float64)          # (k, 6, nb)
    b = np.stack(per_cell_b).astype(np.float64)
    k = a.shape[0]
    if k < 3 or n_boot <= 0:                             # too few cells to resample
        return None
    rng = np.random.default_rng(seed)
    w = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(np.float64)
    d = (_metric((w @ a.reshape(k, -1)).reshape(n_boot, 6, -1), min_pairs, metric)[0]
         - _metric((w @ b.reshape(k, -1)).reshape(n_boot, 6, -1), min_pairs, metric)[0])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # A bin masked in every resample is an all-NaN slice — expected, not an
        # error; it simply has no Δ to report.
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(d, [2.5, 97.5], axis=0)
        n_ok = np.sum(np.isfinite(d), axis=0)
        # (1 + count) / (1 + B), NOT count / B: a finite bootstrap can never
        # justify p = 0, and an exact zero would then survive any FDR procedure
        # unconditionally.  This floors the two-sided p at 2/(B+1).
        p = 2.0 * (1.0 + np.minimum(np.sum(d <= 0, axis=0),
                                    np.sum(d >= 0, axis=0))) / (n_ok + 1.0)
    # Below ~20 usable resamples the tail is unresolvable; report nothing.
    p = np.where(n_ok >= 20, np.minimum(p, 1.0), np.nan)
    return lo, hi, np.where(n_ok >= 20, p, np.nan)


def _fdr(p, method="bh"):
    """Adjusted p (q) over the finite entries of ``p``; NaN stays NaN.

    The figure makes one test per distance bin per measure -- ~144 for a single
    group -- so at raw p<0.05 roughly 7 bins would be starred by chance alone.

    'bh' (Benjamini-Hochberg) is the default and is valid under positive
    regression dependency, which is the right assumption here: adjacent distance
    bins share reads and cells and trace a smooth curve, so the tests are
    positively correlated rather than independent.  'by' (Benjamini-Yekutieli)
    holds under ARBITRARY dependence at the cost of a log(m) penalty.
    """
    p = np.asarray(p, float)
    if method == "none":
        return p.copy()
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    ps = p[ok]
    order = np.argsort(ps)
    c = np.sum(1.0 / np.arange(1, m + 1)) if method == "by" else 1.0
    q = ps[order] * c * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]      # enforce monotonicity in rank
    adj = np.empty(m)
    adj[order] = np.clip(q, 0.0, 1.0)
    out[ok] = adj
    return out


def _boot_parts(entry):
    """(lo, hi, p, q) from a bootstrap entry, tolerating a pre-FDR 3-tuple."""
    if entry is None:
        return None
    if len(entry) == 4:
        return entry
    lo, hi, p = entry
    return lo, hi, p, p


def _stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# --------------------------------------------------------------------------
# Pooled cross-cell null
# --------------------------------------------------------------------------

def _concat_csr(parts, tag):
    """Concatenate per-cell CSR columns, shifting each part's row pointers."""
    if not parts:
        return (np.zeros(0, np.int32), np.zeros(0, np.int8), np.zeros(1, np.int32))
    pos = np.concatenate([np.asarray(p[f"{tag}_pos"]) for p in parts])
    st = np.concatenate([np.asarray(p[f"{tag}_st"]) for p in parts])
    ptrs, off = [], 0
    for p in parts:
        q = np.asarray(p[f"{tag}_ptr"], np.int64)
        ptrs.append(q[:-1] + off)
        off += int(q[-1])
    ptr = np.concatenate(ptrs + [np.array([off], np.int64)])
    return pos.astype(np.int32), st.astype(np.int8), ptr.astype(np.int32)


def _permute_csr(pos, st, ptr, order):
    """Reorder CSR rows — donors must reach the matcher sorted by (chrom, mid)."""
    ptr = ptr.astype(np.int64)
    lens = (ptr[1:] - ptr[:-1])[order]
    new_ptr = np.concatenate([[0], np.cumsum(lens)])
    total = int(new_ptr[-1])
    if total == 0:
        return pos[:0].copy(), st[:0].copy(), new_ptr.astype(np.int32)
    idx = np.repeat(ptr[order] - new_ptr[:-1], lens) + np.arange(total)
    return pos[idx], st[idx], new_ptr.astype(np.int32)


def _pooled_null(res, idxs, edges, args, nb):
    """Null drawn from the read ends of EVERY cell in this group.

    Within one cell a long-range pair rarely has another end within ±window, so
    the within-cell null is built from a small, read-density-biased minority.
    Pooling the donor ends across the group's cells raises donor density by
    roughly the number of cells.  Returns per-cell accumulators so the bootstrap
    can still resample cells.
    """
    kept = [i for i in idxs if res[i][3] is not None
            and len(res[i][3][1]["chrom"]) > 0]
    if not kept:
        return None
    dparts = [res[i][3][0] for i in kept]
    aparts = [res[i][3][1] for i in kept]

    def _col(parts, name, dtype):
        return np.concatenate([np.asarray(p[name], dtype) for p in parts])

    def _cellcol(parts):
        return np.concatenate([np.full(len(np.asarray(p["chrom"])), j, np.int32)
                               for j, p in enumerate(parts)])

    d_chrom, d_mid = _col(dparts, "chrom", np.int32), _col(dparts, "mid", np.int32)
    order = np.lexsort((d_mid, d_chrom))
    donors = {"chrom": d_chrom[order], "mid": d_mid[order],
              "cell": _cellcol(dparts)[order]}
    for tag in ("g", "h"):
        p_, s_, q_ = _permute_csr(*_concat_csr(dparts, tag), order)
        donors[f"{tag}_pos"], donors[f"{tag}_st"], donors[f"{tag}_ptr"] = p_, s_, q_

    anchors = {"chrom": _col(aparts, "chrom", np.int32),
               "mid2": _col(aparts, "mid2", np.int32),
               "cell": _cellcol(aparts)}
    for tag in ("g1", "h1", "g2", "h2"):
        p_, s_, q_ = _concat_csr(aparts, tag)
        anchors[f"{tag}_pos"], anchors[f"{tag}_st"], anchors[f"{tag}_ptr"] = p_, s_, q_

    out = _CPP.cross_cell_null(anchors, donors, [float(e) for e in edges],
                               int(args.null_window),
                               int(args.max_site_pairs_per_read),
                               int(args.seed), len(kept))
    k = len(kept)
    null_pc, match_pc = {}, {}
    for mi, (mkey, _) in enumerate(_MEASURES):
        null_pc[mkey] = [np.asarray(out["null_per_cell"][3 * c + mi], np.float64)
                         for c in range(k)]
        match_pc[mkey] = [np.asarray(out["matched_per_cell"][3 * c + mi], np.float64)
                          for c in range(k)]
    st = [res[i][3][2] for i in kept if len(res[i][3]) > 2]
    return {"null_per_cell": null_pc, "matched_per_cell": match_pc,
            "n_eligible": int(out["n_eligible"]),
            "n_anchors": int(out["n_anchors"]), "n_donors": int(len(d_mid)),
            "n_anchor_long": sum(s["anchor_long"] for s in st),
            "n_pairs_long_seen": sum(s["pairs_long_seen"] for s in st)}


def _build_edges(min_d, max_d, per_decade):
    lo, hi = np.log10(max(min_d, 1)), np.log10(max_d)
    n_edges = max(2, int(round((hi - lo) * per_decade)) + 1)
    e = np.logspace(lo, hi, n_edges)
    e[0], e[-1] = float(min_d), float(max_d)      # exact bounds (FP-safe)
    return e


def _pair_sites(pos_a, st_a, pos_b, st_b, edges, same_array, rng, cap):
    """All (or a capped sample of) cross products between two site sets.

    ``same_array`` → the two sets are the SAME sites (within-read), so take the
    upper triangle (i<j) and never pair a site with itself.  Returns
    ``(bins, x, y)`` for pairs whose separation falls inside the edges.
    """
    na, nb_ = pos_a.size, pos_b.size
    if na == 0 or nb_ == 0:
        return (np.empty(0, np.int64),) * 3
    if same_array:
        ia, ib = np.triu_indices(na, k=1)
    else:
        ia = np.repeat(np.arange(na), nb_)
        ib = np.tile(np.arange(nb_), na)
    if ia.size == 0:
        return (np.empty(0, np.int64),) * 3
    if cap and ia.size > cap:                      # bound the O(n·m) blow-up
        sel = rng.choice(ia.size, size=cap, replace=False)
        ia, ib = ia[sel], ib[sel]
    d = np.abs(pos_b[ib] - pos_a[ia])
    keep = (d >= edges[0]) & (d <= edges[-1])
    if not keep.any():
        return (np.empty(0, np.int64),) * 3
    ia, ib, d = ia[keep], ib[keep], d[keep]
    nbins = len(edges) - 1
    bins = np.clip(np.digitize(d, edges) - 1, 0, nbins - 1).astype(np.int64)
    return bins, st_a[ia], st_b[ib]


def _add_pair(accs, cond, edges, rng, cap, c1, c2, cross_only=False):
    """Accumulate every measure for one (end1, end2) site-call pair."""
    nb = accs[("gg", cond)].shape[1]
    g1, g2 = c1.get("GCH", (np.empty(0), np.empty(0))), c2.get("GCH", (np.empty(0), np.empty(0)))
    h1, h2 = c1.get("HCG", (np.empty(0), np.empty(0))), c2.get("HCG", (np.empty(0), np.empty(0)))

    def go(key, pa, sa, pb, sb, same):
        b, x, y = _pair_sites(pa, sa, pb, sb, edges, same, rng, cap)
        _accumulate(accs[(key, cond)], b, x, y)

    if not cross_only:
        # WITHIN-read arm (short range) — both ends contribute
        go("gg", g1[0], g1[1], g1[0], g1[1], True)
        go("gg", g2[0], g2[1], g2[0], g2[1], True)
        go("hh", h1[0], h1[1], h1[0], h1[1], True)
        go("hh", h2[0], h2[1], h2[0], h2[1], True)
        go("gh", h1[0], h1[1], g1[0], g1[1], False)
        go("gh", h2[0], h2[1], g2[0], g2[1], False)
    # CROSS-end arm (long range)
    go("gg", g1[0], g1[1], g2[0], g2[1], False)
    go("hh", h1[0], h1[1], h2[0], h2[1], False)
    go("gh", h1[0], h1[1], g2[0], g2[1], False)     # CpG(end1) × GpC(end2)
    go("gh", h2[0], h2[1], g1[0], g1[1], False)     # and the reverse ordering
    _ = nb


def prepare_ctx(ctx_positions, blacklist=None):
    """Convert the context tables to the C++ layout ONCE.

    ``arr["pos"]`` on a structured array is a strided view, so ``.astype`` copies
    the whole table (~2 GB for a mammalian GCH+HCG set).  Doing that per cell
    burned that allocation 1577 times per worker; hoisting it here means the
    buffers are built once and, under the thread pool, shared by every worker.
    """
    ctx_g = {c: (np.ascontiguousarray(a["pos"], np.int64),
                 np.ascontiguousarray(a["ref_base"], np.uint8))
             for c, a in ctx_positions.get("GCH", {}).items()}
    ctx_h = {c: (np.ascontiguousarray(a["pos"], np.int64),
                 np.ascontiguousarray(a["ref_base"], np.uint8))
             for c, a in ctx_positions.get("HCG", {}).items()}
    bl = {c: (np.ascontiguousarray(iv[:, 0], np.int64),
              np.ascontiguousarray(iv[:, 1], np.int64))
          for c, iv in (blacklist or {}).items() if len(iv)}
    return ctx_g, ctx_h, bl


def process_cell_cpp(bam_path, prepared, edges, trim_ends, mapq_min,
                     cap, null_window, seed, baseq_min=0,
                     donor_cap=0, anchor_cap=0, anchor_long_cap=0,
                     long_range_bp=1000, strand_min_evidence=1):
    """htslib fast path — same result as :func:`process_cell`, ~100x faster.

    ``prepared`` comes from :func:`prepare_ctx`; the extension releases the GIL,
    so this is safe to run on a thread pool.

    ``donor_cap`` / ``anchor_cap`` additionally reservoir-sample this cell's
    read ends and pairs for the POOLED cross-cell null (see ``--null-scope``).
    """
    ctx_g, ctx_h, bl = prepared
    res = _CPP.parse_cell_site_distance(
        str(bam_path), ctx_g, ctx_h, [float(e) for e in edges],
        int(trim_ends), int(mapq_min), int(null_window), int(cap), int(seed),
        False, int(baseq_min), bl, int(donor_cap), int(anchor_cap),
        int(anchor_long_cap), int(long_range_bp),
        int(strand_min_evidence))
    accs = {}
    for m, _ in _MEASURES:
        for c, _lab in _CONDS:
            accs[(m, c)] = np.asarray(res[f"{m}_{c}"], dtype=np.float64)
    pools = None
    if donor_cap or anchor_cap or anchor_long_cap:
        pools = (dict(res["donors"]), dict(res["anchors"]),
                 {"anchor_long": int(res["n_anchor_long"]),
                  "anchor_short": int(res["n_anchor_short"]),
                  "pairs_long_seen": int(res["n_pairs_long_seen"])})
    strand = {"known": int(res["n_strand_known"]),
              "disagree": int(res["n_strand_disagree"])}
    return accs, int(res["n_pairs"]), int(res["n_null_eligible"]), pools, strand


def process_cell(bam_path, ctx_positions, edges, trim_ends, mapq_min,
                 max_pairs, cap, null_window, seed):
    """One cell → {(measure, condition): (6, nb) sums}.

    Both conditions are built from the SAME parsed pairs (one BAM pass): the
    linked condition uses the true partner end; the null replaces it with a
    different end whose midpoint lies within ±``null_window`` of it.

    Pure-Python/pysam fallback; the htslib backend above is used when built.
    """
    rng = np.random.default_rng(seed)
    nb = len(edges) - 1
    accs = {(m, c): _new_acc(nb) for m, _ in _MEASURES for c, _ in _CONDS}

    pairs = list(iter_cell_read_pairs(
        bam_path, ctx_positions, trim_ends=trim_ends, mapq_min=mapq_min,
        max_pairs_per_cell=max_pairs, rng=rng))
    if not pairs:
        return accs, 0, 0, None, {"known": 0, "disagree": 0}

    for chrom, mid1, mid2, c1, c2 in pairs:
        _add_pair(accs, "linked", edges, rng, cap, c1, c2)

    # ---- NULL: swap in a nearby, non-linked partner ----------------------
    # Index end-2 midpoints per chromosome so "within ±window of the true
    # partner" is a binary search rather than an O(n²) scan.
    by_chrom: dict[str, tuple] = {}
    for chrom in {p[0] for p in pairs}:
        idx = [i for i, p in enumerate(pairs) if p[0] == chrom]
        mids = np.array([pairs[i][2] for i in idx], np.int64)
        order = np.argsort(mids)
        by_chrom[chrom] = (mids[order], np.asarray(idx)[order])

    n_eligible = 0
    for i, (chrom, mid1, mid2, c1, c2) in enumerate(pairs):
        mids, idx = by_chrom[chrom]
        lo = np.searchsorted(mids, mid2 - null_window, side="left")
        hi = np.searchsorted(mids, mid2 + null_window, side="right")
        if hi - lo < 2:                       # only itself in the window
            continue
        # Deterministic non-self pick (see the C++ backend for why the old
        # "4 random attempts, else give up" silently dropped eligible pairs).
        span = int(hi - lo)
        k = int(rng.integers(lo, hi))
        if int(idx[k]) == i:
            k = int(lo) + ((k - int(lo)) + 1) % span
        j = int(idx[k])
        n_eligible += 1
        _add_pair(accs, "null", edges, rng, cap, c1, pairs[j][4],
                  cross_only=True)
        # Same pair, TRUE partner, cross-end only — the like-for-like baseline
        # for the null, since the null can only ever see THESE pairs.
        _add_pair(accs, "matched", edges, rng, cap, c1, c2, cross_only=True)
    return accs, len(pairs), n_eligible, None, {"known": 0, "disagree": 0}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _load_snps_all(dbsnp_vcf, wanted_chroms):
    """``{chrom: sorted 0-based SNP positions}`` for every wanted chromosome.

    Deliberately NOT reference._load_snp_positions, which needs a tabix index
    and returns an EMPTY set on any failure — an unindexed .vcf or a chr-naming
    mismatch then silently disables the filter.  Here an indexed VCF is queried
    per chromosome, an unindexed one is read in a single sequential pass, and
    'chr1' / '1' are reconciled.  Returns {} only when the VCF truly has no
    usable records, which the caller treats as an error.
    """
    import pysam

    def _norm(c):
        return c[3:] if c.startswith("chr") else c

    want = {_norm(c): c for c in wanted_chroms}
    found: dict[str, set] = {c: set() for c in wanted_chroms}
    vcf = pysam.VariantFile(dbsnp_vcf)
    indexed = False
    try:
        indexed = vcf.index is not None
    except (AttributeError, ValueError):
        indexed = False

    n = 0
    if indexed:
        for norm, orig in want.items():
            for name in (orig, norm, f"chr{norm}"):     # try both conventions
                try:
                    it = vcf.fetch(name)
                except (ValueError, KeyError):
                    continue
                for rec in it:
                    found[orig].add(rec.pos - 1)        # VCF 1-based → 0-based
                    n += 1
                if found[orig]:
                    break
    else:
        logger.warning(
            "%s has no tabix index, so it must be read sequentially in ONE "
            "pass — this phase is inherently single-threaded (~1-2 min for a "
            "mammalian dbSNP). bgzip + tabix it to avoid this.", dbsnp_vcf)
        t_scan = time.time()
        for rec in vcf.fetch():                         # whole-file scan
            orig = want.get(_norm(rec.chrom))
            if orig is not None:
                found[orig].add(rec.pos - 1)
                n += 1
                if n % 10_000_000 == 0:                 # keep the phase visible
                    print(f"    dbSNP scan: {n / 1e6:.0f}M records "
                          f"({time.time() - t_scan:.0f}s)", flush=True)
    vcf.close()
    logger.info("dbSNP: loaded %s SNP positions over %s chromosomes",
                f"{n:,}", sum(1 for v in found.values() if v))
    return {c: v for c, v in found.items() if v}


def _drop_snp_sites(ctx_positions, dbsnp_vcf, motif_flank=1):
    """Remove context sites whose MOTIF overlaps a dbSNP locus.

    Checking only the cytosine is not enough: GCH and HCG are 3-bp motifs, and a
    SNP at the flanking G (or at the H) means the locus is not that motif in
    this sample at all — a GCH whose G is polymorphic may really be an HCH, so
    its "accessibility" call is meaningless.  A SNP at the cytosine itself is
    the other failure (an unconverted C and a genomic T are indistinguishable).
    So a site is dropped when ANY of pos-1, pos, pos+1 carries a SNP, which
    covers the motif on either strand (the cytosine is always the middle base).
    """
    out, dropped, kept = {}, 0, 0
    chroms = sorted({c for ctx in ctx_positions.values() for c in ctx})
    snps = _load_snps_all(dbsnp_vcf, chroms)
    if not snps:
        sys.exit(
            f"--dbsnp-vcf {dbsnp_vcf}: no SNP positions matched any of the "
            f"{len(chroms)} context chromosomes (e.g. {chroms[:3]}).\n"
            f"  Check that the VCF's chromosome names match the context BED's "
            f"and that the file is a readable VCF.\n"
            f"  Refusing to continue silently — an unfiltered run would look "
            f"identical to a filtered one.")
    for ctx, per_chrom in ctx_positions.items():
        out[ctx] = {}
        for chrom, arr in per_chrom.items():
            s = snps.get(chrom)
            if not s:
                out[ctx][chrom] = arr
                kept += len(arr)
                continue
            snp_arr = np.fromiter(s, np.int64, len(s))
            pos = arr["pos"]
            # drop if ANY motif base (pos-flank .. pos+flank) hits a SNP
            hit = np.zeros(pos.shape, dtype=bool)
            for off in range(-motif_flank, motif_flank + 1):
                hit |= np.isin(pos + off, snp_arr)
            mask = ~hit
            out[ctx][chrom] = arr[mask]
            dropped += int((~mask).sum())
            kept += int(mask.sum())
    total = dropped + kept
    print(f"  dbSNP filter: dropped {dropped:,} of {total:,} context sites "
          f"({100.0 * dropped / max(total, 1):.2f}%)")
    if dropped == 0:
        print("  WARNING: 0 sites dropped despite SNPs being loaded — check "
              "the coordinate convention of the VCF.", file=sys.stderr)
    return out


_UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9._+-]")


def safe_names(labels):
    """``{label: filename-safe name}``, unique across ``labels``.

    Cell-type labels routinely contain characters that are not filename-safe —
    'IT-L2/3' and 'PC/EC' turn a filename into a path into a directory that
    does not exist.  Slashes and friends collapse to '_', and if two labels
    collapse onto the same name a numeric suffix keeps them distinct.  Only the
    FILENAME is rewritten; titles and the summary JSON keep the real label.
    """
    out, used = {}, {}
    for lab in labels:
        base = _UNSAFE_IN_NAME.sub("_", str(lab).strip()) or "unnamed"
        if not base.strip("."):        # '.' / '..' are path components, not names
            base = "unnamed"
        n = used.get(base, 0)
        used[base] = n + 1
        out[lab] = base if n == 0 else f"{base}_{n + 1}"
    return out


def _logx(ax, lo, hi):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)               # pin it: masked bins must not stretch it
    ticks = [t for t in _TLAB if lo <= t <= hi]
    ax.set_xticks(ticks); ax.set_xticklabels([_TLAB[t] for t in ticks])
    ax.set_xlabel("genomic distance between the two cytosines")


def _delta(curves, key, min_pairs):
    """linked − null, masked where EITHER side is too thin to trust.

    The question the null answers is whether local methylation at one END
    correlates with local methylation at the OTHER end, so the thing to subtract
    it from is the full linked curve over ALL pairs.  ('matched' -- the true
    partner restricted to null-eligible pairs -- is still computed and written
    to the TSV as a check on whether that subset is representative, but it is
    built from subsampled anchors and so is the noisier of the two; with the
    pooled null it is also near-redundant, since eligibility runs close to 100%.)

    Masking jointly matters: a bin can hold millions of linked site pairs and a
    few hundred null ones, and differencing an unmasked noisy null is exactly
    what produces a lone spike in an otherwise smooth curve.
    """
    rl, nl = curves[(key, "linked")]
    rn, nn = curves[(key, "null")]
    ok = (nl >= min_pairs) & (nn >= min_pairs)
    return np.where(ok, rl - rn, np.nan), ok


def _plot(curves, boot, centers, edges, label, n_cells, n_pairs, out, min_pairs,
          plot_min=10.0, metric="phi", envelope=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"gg": "#1f77b4", "hh": "#d62728", "gh": "#2ca02c"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), sharex=True)
    # Below ~10bp the site pairs are adjacent-motif artefacts rather than a
    # distance relationship -- most sharply the distance-1 HCG-HCG point, which
    # is one CpG dyad and can only be seen when the two mates disagree on
    # conversion strand.  Hide them rather than let them anchor the axis.
    lo_x = max(float(edges[0]), float(plot_min))
    vis = np.asarray(centers) >= lo_x

    def _show(v):
        return np.where(vis, v, np.nan)

    for col, (key, title) in enumerate(_MEASURES):
        ax = axes[0, col]
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        for cond, clabel in _CONDS:
            if cond not in _PLOT_CONDS:
                continue
            r, _n = curves[(key, cond)]
            ax.plot(centers, _show(r), color=colors[key], label=clabel,
                    **_CSTYLE[cond])
        if metric == "phi" and envelope is not None and key in envelope:
            # phi cannot leave this band, whatever the association is.  When the
            # curve rides the edge, a near-zero phi is a CEILING, not an absence.
            pmin, pmax = envelope[key]
            ax.plot(centers, _show(pmax), ls=":", lw=1.1, color="0.45",
                    label="attainable phi range (marginals)")
            ax.plot(centers, _show(pmin), ls=":", lw=1.1, color="0.45")
        ax.set_ylabel(_METRICS[metric])
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5, loc="best")

        # ---- delta panel -------------------------------------------------
        axd = axes[1, col]
        axd.axhline(0, color="0.6", lw=0.8, zorder=0)
        d, ok = _delta(curves, key, min_pairs)
        d = _show(d)
        sig = None
        if boot and boot.get(key) is not None:
            lo, hi, _p, q = _boot_parts(boot[key])
            lo = _show(np.where(ok, lo, np.nan))
            hi = _show(np.where(ok, hi, np.nan))
            axd.fill_between(centers, lo, hi, color=colors[key], alpha=0.20,
                             lw=0, label="95% CI (cell bootstrap)")
            sig = np.isfinite(d) & np.isfinite(q) & (q < 0.05)
        axd.plot(centers, d, "-o", ms=3.2, lw=1.6, color=colors[key],
                 label="Δ = linked − null")
        # AFTER the line, or the ring is hidden underneath its own marker.
        if sig is not None and sig.any():
            axd.plot(np.asarray(centers)[sig], d[sig], "o", ms=7.5,
                     mfc="none", mec="0.15", mew=1.2, ls="none",
                     label="q < 0.05 (FDR)", zorder=5)
        _logx(axd, lo_x, edges[-1])
        axd.set_ylabel(f"Δ {_METRICS[metric].split(chr(32))[0]}")
        axd.set_title(f"Δ {title.split('  ')[0]}   (same-molecule excess)",
                      fontsize=9.5)
        axd.legend(fontsize=7, loc="best")

    fig.suptitle(f"Site-level correlation vs distance — {label}  |  "
                 f"{n_cells} cells, {n_pairs:,} read pairs", fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    fig.text(0.5, 0.024, _QC, ha="center", fontsize=7.5, color="0.35")
    fig.text(0.5, 0.008,
             f"x starts at {lo_x:,.0f}bp (closer pairs are adjacent-motif "
             f"artefacts); bins with fewer than {min_pairs:,} site pairs on "
             f"either side are masked; rings mark FDR-adjusted q<0.05 from "
             f"a cell-level bootstrap (site pairs are not independent)",
             ha="center", fontsize=7, color="0.45")
    fig.savefig(out)
    plt.close(fig)


def _write_tsv(curves, boot, centers, edges, path, min_pairs,
               accs=None, metric="phi"):
    rows = {"bin_lo": edges[:-1].astype(np.int64),
            "bin_hi": edges[1:].astype(np.int64),
            "center_bp": np.round(centers).astype(np.int64)}
    for key, _ in _MEASURES:
        for cond, _lab in _CONDS:
            r, n = curves[(key, cond)]
            rows[f"{key}_{cond}_{metric}"] = r
            rows[f"{key}_{cond}_n"] = n.astype(np.int64)
            if accs is not None:
                # All 2x2 measures come free from the same sufficient
                # statistics, so write them all rather than force a re-run to
                # see whether a conclusion survives a different metric.
                for alt in _METRICS:
                    if alt == metric:
                        continue
                    rows[f"{key}_{cond}_{alt}"] = _metric(
                        accs[(key, cond)], min_pairs, alt)[0]
                if cond == "linked":
                    pmin, pmax = _phi_envelope(accs[(key, cond)])
                    rows[f"{key}_phi_min_attainable"] = pmin
                    rows[f"{key}_phi_max_attainable"] = pmax
        d, _ok = _delta(curves, key, min_pairs)
        rows[f"{key}_delta"] = d
        if boot and boot.get(key) is not None:
            lo, hi, praw, q = _boot_parts(boot[key])
            rows[f"{key}_delta_ci_lo"] = np.where(np.isfinite(d), lo, np.nan)
            rows[f"{key}_delta_ci_hi"] = np.where(np.isfinite(d), hi, np.nan)
            rows[f"{key}_delta_p"] = np.where(np.isfinite(d), praw, np.nan)
            rows[f"{key}_delta_q"] = np.where(np.isfinite(d), q, np.nan)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, na_rep="nan")


def _report(accs, n_cells, n_pairs, centers, edges, label, out_dir, args,
            safe=None, n_eligible=0, boot=None, scope_note="",
            n_pairs_for_pct=None):
    curves = {k: _metric(v, args.min_pairs, args.metric)
              for k, v in accs.items()}
    # Build the two paths explicitly: Path.with_suffix() would ALSO mangle a
    # label containing a dot ('A.B' -> '..._A.pdf'), silently.
    stem_name = f"site_distance_corr_{safe or label}"
    fig_path = out_dir / f"{stem_name}.{args.plot_format}"
    tsv_path = out_dir / f"{stem_name}.tsv"
    env = {k: _phi_envelope(accs[(k, "linked")]) for k, _ in _MEASURES}
    _plot(curves, boot, centers, edges, label, n_cells, n_pairs, fig_path,
          args.min_pairs, args.plot_min_distance, args.metric, env)
    _write_tsv(curves, boot, centers, edges, tsv_path, args.min_pairs, accs,
               args.metric)
    print(f"\n=== {label}: {n_cells} cells, {n_pairs:,} read pairs ===")
    if scope_note:
        print(f"  {scope_note}")
    denom = n_pairs_for_pct if n_pairs_for_pct else n_pairs
    pct = 100.0 * n_eligible / denom if denom else 0.0
    print(f"  null-eligible: {n_eligible:,} anchors ({pct:.1f}%) had a "
          f"non-linked end within ±{args.null_window}bp")
    if pct < 25.0:
        print("  NOTE: most anchors have no nearby non-linked end, and the ones "
              "that do sit in read-dense loci.\n"
              "        Compare 'null' against 'matched' (same pairs), NOT "
              "against 'linked' (all pairs).\n"
              "        --null-scope pooled raises this by drawing donor ends "
              "from every cell in the group.")
    deltas = {k: _delta(curves, k, args.min_pairs)[0] for k, _ in _MEASURES}
    print(f"  {'distance':>10}{'n(GG)':>12} | "
          + "".join(f"{m.split('  ')[0]:>30}" for _, m in _MEASURES))
    print(f"  {'':>10}{'':>12} | "
          + "".join(f"{'linked/null        Δ':>30}" for _ in _MEASURES))
    for bi in range(len(centers)):
        if not np.isfinite(curves[("gg", "linked")][0][bi]):
            continue
        cols = ""
        for key, _ in _MEASURES:
            vals = "/".join(
                f"{curves[(key, c)][0][bi]:+.3f}" for c in _PLOT_CONDS)
            dv = deltas[key][bi]
            if np.isfinite(dv):
                star = (_stars(_boot_parts(boot[key])[3][bi])
                        if boot and boot.get(key) is not None else "")
                vals += f"  {dv:+.3f}{star}"
            else:
                vals += "      ."          # masked: too few pairs on one side
            cols += vals.rjust(30)
        print(f"  {centers[bi]:>10,.0f}{int(curves[('gg','linked')][1][bi]):>12,d} | {cols}")
    if boot:
        print(f"  Δ = linked − null;  stars are FDR-adjusted q "
              f"({args.fdr.upper()}): * q<0.05  ** q<0.01  *** q<0.001 "
              f"({args.n_bootstrap} cell-level bootstrap resamples)")
    print(f"  wrote {fig_path} and {tsv_path}")
    return curves


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True,
                    help="TSV with cell_name, bam_path [, group].")
    ap.add_argument("--context-bed", default=None,
                    help="'GCH:/path/GCH.bed.gz,HCG:/path/HCG.bed.gz' — same "
                         "CONTEXT:PATH form as nome_hic_concordance. Either "
                         "this or --reference is required.")
    ap.add_argument("--context-bed-offset", type=int, default=0)
    ap.add_argument("--reference", default=None,
                    help="Reference .2bit — extracts the GCH/HCG positions "
                         "directly (same as nome_hic_concordance Step 1) when "
                         "no --context-bed is given.  Note it does NOT change "
                         "how methylation is called: the state comes from the "
                         "read base vs the site's reference base, which the "
                         "context table already carries.")
    ap.add_argument("--dbsnp-vcf", default=None,
                    help="Tabix-indexed dbSNP VCF; context sites overlapping a "
                         "SNP are dropped (a C/T polymorphism is "
                         "indistinguishable from a bisulfite conversion).")
    ap.add_argument("--blacklist-bed", default=None,
                    help="BED of blacklisted regions; a read PAIR is dropped if "
                         "EITHER end overlaps one.")
    ap.add_argument("--baseq-min", type=int, default=0,
                    help="Drop individual cytosines whose basecall quality is "
                         "below this (0 = off). A miscalled base is a miscalled "
                         "methylation state, so this filters loci, not reads.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--group-column", default=None,
                    help="Column in --samples; also emit a figure per group "
                         "(the pooled 'all' is always written).")
    ap.add_argument("--min-distance", type=int, default=1)
    ap.add_argument("--max-distance", type=int, default=1_000_000)
    ap.add_argument("--bins-per-decade", type=int, default=8)
    ap.add_argument("--null-window", type=int, default=100,
                    help="Replace the partner end with a different end whose "
                         "midpoint is within ±this many bp (default 100).")
    ap.add_argument("--mapq-min", type=int, default=30)
    ap.add_argument("--trim-ends", type=int, default=5)
    ap.add_argument("--max-cells", type=int, default=0,
                    help="0 = all cells in --samples.")
    ap.add_argument("--max-pairs-per-cell", type=int, default=0,
                    help="0 (default) = use ALL read pairs. Only ~1%% of pairs "
                         "are long-range, so capping would starve the "
                         "long-range arm; the htslib backend makes the full "
                         "pass affordable. A non-zero value down-samples "
                         "uniformly (rate from the BAM index), and is only "
                         "honoured by the --backend python fallback.")
    ap.add_argument("--backend", choices=["auto", "htslib", "python"],
                    default="auto",
                    help="'auto' (default) uses the C++/htslib backend when "
                         "built and falls back to pysam otherwise; 'htslib' "
                         "requires it; 'python' forces the pysam path.")
    ap.add_argument("--max-site-pairs-per-read", type=int, default=64,
                    help="Cap site-pair combinations sampled per read pair, "
                         "bounding the O(n·m) blow-up on site-dense reads.")
    ap.add_argument("--min-pairs", type=int, default=2000,
                    help="Mask a distance bin with fewer site pairs than this, "
                         "on EITHER side of a comparison. Thin bins are what "
                         "produce lone spikes in an otherwise smooth curve.")
    ap.add_argument("--null-scope", choices=["pooled", "cell"], default="pooled",
                    help="Where substitute partner ends come from. 'pooled' "
                         "(default) draws them from every cell in the group "
                         "being plotted — the group's own cells for a group "
                         "figure, all cells for the 'all' figure. 'cell' "
                         "restricts them to the anchor's own cell, which at "
                         "single-cell coverage leaves few pairs eligible.")
    ap.add_argument("--null-donors-per-cell", type=int, default=10_000,
                    help="Read ends sampled per cell into the pooled donor "
                         "pool. Raise it if the eligible fraction is low; it "
                         "is the main memory knob (~50 B per donor).")
    ap.add_argument("--null-anchors-per-cell", type=int, default=2_500,
                    help="SHORT-range read pairs sampled per cell as "
                         "pooled-null anchors. Short-range bins have orders of "
                         "magnitude more data than they need, so this stays "
                         "small.")
    ap.add_argument("--null-anchors-long-per-cell", type=int, default=10_000,
                    help="LONG-range (span >= --null-long-range-bp) pairs "
                         "sampled per cell. Sampled separately because only "
                         "~1%% of pairs are long-range: a uniform sample "
                         "starves exactly the bins this analysis is about, "
                         "and for a context as sparse as HCG those bins then "
                         "fall under --min-pairs and go NaN. Long-range pairs "
                         "are rare enough that a high cap keeps nearly all of "
                         "them cheaply.")
    ap.add_argument("--null-long-range-bp", type=int, default=1_000,
                    help="Span at which a pair counts as long-range for the "
                         "stratified anchor sampling above.")
    ap.add_argument("--n-bootstrap", type=int, default=1000,
                    help="Cell-level bootstrap resamples for the Δ confidence "
                         "band and p-value. 0 disables. Note the resolution "
                         "floor: the smallest two-sided p a bootstrap can "
                         "return is 2/(B+1), so with FDR over ~144 tests you "
                         "need B>=5000 before an ISOLATED significant bin can "
                         "clear q<0.05 at all.")
    ap.add_argument("--fdr", choices=["bh", "by", "none"], default="bh",
                    help="Multiple-testing correction across every bin of every "
                         "measure in one figure (~144 tests, so ~7 bins would "
                         "be starred by chance at raw p<0.05). 'bh' is valid "
                         "under positive dependence, which fits adjacent "
                         "distance bins; 'by' holds under arbitrary dependence "
                         "but costs a log(m) penalty. The raw p is always kept "
                         "in the TSV alongside q.")
    ap.add_argument("--strand-min-evidence", type=int, default=1,
                    help="How far C>T must lead G>A (or vice versa) in a read's "
                         "MD tag before its conversion strand is trusted. The "
                         "default 1 is the historical behaviour: a SINGLE "
                         "mismatch, which may be a SNP or a sequencing error, "
                         "decides the strand for the whole read, and a wrong "
                         "call makes every site on it a reading of the "
                         "complementary base. Raise to 2-3 to drop ambiguous "
                         "reads instead (they become strand-unknown and "
                         "contribute nothing).")
    ap.add_argument("--plot-min-distance", type=float, default=10.0,
                    help="Left edge of the x-axis in the FIGURE. Below ~10bp "
                         "the pairs are adjacent-motif artefacts rather than a "
                         "distance relationship — most sharply the distance-1 "
                         "HCG–HCG point, which is a single CpG dyad and only "
                         "observable when the two mates disagree on conversion "
                         "strand. The bins are still computed and written to "
                         "the TSV; use --min-distance to drop them entirely.")
    ap.add_argument("--metric", choices=list(_METRICS), default="phi",
                    help="Association measure for 0/1 calls. 'phi' is Pearson "
                         "r (identical on binary data) and is the historical "
                         "default, but it is BOUNDED BY THE MARGINALS: at HCG "
                         "~0.78 vs GCH ~0.25 it can only reach [-0.92, +0.31], "
                         "and a fixed odds ratio of 9 reads anywhere from 0.10 "
                         "to 0.50 depending on rates alone. 'yule_q' and "
                         "'log_or' are functions of the odds ratio and are "
                         "marginal-free, so they compare across contexts; "
                         "'phi_norm' rescales phi by its attainable bound. "
                         "ALL of them are written to the TSV regardless.")
    ap.add_argument("--plot-format", choices=["png", "pdf", "svg"], default="pdf")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from nome_hic_concordance.cli import _parse_context_bed, load_samples
    from nome_hic_concordance.reference import (extract_context_positions,
                                                load_context_from_bed)
    from nome_hic_concordance.utils import ensure_dir

    samples = load_samples(args.samples)
    if args.max_cells and args.max_cells > 0:
        samples = samples.iloc[: args.max_cells]
    if not args.context_bed and not args.reference:
        sys.exit("provide --context-bed or --reference.")
    if args.context_bed:
        bed = _parse_context_bed(args.context_bed)
        for need in ("GCH", "HCG"):
            if need not in bed:
                sys.exit(f"--context-bed must define {need}: (got {list(bed)}).")
        ctx_positions = load_context_from_bed(
            bed, n_threads=args.n_threads, c_offset=args.context_bed_offset)
        if args.dbsnp_vcf:
            ctx_positions = _drop_snp_sites(ctx_positions, args.dbsnp_vcf)
    else:
        # Step-1 extraction applies --dbsnp-vcf itself.
        ctx_positions = extract_context_positions(
            args.reference, str(Path(args.output_dir) / "context"),
            contexts=["GCH", "HCG"], dbsnp_vcf=args.dbsnp_vcf,
            n_threads=args.n_threads)

    blacklist = None
    if args.blacklist_bed:
        from nome_hic_concordance.blacklist import load_blacklist_bed
        blacklist = load_blacklist_bed(args.blacklist_bed)
        n_iv = sum(len(v) for v in blacklist.values())
        print(f"  blacklist: {n_iv:,} intervals over {len(blacklist)} chroms")
    out_dir = Path(ensure_dir(args.output_dir))
    edges = _build_edges(args.min_distance, args.max_distance,
                         args.bins_per_decade)
    centers = np.sqrt(edges[:-1] * edges[1:])
    use_cpp = _CPP is not None if args.backend == "auto" else args.backend == "htslib"
    if use_cpp and _CPP is None:
        sys.exit("--backend htslib requested but the extension is not built "
                 "(pip install -e . with htslib + pybind11 available).")
    backend = "htslib (C++)" if use_cpp else "python (pysam)"
    if not use_cpp and not args.max_pairs_per_cell:
        print("  NOTE: the pysam fallback on FULL BAMs is ~100x slower than the "
              "htslib backend; build the extension or set --max-pairs-per-cell.",
              file=sys.stderr)
    pairs_note = ("ALL pairs" if not args.max_pairs_per_cell
                  else f"~{args.max_pairs_per_cell:,} pairs/cell")
    pooled_null = args.null_scope == "pooled"
    if pooled_null and not use_cpp:
        print("  NOTE: --null-scope pooled needs the htslib backend; falling "
              "back to the within-cell null.", file=sys.stderr)
        pooled_null = False
    donor_cap = args.null_donors_per_cell if pooled_null else 0
    anchor_cap = args.null_anchors_per_cell if pooled_null else 0
    anchor_long_cap = args.null_anchors_long_per_cell if pooled_null else 0

    print(f"# {_QC}")
    print(f"[site-dist] {len(samples)} cells | {len(edges)-1} bins "
          f"{args.min_distance:,}–{args.max_distance:,} bp | null window ±"
          f"{args.null_window} bp | {pairs_note} | backend {backend}")
    if pooled_null:
        # ~50 B/donor and ~100 B/anchor, measured on the CSR layout below.
        mb = len(samples) * (donor_cap * 2 * 50
                             + (anchor_cap + anchor_long_cap) * 90) / 1e6
        print(f"  null scope: POOLED — substitute ends are drawn from every "
              f"cell in the group\n"
              f"    sampling {donor_cap:,} donor ends per cell, and anchors "
              f"STRATIFIED by span:\n"
              f"      <{args.null_long_range_bp:,}bp: up to {anchor_cap:,}/cell"
              f"   >={args.null_long_range_bp:,}bp: up to "
              f"{anchor_long_cap:,}/cell   (<={mb:,.0f} MB, a ceiling — a cell "
              f"with fewer long-range pairs than the cap uses less)")
    else:
        print("  null scope: WITHIN-CELL — substitute ends come only from the "
              "anchor's own cell")

    if use_cpp:
        prepared = prepare_ctx(ctx_positions, blacklist)
        n_sites = sum(len(v[0]) for v in prepared[0].values()) + \
            sum(len(v[0]) for v in prepared[1].values())
        print(f"  context prepared once: {n_sites:,} sites, shared across "
              f"threads (no per-worker copy)")
        jobs = [(row["bam_path"], prepared, edges, args.trim_ends,
                 args.mapq_min, args.max_site_pairs_per_read,
                 args.null_window, args.seed + i, args.baseq_min,
                 donor_cap, anchor_cap, anchor_long_cap,
                 args.null_long_range_bp, args.strand_min_evidence)
                for i, (_, row) in enumerate(samples.iterrows())]
        fn = process_cell_cpp
    else:
        if args.baseq_min or blacklist:
            print("  NOTE: --baseq-min / --blacklist-bed are applied by the "
                  "htslib backend only; the python fallback ignores them.",
                  file=sys.stderr)
        jobs = [(row["bam_path"], ctx_positions, edges, args.trim_ends,
                 args.mapq_min, args.max_pairs_per_cell,
                 args.max_site_pairs_per_read, args.null_window, args.seed + i)
                for i, (_, row) in enumerate(samples.iterrows())]
        fn = process_cell

    cell_names = samples["cell_name"].tolist()
    n_jobs_total = len(jobs)
    t0 = time.time()

    def _tick(done, idx, n_pairs):
        """One line per finished cell: which cell, how far along, and an ETA."""
        el = time.time() - t0
        eta = el / max(done, 1) * (n_jobs_total - done)
        print(f"  [{done}/{n_jobs_total}] {cell_names[idx]}: {n_pairs:,} pairs "
              f"| {el / 60:.1f} min elapsed, ETA {eta / 60:.1f} min",
              flush=True)

    if use_cpp and args.n_threads > 1 and n_jobs_total > 1:
        # THREADS: the extension releases the GIL, so they run truly in parallel
        # AND share the ~GB context tables (processes would each need a pickled
        # copy).  Driving the pool directly — rather than joblib — is what lets
        # us report "[i/N] <cell>" instead of joblib's opaque "Done N tasks".
        from concurrent.futures import ThreadPoolExecutor, as_completed
        res = [None] * n_jobs_total
        with ThreadPoolExecutor(max_workers=args.n_threads) as ex:
            futs = {ex.submit(fn, *j): i for i, j in enumerate(jobs)}
            for done, fut in enumerate(as_completed(futs), 1):
                i = futs[fut]
                res[i] = fut.result()
                _tick(done, i, res[i][1])
    elif args.n_threads > 1 and n_jobs_total > 1:
        from joblib import Parallel, delayed          # pysam fallback: processes
        res = Parallel(n_jobs=args.n_threads, verbose=10)(
            delayed(fn)(*j) for j in jobs)
    else:
        res = []
        for i, j in enumerate(jobs):
            res.append(fn(*j))
            _tick(i + 1, i, res[-1][1])
    print(f"  parsed {len(res)} cells in {(time.time() - t0) / 60:.1f} min "
          f"({sum(r[1] for r in res):,} read pairs total)")

    # A sub-read-length HCG-HCG pair (distance 1 = the two strands of one CpG
    # dyad) can ONLY form when the two mates disagree on conversion strand.  In
    # a directional library they should agree, so this rate is the denominator
    # for how much of that bin is real.
    sk = sum(r[4]["known"] for r in res)
    sd = sum(r[4]["disagree"] for r in res)
    if sk:
        print(f"  mate conversion strand: {sd:,}/{sk:,} pairs DISAGREE "
              f"({100.0 * sd / sk:.2f}%) at --strand-min-evidence "
              f"{args.strand_min_evidence}")
        print("    only these pairs can produce a sub-read-length HCG-HCG "
              "point (the CpG dyad),\n    so treat the shortest bins as a "
              "selected subset, not a population estimate.")

    keys = [(m, c) for m, _ in _MEASURES for c, _ in _CONDS]
    groups: dict[str, list[int]] = {"all": list(range(len(res)))}
    if args.group_column:
        if args.group_column not in samples.columns:
            sys.exit(f"--group-column {args.group_column!r} not in --samples "
                     f"(columns: {samples.columns.tolist()}).")
        for gi, g in enumerate(samples[args.group_column].to_numpy()):
            groups.setdefault(str(g), []).append(gi)

    summary = {}
    safe = safe_names(groups)          # 'IT-L2/3' -> 'IT-L2_3', collision-proof
    for label, idxs in groups.items():
        accs = {k: sum(res[i][0][k] for i in idxs) for k in keys}
        n_pairs = sum(res[i][1] for i in idxs)
        n_elig = sum(res[i][2] for i in idxs)
        if n_pairs == 0:
            print(f"\n=== {label}: no usable read pairs — skipped ===")
            continue

        # Per-cell accumulators, so the bootstrap can resample cells.  For the
        # within-cell null they come straight from the parse; for the pooled
        # null the matcher returns them keyed by the anchor's cell.
        pc = {(m, c): [res[i][0][(m, c)] for i in idxs]
              for m, _ in _MEASURES for c, _ in _CONDS}
        scope_note = ""
        if pooled_null:
            pooled = _pooled_null(res, idxs, edges, args, len(edges) - 1)
            if pooled is not None:
                for m, _ in _MEASURES:
                    pc[(m, "null")] = pooled["null_per_cell"][m]
                    pc[(m, "matched")] = pooled["matched_per_cell"][m]
                    accs[(m, "null")] = sum(pooled["null_per_cell"][m])
                    accs[(m, "matched")] = sum(pooled["matched_per_cell"][m])
                n_pairs_null = pooled["n_anchors"]
                n_elig = pooled["n_eligible"]
                seen_l = pooled["n_pairs_long_seen"]
                kept_l = pooled["n_anchor_long"]
                frac = f"{100.0 * kept_l / seen_l:.0f}%" if seen_l else "n/a"
                scope_note = (
                    f"pooled null: {pooled['n_donors']:,} donor ends from "
                    f"{len(idxs)} cells; {pooled['n_anchors']:,} anchors "
                    f"({kept_l:,} long-range = {frac} of the "
                    f"{seen_l:,} seen)")
                # eligibility is now relative to the sampled anchors
                n_pairs_for_pct = n_pairs_null
            else:
                n_pairs_for_pct = n_pairs
        else:
            n_pairs_for_pct = n_pairs

        boot = None
        if args.n_bootstrap > 0 and len(idxs) >= 3:
            boot = {}
            for m, _ in _MEASURES:
                boot[m] = _bootstrap_delta(pc[(m, "linked")], pc[(m, "null")],
                                           args.min_pairs, args.n_bootstrap,
                                           args.seed, args.metric)
            # FDR over every bin of every measure in THIS figure — that is the
            # family a reader looks at together.  Correcting per panel would
            # under-count by 3x; correcting across groups would be a different
            # (and much stricter) claim, so it is left to the caller.
            # NOT `keys` — that name holds the (measure, condition) tuples this
            # loop rebuilds `accs` from on EVERY iteration, so shadowing it here
            # let the 'all' figure succeed and then killed the first per-group
            # one with KeyError: 'gg'.
            mkeys = [k for k, _ in _MEASURES if boot.get(k) is not None]
            if mkeys:
                flat = np.concatenate([boot[k][2] for k in mkeys])
                n_test = int(np.isfinite(flat).sum())
                floor = 2.0 / (args.n_bootstrap + 1.0)
                if args.fdr != "none" and n_test and floor * n_test > 0.05:
                    print(f"  NOTE: {args.n_bootstrap:,} resamples floor the "
                          f"two-sided p at {floor:.4f}; over {n_test} tests "
                          f"that is q>={floor * n_test:.3f} at rank 1, so an "
                          f"ISOLATED significant bin cannot reach q<0.05.\n"
                          f"        Pervasive effects still pass. Raise "
                          f"--n-bootstrap to ~{int(np.ceil(2 * n_test / 0.05)):,} "
                          f"to resolve isolated ones.")
                q = _fdr(flat, args.fdr)
                off = 0
                for k in mkeys:
                    nb_k = len(boot[k][2])
                    lo_, hi_, p_ = boot[k]
                    boot[k] = (lo_, hi_, p_, q[off:off + nb_k])
                    off += nb_k
        _report(accs, len(idxs), n_pairs, centers, edges, label, out_dir, args,
                safe=safe[label], n_eligible=n_elig, boot=boot,
                scope_note=scope_note, n_pairs_for_pct=n_pairs_for_pct)
        summary[label] = {"n_cells": len(idxs), "n_read_pairs": int(n_pairs),
                          "null_scope": "pooled" if pooled_null else "cell",
                          "n_null_eligible": int(n_elig),
                          "null_eligible_frac": (round(n_elig / n_pairs_for_pct, 4)
                                                 if n_pairs_for_pct else None),
                          "file_stem": f"site_distance_corr_{safe[label]}"}
    with open(out_dir / "site_distance_corr_summary.json", "w") as fh:
        json.dump({"groups": summary, "qc": _QC,
                   "null_window_bp": args.null_window}, fh, indent=2)
    print("\nRead the figures: below ~read length the pairs are WITHIN-read, which "
          "has no\n  non-linked counterpart (both cytosines are on one molecule), "
          "so the null is\n  blank there and the linked curve alone is the 1-D "
          "co-methylation decay.  At\n  longer range a curve sitting above "
          "its non-linked null is same-molecule\n  coordination rather than mere "
          "genomic proximity.\n"
          "\n  Compare the null against MATCHED, not against LINKED.  A pair can "
          "only enter\n  the null if some other end falls within ±window of its "
          "partner; at single-cell\n  coverage most pairs have no such neighbour "
          "and are EXCLUDED (not NaN), and the\n  ones that survive sit in "
          "read-dense loci.  LINKED is all pairs, so linked-vs-null\n  also "
          "compares two different sets of loci.  MATCHED is the true partner on "
          "exactly\n  the null-eligible pairs, cross-end only, so matched-vs-null "
          "differs in one thing:\n  whether the partner was the real mate.")


if __name__ == "__main__":
    main()
