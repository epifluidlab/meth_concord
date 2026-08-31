#!/usr/bin/env python3
"""Read-PAIR level correlation vs genomic distance, with a re-pairing null.

The companion script, scripts/site_distance_correlation.py, correlates
individual cytosines: one site on end 1 against one site on end 2, binned by
the distance between THOSE TWO SITES, so a pair carrying five GpC per end
contributes twenty-five observations spread over several bins.  That is the
right unit for asking how far methylation state travels along the genome.

This one asks a different question, and the unit is the read pair.  Each
high-quality cis pair gives ONE observation: the mean call on end 1 against
the mean call on end 2, binned by the distance between the two END CENTRES.
That is the unit a Hi-C contact actually is, and the unit the concordance
pipeline works in.

Four panels.  Each reports BOTH a Pearson r and a Spearman rho over the
pairs in a distance bin -- these are bounded, skewed quantities, a posterior
piles up near 0 and 1, and a handful of pairs at the corners can carry a
Pearson that a rank correlation will not:

  GCH-GCH   x = mean GpC call on end 1, y = the same on end 2
  HCG-HCG   x = mean CpG call on end 1, y = the same on end 2
  GCH-GCH vs HCG-HCG
            restricted to pairs carrying BOTH contexts on BOTH ends,
            x = (GpC end 1) - (GpC end 2), y = (CpG end 1) - (CpG end 2).
            Correlating the two DIFFERENCES asks whether the accessibility
            gradient across a contact tracks the methylation gradient.

  GCH-HCG   accessibility at ONE end against methylation at the OTHER.
            Needs GpC on one end and CpG on the other, which is weaker than
            the panel above -- a pair with GpC only on end 1 and CpG only on
            end 2 qualifies here and not there.  Both orientations are
            emitted where both exist, so a pair carrying both contexts on
            both ends contributes two observations here and one to every
            other panel, and its n runs higher.

THE P-VALUE TO READ IS THE EMPIRICAL ONE.  scipy's p for either correlation
assumes the observations are independent draws; many here come from the same
cell, which is the very reason the null is per cell.  The empirical p counts
how often a re-pairing reaches the observed correlation, so every one of
those dependencies is in it.  Both are written to the TSV and the difference
is often large -- on one synthetic run the cross-context panel came back at
a parametric 0.023 and an empirical 0.216.

THE NULL IS A RE-PAIRING, WITHIN CELL AND WITHIN DISTANCE BIN.  The y column
is permuted among the pairs of one (cell, bin), so end 1 of one pair meets
end 2 of another that was the same distance away in the same cell.  Both
marginal distributions survive exactly; only the link is destroyed.

  WITHIN CELL, because cells differ in global accessibility and methylation.
  Pooling across cells makes end 1 and end 2 correlate simply by both being
  high in an open cell, and a null that shuffled across cells would call
  that correlation real.  Permuting inside a cell leaves it in the null,
  where it belongs.

  WITHIN DISTANCE BIN, because the bin is the axis.  A null drawn from other
  bins would answer "is this contact more correlated than a contact at some
  other distance", which is a different and much easier question.

Read filters match bam_parser exactly: uniquely mapped, not a PCR duplicate,
no supplementary alignment, mapQ above the threshold on BOTH ends, both ends
on the same chromosome.  They come from the same site_calls module the
pipeline uses, so this script and Step 1 see the same reads.

WHAT THE VALUES ARE.  Two inputs, and they differ in what an end is worth:

  --context-bed / --reference   a BAM pass, and the value is the MEAN of the
      0/1 calls on the end -- the fraction of its cytosines called accessible
      (GpC) or methylated (CpG).  An end with one site contributes a bare 0
      or 1, which --min-sites-per-end trades pairs to avoid.

  --posteriors                  Step 4's own output, and the value is the
      mixture model's POSTERIOR for the end.  No BAM pass and no context
      table: the site calling and the model already happened.  The third
      panel joins the two contexts on (chrom, pos1, pos2), which is exactly
      "pairs carrying both contexts on both ends", while the first two use
      each context's table whole and keep the pairs the join drops.

--exclude-bed drops a pair when EITHER end lands in one of the given regions
-- a blacklist, a CGI+shore track, or both, merged.  Either end
disqualifies it: a contact is a statement about two loci, so one end in a
CpG island makes the pair a comparison between an island and something else
rather than half a usable observation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_MEASURES = (("gg", "GCH–GCH  r(end1, end2)"),
             ("hh", "HCG–HCG  r(end1, end2)"),
             ("gh", "GCH–GCH vs HCG–HCG  r(ΔGpC, ΔCpG)"),
             ("gx", "GCH–HCG  r(GpC one end, CpG the other)"))


def pair_values(calls1, calls2, min_sites=1):
    """Per-END mean call for each context, NaN where THAT end is short.

    Independently per end, not per pair.  The cross-context measure needs
    GpC on one end and CpG on the other, so blanking a context on both ends
    because one of them lacks it would throw away exactly the pairs that
    measure exists for.  The both-ends requirement belongs to the measures
    that actually have it, and is applied in ``measures_of``.
    """
    def _one(c, ctx):
        v = c.get(ctx)
        if v is None or len(v[1]) < min_sites:
            return np.nan
        return float(np.mean(v[1]))

    return {ctx: (_one(calls1, ctx), _one(calls2, ctx))
            for ctx in ("GCH", "HCG")}


def measures_of(vals):
    """The three (x, y) observations a pair contributes, NaN where ineligible.

    'gg' and 'hh' need their own context on both ends.  'gh' needs BOTH
    contexts on BOTH ends -- that is the question, and a pair missing either
    cannot answer it -- and is a difference rather than a level, so the two
    ends must be ordered consistently.  They are: end 1 is the leftmost, set
    by the caller from the midpoints.
    """
    g1, g2 = vals["GCH"]
    h1, h2 = vals["HCG"]
    out = {"gg": (g1, g2) if np.isfinite([g1, g2]).all()
           else (np.nan, np.nan),
           "hh": (h1, h2) if np.isfinite([h1, h2]).all()
           else (np.nan, np.nan)}
    if np.isfinite([g1, g2, h1, h2]).all():
        out["gh"] = (g1 - g2, h1 - h2)
    else:
        out["gh"] = (np.nan, np.nan)
    # CROSS-CONTEXT, ACROSS ENDS: accessibility at one locus against
    # methylation at the OTHER.  Needs GpC on one end and CpG on the other,
    # which is a weaker requirement than 'gh' -- a pair with GpC only on
    # end 1 and CpG only on end 2 qualifies here and not there.  Both
    # orientations are emitted where both exist, matching the convention in
    # scripts/site_distance_correlation.py; a pair carrying both contexts on
    # both ends therefore contributes two observations to this measure and
    # one to every other, which is why its n runs higher.
    cross = []
    if np.isfinite(g1) and np.isfinite(h2):
        cross.append((g1, h2))
    if np.isfinite(g2) and np.isfinite(h1):
        cross.append((g2, h1))
    out["gx"] = cross or [(np.nan, np.nan)]
    return out


def load_exclude(paths):
    """Merged intervals per chromosome, from one or more BED files."""
    if not paths:
        return None
    import gzip
    by_chrom = {}
    for path in paths:
        op = gzip.open if str(path).endswith(".gz") else open
        with op(path, "rt") as fh:
            for line in fh:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                f = line.split()
                if len(f) < 3:
                    continue
                by_chrom.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    out = {}
    for c, iv in by_chrom.items():
        iv.sort()
        merged = []
        for lo, hi in iv:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        out[c] = (np.array([m[0] for m in merged], np.int64),
                  np.array([m[1] for m in merged], np.int64))
    return out


def excluded(regions, chrom, positions):
    """True where a position falls in any excluded interval.

    EITHER END DISQUALIFIES THE PAIR.  A contact is a statement about two
    loci, so a pair with one end in a CpG island is not half usable -- its
    concordance is a comparison between an island and something else, and
    keeping it would put that comparison in every distance bin the island's
    partners happen to land in.
    """
    if regions is None:
        return np.zeros(len(positions), bool)
    got = regions.get(str(chrom))
    if got is None:
        return np.zeros(len(positions), bool)
    starts, ends = got
    pos = np.asarray(positions, np.int64)
    i = np.searchsorted(starts, pos, side="right") - 1
    ok = i >= 0
    out = np.zeros(pos.shape, bool)
    out[ok] = pos[ok] < ends[i[ok]]
    return out


def collect_cell_posteriors(paths, edges, min_sites, exclude):
    """One cell → {measure: (bins, x, y)} from Step-4 posteriors.

    THE VALUES ARE THE MIXTURE MODEL'S POSTERIORS, not per-end means of 0/1
    calls: this reads what Step 4 wrote rather than re-reading the BAM, so
    an end with two sites carries the model's belief about it instead of a
    bare 0, 0.5 or 1.

    ``paths`` is {context: parquet}.  The third measure needs the SAME read
    pair in both contexts, so the two tables are joined on (chrom, pos1,
    pos2) -- Step 4 writes one row per pair per context, and the join is
    exactly "pairs carrying both contexts on both ends".  The first two
    measures use their own context's table whole, so they keep the pairs the
    join would drop.
    """
    nb = len(edges) - 1
    cols = ["chrom", "pos1", "pos2", "n1", "n2", "posterior1", "posterior2"]
    tabs = {}
    for ctx, path in paths.items():
        if path is None or not Path(path).exists():
            continue
        t = pd.read_parquet(path, columns=cols)
        t = t[(t["n1"] >= min_sites) & (t["n2"] >= min_sites)]
        t = t[np.isfinite(t["posterior1"]) & np.isfinite(t["posterior2"])]
        if exclude is not None and len(t):
            drop = np.zeros(len(t), bool)
            for c, sub in t.groupby("chrom", sort=False):
                idx = sub.index
                bad = (excluded(exclude, c, sub["pos1"].to_numpy())
                       | excluded(exclude, c, sub["pos2"].to_numpy()))
                drop[t.index.get_indexer(idx)] = bad
            t = t[~drop]
        tabs[ctx] = t.reset_index(drop=True)

    def _binned(t):
        d = np.abs(t["pos2"].to_numpy(np.int64) - t["pos1"].to_numpy(np.int64))
        keep = (d >= edges[0]) & (d <= edges[-1])
        b = np.clip(np.digitize(d[keep], edges) - 1, 0, nb - 1).astype(np.int64)
        return keep, b

    got = {}
    for m, ctx in (("gg", "GCH"), ("hh", "HCG")):
        t = tabs.get(ctx)
        if t is None or not len(t):
            got[m] = (np.empty(0, np.int64), np.empty(0), np.empty(0))
            continue
        keep, b = _binned(t)
        # Leftmost end first, so a difference's sign is consistent later.
        p1 = t["posterior1"].to_numpy(float)[keep]
        p2 = t["posterior2"].to_numpy(float)[keep]
        lo = t["pos1"].to_numpy(np.int64)[keep] <= t["pos2"].to_numpy(np.int64)[keep]
        got[m] = (b, np.where(lo, p1, p2), np.where(lo, p2, p1))

    g, h = tabs.get("GCH"), tabs.get("HCG")
    if g is not None and h is not None and len(g) and len(h):
        j = g.merge(h, on=["chrom", "pos1", "pos2"], suffixes=("_g", "_h"))
        if len(j):
            keep, b = _binned(j)
            lo = (j["pos1"].to_numpy(np.int64)[keep]
                  <= j["pos2"].to_numpy(np.int64)[keep])
            gd = (j["posterior1_g"].to_numpy(float)[keep]
                  - j["posterior2_g"].to_numpy(float)[keep])
            hd = (j["posterior1_h"].to_numpy(float)[keep]
                  - j["posterior2_h"].to_numpy(float)[keep])
            got["gh"] = (b, np.where(lo, gd, -gd), np.where(lo, hd, -hd))
        else:
            got["gh"] = (np.empty(0, np.int64), np.empty(0), np.empty(0))
    else:
        got["gh"] = (np.empty(0, np.int64), np.empty(0), np.empty(0))
    # GCH one end against HCG the OTHER.  Step 4 writes a row per pair per
    # context, so a pair qualifies as soon as it appears in both tables --
    # the same join as 'gh', but the value taken across ends rather than
    # within each.  Both orientations, so a pair contributes two rows.
    if g is not None and h is not None and len(g) and len(h):
        j = g.merge(h, on=["chrom", "pos1", "pos2"], suffixes=("_g", "_h"))
        if len(j):
            keep, b = _binned(j)
            g1 = j["posterior1_g"].to_numpy(float)[keep]
            g2 = j["posterior2_g"].to_numpy(float)[keep]
            h1 = j["posterior1_h"].to_numpy(float)[keep]
            h2 = j["posterior2_h"].to_numpy(float)[keep]
            got["gx"] = (np.concatenate([b, b]),
                         np.concatenate([g1, g2]),
                         np.concatenate([h2, h1]))
        else:
            got["gx"] = (np.empty(0, np.int64), np.empty(0), np.empty(0))
    else:
        got["gx"] = (np.empty(0, np.int64), np.empty(0), np.empty(0))
    n_pairs = max((len(t) for t in tabs.values()), default=0)
    return got, n_pairs


def collect_cell(bam_path, ctx_positions, edges, trim_ends, mapq_min,
                 max_pairs, min_sites, seed, exclude=None):
    """One cell → {measure: [(bin, x, y), ...]}.

    The per-pair values are KEPT rather than streamed into running sums,
    because the null permutes them within (cell, bin) and a permutation
    cannot be done on a sum.  One float pair per read pair per measure, so
    the cost is linear and small beside the BAM pass.
    """
    from nome_hic_concordance.site_calls import iter_cell_read_pairs

    rng = np.random.default_rng(seed)
    nb = len(edges) - 1
    got = {m: ([], [], []) for m, _ in _MEASURES}
    n_pairs = 0
    for _chrom, mid1, mid2, c1, c2 in iter_cell_read_pairs(
            bam_path, ctx_positions, trim_ends=trim_ends, mapq_min=mapq_min,
            max_pairs_per_cell=max_pairs, rng=rng):
        if exclude is not None and bool(
                excluded(exclude, _chrom, [mid1, mid2]).any()):
            continue
        d = abs(int(mid2) - int(mid1))
        if d < edges[0] or d > edges[-1]:
            continue
        b = int(np.clip(np.digitize([d], edges)[0] - 1, 0, nb - 1))
        # Leftmost end first, so the sign of a difference means the same
        # thing in every pair.
        if mid1 <= mid2:
            vals = pair_values(c1, c2, min_sites)
        else:
            vals = pair_values(c2, c1, min_sites)
        n_pairs += 1
        for m, v in measures_of(vals).items():
            for x, y in (v if isinstance(v, list) else [v]):
                if np.isfinite(x) and np.isfinite(y):
                    got[m][0].append(b)
                    got[m][1].append(x)
                    got[m][2].append(y)
    return ({m: (np.asarray(v[0], np.int64), np.asarray(v[1], float),
                 np.asarray(v[2], float)) for m, v in got.items()}, n_pairs)


def shuffle_within(bins, y, nb, rng):
    """Permute y among the entries sharing a bin.  Cell scope is the caller's."""
    out = y.copy()
    for b in range(nb):
        idx = np.flatnonzero(bins == b)
        if idx.size > 1:
            out[idx] = y[rng.permutation(idx)]
    return out


def _running(bins, x, y, nb):
    """(6, nb) sums for a Pearson."""
    acc = np.zeros((6, nb))
    if bins.size:
        acc[0] = np.bincount(bins, minlength=nb)
        acc[1] = np.bincount(bins, weights=x, minlength=nb)
        acc[2] = np.bincount(bins, weights=y, minlength=nb)
        acc[3] = np.bincount(bins, weights=x * x, minlength=nb)
        acc[4] = np.bincount(bins, weights=y * y, minlength=nb)
        acc[5] = np.bincount(bins, weights=x * y, minlength=nb)
    return acc


def corr_of(acc, min_pairs):
    """(6, nb) sums → (r, n); NaN where n < min_pairs or either side is flat."""
    n, sx, sy, sxx, syy, sxy = acc
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = np.where(n > 0, n, 1.0)
        cov = sxy - sx * sy / nz
        vx = sxx - sx * sx / nz
        vy = syy - sy * sy / nz
        den = np.sqrt(vx * vy)
        r = np.where(den > 0, cov / np.where(den > 0, den, 1.0), np.nan)
    return np.where(n >= min_pairs, r, np.nan), n


def pool_cells(per_cell, nb):
    """Stack every cell's observations, keeping which cell each came from.

    The raw values are kept rather than folded into running sums, because
    Spearman needs ranks and the null needs a permutation -- neither can be
    done on a sum.  The cell index rides along because the permutation is
    per (cell, bin).
    """
    out = {}
    for m, _ in _MEASURES:
        cid, bb, xx, yy = [], [], [], []
        for ci, cell in enumerate(per_cell):
            b, x, y = cell[m]
            if b.size:
                cid.append(np.full(b.size, ci, np.int32))
                bb.append(b); xx.append(x); yy.append(y)
        out[m] = ((np.concatenate(cid), np.concatenate(bb),
                   np.concatenate(xx), np.concatenate(yy))
                  if bb else (np.empty(0, np.int32), np.empty(0, np.int64),
                              np.empty(0), np.empty(0)))
    return out


def _pearson(x, y, n, sx, sxx, sy, syy):
    """Pearson r from precomputed marginal sums.  One dot product.

    Σy and Σyy are INVARIANT under a permutation of y -- it only reorders
    the same values -- so a permutation loop can hoist them out and touch
    nothing but Σxy.  scipy's pearsonr recomputes all five and a p-value
    that the null discards.
    """
    sxy = float(np.dot(x, y))
    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    return num / den if den > 0 else np.nan


def _bin_result(xb, yb, groups, n_perm, min_pairs, seed):
    """Observed and permuted correlations for ONE distance bin.

    SPEARMAN IS PEARSON ON RANKS, and a permutation of y permutes its ranks
    the same way -- so the ranks are computed ONCE and every null Spearman
    is another dot product.  Ranking inside the loop instead was the whole
    cost: measured on 200,000 observations, scipy's spearmanr takes 41 ms
    against 3.2 ms for the dot, and it was being called n_perm x n_bins
    times.
    """
    from scipy.stats import pearsonr, rankdata, spearmanr
    out = {k: np.nan for k in ("r", "p", "rho", "p_rho", "r_null",
                               "rho_null", "emp_p", "emp_p_rho")}
    out["n"] = float(xb.size)
    if xb.size < max(3, min_pairs) or np.ptp(xb) == 0 or np.ptp(yb) == 0:
        return out
    r, pr = pearsonr(xb, yb)
    rho, prho = spearmanr(xb, yb)
    out["r"], out["p"] = float(r), float(pr)
    out["rho"], out["p_rho"] = float(rho), float(prho)
    if int(n_perm) <= 0:
        return out

    rxb, ryb = rankdata(xb), rankdata(yb)
    n = float(xb.size)
    sx, sxx = float(xb.sum()), float((xb * xb).sum())
    sy, syy = float(yb.sum()), float((yb * yb).sum())
    srx, srxx = float(rxb.sum()), float((rxb * rxb).sum())
    sry, sryy = float(ryb.sum()), float((ryb * ryb).sum())

    rng = np.random.default_rng(seed)
    perm = np.arange(xb.size)
    acc_r = acc_rho = 0.0
    hit_r = hit_rho = 0
    seen = 0
    big = [g for g in groups if g.size > 1]
    for _ in range(int(n_perm)):
        for g in big:
            perm[g] = rng.permutation(g)
        rr = _pearson(xb, yb[perm], n, sx, sxx, sy, syy)
        rr_s = _pearson(rxb, ryb[perm], n, srx, srxx, sry, sryy)
        if np.isfinite(rr):
            acc_r += rr
            hit_r += int(abs(rr) >= abs(out["r"]))
        if np.isfinite(rr_s):
            acc_rho += rr_s
            hit_rho += int(abs(rr_s) >= abs(out["rho"]))
        seen += 1
    if seen:
        out["r_null"] = acc_r / seen
        out["rho_null"] = acc_rho / seen
        # +1 on both sides: beyond every re-pairing is not impossible, only
        # beyond what N of them reached.
        out["emp_p"] = (hit_r + 1.0) / (seen + 1.0)
        out["emp_p_rho"] = (hit_rho + 1.0) / (seen + 1.0)
    return out


def correlations(pooled, nb, n_perm, min_pairs, seed, n_jobs=1):
    """Observed correlations, the re-paired null, and an EMPIRICAL p.

    THE PARAMETRIC P IS NOT THE ONE TO READ, and both are reported so the
    difference is visible.  scipy's p assumes the observations are
    independent draws; here many come from the same cell, and cells differ
    in global level, so it is anti-conservative for exactly the reason the
    null is per cell.  The empirical p counts how often a re-pairing --
    within cell and within bin, so every one of those dependencies survives
    -- reaches the observed correlation.  Its floor is 1/(n_perm + 1).

    Spearman is reported beside Pearson because these are bounded, skewed
    quantities: a posterior piles up near 0 and 1, and a handful of pairs at
    the corners can carry a Pearson that a rank correlation will not.

    Every (measure, bin) is independent, so ``n_jobs`` fans them out.  The
    per-cell read is not the only parallel phase any more: with four
    measures and a handful of bins there is real work here, and it used to
    run on one core while the rest idled.
    """
    tasks, keys = [], []
    for m, _ in _MEASURES:
        cid, bins, x, y = pooled[m]
        for b in range(nb):
            if not bins.size:
                tasks.append((np.empty(0), np.empty(0), [], n_perm,
                              min_pairs, seed + b))
                keys.append((m, b))
                continue
            sel = np.flatnonzero(bins == b)
            xb, yb = x[sel], y[sel]
            # Permutation groups: positions WITHIN this bin, one per cell.
            cb = cid[sel]
            o = np.argsort(cb, kind="stable")
            cs = cb[o]
            starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
            groups = np.split(o, starts[1:]) if o.size else []
            tasks.append((xb, yb, groups, n_perm, min_pairs, seed + b))
            keys.append((m, b))
    if int(n_jobs) > 1 and len(tasks) > 1:
        from joblib import Parallel, delayed
        got = Parallel(n_jobs=int(n_jobs), prefer="processes")(
            delayed(_bin_result)(*t) for t in tasks)
    else:
        got = [_bin_result(*t) for t in tasks]

    out = {m: {k: np.full(nb, np.nan) for k in
               ("r", "p", "rho", "p_rho", "r_null", "rho_null",
                "emp_p", "emp_p_rho")} for m, _ in _MEASURES}
    for m, _ in _MEASURES:
        out[m]["n"] = np.zeros(nb)
    for (m, b), d in zip(keys, got):
        for k, v in d.items():
            out[m][k][b] = v
    return out


def figure(path, centers, res, title):
    """Two rows: the correlations themselves, then linked minus null.

    Pearson solid, Spearman dashed, in the same colour -- they answer the
    same question on different scales, and putting them side by side would
    invite reading a gap between panels as a gap between distances.  A ring
    marks a bin whose EMPIRICAL p clears 0.05: the parametric p is in the
    TSV but is anti-conservative here, since many observations come from the
    same cell.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"gg": "#1f77b4", "hh": "#d62728", "gh": "#7b3f9e",
              "gx": "#2ca02c"}
    n = len(_MEASURES)
    fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 8.4), sharex=True,
                             squeeze=False)
    for col, (key, ttl) in enumerate(_MEASURES):
        d = res[key]
        c = colors[key]
        a = axes[0][col]
        a.axhline(0, color="0.6", lw=0.8, zorder=0)
        a.plot(centers, d["r"], "-o", ms=3, lw=1.6, color=c, label="Pearson r")
        a.plot(centers, d["rho"], "--s", ms=3, lw=1.4, color=c, alpha=0.75,
               label="Spearman ρ")
        a.plot(centers, d["r_null"], ":", lw=1.3, color="0.35",
               label="null, Pearson")
        a.plot(centers, d["rho_null"], ":", lw=1.0, color="0.6",
               label="null, Spearman")
        sig = np.isfinite(d["emp_p"]) & (d["emp_p"] <= 0.05)
        if sig.any():
            a.plot(np.asarray(centers)[sig], d["r"][sig], "o", ms=8,
                   mfc="none", mec=c, mew=1.4, zorder=6)
        a.set_xscale("log")
        a.set_title(ttl.split("  ")[0], fontsize=10)
        if col == 0:
            a.set_ylabel("correlation over read pairs", fontsize=9)
        a.legend(frameon=False, fontsize=7)

        b = axes[1][col]
        b.axhline(0, color="0.6", lw=0.8, zorder=0)
        b.plot(centers, d["r"] - d["r_null"], "-o", ms=3, lw=1.6, color=c,
               label="Δ Pearson")
        b.plot(centers, d["rho"] - d["rho_null"], "--s", ms=3, lw=1.4,
               color=c, alpha=0.75, label="Δ Spearman")
        b.set_xscale("log")
        b.set_xlabel("distance between end centres (bp)", fontsize=9)
        if col == 0:
            b.set_ylabel("Δ = linked − re-paired null", fontsize=9)
        b.legend(frameon=False, fontsize=7)
        tw = b.twinx()
        tw.plot(centers, d["n"], ":", lw=1.0, color="0.6")
        tw.set_yscale("log")
        tw.tick_params(labelsize=6)
        tw.set_ylabel("observations" if col == n - 1 else "", fontsize=7)
        if col != n - 1:
            tw.set_yticklabels([])
    for ax in axes.ravel():
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"{title}   (rings: empirical p ≤ 0.05)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True,
                    help="TSV with cell_name and bam_path.")
    ap.add_argument("--context-bed", default=None,
                    help="'GCH:/path/GCH.bed.gz,HCG:/path/HCG.bed.gz' -- the "
                         "same CONTEXT:PATH form as nome_hic_concordance and "
                         "scripts/site_distance_correlation.py.  Either this "
                         "or --reference is required, and BOTH contexts are "
                         "needed: every panel uses one of them and the third "
                         "uses both.")
    ap.add_argument("--reference", default=None,
                    help=".2bit genome, if --context-bed is not given.")
    ap.add_argument("--context-bed-offset", type=int, default=0)
    ap.add_argument("--dbsnp-vcf", default=None)
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--posteriors", default=None,
                    metavar="GCH:DIR,HCG:DIR",
                    help="Read Step-4 posteriors instead of the BAMs: "
                         "'GCH:.../context_GCH/04_posteriors,"
                         "HCG:.../context_HCG/04_posteriors'.  The values "
                         "are then the mixture model's posterior per END, "
                         "not the per-end mean of 0/1 calls, so an end with "
                         "two sites carries the model's belief rather than a "
                         "bare 0, 0.5 or 1.  The third panel needs the SAME "
                         "read pair in both contexts, so the two tables are "
                         "joined on (chrom, pos1, pos2) -- Step 4 writes one "
                         "row per pair per context and that join IS 'pairs "
                         "carrying both contexts on both ends'.  The first "
                         "two panels use their own table whole, so they keep "
                         "the pairs the join drops.  Cell names come from "
                         "--samples and the files are <cell>.posteriors."
                         "parquet.")
    ap.add_argument("--exclude-bed", action="append", default=None,
                    metavar="PATH",
                    help="Drop a read pair if EITHER end falls in one of "
                         "these regions.  Repeatable, and the intervals are "
                         "merged across files, so a blacklist and a CGI+shore "
                         "track can be given together.  Either end "
                         "disqualifies the pair: a contact is a statement "
                         "about two loci, so one end in a CpG island makes "
                         "the pair a comparison between an island and "
                         "something else rather than half a usable "
                         "observation.")
    ap.add_argument("--min-dist", type=float, default=1e2)
    ap.add_argument("--max-dist", type=float, default=1e7)
    ap.add_argument("--bins-per-decade", type=int, default=1,
                    help="1 (default) gives the decade bins 100-1,000, "
                         "1,000-10,000 and so on.  Raise it for a finer "
                         "axis, at the cost of pairs per bin.")
    ap.add_argument("--min-sites-per-end", type=int, default=1,
                    help="An end with one site contributes a bare 0 or 1; "
                         "raising this trades pairs for a less granular "
                         "value.  Applied per CONTEXT, so the third panel's "
                         "eligible set is the intersection.")
    ap.add_argument("--trim-ends", type=int, default=5)
    ap.add_argument("--mapq-min", type=int, default=30)
    ap.add_argument("--max-pairs-per-cell", type=int, default=0)
    ap.add_argument("--min-pairs", type=int, default=50, metavar="N",
                    help="Bins with fewer eligible pairs report NaN rather "
                         "than a correlation of three points.")
    ap.add_argument("--n-permutations", type=int, default=200, metavar="N",
                    help="Re-pairings, used for BOTH the null curve and the "
                         "empirical p.  The p is what to read: the "
                         "parametric one assumes independent observations "
                         "and many here come from the same cell, so it is "
                         "anti-conservative for exactly the reason the null "
                         "is per cell.  Its floor is 1/(N + 1), so 200 "
                         "resolves 0.005; drop it to 20 for a quick look.")
    ap.add_argument("--exclude-chrom", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--plot-format", default="pdf",
                    choices=["pdf", "png", "svg"])
    ap.add_argument("--title", default="Read-pair correlation vs distance")
    args = ap.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_edges = int(round(np.log10(args.max_dist / args.min_dist)
                        * args.bins_per_decade)) + 1
    edges = np.logspace(np.log10(args.min_dist), np.log10(args.max_dist),
                        max(2, n_edges))
    centers = np.sqrt(edges[:-1] * edges[1:])
    nb = len(edges) - 1

    # Same loaders as scripts/site_distance_correlation.py and Step 1, so the
    # two scripts see the same sites and the same reads.
    from nome_hic_concordance.cli import _parse_context_bed, load_samples
    from nome_hic_concordance.reference import (extract_context_positions,
                                                load_context_from_bed)

    samples = load_samples(args.samples)
    if args.max_cells and args.max_cells > 0:
        samples = samples.iloc[: args.max_cells]
    if args.posteriors:
        # Step 4 already did the site calling and the mixture model, so no
        # context table and no BAM pass are needed here.
        pdirs = _parse_context_bed(args.posteriors)
        for need in ("GCH", "HCG"):
            if need not in pdirs:
                sys.exit(f"--posteriors must define {need}: the third panel "
                         f"joins the two contexts on the same read pair "
                         f"(got {list(pdirs)}).")
        ctx_positions = None
    elif args.context_bed:
        bed = _parse_context_bed(args.context_bed)
        for need in ("GCH", "HCG"):
            if need not in bed:
                sys.exit(f"--context-bed must define {need}: all three "
                         f"panels need both contexts (got {list(bed)}).")
        ctx_positions = load_context_from_bed(
            bed, n_threads=args.threads,
            c_offset=args.context_bed_offset)
    elif args.reference:
        ctx_positions = extract_context_positions(
            args.reference, str(out / "context"), contexts=["GCH", "HCG"],
            dbsnp_vcf=args.dbsnp_vcf, n_threads=args.threads)
    else:
        sys.exit("provide --posteriors, --context-bed or --reference.")
    if args.exclude_chrom and ctx_positions is not None:
        drop = {c.strip() for c in args.exclude_chrom.split(",") if c.strip()}
        ctx_positions = {k: {c: v for c, v in d.items() if c not in drop}
                         for k, d in ctx_positions.items()}

    exclude = load_exclude(args.exclude_bed)
    if exclude is not None:
        n_iv = sum(len(v[0]) for v in exclude.values())
        n_bp = sum(int((v[1] - v[0]).sum()) for v in exclude.values())
        print(f"  --exclude-bed: {n_iv:,} merged intervals over "
              f"{len(exclude)} contigs, {n_bp / 1e6:.1f} Mb; a pair is "
              f"dropped if EITHER end lands in one", file=sys.stderr)
    print(f"  values: "
          + ("Step-4 posteriors per end (--posteriors)"
             if args.posteriors else
             "per-end mean of the 0/1 site calls (BAM pass)"),
          file=sys.stderr)
    print(f"  {len(samples)} cells, {nb} distance bins "
          f"({args.min_dist:g}-{args.max_dist:g} bp, "
          f"{args.bins_per_decade}/decade)", file=sys.stderr)
    if args.posteriors:
        _pd = _parse_context_bed(args.posteriors)
        jobs, missing = [], []
        for _, row in samples.iterrows():
            paths = {c: Path(d) / f"{row['cell_name']}.posteriors.parquet"
                     for c, d in _pd.items()}
            if not any(v.exists() for v in paths.values()):
                missing.append(row["cell_name"])
                continue
            jobs.append((paths, edges, args.min_sites_per_end, exclude))
        if missing:
            print(f"  {len(missing)} of {len(samples)} cells have no "
                  f"posterior file and are skipped (first: "
                  f"{missing[0]})", file=sys.stderr)
        if not jobs:
            sys.exit("--posteriors: no <cell>.posteriors.parquet found for "
                     "any cell in --samples")
        fn = collect_cell_posteriors
    else:
        jobs = [(row["bam_path"], ctx_positions, edges, args.trim_ends,
                 args.mapq_min, args.max_pairs_per_cell,
                 args.min_sites_per_end, args.seed + i, exclude)
                for i, (_, row) in enumerate(samples.iterrows())]
        fn = collect_cell
    if args.threads > 1:
        from joblib import Parallel, delayed
        got = Parallel(n_jobs=args.threads)(delayed(fn)(*j) for j in jobs)
    else:
        got = [fn(*j) for j in jobs]
    per_cell = [g[0] for g in got]
    n_pairs_total = sum(g[1] for g in got)

    pooled = pool_cells(per_cell, nb)
    n_used = sum(1 for c in per_cell
                 if any(c[m][0].size for m, _ in _MEASURES))
    res = correlations(pooled, nb, args.n_permutations, args.min_pairs,
                       args.seed, n_jobs=args.threads)

    rows = {"bin_lo": edges[:-1], "bin_hi": edges[1:], "center": centers}
    for m, _ in _MEASURES:
        d = res[m]
        rows[f"{m}_n"] = d["n"]
        rows[f"{m}_pearson_r"] = d["r"]
        rows[f"{m}_pearson_p"] = d["p"]
        rows[f"{m}_spearman_rho"] = d["rho"]
        rows[f"{m}_spearman_p"] = d["p_rho"]
        rows[f"{m}_pearson_r_null"] = d["r_null"]
        rows[f"{m}_spearman_rho_null"] = d["rho_null"]
        rows[f"{m}_pearson_delta"] = d["r"] - d["r_null"]
        rows[f"{m}_spearman_delta"] = d["rho"] - d["rho_null"]
        rows[f"{m}_empirical_p_pearson"] = d["emp_p"]
        rows[f"{m}_empirical_p_spearman"] = d["emp_p_rho"]
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "pair_distance_correlation.tsv", sep="\t", index=False)
    figure(out / f"pair_distance_correlation.{args.plot_format}",
           centers, res, args.title)

    print(f"  {n_pairs_total:,} high-quality cis pairs from {n_used} cells",
          file=sys.stderr)
    for m, lab in _MEASURES:
        d = res[m]
        ok = np.isfinite(d["r"])
        if ok.any():
            j = int(np.flatnonzero(ok)[-1])
            print(f"  {lab.split('  ')[0]:22s} longest bin with >= "
                  f"{args.min_pairs}: {edges[j]:.0f}-{edges[j+1]:.0f} bp, "
                  f"n = {int(d['n'][j]):,}\n"
                  f"    Pearson  r = {d['r'][j]:+.3f}  null {d['r_null'][j]:+.3f}"
                  f"  Δ = {d['r'][j]-d['r_null'][j]:+.3f}"
                  f"  empirical p = {d['emp_p'][j]:.3g}"
                  f"  (parametric {d['p'][j]:.2g})\n"
                  f"    Spearman ρ = {d['rho'][j]:+.3f}  null "
                  f"{d['rho_null'][j]:+.3f}"
                  f"  Δ = {d['rho'][j]-d['rho_null'][j]:+.3f}"
                  f"  empirical p = {d['emp_p_rho'][j]:.3g}"
                  f"  (parametric {d['p_rho'][j]:.2g})", file=sys.stderr)
    print(f"  -> {out}/pair_distance_correlation.tsv and "
          f".{args.plot_format}", file=sys.stderr)
    print(f"  The null re-pairs WITHIN cell and WITHIN bin, so a correlation "
          f"that comes from cells differing in global level is in the null "
          f"too; read Δ, not r.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
