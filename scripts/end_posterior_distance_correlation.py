#!/usr/bin/env python3
"""Correlation between the two ENDS' posterior levels, versus their separation.

site_distance_correlation.py correlates individual 0/1 site calls, which at one
cytosine on one molecule is close to a coin flip: most of what it measures at
long range is that Bernoulli noise, and the real coordination has to be read
through it.  This script instead uses the Step-4 per-end POSTERIOR — the
probability that the end is open (GCH) or methylated (HCG) given ALL of its
informative sites — so each observation already averages over n sites.

For every read pair the two ends give (p1, p2); pairs are binned by the
separation of their midpoints and a Pearson r is taken per bin.  Three
measures, as in the site-level script:

    gg   GCH(end 1) x GCH(end 2)
    hh   HCG(end 1) x HCG(end 2)
    gh   HCG(one end) x GCH(the other)   -- cross-context

The cross-context measure needs both contexts for the SAME read pair, so the
two posterior tables are joined on (chrom, pos1, pos2); a pair that failed the
site threshold in either context simply does not contribute to gh.

Nulls mirror the site-level script:

    --null-type distance-matched   re-pair two ends of the SAME cell at the
        observed separation (+/- the bin), excluding real ligations.  Answers
        "is this contact more concordant than any two loci that far apart in
        this cell".
    --null-type locus-matched      keep end 1 and swap end 2 for another end
        within +/---null-window of the true partner.  Answers "same two places,
        different molecule".

    --null-scope cell     donors come from the anchor's own cell.
    --null-scope pooled   donors come from every cell in the group, optionally
        restricted to cells sharing --null-match-column (e.g.
        exp_batch,plate_batch,lib_batch), so a donor never crosses a batch.

With --group-column, a figure and table are written per cell-type group as well
as for the pooled "all".

    python scripts/end_posterior_distance_correlation.py \\
        --posteriors GCH=output/signed_250kb_rwr/context_GCH/04_posteriors \\
        --posteriors HCG=output/signed_250kb_rwr/context_HCG/04_posteriors \\
        --samples mouse_cortex_1577_cells.samples.with_batches.tsv \\
        --group-column group --null-type distance-matched --null-scope cell \\
        --output-dir output/end_posterior_dist
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance.distance_stats import (      # noqa: E402
    accumulate, bootstrap_delta, build_edges, corr_from_acc, fdr,
    new_acc, stars)

logger = logging.getLogger("end_posterior_dist")

_MEASURES = (("gg", "GCH–GCH  r(end$_i$, end$_j$)"),
             ("hh", "HCG–HCG  r(end$_i$, end$_j$)"),
             ("gh", "HCG–GCH  r(end$_i$, end$_j$)  [cross-context]"))
_CONDS = (("linked", "linked (true partner)"),
          ("null", "non-linked"))
_COLOR = {"gg": "#1f77b4", "hh": "#d62728", "gh": "#2ca02c"}


# ---------------------------------------------------------------- exclusion --

def _canon(c: str) -> str:
    """Chromosome name without the 'chr' prefix, so a BED and a BAM that
    disagree about it still line up."""
    c = str(c)
    return c[3:] if c.startswith("chr") else c


def load_exclusions(bed_paths, exclude_chrom=()):
    """Merged intervals per chromosome from one or more BED files.

    Merging matters for the lookup below: with non-overlapping, sorted
    intervals a position hits at most the predecessor of its insertion point,
    so the test is one searchsorted rather than a scan.
    """
    excl_chrom = {_canon(c) for c in exclude_chrom if c}
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for path in bed_paths or []:
        with (gzip.open(path, "rt") if str(path).endswith(".gz")
              else open(path)) as fh:
            for line in fh:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                f = line.split()
                if len(f) < 3:
                    continue
                by_chrom.setdefault(_canon(f[0]), []).append(
                    (int(f[1]), int(f[2])))
    merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c, ivs in by_chrom.items():
        ivs.sort()
        out: list[list[int]] = []
        for s0, e0 in ivs:
            if out and s0 <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e0)
            else:
                out.append([s0, e0])
        merged[c] = (np.array([a for a, _ in out], np.int64),
                     np.array([b for _, b in out], np.int64))
    if merged:
        n = sum(len(v[0]) for v in merged.values())
        logger.info(f"Exclusion set: {n:,} merged intervals over "
                    f"{len(merged)} chromosome(s).")
    return merged, excl_chrom


def _keep_mask(df: pd.DataFrame, merged, excl_chrom) -> np.ndarray:
    """Keep a pair only if NEITHER end falls in an excluded interval.

    Pair-level, matching the site-level script's blacklist rule: a contact with
    one end in a dark region is not half-usable.  Only the end midpoints are
    available here (the read spans are not carried in the posteriors), so a
    read overlapping an interval edge can survive -- acceptable for regions
    that are large relative to a read, which is what these exclusions are.

    Chromosomes are handled through categorical CODES.  The obvious
    ``chrom.map(_canon)`` runs a Python call per row -- 1.6 M of them for two
    contexts of a 400 k-pair cell, which profiling showed was the single
    largest cost in the whole script -- and ``np.unique`` on the string column
    then sorts every row again.  Canonicalising the ~20 category labels
    instead makes both costs independent of the number of pairs.
    """
    n = len(df)
    keep = np.ones(n, dtype=bool)
    cat = df["chrom"].astype("category")
    codes = cat.cat.codes.to_numpy()
    canon = np.array([_canon(c) for c in cat.cat.categories])   # ~20 calls

    if excl_chrom:
        bad = np.flatnonzero(np.isin(canon, list(excl_chrom)))
        if bad.size:
            keep &= ~np.isin(codes, bad)
    if not merged:
        return keep

    p1 = df["pos1"].to_numpy(np.int64)
    p2 = df["pos2"].to_numpy(np.int64)
    for ci, c in enumerate(canon):                 # ~20 iterations, not n
        iv = merged.get(c)
        if iv is None:
            continue
        rows = np.flatnonzero(codes == ci)
        if rows.size == 0:
            continue
        starts, ends = iv
        last = len(ends) - 1
        for pos in (p1[rows], p2[rows]):
            # Intervals are merged and sorted, so only the predecessor of the
            # insertion point can contain the position.
            i = np.searchsorted(starts, pos, side="right") - 1
            hit = (i >= 0) & (ends[np.clip(i, 0, last)] > pos)
            keep[rows[hit]] = False
    return keep


# ------------------------------------------------------------------ loading --

def _load_cell(paths: dict[str, Path], cell: str, min_sites: int,
               merged=None, excl_chrom=()):
    """One cell's per-end posteriors, per context, keyed by read pair.

    Returns ``{ctx: DataFrame[chrom,pos1,pos2,distance,p1,p2]}``; contexts with
    no file for this cell are omitted rather than faked.
    """
    out = {}
    for ctx, d in paths.items():
        p = d / f"{cell}.posteriors.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                         "distance", "posterior1",
                                         "posterior2"])
        df = df[(df["n1"] >= min_sites) & (df["n2"] >= min_sites)
                & df["posterior1"].notna() & df["posterior2"].notna()]
        if df.empty:
            continue
        if merged or excl_chrom:
            df = df[_keep_mask(df, merged, excl_chrom)]
            if df.empty:
                continue
        out[ctx] = df.rename(columns={"posterior1": "p1", "posterior2": "p2"})[
            ["chrom", "pos1", "pos2", "distance", "p1", "p2"]]
    return out


def _pair_frames(per_ctx: dict[str, pd.DataFrame]):
    """The (x, y, distance) inputs for each measure.

    gg / hh are within one context.  gh joins the two contexts on the read-pair
    key so the two levels come from the SAME molecule pair; both orderings are
    emitted (HCG@end1 x GCH@end2 and the reverse), matching the site-level
    script's treatment.
    """
    frames: dict[str, list[tuple]] = {"gg": [], "hh": [], "gh": []}
    g, h = per_ctx.get("GCH"), per_ctx.get("HCG")
    if g is not None:
        frames["gg"].append((g["p1"].to_numpy(), g["p2"].to_numpy(),
                             g["distance"].to_numpy()))
    if h is not None:
        frames["hh"].append((h["p1"].to_numpy(), h["p2"].to_numpy(),
                             h["distance"].to_numpy()))
    if g is not None and h is not None:
        j = h.merge(g, on=["chrom", "pos1", "pos2"], suffixes=("_h", "_g"))
        if len(j):
            d = j["distance_h"].to_numpy()
            frames["gh"].append((j["p1_h"].to_numpy(), j["p2_g"].to_numpy(), d))
            frames["gh"].append((j["p2_h"].to_numpy(), j["p1_g"].to_numpy(), d))
    return frames


def joined_pairs(per_ctx: dict[str, pd.DataFrame]):
    """One frame of read pairs carrying BOTH contexts' end posteriors.

    Joined on (chrom, pos1, pos2) so a row's GCH and HCG levels come from the
    same molecule pair.  Rows missing from either context are dropped -- the
    cross-context measure is undefined for them, and using a different pair set
    per measure would make the three panels incomparable.
    """
    g, h = per_ctx.get("GCH"), per_ctx.get("HCG")
    if g is None or h is None:
        return None
    j = h.merge(g, on=["chrom", "pos1", "pos2"], suffixes=("_h", "_g"))
    if j.empty:
        return None
    return j.rename(columns={"distance_h": "distance"})[
        ["chrom", "pos1", "pos2", "distance",
         "p1_h", "p2_h", "p1_g", "p2_g"]]


def end_table(df: pd.DataFrame, cols: dict[str, tuple[str, str]]):
    """Both ends of every pair, sorted by position.

    ``cols`` maps an output name to the (end-1, end-2) column pair, so one call
    builds an end table carrying however many per-end values the caller needs
    (here: the GCH and HCG posteriors together).  Returns
    ``(pos, {name: values}, pair_index)``; the pair index lets a null refuse to
    re-create a real ligation.
    """
    n = len(df)
    pos = np.concatenate([df["pos1"].to_numpy(), df["pos2"].to_numpy()])
    pi = np.concatenate([np.arange(n), np.arange(n)])
    order = np.argsort(pos, kind="stable")
    vals = {name: np.concatenate([df[c1].to_numpy(), df[c2].to_numpy()])[order]
            for name, (c1, c2) in cols.items()}
    return pos[order], vals, pi[order]


# -------------------------------------------------------------------- nulls --

def _null_indices_distance_matched(anchor_pos, anchor_dist, d_pos, d_pi,
                                   edges, rng, window, anchor_window,
                                   same_pool, max_rounds=16):
    """Indices (ka, kb) into the donor end table for a distance-matched null.

    Returning INDICES rather than values is what lets one re-pairing serve
    every measure: gg takes (g[ka], g[kb]), hh takes (h[ka], h[kb]) and the
    cross-context measure takes (h[ka], g[kb]).  Building a separate null per
    measure would compare them against different synthetic pairs.

    Vectorised over anchors, with the retry as an OUTER loop of at most
    ``max_rounds`` fully-vectorised passes over whatever is still unplaced.
    The obvious per-anchor loop is pure Python and holds the GIL for the whole
    run, which made --n-threads do nothing at all.
    """
    n = int(anchor_pos.size)
    if d_pos.size < 2 or n == 0:
        return None
    nb = len(edges) - 1
    in_range = (anchor_dist >= edges[0]) & (anchor_dist <= edges[-1])
    bins = np.clip(np.digitize(anchor_dist, edges) - 1, 0, nb - 1)
    # Target separation = the observed pair's own distance BIN, widened to at
    # least +/-window: the analysis cannot resolve finer than a bin, and an
    # absolute window is unfindable at long range.
    tlo = np.minimum(edges[bins], anchor_dist - window)
    thi = np.maximum(edges[bins + 1], anchor_dist + window)

    if anchor_window > 0:
        alo = np.searchsorted(d_pos, anchor_pos - anchor_window, side="left")
        ahi = np.searchsorted(d_pos, anchor_pos + anchor_window, side="right")
    else:
        alo = np.zeros(n, dtype=np.int64)
        ahi = np.full(n, d_pos.size, dtype=np.int64)

    alive = in_range & ((ahi - alo) >= 2)
    ka_out = np.full(n, -1, dtype=np.int64)
    kb_out = np.full(n, -1, dtype=np.int64)

    for _ in range(max_rounds):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        span = (ahi[idx] - alo[idx]).astype(np.int64)
        ka = alo[idx] + (rng.random(idx.size) * span).astype(np.int64)
        up = rng.random(idx.size) < 0.5
        pa = d_pos[ka]
        lo = np.where(up, pa + tlo[idx], pa - thi[idx])
        hi = np.where(up, pa + thi[idx], pa - tlo[idx])
        blo = np.searchsorted(d_pos, lo, side="left")
        bhi = np.searchsorted(d_pos, hi, side="right")
        has = bhi > blo
        width = np.maximum(bhi - blo, 1).astype(np.int64)
        kb = blo + (rng.random(idx.size) * width).astype(np.int64)
        kb = np.clip(kb, 0, d_pos.size - 1)
        good = has & (kb != ka)
        if same_pool:
            # Two ends of ONE read pair are a real ligation, not a null.
            good &= d_pi[ka] != d_pi[kb]
        sel = idx[good]
        ka_out[sel] = ka[good]
        kb_out[sel] = kb[good]
        alive[sel] = False

    placed = ka_out >= 0
    if not placed.any():
        return None
    ka_f, kb_f = ka_out[placed], kb_out[placed]
    dist = np.abs(d_pos[kb_f].astype(np.int64) - d_pos[ka_f].astype(np.int64))
    return ka_f, kb_f, dist.astype(float)


def _null_indices_locus_matched(anchor_pos2, anchor_pos1, anchor_idx, d_pos,
                                d_pi, rng, window, same_pool, max_rounds=8):
    """Indices for the locus-matched null: keep end 1, swap end 2 for another
    end within +/-window of the true partner.

    The first return value indexes the ANCHOR's own end 1 (not the donor
    table), so the caller pairs anchor-side values with donor-side ones.
    Vectorised the same way as above.
    """
    n = int(anchor_pos2.size)
    if d_pos.size < 2 or n == 0:
        return None
    lo = np.searchsorted(d_pos, anchor_pos2 - window, side="left")
    hi = np.searchsorted(d_pos, anchor_pos2 + window, side="right")
    alive = (hi - lo) >= 2
    kb_out = np.full(n, -1, dtype=np.int64)

    for _ in range(max_rounds):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        span = (hi[idx] - lo[idx]).astype(np.int64)
        k = lo[idx] + (rng.random(idx.size) * span).astype(np.int64)
        k = np.clip(k, 0, d_pos.size - 1)
        good = np.ones(idx.size, dtype=bool)
        if same_pool:
            good = d_pi[k] != anchor_idx[idx]
        sel = idx[good]
        kb_out[sel] = k[good]
        alive[sel] = False

    placed = kb_out >= 0
    if not placed.any():
        return None
    ai = np.flatnonzero(placed)
    kb_f = kb_out[placed]
    dist = np.abs(d_pos[kb_f].astype(np.int64)
                  - anchor_pos1[ai].astype(np.int64))
    return ai, kb_f, dist.astype(float)


# ----------------------------------------------------------------- plotting --

def _figure(curves, centers, label, out_path, args, note="", boot=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    boot = boot or {}

    fig, axes = plt.subplots(2, len(_MEASURES),
                             figsize=(5.2 * len(_MEASURES), 8), sharex=True)
    for j, (key, title) in enumerate(_MEASURES):
        ax = axes[0, j]
        for cond, clab in _CONDS:
            r, n = curves[(key, cond)]
            ax.plot(centers, r, label=clab, color=_COLOR[key], lw=1.6,
                    marker="o" if cond == "linked" else "", ms=3.0,
                    ls="-" if cond == "linked" else "--",
                    alpha=1.0 if cond == "linked" else 0.6)
        ax.axhline(0, color="#9AA0A6", lw=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Pearson r of end posteriors" if j == 0 else "")
        ax.set_xscale("log")
        ax.legend(fontsize=8, frameon=False)

        axd = axes[1, j]
        delta = curves[(key, "linked")][0] - curves[(key, "null")][0]
        if key in boot:
            lo, hi, pv, q = boot[key]
            axd.fill_between(centers, lo, hi, color=_COLOR[key], alpha=0.18,
                             lw=0, label="95% CI (cell bootstrap)")
        axd.plot(centers, delta, color=_COLOR[key], lw=1.6, marker="o", ms=3,
                 label="Δ = linked − null")
        if key in boot:
            # Ring the bins that survive FDR, and only those: one test per bin
            # per measure means several would clear raw p<0.05 by chance.
            lo, hi, pv, q = boot[key]
            sig = np.isfinite(q) & (q < 0.05) & np.isfinite(delta)
            if sig.any():
                axd.plot(centers[sig], delta[sig], "o", ms=8,
                         mfc="none", mec=_COLOR[key], mew=1.5,
                         label=f"q < 0.05  ({int(sig.sum())} bins)")
            span = np.nanmax(np.abs(delta)) or 1.0
            for xi, di, qi in zip(centers, delta, q):
                mark = stars(qi)
                if mark and np.isfinite(di):
                    axd.annotate(mark, (xi, di), textcoords="offset points",
                                 xytext=(0, 7), ha="center", fontsize=8,
                                 color=_COLOR[key])
        axd.axhline(0, color="#9AA0A6", lw=0.8)
        axd.set_xscale("log")
        axd.set_xlabel("genomic distance between the two ends (bp)")
        axd.set_ylabel("Δ = linked − null" if j == 0 else "")
        axd.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{label}: end-posterior correlation vs distance"
                 + (f"   ({note})" if note else ""), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _figure_by_group(per_group, out_path, args, colors=None):
    """One panel per measure, every cell-type group overlaid on a shared scale.

    The per-group figures cannot be compared by eye because each is drawn to
    its own y-range; this is the one that shows whether the groups actually
    differ.  Δ is plotted rather than the raw curves, since that is the
    quantity the per-group panels are read for.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    labels = [g for g in per_group if g != "all"]
    if len(labels) < 2:
        return False
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(labels), 1)))
    colors = colors or {g: cmap[i] for i, g in enumerate(labels)}

    fig, axes = plt.subplots(2, len(_MEASURES),
                             figsize=(5.2 * len(_MEASURES), 8), sharex=True)
    for j, (key, title) in enumerate(_MEASURES):
        for row, what in enumerate(("linked", "delta")):
            ax = axes[row, j]
            for g in labels:
                curves, centers = per_group[g]
                if what == "linked":
                    y = curves[(key, "linked")][0]
                else:
                    y = (curves[(key, "linked")][0]
                         - curves[(key, "null")][0])
                ax.plot(centers, y, lw=1.4, color=colors[g], label=g,
                        marker="o", ms=2.5, alpha=0.9)
            ax.axhline(0, color="#9AA0A6", lw=0.8)
            ax.set_xscale("log")
            if row == 0:
                ax.set_title(title, fontsize=11)
                ax.set_ylabel("linked r" if j == 0 else "")
            else:
                ax.set_xlabel("genomic distance between the two ends (bp)")
                ax.set_ylabel("Δ = linked − null" if j == 0 else "")
        # Shared y-scale per row so the groups are visually comparable.
    for row in range(2):
        ylo = min(ax.get_ylim()[0] for ax in axes[row])
        yhi = max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:
            ax.set_ylim(ylo, yhi)
    axes[0, -1].legend(fontsize=7, frameon=False, ncol=2,
                       loc="upper right")
    fig.suptitle(f"end-posterior correlation by cell-type group "
                 f"({len(labels)} groups, shared y-scale)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# --------------------------------------------------------------------- main --

# Read-only context handed to a worker process ONCE, at start-up.  Passing it
# per task instead would pickle the exclusion intervals and the donor pools
# 1577 times over, which is what made the process backend lose to threads.
_W: dict = {}


def _init_worker(paths, edges, args, pools, batch_of, merged, excl_chrom):
    _W.update(paths=paths, edges=edges, args=args, pools=pools,
              batch_of=batch_of, merged=merged, excl_chrom=excl_chrom)


def _worker(job):
    """Only (cell, index) crosses the process boundary per task."""
    cell, i = job
    return _accumulate_cell(cell, i, _W["paths"], _W["edges"], _W["args"],
                            _W["pools"], _W["batch_of"], _W["merged"],
                            _W["excl_chrom"])


def _accumulate_cell(cell, seed_i, paths, edges, args, pools, batch_of,
                     merged, excl_chrom):
    """One cell's contribution, as its own accumulators.

    Returned rather than added in place so the cells can run on a thread pool
    and be merged afterwards; the accumulators are additive, so summing them is
    exact.  The RNG is seeded per cell rather than shared: a shared generator
    would both race between threads and make the run depend on the order they
    happen to finish in.
    """
    rng = np.random.default_rng(args.seed + seed_i)
    nb = len(edges) - 1
    accs = {(m, c): new_acc(nb) for m, _ in _MEASURES for c, _ in _CONDS}
    n_pairs = n_null = 0

    per_ctx = _load_cell(paths, cell, args.min_sites, merged, excl_chrom)
    j = joined_pairs(per_ctx)
    if j is None or j.empty:
        return accs, 0, 0

    d = j["distance"].to_numpy()
    keep = (d >= edges[0]) & (d <= edges[-1])
    if keep.any():
        b = np.clip(np.digitize(d[keep], edges) - 1, 0, nb - 1)
        # LINKED: the true partner, all three measures off the same pairs.
        accumulate(accs[("gg", "linked")], b,
                   j["p1_g"].to_numpy()[keep], j["p2_g"].to_numpy()[keep])
        accumulate(accs[("hh", "linked")], b,
                   j["p1_h"].to_numpy()[keep], j["p2_h"].to_numpy()[keep])
        # Both orderings of the cross measure, as the site-level script does.
        accumulate(accs[("gh", "linked")], b,
                   j["p1_h"].to_numpy()[keep], j["p2_g"].to_numpy()[keep])
        accumulate(accs[("gh", "linked")], b,
                   j["p2_h"].to_numpy()[keep], j["p1_g"].to_numpy()[keep])
        n_pairs = int(keep.sum())

    cols = {"g": ("p1_g", "p2_g"), "h": ("p1_h", "p2_h")}
    a_pos, a_vals, a_pi = end_table(j, cols)
    bkey = batch_of.get(cell, "") if batch_of else ""
    if args.null_scope == "pooled":
        pool = pools.get(bkey)
        if pool is None:
            return accs, n_pairs, 0
        d_pos, d_vals, d_pi = pool
        same_pool = False
    else:
        d_pos, d_vals, d_pi, same_pool = a_pos, a_vals, a_pi, True

    if args.null_type == "distance-matched":
        got = _null_indices_distance_matched(
            j["pos1"].to_numpy(), d, d_pos, d_pi, edges, rng,
            args.null_window, args.null_anchor_window, same_pool)
        if got is None:
            return accs, n_pairs, 0
        ka, kb, dn = got
        xa = {k: v[ka] for k, v in d_vals.items()}
        xb = {k: v[kb] for k, v in d_vals.items()}
    else:
        got = _null_indices_locus_matched(
            j["pos2"].to_numpy(), j["pos1"].to_numpy(), np.arange(len(j)),
            d_pos, d_pi, rng, args.null_window, same_pool)
        if got is None:
            return accs, n_pairs, 0
        ai, kb, dn = got
        # end 1 stays with the anchor; end 2 comes from the donor pool
        xa = {"g": j["p1_g"].to_numpy()[ai], "h": j["p1_h"].to_numpy()[ai]}
        xb = {k: v[kb] for k, v in d_vals.items()}

    keep = (dn >= edges[0]) & (dn <= edges[-1])
    if keep.any():
        bn = np.clip(np.digitize(dn[keep], edges) - 1, 0, nb - 1)
        # ONE re-pairing feeds every measure, so the three panels are compared
        # against the same synthetic pairs.
        accumulate(accs[("gg", "null")], bn, xa["g"][keep], xb["g"][keep])
        accumulate(accs[("hh", "null")], bn, xa["h"][keep], xb["h"][keep])
        accumulate(accs[("gh", "null")], bn, xa["h"][keep], xb["g"][keep])
        accumulate(accs[("gh", "null")], bn, xa["g"][keep], xb["h"][keep])
        n_null = int(keep.sum())
    return accs, n_pairs, n_null


def _build_pools(cells, paths, args, batch_of, merged, excl_chrom):
    """Donor end pools for --null-scope pooled, one per batch stratum."""
    parts: dict[str, list] = {}
    for cell in cells:
        j = joined_pairs(_load_cell(paths, cell, args.min_sites, merged,
                                    excl_chrom))
        if j is None or j.empty:
            continue
        key = batch_of.get(cell, "") if batch_of else ""
        parts.setdefault(key, []).append(
            end_table(j, {"g": ("p1_g", "p2_g"), "h": ("p1_h", "p2_h")}))
    pools = {}
    for k, ps in parts.items():
        pos = np.concatenate([p[0] for p in ps])
        vals = {name: np.concatenate([p[1][name] for p in ps])
                for name in ("g", "h")}
        # Cross-cell donors: no pair index can collide with the anchor's, so
        # -1 marks "never a self-pair".
        pi = np.full(pos.size, -1)
        o = np.argsort(pos, kind="stable")
        pools[k] = (pos[o], {n_: v[o] for n_, v in vals.items()}, pi[o])
    logger.info(f"  pooled donor ends: "
                f"{ {k: int(v[0].size) for k, v in pools.items()} }")
    return pools


def run_group(cells, paths, edges, args, label, out_dir, batch_of=None,
              merged=None, excl_chrom=(), executor=None):
    nb = len(edges) - 1
    accs = {(m, c): new_acc(nb) for m, _ in _MEASURES for c, _ in _CONDS}
    pools = ({} if args.null_scope != "pooled"
             else _build_pools(cells, paths, args, batch_of, merged,
                               excl_chrom))

    n_pairs = n_null = 0
    per_cell: list = []   # kept for the cell-level bootstrap
    jobs = [(c, i) for i, c in enumerate(cells)]
    if executor is not None and len(cells) > 1:
        # The pool is created once for the whole run and reused across groups.
        chunk = max(1, len(jobs) // (args.n_threads * 8))
        if args.backend == "process" and pools:
            # Pooled donors are per GROUP, so they cannot ride on a pool built
            # once for the whole run: a shared initializer would hand every
            # group the same (or an empty) donor set and silently produce the
            # wrong null.  Threads close over `pools` directly and are fine.
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(
                    max_workers=args.n_threads, initializer=_init_worker,
                    initargs=(paths, edges, args, pools, batch_of, merged,
                              excl_chrom)) as gex:
                out = list(gex.map(_worker, jobs, chunksize=chunk))
        elif args.backend == "process":
            out = list(executor.map(_worker, jobs, chunksize=chunk))
        else:
            out = list(executor.map(
                lambda ci: _accumulate_cell(ci[0], ci[1], paths, edges, args,
                                            pools, batch_of, merged,
                                            excl_chrom), jobs))
        for a, np_, nn_ in out:
            for k in accs:
                accs[k] += a[k]
            per_cell.append(a)
            n_pairs += np_
            n_null += nn_
    else:
        for c, i in jobs:
            a, np_, nn_ = _accumulate_cell(c, i, paths, edges, args, pools,
                                           batch_of, merged, excl_chrom)
            for k in accs:
                accs[k] += a[k]
            per_cell.append(a)
            n_pairs += np_
            n_null += nn_
            if (i + 1) % 200 == 0:
                logger.info(f"  {label}: {i + 1}/{len(cells)} cells")

    curves = {k: corr_from_acc(a, args.min_pairs) for k, a in accs.items()}

    # Cell-level bootstrap of Δ = linked − null, then FDR across the whole
    # figure.  Resampling CELLS rather than end pairs is the honest unit: a
    # cell contributes hundreds of thousands of massively non-independent
    # pairs, so a test treating those as independent would star everything.
    boot = {}
    if args.n_bootstrap > 0 and len(per_cell) >= 3:
        raw = {}
        for m, _ in _MEASURES:
            got = bootstrap_delta([a[(m, "linked")] for a in per_cell],
                                  [a[(m, "null")] for a in per_cell],
                                  args.min_pairs, args.n_bootstrap, args.seed)
            if got is not None:
                raw[m] = got
        if raw:
            # One test per bin per measure, so adjust over the figure as a
            # whole rather than per panel.
            flat = np.concatenate([raw[m][2] for m in raw])
            q_all = fdr(flat, args.fdr)
            off = 0
            for m in raw:
                lo, hi, pv = raw[m]
                boot[m] = (lo, hi, pv, q_all[off:off + pv.size])
                off += pv.size

    centers = np.sqrt(edges[:-1] * edges[1:])
    rows = {"bin_lo": edges[:-1], "bin_hi": edges[1:], "center_bp": centers}
    for (m, c), (r, n) in curves.items():
        rows[f"{m}_{c}_r"] = r
        rows[f"{m}_{c}_n"] = n.astype(np.int64)
    for m, (lo, hi, pv, q) in boot.items():
        rows[f"{m}_delta_lo"] = lo
        rows[f"{m}_delta_hi"] = hi
        rows[f"{m}_delta_p"] = pv
        rows[f"{m}_delta_q"] = q
    tsv = Path(out_dir) / f"end_posterior_dist_{label}.tsv"
    pd.DataFrame(rows).to_csv(tsv, sep="\t", index=False)
    note = (f"{args.null_type}, --null-scope {args.null_scope}"
            + (f", donors matched on {args.null_match_column}"
               if args.null_match_column and args.null_scope == "pooled"
               else ""))
    _figure(curves, centers, label,
            Path(out_dir) / f"end_posterior_dist_{label}.{args.plot_format}",
            args, note, boot)
    logger.info(f"  {label}: {n_pairs:,} linked and {n_null:,} null "
                f"end-pairs; wrote {tsv}")
    return curves, centers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posteriors", action="append", required=True,
                    metavar="CTX=DIR",
                    help="Repeatable, e.g. GCH=.../04_posteriors.")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-column", default="cell_name")
    ap.add_argument("--group-column", default=None,
                    help="Also emit one figure/table per group.")
    ap.add_argument("--null-match-column", default=None,
                    help="With --null-scope pooled, restrict donor ends to "
                         "cells sharing these columns (e.g. "
                         "exp_batch,plate_batch,lib_batch).")
    ap.add_argument("--null-type",
                    choices=["distance-matched", "locus-matched"],
                    default="distance-matched")
    ap.add_argument("--null-scope", choices=["cell", "pooled"], default="cell")
    ap.add_argument("--null-window", type=int, default=1000,
                    help="Tolerance in bp; also the minimum width of the "
                         "distance-matched target band (1 kb).")
    ap.add_argument("--null-anchor-window", type=int, default=10_000_000,
                    help="Distance-matched only: draw the synthetic anchor "
                         "within ±this many bp of the real one (10 Mb); "
                         "0 disables.")
    ap.add_argument("--min-sites", type=int, default=2,
                    help="Minimum informative sites per END (2).")
    ap.add_argument("--min-distance", type=int, default=100)
    ap.add_argument("--max-distance", type=int, default=10_000_000)
    ap.add_argument("--bins-per-decade", type=int, default=8)
    ap.add_argument("--min-pairs", type=int, default=200,
                    help="Mask a bin with fewer end-pairs than this.")
    ap.add_argument("--exclude-bed", action="append", default=None,
                    metavar="BED",
                    help="Repeatable.  Drop a read pair when EITHER end falls "
                         "in one of these intervals — a blacklist, a "
                         "CpG-island+shore set, or any custom region file.  "
                         "Pair-level, matching the site-level script: a "
                         "contact with one end in a dark region is not half "
                         "usable.  .gz is fine.")
    ap.add_argument("--exclude-chrom", default="",
                    help="Comma-separated chromosomes to drop entirely "
                         "(e.g. chrM,chrX,chrY).")
    ap.add_argument("--backend", choices=["thread", "process"],
                    default="thread",
                    help="'thread' (default) measured fastest: on 16 cells x "
                         "250k pairs, 8 threads took 5.8 s against 8.6 s "
                         "serial and 8.0 s with processes, whose worker "
                         "start-up and argument pickling outweigh the GIL they "
                         "escape.  Try 'process' on a machine with many more "
                         "cores, where that trade may reverse.")
    ap.add_argument("--n-threads", type=int, default=4,
                    help="Cells are scored in parallel and their accumulators "
                         "summed; each cell gets its own seeded RNG, so the "
                         "result does not depend on thread count or order.")
    ap.add_argument("--n-bootstrap", type=int, default=500,
                    help="Cell-level bootstrap resamples for the Δ CI and "
                         "p-value (500; 0 disables).  Cells are the unit "
                         "because end pairs within one are not independent.")
    ap.add_argument("--fdr", choices=["bh", "by", "none"], default="bh",
                    help="Multiple-testing adjustment across every bin and "
                         "measure in a figure.  'bh' assumes positive "
                         "dependence, which adjacent distance bins have.")
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot-format", choices=["pdf", "png", "svg"],
                    default="pdf")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    paths = {}
    for spec in args.posteriors:
        if "=" not in spec:
            raise SystemExit(f"--posteriors expects CTX=DIR, got {spec!r}")
        ctx, d = spec.split("=", 1)
        paths[ctx.strip()] = Path(d)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(args.samples, sep="\t")
    cells = samples[args.cell_column].astype(str).tolist()
    if args.max_cells:
        cells = cells[:args.max_cells]
    batch_of = None
    if args.null_match_column:
        cols = [c.strip() for c in args.null_match_column.split(",")
                if c.strip()]
        missing = [c for c in cols if c not in samples.columns]
        if missing:
            raise SystemExit(f"--null-match-column not in --samples: {missing}")
        idx = samples.set_index(args.cell_column)
        batch_of = (idx[cols].astype(str).agg("|".join, axis=1)).to_dict()

    merged, excl_chrom = load_exclusions(
        args.exclude_bed,
        [c.strip() for c in args.exclude_chrom.split(",") if c.strip()])

    edges = build_edges(args.min_distance, args.max_distance,
                        args.bins_per_decade)
    logger.info(f"{len(cells)} cells, {len(edges) - 1} distance bins, "
                f"contexts {sorted(paths)}")

    groups = {"all": cells}
    if args.group_column:
        gmap = samples.set_index(args.cell_column)[args.group_column]
        for c in cells:
            g = str(gmap.get(c, "NA"))
            groups.setdefault(g, []).append(c)
    executor = None
    if args.n_threads > 1 and len(cells) > 1:
        if args.backend == "process":
            from concurrent.futures import ProcessPoolExecutor
            executor = ProcessPoolExecutor(
                max_workers=args.n_threads, initializer=_init_worker,
                initargs=(paths, edges, args, {}, batch_of, merged,
                          excl_chrom))
        else:
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=args.n_threads)
        logger.info(f"Parallel over cells: {args.n_threads} "
                    f"{args.backend}(s), pool reused across groups.")
    try:
        per_group = {}
        for label, gcells in groups.items():
            safe = label.replace("/", "_").replace(" ", "_")
            logger.info(f"--- {label} ({len(gcells)} cells) ---")
            per_group[label] = run_group(
                gcells, paths, edges, args, safe, out_dir, batch_of, merged,
                excl_chrom, executor)
        overlay = out_dir / f"end_posterior_dist_by_group.{args.plot_format}"
        if _figure_by_group(per_group, overlay, args):
            logger.info(f"wrote {overlay}  "
                        f"({len(per_group) - 1} groups overlaid, shared "
                        f"y-scale)")
    finally:
        if executor is not None:
            executor.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
