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
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance.distance_stats import (      # noqa: E402
    build_edges, corr_from_acc, new_acc, accumulate)

logger = logging.getLogger("end_posterior_dist")

_MEASURES = (("gg", "GCH–GCH  r(end$_i$, end$_j$)"),
             ("hh", "HCG–HCG  r(end$_i$, end$_j$)"),
             ("gh", "HCG–GCH  r(end$_i$, end$_j$)  [cross-context]"))
_CONDS = (("linked", "linked (true partner)"),
          ("null", "non-linked"))
_COLOR = {"gg": "#1f77b4", "hh": "#d62728", "gh": "#2ca02c"}


# ------------------------------------------------------------------ loading --

def _load_cell(paths: dict[str, Path], cell: str, min_sites: int):
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
                                   same_pool):
    """Indices (ka, kb) into the donor end table for a distance-matched null.

    Returning INDICES rather than values is what lets one re-pairing serve
    every measure: gg takes (g[ka], g[kb]), hh takes (h[ka], h[kb]) and the
    cross-context measure takes (h[ka], g[kb]).  Building a separate null per
    measure would compare them against different synthetic pairs.
    """
    if d_pos.size < 2 or anchor_pos.size == 0:
        return None
    nb = len(edges) - 1
    bins = np.clip(np.digitize(anchor_dist, edges) - 1, 0, nb - 1)
    ka_l, kb_l, d_l = [], [], []
    for i in range(anchor_pos.size):
        if anchor_dist[i] < edges[0] or anchor_dist[i] > edges[-1]:
            continue
        b = int(bins[i])
        # Target separation = the observed pair's own distance BIN, widened to
        # at least +/-window: the analysis cannot resolve finer than a bin, and
        # an absolute window is unfindable at long range.
        tlo = min(float(edges[b]), anchor_dist[i] - window)
        thi = max(float(edges[b + 1]), anchor_dist[i] + window)
        alo, ahi = 0, d_pos.size
        if anchor_window > 0:
            alo = int(np.searchsorted(d_pos, anchor_pos[i] - anchor_window, "left"))
            ahi = int(np.searchsorted(d_pos, anchor_pos[i] + anchor_window, "right"))
        if ahi - alo < 2:
            continue
        for _ in range(16):
            ka = int(rng.integers(alo, ahi))
            up = bool(rng.integers(0, 2))
            lo = d_pos[ka] + tlo if up else d_pos[ka] - thi
            hi = d_pos[ka] + thi if up else d_pos[ka] - tlo
            blo = int(np.searchsorted(d_pos, lo, "left"))
            bhi = int(np.searchsorted(d_pos, hi, "right"))
            if bhi <= blo:
                continue
            kb = int(rng.integers(blo, bhi))
            if kb == ka:
                continue
            # Two ends of ONE read pair are a real ligation, not a null.
            if same_pool and d_pi[ka] == d_pi[kb]:
                continue
            ka_l.append(ka), kb_l.append(kb)
            d_l.append(abs(int(d_pos[kb]) - int(d_pos[ka])))
            break
    if not ka_l:
        return None
    return (np.asarray(ka_l), np.asarray(kb_l),
            np.asarray(d_l, dtype=float))


def _null_indices_locus_matched(anchor_pos2, anchor_pos1, anchor_idx, d_pos,
                                d_pi, rng, window, same_pool):
    """Indices for the locus-matched null: keep end 1, swap end 2 for another
    end within +/-window of the true partner.

    ``ka`` here indexes the ANCHOR's own end 1 (not the donor table), so the
    caller pairs anchor-side values with donor-side ones.
    """
    if d_pos.size < 2 or anchor_pos2.size == 0:
        return None
    ai_l, kb_l, d_l = [], [], []
    for i in range(anchor_pos2.size):
        lo = int(np.searchsorted(d_pos, anchor_pos2[i] - window, "left"))
        hi = int(np.searchsorted(d_pos, anchor_pos2[i] + window, "right"))
        if hi - lo < 2:
            continue
        for _ in range(8):
            k = int(rng.integers(lo, hi))
            if same_pool and d_pi[k] == anchor_idx[i]:
                continue
            ai_l.append(i), kb_l.append(k)
            d_l.append(abs(int(d_pos[k]) - int(anchor_pos1[i])))
            break
    if not ai_l:
        return None
    return (np.asarray(ai_l), np.asarray(kb_l),
            np.asarray(d_l, dtype=float))


# ----------------------------------------------------------------- plotting --

def _figure(curves, centers, label, out_path, args, note=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    fig, axes = plt.subplots(2, len(_MEASURES), figsize=(5.2 * len(_MEASURES), 8),
                             sharex=True)
    for j, (key, title) in enumerate(_MEASURES):
        ax = axes[0, j]
        for cond, clab in _CONDS:
            r, n = curves[(key, cond)]
            style = dict(color=_COLOR[key], lw=1.6,
                         marker="o" if cond == "linked" else "",
                         ms=3.0, ls="-" if cond == "linked" else "--",
                         alpha=1.0 if cond == "linked" else 0.6)
            ax.plot(centers, r, label=clab, **style)
        ax.axhline(0, color="#9AA0A6", lw=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Pearson r of end posteriors" if j == 0 else "")
        ax.set_xscale("log")
        ax.legend(fontsize=8, frameon=False)

        axd = axes[1, j]
        rl = curves[(key, "linked")][0]
        rn = curves[(key, "null")][0]
        axd.plot(centers, rl - rn, color=_COLOR[key], lw=1.6, marker="o", ms=3)
        axd.axhline(0, color="#9AA0A6", lw=0.8)
        axd.set_xscale("log")
        axd.set_xlabel("genomic distance between the two ends (bp)")
        axd.set_ylabel("Δ = linked − null" if j == 0 else "")
    fig.suptitle(f"{label}: end-posterior correlation vs distance"
                 + (f"   ({note})" if note else ""), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------- main --

def run_group(cells, paths, edges, args, label, out_dir, batch_of=None):
    rng = np.random.default_rng(args.seed)
    nb = len(edges) - 1
    accs = {(m, c): new_acc(nb) for m, _ in _MEASURES for c, _ in _CONDS}

    # Pooled donors are assembled per batch stratum so a substitute end never
    # crosses a batch boundary -- the same reasoning as --null-match-column in
    # the site-level script.
    pools: dict[str, tuple] = {}
    if args.null_scope == "pooled":
        parts: dict[str, list] = {}
        for cell in cells:
            j = joined_pairs(_load_cell(paths, cell, args.min_sites))
            if j is None or j.empty:
                continue
            key = batch_of.get(cell, "") if batch_of else ""
            parts.setdefault(key, []).append(
                end_table(j, {"g": ("p1_g", "p2_g"), "h": ("p1_h", "p2_h")}))
        for k, ps in parts.items():
            pos = np.concatenate([p[0] for p in ps])
            vals = {name: np.concatenate([p[1][name] for p in ps])
                    for name in ("g", "h")}
            # Cross-cell donors: no pair index can collide with the anchor's,
            # so -1 marks "never a self-pair".
            pi = np.full(pos.size, -1)
            o = np.argsort(pos, kind="stable")
            pools[k] = (pos[o], {n_: v[o] for n_, v in vals.items()}, pi[o])
        logger.info(f"  pooled donor ends: "
                    f"{ {k: int(v[0].size) for k, v in pools.items()} }")

    n_pairs = n_null = 0
    for ci, cell in enumerate(cells, 1):
        per_ctx = _load_cell(paths, cell, args.min_sites)
        j = joined_pairs(per_ctx)
        if j is None or j.empty:
            continue
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
            n_pairs += int(keep.sum())

        cols = {"g": ("p1_g", "p2_g"), "h": ("p1_h", "p2_h")}
        a_pos, a_vals, a_pi = end_table(j, cols)
        bkey = batch_of.get(cell, "") if batch_of else ""
        if args.null_scope == "pooled":
            pool = pools.get(bkey)
            if pool is None:
                continue
            d_pos, d_vals, d_pi = pool
            same_pool = False
        else:
            d_pos, d_vals, d_pi, same_pool = a_pos, a_vals, a_pi, True

        if args.null_type == "distance-matched":
            got = _null_indices_distance_matched(
                j["pos1"].to_numpy(), d, d_pos, d_pi, edges, rng,
                args.null_window, args.null_anchor_window, same_pool)
            if got is None:
                continue
            ka, kb, dn = got
            xa = {k: v[ka] for k, v in d_vals.items()}
            xb = {k: v[kb] for k, v in d_vals.items()}
        else:
            got = _null_indices_locus_matched(
                j["pos2"].to_numpy(), j["pos1"].to_numpy(),
                np.arange(len(j)), d_pos, d_pi, rng, args.null_window,
                same_pool)
            if got is None:
                continue
            ai, kb, dn = got
            # end 1 stays with the anchor; end 2 comes from the donor pool
            xa = {"g": j["p1_g"].to_numpy()[ai], "h": j["p1_h"].to_numpy()[ai]}
            xb = {k: v[kb] for k, v in d_vals.items()}

        keep = (dn >= edges[0]) & (dn <= edges[-1])
        if keep.any():
            bn = np.clip(np.digitize(dn[keep], edges) - 1, 0, nb - 1)
            # ONE re-pairing feeds every measure, so the three panels are
            # compared against the same synthetic pairs.
            accumulate(accs[("gg", "null")], bn, xa["g"][keep], xb["g"][keep])
            accumulate(accs[("hh", "null")], bn, xa["h"][keep], xb["h"][keep])
            accumulate(accs[("gh", "null")], bn, xa["h"][keep], xb["g"][keep])
            accumulate(accs[("gh", "null")], bn, xa["g"][keep], xb["h"][keep])
            n_null += int(keep.sum())
        if ci % 100 == 0:
            logger.info(f"  {label}: {ci}/{len(cells)} cells")

    curves = {k: corr_from_acc(a, args.min_pairs) for k, a in accs.items()}
    centers = np.sqrt(edges[:-1] * edges[1:])
    rows = {"bin_lo": edges[:-1], "bin_hi": edges[1:], "center_bp": centers}
    for (m, c), (r, n) in curves.items():
        rows[f"{m}_{c}_r"] = r
        rows[f"{m}_{c}_n"] = n.astype(np.int64)
    tsv = Path(out_dir) / f"end_posterior_dist_{label}.tsv"
    pd.DataFrame(rows).to_csv(tsv, sep="\t", index=False)
    note = (f"{args.null_type}, --null-scope {args.null_scope}"
            + (f", donors matched on {args.null_match_column}"
               if args.null_match_column and args.null_scope == "pooled"
               else ""))
    _figure(curves, centers, label,
            Path(out_dir) / f"end_posterior_dist_{label}.{args.plot_format}",
            args, note)
    logger.info(f"  {label}: {n_pairs:,} linked and {n_null:,} null "
                f"end-pairs; wrote {tsv}")
    return curves


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
    for label, gcells in groups.items():
        safe = label.replace("/", "_").replace(" ", "_")
        logger.info(f"--- {label} ({len(gcells)} cells) ---")
        run_group(gcells, paths, edges, args, safe, out_dir, batch_of)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
