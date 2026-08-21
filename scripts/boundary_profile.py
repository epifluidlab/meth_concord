#!/usr/bin/env python3
"""Density of bin sets around domain boundaries, with a control set.

Asks whether a bin set piles up at the edge of a domain rather than inside
it -- differentiated bins at the early/late replication-timing boundary, at
the cLAD edge, at the PMD edge -- and answers it against a control set run
through the identical geometry, because a profile with no control cannot tell
enrichment at the boundary from any bin set being non-uniform.

    python scripts/boundary_profile.py \\
        --universe out/marker_bins/bin_scores_gch.all.bed \\
        --bed differentiated=out/marker_bins/bin_scores_gch.contrast_x.bed \\
        --bed stable=out/marker_bins/bin_scores_gch.stable.bed \\
        --boundary "early/late RT=annots/rt_4dn/HFFc6_early.bed" \\
        --boundary "cLAD=annots/lad_4dn/cLAD.bed" \\
        --boundary "PMD=annots/cache/*_PMD_coordinates_hg38.bed.gz" \\
        --flank 5000000 --output-dir out/boundary

DENSITY, NOT COUNT.  At each offset the value is the share of the UNIVERSE
bins there that belong to the set.  Counting instead would draw the shape of
how many bins sit at each distance -- domains have a length distribution, so
offsets far from a boundary are rarer -- and that shape would appear in every
set identically and be read as biology.

ORIENTATION.  Distance is signed so that positive is INSIDE the domain.  Both
edges of a domain are boundaries, and folding them together without flipping
the sign at one of them cancels exactly the asymmetry the plot is for.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_bed(path, merge=False):
    """chrom/start/end, header tolerated; merged only when asked.

    Merging is right for a DOMAIN file, where touching intervals are one
    domain and their shared edge is not a boundary.  It is wrong for a bin
    grid, where every bin abuts its neighbour and merging collapses a whole
    chromosome into a single interval -- silently, and leaving a universe of
    one row per contig.
    """
    p = str(path)
    head = pd.read_csv(p, sep="\t", nrows=1, header=None, compression="infer",
                       dtype=str)
    skip = 1 if str(head.iloc[0, 1]).lower() in ("start", "chromstart") else 0
    df = pd.read_csv(p, sep="\t", header=None, skiprows=skip, comment="#",
                     compression="infer", usecols=[0, 1, 2],
                     names=["chrom", "start", "end"],
                     dtype={0: str, 1: np.int64, 2: np.int64})
    df = df.sort_values(["chrom", "start"]).reset_index(drop=True)
    if not merge:
        return df
    out = []
    for chrom, g in df.groupby("chrom", sort=False):
        cur = None
        for s, e in zip(g.start, g.end):
            if cur and s <= cur[1]:
                cur[1] = max(cur[1], e)
            else:
                if cur:
                    out.append((chrom, *cur))
                cur = [s, e]
        if cur:
            out.append((chrom, *cur))
    return pd.DataFrame(out, columns=["chrom", "start", "end"])


def isolated_boundaries(domains, flank, min_width=0):
    """Domains with no OTHER domain within +/- flank -> a filtered frame.

    The point is to keep a NEIGHBOUR out of the window.  Two domains 1 Mb
    apart profiled with a 5 Mb flank put each other's edges inside each
    other's windows, so a bin deep inside one is just outside the next and
    the two signals cancel to a flat line whatever is really there.

    A domain's OWN far edge is a different matter and is not excluded by
    default.  A narrow isolated domain is a real object -- it simply appears
    in the profile as a bump of its own width, which is readable rather than
    confounding.  --min-domain-width brings back the stricter reading, where
    every window is one edge against open ground.
    """
    keep = []
    for _, g in domains.groupby("chrom", sort=False):
        g = g.sort_values("start").reset_index(drop=True)
        gap_before = np.full(len(g), np.inf)
        gap_after = np.full(len(g), np.inf)
        if len(g) > 1:
            gaps = g.start.to_numpy(float)[1:] - g.end.to_numpy(float)[:-1]
            gap_before[1:] = gaps
            gap_after[:-1] = gaps
        ok = (gap_before >= flank) & (gap_after >= flank)
        if min_width:
            ok &= (g.end - g.start).to_numpy(float) >= min_width
        keep.append(g[ok])
    return (pd.concat(keep, ignore_index=True) if keep
            else domains.iloc[:0].copy())


def signed_distance(bins, domains):
    """Signed distance from each bin's centre to the nearest domain edge.

    Positive is inside the domain, negative outside, so the two edges of a
    domain fold onto one another instead of cancelling.  Bins on a contig the
    domains never mention get NaN rather than a distance to nothing.
    """
    centre = ((bins.start + bins.end) / 2.0).to_numpy(float)
    chrom = bins.chrom.astype(str).to_numpy()
    out = np.full(len(bins), np.nan)
    for c, g in domains.groupby("chrom", sort=False):
        m = chrom == str(c)
        if not m.any():
            continue
        edges = np.sort(np.concatenate([g.start.to_numpy(float),
                                        g.end.to_numpy(float)]))
        starts = g.start.to_numpy(float)
        ends = g.end.to_numpy(float)
        x = centre[m]
        # distance to the nearest edge, unsigned
        idx = np.searchsorted(edges, x)
        idx = np.clip(idx, 1, len(edges) - 1)
        d = np.minimum(np.abs(x - edges[idx - 1]), np.abs(x - edges[idx]))
        # inside if the interval starting at or before x also ends after it
        j = np.clip(np.searchsorted(starts, x, side="right") - 1, 0, None)
        inside = (x >= starts[j]) & (x < ends[j])
        out[m] = np.where(inside, d, -d)
    return out


def offset_edges(flank, step):
    edges = np.arange(-flank, flank + step, step, dtype=float)
    return edges, (edges[:-1] + edges[1:]) / 2.0


def profile(dist, in_set, flank, step):
    """Where the SET's own bins sit, as a frequency over offsets.

    The share of bins at an offset that belong to the set is a different
    quantity and a misleading one here: it saturates at 1 wherever the set
    happens to cover everything nearby, so a large set draws a line along the
    top of the panel and its shape says more about set size than position.
    The frequency -- what fraction of the set's bins fall at each offset --
    sums to one for any set, so sets of different sizes are on one axis and
    the curve is a distribution.

    Returns (counts, frequency); the counts matter because the tails rest on
    few bins.
    """
    edges, centres = offset_edges(flank, step)
    ok = np.isfinite(dist) & np.asarray(in_set, bool)
    counts, _ = np.histogram(np.asarray(dist, float)[ok], bins=edges)
    total = counts.sum()
    freq = counts / total if total else np.full(counts.shape, np.nan)
    return counts.astype(int), freq


def random_profile(dist, universe_mask, n_draw, n_rep, flank, step, seed=0):
    """The same frequency for random bin sets of the SAME SIZE.

    A size-matched random draw is the null the eye needs: any set inherits the
    shape of where the universe's bins sit relative to the boundaries -- which
    is not uniform, because domains have a length distribution -- and without
    that shape drawn beside it, the universe's geometry gets read as the set's
    biology.  Averaged over repeats so the control is the expectation rather
    than one arbitrary draw.
    """
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(np.isfinite(dist) & universe_mask)
    if pool.size == 0 or n_draw <= 0:
        _, centres = offset_edges(flank, step)
        return np.full(centres.shape, np.nan)
    n_draw = min(n_draw, pool.size)
    acc = None
    for _ in range(n_rep):
        pick = rng.choice(pool, size=n_draw, replace=False)
        m = np.zeros(dist.shape, bool)
        m[pick] = True
        _, freq = profile(dist, m, flank, step)
        acc = freq if acc is None else acc + freq
    return acc / n_rep


def two_set_test(dist_a, dist_b, near):
    """Fisher on (near the boundary) x (which set), for two bin sets.

    The right test when a matched control is supplied: it asks directly
    whether the query concentrates at boundaries MORE than the control does,
    which is the claim the figure makes.  Testing a set against the rest of
    the universe answers a different question -- whether the set differs from
    everything it was not -- and a matched control is a sharper comparator
    than "everything else".
    """
    from scipy.stats import fisher_exact

    da = np.asarray(dist_a, float)
    db = np.asarray(dist_b, float)
    da, db = da[np.isfinite(da)], db[np.isfinite(db)]
    a = int(np.sum(np.abs(da) < near))
    b = int(da.size - a)
    c = int(np.sum(np.abs(db) < near))
    d = int(db.size - c)
    try:
        orr, p = fisher_exact([[a, b], [c, d]])
    except Exception:
        orr, p = np.nan, np.nan
    return {"n_near_in_set": a, "n_far_in_set": b,
            "n_near_control": c, "n_far_control": d,
            "frac_near": a / max(a + b, 1),
            "frac_near_control": c / max(c + d, 1),
            "odds_ratio": float(orr), "pvalue": float(p)}


def edge_test(dist, in_set, universe_mask, near):
    """Fisher on (near the boundary) x (in the set), against the far bins.

    Used when a universe is given and no control set is: it compares the set
    with everything it is not, within the universe.
    """
    from scipy.stats import fisher_exact

    ok = np.isfinite(dist) & universe_mask
    close = ok & (np.abs(dist) < near)
    far = ok & (np.abs(dist) >= near)
    a = int(np.sum(close & in_set))
    b = int(np.sum(close & ~in_set))
    c = int(np.sum(far & in_set))
    d = int(np.sum(far & ~in_set))
    try:
        orr, p = fisher_exact([[a, b], [c, d]])
    except Exception:
        orr, p = np.nan, np.nan
    return {"n_near_in_set": a, "n_near_out": b, "n_far_in_set": c,
            "n_far_out": d,
            "frac_near": a / max(a + b, 1), "frac_far": c / max(c + d, 1),
            "odds_ratio": float(orr), "pvalue": float(p)}


def profile_figure(results, path, title, near, step):
    """One panel per boundary; the sets overlaid so the control is beside the
    query rather than on another page."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.8),
                             squeeze=False, sharey=True)
    palette = ["#D1495B", "#1F6F8B", "#3E8E7E", "#B07C4F"]
    for ax, bname in zip(axes[0], names):
        block = results[bname]
        colour_of, k = {}, 0
        for sname, r in block["sets"].items():
            base = sname.split(" (random")[0]
            if base not in colour_of:
                colour_of[base] = palette[k % len(palette)]
                k += 1
            x, y = r["offset"], r["density"]
            good = np.isfinite(y)
            is_ctrl = bool(r.get("control"))
            ax.plot(x[good] / 1e6, y[good], color=colour_of[base],
                    lw=1.2 if is_ctrl else 1.9,
                    ls="--" if is_ctrl else "-",
                    alpha=0.75 if is_ctrl else 1.0,
                    label=sname, zorder=2 if is_ctrl else 3)
        ax.axvline(0, color="#374151", lw=1.0, ls="--", zorder=1)
        ax.axvspan(-near / 1e6, near / 1e6, color="#F0C808", alpha=0.10, lw=0,
                   zorder=0)
        ax.set_title(bname, fontsize=10)
        ax.set_xlabel("distance from boundary (Mb)\n"
                      "<- outside      inside ->", fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#EEF0F2", lw=0.7)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("fraction of the set's bins", fontsize=9)
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left",
                       bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _named(specs, flag):
    out = {}
    for spec in specs or []:
        if "=" not in spec:
            sys.exit(f"{flag} must be LABEL=PATH, got {spec!r}")
        k, v = spec.split("=", 1)
        hits = sorted(glob.glob(v.strip())) or [v.strip()]
        out[k.strip()] = hits[0]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", default=None, metavar="BED",
                    help="Every bin that could have been selected -- "
                         "normally bin_scores_<ctx>.all.bed.  Needed only to "
                         "DRAW a random control and to test a set against "
                         "everything it is not.  Omit it when you supply your "
                         "own matched control with --control: the profile "
                         "itself is computed from each set's own bins.")
    ap.add_argument("--control", default=None, metavar="LABEL",
                    help="Which --bed is the control.  The comparison then "
                         "becomes query-vs-control, which is the claim the "
                         "figure makes; without it a set is tested against "
                         "the rest of the universe, a different and blunter "
                         "question.")
    ap.add_argument("--bed", action="append", required=True,
                    metavar="LABEL=BED",
                    help="A bin set to profile.  Repeatable.  Each gets a "
                         "size-matched random control drawn from the universe "
                         "automatically; adding the stable bins as a second "
                         "set is still worth it, since random bins share the "
                         "universe's geometry but not its biology.")
    ap.add_argument("--boundary", action="append", required=True,
                    metavar="LABEL=BED",
                    help="A domain BED whose interval EDGES are the "
                         "boundaries.  Repeatable, one panel each.")
    ap.add_argument("--flank", type=int, default=5_000_000,
                    help="Half-width of the profile (5 Mb).  At a 250 kb bin "
                         "this is 20 bins each way; a flank of tens of Mb "
                         "spans whole chromosome arms and is flat by "
                         "construction.")
    ap.add_argument("--step", type=int, default=250_000,
                    help="Offset bin for the profile (250 kb).  Below the "
                         "bin size it only splits single bins across "
                         "neighbouring offsets.")
    ap.add_argument("--n-random", type=int, default=200, metavar="N",
                    help="Random size-matched draws averaged into the control "
                         "(200).  One draw is an arbitrary sample; the mean "
                         "over many is the expectation a set of that size "
                         "would show if it ignored the boundaries.")
    ap.add_argument("--min-domain-width", type=int, default=0, metavar="BP",
                    help="Also require each domain to be at least this wide "
                         "(0: no requirement).  Off by default because a "
                         "narrow ISOLATED domain is a real object -- it shows "
                         "as a bump of its own width, which is readable.  Set "
                         "it to --flank when every window must be one edge "
                         "against open ground.")
    ap.add_argument("--keep-crowded", action="store_true",
                    help="Profile every domain edge, including those whose "
                         "window contains another domain.  Off by default: a "
                         "domain narrower than twice the flank puts its far "
                         "edge inside its near edge's window, so bins deep "
                         "inside one boundary are just outside the next and "
                         "the two cancel to a flat line.")
    ap.add_argument("--near", type=int, default=500_000,
                    help="|distance| under this counts as AT the boundary for "
                         "the Fisher test (500 kb).  Domains are megabases "
                         "wide, so a generous 'near' takes in much of the "
                         "genome and the near-vs-far contrast stops meaning "
                         "much; the run reports what share it captured.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    sets = _named(args.bed, "--bed")
    if args.universe:
        uni = read_bed(args.universe)
    else:
        # No universe: the profile only needs each set's own bins, so the
        # working grid is their union.  Every automatic control and the
        # against-everything-else test are unavailable and are skipped rather
        # than faked from a grid that is not a universe.
        uni = (pd.concat([read_bed(v) for v in sets.values()],
                         ignore_index=True)
               .drop_duplicates(subset=["chrom", "start"])
               .sort_values(["chrom", "start"]).reset_index(drop=True))
        print(f"  no --universe: profiling on the union of the given sets "
              f"({len(uni):,} bins).  No random control is drawn and no "
              f"test against the rest of the genome is run.", file=sys.stderr)
    if args.control and args.control not in sets:
        sys.exit(f"--control {args.control!r} is not one of the --bed labels "
                 f"({sorted(sets)})")
    bounds = _named(args.boundary, "--boundary")
    # Every set gets a size-matched random control automatically, so one
    # --bed is a complete analysis.  A second real set (the stable bins) is
    # still worth giving: random bins share the universe's geometry but not
    # its biology, and a real set that behaves like the random one is a
    # stronger statement than random alone.

    key = uni.chrom.astype(str) + ":" + uni.start.astype(str)
    membership = {}
    for name, path in sets.items():
        b = read_bed(path)
        k = set(b.chrom.astype(str) + ":" + b.start.astype(str))
        membership[name] = key.isin(k).to_numpy()
        where = "universe bins" if args.universe else "bins in the union"
        print(f"  {name}: {int(membership[name].sum()):,} of "
              f"{len(uni):,} {where}")
        if not membership[name].any():
            sys.exit(f"--bed {name}: not one of its bins matches the "
                     f"{'universe' if args.universe else 'other sets'}.  They "
                     f"must share a bin grid -- the subset BEDs written "
                     f"beside bin_scores_<ctx>.all.bed do.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    universe_mask = np.ones(len(uni), bool)

    results, rows = {}, []
    for bname, bpath in bounds.items():
        dom_all = read_bed(bpath, merge=True)
        dom = (dom_all if args.keep_crowded
               else isolated_boundaries(dom_all, args.flank,
                                        args.min_domain_width))
        if not len(dom):
            print(f"  {bname}: no domain has {args.flank / 1e6:g} Mb of "
                  f"clear space on both sides"
                  + (f" and is {args.min_domain_width / 1e6:g} Mb wide"
                     if args.min_domain_width else "")
                  + ".  Lower --flank, or pass --keep-crowded to profile "
                    "every edge and accept that neighbouring domains fall "
                    "inside the window.", file=sys.stderr)
            continue
        dist = signed_distance(uni, dom)
        n_ok = int(np.isfinite(dist).sum())
        print(f"  {bname}: {len(dom):,} of {len(dom_all):,} domains isolated "
              f"at +/-{args.flank / 1e6:g} Mb, {n_ok:,} bins placed")
        _, centres = offset_edges(args.flank, args.step)
        block = {"sets": {}}
        for sname, in_set in membership.items():
            n_at, freq = profile(dist, in_set, args.flank, args.step)
            block["sets"][sname] = {"offset": centres, "density": freq,
                                    "n": n_at}
            if args.universe and args.n_random > 0 and not args.control:
                ctrl = random_profile(dist, universe_mask, int(in_set.sum()),
                                      args.n_random, args.flank, args.step)
                block["sets"][f"{sname} (random, n matched)"] = {
                    "offset": centres, "density": ctrl,
                    "n": np.zeros(centres.shape, int), "control": True}
            if args.control:
                if sname == args.control:
                    continue
                st = two_set_test(dist[in_set],
                                  dist[membership[args.control]], args.near)
                st.update({"boundary": bname, "set": sname,
                           "control": args.control})
                rows.append(st)
                print(f"    {sname:>16}: {st['frac_near']:.3f} within "
                      f"{args.near / 1e6:g} Mb vs {st['frac_near_control']:.3f}"
                      f" for {args.control}, OR {st['odds_ratio']:.2f}, "
                      f"p {st['pvalue']:.2g}")
                continue
            if not args.universe:
                continue
            st = edge_test(dist, in_set, universe_mask, args.near)
            share = (st["n_near_in_set"] + st["n_near_out"]) / max(
                st["n_near_in_set"] + st["n_near_out"]
                + st["n_far_in_set"] + st["n_far_out"], 1)
            st.update({"boundary": bname, "set": sname})
            rows.append(st)
            st["frac_of_genome_near"] = float(share)
            note = ("   (near covers most of the universe -- lower --near)"
                    if share > 0.4 else "")
            print(f"    {sname:>16}: {st['frac_near']:.3f} within "
                  f"{args.near / 1e6:g} Mb vs {st['frac_far']:.3f} beyond, "
                  f"OR {st['odds_ratio']:.2f}, p {st['pvalue']:.2g}"
                  f"   [near = {share:.0%} of bins]{note}")
        results[bname] = block

    if rows:
        tab = pd.DataFrame(rows)
        lead = [c for c in ("boundary", "set", "control") if c in tab.columns]
        tab = tab[lead + [c for c in tab.columns if c not in lead]]
        tab.to_csv(out / "boundary_enrichment.tsv", sep="\t", index=False)

    long = []
    for bname, block in results.items():
        for sname, r in block["sets"].items():
            long.append(pd.DataFrame({"boundary": bname, "set": sname,
                                      "offset": r["offset"],
                                      "density": r["density"],
                                      "n_universe": r["n"]}))
    pd.concat(long, ignore_index=True).to_csv(
        out / "boundary_profile.tsv", sep="\t", index=False)

    fig = profile_figure(results, out / "boundary_profile.pdf",
                         f"bin density around domain boundaries "
                         f"(+/-{args.flank / 1e6:g} Mb)", args.near, args.step)
    print(f"  -> {Path(fig).name}, boundary_enrichment.tsv, "
          f"boundary_profile.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
