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


def _interval_index(bed):
    """{chrom: (starts, ends)} sorted, for fast overlap queries."""
    idx = {}
    for c, g in bed.groupby("chrom", sort=False):
        g = g.sort_values("start")
        idx[str(c)] = (g.start.to_numpy(np.int64), g.end.to_numpy(np.int64))
    return idx


def _occupied(idx, chrom, lo, hi):
    """Does any interval of the set overlap [lo, hi)?  Vectorised over windows."""
    got = idx.get(str(chrom))
    if got is None:
        return np.zeros(np.shape(lo), bool)
    starts, ends = got
    lo = np.asarray(lo, np.int64)
    hi = np.asarray(hi, np.int64)
    # Intervals are sorted and non-overlapping, so the last one starting
    # before hi is the only candidate that can reach past lo.
    j = np.searchsorted(starts, hi, side="left")
    ok = j > 0
    out = np.zeros(lo.shape, bool)
    out[ok] = ends[np.maximum(j[ok] - 1, 0)] > lo[ok]
    return out


def signed_distance_to_boundaries(bed, boundaries):
    """Signed distance from each interval's centre to the nearest domain edge,
    positive inside, oriented per edge."""
    centre = ((bed.start + bed.end) / 2.0).to_numpy(float)
    chrom = bed.chrom.astype(str).to_numpy()
    out = np.full(len(bed), np.nan)
    for c, g in boundaries.groupby("chrom", sort=False):
        m = chrom == str(c)
        if not m.any():
            continue
        starts = g.start.to_numpy(float)
        ends = g.end.to_numpy(float)
        edges = np.sort(np.concatenate([starts, ends]))
        x = centre[m]
        idx = np.clip(np.searchsorted(edges, x), 1, len(edges) - 1)
        d = np.minimum(np.abs(x - edges[idx - 1]), np.abs(x - edges[idx]))
        j = np.clip(np.searchsorted(starts, x, side="right") - 1, 0, None)
        inside = (x >= starts[j]) & (x < ends[j])
        out[m] = np.where(inside, d, -d)
    return out


def set_fraction_profile(bed, boundaries, flank, step):
    """Fraction of the SET's own bins sitting at each offset.

    Sums to one across the axis, so sets of different sizes share it and the
    curve keeps its dynamic range however large the set is -- which is why it
    is easier to read than occupancy for a set covering much of the genome.

    The cost, and it is not small: the positions AVAILABLE at each distance
    are not uniform -- domains have finite width and uneven spacing -- so
    every set inherits that shape, and a randomly placed control peaks at the
    boundary exactly as the query does.  Nothing here is enrichment until the
    query is compared with a control drawn the same way, so the control is
    not optional in this mode; it is the axis the reading is made against.
    """
    edges, centres = offset_edges(flank, step)
    dist = signed_distance_to_boundaries(bed, boundaries)
    ok = np.isfinite(dist)
    counts, _ = np.histogram(dist[ok], bins=edges)
    total = counts.sum()
    freq = counts / total if total else np.full(counts.shape, np.nan)
    return freq, counts.astype(int)


def occupancy(boundaries, bed, flank, step):
    """Per-boundary occupancy of the set at each offset -> (n_bnd, n_offset).

    THE denominator question.  Histogramming the set's own bins by distance
    to a boundary looks like a profile and is not one: the positions
    AVAILABLE at each distance are not uniform -- domains have finite width
    and uneven spacing -- so any set drawn from the genome inherits that shape
    and peaks at the boundary.  A random control then peaks too, which is the
    tell that the curve is geometry rather than enrichment.

    Here each boundary contributes one row: at every offset the entry is 1 if
    a bin of the set overlaps that window and 0 if not.  Averaging down the
    rows gives the share of boundaries occupied at that offset, which for a
    randomly placed set is its genome-wide coverage at every offset -- flat,
    as it should be.

    Offsets are oriented so positive is INSIDE the domain, which for a right
    edge means walking left.
    """
    edges, centres = offset_edges(flank, step)
    idx = _interval_index(bed)
    rows = []
    for _, d in boundaries.iterrows():
        for pos, sign in ((int(d["start"]), +1), (int(d["end"]), -1)):
            lo = pos + sign * edges[:-1]
            hi = pos + sign * edges[1:]
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            rows.append(_occupied(idx, d["chrom"], lo, hi))
    if not rows:
        return np.zeros((0, centres.size), bool), centres
    return np.vstack(rows), centres


def profile_from_occupancy(occ):
    """Share of boundaries occupied at each offset, and the boundary count."""
    if occ.shape[0] == 0:
        return np.full(occ.shape[1], np.nan), 0
    return occ.mean(axis=0), int(occ.shape[0])


def occupancy_test(occ, centres, near):
    """Occupied share near the boundary against far from it, per set."""
    from scipy.stats import fisher_exact

    if occ.shape[0] == 0:
        return {}
    close = np.abs(centres) < near
    far = ~close
    a = int(occ[:, close].sum())
    b = int(occ[:, close].size - a)
    c = int(occ[:, far].sum())
    d = int(occ[:, far].size - c)
    try:
        orr, p = fisher_exact([[a, b], [c, d]])
    except Exception:
        orr, p = np.nan, np.nan
    return {"n_occupied_near": a, "n_empty_near": b,
            "n_occupied_far": c, "n_empty_far": d,
            "frac_near": a / max(a + b, 1), "frac_far": c / max(c + d, 1),
            "odds_ratio": float(orr), "pvalue": float(p)}


def profile_figure(results, path, title, near, step, ylabel):
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
    axes[0][0].set_ylabel(ylabel, fontsize=9)
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
                    help="Accepted and ignored.  The profile is per "
                         "BOUNDARY -- at each offset it asks whether a bin of "
                         "the set is there -- so it needs no universe, and a "
                         "randomly placed control comes out flat without one. "
                         "Supply the control as another --bed and name it "
                         "with --control.")
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
    ap.add_argument("--density", choices=["set-fraction", "occupancy"],
                    default="set-fraction",
                    help="'set-fraction' (default): of the SET's bins, the "
                         "fraction at each offset -- sums to one, so the "
                         "curve keeps its range however large the set is.  "
                         "But the positions available at each distance are "
                         "not uniform, so EVERY set peaks at the boundary and "
                         "only the gap to a matched control means anything.  "
                         "'occupancy': of the BOUNDARIES, the share with a "
                         "set bin at that offset -- a randomly placed set is "
                         "flat, but the value equals the set's genome "
                         "coverage, so a set covering much of the genome "
                         "sits near that with little room to move.")
    ap.add_argument("--flank", type=int, default=5_000_000,
                    help="Half-width of the FIGURE (5 Mb): the x axis runs "
                         "from -flank to +flank.  At a 250 kb bin that is 20 "
                         "bins each way; a flank of tens of Mb spans whole "
                         "chromosome arms and is flat by construction.")
    ap.add_argument("--step", type=int, default=250_000,
                    help="Offset bin for the profile (250 kb).  Below the "
                         "bin size it only splits single bins across "
                         "neighbouring offsets.")
    ap.add_argument("--n-random", type=int, default=0, metavar="N",
                    help=argparse.SUPPRESS)
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
                    help="ONLY for the Fisher test in the table, never for "
                         "the plot: offsets with |distance| < this count as "
                         "AT the boundary and the rest as away from it "
                         "(500 kb).  --flank sets what the figure spans; "
                         "--near sets where the summary statistic draws its "
                         "line inside that span.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    sets = _named(args.bed, "--bed")
    if args.universe:
        print("  NOTE --universe is ignored: the profile is per boundary, "
              "not per bin of a universe.  Give the control as another --bed "
              "and name it with --control.", file=sys.stderr)
    if True:
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
    if args.density == "set-fraction" and not args.control:
        print("  NOTE --density set-fraction gives every set a peak at the "
              "boundary, because the positions available at each distance are "
              "not uniform.  Without --control naming a matched control there "
              "is nothing to read that peak against.  --density occupancy is "
              "flat for a randomly placed set and needs no control to be "
              "interpretable.", file=sys.stderr)
    if args.control and args.control not in sets:
        sys.exit(f"--control {args.control!r} is not one of the --bed labels "
                 f"({sorted(sets)})")
    bounds = _named(args.boundary, "--boundary")
    # Every set gets a size-matched random control automatically, so one
    # --bed is a complete analysis.  A second real set (the stable bins) is
    # still worth giving: random bins share the universe's geometry but not
    # its biology, and a real set that behaves like the random one is a
    # stronger statement than random alone.

    bed_of = {name: read_bed(path) for name, path in sets.items()}
    for name, b in bed_of.items():
        print(f"  {name}: {len(b):,} intervals")
        if not len(b):
            sys.exit(f"--bed {name}: no intervals")
    # A set covering most of the genome pins the curve near its own coverage
    # at every offset: each offset window is about one bin, so the share of
    # boundaries occupied IS the coverage fraction, and a set at 60% can rise
    # at most 0.4 anywhere.  That is a property of the set, not of the
    # boundaries, and no profile recovers structure from it.
    span = sum((b.end - b.start).sum() for b in bed_of.values())
    for name, b in bed_of.items():
        frac = float((b.end - b.start).sum()) / max(span, 1)
        if len(bed_of) > 1 and frac > 0.45 and len(b) > 1000:
            print(f"  NOTE {name} is large.  The share of boundaries occupied "
                  f"equals the set's genome coverage at every offset, so a "
                  f"set covering much of the genome sits flat near that value "
                  f"with little room to rise.  A profile is informative when "
                  f"the set is a small minority of bins.", file=sys.stderr)
            break

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

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

        block = {"sets": {}}
        n_edges = None
        for sname, bed in bed_of.items():
            occ, centres = occupancy(dom, bed, args.flank, args.step)
            n_edges = int(occ.shape[0])
            if args.density == "occupancy":
                dens, _ = profile_from_occupancy(occ)
                n_at = np.full(centres.shape, n_edges, int)
            else:
                dens, n_at = set_fraction_profile(bed, dom, args.flank,
                                                  args.step)
            block["sets"][sname] = {
                "offset": centres, "density": dens, "n": n_at,
                "control": sname == args.control}
            st = occupancy_test(occ, centres, args.near)
            if st:
                st.update({"boundary": bname, "set": sname})
                rows.append(st)
                print(f"    {sname:>16}: {st['frac_near']:.3f} occupied "
                      f"within {args.near / 1e6:g} Mb vs {st['frac_far']:.3f} "
                      f"beyond, OR {st['odds_ratio']:.2f}, "
                      f"p {st['pvalue']:.2g}")
        if args.density == "set-fraction":
            print("    (the figure is set-fraction; the test above is "
                  "occupancy either way, since that is the one a randomly "
                  "placed set does not pass by construction)",
                  file=sys.stderr)
        print(f"  {bname}: {len(dom):,} of {len(dom_all):,} domains isolated "
              f"at +/-{args.flank / 1e6:g} Mb, {n_edges or 0} edges profiled")
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

    ylabel = ("fraction of the set's bins\nat this offset"
              if args.density == "set-fraction"
              else "share of boundaries with a bin\nof the set at this offset")
    fig = profile_figure(results, out / "boundary_profile.pdf",
                         f"bin density around domain boundaries "
                         f"(+/-{args.flank / 1e6:g} Mb)", args.near,
                         args.step, ylabel)
    print(f"  -> {Path(fig).name}, boundary_enrichment.tsv, "
          f"boundary_profile.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
