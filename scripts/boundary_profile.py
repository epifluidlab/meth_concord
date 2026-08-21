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


def profile(dist, in_set, universe_mask, flank, step):
    """Share of universe bins at each offset that belong to the set.

    Returns offset centres, the density, and the count behind it -- the count
    matters because the tails of a profile rest on few bins and a density
    there swings on one of them.
    """
    edges = np.arange(-flank, flank + step, step, dtype=float)
    centres = (edges[:-1] + edges[1:]) / 2.0
    dens = np.full(centres.shape, np.nan)
    n_at = np.zeros(centres.shape, dtype=int)
    ok = np.isfinite(dist) & universe_mask
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = ok & (dist >= a) & (dist < b)
        n = int(m.sum())
        n_at[i] = n
        if n:
            dens[i] = float(np.mean(in_set[m]))
    return centres, dens, n_at


def edge_test(dist, in_set, universe_mask, near):
    """Fisher on (near the boundary) x (in the set), against the far bins.

    The profile shows a shape; this says whether the shape is bigger than
    counting noise.  Near means |distance| < near, on either side, because a
    boundary effect is about proximity to the edge rather than which side of
    it a bin sits on.
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
        for i, (sname, r) in enumerate(block["sets"].items()):
            x, y, n = r["offset"], r["density"], r["n"]
            good = np.isfinite(y) & (n > 0)
            # Binomial standard error on the share, so the tails show their
            # own uncertainty instead of looking as firm as the centre.
            se = np.sqrt(np.clip(y, 0, 1) * (1 - np.clip(y, 0, 1))
                         / np.maximum(n, 1))
            ax.plot(x[good] / 1e6, y[good], color=palette[i % len(palette)],
                    lw=1.8, label=sname, zorder=3)
            ax.fill_between(x[good] / 1e6, (y - 1.96 * se)[good],
                            (y + 1.96 * se)[good],
                            color=palette[i % len(palette)], alpha=0.16, lw=0)
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
    axes[0][0].set_ylabel("share of bins in the set", fontsize=9)
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
    ap.add_argument("--universe", required=True, metavar="BED",
                    help="Every bin that could have been selected -- normally "
                         "bin_scores_<ctx>.all.bed.  The density at each "
                         "offset is a share of THIS, so a domain-length "
                         "effect does not become a profile.")
    ap.add_argument("--bed", action="append", required=True,
                    metavar="LABEL=BED",
                    help="A bin set to profile.  Repeatable; give the query "
                         "AND a control (the stable bins), since a profile "
                         "with no control cannot separate enrichment at the "
                         "boundary from any set being non-uniform.")
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
    ap.add_argument("--near", type=int, default=500_000,
                    help="|distance| under this counts as AT the boundary for "
                         "the Fisher test (500 kb).  Domains are megabases "
                         "wide, so a generous 'near' takes in much of the "
                         "genome and the near-vs-far contrast stops meaning "
                         "much; the run reports what share it captured.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    uni = read_bed(args.universe)
    sets = _named(args.bed, "--bed")
    bounds = _named(args.boundary, "--boundary")
    if len(sets) < 2:
        print("  NOTE only one --bed given.  Without a control set a profile "
              "cannot distinguish enrichment at the boundary from the set "
              "being non-uniform for any other reason; the stable bins are "
              "the natural control.", file=sys.stderr)

    key = uni.chrom.astype(str) + ":" + uni.start.astype(str)
    membership = {}
    for name, path in sets.items():
        b = read_bed(path)
        k = set(b.chrom.astype(str) + ":" + b.start.astype(str))
        membership[name] = key.isin(k).to_numpy()
        print(f"  {name}: {int(membership[name].sum()):,} of "
              f"{len(uni):,} universe bins")
        if not membership[name].any():
            sys.exit(f"--bed {name}: no bin matches the universe.  Both must "
                     f"be on the same grid -- the subset BEDs written beside "
                     f"bin_scores_<ctx>.all.bed are.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    universe_mask = np.ones(len(uni), bool)

    results, rows = {}, []
    for bname, bpath in bounds.items():
        dom = read_bed(bpath, merge=True)
        dist = signed_distance(uni, dom)
        n_ok = int(np.isfinite(dist).sum())
        print(f"  {bname}: {len(dom):,} domains, {n_ok:,} bins placed")
        block = {"sets": {}}
        for sname, in_set in membership.items():
            x, dens, n_at = profile(dist, in_set, universe_mask, args.flank,
                                    args.step)
            block["sets"][sname] = {"offset": x, "density": dens, "n": n_at}
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

    tab = pd.DataFrame(rows)[
        ["boundary", "set", "n_near_in_set", "n_near_out", "n_far_in_set",
         "n_far_out", "frac_near", "frac_far", "frac_of_genome_near",
         "odds_ratio", "pvalue"]]
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
