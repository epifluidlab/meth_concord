#!/usr/bin/env python3
"""2×2 epigenetic-state diagnostic — GCH accessibility × HCG methylation on the
SAME end, across genomic-distance bands of the linked pairs.

Per-end status is the **Beta-Binomial mixture posterior** already stored in the
``*.posteriors.parquet`` (``a = predict_proba(GpC) = P(accessible)``,
``m = predict_proba(CpG) = P(methylated)``) — coverage-aware, so an end with few
sites is pulled toward the genome-wide prior.  Each end is classified with a
**confidence band** (>= --posterior-high -> positive; <= --posterior-low ->
negative; in between -> AMBIGUOUS, excluded), not a single 0.5 cut, and only
ends confidently called in BOTH modalities enter the 2×2.

                       HCG low (unmeth)        HCG high (meth)
    GCH high (open)    open, unmethylated      open, methylated     (active-meth)
    GCH low  (closed)  closed, unmethylated    closed, methylated

It reports, per distance band, both the **per-end** coupling and the
**region-level** coupling (average a, m per genomic bin, correlate across bins —
the scale the concordance/embedding runs at).  With ``--distance-bands`` it
sweeps several bands in ONE pass (one read of each parquet), so you can see
whether accessibility–methylation anti-correlation holds for short-range linked
pairs and weakens with distance.

Usage
-----
    # single band
    python scripts/joint_state_2x2.py --gch-dir .../04_posteriors \\
        --hcg-dir .../04_posteriors --min-distance 20000 --max-distance 0

    # distance sweep (short -> long) in one pass
    python scripts/joint_state_2x2.py --gch-dir .../04_posteriors \\
        --hcg-dir .../04_posteriors \\
        --distance-bands 1000,20000,100000,1000000,10000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 2×2 cell order: (GCH+, HCG-), (GCH+, HCG+), (GCH-, HCG-), (GCH-, HCG+)
_STATES = ("open_unmeth", "open_meth", "closed_unmeth", "closed_meth")


def _cells(d: Path) -> dict:
    return {p.name.replace(".posteriors.parquet", ""): p
            for p in d.glob("*.posteriors.parquet")}


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                        "posterior1", "posterior2"])
    df["chrom"] = df["chrom"].astype(str)
    return df


def process_cell(gpath, hpath, bands, bsz, min_gpc, min_cpg, hi, lo, rbsz):
    """Load one cell's joined posteriors once and compute the 2×2 + coupling
    stats for every (lo, hi) band in ``bands``.  Returns ``{band_index: (conf,
    counts, stats, binsum)}`` or None if the cell has no joined pairs."""
    g = _load(Path(gpath)).rename(columns={"n1": "n1g", "n2": "n2g",
                                           "posterior1": "a1", "posterior2": "a2"})
    h = _load(Path(hpath)).rename(columns={"n1": "n1h", "n2": "n2h",
                                           "posterior1": "m1", "posterior2": "m2"})
    m = g.merge(h, on=["chrom", "pos1", "pos2"], how="inner")
    if len(m) == 0:
        return None
    pos1 = m["pos1"].to_numpy(np.int64); pos2 = m["pos2"].to_numpy(np.int64)
    dist = np.abs(pos2 - pos1)
    same_bin = (pos1 // bsz == pos2 // bsz)
    chrom = m["chrom"].to_numpy()

    # Per-end both-modality arrays (distance-independent; reused for every band).
    ends = []
    for ng, nh, ac, mc, pc in (("n1g", "n1h", "a1", "m1", pos1),
                               ("n2g", "n2h", "a2", "m2", pos2)):
        a = m[ac].to_numpy(np.float64, na_value=np.nan)
        me = m[mc].to_numpy(np.float64, na_value=np.nan)
        both = ((m[ng].to_numpy(np.float64) >= min_gpc)
                & (m[nh].to_numpy(np.float64) >= min_cpg)
                & np.isfinite(a) & np.isfinite(me))
        ends.append((a, me, both, pc))

    out = {}
    for bi, (lo_d, hi_d) in enumerate(bands):
        band = same_bin & (dist >= lo_d)
        if hi_d and hi_d > 0:
            band = band & (dist <= hi_d)
        conf = {k: 0 for k in _STATES}
        n_both = n_conf = n_acc_amb = n_meth_amb = 0
        sa = sm = saa = smm = sam = 0.0; n_ends = 0
        bch, bid, ba, bm = [], [], [], []
        for a_all, me_all, both, pc in ends:
            ok = both & band
            a = a_all[ok]; me = me_all[ok]
            if a.size == 0:
                continue
            n_both += int(a.size)
            ap, an = a >= hi, a <= lo
            mp, mn = me >= hi, me <= lo
            n_acc_amb += int((~ap & ~an).sum())
            n_meth_amb += int((~mp & ~mn).sum())
            conf["open_unmeth"] += int((ap & mn).sum())
            conf["open_meth"] += int((ap & mp).sum())
            conf["closed_unmeth"] += int((an & mn).sum())
            conf["closed_meth"] += int((an & mp).sum())
            n_conf += int(((ap | an) & (mp | mn)).sum())
            sa += float(a.sum()); sm += float(me.sum())
            saa += float((a * a).sum()); smm += float((me * me).sum())
            sam += float((a * me).sum()); n_ends += int(a.size)
            bch.append(chrom[ok]); bid.append(pc[ok] // rbsz); ba.append(a); bm.append(me)
        if ba:
            be = pd.DataFrame({"chrom": np.concatenate(bch),
                               "bin": np.concatenate(bid),
                               "a": np.concatenate(ba), "m": np.concatenate(bm)})
            binsum = be.groupby(["chrom", "bin"]).agg(
                sa=("a", "sum"), sm=("m", "sum"), n=("a", "size")).reset_index()
        else:
            binsum = pd.DataFrame(columns=["chrom", "bin", "sa", "sm", "n"])
        out[bi] = (conf, (n_both, n_conf, n_acc_amb, n_meth_amb),
                   (sa, sm, saa, smm, sam, n_ends), binsum)
    return out


def _agg_band(per_cell, region_min_ends):
    """Aggregate one band's per-cell tuples → a dict of headline numbers."""
    conf = {k: sum(c[0][k] for c in per_cell) for k in _STATES}
    n_both = sum(c[1][0] for c in per_cell); n_conf = sum(c[1][1] for c in per_cell)
    n_acc_amb = sum(c[1][2] for c in per_cell)
    n_meth_amb = sum(c[1][3] for c in per_cell)
    sa = sum(c[2][0] for c in per_cell); sm = sum(c[2][1] for c in per_cell)
    saa = sum(c[2][2] for c in per_cell); smm = sum(c[2][3] for c in per_cell)
    sam = sum(c[2][4] for c in per_cell); n = sum(c[2][5] for c in per_cell)
    ma = sa / n if n else np.nan; mm = sm / n if n else np.nan
    with np.errstate(invalid="ignore"):
        va = saa / n - ma * ma if n else np.nan
        vm = smm / n - mm * mm if n else np.nan
        r_end = (sam / n - ma * mm) / np.sqrt(va * vm) if (n and va > 0 and vm > 0) \
            else np.nan
    nc = max(n_conf, 1)
    offd = (conf["open_meth"] + conf["closed_unmeth"]) / nc

    r_reg = np.nan; n_bins = 0; reg22 = None
    allbin = [c[3] for c in per_cell if len(c[3])]
    if allbin:
        bs = pd.concat(allbin, ignore_index=True).groupby(["chrom", "bin"]).agg(
            sa=("sa", "sum"), sm=("sm", "sum"), n=("n", "sum")).reset_index()
        bs = bs[bs["n"] >= region_min_ends]
        n_bins = len(bs)
        if n_bins >= 3:
            x = (bs["sa"] / bs["n"]).to_numpy(); y = (bs["sm"] / bs["n"]).to_numpy()
            w = bs["n"].to_numpy().astype(float)
            mx = np.average(x, weights=w); my = np.average(y, weights=w)
            vx = np.average((x - mx) ** 2, weights=w)
            vy = np.average((y - my) ** 2, weights=w)
            cxy = np.average((x - mx) * (y - my), weights=w)
            r_reg = cxy / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan
            ah = x >= mx; mh = y >= my; tw = w.sum()
            reg22 = (w[ah & ~mh].sum() / tw, w[ah & mh].sum() / tw,
                     w[~ah & ~mh].sum() / tw, w[~ah & mh].sum() / tw)
    return dict(conf=conf, n_both=n_both, n_conf=n_conf, n_acc_amb=n_acc_amb,
                n_meth_amb=n_meth_amb, ma=ma, mm=mm, r_end=r_end, offd=offd,
                r_reg=r_reg, n_bins=n_bins, reg22=reg22)


def _print_detail(a):
    nc = max(a["n_conf"], 1); c = a["conf"]; f = {k: c[k] / nc for k in _STATES}
    nb = max(a["n_both"], 1)
    print(f"\n[2x2] ends with both modalities on a linked pair: {a['n_both']:,}")
    print(f"      ambiguous (excluded): accessibility {a['n_acc_amb']/nb:.1%}, "
          f"methylation {a['n_meth_amb']/nb:.1%}  | confidently classified in "
          f"BOTH: {a['n_conf']:,} ({a['n_conf']/nb:.1%})")
    print(f"\n=== 2x2 (mixture-confident; total = {a['n_conf']:,}) ===")
    print(f"  {'':<16}{'HCG- (unmeth)':>20}{'HCG+ (meth)':>18}")
    print(f"  {'GCH+ (open)':<16}{c['open_unmeth']:>11,d} {f['open_unmeth']:>6.1%}"
          f"{c['open_meth']:>10,d} {f['open_meth']:>6.1%}")
    print(f"  {'GCH- (closed)':<16}{c['closed_unmeth']:>11,d} {f['closed_unmeth']:>6.1%}"
          f"{c['closed_meth']:>10,d} {f['closed_meth']:>6.1%}")
    print(f"  canonical (open-unmeth + closed-meth): "
          f"{f['open_unmeth']+f['closed_meth']:.1%}   "
          f"off-diagonal (open-meth + closed-unmeth): {a['offd']:.1%}")
    print(f"  PER-END  : mean a={a['ma']:.3f} m={a['mm']:.3f}  r(a,m) = {a['r_end']:+.3f}")
    if np.isfinite(a["r_reg"]):
        ou, om, cu, cm = a["reg22"]
        print(f"  REGION   : {a['n_bins']:,} bins  r(mean_a,mean_m) = {a['r_reg']:+.3f}"
              f"  (region 2x2 off-diag {om+cu:.1%})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gch-dir", required=True)
    ap.add_argument("--hcg-dir", required=True)
    ap.add_argument("--min-distance", type=int, default=20_000,
                    help="Lower bound of the linked band (single-band mode).")
    ap.add_argument("--max-distance", type=int, default=0,
                    help="Upper bound of the linked band (0 = none). Use a small "
                         "value (e.g. 20000) to test SHORT-range linked pairs.")
    ap.add_argument("--distance-bands", default=None,
                    help="Comma-separated bp edges → consecutive bands, swept in "
                         "ONE pass (e.g. 1000,20000,100000,1000000,10000000). "
                         "Overrides --min/--max-distance.")
    ap.add_argument("--bin-size", type=int, default=250_000,
                    help="Same-bin bin size for the linked filter (caps the "
                         "max separation a 'linked' pair can have).")
    ap.add_argument("--region-bin-size", type=int, default=0,
                    help="Bin size for the REGION-level a-m correlation (0 = use "
                         "--bin-size). Shrink it (e.g. 5000) to test the "
                         "gene-density/scale confound: if r(region) flips from + "
                         "at 250 kb to - at ~5 kb, the positive region coupling "
                         "was the coarse-scale gene-body confound.")
    ap.add_argument("--min-gpc-sites", type=int, default=2)
    ap.add_argument("--min-cpg-sites", type=int, default=2)
    ap.add_argument("--posterior-high", type=float, default=0.6)
    ap.add_argument("--posterior-low", type=float, default=0.4)
    ap.add_argument("--region-min-ends", type=int, default=20)
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--n-threads", type=int, default=4)
    args = ap.parse_args()
    if not (0 <= args.posterior_low <= args.posterior_high <= 1):
        sys.exit("require 0 <= --posterior-low <= --posterior-high <= 1")

    if args.distance_bands:
        edges = [int(x) for x in args.distance_bands.split(",")]
        if len(edges) < 2:
            sys.exit("--distance-bands needs >= 2 edges.")
        bands = list(zip(edges[:-1], edges[1:]))
    else:
        bands = [(args.min_distance, args.max_distance)]

    gch = _cells(Path(args.gch_dir)); hcg = _cells(Path(args.hcg_dir))
    shared = sorted(set(gch) & set(hcg))
    if not shared:
        sys.exit("No cells shared between --gch-dir and --hcg-dir.")
    if args.max_cells and args.max_cells > 0:
        shared = shared[: args.max_cells]
    print(f"[2x2] {len(shared)} shared cells | bin {args.bin_size:,} | min sites "
          f"GpC>={args.min_gpc_sites} CpG>={args.min_cpg_sites} | confidence band "
          f"[{args.posterior_low}, {args.posterior_high}] | "
          f"{len(bands)} distance band(s)")

    rbsz = args.region_bin_size if args.region_bin_size > 0 else args.bin_size
    at = (bands, args.bin_size, args.min_gpc_sites, args.min_cpg_sites,
          args.posterior_high, args.posterior_low, rbsz)
    if args.n_threads > 1 and len(shared) > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(gch[c], hcg[c], *at) for c in shared)
    else:
        res = [process_cell(gch[c], hcg[c], *at) for c in shared]
    res = [r for r in res if r is not None]
    if not res:
        sys.exit("No qualifying ends found.")

    aggs = [_agg_band([r[bi] for r in res], args.region_min_ends)
            for bi in range(len(bands))]

    print(f"\n=== distance-band sweep (per-end vs region coupling) ===")
    print(f"  {'band (bp)':<22}{'n_ends':>10}{'conf%':>7}{'r(end)':>9}"
          f"{'r(region)':>11}{'off-diag':>10}")
    for (loi, hii), a in zip(bands, aggs):
        hi_s = "inf" if not hii or hii <= 0 else f"{hii:,}"
        band_s = f"{loi:,}-{hi_s}"
        cf = (a["n_conf"] / a["n_both"]) if a["n_both"] else 0.0
        re = f"{a['r_end']:+.3f}" if np.isfinite(a["r_end"]) else "   n/a"
        rr = f"{a['r_reg']:+.3f}" if np.isfinite(a["r_reg"]) else "   n/a"
        print(f"  {band_s:<22}{a['n_both']:>10,d}{cf:>7.0%}{re:>9}{rr:>11}"
              f"{a['offd']:>9.1%}")

    if len(bands) == 1:
        _print_detail(aggs[0])
    else:
        print("\n  Read across bands: if r(region) is clearly negative at short")
        print("  range and rises toward 0 with distance, the canonical accessibility")
        print("  -methylation anti-correlation is a SHORT-range/local property that")
        print("  decouples at long range — so the 4-state joint carries genuinely")
        print("  independent information specifically for the long-range contacts.")
        print("  If r(region) ~ 0 in every band, the modalities are independent at")
        print("  all scales (joint = two orthogonal axes; a worse joint embedding is")
        print("  the pipeline, not biology).")


if __name__ == "__main__":
    main()
