#!/usr/bin/env python3
"""2×2 epigenetic-state diagnostic — GCH accessibility × HCG methylation on the
SAME end, for the ends of long-range linked pairs that have BOTH modalities.

Per-end status is the **Beta-Binomial mixture posterior** already stored in the
``*.posteriors.parquet`` (``a = predict_proba(GpC) = P(accessible)``,
``m = predict_proba(CpG) = P(methylated)``).  That posterior is **coverage-aware**
(an end with few sites is pulled toward the genome-wide prior), so it is the
right per-end estimator — *not* a raw rate.

We then classify each end with a **confidence band** instead of a single 0.5 cut:

    posterior >= --posterior-high (0.6)  ->  positive (open / methylated)
    posterior <= --posterior-low  (0.4)  ->  negative (closed / unmethylated)
    in between                           ->  AMBIGUOUS (excluded)

so the 2×2 reflects ends the mixture can confidently call, not the (often
low-coverage) coin-flips a 0.5 threshold would force across, and not the soft
posterior product (which spreads mass off-diagonal purely from uncertainty).

                       HCG low (unmeth)        HCG high (meth)
    GCH high (open)    open, unmethylated      open, methylated     (active-meth)
    GCH low  (closed)  closed, unmethylated    closed, methylated
                       (poised)                (canonical closed)

Question it answers: do accessibility and methylation collapse onto the two
**anti-correlated / canonical** states (open-unmeth + closed-meth) — so a
**2-state** (open/closed) model suffices — or is the **off-diagonal** (open-meth
+ closed-unmeth) genuinely occupied, so the **4-state** model (design §2) carries
distinct information?

Usage
-----
    python scripts/joint_state_2x2.py \\
        --gch-dir output/.../context_GCH/04_posteriors \\
        --hcg-dir output/.../context_HCG/04_posteriors \\
        --min-distance 20000 --bin-size 250000 \\
        --min-gpc-sites 2 --min-cpg-sites 2 \\
        --posterior-high 0.6 --posterior-low 0.4 --max-cells 100
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


def process_cell(gpath, hpath, min_d, max_d, bsz, min_gpc, min_cpg, hi, lo):
    g = _load(Path(gpath)).rename(columns={"n1": "n1g", "n2": "n2g",
                                           "posterior1": "a1", "posterior2": "a2"})
    h = _load(Path(hpath)).rename(columns={"n1": "n1h", "n2": "n2h",
                                           "posterior1": "m1", "posterior2": "m2"})
    m = g.merge(h, on=["chrom", "pos1", "pos2"], how="inner")
    if len(m) == 0:
        return None
    pos1 = m["pos1"].to_numpy(np.int64); pos2 = m["pos2"].to_numpy(np.int64)
    dist = np.abs(pos2 - pos1)
    linked = (pos1 // bsz == pos2 // bsz) & (dist >= min_d)
    if max_d and max_d > 0:
        linked = linked & (dist <= max_d)

    chrom = m["chrom"].to_numpy()
    conf = {k: 0 for k in _STATES}
    n_both = n_conf = n_acc_amb = n_meth_amb = 0
    sa = sm = saa = smm = sam = 0.0; n_ends = 0
    bin_ch, bin_id, bin_a, bin_m = [], [], [], []   # for the region-level test
    for ng, nh, ac, mc, pc in (("n1g", "n1h", "a1", "m1", pos1),
                               ("n2g", "n2h", "a2", "m2", pos2)):
        a = m[ac].to_numpy(np.float64, na_value=np.nan)
        me = m[mc].to_numpy(np.float64, na_value=np.nan)
        ok = (linked & (m[ng].to_numpy(np.float64) >= min_gpc)
              & (m[nh].to_numpy(np.float64) >= min_cpg)
              & np.isfinite(a) & np.isfinite(me))
        a = a[ok]; me = me[ok]
        if a.size == 0:
            continue
        n_both += int(a.size)
        # mixture-posterior confidence band
        ap, an = a >= hi, a <= lo            # accessible+, closed-
        mp, mn = me >= hi, me <= lo          # methylated+, unmethylated-
        n_acc_amb += int((~ap & ~an).sum())
        n_meth_amb += int((~mp & ~mn).sum())
        conf["open_unmeth"] += int((ap & mn).sum())
        conf["open_meth"] += int((ap & mp).sum())
        conf["closed_unmeth"] += int((an & mn).sum())
        conf["closed_meth"] += int((an & mp).sum())
        n_conf += int(((ap | an) & (mp | mn)).sum())
        # coupling (continuous, threshold-free) over all both-modality ends
        sa += float(a.sum()); sm += float(me.sum())
        saa += float((a * a).sum()); smm += float((me * me).sum())
        sam += float((a * me).sum()); n_ends += int(a.size)
        # bin each end for the region-level (a, m) means
        bin_ch.append(chrom[ok]); bin_id.append((pc[ok] // bsz))
        bin_a.append(a); bin_m.append(me)
    if bin_a:
        be = pd.DataFrame({"chrom": np.concatenate(bin_ch),
                           "bin": np.concatenate(bin_id),
                           "a": np.concatenate(bin_a),
                           "m": np.concatenate(bin_m)})
        binsum = be.groupby(["chrom", "bin"]).agg(
            sa=("a", "sum"), sm=("m", "sum"), n=("a", "size")).reset_index()
    else:
        binsum = pd.DataFrame(columns=["chrom", "bin", "sa", "sm", "n"])
    return conf, (n_both, n_conf, n_acc_amb, n_meth_amb), \
        (sa, sm, saa, smm, sam, n_ends), binsum


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gch-dir", required=True)
    ap.add_argument("--hcg-dir", required=True)
    ap.add_argument("--min-distance", type=int, default=20_000)
    ap.add_argument("--max-distance", type=int, default=0)
    ap.add_argument("--bin-size", type=int, default=250_000)
    ap.add_argument("--min-gpc-sites", type=int, default=2)
    ap.add_argument("--min-cpg-sites", type=int, default=2)
    ap.add_argument("--posterior-high", type=float, default=0.6,
                    help="Mixture posterior >= this = positive (open/methylated).")
    ap.add_argument("--posterior-low", type=float, default=0.4,
                    help="Mixture posterior <= this = negative (closed/unmeth); "
                         "in between = ambiguous, excluded.")
    ap.add_argument("--region-min-ends", type=int, default=20,
                    help="Min ends per genomic bin to include it in the "
                         "region-level a-m correlation (default 20).")
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--n-threads", type=int, default=4)
    args = ap.parse_args()
    if not (0 <= args.posterior_low <= args.posterior_high <= 1):
        sys.exit("require 0 <= --posterior-low <= --posterior-high <= 1")

    gch = _cells(Path(args.gch_dir)); hcg = _cells(Path(args.hcg_dir))
    shared = sorted(set(gch) & set(hcg))
    if not shared:
        sys.exit("No cells shared between --gch-dir and --hcg-dir.")
    if args.max_cells and args.max_cells > 0:
        shared = shared[: args.max_cells]
    print(f"[2x2] {len(shared)} shared cells | linked >= {args.min_distance:,} bp "
          f"(bin {args.bin_size:,}) | min sites GpC>={args.min_gpc_sites} "
          f"CpG>={args.min_cpg_sites} | confidence band [{args.posterior_low}, "
          f"{args.posterior_high}] (mixture posterior; ambiguous excluded)")

    at = (args.min_distance, args.max_distance, args.bin_size,
          args.min_gpc_sites, args.min_cpg_sites,
          args.posterior_high, args.posterior_low)
    if args.n_threads > 1 and len(shared) > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(gch[c], hcg[c], *at) for c in shared)
    else:
        res = [process_cell(gch[c], hcg[c], *at) for c in shared]
    res = [r for r in res if r is not None]
    if not res:
        sys.exit("No qualifying ends found.")

    conf = {k: sum(r[0][k] for r in res) for k in _STATES}
    n_both = sum(r[1][0] for r in res); n_conf = sum(r[1][1] for r in res)
    n_acc_amb = sum(r[1][2] for r in res); n_meth_amb = sum(r[1][3] for r in res)
    sa = sum(r[2][0] for r in res); sm = sum(r[2][1] for r in res)
    saa = sum(r[2][2] for r in res); smm = sum(r[2][3] for r in res)
    sam = sum(r[2][4] for r in res); n = sum(r[2][5] for r in res)

    print(f"\n[2x2] ends with both modalities on a linked pair: {n_both:,}")
    print(f"      ambiguous (mixture posterior in the band, excluded): "
          f"accessibility {n_acc_amb/max(n_both,1):.1%}, "
          f"methylation {n_meth_amb/max(n_both,1):.1%}")
    print(f"      confidently classified in BOTH (used in the 2x2): "
          f"{n_conf:,}  ({n_conf/max(n_both,1):.1%})")

    nc = max(n_conf, 1)
    f = {k: conf[k] / nc for k in _STATES}
    print(f"\n=== 2x2 (mixture-confident hard counts; total = {n_conf:,}) ===")
    print(f"  {'':<16}{'HCG- (unmeth)':>20}{'HCG+ (meth)':>18}")
    print(f"  {'GCH+ (open)':<16}{conf['open_unmeth']:>11,d} {f['open_unmeth']:>6.1%}"
          f"{conf['open_meth']:>10,d} {f['open_meth']:>6.1%}")
    print(f"  {'GCH- (closed)':<16}{conf['closed_unmeth']:>11,d} {f['closed_unmeth']:>6.1%}"
          f"{conf['closed_meth']:>10,d} {f['closed_meth']:>6.1%}")
    canon = f["open_unmeth"] + f["closed_meth"]
    offd = f["open_meth"] + f["closed_unmeth"]
    print(f"  canonical / anti-correlated (open-unmeth + closed-meth): {canon:6.1%}")
    print(f"  off-diagonal / aligned      (open-meth + closed-unmeth): {offd:6.1%}")

    ma = sa / n if n else np.nan; mm = sm / n if n else np.nan
    cov = sam / n - ma * mm
    va = saa / n - ma * ma; vm = smm / n - mm * mm
    r = cov / np.sqrt(va * vm) if va > 0 and vm > 0 else np.nan
    print(f"\n=== coupling ===")
    print(f"  PER-END:  mean a={ma:.3f}  mean m={mm:.3f}  "
          f"Pearson r(a,m) across ends = {r:+.3f}")

    # --- REGION level: average a, m per genomic bin, correlate across bins ----
    # Averaging many ends per bin removes the per-end estimation noise, so this
    # separates a true coupling from per-end attenuation, and is the scale the
    # concordance / embedding actually operates at.
    r_reg = np.nan; n_bins = 0; offd_reg = np.nan
    allbin = [r4 for r4 in (rr[3] for rr in res) if len(r4)]
    if allbin:
        bs = pd.concat(allbin, ignore_index=True).groupby(
            ["chrom", "bin"]).agg(sa=("sa", "sum"), sm=("sm", "sum"),
                                  n=("n", "sum")).reset_index()
        bs = bs[bs["n"] >= args.region_min_ends]
        n_bins = len(bs)
        if n_bins >= 3:
            x = (bs["sa"] / bs["n"]).to_numpy()
            y = (bs["sm"] / bs["n"]).to_numpy()
            w = bs["n"].to_numpy().astype(float)
            mx = np.average(x, weights=w); my = np.average(y, weights=w)
            cxy = np.average((x - mx) * (y - my), weights=w)
            vx = np.average((x - mx) ** 2, weights=w)
            vy = np.average((y - my) ** 2, weights=w)
            r_reg = cxy / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan
            # region 2x2: split bins at the (end-weighted) mean of each axis
            ah = x >= mx; mh = y >= my
            tw = w.sum()
            ou = w[ah & ~mh].sum() / tw; om = w[ah & mh].sum() / tw
            cu = w[~ah & ~mh].sum() / tw; cm = w[~ah & mh].sum() / tw
            offd_reg = om + cu
            print(f"  REGION ({args.bin_size:,} bp, {n_bins:,} bins with "
                  f">= {args.region_min_ends} ends):  Pearson r(mean_a, mean_m) "
                  f"= {r_reg:+.3f}")
            print(f"    region 2x2 (split at the mean): open-unmeth {ou:.1%}  "
                  f"open-meth {om:.1%}  closed-unmeth {cu:.1%}  closed-meth "
                  f"{cm:.1%}  (off-diag {offd_reg:.1%})")

    print(f"\n=== verdict ===")
    if offd < 0.10 or (np.isfinite(r) and r < -0.5):
        print("  → strongly ANTI-CORRELATED per end, off-diagonal rare: a 2-STATE")
        print("    (open/closed) model captures it; methylation is largely")
        print("    redundant with accessibility (a joint 4-state would not beat")
        print("    the single contexts).")
    elif offd >= 0.25 or (np.isfinite(r) and r > -0.2):
        print("  → the off-diagonal states are genuinely occupied / a,m not")
        print("    strongly anti-correlated PER END: a 4-state grid is occupied.")
    else:
        print("  → intermediate per end.")
    # The region-level r is what matters for the joint embedding / concordance:
    if np.isfinite(r_reg):
        if r_reg < -0.4:
            print(f"  → BUT region-level r = {r_reg:+.3f}: accessibility and")
            print("    methylation ARE anti-correlated at the bin scale (the per-end")
            print("    independence was noise). Region-wise the two modalities are")
            print("    largely redundant → the joint adds little over the better")
            print("    single context (consistent with the flat joint embedding).")
        elif abs(r_reg) <= 0.2:
            print(f"  → region-level r = {r_reg:+.3f}: accessibility and methylation")
            print("    are independent at the bin scale TOO (not per-end noise). The")
            print("    two axes carry SEPARATE region-level info → a joint embedding")
            print("    SHOULD be >= the better single context; a WORSE joint is then")
            print("    the pipeline (resid_full / log2(O/E) vs signed-diff / 1 Mb /")
            print("    channel dilution), not a biology limit.")
        else:
            print(f"  → region-level r = {r_reg:+.3f}: weak/intermediate coupling at")
            print("    the bin scale; the joint may add modest information.")


if __name__ == "__main__":
    main()
