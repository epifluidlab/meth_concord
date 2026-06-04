#!/usr/bin/env python3
"""2×2 epigenetic-state diagnostic — GCH accessibility × HCG methylation on the
SAME end, for the ends of long-range linked pairs that have BOTH modalities.

For every end of a long-range linked pair (same-bin, separation ≥ --min-distance)
whose end carries both GpC (≥ --min-gpc-sites) and CpG (≥ --min-cpg-sites) sites,
take ``a = P(accessible | GpC)`` (GCH posterior) and ``m = P(methylated | CpG)``
(HCG posterior) and place the end in the 2×2 joint-state grid:

                       HCG low (unmeth)        HCG high (meth)
    GCH high (open)    open, unmethylated      open, methylated     (active-meth)
    GCH low  (closed)  closed, unmethylated    closed, methylated
                       (poised)                (canonical closed)

Reported two ways:
  * HARD  — threshold the posteriors at --threshold (count of ends per cell).
  * SOFT  — posterior mass  a·(1−m), a·m, (1−a)·(1−m), (1−a)·m  (threshold-free).

Question it answers: do accessibility and methylation collapse onto the two
**anti-correlated / canonical** states (open-unmethylated + closed-methylated) —
so a **2-state** (open/closed) model captures the biology — or is the
**off-diagonal** mass (open-methylated + closed-unmethylated) substantial, so the
**4-state** model (design §2) carries real, distinct information?

Usage
-----
    python scripts/joint_state_2x2.py \\
        --gch-dir output/.../context_GCH/04_posteriors \\
        --hcg-dir output/.../context_HCG/04_posteriors \\
        --min-distance 20000 --bin-size 250000 \\
        --min-gpc-sites 1 --min-cpg-sites 1 --max-cells 100
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

# 2×2 cell order: (GCH_high, HCG_low), (GCH_high, HCG_high),
#                 (GCH_low,  HCG_low), (GCH_low,  HCG_high)
_STATES = ("open_unmeth", "open_meth", "closed_unmeth", "closed_meth")
_LABELS = {
    "open_unmeth":   "open, unmethylated   (GCH_high+HCG_low) ",
    "open_meth":     "open, methylated     (GCH_high+HCG_high)",
    "closed_unmeth": "closed, unmethylated (GCH_low +HCG_low) ",
    "closed_meth":   "closed, methylated   (GCH_low +HCG_high)",
}


def _cells(d: Path) -> dict:
    return {p.name.replace(".posteriors.parquet", ""): p
            for p in d.glob("*.posteriors.parquet")}


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                        "posterior1", "posterior2"])
    df["chrom"] = df["chrom"].astype(str)
    return df


def process_cell(gpath, hpath, min_d, max_d, bsz, min_gpc, min_cpg, thr):
    g = _load(Path(gpath)).rename(columns={"n1": "n1g", "n2": "n2g",
                                           "posterior1": "a1", "posterior2": "a2"})
    h = _load(Path(hpath)).rename(columns={"n1": "n1h", "n2": "n2h",
                                           "posterior1": "m1", "posterior2": "m2"})
    m = g.merge(h, on=["chrom", "pos1", "pos2"], how="inner")
    if len(m) == 0:
        return None
    pos1 = m["pos1"].to_numpy(np.int64); pos2 = m["pos2"].to_numpy(np.int64)
    dist = np.abs(pos2 - pos1)
    same_bin = (pos1 // bsz) == (pos2 // bsz)
    linked = same_bin & (dist >= min_d)
    if max_d and max_d > 0:
        linked = linked & (dist <= max_d)

    hard = {k: 0 for k in _STATES}
    soft = {k: 0.0 for k in _STATES}
    sa = sm = saa = smm = sam = 0.0
    n_ends = 0
    for ng, nh, ac, mc in (("n1g", "n1h", "a1", "m1"),
                           ("n2g", "n2h", "a2", "m2")):
        a = m[ac].to_numpy(np.float64, na_value=np.nan)
        me = m[mc].to_numpy(np.float64, na_value=np.nan)
        ok = (linked & (m[ng].to_numpy(np.float64) >= min_gpc)
              & (m[nh].to_numpy(np.float64) >= min_cpg)
              & np.isfinite(a) & np.isfinite(me))
        a = a[ok]; me = me[ok]
        if a.size == 0:
            continue
        gh = a > thr; mh = me > thr
        hard["open_unmeth"] += int((gh & ~mh).sum())
        hard["open_meth"] += int((gh & mh).sum())
        hard["closed_unmeth"] += int((~gh & ~mh).sum())
        hard["closed_meth"] += int((~gh & mh).sum())
        soft["open_unmeth"] += float((a * (1 - me)).sum())
        soft["open_meth"] += float((a * me).sum())
        soft["closed_unmeth"] += float(((1 - a) * (1 - me)).sum())
        soft["closed_meth"] += float(((1 - a) * me).sum())
        sa += float(a.sum()); sm += float(me.sum())
        saa += float((a * a).sum()); smm += float((me * me).sum())
        sam += float((a * me).sum()); n_ends += int(a.size)
    return hard, soft, (sa, sm, saa, smm, sam, n_ends)


def _print_2x2(title, tbl, total):
    """tbl: dict of the 4 states (count or soft mass); prints a 2×2 + fractions."""
    def f(k):
        return tbl[k] / total if total > 0 else 0.0
    print(f"\n{title}  (total = {total:,.0f})")
    print(f"  {'':<14}{'HCG low (unmeth)':>22}{'HCG high (meth)':>22}")
    print(f"  {'GCH high (open)':<14}{f('open_unmeth'):>21.1%}{f('open_meth'):>22.1%}")
    print(f"  {'GCH low (closed)':<14}{f('closed_unmeth'):>21.1%}{f('closed_meth'):>22.1%}")
    canon = f("open_unmeth") + f("closed_meth")
    offdiag = f("open_meth") + f("closed_unmeth")
    print(f"  canonical/anti-correlated (open-unmeth + closed-meth): {canon:6.1%}")
    print(f"  off-diagonal/aligned     (open-meth + closed-unmeth) : {offdiag:6.1%}")
    return canon, offdiag


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gch-dir", required=True,
                    help="Dir of *.posteriors.parquet for GCH (accessibility).")
    ap.add_argument("--hcg-dir", required=True,
                    help="Dir of *.posteriors.parquet for HCG (methylation).")
    ap.add_argument("--min-distance", type=int, default=20_000)
    ap.add_argument("--max-distance", type=int, default=0,
                    help="Optional upper bound on the linked separation (0 = none).")
    ap.add_argument("--bin-size", type=int, default=250_000)
    ap.add_argument("--min-gpc-sites", type=int, default=1)
    ap.add_argument("--min-cpg-sites", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Posterior cut for the HARD table (high vs low). The "
                         "SOFT table is threshold-free.")
    ap.add_argument("--max-cells", type=int, default=0,
                    help="Sample at most this many shared cells (0 = all).")
    ap.add_argument("--n-threads", type=int, default=4)
    args = ap.parse_args()

    gch = _cells(Path(args.gch_dir)); hcg = _cells(Path(args.hcg_dir))
    shared = sorted(set(gch) & set(hcg))
    if not shared:
        sys.exit("No cells shared between --gch-dir and --hcg-dir.")
    if args.max_cells and args.max_cells > 0:
        shared = shared[: args.max_cells]
    print(f"[2x2] {len(shared)} shared cells | linked >= {args.min_distance:,} bp "
          f"(bin {args.bin_size:,}) | min sites GpC>={args.min_gpc_sites} "
          f"CpG>={args.min_cpg_sites} | hard threshold={args.threshold}")

    args_t = (args.min_distance, args.max_distance, args.bin_size,
              args.min_gpc_sites, args.min_cpg_sites, args.threshold)
    if args.n_threads > 1 and len(shared) > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(gch[c], hcg[c], *args_t) for c in shared)
    else:
        res = [process_cell(gch[c], hcg[c], *args_t) for c in shared]
    res = [r for r in res if r is not None]
    if not res:
        sys.exit("No qualifying ends found.")

    hard = {k: sum(r[0][k] for r in res) for k in _STATES}
    soft = {k: sum(r[1][k] for r in res) for k in _STATES}
    sa = sum(r[2][0] for r in res); sm = sum(r[2][1] for r in res)
    saa = sum(r[2][2] for r in res); smm = sum(r[2][3] for r in res)
    sam = sum(r[2][4] for r in res); n = sum(r[2][5] for r in res)

    n_hard = sum(hard.values()); n_soft = sum(soft.values())
    print(f"\n[2x2] qualifying ends (both modalities, linked): {n:,}")
    canon_h, off_h = _print_2x2("=== HARD counts (threshold) ===", hard, n_hard)
    canon_s, off_s = _print_2x2("=== SOFT posterior mass ===", soft, n_soft)

    # Marginals + independence-expected 2×2 (would the joint be the product?).
    ma = sa / n if n else np.nan; mm = sm / n if n else np.nan
    print(f"\n=== marginals + coupling ===")
    print(f"  mean accessibility a = {ma:.3f}   mean methylation m = {mm:.3f}")
    print(f"  expected under INDEPENDENCE (a marg ⊗ m marg):")
    print(f"    open-unmeth a(1-m)={ma*(1-mm):.1%}  open-meth a·m={ma*mm:.1%}  "
          f"closed-unmeth (1-a)(1-m)={(1-ma)*(1-mm):.1%}  "
          f"closed-meth (1-a)m={(1-ma)*mm:.1%}")
    # Pearson correlation of (a, m) across ends.
    cov = sam / n - ma * mm
    va = saa / n - ma * ma; vm = smm / n - mm * mm
    r = cov / np.sqrt(va * vm) if va > 0 and vm > 0 else np.nan
    print(f"  Pearson r(a, m) across ends = {r:+.3f}")

    print(f"\n=== verdict ===")
    print(f"  HARD off-diagonal (thresholded states) = {off_h:.1%}   "
          f"Pearson r(a,m) = {r:+.3f}")
    print(f"  NB: base the 2-vs-4 decision on the HARD table + the correlation. "
          f"The\n      SOFT off-diagonal ({off_s:.1%}) is INFLATED by posterior "
          f"uncertainty —\n      the products a·m, (1−a)(1−m) spread mass whenever "
          f"a,m sit near 0.5,\n      which is smear, not genuine off-diagonal "
          f"biology.")
    if off_h < 0.10 or (np.isfinite(r) and r < -0.5):
        print("  → accessibility and methylation are strongly ANTI-CORRELATED and")
        print("    the off-diagonal states are rare: a 2-STATE (open/closed) model")
        print("    captures the biology; the methylation axis is largely redundant")
        print("    with accessibility on each end.")
    elif off_h >= 0.25 or (np.isfinite(r) and r > -0.2):
        print("  → the off-diagonal states are genuinely occupied (and/or a,m are")
        print("    not strongly anti-correlated): the 4-STATE model is warranted —")
        print("    methylation carries information beyond accessibility on each end.")
    else:
        print("  → intermediate: the off-diagonal states are non-trivial but a "
              "minority.")
        print("    The 4-state model adds some information; weigh it against the "
              "extra")
        print("    sparsity/parameters (see the joint observability check).")


if __name__ == "__main__":
    main()
