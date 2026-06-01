#!/usr/bin/env python3
"""Phase J0 — is the joint GCH+HCG concordance feasible on this data?

The joint multi-state method needs both modalities at both ends of a
contact.  Before building it, this measures how many pairs actually
qualify, by joining the per-cell read pairs from the GCH and HCG context
runs on (chrom, pos1, pos2) and tallying observation completeness.

It reports, pooled over a sample of cells:
  * per-end modality: fraction with GpC-only / CpG-only / both / neither;
  * per (same-bin) pair: fraction fully-observed (strict: both modalities
    at both ends) vs lenient (each end informative in >= 1 modality);
  * per (cell, bin) long-range linked-pair counts under strict vs lenient.

Caveat: the GCH and HCG runs are joined on (chrom, pos1, pos2); this
assumes the read-end midpoints match across the two context runs (they
should, being the same reads).  A pair present in only one context is
treated as having that modality and missing the other.

Usage
-----
    python scripts/joint_observability_check.py \\
        --gch-read-pairs-dir output/.../context_GCH/02_read_pair_counts \\
        --hcg-read-pairs-dir output/.../context_HCG/02_read_pair_counts \\
        --min-sites 2 --min-distance 20000 --bin-size 250000 --max-cells 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _cells(d: Path) -> dict[str, Path]:
    return {p.name.replace(".read_pairs.parquet", ""): p
            for p in d.glob("*.read_pairs.parquet")}


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["chrom", "pos1", "n1", "pos2", "n2",
                                        "distance"])
    df["chrom"] = df["chrom"].astype(str)
    return df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gch-read-pairs-dir", required=True)
    ap.add_argument("--hcg-read-pairs-dir", required=True)
    ap.add_argument("--min-sites", type=int, default=2,
                    help="Min context sites per end to call a modality informative.")
    ap.add_argument("--min-distance", type=int, default=20_000,
                    help="Long-range 'linked' threshold (bp).")
    ap.add_argument("--bin-size", type=int, default=250_000)
    ap.add_argument("--max-cells", type=int, default=40,
                    help="Sample at most this many shared cells (speed).")
    args = ap.parse_args()

    gch = _cells(Path(args.gch_read_pairs_dir))
    hcg = _cells(Path(args.hcg_read_pairs_dir))
    shared = sorted(set(gch) & set(hcg))
    if not shared:
        raise SystemExit("No cells shared between the two read-pair dirs.")
    sample = shared[: args.max_cells]
    print(f"{len(shared)} shared cells; sampling {len(sample)}.")

    thr, mind, bsz = args.min_sites, args.min_distance, args.bin_size
    end_counts = {"both": 0, "gch_only": 0, "cpg_only": 0, "neither": 0}
    n_sb = n_strict = n_lenient = 0           # same-bin pair tallies
    n_sb_link = n_strict_link = n_lenient_link = 0
    per_bin_strict, per_bin_lenient = [], []

    for cell in sample:
        g = _load(gch[cell]).rename(columns={"n1": "n1g", "n2": "n2g"})
        h = _load(hcg[cell]).rename(columns={"n1": "n1h", "n2": "n2h"})
        m = g.merge(h[["chrom", "pos1", "pos2", "n1h", "n2h"]],
                    on=["chrom", "pos1", "pos2"], how="outer")
        for c in ("n1g", "n2g", "n1h", "n2h"):
            m[c] = m[c].fillna(0)
        if "distance" in m:
            m["distance"] = m["distance"].fillna(
                (m["pos2"] - m["pos1"]).abs())
        # per-end informativeness
        g1, h1 = m["n1g"] >= thr, m["n1h"] >= thr
        g2, h2 = m["n2g"] >= thr, m["n2h"] >= thr
        for (gi, hi) in ((g1, h1), (g2, h2)):
            both = (gi & hi).sum()
            go = (gi & ~hi).sum(); ho = (~gi & hi).sum()
            nei = (~gi & ~hi).sum()
            end_counts["both"] += int(both); end_counts["gch_only"] += int(go)
            end_counts["cpg_only"] += int(ho); end_counts["neither"] += int(nei)
        # pair classes
        strict = (g1 & h1) & (g2 & h2)
        lenient = (g1 | h1) & (g2 | h2)
        same_bin = (m["pos1"] // bsz) == (m["pos2"] // bsz)
        linked = same_bin & (m["distance"] >= mind)
        n_sb += int(same_bin.sum())
        n_strict += int((strict & same_bin).sum())
        n_lenient += int((lenient & same_bin).sum())
        n_sb_link += int(linked.sum())
        n_strict_link += int((strict & linked).sum())
        n_lenient_link += int((lenient & linked).sum())
        # per (cell, bin) linked counts
        binid = (m["pos1"] // bsz)
        ls = pd.DataFrame({"bin": binid[linked],
                           "strict": (strict & linked)[linked].astype(int),
                           "lenient": (lenient & linked)[linked].astype(int)})
        if len(ls):
            grp = ls.groupby("bin").sum()
            per_bin_strict += grp["strict"].tolist()
            per_bin_lenient += grp["lenient"].tolist()

    tot_ends = sum(end_counts.values())
    print("\n=== per-end modality (over both ends of all joined pairs) ===")
    for k, v in end_counts.items():
        print(f"  {k:>9s}: {v:>10,d}  ({100*v/max(tot_ends,1):5.1f}%)")

    print("\n=== same-bin pairs ===")
    print(f"  total same-bin pairs : {n_sb:>10,d}")
    print(f"  strict (both mods, both ends) : {n_strict:>10,d}  "
          f"({100*n_strict/max(n_sb,1):5.1f}%)")
    print(f"  lenient (>=1 mod each end)    : {n_lenient:>10,d}  "
          f"({100*n_lenient/max(n_sb,1):5.1f}%)")

    print("\n=== long-range LINKED same-bin pairs (>= {:,} bp) ===".format(mind))
    print(f"  linked pairs (any obs): {n_sb_link:>10,d}")
    print(f"  strict  : {n_strict_link:>10,d}  ({100*n_strict_link/max(n_sb_link,1):5.1f}%)")
    print(f"  lenient : {n_lenient_link:>10,d}  ({100*n_lenient_link/max(n_sb_link,1):5.1f}%)")

    def _med(x):
        return float(np.median(x)) if x else 0.0
    print("\n=== per (cell, bin) linked-pair counts ===")
    print(f"  strict : median {_med(per_bin_strict):.1f}  "
          f"(n bins with >=1: {len(per_bin_strict)})")
    print(f"  lenient: median {_med(per_bin_lenient):.1f}  "
          f"(n bins with >=1: {len(per_bin_lenient)})")

    print("\nInterpretation:")
    print("  • lenient ~ strict and per-(cell,bin) median >~ a few  -> joint method viable.")
    print("  • lenient >> strict  -> most pairs need prior-marginalization (lenient mode")
    print("    is doing the heavy lifting; the baseline subtraction must carry the artifact).")
    print("  • per-(cell,bin) median ~ 0–1  -> too sparse; lean harder on RWR or coarsen bins.")


if __name__ == "__main__":
    main()
