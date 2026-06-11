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
        --min-gpc-sites 2 --min-cpg-sites 2 \\
        --min-distance 20000 --bin-size 250000 --max-cells 40 --n-threads 4
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


def subsample(shared, max_cells, seed):
    """The shared cells to process: **all** of them when ``max_cells`` is
    ``<= 0`` or ``>=`` the number available, else a **reproducible random**
    subsample of size ``max_cells`` (seeded by ``seed``).  Returned sorted so
    downstream order is stable regardless of which cells were drawn."""
    if max_cells and 0 < max_cells < len(shared):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(shared), size=max_cells, replace=False)
        return sorted(shared[i] for i in idx)
    return list(shared)


def process_cell(gpath, hpath, min_gpc, min_cpg, mind, bsz):
    """Tally one cell's observation completeness.

    Returns a dict of **additive** contributions (count dicts, scalar tallies,
    and per-(cell,bin) linked-count lists) so cells processed on different
    threads can simply be summed/concatenated.  An end is GpC-informative at
    ``>= min_gpc`` sites and CpG-informative at ``>= min_cpg`` sites — the two
    modalities have independent thresholds."""
    g = _load(Path(gpath)).rename(columns={"n1": "n1g", "n2": "n2g"})
    h = _load(Path(hpath)).rename(columns={"n1": "n1h", "n2": "n2h"})
    m = g.merge(h[["chrom", "pos1", "pos2", "n1h", "n2h"]],
                on=["chrom", "pos1", "pos2"], how="outer")
    for c in ("n1g", "n2g", "n1h", "n2h"):
        m[c] = m[c].fillna(0)
    if "distance" in m:
        m["distance"] = m["distance"].fillna((m["pos2"] - m["pos1"]).abs())
    # per-end informativeness (independent GpC / CpG thresholds)
    g1, h1 = m["n1g"] >= min_gpc, m["n1h"] >= min_cpg
    g2, h2 = m["n2g"] >= min_gpc, m["n2h"] >= min_cpg
    same_bin = (m["pos1"] // bsz) == (m["pos2"] // bsz)
    linked = same_bin & (m["distance"] >= mind)
    end_counts = {"both": 0, "gch_only": 0, "cpg_only": 0, "neither": 0}
    end_counts_link = {"both": 0, "gch_only": 0, "cpg_only": 0, "neither": 0}
    for (gi, hi) in ((g1, h1), (g2, h2)):
        end_counts["both"] += int((gi & hi).sum())
        end_counts["gch_only"] += int((gi & ~hi).sum())
        end_counts["cpg_only"] += int((~gi & hi).sum())
        end_counts["neither"] += int((~gi & ~hi).sum())
        end_counts_link["both"] += int((gi & hi & linked).sum())
        end_counts_link["gch_only"] += int((gi & ~hi & linked).sum())
        end_counts_link["cpg_only"] += int((~gi & hi & linked).sum())
        end_counts_link["neither"] += int((~gi & ~hi & linked).sum())
    strict = (g1 & h1) & (g2 & h2)
    lenient = (g1 | h1) & (g2 | h2)
    binid = (m["pos1"] // bsz)
    ls = pd.DataFrame({"bin": binid[linked],
                       "strict": (strict & linked)[linked].astype(int),
                       "lenient": (lenient & linked)[linked].astype(int)})
    if len(ls):
        grp = ls.groupby("bin").sum()
        per_bin_strict = grp["strict"].tolist()
        per_bin_lenient = grp["lenient"].tolist()
    else:
        per_bin_strict, per_bin_lenient = [], []
    return {
        "end_counts": end_counts, "end_counts_link": end_counts_link,
        "n_sb": int(same_bin.sum()),
        "n_strict": int((strict & same_bin).sum()),
        "n_lenient": int((lenient & same_bin).sum()),
        "n_sb_link": int(linked.sum()),
        "n_strict_link": int((strict & linked).sum()),
        "n_lenient_link": int((lenient & linked).sum()),
        "per_bin_strict": per_bin_strict, "per_bin_lenient": per_bin_lenient,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gch-read-pairs-dir", required=True)
    ap.add_argument("--hcg-read-pairs-dir", required=True)
    ap.add_argument("--min-sites", type=int, default=2,
                    help="Shared default min sites/end for BOTH modalities "
                         "(used where --min-gpc-sites/--min-cpg-sites are unset).")
    ap.add_argument("--min-gpc-sites", type=int, default=None,
                    help="Min GpC sites per end to call accessibility "
                         "informative (default: --min-sites).")
    ap.add_argument("--min-cpg-sites", type=int, default=None,
                    help="Min CpG sites per end to call methylation "
                         "informative (default: --min-sites).")
    ap.add_argument("--min-distance", type=int, default=20_000,
                    help="Long-range 'linked' threshold (bp).")
    ap.add_argument("--bin-size", type=int, default=250_000)
    ap.add_argument("--max-cells", type=int, default=40,
                    help="Random subsample of at most this many shared cells "
                         "(speed); 0 or negative = use ALL shared cells.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the random cell subsample (reproducible).")
    ap.add_argument("--n-threads", type=int, default=4,
                    help="Parallel worker threads over cells (joblib).")
    args = ap.parse_args()
    min_gpc = args.min_gpc_sites if args.min_gpc_sites is not None else args.min_sites
    min_cpg = args.min_cpg_sites if args.min_cpg_sites is not None else args.min_sites

    gch = _cells(Path(args.gch_read_pairs_dir))
    hcg = _cells(Path(args.hcg_read_pairs_dir))
    shared = sorted(set(gch) & set(hcg))
    if not shared:
        raise SystemExit("No cells shared between the two read-pair dirs.")
    sample = subsample(shared, args.max_cells, args.seed)
    mind, bsz = args.min_distance, args.bin_size
    mode = "all" if len(sample) == len(shared) else f"random, seed {args.seed}"
    print(f"{len(shared)} shared cells; using {len(sample)} ({mode}) | min sites "
          f"GpC>={min_gpc} CpG>={min_cpg} | bin {bsz:,} | {args.n_threads} threads")

    if args.n_threads > 1 and len(sample) > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(gch[c], hcg[c], min_gpc, min_cpg, mind, bsz)
            for c in sample)
    else:
        res = [process_cell(gch[c], hcg[c], min_gpc, min_cpg, mind, bsz)
               for c in sample]

    # additive aggregation across cells (order-independent → thread-invariant)
    keys = ("both", "gch_only", "cpg_only", "neither")
    end_counts = {k: sum(r["end_counts"][k] for r in res) for k in keys}
    # same breakdown, but only over the two ends of LINKED (long-range,
    # same-bin >= min_distance) pairs — the ends the joint method actually uses.
    end_counts_link = {k: sum(r["end_counts_link"][k] for r in res) for k in keys}
    n_sb = sum(r["n_sb"] for r in res)
    n_strict = sum(r["n_strict"] for r in res)
    n_lenient = sum(r["n_lenient"] for r in res)
    n_sb_link = sum(r["n_sb_link"] for r in res)
    n_strict_link = sum(r["n_strict_link"] for r in res)
    n_lenient_link = sum(r["n_lenient_link"] for r in res)
    per_bin_strict = [v for r in res for v in r["per_bin_strict"]]
    per_bin_lenient = [v for r in res for v in r["per_bin_lenient"]]

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

    tot_ends_link = sum(end_counts_link.values())
    print("\n=== per-end modality (over both ends of LINKED pairs only, "
          ">= {:,} bp) ===".format(mind))
    for k, v in end_counts_link.items():
        print(f"  {k:>9s}: {v:>10,d}  ({100*v/max(tot_ends_link,1):5.1f}%)")

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
