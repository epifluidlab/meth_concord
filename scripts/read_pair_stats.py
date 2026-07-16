#!/usr/bin/env python3
"""Read-pair observability statistics: how many linked pairs carry enough GpC /
CpG sites, as stacked barplots + a summary TSV.

Only HIGH-QUALITY read pairs are used: uniquely mapped to the SAME chromosome,
not a PCR duplicate, mapQ > 30 on BOTH sides.  Those criteria are already
enforced upstream when the BAMs are parsed into ``02_read_pair_counts``
(``nome_hic_concordance/bam_parser.py``: the skip-mask drops UNMAPPED /
MATE_UNMAPPED / SECONDARY / DUPLICATE alignments, ``mapq_min`` (default 30) is
applied to both the read and its mate, and pairs are formed within a single
chromosome), so every pair in those parquets already qualifies.  Crucially the
parser emits EVERY such pair — including ones with zero context sites — so the
denominator here is the complete high-quality pair set, not just pairs that
happen to be covered.

Three stacked barplots, each bar splitting the pairs into
``neither side`` / ``one side only`` / ``both sides``:

  1. GpC (GCH accessibility) at >= 1, 2, 3, 4, 5 sites per side
  2. CpG (HCG methylation)   at >= 1, 2, 3 sites per side
  3. JOINT — a side counts only if it has >= t GpC **and** >= t CpG (what the
     joint multi-state method needs), at >= 1, 2, 3

The GCH and HCG runs are joined on (chrom, pos1, pos2) — the same reads, so the
same pair keys; a pair absent from one context simply has 0 sites there.

Usage
-----
    python scripts/read_pair_stats.py \\
        --gch-read-pairs-dir output/.../context_GCH/02_read_pair_counts \\
        --hcg-read-pairs-dir output/.../context_HCG/02_read_pair_counts \\
        --out read_pair_stats.pdf

    # restrict to long-range linked pairs only
    ... --min-distance 20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_QC_NOTE = ("Only high-quality read pairs: uniquely mapped to the same "
            "chromosome, not PCR duplicate, mapQ>30 on both sides "
            "(enforced upstream in bam_parser.py).")
_CATS = ("neither side", "one side only", "both sides")


def _cells(d: Path) -> dict:
    return {p.name.replace(".read_pairs.parquet", ""): p
            for p in d.glob("*.read_pairs.parquet")}


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["chrom", "pos1", "n1", "pos2", "n2",
                                        "distance"])
    df["chrom"] = df["chrom"].astype(str)
    return df


def subsample(shared, max_cells, seed):
    """All cells when ``max_cells <= 0`` or >= available, else a reproducible
    random subsample of that size (sorted for stable ordering)."""
    if max_cells and 0 < max_cells < len(shared):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(shared), size=max_cells, replace=False)
        return sorted(shared[i] for i in idx)
    return list(shared)


def _split(ok1, ok2):
    """#sides qualifying (0/1/2) → counts [neither, one side, both]."""
    k = ok1.astype(np.int64) + ok2.astype(np.int64)
    return np.bincount(k, minlength=3)[:3]


def process_cell(gpath, hpath, gch_thr, hcg_thr, joint_thr, mind, maxd):
    """One cell → (total pairs, gch (T,3), hcg (T,3), joint (T,3)) counts.

    Additive, so cells can be summed across threads."""
    g = _load(Path(gpath)).rename(columns={"n1": "n1g", "n2": "n2g"})
    h = _load(Path(hpath)).rename(columns={"n1": "n1h", "n2": "n2h"})
    m = g.merge(h[["chrom", "pos1", "pos2", "n1h", "n2h"]],
                on=["chrom", "pos1", "pos2"], how="outer")
    for c in ("n1g", "n2g", "n1h", "n2h"):
        m[c] = m[c].fillna(0)
    if "distance" in m:
        m["distance"] = m["distance"].fillna((m["pos2"] - m["pos1"]).abs())
    keep = np.ones(len(m), dtype=bool)
    if mind > 0:
        keep &= m["distance"].to_numpy() >= mind
    if maxd > 0:
        keep &= m["distance"].to_numpy() <= maxd
    if not keep.any():
        z = lambda t: np.zeros((len(t), 3), np.int64)     # noqa: E731
        return 0, z(gch_thr), z(hcg_thr), z(joint_thr)

    n1g = m["n1g"].to_numpy(np.float64)[keep]
    n2g = m["n2g"].to_numpy(np.float64)[keep]
    n1h = m["n1h"].to_numpy(np.float64)[keep]
    n2h = m["n2h"].to_numpy(np.float64)[keep]
    gch = np.stack([_split(n1g >= t, n2g >= t) for t in gch_thr])
    hcg = np.stack([_split(n1h >= t, n2h >= t) for t in hcg_thr])
    # JOINT: a side needs BOTH modalities at >= t
    joint = np.stack([_split((n1g >= t) & (n1h >= t), (n2g >= t) & (n2h >= t))
                      for t in joint_thr])
    return int(keep.sum()), gch, hcg, joint


def _pct(counts):
    tot = counts.sum(axis=1, keepdims=True).astype(np.float64)
    return 100.0 * counts / np.where(tot > 0, tot, 1.0)


def _plot(panels, out, n_cells, total, denom_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#d9d9d9", "#fdae61", "#2c7bb6")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, (title, thr, cnt, xlab) in zip(axes, panels):
        pct = _pct(cnt)
        x = np.arange(len(thr))
        bottom = np.zeros(len(thr))
        for ci, cat in enumerate(_CATS):
            ax.bar(x, pct[:, ci], bottom=bottom, color=colors[ci], label=cat,
                   width=0.66, edgecolor="white", linewidth=0.6)
            for xi in range(len(thr)):
                if pct[xi, ci] >= 4:
                    ax.text(x[xi], bottom[xi] + pct[xi, ci] / 2,
                            f"{pct[xi, ci]:.1f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if ci == 2 else "0.2")
            bottom += pct[:, ci]
        ax.set_xticks(x); ax.set_xticklabels([f"≥{t}" for t in thr])
        ax.set_xlabel(xlab); ax.set_ylim(0, 100)
        ax.set_ylabel(f"% of {denom_label}")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Read-pair observability  |  {n_cells} cells  |  "
                 f"{total:,} {denom_label}", fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.text(0.5, 0.012, _QC_NOTE, ha="center", fontsize=7.5, color="0.35",
             wrap=True)
    fig.savefig(out)
    plt.close(fig)


def _write_tsv(panels, path, total):
    rows = []
    for title, thr, cnt, _ in panels:
        pct = _pct(cnt)
        for ti, t in enumerate(thr):
            for ci, cat in enumerate(_CATS):
                rows.append({"panel": title, "min_sites_per_side": t,
                             "category": cat, "n_pairs": int(cnt[ti, ci]),
                             "pct": round(float(pct[ti, ci]), 4),
                             "total_pairs": total})
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _print_panel(title, thr, cnt):
    pct = _pct(cnt)
    print(f"\n=== {title} ===")
    print(f"  {'min sites/side':>15}{'neither':>22}{'one side':>22}{'both':>22}")
    for ti, t in enumerate(thr):
        cells = "".join(f"{cnt[ti, ci]:>13,d} {pct[ti, ci]:>6.2f}%"
                        for ci in range(3))
        print(f"  {'>= ' + str(t):>15}{cells}")


def _ints(s):
    return [int(x) for x in str(s).split(",") if str(x).strip() != ""]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gch-read-pairs-dir", required=True,
                    help="context_GCH/02_read_pair_counts")
    ap.add_argument("--hcg-read-pairs-dir", required=True,
                    help="context_HCG/02_read_pair_counts")
    ap.add_argument("--out", default="read_pair_stats.pdf",
                    help="Output PDF (a sibling .tsv of the counts is written too).")
    ap.add_argument("--gch-thresholds", default="1,2,3,4,5",
                    help="Min GpC sites per side to sweep (barplot 1).")
    ap.add_argument("--hcg-thresholds", default="1,2,3",
                    help="Min CpG sites per side to sweep (barplot 2).")
    ap.add_argument("--joint-thresholds", default="1,2,3",
                    help="Min GpC AND CpG sites per side to sweep (barplot 3).")
    ap.add_argument("--min-distance", type=int, default=0,
                    help="Restrict to linked pairs at least this far apart "
                         "(0 = every high-quality pair).")
    ap.add_argument("--max-distance", type=int, default=0,
                    help="Upper distance bound (0 = none).")
    ap.add_argument("--max-cells", type=int, default=0,
                    help="Random subsample of at most this many shared cells "
                         "(0 or negative = all).")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the random cell subsample.")
    ap.add_argument("--n-threads", type=int, default=4)
    args = ap.parse_args()
    gch_thr, hcg_thr, joint_thr = (_ints(args.gch_thresholds),
                                   _ints(args.hcg_thresholds),
                                   _ints(args.joint_thresholds))
    if not (gch_thr and hcg_thr and joint_thr):
        sys.exit("each --*-thresholds needs at least one integer")

    gch = _cells(Path(args.gch_read_pairs_dir))
    hcg = _cells(Path(args.hcg_read_pairs_dir))
    shared = sorted(set(gch) & set(hcg))
    if not shared:
        sys.exit("No cells shared between the two read-pair dirs.")
    sample = subsample(shared, args.max_cells, args.seed)
    mode = "all" if len(sample) == len(shared) else f"random, seed {args.seed}"
    band = ""
    if args.min_distance > 0 or args.max_distance > 0:
        hi = f"{args.max_distance:,}" if args.max_distance > 0 else "inf"
        band = f" | linked {args.min_distance:,}-{hi} bp"
    print(f"# {_QC_NOTE}")
    print(f"[read-pair-stats] {len(shared)} shared cells; using {len(sample)} "
          f"({mode}){band} | {args.n_threads} threads")

    jobs = [(gch[c], hcg[c], gch_thr, hcg_thr, joint_thr,
             args.min_distance, args.max_distance) for c in sample]
    if args.n_threads > 1 and len(jobs) > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(*j) for j in jobs)
    else:
        res = [process_cell(*j) for j in jobs]

    total = sum(r[0] for r in res)
    if total == 0:
        sys.exit("No qualifying read pairs found.")
    gch_c = sum(r[1] for r in res)
    hcg_c = sum(r[2] for r in res)
    joint_c = sum(r[3] for r in res)

    denom = ("high-quality linked pairs" if band else "high-quality read pairs")
    panels = [
        ("GpC (GCH accessibility) per side", gch_thr, gch_c,
         "min GpC sites per side"),
        ("CpG (HCG methylation) per side", hcg_thr, hcg_c,
         "min CpG sites per side"),
        ("JOINT: GpC AND CpG per side", joint_thr, joint_c,
         "min GpC and CpG sites per side"),
    ]
    print(f"\ntotal {denom}: {total:,}")
    for title, thr, cnt, _ in panels:
        _print_panel(title, thr, cnt)

    out = Path(args.out)
    _plot(panels, out, len(res), total, denom)
    tsv = out.with_suffix(".tsv")
    _write_tsv(panels, tsv, total)
    print(f"\nwrote {out}\n      {tsv}")
    print("\n'both sides' is the fraction usable by a pairwise method: the GCH "
          "and CpG\n  panels gate one modality, the JOINT panel needs BOTH "
          "modalities on BOTH ends\n  (what the joint multi-state concordance "
          "actually consumes).")


if __name__ == "__main__":
    main()
