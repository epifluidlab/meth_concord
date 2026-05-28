#!/usr/bin/env python3
"""Diagnostic: count effective long-range linked pairs per (cell, bin).

Reads Step-2 per-cell parquets (produced by ``nome_hic_concordance`` Step 2 —
read-quality filters like ``--mapq-min`` / ``--trim-ends`` were already applied
when those parquets were written) and counts pairs that satisfy:

  - both ends on the same chromosome AND in the same bin of size ``--bin-size``
  - distance >= ``--min-distance``
  - both ends informative: n_GCH on each end >= ``--min-gch-sites``

Prints a per-cell summary and the overall (cell, bin) n_linked distribution
so you can decide whether the requested bin size is viable for downstream
concordance computation.

Example:
    python scripts/check_linked_pair_density.py \\
        --input-dir output/betabinom_250kb/GCH/02_read_pair_counts \\
        --bin-size 250000 \\
        --min-distance 1000 \\
        --min-gch-sites 3 \\
        --n-threads 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def process_cell(
    path: Path,
    bin_size: int,
    min_distance: int,
    min_gch_sites: int,
) -> tuple[str, int, pd.Series]:
    """Return (cell_name, n_linked_total, pairs_per_bin) for one cell."""
    cell_name = path.stem.replace(".read_pairs", "")
    df = pd.read_parquet(path, columns=["chrom", "pos1", "pos2", "n1", "n2", "distance"])

    if len(df) == 0:
        return cell_name, 0, pd.Series([], dtype=np.int64)

    chrom = df["chrom"].astype(str).values
    pos1 = df["pos1"].astype(np.int64).values
    pos2 = df["pos2"].astype(np.int64).values
    n1 = df["n1"].astype(np.int32).values
    n2 = df["n2"].astype(np.int32).values
    distance = df["distance"].astype(np.int64).values

    bin1 = (pos1 // bin_size) * bin_size
    bin2 = (pos2 // bin_size) * bin_size

    same_bin = bin1 == bin2
    informative = (n1 >= min_gch_sites) & (n2 >= min_gch_sites)
    long_range = distance >= min_distance

    linked = same_bin & informative & long_range
    n_linked_total = int(linked.sum())

    if n_linked_total == 0:
        return cell_name, 0, pd.Series([], dtype=np.int64)

    chrom_linked = chrom[linked]
    bin_linked = bin1[linked]
    # Use a pandas MultiIndex value_count to avoid numpy str-dtype issues
    # when chrom is object and bin is int.
    pairs_per_bin = (
        pd.DataFrame({"chrom": chrom_linked, "bin": bin_linked})
        .groupby(["chrom", "bin"], sort=False)
        .size()
    )
    return cell_name, n_linked_total, pairs_per_bin


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing per-cell *.read_pairs.parquet files "
             "(typically output/.../02_read_pair_counts/).",
    )
    parser.add_argument("--bin-size", type=int, default=250_000,
                        help="Bin size in bp (default: 250000).")
    parser.add_argument("--min-distance", type=int, default=1_000,
                        help="Minimum end-to-end distance for 'linked' pairs (default: 1000).")
    parser.add_argument("--min-gch-sites", type=int, default=3,
                        help="Minimum GCH sites per end to count as informative (default: 3).")
    parser.add_argument("--n-threads", type=int, default=4,
                        help="Number of parallel workers (default: 4).")
    parser.add_argument("--sample", type=int, default=0,
                        help="If > 0, randomly sample this many cells for a faster check (default: 0 = all).")
    parser.add_argument("--output-prefix", default=None,
                        help="If set, write per-cell + per-bin TSVs to <prefix>.{per_cell,per_bin}.tsv.")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    files = sorted(in_dir.glob("*.read_pairs.parquet"))
    if not files:
        print(f"ERROR: no *.read_pairs.parquet files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    if args.sample > 0 and args.sample < len(files):
        rng = np.random.default_rng(args.random_seed)
        idx = rng.choice(len(files), size=args.sample, replace=False)
        files = [files[i] for i in sorted(idx)]
        print(f"Subsampled {args.sample} of {len(files)} parquets (seed={args.random_seed})")

    print(
        f"Processing {len(files)} cell(s) with "
        f"bin_size={args.bin_size:,}, "
        f"min_distance={args.min_distance:,}, "
        f"min_gch_sites={args.min_gch_sites}"
    )

    if args.n_threads > 1 and len(files) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(p, args.bin_size, args.min_distance, args.min_gch_sites)
            for p in files
        )
    else:
        results = [
            process_cell(p, args.bin_size, args.min_distance, args.min_gch_sites)
            for p in files
        ]

    # ---- Per-cell summary --------------------------------------------------
    per_cell_rows = []
    all_ppb_list: list[pd.Series] = []
    for cell_name, n_linked_total, ppb in results:
        per_cell_rows.append({
            "cell": cell_name,
            "n_linked_total": n_linked_total,
            "n_bins_with_data": int(len(ppb)),
            "median_pairs_per_bin": float(ppb.median()) if len(ppb) > 0 else 0.0,
            "mean_pairs_per_bin":   float(ppb.mean())   if len(ppb) > 0 else 0.0,
            "max_pairs_per_bin":    int(ppb.max())      if len(ppb) > 0 else 0,
        })
        if len(ppb) > 0:
            all_ppb_list.append(ppb)
    per_cell_df = pd.DataFrame(per_cell_rows)

    print("\n" + "=" * 72)
    print("Per-cell summary")
    print("=" * 72)
    summary = per_cell_df[[
        "n_linked_total",
        "n_bins_with_data",
        "median_pairs_per_bin",
        "mean_pairs_per_bin",
        "max_pairs_per_bin",
    ]].describe(percentiles=[0.25, 0.5, 0.75])
    pd.set_option("display.float_format", lambda x: f"{x:.1f}")
    print(summary)

    # ---- (cell, bin) distribution -----------------------------------------
    if not all_ppb_list:
        print("\nNo cell has any qualifying linked pairs at these filters.")
        return

    combined = pd.concat(all_ppb_list, ignore_index=True)
    n_cb = len(combined)
    print("\n" + "=" * 72)
    print(f"n_linked per (cell, bin) — {n_cb:,} (cell, bin) records with n_linked >= 1")
    print("=" * 72)
    for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"  {q*100:5.0f}th percentile: {combined.quantile(q):6.1f}")
    print(f"  mean:               {combined.mean():6.2f}")
    print(f"  max:                {combined.max():6d}")

    print("\nHistogram of n_linked per (cell, bin):")
    buckets = [
        (1, 1), (2, 2), (3, 3), (4, 5),
        (6, 10), (11, 25), (26, 100), (101, None),
    ]
    for lo, hi in buckets:
        if hi is None:
            mask = combined >= lo
            label = f"n_linked >= {lo}"
        elif lo == hi:
            mask = combined == lo
            label = f"n_linked = {lo}"
        else:
            mask = (combined >= lo) & (combined <= hi)
            label = f"n_linked = {lo}-{hi}"
        c = int(mask.sum())
        bar = "#" * int(50 * c / n_cb) if n_cb > 0 else ""
        print(f"  {label:20s}: {c:9,d} ({100*c/n_cb:5.1f}%)  {bar}")

    # ---- min-pairs-per-bin filter implications ----------------------------
    print("\n" + "=" * 72)
    print("How many (cell, bin) records pass --min-pairs-per-bin?")
    print("=" * 72)
    print(f"  Total (cell, bin) with n_linked >= 1: {n_cb:,}")
    for threshold in (1, 2, 3, 5, 10, 20):
        c = int((combined >= threshold).sum())
        print(f"  >= {threshold:3d}: {c:9,d} ({100*c/n_cb:5.1f}%)")

    # ---- per-cell × per-bin density estimate ------------------------------
    n_cells = per_cell_df.shape[0]
    n_unique_bins_seen = combined.index.nunique() if isinstance(combined, pd.Series) else 0
    median_bins_per_cell = float(per_cell_df["n_bins_with_data"].median())
    print("\n" + "=" * 72)
    print("Density estimate")
    print("=" * 72)
    print(f"  Median bins per cell with n_linked >= 1: {median_bins_per_cell:.0f}")
    print(f"  Median pairs per bin (per cell):         "
          f"{per_cell_df['median_pairs_per_bin'].median():.2f}")
    print(f"  Mean pairs per bin (per cell):           "
          f"{per_cell_df['mean_pairs_per_bin'].median():.2f}")

    if args.output_prefix:
        per_cell_df.to_csv(
            f"{args.output_prefix}.per_cell.tsv", sep="\t", index=False
        )
        combined.to_frame("n_linked").to_csv(
            f"{args.output_prefix}.per_bin.tsv", sep="\t", index=False
        )
        print(f"\nWrote {args.output_prefix}.per_cell.tsv and "
              f"{args.output_prefix}.per_bin.tsv")


if __name__ == "__main__":
    main()
