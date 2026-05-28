#!/usr/bin/env python3
"""Diagnostic: linked-pair density, posterior sharpness, low-coverage cells.

Reads per-cell parquets produced by ``nome_hic_concordance`` and reports
three sets of statistics:

  A) Linked-pair density per (cell, bin)
     - both ends on the same chromosome AND in the same bin of size --bin-size
     - distance >= --min-distance
     - both ends informative: n_GCH on each end >= --min-gch-sites

  B) Per-end posterior sharpness  (requires Step-4 ``*.posteriors.parquet``)
     - max(p, 1 - p) across every informative end
     - tells you whether U-3 soft counts behave like hard binarization
       (sharp posteriors → equivalent) or genuinely matter (diffuse posteriors)

  C) Low-coverage cell QC
     - per-cell total linked pairs distribution
     - knee-detection-based suggested cutoffs for filtering bad cells
     - shows how many cells survive each candidate threshold

Works on either Step-2 (``*.read_pairs.parquet``) or Step-4
(``*.posteriors.parquet``) directories.  Section B is skipped if the input
is Step-2 (no posterior columns).

Example:
    # On Step-4 posteriors (recommended: produces all three sections)
    python scripts/check_linked_pair_density.py \\
        --input-dir output/betabinom_250kb/GCH/04_posteriors \\
        --bin-size 250000 \\
        --min-distance 1000 \\
        --min-gch-sites 3 \\
        --n-threads 8

    # On Step-2 read pairs (pair density + low-coverage QC only)
    python scripts/check_linked_pair_density.py \\
        --input-dir output/betabinom_250kb/GCH/02_read_pair_counts \\
        --bin-size 250000 --min-distance 1000 --min-gch-sites 3 \\
        --n-threads 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _detect_schema(path: Path) -> set[str]:
    """Peek at parquet schema without reading data."""
    import pyarrow.parquet as pq
    return set(pq.read_schema(str(path)).names)


def process_cell(
    path: Path,
    bin_size: int,
    min_distance: int,
    min_gch_sites: int,
    has_posteriors: bool,
) -> dict:
    """Compute pair density (and posterior sharpness if available) for one cell."""
    # Strip both possible suffixes so the cell name is the same regardless of
    # which directory we were pointed at.
    cell_name = path.stem
    for suffix in (".read_pairs", ".posteriors"):
        if cell_name.endswith(suffix):
            cell_name = cell_name[: -len(suffix)]
            break

    cols = ["chrom", "pos1", "pos2", "n1", "n2", "distance"]
    if has_posteriors:
        cols += ["posterior1", "posterior2"]
    df = pd.read_parquet(path, columns=cols)

    empty_result = {
        "cell_name": cell_name,
        "n_linked_total": 0,
        "n_informative_ends": 0,
        "pairs_per_bin": pd.Series([], dtype=np.int64),
        "sharpness": np.array([], dtype=np.float32),
    }
    if len(df) == 0:
        return empty_result

    chrom = df["chrom"].astype(str).values
    pos1 = df["pos1"].astype(np.int64).values
    pos2 = df["pos2"].astype(np.int64).values
    n1 = df["n1"].astype(np.int32).values
    n2 = df["n2"].astype(np.int32).values
    distance = df["distance"].astype(np.int64).values

    bin1 = (pos1 // bin_size) * bin_size
    bin2 = (pos2 // bin_size) * bin_size

    same_bin = bin1 == bin2
    inf1 = n1 >= min_gch_sites
    inf2 = n2 >= min_gch_sites
    informative_pair = inf1 & inf2
    long_range = distance >= min_distance

    linked = same_bin & informative_pair & long_range
    n_linked_total = int(linked.sum())

    # Per-cell posterior sharpness across all informative ends.
    sharpness_parts: list[np.ndarray] = []
    if has_posteriors:
        p1 = df["posterior1"].to_numpy(dtype=np.float32, na_value=np.nan)
        p2 = df["posterior2"].to_numpy(dtype=np.float32, na_value=np.nan)
        ok1 = inf1 & np.isfinite(p1)
        ok2 = inf2 & np.isfinite(p2)
        if ok1.any():
            sharpness_parts.append(np.maximum(p1[ok1], 1.0 - p1[ok1]))
        if ok2.any():
            sharpness_parts.append(np.maximum(p2[ok2], 1.0 - p2[ok2]))
    sharpness = (
        np.concatenate(sharpness_parts) if sharpness_parts
        else np.array([], dtype=np.float32)
    )
    n_informative_ends = int(inf1.sum() + inf2.sum())

    if n_linked_total == 0:
        return {
            "cell_name": cell_name,
            "n_linked_total": 0,
            "n_informative_ends": n_informative_ends,
            "pairs_per_bin": pd.Series([], dtype=np.int64),
            "sharpness": sharpness,
        }

    chrom_linked = chrom[linked]
    bin_linked = bin1[linked]
    pairs_per_bin = (
        pd.DataFrame({"chrom": chrom_linked, "bin": bin_linked})
        .groupby(["chrom", "bin"], sort=False)
        .size()
    )
    return {
        "cell_name": cell_name,
        "n_linked_total": n_linked_total,
        "n_informative_ends": n_informative_ends,
        "pairs_per_bin": pairs_per_bin,
        "sharpness": sharpness,
    }


def _knee_index(sorted_ascending: np.ndarray) -> int:
    """Find the 'knee' of an ascending curve as the point of maximum distance
    below the chord from start to end (a simple kneedle-style heuristic).

    Returns 0 if no clear knee can be identified.
    """
    n = len(sorted_ascending)
    if n < 4:
        return 0
    # Use log to make the knee meaningful on highly-skewed coverage distributions.
    y = np.log10(np.asarray(sorted_ascending, dtype=np.float64) + 1.0)
    x = np.arange(n, dtype=np.float64)
    y_norm = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    x_norm = x / (n - 1)
    # Chord from (0, 0) to (1, 1).  Concave curves (slow start, fast end) have
    # the knee at the point furthest BELOW the chord.
    below_chord = x_norm - y_norm
    return int(below_chord.argmax())


def _suggest_cutoffs(per_cell_df: pd.DataFrame, metric: str = "n_linked_total") -> dict:
    """Return a dict of candidate low-coverage-cell cutoffs and their effects."""
    vals = per_cell_df[metric].to_numpy()
    sorted_asc = np.sort(vals)
    n = len(sorted_asc)

    median = float(np.median(sorted_asc))
    mad = float(np.median(np.abs(sorted_asc - median)))

    knee_idx = _knee_index(sorted_asc)
    knee_value = float(sorted_asc[knee_idx]) if n > 0 else 0.0

    candidates = {
        "5th percentile":          float(np.quantile(sorted_asc, 0.05)),
        "10th percentile":         float(np.quantile(sorted_asc, 0.10)),
        "25th percentile":         float(np.quantile(sorted_asc, 0.25)),
        "median - 3*MAD":          max(median - 3 * mad, 1.0),
        "median / 10":             median / 10.0,
        "median / 5":              median / 5.0,
        "knee detection (log)":    knee_value,
        "100":                     100.0,
        "500":                     500.0,
    }
    return {"sorted": sorted_asc, "candidates": candidates, "knee_idx": knee_idx}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing per-cell parquets — either Step-2 "
             "(*.read_pairs.parquet, output/.../02_read_pair_counts/) or "
             "Step-4 (*.posteriors.parquet, output/.../04_posteriors/). "
             "Point at Step-4 to include the posterior-sharpness analysis.",
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
    # Accept either Step-2 (*.read_pairs.parquet) or Step-4 (*.posteriors.parquet).
    files = sorted(in_dir.glob("*.read_pairs.parquet"))
    if not files:
        files = sorted(in_dir.glob("*.posteriors.parquet"))
    if not files:
        print(
            f"ERROR: no *.read_pairs.parquet or *.posteriors.parquet files "
            f"found in {in_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.sample > 0 and args.sample < len(files):
        rng = np.random.default_rng(args.random_seed)
        idx = rng.choice(len(files), size=args.sample, replace=False)
        files = [files[i] for i in sorted(idx)]
        print(f"Subsampled {args.sample} of {len(files)} parquets (seed={args.random_seed})")

    # Peek at the first file's schema to see whether posteriors are present.
    available = _detect_schema(files[0])
    has_posteriors = {"posterior1", "posterior2"}.issubset(available)
    print(
        f"Processing {len(files)} cell(s) with "
        f"bin_size={args.bin_size:,}, "
        f"min_distance={args.min_distance:,}, "
        f"min_gch_sites={args.min_gch_sites}"
    )
    print(f"  posterior columns present: {has_posteriors}")

    if args.n_threads > 1 and len(files) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(
                p, args.bin_size, args.min_distance, args.min_gch_sites, has_posteriors,
            )
            for p in files
        )
    else:
        results = [
            process_cell(p, args.bin_size, args.min_distance, args.min_gch_sites, has_posteriors)
            for p in files
        ]

    # ---- Per-cell summary --------------------------------------------------
    per_cell_rows = []
    all_ppb_list: list[pd.Series] = []
    all_sharpness_parts: list[np.ndarray] = []
    for r in results:
        ppb = r["pairs_per_bin"]
        per_cell_rows.append({
            "cell": r["cell_name"],
            "n_linked_total":   r["n_linked_total"],
            "n_informative_ends": r["n_informative_ends"],
            "n_bins_with_data": int(len(ppb)),
            "median_pairs_per_bin": float(ppb.median()) if len(ppb) > 0 else 0.0,
            "mean_pairs_per_bin":   float(ppb.mean())   if len(ppb) > 0 else 0.0,
            "max_pairs_per_bin":    int(ppb.max())      if len(ppb) > 0 else 0,
            "median_sharpness":     float(np.median(r["sharpness"])) if len(r["sharpness"]) > 0 else np.nan,
        })
        if len(ppb) > 0:
            all_ppb_list.append(ppb)
        if len(r["sharpness"]) > 0:
            all_sharpness_parts.append(r["sharpness"])
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
    median_bins_per_cell = float(per_cell_df["n_bins_with_data"].median())
    print("\n" + "=" * 72)
    print("Density estimate")
    print("=" * 72)
    print(f"  Median bins per cell with n_linked >= 1: {median_bins_per_cell:.0f}")
    print(f"  Median pairs per bin (per cell):         "
          f"{per_cell_df['median_pairs_per_bin'].median():.2f}")
    print(f"  Mean pairs per bin (per cell):           "
          f"{per_cell_df['mean_pairs_per_bin'].median():.2f}")

    # ---- (B) posterior sharpness -----------------------------------------
    if has_posteriors and all_sharpness_parts:
        all_sharp = np.concatenate(all_sharpness_parts)
        n_ends = len(all_sharp)
        print("\n" + "=" * 72)
        print(f"Per-end posterior sharpness — max(p, 1-p) across "
              f"{n_ends:,} informative ends")
        print("=" * 72)
        print("  Higher = more confident open/closed call.")
        print("  Sharp posteriors (≥0.9) ⇒ U-3 soft counts ≈ hard binarization.")
        print("  Diffuse posteriors (<0.7) ⇒ U-3 genuinely propagates uncertainty.\n")
        for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
            print(f"  {q*100:5.0f}th percentile: {np.quantile(all_sharp, q):6.3f}")
        print(f"  mean:               {all_sharp.mean():6.3f}")
        print(f"  fraction ≥ 0.95:    {(all_sharp >= 0.95).mean()*100:5.1f}%")
        print(f"  fraction ≥ 0.90:    {(all_sharp >= 0.90).mean()*100:5.1f}%")
        print(f"  fraction ≥ 0.80:    {(all_sharp >= 0.80).mean()*100:5.1f}%")
        print(f"  fraction ≥ 0.70:    {(all_sharp >= 0.70).mean()*100:5.1f}%")
        print(f"  fraction < 0.60:    {(all_sharp <  0.60).mean()*100:5.1f}%")

        # Sharpness histogram (the diffuse end of the spectrum matters most).
        print("\n  Histogram of max(p, 1-p):")
        bins_edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                      0.85, 0.90, 0.95, 1.0001]
        for lo, hi in zip(bins_edges[:-1], bins_edges[1:]):
            mask = (all_sharp >= lo) & (all_sharp < hi)
            c = int(mask.sum())
            bar = "#" * int(50 * c / n_ends) if n_ends > 0 else ""
            label = f"[{lo:.2f}, {hi:.2f})"
            print(f"    {label:14s}: {c:9,d} ({100*c/n_ends:5.1f}%)  {bar}")

        # Per-cell median sharpness distribution.
        med_sharp = per_cell_df["median_sharpness"].dropna()
        if len(med_sharp) > 0:
            print(f"\n  Per-cell median sharpness — across {len(med_sharp)} cells:")
            for q in (0.05, 0.25, 0.50, 0.75, 0.95):
                print(f"    {q*100:5.0f}th percentile: {med_sharp.quantile(q):.3f}")
    elif not has_posteriors:
        print("\n[posterior sharpness section skipped: posteriors not in input "
              "— point --input-dir at output/.../04_posteriors/ to compute this]")

    # ---- (C) low-coverage cell QC ----------------------------------------
    print("\n" + "=" * 72)
    print("Low-coverage cell QC")
    print("=" * 72)
    suggest = _suggest_cutoffs(per_cell_df, metric="n_linked_total")
    sorted_asc = suggest["sorted"]
    n_cells = len(sorted_asc)
    knee_idx = suggest["knee_idx"]

    print(f"\nBottom 20 cells by n_linked_total:")
    bottom = per_cell_df.nsmallest(20, "n_linked_total")[
        ["cell", "n_linked_total", "n_informative_ends", "n_bins_with_data"]
    ]
    print(bottom.to_string(index=False))

    print(f"\nCandidate cutoffs on n_linked_total:")
    print(f"  {'method':24s} {'cutoff':>10s} {'n_dropped':>12s} {'n_kept':>10s}")
    for name, value in sorted(suggest["candidates"].items(), key=lambda kv: kv[1]):
        n_dropped = int((sorted_asc < value).sum())
        n_kept = n_cells - n_dropped
        marker = "  ← knee" if name == "knee detection (log)" else ""
        print(f"  {name:24s} {value:10.0f} {n_dropped:12d} {n_kept:10d}{marker}")

    print(f"\n  (knee at sorted-ascending index {knee_idx} of {n_cells} cells = "
          f"value {sorted_asc[knee_idx]:.0f})")
    print("\nRecommended starting point:")
    knee_value = float(sorted_asc[knee_idx])
    knee_kept = n_cells - int((sorted_asc < knee_value).sum())
    if knee_value >= 100:
        print(f"  --min-linked-pairs-per-cell {int(knee_value)}  "
              f"(keeps {knee_kept}/{n_cells} cells; from knee detection)")
    else:
        # Fall back to a sensible floor if the knee is too low to be meaningful.
        fallback = max(100, int(np.quantile(sorted_asc, 0.05)))
        fallback_kept = n_cells - int((sorted_asc < fallback).sum())
        print(f"  --min-linked-pairs-per-cell {fallback}  "
              f"(keeps {fallback_kept}/{n_cells} cells; 5th percentile / 100 floor — "
              f"knee was too low to be a clear elbow)")

    if args.output_prefix:
        per_cell_df.to_csv(
            f"{args.output_prefix}.per_cell.tsv", sep="\t", index=False
        )
        combined.to_frame("n_linked").to_csv(
            f"{args.output_prefix}.per_bin.tsv", sep="\t", index=False
        )
        if has_posteriors and all_sharpness_parts:
            np.savetxt(
                f"{args.output_prefix}.posterior_sharpness.tsv",
                np.concatenate(all_sharpness_parts),
                fmt="%.4f", header="max_p_or_1minusp", comments="",
            )
        print(f"\nWrote {args.output_prefix}.per_cell.tsv, "
              f"{args.output_prefix}.per_bin.tsv"
              + (f", {args.output_prefix}.posterior_sharpness.tsv"
                 if has_posteriors and all_sharpness_parts else ""))


if __name__ == "__main__":
    main()
