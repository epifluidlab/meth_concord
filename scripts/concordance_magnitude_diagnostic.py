#!/usr/bin/env python3
"""Diagnostic: is long-range concordance *real in magnitude* (vs random),
independent of whether it carries cell-type information?

The signed/clustering pipeline answers "does per-(cell,bin) concordance VARY
across cells in a cell-type way?" — and is blind to the average MAGNITUDE of
concordance, because its embedding stack (per-cell rank -> per-bin QN ->
recenter) strips location and scale.  A signal can therefore be **real and
strong but cell-type-invariant** (high mean, low cross-cell variance) and
still show weak kNN/silhouette.  This script tests that magnitude directly.

For every informative same-bin pair it computes the **soft agreement**
(co-state concordance):

    a = P(open_i), b = P(open_j)
    agreement = a*b + (1-a)*(1-b)        # = co-open + co-closed = 1 - mixed
    mixed     = a + b - 2ab              # discordance

and aggregates it for three pools, **stratified by genomic distance**:

  - linked     : long-range contacts (same bin, distance >= --min-distance)
  - short-range: local pairs          (same bin, distance <  --min-distance)
  - random     : distance-matched NON-contact end-pairs (all triu pairs of
                 the bin's informative ends) — "coincidence at that distance"

It then reports, per cell and pooled:

  (a) is long-range agreement > the distance-matched random null?
      (per-cell distance-matched excess A_long - A_rand; paired test vs 0)
  (b) is long-range agreement comparable to the short-range level?
      (A_long vs A_short, both absolute and as excess-over-random)

A positive, consistent excess with near-zero clustering is exactly the
signature of a **real but constitutive** long-range concordance (the likely
GCH story), as opposed to no long-range concordance at all.

Example:
    python scripts/concordance_magnitude_diagnostic.py \\
        --input-dir output/betabinom_250kb/GCH/04_posteriors \\
        --bin-size 250000 --min-distance 1000 --min-gch-sites 3 \\
        --n-threads 8 --output-prefix gch_magnitude

Run it once for GCH and once for HCG and compare the excess curves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _soft_costate(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (agreement, mixed) for per-end open-probabilities a, b."""
    mixed = a + b - 2.0 * a * b
    return 1.0 - mixed, mixed


def process_cell(
    path: Path,
    bin_size: int,
    min_distance: int,
    min_gch_sites: int,
    edges: np.ndarray,
    max_ends: int,
    seed: int,
) -> dict | None:
    """Per-cell distance-stratified soft agreement for the three pools.

    Returns per-stratum (sumA, sumM, count) arrays for linked / short / random
    plus per-cell scalar summaries, or None if the cell has no informative
    same-bin pairs.
    """
    cell_name = path.stem
    if cell_name.endswith(".posteriors"):
        cell_name = cell_name[: -len(".posteriors")]

    cols = ["chrom", "pos1", "pos2", "n1", "n2", "distance",
            "posterior1", "posterior2"]
    df = pd.read_parquet(path, columns=cols)
    if len(df) == 0:
        return None

    chrom = df["chrom"].astype(str).values
    pos1 = df["pos1"].to_numpy(dtype=np.int64)
    pos2 = df["pos2"].to_numpy(dtype=np.int64)
    n1 = df["n1"].to_numpy(dtype=np.int32)
    n2 = df["n2"].to_numpy(dtype=np.int32)
    p1 = df["posterior1"].to_numpy(dtype=np.float64, na_value=np.nan)
    p2 = df["posterior2"].to_numpy(dtype=np.float64, na_value=np.nan)
    dist = df["distance"].to_numpy(dtype=np.int64)
    bsz = int(bin_size)

    inf1 = (n1 >= min_gch_sites) & np.isfinite(p1)
    inf2 = (n2 >= min_gch_sites) & np.isfinite(p2)
    same_bin = (pos1 // bsz) == (pos2 // bsz)
    pair_ok = inf1 & inf2 & same_bin

    n_strata = len(edges) - 1
    z = lambda: np.zeros(n_strata, dtype=np.float64)
    out = {p: {"A": z(), "M": z(), "n": z()} for p in ("linked", "short", "random")}

    # ---- observed pools: linked (d>=min_dist) and short-range (d<min_dist) --
    if pair_ok.any():
        a = p1[pair_ok]; b = p2[pair_ok]
        agree, mixed = _soft_costate(a, b)
        d = dist[pair_ok]
        s = np.searchsorted(edges, d, side="right") - 1
        s = np.clip(s, 0, n_strata - 1)
        is_long = d >= int(min_distance)
        for pool, sel in (("linked", is_long), ("short", ~is_long)):
            ss = s[sel]
            out[pool]["A"] += np.bincount(ss, weights=agree[sel], minlength=n_strata)
            out[pool]["M"] += np.bincount(ss, weights=mixed[sel], minlength=n_strata)
            out[pool]["n"] += np.bincount(ss, minlength=n_strata)

    # ---- random null: triu end-pairs of each bin's informative ends --------
    e_chrom = np.concatenate([chrom[inf1], chrom[inf2]])
    e_pos = np.concatenate([pos1[inf1], pos2[inf2]])
    e_p = np.concatenate([p1[inf1], p2[inf2]])
    e_bin = e_pos // bsz
    ends = pd.DataFrame({"chrom": e_chrom, "bin": e_bin, "pos": e_pos, "p": e_p})
    rng = np.random.default_rng(seed)
    for (_, _), g in ends.groupby(["chrom", "bin"], sort=False):
        m = len(g)
        if m < 2:
            continue
        epos = g["pos"].to_numpy(dtype=np.int64)
        ep = g["p"].to_numpy(dtype=np.float64)
        if m > max_ends:
            idx = rng.choice(m, size=max_ends, replace=False)
            epos, ep = epos[idx], ep[idx]
            m = max_ends
        iu, ju = np.triu_indices(m, k=1)
        d = np.abs(epos[iu] - epos[ju])
        agree, mixed = _soft_costate(ep[iu], ep[ju])
        s = np.searchsorted(edges, d, side="right") - 1
        s = np.clip(s, 0, n_strata - 1)
        out["random"]["A"] += np.bincount(s, weights=agree, minlength=n_strata)
        out["random"]["M"] += np.bincount(s, weights=mixed, minlength=n_strata)
        out["random"]["n"] += np.bincount(s, minlength=n_strata)

    # ---- per-cell scalar summaries -----------------------------------------
    def _mean(pool):
        n = out[pool]["n"].sum()
        return (out[pool]["A"].sum() / n) if n > 0 else np.nan

    # Distance-matched random for THIS cell: weight per-stratum random
    # agreement by the cell's own linked counts per stratum.
    nl = out["linked"]["n"]
    nr = out["random"]["n"]
    rand_rate = np.divide(out["random"]["A"], nr, out=np.full(n_strata, np.nan),
                          where=nr > 0)
    valid = (nl > 0) & np.isfinite(rand_rate)
    a_rand_matched = (np.sum(nl[valid] * rand_rate[valid]) / np.sum(nl[valid])
                      if valid.any() and np.sum(nl[valid]) > 0 else np.nan)

    return {
        "cell": cell_name,
        "strata": out,
        "A_long": _mean("linked"),
        "A_short": _mean("short"),
        "A_rand_global": _mean("random"),
        "A_rand_matched": a_rand_matched,
        "n_long": float(out["linked"]["n"].sum()),
        "n_short": float(out["short"]["n"].sum()),
    }


def _fmt(x: float) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input-dir", required=True,
                    help="Directory of *.posteriors.parquet (Step-4 output).")
    ap.add_argument("--bin-size", type=int, default=250_000)
    ap.add_argument("--min-distance", type=int, default=1_000)
    ap.add_argument("--min-gch-sites", type=int, default=3)
    ap.add_argument("--n-long-bins", type=int, default=5,
                    help="Number of log-spaced long-range distance strata "
                         "between --min-distance and --bin-size (default 5).")
    ap.add_argument("--max-ends", type=int, default=200,
                    help="Cap on a bin's informative ends used to form the "
                         "random end-pair null (default 200).")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--output-prefix", default=None,
                    help="If set, write <prefix>.curves.csv and (if matplotlib "
                         "is available) <prefix>.png.")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    files = sorted(in_dir.glob("*.posteriors.parquet"))
    if not files:
        sys.exit(f"No *.posteriors.parquet found under {in_dir}")
    print(f"[magnitude] {len(files)} cells | bin={args.bin_size:,} "
          f"min_dist={args.min_distance:,} min_gch={args.min_gch_sites}")

    # Distance strata: stratum 0 = [0, min_distance) (short-range);
    # strata 1..n = log-spaced long-range up to bin_size.
    edges = np.concatenate([
        [0.0],
        np.logspace(np.log10(args.min_distance),
                    np.log10(args.bin_size), args.n_long_bins + 1),
    ])
    n_strata = len(edges) - 1

    if args.n_threads > 1 and len(files) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(
                f, args.bin_size, args.min_distance, args.min_gch_sites,
                edges, args.max_ends, args.random_seed + i,
            ) for i, f in enumerate(files)
        )
    else:
        results = [
            process_cell(f, args.bin_size, args.min_distance,
                         args.min_gch_sites, edges, args.max_ends,
                         args.random_seed + i)
            for i, f in enumerate(files)
        ]
    results = [r for r in results if r is not None]
    if not results:
        sys.exit("No cells with informative same-bin pairs.")
    print(f"[magnitude] {len(results)} cells with usable pairs")

    # ---- per-cell distance-matched excess test -----------------------------
    A_long = np.array([r["A_long"] for r in results], dtype=np.float64)
    A_short = np.array([r["A_short"] for r in results], dtype=np.float64)
    A_rm = np.array([r["A_rand_matched"] for r in results], dtype=np.float64)
    A_rg = np.array([r["A_rand_global"] for r in results], dtype=np.float64)

    ok = np.isfinite(A_long) & np.isfinite(A_rm)
    excess = A_long[ok] - A_rm[ok]                      # contact vs random (matched)
    log2_oe = np.log2(np.clip(A_long[ok], 1e-9, None) /
                      np.clip(A_rm[ok], 1e-9, None))
    oks = np.isfinite(A_long) & np.isfinite(A_short)
    long_minus_short = A_long[oks] - A_short[oks]

    def _ttest(x):
        x = x[np.isfinite(x)]
        if len(x) < 2 or np.std(x) == 0:
            return float(np.mean(x)) if len(x) else np.nan, np.nan
        t = np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x)))
        return float(np.mean(x)), float(t)

    mean_ex, t_ex = _ttest(excess)
    frac_pos = float(np.mean(excess > 0)) if len(excess) else np.nan

    print("\n================  MAGNITUDE (per-cell, distance-matched)  ============")
    print(f"  cells tested                 : {ok.sum()}")
    print(f"  mean A_long  (contacts)      : {_fmt(np.nanmean(A_long))}")
    print(f"  mean A_rand  (matched null)  : {_fmt(np.nanmean(A_rm))}")
    print(f"  mean A_rand  (global null)   : {_fmt(np.nanmean(A_rg))}")
    print(f"  mean A_short (local pairs)   : {_fmt(np.nanmean(A_short))}")
    print(f"  --- claim (a): contacts > random coincidence at matched distance")
    print(f"  mean excess  A_long - A_rand : {_fmt(mean_ex)}   (t~{_fmt(t_ex)})")
    print(f"  fraction of cells excess > 0 : {_fmt(frac_pos)}")
    print(f"  mean log2(A_long / A_rand)   : {_fmt(np.nanmean(log2_oe))}")
    print(f"  --- claim (b): contacts comparable to local short-range level")
    print(f"  mean A_long - A_short        : {_fmt(np.nanmean(long_minus_short))}")
    ratio = (np.nanmean(A_long) / np.nanmean(A_short)
             if np.nanmean(A_short) else np.nan)
    print(f"  A_long / A_short             : {_fmt(ratio)}")
    print("  (interpretation: excess>0 with t large => real long-range")
    print("   concordance above random; A_long~A_short => contact restores")
    print("   near-local concordance.  Both are MAGNITUDE statements and are")
    print("   compatible with weak cell-type clustering.)")

    # ---- pooled distance-decay curves --------------------------------------
    def pool_rate(key):
        A = np.sum([r["strata"][key]["A"] for r in results], axis=0)
        n = np.sum([r["strata"][key]["n"] for r in results], axis=0)
        return np.divide(A, n, out=np.full(n_strata, np.nan), where=n > 0), n

    A_l, n_l = pool_rate("linked")
    A_s, n_s = pool_rate("short")
    A_r, n_r = pool_rate("random")
    lo = edges[:-1]; hi = edges[1:]
    rows = []
    print("\n================  DISTANCE-DECAY of soft agreement  ==================")
    print(f"  {'stratum (bp)':>22} {'n_contact':>11} {'A_contact':>9} "
          f"{'A_random':>9} {'excess':>8} {'log2(O/E)':>9}")
    for k in range(n_strata):
        label = (f"<{int(args.min_distance):,} (short)" if k == 0
                 else f"{int(lo[k]):,}-{int(hi[k]):,}")
        a_c = A_l[k] if k > 0 else A_s[k]      # stratum 0 shows short-range obs
        a_rd = A_r[k]
        ex = (a_c - a_rd) if (np.isfinite(a_c) and np.isfinite(a_rd)) else np.nan
        l2 = (np.log2(max(a_c, 1e-9) / max(a_rd, 1e-9))
              if np.isfinite(a_c) and np.isfinite(a_rd) else np.nan)
        nc = n_s[k] if k == 0 else n_l[k]
        print(f"  {label:>22} {int(nc):>11,} {_fmt(a_c):>9} "
              f"{_fmt(a_rd):>9} {_fmt(ex):>8} {_fmt(l2):>9}")
        rows.append({"stratum_lo": lo[k], "stratum_hi": hi[k],
                     "is_short": k == 0, "n_contact": nc,
                     "A_observed": a_c, "A_random": a_rd,
                     "excess": ex, "log2_OE": l2})

    if args.output_prefix:
        curves = pd.DataFrame(rows)
        csv = f"{args.output_prefix}.curves.csv"
        curves.to_csv(csv, index=False)
        print(f"\n[magnitude] wrote {csv}")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            mid = np.sqrt(np.clip(lo, 1, None) * hi)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(mid[1:], A_l[1:], "o-", label="long-range contacts (O)")
            ax.plot(mid[1:], A_r[1:], "s--", label="distance-matched random (E)")
            ax.axhline(A_s[0], color="green", ls=":",
                       label="short-range local pairs")
            ax.set_xscale("log")
            ax.set_xlabel("genomic distance (bp)")
            ax.set_ylabel("soft agreement  (co-open + co-closed)")
            ax.set_title("Long-range concordance vs random vs local")
            ax.legend(fontsize=8)
            fig.tight_layout()
            png = f"{args.output_prefix}.png"
            fig.savefig(png, dpi=150)
            print(f"[magnitude] wrote {png}")
        except Exception as e:  # pragma: no cover - plotting is optional
            print(f"[magnitude] plot skipped ({e})")


if __name__ == "__main__":
    main()
