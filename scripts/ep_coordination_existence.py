#!/usr/bin/env python3
"""Step 1: clean existence test for cell-type-specific long-range E–P
coordination — WITHOUT cross-cell imputation.

The RWR-based pipeline can manufacture apparent cell-type signal because it
smooths concordance on a cell graph built from the marginal accessibility /
methylation (leakage).  This script avoids that entirely: it densifies the
sparse per-cell signal by **pooling pairs within a cell** (clean) rather than
**borrowing across cells** (leaky), and it holds the distance scale fixed at
the **E–P regulatory band** (default 20–200 kb) so coarse pooling does not
smear the signal into the bulk contact-decay.

For each cell and each genomic **region** (a coarse partition — per chromosome
or per N-Mb window) it computes a distance-matched **log2(O/E)** concordance:

  O = soft co-state of OBSERVED contacts (read pairs) with separation in the
      band, pooled over the region
  E = soft co-state of distance-matched NON-contact end-pairs in the band
      (the region's own read ends, window-paired at the same separations)

  Δ_region = [log2(O_open/E_open)] − [log2(O_closed/E_closed)]

The O/E removes the region's marginal openness + the distance at the pair
level, so Δ is **marginal-controlled by construction**.  Pooling is over
PAIRS, so there is no cross-cell imputation and no leakage.

It then asks, at several spatial granularities, whether the per-(cell, region)
Δ matrix **classifies cell type above chance** (5-fold CV OvR-macro AUC,
chance = 0.5), alongside the per-region mean-openness (marginal) AUC as a
reference:

  - Δ AUC > 0.5 at some granularity  → real, marginal-independent E–P
    coordination signal exists (then Step 2: a held-out concordance-graph
    imputation can chase it at finer resolution).
  - Δ AUC ≈ 0.5 at every granularity → no detectable signal even when richly
    pooled; the RWR result was an imputation artifact.

Example:
    python scripts/ep_coordination_existence.py \\
        --input-dir output/.../context_GCH/04_posteriors \\
        --samples samples.with_batches.tsv \\
        --min-distance 20000 --max-distance 200000 \\
        --granularities chrom,25,10,5 --n-threads 8 \\
        --output-prefix gch_ep_existence

Run once per context (GCH, HCG).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_labels(path: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "cell_name" not in df.columns:
        df = df.rename(columns={"cell": "cell_name"} if "cell" in df.columns
                       else {df.columns[0]: "cell_name"})
    if "group" not in df.columns:
        raise SystemExit(f"{path} has no 'group' column.")
    return df.set_index("cell_name")["group"]


def _window_pairs(pos: np.ndarray, p: np.ndarray, min_d: int, max_d: int,
                  cap: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All same-chrom end-pairs with separation in [min_d, max_d].

    Returns (left_pos, e_open, e_closed) per null pair (left end = the smaller
    position, used for region assignment).  Subsamples ends to ``cap`` first.
    """
    n = pos.shape[0]
    if n < 2:
        return np.empty(0, np.int64), np.empty(0), np.empty(0)
    if n > cap:
        idx = np.sort(rng.choice(n, size=cap, replace=False))
        pos, p = pos[idx], p[idx]
        n = cap
    order = np.argsort(pos, kind="mergesort")
    pos, p = pos[order], p[order]
    lo = np.searchsorted(pos, pos + int(min_d), side="left")
    hi = np.searchsorted(pos, pos + int(max_d), side="right")
    cnt = hi - lo
    total = int(cnt.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0), np.empty(0)
    left = np.repeat(np.arange(n), cnt)
    right = np.concatenate([np.arange(lo[i], hi[i]) for i in range(n)
                            if cnt[i] > 0])
    pa, pb = p[left], p[right]
    return pos[left], pa * pb, (1.0 - pa) * (1.0 - pb)


def _region_key(chrom: np.ndarray, pos: np.ndarray, gran) -> np.ndarray:
    if gran == "chrom":
        return chrom.astype(object)
    size = int(gran) * 1_000_000
    return np.char.add(np.char.add(chrom.astype(str), ":"),
                       (pos // size).astype(np.int64).astype(str))


def process_cell(path: Path, granularities: list, min_d: int, max_d: int,
                 min_gch: int, cap: int, seed: int):
    """Per-(region) Δ_signed (log2 O/E) and mean openness for one cell, at
    each granularity.  Returns (cell_name, {gran: {region: (delta, marg, n_o)}})."""
    cell = path.stem
    if cell.endswith(".posteriors"):
        cell = cell[: -len(".posteriors")]
    df = pd.read_parquet(path, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                         "distance", "posterior1", "posterior2"])
    if len(df) == 0:
        return cell, {}
    chrom = df["chrom"].astype(str).values
    pos1 = df["pos1"].to_numpy(np.int64); pos2 = df["pos2"].to_numpy(np.int64)
    n1 = df["n1"].to_numpy(np.int32); n2 = df["n2"].to_numpy(np.int32)
    p1 = df["posterior1"].to_numpy(np.float64, na_value=np.nan)
    p2 = df["posterior2"].to_numpy(np.float64, na_value=np.nan)
    inf1 = (n1 >= min_gch) & np.isfinite(p1)
    inf2 = (n2 >= min_gch) & np.isfinite(p2)

    # --- OBSERVED contacts in the band -----------------------------------
    lo = np.minimum(pos1, pos2); hi = np.maximum(pos1, pos2)
    sep = hi - lo
    cm = inf1 & inf2 & (sep >= min_d) & (sep <= max_d)
    O = pd.DataFrame({"chrom": chrom[cm], "pos": lo[cm],
                      "o_open": (p1 * p2)[cm],
                      "o_closed": ((1 - p1) * (1 - p2))[cm]})

    # --- end pool (for the null + the marginal) --------------------------
    e_chrom = np.concatenate([chrom[inf1], chrom[inf2]])
    e_pos = np.concatenate([pos1[inf1], pos2[inf2]])
    e_p = np.concatenate([p1[inf1], p2[inf2]])
    ends = pd.DataFrame({"chrom": e_chrom, "pos": e_pos, "p": e_p})

    # --- NULL end-pairs in the band, per chromosome ----------------------
    rng = np.random.default_rng(seed)
    e_rows = []
    for c, g in ends.groupby("chrom", sort=False):
        lp, eo, ec = _window_pairs(g["pos"].to_numpy(np.int64),
                                   g["p"].to_numpy(np.float64),
                                   min_d, max_d, cap, rng)
        if lp.size:
            e_rows.append(pd.DataFrame({"chrom": c, "pos": lp,
                                        "e_open": eo, "e_closed": ec}))
    E = pd.concat(e_rows, ignore_index=True) if e_rows else pd.DataFrame(
        columns=["chrom", "pos", "e_open", "e_closed"])

    eps = 1e-6
    out: dict = {}
    for gran in granularities:
        o_reg = _region_key(O["chrom"].to_numpy(), O["pos"].to_numpy(), gran) \
            if len(O) else np.array([], object)
        e_reg = _region_key(E["chrom"].to_numpy(), E["pos"].to_numpy(), gran) \
            if len(E) else np.array([], object)
        m_reg = _region_key(ends["chrom"].to_numpy(), ends["pos"].to_numpy(), gran)

        og = pd.DataFrame({"r": o_reg, "o_open": O["o_open"].to_numpy(),
                           "o_closed": O["o_closed"].to_numpy()}).groupby("r").agg(
            o_open=("o_open", "sum"), o_closed=("o_closed", "sum"),
            n_o=("o_open", "size")) if len(O) else pd.DataFrame()
        egg = pd.DataFrame({"r": e_reg, "e_open": E["e_open"].to_numpy(),
                            "e_closed": E["e_closed"].to_numpy()}).groupby("r").agg(
            e_open=("e_open", "sum"), e_closed=("e_closed", "sum"),
            n_e=("e_open", "size")) if len(E) else pd.DataFrame()
        mg = pd.DataFrame({"r": m_reg, "p": ends["p"].to_numpy()}).groupby(
            "r").agg(marg=("p", "mean"))

        reg_out: dict = {}
        for r in set(og.index) | set(mg.index):
            marg = float(mg.loc[r, "marg"]) if r in mg.index else np.nan
            delta = np.nan
            if r in og.index and r in egg.index:
                no = og.loc[r, "n_o"]; ne = egg.loc[r, "n_e"]
                if no > 0 and ne > 0:
                    oo = og.loc[r, "o_open"] / no; oc = og.loc[r, "o_closed"] / no
                    ee = egg.loc[r, "e_open"] / ne; ec = egg.loc[r, "e_closed"] / ne
                    delta = (np.log2(max(oo, eps)) - np.log2(max(ee, eps))) \
                        - (np.log2(max(oc, eps)) - np.log2(max(ec, eps)))
            n_o = int(og.loc[r, "n_o"]) if r in og.index else 0
            reg_out[r] = (delta, marg, n_o)
        out[str(gran)] = reg_out
    return cell, out


def _standardize(M: pd.DataFrame) -> np.ndarray:
    X = M.to_numpy(np.float64)
    mu = np.nanmean(X, axis=0, keepdims=True)
    X = np.where(np.isnan(X), mu, X)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return np.nan_to_num((X - mu) / sd, nan=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True,
                    help="Directory of *.posteriors.parquet (one context).")
    ap.add_argument("--samples", required=True, help="TSV with cell + group.")
    ap.add_argument("--min-distance", type=int, default=20_000)
    ap.add_argument("--max-distance", type=int, default=200_000,
                    help="Upper bound of the E–P band (default 200 kb).")
    ap.add_argument("--min-gch-sites", type=int, default=3)
    ap.add_argument("--granularities", default="chrom,25,10,5",
                    help="Comma list: 'chrom' or N (Mb window). Default "
                         "chrom,25,10,5.")
    ap.add_argument("--max-ends-per-chrom", type=int, default=4000,
                    help="Cap on a chromosome's ends used for the null pool.")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=50,
                    help="Label-permutation null for the concordance AUC "
                         "(chance floor accounting for #cells/#features). "
                         "0 disables. Default 50.")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--output-prefix", default=None)
    args = ap.parse_args()

    from nome_hic_concordance.pair_concordance.component_analysis import _cv_auc

    grans = [g if g == "chrom" else int(g)
             for g in args.granularities.split(",")]
    in_dir = Path(args.input_dir)
    files = sorted(in_dir.glob("*.posteriors.parquet"))
    if not files:
        sys.exit(f"No *.posteriors.parquet under {in_dir}")
    labels = _load_labels(args.samples)
    n_classes = labels.loc[labels.index.isin(
        [f.stem.replace(".posteriors", "") for f in files])].nunique()
    print(f"[ep-existence] {len(files)} cells | band "
          f"{args.min_distance:,}–{args.max_distance:,} bp | "
          f"{n_classes} cell types (chance AUC=0.5) | grans={grans}")

    if args.n_threads > 1 and len(files) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(f, grans, args.min_distance, args.max_distance,
                                   args.min_gch_sites, args.max_ends_per_chrom,
                                   args.random_seed + i)
            for i, f in enumerate(files))
    else:
        results = [process_cell(f, grans, args.min_distance, args.max_distance,
                                args.min_gch_sites, args.max_ends_per_chrom,
                                args.random_seed + i)
                   for i, f in enumerate(files)]
    results = [(c, d) for c, d in results if d]
    cells = [c for c, _ in results]
    y_all = labels.reindex(cells)
    keep = y_all.notna().to_numpy()
    if keep.sum() < 10:
        sys.exit("Too few labeled cells.")

    print(f"\n{'granularity':>12} {'n_regions':>10} {'med_pairs/reg':>13} "
          f"{'Δ_concord AUC':>16} {'perm_null':>10} {'p':>7} {'marginal AUC':>14}")
    rows = []
    for gran in grans:
        gk = str(gran)
        # cells × regions matrices for Δ and the marginal.
        all_regions = sorted({r for _, d in results for r in d.get(gk, {})})
        if not all_regions:
            continue
        ridx = {r: j for j, r in enumerate(all_regions)}
        D = np.full((len(cells), len(all_regions)), np.nan)
        Mg = np.full((len(cells), len(all_regions)), np.nan)
        npairs = []
        for i, (_, d) in enumerate(results):
            for r, (delta, marg, n_o) in d.get(gk, {}).items():
                j = ridx[r]
                D[i, j] = delta; Mg[i, j] = marg
                if n_o > 0:
                    npairs.append(n_o)
        Ddf = pd.DataFrame(D, index=cells, columns=all_regions)
        Mdf = pd.DataFrame(Mg, index=cells, columns=all_regions)
        y = y_all[keep].to_numpy()
        Xc = _standardize(Ddf.iloc[keep])
        Xm = _standardize(Mdf.iloc[keep])
        cmean, cstd = _cv_auc(Xc, y, n_folds=args.n_folds,
                              random_state=args.random_seed)
        mmean, mstd = _cv_auc(Xm, y, n_folds=args.n_folds,
                              random_state=args.random_seed)
        # Label-permutation null for the concordance AUC: the honest chance
        # floor for this #cells/#features (a single noisy feature can sit
        # well above 0.5 by chance, so 0.5 is not the right reference).
        perm_null, perm_p = np.nan, np.nan
        if args.n_perm > 0:
            prng = np.random.default_rng(args.random_seed)
            null = np.array([_cv_auc(Xc, prng.permutation(y),
                                     n_folds=args.n_folds,
                                     random_state=args.random_seed)[0]
                             for _ in range(int(args.n_perm))])
            null = null[np.isfinite(null)]
            if null.size:
                perm_null = float(null.mean())
                perm_p = float((np.sum(null >= cmean) + 1) / (null.size + 1))
        med = int(np.median(npairs)) if npairs else 0
        print(f"{gk:>12} {len(all_regions):>10} {med:>13} "
              f"{cmean:>10.3f}±{cstd:.3f} {perm_null:>10.3f} {perm_p:>7.3f} "
              f"{mmean:>8.3f}±{mstd:.3f}")
        rows.append({"granularity": gk, "n_regions": len(all_regions),
                     "median_pairs_per_region": med,
                     "concordance_auc": cmean, "concordance_auc_std": cstd,
                     "concordance_perm_null": perm_null,
                     "concordance_perm_p": perm_p,
                     "marginal_auc": mmean, "marginal_auc_std": mstd})

    print("\n  Read: Δ_concord AUC > 0.5 (beyond a few std) at any granularity")
    print("  => real, marginal-independent E–P coordination signal exists.")
    print("  Δ_concord AUC ≈ 0.5 everywhere => no detectable signal even when")
    print("  richly pooled (the RWR result was an imputation artifact).")
    if args.output_prefix:
        out = f"{args.output_prefix}.ep_existence.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n[ep-existence] wrote {out}")


if __name__ == "__main__":
    main()
