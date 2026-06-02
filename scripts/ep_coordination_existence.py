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
Δ matrix **classifies cell type above chance** (SVD → 5-fold CV OvR-macro AUC),
judged against a **label-permutation null**, alongside the per-region
mean-openness (marginal) AUC as a reference:

  - Δ AUC > permutation null at some granularity → real, marginal-independent
    E–P coordination signal exists (then Step 2: a held-out concordance-graph
    imputation can chase it at finer resolution).
  - Δ AUC ≈ null at every granularity → no detectable signal even when richly
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

_MULT = 100_000_000      # region_id = chrom_code * _MULT + bin_index


def _load_labels(path: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "cell_name" not in df.columns:
        df = df.rename(columns={"cell": "cell_name"} if "cell" in df.columns
                       else {df.columns[0]: "cell_name"})
    if "group" not in df.columns:
        raise SystemExit(f"{path} has no 'group' column.")
    return df.set_index("cell_name")["group"]


def _ranges_concat(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized concat of half-open ranges [lo[i], hi[i]).

    Returns (left_rep, right) where left_rep[k] is the source index i and
    right[k] runs over lo[i]..hi[i]-1 — no Python loop.
    """
    counts = (hi - lo).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    n = lo.shape[0]
    left_rep = np.repeat(np.arange(n), counts)
    starts = np.cumsum(counts) - counts
    within = np.arange(total) - np.repeat(starts, counts)
    right = np.repeat(lo, counts) + within
    return left_rep, right


def _window_pairs(pos: np.ndarray, p: np.ndarray, min_d: int, max_d: int,
                  cap: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All same-chrom end-pairs with separation in [min_d, max_d] (vectorized).

    Returns (left_pos, e_open, e_closed); left = smaller position (for region
    assignment).  Subsamples ends to ``cap`` first."""
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
    li, ri = _ranges_concat(lo, hi)
    if li.size == 0:
        return np.empty(0, np.int64), np.empty(0), np.empty(0)
    pa, pb = p[li], p[ri]
    return pos[li], pa * pb, (1.0 - pa) * (1.0 - pb)


def _region_ids(codes: np.ndarray, pos: np.ndarray, gran) -> np.ndarray:
    """Integer region id = chrom_code*_MULT + bin (bin=0 for 'chrom')."""
    if gran == "chrom":
        return codes.astype(np.int64) * _MULT
    size = int(gran) * 1_000_000
    return codes.astype(np.int64) * _MULT + (pos // size).astype(np.int64)


def _agg(ids: np.ndarray, weights: list) -> tuple:
    """(unique_ids, [weighted sums...], counts) via np.bincount — vectorized."""
    if ids.size == 0:
        return (np.empty(0, np.int64), [np.empty(0) for _ in weights],
                np.empty(0, np.int64))
    uniq, inv = np.unique(ids, return_inverse=True)
    R = uniq.shape[0]
    sums = [np.bincount(inv, weights=w, minlength=R) for w in weights]
    cnt = np.bincount(inv, minlength=R)
    return uniq, sums, cnt


def process_cell(path: Path, granularities: list, min_d: int, max_d: int,
                 min_gch: int, cap: int, seed: int):
    """Per-(region) Δ_signed (log2 O/E) + mean openness for one cell, at each
    granularity.  Fully vectorized.  Returns (cell, {gran: {region:(Δ,marg,n_o)}})."""
    cell = path.stem
    if cell.endswith(".posteriors"):
        cell = cell[: -len(".posteriors")]
    df = pd.read_parquet(path, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                         "posterior1", "posterior2"])
    if len(df) == 0:
        return cell, {}
    chrom = df["chrom"].astype(str).values
    pos1 = df["pos1"].to_numpy(np.int64); pos2 = df["pos2"].to_numpy(np.int64)
    n1 = df["n1"].to_numpy(np.int32); n2 = df["n2"].to_numpy(np.int32)
    p1 = df["posterior1"].to_numpy(np.float64, na_value=np.nan)
    p2 = df["posterior2"].to_numpy(np.float64, na_value=np.nan)
    inf1 = (n1 >= min_gch) & np.isfinite(p1)
    inf2 = (n2 >= min_gch) & np.isfinite(p2)

    # End pool (null + marginal): every informative end.
    e_chrom = np.concatenate([chrom[inf1], chrom[inf2]])
    e_pos = np.concatenate([pos1[inf1], pos2[inf2]])
    e_p = np.concatenate([p1[inf1], p2[inf2]])
    if e_pos.size == 0:
        return cell, {}
    uniq_chroms = np.unique(e_chrom)                      # chrom -> code map
    code_of = lambda arr: np.searchsorted(uniq_chroms, arr).astype(np.int64)

    # OBSERVED contacts in band.
    lo = np.minimum(pos1, pos2); hi = np.maximum(pos1, pos2)
    sep = hi - lo
    cm = inf1 & inf2 & (sep >= min_d) & (sep <= max_d)
    o_codes = code_of(chrom[cm]); o_pos = lo[cm]
    o_open = (p1 * p2)[cm]; o_closed = ((1 - p1) * (1 - p2))[cm]

    # NULL end-pairs in band, per chromosome (windowed, vectorized).
    rng = np.random.default_rng(seed)
    ep_pos_l, ep_open_l, ep_closed_l, ep_code_l = [], [], [], []
    for ci, c in enumerate(uniq_chroms):
        sel = e_chrom == c
        lp, eo, ec = _window_pairs(e_pos[sel], e_p[sel], min_d, max_d, cap, rng)
        if lp.size:
            ep_pos_l.append(lp); ep_open_l.append(eo); ep_closed_l.append(ec)
            ep_code_l.append(np.full(lp.size, ci, np.int64))
    if ep_pos_l:
        e_lpos = np.concatenate(ep_pos_l); e_open = np.concatenate(ep_open_l)
        e_closed = np.concatenate(ep_closed_l); e_codes = np.concatenate(ep_code_l)
    else:
        e_lpos = np.empty(0, np.int64); e_open = np.empty(0)
        e_closed = np.empty(0); e_codes = np.empty(0, np.int64)
    m_codes = code_of(e_chrom)

    eps = 1e-6
    out: dict = {}
    for gran in granularities:
        gk = str(gran)
        ro = _region_ids(o_codes, o_pos, gran)
        re = _region_ids(e_codes, e_lpos, gran)
        rm = _region_ids(m_codes, e_pos, gran)
        ids_o, (oo_s, oc_s), no = _agg(ro, [o_open, o_closed])
        ids_e, (eo_s, ec_s), ne = _agg(re, [e_open, e_closed])
        ids_m, (psum,), nm = _agg(rm, [e_p])
        all_ids = np.unique(np.concatenate([ids_o, ids_e, ids_m]))
        R = all_ids.shape[0]
        if R == 0:
            out[gk] = {}
            continue

        def place(ids, vals, fill=0.0):
            arr = np.full(R, fill, np.float64)
            if ids.size:
                arr[np.searchsorted(all_ids, ids)] = vals
            return arr

        n_o = place(ids_o, no.astype(float)); n_e = place(ids_e, ne.astype(float))
        with np.errstate(divide="ignore", invalid="ignore"):
            oo = place(ids_o, oo_s) / n_o; oc = place(ids_o, oc_s) / n_o
            ee = place(ids_e, eo_s) / n_e; ec = place(ids_e, ec_s) / n_e
            delta = (np.log2(np.clip(oo, eps, None)) - np.log2(np.clip(ee, eps, None))) \
                - (np.log2(np.clip(oc, eps, None)) - np.log2(np.clip(ec, eps, None)))
        delta = np.where((n_o > 0) & (n_e > 0), delta, np.nan)
        marg = place(ids_m, psum, fill=np.nan)
        nm_arr = place(ids_m, nm.astype(float))
        marg = np.where(nm_arr > 0, marg / np.where(nm_arr > 0, nm_arr, 1.0), np.nan)

        codes = (all_ids // _MULT).astype(int)
        bins = (all_ids % _MULT).astype(int)
        names = uniq_chroms[codes]
        if gran == "chrom":
            keys = names
        else:
            keys = np.char.add(np.char.add(names.astype(str), ":"),
                               bins.astype(str))
        out[gk] = {str(keys[i]): (float(delta[i]), float(marg[i]), int(n_o[i]))
                   for i in range(R)}
    return cell, out


def _standardize(M: pd.DataFrame) -> np.ndarray:
    X = M.to_numpy(np.float64)
    mu = np.nanmean(X, axis=0, keepdims=True)
    X = np.where(np.isnan(X), mu, X)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return np.nan_to_num((X - mu) / sd, nan=0.0)


def _svd_reduce(X: np.ndarray, n_svd: int) -> np.ndarray:
    """Truncated-SVD reduction so the classifier + permutations are fast and
    not overfit on thousands of region features (matches the comparison harness)."""
    if n_svd <= 0 or X.shape[1] <= n_svd:
        return X
    from sklearn.decomposition import TruncatedSVD
    k = max(1, min(n_svd, X.shape[0] - 1, X.shape[1]))
    return TruncatedSVD(n_components=k, random_state=0).fit_transform(X)


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
    ap.add_argument("--n-svd", type=int, default=30,
                    help="SVD components for the classifier (0 = raw features). "
                         "Default 30.")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=50,
                    help="Label-permutation null for the concordance AUC. "
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
          f"{n_classes} cell types | grans={grans} | n_svd={args.n_svd}")

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
        y = y_all[keep].to_numpy()
        Xc = _svd_reduce(_standardize(pd.DataFrame(D).iloc[keep]), args.n_svd)
        Xm = _svd_reduce(_standardize(pd.DataFrame(Mg).iloc[keep]), args.n_svd)
        cmean, cstd = _cv_auc(Xc, y, n_folds=args.n_folds,
                              random_state=args.random_seed)
        mmean, mstd = _cv_auc(Xm, y, n_folds=args.n_folds,
                              random_state=args.random_seed)
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

    print("\n  Read: Δ_concord AUC > perm_null (p<0.05) at any granularity")
    print("  => real, marginal-independent E–P coordination signal exists.")
    print("  Δ_concord AUC ≈ perm_null everywhere => no detectable signal even")
    print("  when richly pooled (the RWR result was an imputation artifact).")
    if args.output_prefix:
        out = f"{args.output_prefix}.ep_existence.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n[ep-existence] wrote {out}")


if __name__ == "__main__":
    main()
