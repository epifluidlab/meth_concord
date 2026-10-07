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

THE DRAWN NULL (--null).  None of the ladder's controls can tell coordination
from the methylation level at 1-3 pairs per region.  There Δ is a log ratio
of very small sums, and how widely it SPREADS depends on the region's level;
regressing out level, level^2 and coverage removes only its average, and
split-half, batch and label permutation test whether there is an association,
not where it comes from.  So a random forest separates cell types that differ
in methylation level alone -- Δ|m+cov 0.85 at 1 Mb on synthetic cells with no
coordination anywhere, where L2-logistic stays at 0.56.  --null runs the
identical ladder on --null-draws draws in which every contact is replaced by
a non-contact pair of the cell's own ends, in the same region at a matched
distance: same depth, coverage, methylation and distance mix, no
coordination.  Only real clearly above the draws counts; its rank among them
is the p-value.

    python scripts/ep_coordination_existence.py \\
        --input-dir output/.../context_HCG/04_posteriors \\
        --samples samples.with_batches.tsv \\
        --min-distance 20000 --max-distance 2000000 --min-gch-sites 1 \\
        --granularities 5,2,1 --classifier rf --min-region-frac 0.5 \\
        --split-half --batch-column exp_batch,lib_batch \\
        --null --null-draws 9 --n-perm 0 --n-threads 10 \\
        --output-prefix hcg_ep_existence_null
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nome_hic_concordance.utils import manifest_text

_MULT = 100_000_000      # region_id = chrom_code * _MULT + bin_index


def _load_labels(path: str) -> pd.Series:
    df = pd.read_csv(manifest_text(path), sep="\t", dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "cell_name" not in df.columns:
        df = df.rename(columns={"cell": "cell_name"} if "cell" in df.columns
                       else {df.columns[0]: "cell_name"})
    if "group" not in df.columns:
        raise SystemExit(f"{path} has no 'group' column.")
    return df.set_index("cell_name")["group"]


def _load_column(path: str, col: str) -> pd.Series:
    """Read an arbitrary per-cell column (e.g. batch) from the samples TSV."""
    df = pd.read_csv(manifest_text(path), sep="\t", dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "cell_name" not in df.columns:
        df = df.rename(columns={"cell": "cell_name"} if "cell" in df.columns
                       else {df.columns[0]: "cell_name"})
    col = col.strip().lower().replace(" ", "_")
    if col not in df.columns:
        raise SystemExit(f"{path} has no '{col}' column (have {df.columns.tolist()}).")
    return df.set_index("cell_name")[col]


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
    empty = (np.empty(0, np.int64), np.empty(0), np.empty(0), np.empty(0, np.int64))
    n = pos.shape[0]
    if n < 2:
        return empty
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
        return empty
    pa, pb = p[li], p[ri]
    return (pos[li], pa * pb, (1.0 - pa) * (1.0 - pb), pos[ri] - pos[li])


_PAIR_KEY = np.int64(1) << np.int64(32)


def _pair_keys(lo, hi) -> np.ndarray:
    """One sorted int64 per read pair (lo << 32 | hi), for refusing them."""
    return np.unique(np.asarray(lo, np.int64) * _PAIR_KEY
                     + np.asarray(hi, np.int64))


def _is_pair(lo, hi, keys) -> np.ndarray:
    """Whether each end-pair (lo[k], hi[k]) is one of the read pairs ``keys``."""
    lo = np.asarray(lo, np.int64)
    if keys is None or len(keys) == 0 or lo.size == 0:
        return np.zeros(lo.shape, dtype=bool)
    q = lo * _PAIR_KEY + np.asarray(hi, np.int64)
    at = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
    return keys[at] == q


def _null_dbins(min_d: int, max_d: int, per_decade: int = 8) -> np.ndarray:
    """Log-spaced distance bins across the band, for matching null pairs."""
    lo, hi = np.log10(max(min_d, 1)), np.log10(max(max_d, min_d + 1))
    n = max(1, int(np.ceil((hi - lo) * per_decade)))
    return np.logspace(lo, hi, n + 1)


def _draw_null(c_reg, c_dist, b_reg, b_dist, dbins, rng):
    """One null pair per contact: a pair from the null pool in the SAME
    region at a matched distance (same log bin), drawn with replacement;
    failing that, any pair in the region.  Returns (pool index or -1,
    whether it fell back to an unmatched distance)."""
    c_reg = np.asarray(c_reg, np.int64)
    n = c_reg.size
    if n == 0 or np.asarray(b_reg).size == 0:
        return np.full(n, -1, np.int64), np.zeros(n, dtype=bool)
    nb = len(dbins) - 1

    def dbin(d):
        return np.clip(np.searchsorted(dbins, d, side="right") - 1, 0, nb - 1)

    bkey = np.asarray(b_reg, np.int64) * nb + dbin(b_dist)
    order = np.argsort(bkey, kind="stable")
    sk = bkey[order]
    ckey = c_reg * nb + dbin(c_dist)
    lo = np.searchsorted(sk, ckey, side="left")
    hi = np.searchsorted(sk, ckey, side="right")
    rlo = np.searchsorted(sk, c_reg * nb, side="left")
    rhi = np.searchsorted(sk, c_reg * nb + nb, side="left")
    matched = hi > lo
    L = np.where(matched, lo, rlo)
    H = np.where(matched, hi, rhi)
    ok = H > L
    pick = np.minimum(L + (rng.random(n) * (H - L)).astype(np.int64),
                      order.size - 1)
    return np.where(ok, order[pick], -1), ok & ~matched


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
                 min_gch: int, cap: int, seed: int, decay_dbins: np.ndarray = None,
                 null: bool = False, null_draws: int = 1):
    """Per-(region) Δ_signed (log2 O/E) + mean openness for one cell, at each
    granularity.  Fully vectorized.  Returns (cell, {gran: {region:(Δ,marg,n_o)}},
    decay, null_part) where ``decay`` holds distance-binned soft-agreement
    sums/counts for contacts (O) and the random null (E) — for the Hi-C-style
    distance plot.

    ``null``: also the same Δ with each contact replaced by a NON-contact pair
    of this cell's ends in the same region at a matched distance -- what the
    statistic looks like with no coordination, at the same depth, coverage and
    methylation.  The ends are split into two disjoint halves per chromosome:
    E is built from one, the null pairs from the other, and neither pool may
    hold one of the cell's read pairs.  The real Δ is then computed against
    that same E, so real and null differ only in their O pairs.  The split
    is fixed and ``null_draws`` independent sets of null pairs are drawn from
    it, so the real Δ is the same for every draw.  ``null_part`` is ``{"out":
    [{gran: {region: (Δ_null, marg, n_null)}} per draw], "stats": {gran:
    (matched, other distance, none)}}``, or None."""
    cell = path.stem
    if cell.endswith(".posteriors"):
        cell = cell[: -len(".posteriors")]
    df = pd.read_parquet(path, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                         "posterior1", "posterior2"])
    if len(df) == 0:
        return cell, {}, None
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
        return cell, {}, None
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
    ep_pos_l, ep_open_l, ep_closed_l, ep_code_l, ep_dist_l = [], [], [], [], []
    nb_pos_l, nb_open_l, nb_closed_l, nb_code_l, nb_dist_l = [], [], [], [], []
    if null:
        # The cell's read pairs, which neither E nor the null may re-form.
        both = inf1 & inf2
        rp_keys = {c: _pair_keys(lo[both][chrom[both] == c],
                                 hi[both][chrom[both] == c])
                   for c in np.unique(chrom[both])}
        nrng = np.random.default_rng(seed + 7919)
    for ci, c in enumerate(uniq_chroms):
        sel = e_chrom == c
        if not null:
            lp, eo, ec, ed = _window_pairs(e_pos[sel], e_p[sel], min_d, max_d,
                                           cap, rng)
            if lp.size:
                ep_pos_l.append(lp); ep_open_l.append(eo); ep_closed_l.append(ec)
                ep_code_l.append(np.full(lp.size, ci, np.int64))
                ep_dist_l.append(ed)
            continue
        cpos, cp = e_pos[sel], e_p[sel]
        half = np.zeros(cpos.size, dtype=bool)
        half[nrng.permutation(cpos.size)[: cpos.size // 2]] = True
        for keep_, lists in ((half, (ep_pos_l, ep_open_l, ep_closed_l,
                                     ep_code_l, ep_dist_l)),
                             (~half, (nb_pos_l, nb_open_l, nb_closed_l,
                                      nb_code_l, nb_dist_l))):
            lp, eo, ec, ed = _window_pairs(cpos[keep_], cp[keep_], min_d,
                                           max_d, cap, rng)
            ok = ~_is_pair(lp, lp + ed, rp_keys.get(c))
            if ok.any():
                for lst, arr in zip(lists, (lp[ok], eo[ok], ec[ok],
                                            np.full(int(ok.sum()), ci, np.int64),
                                            ed[ok])):
                    lst.append(arr)
    if ep_pos_l:
        e_lpos = np.concatenate(ep_pos_l); e_open = np.concatenate(ep_open_l)
        e_closed = np.concatenate(ep_closed_l); e_codes = np.concatenate(ep_code_l)
        e_dist = np.concatenate(ep_dist_l)
    else:
        e_lpos = np.empty(0, np.int64); e_open = np.empty(0)
        e_closed = np.empty(0); e_codes = np.empty(0, np.int64); e_dist = np.empty(0)
    m_codes = code_of(e_chrom)
    if null:
        cat = (lambda lst, dt: np.concatenate(lst) if lst
               else np.empty(0, dt))
        b_lpos, b_open = cat(nb_pos_l, np.int64), cat(nb_open_l, np.float64)
        b_closed = cat(nb_closed_l, np.float64)
        b_codes, b_dist = cat(nb_code_l, np.int64), cat(nb_dist_l, np.int64)
        o_sep = sep[cm]
        null_dbins = _null_dbins(min_d, max_d)

    # Distance-decay accumulators (Hi-C-style), over the SEPARATE decay range
    # [decay_dbins[0], decay_dbins[-1]] using ALL informative pairs in that
    # range (independent of the analysis --min/--max-distance band).  Soft
    # agreement (= co-open + co-closed) summed over log-distance bins for
    # contacts (O) and the distance-matched random null (E).
    decay = None
    if decay_dbins is not None:
        dmin, dmax = float(decay_dbins[0]), float(decay_dbins[-1])
        nb = len(decay_dbins) - 1

        def _bins(dist):
            return np.clip(np.searchsorted(decay_dbins, dist, side="right") - 1,
                           0, nb - 1)

        # O: every informative pair in the decay range — co-open & co-closed
        # tracked SEPARATELY (so a signed log2(O/E) can be formed per bin).
        dm = inf1 & inf2 & (sep >= dmin) & (sep <= dmax)
        ob = _bins(sep[dm])
        O_open = np.bincount(ob, weights=(p1 * p2)[dm], minlength=nb)
        O_closed = np.bincount(ob, weights=((1 - p1) * (1 - p2))[dm], minlength=nb)
        O_cnt = np.bincount(ob, minlength=nb).astype(float)
        # E: windowed non-contact null over the decay range, per chromosome.
        eo_l, ec_l, ed_l = [], [], []
        for c in uniq_chroms:
            sel = e_chrom == c
            _, eo, ec, ed = _window_pairs(e_pos[sel], e_p[sel], dmin, dmax,
                                          cap, rng)
            if ed.size:
                eo_l.append(eo); ec_l.append(ec); ed_l.append(ed)
        if ed_l:
            eb = _bins(np.concatenate(ed_l))
            E_open = np.bincount(eb, weights=np.concatenate(eo_l), minlength=nb)
            E_closed = np.bincount(eb, weights=np.concatenate(ec_l), minlength=nb)
            E_cnt = np.bincount(eb, minlength=nb).astype(float)
        else:
            E_open = np.zeros(nb); E_closed = np.zeros(nb); E_cnt = np.zeros(nb)
        decay = {"O_open_sum": O_open, "O_closed_sum": O_closed, "O_cnt": O_cnt,
                 "E_open_sum": E_open, "E_closed_sum": E_closed, "E_cnt": E_cnt}

    eps = 1e-6
    out: dict = {}
    n_draws = max(1, int(null_draws)) if null else 0
    out_null: list = [{} for _ in range(n_draws)]
    stats: dict = {}
    draw_rngs = [np.random.default_rng(seed + 104_729 + 1_000_003 * d)
                 for d in range(n_draws)]
    for gran in granularities:
        gk = str(gran)
        ro = _region_ids(o_codes, o_pos, gran)
        re = _region_ids(e_codes, e_lpos, gran)
        rm = _region_ids(m_codes, e_pos, gran)
        ids_o, (oo_s, oc_s), no = _agg(ro, [o_open, o_closed])
        ids_e, (eo_s, ec_s), ne = _agg(re, [e_open, e_closed])
        ids_m, (psum,), nm = _agg(rm, [e_p])
        draws_ = []
        if null:
            b_reg = _region_ids(b_codes, b_lpos, gran)
            tally = np.zeros(3, np.int64)
            for drng in draw_rngs:
                pick, other = _draw_null(ro, o_sep, b_reg, b_dist,
                                         null_dbins, drng)
                got = pick >= 0
                draws_.append(_agg(ro[got], [b_open[pick[got]],
                                             b_closed[pick[got]]]))
                tally += (int(got.sum() - other.sum()), int(other.sum()),
                          int((~got).sum()))
            stats[gk] = tuple(int(x) for x in tally)
        all_ids = np.unique(np.concatenate([ids_o, ids_e, ids_m]))
        R = all_ids.shape[0]
        if R == 0:
            out[gk] = {}
            for od in out_null:
                od[gk] = {}
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
        # Same E, same formula; only the O pairs differ.
        for od, (ids_n, (no_s, nc_s), nn) in zip(out_null, draws_):
            n_n = place(ids_n, nn.astype(float))
            with np.errstate(divide="ignore", invalid="ignore"):
                no_ = place(ids_n, no_s) / n_n
                nc_ = place(ids_n, nc_s) / n_n
                dnull = ((np.log2(np.clip(no_, eps, None))
                          - np.log2(np.clip(ee, eps, None)))
                         - (np.log2(np.clip(nc_, eps, None))
                            - np.log2(np.clip(ec, eps, None))))
            dnull = np.where((n_n > 0) & (n_e > 0), dnull, np.nan)
            od[gk] = {str(keys[i]): (float(dnull[i]), float(marg[i]),
                                     int(n_n[i])) for i in range(R)}
    null_part = {"out": out_null, "stats": stats} if null else None
    return cell, out, decay, null_part


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


def _zcols(X: np.ndarray) -> np.ndarray:
    """Per-column z-score; constant columns (std≈0) → all zeros (no influence)."""
    Xc = np.nan_to_num(X - np.nanmean(X, axis=0, keepdims=True))
    sd = Xc.std(axis=0, keepdims=True)
    return np.where(sd < 1e-12, 0.0, Xc / np.where(sd < 1e-12, 1.0, sd))


def _cv_rf(X: np.ndarray, y: np.ndarray, n_folds: int, seed: int,
           n_jobs: int = 1):
    """5-fold CV AUC with a random forest (binary → AUC, multiclass → OvR-macro).

    Tree ensembles weight any feature regardless of its variance, so — unlike
    SVD→LR — they recover label signal that lives in low-variance directions
    (the concern when classifying the marginal-residualized Δ).
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    le = LabelEncoder(); yy = le.fit_transform(y); nc = len(le.classes_)
    if nc < 2:
        return np.nan, np.nan
    folds = min(n_folds, int(np.bincount(yy).min()))
    if folds < 2:
        return np.nan, np.nan
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, yy):
        if len(np.unique(yy[tr])) < 2 or len(np.unique(yy[te])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                     n_jobs=n_jobs, random_state=seed)
        clf.fit(X[tr], yy[tr]); prob = clf.predict_proba(X[te])
        aucs.append(roc_auc_score(yy[te], prob[:, 1]) if nc == 2 else
                    roc_auc_score(yy[te], prob, multi_class="ovr", average="macro"))
    return (float(np.mean(aucs)), float(np.std(aucs))) if aucs else (np.nan, np.nan)


def _residualize(D: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Per-region (per-column) residual of Δ (D) on the marginal (M).

    The O/E controls the marginal at the pair level, but imperfectly — and
    when the marginal AUC is near 1.0 even a small residual correlation makes
    Δ classify cell type.  This removes the linear Δ–marginal relationship in
    each region, so a surviving AUC is marginal-INDEPENDENT coordination and a
    collapsing AUC means the Δ signal was residual marginal leakage.

    Done in standardized space so the coefficient is a bounded per-column
    correlation b∈[-1,1] (robust when a region's marginal has tiny variance —
    a plain cov/var would blow up and destroy real signal there).  A constant
    marginal column → b=0 → Δ passes through unchanged.
    """
    Ds, Ms = _zcols(D), _zcols(M)
    b = (Ds * Ms).mean(axis=0)            # per-column correlation, in [-1, 1]
    return Ds - b[None, :] * Ms


def _residualize_multi(D: np.ndarray, covars: list) -> np.ndarray:
    """Per-region residual of Δ on MULTIPLE covariates (each cells × regions).

    Fuller leakage control: regress each region's Δ across cells on
    [marginal, marginal², coverage] and keep the residual.  Removes nonlinear
    marginal dependence (Δ=log2(O/E) is nonlinear in the mean-openness
    marginal) and the coverage/sparsity confound (a few-pair Δ estimate's
    bias/noise tracks the per-region pair count, which itself differs by cell
    type).  Vectorized per-column OLS in standardized space (ridge 1e-6).
    """
    Ds = _zcols(D)
    Cs = [_zcols(C) for C in covars]
    if not Cs:
        return Ds
    K, R = len(Cs), Ds.shape[1]
    A = np.stack(Cs, axis=1)                       # cells × K × regions
    AtA = np.einsum("ikr,ilr->klr", A, A)          # K × K × regions
    Atd = np.einsum("ikr,ir->kr", A, Ds)           # K × regions
    AtA_b = np.moveaxis(AtA, 2, 0) + 1e-6 * np.eye(K)     # regions × K × K
    Atd_b = np.moveaxis(Atd, 1, 0)[:, :, None]           # regions × K × 1
    beta = np.linalg.solve(AtA_b, Atd_b)[:, :, 0]        # regions × K
    fit = np.einsum("ikr,rk->ir", A, beta)               # cells × regions
    return Ds - fit


def _residualize_against(R: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Remove the projection of R (cells × regions) onto the column space of a
    PER-CELL design B (cells × k, e.g. batch one-hot — same for every region).

    Used as a second stage on the marginal+coverage residual to also strip
    per-batch offsets: a single least-squares projection (B is shared across
    regions), so a surviving AUC is batch-independent coordination.
    """
    Bz = _zcols(B)
    if Bz.shape[1] == 0 or not np.isfinite(Bz).any():
        return R
    beta, *_ = np.linalg.lstsq(Bz, R, rcond=None)        # k × regions
    return R - Bz @ beta


def _make_clf(method: str, seed: int, n_jobs: int = 1):
    if method == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                      n_jobs=n_jobs, random_state=seed)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="ovr",
                              C=1.0)


def _cv_predict(X: np.ndarray, y: np.ndarray, method: str, n_folds: int,
                seed: int, n_jobs: int = 1):
    """Pooled out-of-fold predictions for an ROC curve.

    Binary → (y∈{0,1}, positive-class score).  Multiclass → micro-averaged
    OvR: (one-hot label flattened, predicted-proba flattened).  Returns None
    if CV isn't possible.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, label_binarize
    le = LabelEncoder(); yy = le.fit_transform(y); nc = len(le.classes_)
    if nc < 2 or X.shape[1] < 1:
        return None
    folds = min(n_folds, int(np.bincount(yy).min()))
    if folds < 2:
        return None
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    if nc == 2:
        score = np.full(len(yy), np.nan)
        for tr, te in skf.split(X, yy):
            clf = _make_clf(method, seed, n_jobs).fit(X[tr], yy[tr])
            score[te] = clf.predict_proba(X[te])[:, 1]
        ok = np.isfinite(score)
        return yy[ok].astype(float), score[ok]
    Yb = label_binarize(yy, classes=list(range(nc)))
    P = np.full((len(yy), nc), np.nan)
    for tr, te in skf.split(X, yy):
        clf = _make_clf(method, seed, n_jobs).fit(X[tr], yy[tr])
        P[te] = clf.predict_proba(X[te])
    ok = np.isfinite(P).all(axis=1)
    return Yb[ok].ravel().astype(float), P[ok].ravel()


def _plot_roc(roc_data: dict, classifier: str, output: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    grans = [g for g in roc_data if any(roc_data[g].values())]
    if not grans:
        return
    ncol = min(3, len(grans)); nrow = (len(grans) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow),
                             squeeze=False)
    for idx, gk in enumerate(grans):
        ax = axes[idx // ncol][idx % ncol]
        for name in ("Δ", "Δ|m", "Δ|m+cov"):
            dat = roc_data[gk].get(name)
            if dat is None:
                continue
            ytrue, yscore = dat
            fpr, tpr, _ = roc_curve(ytrue, yscore)
            ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(f"granularity = {gk}", fontsize=10)
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.legend(fontsize=8, loc="lower right")
    for j in range(len(grans), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Cross-validated ROC ({classifier})  —  Δ vs Δ|m vs Δ|m+cov",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output)
    plt.close(fig)


def _decade_fmt(x: float, pos=None) -> str:
    """Tick label = log10 exponent + human distance, e.g. '5 (100kb)'."""
    e = int(round(np.log10(x)))
    human = {3: "1kb", 4: "10kb", 5: "100kb", 6: "1Mb", 7: "10Mb",
             8: "100Mb"}.get(e, f"1e{e}")
    return f"{e} ({human})"


def _decade_ticks(ax):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(3, 9)]))
    ax.xaxis.set_major_formatter(FuncFormatter(_decade_fmt))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel("log10(distance between ends)  —  exponent (distance)")


def _plot_distance_decay(mid: np.ndarray, curves: dict, dbins: np.ndarray,
                         output: str) -> None:
    """Hi-C-style 3-panel distance decay with 95% CI shading (mean ± 1.96·SE
    across cells, the independent unit):
      (1) agreement      = P(ends share state) = co-open + co-closed,
      (2) signed log2(O/E) = log2(O_open/E_open) − log2(O_closed/E_closed),
      (3) residual         = O − E (agreement excess over the random null)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def band(ax, m, se, allcells=False, **kw):
        style = dict(lw=2.4, color="black", zorder=5) if allcells \
            else dict(lw=1.3, marker="o", ms=3)
        ax.plot(mid, m, **style, **kw)
        col = ax.lines[-1].get_color()
        ax.fill_between(mid, m - 1.96 * se, m + 1.96 * se,
                        where=np.isfinite(se), color=col,
                        alpha=0.12 if allcells else 0.2, linewidth=0)

    order = [g for g in sorted(curves) if g != "all cells"] + \
            (["all cells"] if "all cells" in curves else [])
    panels = [
        ("agree", "concordance = P(ends share state)\n= co-open + co-closed",
         "Concordance (soft agreement) vs distance", False),
        ("signed", "signed log2(O/E)\nlog2(O_open/E_open) − log2(O_closed/E_closed)",
         "Directional coordination log2(O/E) vs distance", True),
        ("resid", "residual agreement  (O − E)",
         "Excess concordance over distance-matched random", True),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7, 11))
    for ax, (key, ylab, title, zero) in zip(axes, panels):
        for g in order:
            band(ax, curves[g][f"{key}_mean"], curves[g][f"{key}_se"],
                 allcells=(g == "all cells"), label=g)
        if zero:
            ax.axhline(0.0, color="grey", lw=0.8)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlim(dbins[0], dbins[-1])
        _decade_ticks(ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main(argv=None) -> None:
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
    ap.add_argument("--classifier", choices=["svd-lr", "lr", "rf"], default="lr",
                    help="Classifier for the AUC.  'lr' (default) = L2 logistic "
                         "on the FULL standardized region features (no variance-"
                         "based reduction).  'svd-lr' = SVD→L2-LR (fast, but the "
                         "unsupervised SVD can discard low-variance label signal "
                         "— a concern for the marginal-residualized Δ).  'rf' = "
                         "random forest (recovers low-variance/non-linear signal). "
                         "Run >1 and triangulate; the permutation null calibrates "
                         "each.")
    ap.add_argument("--n-svd", type=int, default=30,
                    help="SVD components (only used by --classifier svd-lr; "
                         "0 = raw features). Default 30.")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=50,
                    help="Label-permutation null for the concordance AUC. "
                         "0 disables. Default 50.")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--min-region-frac", type=float, default=0.0,
                    help="Keep only regions with a Δ observed in >= this "
                         "fraction of labeled cells (default 0 = all).  Set "
                         "e.g. 0.5–0.8 to remove the missingness/imputation "
                         "confound: on well-observed regions there is little "
                         "NaN for the classifier to exploit, so a surviving "
                         "Δ|m+cov AUC is genuine concordance, not contact "
                         "presence.")
    ap.add_argument("--decay-min-distance", type=int, default=1000,
                    help="Distance-decay plot ONLY: lower bound (default 1 kb). "
                         "Independent of --min-distance; uses ALL read pairs in "
                         "[--decay-min-distance, --decay-max-distance].")
    ap.add_argument("--decay-max-distance", type=int, default=1_000_000,
                    help="Distance-decay plot ONLY: upper bound (default 1 Mb).")
    ap.add_argument("--decay-bins", type=int, default=40,
                    help="Number of log-spaced distance bins for the decay plot "
                         "(default 40 ≈ 10 per decade → smooth Hi-C-style curve).")
    ap.add_argument("--batch-column", action="append", default=[],
                    metavar="COLUMN",
                    help="Per-cell batch column(s) in --samples (e.g. exp_batch, "
                         "lib_batch).  Repeatable AND comma-separated: "
                         "'--batch-column exp_batch --batch-column lib_batch' or "
                         "'--batch-column exp_batch,lib_batch'.  Residualizes Δ "
                         "ALSO against the UNION of their one-hots (additive — "
                         "removes each factor's per-batch offsets), on top of "
                         "marginal+coverage → a Δ|m+cov+batch AUC.  Survives => "
                         "batch-independent; collapses => a batch effect. "
                         "(split-half can't catch batch — it's global per-cell.)")
    ap.add_argument("--split-half", action="store_true",
                    help="Genomic split-half consistency: classify Δ|m+cov on "
                         "even- vs odd-indexed chromosomes independently. If "
                         "BOTH halves are significant, the signal is distributed "
                         "genome-wide (not a few-region artifact). Adds two "
                         "columns (evenchr/oddchr AUC + perm-p).")
    ap.add_argument("--null", action="store_true",
                    help="Also run the identical ladder on a DRAWN NULL: in "
                         "each cell, every contact replaced by a non-contact "
                         "pair of the cell's own ends in the same region at a "
                         "matched distance (same depth, coverage, methylation "
                         "and distance mix; no coordination).  Each cell's "
                         "ends are split in two disjoint halves, one for E and "
                         "one for the null, and the real Δ is computed against "
                         "that same E.  WHY: at 1-3 pairs per region Δ's SPREAD "
                         "depends on the methylation level, and regressing out "
                         "level, level^2 and coverage removes only its average "
                         "-- a random forest can still separate cell types "
                         "that differ in level alone.  Only real >> null is "
                         "evidence of coordination.  With --output-prefix also "
                         "writes the per-cell x region matrices, real and "
                         "null.")
    ap.add_argument("--null-draws", type=int, default=5, metavar="K",
                    help="Independent null draws with --null (default 5).  "
                         "With no coordination the real Δ is just one more "
                         "draw, so its rank among them is a p-value: "
                         "(1 + #null >= real) / (K + 1).  K=19 resolves 0.05.  "
                         "Each draw is another pass of the ladder.")
    ap.add_argument("--output-prefix", default=None)
    args = ap.parse_args(argv)

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
    feat = (f"SVD top-{args.n_svd}" if args.classifier == "svd-lr"
            else "ALL region features (no SVD)")
    print(f"[ep-existence] {len(files)} cells | band "
          f"{args.min_distance:,}–{args.max_distance:,} bp | "
          f"{n_classes} cell types | grans={grans} | "
          f"classifier={args.classifier} → {feat}")

    # Log-spaced distance bins for the Hi-C-style distance-decay plot — over
    # its OWN range (default 1 kb–10 Mb), independent of the analysis band.
    dbins = np.logspace(np.log10(max(args.decay_min_distance, 1)),
                        np.log10(max(args.decay_max_distance,
                                     args.decay_min_distance + 1)),
                        int(args.decay_bins) + 1)

    if args.n_threads > 1 and len(files) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(f, grans, args.min_distance, args.max_distance,
                                   args.min_gch_sites, args.max_ends_per_chrom,
                                   args.random_seed + i, dbins, args.null,
                                   args.null_draws)
            for i, f in enumerate(files))
    else:
        results = [process_cell(f, grans, args.min_distance, args.max_distance,
                                args.min_gch_sites, args.max_ends_per_chrom,
                                args.random_seed + i, dbins, args.null,
                                args.null_draws)
                   for i, f in enumerate(files)]
    nulls = {c: nl for c, d, _dec, nl in results if d}
    results = [(c, d, dec) for c, d, dec, _nl in results if d]
    cells = [c for c, _, _ in results]
    y_all = labels.reindex(cells)
    keep = y_all.notna().to_numpy()
    if keep.sum() < 10:
        sys.exit("Too few labeled cells.")

    y = y_all[keep].to_numpy()
    method = args.classifier

    # Optional batch covariates (one-hot over the labeled cells) for the
    # Δ|m+cov+batch control — UNION of one or more batch factors (additive).
    batch_oh = None
    batch_cols = list(dict.fromkeys(
        c.strip() for item in args.batch_column for c in item.split(",")
        if c.strip()))
    if batch_cols:
        ohs = []
        for col in batch_cols:
            bvals = _load_column(args.samples, col).reindex(cells).to_numpy()
            oh = pd.get_dummies(pd.Series(bvals[keep]), prefix=col,
                                drop_first=True, dummy_na=False)
            if oh.shape[1]:
                ohs.append(oh.to_numpy(dtype=np.float64))
        if ohs:
            batch_oh = np.hstack(ohs)
            print(f"[ep-existence] batch control on {batch_cols}: "
                  f"{batch_oh.shape[1]} indicator columns (additive union)")
        else:
            print(f"[ep-existence] batch control {batch_cols}: no usable "
                  f"levels — skipped")

    def _score(X, yy):
        if method == "rf":
            # Permutations already run one process per thread; without
            # them the forest gets the threads.  Same trees either way.
            return _cv_rf(X, yy, args.n_folds, args.random_seed,
                          n_jobs=1 if args.n_perm > 0 else args.n_threads)
        return _cv_auc(X, yy, n_folds=args.n_folds, random_state=args.random_seed)

    def _prep(Xstd):
        # SVD only for svd-lr; lr/rf classify the FULL standardized features so
        # the (unsupervised) SVD can't discard low-variance label signal.
        return _svd_reduce(Xstd, args.n_svd) if method == "svd-lr" else Xstd

    def _auc_perm(X):
        mean, std = _score(X, y)
        p = np.nan
        if args.n_perm > 0 and np.isfinite(mean):
            prng = np.random.default_rng(args.random_seed)
            yperms = [prng.permutation(y) for _ in range(int(args.n_perm))]
            if args.n_threads > 1:
                from joblib import Parallel, delayed
                res = Parallel(n_jobs=args.n_threads)(
                    delayed(_score)(X, yp) for yp in yperms)
                null = np.array([m for m, _ in res])
            else:
                null = np.array([_score(X, yp)[0] for yp in yperms])
            null = null[np.isfinite(null)]
            if null.size:
                p = float((np.sum(null >= mean) + 1) / (null.size + 1))
        return mean, std, p

    print(f"\n[classifier={method}]  Δ|m = Δ resid. on marginal; Δ|m+cov = on "
          f"marginal+marginal²+coverage;  miss = AUC of the contact-PRESENCE "
          f"mask (scHi-C-like reference)")
    print(f"{'granularity':>11} {'n_reg':>6} {'pairs/reg':>9} "
          f"{'Δ AUC':>8} {'p':>6} {'Δ|m':>7} {'p':>6} "
          f"{'Δ|m+cov':>8} {'p':>6} {'miss':>6} {'marg':>6}")
    rows = []
    roc_data: dict = {}
    marg_auc: dict = {}

    def _ladder(gk, label, D, Mk, Cov, region_keys):
        """The confound ladder on one cells x regions Δ matrix; prints its
        line and returns (CSV row, the Δ|m+cov residual)."""
        n_reg = D.shape[1]
        covk_obs = Cov[keep][np.isfinite(Cov[keep])]
        med = int(np.median(covk_obs)) if covk_obs.size else 0
        Dk = D[keep]
        Covk = np.log1p(np.nan_to_num(Cov[keep], nan=0.0))
        # Missingness mask: which (cell, region) have a Δ at all (contact
        # PRESENCE, scHi-C-like).  A reference benchmark: if the residual AUC
        # ≈ the missingness AUC, the "signal" is just which regions a cell has
        # contacts in (coverage/contact-map), NOT epigenetic concordance.
        Miss = np.isfinite(Dk).astype(np.float64)
        Xc = _prep(_standardize(pd.DataFrame(Dk)))
        Xmiss = _prep(_standardize(pd.DataFrame(Miss)))
        # Leakage controls: linear marginal, then marginal+marginal²+coverage.
        Xr = _prep(_standardize(pd.DataFrame(_residualize(Dk, Mk))))
        R2resid = _residualize_multi(Dk, [Mk, Mk * Mk, Covk])
        Xr2 = _prep(_standardize(pd.DataFrame(R2resid)))
        cmean, cstd, cp = _auc_perm(Xc)
        rmean, rstd, rp = _auc_perm(Xr)
        r2mean, r2std, r2p = _auc_perm(Xr2)
        if gk not in marg_auc:            # the same for the real and the null
            marg_auc[gk] = _score(_prep(_standardize(pd.DataFrame(Mk))), y)
        mmean, mstd = marg_auc[gk]
        missmean, _ = _score(Xmiss, y)
        # Batch control: also strip per-batch offsets from Δ|m+cov.
        b3mean = b3p = np.nan
        if batch_oh is not None:
            b3mean, _, b3p = _auc_perm(_prep(_standardize(
                pd.DataFrame(_residualize_against(R2resid, batch_oh)))))
        # Split-half genomic consistency: interleave chromosomes into two
        # disjoint halves; classify Δ|m+cov on each independently.  Both halves
        # significant => the signal is distributed genome-wide, not driven by a
        # few regions (rules out a region-specific artifact).
        shA = shAp = shB = shBp = np.nan
        if args.split_half:
            chrom_of = np.array([k.split(":")[0] for k in region_keys])
            uchr = sorted(set(chrom_of))
            setA = set(uchr[0::2])
            mA = np.array([c in setA for c in chrom_of])
            if mA.sum() >= 2:
                shA, _, shAp = _auc_perm(_prep(_standardize(
                    pd.DataFrame(R2resid[:, mA]))))
            if (~mA).sum() >= 2:
                shB, _, shBp = _auc_perm(_prep(_standardize(
                    pd.DataFrame(R2resid[:, ~mA]))))
        if args.output_prefix and label == "real":   # out-of-fold preds for ROC
            roc_data[gk] = {
                name: _cv_predict(X_, y, method, args.n_folds,
                                  args.random_seed, args.n_threads)
                for name, X_ in (("Δ", Xc), ("Δ|m", Xr), ("Δ|m+cov", Xr2))}
        sh = (f"  | even {shA:.3f}({shAp:.3f}) odd {shB:.3f}({shBp:.3f})"
              if args.split_half else "")
        bt = (f"  | Δ|m+cov+batch {b3mean:.3f}({b3p:.3f})"
              if batch_oh is not None else "")
        tag = gk if label == "real" else label.replace("drawn_", "")
        print(f"{tag:>11} {n_reg:>6} {med:>9} "
              f"{cmean:>7.3f} {cp:>6.3f} {rmean:>7.3f} {rp:>6.3f} "
              f"{r2mean:>7.3f} {r2p:>6.3f} {missmean:>6.3f} {mmean:>6.3f}{sh}{bt}")
        # 'drawn_null<d>', never a bare "null": pandas reads that as missing.
        return ({"classifier": method,
                 "data": "real" if label == "real" else "drawn_null",
                 "draw": -1 if label == "real" else int(label[10:]),
                 "granularity": gk, "n_regions": n_reg,
                 "median_pairs_per_region": med,
                 "concordance_auc": cmean, "concordance_auc_std": cstd,
                 "concordance_perm_p": cp,
                 "concordance_resid_auc": rmean,
                 "concordance_resid_std": rstd,
                 "concordance_resid_perm_p": rp,
                 "concordance_resid_full_auc": r2mean,
                 "concordance_resid_full_std": r2std,
                 "concordance_resid_full_perm_p": r2p,
                 "resid_full_evenchr_auc": shA,
                 "resid_full_evenchr_perm_p": shAp,
                 "resid_full_oddchr_auc": shB,
                 "resid_full_oddchr_perm_p": shBp,
                 "resid_full_batch_auc": b3mean,
                 "resid_full_batch_perm_p": b3p,
                 "missingness_auc": missmean,
                 "marginal_auc": mmean, "marginal_auc_std": mstd}, R2resid)

    def _write_matrices(gk, label, D, R2resid, region_keys):
        """Per-cell x region Δ (all cells, NaN = no contact) and its Δ|m+cov
        residual (the labelled cells the ladder classifies)."""
        def bp(key):
            if gk == "chrom":
                return key
            chrom_, b = key.rsplit(":", 1)
            return f"{chrom_}:{int(b) * int(gk) * 1_000_000}"
        cols = [bp(k) for k in region_keys]
        tag = "chrom" if gk == "chrom" else f"{gk}mb"
        sfx = "" if label == "real" else "_null"
        for stem, M, idx in (("delta", D, cells),
                             ("delta_mcov", R2resid,
                              [c for c, k in zip(cells, keep) if k])):
            path = f"{args.output_prefix}.{tag}.{stem}{sfx}.parquet"
            pd.DataFrame(M, index=pd.Index(idx, name="cell"),
                         columns=cols).to_parquet(path)

    for gran in grans:
        gk = str(gran)
        all_regions = sorted({r for _, d, _ in results for r in d.get(gk, {})})
        if not all_regions:
            continue
        ridx = {r: j for j, r in enumerate(all_regions)}

        def _matrices(per_cell):
            D = np.full((len(cells), len(all_regions)), np.nan)
            Mg = np.full((len(cells), len(all_regions)), np.nan)
            Cov = np.full((len(cells), len(all_regions)), np.nan)
            for i, d in enumerate(per_cell):
                for r, (delta, marg, n_o) in d.get(gk, {}).items():
                    j = ridx[r]
                    D[i, j] = delta; Mg[i, j] = marg; Cov[i, j] = n_o
            return D, Mg, Cov

        D, Mg, Cov = _matrices([d for _, d, _ in results])
        data = [("real", D, Cov)]
        if args.null:
            for d_ in range(max(1, int(args.null_draws))):
                Dn, _, Covn = _matrices([nulls[c]["out"][d_] if nulls[c]
                                         else {} for c in cells])
                data.append((f"drawn_null{d_}", Dn, Covn))
            st = np.sum([nulls[c]["stats"].get(gk, (0, 0, 0))
                         for c in cells if nulls[c]], axis=0)
            tot = max(int(np.sum(st)), 1)
            print(f"[ep-existence] null pairs at {gk}: {st[0] / tot:.1%} at a "
                  f"matched distance, {st[1] / tot:.1%} in the region at "
                  f"another distance, {st[2] / tot:.1%} none (left "
                  f"unobserved)")
        region_keys = all_regions             # aligned with D columns
        # Region-observation filter: drop sparsely-observed regions so the
        # classifier can't exploit the missingness/imputation pattern.  Set
        # on the REAL Δ and applied to the null too, so both see the same
        # regions.
        if args.min_region_frac > 0:
            of = np.isfinite(D[keep]).mean(axis=0)
            cols = of >= args.min_region_frac
            if int(cols.sum()) < 2:
                print(f"{gk:>11} {int(cols.sum()):>6}  (too few regions after "
                      f"--min-region-frac {args.min_region_frac})")
                continue
            Mg = Mg[:, cols]
            data = [(lab, DD[:, cols], CC[:, cols]) for lab, DD, CC in data]
            region_keys = [r for r, c in zip(all_regions, cols) if c]
        for label, DD, CC in data:
            row, R2resid = _ladder(gk, label, DD, Mg[keep], CC, region_keys)
            rows.append(row)
            if args.output_prefix and label in ("real", "drawn_null0"):
                _write_matrices(gk, label, DD, R2resid, region_keys)

    if args.null:
        print("\n  REAL vs its drawn NULL (same cells, regions, depth, distance "
              "mix and E; no coordination):")
        tab = pd.DataFrame(rows)
        cols_ = [("concordance_resid_full_auc", "Δ|m+cov")]
        if batch_oh is not None:
            cols_.append(("resid_full_batch_auc", "+batch"))
        for gk in dict.fromkeys(tab["granularity"]):
            t = tab[tab["granularity"] == gk]
            real = t[t["data"] == "real"]
            nul = t[t["data"] == "drawn_null"]
            if real.empty or nul.empty:
                continue
            parts = []
            for col, name in cols_:
                r = float(real[col].iloc[0])
                v = nul[col].astype(float).to_numpy()
                v = v[np.isfinite(v)]
                if not v.size or not np.isfinite(r):
                    continue
                p_ = (1 + int(np.sum(v >= r))) / (v.size + 1)
                sd = float(np.std(v, ddof=1)) if v.size > 1 else float("nan")
                parts.append(f"{name} real {r:.3f} vs null {v.mean():.3f}"
                             f" ± {sd:.3f} (max {v.max():.3f}; rank p = "
                             f"{p_:.3f}, {v.size} draws)")
            print(f"  {gk:>6}  " + "  |  ".join(parts))
        print("  real clearly above null  => coordination beyond what depth,")
        print("            coverage and methylation level give on their own;")
        print("  real ≈ null              => the AUC is what the ladder finds")
        print("            WITHOUT coordination (e.g. the level, through Δ's")
        print("            spread at 1-3 pairs per region).")
    print("\n  Δ|m+cov stays high (p<0.05) AND clearly > miss  => REAL")
    print("            marginal-, coverage- and presence-independent coordination.")
    print("  Δ|m+cov -> its null          => nonlinear-marginal leak or a")
    print("            coverage/sparsity artifact.")
    print("  Δ|m+cov ≈ miss (both high)    => the 'signal' is just contact")
    print("            PRESENCE (which regions a cell has data in; scHi-C-like),")
    print("            NOT epigenetic concordance.  Definitive test: held-out")
    print("            genomic split (train even chr, test odd chr).")
    if args.output_prefix:
        out = f"{args.output_prefix}.ep_existence.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n[ep-existence] wrote {out}")

        # ---- ROC curves (Δ vs Δ|m vs Δ|m+cov, per granularity) ----------
        try:
            roc_out = f"{args.output_prefix}.roc.pdf"
            _plot_roc(roc_data, method, roc_out)
            print(f"[ep-existence] wrote {roc_out}")
        except Exception as e:                       # pragma: no cover
            print(f"[ep-existence] ROC plot skipped ({e})")

        # ---- Distance-decay (Hi-C-style, 3 panels): agreement, signed
        # log2(O/E), and residual O−E vs distance.  Curves are POOLED over all
        # reads (Σsum/Σcnt → smooth); 95% CI = ±1.96·SE across CELLS.  An
        # "all cells" line (pooled over every cell) is added to each panel.
        nb = len(dbins) - 1
        mid = np.sqrt(dbins[:-1] * dbins[1:])
        groups = y_all.to_numpy()
        eps = 1e-6
        _l2 = lambda x: np.log2(np.clip(x, eps, None))

        def _rate(s, c):
            return np.divide(s, c, out=np.full(nb, np.nan), where=c > 0)

        def _metrics(oo, oc, eo, ec, ocnt, ecnt):
            """(agreement, signed log2(O/E), residual O−E) from open/closed
            sums + counts."""
            o_op, o_cl = _rate(oo, ocnt), _rate(oc, ocnt)
            e_op, e_cl = _rate(eo, ecnt), _rate(ec, ecnt)
            agree = o_op + o_cl
            signed = (_l2(o_op) - _l2(e_op)) - (_l2(o_cl) - _l2(e_cl))
            resid = (o_op + o_cl) - (e_op + e_cl)
            return agree, signed, resid

        def _pooled_metrics(decs):
            s = lambda k: np.sum([d[k] for d in decs], axis=0)
            return _metrics(s("O_open_sum"), s("O_closed_sum"),
                            s("E_open_sum"), s("E_closed_sum"),
                            s("O_cnt"), s("E_cnt"))

        def _cell_metrics(d):
            return _metrics(d["O_open_sum"], d["O_closed_sum"],
                            d["E_open_sum"], d["E_closed_sum"],
                            d["O_cnt"], d["E_cnt"])

        def _se(arrs):
            A = np.vstack(arrs)
            n = np.sum(np.isfinite(A), axis=0)
            with np.errstate(invalid="ignore"):
                sd = np.nanstd(A, axis=0, ddof=1)
            return np.where(n >= 2, sd / np.sqrt(np.maximum(n, 1)), np.nan)

        decs_by_g: dict = {}; decs_all = []
        for (_, _, dec), g in zip(results, groups):
            if dec is None:
                continue
            decs_all.append(dec)
            if isinstance(g, str):
                decs_by_g.setdefault(g, []).append(dec)
        if decs_all:
            groups_plot = dict(sorted(decs_by_g.items()))
            groups_plot["all cells"] = decs_all
            curves = {}
            for g, decs in groups_plot.items():
                ag, sg, rs = _pooled_metrics(decs)
                cm = [_cell_metrics(d) for d in decs]
                curves[g] = dict(
                    agree_mean=ag, agree_se=_se([c[0] for c in cm]),
                    signed_mean=sg, signed_se=_se([c[1] for c in cm]),
                    resid_mean=rs, resid_se=_se([c[2] for c in cm]))
            # values CSV (x = dist_mid_bp; y = the 3 metrics + 95% CI per group)
            dd = pd.DataFrame({
                "dist_lo_bp": dbins[:-1].astype(np.int64),
                "dist_hi_bp": dbins[1:].astype(np.int64),
                "dist_mid_bp": mid.astype(np.int64),
                "log10_dist_mid": np.log10(mid),
            })
            for g in sorted(curves):
                c = curves[g]; gs = g.replace(" ", "_")
                for key in ("agree", "signed", "resid"):
                    dd[f"{key}_{gs}_mean"] = c[f"{key}_mean"]
                    dd[f"{key}_{gs}_ci_lo"] = c[f"{key}_mean"] - 1.96 * c[f"{key}_se"]
                    dd[f"{key}_{gs}_ci_hi"] = c[f"{key}_mean"] + 1.96 * c[f"{key}_se"]
            dd_csv = f"{args.output_prefix}.distance_decay.values.csv"
            dd.to_csv(dd_csv, index=False)
            print(f"[ep-existence] wrote {dd_csv}")
            try:
                dd_out = f"{args.output_prefix}.distance_decay.pdf"
                _plot_distance_decay(mid, curves, dbins, dd_out)
                print(f"[ep-existence] wrote {dd_out}")
            except Exception as e:                   # pragma: no cover
                print(f"[ep-existence] distance-decay plot skipped ({e})")


if __name__ == "__main__":
    main()
