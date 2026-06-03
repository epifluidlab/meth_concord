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
                 min_gch: int, cap: int, seed: int, dbins: np.ndarray = None):
    """Per-(region) Δ_signed (log2 O/E) + mean openness for one cell, at each
    granularity.  Fully vectorized.  Returns (cell, {gran: {region:(Δ,marg,n_o)}},
    decay) where ``decay`` holds distance-binned soft-agreement sums/counts for
    contacts (O) and the random null (E) — for the Hi-C-style distance plot."""
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
    for ci, c in enumerate(uniq_chroms):
        sel = e_chrom == c
        lp, eo, ec, ed = _window_pairs(e_pos[sel], e_p[sel], min_d, max_d, cap, rng)
        if lp.size:
            ep_pos_l.append(lp); ep_open_l.append(eo); ep_closed_l.append(ec)
            ep_code_l.append(np.full(lp.size, ci, np.int64)); ep_dist_l.append(ed)
    if ep_pos_l:
        e_lpos = np.concatenate(ep_pos_l); e_open = np.concatenate(ep_open_l)
        e_closed = np.concatenate(ep_closed_l); e_codes = np.concatenate(ep_code_l)
        e_dist = np.concatenate(ep_dist_l)
    else:
        e_lpos = np.empty(0, np.int64); e_open = np.empty(0)
        e_closed = np.empty(0); e_codes = np.empty(0, np.int64); e_dist = np.empty(0)
    m_codes = code_of(e_chrom)

    # Distance-decay accumulators (Hi-C-style): soft agreement (= co-open +
    # co-closed) summed over log-distance bins, for contacts (O) and null (E).
    decay = None
    if dbins is not None:
        nb = len(dbins) - 1
        def _dacc(dist, agree):
            if dist.size == 0:
                return np.zeros(nb), np.zeros(nb)
            bi = np.clip(np.searchsorted(dbins, dist, side="right") - 1, 0, nb - 1)
            return (np.bincount(bi, weights=agree, minlength=nb),
                    np.bincount(bi, minlength=nb).astype(float))
        o_s, o_c = _dacc(sep[cm], o_open + o_closed)
        e_s, e_c = _dacc(e_dist, e_open + e_closed)
        decay = {"O_sum": o_s, "O_cnt": o_c, "E_sum": e_s, "E_cnt": e_c}

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
    return cell, out, decay


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


def _cv_rf(X: np.ndarray, y: np.ndarray, n_folds: int, seed: int):
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
                                     n_jobs=1, random_state=seed)
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


def _make_clf(method: str, seed: int):
    if method == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                      n_jobs=1, random_state=seed)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="ovr",
                              C=1.0)


def _cv_predict(X: np.ndarray, y: np.ndarray, method: str, n_folds: int, seed: int):
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
            clf = _make_clf(method, seed).fit(X[tr], yy[tr])
            score[te] = clf.predict_proba(X[te])[:, 1]
        ok = np.isfinite(score)
        return yy[ok].astype(float), score[ok]
    Yb = label_binarize(yy, classes=list(range(nc)))
    P = np.full((len(yy), nc), np.nan)
    for tr, te in skf.split(X, yy):
        clf = _make_clf(method, seed).fit(X[tr], yy[tr])
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


def _plot_distance_decay(dbins: np.ndarray, per_group: dict, pooled: dict,
                         output: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mid = np.sqrt(dbins[:-1] * dbins[1:])         # geometric mid of each log bin

    def rate(d, key):
        s, c = d[key + "_sum"], d[key + "_cnt"]
        return np.divide(s, c, out=np.full(len(s), np.nan), where=c > 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for g, d in sorted(per_group.items()):
        ax1.plot(mid, rate(d, "O"), "-o", ms=3, lw=1.3, label=f"contacts (O) — {g}")
    ax1.plot(mid, rate(pooled, "E"), "k--", lw=1.3,
             label="distance-matched random (E)")
    ax1.set_xscale("log"); ax1.set_ylabel("concordance  (soft agreement)")
    ax1.set_title("Concordance vs genomic distance (Hi-C-style)")
    ax1.legend(fontsize=8)
    for g, d in sorted(per_group.items()):
        ax2.plot(mid, rate(d, "O") - rate(d, "E"), "-o", ms=3, lw=1.3,
                 label=f"O − E — {g}")
    ax2.axhline(0.0, color="grey", lw=0.8)
    ax2.set_xscale("log"); ax2.set_xlabel("genomic distance between ends (bp, log)")
    ax2.set_ylabel("residual concordance  (O − E)")
    ax2.set_title("Residual concordance (contacts vs distance-matched random)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


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
    feat = (f"SVD top-{args.n_svd}" if args.classifier == "svd-lr"
            else "ALL region features (no SVD)")
    print(f"[ep-existence] {len(files)} cells | band "
          f"{args.min_distance:,}–{args.max_distance:,} bp | "
          f"{n_classes} cell types | grans={grans} | "
          f"classifier={args.classifier} → {feat}")

    # Log-spaced distance bins for the Hi-C-style distance-decay plot.
    dbins = np.logspace(np.log10(max(args.min_distance, 1)),
                        np.log10(max(args.max_distance, args.min_distance + 1)), 31)

    if args.n_threads > 1 and len(files) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(f, grans, args.min_distance, args.max_distance,
                                   args.min_gch_sites, args.max_ends_per_chrom,
                                   args.random_seed + i, dbins)
            for i, f in enumerate(files))
    else:
        results = [process_cell(f, grans, args.min_distance, args.max_distance,
                                args.min_gch_sites, args.max_ends_per_chrom,
                                args.random_seed + i, dbins)
                   for i, f in enumerate(files)]
    results = [(c, d, dec) for c, d, dec in results if d]
    cells = [c for c, _, _ in results]
    y_all = labels.reindex(cells)
    keep = y_all.notna().to_numpy()
    if keep.sum() < 10:
        sys.exit("Too few labeled cells.")

    y = y_all[keep].to_numpy()
    method = args.classifier

    def _score(X, yy):
        if method == "rf":
            return _cv_rf(X, yy, args.n_folds, args.random_seed)
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
    for gran in grans:
        gk = str(gran)
        all_regions = sorted({r for _, d, _ in results for r in d.get(gk, {})})
        if not all_regions:
            continue
        ridx = {r: j for j, r in enumerate(all_regions)}
        D = np.full((len(cells), len(all_regions)), np.nan)
        Mg = np.full((len(cells), len(all_regions)), np.nan)
        Cov = np.full((len(cells), len(all_regions)), np.nan)
        for i, (_, d, _) in enumerate(results):
            for r, (delta, marg, n_o) in d.get(gk, {}).items():
                j = ridx[r]
                D[i, j] = delta; Mg[i, j] = marg; Cov[i, j] = n_o
        # Region-observation filter: drop sparsely-observed regions so the
        # classifier can't exploit the missingness/imputation pattern.
        if args.min_region_frac > 0:
            of = np.isfinite(D[keep]).mean(axis=0)
            cols = of >= args.min_region_frac
            if int(cols.sum()) < 2:
                print(f"{gk:>11} {int(cols.sum()):>6}  (too few regions after "
                      f"--min-region-frac {args.min_region_frac})")
                continue
            D, Mg, Cov = D[:, cols], Mg[:, cols], Cov[:, cols]
        n_reg = D.shape[1]
        covk_obs = Cov[keep][np.isfinite(Cov[keep])]
        med = int(np.median(covk_obs)) if covk_obs.size else 0
        Dk, Mk = D[keep], Mg[keep]
        Covk = np.log1p(np.nan_to_num(Cov[keep], nan=0.0))
        # Missingness mask: which (cell, region) have a Δ at all (contact
        # PRESENCE, scHi-C-like).  A reference benchmark: if the residual AUC
        # ≈ the missingness AUC, the "signal" is just which regions a cell has
        # contacts in (coverage/contact-map), NOT epigenetic concordance.
        Miss = np.isfinite(Dk).astype(np.float64)
        Xc = _prep(_standardize(pd.DataFrame(Dk)))
        Xm = _prep(_standardize(pd.DataFrame(Mk)))
        Xmiss = _prep(_standardize(pd.DataFrame(Miss)))
        # Leakage controls: linear marginal, then marginal+marginal²+coverage.
        Xr = _prep(_standardize(pd.DataFrame(_residualize(Dk, Mk))))
        Xr2 = _prep(_standardize(pd.DataFrame(
            _residualize_multi(Dk, [Mk, Mk * Mk, Covk]))))
        cmean, cstd, cp = _auc_perm(Xc)
        rmean, rstd, rp = _auc_perm(Xr)
        r2mean, r2std, r2p = _auc_perm(Xr2)
        mmean, mstd = _score(Xm, y)
        missmean, _ = _score(Xmiss, y)
        if args.output_prefix:           # collect out-of-fold preds for ROC
            roc_data[gk] = {
                "Δ": _cv_predict(Xc, y, method, args.n_folds, args.random_seed),
                "Δ|m": _cv_predict(Xr, y, method, args.n_folds, args.random_seed),
                "Δ|m+cov": _cv_predict(Xr2, y, method, args.n_folds,
                                       args.random_seed)}
        print(f"{gk:>11} {n_reg:>6} {med:>9} "
              f"{cmean:>7.3f} {cp:>6.3f} {rmean:>7.3f} {rp:>6.3f} "
              f"{r2mean:>7.3f} {r2p:>6.3f} {missmean:>6.3f} {mmean:>6.3f}")
        rows.append({"classifier": method,
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
                     "missingness_auc": missmean,
                     "marginal_auc": mmean, "marginal_auc_std": mstd})

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

        # ---- Distance-decay (Hi-C-style): concordance & residual vs dist --
        keys = ("O_sum", "O_cnt", "E_sum", "E_cnt")
        nb = len(dbins) - 1
        pooled = {k: np.zeros(nb) for k in keys}
        per_group: dict = {}
        groups = y_all.to_numpy()
        for (_, _, dec), g in zip(results, groups):
            if dec is None:
                continue
            for k in keys:
                pooled[k] += dec[k]
            if isinstance(g, str):                   # valid cell-type label
                pg = per_group.setdefault(g, {k: np.zeros(nb) for k in keys})
                for k in keys:
                    pg[k] += dec[k]
        if per_group:
            try:
                dd_out = f"{args.output_prefix}.distance_decay.pdf"
                _plot_distance_decay(dbins, per_group, pooled, dd_out)
                print(f"[ep-existence] wrote {dd_out}")
            except Exception as e:                   # pragma: no cover
                print(f"[ep-existence] distance-decay plot skipped ({e})")


if __name__ == "__main__":
    main()
