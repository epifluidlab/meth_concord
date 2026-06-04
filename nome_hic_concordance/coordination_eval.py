"""Shared evaluation helpers for long-range coordination existence tests.

Channel-agnostic numerics used by both the single-context existence test
(``scripts/ep_coordination_existence.py``) and the joint GCH+HCG test
(``scripts/joint_coordination_existence.py``): standardization, the confound
ladder (marginal / marginal+coverage / batch residualization), the classifier
panel (full-feature L2-LR, SVD→LR, random forest), cross-validated AUC with a
label-permutation null, and ROC plotting.

Each function takes an already-built ``cells × features`` matrix, so it knows
nothing about how many co-state channels produced those features — the joint
test simply hands it a wider matrix (regions × 4 channels) than the
single-context test (regions × 1).  Keeping them here means the joint build
reuses the exact, validated confound-ladder logic rather than a re-derivation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Standardization / reduction
# --------------------------------------------------------------------------

def standardize(M: pd.DataFrame) -> np.ndarray:
    """NaN-impute (column mean) then z-score a ``cells × features`` matrix."""
    X = M.to_numpy(np.float64)
    mu = np.nanmean(X, axis=0, keepdims=True)
    X = np.where(np.isnan(X), mu, X)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return np.nan_to_num((X - mu) / sd, nan=0.0)


def svd_reduce(X: np.ndarray, n_svd: int) -> np.ndarray:
    """Truncated-SVD reduction so the classifier + permutations are fast and not
    overfit on thousands of region features (matches the comparison harness)."""
    if n_svd <= 0 or X.shape[1] <= n_svd:
        return X
    from sklearn.decomposition import TruncatedSVD
    k = max(1, min(n_svd, X.shape[0] - 1, X.shape[1]))
    return TruncatedSVD(n_components=k, random_state=0).fit_transform(X)


def prep(Xstd: np.ndarray, method: str, n_svd: int) -> np.ndarray:
    """SVD only for ``svd-lr``; ``lr``/``rf`` classify the FULL standardized
    features so the (unsupervised) SVD can't discard low-variance label signal."""
    return svd_reduce(Xstd, n_svd) if method == "svd-lr" else Xstd


def zcols(X: np.ndarray) -> np.ndarray:
    """Per-column z-score; constant columns (std≈0) → all zeros (no influence)."""
    Xc = np.nan_to_num(X - np.nanmean(X, axis=0, keepdims=True))
    sd = Xc.std(axis=0, keepdims=True)
    return np.where(sd < 1e-12, 0.0, Xc / np.where(sd < 1e-12, 1.0, sd))


# --------------------------------------------------------------------------
# Confound ladder — residualization
# --------------------------------------------------------------------------

def residualize(D: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Per-region (per-column) residual of Δ (D) on the marginal (M).

    The O/E controls the marginal at the pair level, but imperfectly — and when
    the marginal AUC is near 1.0 even a small residual correlation makes Δ
    classify cell type.  This removes the linear Δ–marginal relationship in each
    region, so a surviving AUC is marginal-INDEPENDENT coordination.

    Done in standardized space so the coefficient is a bounded per-column
    correlation b∈[-1,1] (robust when a region's marginal has tiny variance — a
    plain cov/var would blow up and destroy real signal there)."""
    Ds, Ms = zcols(D), zcols(M)
    b = (Ds * Ms).mean(axis=0)            # per-column correlation, in [-1, 1]
    return Ds - b[None, :] * Ms


def residualize_multi(D: np.ndarray, covars: list) -> np.ndarray:
    """Per-region residual of Δ on MULTIPLE covariates (each cells × regions).

    Regress each region's Δ across cells on [marginal, marginal², coverage] and
    keep the residual: removes nonlinear marginal dependence (Δ=log2(O/E) is
    nonlinear in the mean-openness marginal) and the coverage/sparsity confound.
    Vectorized per-column OLS in standardized space (ridge 1e-6)."""
    Ds = zcols(D)
    Cs = [zcols(C) for C in covars]
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


def residualize_against(R: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Remove the projection of R (cells × regions) onto the column space of a
    PER-CELL design B (cells × k, e.g. batch one-hot — same for every region).

    A single least-squares projection (B is shared across regions), so a
    surviving AUC is batch-independent coordination."""
    Bz = zcols(B)
    if Bz.shape[1] == 0 or not np.isfinite(Bz).any():
        return R
    beta, *_ = np.linalg.lstsq(Bz, R, rcond=None)        # k × regions
    return R - Bz @ beta


# --------------------------------------------------------------------------
# Classifiers + cross-validated AUC
# --------------------------------------------------------------------------

def make_clf(method: str, seed: int):
    if method == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                      n_jobs=1, random_state=seed)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="ovr",
                              C=1.0)


def cv_rf(X: np.ndarray, y: np.ndarray, n_folds: int, seed: int):
    """5-fold CV AUC with a random forest (binary → AUC, multiclass → OvR-macro).

    Tree ensembles weight any feature regardless of its variance, so — unlike
    SVD→LR — they recover label signal that lives in low-variance directions
    (the concern when classifying the marginal-residualized Δ)."""
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


def score(X: np.ndarray, y: np.ndarray, method: str, n_folds: int, seed: int):
    """Unified (mean, std) CV AUC: random forest for ``rf``, else OvR L2-LR."""
    if method == "rf":
        return cv_rf(X, y, n_folds, seed)
    from nome_hic_concordance.pair_concordance.component_analysis import _cv_auc
    return _cv_auc(X, y, n_folds=n_folds, random_state=seed)


def auc_perm(X: np.ndarray, y: np.ndarray, method: str, n_folds: int, seed: int,
             n_perm: int, n_threads: int):
    """(mean, std, perm_p) — CV AUC plus a label-permutation null p-value.

    The permutation null calibrates the AUC against the actual chance ceiling
    for this (n cells, class balance, feature count) — essential because with
    p≫n a high AUC can arise by chance.  p = (#{null ≥ observed} + 1)/(M + 1)."""
    mean, std = score(X, y, method, n_folds, seed)
    p = np.nan
    if n_perm > 0 and np.isfinite(mean):
        prng = np.random.default_rng(seed)
        yperms = [prng.permutation(y) for _ in range(int(n_perm))]
        if n_threads > 1:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=n_threads)(
                delayed(score)(X, yp, method, n_folds, seed) for yp in yperms)
            null = np.array([m for m, _ in res])
        else:
            null = np.array([score(X, yp, method, n_folds, seed)[0]
                             for yp in yperms])
        null = null[np.isfinite(null)]
        if null.size:
            p = float((np.sum(null >= mean) + 1) / (null.size + 1))
    return mean, std, p


def cv_predict(X: np.ndarray, y: np.ndarray, method: str, n_folds: int, seed: int):
    """Pooled out-of-fold predictions for an ROC curve.

    Binary → (y∈{0,1}, positive-class score).  Multiclass → micro-averaged OvR:
    (one-hot label flattened, predicted-proba flattened).  None if CV isn't
    possible."""
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
        sc = np.full(len(yy), np.nan)
        for tr, te in skf.split(X, yy):
            clf = make_clf(method, seed).fit(X[tr], yy[tr])
            sc[te] = clf.predict_proba(X[te])[:, 1]
        ok = np.isfinite(sc)
        return yy[ok].astype(float), sc[ok]
    Yb = label_binarize(yy, classes=list(range(nc)))
    P = np.full((len(yy), nc), np.nan)
    for tr, te in skf.split(X, yy):
        clf = make_clf(method, seed).fit(X[tr], yy[tr])
        P[te] = clf.predict_proba(X[te])
    ok = np.isfinite(P).all(axis=1)
    return Yb[ok].ravel().astype(float), P[ok].ravel()


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def decade_fmt(x: float, pos=None) -> str:
    """Tick label = log10 exponent + human distance, e.g. '5 (100kb)'."""
    e = int(round(np.log10(x)))
    human = {3: "1kb", 4: "10kb", 5: "100kb", 6: "1Mb", 7: "10Mb",
             8: "100Mb"}.get(e, f"1e{e}")
    return f"{e} ({human})"


def decade_ticks(ax):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(3, 9)]))
    ax.xaxis.set_major_formatter(FuncFormatter(decade_fmt))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel("log10(distance between ends)  —  exponent (distance)")
