#!/usr/bin/env python
"""Which way of filling sparse per-cell coordination separates cell types --
and does its drawn null?

Reads step 5's raw per-(cell, bin) counts (``05_concordance/counts/``, written
by ``--emit-counts``; the null's by ``--emit-null``) and fills the sparse
cells x bins coordination matrix C = O - E several ways.  Every method is run
on the real counts and, separately, on each drawn null's counts (step 5 writes
``--null-draws`` of them) -- the same cells, cell-bins and expected, with
pairs that are not contacts.  Each result is scored the way
celltype_metric_barplots.py scores a bar (kNN purity, ARI, NMI on SVD scores)
plus a cross-validated classifier AUC.  The nulls are summarised as a mean, a
spread and a max: two draws of the same data differ by chance, so one null
draw is too noisy a reference.

WHY THE NULL COLUMN IS THE RESULT.  A method that borrows across cells can
put cell-type structure into ANY matrix, coordination or not: RWR over the
pipeline's methylation graph scored the same purity for C and for its null.
So a method is better only if the real counts separate cell types AND its
null does not.  Read the "real - null" gap, never the real column alone.  C itself carries
some of the level even with no coordination -- at one contact it takes one of
two values that depend on the level -- so even an un-imputed null can sit
above chance.  Only the comparison with the null removes that.

Methods (``--methods``, comma-separated):

  none          Per-bin C from the raw counts; a cell-bin with no contact
                stays missing (0 after the shared prep).  Nothing borrowed.
  pool:<size>   Per cell, the counts summed over consecutive bins into
                regions of <size> (pool:5mb, pool:10mb, pool:2500kb), then C
                on the pooled counts.  Pairs are pooled within a cell, never
                across cells -- the design-doc 6b approach.
  class         The same, pooled over ``--bin-classes`` (bin -> class),
                which must be defined without the cell labels.
  lowrank       A binomial low-rank model of each cell-bin's concordant
                count, with E as the offset, fitted on observed cell-bins
                only:  logit P = logit E + a_cell + c_bin + U_cell . V_bin.
                The bin loadings V are learned from all cells together, so a
                cell needs only its own observed bins to be placed -- not
                bins in common with other cells, which is what a k-NN graph
                needs and sparse data cannot give.  Output: a_cell +
                U_cell . V_bin at every cell-bin.
  rwr-c         RWR over a k-NN graph built from the raw C itself (what
                --rwr-init coordination does at pass 0).
  rwr-acc       RWR over a k-NN graph built from ``--accessibility`` (the
                pipeline's default pass-0 graph, without its batch
                correction).  The reference for the leak.

For lowrank and rwr-c the null gets its own fit / its own graph.  rwr-acc's
graph is the same for both, as the pipeline's pass 0 is.

Example (2.5 Mb HCG):

    python scripts/imputation_benchmark.py \\
        --counts-dir output/final_25mb/context_HCG/05_concordance/counts \\
        --accessibility output/final_25mb/context_HCG/05_concordance/accessibility_mean.parquet \\
        --samples mouse_cortex_1577_cells.samples.with_batches.fixed.tsv \\
        --batch-column exp_batch --batch-correction harmony \\
        --methods none,pool:10mb,pool:25mb,lowrank,rwr-c,rwr-acc \\
        --matrix-norm none --n-svd 5 --exclude-chrom chrM,chrX,chrY \\
        --output-dir output/final_25mb/10_imputation_benchmark_HCG
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nome_hic_concordance import comparison  # noqa: E402
from nome_hic_concordance import metrics as _metrics  # noqa: E402

logger = logging.getLogger("imputation_benchmark")

_O = ("k_open", "k_closed", "k_mixed", "n_linked")
_E = ("k_dm_open", "k_dm_closed", "k_dm_mixed", "n_dm")
METRICS = ("auc", "knn_purity", "ari", "nmi")


# ------------------------------------------------------------------- inputs --

def _null_keys(draw: int) -> tuple[str, ...]:
    from nome_hic_concordance.concordance import _null_keys as nk
    return nk(int(draw))


def _drop_chroms(df: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    if not exclude:
        return df
    keep = [c for c in df.columns if str(c).split(":", 1)[0] not in exclude]
    return df[keep]


def available_draws(counts_dir) -> list[int]:
    """The drawn-null replicates step 5 wrote into ``counts_dir``."""
    d, out = Path(counts_dir), []
    while (d / f"{_null_keys(len(out))[0]}.parquet").exists():
        out.append(len(out))
    return out


def load_counts(counts_dir, draws=(0,), exclude=()):
    """(real, {draw: null}) count dicts, cells x bins.  Each null takes its O
    keys from its draw and shares E -- as step 5's own null does."""
    d = Path(counts_dir)
    exclude = set(exclude)

    def rd(key):
        p = d / f"{key}.parquet"
        if not p.exists():
            raise SystemExit(
                f"{p} not found.  Step 5 writes these with --emit-counts, "
                f"and the null's with --emit-null as well.")
        return _drop_chroms(pd.read_parquet(p).astype(float), exclude)

    real = {k: rd(k) for k in _O + _E}
    nulls = {}
    for draw in draws:
        nulls[int(draw)] = {o: rd(k) for o, k in zip(_O, _null_keys(draw))}
        nulls[int(draw)].update({k: real[k] for k in _E})
    cells, bins = real["n_linked"].index, real["n_linked"].columns
    for dct in (real, *nulls.values()):
        for k in list(dct):
            dct[k] = dct[k].reindex(index=cells, columns=bins).fillna(0.0)
    return real, nulls


# --------------------------------------------------------------- statistic --

def coordination(cnt: dict, observed_only: bool = True) -> pd.DataFrame:
    """The pipeline's MLE coordination, (O_open + O_closed) - (E_open +
    E_closed).  ``observed_only`` leaves a cell-bin with no contact (or no
    expected) missing instead of reading it as 0."""
    n = cnt["n_linked"].to_numpy(float)
    ne = cnt["n_dm"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        o = (cnt["k_open"].to_numpy(float)
             + cnt["k_closed"].to_numpy(float)) / n
        e = (cnt["k_dm_open"].to_numpy(float)
             + cnt["k_dm_closed"].to_numpy(float)) / ne
    ok = (n > 0) & (ne > 0)
    c = np.where(ok, o - e, np.nan if observed_only else 0.0)
    return pd.DataFrame(c, index=cnt["n_linked"].index,
                        columns=cnt["n_linked"].columns)


# ----------------------------------------------------------------- methods --

def parse_size(text: str) -> int:
    m = re.fullmatch(r"\s*([0-9.]+)\s*(bp|kb|mb)?\s*", str(text).lower())
    if not m:
        raise SystemExit(f"cannot read a region size from {text!r}")
    mult = {"bp": 1, None: 1, "kb": 1_000, "mb": 1_000_000}[m.group(2)]
    return int(round(float(m.group(1)) * mult))


def region_groups(bins, size: int) -> pd.Series:
    """bin 'chr:start' -> its region 'chr:start' of ``size`` bp."""
    out = {}
    for b in bins:
        chrom, start = str(b).rsplit(":", 1)
        out[b] = f"{chrom}:{int(start) // size * size}"
    return pd.Series(out)


def pool(cnt: dict, groups: pd.Series) -> dict:
    """Each cell's counts summed over the bins of each group."""
    g = groups.reindex(cnt["n_linked"].columns)
    keep = g.notna().to_numpy()
    keys = g[keep].to_numpy()
    return {k: v.loc[:, keep].T.groupby(keys, sort=True).sum().T
            for k, v in cnt.items()}


def lowrank(cnt: dict, rank: int = 10, l2: float = 1.0,
            max_iter: int = 500, seed: int = 0):
    """Binomial low-rank fit; returns (a + U V^T as a DataFrame, info)."""
    import scipy.sparse as sp
    from scipy.optimize import minimize
    from scipy.special import expit

    n = cnt["n_linked"].to_numpy(float)
    ne = cnt["n_dm"].to_numpy(float)
    obs = (n > 0) & (ne > 0)
    i, b = np.nonzero(obs)
    N, B, r = n.shape[0], n.shape[1], int(rank)
    if i.size == 0:
        raise ValueError("no observed cell-bin to fit")
    nn = n[obs]
    y = np.clip((cnt["k_open"].to_numpy(float)
                 + cnt["k_closed"].to_numpy(float))[obs], 0.0, nn)
    e = ((cnt["k_dm_open"].to_numpy(float)
          + cnt["k_dm_closed"].to_numpy(float))[obs] / ne[obs])
    e = np.clip(e, 1e-4, 1.0 - 1e-4)
    off = np.log(e) - np.log1p(-e)

    def unpack(x):
        a = x[:N]
        c = x[N:N + B]
        U = x[N + B:N + B + N * r].reshape(N, r)
        V = x[N + B + N * r:].reshape(B, r)
        return a, c, U, V

    def f(x):
        a, c, U, V = unpack(x)
        eta = off + a[i] + c[b] + np.einsum("ij,ij->i", U[i], V[b])
        nll = float(np.sum(y * np.logaddexp(0.0, -eta)
                           + (nn - y) * np.logaddexp(0.0, eta)))
        res = nn * expit(eta) - y
        R = sp.csr_matrix((res, (i, b)), shape=(N, B))
        g = np.concatenate([
            np.bincount(i, weights=res, minlength=N) + l2 * a,
            np.bincount(b, weights=res, minlength=B) + l2 * c,
            (R @ V + l2 * U).ravel(),
            (R.T @ U + l2 * V).ravel()])
        reg = 0.5 * l2 * float(a @ a + c @ c + np.sum(U * U) + np.sum(V * V))
        return nll + reg, g

    rng = np.random.default_rng(seed)
    x0 = np.concatenate([np.zeros(N + B),
                         0.01 * rng.standard_normal((N + B) * r)])
    out = minimize(f, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": int(max_iter)})
    a, _c, U, V = unpack(out.x)
    Z = a[:, None] + U @ V.T
    info = {"converged": bool(out.success), "n_iter": int(out.nit),
            "n_obs": int(i.size)}
    return (pd.DataFrame(Z, index=cnt["n_linked"].index,
                         columns=cnt["n_linked"].columns), info)


def rwr(cnt: dict, features: pd.DataFrame, k: int = 15,
        restart: float = 0.5) -> pd.DataFrame:
    """C from counts imputed over a k-NN graph built from ``features``."""
    from nome_hic_concordance.imputation import (
        build_knn_graph, quantile_normalize_for_knn, rwr_impute_matrices)
    ref = cnt["n_linked"]
    X = features.reindex(index=ref.index).to_numpy(float)
    _, P = build_knn_graph(quantile_normalize_for_knn(X), k=k)
    imp, _ = rwr_impute_matrices({key: v.to_numpy(float)
                                  for key, v in cnt.items()},
                                 P, restart_prob=restart)
    return coordination({key: pd.DataFrame(v, index=ref.index,
                                           columns=ref.columns)
                         for key, v in imp.items()})


def run_method(spec: str, cnt: dict, args, accessibility=None,
               bin_classes=None):
    """One method on one count dict -> (cells x features matrix, info)."""
    if spec == "none":
        return coordination(cnt), {}
    if spec.startswith("pool:"):
        size = parse_size(spec.split(":", 1)[1])
        pooled = pool(cnt, region_groups(cnt["n_linked"].columns, size))
        return coordination(pooled), {"n_features": pooled["n_linked"].shape[1]}
    if spec == "class":
        if bin_classes is None:
            raise SystemExit("method 'class' needs --bin-classes")
        pooled = pool(cnt, bin_classes)
        return coordination(pooled), {"n_features": pooled["n_linked"].shape[1]}
    if spec == "lowrank":
        return lowrank(cnt, rank=args.rank, l2=args.lowrank_l2,
                       max_iter=args.lowrank_max_iter, seed=args.random_seed)
    if spec == "rwr-c":
        return rwr(cnt, coordination(cnt), k=args.rwr_k,
                   restart=args.rwr_restart), {}
    if spec == "rwr-acc":
        if accessibility is None:
            raise SystemExit("method 'rwr-acc' needs --accessibility")
        return rwr(cnt, accessibility, k=args.rwr_k,
                   restart=args.rwr_restart), {}
    raise SystemExit(f"unknown method {spec!r}")


# ------------------------------------------------------------------ scoring --

def _batch_design(batch: pd.DataFrame | None, cells) -> np.ndarray | None:
    """Additive one-hot of every batch column, cells in ``cells`` order."""
    if batch is None:
        return None
    parts = [pd.get_dummies(batch[c].reindex(cells).astype(str),
                            drop_first=True).to_numpy(float)
             for c in batch.columns]
    D = np.hstack(parts) if parts else np.zeros((len(cells), 0))
    return D if D.shape[1] else None


def _cv_auc(X, y, classifier: str, n_folds: int, seed: int) -> float:
    """Out-of-fold OvR-macro AUC (binary: plain AUC) on all features."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    yy = LabelEncoder().fit_transform(y)
    nc = int(yy.max()) + 1
    folds = min(int(n_folds), int(np.bincount(yy).min()))
    if nc < 2 or folds < 2:
        return float("nan")
    if classifier == "rf":
        from sklearn.ensemble import RandomForestClassifier

        def make():
            return RandomForestClassifier(n_estimators=300,
                                          max_features="sqrt", n_jobs=1,
                                          random_state=seed)
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        def make():
            return OneVsRestClassifier(LogisticRegression(
                max_iter=2000, C=1.0, solver="lbfgs"))
    P = np.zeros((len(yy), nc))
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=seed).split(X, yy):
        P[te] = make().fit(X[tr], yy[tr]).predict_proba(X[te])
    if nc == 2:
        return float(roc_auc_score(yy, P[:, 1]))
    return float(roc_auc_score(yy, P, multi_class="ovr", average="macro"))


def score(mat: pd.DataFrame, name: str, labels: pd.Series,
          batch: pd.DataFrame | None, args, out_dir) -> dict:
    """The barplot's scoring, plus a classifier AUC on the same features."""
    norm = comparison._prep_matrix(
        mat, name, str(out_dir), args.matrix_norm, args.min_cell_coverage,
        args.per_cell_norm, args.post_norm_recenter)
    common = norm.index.intersection(labels.index)
    out = {"n_features": float(norm.shape[1]),
           "observed_frac": float(np.isfinite(mat.to_numpy(float)).mean())}
    if norm.shape[1] < 2 or len(common) < 3:
        return {**out, "knn_chance": float("nan"),
                **{m: float("nan") for m in METRICS}}
    X = norm.loc[common]
    scores, _ = comparison._svd_scores(X, args.n_svd,
                                       skip_top=args.skip_top_svd,
                                       random_state=args.random_seed)
    if batch is not None and args.batch_correction == "harmony":
        from nome_hic_concordance import batch_correction as _bc
        scores = _bc.harmonize_embedding(
            scores, batch.iloc[:, 0].reindex(common).reset_index(drop=True),
            random_state=args.random_seed, theta=None,
            n_threads=args.n_threads)
    out.update(_metrics.label_agreement(
        scores, labels.loc[common], k=args.knn_k,
        random_state=args.random_seed, n_perm=args.n_perm))
    # AUC on ALL the features, not the SVD scores: 6b's lesson is that the
    # coordination signal can sit in low-variance directions an SVD drops.
    Z = X.to_numpy(float)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) < 1e-12, 1.0, Z.std(0))
    D = _batch_design(batch, common)
    if D is not None:                      # strip per-batch offsets
        D = D - D.mean(0)
        beta, *_ = np.linalg.lstsq(D, Z, rcond=None)
        Z = Z - D @ beta
    out["auc"] = _cv_auc(Z, labels.loc[common].to_numpy(), args.classifier,
                         args.n_folds, args.random_seed)
    return out


def summarise(tab: pd.DataFrame) -> pd.DataFrame:
    """Per method: the real value, the nulls' mean / sd / max, and the gap
    real - mean(null), also in units of the nulls' sd when there are two or
    more draws."""
    rows = []
    for meth in dict.fromkeys(tab["method"]):
        sub_ = tab[tab["method"] == meth]
        real = sub_[sub_["data"] == "real"]
        null = sub_[sub_["data"] != "real"]
        row = {"method": meth, "n_null_draws": len(null)}
        for m in METRICS:
            r = float(real[m].astype(float).mean())
            v = null[m].astype(float).to_numpy()
            mu = float(np.nanmean(v)) if v.size else float("nan")
            sd = (float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1
                  else float("nan"))
            row.update({f"{m}_real": r, f"{m}_null_mean": mu,
                        f"{m}_null_sd": sd,
                        f"{m}_null_max": (float(np.nanmax(v)) if v.size
                                          else float("nan")),
                        f"{m}_gap": r - mu,
                        f"{m}_gap_z": ((r - mu) / sd if sd and sd > 0
                                       else float("nan"))})
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


# --------------------------------------------------------------------- plot --

def plot(tab: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(dict.fromkeys(tab["method"]))
    ref = {"auc": 0.5, "ari": 0.0, "nmi": 0.0}
    fig, axes = plt.subplots(1, len(METRICS),
                             figsize=(3.2 * len(METRICS),
                                      2.2 + 0.25 * len(methods)))
    x = np.arange(len(methods))
    for ax, m in zip(axes, METRICS):
        real = [tab.loc[(tab.method == me) & (tab.data == "real"), m]
                .astype(float).mean() for me in methods]
        nv = [tab.loc[(tab.method == me) & (tab.data != "real"), m]
              .astype(float).to_numpy() for me in methods]
        mu = np.array([np.nanmean(v) if v.size else np.nan for v in nv])
        lo = np.array([mu[i] - np.nanmin(v) if v.size else 0.0
                       for i, v in enumerate(nv)])
        hi = np.array([np.nanmax(v) - mu[i] if v.size else 0.0
                       for i, v in enumerate(nv)])
        ax.bar(x - 0.19, real, 0.38, color="#2F6B9A", edgecolor="#555",
               linewidth=0.5, label="real")
        ax.bar(x + 0.19, mu, 0.38, color="#C9CED6", hatch="//",
               edgecolor="#555", linewidth=0.5,
               yerr=np.vstack([lo, hi]), capsize=2,
               error_kw={"elinewidth": 0.8}, label="drawn null (range)")
        line = (ref.get(m) if m != "knn_purity"
                else float(tab["knn_chance"].astype(float).mean()))
        if line is not None and np.isfinite(line):
            ax.axhline(line, color="#888", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
        ax.set_title({"auc": "classifier AUC", "knn_purity": "kNN purity",
                      "ari": "ARI", "nmi": "NMI"}[m], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Real coordination vs its drawn null, by imputation method "
                 "(dashed: chance)", fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------- main --

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts-dir", required=True,
                    help="step 5's 05_concordance/counts/ (--emit-counts).")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-column", default="cell_name")
    ap.add_argument("--group-column", default="group")
    ap.add_argument("--batch-column", default=None,
                    help="Comma-separated.  Stripped from the AUC features as "
                         "additive offsets; with --batch-correction harmony "
                         "the first column also corrects the SVD scores.")
    ap.add_argument("--batch-correction", choices=["none", "harmony"],
                    default="none")
    ap.add_argument("--methods",
                    default="none,pool:5mb,pool:10mb,lowrank,rwr-c")
    ap.add_argument("--null-draws", default="all",
                    help="Which drawn-null replicates to score: 'all' "
                         "(default, every one in --counts-dir) or a "
                         "comma-separated list such as 0,1.")
    ap.add_argument("--accessibility", default=None,
                    help="accessibility_mean.parquet, for rwr-acc.")
    ap.add_argument("--bin-classes", default=None,
                    help="TSV of bin<TAB>class, for 'class'.")
    ap.add_argument("--exclude-chrom", default="")
    ap.add_argument("--min-group-size", type=int, default=10,
                    help="Groups with fewer labelled cells are dropped.")
    ap.add_argument("--rank", type=int, default=10)
    ap.add_argument("--lowrank-l2", type=float, default=1.0)
    ap.add_argument("--lowrank-max-iter", type=int, default=500)
    ap.add_argument("--rwr-k", type=int, default=15)
    ap.add_argument("--rwr-restart", type=float, default=0.5)
    ap.add_argument("--matrix-norm", choices=["quantile", "zscore", "none"],
                    default="quantile")
    ap.add_argument("--per-cell-norm",
                    choices=["off", "center", "zscore", "rank"],
                    default="rank")
    ap.add_argument("--post-norm-recenter", action="store_true")
    ap.add_argument("--min-cell-coverage", type=float, default=0.0)
    ap.add_argument("--n-svd", type=int, default=10)
    ap.add_argument("--skip-top-svd", type=int, default=0)
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--n-perm", type=int, default=0,
                    help="Label permutations for the kNN-purity p-value.")
    ap.add_argument("--classifier", choices=["lr", "rf"], default="lr")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--plot-format", choices=["pdf", "png", "svg"],
                    default="pdf")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    excl = {c.strip() for c in args.exclude_chrom.split(",") if c.strip()}

    draws = (available_draws(args.counts_dir)
             if args.null_draws.strip().lower() == "all"
             else [int(x) for x in args.null_draws.split(",") if x.strip()])
    if not draws:
        raise SystemExit(f"no drawn null in {args.counts_dir}: run step 5 "
                         f"with --emit-null as well as --emit-counts.")
    real, nulls = load_counts(args.counts_dir, draws, excl)
    cells = real["n_linked"].index
    samples = pd.read_csv(args.samples, sep="\t").set_index(args.cell_column)
    labels = samples[args.group_column].reindex(cells).dropna()
    sizes = labels.value_counts()
    small = sizes[sizes < args.min_group_size]
    if len(small):
        logger.info(f"Dropping {len(small)} group(s) under "
                    f"{args.min_group_size} cells: {dict(small)}")
        labels = labels[~labels.isin(small.index)]
    batch = None
    if args.batch_column:
        cols = [c.strip() for c in args.batch_column.split(",") if c.strip()]
        missing = [c for c in cols if c not in samples.columns]
        if missing:
            raise SystemExit(f"--batch-column not in {args.samples}: "
                             f"{missing}")
        batch = samples[cols].reindex(cells)
    n_obs = (real["n_linked"].to_numpy() > 0).mean()
    logger.info(f"{len(cells)} cells, {real['n_linked'].shape[1]} bins, "
                f"{n_obs:.1%} of cell-bins with a contact; {len(labels)} "
                f"labelled cells in {labels.nunique()} groups; "
                f"{len(draws)} null draw(s).")

    accessibility = None
    if args.accessibility:
        accessibility = _drop_chroms(
            pd.read_parquet(args.accessibility).astype(float), excl
        ).reindex(index=cells, columns=real["n_linked"].columns)
    bin_classes = None
    if args.bin_classes:
        t = pd.read_csv(args.bin_classes, sep="\t", header=None,
                        names=["bin", "class"], dtype=str)
        bin_classes = t.set_index("bin")["class"]

    rows = []
    for spec in [m.strip() for m in args.methods.split(",") if m.strip()]:
        for data, cnt in (("real", real),
                          *((f"null{d}", nulls[d]) for d in draws)):
            t0 = time.time()
            mat, info = run_method(spec, cnt, args, accessibility,
                                   bin_classes)
            res = score(mat, f"{spec.replace(':', '_')}_{data}", labels,
                        batch, args, out_dir)
            rows.append({"method": spec, "data": data, **res,
                         **{f"info_{k}": v for k, v in info.items()},
                         "seconds": round(time.time() - t0, 1)})
            logger.info(f"  {spec:12s} {data:4s}  AUC={res['auc']:.3f}  "
                        f"purity={res.get('knn_purity', float('nan')):.3f}  "
                        f"ARI={res.get('ari', float('nan')):.3f}  "
                        f"NMI={res.get('nmi', float('nan')):.3f}  "
                        f"({rows[-1]['seconds']} s)")

    tab = pd.DataFrame(rows)
    summ = summarise(tab)
    tab.to_csv(out_dir / "imputation_benchmark.tsv", sep="\t", index=False)
    summ.to_csv(out_dir / "imputation_benchmark_summary.tsv", sep="\t")
    plot(tab, out_dir / f"imputation_benchmark.{args.plot_format}")
    chance = float(tab["knn_chance"].astype(float).mean())
    logger.info(f"Real vs mean of {len(draws)} null draw(s) (kNN chance "
                f"{chance:.3f}; AUC chance 0.5):")
    for meth, r in summ.iterrows():
        logger.info("  " + f"{meth:12s}" + "  ".join(
            f"{lab} {r[f'{m}_real']:.3f} vs {r[f'{m}_null_mean']:.3f}"
            for m, lab in (("auc", "AUC"), ("knn_purity", "purity"),
                           ("ari", "ARI"), ("nmi", "NMI"))))
    logger.info(f"wrote {out_dir / 'imputation_benchmark_summary.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
