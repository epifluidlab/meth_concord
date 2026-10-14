#!/usr/bin/env python3
"""Is concordance cell-type specific, class pair by class pair?

From an annotation_class_concordance.py run (percell.tsv.gz,
cell_covariates.tsv), per context and class pair of one system -- e.g.
chromHMM state x chromHMM state, the pairs across states (promoter x
enhancer) as well as within -- two questions, each asked of the statistic
(C, 'delta'; or signed: C x w, 'signed_rel'; C x sign w, 'signed_sgn') and
of its control (the same on reshuffled links alone, which carries cell type
only through its construction):

  * does it differ between cell types (class_pair_variation.tsv)?  An OLS
    F-test of cell type over batch (--batch-columns), the pair's
    methylation / accessibility level and its square, the cell's level and
    its contacts in the pair (log), BH over the pairs -- and again of the
    pair less the cell's own genome-wide value ('_specific'): a type more
    concordant everywhere is more concordant in every pair; and the types'
    means
    with their intervals (class_pair_type_means.tsv), as is and less the
    type's own genome-wide value ('all regions'), so a type that is more
    concordant everywhere does not count;
  * does it tell a cell's type (class_pair_auc.tsv)?  Out-of-fold one-vs-rest
    macro AUC of an L2-logistic and a random forest -- each pair alone, each
    --sets set of pairs (pairs with both ends in the set's states, e.g.
    regulatory=promoter,enhancer), all pairs -- beside the control and the
    level alone.

A population difference need not make a classifier: a shift of a point
between types under a per-cell spread of several points is significant over
hundreds of cells and still near 0.5 cell by cell.  Figure:
class_pair_celltype_<context>.

    python scripts/class_pair_celltype.py \\
        --run-dir output/annotation_class_concordance_v10 \\
        --system chromhmm --batch-columns exp_batch,lib_batch \\
        --sets regulatory=promoter,enhancer \\
        --sets repressive=polycomb,heterochromatin \\
        --output-dir output/class_pair_celltype
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("class_pair_celltype")

#: Each statistic's control in percell.tsv.gz: the same on reshuffled links.
CONTROL_OF = {"delta": "control", "signed_rel": "signed_rel_control",
              "signed_sgn": "signed_sgn_control", "delta_r": "control_r"}
ALL_REGIONS = "all regions"


def fdr_bh(p) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size:
        o = ok[np.argsort(p[ok])]
        v = p[o] * o.size / np.arange(1, o.size + 1)
        q[o] = np.minimum(np.minimum.accumulate(v[::-1])[::-1], 1.0)
    return q


def read_run(percell, covariates, system: str, stats, band=None):
    """The run's per-cell rows of ``system`` in its all-distances band (or
    ``band``), those with enough pairs, with each cell's covariates."""
    want = ["context", "system", "cell", "cell_type", "anchor", "partner",
            "band", "pairs", "enough_pairs", "mu_anchor", "mu_partner"]
    head = pd.read_csv(percell, sep="\t", nrows=0).columns
    for s in stats:
        for c in (s, CONTROL_OF[s]):
            if c not in head:
                sys.exit(f"{percell}: no column {c!r} (--stats)")
            want.append(c)
    p = pd.read_csv(percell, sep="\t", usecols=list(dict.fromkeys(want)),
                    dtype={"cell": str, "cell_type": str})
    have = list(dict.fromkeys(p["system"]))
    p = p[p["system"] == system]
    if p.empty:
        sys.exit(f"{percell}: no rows of --system {system!r} (have "
                 f"{', '.join(have)})")
    b = band or next((x for x in dict.fromkeys(p["band"])
                      if str(x).startswith("all")), None)
    p = p[(p["band"] == b) & p["enough_pairs"].astype(bool)]
    if covariates is not None and Path(covariates).exists():
        cov = pd.read_csv(covariates, sep="\t", dtype={"cell": str})
        cov = cov.drop(columns=["cell_type"], errors="ignore")
        p = p.merge(cov, on="cell", how="left")
    p["pair"] = p["anchor"] + " × " + p["partner"]
    return p, b


def _cv_auc(X, y, clf: str, folds: int = 5, seed: int = 0) -> float:
    """Out-of-fold one-vs-rest macro AUC of ``clf`` ('lr': L2-logistic on
    standardised features; 'rf': 300 trees, leaves of >= 5 cells)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    labs, yy = np.unique(np.asarray(y), return_inverse=True)
    folds = min(folds, int(np.bincount(yy).min()))
    if labs.size < 2 or folds < 2:
        return np.nan
    P = np.zeros((yy.size, labs.size))
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=seed).split(X, yy):
        if clf == "lr":
            m = OneVsRestClassifier(make_pipeline(
                StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)))
        else:
            m = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                       min_samples_leaf=5, n_jobs=1,
                                       random_state=seed)
        m.fit(X[tr], yy[tr])
        P[np.ix_(te, m.classes_)] = m.predict_proba(X[te])
    return float(np.mean([roc_auc_score(yy == j, P[:, j])
                          for j in range(labs.size)]))


def auc_table(p: pd.DataFrame, stats, sets: dict, classifiers,
              min_type_cells: int = 10, n_threads: int = 1,
              seed: int = 0) -> pd.DataFrame:
    """Per context and group of pairs -- all, each --sets set, each pair --
    the cross-validated AUC of cell type from the statistic, its control,
    and the level (the pairs' two ends' mean posteriors)."""
    from joblib import Parallel, delayed
    tasks = []
    for ctx, q in p.groupby("context", sort=False):
        q = q[q["pair"] != f"{ALL_REGIONS} × {ALL_REGIONS}"]
        pairs = sorted(q["pair"].unique())
        groups = {"all pairs": pairs}
        for nm, states in sets.items():
            groups[nm] = [x for x in pairs
                          if all(s in states for s in x.split(" × "))]
        groups.update({x: [x] for x in pairs})
        types = q.drop_duplicates("cell").set_index("cell")["cell_type"]
        for gname, gp in groups.items():
            if not gp:
                continue
            qq = q[q["pair"].isin(gp)]
            feats = {(s, "concordance"): [s] for s in stats}
            feats.update({(s, "control"): [CONTROL_OF[s]] for s in stats})
            feats[("level", "level")] = ["mu_anchor", "mu_partner"]
            for (st, kind), cols in feats.items():
                W = pd.concat([qq.pivot_table(index="cell", columns="pair",
                                              values=c, aggfunc="first")
                               .add_suffix(f" {c}") for c in cols], axis=1)
                y = types.reindex(W.index)
                cnt = y.value_counts()
                keep = y.isin(cnt.index[cnt >= min_type_cells]).to_numpy()
                W, y = W[keep], y[keep]
                if y.nunique() < 2:
                    continue
                X = W.to_numpy(float)
                med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), 0)
                X = np.where(np.isfinite(X), X, np.nan_to_num(med)[None, :])
                npairs = float(qq[qq["cell"].isin(W.index)].groupby(
                    "cell")["pairs"].sum().median())
                for clf in classifiers:
                    tasks.append(((ctx, gname, st, kind, clf, len(y),
                                   y.nunique(), npairs), X, y.to_numpy()))
    res = Parallel(n_jobs=n_threads)(
        delayed(_cv_auc)(X, y, k[4], seed=seed) for k, X, y in tasks)
    return pd.DataFrame([{"context": k[0], "group": k[1], "statistic": k[2],
                          "features": k[3], "classifier": k[4],
                          "cells": k[5], "types": k[6], "median_pairs": k[7],
                          "auc": a} for (k, _, _), a in zip(tasks, res)])


def variation_table(p: pd.DataFrame, stats, batch_cols=(),
                    min_type_cells: int = 10, n_boot: int = 500,
                    seed: int = 0):
    """Per context and pair: does the statistic -- and its control --
    differ between cell types beyond batch, level and contacts (an OLS
    F-test of cell type, BH over the pairs)?  And per type its mean with a
    95% bootstrap interval, as is and less the type's own all-regions
    value."""
    import statsmodels.api as sm
    rng = np.random.default_rng(seed)
    rows, means = [], []
    whole = f"{ALL_REGIONS} × {ALL_REGIONS}"
    for ctx, q in p.groupby("context", sort=False):
        lvl_col = f"level_{ctx}" if f"level_{ctx}" in q else None
        allr = q[q["pair"] == whole].groupby("cell_type")
        # each cell's genome-wide value: a pair's type effect beyond it is
        # the pair's own
        own = q[q["pair"] == whole].set_index("cell")
        for pr, g in q.groupby("pair", sort=False):
            cnt = g["cell_type"].value_counts()
            g = g[g["cell_type"].isin(cnt.index[cnt >= min_type_cells])]
            if g["cell_type"].nunique() < 2 or len(g) < 30:
                continue
            mu = 0.5 * (g["mu_anchor"] + g["mu_partner"])
            Z = [mu, mu ** 2, np.log10(g["pairs"].clip(lower=1))]
            if lvl_col:
                Z.append(g[lvl_col])
            Z = pd.concat(Z, axis=1)
            for b in batch_cols:
                if b in g and g[b].nunique() > 1:
                    Z = pd.concat([Z, pd.get_dummies(g[b].astype(str),
                                                     prefix=b,
                                                     drop_first=True)],
                                  axis=1)
            T = pd.get_dummies(g["cell_type"], prefix="t", drop_first=True)
            Z = Z.astype(float).fillna(Z.astype(float).mean())
            r = {"context": ctx, "pair": pr, "cells": len(g),
                 "types": g["cell_type"].nunique(),
                 "median_pairs": float(g["pairs"].median())}
            for s in stats:
                for kind, col in (("", s), ("control_", CONTROL_OF[s])):
                    y = g[col].to_numpy(float)
                    ok = np.isfinite(y)
                    m = g.groupby("cell_type")[col].mean()
                    r[f"{kind}{s}_range"] = float(m.max() - m.min())
                    for tag in ("", "_specific"):
                        yy, okk = y, ok
                        if tag:
                            # the pair less the cell's genome-wide value: a
                            # type shifted everywhere cancels (a covariate
                            # measured with noise would only partly take it)
                            if pr == whole or col not in own:
                                continue
                            w_ = own[col].reindex(g["cell"]).to_numpy(float)
                            yy = y - w_
                            okk = np.isfinite(yy)
                        X0 = sm.add_constant(Z[okk].to_numpy(float),
                                             has_constant="add")
                        X1 = np.column_stack([X0, T[okk].to_numpy(float)])
                        f0 = sm.OLS(yy[okk], X0).fit()
                        f1 = sm.OLS(yy[okk], X1).fit()
                        try:
                            pv = float(f1.compare_f_test(f0)[1])
                        except ValueError:
                            pv = np.nan
                        r[f"{kind}{s}{tag}_p"] = pv
            rows.append(r)
            for t, x in g.groupby("cell_type"):
                base = (allr.get_group(t) if t in allr.groups else None)
                for s in stats:
                    v = x[s].dropna().to_numpy()
                    if not v.size:
                        continue
                    bt = [v[rng.integers(0, v.size, v.size)].mean()
                          for _ in range(n_boot)]
                    lo, hi = np.percentile(bt, [2.5, 97.5])
                    ref = (float(base[s].mean()) if base is not None
                           and s in base else np.nan)
                    means.append({"context": ctx, "pair": pr, "cell_type": t,
                                  "statistic": s, "cells": int(v.size),
                                  "mean": float(v.mean()), "lo": lo, "hi": hi,
                                  "control_mean": float(
                                      x[CONTROL_OF[s]].mean()),
                                  "minus_all_regions": float(v.mean()) - ref})
    var = pd.DataFrame(rows)
    for c in [c for c in var.columns if c.endswith("_p")]:
        var[c[:-2] + "_q"] = np.nan
        for ctx, g in var.groupby("context"):
            var.loc[g.index, c[:-2] + "_q"] = fdr_bh(g[c].to_numpy())
    return var, pd.DataFrame(means)


def plot_context(auc: pd.DataFrame, var: pd.DataFrame, means: pd.DataFrame,
                 ctx: str, stat: str, path, q_cut: float = 0.05):
    """a: the AUC of cell type per group of pairs -- the statistic, its
    control, the level -- logistic and forest; b: per cell type and pair
    the statistic less the type's genome-wide value, pairs with a type
    effect beyond it (q < q_cut) starred; c: -log10 q of the type effect --
    beyond batch and level, and beyond the cell's genome-wide value too --
    against the control's."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = auc[(auc["context"] == ctx)
            & auc["statistic"].isin([stat, "level"])]
    v = var[var["context"] == ctx]
    m = means[(means["context"] == ctx) & (means["statistic"] == stat)
              & (means["pair"] != f"{ALL_REGIONS} × {ALL_REGIONS}")]
    if a.empty and v.empty:
        return
    groups = list(dict.fromkeys(a["group"])) if len(a) else []
    key = (f"{stat}_specific_p" if f"{stat}_specific_p" in v
           else f"{stat}_p")
    pairs = (v.sort_values(key)["pair"].tolist() if len(v) else [])
    pairs = [x for x in pairs if x != f"{ALL_REGIONS} × {ALL_REGIONS}"]
    fig = plt.figure(figsize=(7.6, 3.2 + 0.16 * max(len(groups), len(pairs))))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.4, 0.75), wspace=1.25)
    ax = fig.add_subplot(gs[0])
    sty = {("concordance", "lr"): ("o", "#b2182b", "C, logistic"),
           ("concordance", "rf"): ("s", "#ef8a62", "C, forest"),
           ("control", "lr"): ("o", "0.55", "control, logistic"),
           ("control", "rf"): ("s", "0.75", "control, forest"),
           ("level", "lr"): ("D", "#2166ac", "level, logistic"),
           ("level", "rf"): ("D", "#92c5de", "level, forest")}
    for i, g in enumerate(groups):
        for (kind, clf), (mk, col, lab) in sty.items():
            r = a[(a["group"] == g) & (a["features"] == kind)
                  & (a["classifier"] == clf)]
            if len(r):
                ax.scatter(r["auc"], [-i], marker=mk, s=12, color=col,
                           label=lab if i == 0 else None, zorder=3)
    ax.axvline(0.5, color="0.6", lw=0.5)
    ax.set_yticks([-i for i in range(len(groups))])
    ax.set_yticklabels(groups, fontsize=5)
    ax.set_xlabel("cell type AUC (macro, out of fold)", fontsize=6)
    ax.set_title("a  a cell's type", loc="left", fontsize=6.5)
    ax.legend(fontsize=4.5, frameon=False, loc="lower right")
    ax.tick_params(labelsize=5.5)
    ax = fig.add_subplot(gs[1])
    if len(m) and pairs:
        H = m.pivot_table(index="pair", columns="cell_type",
                          values="minus_all_regions").reindex(pairs)
        lim = np.nanmax(np.abs(H.to_numpy())) if H.size else 1
        im = ax.imshow(100 * H.to_numpy(), aspect="auto", cmap="RdBu_r",
                       vmin=-100 * lim, vmax=100 * lim)
        ax.set_yticks(range(len(pairs)))
        q = v.set_index("pair")[key[:-2] + "_q"].reindex(pairs)
        ax.set_yticklabels([f"{x} {'*' if q.get(x, 1) < q_cut else ''}"
                            for x in pairs], fontsize=4.6)
        ax.set_xticks(range(H.shape[1]))
        ax.set_xticklabels(H.columns, rotation=90, fontsize=5)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(f"{stat} less the type's all-regions, pp", fontsize=5)
        cb.ax.tick_params(labelsize=5)
    ax.set_title(f"b  by cell type (* beyond its genome-wide, "
                 f"q < {q_cut:g})", loc="left", fontsize=6.5)
    ax = fig.add_subplot(gs[2])
    if len(v) and pairs:
        vv = v.set_index("pair").reindex(pairs)
        y = -np.arange(len(pairs))
        bars = [(f"{stat}_q", "#f4a582", "beyond batch, level"),
                (f"{stat}_specific_q", "#b2182b", "and genome-wide"),
                (f"control_{stat}_q", "0.6", "control")]
        bars = [b for b in bars if b[0] in vv]
        hgt = 0.8 / len(bars)
        for i, (c, col, lab) in enumerate(bars):
            ax.barh(y + (len(bars) - 1) / 2 * hgt - i * hgt,
                    -np.log10(vv[c]), height=hgt, color=col, label=lab)
        ax.axvline(-np.log10(q_cut), color="0.3", lw=0.5, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_ylim(-len(pairs) + 0.5, 0.5)     # b's rows
        ax.legend(fontsize=5, frameon=False)
    ax.set_xlabel("−log10 q, cell type effect", fontsize=6)
    ax.set_title("c  the type effect", loc="left", fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    fig.suptitle(f"{ctx}: is {stat} cell-type specific, pair by pair?",
                 fontsize=7)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _sets(values) -> dict:
    out = {}
    for v in values or []:
        nm, _, states = str(v).partition("=")
        out[nm.strip()] = {s.strip() for s in states.split(",") if s.strip()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir", default=None,
                    help="An annotation_class_concordance.py output "
                         "directory (percell.tsv.gz, cell_covariates.tsv).")
    ap.add_argument("--percell", default=None)
    ap.add_argument("--covariates", default=None)
    ap.add_argument("--system", default="chromhmm")
    ap.add_argument("--band", default=None,
                    help="percell's band (default: its all-distances one).")
    ap.add_argument("--stats", default="delta,signed_rel,signed_sgn")
    ap.add_argument("--classifiers", default="lr,rf")
    ap.add_argument("--sets", action="append", default=None,
                    metavar="NAME=STATE,STATE",
                    help="A set of pairs: both ends in these states "
                         "(repeatable).")
    ap.add_argument("--batch-columns", default="",
                    help="Covariates' batch columns, for the type effect.")
    ap.add_argument("--min-type-cells", type=int, default=10)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-format", default="pdf")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    rd = Path(args.run_dir) if args.run_dir else None
    percell = args.percell or (rd / "percell.tsv.gz" if rd else None)
    cov = args.covariates or (rd / "cell_covariates.tsv" if rd else None)
    if percell is None:
        ap.error("--run-dir or --percell")
    stats = [s.strip() for s in args.stats.split(",") if s.strip()]
    bad = [s for s in stats if s not in CONTROL_OF]
    if bad:
        ap.error(f"--stats {', '.join(bad)}: one of {', '.join(CONTROL_OF)}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p, band = read_run(percell, cov, args.system, stats, args.band)
    logger.info(f"{args.system}, {band}: {p['cell'].nunique():,} cells, "
                f"{p['pair'].nunique()} pairs")
    batch = [b.strip() for b in args.batch_columns.split(",") if b.strip()]
    var, means = variation_table(p, stats, batch, args.min_type_cells,
                                 seed=args.seed)
    var.to_csv(out / "class_pair_variation.tsv", sep="\t", index=False)
    means.to_csv(out / "class_pair_type_means.tsv", sep="\t", index=False)
    auc = auc_table(p, stats, _sets(args.sets),
                    [c.strip() for c in args.classifiers.split(",")],
                    args.min_type_cells, args.n_threads, args.seed)
    auc.to_csv(out / "class_pair_auc.tsv", sep="\t", index=False)
    for ctx in dict.fromkeys(p["context"]):
        for s in stats:
            plot_context(auc, var, means, ctx, s,
                         out / f"class_pair_celltype_{ctx}_{s}."
                               f"{args.plot_format}")
        v = var[var["context"] == ctx]
        if len(v):
            st = stats[0]
            top = v.sort_values(f"{st}_p").head(6)
            logger.info(f"{ctx}: pairs whose {st} differs between cell types "
                        f"(q; control's q):\n" + "\n".join(
                            f"  {r['pair']:<36} {r[f'{st}_q']:.3g}; "
                            f"{r[f'control_{st}_q']:.3g}"
                            for _, r in top.iterrows()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
