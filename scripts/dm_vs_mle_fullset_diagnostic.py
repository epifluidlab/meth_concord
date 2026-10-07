#!/usr/bin/env python3
"""DM vs MLE on the FULL cell set — the comparison done right.

Earlier validations (the 2x2, the batch sweep) measured cell-type AUC on
the high-coverage HCG-labeled subset, where PC0 is always cell type.  That
hid the real situation: on the FULL set the low-coverage cells put a
non-cell-type confound on PC0 for BOTH estimators, so both need
--skip-top-svd 1, and the question becomes which estimator clusters better
in the retained (post-skip) space.

Single-PC AUC can't settle it: DM gives a higher-AUC but thin (~2.6% var)
cell-type PC, MLE a lower-AUC but dominant (~8.7% var) one — opposite
pulls.  The metric that predicts UMAP/clustering quality is leave-one-out
kNN cell-type accuracy in the retained PC space, evaluated on ALL cells
with claimed labels.

For each {estimator} x {batch-correction} x {skip 0,1} this reports that
kNN accuracy plus the leading retained-PC AUC + variance.  Reuses the real
pipeline functions; loads one run's saved accessibility + soft counts.

Usage
-----
    # Evaluate on the WHOLE sample's experimental labels (all cells), which
    # is required for a full-set assessment — the HCG-derived labels only
    # cover the high-coverage subset and would re-introduce the bias.
    python scripts/dm_vs_mle_fullset_diagnostic.py \\
        --concordance-dir output/signed_250kb_rwr_v3/context_GCH/05_concordance \\
        --samples samples.filtered.with_batches.tsv --batch-column batch
        # samples file columns: 'cell' (name), 'group' (label), 'batch'.
        # --hcg-labels is OPTIONAL and only adds a high-cov-subset breakdown.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance import batch_correction as bc
from nome_hic_concordance.utils import manifest_text
from nome_hic_concordance import imputation

logger = logging.getLogger("dm_vs_mle")
SOFT_KEYS = ["k_open", "k_closed", "k_mixed", "n_linked",
             "k_u_open", "k_u_closed", "k_u_mixed", "n_unlinked"]


def _load_samples(path):
    df = pd.read_csv(manifest_text(path), sep="\t", dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "cell_name" not in df.columns:
        # Accept 'cell' (e.g. samples.filtered.with_batches.tsv) or fall back
        # to the first column.
        if "cell" in df.columns:
            df = df.rename(columns={"cell": "cell_name"})
        else:
            df = df.rename(columns={df.columns[0]: "cell_name"})
    return df


def _mle_delta(m):
    def side(ko, kc, n):
        safe = np.where(n > 0, n, np.nan)
        return np.nan_to_num((ko - kc) / safe, nan=0.0)
    return (side(m["k_open"], m["k_closed"], m["n_linked"])
            - side(m["k_u_open"], m["k_u_closed"], m["n_unlinked"]))


def _fit_dm_eb(k_o, k_c, k_m, n, min_obs=50, nu_floor=2.0):
    mask = n.ravel() >= 1
    if int(mask.sum()) < min_obs:
        return {"open": 1.0, "closed": 1.0, "mixed": 1.0}
    n_arr = n.ravel()[mask]; n_sum = float(n_arr.sum())
    mus = {"open": float(k_o.ravel()[mask].sum() / n_sum),
           "closed": float(k_c.ravel()[mask].sum() / n_sum),
           "mixed": float(k_m.ravel()[mask].sum() / n_sum)}
    best = max(mus, key=mus.get)
    bk = {"open": k_o, "closed": k_c, "mixed": k_m}[best].ravel()[mask]
    w = n_arr / n_sum; mu = mus[best]
    s2 = float(np.sum(w * (bk / n_arr - mu) ** 2))
    n_bar = float(np.sum(w * n_arr)); mu1mu = mu * (1 - mu)
    numer, denom = mu1mu - s2, s2 - mu1mu / max(n_bar, 1.0)
    nu = 100.0 if denom <= 0 else (nu_floor if numer <= 0 else max(numer / denom, nu_floor))
    return {c: mu_c * nu for c, mu_c in mus.items()}


def _dm_delta(m):
    al = _fit_dm_eb(m["k_open"], m["k_closed"], m["k_mixed"], m["n_linked"])
    au = _fit_dm_eb(m["k_u_open"], m["k_u_closed"], m["k_u_mixed"], m["n_unlinked"])
    atl, atu = sum(al.values()), sum(au.values())
    Sl = ((m["k_open"] + al["open"]) - (m["k_closed"] + al["closed"])) / (m["n_linked"] + atl)
    Su = ((m["k_u_open"] + au["open"]) - (m["k_u_closed"] + au["closed"])) / (m["n_unlinked"] + atu)
    return Sl - Su


def _two_pass(acc_bc, soft, estimator, k=15, restart=0.5):
    """Pass-0 (accessibility k-NN) + pass-1 (postnorm-Δ k-NN), then Step-7 norm."""
    delta_fn = _mle_delta if estimator == "mle" else _dm_delta
    qn0 = imputation.quantile_normalize_for_knn(acc_bc.to_numpy(dtype=np.float64))
    _, P0 = imputation.build_knn_graph(qn0, k=k)
    imp0, _ = imputation.rwr_impute_matrices(soft, P0, restart_prob=restart)
    D1 = imputation.post_norm_for_knn(delta_fn(imp0))
    qn1 = imputation.quantile_normalize_for_knn(D1)
    _, P1 = imputation.build_knn_graph(qn1, k=k)
    imp1, _ = imputation.rwr_impute_matrices(soft, P1, restart_prob=restart)
    return imputation.post_norm_for_knn(delta_fn(imp1))


def _loo_knn_acc(X, y, k=15):
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]; k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X); idx = idx[:, 1:]
    pred = np.array([pd.Series(y[idx[i]]).mode()[0] for i in range(n)])
    return float((pred == y).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concordance-dir", required=True)
    ap.add_argument("--samples", required=True,
                    help="Whole-sample labels for ALL cells, e.g. "
                         "samples.filtered.with_batches.tsv with columns "
                         "'cell','group' (label),'batch'.  Used for the "
                         "headline full-set kNN accuracy.")
    ap.add_argument("--batch-column", default=None)
    ap.add_argument("--hcg-labels", default=None,
                    help="OPTIONAL high-coverage-subset breakdown only (e.g. "
                         "hcg_labels.umap_coords.tsv).  Do NOT use this as the "
                         "ground truth — it covers only high-cov cells and "
                         "biases toward the easy subset.  Omit for a clean "
                         "full-set assessment.")
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cdir = Path(args.concordance_dir)

    acc = pd.read_parquet(cdir / "accessibility_mean.parquet")
    soft_long = pd.read_parquet(cdir / "soft_counts_raw.parquet")
    cells = sorted(soft_long["cell"].unique())
    bins = sorted(soft_long["bin"].unique(),
                  key=lambda b: (b.split(":")[0], int(b.split(":")[1])))
    soft = {k: (soft_long.pivot_table(index="cell", columns="bin", values=k,
                                      aggfunc="first", fill_value=0.0)
                         .reindex(index=cells, columns=bins, fill_value=0.0)
                         .to_numpy(dtype=np.float64)) for k in SOFT_KEYS}
    acc = acc.reindex(index=cells)
    acc = acc.apply(lambda c: c.fillna(c.mean()), axis=0).fillna(0.0)

    s = _load_samples(args.samples)
    grp = s.set_index("cell_name")["group"].reindex(cells)
    y_full = grp.fillna("NA").to_numpy()
    batch = None
    if args.batch_column and args.batch_column in s.columns:
        batch = s.set_index("cell_name")[args.batch_column].reindex(cells)
    hcg_mask = None
    if args.hcg_labels:
        hcg = pd.read_csv(args.hcg_labels, sep="\t")
        hset = set(hcg["cell"])
        hcg_mask = np.array([c in hset for c in cells])
    logger.info(f"cells={len(cells)} bins={len(bins)} "
                f"(HCG-labeled={int(hcg_mask.sum()) if hcg_mask is not None else 'NA'})")

    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import roc_auc_score

    methods = ["none"] + (["harmony"] if batch is not None else [])
    rows = []
    for bcm in methods:
        acc_bc = acc if bcm == "none" else bc.apply_batch_correction(acc, batch, method=bcm)
        for est in ["dm", "mle"]:
            M = _two_pass(acc_bc, soft, est, k=args.k)
            svd = TruncatedSVD(n_components=11, random_state=42).fit(M)
            emb = svd.transform(M); vr = svd.explained_variance_ratio_
            for skip in (0, 1):
                ret = emb[:, skip:skip + 10]
                acc_all = _loo_knn_acc(ret, y_full, k=args.k)
                lead_y = (y_full == pd.Series(y_full).mode()[0]).astype(int)
                lead_auc = max(roc_auc_score(lead_y, ret[:, 0]),
                               roc_auc_score(lead_y, -ret[:, 0]))
                row = {"batch_corr": bcm, "estimator": est, "skip": skip,
                       "lead_PC_var%": float(vr[skip] * 100),
                       "lead_PC_AUC": float(lead_auc),
                       "kNN_acc_full": acc_all}
                if hcg_mask is not None:
                    row["kNN_acc_HCGsub"] = _loo_knn_acc(ret[hcg_mask],
                                                         y_full[hcg_mask], k=args.k)
                rows.append(row)

    summ = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print("DM vs MLE on the FULL cell set (kNN cell-type accuracy, claimed labels)")
    print("=" * 88)
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}",
                           "display.width", 200):
        print(summ.to_string(index=False))
    best = summ.loc[summ["kNN_acc_full"].idxmax()]
    print(f"\nBest full-set clustering: {best['estimator']} + "
          f"batch={best['batch_corr']} + skip={int(best['skip'])} "
          f"→ kNN acc {best['kNN_acc_full']:.3f}")
    print("\nThis (full-set retained-space kNN accuracy), not single-PC AUC, is "
          "what predicts UMAP quality.  Pick the row that maximizes kNN_acc_full.")


if __name__ == "__main__":
    main()
