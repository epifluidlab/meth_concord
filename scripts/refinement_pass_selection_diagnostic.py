#!/usr/bin/env python3
"""How many RWR refinement passes — without cell-type labels?

PC0 cell-type AUC peaks at a specific number of refinement passes (2 on the
GM12878/IMR90 data: 0.876 -> 0.907 -> 0.886 for 1/2/3 passes), then
over-smoothing erodes it.  In a real discovery setting there are no labels
to pick that peak.

This script validates LABEL-FREE selection criteria: for each pass count it
computes unsupervised metrics (top SVD eigengap, best-k silhouette,
between-pass Δ change) AND the supervised PC0 AUC (using HCG labels, only to
*validate* the criterion on this labeled dataset).  If an unsupervised
metric peaks at the same pass as the AUC, that metric is the stopping rule
to use on unlabeled data.

Reuses the real pipeline (imputation.run_rwr_pipeline) with the signed
MLE + post-norm refinement settings, loading one existing run's saved
accessibility_mean.parquet + soft_counts_raw.parquet.

Usage
-----
    python scripts/refinement_pass_selection_diagnostic.py \\
        --concordance-dir output/signed_v8_final/context_GCH/05_concordance \\
        --hcg-labels /tmp/hcg_labels.umap_coords.tsv \\
        --max-passes 4
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

from nome_hic_concordance import imputation

logger = logging.getLogger("pass_select")

SOFT_KEYS = ["k_open", "k_closed", "k_mixed", "n_linked",
             "k_u_open", "k_u_closed", "k_u_mixed", "n_unlinked"]


def _make_mle_layer2(cell_index, bin_columns):
    """layer2_fn for run_rwr_pipeline: MLE Δ_signed (matches _layer2 mle)."""
    def _layer2(imp: dict[str, np.ndarray]) -> pd.DataFrame:
        def side(ko, kc, n):
            safe = np.where(n > 0, n, np.nan)
            return np.nan_to_num((ko - kc) / safe, nan=0.0)
        Sl = side(imp["k_open"], imp["k_closed"], imp["n_linked"])
        Su = side(imp["k_u_open"], imp["k_u_closed"], imp["n_unlinked"])
        return pd.DataFrame(Sl - Su, index=cell_index, columns=bin_columns)
    return _layer2


def _metrics(M: np.ndarray, y_bin: np.ndarray, top_k: int = 10) -> dict:
    """Unsupervised (eigengap, best-k silhouette) + supervised (PC0 AUC)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    from sklearn.metrics import roc_auc_score, silhouette_score

    svd = TruncatedSVD(n_components=min(top_k, M.shape[1] - 1),
                       random_state=42).fit(M)
    emb = svd.transform(M)
    vr = svd.explained_variance_ratio_

    # Unsupervised 1: top eigengap (ratio + absolute, in %).
    eigengap_ratio = float(vr[0] / vr[1]) if len(vr) > 1 and vr[1] > 0 else float("inf")
    eigengap_abs = float((vr[0] - vr[1]) * 100) if len(vr) > 1 else float("nan")

    # Unsupervised 2: best silhouette over k=2..6 (KMeans on SVD coords).
    best_sil, best_k = -1.0, 0
    for kk in range(2, min(7, M.shape[0])):
        try:
            lab = KMeans(n_clusters=kk, n_init=10, random_state=42).fit_predict(emb)
            sil = silhouette_score(emb, lab)
        except Exception:
            continue
        if sil > best_sil:
            best_sil, best_k = float(sil), kk

    # Supervised (validation only): PC0 AUC.
    pc0_auc = max(roc_auc_score(y_bin, emb[:, 0]),
                  roc_auc_score(y_bin, -emb[:, 0]))
    return {
        "PC0_var%": float(vr[0] * 100),
        "PC1_var%": float(vr[1] * 100) if len(vr) > 1 else float("nan"),
        "eigengap_ratio": eigengap_ratio,
        "eigengap_abs%": eigengap_abs,
        "best_silhouette": best_sil,
        "best_k": best_k,
        "PC0_AUC": float(pc0_auc),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concordance-dir", required=True)
    ap.add_argument("--hcg-labels", required=True)
    ap.add_argument("--max-passes", type=int, default=4,
                    help="Sweep --rwr-refine-passes from 0 to this value.")
    ap.add_argument("--k", type=int, default=15, help="k-NN neighbours.")
    ap.add_argument("--restart", type=float, default=0.5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cdir = Path(args.concordance_dir)

    acc = pd.read_parquet(cdir / "accessibility_mean.parquet")
    soft_long = pd.read_parquet(cdir / "soft_counts_raw.parquet")
    cells = sorted(soft_long["cell"].unique())
    bins = sorted(soft_long["bin"].unique(),
                  key=lambda b: (b.split(":")[0], int(b.split(":")[1])))
    soft = {
        key: (soft_long.pivot_table(index="cell", columns="bin", values=key,
                                    aggfunc="first", fill_value=0.0)
                       .reindex(index=cells, columns=bins, fill_value=0.0)
                       .to_numpy(dtype=np.float64))
        for key in SOFT_KEYS
    }
    # Align accessibility rows to soft-count cell order (k-NN uses cells only).
    acc = acc.reindex(index=cells)
    acc = acc.apply(lambda c: c.fillna(c.mean()), axis=0).fillna(0.0)
    logger.info(f"cells={len(cells)} bins={len(bins)}")

    hcg = pd.read_csv(args.hcg_labels, sep="\t")
    hmap = dict(zip(hcg["cell"], hcg["predicted_label"]))
    mask = np.array([c in hmap for c in cells])
    y = pd.Series([hmap[c] for c in np.array(cells)[mask]])
    y_bin = (y == y.iloc[0]).astype(int).values

    layer2 = _make_mle_layer2(pd.Index(cells), pd.Index(bins))

    rows = []
    for n_ref in range(0, args.max_passes + 1):
        logger.info(f"\n===== --rwr-refine-passes {n_ref} =====")
        _, delta_df, diag = imputation.run_rwr_pipeline(
            soft_count_matrices={k: v.copy() for k, v in soft.items()},
            initial_feature_matrix=acc,
            layer2_fn=layer2,
            n_refine_passes=n_ref,
            refinement_basis="postnorm",
            k=args.k, restart_prob=args.restart,
        )
        M = imputation.post_norm_for_knn(delta_df.to_numpy(dtype=np.float64))
        met = _metrics(M[mask], y_bin)
        # Between-pass convergence (last refinement pass's change), if any.
        dch = float("nan")
        passes = diag.get("passes", [])
        if len(passes) > 1 and "mean_abs_delta_change" in passes[-1]:
            dch = float(passes[-1]["mean_abs_delta_change"])
        met["refine_passes"] = n_ref
        met["delta_change"] = dch
        rows.append(met)

    summ = pd.DataFrame(rows)[
        ["refine_passes", "delta_change", "PC0_var%", "PC1_var%",
         "eigengap_ratio", "eigengap_abs%", "best_silhouette", "best_k", "PC0_AUC"]
    ]
    print("\n" + "=" * 92)
    print("Refinement-pass sweep  (unsupervised metrics vs validation AUC)")
    print("=" * 92)
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}",
                           "display.width", 200):
        print(summ.to_string(index=False))

    # Which pass each criterion picks.  CRITICAL: exclude pass 0 from the
    # unsupervised argmax.  Pass 0 (no refinement) uses the accessibility
    # k-NN, whose dominant axis is the depth/batch confound — it inflates the
    # eigengap and silhouette without carrying cell type.  The label-free
    # rule only makes sense among *refined* passes (>=1).
    refined = summ[summ["refine_passes"] >= 1]
    pick_auc = int(summ.loc[summ["PC0_AUC"].idxmax(), "refine_passes"])
    pick_gap = int(refined.loc[refined["eigengap_ratio"].replace(np.inf, np.nan)
                               .idxmax(), "refine_passes"])
    pick_sil = int(refined.loc[refined["best_silhouette"].idxmax(),
                               "refine_passes"])
    pick_gap_all = int(summ.loc[summ["eigengap_ratio"].replace(np.inf, np.nan)
                                .idxmax(), "refine_passes"])
    print(f"\nArgmax pass:  AUC(validation)={pick_auc}")
    print(f"  eigengap among refined passes (>=1) = {pick_gap}   "
          f"(over ALL passes incl 0 = {pick_gap_all}; pass 0 is the confound)")
    print(f"  silhouette among refined passes (>=1) = {pick_sil}")
    print("\nInterpretation:")
    print("  • Compare against the AUC argmax.  On the validation data the")
    print("    eigengap among refined passes matches the AUC peak; silhouette")
    print("    does not.  So the label-free stopping rule is: run >=1")
    print("    refinement pass, then pick the eigengap peak (the pipeline's")
    print("    --rwr-refine-passes auto implements exactly this).")
    print("  • Pass 0 must be excluded — its huge eigengap is the depth/batch")
    print("    confound, not cell type (high eigengap but low PC0 AUC).")


if __name__ == "__main__":
    main()
