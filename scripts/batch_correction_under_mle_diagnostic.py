#!/usr/bin/env python3
"""Compare batch-correction methods under the MLE signed estimator.

Background
----------
The signed Δ pipeline switched its default per-(cell,bin) estimator from
the Dirichlet-Multinomial posterior to plain MLE (commit 6ef1b54).  Under
MLE the cell-type axis lands on the dominant PC0 (no --skip-top-svd
needed).  Batch correction (nome_hic_concordance/batch_correction.py) is
applied to the mean-accessibility matrix BEFORE the pass-0 RWR k-NN graph,
and was validated under the OLD DM estimator.  This script checks whether
each batch-correction method still helps — without hurting the now-on-PC0
cell-type signal — under MLE.

Why this is faithful AND fast
-----------------------------
Batch correction transforms only the accessibility matrix, upstream of the
RWR graph.  ``accessibility_mean.parquet`` is saved PRE-correction (the
correction happens in-memory in Step 3.5).  So we can load that single
file plus ``soft_counts_raw.parquet`` from ONE existing signed run, apply
each batch method in-memory, and run the exact MLE + post-norm two-pass
RWR — reusing the pipeline's own functions.  No 4x full reruns needed.

The two-pass RWR here mirrors the signed auto-defaults:
    pass 0 : k-NN from QN(batch-corrected accessibility)
    pass 1 : k-NN from QN(post_norm(Δ_signed pass 0))   [refinement-basis=postnorm]
    Δ      : MLE   [concordance-formula=mle]
    final  : per-cell rank → per-bin QN → row recenter   [matrix_ops Step 7]

Usage
-----
    python scripts/batch_correction_under_mle_diagnostic.py \\
        --concordance-dir output/signed_v8_final/context_GCH/05_concordance \\
        --samples samples.filtered.tsv \\
        --batch-column batch \\
        --hcg-labels /tmp/hcg_labels.umap_coords.tsv \\
        --methods none,center,zscore,combat

Outputs a table: method | ANOVA-F pre/post (median, 90pct) | PC0 var% |
PC0 AUC | best-PC AUC.  Decision rule printed at the end.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the package importable when run as `python scripts/this.py` (the
# script dir, not the repo root, is on sys.path[0] by default).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance import batch_correction as bc
from nome_hic_concordance import imputation

logger = logging.getLogger("bc_under_mle")

SOFT_KEYS = ["k_open", "k_closed", "k_mixed", "n_linked",
             "k_u_open", "k_u_closed", "k_u_mixed", "n_unlinked"]


def _load_samples(path: str) -> pd.DataFrame:
    with open(path) as fh:
        first = fh.readline().strip()
    first_field = first.split("\t")[0].strip().lower()
    has_header = first_field in (
        "cell_name", "cellname", "cell", "sample", "sample_name",
    )
    if has_header:
        df = pd.read_csv(path, sep="\t", comment="#", dtype=str)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "cellname" in df.columns:
            df = df.rename(columns={"cellname": "cell_name"})
        if "sample" in df.columns and "cell_name" not in df.columns:
            df = df.rename(columns={"sample": "cell_name"})
    else:
        df = pd.read_csv(
            path, sep="\t", header=None, comment="#",
            names=["cell_name", "bam_path", "group"], dtype=str,
        )
    return df


def _mle_delta(imp: dict[str, np.ndarray]) -> np.ndarray:
    """MLE Δ_signed, matching concordance._layer2's mle branch (n=0 -> 0)."""
    def side(ko, kc, n):
        safe = np.where(n > 0, n, np.nan)
        return np.nan_to_num((ko - kc) / safe, nan=0.0)
    Sl = side(imp["k_open"], imp["k_closed"], imp["n_linked"])
    Su = side(imp["k_u_open"], imp["k_u_closed"], imp["n_unlinked"])
    return Sl - Su


def _two_pass_mle(acc_bc: pd.DataFrame, soft: dict[str, np.ndarray],
                  k: int, restart: float) -> np.ndarray:
    """Pass-0 (accessibility k-NN) + pass-1 (postnorm-Δ k-NN), MLE Δ, then
    the Step-7 post-norm.  Returns the final cells × bins matrix for SVD."""
    # Pass 0.
    qn_init = imputation.quantile_normalize_for_knn(
        acc_bc.to_numpy(dtype=np.float64))
    _, P0 = imputation.build_knn_graph(qn_init, k=k)
    imp0, _ = imputation.rwr_impute_matrices(soft, P0, restart_prob=restart)
    D1 = _mle_delta(imp0)
    # Pass 1 (refinement-basis = postnorm).
    qn_ref = imputation.quantile_normalize_for_knn(
        imputation.post_norm_for_knn(D1))
    _, P1 = imputation.build_knn_graph(qn_ref, k=k)
    imp1, _ = imputation.rwr_impute_matrices(soft, P1, restart_prob=restart)
    D2 = _mle_delta(imp1)
    # Step-7 normalization (per-cell rank -> per-bin QN -> row recenter).
    return imputation.post_norm_for_knn(D2)


def _pc_auc(M: np.ndarray, y_bin: np.ndarray, top_k: int = 10) -> pd.DataFrame:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import roc_auc_score
    svd = TruncatedSVD(n_components=min(top_k, M.shape[1] - 1),
                       random_state=42).fit(M)
    emb = svd.transform(M)
    rows = []
    for i in range(emb.shape[1]):
        auc = max(roc_auc_score(y_bin, emb[:, i]),
                  roc_auc_score(y_bin, -emb[:, i]))
        rows.append({"PC": i,
                     "var_frac": float(svd.explained_variance_ratio_[i]),
                     "auc": float(auc)})
    return pd.DataFrame(rows)


def _batch_auc(M: np.ndarray, batch_vals: np.ndarray, top_k: int = 10) -> float:
    """Max one-batch-vs-rest AUC across the top PCs of M — i.e. how much
    batch still structures the final Δ_signed.  High = batch leaked into the
    result (bad); ~0.5-0.6 = batch removed (good)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import roc_auc_score
    svd = TruncatedSVD(n_components=min(top_k, M.shape[1] - 1),
                       random_state=42).fit(M)
    emb = svd.transform(M)
    best = 0.0
    for b in pd.unique(batch_vals):
        yb = (batch_vals == b).astype(int)
        if yb.sum() < 2 or yb.sum() > len(yb) - 2:
            continue
        for i in range(emb.shape[1]):
            a = max(roc_auc_score(yb, emb[:, i]), roc_auc_score(yb, -emb[:, i]))
            best = max(best, a)
    return float(best)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concordance-dir", required=True,
                    help="Dir with accessibility_mean.parquet + "
                         "soft_counts_raw.parquet (a 05_concordance dir).")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--batch-column", required=True,
                    help="Column name in --samples holding batch labels.")
    ap.add_argument("--hcg-labels", required=True,
                    help="HCG label TSV (cols 'cell', 'predicted_label').")
    ap.add_argument("--methods", default="none,center,zscore,combat,harmony",
                    help="Comma-separated batch-correction methods to compare.")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--restart", type=float, default=0.5)
    ap.add_argument("--top-k-pcs", type=int, default=10)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cdir = Path(args.concordance_dir)

    # ---- load pre-correction accessibility + raw soft counts ----
    acc = pd.read_parquet(cdir / "accessibility_mean.parquet")
    soft_long = pd.read_parquet(cdir / "soft_counts_raw.parquet")
    logger.info(f"accessibility: {acc.shape}, soft_counts rows: {len(soft_long)}")

    cells_soft = sorted(soft_long["cell"].unique())
    bins = sorted(soft_long["bin"].unique(),
                  key=lambda b: (b.split(":")[0], int(b.split(":")[1])))
    soft = {
        key: (soft_long.pivot_table(index="cell", columns="bin", values=key,
                                    aggfunc="first", fill_value=0.0)
                       .reindex(index=cells_soft, columns=bins, fill_value=0.0)
                       .to_numpy(dtype=np.float64))
        for key in SOFT_KEYS
    }

    # ---- batch labels ----
    samples = _load_samples(args.samples)
    if args.batch_column not in samples.columns:
        raise SystemExit(
            f"--batch-column {args.batch_column!r} not in samples columns: "
            f"{samples.columns.tolist()}")
    batch_labels = samples.set_index("cell_name")[args.batch_column]

    # ---- align all cells (accessibility ∩ soft counts ∩ batch) ----
    common = [c for c in cells_soft
              if c in acc.index and c in batch_labels.index]
    acc = acc.loc[common]
    keep_idx = np.array([cells_soft.index(c) for c in common])
    soft = {key: v[keep_idx] for key, v in soft.items()}
    batch_labels = batch_labels.loc[common]
    logger.info(f"cells with accessibility + soft counts + batch: {len(common)}")
    logger.info(f"batch counts: {dict(batch_labels.value_counts())}")

    # ---- HCG ground-truth labels ----
    hcg = pd.read_csv(args.hcg_labels, sep="\t")
    hcg_map = dict(zip(hcg["cell"], hcg["predicted_label"]))
    mask = np.array([c in hcg_map for c in common])
    y = pd.Series([hcg_map[c] for c in np.array(common)[mask]])
    y_bin = (y == y.iloc[0]).astype(int).values
    logger.info(f"HCG-labelled cells: {int(mask.sum())} / {len(common)}")

    # ---- per-method comparison ----
    summary = []
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        logger.info(f"\n===== batch-correction = {method} =====")
        try:
            acc_bc = bc.apply_batch_correction(acc, batch_labels, method=method)
        except ImportError as exc:
            logger.warning(f"  skipping {method!r}: {exc}")
            continue

        f_pre = bc.batch_anova_f_per_bin(acc, batch_labels)
        f_post = bc.batch_anova_f_per_bin(acc_bc, batch_labels)

        M = _two_pass_mle(acc_bc, soft, k=args.k, restart=args.restart)
        auc_df = _pc_auc(M[mask], y_bin, top_k=args.top_k_pcs)
        # Residual batch structure in the FINAL Δ_signed (over batch-labelled
        # cells).  Want this LOW (batch removed) while PC0_AUC stays HIGH.
        b_ok = batch_labels.notna().values
        batch_auc = _batch_auc(M[b_ok], batch_labels[b_ok].values,
                               top_k=args.top_k_pcs)

        pc0 = auc_df.iloc[0]
        best = auc_df.loc[auc_df["auc"].idxmax()]
        print(f"\n--- {method} ---")
        print(f"{'PC':>3s} {'var %':>8s} {'AUC':>7s}")
        for _, r in auc_df.iterrows():
            print(f"{int(r['PC']):>3d} {r['var_frac']*100:>7.2f}% {r['auc']:>7.3f}")
        print(f"  cell-type PC0 AUC = {pc0['auc']:.3f}   "
              f"residual batch AUC = {batch_auc:.3f}")
        summary.append({
            "method": method,
            "F_pre_med": float(np.nanmedian(f_pre)),
            "F_post_med": float(np.nanmedian(f_post)),
            "PC0_AUC": float(pc0["auc"]),
            "best_PC": int(best["PC"]),
            "best_AUC": float(best["auc"]),
            "batch_AUC": batch_auc,
        })

    # ---- comparison table + decision ----
    summ = pd.DataFrame(summary)
    print("\n" + "=" * 78)
    print("SUMMARY — batch correction under MLE estimator")
    print("=" * 78)
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(summ.to_string(index=False))

    print("\nDecision guide (the two columns that matter: PC0_AUC, batch_AUC):")
    print("  • IDEAL method: HIGH PC0_AUC (cell type preserved) AND LOW")
    print("    batch_AUC (~0.5-0.6, batch removed from the final Δ_signed).")
    print("  • 'none'   : high PC0_AUC but HIGH batch_AUC → batch still")
    print("              fragments the embedding (the UMAP islands).")
    print("  • combat/center/zscore : low batch_AUC but LOW PC0_AUC → they")
    print("              ate the cell-type signal (composition imbalance).")
    print("  • harmony  : the protected method — soft-clusters preserve cell")
    print("              type while removing within-cluster batch.  If it keeps")
    print("              PC0_AUC high AND drops batch_AUC, it's the winner →")
    print("              set the signed --batch-correction auto-default to harmony.")
    print("  • If NOTHING keeps PC0_AUC high while dropping batch_AUC, fall back")
    print("    to --batch-correction none (accept batch fragmentation) and lean")
    print("    on protected ComBat (provisional-label mod=) as a follow-up.")


if __name__ == "__main__":
    main()
