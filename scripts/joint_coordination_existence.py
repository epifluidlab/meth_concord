#!/usr/bin/env python3
"""Joint GCH+HCG multi-state long-range coordination existence test (J1/J2/J4).

Does combining **accessibility** (GCH) and **methylation** (HCG) into a per-end
joint *state* carry cell-type information **beyond either context alone** — and
beyond marginal levels?  This is the clean, leakage-free existence test
(``ep_coordination_existence.py``) generalized to the joint multi-state grid of
``design/JOINT_MULTISTATE_CONCORDANCE_DESIGN.md`` (§2–§4).

For each cell it joins the GCH and HCG posterior context runs per end (§7),
builds the soft 4-state distribution ``q = (open, closed, poised,
active_methylated)`` (§2; lenient global-prior imputation of the missing
modality, §2.1), and forms a distance-matched **log2(O/E)** concordance per
co-state channel pooled per region in the E–P band — pooling PAIRS not CELLS, so
there is no cross-cell imputation and no leakage.

On the SAME end pool and SAME null it scores three feature sets through the SAME
confound ladder, so the comparison is apples-to-apples:

  joint  — 4 channels Δ_s = log2(O_s/E_s)        (the multi-state object)
  gch    — accessibility signed contrast          (single-context reference)
  hcg    — methylation signed contrast            (single-context reference)

The headline question: does **joint** beat **max(gch, hcg)** at the marginal-
and coverage-controlled rung (Δ|m+cov), above the permutation null?  Per-channel
joint AUCs (open / closed / poised / active-methylated) show WHICH co-state
carries identity (design §6 diagnostic).

Example
-------
    python scripts/joint_coordination_existence.py \\
        --gch-dir output/.../context_GCH/04_posteriors \\
        --hcg-dir output/.../context_HCG/04_posteriors \\
        --samples samples.with_batches.tsv \\
        --min-distance 20000 --max-distance 200000 \\
        --granularities chrom,25,10,5 --classifier lr \\
        --batch-column exp_batch --batch-column lib_batch \\
        --n-threads 8 --output-prefix joint_ep_existence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance import coordination_eval as ce
from nome_hic_concordance.utils import manifest_text
from nome_hic_concordance.joint_coordination import (
    JOINT_CHANNELS, build_matrices, process_cell, scan_priors,
)


# --------------------------------------------------------------------------
# Sample-sheet IO
# --------------------------------------------------------------------------

def _read_sheet(path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_text(path), sep="\t", dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "cell_name" not in df.columns:
        df = df.rename(columns={"cell": "cell_name"} if "cell" in df.columns
                       else {df.columns[0]: "cell_name"})
    return df.set_index("cell_name")


def _labels(path: str) -> pd.Series:
    df = _read_sheet(path)
    if "group" not in df.columns:
        raise SystemExit(f"{path} has no 'group' column.")
    return df["group"]


def _column(path: str, col: str) -> pd.Series:
    df = _read_sheet(path)
    col = col.strip().lower().replace(" ", "_")
    if col not in df.columns:
        raise SystemExit(f"{path} has no '{col}' column (have {df.columns.tolist()}).")
    return df[col]


def _cells(d: Path) -> dict:
    return {p.name[: -len(".posteriors.parquet")]: p
            for p in d.glob("*.posteriors.parquet")}


# --------------------------------------------------------------------------
# The confound ladder on one feature set (joint / gch / hcg / a channel)
# --------------------------------------------------------------------------

def _ladder(Dk, Mk, Covk, y, args, batch_oh):
    """Run Δ → Δ|m → Δ|m+cov (+batch) + marg/miss on a cells × features set."""
    m, nf = args.classifier, Dk.shape[1]
    sc = lambda X: ce.score(X, y, m, args.n_folds, args.random_seed)
    ap = lambda X: ce.auc_perm(X, y, m, args.n_folds, args.random_seed,
                               args.n_perm, args.n_threads)
    prp = lambda X: ce.prep(ce.standardize(pd.DataFrame(X)), m, args.n_svd)

    cmean, cstd, cp = ap(prp(Dk))
    rmean, rstd, rp = ap(prp(ce.residualize(Dk, Mk)))
    R2 = ce.residualize_multi(Dk, [Mk, Mk * Mk, np.log1p(np.nan_to_num(Covk))])
    r2mean, r2std, r2p = ap(prp(R2))
    mmean, _ = sc(prp(Mk))
    missmean, _ = sc(prp(np.isfinite(Dk).astype(float)))
    b3mean = b3p = np.nan
    if batch_oh is not None:
        b3mean, _, b3p = ap(prp(ce.residualize_against(R2, batch_oh)))
    return {"n_features": nf,
            "delta_auc": cmean, "delta_std": cstd, "delta_perm_p": cp,
            "resid_auc": rmean, "resid_std": rstd, "resid_perm_p": rp,
            "resid_full_auc": r2mean, "resid_full_std": r2std,
            "resid_full_perm_p": r2p,
            "resid_full_batch_auc": b3mean, "resid_full_batch_perm_p": b3p,
            "marginal_auc": mmean, "missingness_auc": missmean,
            "_R2": R2}


def _fmt_row(name, r, with_batch):
    bt = (f" {r['resid_full_batch_auc']:>6.3f} {r['resid_full_batch_perm_p']:>5.3f}"
          if with_batch else "")
    return (f"  {name:<16} {r['n_features']:>5} "
            f"{r['delta_auc']:>6.3f} {r['delta_perm_p']:>5.3f} "
            f"{r['resid_auc']:>6.3f} {r['resid_perm_p']:>5.3f} "
            f"{r['resid_full_auc']:>7.3f} {r['resid_full_perm_p']:>5.3f}"
            f"{bt} {r['marginal_auc']:>6.3f} {r['missingness_auc']:>6.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gch-dir", required=True,
                    help="Dir of *.posteriors.parquet for the GCH (accessibility) run.")
    ap.add_argument("--hcg-dir", required=True,
                    help="Dir of *.posteriors.parquet for the HCG (methylation) run.")
    ap.add_argument("--samples", required=True, help="TSV with cell + group.")
    ap.add_argument("--min-distance", type=int, default=20_000)
    ap.add_argument("--max-distance", type=int, default=200_000)
    ap.add_argument("--min-gpc-sites", type=int, default=3,
                    help="Min GpC sites for accessibility to be observed (else "
                         "imputed to the global prior). Default 3.")
    ap.add_argument("--min-cpg-sites", type=int, default=1,
                    help="Min CpG sites for methylation to be observed (else "
                         "imputed to the global prior). Default 1 — methylation "
                         "is the scarce modality, so keep its observations.")
    ap.add_argument("--granularities", default="chrom,25,10,5")
    ap.add_argument("--max-ends-per-chrom", type=int, default=4000)
    ap.add_argument("--classifier", choices=["svd-lr", "lr", "rf"], default="lr",
                    help="lr (default) = L2-LR on FULL standardized features; "
                         "svd-lr = SVD→LR; rf = random forest.")
    ap.add_argument("--n-svd", type=int, default=30)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--min-region-frac", type=float, default=0.0,
                    help="Drop regions whose Δ is observed in < this fraction of "
                         "labeled cells (removes the missingness/contact-presence "
                         "confound).")
    ap.add_argument("--batch-column", action="append", default=[], metavar="COLUMN",
                    help="Per-cell batch column(s) in --samples (repeatable and "
                         "comma-separated). Residualizes Δ|m+cov against the UNION "
                         "of their one-hots → a Δ|m+cov+batch AUC.")
    ap.add_argument("--per-channel", action="store_true",
                    help="Also report each joint channel (open/closed/poised/"
                         "active_methylated) alone at the Δ|m+cov rung — WHICH "
                         "co-state carries cell-type identity (design §6).")
    ap.add_argument("--output-prefix", default=None)
    args = ap.parse_args()

    grans = [g if g == "chrom" else int(g) for g in args.granularities.split(",")]
    gch_cells = _cells(Path(args.gch_dir))
    hcg_cells = _cells(Path(args.hcg_dir))
    shared = sorted(set(gch_cells) & set(hcg_cells))
    if not shared:
        sys.exit("No cells shared between --gch-dir and --hcg-dir.")
    labels = _labels(args.samples)

    # Global priors (leak-safe shared prior for lenient imputation, §2.1).
    pri_sample = shared[: min(len(shared), 40)]
    a_prior = scan_priors([gch_cells[c] for c in pri_sample],
                          args.min_gpc_sites, args.n_threads)
    m_prior = scan_priors([hcg_cells[c] for c in pri_sample],
                          args.min_cpg_sites, args.n_threads)
    n_classes = labels.reindex(shared).nunique()
    feat = ("SVD top-%d" % args.n_svd if args.classifier == "svd-lr"
            else "ALL region features (no SVD)")
    print(f"[joint-ep] {len(shared)} shared cells | band "
          f"{args.min_distance:,}–{args.max_distance:,} bp | {n_classes} cell "
          f"types | grans={grans} | classifier={args.classifier} → {feat}")
    print(f"[joint-ep] global priors: accessibility a={a_prior:.3f}  "
          f"methylation m={m_prior:.3f}")

    if args.n_threads > 1 and len(shared) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(
                gch_cells[c], hcg_cells[c], grans, args.min_distance,
                args.max_distance, args.min_gpc_sites, args.min_cpg_sites,
                a_prior, m_prior, args.max_ends_per_chrom, args.random_seed + i)
            for i, c in enumerate(shared))
    else:
        results = [process_cell(
            gch_cells[c], hcg_cells[c], grans, args.min_distance,
            args.max_distance, args.min_gpc_sites, args.min_cpg_sites,
            a_prior, m_prior, args.max_ends_per_chrom, args.random_seed + i)
            for i, c in enumerate(shared)]
    results = [(c, d, qc) for c, d, qc in results if d]
    cells = [c for c, _, _ in results]
    y_all = labels.reindex(cells)
    keep = y_all.notna().to_numpy()
    if keep.sum() < 10:
        sys.exit("Too few labeled cells.")
    y = y_all[keep].to_numpy()

    # QC (§2.1): observation completeness on the analysis pool.
    qcs = [qc for _, _, qc in results if qc]
    if qcs:
        def _mean(k):
            v = np.array([q.get(k, np.nan) for q in qcs], float)
            return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        print(f"[joint-ep] QC: per-end both={_mean('frac_ends_both'):.1%} "
              f"gch_only={_mean('frac_ends_gch_only'):.1%} "
              f"hcg_only={_mean('frac_ends_hcg_only'):.1%} | "
              f"contact pairs fully-observed={_mean('frac_pairs_full'):.1%}")

    # Batch one-hot (union of factors) over the labeled cells.
    batch_oh = None
    batch_cols = list(dict.fromkeys(
        c.strip() for item in args.batch_column for c in item.split(",")
        if c.strip()))
    if batch_cols:
        ohs = []
        for col in batch_cols:
            bvals = _column(args.samples, col).reindex(cells).to_numpy()
            oh = pd.get_dummies(pd.Series(bvals[keep]), prefix=col,
                                drop_first=True, dummy_na=False)
            if oh.shape[1]:
                ohs.append(oh.to_numpy(np.float64))
        if ohs:
            batch_oh = np.hstack(ohs)
            print(f"[joint-ep] batch control on {batch_cols}: "
                  f"{batch_oh.shape[1]} indicator columns")

    print(f"\n[classifier={args.classifier}]  Δ|m = resid on marginal; Δ|m+cov "
          f"= on marginal+marginal²+coverage; miss = contact-presence AUC.")
    print(f"  HEADLINE: does joint(4ch) Δ|m+cov beat max(gch,hcg) above its "
          f"permutation null?\n")

    rows = []
    roc_data: dict = {}
    for gran in grans:
        mats = build_matrices(results, gran)
        region_keys = mats["region_keys"]
        if not region_keys:
            continue

        def _prep_set(name):
            D, M, C = mats[name]
            Dk, Mk, Ck = D[keep], M[keep], C[keep]
            if args.min_region_frac > 0:
                of = np.isfinite(Dk).mean(axis=0)
                cols = of >= args.min_region_frac
                if int(cols.sum()) >= 2:
                    Dk, Mk, Ck = Dk[:, cols], Mk[:, cols], Ck[:, cols]
            return Dk, Mk, Ck

        Djk, Mjk, Cjk = _prep_set("joint")
        Dgk, Mgk, Cgk = _prep_set("gch")
        Dhk, Mhk, Chk = _prep_set("hcg")
        if Djk.shape[1] < 4:
            continue
        med = int(np.median(Cjk[np.isfinite(Cjk)])) if np.isfinite(Cjk).any() else 0

        res_joint = _ladder(Djk, Mjk, Cjk, y, args, batch_oh)
        res_gch = _ladder(Dgk, Mgk, Cgk, y, args, batch_oh)
        res_hcg = _ladder(Dhk, Mhk, Chk, y, args, batch_oh)

        wb = batch_oh is not None
        print(f"granularity = {gran}   regions={len(region_keys)}  "
              f"joint_feats={res_joint['n_features']}  median_pairs/region={med}")
        print(f"  {'feature':<16} {'nf':>5} {'Δ':>6} {'p':>5} {'Δ|m':>6} {'p':>5} "
              f"{'Δ|m+cov':>7} {'p':>5}{'  +batch     p' if wb else ''} "
              f"{'marg':>6} {'miss':>6}")
        print(_fmt_row("joint(4ch)", res_joint, wb))
        print(_fmt_row("gch(access)", res_gch, wb))
        print(_fmt_row("hcg(methyl)", res_hcg, wb))

        for nm, r in (("joint", res_joint), ("gch", res_gch), ("hcg", res_hcg)):
            rows.append({"feature": nm, "granularity": str(gran),
                         "n_regions": len(region_keys),
                         "median_pairs_per_region": med,
                         **{k: v for k, v in r.items() if not k.startswith("_")}})

        # Per-channel joint diagnostic (which co-state carries identity).
        if args.per_channel:
            short = {"active_methylated": "act-meth"}
            for s, ch in enumerate(JOINT_CHANNELS):
                rc = _ladder(Djk[:, s::4], Mjk[:, s::4], Cjk[:, s::4], y, args,
                             batch_oh)
                print(_fmt_row(f"  ·{short.get(ch, ch)}", rc, wb))
                rows.append({"feature": f"joint:{ch}", "granularity": str(gran),
                             "n_regions": len(region_keys),
                             "median_pairs_per_region": med,
                             **{k: v for k, v in rc.items() if not k.startswith("_")}})
        print()

        if args.output_prefix:
            roc_data[str(gran)] = {
                "joint": ce.cv_predict(
                    ce.prep(ce.standardize(pd.DataFrame(res_joint["_R2"])),
                            args.classifier, args.n_svd),
                    y, args.classifier, args.n_folds, args.random_seed),
                "gch": ce.cv_predict(
                    ce.prep(ce.standardize(pd.DataFrame(res_gch["_R2"])),
                            args.classifier, args.n_svd),
                    y, args.classifier, args.n_folds, args.random_seed),
                "hcg": ce.cv_predict(
                    ce.prep(ce.standardize(pd.DataFrame(res_hcg["_R2"])),
                            args.classifier, args.n_svd),
                    y, args.classifier, args.n_folds, args.random_seed)}

    print("  joint Δ|m+cov > max(gch,hcg) AND p<0.05 AND > miss  => the JOINT")
    print("       multi-state adds cell-type info beyond either context alone.")
    print("  joint ≈ max(gch,hcg)  => the combination adds nothing the better")
    print("       single context didn't already carry (expected if methylation")
    print("       is too sparse — see QC per-end both%).")

    if args.output_prefix:
        out = f"{args.output_prefix}.joint_existence.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n[joint-ep] wrote {out}")
        try:
            _plot_compare_roc(roc_data, args.classifier,
                              f"{args.output_prefix}.joint_roc.pdf")
            print(f"[joint-ep] wrote {args.output_prefix}.joint_roc.pdf")
        except Exception as e:                       # pragma: no cover
            print(f"[joint-ep] ROC plot skipped ({e})")


def _plot_compare_roc(roc_data: dict, classifier: str, output: str) -> None:
    """ROC of joint vs gch vs hcg at the Δ|m+cov rung, per granularity."""
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
        for name in ("joint", "gch", "hcg"):
            dat = roc_data[gk].get(name)
            if dat is None:
                continue
            yt, ys = dat
            fpr, tpr, _ = roc_curve(yt, ys)
            ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(f"granularity = {gk}", fontsize=10)
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.legend(fontsize=8, loc="lower right")
    for j in range(len(grans), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Joint vs GCH vs HCG  —  Δ|m+cov  ({classifier})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
