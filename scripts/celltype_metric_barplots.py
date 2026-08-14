#!/usr/bin/env python3
"""Bar plots of cell-type agreement across the concordance arms and controls.

One bar per arm, one PDF per metric (kNN purity, ARI, NMI by default), for one
context (GCH or HCG):

    1D baseline                    the accessibility / methylation level matrix
    D                              directional coordination
    D − 1D,3D (linear)             D after the baselines are regressed out
    D − 1D,3D (quadratic)          ... with squares + interactions
    C                              coordination magnitude
    C − 1D,3D (linear)
    C − 1D,3D (quadratic)
    random control                 per-bin shuffle of the concordance matrix

Every arm goes through the SAME pipeline as nome_hic_compare, by calling that
module's own functions rather than reimplementing them:

    comparison._prep_matrix   coverage filter → impute → per-cell → per-bin norm
    comparison._svd_scores    column-standardize → TruncatedSVD → drop skip_top
    metrics.label_agreement   silhouette / kNN purity / ARI / NMI

Residual arms additionally run comparison.variance_partition on the NORMALIZED
target and baselines (the order run_comparison uses), then re-prep the residual
before scoring — again matching comparison.residual_umap.

THREE THINGS TO KNOW
--------------------
1. nome_hic_compare computes its reported metrics from the SVD scores WITHOUT
   batch correction -- harmony is applied only inside the UMAP call, for the
   picture.  This script therefore defaults to ``--batch-correction none`` so
   its numbers reproduce the ones in comparison_metrics.  Passing
   ``--batch-correction harmony`` here corrects the scores before scoring,
   which is a different (and stricter) question.
1b. ``--concordance-batch-correction`` is a THIRD, earlier stage, on the C and
   D matrices themselves rather than on the scores, matching the flag of the
   same name in nome_hic_concordance and nome_hic_compare.  It reaches the C
   and D arms, their residuals and the random control, and deliberately not
   the 1D/3D baselines those arms are measured against.
2. The random control is a per-bin permutation of the concordance matrix
   across cells: it preserves every bin's value distribution and the exact NaN
   pattern, and destroys only the cell association.  It is the floor for this
   whole pipeline, so it also absorbs any apparent structure the imputation and
   normalization introduce on their own.

Example (GCH, matching the nome_hic_compare invocation with skip_top_svd = 1)

    python scripts/celltype_metric_barplots.py \\
        --concordance-c out/context_GCH/05_concordance/concordance_coordination.parquet \\
        --concordance-d out/context_GCH/05_concordance/concordance_directional.parquet \\
        --baseline-1d  out/context_GCH/05_concordance/accessibility_mean.parquet \\
        --baseline-3d  out/compartments/compartment_scores.parquet \\
        --samples mouse_cortex_1577_cells.samples.with_batches.tsv \\
        --context GCH --matrix-norm quantile --n-svd 10 --skip-top-svd 1 \\
        --per-cell-norm rank --post-norm-recenter \\
        --exclude-chrom chrM,chrX,chrY \\
        --batch-column exp_batch,plate_batch,lib_batch \\
        --output-dir out/09_metric_barplots_GCH

For HCG, point at the HCG matrices and drop ``--skip-top-svd``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nome_hic_concordance import comparison, metrics as _metrics

logger = logging.getLogger("metric_barplots")

# arm key -> (display label, bar colour).  Order here is the plotting order.
ARM_STYLE = [
    ("baseline_1d",   "1D baseline",                 "#8A94A6"),
    ("D",             "D",                           "#5F4BB6"),
    ("D_linear",      "D − 1D,3D  (linear)",         "#8A79D6"),
    ("D_quadratic",   "D − 1D,3D  (quadratic)",      "#B3A8E3"),
    ("C",             "C",                           "#1F8A7B"),
    ("C_linear",      "C − 1D,3D  (linear)",         "#4FB3A4"),
    ("C_quadratic",   "C − 1D,3D  (quadratic)",      "#9BD9CE"),
    ("random",        "random control",              "#C9CED6"),
]

METRIC_LABEL = {
    "knn_purity": "kNN purity",
    "purity_index": "purity index  (chance-normalized)",
    "ari": "ARI",
    "nmi": "NMI",
    "silhouette": "silhouette (ASW)",
}


# ------------------------------------------------------------------- inputs --

def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    df = (pd.read_parquet(p) if p.suffix in (".parquet", ".pq")
          else pd.read_csv(p, sep="\t", index_col=0))
    return df.astype(float)


def _drop_chroms(df: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    if not exclude:
        return df
    keep = [c for c in df.columns if str(c).split(":", 1)[0] not in exclude]
    if len(keep) != df.shape[1]:
        logger.info(f"Excluded {df.shape[1] - len(keep)} bins on "
                    f"{sorted(exclude)}.")
    return df[keep]


def _shuffle_within_bins(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute each bin's finite values across cells.

    Preserves the per-bin value distribution and the NaN pattern exactly, so
    the control differs from the real matrix ONLY in the cell association.
    """
    rng = np.random.default_rng(seed)
    X = df.to_numpy(dtype=float).copy()
    for j in range(X.shape[1]):
        idx = np.flatnonzero(np.isfinite(X[:, j]))
        if idx.size > 1:
            X[idx, j] = X[rng.permutation(idx), j]
    return pd.DataFrame(X, index=df.index, columns=df.columns)


# ------------------------------------------------------------------ scoring --

def score_matrix(raw, name, labels, args, out_dir, batch_labels=None):
    """prep → SVD scores → label-agreement metrics, the nome_hic_compare way."""
    norm = comparison._prep_matrix(
        raw, name, str(out_dir), args.matrix_norm, args.min_cell_coverage,
        args.per_cell_norm, args.post_norm_recenter)
    if norm.shape[1] < comparison._MIN_BINS:
        logger.warning(f"[{name}] only {norm.shape[1]} bin(s) after the "
                       f"coverage filter — skipping.")
        return None
    common = norm.index.intersection(labels.index)
    scores, _ = comparison._svd_scores(
        norm.loc[common], args.n_svd, skip_top=args.skip_top_svd,
        random_state=args.random_seed)
    # nome_hic_compare scores WITHOUT batch correction; only apply it if the
    # user explicitly asks, and then on the same scores the metrics see.
    if batch_labels is not None and args.batch_correction != "none":
        from nome_hic_concordance import batch_correction as _bc
        if args.batch_correction == "harmony":
            bl = batch_labels.reindex(common).reset_index(drop=True)
            scores = _bc.harmonize_embedding(
                scores, bl, random_state=args.random_seed,
                theta=args.harmony_theta, n_threads=args.n_threads)
        else:
            scores = _bc.apply_batch_correction(
                pd.DataFrame(scores, index=common),
                batch_labels.reindex(common), method=args.batch_correction,
                random_state=args.random_seed).to_numpy()
    m = _metrics.label_agreement(scores, labels.loc[common], k=args.knn_k,
                                 random_state=args.random_seed,
                                 n_perm=args.n_perm)
    m["n_bins"] = float(norm.shape[1])
    return m


def build_arms(mats, labels, args, out_dir, batch_labels,
               concordance_batch_labels=None):
    """Every arm's metric dict, keyed as in ARM_STYLE."""
    results: dict[str, dict] = {}

    def correct(df, nm):
        """The concordance-stage correction, on the raw per-bin values.

        Only C and D get it, never the baselines: they come off a different
        stage, and correcting them with a setting tuned for concordance would
        weaken the very control these bars exist to measure against.  Same
        split, and the same shared implementation, as nome_hic_compare.
        """
        if (args.concordance_batch_correction == "none"
                or concordance_batch_labels is None):
            return df
        from nome_hic_concordance import batch_correction as _bc
        return _bc.correct_and_report(
            df, concordance_batch_labels, args.concordance_batch_correction,
            args.random_seed, nm)

    mats = dict(mats)
    src_raw = mats[args.random_source.upper()]      # before the correction
    mats["C"] = correct(mats["C"], "C")
    mats["D"] = correct(mats["D"], "D")

    results["baseline_1d"] = score_matrix(
        mats["1D"], "baseline_1d", labels, args, out_dir, batch_labels)
    for key, raw in (("C", mats["C"]), ("D", mats["D"])):
        results[key] = score_matrix(raw, key, labels, args, out_dir,
                                    batch_labels)

    # Residual arms.  variance_partition consumes the NORMALIZED target and
    # baselines (the order run_comparison uses), and residual_umap re-preps the
    # residual before scoring — score_matrix does that second prep.
    prep = lambda df, nm: comparison._prep_matrix(   # noqa: E731
        df, nm, str(out_dir), args.matrix_norm, args.min_cell_coverage,
        args.per_cell_norm, args.post_norm_recenter)
    base_norm = {"accessibility": prep(mats["1D"], "vp_1d"),
                 "compartment": prep(mats["3D"], "vp_3d")}
    for key, raw in (("C", mats["C"]), ("D", mats["D"])):
        tnorm = prep(raw, f"vp_target_{key}")
        for level in ("linear", "quadratic"):
            name = f"{key}_{level}"
            resid = comparison.variance_partition(
                tnorm, base_norm, name, str(out_dir / "variance_partition"),
                level_control=level, plot_format=args.plot_format,
                batch_labels=batch_labels)
            results[name] = score_matrix(resid, f"residual_{name}", labels,
                                         args, out_dir, batch_labels)

    # The floor has to travel the SAME road as the real arms, correction
    # included, or it stops being a floor: harmony in particular can invent
    # clustering, and only an arm that also passes through it can absorb that.
    # Shuffled FIRST, so the correction meets a matrix with no cell
    # association left, exactly as it would if the data carried no signal.
    results["random"] = score_matrix(
        correct(_shuffle_within_bins(src_raw, args.random_seed), "random"),
        "random", labels, args, out_dir, batch_labels)
    return results


# ----------------------------------------------------------------- plotting --

def barplot(results, metric, context, out_path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    rows = [(lab, col, (results.get(k) or {}).get(metric, np.nan))
            for k, lab, col in ARM_STYLE]
    labels = [r[0] for r in rows][::-1]         # first arm at the top
    colors = [r[1] for r in rows][::-1]
    vals = [r[2] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)

    finite = [v for v in vals if np.isfinite(v)]
    span = (max(finite) - min(min(finite), 0)) if finite else 1.0
    for yi, v in zip(y, vals):
        if np.isfinite(v):
            ax.text(v + 0.012 * max(span, 1e-9), yi, f"{v:.3f}",
                    va="center", fontsize=8.8, color="#23262B")
        else:
            ax.text(0, yi, "  n/a", va="center", fontsize=8.8, color="#9AA0A6")

    # kNN purity is uninterpretable without its chance level; ARI/NMI are
    # already referenced to 0, and the axvline at 0 is the spine there.
    if metric == "knn_purity":
        chance = np.nanmedian([
            (results.get(k) or {}).get("knn_chance", np.nan)
            for k, *_ in ARM_STYLE])
        if np.isfinite(chance):
            ax.axvline(chance, color="#B0453A", ls="--", lw=1.2,
                       label=f"chance = {chance:.3f}")
            ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.axvline(0, color="#9AA0A6", lw=0.8, zorder=0)

    ax.set_xlabel(METRIC_LABEL.get(metric, metric))
    ax.set_title(f"{context}: {METRIC_LABEL.get(metric, metric)} by feature "
                 f"matrix", fontsize=11.5, pad=16)
    sub = (f"SVD-{args.n_svd}"
           + (f", skip top {args.skip_top_svd}" if args.skip_top_svd else "")
           + f", {args.matrix_norm} norm, k = {args.knn_k}")
    ax.annotate(sub, xy=(0.5, 1.0), xytext=(0, 5), xycoords="axes fraction",
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8, color="#6B7280", style="italic")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------- main --

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concordance-c", required=True)
    ap.add_argument("--concordance-d", required=True)
    ap.add_argument("--baseline-1d", required=True)
    ap.add_argument("--baseline-3d", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-column", default="cell_name")
    ap.add_argument("--group-column", default="group")
    ap.add_argument("--batch-column", default=None,
                    help="Comma-separated batch columns, passed to the "
                         "variance-partition regression as nuisance "
                         "covariates (as nome_hic_compare does).")
    ap.add_argument("--context", default="GCH")
    ap.add_argument("--matrix-norm", choices=["quantile", "zscore"],
                    default="quantile")
    ap.add_argument("--per-cell-norm",
                    choices=["off", "center", "zscore", "rank"], default="rank")
    ap.add_argument("--post-norm-recenter", action="store_true")
    ap.add_argument("--n-svd", type=int, default=10)
    ap.add_argument("--skip-top-svd", type=int, default=0)
    ap.add_argument("--min-cell-coverage", type=float, default=0.0)
    ap.add_argument("--exclude-chrom", default="")
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--batch-correction",
                    choices=["none", "center", "zscore", "combat", "harmony"],
                    default="none",
                    help="Applied to the SVD scores BEFORE scoring.  Default "
                         "'none' reproduces nome_hic_compare, which batch-"
                         "corrects only the UMAP, not the reported metrics.")
    ap.add_argument("--concordance-batch-column", default=None,
                    help="Batch variable(s) for --concordance-batch-correction "
                         "when they should differ from --batch-column; "
                         "comma-separated.  Defaults to --batch-column.")
    ap.add_argument("--concordance-batch-correction",
                    choices=["none", "center", "zscore", "combat", "harmony"],
                    default="none",
                    help="Batch-correct the C and D matrices themselves, per "
                         "bin on their raw values, before the coverage filter "
                         "and every normalization.  A different stage from "
                         "--batch-correction, which acts on the SVD scores: "
                         "batch is already mixed into the concordance bins "
                         "(per-cell coverage drives the DM shrinkage and what "
                         "RWR imputes), and ten SVD components cannot separate "
                         "it out afterwards.  Same flag, same shared "
                         "implementation and same semantics as in "
                         "nome_hic_concordance and nome_hic_compare, so a "
                         "setting chosen there transfers here unchanged.  The "
                         "1D and 3D baselines are deliberately left alone — "
                         "they are what the concordance arms are measured "
                         "against.  The random control DOES get it, since a "
                         "floor that skipped the correction could not absorb "
                         "structure the correction itself creates.")
    ap.add_argument("--harmony-theta", type=float, default=None)
    ap.add_argument("--random-source", choices=["c", "d"], default="d",
                    help="Which concordance matrix the random control shuffles.")
    ap.add_argument("--metrics", default="knn_purity,ari,nmi",
                    help="Comma-separated; one PDF each.  Also available: "
                         "purity_index, silhouette.")
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
    mats = {"C": _load(args.concordance_c), "D": _load(args.concordance_d),
            "1D": _load(args.baseline_1d), "3D": _load(args.baseline_3d)}
    mats = {k: _drop_chroms(v, excl) for k, v in mats.items()}

    samples = pd.read_csv(args.samples, sep="\t")
    labels = samples.set_index(args.cell_column)[args.group_column].dropna()
    idx = samples.set_index(args.cell_column)

    def _batch_series(spec, flag):
        cols = [c.strip() for c in spec.split(",") if c.strip()]
        missing = [c for c in cols if c not in idx.columns]
        if missing:
            raise SystemExit(f"{flag} not in {args.samples}: {missing}")
        return idx[cols[0]] if len(cols) == 1 else idx[cols]

    batch_labels = (_batch_series(args.batch_column, "--batch-column")
                    if args.batch_column else None)
    # Falls back to --batch-column, so the common case stays one flag.
    concordance_batch_labels = (
        _batch_series(args.concordance_batch_column,
                      "--concordance-batch-column")
        if args.concordance_batch_column else batch_labels)
    if (args.concordance_batch_correction != "none"
            and concordance_batch_labels is None):
        raise SystemExit(
            f"--concordance-batch-correction "
            f"{args.concordance_batch_correction!r} requires --batch-column "
            f"or --concordance-batch-column.")
    logger.info(f"{len(labels)} labelled cells, "
                f"{labels.nunique()} groups; context {args.context}.")

    results = build_arms(mats, labels, args, out_dir, batch_labels,
                         concordance_batch_labels)

    tab = pd.DataFrame(
        [{"arm": key, "label": lab, **(results.get(key) or {})}
         for key, lab, _ in ARM_STYLE])
    tsv = out_dir / f"{args.context}_metrics.tsv"
    tab.to_csv(tsv, sep="\t", index=False)
    logger.info(f"wrote {tsv}")
    for _, r in tab.iterrows():
        logger.info(f"  {r['label']:28s} "
                    f"purity={r.get('knn_purity', float('nan')):.3f}  "
                    f"ARI={r.get('ari', float('nan')):.3f}  "
                    f"NMI={r.get('nmi', float('nan')):.3f}")

    for metric in [m.strip() for m in args.metrics.split(",") if m.strip()]:
        path = out_dir / f"{args.context}_{metric}_barplot.{args.plot_format}"
        barplot(results, metric, args.context, path, args)
        logger.info(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
