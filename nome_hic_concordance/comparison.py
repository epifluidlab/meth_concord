"""Comparison harness (design doc Part C / Phase 3).

Answers the question: does long-range GCH/HCG concordance carry cell-type
information *beyond* what mean accessibility and scHi-C compartments
already provide?

Four analyses (§4.2), each emitting PNG + TSV/parquet:

  (a) Marginal UMAPs        — per-matrix matrix_ops -> SVD -> UMAP, plus
                              silhouette + kNN-purity vs cell-type labels.
  (b) Variance-partition     — per bin, regress each concordance matrix on
                              the baselines (accessibility + compartment);
                              R² per bin + coefficients + residual matrix.
  (c) Residual UMAP          — UMAP of the residual; if it still separates
                              cell types, concordance carries unique info.
  (d) Classifier comparison  — cross-validated cell-type AUC for each
                              feature set: each matrix alone, the baselines,
                              baselines+each concordance, and all combined.
                              The headline 'additional info' number.

All matrices are cells × bins parquets.  Matrices may live on different
bin grids (e.g. concordance/accessibility at 250 kb, compartments at
1 Mb); the variance-partition step broadcasts coarser predictors onto the
target's finer grid.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import matrix_ops, visualization
from .blacklist import infer_bin_size
from .pair_concordance.component_analysis import _cv_auc
from .utils import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared prep: normalize a matrix and SVD-reduce it
# ---------------------------------------------------------------------------

def _safe_umap(*args, **kwargs):
    """Run the UMAP plot, but never let a missing umap-learn (or a plotting
    error) abort the quantitative analyses, which don't need it."""
    try:
        return visualization.run_umap_and_plot(*args, **kwargs)
    except ImportError as exc:
        logger.warning(
            f"UMAP plot skipped (umap-learn not installed: {exc}); "
            f"silhouette / kNN-purity / classifier still computed.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"UMAP plot failed ({exc}); continuing.")
    return None


def _prep_matrix(
    df: pd.DataFrame,
    name: str,
    output_dir: str,
    matrix_norm: str,
    min_cell_coverage: float,
) -> pd.DataFrame:
    """Coverage-filter + impute + per-bin normalize a cells × bins matrix.

    Reuses :func:`matrix_ops.process_concordance_matrix` so every feature
    matrix gets the identical treatment (fairness).  Returns the normalized
    cells × bins DataFrame ('zscore' key, regardless of matrix_norm).
    """
    res = matrix_ops.process_concordance_matrix(
        df, min_cell_coverage=min_cell_coverage,
        output_dir=str(Path(output_dir) / "prep" / name),
        matrix_norm=matrix_norm,
    )
    return res["zscore"]


def _svd_scores(normalized: pd.DataFrame, n_svd: int) -> tuple[np.ndarray, np.ndarray]:
    """Column-standardize then TruncatedSVD; return (scores, var_ratio)."""
    from sklearn.decomposition import TruncatedSVD

    X = normalized.to_numpy(dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (X - mu) / sd
    k = max(1, min(n_svd, Z.shape[0] - 1, Z.shape[1]))
    svd = TruncatedSVD(n_components=k, random_state=42).fit(Z)
    return svd.transform(Z), svd.explained_variance_ratio_


def _silhouette(scores: np.ndarray, labels: pd.Series) -> float:
    from sklearn.metrics import silhouette_score
    y = labels.to_numpy()
    if len(np.unique(y)) < 2 or scores.shape[0] < 3:
        return float("nan")
    try:
        return float(silhouette_score(scores, y))
    except Exception:
        return float("nan")


def _knn_purity(scores: np.ndarray, labels: pd.Series, k: int = 15) -> float:
    """Fraction of each cell's k nearest neighbours sharing its label."""
    from sklearn.neighbors import NearestNeighbors
    y = labels.to_numpy()
    n = scores.shape[0]
    if n < 3:
        return float("nan")
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(scores)
    _, idx = nn.kneighbors(scores)
    idx = idx[:, 1:]
    same = np.array([(y[idx[i]] == y[i]).mean() for i in range(n)])
    return float(same.mean())


# ---------------------------------------------------------------------------
# Bin-grid alignment (broadcast a coarser predictor onto a finer target)
# ---------------------------------------------------------------------------

def _parse_bin(label: str) -> tuple[str, int]:
    chrom, pos = label.rsplit(":", 1)
    return chrom, int(pos)


def align_to_grid(
    predictor: pd.DataFrame,
    target_cols: list[str],
) -> pd.DataFrame:
    """Reindex ``predictor`` onto ``target_cols`` (a finer/equal bin grid).

    Each target bin inherits the predictor value of the coarse bin that
    contains it (floor of the target position to the predictor's bin size).
    Target bins with no covering predictor bin become NaN.
    """
    pred_bin = infer_bin_size(list(predictor.columns))
    # Map (chrom, coarse_pos) -> predictor column.
    pred_lookup = {}
    for col in predictor.columns:
        c, p = _parse_bin(col)
        pred_lookup[(c, p)] = col

    src_cols = []
    for tcol in target_cols:
        c, p = _parse_bin(tcol)
        parent = (c, (p // pred_bin) * pred_bin)
        src_cols.append(pred_lookup.get(parent))

    out = pd.DataFrame(index=predictor.index, columns=target_cols, dtype=float)
    for tcol, scol in zip(target_cols, src_cols):
        if scol is not None:
            out[tcol] = predictor[scol].values
    return out


# ---------------------------------------------------------------------------
# (a) Marginal UMAPs + cluster metrics
# ---------------------------------------------------------------------------

def marginal_analysis(
    matrices: dict[str, pd.DataFrame],
    group_labels: pd.Series | None,
    output_dir: str,
    matrix_norm: str,
    n_svd: int,
    min_cell_coverage: float,
    random_state: int,
) -> dict[str, dict]:
    """Per-matrix UMAP + silhouette/kNN-purity.  Returns metrics per matrix."""
    out = ensure_dir(output_dir)
    metrics: dict[str, dict] = {}
    for name, df in matrices.items():
        logger.info(f"[marginal] {name}: {df.shape}")
        norm = _prep_matrix(df, name, output_dir, matrix_norm, min_cell_coverage)
        visualization.run_svd_scree(
            norm, output_prefix=f"svd_scree_{name}",
            output_dir=str(out), random_state=random_state,
        )
        _safe_umap(
            norm, group_labels, title=f"UMAP ({name})",
            output_prefix=f"umap_{name}", output_dir=str(out),
            n_svd=n_svd, n_threads=1, random_state=random_state,
        )
        m = {"n_cells": int(norm.shape[0]), "n_bins": int(norm.shape[1])}
        if group_labels is not None:
            common = norm.index.intersection(group_labels.index)
            scores, vr = _svd_scores(norm.loc[common], n_svd)
            lab = group_labels.loc[common]
            m["silhouette"] = _silhouette(scores, lab)
            m["knn_purity"] = _knn_purity(scores, lab)
            m["top_svd_var_frac"] = float(vr[0])
        metrics[name] = m
    return metrics


# ---------------------------------------------------------------------------
# (b) Per-bin variance-partition regression + (c) residual UMAP
# ---------------------------------------------------------------------------

def variance_partition(
    target: pd.DataFrame,
    predictors: dict[str, pd.DataFrame],
    target_name: str,
    output_dir: str,
) -> pd.DataFrame:
    """Per-bin OLS of ``target`` on ``predictors`` (aligned to target grid).

    Writes regression_R2_per_bin_{target}.parquet, the R² histogram, and
    delta_residual_{target}.parquet.  Returns the residual matrix
    (cells × bins) — target minus its fit from the predictors.
    """
    out = ensure_dir(output_dir)
    cells = target.index
    bins = list(target.columns)

    # Align predictors onto the target grid + common cells.
    aligned = {}
    for pname, pdf in predictors.items():
        a = align_to_grid(pdf.reindex(index=cells), bins)
        aligned[pname] = a

    Y = target.to_numpy(dtype=np.float64)
    n_cells, n_bins = Y.shape
    pred_arrays = {pn: a.to_numpy(dtype=np.float64) for pn, a in aligned.items()}

    r2 = np.full(n_bins, np.nan)
    coefs = {pn: np.full(n_bins, np.nan) for pn in predictors}
    residual = Y.copy()

    for j in range(n_bins):
        y = Y[:, j]
        cols = [pred_arrays[pn][:, j] for pn in predictors]
        X = np.column_stack(cols) if cols else np.empty((n_cells, 0))
        # Drop cells with any NaN in y or X for this bin.
        good = np.isfinite(y) & np.all(np.isfinite(X), axis=1) if X.size else np.isfinite(y)
        if good.sum() < (X.shape[1] + 2) or np.nanstd(y[good]) < 1e-12:
            continue
        Xg = np.column_stack([np.ones(good.sum()), X[good]])
        yg = y[good]
        beta, *_ = np.linalg.lstsq(Xg, yg, rcond=None)
        yhat = Xg @ beta
        ss_res = float(np.sum((yg - yhat) ** 2))
        ss_tot = float(np.sum((yg - yg.mean()) ** 2))
        r2[j] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        for pi, pn in enumerate(predictors):
            coefs[pn][j] = beta[1 + pi]
        # Residual for the good cells; leave others as the original (centered).
        residual[good, j] = yg - yhat

    reg_df = pd.DataFrame({"bin": bins, "R2": r2,
                           **{f"beta_{pn}": coefs[pn] for pn in predictors}})
    reg_df.to_parquet(out / f"regression_R2_per_bin_{target_name}.parquet")

    residual_df = pd.DataFrame(residual, index=cells, columns=bins)
    residual_df.to_parquet(out / f"delta_residual_{target_name}.parquet")

    _r2_hist(r2, target_name, out)
    med = float(np.nanmedian(r2))
    logger.info(
        f"[variance-partition] {target_name} ~ {'+'.join(predictors)}: "
        f"median R² = {med:.3f} "
        f"(low R² ⇒ concordance is NOT explained by the baselines)."
    )
    return residual_df


def _r2_hist(r2: np.ndarray, target_name: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vals = r2[np.isfinite(r2)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vals, bins=40, color="steelblue", edgecolor="none")
    ax.axvline(float(np.median(vals)) if vals.size else 0.0,
               color="firebrick", ls="--",
               label=f"median {np.median(vals):.3f}" if vals.size else "n/a")
    ax.set_xlabel(f"per-bin R²  ({target_name} ~ baselines)")
    ax.set_ylabel("bins")
    ax.set_title(f"How much of {target_name} the baselines explain")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / f"regression_R2_hist_{target_name}.png", dpi=150)
    plt.close(fig)


def residual_umap(
    residual: pd.DataFrame,
    group_labels: pd.Series | None,
    target_name: str,
    output_dir: str,
    matrix_norm: str,
    n_svd: int,
    min_cell_coverage: float,
    random_state: int,
) -> dict:
    """UMAP + cluster metrics on the residual matrix."""
    out = ensure_dir(output_dir)
    norm = _prep_matrix(residual, f"residual_{target_name}", output_dir,
                        matrix_norm, min_cell_coverage)
    _safe_umap(
        norm, group_labels, title=f"UMAP (residual: {target_name} | baselines)",
        output_prefix=f"umap_residual_{target_name}", output_dir=str(out),
        n_svd=n_svd, n_threads=1, random_state=random_state,
    )
    m = {"n_bins": int(norm.shape[1])}
    if group_labels is not None:
        common = norm.index.intersection(group_labels.index)
        scores, _ = _svd_scores(norm.loc[common], n_svd)
        lab = group_labels.loc[common]
        m["silhouette"] = _silhouette(scores, lab)
        m["knn_purity"] = _knn_purity(scores, lab)
    return m


# ---------------------------------------------------------------------------
# (d) Cross-validated classifier comparison
# ---------------------------------------------------------------------------

def classifier_comparison(
    matrices: dict[str, pd.DataFrame],
    group_labels: pd.Series,
    baseline_names: list[str],
    concordance_names: list[str],
    output_dir: str,
    matrix_norm: str,
    n_svd: int,
    min_cell_coverage: float,
    n_folds: int,
    random_state: int,
) -> pd.DataFrame:
    """CV cell-type AUC for individual + combined feature sets.

    The headline rows: 'baselines' vs 'baselines+<concordance>' and 'all'.
    Each feature set is the concatenation of per-matrix SVD scores.
    """
    out = ensure_dir(output_dir)
    # Per-matrix SVD feature blocks on the common cells.
    common = group_labels.index
    for df in matrices.values():
        common = common.intersection(df.index)
    common = sorted(common)
    y = group_labels.loc[common].to_numpy()

    blocks: dict[str, np.ndarray] = {}
    for name, df in matrices.items():
        norm = _prep_matrix(df, name, output_dir, matrix_norm, min_cell_coverage)
        scores, _ = _svd_scores(norm.loc[common], n_svd)
        blocks[name] = scores

    def _auc(names: list[str]) -> tuple[float, float]:
        X = np.concatenate([blocks[n] for n in names], axis=1)
        return _cv_auc(X, y, n_folds=n_folds, random_state=random_state)

    rows = []

    def _add(label, names):
        mean, std = _auc(names)
        rows.append({"feature_set": label, "matrices": "+".join(names),
                     "auc_mean": mean, "auc_std": std, "n_features": sum(
                         blocks[n].shape[1] for n in names)})
        return mean

    # Individual matrices.
    for name in matrices:
        _add(name, [name])
    # Baselines combined (if >1 baseline present).
    present_baselines = [b for b in baseline_names if b in matrices]
    base_auc = None
    if present_baselines:
        base_auc = _add("baselines", present_baselines)
    # Baselines + each concordance.
    for cn in concordance_names:
        if cn in matrices and present_baselines:
            _add(f"baselines+{cn}", present_baselines + [cn])
    # All matrices.
    if len(matrices) > 1:
        _add("all", list(matrices.keys()))

    tab = pd.DataFrame(rows)
    # Gain of each concordance over the baselines.
    if base_auc is not None:
        gains = []
        for cn in concordance_names:
            row = tab.loc[tab["feature_set"] == f"baselines+{cn}"]
            if len(row):
                gains.append({"concordance": cn,
                              "auc_gain_over_baselines":
                                  float(row["auc_mean"].iloc[0]) - base_auc})
        if gains:
            pd.DataFrame(gains).to_csv(
                out / "classifier_auc_gain.tsv", sep="\t", index=False)
    tab.to_csv(out / "classifier_comparison.tsv", sep="\t", index=False)
    logger.info(f"[classifier] wrote {out / 'classifier_comparison.tsv'}")
    return tab


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_comparison(
    matrices: dict[str, pd.DataFrame],
    group_labels: pd.Series | None,
    baseline_names: list[str],
    concordance_names: list[str],
    output_dir: str,
    matrix_norm: str = "quantile",
    n_svd: int = 10,
    min_cell_coverage: float = 0.5,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Run the full §4.2 suite; write summary.json; return the headline dict."""
    out = ensure_dir(output_dir)
    # Restrict every matrix to their common cells (+ labels).
    common = None
    for df in matrices.values():
        common = df.index if common is None else common.intersection(df.index)
    if group_labels is not None:
        common = common.intersection(group_labels.index)
    common = sorted(common)
    if len(common) < 3:
        raise ValueError(
            f"Only {len(common)} shared cells across the input matrices "
            f"(need >= 3); check that cell names match.")
    matrices = {n: df.loc[common] for n, df in matrices.items()}
    labels = group_labels.loc[common] if group_labels is not None else None
    logger.info(f"Comparison on {len(common)} shared cells, "
                f"matrices: {list(matrices)}")

    summary: dict = {"n_cells": len(common), "matrices": list(matrices)}

    # (a) Marginal.
    summary["marginal"] = marginal_analysis(
        matrices, labels, output_dir, matrix_norm, n_svd,
        min_cell_coverage, random_state)

    # (b)+(c) Variance partition + residual UMAP for each concordance matrix.
    present_baselines = [b for b in baseline_names if b in matrices]
    summary["residual"] = {}
    if present_baselines:
        base_norm = {b: _prep_matrix(matrices[b], b, output_dir, matrix_norm,
                                     min_cell_coverage) for b in present_baselines}
        for cn in concordance_names:
            if cn not in matrices:
                continue
            tnorm = _prep_matrix(matrices[cn], cn, output_dir, matrix_norm,
                                 min_cell_coverage)
            resid = variance_partition(tnorm, base_norm, cn, output_dir)
            summary["residual"][cn] = residual_umap(
                resid, labels, cn, output_dir, matrix_norm, n_svd,
                min_cell_coverage, random_state)
    else:
        logger.warning("No baseline matrices present; skipping (b)/(c).")

    # (d) Classifier comparison.
    if labels is not None:
        tab = classifier_comparison(
            matrices, labels, baseline_names, concordance_names, output_dir,
            matrix_norm, n_svd, min_cell_coverage, n_folds, random_state)
        summary["classifier"] = tab.to_dict(orient="records")
    else:
        logger.warning("No --samples group labels; skipping classifier + metrics.")

    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    logger.info(f"Wrote {out / 'summary.json'}")
    return summary
