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

from . import matrix_ops, metrics as _metrics, visualization
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
    per_cell_norm: str = "off",
    post_norm_recenter: bool = False,
) -> pd.DataFrame:
    """Coverage-filter + impute + (optional per-cell) + per-bin normalize a
    cells × bins matrix.

    Reuses :func:`matrix_ops.process_concordance_matrix` so every feature
    matrix gets the identical treatment (fairness).  ``per_cell_norm`` and
    ``post_norm_recenter`` default to off, reproducing the historical
    compare behaviour; set them to mirror the nome_hic_concordance step-7 stack
    (per-cell rank + recenter).  Returns the normalized cells × bins DataFrame
    ('zscore' key, regardless of matrix_norm).
    """
    res = matrix_ops.process_concordance_matrix(
        df, min_cell_coverage=min_cell_coverage,
        output_dir=str(Path(output_dir) / "prep" / name),
        matrix_norm=matrix_norm,
        per_cell_norm=per_cell_norm,
        post_norm_recenter=post_norm_recenter,
    )
    return res["zscore"]


# A feature matrix needs at least this many bins (after the coverage filter)
# to be usable for SVD/UMAP/classification.  Concordance computed WITHOUT RWR
# imputation at fine resolution is so sparse that almost no bin clears the
# coverage filter (e.g. 100-kb no-RWR concordance → 1–28 bins), which would
# otherwise crash TruncatedSVD; such matrices are skipped with a warning.
_MIN_BINS = 2


def _svd_scores(normalized: pd.DataFrame, n_svd: int,
                skip_top: int = 0,
                random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Column-standardize then TruncatedSVD; return (scores, var_ratio).

    ``skip_top`` drops the leading component(s) from the returned SCORES — the
    per-cell depth / batch axis — matching ``run_umap_and_plot(skip_top_svd=)``.
    At least one component is always kept.  ``var_ratio`` is deliberately NOT
    sliced, so ``top_svd_var_frac`` still reports what the dropped component
    carried.
    """
    from sklearn.decomposition import TruncatedSVD

    X = normalized.to_numpy(dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (X - mu) / sd
    k = max(1, min(n_svd, Z.shape[0] - 1, Z.shape[1]))
    svd = TruncatedSVD(n_components=k, random_state=random_state).fit(Z)
    s = min(max(int(skip_top), 0), max(k - 1, 0))
    return svd.transform(Z)[:, s:], svd.explained_variance_ratio_


# Single implementation lives in `metrics` (visualization needs it too, and
# this module imports visualization — so it cannot be the other way round).
_silhouette = _metrics.silhouette
_knn_purity = _metrics.knn_purity


def _knn_purity_stats(
    scores: np.ndarray, labels: pd.Series, k: int = 15,
    n_perm: int = 500, seed: int = 0,
) -> dict:
    """kNN-purity WITH its null, so an absolute value is interpretable.

    A raw purity is meaningless without a reference: under random neighbour
    assignment the expected purity is ``chance = Σ_g p_g²`` (= 0.5 only for a
    balanced 2-class problem; higher when imbalanced, lower with more
    classes).  We also run a label-permutation null on the *fixed* neighbour
    graph (relabel only — no refit) to get a calibrated p-value and z-score.

    Returns {purity, chance, perm_mean, perm_sd, z, perm_p, excess}.
    ``perm_p`` is one-sided (P[null purity >= observed]); ``excess`` is
    ``purity − chance``.
    """
    from sklearn.neighbors import NearestNeighbors
    nan = float("nan")
    y = labels.to_numpy()
    n = scores.shape[0]
    if n < 3 or len(np.unique(y)) < 2:
        return {"purity": nan, "chance": nan, "perm_mean": nan,
                "perm_sd": nan, "z": nan, "perm_p": nan, "excess": nan}
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(scores)
    _, idx = nn.kneighbors(scores)
    idx = idx[:, 1:]                                  # drop self
    obs = float((y[idx] == y[:, None]).mean())
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    chance = float(np.sum(p * p))
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm), dtype=np.float64)
    for t in range(int(n_perm)):
        yp = rng.permutation(y)                       # reuse fixed graph
        null[t] = (yp[idx] == yp[:, None]).mean()
    perm_mean = float(null.mean())
    perm_sd = float(null.std(ddof=1)) if n_perm > 1 else nan
    z = float((obs - perm_mean) / perm_sd) if perm_sd and perm_sd > 0 else nan
    perm_p = float((np.sum(null >= obs) + 1) / (int(n_perm) + 1))
    return {"purity": obs, "chance": chance, "perm_mean": perm_mean,
            "perm_sd": perm_sd, "z": z, "perm_p": perm_p,
            "excess": obs - chance}


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
    group_colors: dict | None = None,
    group_order: list | None = None,
    skip_top_svd: dict | None = None,
    per_cell_norm: str = "off",
    post_norm_recenter: bool = False,
    plot_format: str = "png",
    batch_labels=None,
    batch_correction: str = "none",
    harmony_theta=None,
) -> dict[str, dict]:
    """Per-matrix UMAP + silhouette/kNN-purity.  Returns metrics per matrix."""
    skip_map = skip_top_svd or {}
    out = ensure_dir(output_dir)
    metrics: dict[str, dict] = {}
    for name, df in matrices.items():
        logger.info(f"[marginal] {name}: {df.shape}")
        norm = _prep_matrix(df, name, output_dir, matrix_norm, min_cell_coverage,
                            per_cell_norm, post_norm_recenter)
        if norm.shape[1] < _MIN_BINS:
            logger.warning(
                f"[marginal] {name}: only {norm.shape[1]} bin(s) survive the "
                f"coverage filter — skipping SVD/UMAP/metrics.  This matrix is "
                f"too sparse to analyze (typical of concordance without "
                f"--impute rwr at fine resolution; use coarser bins).")
            metrics[name] = {"n_cells": int(norm.shape[0]),
                             "n_bins": int(norm.shape[1]),
                             "skipped": "too_few_bins_after_coverage_filter"}
            continue
        visualization.run_svd_scree(
            norm, output_prefix=f"svd_scree_{name}",
            output_dir=str(out), random_state=random_state,
            plot_format=plot_format,
        )
        skip = int(skip_map.get(name, 0))
        _safe_umap(
            norm, group_labels, title=f"UMAP ({name})",
            output_prefix=f"umap_{name}", output_dir=str(out),
            n_svd=n_svd, skip_top_svd=skip, n_threads=1,
            random_state=random_state,
            group_colors=group_colors, group_order=group_order,
            plot_format=plot_format,
            batch_labels=batch_labels, batch_correction=batch_correction,
            harmony_theta=harmony_theta,
        )
        m = {"n_cells": int(norm.shape[0]), "n_bins": int(norm.shape[1]),
             "skip_top_svd": skip}
        if group_labels is not None:
            common = norm.index.intersection(group_labels.index)
            scores, vr = _svd_scores(norm.loc[common], n_svd, skip_top=skip,
                                     random_state=random_state)
            lab = group_labels.loc[common]
            m["silhouette"] = _silhouette(scores, lab)
            ks = _knn_purity_stats(scores, lab)
            m["knn_purity"] = ks["purity"]
            m["knn_chance"] = ks["chance"]
            m["knn_excess"] = ks["excess"]
            m["knn_z"] = ks["z"]
            m["knn_perm_p"] = ks["perm_p"]
            m["top_svd_var_frac"] = float(vr[0])
        metrics[name] = m
    return metrics


# ---------------------------------------------------------------------------
# (b) Per-bin variance-partition regression + (c) residual UMAP
# ---------------------------------------------------------------------------

def _design_matrix(Xcols: np.ndarray, level_control: str) -> np.ndarray:
    """Per-bin regression design from predictor columns (n_cells × n_pred).

    Predictors are standardized (center + scale) for numerical stability —
    this leaves OLS fitted values / R² / residuals unchanged vs raw
    predictors (only coefficients rescale), so 'linear' reproduces the
    plain OLS exactly.  'quadratic' adds the squares + pairwise
    interactions, removing any *nonlinear* dependence on the levels so the
    residual cannot retain a curved level relationship masquerading as
    independent coordination signal.
    """
    n, p = Xcols.shape
    parts = [np.ones((n, 1))]
    if p == 0:
        return parts[0]
    mu = Xcols.mean(axis=0, keepdims=True)
    sd = Xcols.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (Xcols - mu) / sd
    parts.append(Z)
    if level_control == "quadratic":
        parts.append(Z ** 2)
        for a in range(p):
            for b in range(a + 1, p):
                parts.append((Z[:, a] * Z[:, b]).reshape(-1, 1))
    elif level_control != "linear":
        raise ValueError(
            f"level_control must be 'linear' or 'quadratic'; got {level_control!r}")
    return np.column_stack(parts)


def _batch_design(batch_labels, cells: pd.Index):
    """One-hot batch design (n_cells, k), first level dropped, mean-centred.

    Added to the residual regression as a NUISANCE covariate so the residual is
    'concordance not explained by 1D + 3D + BATCH', which is what the figure
    actually claims.  Without it the residual regresses concordance on the RAW,
    batch-contaminated baselines and their batch leaks straight back into the
    residual -- the reason a residual UMAP can be MORE batch-split than the
    concordance it came from.
    """
    if batch_labels is None:
        return None
    df = (batch_labels.to_frame() if isinstance(batch_labels, pd.Series)
          else pd.DataFrame(batch_labels))
    df = df.reindex(cells).astype(str)
    parts = []
    for c in df.columns:
        d = pd.get_dummies(df[c], prefix=str(c), drop_first=True, dummy_na=False)
        if d.shape[1]:
            parts.append(d)
    if not parts:
        return None
    B = pd.concat(parts, axis=1).to_numpy(dtype=np.float64)
    return B - B.mean(axis=0, keepdims=True)      # centre: leave intercept free


def _corr(a, b):
    """Pearson r over finite pairs; NaN if <3 pairs or either side constant."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return np.nan
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def variance_partition(
    target: pd.DataFrame,
    predictors: dict[str, pd.DataFrame],
    target_name: str,
    output_dir: str,
    level_control: str = "linear",
    plot_format: str = "png",
    batch_labels=None,
) -> pd.DataFrame:
    """Per-bin OLS of ``target`` on ``predictors`` (aligned to target grid).

    Writes regression_R2_per_bin_{target}.parquet, the R² histogram, and
    delta_residual_{target}.parquet.  Returns the residual matrix
    (cells × bins) — target minus its fit from the predictors.

    ``level_control='quadratic'`` expands each bin's design with the
    predictors' squares + pairwise interactions, so the residual is
    cleansed of *nonlinear* level dependence (the conservative control a
    reviewer will ask for).
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

    B = _batch_design(batch_labels, cells)          # nuisance covariate or None
    r2 = np.full(n_bins, np.nan)
    coefs = {pn: np.full(n_bins, np.nan) for pn in predictors}
    residual = Y.copy()
    # Per-bin corr with each baseline: concordance (before) vs the RAW residual
    # (after), over the cells actually fit, for the decorrelation QC plot.
    corr_conc = {pn: np.full(n_bins, np.nan) for pn in predictors}
    corr_resid = {pn: np.full(n_bins, np.nan) for pn in predictors}

    for j in range(n_bins):
        y = Y[:, j]
        cols = [pred_arrays[pn][:, j] for pn in predictors]
        X = np.column_stack(cols) if cols else np.empty((n_cells, 0))
        # Drop cells with any NaN in y or X for this bin.
        good = np.isfinite(y) & np.all(np.isfinite(X), axis=1) if X.size else np.isfinite(y)
        if np.nanstd(y[good]) < 1e-12:
            continue
        Xg = _design_matrix(X[good] if X.size else np.empty((int(good.sum()), 0)),
                            level_control)
        if good.sum() < Xg.shape[1] + 1:
            continue
        yg = y[good]
        # R² and the reported coefficients come from the BASELINE-ONLY fit, so
        # "1D/3D explain X%" keeps its meaning.
        beta, *_ = np.linalg.lstsq(Xg, yg, rcond=None)
        yhat = Xg @ beta
        ss_tot = float(np.sum((yg - yg.mean()) ** 2))
        r2[j] = 1.0 - float(np.sum((yg - yhat) ** 2)) / ss_tot if ss_tot > 0 else np.nan
        for pi, pn in enumerate(predictors):
            coefs[pn][j] = beta[1 + pi]
        # The RESIDUAL additionally regresses out batch, so it is what 1D + 3D
        # AND batch leave behind.  Batch enters only here, never in the R²/coef
        # accounting above.
        yhat_r = yhat
        if B is not None:
            Xf = np.column_stack([Xg, B[good]])
            if good.sum() >= Xf.shape[1] + 1:
                bf, *_ = np.linalg.lstsq(Xf, yg, rcond=None)
                yhat_r = Xf @ bf
        res_j = yg - yhat_r
        residual[good, j] = res_j
        # corr with each baseline, over the fitted cells: concordance vs the raw
        # residual.  The residual is the OLS residual, so it is orthogonal to the
        # baselines by construction (after ~0) — the QC that the linear 1D/3D
        # dependence is gone; the concordance (before) spread shows how much.
        for pn in predictors:
            xj = pred_arrays[pn][good, j]
            corr_conc[pn][j] = _corr(yg, xj)
            corr_resid[pn][j] = _corr(res_j, xj)

    reg_df = pd.DataFrame({"bin": bins, "R2": r2,
                           **{f"beta_{pn}": coefs[pn] for pn in predictors}})
    reg_df.to_parquet(out / f"regression_R2_per_bin_{target_name}.parquet")

    residual_df = pd.DataFrame(residual, index=cells, columns=bins)
    residual_df.to_parquet(out / f"delta_residual_{target_name}.parquet")

    # Decorrelation QC: per-bin correlation with each baseline, concordance vs
    # residual (parquet + before/after density plot).
    dec = {"bin": bins}
    for pn in predictors:
        dec[f"corr_concordance_{pn}"] = corr_conc[pn]
        dec[f"corr_residual_{pn}"] = corr_resid[pn]
    pd.DataFrame(dec).to_parquet(
        out / f"baseline_decorrelation_{target_name}.parquet")
    _decorr_density(corr_conc, corr_resid, target_name, out, plot_format)
    for pn in predictors:
        mb = float(np.nanmedian(np.abs(corr_conc[pn])))
        ma = float(np.nanmedian(np.abs(corr_resid[pn])))
        logger.info(f"[decorrelation] {target_name} vs {pn}: per-bin median |r| "
                    f"{mb:.3f} (concordance) → {ma:.3f} (residual).")

    _r2_hist(r2, target_name, out, plot_format)
    med = float(np.nanmedian(r2))
    logger.info(
        f"[variance-partition] {target_name} ~ {'+'.join(predictors)} "
        f"({level_control}): median R² = {med:.3f} "
        f"(low R² ⇒ concordance is NOT explained by the baselines)."
    )
    return residual_df


def _r2_hist(r2: np.ndarray, target_name: str, out: Path,
             plot_format: str = "png") -> None:
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
    plt.savefig(out / f"regression_R2_hist_{target_name}.{plot_format}", dpi=150)
    plt.close(fig)


# Display labels for the baselines: mean accessibility/methylation is the 1D
# signal, scHi-C compartment the 3D signal.
_BASELINE_LABEL = {"accessibility": "1D", "compartment": "3D"}


def _decorr_density(before, after, target_name, out, plot_format="png"):
    """Density of per-bin corr(matrix, baseline): concordance vs residual.

    One panel per baseline.  ``before`` (blue) is the per-bin correlation of the
    concordance with the baseline; ``after`` (red) is the correlation of the RAW
    regression residual (concordance minus its least-squares fit on the
    baselines).  The residual is orthogonal to the baselines by construction, so
    ``after`` sits at ~0 — the QC that the linear 1D/3D dependence has been
    removed — while ``before`` shows how much correlation there was.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pns = list(before)
    if not pns:
        return
    fig, axes = plt.subplots(1, len(pns), figsize=(5.4 * len(pns), 4.4),
                             squeeze=False)
    edges = np.linspace(-1, 1, 61)
    grid = np.linspace(-1, 1, 400)
    for ax, pn in zip(axes[0], pns):
        disp = _BASELINE_LABEL.get(pn, pn)
        for vals, col, lab in ((before[pn], "#4C72B0", "concordance"),
                               (after[pn], "#C44E52", "residual")):
            v = vals[np.isfinite(vals)]
            if v.size == 0:
                continue
            ax.hist(v, bins=edges, density=True, color=col, alpha=0.45,
                    label=f"{lab}  (median |r|={np.median(np.abs(v)):.3f}, "
                          f"sd={v.std():.3f})")
            if v.std() > 0.02 and v.size > 5:      # smooth curve when spread
                try:
                    from scipy.stats import gaussian_kde
                    ax.plot(grid, gaussian_kde(v)(grid), color=col, lw=1.8)
                except Exception:
                    pass
        ax.axvline(0, color="0.35", lw=1.0, ls="--")
        ax.set_xlim(-1, 1)
        ax.set_xlabel(f"per-bin Pearson r  vs  {disp} baseline")
        ax.set_ylabel("density (over bins)")
        ax.set_title(disp)
        ax.legend(fontsize=8)
    fig.suptitle(f"{target_name}: per-bin correlation with baseline — "
                 f"concordance vs residual", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"baseline_decorrelation_{target_name}.{plot_format}",
                dpi=150, bbox_inches="tight")
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
    group_colors: dict | None = None,
    group_order: list | None = None,
    skip_top: int = 0,
    per_cell_norm: str = "off",
    post_norm_recenter: bool = False,
    plot_format: str = "png",
    batch_labels=None,
    batch_correction: str = "none",
    harmony_theta=None,
) -> dict:
    """UMAP + cluster metrics on the residual matrix."""
    out = ensure_dir(output_dir)
    norm = _prep_matrix(residual, f"residual_{target_name}", output_dir,
                        matrix_norm, min_cell_coverage,
                        per_cell_norm, post_norm_recenter)
    if norm.shape[1] < _MIN_BINS:
        logger.warning(f"[residual] {target_name}: {norm.shape[1]} bin(s) — "
                       f"skipping residual UMAP/metrics.")
        return {"n_bins": int(norm.shape[1]), "skipped": "too_few_bins"}
    _safe_umap(
        norm, group_labels, title=f"UMAP (residual: {target_name} | baselines)",
        output_prefix=f"umap_residual_{target_name}", output_dir=str(out),
        n_svd=n_svd, skip_top_svd=int(skip_top), n_threads=1,
        random_state=random_state,
        group_colors=group_colors, group_order=group_order,
        plot_format=plot_format,
        batch_labels=batch_labels, batch_correction=batch_correction,
        harmony_theta=harmony_theta,
    )
    m = {"n_bins": int(norm.shape[1]), "skip_top_svd": int(skip_top)}
    if group_labels is not None:
        common = norm.index.intersection(group_labels.index)
        scores, _ = _svd_scores(norm.loc[common], n_svd, skip_top=int(skip_top),
                                random_state=random_state)
        lab = group_labels.loc[common]
        m["silhouette"] = _silhouette(scores, lab)
        ks = _knn_purity_stats(scores, lab)
        m["knn_purity"] = ks["purity"]
        m["knn_chance"] = ks["chance"]
        m["knn_excess"] = ks["excess"]
        m["knn_z"] = ks["z"]
        m["knn_perm_p"] = ks["perm_p"]
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
    skip_top_svd: dict | None = None,
    per_cell_norm: str = "off",
    post_norm_recenter: bool = False,
) -> pd.DataFrame:
    """CV cell-type AUC for individual + combined feature sets.

    The headline rows: 'baselines' vs 'baselines+<concordance>' and 'all'.
    Each feature set is the concatenation of per-matrix SVD scores, so a
    per-matrix ``skip_top_svd`` applies unambiguously to that matrix's block.
    """
    skip_map = skip_top_svd or {}
    out = ensure_dir(output_dir)
    # Per-matrix SVD feature blocks on the common cells.
    common = group_labels.index
    for df in matrices.values():
        common = common.intersection(df.index)
    common = sorted(common)
    y = group_labels.loc[common].to_numpy()

    blocks: dict[str, np.ndarray] = {}
    for name, df in matrices.items():
        norm = _prep_matrix(df, name, output_dir, matrix_norm, min_cell_coverage,
                            per_cell_norm, post_norm_recenter)
        if norm.shape[1] < _MIN_BINS:
            logger.warning(
                f"[classifier] {name}: only {norm.shape[1]} bin(s) after "
                f"coverage filter — excluding from classification (too sparse).")
            continue
        scores, _ = _svd_scores(norm.loc[common], n_svd,
                                skip_top=int(skip_map.get(name, 0)),
                                random_state=random_state)
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

    # Individual matrices (only those that survived).
    for name in blocks:
        _add(name, [name])
    # Baselines combined (if >1 baseline present).
    present_baselines = [b for b in baseline_names if b in blocks]
    base_auc = None
    if present_baselines:
        base_auc = _add("baselines", present_baselines)
    # Baselines + each concordance.
    for cn in concordance_names:
        if cn in blocks and present_baselines:
            _add(f"baselines+{cn}", present_baselines + [cn])
    # Concordance added to the 3D-genome baseline ALONE — scHi-C compartments
    # are typically far from ceiling, so this isolates "does concordance add
    # cell-type info beyond scHi-C?" with real headroom (unlike the
    # level-saturated full-baseline gain).
    if "compartment" in blocks:
        for cn in concordance_names:
            if cn in blocks:
                _add(f"compartment+{cn}", ["compartment", cn])
    # All matrices.
    if len(blocks) > 1:
        _add("all", list(blocks.keys()))

    tab = pd.DataFrame(rows)
    # Gain of each concordance over the baselines.
    if base_auc is not None:
        gains = []
        for cn in concordance_names:
            if cn not in blocks:
                continue
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


def classifier_comparison_pairwise(
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
    min_class_cells: int = 6,
    skip_top_svd: dict | None = None,
    per_cell_norm: str = "off",
    post_norm_recenter: bool = False,
) -> pd.DataFrame:
    """One-vs-one CV AUC: baselines vs baselines+concordance, per cell-type
    PAIR — the headroom-sensitive, leakage-robust version of (d).

    When marginal levels saturate the aggregate AUC (ceiling), concordance
    cannot show a gain there even if it carries unique information.  This
    restricts to each pair of cell types and asks whether concordance lifts
    the AUC of the pairs the baselines *confuse*.  Improving a pair the
    baselines already fail on is the additional-info signal the aggregate
    hides — and it is largely leakage-robust: structure leaked into
    concordance from a baseline's smoothing graph cannot separate cells that
    baseline itself cannot separate.

    Writes classifier_pairwise_auc.tsv (one row per pair, sorted by baseline
    difficulty) and returns it.
    """
    import itertools
    out = ensure_dir(output_dir)
    common = group_labels.index
    for df in matrices.values():
        common = common.intersection(df.index)
    common = sorted(common)
    labels = group_labels.loc[common]
    cells_index = pd.Index(common)

    # Per-matrix SVD blocks on the full common set (same feature space as the
    # aggregate run); rows are subset per pair below.
    blocks: dict[str, pd.DataFrame] = {}
    for name, df in matrices.items():
        norm = _prep_matrix(df, name, output_dir, matrix_norm, min_cell_coverage,
                            per_cell_norm, post_norm_recenter)
        if norm.shape[1] < _MIN_BINS:
            continue                              # too sparse — skip (warned elsewhere)
        scores, _ = _svd_scores(norm.loc[common], n_svd,
                                skip_top=int((skip_top_svd or {}).get(name, 0)),
                                random_state=random_state)
        blocks[name] = pd.DataFrame(scores, index=cells_index)

    present_baselines = [b for b in baseline_names if b in blocks]
    conc = [c for c in concordance_names if c in blocks]
    if not present_baselines or not conc:
        return pd.DataFrame()
    classes = sorted(pd.unique(labels))

    rows = []
    for g1, g2 in itertools.combinations(classes, 2):
        sel = labels[labels.isin([g1, g2])].index
        y = labels.loc[sel].to_numpy()
        cnt = pd.Series(y).value_counts()
        if cnt.min() < min_class_cells:
            continue

        def _auc(names):
            X = np.concatenate([blocks[n].loc[sel].to_numpy() for n in names],
                               axis=1)
            return _cv_auc(X, y, n_folds=n_folds, random_state=random_state)

        b_mean, _ = _auc(present_baselines)
        row = {"pair": f"{g1}|{g2}", "n1": int(cnt.get(g1, 0)),
               "n2": int(cnt.get(g2, 0)), "baseline_auc": b_mean}
        for cn in conc:
            m, _ = _auc(present_baselines + [cn])
            row[f"auc_with_{cn}"] = m
            row[f"gain_{cn}"] = (m - b_mean if np.isfinite(m)
                                 and np.isfinite(b_mean) else float("nan"))
        rows.append(row)

    tab = pd.DataFrame(rows)
    if len(tab):
        tab = tab.sort_values("baseline_auc").reset_index(drop=True)
        tab.to_csv(out / "classifier_pairwise_auc.tsv", sep="\t", index=False)
        logger.info(f"[classifier] wrote {out / 'classifier_pairwise_auc.tsv'}")
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
    level_control: str = "linear",
    group_colors: dict | None = None,
    group_order: list | None = None,
    skip_top_svd: dict | None = None,
    batch_labels=None,
    batch_correction: str = "none",
    per_cell_norm: str = "off",
    post_norm_recenter: bool = False,
    plot_format: str = "png",
    harmony_theta=None,
    concordance_batch_labels=None,
    concordance_batch_correction: str = "none",
    marginal_matrix=None,
    marginal_correct: bool = False,
    marginal_correct_k: int = 5,
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

    # (a0) Batch-correct EVERY matrix up front, so the marginal UMAPs, the
    # variance-partition regression (and hence the residual) and the classifier
    # all see harmonised inputs — mirroring nome_hic_concordance, which corrects
    # the mean-accessibility matrix before it is normalised/consumed.
    # Batch correction is applied on the UMAP SCORES (see run_umap_and_plot),
    # NOT on the raw matrix here.  The old placement -- correct the matrix, then
    # per-cell rank / quantile normalise -- was undone by those normalisations,
    # so a 'corrected' UMAP still split every cell type by batch.  Harmony now
    # runs on the SVD scores that feed each UMAP, after all normalisation.  The
    # regression below therefore sees RAW (un-harmonised) inputs, which also
    # avoids the separate-PC-space leak of harmonising each matrix independently
    # and then differencing them.
    if batch_labels is not None and batch_correction != "none":
        summary["batch_correction"] = {
            "method": batch_correction, "applied_on": "umap_svd_scores",
            "matrices": sorted(matrices)}

    # (a0') The concordance matrices get their own, EARLIER correction, on the
    # raw per-bin values rather than the scores, and only on them.  The
    # baselines come off a different stage of a different pipeline, so a
    # setting tuned for one is not the setting for the other; this is the same
    # split nome_hic_concordance draws between --batch-correction and
    # --concordance-batch-correction, and it is what lets a winning
    # concordance-stage setting be reproduced here.  Applied after the
    # common-cell intersection so the batch means are those of the cells the
    # analysis actually uses, and before every normalization, since per-bin QN
    # turns a batch offset into a batch-ordered ranking and preserves it.
    if (concordance_batch_labels is not None
            and concordance_batch_correction != "none"):
        from . import batch_correction as _bc
        touched = [n for n in concordance_names if n in matrices]
        for name in touched:
            matrices[name] = _bc.correct_and_report(
                matrices[name], concordance_batch_labels,
                concordance_batch_correction, random_state, name)
        summary["concordance_batch_correction"] = {
            "method": concordance_batch_correction,
            "applied_on": "raw_matrix_per_bin", "matrices": sorted(touched)}

    # Marginal correction, at the same stage and for the same reason: the
    # batch is in the concordance VALUES, so it has to go before any
    # normalisation.  Different from --concordance-batch-correction in what
    # it removes -- that one takes a batch term, this one takes the leading
    # factors of the pair-derived marginal the values inherit.  On GCH the
    # batch term family topped out at cross-replicate purity 0.29 of the
    # within-replicate value while this reached 0.72.
    if marginal_matrix is not None and marginal_correct:
        from .marginal_correction import two_stage
        if batch_labels is None:
            logger.warning(
                "--marginal-correct needs --batch-column to know which cells "
                "are which replicate; skipping.")
        else:
            _bl = (batch_labels.iloc[:, 0]
                   if isinstance(batch_labels, pd.DataFrame)
                   else batch_labels).astype(str)
            _base = next((matrices[b] for b in baseline_names
                          if b in matrices), None)
            if _base is None:
                logger.warning(
                    "--marginal-correct has no baseline matrix to protect the "
                    "biology with; stage 1 INVERTS unprotected (measured "
                    "0.142 -> 0.083), so this is skipped.  Pass "
                    "--baseline-matrix.")
            else:
                done = []
                for name in [n for n in concordance_names if n in matrices]:
                    matrices[name] = two_stage(
                        matrices[name], marginal_matrix, _bl,
                        baseline=_base, k=marginal_correct_k)
                    done.append(name)
                summary["marginal_correct"] = {
                    "k": int(marginal_correct_k),
                    "applied_on": "raw_matrix_before_normalization",
                    "matrices": sorted(done)}
                logger.info(f"Marginal correction (k={marginal_correct_k}) "
                            f"applied to {sorted(done)}")

    # (a) Marginal.
    summary["marginal"] = marginal_analysis(
        matrices, labels, output_dir, matrix_norm, n_svd,
        min_cell_coverage, random_state,
        group_colors=group_colors, group_order=group_order,
        skip_top_svd=skip_top_svd, per_cell_norm=per_cell_norm,
        post_norm_recenter=post_norm_recenter, plot_format=plot_format,
        batch_labels=batch_labels, batch_correction=batch_correction,
        harmony_theta=harmony_theta)

    # (b)+(c) Variance partition + residual UMAP for each concordance matrix.
    present_baselines = [b for b in baseline_names if b in matrices]
    summary["residual"] = {}
    # Drop baselines too sparse to use as regressors.
    base_norm = {}
    for b in present_baselines:
        bn = _prep_matrix(matrices[b], b, output_dir, matrix_norm,
                          min_cell_coverage, per_cell_norm, post_norm_recenter)
        if bn.shape[1] < _MIN_BINS:
            logger.warning(f"[residual] baseline {b}: {bn.shape[1]} bin(s) — "
                           f"excluding from the variance-partition regression.")
            continue
        base_norm[b] = bn
    if base_norm:
        for cn in concordance_names:
            if cn not in matrices:
                continue
            tnorm = _prep_matrix(matrices[cn], cn, output_dir, matrix_norm,
                                 min_cell_coverage, per_cell_norm,
                                 post_norm_recenter)
            if tnorm.shape[1] < _MIN_BINS:
                logger.warning(f"[residual] {cn}: {tnorm.shape[1]} bin(s) after "
                               f"coverage filter — skipping residual analysis "
                               f"(too sparse; e.g. no-RWR concordance at fine "
                               f"resolution).")
                summary["residual"][cn] = {"skipped": "too_few_bins"}
                continue
            resid = variance_partition(tnorm, base_norm, cn, output_dir,
                                       level_control=level_control,
                                       plot_format=plot_format,
                                       batch_labels=(batch_labels
                                                     if batch_correction != "none"
                                                     else None))
            summary["residual"][cn] = residual_umap(
                resid, labels, cn, output_dir, matrix_norm, n_svd,
                min_cell_coverage, random_state,
                group_colors=group_colors, group_order=group_order,
                skip_top=int((skip_top_svd or {}).get(f"residual_{cn}", 0)),
                per_cell_norm=per_cell_norm, post_norm_recenter=post_norm_recenter,
                plot_format=plot_format,
                batch_labels=batch_labels, batch_correction=batch_correction,
                harmony_theta=harmony_theta)
    else:
        logger.warning("No usable baseline matrices; skipping (b)/(c).")

    # (d) Classifier comparison.
    if labels is not None:
        tab = classifier_comparison(
            matrices, labels, baseline_names, concordance_names, output_dir,
            matrix_norm, n_svd, min_cell_coverage, n_folds, random_state,
            skip_top_svd=skip_top_svd, per_cell_norm=per_cell_norm,
            post_norm_recenter=post_norm_recenter)
        summary["classifier"] = tab.to_dict(orient="records")
        # (d2) One-vs-one pairwise AUC — headroom-sensitive when the aggregate
        # is ceiling-limited by saturating marginal levels.
        if labels.nunique() > 2:
            ptab = classifier_comparison_pairwise(
                matrices, labels, baseline_names, concordance_names,
                output_dir, matrix_norm, n_svd, min_cell_coverage, n_folds,
                random_state, skip_top_svd=skip_top_svd,
                per_cell_norm=per_cell_norm, post_norm_recenter=post_norm_recenter)
            if len(ptab):
                summary["pairwise"] = ptab.to_dict(orient="records")
    else:
        logger.warning("No --samples group labels; skipping classifier + metrics.")

    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    logger.info(f"Wrote {out / 'summary.json'}")
    return summary
