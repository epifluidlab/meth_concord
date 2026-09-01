#!/usr/bin/env python3
"""Use chrM as an enzyme-efficiency covariate for GCH concordance.

WHY chrM.  Mitochondrial DNA is not chromatinised, so its GpC methylation is
close to a pure readout of M.CviPI efficiency per cell -- the NOMe-seq
internal control -- with none of the accessibility biology that contaminates
a genome-wide marginal.  That makes it the cleanest technical covariate
available for the GCH arm, and an INDEPENDENT one: unlike the pair-derived
marginal, it is not computed from the same reads as the target.

WHY IT IS REGRESSED PER BIN.  The covariate is one number per cell, but a
per-cell covariate removed uniformly does nothing here -- ``--per-cell-norm
rank`` already removes uniform per-cell shifts, which is why regressing the
global sequencing depth changed cross-replicate purity from 0.244 to 0.246.
Fitting a separate slope PER BIN is what makes it bite: bins sit at different
points on the non-linear L(p_bar) curve, so one efficiency change moves them
by different amounts, and that is not a uniform shift.

READ THE COVARIATE ITSELF, NOT ONLY THE CORRECTION.  The per-replicate chrM
means and the AUC for predicting replicate from chrM alone are printed
first.  A large gap there is a direct measurement of enzyme-efficiency drift
between preparations, independent of whether any correction helps -- and it
is the evidence a mechanistic claim needs.  Genome-wide p_bar showed 0.267
against 0.346, but that confounds efficiency with real accessibility; chrM
does not.

Two sources are worth trying, and they answer different questions:

  concordance_marginal.parquet   p_bar on chrM, the SAME estimator as the
                                 target, so it carries the same sparse-pair
                                 pathology -- if this works the fix is
                                 estimator-internal
  accessibility_mean.parquet     dense chrM accessibility, an independent
                                 measurement -- if this works you have a
                                 genuinely external technical covariate,
                                 which is the stronger claim

Usage:
  chrm_regress.py SOURCE.parquet TARGET.parquet SAMPLES.tsv \\
      --replicate-col exp_batch --celltype-col group \\
      [--baseline accessibility_mean.parquet] [--out OUT.parquet]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance.marginal_correction import (      # noqa: E402
    DEFAULT_PROTECT_PCS, protect_design)
from nome_hic_concordance.replicate_metrics import (        # noqa: E402
    batch_separation, purity, svd_scores)

#: chrM under the naming conventions bins actually use.
CHRM_PATTERN = r"(^|[^A-Za-z0-9])(chrM|chrMT|MT|M)([^A-Za-z0-9]|$)"


def chrm_covariate(source: pd.DataFrame, pattern: str = CHRM_PATTERN):
    """Per-cell chrM signal, and the bin names it came from."""
    rx = re.compile(pattern, re.I)
    cols = [c for c in source.columns if rx.search(str(c))]
    if not cols:
        return None, []
    X = source[cols].to_numpy(float)
    # A cell with no chrM coverage at all is expected, not exceptional, and
    # must come back NaN rather than 0 -- zero would read as "no enzyme
    # activity" instead of "not measured" and drag that cell the wrong way.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=1)
    return v, cols


def regress_per_bin(Y: np.ndarray, v: np.ndarray,
                    protect: np.ndarray | None = None) -> np.ndarray:
    """Per-bin slope on the per-cell covariate; subtract only that term."""
    v = np.asarray(v, float)
    v = (v - v.mean()) / (v.std() + 1e-12)
    D = (np.ones((len(v), 1)) if protect is None
         else np.column_stack([np.ones(len(v)), protect]))
    beta_d, *_ = np.linalg.lstsq(D, Y, rcond=None)
    vr = v - (D @ np.linalg.lstsq(D, v[:, None], rcond=None)[0])[:, 0]
    Yr = Y - D @ beta_d
    den = float((vr * vr).sum())
    slope = (vr @ Yr) / den if den > 1e-12 else np.zeros(Y.shape[1])
    return Y - np.outer(v, slope)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="Matrix to take the chrM signal FROM.")
    ap.add_argument("target", help="Concordance matrix to correct.")
    ap.add_argument("samples")
    ap.add_argument("--replicate-col", required=True)
    ap.add_argument("--celltype-col", required=True)
    ap.add_argument("--baseline", default=None,
                    help="Matrix whose components protect the biology.")
    ap.add_argument("--chrm-pattern", default=CHRM_PATTERN)
    ap.add_argument("--protect-pcs", type=int, default=DEFAULT_PROTECT_PCS)
    ap.add_argument("--n-svd", type=int, default=20)
    ap.add_argument("-k", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    src = pd.read_parquet(a.source)
    M = pd.read_parquet(a.target)
    s = pd.read_csv(a.samples, sep="\t")
    cell = next((c for c in ("cell_name", "cell", "cell_id")
                 if c in s.columns), s.columns[0])
    for c in (a.replicate_col, a.celltype_col):
        if c not in s.columns:
            raise SystemExit(f"{c!r} not in {a.samples} "
                             f"(has {list(s.columns)})")
    meta = s.set_index(s[cell].astype(str))
    common = src.index.astype(str).intersection(
        M.index.astype(str)).intersection(meta.index)
    if len(common) < 20:
        raise SystemExit(f"only {len(common)} shared cells; check cell names")
    src, M = src.loc[common], M.loc[common]
    rp = meta.loc[common, a.replicate_col].astype(str).values
    ct = meta.loc[common, a.celltype_col].astype(str).values

    v, cols = chrm_covariate(src, a.chrm_pattern)
    print(f"cells {len(common)}   target bins {M.shape[1]}")
    if v is None:
        raise SystemExit(
            f"no chrM bins matched /{a.chrm_pattern}/ in "
            f"{Path(a.source).name}.  Column names look like "
            f"{list(map(str, src.columns[:4]))}.\n"
            f"  If chrM was excluded when that matrix was built the covariate "
            f"is not available here --\n  rebuild without --exclude-chrom "
            f"chrM, or point at a matrix that kept it.")
    print(f"chrM bins matched: {len(cols)}   e.g. {cols[:3]}")
    if not np.isfinite(v).all():
        n_bad = int((~np.isfinite(v)).sum())
        print(f"  WARNING: {n_bad} cells have no finite chrM signal; filled "
              f"with the cohort mean.  Their correction is a guess.")
        v = np.where(np.isfinite(v), v, np.nanmean(v))

    lv = sorted(pd.unique(rp))
    print("\n  chrM signal per replicate (the enzyme-efficiency readout):")
    for x in lv:
        z = v[rp == x]
        print(f"    {x:10s} n={int((rp == x).sum()):5d}  mean {z.mean():.4f}"
              f"  median {np.median(z):.4f}  sd {z.std():.4f}")
    if len(lv) == 2:
        gap = abs(v[rp == lv[0]].mean() - v[rp == lv[1]].mean()) / (
            v.std() + 1e-12)
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        auc = cross_val_score(LogisticRegression(max_iter=2000), v[:, None],
                              (rp == lv[0]).astype(int), cv=5,
                              scoring="roc_auc").mean()
        print(f"    gap {gap:.3f} sd   replicate predictable from chrM alone: "
              f"AUC {auc:.3f}")
        print("    ^ a large gap is a DIRECT measurement of efficiency drift, "
              "whether or not\n      the correction below helps.")

    Y = np.nan_to_num(M.to_numpy(float), nan=0.0)
    prot = (protect_design(pd.read_parquet(a.baseline), M.index,
                           a.protect_pcs) if a.baseline else None)
    rows = [("uncorrected", Y), ("chrM regressed out", regress_per_bin(Y, v))]
    if prot is not None:
        rows.append(("chrM out, biology protected",
                     regress_per_bin(Y, v, prot)))

    def wrap(A):
        return pd.DataFrame(A, index=M.index, columns=M.columns)

    print(f"\n  {'':30s} {'within':>8s} {'cross':>8s} {'cross/within':>13s}")
    best, w0 = None, None
    for name, A in rows:
        p = purity(svd_scores(wrap(A), n_svd=a.n_svd), ct, rp, k=a.k)
        if name == "uncorrected":
            w0 = p["within"]
        ok = name != "uncorrected" and p["within"] >= 0.97 * w0
        mark = "" if ok or name == "uncorrected" else "   (within eroded)"
        print(f"  {name:30s} {p['within']:8.3f} {p['cross']:8.3f} "
              f"{p['ratio']:13.3f}{mark}")
        if ok and (best is None or p["ratio"] > best[1]):
            best = (A, p["ratio"], name)
    print(f"\n  chance purity {purity(np.zeros((len(common), 2)), ct, rp, k=a.k)['chance']:.3f}"
          "\n  -- 'best' considers only rows keeping >=97% of the uncorrected"
          "\n  within-purity: a ratio can always be raised by flattening cell "
          "state.")
    print("\n  QC -- did it change anything, and in which direction?")
    b0 = batch_separation(svd_scores(wrap(Y), n_svd=a.n_svd), rp)
    print(f"    replicate separation before {b0:.3f}")
    for name, A in rows[1:]:
        frac = float(((Y - A) ** 2).sum()
                     / max(((Y - Y.mean(0)) ** 2).sum(), 1e-12))
        b1 = batch_separation(svd_scores(wrap(A), n_svd=a.n_svd), rp)
        note = "   <- changed almost nothing" if frac < 0.005 else ""
        print(f"    {name:30s} variance removed {frac:7.1%}   "
              f"replicate separation {b1:.3f}{note}")
    if a.out and best:
        wrap(best[0]).to_parquet(a.out)
        print(f"\n  wrote {a.out}  (best: {best[2]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
