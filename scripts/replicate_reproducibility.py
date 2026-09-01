#!/usr/bin/env python3
"""Does a matrix reproduce across biological replicates, and if not, why?

TWO MODES.

  score   cell-type purity within and ACROSS replicates, for one or more
          matrices.  Ordinary kNN purity cannot see this failure -- a cell
          whose neighbours are all its own type from its OWN replicate scores
          1.0 -- so the cross-replicate number is the one that matters.

  probe   where the difference lives, when a matrix fails to reproduce.
          Reports, for each candidate carrier: whether the replicates differ
          in LOCATION or in SCALE per bin, how concentrated the difference is
          across components, and whether a supplied covariate matrix (e.g.
          the coverage counts) is a replicate fingerprint on its own.

Accepts either UMAP coordinate TSVs (from nome_hic_compare) or cells x bins
parquet matrices; a matrix is scored in its own per-cell-rank + SVD space,
which is what the pipeline embeds.

Usage:
  replicate_reproducibility.py score SAMPLES.tsv --replicate-col exp_batch \\
      --celltype-col group --matrix a=A.parquet --matrix b=B.parquet
  replicate_reproducibility.py probe SAMPLES.tsv --replicate-col exp_batch \\
      --celltype-col group --matrix conc=C.parquet --matrix base=B.parquet
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

from nome_hic_concordance.replicate_metrics import (      # noqa: E402
    batch_separation, purity, svd_scores)


def _load_named(spec: str):
    if "=" not in spec:
        raise SystemExit(f"--matrix expects NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{name}: {p} not found")
    if p.suffix in (".tsv", ".txt", ".csv"):
        sep = "," if p.suffix == ".csv" else "\t"
        return name, pd.read_csv(p, sep=sep), True
    return name, pd.read_parquet(p), False


def _scores_of(df, is_coords, n_svd, cells):
    """UMAP coordinates are used as they are; a matrix gets the pipeline's
    own space (per-cell rank then SVD), never raw values."""
    if is_coords:
        cc = next(c for c in df.columns if c.lower() in
                  ("cell", "cell_name", "cell_id"))
        d = df.set_index(cc)
        xy = [c for c in d.columns if re.fullmatch(r"UMAP\d", str(c), re.I)]
        if len(xy) < 2:
            raise SystemExit(f"expected UMAP1/UMAP2 columns, found "
                             f"{list(d.columns)}")
        d = d.reindex(cells)
        return d[xy[:2]].to_numpy(float)
    return svd_scores(df.reindex(cells), n_svd=n_svd)


def _common_cells(items, meta):
    idx = None
    for _, df, is_coords in items:
        if is_coords:
            cc = next(c for c in df.columns if c.lower() in
                      ("cell", "cell_name", "cell_id"))
            got = pd.Index(df[cc].astype(str))
        else:
            got = df.index.astype(str)
        idx = got if idx is None else idx.intersection(got)
    return idx.intersection(meta.index.astype(str))


def _meta(samples, rep_col, ct_col):
    s = pd.read_csv(samples, sep="\t")
    cell = next((c for c in ("cell_name", "cell", "cell_id")
                 if c in s.columns), s.columns[0])
    for c in (rep_col, ct_col):
        if c not in s.columns:
            raise SystemExit(f"{c!r} not in {samples} (has {list(s.columns)})")
    m = s.set_index(s[cell].astype(str))
    return m


def cmd_score(a) -> int:
    meta = _meta(a.samples, a.replicate_col, a.celltype_col)
    items = [_load_named(x) for x in a.matrix]
    cells = _common_cells(items, meta)
    if len(cells) < 20:
        raise SystemExit(f"only {len(cells)} shared cells; check cell names")
    ct = meta.loc[cells, a.celltype_col].astype(str).values
    rp = meta.loc[cells, a.replicate_col].astype(str).values
    print(f"cells {len(cells)}   replicates {sorted(pd.unique(rp))}   "
          f"cell types {len(set(ct))}")
    print(f"\n  {'':22s} {'within':>8s} {'cross':>8s} {'cross/within':>13s} "
          f"{'batch sep':>10s}")
    for name, df, is_coords in items:
        S = _scores_of(df, is_coords, a.n_svd, cells)
        p = purity(S, ct, rp, k=a.k)
        print(f"  {name:22s} {p['within']:8.3f} {p['cross']:8.3f} "
              f"{p['ratio']:13.3f} {batch_separation(S, rp):10.3f}")
    print(f"\n  chance purity {purity(np.zeros((len(cells), 2)), ct, rp, k=a.k)['chance']:.3f}"
          "   -- read cross AND within: a ratio raised while within collapses\n"
          "  is a correction erasing cell state, not fixing the replicate.")
    return 0


def cmd_probe(a) -> int:
    meta = _meta(a.samples, a.replicate_col, a.celltype_col)
    items = [_load_named(x) for x in a.matrix]
    if any(c for _, _, c in items):
        raise SystemExit("probe needs parquet matrices, not coordinate TSVs")
    cells = _common_cells(items, meta)
    rp = meta.loc[cells, a.replicate_col].astype(str).values
    lv = sorted(pd.unique(rp))
    if len(lv) != 2:
        raise SystemExit(f"probe compares exactly 2 replicates; got {lv}")
    for name, df, _ in items:
        M = df.reindex(cells)
        X = M.to_numpy(float)
        A, B = X[rp == lv[0]], X[rp == lv[1]]
        pooled = np.nanstd(X, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            loc = np.abs((np.nanmean(A, 0) - np.nanmean(B, 0))
                         / np.where(pooled > 0, pooled, np.nan))
            sca = np.abs(np.log2(np.nanstd(A, 0)
                                 / np.where(np.nanstd(B, 0) > 0,
                                            np.nanstd(B, 0), np.nan)))
        S = svd_scores(M, n_svd=a.n_svd)
        d = np.abs(S[rp == lv[0]].mean(0) - S[rp == lv[1]].mean(0)) / (
            S.std(0) + 1e-12)
        share = d ** 2 / max((d ** 2).sum(), 1e-12)
        cum = np.cumsum(np.sort(share)[::-1])
        n80 = int(np.searchsorted(cum, 0.80) + 1)
        print(f"\n  {name}  ({M.shape[0]} cells x {M.shape[1]} bins)")
        print(f"    LOCATION |dmean|/sd   median {np.nanmedian(loc):.3f}"
              f"   90th {np.nanpercentile(loc, 90):.3f}")
        print(f"    SCALE |log2 sd ratio| median {np.nanmedian(sca):.3f}"
              f"   90th {np.nanpercentile(sca, 90):.3f}")
        print(f"    components holding 80% of the replicate separation: {n80}"
              f"   (top component {share.max():.1%})")
        # is the matrix a replicate fingerprint on its own?
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        auc = cross_val_score(LogisticRegression(max_iter=2000), S,
                              (rp == lv[0]).astype(int), cv=5,
                              scoring="roc_auc").mean()
        print(f"    replicate predictable from it: AUC {auc:.3f}")
    print("\n  A carrier looks like: high AUC, a LOCATION difference, and the"
          "\n  separation concentrated in few components.  Compare a matrix"
          "\n  that fails to reproduce against one that does -- the difference"
          "\n  between them is the thing to explain.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("score", cmd_score), ("probe", cmd_probe)):
        q = sub.add_parser(name, help=fn.__doc__)
        q.add_argument("samples")
        q.add_argument("--matrix", action="append", required=True,
                       metavar="NAME=PATH",
                       help="Repeatable.  .parquet matrix or UMAP "
                            "coordinates .tsv (score only).")
        q.add_argument("--replicate-col", required=True)
        q.add_argument("--celltype-col", required=True)
        q.add_argument("--n-svd", type=int, default=20)
        q.add_argument("-k", type=int, default=30,
                       help="Neighbours per replicate (30).")
        q.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
