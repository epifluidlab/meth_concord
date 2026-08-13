#!/usr/bin/env python3
"""Grid-search the batch-removal settings and rank them by kNN purity.

Sweeps every combination of

    --concordance-batch-correction   the Δ matrix itself, before normalization
    --batch-correction               the SVD scores, just before UMAP
    --batch-column                   every non-empty subset of the named columns
    --skip-top-svd                   0 or 1

and scores each one against the cell-type labels with kNN purity, ARI, NMI,
silhouette and the chance-normalized purity index.  Results go to a TSV sorted
by kNN purity, with the winning combination printed as a ready-to-paste flag
string.

Scoring reuses the pipeline's own code -- ``matrix_ops.process_concordance_matrix``
for the normalization and ``visualization.umap_input_scores`` for the SVD plus
score-level correction -- so a combination's number here is what the real run
would produce, not a re-implementation that can drift.  UMAP itself is skipped:
the metrics are computed on the array UMAP would be handed, which is where the
pipeline computes them too, and 2-D layout would only add minutes per cell.

Work is shared across the grid rather than repeated: normalization depends only
on (concordance correction, batch columns), and the SVD only additionally on
skip, so those run once each and the cheap score-level correction plus scoring
runs per combination.

    python scripts/batch_correction_grid.py \\
        --concordance-matrix output/signed_250kb_rwr/context_GCH/05_concordance/concordance_shrunk.parquet \\
        --samples mouse_cortex_1577_cells.samples.with_batches.fixed.tsv \\
        --batch-column exp_batch,plate_batch,lib_batch \\
        --matrix-norm quantile --per-cell-norm rank --post-norm-recenter \\
        --n-svd 10 --exclude-chrom chrM,chrX,chrY \\
        --output-dir output/signed_250kb_rwr/11_batch_grid

READ THE RANKING WITH THE TOP-N, NOT THE MAX
--------------------------------------------
The criterion is agreement with the TRUE cell-type labels, so a correction that
removes biology along with batch is penalised rather than rewarded -- that part
is sound.  What is not sound is reading the single best row: the maximum over
a few hundred combinations is optimistically biased by selection alone.  Prefer
a setting that sits in a BAND of similar-scoring neighbours (its own family
does well) over a lone spike, and treat a winner that beats its neighbours by
less than the spread within a family as a tie.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance import batch_correction as _bc      # noqa: E402
from nome_hic_concordance import matrix_ops, metrics, visualization  # noqa: E402

logger = logging.getLogger("batch_grid")

_METRIC_COLS = ("knn_purity", "purity_index", "ari", "nmi", "silhouette",
                "knn_chance")

# Metrics that carry INDEPENDENT information, for the composite rankings.
# purity_index is deliberately excluded: chance is fixed across the grid (same
# cells, same labels), so it is a monotone transform of knn_purity and ranks
# identically — including both would simply weight purity twice.
_RANK_METRICS = ("knn_purity", "ari", "nmi", "silhouette")


def add_composite_ranks(res: pd.DataFrame) -> pd.DataFrame:
    """Two scale-free ways to combine the metrics, plus their per-metric ranks.

    A mean (or sum, or median) of the RAW values is not a defensible composite:
    the metrics live on different scales with different spreads, so whichever
    happens to vary most across the grid dominates the total.  Both composites
    here remove that.

    ``mean_rank``   — average of the per-metric ranks; scale-free and robust to
                      outliers, but ignores how MUCH better a setting is.
    ``scaled_mean`` — min-max each metric across the grid, then average; keeps
                      the margins, at the cost of being defined relative to
                      this particular grid.
    """
    r = res.copy()
    have = [m for m in _RANK_METRICS if m in r.columns and r[m].notna().any()]
    for m in have:
        r[f"rank_{m}"] = r[m].rank(ascending=False, method="average")
    r["mean_rank"] = r[[f"rank_{m}" for m in have]].mean(axis=1)
    scaled = []
    for m in have:
        v = r[m].astype(float)
        span = float(np.nanmax(v) - np.nanmin(v))
        scaled.append((v - np.nanmin(v)) / span if span > 1e-12
                      else pd.Series(1.0, index=v.index))
    r["scaled_mean"] = pd.concat(scaled, axis=1).mean(axis=1)
    return r


def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    df = (pd.read_parquet(p) if p.suffix in (".parquet", ".pq")
          else pd.read_csv(p, sep="\t", index_col=0))
    return df.astype(float)


def _drop_chroms(df: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    if not exclude:
        return df
    keep = [c for c in df.columns if str(c).split(":", 1)[0] not in exclude]
    logger.info(f"Excluded {df.shape[1] - len(keep)} bins on {sorted(exclude)}.")
    return df[keep]


def _column_subsets(cols: list[str]) -> list[tuple[str, ...]]:
    """Every non-empty subset, shortest first (ties in source order)."""
    out: list[tuple[str, ...]] = []
    for r in range(1, len(cols) + 1):
        out.extend(itertools.combinations(cols, r))
    return out


def _batch_frame(samples: pd.DataFrame, cell_col: str, cols: tuple[str, ...]):
    idx = samples.set_index(cell_col)
    return idx[cols[0]] if len(cols) == 1 else idx[list(cols)]


def _usable_methods(methods, where):
    """Drop methods whose optional dependency is missing.

    ComBat needs `combat` or `inmoose`, harmony needs `harmonypy`.  A sweep
    that dies on the first such combination wastes the whole run, and one that
    swallows the error silently would report a method as merely losing when it
    never ran.  Probe each once and say plainly which were dropped.
    """
    import contextlib
    import io
    import logging as _logging

    # The probe must be big enough for every method to run on it.  Harmony
    # clusters internally and raises on a handful of cells, so an 8x4 probe
    # reported it as unavailable and silently removed it from the grid -- a
    # false negative that looked exactly like a missing package.
    n_probe, p_probe = 80, 12
    rng = np.random.default_rng(0)
    probe = pd.DataFrame(rng.normal(0, 1, (n_probe, p_probe)),
                         index=[f"c{i}" for i in range(n_probe)])
    bl = pd.Series(["a", "b"] * (n_probe // 2), index=probe.index)
    ok, dropped = [], []
    for m in methods:
        if m == "none":
            ok.append(m)
            continue
        try:
            # ComBat narrates to stdout and the correction logs its own banner,
            # so an 8x4 probe would otherwise print a full run report that
            # reads as if it were the real data.
            _logging.disable(_logging.INFO)
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                _bc.apply_batch_correction(probe, bl, method=m, random_state=0)
            ok.append(m)
        except Exception as exc:                      # missing optional dep
            dropped.append((m, f"{type(exc).__name__}: "
                               f"{str(exc).splitlines()[0]}"))
        finally:
            _logging.disable(_logging.NOTSET)
    for m, why in dropped:
        logger.warning(f"  {where}: DROPPING {m!r} — {why}")
    if dropped:
        logger.warning(
            f"  {where}: {len(dropped)} method(s) will NOT appear in the "
            f"ranking. Install the missing package and re-run if you need "
            f"them compared (combat/inmoose for ComBat, harmonypy for "
            f"harmony) — a method that never ran is absent, not beaten.")
    if not ok:
        raise SystemExit(f"No usable methods for {where}.")
    logger.info(f"  {where}: using {ok}")
    return ok


def run_grid(delta, labels, samples, args):
    cols_all = [c.strip() for c in args.batch_column.split(",") if c.strip()]
    subsets = _column_subsets(cols_all)
    cbc_methods = _usable_methods(
        [m.strip() for m in args.concordance_methods.split(",")],
        "--concordance-methods")
    sbc_methods = _usable_methods(
        [m.strip() for m in args.score_methods.split(",")], "--score-methods")
    skips = [int(s) for s in args.skip_top_svd.split(",")]

    rows: list[dict] = []
    # A correction of "none" ignores the batch columns entirely, so pairing it
    # with each subset would run the same computation 7 times over.
    cbc_keys = [(m, s) for m in cbc_methods for s in subsets
                if not (m == "none" and s != subsets[0])]
    total = sum(len(skips) * len([(m, s) for m in sbc_methods for s in subsets
                                  if not (m == "none" and s != subsets[0])])
                for _ in cbc_keys)
    logger.info(f"Grid: {len(cbc_keys)} normalizations x {len(skips)} skips "
                f"x score-level settings = {total} combinations.")
    if args.dry_run:
        return pd.DataFrame()

    t0 = time.time()
    done = 0
    for cbc, cbc_cols in cbc_keys:
        # ---- stage 1: correct the concordance, then normalize (expensive) --
        mat = delta
        if cbc != "none":
            bl = _batch_frame(samples, args.cell_column, cbc_cols)
            mat = _bc.apply_batch_correction(
                delta, bl.reindex(delta.index), method=cbc,
                random_state=args.random_seed)
        norm = matrix_ops.process_concordance_matrix(
            mat, min_cell_coverage=args.min_cell_coverage,
            output_dir="", matrix_norm=args.matrix_norm,
            per_cell_norm=args.per_cell_norm,
            post_norm_recenter=args.post_norm_recenter,
            save_outputs=False)["zscore"]
        common = norm.index.intersection(labels.index)
        y = labels.loc[common]

        for skip in skips:
            # The SVD is the expensive half and depends only on (matrix, skip),
            # so fit it ONCE here and let each score-level correction reuse it.
            X0, _ = visualization.umap_input_scores(
                norm.loc[common], n_svd=args.n_svd, skip_top_svd=skip,
                n_threads=args.n_threads, random_state=args.random_seed,
                batch_correction="none")
            for sbc in sbc_methods:
                for sbc_cols in (subsets if sbc != "none" else [subsets[0]]):
                    bl2 = (None if sbc == "none"
                           else _batch_frame(samples, args.cell_column,
                                             sbc_cols).reindex(common))
                    # A method can clear the probe and still fail on the real
                    # data (a stratum with one cell, a degenerate component).
                    # Record it as failed rather than losing the whole sweep,
                    # and never let it be read as a low score.
                    try:
                        X = visualization.correct_scores(
                            X0, common, bl2, sbc,
                            random_state=args.random_seed,
                            n_threads=args.n_threads,
                            harmony_theta=args.harmony_theta)
                        m = metrics.label_agreement(
                            X, y, k=args.knn_k, random_state=args.random_seed,
                            n_perm=0)
                        err = ""
                    except Exception as exc:
                        m = {k: float("nan") for k in _METRIC_COLS}
                        err = f"{type(exc).__name__}: {exc}"
                        logger.warning(
                            f"  failed: cbc={cbc}/{'+'.join(cbc_cols)} "
                            f"sbc={sbc}/{'+'.join(sbc_cols)} skip={skip} — "
                            f"{err}")
                    rows.append({
                        "error": err,
                        "concordance_batch_correction": cbc,
                        "concordance_batch_column": ("-" if cbc == "none"
                                                     else "+".join(cbc_cols)),
                        "batch_correction": sbc,
                        "batch_column": ("-" if sbc == "none"
                                         else "+".join(sbc_cols)),
                        "skip_top_svd": skip,
                        **{k: m[k] for k in _METRIC_COLS},
                    })
                    done += 1
                    if done % 25 == 0 or done == total:
                        el = time.time() - t0
                        logger.info(
                            f"  [{done}/{total}] {el / 60:.1f} min elapsed, "
                            f"ETA {el / done * (total - done) / 60:.1f} min")
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concordance-matrix", required=True,
                    help="concordance_shrunk.parquet from Step 6.")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-column", default="cell_name")
    ap.add_argument("--group-column", default="group")
    ap.add_argument("--batch-column", required=True,
                    help="Comma-separated; every non-empty SUBSET is tried.")
    ap.add_argument("--concordance-methods",
                    default="none,center,zscore,combat",
                    help="Values to try for --concordance-batch-correction. "
                         "harmony is omitted by default: it is built for "
                         "low-dimensional embeddings, and on a full bin matrix "
                         "it is slow and most likely to remove cell type.")
    ap.add_argument("--score-methods",
                    default="none,center,zscore,combat,harmony",
                    help="Values to try for --batch-correction (SVD scores).")
    ap.add_argument("--skip-top-svd", default="0,1")
    ap.add_argument("--matrix-norm", choices=["quantile", "zscore"],
                    default="quantile")
    ap.add_argument("--per-cell-norm",
                    choices=["off", "center", "zscore", "rank"], default="rank")
    ap.add_argument("--post-norm-recenter", action="store_true")
    ap.add_argument("--n-svd", type=int, default=10)
    ap.add_argument("--min-cell-coverage", type=float, default=0.0)
    ap.add_argument("--exclude-chrom", default="")
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--harmony-theta", type=float, default=None)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--rank-by",
                    choices=["knn_purity", "purity_index", "ari", "nmi",
                             "silhouette", "mean_rank", "scaled_mean"],
                    default="knn_purity",
                    help="Criterion the table is sorted by and the winner "
                         "chosen on.  Single metrics answer different "
                         "questions: knn_purity is local and blind to a type "
                         "split across islands, silhouette is the one that "
                         "punishes that split, ARI/NMI ask whether k-means "
                         "recovers the labels.  'mean_rank' and 'scaled_mean' "
                         "combine them scale-free — do NOT average the raw "
                         "values, whose spreads differ severalfold.")
    ap.add_argument("--top", type=int, default=15,
                    help="Rows to print (default 15).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the grid size and exit.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excl = {c.strip() for c in args.exclude_chrom.split(",") if c.strip()}
    delta = _drop_chroms(_load(args.concordance_matrix), excl)
    samples = pd.read_csv(args.samples, sep="\t")
    labels = samples.set_index(args.cell_column)[args.group_column].dropna()
    labels = labels.loc[labels.index.intersection(delta.index)]
    logger.info(f"{delta.shape[0]} cells x {delta.shape[1]} bins; "
                f"{len(labels)} labelled, {labels.nunique()} cell types.")

    res = run_grid(delta, labels, samples, args)
    if res.empty:
        return 0

    res = add_composite_ranks(res)
    ascending = args.rank_by == "mean_rank"      # a rank: lower is better
    res = res.sort_values(args.rank_by, ascending=ascending
                          ).reset_index(drop=True)
    tsv = out_dir / "batch_correction_grid.tsv"
    res.to_csv(tsv, sep="\t", index=False)

    # Whether the metrics agree is itself the finding: if they do, the choice
    # of criterion is immaterial; if they do not, no single composite can be
    # right for every purpose and the disagreement has to be read directly.
    ok = res[[m for m in _RANK_METRICS if m in res]].dropna()
    if len(ok) > 2:
        print("\n=== rank agreement between metrics (Spearman) ===")
        print(ok.corr(method="spearman").round(2).to_string())
        print("  (purity_index is omitted: at fixed chance it ranks "
              "identically to knn_purity)")

    show = ["concordance_batch_correction", "concordance_batch_column",
            "batch_correction", "batch_column", "skip_top_svd",
            "knn_purity", "ari", "nmi", "silhouette",
            "mean_rank", "scaled_mean"]
    show = [c for c in show if c in res.columns]
    pd.set_option("display.width", 220)
    print(f"\n=== top {args.top} by {args.rank_by} "
          f"(purity chance = {res['knn_chance'].iloc[0]:.3f}) ===")
    print(res[show].head(args.top).to_string(index=False))

    best = res.iloc[0]
    print(f"\n=== best by {args.rank_by} ===")
    print(f"  --concordance-batch-correction {best['concordance_batch_correction']}"
          + ("" if best["concordance_batch_correction"] == "none"
             else f" (columns {best['concordance_batch_column'].replace('+', ',')})"))
    print(f"  --batch-correction {best['batch_correction']}"
          + ("" if best["batch_correction"] == "none"
             else f" (columns {best['batch_column'].replace('+', ',')})"))
    print(f"  --skip-top-svd {int(best['skip_top_svd'])}")
    print(f"  kNN purity {best['knn_purity']:.4f}  "
          f"purity index {best['purity_index']:.4f}  "
          f"ARI {best['ari']:.4f}  NMI {best['nmi']:.4f}  "
          f"ASW {best['silhouette']:.4f}")
    # A tie band on the chosen criterion, in its own units.
    col = args.rank_by
    if col == "mean_rank":
        near = int((res[col] <= best[col] + 0.05 * len(res)).sum())
        unit = f"within 5% of the grid in mean rank"
    else:
        near = int((res[col] >= best[col] - 0.01).sum())
        unit = f"within 0.01 {col}"
    print(f"\n  {near} combination(s) {unit} of the best — the max over "
          f"{len(res)} settings is optimistically biased by selection alone, "
          f"so prefer a setting inside a band of similar scores over a lone "
          f"spike.")
    print(f"\nwrote {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
