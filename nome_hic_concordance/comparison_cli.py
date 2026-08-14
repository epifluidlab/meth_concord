"""CLI for the comparison harness (`nome_hic_compare`) — design doc Part C.

Compares long-range GCH/HCG concordance against baseline feature matrices
(mean accessibility, scHi-C compartments) to answer: does concordance carry
cell-type information beyond the baselines?

Example
-------
    nome_hic_compare \\
        --accessibility-matrix output/.../context_GCH/05_concordance/accessibility_mean.parquet \\
        --compartment-matrix   output/compartments/compartment_scores.parquet \\
        --concordance-matrix gch=output/.../context_GCH/05_concordance/concordance_signed.parquet \\
        --concordance-matrix hcg=output/.../context_HCG/05_concordance/concordance_signed.parquet \\
        --samples samples.filtered.with_batches.tsv \\
        --output-dir output/.../08_comparison \\
        --matrix-norm quantile --n-svd 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import comparison
from .utils import (add_exclude_chrom_arg, bin_label_chrom,
                    chrom_excluded, chrom_key, parse_exclude_chroms)

logger = logging.getLogger("nome_hic_compare")


def _load_samples_sheet(path: str) -> pd.DataFrame:
    """Read the samples TSV (cell/cell_name + group [+ colour/batch columns]).

    Uses cli._manifest_text rather than pandas' ``comment='#'``: the latter also
    strips INLINE ``#...``, which truncates a row at a hex colour like
    '#FF9A4C' and blanks every column to its right.
    """
    from .cli import _manifest_text

    df = pd.read_csv(_manifest_text(path), sep="\t", dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "cell_name" not in df.columns:
        if "cell" in df.columns:
            df = df.rename(columns={"cell": "cell_name"})
        else:
            df = df.rename(columns={df.columns[0]: "cell_name"})
    if "group" not in df.columns:
        raise ValueError(
            f"{path} has no 'group' column (columns: {df.columns.tolist()})."
        )
    return df


def _load_group_labels(path: str) -> pd.Series:
    """cell -> group label."""
    return _load_samples_sheet(path).set_index("cell_name")["group"]


def _parse_skip_top_svd(specs: list[str]) -> dict[str, int]:
    """``['accessibility=1', 'gch=1']`` -> ``{'accessibility': 1, 'gch': 1}``."""
    out: dict[str, int] = {}
    for s in specs:
        if "=" not in s:
            raise ValueError(
                f"--skip-top-svd expects NAME=N (e.g. accessibility=1), got {s!r}."
            )
        name, val = s.split("=", 1)
        try:
            n = int(val)
        except ValueError:
            raise ValueError(f"--skip-top-svd {s!r}: N must be an integer.")
        if n < 0:
            raise ValueError(f"--skip-top-svd {s!r}: N must be >= 0.")
        out[name.strip()] = n
    return out


def _parse_named(spec: str, default_name: str = "concordance",
                 flag: str = "--concordance-matrix") -> tuple[str, str]:
    """Parse 'name=path'; a bare path falls back to ``default_name``.

    The fallback must follow the FLAG, not the parser: an unnamed
    --baseline-matrix used to come out called 'concordance', which labelled a
    baseline as the thing it is the control for in every figure, table and
    UMAP filename.
    """
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), path.strip()
    logger.warning(
        "%s was given a bare path, so it is named %r — every figure, table and "
        "UMAP filename will use that.  Pass NAME=PATH to name it yourself, "
        "e.g. %s gch_mean=%s",
        flag, default_name, flag, spec.strip())
    return default_name, spec.strip()


def _plain_path(value: str, flag: str) -> str:
    """Reject a NAME=PATH argument to a flag that takes a bare path.

    Only --baseline-matrix and --concordance-matrix are named; the two fixed
    baselines are not.  Passing 'name=path' to them otherwise surfaces as a
    FileNotFoundError quoting a path with the name glued to its front, which
    reads like a corrupt path rather than the wrong flag.
    """
    if "=" in value and not Path(value).exists():
        name = value.split("=", 1)[0].strip()
        raise ValueError(
            f"{flag} takes a bare PATH, not NAME=PATH — only "
            f"--baseline-matrix and --concordance-matrix are named.  To load "
            f"this matrix as a baseline called {name!r}, use:\n"
            f"    --baseline-matrix {value}")
    return value


class _DefaultsRawFormatter(argparse.ArgumentDefaultsHelpFormatter,
                            argparse.RawDescriptionHelpFormatter):
    """Keep the raw multi-line description AND mark every option's default."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nome_hic_compare", description=__doc__,
        formatter_class=_DefaultsRawFormatter,
    )
    p.add_argument("--accessibility-matrix", default=None,
                   help="cells × bins mean-accessibility parquet (baseline).")
    p.add_argument("--compartment-matrix", default=None,
                   help="cells × bins scHi-C compartment-score parquet (baseline).")
    p.add_argument("--baseline-matrix", action="append", default=[],
                   metavar="NAME=PATH",
                   help="Additional baseline matrix as NAME=PATH (repeatable). "
                        "Use to add e.g. mean HCG methylation "
                        "(hcg_mean=.../context_HCG/05_concordance/accessibility_mean.parquet) "
                        "so concordance must beat methylation LEVEL, not just "
                        "accessibility — the coordination-vs-level control.")
    p.add_argument("--concordance-matrix", action="append", default=[],
                   metavar="NAME=PATH",
                   help="cells × bins concordance parquet to test, as "
                        "NAME=PATH (repeatable), e.g. gch=.../concordance_signed.parquet "
                        "hcg=.../concordance_signed.parquet.")
    add_exclude_chrom_arg(p)
    p.add_argument("--samples", default=None,
                   help="samples TSV with 'cell'/'cell_name' + 'group' "
                        "labels (enables silhouette/kNN-purity + classifier).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--skip-top-svd", action="append", default=[],
                   metavar="NAME=N",
                   help="Drop the first N SVD components for ONE matrix "
                        "(repeatable; default 0 for every matrix).  N applies to "
                        "that matrix's UMAP *and* its silhouette / kNN-purity / "
                        "classifier block.  NAME is a matrix name — "
                        "'accessibility', 'compartment', a --baseline-matrix or "
                        "--concordance-matrix name — or 'residual_<name>' for a "
                        "residual UMAP.  e.g. --skip-top-svd accessibility=1 "
                        "(the top component is the per-cell depth/batch axis).")
    p.add_argument("--batch-column", type=str, default=None,
                   help="Column name(s) in --samples holding batch labels; "
                        "accepts a COMMA-SEPARATED list (e.g. "
                        "'exp_batch,lib_batch').  Required by "
                        "--batch-correction.")
    p.add_argument("--batch-correction",
                   choices=["none", "center", "zscore", "combat", "harmony"],
                   default="none",
                   help="Correct EVERY input matrix before any analysis, so the "
                        "marginal UMAPs, the variance-partition regression (and "
                        "hence the residual) and the classifier all see "
                        "harmonised inputs — mirroring nome_hic_concordance, "
                        "which corrects the mean-accessibility matrix before it "
                        "is normalised.")
    p.add_argument("--concordance-batch-column", type=str, default=None,
                   help="Batch variable(s) for --concordance-batch-correction "
                        "when they should differ from --batch-column; accepts "
                        "a COMMA-SEPARATED list.  Defaults to --batch-column, "
                        "so the common case stays one flag.")
    p.add_argument("--concordance-batch-correction",
                   choices=["none", "center", "zscore", "combat", "harmony"],
                   default="none",
                   help="Batch-correct ONLY the --concordance-matrix inputs, "
                        "per bin on their raw values, before any "
                        "normalization.  This is a different stage from "
                        "--batch-correction, which acts on the UMAP SVD scores "
                        "of every matrix: batch is already mixed into the "
                        "concordance bins (per-cell coverage drives the DM "
                        "shrinkage and what RWR imputes), and ten SVD "
                        "components cannot separate it out afterwards.  It has "
                        "to precede the per-bin normalization, since QN turns a "
                        "batch offset into a batch-ordered ranking and keeps "
                        "it.  Same flag and semantics as in "
                        "nome_hic_concordance, so a setting chosen there can be "
                        "reproduced here.  Baselines are deliberately left "
                        "alone — they come off a different stage, and "
                        "correcting them with a setting tuned for concordance "
                        "would weaken the control the comparison rests on.  "
                        "Needs --batch-column or --concordance-batch-column.")
    p.add_argument("--harmony-theta", type=float, default=None,
                   help="Harmony batch-correction STRENGTH (diversity-clustering "
                        "penalty) for --batch-correction harmony.  Higher mixes "
                        "batches more aggressively; lower is gentler and keeps "
                        "more of the original structure.  Broadcast to every "
                        "--batch-column variable; omit to use harmonypy's "
                        "default (2.0).")
    p.add_argument("--color-column", type=str, default=None,
                   help="Column in --samples holding the plot colour per cell "
                        "(hex like '#1f77b4' or a matplotlib name).  Each group "
                        "should carry one colour; the group's first wins.  Used "
                        "for every UMAP legend here (marginal + residual).")
    p.add_argument("--group-order", type=str, default=None,
                   help="COMMA-SEPARATED group order for the UMAP legends, e.g. "
                        "'IT-L2/3,IT-L4,ODC'.  Groups not listed are appended in "
                        "data order; names not in the data are ignored.")
    p.add_argument("--matrix-norm", choices=["zscore", "quantile"],
                   default="quantile",
                   help="Per-bin normalization applied to every matrix before "
                        "the UMAP.  'quantile' = rank-based QN (matches the "
                        "nome_hic_concordance signed default); 'zscore' = "
                        "classical.")
    p.add_argument("--per-cell-norm", choices=["off", "rank", "center"],
                   default="rank",
                   help="Per-cell normalization applied to EVERY matrix before "
                        "the per-bin norm.  'rank' mirrors the "
                        "nome_hic_concordance step-7 stack so the UMAPs are "
                        "directly comparable; 'off' is the historical compare "
                        "behaviour.")
    p.add_argument("--post-norm-recenter", choices=["off", "on"], default="on",
                   help="Subtract each cell's row-mean after normalization "
                        "(fixes the non-commutativity of per-cell rank + per-bin "
                        "QN).  Use 'on' with --per-cell-norm rank to match "
                        "nome_hic_concordance's signed step-7 stack.")
    p.add_argument("--plot-format", choices=["png", "pdf", "svg"], default="png",
                   help="Figure format for the UMAP and SVD-scree plots. "
                        "'pdf'/'svg' write vector figures.")
    p.add_argument("--n-svd", type=int, default=10,
                   help="SVD components for the UMAP pre-reduction.")
    p.add_argument("--min-cell-coverage", type=float, default=0.5,
                   help="Fraction of cells non-NA required to retain a bin.")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Cross-validation folds for the cell-type classifier.")
    p.add_argument("--level-control", choices=["linear", "quadratic"],
                   default="linear",
                   help="Variance-partition regression of concordance on the "
                        "baselines.  'quadratic' adds squares + interactions so "
                        "the residual is cleansed of NONLINEAR level dependence "
                        "— the conservative control for 'does concordance carry "
                        "structure beyond marginal levels?'.")
    p.add_argument("--random-seed", type=int, default=42,
                   help="Random seed (SVD / UMAP / CV).")
    p.add_argument("--log-level", default="INFO",
                   help="Logging verbosity (DEBUG, INFO, WARNING, ...).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "compare_config.json", "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)

    matrices: dict[str, pd.DataFrame] = {}
    baseline_names: list[str] = []
    concordance_names: list[str] = []

    if args.accessibility_matrix:
        matrices["accessibility"] = pd.read_parquet(
            _plain_path(args.accessibility_matrix, "--accessibility-matrix"))
        baseline_names.append("accessibility")
    if args.compartment_matrix:
        matrices["compartment"] = pd.read_parquet(
            _plain_path(args.compartment_matrix, "--compartment-matrix"))
        baseline_names.append("compartment")
    def _register(spec, default_name, flag, into):
        name, path = _parse_named(spec, default_name, flag)
        if name in matrices:
            # Silently overwriting is the dangerous outcome: two bare paths
            # both land on the same auto-name, one matrix replaces the other,
            # and it ends up in BOTH baseline_names and concordance_names --
            # so the variance partition regresses that matrix on itself and
            # reports R2 = 1 with an all-zero residual.
            held = ("a baseline" if name in baseline_names
                    else "a concordance matrix")
            raise ValueError(
                f"{flag}: matrix name {name!r} is already taken by {held}.  "
                f"Every matrix needs a distinct name — pass NAME=PATH, e.g. "
                f"{flag} {name}2={path}")
        matrices[name] = pd.read_parquet(path)
        into.append(name)

    for spec in args.baseline_matrix:
        _register(spec, "baseline", "--baseline-matrix", baseline_names)
    for spec in args.concordance_matrix:
        _register(spec, "concordance", "--concordance-matrix",
                  concordance_names)

    # Every matrix is cells x bins with 'chrom:start' columns, so the filter
    # is a column drop -- applied to ALL of them so baselines and concordance
    # stay on identical bins (a mismatch would silently shrink the overlap).
    excluded = parse_exclude_chroms(getattr(args, "exclude_chrom", None))
    if excluded and matrices:
        seen = set()
        for name, df in matrices.items():
            keep = [c for c in df.columns
                    if not chrom_excluded(bin_label_chrom(c), excluded)]
            seen.update(bin_label_chrom(c) for c in df.columns)
            dropped = df.shape[1] - len(keep)
            if dropped:
                logger.info(f"{name}: dropped {dropped:,} of {df.shape[1]:,} "
                            f"bins on excluded contigs")
            matrices[name] = df[keep]
        unmatched = excluded - {chrom_key(c) for c in seen}
        if unmatched:
            logger.warning("--exclude-chrom matched no bin for: "
                           + ", ".join(sorted(unmatched)))

    if not matrices:
        logger.error("No matrices provided.")
        return 1
    if len(matrices) < 2:
        logger.warning(
            "Only one matrix provided — running marginal analysis only; "
            "cross-method comparison needs >= 2.")

    group_labels = group_colors = batch_labels = None
    concordance_batch_labels = None
    if args.samples:
        from .cli import group_colors_from_samples

        sheet = _load_samples_sheet(args.samples)
        group_labels = sheet.set_index("cell_name")["group"]
        group_colors = group_colors_from_samples(sheet, args.color_column)
        idx = sheet.set_index("cell_name")

        def _batch_series(spec: str, flag: str):
            cols = [c.strip() for c in spec.split(",") if c.strip()]
            miss = [c for c in cols if c not in sheet.columns]
            if miss:
                raise ValueError(
                    f"{flag} {miss} not in {args.samples}. "
                    f"Available columns: {sheet.columns.tolist()}"
                )
            return idx[cols[0]] if len(cols) == 1 else idx[cols]

        if args.batch_column:
            batch_labels = _batch_series(args.batch_column, "--batch-column")
        # Falls back to --batch-column, so one flag covers the common case.
        concordance_batch_labels = (
            _batch_series(args.concordance_batch_column,
                          "--concordance-batch-column")
            if args.concordance_batch_column else batch_labels)
    elif (args.color_column or args.batch_column
          or args.concordance_batch_column):
        raise ValueError("--color-column / --batch-column / "
                         "--concordance-batch-column require --samples.")
    if args.batch_correction != "none" and batch_labels is None:
        raise ValueError(
            f"--batch-correction {args.batch_correction!r} requires --batch-column."
        )
    if (args.concordance_batch_correction != "none"
            and concordance_batch_labels is None):
        raise ValueError(
            f"--concordance-batch-correction "
            f"{args.concordance_batch_correction!r} requires --batch-column "
            f"or --concordance-batch-column."
        )
    if args.concordance_batch_correction != "none" and not concordance_names:
        logger.warning(
            "--concordance-batch-correction %r has nothing to act on: no "
            "--concordance-matrix was given.  Baselines are left uncorrected "
            "by design; use --batch-correction for those.",
            args.concordance_batch_correction)
    group_order = ([s.strip() for s in args.group_order.split(",") if s.strip()]
                   if args.group_order else None)
    skip_top_svd = _parse_skip_top_svd(args.skip_top_svd)

    summary = comparison.run_comparison(
        matrices, group_labels,
        baseline_names=baseline_names,
        concordance_names=concordance_names,
        output_dir=str(out),
        matrix_norm=args.matrix_norm,
        n_svd=args.n_svd,
        min_cell_coverage=args.min_cell_coverage,
        n_folds=args.n_folds,
        random_state=args.random_seed,
        level_control=args.level_control,
        group_colors=group_colors,
        group_order=group_order,
        skip_top_svd=skip_top_svd,
        batch_labels=batch_labels,
        batch_correction=args.batch_correction,
        concordance_batch_labels=concordance_batch_labels,
        concordance_batch_correction=args.concordance_batch_correction,
        harmony_theta=args.harmony_theta,
        per_cell_norm=args.per_cell_norm,
        post_norm_recenter=args.post_norm_recenter == "on",
        plot_format=args.plot_format,
    )

    # Headline to stdout.
    if "classifier" in summary:
        print("\n=== cell-type classification AUC (5-fold CV) ===")
        tab = pd.DataFrame(summary["classifier"])
        with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
            print(tab[["feature_set", "auc_mean", "auc_std"]].to_string(index=False))
        gain_path = out / "classifier_auc_gain.tsv"
        if gain_path.exists():
            print("\n=== AUC gain of concordance over baselines ===")
            print(pd.read_csv(gain_path, sep="\t").to_string(index=False))

    pw_path = out / "classifier_pairwise_auc.tsv"
    if pw_path.exists():
        pw = pd.read_csv(pw_path, sep="\t")
        gain_cols = [c for c in pw.columns if c.startswith("gain_")]
        print("\n=== one-vs-one AUC: hardest cell-type pairs (headroom test) ===")
        with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
            print(pw.head(8).to_string(index=False))
        # Mean concordance gain on the CONFUSABLE pairs (where baselines miss).
        conf = pw[pw["baseline_auc"] < 0.9]
        if len(conf) and gain_cols:
            print(f"\n  confusable pairs (baseline_auc < 0.9): {len(conf)} of {len(pw)}")
            for gc in gain_cols:
                cn = gc[len("gain_"):]
                print(f"    mean gain on confusable pairs [{cn}]: "
                      f"{conf[gc].mean():+.4f}  "
                      f"(pairs improved: {(conf[gc] > 0.01).sum()}/{len(conf)})")
        else:
            print("  (no pairs with baseline_auc < 0.9 — baselines separate "
                  "all pairs; no headroom on this dataset.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
