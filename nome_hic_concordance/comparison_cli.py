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


def _parse_named(spec: str) -> tuple[str, str]:
    """Parse 'name=path' (or bare 'path' -> name 'concordance')."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), path.strip()
    return "concordance", spec.strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nome_hic_compare", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    p.add_argument("--samples", default=None,
                   help="samples TSV with 'cell'/'cell_name' + 'group' "
                        "labels (enables silhouette/kNN-purity + classifier).")
    p.add_argument("--output-dir", required=True)
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
                   default="quantile")
    p.add_argument("--n-svd", type=int, default=10)
    p.add_argument("--min-cell-coverage", type=float, default=0.5)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--level-control", choices=["linear", "quadratic"],
                   default="linear",
                   help="Variance-partition regression of concordance on the "
                        "baselines.  'quadratic' adds squares + interactions so "
                        "the residual is cleansed of NONLINEAR level dependence "
                        "— the conservative control for 'does concordance carry "
                        "structure beyond marginal levels?'.")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
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
        matrices["accessibility"] = pd.read_parquet(args.accessibility_matrix)
        baseline_names.append("accessibility")
    if args.compartment_matrix:
        matrices["compartment"] = pd.read_parquet(args.compartment_matrix)
        baseline_names.append("compartment")
    for spec in args.baseline_matrix:
        name, path = _parse_named(spec)
        matrices[name] = pd.read_parquet(path)
        baseline_names.append(name)
    for spec in args.concordance_matrix:
        name, path = _parse_named(spec)
        matrices[name] = pd.read_parquet(path)
        concordance_names.append(name)

    if not matrices:
        logger.error("No matrices provided.")
        return 1
    if len(matrices) < 2:
        logger.warning(
            "Only one matrix provided — running marginal analysis only; "
            "cross-method comparison needs >= 2.")

    group_labels = group_colors = None
    if args.samples:
        from .cli import group_colors_from_samples

        sheet = _load_samples_sheet(args.samples)
        group_labels = sheet.set_index("cell_name")["group"]
        group_colors = group_colors_from_samples(sheet, args.color_column)
    elif args.color_column:
        raise ValueError("--color-column requires --samples.")
    group_order = ([s.strip() for s in args.group_order.split(",") if s.strip()]
                   if args.group_order else None)

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
