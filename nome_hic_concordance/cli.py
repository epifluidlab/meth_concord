"""Command-line entry point for the NOMe-HiC concordance pipeline.

Invoked via the ``nome_hic_concordance`` console script (see pyproject.toml)
or with ``python -m nome_hic_concordance``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import (
    bam_parser,
    concordance as concordance_mod,
    distance_analysis,
    matrix_ops,
    mixture_model,
    posterior,
    reference,
    shrinkage,
    visualization,
)
from .utils import ensure_dir, setup_logging

logger = logging.getLogger("nome_hic_concordance")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nome_hic_concordance",
        description="NOMe-HiC Long-Range Chromatin Accessibility Concordance Pipeline",
    )

    # Required
    parser.add_argument("--samples", required=True,
                        help="Path to sample manifest TSV (cell_name, bam_path, [group]).")
    parser.add_argument("--reference", default=None,
                        help="Path to .2bit reference genome. Required when "
                             "extracting contexts from the reference (no "
                             "--context-bed). Optional when --context-bed is "
                             "provided (chromosome sizes are read from the "
                             "BAM headers instead).")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory (will be created).")

    # Filtering
    parser.add_argument("--min-distance", type=int, default=20_000,
                        help="Min genomic distance for qualifying pairs (default: 20000).")
    parser.add_argument("--min-gch-sites", type=int, default=3,
                        help="Min GCH sites per read end (default: 3).")
    parser.add_argument("--bin-size", type=int, default=5_000_000,
                        help="Genomic bin size in bp (default: 5000000).")
    parser.add_argument("--min-pairs-per-bin", type=int, default=5,
                        help="Min pairs per bin to compute r (default: 5).")
    parser.add_argument("--min-cell-coverage", type=float, default=0.8,
                        help="Fraction of cells non-NA to retain a bin (default: 0.8).")

    # EM
    parser.add_argument("--em-max-iter", type=int, default=200)
    parser.add_argument("--em-tol", type=float, default=1e-6)
    parser.add_argument("--em-restarts", type=int, default=5,
                        help="Random restarts for EM fitting (default: 5).")
    parser.add_argument("--short-range-max", type=int, default=1_000,
                        help="Max distance for short-range diagnostic class (default: 1000).")

    # Shrinkage
    parser.add_argument("--shrinkage-n0", type=str, default="auto",
                        help='Shrinkage pseudocount (float or "auto" = empirical Bayes).')

    # SNP / context
    parser.add_argument("--dbsnp-vcf", type=str, default=None,
                        help="Tabix-indexed dbSNP VCF for SNP filtering of context sites.")
    parser.add_argument("--contexts", type=str, default="GCH",
                        help='Comma-separated IUPAC cytosine contexts to analyse (default: "GCH").')
    parser.add_argument(
        "--context-bed", type=str, default=None,
        help=(
            "Load context positions from tabix-indexed BED files instead "
            "of extracting from the .2bit reference (much faster). "
            "Format: CONTEXT:PATH[,CONTEXT:PATH,...] e.g. "
            '"GCH:/data/GCH.bed.gz" or '
            '"GCH:/data/GCH.bed.gz,HCG:/data/HCG.bed.gz". '
            "Each BED must be bgzipped with a .tbi index. "
            "Accepted columns: (chrom, start, end, strand_or_refbase) "
            "or standard BED6. When provided, Step 1 .2bit extraction "
            "is skipped and --contexts is inferred from the BED paths."
        ),
    )
    parser.add_argument(
        "--context-bed-offset", type=int, default=0,
        help=(
            "Offset from BED start (col 2, 0-based) to the cytosine "
            "position. Use 0 (default) when the BED interval already "
            "spans the C itself (e.g. start=10481 end=10482 means C "
            "is at 0-based position 10481). Use 1 if the BED start is "
            "the first base of the full trinucleotide motif (the G of "
            "GCH), so the C is at start+1."
        ),
    )

    # Read processing
    parser.add_argument("--trim-ends", type=int, default=5,
                        help="Bases to trim from read ends (default: 5).")
    parser.add_argument("--mapq-min", type=int, default=30,
                        help="Minimum mapping quality on both ends (default: 30).")

    # EM fitting options
    parser.add_argument("--em-subsample", type=int, default=10_000_000,
                        help="Max read ends for EM fitting (default: 10000000). "
                             "Use 0 to disable subsampling and fit on all read ends.")
    parser.add_argument("--em-cell-specific", action="store_true",
                        help="Also fit EM per-cell for diagnostic comparison.")

    # Execution
    parser.add_argument("--n-threads", type=int, default=4,
                        help="Number of parallel threads (default: 4).")
    parser.add_argument("--steps", type=str, default="all",
                        help='Steps to run: comma-separated (1-7) or "all".')
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-file", type=str, default=None,
                        help="Additional log file (default: stderr only).")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_samples(path: str) -> pd.DataFrame:
    """Load a TSV manifest: cell_name, bam_path, optional group.

    Accepts files with or without a header row.  If the first row looks
    like a header (first field equals ``cell_name``), it is consumed as
    column names; otherwise the file is treated as headerless.
    """
    # Peek at the first line to decide whether a header is present.
    with open(path) as fh:
        first_line = fh.readline().strip()
    first_field = first_line.split("\t")[0].strip().lower()
    has_header = first_field in ("cell_name", "cellname", "cell", "sample", "sample_name")

    if has_header:
        df = pd.read_csv(path, sep="\t", comment="#", dtype=str)
        # Normalise column names coming from the header.
        col_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
        df.rename(columns=col_map, inplace=True)
        # Accept common synonyms.
        if "cellname" in df.columns:
            df.rename(columns={"cellname": "cell_name"}, inplace=True)
        if "sample" in df.columns and "cell_name" not in df.columns:
            df.rename(columns={"sample": "cell_name"}, inplace=True)
        if "sample_name" in df.columns and "cell_name" not in df.columns:
            df.rename(columns={"sample_name": "cell_name"}, inplace=True)
    else:
        df = pd.read_csv(
            path, sep="\t", header=None, comment="#",
            names=["cell_name", "bam_path", "group"],
            engine="python", dtype=str,
        )

    if "bam_path" not in df.columns:
        raise ValueError(
            f"Sample manifest {path} must have a 'bam_path' column (or be a "
            f"headerless TSV with columns: cell_name, bam_path, [group])."
        )

    if "group" not in df.columns:
        df["group"] = "unknown"
    df["group"] = df["group"].fillna("unknown")

    n_groups = df["group"].nunique()
    logger.info(f"Loaded {len(df)} samples, {n_groups} group(s).")
    return df


def _parse_context_bed(spec: str) -> dict[str, str]:
    """Parse ``'GCH:/path/to/GCH.bed.gz,HCG:/path/to/HCG.bed.gz'`` into a dict."""
    result: dict[str, str] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"--context-bed entries must be CONTEXT:PATH, got {token!r}"
            )
        ctx, path = token.split(":", 1)
        ctx = ctx.strip().upper()
        path = path.strip()
        if not path:
            raise ValueError(f"Empty path for context {ctx!r} in --context-bed")
        result[ctx] = path
    return result


def _parse_steps(spec: str) -> set[int]:
    """Parse ``'all'`` or e.g. ``'2,3,4'`` into an explicit set of steps."""
    spec = spec.strip().lower()
    if spec == "all":
        return set(range(1, 8))
    return {int(s) for s in spec.split(",") if s.strip()}


def _get_chrom_sizes(reference_path: str) -> dict[str, int]:
    """Read chromosome sizes from a .2bit reference."""
    import py2bit
    tb = py2bit.open(reference_path)
    try:
        return dict(tb.chroms())
    finally:
        tb.close()


def _get_chrom_sizes_from_bam(bam_path: str) -> dict[str, int]:
    """Read chromosome sizes from a BAM file's @SQ header lines."""
    import pysam
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return {
            sq["SN"]: sq["LN"]
            for sq in bam.header.to_dict().get("SQ", [])
        }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_context(
    context: str,
    context_positions: dict[str, np.ndarray],
    samples: pd.DataFrame,
    ctx_out: Path,
    steps: set[int],
    args: argparse.Namespace,
    chrom_sizes: dict[str, int],
) -> None:
    """Run the per-context pipeline (steps 2-7) for a single context."""
    logger.info("#" * 60)
    logger.info(f"Processing context: {context}")
    logger.info("#" * 60)

    ensure_dir(ctx_out)

    read_pair_dir = ctx_out / "02_read_pair_counts"
    mixture_dir = ctx_out / "03_mixture_model"
    posterior_dir = ctx_out / "04_posteriors"
    concordance_dir = ctx_out / "05_concordance"
    umap_dir = ctx_out / "06_umap"

    # ---- Step 2: BAM parsing ----------------------------------------------
    if 2 in steps:
        logger.info("=" * 60)
        logger.info(f"Step 2: Parsing BAM files for {context}")
        logger.info("=" * 60)
        cell_parquets = bam_parser.parse_all_cells(
            samples, context_positions,
            str(read_pair_dir),
            n_threads=args.n_threads,
            trim_ends=args.trim_ends, mapq_min=args.mapq_min,
        )
    else:
        logger.info(f"Step 2: Skipped (loading per-cell parquets from {read_pair_dir})")
        cell_parquets = {
            row["cell_name"]: str(read_pair_dir / f"{row['cell_name']}.read_pairs.parquet")
            for _, row in samples.iterrows()
        }
        missing = [c for c, p in cell_parquets.items() if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"Read-pair parquets missing for {len(missing)} cell(s); "
                f"re-run step 2 or populate {read_pair_dir}."
            )

    # ---- Step 3: Mixture model -------------------------------------------
    if 3 in steps:
        logger.info("=" * 60)
        logger.info(f"Step 3: Fitting beta-binomial mixture for {context}")
        logger.info("=" * 60)

        # Stream-subsample: reads one cell at a time instead of loading all
        # parquets into memory.  Peak RAM ≈ one parquet + subsample pool
        # (~400 MB) instead of ~60–95 GB for the full concatenation.
        fit_results = mixture_model.fit_and_compare_from_parquets(
            cell_parquets,
            short_range_max=args.short_range_max,
            min_distance=args.min_distance,
            output_dir=str(mixture_dir),
            max_subsample=args.em_subsample,
            random_state=args.random_seed,
            max_iter=args.em_max_iter,
            tol=args.em_tol,
            n_restarts=args.em_restarts,
        )
        fitted_model = fit_results["models"]["all"]

        if args.em_cell_specific:
            logger.info("Fitting cell-specific mixture models for diagnostic...")
            mixture_model.fit_cell_specific_models(
                cell_parquets, fitted_model,
                n_threads=args.n_threads,
                output_dir=str(mixture_dir),
                group_labels=samples.set_index("cell_name")["group"],
            )
    else:
        em_fit_path = mixture_dir / "em_fit_all.json"
        if not em_fit_path.exists():
            raise FileNotFoundError(
                f"Mixture fit JSON not found at {em_fit_path}; re-run step 3."
            )
        logger.info(f"Step 3: Skipped (loading fit from {em_fit_path})")
        fitted_model = mixture_model.BetaBinomialMixture.from_json(str(em_fit_path))

    # ---- Step 4: Posteriors ----------------------------------------------
    if 4 in steps:
        logger.info("=" * 60)
        logger.info(f"Step 4: Computing posteriors for {context}")
        logger.info("=" * 60)
        posterior_paths = posterior.compute_all_posteriors(
            cell_parquets, fitted_model, str(posterior_dir),
            n_threads=args.n_threads,
        )
    else:
        posterior_paths = {
            row["cell_name"]: str(posterior_dir / f"{row['cell_name']}.posteriors.parquet")
            for _, row in samples.iterrows()
        }
        missing = [c for c, p in posterior_paths.items() if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"Posterior parquets missing for {len(missing)} cell(s); re-run step 4."
            )
        logger.info(f"Step 4: Skipped (loading posteriors from {posterior_dir})")

    # ---- Step 5: Concordance + distance decay -----------------------------
    if 5 in steps:
        logger.info("=" * 60)
        logger.info(f"Step 5: Computing per-bin concordance for {context}")
        logger.info("=" * 60)

        concordance_raw, pair_counts = concordance_mod.compute_concordance_matrix(
            posterior_paths, chrom_sizes,
            bin_size=args.bin_size,
            min_distance=args.min_distance,
            min_gch_sites=args.min_gch_sites,
            min_pairs_per_bin=args.min_pairs_per_bin,
            n_threads=args.n_threads,
            output_dir=str(concordance_dir),
        )

        logger.info("Computing concordance vs. distance decay...")
        distance_analysis.concordance_vs_distance(
            posterior_paths,
            min_gch_sites=args.min_gch_sites,
            n_threads=args.n_threads,
            output_dir=str(concordance_dir),
        )
    else:
        raw_path = concordance_dir / "concordance_raw.parquet"
        cnt_path = concordance_dir / "pair_counts_per_bin.parquet"
        if not raw_path.exists() or not cnt_path.exists():
            raise FileNotFoundError(
                f"Concordance matrix not found at {raw_path}; re-run step 5."
            )
        logger.info(f"Step 5: Skipped (loading concordance matrix from {concordance_dir})")
        concordance_raw = pd.read_parquet(raw_path)
        pair_counts = pd.read_parquet(cnt_path)

    # ---- Step 6: Shrinkage -----------------------------------------------
    if 6 in steps:
        logger.info("=" * 60)
        logger.info(f"Step 6: Shrinkage analysis for {context}")
        logger.info("=" * 60)

        if args.shrinkage_n0 == "auto":
            shrink_result = shrinkage.estimate_shrinkage_n0(
                concordance_raw, pair_counts,
                output_dir=str(concordance_dir),
            )
            n0 = shrink_result["n0"]
        else:
            n0 = float(args.shrinkage_n0)
            logger.info(f"Using user-supplied shrinkage n0 = {n0}")
            with open(concordance_dir / "shrinkage_n0.json", "w") as f:
                json.dump({"n0": n0, "source": "user"}, f, indent=2)

        concordance_shrunk = shrinkage.apply_shrinkage(concordance_raw, pair_counts, n0)
        concordance_shrunk.to_parquet(concordance_dir / "concordance_shrunk.parquet")
        concordance_shrunk.to_csv(concordance_dir / "concordance_shrunk.tsv", sep="\t")
    else:
        shrunk_path = concordance_dir / "concordance_shrunk.parquet"
        if not shrunk_path.exists():
            raise FileNotFoundError(
                f"Shrunk concordance not found at {shrunk_path}; re-run step 6."
            )
        logger.info(f"Step 6: Skipped (loading shrunk concordance from {shrunk_path})")
        concordance_shrunk = pd.read_parquet(shrunk_path)

    # ---- Step 7: Matrix ops + UMAP ---------------------------------------
    if 7 in steps:
        logger.info("=" * 60)
        logger.info(f"Step 7: Matrix processing + UMAP for {context}")
        logger.info("=" * 60)

        if concordance_shrunk.shape[0] == 0 or concordance_shrunk.shape[1] == 0:
            logger.warning(
                "Concordance matrix is empty (0 cells or 0 bins). "
                "This usually means no read pairs passed the filters "
                "(--min-distance, --min-gch-sites). Check that the context "
                "BED positions match the reference used for alignment, and "
                "that --context-bed-offset is correct. Skipping matrix "
                "processing and UMAP."
            )
            return

        matrices = matrix_ops.process_concordance_matrix(
            concordance_shrunk,
            min_cell_coverage=args.min_cell_coverage,
            output_dir=str(concordance_dir),
        )

        group_labels = samples.set_index("cell_name")["group"]

        if matrices["imputed"].shape[1] == 0:
            logger.warning(
                "No bins retained after coverage filter; skipping UMAP."
            )
        elif matrices["imputed"].shape[0] < 2:
            logger.warning(
                "Fewer than 2 cells in the imputed matrix; skipping UMAP."
            )
        else:
            visualization.run_umap_and_plot(
                matrices["imputed"], group_labels,
                title=f"UMAP (no z-score, {context})",
                output_prefix="umap_no_zscore",
                output_dir=str(umap_dir),
                n_threads=args.n_threads,
                random_state=args.random_seed,
            )
            visualization.run_umap_and_plot(
                matrices["zscore"], group_labels,
                title=f"UMAP (z-scored, {context})",
                output_prefix="umap_zscore",
                output_dir=str(umap_dir),
                n_threads=args.n_threads,
                random_state=args.random_seed,
            )
    else:
        logger.info("Step 7: Skipped.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    out = ensure_dir(args.output_dir)
    log_file = args.log_file
    if log_file is None:
        log_file = str(out / "pipeline.log")
    setup_logging(logging.INFO, log_file=log_file)

    # Save full config up front.
    config = vars(args).copy()
    with open(out / "pipeline_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    steps = _parse_steps(args.steps)
    logger.info(f"Running steps: {sorted(steps)}")

    samples = load_samples(args.samples)

    # Resolve contexts: --context-bed overrides --contexts when provided.
    context_bed_paths: dict[str, str] | None = None
    if args.context_bed:
        context_bed_paths = _parse_context_bed(args.context_bed)
        contexts = list(context_bed_paths.keys())
        logger.info(
            f"Using pre-built BED files for contexts: "
            + ", ".join(f"{c}={p}" for c, p in context_bed_paths.items())
        )
    else:
        contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
        # Without --context-bed the .2bit reference is required for extraction.
        if args.reference is None:
            raise ValueError(
                "--reference is required when not using --context-bed."
            )
    if not contexts:
        raise ValueError("--contexts (or --context-bed) produced an empty list.")

    # ---- Step 1: context positions ----------------------------------------
    if context_bed_paths is not None:
        # Fast path: load from tabix-indexed BED files.
        # Always runs (context positions are needed by all downstream steps).
        logger.info("Loading context positions from BED files...")
        all_context_positions = reference.load_context_from_bed(
            context_bed_paths,
            n_threads=args.n_threads,
            c_offset=args.context_bed_offset,
        )
    elif 1 in steps:
        logger.info("=" * 60)
        logger.info("Step 1: Extracting context positions from reference")
        logger.info("=" * 60)
        all_context_positions = reference.extract_context_positions(
            args.reference,
            str(out / "01_context_extraction"),
            contexts=contexts,
            dbsnp_vcf=args.dbsnp_vcf,
            n_threads=args.n_threads,
        )
    else:
        logger.info("Step 1: Skipped (loading precomputed context positions)")
        all_context_positions = reference.load_context_positions(
            str(out / "01_context_extraction"), contexts
        )

    # Chromosome sizes — used for bin sorting in Step 5.
    if args.reference is not None:
        chrom_sizes = _get_chrom_sizes(args.reference)
    else:
        # Fall back to the first BAM file's @SQ header.
        first_bam = samples["bam_path"].iloc[0]
        logger.info(f"Reading chromosome sizes from BAM header: {first_bam}")
        chrom_sizes = _get_chrom_sizes_from_bam(first_bam)

    # ---- Per-context processing (steps 2..7) -----------------------------
    for context in contexts:
        ctx_positions = all_context_positions.get(context, {})
        ctx_out = out / f"context_{context}"
        run_context(
            context=context,
            context_positions=ctx_positions,
            samples=samples,
            ctx_out=ctx_out,
            steps=steps,
            args=args,
            chrom_sizes=chrom_sizes,
        )

    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
