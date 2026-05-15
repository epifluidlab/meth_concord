"""CLI entry point for the pair-concordance pipeline (Steps 5p–9p).

Invoked via ``nome_hic_pair_concordance`` or
``python -m nome_hic_concordance.pair_concordance``.

Operates on Step 4 posteriors from the main nome_hic_concordance pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("nome_hic_concordance.pair_concordance")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nome_hic_pair_concordance",
        description=(
            "Pair-concordance pipeline: contact-aware chromatin coordination "
            "analysis on top of nome_hic_concordance Step 4 posteriors."
        ),
    )

    # Required
    parser.add_argument(
        "--posteriors-dir", required=True,
        help="Directory with {cell}.posteriors.parquet from Step 4.",
    )
    parser.add_argument(
        "--pair-panel", required=True,
        help="Path to pair panel TSV/BEDPE (pair_id, chrom_i, start_i, end_i, "
             "chrom_j, start_j, end_j, distance, category, distance_band).",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory.",
    )

    # Optional inputs
    parser.add_argument(
        "--ground-truth", default=None,
        help="TSV with cell_name, true_label columns (for validation mode).",
    )
    parser.add_argument(
        "--samples", default=None,
        help="Sample manifest TSV (for cell list and group labels). "
             "If not provided, discovers cells from posteriors-dir.",
    )

    # Model parameters
    parser.add_argument(
        "--posterior-threshold", type=float, default=0.5,
        help="Threshold for binarising posteriors (default: 0.5).",
    )
    parser.add_argument(
        "--min-gch-sites", type=int, default=3,
        help="Minimum GCH sites per read end (default: 3).",
    )
    parser.add_argument(
        "--min-unlinked", type=int, default=5,
        help="Minimum unlinked reads per anchor for reliable p_i/p_j (default: 5).",
    )
    parser.add_argument(
        "--eb-pool-by", choices=["category", "category_band"], default="category",
        help="EB prior pooling level (default: category).",
    )

    # Embedding
    parser.add_argument(
        "--n-pca", type=int, default=20,
        help="Number of PCA components for embeddings (default: 20).",
    )
    parser.add_argument(
        "--pca-method", choices=["masked", "mean_impute"], default="masked",
        help="PCA method for E_coord (default: masked).",
    )

    # Execution
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument(
        "--steps", type=str, default="all",
        help="Steps to run: comma-separated (5p,6p,7p,8p,9p) or 'all'.",
    )

    return parser.parse_args(argv)


def _parse_steps(spec: str) -> set[str]:
    spec = spec.strip().lower()
    if spec == "all":
        return {"5p", "6p", "7p", "8p", "9p"}
    return {s.strip() for s in spec.split(",") if s.strip()}


def _discover_cells(posteriors_dir: str) -> list[str]:
    """Discover cell names from posteriors parquet filenames."""
    p = Path(posteriors_dir)
    files = sorted(p.glob("*.posteriors.parquet"))
    return [f.stem.replace(".posteriors", "") for f in files]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out / "pipeline_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    steps = _parse_steps(args.steps)
    logger.info(f"Running steps: {sorted(steps)}")

    # Discover cells
    if args.samples:
        samples = pd.read_csv(args.samples, sep="\t", dtype=str)
        if "cell_name" not in samples.columns:
            samples.columns = ["cell_name", "bam_path"] + list(samples.columns[2:])
        cell_names = samples["cell_name"].tolist()
        labels = samples.set_index("cell_name").get("group")
    else:
        cell_names = _discover_cells(args.posteriors_dir)
        labels = None

    if args.ground_truth:
        gt = pd.read_csv(args.ground_truth, sep="\t", dtype=str)
        if "cell_name" in gt.columns and "true_label" in gt.columns:
            labels = gt.set_index("cell_name")["true_label"]
        elif gt.shape[1] >= 2:
            gt.columns = ["cell_name", "true_label"] + list(gt.columns[2:])
            labels = gt.set_index("cell_name")["true_label"]

    logger.info(f"Found {len(cell_names)} cells")

    # ---- Step 5p: Load pair panel ----------------------------------------
    from . import panel as panel_mod

    logger.info("=" * 60)
    logger.info("Step 5p: Loading pair panel")
    logger.info("=" * 60)
    pair_panel = panel_mod.load_panel(args.pair_panel)
    logger.info(f"  Panel: {len(pair_panel)} pairs, "
                f"{pair_panel['category'].nunique()} categories")

    # ---- Step 6p: Linked/unlinked aggregation ----------------------------
    from . import classify as classify_mod

    counts_dir = out / "06p_pair_counts"
    if "6p" in steps:
        logger.info("=" * 60)
        logger.info("Step 6p: Aggregating linked/unlinked counts per cell")
        logger.info("=" * 60)

        cell_count_paths = classify_mod.aggregate_all_cells(
            posteriors_dir=args.posteriors_dir,
            cell_names=cell_names,
            panel=pair_panel,
            output_dir=str(counts_dir),
            n_threads=args.n_threads,
            posterior_threshold=args.posterior_threshold,
            min_gch_sites=args.min_gch_sites,
        )
    else:
        logger.info("Step 6p: Skipped (loading existing pair counts)")
        cell_count_paths = {
            c: str(counts_dir / f"{c}.pair_counts.parquet")
            for c in cell_names
        }

    # Stack all cell counts
    all_counts = []
    for cell, path in cell_count_paths.items():
        if Path(path).exists():
            df = pd.read_parquet(path)
            all_counts.append(df)
    if all_counts:
        pair_counts_stacked = pd.concat(all_counts, ignore_index=True)
    else:
        pair_counts_stacked = pd.DataFrame()
    logger.info(f"  Total pair-cell observations: {len(pair_counts_stacked):,}")

    # ---- Step 7p: EB priors + hierarchical Δ -----------------------------
    from . import empirical_bayes as eb_mod
    from . import hierarchical as hier_mod

    delta_dir = out / "07p_delta"
    if "7p" in steps:
        logger.info("=" * 60)
        logger.info("Step 7p: Empirical Bayes priors + Δ computation")
        logger.info("=" * 60)

        delta_dir.mkdir(parents=True, exist_ok=True)

        # Fit priors
        priors = eb_mod.fit_priors(
            pair_counts_stacked, pair_panel, pool_by=args.eb_pool_by,
        )
        priors.to_csv(delta_dir / "eb_priors.tsv", sep="\t", index=False)
        logger.info(f"  Fitted priors for {len(priors)} groups")

        # Compute Δ for all cells
        delta_matrix, delta_var_matrix = hier_mod.compute_all_deltas(
            cell_pair_count_paths=cell_count_paths,
            priors=priors,
            panel=pair_panel,
            output_dir=str(delta_dir),
            n_threads=args.n_threads,
            min_unlinked=args.min_unlinked,
        )
    else:
        logger.info("Step 7p: Skipped (loading existing Δ matrices)")
        delta_matrix = pd.read_parquet(delta_dir / "delta_matrix.parquet")
        delta_var_matrix = pd.read_parquet(delta_dir / "delta_var_matrix.parquet")

    # ---- Step 8p: Three embeddings + UMAP --------------------------------
    from . import embed as embed_mod
    from . import visualization as viz_mod

    embed_dir = out / "08p_embeddings"
    if "8p" in steps:
        logger.info("=" * 60)
        logger.info("Step 8p: Building embeddings (E_contact, E_coord, E_full)")
        logger.info("=" * 60)

        embed_dir.mkdir(parents=True, exist_ok=True)

        # E_contact
        logger.info("  Building E_contact...")
        emb_contact = embed_mod.build_contact_embedding(
            pair_counts_stacked, pair_panel,
            n_pca=args.n_pca, random_state=args.random_seed,
        )

        # E_coord
        logger.info("  Building E_coord...")
        # Build delta_z matrix from delta and delta_var
        delta_z_matrix = delta_matrix / np.sqrt(delta_var_matrix.replace(0, np.nan))
        emb_coord = embed_mod.build_coord_embedding(
            delta_z_matrix,
            n_pca=args.n_pca,
            pca_method=args.pca_method,
            random_state=args.random_seed,
        )

        # E_full
        logger.info("  Building E_full...")
        # Align cell sets
        shared_cells = sorted(
            set(emb_contact["cell_names"]) & set(emb_coord["cell_names"])
        )
        if len(shared_cells) >= 2:
            c_idx = [emb_contact["cell_names"].index(c) for c in shared_cells]
            d_idx = [emb_coord["cell_names"].index(c) for c in shared_cells]
            contact_feat = emb_contact["matrix"][c_idx]
            coord_feat = emb_coord["matrix"][d_idx]
            emb_full = embed_mod.build_full_embedding(
                contact_feat, coord_feat, shared_cells,
                n_pca=args.n_pca, random_state=args.random_seed,
            )
        else:
            logger.warning("Too few shared cells for E_full")
            emb_full = {"umap": None, "cell_names": [], "pca": None}

        embeddings = {
            "E_contact": emb_contact,
            "E_coord": emb_coord,
            "E_full": emb_full,
        }

        # Save & plot
        viz_mod.plot_three_umaps(embeddings, labels, str(embed_dir))

        # Coverage diagnostic
        viz_mod.plot_coverage_diagnostic(
            pair_counts_stacked, pair_panel,
            str(embed_dir / "coverage_diagnostic.png"),
        )

        # Δ distribution
        viz_mod.plot_delta_distribution(
            delta_matrix, pair_panel,
            str(embed_dir / "delta_distribution.png"),
        )

    # ---- Step 9p: Component-contribution analysis ------------------------
    from . import component_analysis as comp_mod

    analysis_dir = out / "09p_analysis"
    if "9p" in steps and labels is not None:
        logger.info("=" * 60)
        logger.info("Step 9p: Component-contribution analysis")
        logger.info("=" * 60)

        analysis_dir.mkdir(parents=True, exist_ok=True)

        # AUC per category for each embedding
        for emb_name, emb in embeddings.items():
            if emb.get("pca") is None:
                continue
            logger.info(f"  AUC analysis for {emb_name}...")
            auc_df = comp_mod.auc_per_category(
                emb, labels, pair_panel,
                delta_matrix=delta_z_matrix if emb_name == "E_coord" else None,
                pair_counts=pair_counts_stacked if emb_name == "E_contact" else None,
                random_state=args.random_seed,
            )
            auc_df["embedding"] = emb_name
            auc_df.to_csv(
                analysis_dir / f"auc_{emb_name}.tsv", sep="\t", index=False,
            )
            viz_mod.plot_auc_forest(
                auc_df, str(analysis_dir / f"auc_forest_{emb_name}.png"),
            )

        # Marginal contribution
        if emb_contact.get("pca") is not None and emb_coord.get("pca") is not None:
            logger.info("  Marginal contribution of E_coord over E_contact...")
            # Align PCA arrays
            if len(shared_cells) >= 10:
                c_pca = emb_contact["pca"][c_idx]
                d_pca = emb_coord["pca"][d_idx]
                shared_labels = labels.reindex(shared_cells).dropna()
                if len(shared_labels.unique()) >= 2:
                    mc = comp_mod.marginal_contribution(
                        c_pca, d_pca, shared_labels,
                        random_state=args.random_seed,
                    )
                    with open(analysis_dir / "marginal_contribution.json", "w") as f:
                        json.dump(mc, f, indent=2)
                    logger.info(
                        f"    ΔAUC = {mc['delta_auc']:.4f}, "
                        f"LR p = {mc['lr_pvalue']:.2e}"
                    )

    elif "9p" in steps:
        logger.warning("Step 9p: Skipped (no ground-truth labels for AUC analysis)")

    logger.info("Pair-concordance pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
