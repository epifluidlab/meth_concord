"""CLI for the scHi-C compartment baseline (`nome_hic_compartments`).

Builds a cells × bins A/B-compartment-score matrix from the per-cell Hi-C
contacts parsed in Step 2 (``02_read_pair_counts/{cell}.read_pairs.parquet``)
— the scHi-C feature set consumed by ``nome_hic_compare``.

The compartment math is scHiCluster's recipe (per-cell RWR-imputed contact
matrix → O/E → correlation → PC1), implemented self-contained so it needs
no external scHiCluster install.

Example
-------
    nome_hic_compartments \\
        --read-pairs-dir output/.../context_GCH/02_read_pair_counts \\
        --reference hg38.2bit \\
        --resolution 1000000 \\
        --output-dir output/compartments
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import compartments

logger = logging.getLogger("nome_hic_compartments")


def _load_chrom_sizes(chrom_sizes_path: str | None,
                      reference_path: str | None) -> dict[str, int]:
    if chrom_sizes_path:
        sizes = {}
        with open(chrom_sizes_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                sizes[parts[0]] = int(parts[1])
        return sizes
    if reference_path:
        import py2bit
        tb = py2bit.open(reference_path)
        try:
            return dict(tb.chroms())
        finally:
            tb.close()
    raise ValueError("Provide --chrom-sizes or --reference.")


def _discover_cells(read_pairs_dir: str) -> dict[str, str]:
    d = Path(read_pairs_dir)
    paths = sorted(d.glob("*.read_pairs.parquet"))
    if not paths:
        raise ValueError(f"No *.read_pairs.parquet files in {read_pairs_dir}")
    return {p.name.replace(".read_pairs.parquet", ""): str(p) for p in paths}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nome_hic_compartments", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--read-pairs-dir", required=True,
                   help="Step-2 dir with {cell}.read_pairs.parquet files.")
    p.add_argument("--chrom-sizes", default=None,
                   help="Two-column TSV: chrom<TAB>size.")
    p.add_argument("--reference", default=None,
                   help=".2bit reference (alternative to --chrom-sizes).")
    p.add_argument("--resolution", type=int, default=1_000_000,
                   help="Bin size in bp (default 1 Mb, the compartment scale).")
    p.add_argument("--chroms", default=None,
                   help="Comma-separated chromosomes (default: 22 autosomes).")
    p.add_argument("--impute", choices=["rwr", "none"], default="rwr",
                   help="Per-cell contact-graph RWR denoising (scHiCluster "
                        "style) before compartment calling.")
    p.add_argument("--restart", type=float, default=0.5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-threads", type=int, default=4)
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
    with open(out / "compartments_config.json", "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)

    chrom_sizes = _load_chrom_sizes(args.chrom_sizes, args.reference)
    cell_paths = _discover_cells(args.read_pairs_dir)
    chroms = ([c.strip() for c in args.chroms.split(",")]
              if args.chroms else None)

    comp = compartments.compute_compartment_matrix(
        cell_paths, chrom_sizes,
        resolution=args.resolution, chroms=chroms,
        impute=args.impute, restart=args.restart,
        n_threads=args.n_threads, output_dir=str(out), save_outputs=True,
    )
    logger.info(f"compartment matrix: {comp.shape} → {out}")
    print(f"compartment_scores.parquet: {comp.shape[0]} cells × "
          f"{comp.shape[1]} bins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
