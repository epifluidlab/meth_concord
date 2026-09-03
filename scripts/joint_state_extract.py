#!/usr/bin/env python3
"""Part A: BAM -> per-read-end joint GCH/HCG/HCH counts and per-end posteriors.

    scripts/joint_state_extract.py \
        --samples mouse_cortex_1577_cells.samples.with_batches.fixed.tsv \
        --group-order "IT-L2/3,IT-L4,IT-L5/6,CT-L6,Inhib,ODC,OPC,ASC,MGC,PC/EC" \
        --context-bed GCH:GCH.bed.gz,HCG:HCG.bed.gz \
        -o output/joint

Two stages, because the prior is fitted across cells and the posterior is
evaluated per end:

  1. extract   every cell's BAM -> 02j_read_end_joint_counts/chrom=*/ , while
               accumulating the pooled sufficient statistics for stage 2;
  2. baseline  fit the aligned 2-D prior, evaluate per-end posteriors, and
               write them back alongside the raw counts.

``--steps`` runs either alone, so the baseline can be refitted -- a different
``--bin-size``, ``--min-obs`` or ``--gch-norm`` -- without re-reading a BAM.

The counts (k, n) are persisted next to every posterior (A.3.2).  Part B's
measurement-error correction is unambiguous with raw counts and needs an
empirical diagnostic without them, so they must survive any later cleanup.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nome_hic_concordance import joint_baseline as jb           # noqa: E402
from nome_hic_concordance.joint_state_extract import (          # noqa: E402
    extract_cell_to_shards)
from nome_hic_concordance.samples_manifest import (             # noqa: E402
    add_manifest_args, load_from_args)
from nome_hic_concordance.utils import (                        # noqa: E402
    add_exclude_chrom_arg, chrom_excluded, ensure_dir,
    parse_exclude_chroms, setup_logging)

logger = logging.getLogger("joint_state_extract")

COUNTS_DIR = "02j_read_end_joint_counts"
POSTERIOR_DIR = "03j_read_end_posteriors"
#: The prior sits BESIDE the partitioned tree, not inside it: anything dropped
#: into a Hive-partitioned directory is read back as another shard, and a table
#: with different columns comes back as a block of all-NaN rows rather than an
#: error.
PRIOR_NAME = "03j_prior.parquet"
META_NAME = "_metadata.json"


# --------------------------------------------------------------- arguments ---

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_manifest_args(p)

    g = p.add_argument_group("contexts")
    g.add_argument("--reference", help=".2bit reference; required without "
                                      "--context-bed.")
    g.add_argument("--context-bed", help="MARK:PATH[,MARK:PATH]; precomputed "
                                        "site BEDs, as the main pipeline uses.")
    g.add_argument("--context-bed-offset", type=int, default=0)
    g.add_argument("--contexts", default="GCH,HCG",
                   help="Marks to score, in output column order (default: "
                        "GCH,HCG).  HCH is NOT listed here -- it is derived "
                        "from each read's own reference span.")
    g.add_argument("--dbsnp-vcf", default=None,
                   help="Mask these positions out of every context, and count "
                        "them per end as n_masked_snp.")

    g = p.add_argument_group("extraction")
    g.add_argument("--mapq-min", type=int, default=30)
    g.add_argument("--trim-ends", type=int, default=5)
    g.add_argument("--baseq-min", type=int, default=0)
    g.add_argument("--strand-min-evidence", type=int, default=1)
    g.add_argument("--min-distance", type=int, default=1000,
                   help="Contact separation floor (default 1000).  Residual "
                        "self-ligation is handled in Part B's distance "
                        "stratification, not by raising this.")
    g.add_argument("--keep-supplementary", action="store_true")
    g.add_argument("--include-trans", action="store_true",
                   help="Keep trans contacts (is_cis False).  Off by default: "
                        "buffering unpaired reads across the whole BAM costs "
                        "memory proportional to the trans rate.")
    g.add_argument("--blacklist-bed", default=None)
    add_exclude_chrom_arg(p, default=["chrM,chrX,chrY"])

    g = p.add_argument_group("per-end requirements")
    g.add_argument("--min-gch-sites", type=int, default=2)
    g.add_argument("--min-hcg-sites", type=int, default=1)
    g.add_argument("--min-conversion", type=float, default=0.95,
                   help="Per-MOLECULE HCH conversion floor.  Ends are MARKED "
                        "(conversion_pass), never dropped: A.3.5 requires "
                        "k_hch/n_hch to survive so the threshold can be "
                        "revisited without re-extraction.")

    g = p.add_argument_group("baseline (A.3.2-A.3.4)")
    g.add_argument("--bin-size", type=int, default=250_000)
    g.add_argument("--strata-per-decade", type=int, default=3)
    g.add_argument("--min-obs", type=int, default=50,
                   help="Informative ends a (bin x stratum) cell needs, for "
                        "EVERY mark, before it gets its own prior.")
    g.add_argument("--gch-norm", choices=jb.GCH_NORM_CHOICES,
                   default="location",
                   help="A.3.4: location only.  'full' exists as the B.7.6 "
                        "ablation arm and is not a sensible default.")

    g = p.add_argument_group("run")
    g.add_argument("-o", "--output", required=True)
    g.add_argument("--threads", type=int, default=4)
    g.add_argument("--steps", default="extract,baseline",
                   help="Comma-delimited subset of extract,baseline.")
    g.add_argument("--overwrite", action="store_true",
                   help="Re-extract cells whose shard already exists.")
    return p


# ----------------------------------------------------------------- contexts ---

def _load_contexts(args, marks, keep_chrom):
    """``{mark: {chrom: structured array}}`` plus the SNP positions."""
    from nome_hic_concordance.reference import (
        extract_context_positions, load_context_from_bed)

    if args.context_bed:
        from nome_hic_concordance.cli import _parse_context_bed
        spec = _parse_context_bed(args.context_bed)
        missing = [m for m in marks if m not in spec]
        if missing:
            raise SystemExit(f"--context-bed has no entry for {missing}; "
                             f"it provides {sorted(spec)}")
        cp = load_context_from_bed(spec, n_threads=args.threads,
                                   c_offset=args.context_bed_offset)
    else:
        if not args.reference:
            raise SystemExit("--reference is required without --context-bed.")
        cp = extract_context_positions(args.reference, contexts=list(marks),
                                       dbsnp_vcf=args.dbsnp_vcf,
                                       n_threads=args.threads)

    cp = {m: {c: a for c, a in cp[m].items() if keep_chrom(c)} for m in marks}
    for m in marks:
        total = sum(len(a) for a in cp[m].values())
        logger.info("  %s: %s sites on %d contigs", m, f"{total:,}", len(cp[m]))
        if total == 0:
            raise SystemExit(
                f"{m} has no sites after --exclude-chrom.  Check "
                f"--context-bed-offset and the contig naming.")

    snp = {}
    if args.dbsnp_vcf and args.context_bed:
        logger.warning(
            "--dbsnp-vcf with --context-bed: the BEDs were built upstream, so "
            "masking is only counted (n_masked_snp), not applied.  Rebuild the "
            "BEDs with the VCF to actually remove those sites.")
    return cp, snp


# ------------------------------------------------------------------ stage 1 ---

def _extract_args(args) -> dict:
    return dict(trim_ends=args.trim_ends, mapq_min=args.mapq_min,
                baseq_min=args.baseq_min,
                strand_min_evidence=args.strand_min_evidence,
                keep_supplementary=args.keep_supplementary,
                min_distance=args.min_distance,
                cis_only=not args.include_trans)


def run_extract(args, manifest, cp, snp, marks, root: Path):
    from joblib import Parallel, delayed

    rows = [r for _, r in manifest.table.iterrows()]
    if not args.overwrite:
        done = {p.stem for p in root.glob("chrom=*/*.parquet")}
        skip = [r for r in rows if r["name"] in done]
        if skip:
            logger.info("  %d cell(s) already extracted; --overwrite to redo",
                        len(skip))
        rows = [r for r in rows if r["name"] not in done]
    if not rows:
        return {}

    logger.info("Extracting %d cell(s) on %d thread(s)", len(rows), args.threads)
    kw = dict(marks=marks, snp_positions=snp,
              min_gch_sites=args.min_gch_sites,
              min_hcg_sites=args.min_hcg_sites,
              min_conversion=args.min_conversion,
              blacklist_bed=args.blacklist_bed, **_extract_args(args))
    fn = delayed(extract_cell_to_shards)
    jobs = [fn(r["bam"], r["name"], cp, str(root), **kw) for r in rows]
    out = (Parallel(n_jobs=args.threads)(jobs)
           if args.threads > 1 and len(jobs) > 1
           else [extract_cell_to_shards(r["bam"], r["name"], cp, str(root), **kw)
                 for r in rows])

    stats = {}
    for cell, _, st in out:
        stats[cell] = st
        logger.info("  %s: %s/%s ends kept (%ss)", cell,
                    f"{st['n_ends_written']:,}", f"{st['n_ends_raw']:,}",
                    st["elapsed_s"])
    return stats


# ------------------------------------------------------------------ stage 2 ---

def _chrom_sizes(cp, marks) -> dict[str, int]:
    """Contig extents from the context arrays themselves.

    Deriving them from a BAM header instead would make the bin grid depend on
    which cell happened to be read first.
    """
    sizes: dict[str, int] = {}
    for m in marks:
        for c, arr in cp[m].items():
            if len(arr):
                sizes[c] = max(sizes.get(c, 0), int(arr["pos"][-1]) + 1)
    return dict(sorted(sizes.items()))


def _bin_grid(sizes, bin_size):
    """Global bin numbering: ``{chrom: (offset, n_bins)}`` and the total."""
    grid, off = {}, 0
    for c, ln in sizes.items():
        nb = int(ln // bin_size) + 1
        grid[c] = (off, nb)
        off += nb
    return grid, off


def _shard_chrom(path: Path) -> str:
    """The chromosome a shard holds, from its Hive partition directory."""
    name = path.parent.name
    if not name.startswith("chrom="):
        raise SystemExit(f"{path} is not under a chrom=... partition")
    return name.split("=", 1)[1]


def _cell_ids(df, chrom, grid, bin_size, edges, n_bins, n_strata):
    """Grid-cell index per read end, and its distance stratum.

    One shard holds one chromosome, so the bin offset is a single lookup.
    """
    pos = np.minimum(df["pos_5p"].to_numpy(), df["pos_3p"].to_numpy())
    off, nb = grid.get(chrom, (-1, 0))
    if off < 0:
        bin_id = np.full(len(df), -1, dtype=np.int64)
    else:
        local = pos // bin_size
        bin_id = np.where(local < nb, off + local, -1)
    strat = jb.distance_strata(df["contact_dist"].to_numpy(), edges)
    cell, _, _, _ = jb.grid_index(bin_id, strat, n_bins, n_strata)
    return cell, strat


def run_baseline(args, cp, marks, root: Path, out_root: Path,
                 prior_path: Path):
    from nome_hic_concordance.distance_stats import build_edges

    sizes = _chrom_sizes(cp, marks)
    grid, n_bins = _bin_grid(sizes, args.bin_size)
    edges = build_edges(args.min_distance, max(sizes.values()),
                        args.strata_per_decade)
    n_strata = len(edges)          # last slot is trans
    logger.info("Baseline grid: %d bins x %d distance strata = %s cells",
                n_bins, n_strata, f"{n_bins * n_strata:,}")

    shards = sorted(root.glob("chrom=*/*.parquet"))
    if not shards:
        raise SystemExit(f"no shards under {root}; run --steps extract first")
    cols = ["pos_5p", "pos_3p", "contact_dist"] + \
           [f"{p}_{m.lower()}" for m in marks for p in ("k", "n")]

    # Pass 1 -- the global mean per mark, which is the origin the accumulator
    # centres on.  Cheap: two sums per shard.
    tot_k = np.zeros(len(marks))
    tot_n = np.zeros(len(marks))
    for s_ in shards:
        d = pd.read_parquet(s_, columns=[c for c in cols if c.startswith(("k_", "n_"))])
        for i, m in enumerate(marks):
            tot_k[i] += float(d[f"k_{m.lower()}"].sum())
            tot_n[i] += float(d[f"n_{m.lower()}"].sum())
    center = np.where(tot_n > 0, tot_k / np.maximum(tot_n, 1), 0.5)
    logger.info("  global rates: %s",
                ", ".join(f"{m}={c:.4f}" for m, c in zip(marks, center)))

    # Pass 2 -- accumulate per grid cell, in sorted shard order so the fit does
    # not depend on how the shards happened to be listed.
    n_cells = n_bins * n_strata
    accs = [jb.PriorAccumulator.zeros(n_cells, center[i])
            for i in range(len(marks))]
    for s_ in shards:
        d = pd.read_parquet(s_, columns=cols)
        cid, _ = _cell_ids(d, _shard_chrom(s_), grid, args.bin_size, edges,
                           n_bins, n_strata)
        for i, m in enumerate(marks):
            accs[i].add(d[f"k_{m.lower()}"], d[f"n_{m.lower()}"], cid)

    alpha = np.stack([a.fit(args.min_obs)[0] for a in accs])
    beta = np.stack([a.fit(args.min_obs)[1] for a in accs])
    n_obs = np.stack([a.cnt for a in accs])
    usable = np.isfinite(alpha).all(axis=0) & np.isfinite(beta).all(axis=0)
    g_alpha = np.empty(len(marks))
    g_beta = np.empty(len(marks))
    for i in range(len(marks)):
        nu = jb.NU_TIGHT
        g_alpha[i], g_beta[i] = center[i] * nu, (1 - center[i]) * nu
        alpha[i] = np.where(usable, alpha[i], g_alpha[i])
        beta[i] = np.where(usable, beta[i], g_beta[i])
    logger.info("  %d/%d grid cells have their own prior for BOTH marks",
                int(usable.sum()), n_cells)

    bl = jb.JointBaseline(
        marks=tuple(marks), alpha=alpha, beta=beta, n_obs=n_obs, usable=usable,
        global_alpha=g_alpha, global_beta=g_beta,
        bin_of_cell=np.repeat(np.arange(n_bins), n_strata),
        stratum_of_cell=np.tile(np.arange(n_strata), n_bins))

    # Pass 3 -- per-end posteriors, written beside the counts.
    ensure_dir(out_root)
    n_written = 0
    for s_ in shards:
        d = pd.read_parquet(s_)
        cid, strat = _cell_ids(d, _shard_chrom(s_), grid, args.bin_size,
                               edges, n_bins, n_strata)
        d["grid_cell"] = cid
        d["distance_stratum"] = strat
        for i, m in enumerate(marks):
            key = m.lower()
            mean, var = bl.evaluate(m, d[f"k_{key}"], d[f"n_{key}"], cid)
            mu = np.where(cid >= 0, bl.mu(m)[np.maximum(cid, 0)],
                          g_alpha[i] / (g_alpha[i] + g_beta[i]))
            d[f"theta_{key}"] = mean
            d[f"var_{key}"] = var
            d[f"mu_{key}"] = mu
            mode = args.gch_norm if m.upper() == "GCH" else "location"
            d[f"centred_{key}"] = jb.normalize(mean, mu, var, mode=mode)
        out = out_root / s_.parent.name / s_.name
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(out, index=False)
        n_written += len(d)
    logger.info("  wrote %s ends with posteriors to %s", f"{n_written:,}",
                out_root)

    pd.DataFrame({
        "grid_cell": np.arange(n_cells),
        "bin": bl.bin_of_cell, "stratum": bl.stratum_of_cell,
        "usable": usable,
        **{f"alpha_{m.lower()}": alpha[i] for i, m in enumerate(marks)},
        **{f"beta_{m.lower()}": beta[i] for i, m in enumerate(marks)},
        **{f"n_obs_{m.lower()}": n_obs[i] for i, m in enumerate(marks)},
    }).to_parquet(prior_path, index=False)
    return bl, {"edges": edges.tolist(), "n_bins": n_bins,
                "n_strata": int(n_strata), "global_rate": center.tolist()}


# --------------------------------------------------------------------- main ---

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    unknown = steps - {"extract", "baseline"}
    if unknown:
        raise SystemExit(f"--steps: unknown {sorted(unknown)}")

    manifest = load_from_args(args)
    manifest.report()

    marks = [m.strip() for m in args.contexts.split(",") if m.strip()]
    if len(marks) < 2:
        raise SystemExit("--contexts needs at least two marks; the analysis is "
                         "a covariance between them.")
    excluded = parse_exclude_chroms(args.exclude_chrom)
    keep = lambda c: not chrom_excluded(c, excluded)          # noqa: E731

    out = ensure_dir(args.output)
    root = ensure_dir(out / COUNTS_DIR)
    cp, snp = _load_contexts(args, marks, keep)

    meta = {
        "command": " ".join(sys.argv),
        "marks": marks,
        "hch": "derived per read from CIGAR+MD, not from a site table",
        "gch_norm": args.gch_norm,
        "min_gch_sites": args.min_gch_sites,
        "min_hcg_sites": args.min_hcg_sites,
        "min_conversion": args.min_conversion,
        "min_conversion_applied": "marked as conversion_pass, not dropped",
        "min_distance": args.min_distance,
        "bin_size": args.bin_size,
        "min_obs": args.min_obs,
        "cis_only": not args.include_trans,
        "manifest": manifest.metadata(),
    }

    if "extract" in steps:
        meta["extract_stats"] = run_extract(args, manifest, cp, snp, marks, root)
    if "baseline" in steps:
        bl, grid_meta = run_baseline(args, cp, marks, root,
                                     ensure_dir(out / POSTERIOR_DIR),
                                     out / PRIOR_NAME)
        meta["baseline"] = {**bl.metadata(), **grid_meta}

    (out / META_NAME).write_text(json.dumps(meta, indent=2, default=str))
    logger.info("Done -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
