#!/usr/bin/env python3
"""Per cell, the methylation level of each cytosine context, from its BAM.

NOMe-seq's GpC labeling marks GCA and GCY alike, so in a cell without
non-CpG methylation the two read the same.  A neuron's DNA carries mCA, and
at a GCA -- also a CpA -- it adds to the label: GCA less GCY is that cell's
endogenous mCA as GCH sees it (split_context_bed.py makes the two BEDs).
Per cell and context: the sites methylated, those read, the level
(context_levels.tsv); per group (--group-column) each context's median and
GCA less GCY with its 95% interval over cells (context_levels_groups.tsv);
the figure context_levels.<fmt>.

Reads as the pipeline reads them (nome_hic_concordance.bam_parser): its
flags, --mapq-min, the conversion strand from the read's mismatches, the
sites of that strand, --trim-ends.  --chroms keeps it quick; a few
chromosomes hold ample sites for one cell's level.

    python scripts/context_levels.py --samples samples.tsv \\
        --group-column cell_type \\
        --context-bed GCA:GCH_motif.mm10_GCA.bed.gz,GCY:GCH_motif.mm10_GCY.bed.gz,HCG:HCG_motif.mm10.bed.gz \\
        --chroms chr1,chr2 --n-threads 10 --output-dir output/context_levels
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance import bam_parser as _bp  # noqa: E402
from nome_hic_concordance import reference as _ref  # noqa: E402
from nome_hic_concordance.utils import manifest_text  # noqa: E402

logger = logging.getLogger("context_levels")


def cell_levels(bam_path, sites: dict, chroms, trim_ends: int,
                mapq_min: int) -> dict:
    """One cell: per context the sites methylated and read, over
    ``chroms``."""
    import pysam
    pysam.set_verbosity(0)
    out = {ctx: [0, 0] for ctx in sites}
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for chrom in chroms:
            here = {ctx: s.get(chrom) for ctx, s in sites.items()}
            if all(h is None for h in here.values()):
                continue
            try:
                reads = bam.fetch(chrom)
            except ValueError:
                continue
            for read in reads:
                if read.flag & _bp._SKIP_FLAGS or \
                        read.mapping_quality < mapq_min:
                    continue
                for ctx, h in here.items():
                    if h is None:
                        continue
                    k, n = _bp._count_context_in_read(read, h[0], h[1],
                                                      trim_ends)
                    out[ctx][0] += k
                    out[ctx][1] += n
    return out


def _task(cell, group, *args):
    return cell, group, cell_levels(*args)


def group_table(t: pd.DataFrame, contexts, pairs, n_boot: int = 1000,
                seed: int = 0) -> pd.DataFrame:
    """Per group each context's median level, and each ``pairs`` (a, b)
    difference a less b: its median over cells and the 95% interval of
    that median."""
    rng = np.random.default_rng(seed)
    rows = []
    for g, x in t.groupby("group", sort=False):
        r = {"group": g, "cells": len(x)}
        for c in contexts:
            r[f"{c}_median"] = float(x[f"{c}_level"].median())
        for a, b in pairs:
            d = (x[f"{a}_level"] - x[f"{b}_level"]).dropna().to_numpy()
            if not d.size:
                continue
            bt = [np.median(d[rng.integers(0, d.size, d.size)])
                  for _ in range(n_boot)]
            lo, hi = np.percentile(bt, [2.5, 97.5])
            r[f"{a}_minus_{b}"] = float(np.median(d))
            r[f"{a}_minus_{b}_lo"], r[f"{a}_minus_{b}_hi"] = lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def plot_levels(t: pd.DataFrame, pairs, path):
    """Per group, each cell's level difference of each pair (pp)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not pairs:
        return
    groups = list(dict.fromkeys(t["group"]))
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.4 * len(pairs), 2.8),
                             squeeze=False)
    for ax, (a, b) in zip(axes[0], pairs):
        v = [100 * (t.loc[t["group"] == g, f"{a}_level"]
                    - t.loc[t["group"] == g, f"{b}_level"]).dropna()
             for g in groups]
        ax.boxplot(v, showfliers=False)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel(f"{a} − {b} level, pp", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-column", default="cell_name")
    ap.add_argument("--bam-column", default="bam_path")
    ap.add_argument("--group-column", default="group")
    ap.add_argument("--context-bed", required=True,
                    help="CTX:PATH[,CTX:PATH] tabix-indexed site BEDs.")
    ap.add_argument("--context-bed-offset", type=int, default=0)
    ap.add_argument("--chroms", default="chr1,chr2",
                    help="Chromosomes to read (comma-separated).")
    ap.add_argument("--pairs", default="GCA-GCY",
                    help="Context differences per cell, A-B (comma-"
                         "separated); those of contexts not given are "
                         "skipped.")
    ap.add_argument("--trim-ends", type=int, default=5)
    ap.add_argument("--mapq-min", type=int, default=30)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--plot-format", default="pdf")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    from joblib import Parallel, delayed
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    beds = {}
    for tok in args.context_bed.split(","):
        ctx, _, p = tok.strip().partition(":")
        beds[ctx.strip()] = p.strip()
    contexts = list(beds)
    chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
    raw = _ref.load_context_from_bed(beds, chroms=chroms,
                                     n_threads=args.n_threads,
                                     c_offset=args.context_bed_offset)
    sites = {ctx: {c: (a["pos"], a["ref_base"]) for c, a in d.items()}
             for ctx, d in raw.items()}
    del raw
    samples = pd.read_csv(manifest_text(args.samples), sep="\t", dtype=str)
    for col in (args.cell_column, args.bam_column, args.group_column):
        if col not in samples:
            sys.exit(f"--samples has no column {col!r}")
    res = Parallel(n_jobs=args.n_threads)(
        delayed(_task)(c, g, b, sites, chroms, args.trim_ends,
                       args.mapq_min)
        for c, g, b in zip(samples[args.cell_column],
                           samples[args.group_column],
                           samples[args.bam_column]))
    rows = []
    for cell, g, d in res:
        r = {"cell": cell, "group": g}
        for ctx in contexts:
            k, n = d[ctx]
            r[f"{ctx}_methylated"], r[f"{ctx}_sites"] = k, n
            r[f"{ctx}_level"] = k / n if n else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / "context_levels.tsv", sep="\t", index=False)
    pairs = [tuple(x.split("-", 1)) for x in args.pairs.split(",")
             if "-" in x]
    pairs = [(a, b) for a, b in pairs if a in contexts and b in contexts]
    gt = group_table(t, contexts, pairs)
    gt.to_csv(out / "context_levels_groups.tsv", sep="\t", index=False)
    plot_levels(t, pairs, out / f"context_levels.{args.plot_format}")
    logger.info("per group (median over cells):\n" + gt.to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
