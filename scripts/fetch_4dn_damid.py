#!/usr/bin/env python3
"""Fetch 4DN LaminB1 DamID-seq tracks, and optionally derive LAD calls.

The 4DN portal has LaminB1 DamID for seven human cell types, processed to
DAM-ONLY NORMALIZED bigWigs at several bin sizes including 250 kb -- already
the grid the concordance matrices use.  The signal is a signed log ratio of
Dam-LaminB1 over the Dam-only control, so positive means lamina-associated and
the sign is the natural cut.  Published LAD calls fit a two-state HMM along the
chromosome instead (Guelen et al. Nature 2008), which borrows strength between
neighbouring bins; --derive-lad-bed does not, so its borders are noisier than a
released LAD track even though the genome fraction comes out about right.

Two things make this awkward by hand, which is why the script exists:

  * the portal's own download URL returns 403 without an account, while the
    same file is served unauthenticated from the ``open_data_url`` S3 mirror
    recorded on each file;
  * picking the right files means filtering ~933 DamID bigWigs down to the
    LaminB1 ones (the rest are nucleolar targeting, which gives NADs, not
    LADs), at one bin size, replicate-averaged.

    python scripts/fetch_4dn_damid.py --output-dir annots/lad_4dn \
        --derive-lad-bed

Use the bigWigs as CONTINUOUS tracks -- ``--signal-track`` in
marker_bin_enrichment.py or bin_set_signal_summary.py -- rather than as a
region annotation: no threshold has to be chosen, and lamina association is a
graded quantity.  ``--derive-lad-bed`` additionally writes binary calls per
cell type plus the constitutive / facultative / constitutive-inter partition
across them, for the coverage-style enrichment.

The signal is SIGNED, so in bin_set_signal_summary --layout single use
--normalize zscore; log2-median refuses it, correctly, since it is already a
log ratio.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _4dn import cell_name, download, search  # noqa: E402


logger = logging.getLogger("fetch_4dn_damid")


def query_files(assembly="GRCh38", timeout=180):
    """Every DamID-seq bigWig record the portal lists, as JSON."""
    return search({"type": "FileProcessed",
                   "file_format.file_format": "bw",
                   "track_and_facet_info.experiment_type": "DamID-seq"},
                  timeout=timeout)


def select(records, assembly="GRCh38", bin_size="250kb", replicate_mean=True):
    """LaminB1, dam-only normalised, one bin size, one file per cell type.

    LaminB1 and not the nucleolar-targeting DamID in the same collection: those
    give NADs, which overlap LADs substantially but are a different thing.
    """
    out = {}
    for r in records:
        tf = r.get("track_and_facet_info") or {}
        desc = r.get("description") or ""
        if tf.get("condition") != "LaminB1-Dam":
            continue
        if r.get("genome_assembly") != assembly:
            continue
        if not desc.startswith(f"{bin_size} bin dam-only normalized"):
            continue
        if replicate_mean and "mean of replicates" not in desc:
            continue
        cell = cell_name(tf.get("biosource_name"))
        out.setdefault(cell, r)
    return dict(sorted(out.items()))


def bigwig_grid(path, grid_bp, chroms=None):
    """Per-bin mean on a fixed grid → (bins DataFrame, values array).

    A fixed grid rather than the file's own intervals: the point is to compare
    cell types bin by bin, and that needs one grid they all share even if their
    stored intervals differ at the ends.
    """
    import pandas as pd
    try:
        import pyBigWig
    except ImportError:
        raise SystemExit("reading a bigWig needs pyBigWig (pip install pyBigWig)")
    bw = pyBigWig.open(str(path))
    rows, vals = [], []
    for chrom, size in bw.chroms().items():
        if chroms and chrom not in chroms:
            continue
        starts = np.arange(0, size, grid_bp, dtype=np.int64)
        ends = np.minimum(starts + grid_bp, size)
        stat = bw.stats(chrom, 0, int(size), type="mean",
                        nBins=int(len(starts)))
        for s, e, v in zip(starts, ends, stat):
            rows.append((chrom, int(s), int(e)))
            vals.append(np.nan if v is None else float(v))
    bw.close()
    bins = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    return bins, np.asarray(vals, float)


def lad_partition(calls):
    """{cell: boolean LAD mask} → (constitutive, facultative, inter) masks.

    fLAD is the UNION of the per-cell-type calls minus the constitutive ones --
    a bin that is a LAD in at least one cell type but not in all.  All three
    come from the same masks, so they partition the grid and their fractions do
    sum to one; that is correct here, and is only a warning sign when ciLAD
    comes from a published list instead, since then a bin can be in neither the
    observed union nor that list.

    What the counts DO depend on is how many cell types went in: a bin has to
    be a LAD in all of them to be constitutive, so cLAD shrinks and fLAD grows
    as cell types are added.  The three sets are not comparable across runs
    with different --cell-type lists.
    """
    stack = np.vstack([np.asarray(v, bool) for v in calls.values()])
    n = stack.shape[0]
    hits = stack.sum(axis=0)
    constitutive = hits == n
    union = hits > 0
    return constitutive, union & ~constitutive, hits == 0


def merge_bins(frame):
    """Contiguous bins -> the domains they compose.

    These BEDs are built by selecting bins, so without this a file called
    cLAD.bed holds the 250 kb pieces of a cLAD rather than the cLAD.  Every
    consumer then has to know to merge -- bedtools, IGV and a reader counting
    "how many domains" all take each row at face value, and a boundary
    analysis sees a boundary at every bin edge inside a domain.
    """
    if not len(frame):
        return frame
    out = []
    for chrom, g in frame.groupby("chrom", sort=False):
        g = g.sort_values("start")
        cur = None
        for s, e in zip(g.start.to_numpy(), g.end.to_numpy()):
            if cur is not None and s <= cur[1]:
                cur[1] = max(cur[1], e)
            else:
                if cur is not None:
                    out.append((chrom, *cur))
                cur = [s, e]
        if cur is not None:
            out.append((chrom, *cur))
    return pd.DataFrame(out, columns=["chrom", "start", "end"])


def write_bed(bins, mask, path):
    """Selected bins, merged into the domains they compose."""
    sel = bins.loc[np.asarray(mask, bool), ["chrom", "start", "end"]]
    dom = merge_bins(sel.sort_values(["chrom", "start"]))
    dom.to_csv(path, sep="\t", index=False, header=False)
    return int(np.sum(mask))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--bin-size", default="250kb",
                    help="Which processed bin size to take (250kb, 100kb, ...); "
                         "250 kb matches the concordance grid.")
    ap.add_argument("--cell-type", action="append", default=None,
                    help="Restrict to these (repeatable); default all seven.")
    ap.add_argument("--derive-lad-bed", action="store_true",
                    help="Also threshold each track into a binary LAD BED and "
                         "write the constitutive / facultative / "
                         "constitutive-inter partition across the cell types.")
    ap.add_argument("--lad-threshold", type=float, default=0.0,
                    help="Signal above this is a LAD (0.0: the signal is a log "
                         "ratio of LaminB1 over the Dam-only control, so zero "
                         "is the natural cut and gives the ~35%% genome "
                         "coverage LADs are known for).")
    ap.add_argument("--grid-bp", type=int, default=250_000,
                    help="Grid for the LAD calls and the partition.")
    ap.add_argument("--chroms", default=None,
                    help="Comma-separated; default every contig in the file.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    picked = select(query_files(args.assembly), args.assembly, args.bin_size)
    if args.cell_type:
        want = {c.strip() for c in args.cell_type}
        missing = want - set(picked)
        if missing:
            sys.exit(f"--cell-type {sorted(missing)} not among {sorted(picked)}")
        picked = {k: v for k, v in picked.items() if k in want}
    if not picked:
        sys.exit(f"no LaminB1 dam-only-normalized {args.bin_size} bigWig for "
                 f"{args.assembly}; try --bin-size 100kb.")
    logger.info(f"{len(picked)} cell types: {', '.join(picked)}")

    paths = {}
    for cell, rec in picked.items():
        paths[cell] = download(rec, out / f"{cell}_LMNB1_{args.bin_size}.bw")

    flags = " ".join(f"--signal-track LAD_{c}={p}" for c, p in paths.items())
    (out / "signal_track_flags.txt").write_text(flags + "\n")
    logger.info(f"wrote {out / 'signal_track_flags.txt'}")

    if args.derive_lad_bed:
        chroms = ({c.strip() for c in args.chroms.split(",")}
                  if args.chroms else None)
        bins = calls = None
        for cell, p in paths.items():
            b, v = bigwig_grid(p, args.grid_bp, chroms)
            if bins is None:
                bins, calls = b, {}
            elif not b[["chrom", "start"]].equals(bins[["chrom", "start"]]):
                sys.exit(f"{cell}: grid differs from the first track's")
            calls[cell] = np.isfinite(v) & (v > args.lad_threshold)
            n = write_bed(b, calls[cell], out / f"{cell}_LAD.bed")
            logger.info(f"  {cell}: {n:,}/{len(b):,} bins LAD "
                        f"({100 * n / len(b):.1f}%)")
        con, fac, inter = lad_partition(calls)
        man = []
        for name, mask in (("cLAD", con), ("fLAD", fac), ("ciLAD", inter)):
            n = write_bed(bins, mask, out / f"{name}.bed")
            logger.info(f"  {name}: {n:,} bins ({100 * n / len(bins):.1f}%)")
            man.append(f"{name}\t{(out / (name + '.bed')).resolve()}")
        (out / "annotation_manifest.tsv").write_text(
            "# LAD partition across " + ", ".join(paths) + "\n"
            + "\n".join(man) + "\n")
        logger.info(f"wrote {out / 'annotation_manifest.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
