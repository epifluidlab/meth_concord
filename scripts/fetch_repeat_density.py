#!/usr/bin/env python3
"""Per-bin repeat density from the UCSC RepeatMasker table.

Repeat classes are the right SHAPE for a 250 kb bin, but not as regions.  An
individual element is a few hundred bp, so no bin is ever half covered by one
and a region test at majority overlap returns an empty table.  What is
domain-scale is their DENSITY: LINEs cover ~20% of a chromosome and SINEs
~14%, but that coverage is far from uniform -- LINE-rich and Alu-rich
territory are megabase-scale and track compartment, lamina association and
replication timing.  So each class is written as a continuous track, to be
passed with --signal-track rather than --annotation.

    python scripts/fetch_repeat_density.py --genome hg38 --output-dir annots/repeats

--family adds family-level tracks (L1, Alu, MIR, ...) and --name adds named
subfamilies.  The young ones are the interesting ones: L1HS and AluY are
still mobile, and where they sit is not where the ancient L1M/AluJ copies
sit, so collapsing them into "LINE" averages two different distributions.

Density is a FRACTION of the bin, computed on merged intervals so overlapping
annotations of the same class cannot push it above 1.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("fetch_repeat_density")

_UCSC = "https://hgdownload.soe.ucsc.edu/goldenPath"

# rmsk.txt.gz columns (0-based) -- the table has a leading 'bin' column.
_CHROM, _START, _END, _NAME, _CLASS, _FAMILY = 5, 6, 7, 10, 11, 12

#: Classes worth a track by default.  The '?' variants are RepeatMasker's
#: uncertain calls and are folded into their parent rather than given a track.
_DEFAULT_CLASSES = ("LINE", "SINE", "LTR", "DNA", "Satellite",
                    "Simple_repeat", "Low_complexity")


def download(url, dest: Path, timeout=900):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("  cached %s", dest.name)
        return dest
    logger.info("  downloading %s ...", dest.name)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        tmp.write_bytes(fh.read())
    tmp.replace(dest)
    return dest


def chrom_sizes(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 2:
                out[p[0]] = int(p[1])
    return out


def accumulate(rmsk_path, sizes, grid_bp, want_class, want_family, want_name,
               chroms=None):
    """Stream rmsk and sum covered bp per (track, chrom, bin).

    Streamed rather than loaded: the hg38 table is 155 MB gzipped and ~5.6 M
    rows, and only three columns of it are ever needed.

    Intervals are clipped to the bin they start in and carried across bin
    boundaries, so an element spanning a boundary contributes to both bins in
    the proportion it actually covers -- not wholly to the first.
    """
    acc = defaultdict(lambda: defaultdict(float))
    keep = set(chroms) if chroms else None
    n = 0
    with gzip.open(rmsk_path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            chrom = f[_CHROM]
            if chrom not in sizes or (keep and chrom not in keep):
                continue
            cls, fam, nm = f[_CLASS], f[_FAMILY], f[_NAME]
            tracks = []
            if cls in want_class:
                tracks.append(cls)
            if fam in want_family:
                tracks.append(f"fam_{fam}")
            if nm in want_name:
                tracks.append(f"rep_{nm}")
            if not tracks:
                continue
            start, end = int(f[_START]), int(f[_END])
            n += 1
            b0, b1 = start // grid_bp, (end - 1) // grid_bp
            for b in range(b0, b1 + 1):
                lo = max(start, b * grid_bp)
                hi = min(end, (b + 1) * grid_bp)
                if hi > lo:
                    for t in tracks:
                        acc[t][(chrom, b)] += hi - lo
    logger.info("  %d intervals contributed to %d tracks", n, len(acc))
    return acc


def to_bedgraph(counts, sizes, grid_bp, path):
    """{(chrom, bin): bp} -> a bedGraph of the covered FRACTION of each bin.

    Every bin of every contig is emitted, including the zeros: a bin with no
    element of this class has density 0, which is a measurement, and dropping
    it would let the enrichment treat those bins as missing instead.
    """
    rows = []
    for chrom, length in sizes.items():
        nb = -(-length // grid_bp)
        starts = np.arange(nb) * grid_bp
        ends = np.minimum(starts + grid_bp, length)
        bp = np.array([counts.get((chrom, b), 0.0) for b in range(nb)])
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(ends > starts, bp / (ends - starts), np.nan)
        rows.append(pd.DataFrame({"chrom": chrom, "start": starts,
                                  "end": ends, "value": frac}))
    df = pd.concat(rows, ignore_index=True)
    df = df[np.isfinite(df.value)]
    df.to_csv(path, sep="\t", index=False, header=False, float_format="%.5f")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--genome", default="hg38",
                    help="UCSC db name (hg38, hg19, mm10, mm39).")
    ap.add_argument("--grid-bp", type=int, default=250_000)
    ap.add_argument("--cache-dir", default=None,
                    help="Where rmsk.txt.gz and chrom.sizes are kept "
                         "(default: <output-dir>/cache).  The hg38 table is "
                         "155 MB, so it is worth reusing.")
    ap.add_argument("--class", dest="classes", action="append", default=None,
                    metavar="CLASS",
                    help=f"Repeat class track (repeatable).  Default: "
                         f"{', '.join(_DEFAULT_CLASSES)}.")
    ap.add_argument("--family", action="append", default=None, metavar="FAMILY",
                    help="Family-level track, e.g. L1, Alu, MIR, ERVL "
                         "(repeatable).")
    ap.add_argument("--name", action="append", default=None, metavar="NAME",
                    help="Named subfamily, e.g. L1HS, AluY (repeatable).  The "
                         "young elements are worth separating: they are still "
                         "mobile and do not sit where the ancient copies sit.")
    ap.add_argument("--chroms", default=None,
                    help="Comma-separated; default every contig in "
                         "chrom.sizes.")
    ap.add_argument("--main-chroms-only", action="store_true",
                    help="Drop the alts, randoms and chrUn, which carry "
                         "extreme repeat density and no bins to match it.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir) if args.cache_dir else out / "cache"

    sizes_path = download(f"{_UCSC}/{args.genome}/bigZips/{args.genome}"
                          f".chrom.sizes", cache / f"{args.genome}.chrom.sizes")
    rmsk_path = download(f"{_UCSC}/{args.genome}/database/rmsk.txt.gz",
                         cache / f"rmsk_{args.genome}.txt.gz")
    sizes = chrom_sizes(sizes_path)
    if args.main_chroms_only:
        sizes = {c: n for c, n in sizes.items()
                 if "_" not in c and c not in ("chrM",)}
    if args.chroms:
        want = {c.strip() for c in args.chroms.split(",")}
        missing = want - set(sizes)
        if missing:
            raise SystemExit(f"not in chrom.sizes: {sorted(missing)}")
        sizes = {c: n for c, n in sizes.items() if c in want}

    want_class = set(args.classes or _DEFAULT_CLASSES)
    want_family = set(args.family or [])
    want_name = set(args.name or [])
    logger.info("%d contigs, %d kb bins; classes=%s families=%s names=%s",
                len(sizes), args.grid_bp // 1000, sorted(want_class),
                sorted(want_family) or "-", sorted(want_name) or "-")

    acc = accumulate(rmsk_path, sizes, args.grid_bp, want_class, want_family,
                     want_name, chroms=set(sizes))
    missing = ((want_class | {f"fam_{f}" for f in want_family}
                | {f"rep_{n}" for n in want_name}) - set(acc))
    if missing:
        raise SystemExit(
            f"no interval matched {sorted(missing)}.  Class, family and name "
            f"are matched exactly and are case-sensitive in the UCSC table "
            f"(LINE / L1 / L1HS, not line / l1).")

    tracks = {}
    for track, counts in sorted(acc.items()):
        path = out / f"{track}_density_{args.grid_bp // 1000}kb.bedGraph"
        df = to_bedgraph(counts, sizes, args.grid_bp, path)
        tracks[f"rpt_{track}"] = path
        logger.info("  %-18s mean %.3f  median %.3f  max %.3f  -> %s",
                    track, df.value.mean(), df.value.median(),
                    df.value.max(), path.name)

    flags = " ".join(f"--signal-track {k}={v.resolve()}"
                     for k, v in sorted(tracks.items()))
    (out / "signal_track_flags.txt").write_text(flags + "\n")
    logger.info("wrote %d tracks + signal_track_flags.txt to %s",
                len(tracks), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
