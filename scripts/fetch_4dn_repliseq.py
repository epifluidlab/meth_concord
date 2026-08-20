#!/usr/bin/env python3
"""Fetch 4DN 2-stage Repli-seq and build log2(early/late) timing tracks.

4DN has 2-stage Repli-seq for twelve human cell types plus an engineered iPSC
line.  Unlike the DamID collection it does NOT publish a replication-timing
track: what is released is the two fractions SEPARATELY, as 5 kb RPKM
bedGraphs (``condition`` = early / late).  The ratio has to be built here.

    python scripts/fetch_4dn_repliseq.py --output-dir annots/rt_4dn

WHY THE RATIO IS BUILT AT 250 kb AND NOT AT 5 kb.  The order of operations is
the whole game.  At native 5 kb resolution 14% of early bins and 10% of late
bins are exactly zero, so a per-5-kb log2 ratio is dominated by whatever
pseudocount is chosen -- and averaging those ratios up to a coarser grid
averages the pseudocount, not the biology.  Summing each fraction over the
target bin FIRST and taking one ratio of the totals is the ratio of counts
that was actually measured: at 250 kb, 94% of bins are non-zero in both
fractions and the pseudocount stops mattering.  Ratio-of-means and
mean-of-ratios are not the same number, and only the first one is meaningful
here.

Replicates are averaged within a fraction before the ratio, for the same
reason -- the mean of the per-replicate totals, so the result does not depend
on how many replicates a cell type happens to have.

The tracks are written as bedGraph, which marker_bin_enrichment.py and
bin_set_signal_summary.py both read directly via --signal-track.  The score is
a log ratio, so it is SIGNED: positive is early-replicating, and
bin_set_signal_summary needs --normalize zscore rather than log2-median.

--derive-domain-bed additionally writes early/late domain BEDs at the sign cut,
for coverage-style enrichment.  That is a cut through a continuous quantity
with no gap at zero, so prefer the track unless a discrete set is required.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _4dn import cell_name, download, search, write_flags  # noqa: E402

logger = logging.getLogger("fetch_4dn_repliseq")

_FRACTION = re.compile(r"\b(early|late)\b", re.I)


def query_files(assembly="GRCh38", timeout=180):
    """Every 2-stage Repli-seq bedGraph the portal lists."""
    return search({"type": "FileProcessed",
                   "file_format.file_format": "bg",
                   "track_and_facet_info.experiment_type": "2-stage Repli-seq"},
                  timeout=timeout)


def group(records, assembly="GRCh38"):
    """-> {label: {"early": [rec, ...], "late": [rec, ...]}}.

    ``condition`` is 'early'/'late' for the ordinary cell types but carries a
    genotype for the engineered iPSC line ('IZ boundary deletion early
    fraction').  That genotype is kept in the label, because collapsing the
    three genotypes of one line into a single name would silently average
    edited and unedited cells together.  Records whose condition names neither
    fraction -- the pre-computed E/L ratios released for that line only -- are
    left out; this script builds the ratio itself.
    """
    out = {}
    for rec in records:
        if rec.get("genome_assembly") != assembly:
            continue
        info = rec.get("track_and_facet_info") or {}
        cond = str(info.get("condition") or "")
        hit = _FRACTION.search(cond)
        if not hit:
            continue
        frac = hit.group(1).lower()
        # Whatever the condition says besides the fraction is a genotype or
        # treatment and belongs in the label.
        extra = _FRACTION.sub("", cond).replace("fraction", "").strip(" -_")
        label = cell_name(info.get("biosource_name"))
        if extra:
            label = f"{label}_{re.sub(r'[^0-9A-Za-z]+', '_', extra).strip('_')}"
        out.setdefault(label, {}).setdefault(frac, []).append(rec)
    # A fraction on its own cannot make a ratio.
    return {k: v for k, v in sorted(out.items())
            if v.get("early") and v.get("late")}


def binned_totals(path, grid_bp, chroms=None):
    """Sum a bedGraph's values into ``grid_bp`` bins -> Series on (chrom, idx).

    The 4DN grid is 5 kb and the useful targets are multiples of it, so an
    interval sits inside one output bin and integer division places it
    exactly.  A grid that does not divide the native width would split
    intervals, and rather than apportion them silently that case raises.
    """
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     compression="infer", dtype={0: str},
                     names=["chrom", "start", "end", "value"])
    if chroms:
        df = df[df.chrom.isin(set(chroms))]
    width = (df.end - df.start).mode()
    native = int(width.iloc[0]) if len(width) else grid_bp
    if grid_bp % native:
        raise SystemExit(
            f"--grid-bp {grid_bp:,} is not a multiple of the {native:,} bp "
            f"native bin, so intervals would straddle output bins.  Use a "
            f"multiple of {native:,}.")
    idx = df.start.to_numpy() // grid_bp
    return df.groupby([df.chrom.to_numpy(), idx]).value.sum()


def timing(early_paths, late_paths, grid_bp, chroms=None, min_total=1e-6):
    """log2(early/late) per bin, replicates averaged within each fraction.

    Returns (frame, n_dropped).  Bins where either fraction totals ~zero are
    dropped rather than pseudocounted: at 250 kb those are unmappable regions
    (centromeres, gaps), and a pseudocount would turn them into confident
    extreme values instead of the missing data they are.
    """
    def mean_of(paths):
        parts = [binned_totals(p, grid_bp, chroms) for p in paths]
        return pd.concat(parts, axis=1).fillna(0.0).mean(axis=1)

    early, late = mean_of(early_paths), mean_of(late_paths)
    both = early.index.intersection(late.index)
    early, late = early.loc[both], late.loc[both]
    ok = (early > min_total) & (late > min_total)
    out = pd.DataFrame({
        "chrom": [i[0] for i in both[ok]],
        "start": np.array([i[1] for i in both[ok]], dtype=np.int64) * grid_bp,
        "value": np.log2(early[ok].to_numpy() / late[ok].to_numpy())})
    out["end"] = out.start + grid_bp
    out = out[["chrom", "start", "end", "value"]].sort_values(
        ["chrom", "start"]).reset_index(drop=True)
    return out, int((~ok).sum())


def write_bedgraph(df, path):
    df.to_csv(path, sep="\t", index=False, header=False,
              float_format="%.5f")
    return len(df)


def write_domains(df, path, threshold=0.0, early=True):
    sel = df.value > threshold if early else df.value < threshold
    df.loc[sel, ["chrom", "start", "end"]].to_csv(
        path, sep="\t", index=False, header=False)
    return int(sel.sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--assembly", default="GRCh38",
                    help="GRCh38 (human) or GRCm38 (mouse).")
    ap.add_argument("--cell-type", action="append", default=None,
                    help="Restrict to these (repeatable); default all.")
    ap.add_argument("--grid-bp", type=int, default=250_000,
                    help="Bin for the ratio.  Must be a multiple of the 5 kb "
                         "native bin; 250 kb matches the concordance grid and "
                         "is coarse enough that the zero bins stop mattering.")
    ap.add_argument("--chroms", default=None,
                    help="Comma-separated; default every contig in the file.")
    ap.add_argument("--keep-fractions", action="store_true",
                    help="Also write the binned early and late totals, for "
                         "checking the ratio against its inputs.")
    ap.add_argument("--derive-domain-bed", action="store_true",
                    help="Also write early/late domain BEDs at the sign cut.")
    ap.add_argument("--list", action="store_true",
                    help="List what is available and exit without downloading.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    chroms = ([c.strip() for c in args.chroms.split(",")]
              if args.chroms else None)

    picked = group(query_files(args.assembly), args.assembly)
    if not picked:
        raise SystemExit(f"no 2-stage Repli-seq bedGraphs for {args.assembly}")
    if args.cell_type:
        want = {c.strip() for c in args.cell_type}
        missing = want - set(picked)
        if missing:
            raise SystemExit(f"not available: {', '.join(sorted(missing))}\n"
                             f"have: {', '.join(sorted(picked))}")
        picked = {k: v for k, v in picked.items() if k in want}

    if args.list:
        for label, fr in picked.items():
            print(f"{label}: {len(fr['early'])} early / {len(fr['late'])} late")
        return 0

    out = Path(args.output_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    logger.info("%d cell types: %s", len(picked), ", ".join(picked))

    tracks = {}
    for label, fr in picked.items():
        paths = {}
        for frac in ("early", "late"):
            paths[frac] = [
                download(r, out / "raw" / f"{r['accession']}.bedGraph.gz")
                for r in fr[frac]]
        df, dropped = timing(paths["early"], paths["late"], args.grid_bp,
                             chroms)
        dest = out / f"{label}_RT_{args.grid_bp // 1000}kb.bedGraph"
        write_bedgraph(df, dest)
        tracks[f"RT_{label}"] = dest
        logger.info("  %s: %d bins (%d dropped, one fraction empty), "
                    "log2(E/L) %+.2f .. %+.2f, %.0f%% early",
                    label, len(df), dropped, df.value.min(), df.value.max(),
                    100 * (df.value > 0).mean())

        if args.keep_fractions:
            for frac in ("early", "late"):
                tot = pd.concat(
                    [binned_totals(p, args.grid_bp, chroms)
                     for p in paths[frac]], axis=1).fillna(0.0).mean(axis=1)
                f = pd.DataFrame({
                    "chrom": [i[0] for i in tot.index],
                    "start": np.array([i[1] for i in tot.index],
                                      dtype=np.int64) * args.grid_bp,
                    "value": tot.to_numpy()})
                f["end"] = f.start + args.grid_bp
                write_bedgraph(f[["chrom", "start", "end", "value"]]
                               .sort_values(["chrom", "start"]),
                               out / f"{label}_{frac}_"
                                     f"{args.grid_bp // 1000}kb.bedGraph")

        if args.derive_domain_bed:
            ne = write_domains(df, out / f"{label}_early.bed", early=True)
            nl = write_domains(df, out / f"{label}_late.bed", early=False)
            logger.info("    domains: %d early / %d late", ne, nl)

    write_flags(tracks, out / "signal_track_flags.txt")
    logger.info("wrote %d tracks + signal_track_flags.txt to %s",
                len(tracks), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
