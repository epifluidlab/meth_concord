#!/usr/bin/env python3
"""Split GCH sites into GCA and GCY (GCC, GCT) by the base after the C.

NOMe-seq reads M.CviPI's GpC methylation as accessibility at GCH sites, but
a GCA is also a CpA: in neurons, whose DNA carries abundant non-CpG
methylation (mCA), endogenous mCA at GCA reads as accessible.  So neuron
studies count accessibility at GCY alone -- snmCAT-seq (Luo et al. 2022,
Cell Genomics) "the GCY (Y = T, C) context ... as the chromatin
accessibility signal".  This writes a GCH BED's sites as two BEDs, GCA and
GCY, bgzipped and tabix-indexed for nome_hic_concordance --context-bed,
every line as it was.

The C is at start + --c-offset (as the pipeline's --context-bed-offset),
its strand from column 6 (BED6) or column 4 (+/-, C/G), else plus; the base
after it on its own strand from --fasta -- plus: the next base; minus: the
previous one, complemented.  A site whose own base is not the C (plus) or
G (minus), or not after a G on its strand -- a wrong offset -- is counted
and left out.

    python scripts/split_context_bed.py \\
        --bed GCH_motif_seqkit.mm10.bed.gz --fasta mm10.fa --c-offset 1 \\
        --out-prefix GCH_motif_seqkit.mm10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("split_context_bed")

_COMP = bytes.maketrans(b"ACGTN", b"TGCAN")


def classify(seq: bytes, pos, minus) -> np.ndarray:
    """Per site (the C at ``pos``, 0-based; ``minus`` its strand): 'GCA',
    'GCY', or '' -- not a GpC on its strand, or GCG."""
    arr = np.frombuffer(seq.upper(), dtype=np.uint8)
    n = arr.size
    pos = np.asarray(pos, np.int64)
    ok = (pos >= 1) & (pos < n - 1)
    p = np.clip(pos, 1, n - 2)
    here, before, after = arr[p], arr[p - 1], arr[p + 1]
    comp = np.frombuffer(_COMP, dtype=np.uint8)
    # on the site's own strand: the C, the G before it, the base after it
    own = np.where(minus, comp[here], here)
    prev = np.where(minus, comp[after], before)
    nxt = np.where(minus, comp[before], after)
    good = ok & (own == ord("C")) & (prev == ord("G"))
    out = np.full(pos.size, "", dtype=object)
    out[good & (nxt == ord("A"))] = "GCA"
    out[good & ((nxt == ord("C")) | (nxt == ord("T")))] = "GCY"
    return out


def _strand(df: pd.DataFrame) -> np.ndarray:
    """Each row's minus-strandedness, as the pipeline reads it."""
    if df.shape[1] >= 6 and df[5].isin(["+", "-"]).all():
        return (df[5] == "-").to_numpy()
    if df.shape[1] >= 4 and df[3].isin(["+", "-", "C", "G"]).all():
        return df[3].isin(["-", "G"]).to_numpy()
    return np.zeros(len(df), bool)


def split_bed(bed, fasta, c_offset: int, out_prefix, chunk: int = 2_000_000):
    """Write ``out_prefix``_GCA.bed.gz and _GCY.bed.gz (tabix-indexed);
    returns the counts per class, '' those left out."""
    import pysam
    fa = pysam.FastaFile(str(fasta))
    outs = {k: Path(f"{out_prefix}_{k}.bed") for k in ("GCA", "GCY")}
    fh = {k: open(p, "w") for k, p in outs.items()}
    counts = {"GCA": 0, "GCY": 0, "": 0}
    cache = {}
    for df in pd.read_csv(bed, sep="\t", header=None, comment="#", dtype=str,
                          chunksize=chunk, compression="infer"):
        df = df[~df[0].str.startswith(("track", "browser"))]
        minus = _strand(df)
        start = df[1].astype(np.int64).to_numpy()
        cls = np.full(len(df), "", dtype=object)
        for c in pd.unique(df[0]):
            m = (df[0] == c).to_numpy()
            if c not in fa.references:
                continue
            if c not in cache:
                cache.clear()               # one chromosome at a time
                cache[c] = fa.fetch(c).encode()
            cls[m] = classify(cache[c], start[m] + c_offset, minus[m])
        for k in ("GCA", "GCY"):
            sel = cls == k
            counts[k] += int(sel.sum())
            if sel.any():
                df[sel].to_csv(fh[k], sep="\t", header=False, index=False)
        counts[""] += int((cls == "").sum())
    for f in fh.values():
        f.close()
    for k, p in outs.items():
        pysam.tabix_compress(str(p), str(p) + ".gz", force=True)
        pysam.tabix_index(str(p) + ".gz", preset="bed", force=True)
        p.unlink()
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bed", required=True, help="The GCH sites' BED.")
    ap.add_argument("--fasta", required=True,
                    help="The genome (indexed FASTA) the BED is on.")
    ap.add_argument("--c-offset", type=int, default=1,
                    help="The C's offset from the BED start, as the "
                         "pipeline's --context-bed-offset (seqkit locate: "
                         "1).")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    n = split_bed(args.bed, args.fasta, args.c_offset, args.out_prefix)
    tot = sum(n.values())
    logger.info(f"{tot:,} sites: GCA {n['GCA']:,} ({n['GCA'] / max(tot, 1):.1%}),"
                f" GCY {n['GCY']:,} ({n['GCY'] / max(tot, 1):.1%}); left out "
                f"{n['']:,} ({n[''] / max(tot, 1):.2%}) -- not a GpC on their "
                f"strand at this offset, or GCG")
    if n[""] > 0.05 * max(tot, 1):
        logger.warning("over 5% of the sites are not a GpC on their strand: "
                       "is --c-offset right?")
    for k in ("GCA", "GCY"):
        logger.info(f"  {args.out_prefix}_{k}.bed.gz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
