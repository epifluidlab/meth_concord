#!/usr/bin/env python3
"""Export the 250 kb-bin signals as genome-browser bigWig (+ bedGraph) tracks.

Each input matrix is cells x 250 kb-bins (the pipeline's concordance_signed /
accessibility_mean / methylation_mean / compartment_scores parquet).  For every
signal this writes one track per cell group AND one for all cells merged, taking
the per-bin MEAN over the cells in that group.

Resolution.  The 250 kb-bin value is written out at a fine bin size (default
1 kb): each output bin inherits the value of the 250 kb bin that CONTAINS its
centre (a step track -- no invented values), stored as one interval per source
bin (equal 1 kb bins merged, an identical but far smaller track).  ``--smooth``
instead linearly interpolates between adjacent 250 kb-bin centres to give a
smooth 1 kb track.

Outputs ``<signal>.<group>.bw`` (needs pyBigWig) and ``<signal>.<group>.bedgraph``
(always).  Chromosome names are matched with a leading 'chr' stripped.

Example
-------
    concordance_bigwig.py \\
      --signal gch_concordance=out/context_GCH/05_concordance/concordance_signed.parquet \\
      --signal gch_accessibility=out/context_GCH/05_concordance/accessibility_mean.parquet \\
      --signal hcg_concordance=out/context_HCG/05_concordance/concordance_signed.parquet \\
      --signal hcg_methylation=out/context_HCG/05_concordance/accessibility_mean.parquet \\
      --signal compartment=out/compartments/compartment_scores.parquet \\
      --samples mouse_cortex.samples.tsv --cell-type-column group \\
      --chrom-sizes mm10.chrom.sizes --output-dir out/bigwig
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _canon(chrom) -> str:
    s = str(chrom)
    return s[3:] if s[:3].lower() == "chr" else s


def _chrom_key(chrom):
    c = _canon(chrom)
    return (0, int(c)) if c.isdigit() else (1, c)


def _read_chrom_sizes(path) -> dict:
    """{canonical contig: length} from a chrom.sizes / .fai (first 2 columns)."""
    sizes = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 2:
                try:
                    sizes[_canon(p[0])] = int(p[1])
                except ValueError:
                    continue
    return sizes


def _bin_grid(columns) -> pd.DataFrame:
    """Parse bin labels 'chrom:start' -> DataFrame(chrom, start, width) aligned to
    ``columns``; width is the inferred (modal) bin size."""
    chrom, start = [], []
    for c in columns:
        s = str(c)
        if ":" in s:
            a, b = s.rsplit(":", 1)
            try:
                start.append(int(b))
                chrom.append(a)
                continue
            except ValueError:
                pass
        chrom.append(s)
        start.append(0)
    g = pd.DataFrame({"chrom": chrom, "start": start})
    diffs = (g.sort_values(["chrom", "start"]).groupby("chrom")["start"].diff()
             .dropna())
    diffs = diffs[diffs > 0]
    width = int(diffs.mode().iloc[0]) if len(diffs) else 1
    g["width"] = width
    return g


def _group_means(matrix: pd.DataFrame, groups: pd.Series, which: str,
                 min_cells: int):
    """{group_label: per-bin mean Series}.  'all' = every cell in the matrix;
    per-group = cells whose --cell-type-column matches (>= min_cells)."""
    out = {}
    if which in ("both", "merged"):
        out["all"] = matrix.mean(axis=0, skipna=True)
    if which in ("both", "per-group"):
        g = groups.reindex(matrix.index)
        for grp, idx in matrix.groupby(g.values).groups.items():
            if grp is None or str(grp) in ("nan", "None", ""):
                continue
            if len(idx) < min_cells:
                continue
            out[str(grp)] = matrix.loc[idx].mean(axis=0, skipna=True)
    return out


def _intervals(values: pd.Series, grid: pd.DataFrame, sizes: dict,
               smooth: bool, bin_size: int):
    """(chrom, start, end, value) intervals for one track, sorted by chrom/start
    and clamped to the contig length.  Step (one per source bin) unless
    ``smooth``, which interpolates to ``bin_size`` between source-bin centres."""
    v = values.values.astype(float)
    df = grid.assign(value=v)
    out = []
    for chrom, gd in df.groupby("chrom"):
        cap = sizes.get(_canon(chrom))
        gd = gd.sort_values("start")
        s = gd["start"].values
        w = gd["width"].values
        val = gd["value"].values
        fin = np.isfinite(val)
        if not fin.any():
            continue
        if not smooth:
            for i in np.flatnonzero(fin):
                end = s[i] + w[i]
                if cap is not None:
                    end = min(end, cap)
                if end > s[i]:
                    out.append((chrom, int(s[i]), int(end), float(val[i])))
        else:
            centres = (s + w / 2.0)[fin]
            fval = val[fin]
            if centres.size < 2:
                continue
            lo = int(s[fin].min())
            hi = int((s[fin] + w[fin]).max())
            if cap is not None:
                hi = min(hi, cap)
            starts = np.arange(lo, hi, bin_size)
            mids = starts + bin_size / 2.0
            iv = np.interp(mids, centres, fval)
            ends = np.minimum(starts + bin_size, hi)
            for st, en, vv in zip(starts, ends, iv):
                if en > st:
                    out.append((chrom, int(st), int(en), float(vv)))
    out.sort(key=lambda r: (_chrom_key(r[0]), r[1]))
    return out


def _write_bedgraph(path: Path, intervals):
    with open(path, "w") as fh:
        for chrom, s, e, v in intervals:
            fh.write(f"{chrom}\t{s}\t{e}\t{v:.6g}\n")


def _write_bigwig(path: Path, intervals, sizes: dict) -> bool:
    """Write a bigWig via pyBigWig; return False (with a note) if unavailable."""
    try:
        import pyBigWig
    except ImportError:
        return False
    # Header: contigs that carry intervals, in natural order, with their length.
    present = {}
    for chrom, _, e, _ in intervals:
        cap = sizes.get(_canon(chrom))
        present[chrom] = max(present.get(chrom, 0), e if cap is None else cap)
    header = [(c, int(present[c])) for c in sorted(present, key=_chrom_key)]
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader(header)
    for chrom, _ in header:
        rows = [iv for iv in intervals if iv[0] == chrom]
        if rows:
            bw.addEntries([chrom] * len(rows), [r[1] for r in rows],
                          ends=[r[2] for r in rows],
                          values=[float(r[3]) for r in rows])
    bw.close()
    return True


def _safe(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(name))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", action="append", required=True,
                    metavar="NAME=PATH",
                    help="cells x 250 kb-bin parquet, e.g. "
                         "gch_concordance=.../concordance_signed.parquet. "
                         "Repeatable.")
    ap.add_argument("--samples", required=True,
                    help="Samples sheet mapping cell -> group.")
    ap.add_argument("--cell-type-column", default="cell_type",
                    help="Group column in --samples (default cell_type).")
    ap.add_argument("--chrom-sizes", required=True, metavar="FILE",
                    help="chrom.sizes / .fai; the bigWig header + interval "
                         "clamping.")
    ap.add_argument("--exclude-chrom", action="append", default=None)
    ap.add_argument("--groups", choices=["both", "merged", "per-group"],
                    default="both",
                    help="Write per-cell-group tracks, the merged all-cells "
                         "track, or both (default).")
    ap.add_argument("--min-cells", type=int, default=1,
                    help="Skip a group with fewer than this many cells.")
    ap.add_argument("--smooth", action="store_true",
                    help="Linearly interpolate the 250 kb-bin-centre values to "
                         "--bin-size (a smooth track) instead of a step track.")
    ap.add_argument("--bin-size", type=int, default=1000,
                    help="Fine output resolution in bp for --smooth "
                         "(default 1000).")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    from nome_hic_concordance.cli import load_samples
    from nome_hic_concordance.utils import (bin_label_chrom, chrom_excluded,
                                            parse_exclude_chroms)

    sizes = _read_chrom_sizes(args.chrom_sizes)
    excl = parse_exclude_chroms(args.exclude_chrom)
    samples = load_samples(args.samples)
    if args.cell_type_column not in samples.columns:
        sys.exit(f"--cell-type-column {args.cell_type_column!r} not in "
                 f"{args.samples} (columns: {samples.columns.tolist()}).")
    samples["cell_name"] = samples["cell_name"].astype(str)
    groups = samples.set_index("cell_name")[args.cell_type_column].astype(str)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_bw = n_bg = 0
    warned_bw = False
    for spec in args.signal:
        if "=" not in spec:
            sys.exit(f"--signal must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not Path(path).exists():
            sys.exit(f"no such signal matrix: {path}")
        M = pd.read_parquet(path)
        M.index = M.index.astype(str)
        if excl:
            M = M[[b for b in M.columns
                   if not chrom_excluded(bin_label_chrom(b), excl)]]
        grid = _bin_grid(M.columns)
        tracks = _group_means(M, groups, args.groups, args.min_cells)
        print(f"{name}: {M.shape[0]} cells x {M.shape[1]} bins "
              f"({grid['width'].iloc[0] if len(grid) else '?'} bp), "
              f"{len(tracks)} tracks", file=sys.stderr)
        for grp, vals in tracks.items():
            ivs = _intervals(vals, grid, sizes, args.smooth, args.bin_size)
            if not ivs:
                continue
            stem = f"{_safe(name)}.{_safe(grp)}"     # e.g. gch_concordance.all
            _write_bedgraph(out_dir / f"{stem}.bedgraph", ivs)
            n_bg += 1
            if _write_bigwig(out_dir / f"{stem}.bw", ivs, sizes):
                n_bw += 1
            elif not warned_bw:
                warned_bw = True
                print("  pyBigWig not installed -> wrote bedGraph only.  For "
                      "bigWig: `pip install pyBigWig`, or convert with "
                      "`bedGraphToBigWig <sorted.bedgraph> "
                      f"{args.chrom_sizes} out.bw`.", file=sys.stderr)
    print(f"wrote {n_bg} bedGraph and {n_bw} bigWig tracks to {out_dir}")


if __name__ == "__main__":
    main()
