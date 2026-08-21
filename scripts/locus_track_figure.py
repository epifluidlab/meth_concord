#!/usr/bin/env python3
"""A genome-browser panel for one locus: every cell, then the annotation tracks.

Picks the region where a contrast separates the arms most cleanly and draws it
the way a browser would -- single cells as heatmap rows grouped by cell type,
then the continuous tracks (replication timing, lamina association) and the
genes underneath, all on one x axis.

    python scripts/locus_track_figure.py \\
        --matrix gch=.../concordance_directional.parquet \\
        --scores out/marker_bins/bin_scores_gch.all.bed \\
        --samples samples.tsv --cell-type-column group \\
        --low tumor1,tumor2 --high GM12878,IMR-90,monocyte \\
        --signal-track "RT early=rt/HFFc6_early_250kb.bedGraph" \\
        --signal-track "RT late=rt/HFFc6_late_250kb.bedGraph" \\
        --signal-track "LaminB1=lad/HFFc6_LMNB1_250kb.bw" \\
        --gene-annotation gencode.v44.annotation.gtf.gz \\
        --output-dir out/locus

REGION CHOICE.  A bin scores on the GAP between the arms -- the highest of the
"low" group means subtracted from the lowest of the "high" group means -- so a
bin only scores when EVERY low group sits below EVERY high one.  A mean
difference would let one tumour population carry a bin the other does not
share, which is the thing this figure would be read as ruling out.  Runs of
adjacent qualifying bins are ranked by total gap, and the winner is drawn with
--flank-bins of context on each side so the edges of the domain are visible
rather than cropped to the claim.

SMOOTHING IS INTERPOLATION, NOT RESOLUTION.  D is stored per 250 kb bin, so a
250 kb window stepped every 10 kb is a rolling mean over overlapping bins: it
makes the track continuous instead of blocky and it moves no boundary it was
not already given.  Real 10 kb detail would mean recomputing concordance from
the read pairs at that step.  The axis label says which one this is.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _enrich():
    """The sibling enrichment script, for its readers."""
    p = Path(__file__).resolve().parent / "marker_bin_enrichment.py"
    spec = importlib.util.spec_from_file_location("_mbe", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_mbe"] = m
    spec.loader.exec_module(m)
    return m


def group_means(M, ct, groups):
    """{group: per-bin mean over its cells}, on the raw scale."""
    out = {}
    for g in groups:
        rows = M.loc[ct.index[ct == g]] if len(ct.index[ct == g]) else None
        if rows is None or not len(rows):
            continue
        out[g] = rows.mean(axis=0, skipna=True)
    return out


def separation_score(means, low, high):
    """Lowest high-group mean minus highest low-group mean, per bin.

    Positive only where EVERY low group sits below EVERY high one, which is
    what "consistently down in both tumours and up in the rest" means.  An
    average of the two sides would score a bin one tumour drives alone.
    """
    lo = [means[g] for g in low if g in means]
    hi = [means[g] for g in high if g in means]
    if not lo or not hi:
        raise SystemExit(
            f"need at least one group on each side; have low={list(lo)} "
            f"high={list(hi)} out of {sorted(means)}")
    worst_low = pd.concat(lo, axis=1).max(axis=1)
    worst_high = pd.concat(hi, axis=1).min(axis=1)
    return worst_high - worst_low


def best_run(bins, score, min_bins=2, require=None, min_gap=0.0):
    """Longest-then-strongest run of adjacent bins with a positive score.

    Adjacent in GENOMIC order on one chromosome, so a run cannot jump a gap or
    a centromere and be reported as one domain.
    """
    d = bins.copy()
    d["score"] = np.asarray(score, float)
    ok = np.isfinite(d["score"]) & (d["score"] > max(min_gap, 0.0))
    if require is not None:
        ok &= np.asarray(require, bool)
    d["ok"] = ok
    d = d.sort_values(["chrom", "start"]).reset_index(drop=True)
    runs, cur = [], None
    for i, r in d.iterrows():
        contiguous = (cur is not None and r["chrom"] == cur["chrom"]
                      and r["start"] == cur["end"])
        if r["ok"] and contiguous:
            cur["end"], cur["n"] = r["end"], cur["n"] + 1
            cur["total"] += r["score"]
        elif r["ok"]:
            if cur is not None:
                runs.append(cur)
            cur = {"chrom": r["chrom"], "start": r["start"], "end": r["end"],
                   "n": 1, "total": r["score"]}
        else:
            if cur is not None:
                runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    runs = [r for r in runs if r["n"] >= min_bins]
    if not runs:
        return None
    return max(runs, key=lambda r: (r["total"], r["n"]))


def smooth(values, starts, step, window):
    """Rolling mean of binned values onto a finer grid -> (positions, values).

    Interpolation between bins that already exist, which is a display choice.
    It cannot sharpen a boundary the 250 kb binning already blurred.
    """
    starts = np.asarray(starts, float)
    v = np.asarray(values, float)
    lo, hi = starts.min(), starts.max() + window
    grid = np.arange(lo, hi + step, step, dtype=float)
    out = np.full(grid.shape, np.nan)
    for i, g in enumerate(grid):
        m = (starts + window > g) & (starts < g + window)
        if m.any() and np.isfinite(v[m]).any():
            out[i] = np.nanmean(v[m])
    return grid, out


def robust_limits(values, vmin=None, vmax=None, lo_pct=1.0, hi_pct=99.0):
    """Colour limits from the data actually present.

    Forcing the scale symmetric about zero throws away half the range
    whenever the values do not straddle zero -- D over one locus often sits
    in a narrow positive band, and a -|max|..+|max| scale then paints every
    cell the same colour and hides exactly the differences the figure is for.
    Symmetric is kept only when the data really does span both signs, where
    zero is a meaningful midpoint.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0) if vmin is None else (vmin, vmax)
    lo = np.percentile(v, lo_pct) if vmin is None else vmin
    hi = np.percentile(v, hi_pct) if vmax is None else vmax
    if vmin is None and vmax is None and lo < 0 < hi:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    if hi <= lo:
        hi = lo + 1e-9
    return float(lo), float(hi)


def _gene_rows(genes, chrom, lo, hi, max_rows=4):
    """Genes overlapping the window, packed into non-overlapping rows.

    Packed rather than stacked one-per-row: a gene-dense locus would otherwise
    need a panel taller than the heatmap, and the point of the track is where
    the genes are, not how many rows they need.
    """
    g = genes[(genes.chrom.astype(str) == str(chrom))
              & (genes.end > lo) & (genes.start < hi)].copy()
    if not len(g):
        return g.assign(row=[])
    g = g.sort_values("start").reset_index(drop=True)
    ends = []
    rows = []
    span = max(hi - lo, 1)
    for _, r in g.iterrows():
        # Reserve label width so a name does not collide with the next gene.
        need = r["start"] - 0.06 * span
        for i, e in enumerate(ends):
            if e <= need:
                rows.append(i)
                ends[i] = r["end"]
                break
        else:
            rows.append(len(ends))
            ends.append(r["end"])
    g["row"] = np.minimum(rows, max_rows - 1)
    return g


def locus_figure(M, ct, order, region, tracks, genes, path, title,
                 ct_colors=None, step=10_000, window=250_000, bin_bp=250_000,
                 highlight=None, heat_groups=None, cytoband=None,
                 vmin=None, vmax=None, cmap="RdYlBu_r"):
    """Cells over the locus, then the tracks, then the genes -- one x axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chrom, lo, hi = region
    cols = [c for c in M.columns
            if str(c).rsplit(":", 1)[0] == str(chrom)
            and lo <= int(str(c).rsplit(":", 1)[1]) < hi]
    if not cols:
        raise SystemExit(f"no bins of {M.shape[1]} fall in "
                         f"{chrom}:{lo:,}-{hi:,}")
    starts = np.array([int(str(c).rsplit(":", 1)[1]) for c in cols], float)
    keep = ct.index.intersection(M.index)
    ct = ct.loc[keep]
    cells = [c for g in order for c in ct.index[ct == g]]
    sub = M.loc[cells, cols]

    # Every cell smoothed onto the same fine grid, so the heatmap and the
    # tracks below it share one coordinate system.
    grid, _ = smooth(sub.iloc[0].to_numpy(), starts, step, window)
    img = np.vstack([smooth(sub.loc[c].to_numpy(), starts, step, window)[1]
                     for c in cells])

    heat_groups = heat_groups or {}
    n_track = len(tracks)
    heights = ([len(cells) * 0.055 + 1.2]
               + [max(0.30 * len(v) + 0.30, 0.7) for v in heat_groups.values()]
               + [0.62] * n_track
               + [0.55]                      # dense gene track
               + ([0.42] if cytoband is not None else []))
    # x is shared by the locus panels only.  The ideogram shows the WHOLE
    # chromosome -- that is the point of it -- so sharing would force it to
    # the window and leave a bare rule where the chromosome should be.
    n_locus = len(heights) - (1 if cytoband is not None else 0)
    fig = plt.figure(figsize=(11.5, sum(heights) + 1.2))
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.16)
    axes = [fig.add_subplot(gs[0])]
    axes += [fig.add_subplot(gs[i], sharex=axes[0])
             for i in range(1, n_locus)]
    if cytoband is not None:
        axes.append(fig.add_subplot(gs[len(heights) - 1]))
    ax0 = axes[0]

    vlo, vhi = robust_limits(img, vmin, vmax)
    extent = [grid[0], grid[-1], len(cells), 0]
    im = ax0.imshow(img, aspect="auto", cmap=cmap, vmin=vlo, vmax=vhi,
                    extent=extent, interpolation="nearest")
    ax0.set_yticks([])
    ax0.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax0, fraction=0.012, pad=0.005)
    cb.set_label("concordance D", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Cell-type blocks down the left edge, and a rule between them, so a band
    # in the heatmap can be attributed without counting rows.
    pal = dict(ct_colors or {})
    fallback = plt.get_cmap("tab10")
    for i, g in enumerate(order):
        pal.setdefault(g, fallback(i % 10))
    # Blocks OUTSIDE the axes, in axes coordinates, so they never cover the
    # first column of data and the names have room to sit beside them.
    span = grid[-1] - grid[0]
    y = 0
    for g in order:
        n = int((ct == g).sum())
        if not n:
            continue
        ax0.add_patch(plt.Rectangle(
            (grid[0] - 0.022 * span, y), 0.014 * span, n,
            facecolor=pal[g], edgecolor="none", clip_on=False))
        ax0.text(grid[0] - 0.030 * span, y + n / 2, f"{g}\n(n={n})",
                 ha="right", va="center", fontsize=8, color=pal[g],
                 linespacing=1.25)
        y += n
        if y < len(cells):
            ax0.axhline(y, color="white", lw=1.2)

    nxt = 1
    for gname, members in heat_groups.items():
        ax = axes[nxt]
        nxt += 1
        rows, labels = [], []
        for lab, vals in members.items():
            rows.append(smooth(vals, starts, step, window)[1])
            labels.append(lab)
        block = np.vstack(rows)
        glo, ghi = robust_limits(block)
        gim = ax.imshow(block, aspect="auto", cmap="RdYlBu_r",
                        vmin=glo, vmax=ghi,
                        extent=[grid[0], grid[-1], len(labels), 0],
                        interpolation="nearest")
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_yticklabels(labels, fontsize=7)
        ax.tick_params(axis="y", length=0)
        ax.set_ylabel(gname, fontsize=9, rotation=0, ha="right",
                      va="center", labelpad=42)
        gcb = fig.colorbar(gim, ax=ax, fraction=0.012, pad=0.005)
        gcb.ax.tick_params(labelsize=6)
        for sp in ax.spines.values():
            sp.set_visible(False)

    for ax, (label, values) in zip(axes[nxt:nxt + n_track], tracks.items()):
        tg, tv = smooth(values, starts, step, window)
        ok = np.isfinite(tv)
        ax.fill_between(tg[ok], 0, tv[ok], color="#4B5563", lw=0)
        ax.axhline(0, color="#9AA0A6", lw=0.6)
        ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Dense, unlabelled: at a megabase-scale locus the names collide into an
    # unreadable band, and what the track is for is where the genes ARE.
    axg = axes[nxt + n_track]
    gr = _gene_rows(genes, chrom, grid[0], grid[-1], max_rows=3) \
        if genes is not None else pd.DataFrame()
    for _, r in gr.iterrows():
        axg.add_patch(plt.Rectangle((r["start"], -r["row"] * 0.32 - 0.28),
                                    max(r["end"] - r["start"], 1), 0.24,
                                    facecolor="#2C6FAF", edgecolor="none",
                                    alpha=0.85))
    axg.set_ylim(-1.05, 0.05)
    axg.set_yticks([])
    axg.set_ylabel(f"genes (n={len(gr)})", fontsize=8, rotation=0,
                   ha="right", va="center")
    for sp in ("top", "right", "left"):
        axg.spines[sp].set_visible(False)

    # Ideogram: the whole chromosome with the window marked, so the locus has
    # a position rather than only coordinates.
    if cytoband is not None:
        axi = axes[-1]
        cb_ = cytoband[cytoband.chrom.astype(str) == str(chrom)]
        clen = float(cb_.end.max()) if len(cb_) else float(grid[-1])
        stain = {"gneg": "#FFFFFF", "gpos25": "#C8C8C8", "gpos50": "#969696",
                 "gpos75": "#646464", "gpos100": "#1E1E1E",
                 "acen": "#B03030", "gvar": "#8FA5C8", "stalk": "#5F8FBF"}
        for _, r in cb_.iterrows():
            axi.add_patch(plt.Rectangle(
                (r["start"], 0.16), max(r["end"] - r["start"], 1), 0.34,
                facecolor=stain.get(str(r.get("stain", "gneg")), "#C8C8C8"),
                edgecolor="#4B5563", linewidth=0.25))
        axi.add_patch(plt.Rectangle((grid[0], 0.06),
                                    max(grid[-1] - grid[0], clen * 0.004),
                                    0.54, fill=False, edgecolor="#D1495B",
                                    linewidth=1.4))
        axi.set_xlim(0, clen)
        axi.set_ylim(0, 0.7)
        axi.set_yticks([])
        axi.set_ylabel(chrom, fontsize=8, rotation=0, ha="right", va="center")
        axi.tick_params(labelsize=7)
        axi.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}"))
        for sp in axi.spines.values():
            sp.set_visible(False)

    # Ticks on the bottom locus panel only; repeated on every panel they read
    # as separate axes rather than one coordinate system.
    for ax in axes[:nxt + n_track]:
        ax.tick_params(axis="x", labelbottom=False, length=0)
    if highlight is not None:
        for ax in axes[:nxt + n_track + 1]:
            ax.axvspan(highlight[0], highlight[1], color="#F0C808",
                       alpha=0.13, lw=0, zorder=0)
    axg.set_xlim(grid[0], grid[-1])
    axg.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}"))
    axg.set_xlabel(f"{chrom} (Mb)", fontsize=9, labelpad=1)
    axg.tick_params(axis="x", labelsize=8, pad=1)
    # The caveat goes under the whole figure, not on an axis it would crowd.
    fig.text(0.5, 0.005,
             f"{window // 1000} kb window every {step // 1000} kb: "
             f"interpolation between {bin_bp // 1000} kb bins, not "
             f"{step // 1000} kb resolution",
             ha="center", fontsize=7.5, color="#4B5563")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _named(specs, flag):
    out = {}
    for spec in specs or []:
        if "=" not in spec:
            sys.exit(f"{flag} must be LABEL=PATH, got {spec!r}")
        k, v = spec.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", required=True, metavar="NAME=PATH",
                    help="cells x bins concordance parquet (the RAW one: the "
                         "figure shows D, and per-bin normalisation destroys "
                         "its scale).")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-type-column", default="cell_type")
    ap.add_argument("--color-column", default=None,
                    help="Column giving each cell type's colour, as in "
                         "celltype_marker_bins.py, so the blocks match your "
                         "other figures.")
    ap.add_argument("--low", required=True,
                    help="Comma-separated groups expected LOW (e.g. "
                         "tumor1,tumor2).  Every one must sit below every "
                         "--high group for a bin to score.")
    ap.add_argument("--high", required=True,
                    help="Comma-separated groups expected HIGH.")
    ap.add_argument("--scores", default=None, metavar="BED",
                    help="bin_scores_<ctx>.all.bed, to require significance "
                         "as well as separation.  Without it the region is "
                         "chosen on the group means alone.")
    ap.add_argument("--fdr-column", default=None,
                    help="Column in --scores to threshold (default: the "
                         "first contrast_*_fdr found).")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--signal-track", action="append", default=None,
                    metavar="LABEL=PATH",
                    help="bigWig or bedGraph drawn under the heatmap "
                         "(RT early, RT late, LaminB1, ...).  Repeatable.")
    ap.add_argument("--heatmap-track", action="append", default=None,
                    metavar="GROUP:LABEL=PATH",
                    help="A signal drawn as a HEATMAP ROW inside a named "
                         "group panel, e.g. 'RT:HFFc6=rt/HFFc6_RT_250kb"
                         ".bedGraph'.  Repeatable; every track sharing a "
                         "GROUP becomes one panel, so the transition can be "
                         "read across cell types instead of one at a time.")
    ap.add_argument("--gene-annotation", default=None, metavar="GTF|BED")
    ap.add_argument("--cytoband", default=None, metavar="FILE",
                    help="UCSC cytoBand.txt(.gz) -- draws the whole "
                         "chromosome with the window marked, so the locus has "
                         "a position and not only coordinates.")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None,
                    help="Colour limits for the cell heatmap.  Default is the "
                         "1st-99th percentile of the values actually present, "
                         "symmetric about zero only when they straddle it -- "
                         "a fixed -|max|..+|max| scale wastes half the range "
                         "when D sits in a narrow positive band and paints "
                         "every cell alike.")
    ap.add_argument("--cmap", default="RdYlBu_r")
    ap.add_argument("--region", default=None, metavar="chr:start-end",
                    help="Draw this locus instead of searching for one.")
    ap.add_argument("--flank-bins", type=int, default=6,
                    help="Bins of context each side of the chosen run, so the "
                         "edges of the domain are visible rather than cropped "
                         "to the claim (6).")
    ap.add_argument("--min-run-bins", type=int, default=2)
    ap.add_argument("--min-gap", type=float, default=0.0, metavar="D",
                    help="Require this much separation, in D units, in EVERY "
                         "bin of the run.  The ordering alone is a weak "
                         "condition: a bin where the arms differ by 0.0001 "
                         "satisfies 'every low below every high' and is "
                         "indistinguishable from noise, so asking for the top "
                         "N without this returns real loci first and "
                         "arbitrary ones after.  A tenth of the spread of the "
                         "per-bin separation is a reasonable place to start; "
                         "the run prints that number.")
    ap.add_argument("--step", type=int, default=10_000)
    ap.add_argument("--window", type=int, default=250_000)
    ap.add_argument("--top", type=int, default=1,
                    help="Draw this many runs, best first.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    mbe = _enrich()
    name, mpath = (args.matrix.split("=", 1) if "=" in args.matrix
                   else ("gch", args.matrix))
    M = pd.read_parquet(mpath)
    samples = pd.read_csv(args.samples, sep="\t")
    if args.cell_type_column not in samples.columns:
        sys.exit(f"--cell-type-column {args.cell_type_column!r} not in "
                 f"--samples ({list(samples.columns)})")
    ct = samples.set_index("cell_name")[args.cell_type_column].astype(str)
    ct = ct.reindex(M.index).dropna()
    M = M.loc[ct.index]

    low = [g.strip() for g in args.low.split(",") if g.strip()]
    high = [g.strip() for g in args.high.split(",") if g.strip()]
    missing = [g for g in low + high if g not in set(ct)]
    if missing:
        sys.exit(f"not in --samples {args.cell_type_column!r}: {missing}\n"
                 f"  have: {sorted(set(ct))}")

    bins = pd.DataFrame({
        "chrom": [str(c).rsplit(":", 1)[0] for c in M.columns],
        "start": [int(str(c).rsplit(":", 1)[1]) for c in M.columns]})
    step_bp = int(np.median(np.diff(np.sort(bins.start.unique())))) or 250_000
    bins["end"] = bins.start + step_bp
    bins["bin"] = list(M.columns)

    means = group_means(M, ct, low + high)
    score = separation_score(means, low, high).reindex(M.columns).to_numpy()

    require = None
    if args.scores:
        sc = pd.read_csv(args.scores, sep="\t")
        cands = [c for c in sc.columns
                 if c.startswith("contrast_") and c.endswith("_fdr")]
        col = args.fdr_column
        if col is None and len(cands) > 1:
            # Picking the first silently answers a different question from the
            # one the figure asks, and the count in the log then disagrees
            # with whichever volcano the reader has open.
            sys.exit(
                f"--scores {args.scores} holds {len(cands)} contrasts:\n"
                + "\n".join(f"    {c}" for c in cands)
                + "\n  pass --fdr-column to say which one, or drop --scores "
                  "and let --low/--high choose the region on separation "
                  "alone -- which is the stricter test here, since it wants "
                  "every high group above every low one rather than one "
                  "pooled reference.")
        col = col or (cands[0] if cands else None)
        if col is None:
            sys.exit(f"--scores {args.scores}: no contrast_*_fdr column "
                     f"({list(sc.columns)}); pass --fdr-column")
        if col not in sc.columns:
            sys.exit(f"--fdr-column {col!r} not in {args.scores} "
                     f"(have: {cands or list(sc.columns)})")
        key = sc["chrom"].astype(str) + ":" + sc["start"].astype(str)
        sig = pd.Series(sc[col].to_numpy(), index=key.to_numpy())
        q = sig.reindex(M.columns).to_numpy(float)
        require = np.isfinite(q) & (q < args.fdr)
        print(f"  {int(np.nansum(require)):,} bins at {col} < {args.fdr}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    heat_spec = {}
    for spec in (args.heatmap_track or []):
        if "=" not in spec or ":" not in spec.split("=", 1)[0]:
            sys.exit(f"--heatmap-track must be GROUP:LABEL=PATH, got {spec!r}")
        key, path = spec.split("=", 1)
        grp, lab = key.split(":", 1)
        heat_spec.setdefault(grp.strip(), {})[lab.strip()] = path.strip()

    tracks_spec = _named(args.signal_track, "--signal-track")
    # Checked before the region search, not when the first track is drawn:
    # the search is the slow part, and failing after it wastes the work and
    # makes a missing file look like a plotting bug.
    absent = {k: v for k, v in tracks_spec.items()
              if "://" not in v and not Path(v).exists()}
    absent.update({f"{g}:{k}": v for g, mem in heat_spec.items()
                   for k, v in mem.items()
                   if "://" not in v and not Path(v).exists()})
    if absent:
        lines = [f"    {k} = {v}" for k, v in absent.items()]
        hint = ""
        if any("_early_" in v or "_late_" in v for v in absent.values()):
            hint = ("\n  fetch_4dn_repliseq.py writes the per-fraction "
                    "bedGraphs only with --keep-fractions; without it the "
                    "directory holds <cell>_RT_<grid>kb.bedGraph, the "
                    "log2(early/late) ratio.  Either re-run the fetcher with "
                    "--keep-fractions, or pass the single RT track instead of "
                    "the two fractions.")
        sys.exit("--signal-track: file(s) not found:\n"
                 + "\n".join(lines) + hint)
    genes = (mbe._read_genes(args.gene_annotation)
             if args.gene_annotation else None)
    cyto = None
    if args.cytoband:
        cyto = pd.read_csv(args.cytoband, sep="\t", header=None,
                           compression="infer",
                           names=["chrom", "start", "end", "band", "stain"])
    ct_colors = {}
    if args.color_column and args.color_column in samples.columns:
        for g, c in zip(samples[args.cell_type_column].astype(str),
                        samples[args.color_column].astype(str)):
            if c and c.lower() not in ("nan", "none", ""):
                ct_colors.setdefault(g, c)

    regions = []
    if args.region:
        chrom, rng = args.region.split(":")
        a, b = rng.replace(",", "").split("-")
        regions.append({"chrom": chrom, "start": int(a), "end": int(b),
                        "n": 0, "total": np.nan})
    else:
        pool_score, pool_req = score.copy(), (require.copy()
                                              if require is not None else None)
        for _ in range(args.top):
            run = best_run(bins, pool_score, args.min_run_bins, pool_req,
                           args.min_gap)
            if run is None:
                break
            regions.append(run)
            inside = ((bins.chrom == run["chrom"]) & (bins.start >= run["start"])
                      & (bins.start < run["end"])).to_numpy()
            pool_score[inside] = np.nan
        if not regions:
            sys.exit(
                "no run of adjacent bins has every --low group below every "
                "--high group"
                + (f" at FDR<{args.fdr}" if require is not None else "")
                + ".  Loosen --fdr, lower --min-run-bins, or check that the "
                  "group names are the ones you meant.")

    # What "a real gap" looks like on this data, so --min-gap is a number the
    # user can choose rather than guess.
    pos = score[np.isfinite(score) & (score > 0)]
    if pos.size:
        print(f"  separation where positive: median {np.median(pos):.3g}, "
              f"90th pct {np.percentile(pos, 90):.3g} in D "
              f"(--min-gap is currently {args.min_gap:g})")

    order = low + high + [g for g in sorted(set(ct)) if g not in low + high]
    for i, run in enumerate(regions, 1):
        pad = args.flank_bins * step_bp
        lo, hi = max(run["start"] - pad, 0), run["end"] + pad
        # The tracks must be sampled on the bins the MATRIX has, not on every
        # bin the window spans.  A matrix that lost bins to a coverage filter
        # is the normal case, and sampling the wider grid gives the tracks
        # more points than the heatmap has columns -- which used to crash on
        # the first locus with a gap, after the earlier ones had already been
        # written and looked fine.
        have = set(M.columns)
        starts_here = [x for x in np.arange(lo, hi, step_bp)
                       if f"{run['chrom']}:{x}" in have]
        if not starts_here:
            print(f"  {run['chrom']}:{run['start']:,}: no bin of the matrix "
                  f"falls in the padded window -- skipped", file=sys.stderr)
            continue
        win = pd.DataFrame({"chrom": run["chrom"], "start": starts_here})
        win["end"] = win.start + step_bp
        tracks = {}
        for label, path in tracks_spec.items():
            tracks[label] = mbe._signal_bin_means(win, path)
        heat = {g: {lab: mbe._signal_bin_means(win, pth)
                    for lab, pth in mem.items()}
                for g, mem in heat_spec.items()}
        cols = [f"{run['chrom']}:{x}" for x in starts_here]
        sub = M[cols]
        dest = out / f"locus_{name}_{i}_{run['chrom']}_{run['start']}.pdf"
        gap = run["total"] / max(run["n"], 1)
        locus_figure(
            sub, ct, order, (run["chrom"], lo, hi), tracks, genes, dest,
            f"{name}  {run['chrom']}:{run['start']:,}-{run['end']:,}  "
            f"({run['n']} bins, {len(M):,} cells, mean gap {gap:.3g} in D "
            f"between {'+'.join(low)} and {'+'.join(high)})",
            ct_colors=ct_colors, step=args.step, window=args.window,
            bin_bp=step_bp, highlight=(run["start"], run["end"]),
            heat_groups=heat, cytoband=cyto, vmin=args.vmin, vmax=args.vmax,
            cmap=args.cmap)
        flag = ""
        if pos.size and gap < 0.1 * np.median(pos):
            flag = "   <- gap is a tenth of the typical one; likely noise"
        print(f"  {run['chrom']}:{run['start']:,}-{run['end']:,}  "
              f"{run['n']} bins, mean gap {gap:.3g} -> {dest.name}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
