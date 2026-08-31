#!/usr/bin/env python3
"""Read-PAIR level correlation vs genomic distance, with a re-pairing null.

The companion script, scripts/site_distance_correlation.py, correlates
individual cytosines: one site on end 1 against one site on end 2, binned by
the distance between THOSE TWO SITES, so a pair carrying five GpC per end
contributes twenty-five observations spread over several bins.  That is the
right unit for asking how far methylation state travels along the genome.

This one asks a different question, and the unit is the read pair.  Each
high-quality cis pair gives ONE observation: the mean call on end 1 against
the mean call on end 2, binned by the distance between the two END CENTRES.
That is the unit a Hi-C contact actually is, and the unit the concordance
pipeline works in.

Three panels, each a Pearson r over the pairs in a distance bin:

  GCH-GCH   x = mean GpC call on end 1, y = the same on end 2
  HCG-HCG   x = mean CpG call on end 1, y = the same on end 2
  GCH-GCH vs HCG-HCG
            restricted to pairs carrying BOTH contexts on BOTH ends,
            x = (GpC end 1) - (GpC end 2), y = (CpG end 1) - (CpG end 2).
            Correlating the two DIFFERENCES asks whether the accessibility
            gradient across a contact tracks the methylation gradient.

THE NULL IS A RE-PAIRING, WITHIN CELL AND WITHIN DISTANCE BIN.  The y column
is permuted among the pairs of one (cell, bin), so end 1 of one pair meets
end 2 of another that was the same distance away in the same cell.  Both
marginal distributions survive exactly; only the link is destroyed.

  WITHIN CELL, because cells differ in global accessibility and methylation.
  Pooling across cells makes end 1 and end 2 correlate simply by both being
  high in an open cell, and a null that shuffled across cells would call
  that correlation real.  Permuting inside a cell leaves it in the null,
  where it belongs.

  WITHIN DISTANCE BIN, because the bin is the axis.  A null drawn from other
  bins would answer "is this contact more correlated than a contact at some
  other distance", which is a different and much easier question.

Read filters match bam_parser exactly: uniquely mapped, not a PCR duplicate,
no supplementary alignment, mapQ above the threshold on BOTH ends, both ends
on the same chromosome.  They come from the same site_calls module the
pipeline uses, so this script and Step 1 see the same reads.

WHAT THE VALUES ARE.  The mean of the 0/1 calls on an end -- the fraction of
its cytosines called accessible (GpC) or methylated (CpG).  Not the mixture
model's posterior: that is Step 3/4 output and lives in 04_posteriors, while
this script reads BAMs.  An end with one site therefore contributes a 0 or a
1, and --min-sites-per-end trades pairs for a less granular value.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_MEASURES = (("gg", "GCH–GCH  r(end1, end2)"),
             ("hh", "HCG–HCG  r(end1, end2)"),
             ("gh", "GCH–GCH vs HCG–HCG  r(ΔGpC, ΔCpG)"))


def pair_values(calls1, calls2, min_sites=1):
    """Per-end mean call for each context, or NaN where too few sites."""
    out = {}
    for ctx in ("GCH", "HCG"):
        a, b = calls1.get(ctx), calls2.get(ctx)
        if (a is None or b is None or len(a[1]) < min_sites
                or len(b[1]) < min_sites):
            out[ctx] = (np.nan, np.nan)
        else:
            out[ctx] = (float(np.mean(a[1])), float(np.mean(b[1])))
    return out


def measures_of(vals):
    """The three (x, y) observations a pair contributes, NaN where ineligible.

    'gg' and 'hh' need their own context on both ends.  'gh' needs BOTH
    contexts on BOTH ends -- that is the question, and a pair missing either
    cannot answer it -- and is a difference rather than a level, so the two
    ends must be ordered consistently.  They are: end 1 is the leftmost, set
    by the caller from the midpoints.
    """
    g1, g2 = vals["GCH"]
    h1, h2 = vals["HCG"]
    out = {"gg": (g1, g2), "hh": (h1, h2)}
    if np.isfinite([g1, g2, h1, h2]).all():
        out["gh"] = (g1 - g2, h1 - h2)
    else:
        out["gh"] = (np.nan, np.nan)
    return out


def collect_cell(bam_path, ctx_positions, edges, trim_ends, mapq_min,
                 max_pairs, min_sites, seed):
    """One cell → {measure: [(bin, x, y), ...]}.

    The per-pair values are KEPT rather than streamed into running sums,
    because the null permutes them within (cell, bin) and a permutation
    cannot be done on a sum.  One float pair per read pair per measure, so
    the cost is linear and small beside the BAM pass.
    """
    from nome_hic_concordance.site_calls import iter_cell_read_pairs

    rng = np.random.default_rng(seed)
    nb = len(edges) - 1
    got = {m: ([], [], []) for m, _ in _MEASURES}
    n_pairs = 0
    for _chrom, mid1, mid2, c1, c2 in iter_cell_read_pairs(
            bam_path, ctx_positions, trim_ends=trim_ends, mapq_min=mapq_min,
            max_pairs_per_cell=max_pairs, rng=rng):
        d = abs(int(mid2) - int(mid1))
        if d < edges[0] or d > edges[-1]:
            continue
        b = int(np.clip(np.digitize([d], edges)[0] - 1, 0, nb - 1))
        # Leftmost end first, so the sign of a difference means the same
        # thing in every pair.
        if mid1 <= mid2:
            vals = pair_values(c1, c2, min_sites)
        else:
            vals = pair_values(c2, c1, min_sites)
        n_pairs += 1
        for m, (x, y) in measures_of(vals).items():
            if np.isfinite(x) and np.isfinite(y):
                got[m][0].append(b)
                got[m][1].append(x)
                got[m][2].append(y)
    return ({m: (np.asarray(v[0], np.int64), np.asarray(v[1], float),
                 np.asarray(v[2], float)) for m, v in got.items()}, n_pairs)


def shuffle_within(bins, y, nb, rng):
    """Permute y among the entries sharing a bin.  Cell scope is the caller's."""
    out = y.copy()
    for b in range(nb):
        idx = np.flatnonzero(bins == b)
        if idx.size > 1:
            out[idx] = y[rng.permutation(idx)]
    return out


def _running(bins, x, y, nb):
    """(6, nb) sums for a Pearson."""
    acc = np.zeros((6, nb))
    if bins.size:
        acc[0] = np.bincount(bins, minlength=nb)
        acc[1] = np.bincount(bins, weights=x, minlength=nb)
        acc[2] = np.bincount(bins, weights=y, minlength=nb)
        acc[3] = np.bincount(bins, weights=x * x, minlength=nb)
        acc[4] = np.bincount(bins, weights=y * y, minlength=nb)
        acc[5] = np.bincount(bins, weights=x * y, minlength=nb)
    return acc


def corr_of(acc, min_pairs):
    """(6, nb) sums → (r, n); NaN where n < min_pairs or either side is flat."""
    n, sx, sy, sxx, syy, sxy = acc
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = np.where(n > 0, n, 1.0)
        cov = sxy - sx * sy / nz
        vx = sxx - sx * sx / nz
        vy = syy - sy * sy / nz
        den = np.sqrt(vx * vy)
        r = np.where(den > 0, cov / np.where(den > 0, den, 1.0), np.nan)
    return np.where(n >= min_pairs, r, np.nan), n


def accumulate_cells(per_cell, nb, n_perm, seed):
    """Observed and null running sums, pooled ACROSS cells.

    The observed correlation pools every eligible pair, which is what "all
    the pairs in this distance range" means.  The null pools too -- but each
    cell's y column is permuted WITHIN that cell and within the bin first,
    so whatever correlation the pooling itself creates (cells differ in
    global accessibility, so both ends run high together in an open cell)
    survives into the null instead of being counted as signal.

    ``n_perm`` permutations are averaged.  One is an unbiased null but a
    noisy one, and the null curve is subtracted from every point of the
    observed.
    """
    rng = np.random.default_rng(seed)
    obs = {m: np.zeros((6, nb)) for m, _ in _MEASURES}
    null = {m: np.zeros((6, nb)) for m, _ in _MEASURES}
    n_cells_used = 0
    for cell in per_cell:
        used = False
        for m, _ in _MEASURES:
            b, x, y = cell[m]
            if not b.size:
                continue
            used = True
            obs[m] += _running(b, x, y, nb)
            for _ in range(max(1, int(n_perm))):
                null[m] += _running(b, x, shuffle_within(b, y, nb, rng), nb)
        n_cells_used += int(used)
    for m, _ in _MEASURES:                       # average the permutations
        null[m] /= max(1, int(n_perm))
    return obs, null, n_cells_used


def figure(path, centers, curves, title, min_pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"gg": "#1f77b4", "hh": "#d62728", "gh": "#7b3f9e"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharex=True)
    for col, (key, ttl) in enumerate(_MEASURES):
        r_o, n_o = curves[(key, "linked")]
        r_n, _ = curves[(key, "null")]
        a = axes[0, col]
        a.axhline(0, color="0.6", lw=0.8, zorder=0)
        a.plot(centers, r_o, "-o", ms=3, lw=1.6, color=colors[key],
               label="linked (true pairing)")
        a.plot(centers, r_n, "--", lw=1.4, color="0.35",
               label="null (re-paired within cell & bin)")
        a.set_xscale("log")
        a.set_title(ttl.split("  ")[0], fontsize=10)
        if col == 0:
            a.set_ylabel("Pearson r over read pairs", fontsize=9)
        a.legend(frameon=False, fontsize=7.5)
        b = axes[1, col]
        b.axhline(0, color="0.6", lw=0.8, zorder=0)
        b.plot(centers, r_o - r_n, "-o", ms=3, lw=1.6, color=colors[key])
        b.set_xscale("log")
        b.set_xlabel("distance between end centres (bp)", fontsize=9)
        if col == 0:
            b.set_ylabel("Δ = linked − null", fontsize=9)
        # n on the delta panel: the correlation is only as good as the pairs
        # behind it, and the pair count falls off fast at long range.
        tw = b.twinx()
        tw.plot(centers, n_o, ":", lw=1.0, color="0.6")
        tw.set_yscale("log")
        tw.set_ylabel("pairs" if col == 2 else "", fontsize=7)
        tw.tick_params(labelsize=6)
        if col != 2:
            tw.set_yticklabels([])
        b.axhline(0, color="0.6", lw=0.8, zorder=0)
    for ax in axes.ravel():
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True,
                    help="TSV with cell_name and bam_path.")
    ap.add_argument("--context-bed", default=None,
                    help="'GCH:/path/GCH.bed.gz,HCG:/path/HCG.bed.gz' -- the "
                         "same CONTEXT:PATH form as nome_hic_concordance and "
                         "scripts/site_distance_correlation.py.  Either this "
                         "or --reference is required, and BOTH contexts are "
                         "needed: every panel uses one of them and the third "
                         "uses both.")
    ap.add_argument("--reference", default=None,
                    help=".2bit genome, if --context-bed is not given.")
    ap.add_argument("--context-bed-offset", type=int, default=0)
    ap.add_argument("--dbsnp-vcf", default=None)
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--min-dist", type=float, default=1e2)
    ap.add_argument("--max-dist", type=float, default=1e7)
    ap.add_argument("--bins-per-decade", type=int, default=1,
                    help="1 (default) gives the decade bins 100-1,000, "
                         "1,000-10,000 and so on.  Raise it for a finer "
                         "axis, at the cost of pairs per bin.")
    ap.add_argument("--min-sites-per-end", type=int, default=1,
                    help="An end with one site contributes a bare 0 or 1; "
                         "raising this trades pairs for a less granular "
                         "value.  Applied per CONTEXT, so the third panel's "
                         "eligible set is the intersection.")
    ap.add_argument("--trim-ends", type=int, default=5)
    ap.add_argument("--mapq-min", type=int, default=30)
    ap.add_argument("--max-pairs-per-cell", type=int, default=0)
    ap.add_argument("--min-pairs", type=int, default=50, metavar="N",
                    help="Bins with fewer eligible pairs report NaN rather "
                         "than a correlation of three points.")
    ap.add_argument("--n-permutations", type=int, default=5, metavar="N",
                    help="Re-pairings averaged into the null.  One is "
                         "unbiased but noisy, and the null is subtracted "
                         "from every point.")
    ap.add_argument("--exclude-chrom", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--plot-format", default="pdf",
                    choices=["pdf", "png", "svg"])
    ap.add_argument("--title", default="Read-pair correlation vs distance")
    args = ap.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_edges = int(round(np.log10(args.max_dist / args.min_dist)
                        * args.bins_per_decade)) + 1
    edges = np.logspace(np.log10(args.min_dist), np.log10(args.max_dist),
                        max(2, n_edges))
    centers = np.sqrt(edges[:-1] * edges[1:])
    nb = len(edges) - 1

    # Same loaders as scripts/site_distance_correlation.py and Step 1, so the
    # two scripts see the same sites and the same reads.
    from nome_hic_concordance.cli import _parse_context_bed, load_samples
    from nome_hic_concordance.reference import (extract_context_positions,
                                                load_context_from_bed)

    samples = load_samples(args.samples)
    if args.max_cells and args.max_cells > 0:
        samples = samples.iloc[: args.max_cells]
    if args.context_bed:
        bed = _parse_context_bed(args.context_bed)
        for need in ("GCH", "HCG"):
            if need not in bed:
                sys.exit(f"--context-bed must define {need}: all three "
                         f"panels need both contexts (got {list(bed)}).")
        ctx_positions = load_context_from_bed(
            bed, n_threads=args.threads,
            c_offset=args.context_bed_offset)
    elif args.reference:
        ctx_positions = extract_context_positions(
            args.reference, str(out / "context"), contexts=["GCH", "HCG"],
            dbsnp_vcf=args.dbsnp_vcf, n_threads=args.threads)
    else:
        sys.exit("provide --context-bed or --reference.")
    if args.exclude_chrom:
        drop = {c.strip() for c in args.exclude_chrom.split(",") if c.strip()}
        ctx_positions = {k: {c: v for c, v in d.items() if c not in drop}
                         for k, d in ctx_positions.items()}

    print(f"  {len(samples)} cells, {nb} distance bins "
          f"({args.min_dist:g}-{args.max_dist:g} bp, "
          f"{args.bins_per_decade}/decade)", file=sys.stderr)
    jobs = [(row["bam_path"], ctx_positions, edges, args.trim_ends,
             args.mapq_min, args.max_pairs_per_cell, args.min_sites_per_end,
             args.seed + i) for i, (_, row) in enumerate(samples.iterrows())]
    if args.threads > 1:
        from joblib import Parallel, delayed
        got = Parallel(n_jobs=args.threads)(
            delayed(collect_cell)(*j) for j in jobs)
    else:
        got = [collect_cell(*j) for j in jobs]
    per_cell = [g[0] for g in got]
    n_pairs_total = sum(g[1] for g in got)

    obs, null, n_used = accumulate_cells(per_cell, nb, args.n_permutations,
                                         args.seed)
    curves = {}
    for m, _ in _MEASURES:
        curves[(m, "linked")] = corr_of(obs[m], args.min_pairs)
        curves[(m, "null")] = corr_of(null[m], args.min_pairs)

    rows = {"bin_lo": edges[:-1], "bin_hi": edges[1:], "center": centers}
    for m, _ in _MEASURES:
        rows[f"{m}_r"] = curves[(m, "linked")][0]
        rows[f"{m}_n"] = curves[(m, "linked")][1]
        rows[f"{m}_r_null"] = curves[(m, "null")][0]
        rows[f"{m}_delta"] = curves[(m, "linked")][0] - curves[(m, "null")][0]
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "pair_distance_correlation.tsv", sep="\t", index=False)
    figure(out / f"pair_distance_correlation.{args.plot_format}",
           centers, curves, args.title, args.min_pairs)

    print(f"  {n_pairs_total:,} high-quality cis pairs from {n_used} cells",
          file=sys.stderr)
    for m, lab in _MEASURES:
        r, n = curves[(m, "linked")]
        rn, _ = curves[(m, "null")]
        ok = np.isfinite(r)
        if ok.any():
            j = int(np.flatnonzero(ok)[-1])
            print(f"  {lab.split('  ')[0]:22s} longest bin with >= "
                  f"{args.min_pairs} pairs: {edges[j]:.0f}-{edges[j+1]:.0f} bp, "
                  f"r = {r[j]:+.3f}, null {rn[j]:+.3f}, "
                  f"Δ = {r[j]-rn[j]:+.3f}  (n = {int(n[j]):,})",
                  file=sys.stderr)
    print(f"  -> {out}/pair_distance_correlation.tsv and "
          f".{args.plot_format}", file=sys.stderr)
    print(f"  The null re-pairs WITHIN cell and WITHIN bin, so a correlation "
          f"that comes from cells differing in global level is in the null "
          f"too; read Δ, not r.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
