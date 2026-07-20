#!/usr/bin/env python3
"""Epigenetic concordance vs SPATIAL proximity (contact frequency).

The sibling ``site_distance_correlation.py`` asks how concordance decays with
LINEAR separation.  This asks the other question: at comparable linear
separation, are loci that are CLOSE IN 3D more epigenetically concordant?

Spatial proximity is read off the contacts themselves.  Two loci that ligate
together often are close in 3D, so pooled contact frequency is the proxy.  The
axis is observed/expected, NOT the raw count:

    O/E = (pairs linking bins i and j) / (mean pairs per bin-pair at |i-j|)

Raw contact frequency is dominated by genomic distance -- contacts decay ~1/s --
so a raw-count axis would largely re-plot the linear-distance curve.  Dividing
by the expected count at the SAME separation removes that trend, leaving how
much more often a locus pair touches than its separation predicts.  That is the
quantity that means "spatially close beyond what linear distance explains", and
it is the same observed/expected normalisation ``compartments.py`` already uses.

Being a ratio also makes the axis robust to --max-pairs-per-cell: uniform
subsampling scales observed and expected alike and cancels.

Only pairs at least --min-contact-span apart (10kb) enter, and only bin
separations where that filter is saturated -- at 25kb resolution a 10kb cut
bites into separations 0 and 1, which would deflate their expected and inflate
their O/E, so they are dropped rather than plotted as a spurious rise.

Contact frequency is only meaningful POOLED: one cell sees each long-range
contact about once.  So this is a genuine two-pass analysis -- parse and keep
every long-range pair, then count contacts across the cells of the group, then
bin by O/E and correlate.  Bin edges come from the pooled 'all' distribution and
are reused for every group, so groups are directly comparable.

The headline statistic is the TREND: Spearman r between the O/E bin and the
concordance, with a cell-level bootstrap p (site pairs are not independent, so
the resampling unit is the cell) and BH correction over the three measures.

Usage
-----
    python scripts/contact_concordance.py \\
        --samples samples.tsv \\
        --context-bed GCH:/ref/GCH.bed.gz,HCG:/ref/HCG.bed.gz \\
        --output-dir output/contact_concordance \\
        --resolution 25000 --group-column group --n-threads 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _sdc():
    """The distance script's helpers — reuse the tested ones rather than fork."""
    path = Path(__file__).resolve().parent / "site_distance_correlation.py"
    spec = importlib.util.spec_from_file_location("_sdc_helpers", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_sdc_helpers"] = m
    spec.loader.exec_module(m)
    return m


try:
    import pysam as _pysam
    _ = _pysam
    from nome_hic_concordance import _site_dist_core as _CPP
except Exception:                                           # pragma: no cover
    _CPP = None

_MEASURES = (("gg", "GCH–GCH  r(site_i, site_j)"),
             ("hh", "HCG–HCG  r(site_i, site_j)"),
             ("gh", "HCG–GCH  r(CpG, GpC)  [cross-context]"))
_COLORS = {"gg": "#1f77b4", "hh": "#d62728", "gh": "#2ca02c"}
_QC = ("high-quality read pairs only: uniquely mapped, no PCR duplicate, "
       "mapQ>30 both ends, same chromosome")


# --------------------------------------------------------------------------
# Contact matrix → observed/expected
# --------------------------------------------------------------------------

def _bins_per_chrom(ctx_positions, chrom_names, resolution):
    """Bins per chromosome, from the extent of the context sites.

    The expected count needs a DENOMINATOR of possible bin-pairs, which means
    knowing how many bins each chromosome has.  Context-site extent is the
    right measure: it is the region the analysis could actually observe.
    """
    out = np.zeros(len(chrom_names), np.int64)
    for i, c in enumerate(chrom_names):
        hi = 0
        for ctx in ctx_positions.values():
            arr = ctx.get(c)
            if arr is not None and len(arr["pos"]):
                hi = max(hi, int(arr["pos"][-1]))
        out[i] = hi // resolution + 1
    return out


def _observed_expected(chrom, mid1, mid2, n_bins_chrom, resolution,
                       min_sep_bins):
    """O/E per contact, plus the per-separation expected curve.

    expected[d] = (all contacts at separation d) / (all POSSIBLE bin-pairs at
    separation d).  The denominator counts unobserved bin-pairs too -- without
    that, 'expected' would be the mean over only the pairs we happened to see,
    which is not a background at all.
    """
    b1 = mid1 // resolution
    b2 = mid2 // resolution
    lo = np.minimum(b1, b2).astype(np.int64)
    hi = np.maximum(b1, b2).astype(np.int64)
    sep = hi - lo

    keep = sep >= min_sep_bins
    if not keep.any():
        return None
    idx = np.flatnonzero(keep)
    c_k, lo_k, hi_k, sep_k = chrom[keep], lo[keep], hi[keep], sep[keep]

    # count contacts per (chrom, lo, hi); 21 bits holds 2M bins @25kb = 52 Gb
    key = (c_k.astype(np.int64) << 42) | (lo_k << 21) | hi_k
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    observed = cnt[inv].astype(np.float64)          # M[i][j] for each contact

    max_d = int(sep_k.max())
    obs_d = np.bincount(sep_k, minlength=max_d + 1).astype(np.float64)
    poss_d = np.zeros(max_d + 1, np.float64)
    for n in n_bins_chrom:
        d = np.arange(0, min(int(n), max_d + 1))
        poss_d[:len(d)] += np.maximum(int(n) - d, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        exp_d = np.where(poss_d > 0, obs_d / np.maximum(poss_d, 1.0), np.nan)

    e = exp_d[sep_k]
    oe = np.where(e > 0, observed / np.where(e > 0, e, 1.0), np.nan)
    return {"idx": idx, "oe": oe, "sep": sep_k, "observed": observed,
            "exp_d": exp_d, "n_contacts_kept": int(keep.sum()),
            "n_contacts_dropped": int((~keep).sum()),
            "n_bin_pairs": int(len(uniq))}


def _quantile_edges(x, n_bins):
    """Equal-count bins: every O/E bin then carries comparable weight.

    Fixed-width bins on a heavy-tailed O/E would put almost everything in one
    bin and leave the tail bins too thin to correlate.
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    q = np.linspace(0, 100, n_bins + 1)
    e = np.unique(np.percentile(x, q))
    return e if e.size >= 3 else None


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _spearman(x, y):
    """Spearman r over the finite entries; NaN if fewer than 4 usable bins."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    xr = pd.Series(x[ok]).rank().to_numpy()
    yr = pd.Series(y[ok]).rank().to_numpy()
    if xr.std() == 0 or yr.std() == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def _bootstrap_curve(per_cell, centers, min_pairs, n_boot, seed, corr_stack):
    """Cell-level bootstrap → (lo, hi) per bin and the trend statistic.

    The unit is the CELL: site pairs are massively non-independent (tens per
    read, ~1e6 reads per cell), so resampling them would manufacture certainty.
    """
    a = np.stack(per_cell).astype(np.float64)          # (k, 6, nb)
    k = a.shape[0]
    if k < 3 or n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    w = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(np.float64)
    r = corr_stack((w @ a.reshape(k, -1)).reshape(n_boot, 6, -1), min_pairs)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(r, [2.5, 97.5], axis=0)
    rho = np.array([_spearman(centers, r[b]) for b in range(n_boot)])
    ok = np.isfinite(rho)
    if ok.sum() < 20:
        return lo, hi, np.nan, np.nan
    # (1 + count)/(1 + B): a finite bootstrap cannot justify p = 0
    p = 2.0 * (1.0 + min((rho[ok] <= 0).sum(), (rho[ok] >= 0).sum())) \
        / (ok.sum() + 1.0)
    return lo, hi, float(np.nanmedian(rho)), float(min(p, 1.0))


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _plot(curves, boot, centers, edges, label, n_cells, n_contacts, mean_sep,
          resolution, out, min_pairs, stars):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))

    for col, (key, title) in enumerate(_MEASURES):
        ax = axes[0, col]
        r, n = curves[key]
        if boot and boot.get(key) is not None:
            lo, hi, rho, p = boot[key]
            ax.fill_between(centers, lo, hi, color=_COLORS[key], alpha=0.20,
                            lw=0, label="95% CI (cell bootstrap)")
        ax.plot(centers, r, "-o", ms=4, lw=1.7, color=_COLORS[key],
                label="concordance")
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        ax.set_xlabel("log$_{10}$ observed/expected contacts  →  spatially closer")
        ax.set_ylabel("Pearson r (phi) of 0/1 calls")
        sub = ""
        if boot and boot.get(key) is not None:
            rho, p = boot[key][2], boot[key][3]
            if np.isfinite(rho):
                sub = f"\ntrend ρ={rho:+.2f}{stars.get(key, '')}"
        ax.set_title(title.split("  ")[0] + sub, fontsize=10)
        ax.legend(fontsize=7.5, loc="best")

    # ---- diagnostics -----------------------------------------------------
    ax = axes[1, 0]
    ax.plot(centers, mean_sep / 1e3, "-o", ms=4, lw=1.5, color="0.25")
    ax.set_xlabel("log$_{10}$ observed/expected")
    ax.set_ylabel("mean genomic separation (kb)")
    ax.set_yscale("log")
    # A number, not just a shape: on a log axis a 1.2x range can look dramatic,
    # so state the fold-range outright.  Near 1x => the O/E axis is not just
    # genomic distance wearing a different label.
    fin = mean_sep[np.isfinite(mean_sep)]
    fold = (fin.max() / fin.min()) if fin.size and fin.min() > 0 else np.nan
    ax.set_title(f"decoupling check: separation vs O/E "
                 f"({fold:.2f}× range)" if np.isfinite(fold)
                 else "decoupling check", fontsize=9.5)
    ax.text(0.03, 0.05,
            "near 1× ⇒ the axis is spatial, not linear\n"
            "a strong trend here ⇒ still distance-confounded",
            transform=ax.transAxes, fontsize=7, color="0.4", va="bottom")

    ax = axes[1, 1]
    for key, _ in _MEASURES:
        ax.plot(centers, curves[key][1], "-o", ms=3, lw=1.3,
                color=_COLORS[key], label=key.upper())
    ax.axhline(min_pairs, color="0.6", ls=":", lw=1.0,
               label=f"--min-pairs {min_pairs:,}")
    ax.set_yscale("log")
    ax.set_xlabel("log$_{10}$ observed/expected")
    ax.set_ylabel("site pairs per bin")
    ax.set_title("weight per bin", fontsize=9.5)
    ax.legend(fontsize=7, loc="best")

    ax = axes[1, 2]
    for key, title in _MEASURES:
        ax.plot(centers, curves[key][0], "-o", ms=3.5, lw=1.5,
                color=_COLORS[key], label=title.split("  ")[0])
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    ax.set_xlabel("log$_{10}$ observed/expected")
    ax.set_ylabel("Pearson r (phi)")
    ax.set_title("all measures", fontsize=9.5)
    ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"Concordance vs SPATIAL proximity — {label}  |  {n_cells} "
                 f"cells, {n_contacts:,} long-range contacts @ "
                 f"{resolution / 1e3:.0f}kb", fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    fig.text(0.5, 0.024, _QC, ha="center", fontsize=7.5, color="0.35")
    fig.text(0.5, 0.008,
             "x is observed/expected contacts at matched genomic separation, so "
             "the linear-distance trend is divided out; trend p is a cell-level "
             "bootstrap, BH-corrected over the three measures",
             ha="center", fontsize=7, color="0.45")
    fig.savefig(out)
    plt.close(fig)


def _write_tsv(curves, boot, centers, edges, mean_sep, path):
    rows = {"oe_lo": edges[:-1], "oe_hi": edges[1:],
            "log10_oe_center": centers,
            "mean_separation_bp": mean_sep}
    for key, _ in _MEASURES:
        r, n = curves[key]
        rows[f"{key}_r"] = r
        rows[f"{key}_n"] = n.astype(np.int64)
        if boot and boot.get(key) is not None:
            lo, hi, _rho, _p = boot[key]
            rows[f"{key}_ci_lo"] = lo
            rows[f"{key}_ci_hi"] = hi
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, na_rep="nan")


# --------------------------------------------------------------------------

def _add_exclude(ap):
    from nome_hic_concordance.utils import add_exclude_chrom_arg
    add_exclude_chrom_arg(ap)


def main():
    sdc = _sdc()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--context-bed", default=None)
    ap.add_argument("--context-bed-offset", type=int, default=0)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--dbsnp-vcf", default=None)
    ap.add_argument("--blacklist-bed", default=None)
    _add_exclude(ap)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--group-column", default=None)
    ap.add_argument("--resolution", type=int, default=25_000,
                    help="Contact-matrix bin size in bp. The resolution at "
                         "which two loci count as the same contact; also sets "
                         "the smallest separation the O/E axis can speak "
                         "about. Default 25000.")
    ap.add_argument("--min-contact-span", type=int, default=10_000,
                    help="Only read pairs at least this far apart are counted "
                         "as contacts. Default 10000.")
    ap.add_argument("--min-separation-bins", type=int, default=0,
                    help="Drop bin separations below this. 0 (default) picks "
                         "ceil(min_contact_span/resolution)+1, the smallest "
                         "separation at which the span filter no longer bites "
                         "-- below it the expected count is deflated and O/E "
                         "is spuriously inflated.")
    ap.add_argument("--oe-bins", type=int, default=12,
                    help="Number of equal-count O/E bins. Edges are taken from "
                         "the pooled 'all' distribution and reused for every "
                         "group, so groups stay comparable.")
    ap.add_argument("--min-pairs", type=int, default=2000,
                    help="Mask an O/E bin with fewer site pairs than this.")
    ap.add_argument("--max-pairs-per-cell", type=int, default=0,
                    help="Cap long-range pairs kept per cell (0 = all). O/E is "
                         "a ratio, so uniform subsampling cancels and only the "
                         "count noise grows. This is the memory knob: ~140 B "
                         "per kept pair.")
    ap.add_argument("--max-site-pairs-per-contact", type=int, default=64)
    ap.add_argument("--mapq-min", type=int, default=30)
    ap.add_argument("--baseq-min", type=int, default=0)
    ap.add_argument("--trim-ends", type=int, default=5)
    ap.add_argument("--strand-min-evidence", type=int, default=1)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--fdr", choices=["bh", "by", "none"], default="bh")
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--plot-format", choices=["png", "pdf", "svg"], default="pdf")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if _CPP is None:
        sys.exit("the htslib extension is required; build with `pip install -e .`")

    from nome_hic_concordance.cli import _parse_context_bed, load_samples
    from nome_hic_concordance.reference import (extract_context_positions,
                                                load_context_from_bed)
    from nome_hic_concordance.utils import ensure_dir

    samples = load_samples(args.samples)
    if args.max_cells and args.max_cells > 0:
        samples = samples.iloc[: args.max_cells]
    if not args.context_bed and not args.reference:
        sys.exit("provide --context-bed or --reference.")
    if args.context_bed:
        bed = _parse_context_bed(args.context_bed)
        for need in ("GCH", "HCG"):
            if need not in bed:
                sys.exit(f"--context-bed must define {need} (got {list(bed)}).")
        ctx_positions = load_context_from_bed(
            bed, n_threads=args.n_threads, c_offset=args.context_bed_offset)
    else:
        ctx_positions = extract_context_positions(args.reference,
                                                  ["GCH", "HCG"])
    if args.dbsnp_vcf:
        ctx_positions = sdc._drop_snp_sites(ctx_positions, args.dbsnp_vcf)
    if args.exclude_chrom:
        ctx_positions, dropped = sdc._drop_chroms(ctx_positions,
                                                  args.exclude_chrom)
        for k, v in sorted(dropped.items()):
            print(f"  excluded {k}: {v:,} context sites")
        if not dropped:
            print(f"  NOTE: --exclude-chrom {args.exclude_chrom} matched no "
                  f"contig.", file=sys.stderr)

    blacklist = None
    if args.blacklist_bed:
        from nome_hic_concordance.blacklist import load_blacklist_bed
        blacklist = load_blacklist_bed(args.blacklist_bed)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    min_sep = args.min_separation_bins or (
        int(np.ceil(args.min_contact_span / args.resolution)) + 1)
    print(f"# {_QC}")
    print(f"[contact-conc] {len(samples)} cells | resolution "
          f"{args.resolution:,} bp | contacts >= {args.min_contact_span:,} bp")
    print(f"  O/E axis: observed / expected-at-matched-separation "
          f"(raw counts track genomic distance, so they are not used)")
    print(f"  dropping bin separations < {min_sep} "
          f"(< {min_sep * args.resolution:,} bp): the span filter deflates "
          f"their expected")

    ctx_g, ctx_h, bl = sdc.prepare_ctx(ctx_positions, blacklist)
    jobs = [(row["bam_path"], i) for i, (_, row) in enumerate(samples.iterrows())]
    cell_names = samples["cell_name"].tolist()
    t0 = time.time()

    def _one(bam_path, i):
        return _CPP.parse_cell_contacts(
            str(bam_path), ctx_g, ctx_h, int(args.trim_ends),
            int(args.mapq_min), int(args.min_contact_span), False,
            int(args.baseq_min), bl, int(args.strand_min_evidence),
            int(args.max_pairs_per_cell), int(args.seed + i))

    res = [None] * len(jobs)
    if args.n_threads > 1 and len(jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.n_threads) as ex:
            futs = {ex.submit(_one, *j): j[1] for j in jobs}
            for done, fut in enumerate(as_completed(futs), 1):
                i = futs[fut]
                res[i] = fut.result()
                el = time.time() - t0
                print(f"  [{done}/{len(jobs)}] {cell_names[i]}: "
                      f"{int(res[i]['n_long_range']):,} long-range | "
                      f"{el / 60:.1f} min, ETA "
                      f"{el / done * (len(jobs) - done) / 60:.1f} min", flush=True)
    else:
        for n, (bam, i) in enumerate(jobs, 1):
            res[i] = _one(bam, i)
            print(f"  [{n}/{len(jobs)}] {cell_names[i]}: "
                  f"{int(res[i]['n_long_range']):,} long-range", flush=True)

    chrom_names = list(res[0]["chrom_names"]) if res else []
    n_bins_chrom = _bins_per_chrom(ctx_positions, chrom_names, args.resolution)
    total = sum(len(np.asarray(r["chrom"])) for r in res)
    print(f"  parsed {len(res)} cells in {(time.time() - t0) / 60:.1f} min "
          f"({total:,} long-range pairs kept)")
    if total == 0:
        sys.exit("no long-range pairs survived the filters.")

    groups = {"all": list(range(len(res)))}
    if args.group_column:
        if args.group_column not in samples.columns:
            sys.exit(f"--group-column {args.group_column!r} not in --samples "
                     f"(columns: {samples.columns.tolist()}).")
        for gi, g in enumerate(samples[args.group_column].to_numpy()):
            groups.setdefault(str(g), []).append(gi)

    safe = sdc.safe_names(groups)
    summary, edges = {}, None
    for label, idxs in groups.items():
        parts = [res[i] for i in idxs if len(np.asarray(res[i]["chrom"]))]
        if not parts:
            print(f"\n=== {label}: no long-range pairs — skipped ===")
            continue
        chrom = np.concatenate([np.asarray(p["chrom"], np.int64) for p in parts])
        mid1 = np.concatenate([np.asarray(p["mid1"], np.int64) for p in parts])
        mid2 = np.concatenate([np.asarray(p["mid2"], np.int64) for p in parts])
        cell = np.concatenate([np.full(len(np.asarray(p["chrom"])), j, np.int32)
                               for j, p in enumerate(parts)])

        oe = _observed_expected(chrom, mid1, mid2, n_bins_chrom,
                                args.resolution, min_sep)
        if oe is None:
            print(f"\n=== {label}: no contacts past the separation floor ===")
            continue
        log_oe = np.log10(np.where(oe["oe"] > 0, oe["oe"], np.nan))
        if edges is None:                       # 'all' sets the shared binning
            edges = _quantile_edges(log_oe, args.oe_bins)
            if edges is None:
                sys.exit("could not build O/E bins — too few distinct contacts.")
        nb = len(edges) - 1
        b = np.digitize(log_oe, edges[1:-1])
        b = np.where(np.isfinite(log_oe), b, -1).astype(np.int32)

        sel = oe["idx"]
        cpp_pairs = {}
        for tag in ("g1", "h1", "g2", "h2"):
            p_, s_, q_ = sdc._concat_csr(parts, tag)
            p2, s2, q2 = sdc._permute_csr(p_, s_, q_, sel)
            cpp_pairs[f"{tag}_pos"], cpp_pairs[f"{tag}_st"] = p2, s2
            cpp_pairs[f"{tag}_ptr"] = q2

        acc = _CPP.accumulate_contacts(
            cpp_pairs, b, cell[sel].astype(np.int32), int(nb), len(parts),
            int(args.max_site_pairs_per_contact), int(args.seed))

        pc = {k: [np.asarray(acc[f"{k}_per_cell"][c], np.float64)
                  for c in range(len(parts))] for k, _ in _MEASURES}
        curves = {k: sdc._corr(sum(pc[k]), args.min_pairs) for k, _ in _MEASURES}
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean_sep = np.array([
            (oe["sep"][b == i].mean() * args.resolution) if (b == i).any()
            else np.nan for i in range(nb)])

        boot = None
        if args.n_bootstrap > 0 and len(parts) >= 3:
            boot = {k: _bootstrap_curve(pc[k], centers, args.min_pairs,
                                        args.n_bootstrap, args.seed,
                                        sdc._corr_stack)
                    for k, _ in _MEASURES}
            ps = np.array([boot[k][3] if boot[k] is not None else np.nan
                           for k, _ in _MEASURES])
            qs = sdc._fdr(ps, args.fdr)
            stars = {k: sdc._stars(qs[i]) for i, (k, _) in enumerate(_MEASURES)}
        else:
            stars = {}

        stem = f"contact_concordance_{safe[label]}"
        fig_path = out_dir / f"{stem}.{args.plot_format}"
        _plot(curves, boot, centers, edges, label, len(parts),
              oe["n_contacts_kept"], mean_sep, args.resolution, fig_path,
              args.min_pairs, stars)
        _write_tsv(curves, boot, centers, edges, mean_sep,
                   out_dir / f"{stem}.tsv")

        print(f"\n=== {label}: {len(parts)} cells, "
              f"{oe['n_contacts_kept']:,} contacts over "
              f"{oe['n_bin_pairs']:,} bin-pairs "
              f"({oe['n_contacts_dropped']:,} below the separation floor) ===")
        print(f"  {'log10 O/E':>10}{'mean sep':>12}{'n(GG)':>14} | "
              + "".join(f"{k.upper():>10}" for k, _ in _MEASURES))
        for i in range(nb):
            if not np.isfinite(curves["gg"][0][i]):
                continue
            sep_kb = mean_sep[i] / 1e3
            print(f"  {centers[i]:>10.2f}{sep_kb:>10,.0f}kb"
                  f"{int(curves['gg'][1][i]):>14,d} | "
                  + "".join(f"{curves[k][0][i]:>+10.3f}" for k, _ in _MEASURES))
        if boot:
            for k, _ in _MEASURES:
                if boot[k] is None:
                    continue
                _lo, _hi, rho, p = boot[k]
                if np.isfinite(rho):
                    print(f"  trend {k.upper()}: Spearman ρ={rho:+.3f} "
                          f"(p={p:.4f}{stars.get(k, '')}) — concordance vs "
                          f"spatial proximity")
        print(f"  wrote {fig_path}")

        summary[label] = {
            "n_cells": len(parts),
            "n_contacts": oe["n_contacts_kept"],
            "n_contacts_below_separation_floor": oe["n_contacts_dropped"],
            "n_bin_pairs": oe["n_bin_pairs"],
            "file_stem": stem,
            "trend": {k: (None if not boot or boot[k] is None
                          or not np.isfinite(boot[k][2])
                          else {"spearman_rho": boot[k][2], "p": boot[k][3]})
                      for k, _ in _MEASURES},
        }

    with open(out_dir / "contact_concordance_summary.json", "w") as fh:
        json.dump({"groups": summary, "qc": _QC,
                   "resolution_bp": args.resolution,
                   "min_contact_span_bp": args.min_contact_span,
                   "min_separation_bins": min_sep,
                   "normalization": "observed/expected at matched separation",
                   "oe_bin_edges_log10": (edges.tolist() if edges is not None
                                          else None)}, fh, indent=2)
    print("\nRead the figure: a RISING curve means loci that touch more often "
          "than their\n  genomic separation predicts are also more "
          "epigenetically concordant — spatial\n  proximity, not linear "
          "proximity.  Check the 'decoupling' panel first: if mean\n  "
          "separation is flat across O/E bins, the axis is genuinely spatial.")


if __name__ == "__main__":
    main()
