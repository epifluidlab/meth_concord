#!/usr/bin/env python3
"""Site-level correlation vs genomic distance: GCH–GCH, HCG–HCG, HCG–GCH.

Correlates the methylation state of INDIVIDUAL cytosines (not read-end
averages) as a function of the genomic distance between them, and compares each
curve against a distance-matched, NON-LINKED null.

Only high-quality read pairs are used — uniquely mapped, not a PCR duplicate,
mapQ > 30 on BOTH ends, both ends on the same chromosome.  Those filters are
applied here exactly as ``bam_parser`` applies them upstream (shared skip-mask
and strand logic; see ``nome_hic_concordance/site_calls.py``).

Three site pairings, each a Pearson r on the 0/1 calls (the phi coefficient):

  * GCH–GCH : two GpC sites
  * HCG–HCG : two CpG sites
  * HCG–GCH : a CpG site vs a GpC site (cross-context; the shortest-distance
              bin is the "HCG and its next GCH" case)

Site pairs come from BOTH arms, giving one continuous curve:
  * WITHIN-read  — two sites on the same read  → the short range (≲ read length)
  * CROSS-end    — a site on end 1 × a site on end 2 → the long range (contact)

Two conditions:
  * LINKED  — the read pair's true partner end (the real ligation)
  * NULL    — the partner end is replaced by a DIFFERENT end within
              ±--null-window bp of it (same distance, different molecule), so a
              linked-minus-null gap is same-molecule coordination rather than
              genomic proximity.

Because the null swaps the PARTNER END, it applies to the cross-end arm only.
Within-read pairs have no non-linked counterpart — both cytosines sit on the
same molecule by construction — so the null curve is undefined (NaN) below
roughly the read length, and those bins show the linked curve alone.

By default every cell in --samples is pooled.  With --group-column, a separate
figure + stats file is also written per group (plus the pooled "all").

Usage
-----
    python scripts/site_distance_correlation.py \\
        --samples samples.tsv \\
        --context-bed GCH:/ref/GCH.bed.gz,HCG:/ref/HCG.bed.gz \\
        --output-dir output/site_distance_corr \\
        --max-cells 50 --max-pairs-per-cell 20000 --n-threads 8
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("site_distance")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance.site_calls import iter_cell_read_pairs  # noqa: E402

# htslib backend: does the whole hot path (parse → sites → pair → bin →
# accumulate) in C++.  Essential here — only ~1% of pairs are long-range, so the
# long-range arm needs FULL BAMs, and the Python loop cannot carry ~1e6
# pairs/cell.  Verified bit-identical to the pysam path, ~100x faster.
try:                                                        # pysam first: its
    import pysam as _pysam                                  # bundled libhts
    _ = _pysam                                              # supplies symbols
    from nome_hic_concordance import _site_dist_core as _CPP
except Exception:                                           # pragma: no cover
    _CPP = None

_MEASURES = (("gg", "GCH–GCH  r(site_i, site_j)"),
             ("hh", "HCG–HCG  r(site_i, site_j)"),
             ("gh", "HCG–GCH  r(CpG, GpC)  [cross-context]"))
_CONDS = (("linked", "linked (true partner)"),
          ("null", "non-linked (±window)"))
_TLAB = {1: "1bp", 10: "10bp", 100: "100bp", 1_000: "1kb", 10_000: "10kb",
         100_000: "100kb", 1_000_000: "1Mb", 10_000_000: "10Mb"}
_QC = ("high-quality read pairs only: uniquely mapped, no PCR duplicate, "
       "mapQ>30 both ends, same chromosome")


# --------------------------------------------------------------------------
# Sufficient statistics
# --------------------------------------------------------------------------

def _new_acc(nb):
    """(6, nb) raw sums: n, Σx, Σy, Σx², Σy², Σxy."""
    return np.zeros((6, nb))


def _accumulate(acc, bins, x, y):
    """Add site pairs (already binned) into the (6, nb) sums."""
    if bins.size == 0:
        return
    nb = acc.shape[1]
    x = x.astype(np.float64); y = y.astype(np.float64)
    acc[0] += np.bincount(bins, minlength=nb)
    acc[1] += np.bincount(bins, weights=x, minlength=nb)
    acc[2] += np.bincount(bins, weights=y, minlength=nb)
    acc[3] += np.bincount(bins, weights=x * x, minlength=nb)
    acc[4] += np.bincount(bins, weights=y * y, minlength=nb)
    acc[5] += np.bincount(bins, weights=x * y, minlength=nb)


def _corr(acc, min_pairs):
    """(6, nb) sums → (r, n) per bin; NaN where n < min_pairs or no variance."""
    n, sx, sy, sxx, syy, sxy = acc
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = np.where(n > 0, n, 1.0)
        cov = sxy - sx * sy / nz
        vx = sxx - sx * sx / nz
        vy = syy - sy * sy / nz
        den = np.sqrt(vx * vy)
        r = np.where((n >= min_pairs) & (den > 0), cov / np.where(den > 0, den, 1.0),
                     np.nan)
    return r, n


def _build_edges(min_d, max_d, per_decade):
    lo, hi = np.log10(max(min_d, 1)), np.log10(max_d)
    n_edges = max(2, int(round((hi - lo) * per_decade)) + 1)
    e = np.logspace(lo, hi, n_edges)
    e[0], e[-1] = float(min_d), float(max_d)      # exact bounds (FP-safe)
    return e


def _pair_sites(pos_a, st_a, pos_b, st_b, edges, same_array, rng, cap):
    """All (or a capped sample of) cross products between two site sets.

    ``same_array`` → the two sets are the SAME sites (within-read), so take the
    upper triangle (i<j) and never pair a site with itself.  Returns
    ``(bins, x, y)`` for pairs whose separation falls inside the edges.
    """
    na, nb_ = pos_a.size, pos_b.size
    if na == 0 or nb_ == 0:
        return (np.empty(0, np.int64),) * 3
    if same_array:
        ia, ib = np.triu_indices(na, k=1)
    else:
        ia = np.repeat(np.arange(na), nb_)
        ib = np.tile(np.arange(nb_), na)
    if ia.size == 0:
        return (np.empty(0, np.int64),) * 3
    if cap and ia.size > cap:                      # bound the O(n·m) blow-up
        sel = rng.choice(ia.size, size=cap, replace=False)
        ia, ib = ia[sel], ib[sel]
    d = np.abs(pos_b[ib] - pos_a[ia])
    keep = (d >= edges[0]) & (d <= edges[-1])
    if not keep.any():
        return (np.empty(0, np.int64),) * 3
    ia, ib, d = ia[keep], ib[keep], d[keep]
    nbins = len(edges) - 1
    bins = np.clip(np.digitize(d, edges) - 1, 0, nbins - 1).astype(np.int64)
    return bins, st_a[ia], st_b[ib]


def _add_pair(accs, cond, edges, rng, cap, c1, c2, cross_only=False):
    """Accumulate every measure for one (end1, end2) site-call pair."""
    nb = accs[("gg", cond)].shape[1]
    g1, g2 = c1.get("GCH", (np.empty(0), np.empty(0))), c2.get("GCH", (np.empty(0), np.empty(0)))
    h1, h2 = c1.get("HCG", (np.empty(0), np.empty(0))), c2.get("HCG", (np.empty(0), np.empty(0)))

    def go(key, pa, sa, pb, sb, same):
        b, x, y = _pair_sites(pa, sa, pb, sb, edges, same, rng, cap)
        _accumulate(accs[(key, cond)], b, x, y)

    if not cross_only:
        # WITHIN-read arm (short range) — both ends contribute
        go("gg", g1[0], g1[1], g1[0], g1[1], True)
        go("gg", g2[0], g2[1], g2[0], g2[1], True)
        go("hh", h1[0], h1[1], h1[0], h1[1], True)
        go("hh", h2[0], h2[1], h2[0], h2[1], True)
        go("gh", h1[0], h1[1], g1[0], g1[1], False)
        go("gh", h2[0], h2[1], g2[0], g2[1], False)
    # CROSS-end arm (long range)
    go("gg", g1[0], g1[1], g2[0], g2[1], False)
    go("hh", h1[0], h1[1], h2[0], h2[1], False)
    go("gh", h1[0], h1[1], g2[0], g2[1], False)     # CpG(end1) × GpC(end2)
    go("gh", h2[0], h2[1], g1[0], g1[1], False)     # and the reverse ordering
    _ = nb


def prepare_ctx(ctx_positions, blacklist=None):
    """Convert the context tables to the C++ layout ONCE.

    ``arr["pos"]`` on a structured array is a strided view, so ``.astype`` copies
    the whole table (~2 GB for a mammalian GCH+HCG set).  Doing that per cell
    burned that allocation 1577 times per worker; hoisting it here means the
    buffers are built once and, under the thread pool, shared by every worker.
    """
    ctx_g = {c: (np.ascontiguousarray(a["pos"], np.int64),
                 np.ascontiguousarray(a["ref_base"], np.uint8))
             for c, a in ctx_positions.get("GCH", {}).items()}
    ctx_h = {c: (np.ascontiguousarray(a["pos"], np.int64),
                 np.ascontiguousarray(a["ref_base"], np.uint8))
             for c, a in ctx_positions.get("HCG", {}).items()}
    bl = {c: (np.ascontiguousarray(iv[:, 0], np.int64),
              np.ascontiguousarray(iv[:, 1], np.int64))
          for c, iv in (blacklist or {}).items() if len(iv)}
    return ctx_g, ctx_h, bl


def process_cell_cpp(bam_path, prepared, edges, trim_ends, mapq_min,
                     cap, null_window, seed, baseq_min=0):
    """htslib fast path — same result as :func:`process_cell`, ~100x faster.

    ``prepared`` comes from :func:`prepare_ctx`; the extension releases the GIL,
    so this is safe to run on a thread pool.
    """
    ctx_g, ctx_h, bl = prepared
    res = _CPP.parse_cell_site_distance(
        str(bam_path), ctx_g, ctx_h, [float(e) for e in edges],
        int(trim_ends), int(mapq_min), int(null_window), int(cap), int(seed),
        False, int(baseq_min), bl)
    accs = {}
    for m, _ in _MEASURES:
        for c, _lab in _CONDS:
            accs[(m, c)] = np.asarray(res[f"{m}_{c}"], dtype=np.float64)
    return accs, int(res["n_pairs"])


def process_cell(bam_path, ctx_positions, edges, trim_ends, mapq_min,
                 max_pairs, cap, null_window, seed):
    """One cell → {(measure, condition): (6, nb) sums}.

    Both conditions are built from the SAME parsed pairs (one BAM pass): the
    linked condition uses the true partner end; the null replaces it with a
    different end whose midpoint lies within ±``null_window`` of it.

    Pure-Python/pysam fallback; the htslib backend above is used when built.
    """
    rng = np.random.default_rng(seed)
    nb = len(edges) - 1
    accs = {(m, c): _new_acc(nb) for m, _ in _MEASURES for c, _ in _CONDS}

    pairs = list(iter_cell_read_pairs(
        bam_path, ctx_positions, trim_ends=trim_ends, mapq_min=mapq_min,
        max_pairs_per_cell=max_pairs, rng=rng))
    if not pairs:
        return accs, 0

    for chrom, mid1, mid2, c1, c2 in pairs:
        _add_pair(accs, "linked", edges, rng, cap, c1, c2)

    # ---- NULL: swap in a nearby, non-linked partner ----------------------
    # Index end-2 midpoints per chromosome so "within ±window of the true
    # partner" is a binary search rather than an O(n²) scan.
    by_chrom: dict[str, tuple] = {}
    for chrom in {p[0] for p in pairs}:
        idx = [i for i, p in enumerate(pairs) if p[0] == chrom]
        mids = np.array([pairs[i][2] for i in idx], np.int64)
        order = np.argsort(mids)
        by_chrom[chrom] = (mids[order], np.asarray(idx)[order])

    for i, (chrom, mid1, mid2, c1, _c2) in enumerate(pairs):
        mids, idx = by_chrom[chrom]
        lo = np.searchsorted(mids, mid2 - null_window, side="left")
        hi = np.searchsorted(mids, mid2 + null_window, side="right")
        if hi - lo < 2:                       # only itself in the window
            continue
        for _ in range(4):                    # a few draws to avoid self
            j = int(idx[rng.integers(lo, hi)])
            if j != i:
                break
        else:
            continue
        _add_pair(accs, "null", edges, rng, cap, c1, pairs[j][4],
                  cross_only=True)
    return accs, len(pairs)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _load_snps_all(dbsnp_vcf, wanted_chroms):
    """``{chrom: sorted 0-based SNP positions}`` for every wanted chromosome.

    Deliberately NOT reference._load_snp_positions, which needs a tabix index
    and returns an EMPTY set on any failure — an unindexed .vcf or a chr-naming
    mismatch then silently disables the filter.  Here an indexed VCF is queried
    per chromosome, an unindexed one is read in a single sequential pass, and
    'chr1' / '1' are reconciled.  Returns {} only when the VCF truly has no
    usable records, which the caller treats as an error.
    """
    import pysam

    def _norm(c):
        return c[3:] if c.startswith("chr") else c

    want = {_norm(c): c for c in wanted_chroms}
    found: dict[str, set] = {c: set() for c in wanted_chroms}
    vcf = pysam.VariantFile(dbsnp_vcf)
    indexed = False
    try:
        indexed = vcf.index is not None
    except (AttributeError, ValueError):
        indexed = False

    n = 0
    if indexed:
        for norm, orig in want.items():
            for name in (orig, norm, f"chr{norm}"):     # try both conventions
                try:
                    it = vcf.fetch(name)
                except (ValueError, KeyError):
                    continue
                for rec in it:
                    found[orig].add(rec.pos - 1)        # VCF 1-based → 0-based
                    n += 1
                if found[orig]:
                    break
    else:
        logger.warning(
            "%s has no tabix index, so it must be read sequentially in ONE "
            "pass — this phase is inherently single-threaded (~1-2 min for a "
            "mammalian dbSNP). bgzip + tabix it to avoid this.", dbsnp_vcf)
        t_scan = time.time()
        for rec in vcf.fetch():                         # whole-file scan
            orig = want.get(_norm(rec.chrom))
            if orig is not None:
                found[orig].add(rec.pos - 1)
                n += 1
                if n % 10_000_000 == 0:                 # keep the phase visible
                    print(f"    dbSNP scan: {n / 1e6:.0f}M records "
                          f"({time.time() - t_scan:.0f}s)", flush=True)
    vcf.close()
    logger.info("dbSNP: loaded %s SNP positions over %s chromosomes",
                f"{n:,}", sum(1 for v in found.values() if v))
    return {c: v for c, v in found.items() if v}


def _drop_snp_sites(ctx_positions, dbsnp_vcf, motif_flank=1):
    """Remove context sites whose MOTIF overlaps a dbSNP locus.

    Checking only the cytosine is not enough: GCH and HCG are 3-bp motifs, and a
    SNP at the flanking G (or at the H) means the locus is not that motif in
    this sample at all — a GCH whose G is polymorphic may really be an HCH, so
    its "accessibility" call is meaningless.  A SNP at the cytosine itself is
    the other failure (an unconverted C and a genomic T are indistinguishable).
    So a site is dropped when ANY of pos-1, pos, pos+1 carries a SNP, which
    covers the motif on either strand (the cytosine is always the middle base).
    """
    out, dropped, kept = {}, 0, 0
    chroms = sorted({c for ctx in ctx_positions.values() for c in ctx})
    snps = _load_snps_all(dbsnp_vcf, chroms)
    if not snps:
        sys.exit(
            f"--dbsnp-vcf {dbsnp_vcf}: no SNP positions matched any of the "
            f"{len(chroms)} context chromosomes (e.g. {chroms[:3]}).\n"
            f"  Check that the VCF's chromosome names match the context BED's "
            f"and that the file is a readable VCF.\n"
            f"  Refusing to continue silently — an unfiltered run would look "
            f"identical to a filtered one.")
    for ctx, per_chrom in ctx_positions.items():
        out[ctx] = {}
        for chrom, arr in per_chrom.items():
            s = snps.get(chrom)
            if not s:
                out[ctx][chrom] = arr
                kept += len(arr)
                continue
            snp_arr = np.fromiter(s, np.int64, len(s))
            pos = arr["pos"]
            # drop if ANY motif base (pos-flank .. pos+flank) hits a SNP
            hit = np.zeros(pos.shape, dtype=bool)
            for off in range(-motif_flank, motif_flank + 1):
                hit |= np.isin(pos + off, snp_arr)
            mask = ~hit
            out[ctx][chrom] = arr[mask]
            dropped += int((~mask).sum())
            kept += int(mask.sum())
    total = dropped + kept
    print(f"  dbSNP filter: dropped {dropped:,} of {total:,} context sites "
          f"({100.0 * dropped / max(total, 1):.2f}%)")
    if dropped == 0:
        print("  WARNING: 0 sites dropped despite SNPs being loaded — check "
              "the coordinate convention of the VCF.", file=sys.stderr)
    return out


_UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9._+-]")


def safe_names(labels):
    """``{label: filename-safe name}``, unique across ``labels``.

    Cell-type labels routinely contain characters that are not filename-safe —
    'IT-L2/3' and 'PC/EC' turn a filename into a path into a directory that
    does not exist.  Slashes and friends collapse to '_', and if two labels
    collapse onto the same name a numeric suffix keeps them distinct.  Only the
    FILENAME is rewritten; titles and the summary JSON keep the real label.
    """
    out, used = {}, {}
    for lab in labels:
        base = _UNSAFE_IN_NAME.sub("_", str(lab).strip()) or "unnamed"
        if not base.strip("."):        # '.' / '..' are path components, not names
            base = "unnamed"
        n = used.get(base, 0)
        used[base] = n + 1
        out[lab] = base if n == 0 else f"{base}_{n + 1}"
    return out


def _logx(ax, lo, hi):
    ax.set_xscale("log")
    ticks = [t for t in _TLAB if lo <= t <= hi]
    ax.set_xticks(ticks); ax.set_xticklabels([_TLAB[t] for t in ticks])
    ax.set_xlabel("genomic distance between the two cytosines")


def _plot(curves, centers, edges, label, n_cells, n_pairs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"gg": "#1f77b4", "hh": "#d62728", "gh": "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    panels = dict(zip(("gg", "hh", "gh"), axes.flat[:3]))
    for key, title in _MEASURES:
        ax = panels[key]
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        for cond, clabel in _CONDS:
            r, n = curves[(key, cond)]
            ax.plot(centers, r, "-o" if cond == "linked" else "--",
                    color=colors[key], ms=3.2, lw=1.5,
                    alpha=1.0 if cond == "linked" else 0.65, label=clabel)
        _logx(ax, edges[0], edges[-1])
        ax.set_ylabel("Pearson r (phi) of 0/1 calls")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="best")
    ax = axes.flat[3]
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    for key, title in _MEASURES:
        ax.plot(centers, curves[(key, "linked")][0], "-o", ms=3, lw=1.5,
                color=colors[key], label=title.split("  ")[0])
    _logx(ax, edges[0], edges[-1])
    ax.set_ylabel("Pearson r (phi) of 0/1 calls")
    ax.set_title("linked curves overlaid", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    axn = ax.twinx()
    axn.plot(centers, curves[("gg", "linked")][1], color="0.5", lw=0.9, alpha=0.5)
    axn.set_yscale("log")
    axn.set_ylabel("GCH–GCH site-pair count", color="0.5", fontsize=8)
    axn.tick_params(axis="y", colors="0.5", labelsize=7)
    fig.suptitle(f"Site-level correlation vs distance — {label}  |  "
                 f"{n_cells} cells, {n_pairs:,} read pairs", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.text(0.5, 0.012, _QC, ha="center", fontsize=7.5, color="0.35")
    fig.savefig(out)
    plt.close(fig)


def _write_tsv(curves, centers, edges, path):
    rows = {"bin_lo": edges[:-1].astype(np.int64),
            "bin_hi": edges[1:].astype(np.int64),
            "center_bp": np.round(centers).astype(np.int64)}
    for key, _ in _MEASURES:
        for cond, _lab in _CONDS:
            r, n = curves[(key, cond)]
            rows[f"{key}_{cond}_r"] = r
            rows[f"{key}_{cond}_n"] = n.astype(np.int64)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, na_rep="nan")


def _report(accs, n_cells, n_pairs, centers, edges, label, out_dir, args,
            safe=None):
    curves = {k: _corr(v, args.min_pairs) for k, v in accs.items()}
    # Build the two paths explicitly: Path.with_suffix() would ALSO mangle a
    # label containing a dot ('A.B' -> '..._A.pdf'), silently.
    stem_name = f"site_distance_corr_{safe or label}"
    fig_path = out_dir / f"{stem_name}.{args.plot_format}"
    tsv_path = out_dir / f"{stem_name}.tsv"
    _plot(curves, centers, edges, label, n_cells, n_pairs, fig_path)
    _write_tsv(curves, centers, edges, tsv_path)
    print(f"\n=== {label}: {n_cells} cells, {n_pairs:,} read pairs ===")
    print(f"  {'distance':>10}{'n(GG)':>12} | "
          + "".join(f"{m.split('  ')[0]:>22}" for _, m in _MEASURES))
    for bi in range(len(centers)):
        if not np.isfinite(curves[("gg", "linked")][0][bi]):
            continue
        cols = ""
        for key, _ in _MEASURES:
            rl = curves[(key, "linked")][0][bi]
            rn = curves[(key, "null")][0][bi]
            cols += f"{rl:+.3f}/{rn:+.3f}".rjust(22)
        print(f"  {centers[bi]:>10,.0f}{int(curves[('gg','linked')][1][bi]):>12,d} | {cols}")
    print(f"  wrote {fig_path} and {tsv_path}")
    return curves


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True,
                    help="TSV with cell_name, bam_path [, group].")
    ap.add_argument("--context-bed", default=None,
                    help="'GCH:/path/GCH.bed.gz,HCG:/path/HCG.bed.gz' — same "
                         "CONTEXT:PATH form as nome_hic_concordance. Either "
                         "this or --reference is required.")
    ap.add_argument("--context-bed-offset", type=int, default=0)
    ap.add_argument("--reference", default=None,
                    help="Reference .2bit — extracts the GCH/HCG positions "
                         "directly (same as nome_hic_concordance Step 1) when "
                         "no --context-bed is given.  Note it does NOT change "
                         "how methylation is called: the state comes from the "
                         "read base vs the site's reference base, which the "
                         "context table already carries.")
    ap.add_argument("--dbsnp-vcf", default=None,
                    help="Tabix-indexed dbSNP VCF; context sites overlapping a "
                         "SNP are dropped (a C/T polymorphism is "
                         "indistinguishable from a bisulfite conversion).")
    ap.add_argument("--blacklist-bed", default=None,
                    help="BED of blacklisted regions; a read PAIR is dropped if "
                         "EITHER end overlaps one.")
    ap.add_argument("--baseq-min", type=int, default=0,
                    help="Drop individual cytosines whose basecall quality is "
                         "below this (0 = off). A miscalled base is a miscalled "
                         "methylation state, so this filters loci, not reads.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--group-column", default=None,
                    help="Column in --samples; also emit a figure per group "
                         "(the pooled 'all' is always written).")
    ap.add_argument("--min-distance", type=int, default=1)
    ap.add_argument("--max-distance", type=int, default=1_000_000)
    ap.add_argument("--bins-per-decade", type=int, default=8)
    ap.add_argument("--null-window", type=int, default=100,
                    help="Replace the partner end with a different end whose "
                         "midpoint is within ±this many bp (default 100).")
    ap.add_argument("--mapq-min", type=int, default=30)
    ap.add_argument("--trim-ends", type=int, default=5)
    ap.add_argument("--max-cells", type=int, default=0,
                    help="0 = all cells in --samples.")
    ap.add_argument("--max-pairs-per-cell", type=int, default=0,
                    help="0 (default) = use ALL read pairs. Only ~1%% of pairs "
                         "are long-range, so capping would starve the "
                         "long-range arm; the htslib backend makes the full "
                         "pass affordable. A non-zero value down-samples "
                         "uniformly (rate from the BAM index), and is only "
                         "honoured by the --backend python fallback.")
    ap.add_argument("--backend", choices=["auto", "htslib", "python"],
                    default="auto",
                    help="'auto' (default) uses the C++/htslib backend when "
                         "built and falls back to pysam otherwise; 'htslib' "
                         "requires it; 'python' forces the pysam path.")
    ap.add_argument("--max-site-pairs-per-read", type=int, default=64,
                    help="Cap site-pair combinations sampled per read pair, "
                         "bounding the O(n·m) blow-up on site-dense reads.")
    ap.add_argument("--min-pairs", type=int, default=100,
                    help="Drop a distance bin with fewer site pairs than this.")
    ap.add_argument("--plot-format", choices=["png", "pdf", "svg"], default="pdf")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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
                sys.exit(f"--context-bed must define {need}: (got {list(bed)}).")
        ctx_positions = load_context_from_bed(
            bed, n_threads=args.n_threads, c_offset=args.context_bed_offset)
        if args.dbsnp_vcf:
            ctx_positions = _drop_snp_sites(ctx_positions, args.dbsnp_vcf)
    else:
        # Step-1 extraction applies --dbsnp-vcf itself.
        ctx_positions = extract_context_positions(
            args.reference, str(Path(args.output_dir) / "context"),
            contexts=["GCH", "HCG"], dbsnp_vcf=args.dbsnp_vcf,
            n_threads=args.n_threads)

    blacklist = None
    if args.blacklist_bed:
        from nome_hic_concordance.blacklist import load_blacklist_bed
        blacklist = load_blacklist_bed(args.blacklist_bed)
        n_iv = sum(len(v) for v in blacklist.values())
        print(f"  blacklist: {n_iv:,} intervals over {len(blacklist)} chroms")
    out_dir = Path(ensure_dir(args.output_dir))
    edges = _build_edges(args.min_distance, args.max_distance,
                         args.bins_per_decade)
    centers = np.sqrt(edges[:-1] * edges[1:])
    use_cpp = _CPP is not None if args.backend == "auto" else args.backend == "htslib"
    if use_cpp and _CPP is None:
        sys.exit("--backend htslib requested but the extension is not built "
                 "(pip install -e . with htslib + pybind11 available).")
    backend = "htslib (C++)" if use_cpp else "python (pysam)"
    if not use_cpp and not args.max_pairs_per_cell:
        print("  NOTE: the pysam fallback on FULL BAMs is ~100x slower than the "
              "htslib backend; build the extension or set --max-pairs-per-cell.",
              file=sys.stderr)
    pairs_note = ("ALL pairs" if not args.max_pairs_per_cell
                  else f"~{args.max_pairs_per_cell:,} pairs/cell")
    print(f"# {_QC}")
    print(f"[site-dist] {len(samples)} cells | {len(edges)-1} bins "
          f"{args.min_distance:,}–{args.max_distance:,} bp | null window ±"
          f"{args.null_window} bp | {pairs_note} | backend {backend}")

    if use_cpp:
        prepared = prepare_ctx(ctx_positions, blacklist)
        n_sites = sum(len(v[0]) for v in prepared[0].values()) + \
            sum(len(v[0]) for v in prepared[1].values())
        print(f"  context prepared once: {n_sites:,} sites, shared across "
              f"threads (no per-worker copy)")
        jobs = [(row["bam_path"], prepared, edges, args.trim_ends,
                 args.mapq_min, args.max_site_pairs_per_read,
                 args.null_window, args.seed + i, args.baseq_min)
                for i, (_, row) in enumerate(samples.iterrows())]
        fn = process_cell_cpp
    else:
        if args.baseq_min or blacklist:
            print("  NOTE: --baseq-min / --blacklist-bed are applied by the "
                  "htslib backend only; the python fallback ignores them.",
                  file=sys.stderr)
        jobs = [(row["bam_path"], ctx_positions, edges, args.trim_ends,
                 args.mapq_min, args.max_pairs_per_cell,
                 args.max_site_pairs_per_read, args.null_window, args.seed + i)
                for i, (_, row) in enumerate(samples.iterrows())]
        fn = process_cell

    cell_names = samples["cell_name"].tolist()
    n_jobs_total = len(jobs)
    t0 = time.time()

    def _tick(done, idx, n_pairs):
        """One line per finished cell: which cell, how far along, and an ETA."""
        el = time.time() - t0
        eta = el / max(done, 1) * (n_jobs_total - done)
        print(f"  [{done}/{n_jobs_total}] {cell_names[idx]}: {n_pairs:,} pairs "
              f"| {el / 60:.1f} min elapsed, ETA {eta / 60:.1f} min",
              flush=True)

    if use_cpp and args.n_threads > 1 and n_jobs_total > 1:
        # THREADS: the extension releases the GIL, so they run truly in parallel
        # AND share the ~GB context tables (processes would each need a pickled
        # copy).  Driving the pool directly — rather than joblib — is what lets
        # us report "[i/N] <cell>" instead of joblib's opaque "Done N tasks".
        from concurrent.futures import ThreadPoolExecutor, as_completed
        res = [None] * n_jobs_total
        with ThreadPoolExecutor(max_workers=args.n_threads) as ex:
            futs = {ex.submit(fn, *j): i for i, j in enumerate(jobs)}
            for done, fut in enumerate(as_completed(futs), 1):
                i = futs[fut]
                res[i] = fut.result()
                _tick(done, i, res[i][1])
    elif args.n_threads > 1 and n_jobs_total > 1:
        from joblib import Parallel, delayed          # pysam fallback: processes
        res = Parallel(n_jobs=args.n_threads, verbose=10)(
            delayed(fn)(*j) for j in jobs)
    else:
        res = []
        for i, j in enumerate(jobs):
            res.append(fn(*j))
            _tick(i + 1, i, res[-1][1])
    print(f"  parsed {len(res)} cells in {(time.time() - t0) / 60:.1f} min "
          f"({sum(r[1] for r in res):,} read pairs total)")

    keys = [(m, c) for m, _ in _MEASURES for c, _ in _CONDS]
    groups: dict[str, list[int]] = {"all": list(range(len(res)))}
    if args.group_column:
        if args.group_column not in samples.columns:
            sys.exit(f"--group-column {args.group_column!r} not in --samples "
                     f"(columns: {samples.columns.tolist()}).")
        for gi, g in enumerate(samples[args.group_column].to_numpy()):
            groups.setdefault(str(g), []).append(gi)

    summary = {}
    safe = safe_names(groups)          # 'IT-L2/3' -> 'IT-L2_3', collision-proof
    for label, idxs in groups.items():
        accs = {k: sum(res[i][0][k] for i in idxs) for k in keys}
        n_pairs = sum(res[i][1] for i in idxs)
        if n_pairs == 0:
            print(f"\n=== {label}: no usable read pairs — skipped ===")
            continue
        _report(accs, len(idxs), n_pairs, centers, edges, label, out_dir, args,
                safe=safe[label])
        summary[label] = {"n_cells": len(idxs), "n_read_pairs": int(n_pairs),
                          "file_stem": f"site_distance_corr_{safe[label]}"}
    with open(out_dir / "site_distance_corr_summary.json", "w") as fh:
        json.dump({"groups": summary, "qc": _QC,
                   "null_window_bp": args.null_window}, fh, indent=2)
    print("\nRead the figures: below ~read length the pairs are WITHIN-read, which "
          "has no\n  non-linked counterpart (both cytosines are on one molecule), "
          "so the null is\n  blank there and the linked curve alone is the 1-D "
          "co-methylation decay.  At\n  longer range a LINKED curve sitting above "
          "its non-linked null is same-molecule\n  coordination rather than mere "
          "genomic proximity.")


if __name__ == "__main__":
    main()
