#!/usr/bin/env python3
"""One figure: the signal at several BIN SETS, against several tracks.

Built for the result it is named after -- the invariant (``stable``) bins sit in
EARLY-replicating sequence and the cell-type-variable (``selected``) bins in
LATE, in both contexts -- which otherwise lives across four separate enrichment
PDFs.

The plot is the conventional one for a continuous track at a set of regions: the
DISTRIBUTION of the per-bin signal, background against each bin set, in the
track's own units, with a Mann-Whitney test of each set against the background.
By default one panel per (timing group x context), so every panel has its own
y-axis and tracks on different scales are never forced onto a shared one.

``--layout single`` puts everything in ONE panel instead, ordered context-major
so each context's early/late pair sits together -- for the compact version of the
figure with a single sample.  That shares one y-axis across tracks, so it needs
``--normalize``: ``log2-median`` is the conventional enrichment axis, log2 of the
signal over the genome-wide median, which puts the background at 0 and makes each
set read as fold-enrichment over the genome average.  The stats TSV always
reports the RAW units, whatever the figure is drawn in.

    python scripts/bin_set_signal_summary.py \\
        --background GCH=out/marker_gch/bin_scores_gch.all.bed \\
        --background HCG=out/marker_hcg/bin_scores_hcg.all.bed \\
        --bin-set GCH:variable=out/marker_gch/bin_scores_gch.selected.bed \\
        --bin-set GCH:invariant=out/marker_gch/bin_scores_gch.stable.bed \\
        --bin-set HCG:variable=out/marker_hcg/bin_scores_hcg.selected.bed \\
        --bin-set HCG:invariant=out/marker_hcg/bin_scores_hcg.stable.bed \\
        --signal-track Cas129_NPC_rep1_early=rt/Cas129_rep1_early.bw \\
        --signal-track Cas129_NPC_rep1_late=rt/Cas129_rep1_late.bw \\
        ... \\
        --output-dir out/11_replication_summary

Each background BED is the universe for its context: every track is summarised
per bin ONCE on that universe, and each bin set is then a subset of it matched by
coordinate.  So the sets and the background are guaranteed to be the same bins
measured the same way, and a set bin missing from the universe is reported rather
than silently dropped.

Tracks are grouped by an ``_early`` / ``_late`` token in the name, which is how
Repli-seq tracks are conventionally named; ``--track-group`` overrides that for
anything else.  With no group at all the figure is one row.

One caveat the figure cannot show.  The p-values treat bins as independent, and
replication timing is organised into domains far larger than a 250 kb bin, so
they are anticonservative -- with thousands of bins almost anything reaches
significance.  Read the box positions, not the number of stars;
marker_bin_enrichment reports a block-bootstrap interval for the effect if a
defensible interval is wanted.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("bin_set_signal_summary")

# Background grey, then one colour per set: variable warm, invariant cool, as in
# the marker-bin figures.
_BG_COLOR = "#B8BDC4"
_SET_COLOR = {"variable": "#D1495B", "selected": "#D1495B",
              "invariant": "#1F6F8B", "stable": "#1F6F8B"}
_FALLBACK = ["#8C5E8A", "#3E8E7E", "#B07C4F", "#5F4BB6"]


def _enrich_mod():
    """The sibling enrichment script, for its BED reader and track summariser."""
    path = Path(__file__).resolve().parent / "marker_bin_enrichment.py"
    spec = importlib.util.spec_from_file_location("_mbe", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_mbe"] = m
    spec.loader.exec_module(m)
    return m


def _parse_labelled(spec, what, sep=":"):
    """'CONTEXT:LABEL=PATH' → (context, label, path)."""
    if "=" not in spec:
        sys.exit(f"{what} expects CONTEXT{sep}LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    if sep not in label:
        sys.exit(f"{what} label expects CONTEXT{sep}LABEL, got {label!r}")
    ctx, name = label.split(sep, 1)
    return ctx.strip(), name.strip(), path.strip()


def _parse_pair(spec, what):
    if "=" not in spec:
        sys.exit(f"{what} expects NAME=PATH, got {spec!r}")
    k, v = spec.split("=", 1)
    return k.strip(), v.strip()


def track_group(name, rules=None) -> str:
    """early / late / signal, from --track-group or the conventional suffix."""
    if rules and name in rules:
        return rules[name]
    n = str(name).lower()
    if re.search(r"(^|[_\-.])early([_\-.]|$)", n):
        return "early"
    if re.search(r"(^|[_\-.])late([_\-.]|$)", n):
        return "late"
    return "signal"


def normalise(val, mode, background_mask):
    """Put a track on a scale that can share an axis with another track.

    Necessary for a single-panel layout and only there: with one panel per
    track, each keeps its own axis and its own units, which is always the more
    faithful thing to plot.  Two tracks in one panel do not have that option --
    an early Repli-seq track and a late one are not on the same scale, and a
    shared raw axis would show the difference between the two ASSAYS rather than
    the difference between the bin sets.

    'log2-median' is the conventional enrichment axis: log2 of the signal over
    the genome-wide (background) median, so the background sits at 0 and a set's
    box reads directly as fold-enrichment over the genome average.  It needs
    positive values, so it refuses a track that is already a log ratio.
    'zscore' works whatever the sign, at the cost of a less concrete axis.
    """
    v = np.asarray(val, float)
    if mode == "none":
        return v, "per-bin mean signal"
    ref = v[background_mask & np.isfinite(v)]
    if ref.size == 0:
        return v, "per-bin mean signal"
    if mode == "log2-median":
        med = float(np.median(ref))
        if med <= 0 or np.nanmin(v) <= 0:
            sys.exit(
                "--normalize log2-median needs strictly positive signal (its "
                "reference is a ratio to the genome-wide median), and this "
                "track has values at or below zero -- it is probably already a "
                "log ratio.  Use --normalize zscore, or --layout grid, which "
                "needs no normalisation at all.")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log2(v / med), "log$_2$(signal / genome median)"
    if mode == "zscore":
        mu, sd = float(np.mean(ref)), float(np.std(ref))
        if sd <= 0:
            return v - mu, "signal − genome mean"
        return (v - mu) / sd, "signal, z-score across bins"
    raise ValueError(mode)


def gather(backgrounds, bin_sets, tracks, mbe, normalize="none"):
    """Per-bin signal for every (context, track), plus each set's membership.

    Returns ``{context: {"values": {track: array}, "sets": {label: mask},
    "bins": DataFrame}}`` over that context's universe.
    """
    out = {}
    for ctx, bg_path in backgrounds.items():
        bins = mbe._read_bins_any(bg_path)
        key = bins[["chrom", "start", "end"]].astype(str).agg("|".join, axis=1)
        pos = {k: i for i, k in enumerate(key)}
        raw = {label: mbe._signal_bin_means(bins, path)
               for label, path in tracks.items()}
        sets = {}
        for (c, label), sp in bin_sets.items():
            if c != ctx:
                continue
            sub = mbe._read_bins_any(sp)
            sk = sub[["chrom", "start", "end"]].astype(str).agg("|".join, axis=1)
            idx = [pos[k] for k in sk if k in pos]
            missing = len(sk) - len(idx)
            if missing:
                logger.warning(
                    f"{ctx}:{label}: {missing:,} of {len(sk):,} bins are not in "
                    f"the {ctx} background and were dropped — the set and the "
                    f"background must be the same bin grid for the comparison "
                    f"to mean anything.")
            if not idx:
                sys.exit(f"{ctx}:{label}: no bin overlaps the {ctx} background "
                         f"({bg_path}).  Same grid, same coordinates?")
            m = np.zeros(len(bins), bool)
            m[idx] = True
            sets[label] = m
        # The normalisation reference is the background -- every bin outside
        # every named set -- so a set cannot move its own baseline.
        any_set = np.zeros(len(bins), bool)
        for m in sets.values():
            any_set |= m
        values, ylab = {}, "per-bin mean signal"
        for label, v in raw.items():
            values[label], ylab = normalise(v, normalize, ~any_set)
        out[ctx] = {"bins": bins, "values": values, "sets": sets,
                    "raw": raw, "ylabel": ylab}
    return out


def stats_table(data, groups):
    """Long table: one row per (context, track, set) with the test vs background."""
    from scipy.stats import mannwhitneyu
    rows = []
    for ctx, d in data.items():
        # Raw values: the table is the record of what was measured, and a
        # figure-side rescaling should not change the numbers reported.
        for track, val in d.get("raw", d["values"]).items():
            ok = np.isfinite(val)
            for label, mask in d["sets"].items():
                a = val[mask & ok]
                b = val[(~mask) & ok]           # background = universe minus set
                p = np.nan
                if a.size and b.size:
                    try:
                        p = float(mannwhitneyu(a, b,
                                               alternative="two-sided").pvalue)
                    except ValueError:
                        p = np.nan
                rows.append({
                    "context": ctx, "track": track,
                    "track_group": track_group(track, groups), "bin_set": label,
                    "n_set": int(a.size), "n_background": int(b.size),
                    "mean_set": float(a.mean()) if a.size else np.nan,
                    "mean_background": float(b.mean()) if b.size else np.nan,
                    "median_set": float(np.median(a)) if a.size else np.nan,
                    "median_background": (float(np.median(b)) if b.size
                                          else np.nan),
                    "difference_of_means": ((float(a.mean() - b.mean()))
                                            if a.size and b.size else np.nan),
                    "pvalue": p})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["fdr"] = _bh(df["pvalue"].to_numpy())
    return df


def _bh(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if idx.size == 0:
        return out
    pv = p[idx]
    order = np.argsort(pv)
    m = pv.size
    q = pv[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[idx[order]] = np.clip(q, 0.0, 1.0)
    return out


def _stars(q):
    if not np.isfinite(q):
        return "n/a"
    return ("***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05
            else "ns")


def _draw_slot(ax, d, sets, colors, val, base, per):
    """Background box then one per set, at consecutive positions from ``base``.

    Background first so the eye reads left to right from "every bin" to "this
    subset", and the background EXCLUDES every named set -- otherwise the subsets
    sit on both sides of their own comparison.
    """
    ok = np.isfinite(val)
    series = [("background", val[ok & ~_any_set(d, sets)], _BG_COLOR)]
    for s_ in sets:
        series.append((s_, val[ok & d["sets"][s_]], colors[s_]))
    drawn = []
    for k, (lab, v, col) in enumerate(series):
        if v.size == 0:
            continue
        bp = ax.boxplot([v], positions=[base + k], widths=0.72,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="#111827", lw=1.4),
                        whiskerprops=dict(color="#4B5563"),
                        capprops=dict(color="#4B5563"))
        bp["boxes"][0].set(facecolor=col, alpha=0.85, lw=0.6,
                           edgecolor="#374151")
        if lab != "background":
            drawn.append((base + k, lab, col))
    return drawn


def _mark(ax, marks, q_of, key_of):
    """Significance stars, after every box exists.

    The y-limit is not final until then; annotating against a partial one
    scatters the stars across the panel and into the title.
    """
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.14 * (hi - lo))
    top = ax.get_ylim()[1] - 0.045 * (hi - lo)
    for x, lab, col in marks:
        ax.annotate(_stars(q_of.get(key_of(x) + (lab,), np.nan)), (x, top),
                    ha="center", va="center", fontsize=7.6, color=col)


def figure(data, stats, out_path, args):
    """Boxplots of the signal: background against each set, per track."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    contexts = list(data)
    sets = list(dict.fromkeys(stats["bin_set"]))
    colors = {s: _SET_COLOR.get(s.lower(), _FALLBACK[i % len(_FALLBACK)])
              for i, s in enumerate(sets)}
    order = ["early", "late", "signal"]
    groups = [g for g in order if (stats["track_group"] == g).any()]
    tracks_by_group = {
        g: sorted(stats.loc[stats["track_group"] == g, "track"].unique())
        for g in groups}
    q_of = {(r["context"], r["track"], r["bin_set"]): r["fdr"]
            for _, r in stats.iterrows()}
    per = len(sets) + 1
    ylab = data[contexts[0]]["ylabel"]

    if args.layout == "single":
        # Slots run context-major, timing-minor: HCG early, HCG late, then GCH
        # early, GCH late -- the reading order the comparison is stated in.
        slots = [(ctx, t) for ctx in contexts
                 for g in groups for t in tracks_by_group[g]]
        fig, ax = plt.subplots(figsize=(1.45 * per * len(slots) / 3 + 2.6, 5.4))
        marks, centres, labels = [], [], []
        for i, (ctx, track) in enumerate(slots):
            base = i * (per + 1)
            marks += _draw_slot(ax, data[ctx], sets, colors,
                                data[ctx]["values"][track], base, per)
            centres.append(base + (per - 1) / 2)
            labels.append(track)
        _mark(ax, marks, q_of,
              lambda x: slots[int(round(x)) // (per + 1)])
        ax.set_xticks(centres)
        ax.set_xticklabels(labels, fontsize=8.6, rotation=18, ha="right")
        ax.set_ylabel(ylab)
        if args.normalize != "none":
            ax.axhline(0.0, color="#9AA0A6", lw=1.0, ls="--", zorder=0)
        # A second row of labels for the context, spanning its own slots, and a
        # divider between contexts so the two comparisons are visually separate.
        n_per_ctx = len(slots) // max(len(contexts), 1)
        for ci, ctx in enumerate(contexts):
            lo = ci * n_per_ctx * (per + 1)
            hi = lo + (n_per_ctx - 1) * (per + 1) + per - 1
            ax.annotate(f"{ctx}–{ctx} concordance", ((lo + hi) / 2, -0.155),
                        xycoords=("data", "axes fraction"), ha="center",
                        fontsize=10, fontweight="bold")
            if ci:
                ax.axvline(lo - 1.0, color="#D1D5DB", lw=1.0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    else:
        n_row, n_col = len(groups), len(contexts)
        widths = [max(2.6, 1.35 * len(tracks_by_group[g])) for g in groups]
        fig, axes = plt.subplots(
            n_row, n_col, squeeze=False,
            figsize=(max(widths) * n_col + 1.0, 3.5 * n_row + 0.8))
        for ri, g in enumerate(groups):
            for ci, ctx in enumerate(contexts):
                ax = axes[ri][ci]
                tracks = tracks_by_group[g]
                marks, centres, labels = [], [], []
                for ti, track in enumerate(tracks):
                    base = ti * (per + 1)
                    marks += _draw_slot(ax, data[ctx], sets, colors,
                                        data[ctx]["values"][track], base, per)
                    centres.append(base + (per - 1) / 2)
                    labels.append(track)
                _mark(ax, marks, q_of,
                      lambda x, _c=ctx, _t=labels: (
                          _c, _t[int(round(x)) // (per + 1)]))
                ax.set_xticks(centres)
                ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
                ax.set_ylabel(ylab if ci == 0 else "")
                ax.set_title(f"{ctx}–{ctx} concordance · {g.upper()}",
                             fontsize=10.5)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=_BG_COLOR, alpha=0.85,
                             edgecolor="#374151", label="background (all bins)")]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor=colors[s], alpha=0.85,
                              edgecolor="#374151", label=s) for s in sets]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.99),
               ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(args.title or "Signal at each bin set", fontsize=12.5, y=1.04)
    fig.text(0.5, -0.055 if args.layout == "single" else -0.015,
             "boxes: median, quartiles, 1.5 IQR whiskers; outliers hidden.  "
             "Stars: Mann-Whitney of the set against the rest of the bins, "
             "BH-adjusted over the figure — anticonservative, since bins inside "
             "a replication domain are not independent.",
             ha="center", fontsize=7.8, color="#4B5563")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _any_set(d, sets):
    """Membership of ANY named set, so 'background' excludes all of them."""
    m = np.zeros(len(d["bins"]), bool)
    for s in sets:
        if s in d["sets"]:
            m |= d["sets"][s]
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background", action="append", required=True,
                    metavar="CONTEXT=BED",
                    help="The universe for one context, normally its "
                         "bin_scores_<ctx>.all.bed.  Repeatable.")
    ap.add_argument("--bin-set", action="append", required=True,
                    metavar="CONTEXT:LABEL=BED",
                    help="A bin set to compare against that context's "
                         "background, e.g. "
                         "GCH:invariant=out/bin_scores_gch.stable.bed.  "
                         "Repeatable.")
    ap.add_argument("--signal-track", action="append", required=True,
                    metavar="LABEL=BW|BEDGRAPH",
                    help="A continuous track (bigWig or bedGraph).  Repeatable.")
    ap.add_argument("--track-group", action="append", default=None,
                    metavar="GROUP=t1,t2",
                    help="Group the tracks into panel rows explicitly.  Without "
                         "it an '_early' / '_late' token in the name is used.")
    ap.add_argument("--layout", choices=["grid", "single"], default="grid",
                    help="'grid' (default) is one panel per (timing group x "
                         "context), each with its own y-axis, which is the "
                         "faithful default because two tracks are rarely on the "
                         "same scale.  'single' puts every track in ONE panel, "
                         "ordered context-major so each context's early/late "
                         "pair sits together; it needs --normalize to make the "
                         "shared axis mean anything.")
    ap.add_argument("--normalize", choices=["none", "log2-median", "zscore"],
                    default="none",
                    help="Rescale each track before plotting.  'log2-median' is "
                         "the conventional enrichment axis -- log2 of the signal "
                         "over the genome-wide median, so the background sits at "
                         "0 and each set reads as fold-enrichment over the "
                         "genome average; it needs strictly positive signal.  "
                         "'zscore' works whatever the sign.  The stats TSV always "
                         "reports the RAW units.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--plot-format", choices=["pdf", "png", "svg"],
                    default="pdf")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backgrounds = dict(_parse_pair(s, "--background") for s in args.background)
    tracks = dict(_parse_pair(s, "--signal-track") for s in args.signal_track)
    bin_sets = {}
    for spec in args.bin_set:
        ctx, label, path = _parse_labelled(spec, "--bin-set")
        if ctx not in backgrounds:
            sys.exit(f"--bin-set {ctx}:{label} has no --background {ctx}=<bed>.")
        bin_sets[(ctx, label)] = path
    rules = None
    if args.track_group:
        rules = {}
        for spec in args.track_group:
            grp, names = _parse_pair(spec, "--track-group")
            for t in names.split(","):
                if t.strip():
                    rules[t.strip()] = grp

    if args.layout == "single" and args.normalize == "none" and len(tracks) > 1:
        logger.warning(
            "--layout single with --normalize none puts every track on ONE raw "
            "axis.  That is only honest if the tracks share a scale; an early "
            "and a late Repli-seq track do not, and the panel would then show "
            "the difference between the two assays rather than between the bin "
            "sets.  --normalize log2-median is the usual fix.")
    mbe = _enrich_mod()
    data = gather(backgrounds, bin_sets, tracks, mbe, args.normalize)
    stats = stats_table(data, rules)
    stats.to_csv(out_dir / "bin_set_signal_stats.tsv", sep="\t", index=False)
    path = out_dir / f"bin_set_signal_summary.{args.plot_format}"
    figure(data, stats, path, args)
    for _, r in stats.sort_values(["context", "bin_set", "track"]).iterrows():
        logger.info(f"{r['context']} · {r['bin_set']} · {r['track']}: "
                    f"median {r['median_set']:+.4g} vs background "
                    f"{r['median_background']:+.4g}  "
                    f"(mean diff {r['difference_of_means']:+.4g}, "
                    f"{_stars(r['fdr'])})")
    logger.info(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
