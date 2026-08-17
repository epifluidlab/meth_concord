#!/usr/bin/env python3
"""One panel comparing several BIN SETS against several signal tracks.

Built for the result it is named after: the invariant (``stable``) bins sit in
EARLY-replicating sequence and the cell-type-variable (``selected``) bins in
LATE, in both contexts.  That claim is a 2x2 interaction -- (bin set) x (early
vs late) -- repeated per context, and the figure is laid out to make the
interaction the thing the eye lands on rather than something the reader has to
compute from four separate enrichment PDFs.

    python scripts/bin_set_signal_summary.py \\
        --enrichment GCH:variable=out/enrich_gch_selected/gch_signal_enrichment.tsv \\
        --enrichment GCH:invariant=out/enrich_gch_stable/gch_signal_enrichment.tsv \\
        --enrichment HCG:variable=out/enrich_hcg_selected/hcg_signal_enrichment.tsv \\
        --enrichment HCG:invariant=out/enrich_hcg_stable/hcg_signal_enrichment.tsv \\
        --output-dir out/11_replication_summary

WHY AUC AND NOT THE DIFFERENCE OF MEANS
---------------------------------------
``difference`` is in the track's own units, so it cannot be compared between an
early track and a late one, or between two samples normalised differently -- and
"enriched in early, depleted in late" read off raw differences could be partly a
scale artifact.  The AUC (P(a random foreground bin exceeds a random background
one), 0.5 = nothing) is the effect size belonging to the Mann-Whitney p-value
that marker_bin_enrichment already reports, is scale-free, and so is comparable
down a column of tracks.  Its interval is the block bootstrap that script
computes over megabase blocks, because replication timing is organised into
domains far larger than a bin and a per-bin interval would be far too narrow.

LAYOUT
------
Left, the per-track effects: one row per track, grouped by early / late, one
colour per bin set, a marker at the AUC with its CI, and a line at 0.5.  A
reversal is then two colours on opposite sides of that line, with the early and
late blocks mirroring each other.

Right, the interaction distilled to one number per (context, bin set): the mean
AUC over early tracks minus the mean over late tracks.  That single contrast is
the claim -- positive for a set that prefers early, negative for one that
prefers late -- and the two sets landing on opposite sides of zero IS the
"reversed" observation, with no eye-balling of six rows required.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("bin_set_signal_summary")

# Colour-blind-safe and consistent with the rest of the figures: the variable
# set warm, the invariant set cool.
_SET_COLOR = {"variable": "#D1495B", "invariant": "#1F6F8B",
              "selected": "#D1495B", "stable": "#1F6F8B"}
_FALLBACK = ["#8C5E8A", "#3E8E7E", "#B07C4F", "#5F4BB6"]


def _parse_enrichment(spec: str):
    """'CONTEXT:SET=PATH' → (context, set, path)."""
    if "=" not in spec:
        sys.exit(f"--enrichment expects CONTEXT:SET=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    if ":" not in label:
        sys.exit(f"--enrichment label expects CONTEXT:SET, got {label!r}")
    ctx, bset = label.split(":", 1)
    return ctx.strip(), bset.strip(), path.strip()


def load(specs):
    """Every enrichment TSV as one long frame."""
    frames = []
    for spec in specs:
        ctx, bset, path = _parse_enrichment(spec)
        if not Path(path).exists():
            sys.exit(f"no such enrichment TSV: {path}")
        df = pd.read_csv(path, sep="\t")
        need = {"annotation", "auc"}
        if not need <= set(df.columns):
            sys.exit(
                f"{path}: expected the columns marker_bin_enrichment writes for "
                f"--signal-track (annotation, auc, auc_lo, auc_hi, fdr); got "
                f"{list(df.columns)}.  Re-run marker_bin_enrichment: the AUC "
                f"and its block-bootstrap interval were added for this figure, "
                f"and an older TSV has only the difference of means, which is "
                f"not comparable between tracks.")
        df["context"] = ctx
        df["bin_set"] = bset
        frames.append(df)
    if not frames:
        sys.exit("give at least one --enrichment CONTEXT:SET=PATH.")
    return pd.concat(frames, ignore_index=True)


def assign_groups(df, rules=None):
    """Label each track early / late / other.

    Auto by an ``_early`` / ``_late`` token in the name, which is how Repli-seq
    tracks are conventionally named, with --track-group as the override for
    anything else.
    """
    if rules:
        mapping = {}
        for spec in rules:
            if "=" not in spec:
                sys.exit(f"--track-group expects GROUP=t1,t2, got {spec!r}")
            grp, names = spec.split("=", 1)
            for t in names.split(","):
                if t.strip():
                    mapping[t.strip()] = grp.strip()
        df["track_group"] = df["annotation"].map(mapping).fillna("other")
    else:
        def auto(name):
            n = str(name).lower()
            if re.search(r"(^|[_\-.])early([_\-.]|$)", n):
                return "early"
            if re.search(r"(^|[_\-.])late([_\-.]|$)", n):
                return "late"
            return "other"
        df["track_group"] = df["annotation"].map(auto)
    return df


def interaction(df, a="early", b="late"):
    """Mean AUC over group ``a`` minus over group ``b``, per (context, set).

    The bootstrap intervals of the individual tracks are not combined into one
    for this contrast: the tracks are replicates of the same measurement on the
    same bins, so their errors are strongly dependent and treating them as
    independent would understate the interval.  The SPREAD ACROSS TRACKS is
    shown instead, which is the honest statement available from summaries --
    an interval on the contrast would need the per-bin values, not these TSVs.
    """
    rows = []
    for (ctx, bset), sub in df.groupby(["context", "bin_set"], sort=False):
        ea = sub.loc[sub["track_group"] == a, "auc"].to_numpy(float)
        la = sub.loc[sub["track_group"] == b, "auc"].to_numpy(float)
        ea, la = ea[np.isfinite(ea)], la[np.isfinite(la)]
        if not ea.size or not la.size:
            continue
        rows.append({
            "context": ctx, "bin_set": bset,
            f"mean_auc_{a}": float(ea.mean()), f"mean_auc_{b}": float(la.mean()),
            "contrast": float(ea.mean() - la.mean()),
            f"n_{a}": int(ea.size), f"n_{b}": int(la.size),
            f"min_auc_{a}": float(ea.min()), f"max_auc_{a}": float(ea.max()),
            f"min_auc_{b}": float(la.min()), f"max_auc_{b}": float(la.max())})
    return pd.DataFrame(rows)


def _stars(q):
    if not np.isfinite(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def figure(df, inter, out_path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    contexts = list(dict.fromkeys(df["context"]))
    sets = list(dict.fromkeys(df["bin_set"]))
    colors = {s: _SET_COLOR.get(s.lower(), _FALLBACK[i % len(_FALLBACK)])
              for i, s in enumerate(sets)}
    groups = [g for g in ("early", "late", "other")
              if (df["track_group"] == g).any()]
    # Rows are tracks, ordered by group then name, shared down the columns so a
    # track sits at the same height in every context.
    order = []
    for g in groups:
        order += sorted(df.loc[df["track_group"] == g, "annotation"].unique())
    ypos = {t: len(order) - 1 - i for i, t in enumerate(order)}

    ncol = len(contexts) + 1
    fig, axes = plt.subplots(
        1, ncol, figsize=(5.0 * len(contexts) + 4.0, 0.42 * len(order) + 3.2),
        gridspec_kw={"width_ratios": [1.0] * len(contexts) + [0.85]})
    axes = np.atleast_1d(axes)

    for k, ctx in enumerate(contexts):
        ax = axes[k]
        sub = df[df["context"] == ctx]
        for si, bset in enumerate(sets):
            d = sub[sub["bin_set"] == bset]
            off = (si - (len(sets) - 1) / 2) * 0.26
            for _, r in d.iterrows():
                if r["annotation"] not in ypos:
                    continue
                y = ypos[r["annotation"]] + off
                lo, hi = r.get("auc_lo", np.nan), r.get("auc_hi", np.nan)
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.plot([lo, hi], [y, y], color=colors[bset], lw=2.0,
                            alpha=0.85, zorder=2, solid_capstyle="round")
                ax.plot([r["auc"]], [y], "o", ms=7, color=colors[bset],
                        zorder=3)
                m = _stars(r.get("fdr", np.nan))
                if m:
                    ax.annotate(m, (max(r["auc"], hi if np.isfinite(hi) else
                                        r["auc"]), y),
                                xytext=(4, 0), textcoords="offset points",
                                va="center", fontsize=8, color=colors[bset])
        ax.axvline(0.5, color="#9AA0A6", lw=1.1, ls="--", zorder=0)
        for g in groups[:-1]:                       # separators between groups
            n_above = sum(1 for t in order
                          if df.loc[df["annotation"] == t,
                                    "track_group"].iloc[0] in
                          groups[:groups.index(g) + 1])
            ax.axhline(len(order) - n_above - 0.5, color="#E5E7EB", lw=1.0)
        ax.set_yticks([ypos[t] for t in order])
        ax.set_yticklabels(order if k == 0 else [""] * len(order), fontsize=8.4)
        ax.set_ylim(-0.8, len(order) - 0.2)
        ax.set_xlabel("AUC: P(foreground bin > background bin)")
        ax.set_title(f"{ctx}–{ctx} concordance", fontsize=11)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for g in groups:
            ts = [t for t in order
                  if df.loc[df["annotation"] == t, "track_group"].iloc[0] == g]
            if ts:
                ax.annotate(g.upper(), (0.015, np.mean([ypos[t] for t in ts])),
                            xycoords=("axes fraction", "data"), fontsize=8.6,
                            color="#6B7280", va="center", rotation=90)

    # The interaction, one number per (context, set).
    ax = axes[-1]
    labels, k = [], 0
    for ctx in contexts:
        for bset in sets:
            row = inter[(inter["context"] == ctx) & (inter["bin_set"] == bset)]
            if row.empty:
                continue
            r = row.iloc[0]
            ax.plot([r["contrast"]], [k], "o", ms=9, color=colors[bset],
                    zorder=3)
            lo = min(r["min_auc_early"] - r["max_auc_late"], r["contrast"])
            hi = max(r["max_auc_early"] - r["min_auc_late"], r["contrast"])
            ax.plot([lo, hi], [k, k], color=colors[bset], lw=2.0, alpha=0.45,
                    zorder=2, solid_capstyle="butt")
            labels.append(f"{ctx} · {bset}")
            k += 1
    ax.axvline(0.0, color="#9AA0A6", lw=1.1, ls="--", zorder=0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(max(len(labels) - 0.3, 0.7), -0.7)      # top-down, as at left
    ax.set_xlabel("mean AUC(early) − mean AUC(late)\n"
                  "> 0 prefers early · < 0 prefers late")
    ax.set_title("the interaction", fontsize=11)
    ax.annotate("bar = range across tracks,\nnot a confidence interval",
                xy=(0.5, -0.24), xycoords="axes fraction", ha="center",
                fontsize=7.6, color="#6B7280")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [plt.Line2D([], [], marker="o", ls="-", lw=2.0, ms=7,
                          color=colors[s_], label=s_) for s_ in sets]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.965),
               ncol=len(sets), frameon=False, fontsize=9)
    fig.suptitle(args.title or "Bin-set preference across signal tracks",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enrichment", action="append", required=True,
                    metavar="CONTEXT:SET=PATH",
                    help="A <ctx>_signal_enrichment.tsv from "
                         "marker_bin_enrichment, labelled by context and bin "
                         "set, e.g. GCH:invariant=out/enrich_gch_stable/"
                         "gch_signal_enrichment.tsv.  Repeatable.")
    ap.add_argument("--track-group", action="append", default=None,
                    metavar="GROUP=t1,t2",
                    help="Group the tracks explicitly, e.g. "
                         "early=Cas129_NPC_rep1_early,F121_NPC_early.  Without "
                         "this, an '_early' / '_late' token in the track name is "
                         "used, which is how Repli-seq tracks are usually named.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--plot-format", choices=["pdf", "png", "svg"],
                    default="pdf")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = assign_groups(load(args.enrichment), args.track_group)
    n_other = int((df["track_group"] == "other").sum())
    if n_other:
        logger.warning(
            f"{n_other} track/set rows fall in neither early nor late and are "
            f"drawn as 'other' but left out of the interaction; name them with "
            f"--track-group if they belong to one: "
            f"{sorted(df.loc[df['track_group'] == 'other', 'annotation'].unique())}")
    inter = interaction(df)
    df.to_csv(out_dir / "bin_set_signal_long.tsv", sep="\t", index=False)
    inter.to_csv(out_dir / "bin_set_signal_interaction.tsv", sep="\t",
                 index=False)
    path = out_dir / f"bin_set_signal_summary.{args.plot_format}"
    figure(df, inter, path, args)
    for _, r in inter.iterrows():
        logger.info(f"{r['context']} · {r['bin_set']}: mean AUC early "
                    f"{r['mean_auc_early']:.3f} vs late {r['mean_auc_late']:.3f}"
                    f"  →  contrast {r['contrast']:+.3f}")
    logger.info(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
