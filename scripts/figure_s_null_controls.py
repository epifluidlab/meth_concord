#!/usr/bin/env python3
"""Supplementary figure: Fig. 5a-c under each of the three non-linked
controls of Supplementary Fig. 9c, so that their conclusion can be seen not
to rest on one of them.

One row per control, from its end_posterior_distance_correlation.py run:

  a  the control (as Supplementary Fig. 9c)
  b  Δr by distance, linked less non-linked, GCH-GCH, HCG-HCG and HCG-GCH,
     with 95% intervals and rings where q < 0.05 (Fig. 5a, bottom)
  c  per 10-Mb region, HCG-HCG against GCH-GCH Δr (Fig. 5b)
  d  per cell, Δr against the control draw (Fig. 5c)

    python scripts/figure_s_null_controls.py \\
        --run 1=end_posterior_dist.null_link_shuffled_match_per_cell.cell_center.excl1kb \\
        --run 2=null_link_shuffled_match_per_cell.cell_center \\
        --run 3=end_posterior_dist.null_locus_matched_pooled_type_batch.cell_center \\
        --output FigureS_null_controls.pdf

``--run K=DIR``: control K (1, 2 or 3 of Supplementary Fig. 9c) and the run
made with it; a control without a run is drawn with its row empty.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


f5 = _load("figure5_cortex")
s9 = _load("figure_s9_concordance_model")

TABLES = ("end_posterior_dist_all.tsv", "end_posterior_regions.tsv",
          "end_posterior_regions_summary.tsv", "end_posterior_percell.tsv")


def read_run(d) -> dict:
    """A run's four tables, or stop naming the missing ones."""
    d = Path(d)
    miss = [str(d / f) for f in TABLES if not (d / f).exists()]
    if miss:
        sys.exit("missing: " + ", ".join(miss))
    return {"dist": pd.read_csv(d / TABLES[0], sep="\t"),
            "regions": pd.read_csv(d / TABLES[1], sep="\t"),
            "summary": pd.read_csv(d / TABLES[2], sep="\t").iloc[0],
            "percell": pd.read_csv(d / TABLES[3], sep="\t")}


def panel_delta(ax, dist: pd.DataFrame, legend: bool):
    """Δr by distance for the three measures on one axes, each over its
    matched_bins, with its interval and rings where q < 0.05."""
    x = dist["center_bp"].to_numpy(float)
    lo_x = []
    for k, title, col in f5.A_MEASURES:
        keep = f5.matched_bins(dist, k)
        if not keep.any():
            continue
        d = np.where(keep, dist[f"{k}_matched_r"] - dist[f"{k}_null_r"],
                     np.nan)
        ax.fill_between(x, np.where(keep, dist[f"{k}_delta_lo"], np.nan),
                        np.where(keep, dist[f"{k}_delta_hi"], np.nan),
                        color=col, alpha=0.18, lw=0)
        ax.plot(x, d, "-", color=col, lw=0.8,
                label=title.split(":")[0])
        q = dist[f"{k}_delta_q"].to_numpy(float)
        sig = keep & np.isfinite(q) & (q < 0.05)
        ax.plot(x[sig], d[sig], "o", ms=2.2, mfc="none", mec=col, mew=0.4)
        lo_x.append(x[keep].min())
    ax.axhline(0, color="0.6", lw=0.4)
    ax.set_xscale("log")
    ax.set_xlim(90, 1.3e7)
    if lo_x and min(lo_x) < f5.ONE_MOLECULE_BP:
        ax.axvspan(90, f5.ONE_MOLECULE_BP, color="0.93", lw=0, zorder=0)
    ax.set_ylabel("Δr, linked − non-linked")
    ax.set_xlabel("distance between the two ends (bp)")
    if legend:
        ax.legend(frameon=False, fontsize=4.6, loc="upper left",
                  handlelength=1.4, borderaxespad=0.2)


def build_figure(runs: dict):
    """``runs``: {control 0-2: tables or None}."""
    plt = f5._style()
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(7.2, 7.8))
    gs = GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.42,
                  width_ratios=[1.3, 1.35, 1.0, 1.0], left=0.06, right=0.985,
                  top=0.95, bottom=0.06)
    first, deltas = {}, []
    for k in range(3):
        a = fig.add_subplot(gs[k, 0])
        a.set_xlim(0, 10)
        a.set_ylim(0, 10)
        a.axis("off")
        s9.draw_control(a, k)
        first.setdefault("a", a)
        t = runs.get(k)
        if t is None:
            a2 = fig.add_subplot(gs[k, 1:])
            a2.axis("off")
            a2.text(0.5, 0.5, "run pending", ha="center", va="center",
                    fontsize=7, color="0.5", transform=a2.transAxes)
            continue
        ab = fig.add_subplot(gs[k, 1])
        panel_delta(ab, t["dist"], legend=(k == min(runs)))
        deltas.append(ab)
        first.setdefault("b", ab)
        ac = fig.add_subplot(gs[k, 2])
        f5.panel_b(ac, t["regions"], t["summary"])
        first.setdefault("c", ac)
        gd = GridSpecFromSubplotSpec(2, 1, gs[k, 3], hspace=0.95)
        axd = [fig.add_subplot(gd[i]) for i in range(2)]
        f5.panel_c(axd, t["percell"])
        first.setdefault("d", axd[0])
    # one scale for every control's Δr, so their sizes compare
    if deltas:
        lo = min(a.get_ylim()[0] for a in deltas)
        hi = max(a.get_ylim()[1] for a in deltas)
        for a in deltas:
            a.set_ylim(lo, hi)
    fig.canvas.draw()
    for lab, ax in first.items():
        bb = ax.get_position()
        fig.text(bb.x0 - (0.0 if lab == "a" else 0.045), 0.975, lab,
                 fontsize=8, fontweight="bold", va="bottom", ha="left")
    return fig


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="K=DIR",
                    help="Control K (1, 2 or 3 of Supplementary Fig. 9c) "
                         "and its end_posterior_distance_correlation.py "
                         "output directory.")
    ap.add_argument("--output", required=True)
    return ap.parse_args(argv)


def runs_of(specs) -> dict:
    runs = {}
    for spec in specs:
        k, _, d = str(spec).partition("=")
        if k.strip() not in ("1", "2", "3") or not d.strip():
            sys.exit(f"--run {spec}: K=DIR, K one of 1, 2, 3")
        runs[int(k) - 1] = read_run(d.strip())
    if not runs:
        sys.exit("give at least one --run")
    return runs


def main(argv=None) -> int:
    args = parse_args(argv)
    fig = build_figure(runs_of(args.run))
    import matplotlib.pyplot as plt
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
