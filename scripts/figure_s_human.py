#!/usr/bin/env python3
"""Supplementary figure: Fig. 5a-d in human cells -- the GM12878 and IMR-90
lines, and a meningioma's tumour cells and macrophages -- so that the long-
range concordance of the mouse cortex can be seen in other cells too.

Each dataset's end posteriors, its cells as its samples sheet lists them,
are put through end_posterior_distance_correlation.py as the cortex's were
for Fig. 5 (reproduce/figure5_cortex.ipynb, Step 3): once with the null
keeping the contacts' near-copies (--distance-ab: a and b), once refusing
them (--null-exclude 1000; --distance: c and d), on the autosomes, the
blacklist left out; each run into its own folder:

    python scripts/end_posterior_distance_correlation.py \\
        --posteriors GCH=pipeline/context_GCH/04_posteriors \\
        --posteriors HCG=pipeline/context_HCG/04_posteriors \\
        --samples SHEET --group-column group \\
        --null-scope cell --null-type link-shuffle [--null-exclude 1000] \\
        --exclude-chrom chrM,chrX,chrY \\
        --exclude-bed GRCh38_unified_blacklist.ENCFF356LFX.bed \\
        --min-sites-gch 1 --min-sites-hcg 1 --delta-arm matched \\
        --cell-center on --seed 42 --plot-format pdf --output-dir RUN

One block per dataset, top to bottom, under its name, its letters running
on from the block above; each panel drawn by figure5_cortex.py's own
functions, as large and where it is in Fig. 5:

  a  the end posteriors' correlation by distance, linked pairs against
     non-linked ones, and their difference, for GCH-GCH, HCG-HCG and
     HCG-GCH (a's reshuffle schematics are left to Fig. 5a)
  b  per 10-Mb region, HCG-HCG against GCH-GCH Δr
  c  per cell, concordance against the negative control
  d  the same by cell line or cell type

    python scripts/figure_s_human.py \\
        --dataset "GM12878 and IMR-90" \\
        --distance output/fig5ad_lines.near_copies_refused \\
        --distance-ab output/fig5ad_lines.near_copies_kept \\
        --dataset "Meningioma" \\
        --distance output/fig5ad_meningioma.near_copies_refused \\
        --distance-ab output/fig5ad_meningioma.near_copies_kept \\
        --samples all_human.samples.fixed_imr90_color.tsv \\
        --output FigureS_human.pdf

--dataset, --distance and --distance-ab go once per dataset, in order; the
same folder twice draws a and b from the run refusing near-copies.  d
draws the cell types in TYPES's order, named as NAMES has them, in a
--samples table's colours where one gives them, else TYPES's.
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


sg = _load("figure_s_gcy")
f5 = sg.f5

#: d's cell types in order -- the lines, then the meningioma's -- and their
#: colours where no --samples table gives one.
TYPES = {"GM12878": "#7b4fa0", "IMR90": "#d9822b", "tumor1": "#b2443a",
         "tumor2": "#e0a33e", "monocyte": "#2a9d8f"}
#: How d names them; the sheets' monocytes are the meningioma's macrophages.
NAMES = {"IMR90": "IMR-90", "tumor1": "tumor 1", "tumor2": "tumor 2",
         "monocyte": "macrophage"}
#: Above each block, room in inches for its name over a's letter and
#: titles.
HEAD = 0.18
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def read_datasets(args) -> list:
    """[(name, tables)] in order, each dataset's tables as figure_s_gcy
    reads them; or stop, naming what is missing or unpaired."""
    names, dd, ab = args.dataset, args.distance, args.distance_ab
    if not names or not len(names) == len(dd) == len(ab):
        sys.exit("give --dataset, --distance and --distance-ab once per "
                 f"dataset: {len(names)}, {len(dd)} and {len(ab)} given")
    return [(n, sg.read_tables(argparse.Namespace(distance=d, distance_ab=a)))
            for n, d, a in zip(names, dd, ab)]


def type_colors(args, groups) -> tuple:
    """d's cell types, TYPES's first, then any others, and their colours:
    a --samples table's (the first of --color-column it has) over TYPES's."""
    colors = dict(TYPES)
    cols = [c.strip() for c in args.color_column.split(",") if c.strip()]
    for path in args.samples:
        s = pd.read_csv(path, sep="\t")
        col = next((c for c in cols if c in s.columns), None)
        if col and args.group_column in s.columns:
            colors.update(s.dropna(subset=[col])
                          .groupby(args.group_column)[col].first().to_dict())
    groups = set(groups)
    order = [g for g in TYPES if g in groups] + sorted(groups - set(TYPES))
    return order, colors


def named(d: dict, order, colors) -> tuple:
    """The tables, the order and the colours with the cell types named as
    NAMES has them."""
    pc = d["percell"].assign(group=d["percell"]["group"].replace(NAMES))
    return (dict(d, percell=pc), [NAMES.get(g, g) for g in order],
            {NAMES.get(g, g): c for g, c in colors.items()})


def build_figure(blocks):
    """``blocks``: [(name, tables, order, colours)], drawn top to bottom."""
    plt = f5._style()
    above, span, hspace = sg.ad_inches()
    per = HEAD + above + span + sg.FOOT
    H = float(np.ceil(len(blocks) * per * 4) / 4)
    fig = plt.figure(figsize=(7.2, H))
    letters = []
    for k, (name, d, order, colors) in enumerate(blocks):
        # the block's axes, in inches down from the page's top
        top = k * per + HEAD + above
        gs = sg.ad_grid(fig, 1 - top / H, 1 - (top + span) / H, hspace)
        ad = sg.draw_ad(fig, gs, d, order, colors, schematics=False)
        letters += [(LETTERS[4 * k + i], ax, dx)
                    for i, (_, ax, dx) in enumerate(ad)]
        # its name above a's letter, where the letter is
        x = ad[0][1].get_position().x0 - ad[0][2]
        fig.text(x, 1 - (k * per + HEAD) / H, name, fontsize=7,
                 fontweight="bold", ha="left", va="bottom")
    sg.put_letters(fig, letters, H)
    return fig


def figure_from_args(args):
    blocks = []
    for name, d in read_datasets(args):
        order, colors = type_colors(args, d["percell"]["group"].dropna())
        blocks.append((name, *named(d, order, colors)))
    return build_figure(blocks)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", action="append", default=[],
                    metavar="NAME",
                    help="A dataset's name, as its block's heading; once "
                         "per dataset, in order.")
    ap.add_argument("--distance", action="append", default=[],
                    metavar="DIR",
                    help="The dataset's end_posterior_distance_correlation"
                         ".py run, its null refusing the contacts' near-"
                         "copies (--null-exclude 1000): c and d, as Fig. "
                         "5's.")
    ap.add_argument("--distance-ab", action="append", default=[],
                    metavar="DIR",
                    help="The same with near-copies kept: a and b, as Fig. "
                         "5's.")
    ap.add_argument("--samples", action="append", default=[],
                    help="A samples table giving cell types' colours "
                         "(--group-column, --color-column); repeatable.")
    ap.add_argument("--group-column", default="group")
    ap.add_argument("--color-column", default="color,cell_type_joint_color",
                    help="The colour column: the first of these a table "
                         "has (default %(default)s).")
    ap.add_argument("--output", required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    fig = figure_from_args(args)
    import matplotlib.pyplot as plt
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
