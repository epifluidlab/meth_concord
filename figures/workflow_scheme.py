#!/usr/bin/env python3
"""Nature-Methods-style *workflow* schematic for the long-range epigenetic
concordance analysis — the four tools and how data flows between them.

Complements ``method_schematic.py`` (which draws the per-cell *estimator* math):
this figure is the software/data-flow view used as the overview panel (Fig 5a).

Flow
----
    single-cell NOMe-HiC BAMs + .2bit reference (+ dbSNP)
        ├── site_distance_correlation   (site-level phi vs distance)        -> F5B
        └── nome_hic_concordance        (7-step, per GCH/HCG context)
                 └── cells x bins signed-Δ concordance matrix (+ mean access.)
                        ├── celltype_marker_bins  (marker heatmap + control) -> F5C
                        └── nome_hic_compare       (regress out 1D+3D)       -> F5D

Emits a vector PDF + a 300-dpi PNG into figures/.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

# ---- palette (shared with method_schematic.py) ---------------------------
OPEN   = "#E76F51"   # accessible / open  · nome_hic_compare accent
CLOSED = "#264653"
DNA    = "#9AA0A6"
PANEL  = "#FFFFFF"
CARD   = "#F6F7F9"
EDGE   = "#D5D8DF"
INK    = "#23262B"
MUT    = "#6B7280"
CTA    = "#2A9D8F"   # cell type A  · nome_hic_concordance (core)
CTB    = "#8E5BB5"   # cell type B  · celltype_marker_bins
ACC    = "#3D7DCA"   # accent / arrows · site_distance_correlation

# per-tool header colour
C_CORE = CTA          # nome_hic_concordance
C_SITE = ACC          # site_distance_correlation
C_MARK = CTB          # celltype_marker_bins
C_CMP  = OPEN         # nome_hic_compare

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "text.color": INK, "axes.edgecolor": INK,
})

FIG_W, FIG_H = 16.6, 10.7
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 166)
ax.set_ylim(0, 107)
ax.axis("off")


# ---- primitives ----------------------------------------------------------
def card(x, y, w, h, fc=CARD, ec=EDGE, lw=1.2, z=1, r=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, zorder=z))


def tool(x, y, w, h, color, name, tag=None):
    """A tool box: coloured edge + bold coloured title + rule (+ Fig tag)."""
    card(x, y, w, h, fc=PANEL, ec=color, lw=1.8, z=3)
    ax.text(x + 3.0, y + h - 3.4, name, fontsize=11.2, fontweight="bold",
            color=color, va="top", ha="left", zorder=6, family="monospace")
    ax.plot([x + 3.0, x + w - 3.0], [y + h - 6.6, y + h - 6.6],
            color=color, lw=1.4, zorder=6)
    if tag:
        tw, th = 11.0, 5.0
        ax.add_patch(FancyBboxPatch((x + w - tw - 2.2, y + 2.0), tw, th,
                     boxstyle="round,pad=0.0,rounding_size=1.1",
                     fc=color, ec="none", zorder=7))
        ax.text(x + w - tw / 2 - 2.2, y + 2.0 + th / 2, tag, fontsize=8.4,
                fontweight="bold", color="white", ha="center", va="center",
                zorder=8)


def flow(x1, y1, x2, y2, rad=0.0, color=INK, lw=2.2, mut=15):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                 mutation_scale=mut, lw=lw, color=color, zorder=5))


def stage(y, label):
    ax.text(5.2, y, label.upper(), rotation=90, va="center", ha="center",
            fontsize=10.5, fontweight="bold", color=MUT, family="monospace")


def lolly(cx, cy, state, r=0.8, stem=2.0):
    ax.plot([cx, cx], [cy, cy + stem], color=DNA, lw=1.1, zorder=3)
    fc = OPEN if state else "white"
    ec = OPEN if state else CLOSED
    ax.add_patch(Circle((cx, cy + stem + r), r, fc=fc, ec=ec, lw=1.2, zorder=4))


def step_chip(x, y, w, h, num, lines, color):
    card(x, y, w, h, fc=CARD, ec=EDGE, lw=1.0, z=4, r=1.2)
    ax.add_patch(Circle((x + 3.4, y + h - 3.4), 2.05, fc=color, ec="none",
                        zorder=6))
    ax.text(x + 3.4, y + h - 3.4, str(num), fontsize=8.0, fontweight="bold",
            color="white", ha="center", va="center", zorder=7)
    ax.text(x + w / 2, y + h - 8.2, lines, fontsize=7.7, color=INK,
            ha="center", va="top", zorder=6, linespacing=1.28)


# =========================================================================
# Title
# =========================================================================
ax.text(84, 104.4, "Long-range epigenetic concordance — analysis workflow",
        fontsize=16.5, fontweight="bold", ha="center", va="center")
ax.text(84, 100.4,
        "single-cell NOMe-HiC  ·  GCH = GpC accessibility  ·  HCG = CpG methylation  "
        "·  Δ$_{signed}$ = (co-open − co-closed): long-range contact − short-range baseline",
        fontsize=9.2, color=MUT, ha="center", va="center", style="italic")

# =========================================================================
# INPUT bar
# =========================================================================
stage(92, "input")
ix, iy, iw, ih = 14, 89, 138, 7.2
card(ix, iy, iw, ih, fc=CARD, ec=EDGE, lw=1.3, z=2)
# contact glyph on the left of the bar
gx = ix + 8
for lx0, lx1, states in [(gx, gx + 10, [1, 1, 0]), (gx + 16, gx + 26, [1, 0, 1])]:
    ax.plot([lx0, lx1], [iy + 2.6, iy + 2.6], color=DNA, lw=2.4,
            solid_capstyle="round", zorder=3)
    for xx, s in zip(np.linspace(lx0 + 1.4, lx1 - 1.4, len(states)), states):
        lolly(xx, iy + 2.6, s)
ax.add_patch(FancyArrowPatch((gx + 5, iy + 3.0), (gx + 21, iy + 3.0),
             connectionstyle="arc3,rad=-0.55", arrowstyle="-",
             mutation_scale=1, lw=1.4, color=ACC, zorder=2))
ax.text(gx + 13, iy + 6.6, "contact", ha="center", va="center", fontsize=7.4,
        color=ACC, style="italic", zorder=4)
ax.text(gx + 34, iy + ih / 2,
        "single-cell NOMe-HiC BAMs   ·   reference genome (.2bit)   ·   dbSNP (optional)",
        fontsize=10.4, va="center", ha="left", fontweight="bold")

# =========================================================================
# MEASURE stage — two parallel tracks branching from the input
# =========================================================================
stage(70.5, "measure")

# ---- site_distance_correlation (left, site-level) -----------------------
sx, sy, sw, sh = 14, 53, 40, 33
tool(sx, sy, sw, sh, C_SITE, "site_distance_\ncorrelation", tag="Fig 5B")
ax.text(sx + sw / 2, sy + sh - 9.4,
        "individual-cytosine phi (0/1)\nvs. genomic distance",
        fontsize=8.2, color=INK, ha="center", va="top", linespacing=1.28)
# mini distance-decay plot: linked above a distance-matched null (raised so the
# axis label clears the Fig-tag corner)
px0, px1 = sx + 7, sx + sw - 5
py0, py1 = sy + 10.5, sy + 19.5
ax.plot([px0, px1], [py0, py0], color=INK, lw=1.0)
ax.plot([px0, px0], [py0, py1], color=INK, lw=1.0)
xx = np.linspace(0, 1, 60)
linked = py0 + (py1 - py0) * (0.92 * np.exp(-2.1 * xx) + 0.06)
null = py0 + (py1 - py0) * (0.42 * np.exp(-2.6 * xx) + 0.03)
ax.fill_between(px0 + xx * (px1 - px0), null, linked, color=C_SITE, alpha=0.14,
                zorder=2)
ax.plot(px0 + xx * (px1 - px0), linked, color=C_SITE, lw=2.0, zorder=3)
ax.plot(px0 + xx * (px1 - px0), null, color=MUT, lw=1.6, ls=(0, (3, 2)),
        zorder=3)
ax.text(px1, linked[-1] + 1.0, "linked", fontsize=6.9, color=C_SITE,
        ha="right", va="bottom", fontweight="bold")
ax.text(px0 + 2.2, null[0] + 2.2, "matched null", fontsize=6.9, color=MUT,
        ha="left", va="bottom")
ax.text((px0 + px1) / 2, py0 - 2.0, "distance  (log)", fontsize=6.8,
        color=MUT, ha="center", va="top")
ax.text(px0 - 1.4, (py0 + py1) / 2, "concordance", fontsize=6.8, color=MUT,
        rotation=90, ha="right", va="center")
ax.text(sx + 3.4, sy + 3.6, "GCH·GCH  HCG·HCG  HCG·GCH\nΔ = linked − null, bootstrap",
        fontsize=6.9, color=MUT, ha="left", va="center", style="italic",
        linespacing=1.3)

# =========================================================================
# nome_hic_concordance (right, the 7-step core) — per GCH/HCG context
# =========================================================================
cx, cy, cw, ch = 60, 53, 92, 33
tool(cx, cy, cw, ch, C_CORE, "nome_hic_concordance", tag=None)
ax.text(cx + cw - 4, cy + ch - 4.0,
        "run per context:  GCH  +  HCG", fontsize=8.4, color=C_CORE,
        ha="right", va="top", style="italic", fontweight="bold")
steps = [
    ("GCH / HCG\ncontexts\n(excl. GCG,\ndbSNP)"),
    ("parse HQ\ndistal pairs\n(mapQ>30,\n≥20 kb)"),
    ("β-binomial\nmixture\n(EM)"),
    ("per-end\nposterior\nP(open|k,n)"),
    ("signed Δ\nper (cell,\nbin)"),
    ("empirical-\nBayes\nshrinkage"),
    ("norm · RWR\nimpute ·\nSVD → UMAP"),
]
n = len(steps)
pad = 4.0
gap = 1.6
inner = cw - 2 * pad - (n - 1) * gap
scw = inner / n
sch = 20.5
sy0 = cy + 4.0
for i, txt in enumerate(steps):
    xchip = cx + pad + i * (scw + gap)
    step_chip(xchip, sy0, scw, sch, i + 1, txt, C_CORE)
    if i < n - 1:
        flow(xchip + scw + 0.1, sy0 + sch / 2, xchip + scw + gap - 0.1,
             sy0 + sch / 2, color=C_CORE, lw=1.3, mut=8)

# =========================================================================
# concordance matrix — the shared data object
# =========================================================================
mx, my, mw, mh = 81, 40, 46, 10.5
card(mx, my, mw, mh, fc="#FBFCFD", ec=INK, lw=1.4, z=4)
# mini cells x bins heatmap glyph
rng = np.random.default_rng(2)
blk = np.vstack([rng.normal(0.6, 0.3, (5, 16)), rng.normal(-0.6, 0.3, (5, 16))])
blk[:, ::4] += rng.normal(0, 0.5, (10, 4))
hx0, hy0, hw, hh = mx + 3.2, my + 2.2, 15, 6.2
ax.imshow(blk, cmap="RdBu_r", vmin=-1.3, vmax=1.3, aspect="auto",
          extent=[hx0, hx0 + hw, hy0, hy0 + hh], zorder=6)
ax.add_patch(Rectangle((hx0, hy0), hw, hh, fc="none", ec=INK, lw=0.9, zorder=7))
ax.text(hx0 + hw / 2, hy0 - 1.4, "bins", fontsize=6.6, color=MUT, ha="center",
        va="top", zorder=7)
ax.text(hx0 - 1.2, hy0 + hh / 2, "cells", fontsize=6.6, color=MUT, rotation=90,
        ha="right", va="center", zorder=7)
ax.text(mx + 21, my + mh / 2 + 1.3,
        "cells × bins  signed-Δ\nconcordance matrix", fontsize=8.6,
        fontweight="bold", color=INK, ha="left", va="center", zorder=7)
ax.text(mx + 21, my + 1.9, "GCH · HCG  (+ mean accessibility)", fontsize=7.4,
        color=MUT, ha="left", va="center", zorder=7)

# =========================================================================
# INTERPRET stage — two tools consuming the matrix
# =========================================================================
stage(22.5, "interpret")

# ---- celltype_marker_bins (F5C) -----------------------------------------
kx, ky, kw, kh = 46, 8, 55, 29
tool(kx, ky, kw, kh, C_MARK, "celltype_marker_bins", tag="Fig 5C")
ax.text(kx + 3.4, ky + kh - 9.0,
        "bins with high between-cell-type\n"
        "variance, low within-type batch\n"
        "variance  →  marker heatmap",
        fontsize=8.0, color=INK, ha="left", va="top", linespacing=1.3)
# mini heatmap with a cell-type colour strip + tiny dendrogram (upper-right so
# it clears the Fig-tag corner; kept below the title rule)
bx0, by0, bw, bh = kx + 31, ky + 8.5, 20, 11.0
ct_cols = [CTA, CTB, OPEN, "#4C6EF5"]
strip = np.array([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3]])
ax.imshow(strip, cmap=matplotlib.colors.ListedColormap(ct_cols), aspect="auto",
          extent=[bx0, bx0 + bw, by0 + bh + 0.3, by0 + bh + 1.8], zorder=6)
mk = np.vstack([
    np.repeat([[1.2, 1.2, 1.2, -1, -1, -1, .2, .2, -.8, -.8]], 4, 0),
    np.repeat([[-1, -1, -1, 1.2, 1.2, 1.2, -.6, -.6, .3, .3]], 4, 0),
    np.repeat([[.1, .1, .1, -.7, -.7, -.7, 1.3, 1.3, -1, -1]], 3, 0),
]) + rng.normal(0, 0.16, (11, 10))
ax.imshow(mk, cmap="PuOr_r", vmin=-1.4, vmax=1.4, aspect="auto",
          extent=[bx0, bx0 + bw, by0, by0 + bh], zorder=6)
ax.add_patch(Rectangle((bx0, by0), bw, bh, fc="none", ec=INK, lw=0.8, zorder=7))
# tiny column dendrogram above the strip
dy = by0 + bh + 2.2
for a, b, hgt in [(2, 5, 2.0), (7.5, 9, 1.4)]:
    xa = bx0 + a / 10 * bw
    xb = bx0 + b / 10 * bw
    ax.plot([xa, xa, xb, xb], [dy, dy + hgt, dy + hgt, dy], color="0.45",
            lw=0.8, zorder=7)
ax.text(bx0 + bw / 2 - 3.0, by0 - 1.3, "cells", fontsize=6.4, color=MUT,
        ha="center", va="top", zorder=7)
ax.text(kx + 3.4, ky + 3.0,
        "control: non-selected bins\nstill separate types (perm. p)",
        fontsize=7.0, color=MUT, ha="left", va="bottom", linespacing=1.25,
        style="italic")

# ---- nome_hic_compare (F5D) ---------------------------------------------
qx, qy, qw, qh = 108, 8, 44, 29
tool(qx, qy, qw, qh, C_CMP, "nome_hic_compare", tag="Fig 5D")
ax.text(qx + qw / 2, qy + qh - 9.0,
        "regress Δ on 1D mean accessibility\n+ 3D scHi-C compartments",
        fontsize=8.0, color=INK, ha="center", va="top", linespacing=1.3)
# residual equation
ax.text(qx + qw / 2, qy + 15.4,
        r"$\Delta \;=\; \beta_{1D}\,\mathrm{access} + \beta_{3D}\,\mathrm{comp} + \mathbf{residual}$",
        fontsize=7.8, color=INK, ha="center", va="center")
ax.text(qx + qw / 2, qy + 11.2,
        "does the residual still cluster?", fontsize=7.6, color=MUT,
        ha="center", va="center", style="italic")
# residual UMAP still separates two cell types (centred; tag corner kept clear)
ux, uy = qx + qw / 2 + 1, qy + 5.4
cA = np.random.default_rng(21).normal([ux - 3.4, uy + 0.6], [1.05, 1.4], (22, 2))
cB = np.random.default_rng(22).normal([ux + 3.4, uy - 0.6], [1.05, 1.4], (22, 2))
ax.scatter(cA[:, 0], cA[:, 1], s=7, c=CTA, edgecolors="none", zorder=6)
ax.scatter(cB[:, 0], cB[:, 1], s=7, c=CTB, edgecolors="none", zorder=6)
ax.text(qx + 4.0, qy + 3.0, "residual UMAP →\nnot reducible to\n1D or 3D alone",
        fontsize=7.3, color=C_CMP, ha="left", va="center", fontweight="bold",
        linespacing=1.22)

# =========================================================================
# flow arrows between stages
# =========================================================================
# input -> the two measure tracks
flow(34, iy, 34, sy + sh, color=INK, lw=2.0)                 # -> site_distance
flow(106, iy, 106, cy + ch, color=INK, lw=2.0)              # -> concordance core
# concordance core -> matrix
flow(106, cy, 104, my + mh, color=C_CORE, lw=2.2)
# matrix -> the two interpret tools
flow(mx + 6, my, kx + kw / 2, ky + kh, rad=0.12, color=INK, lw=2.0)   # -> marker_bins
flow(mx + mw - 6, my, qx + qw / 2, qy + qh, rad=-0.12, color=INK, lw=2.0)  # -> compare

fig.savefig("figures/workflow_scheme.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig("figures/workflow_scheme.pdf", bbox_inches="tight",
            facecolor="white")
print("wrote figures/workflow_scheme.png and .pdf")
