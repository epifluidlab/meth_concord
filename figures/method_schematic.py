#!/usr/bin/env python3
"""Nature-Methods-style schematic of the single-cell long-range GCH/HCG
concordance method.  Emits a vector PDF + high-res PNG.

Panels:
  a  scNOMe-HiC contact — two distal loci + per-end methylation/accessibility
  b  per-end Beta-Binomial posterior  p = P(open | k, n)
  c  U-3 soft co-state probabilities (co-open / co-closed / mixed)
  d  distance stratification: long-range "linked" vs short-range "unlinked"
  e  signed long-range concordance Delta_signed per (cell, bin)
  f  RWR denoising across cells -> UMAP separates cell types
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.stats import beta as betadist

# ---- palette -------------------------------------------------------------
OPEN   = "#E76F51"   # accessible / open
CLOSED = "#264653"   # inaccessible / closed
DNA    = "#9AA0A6"
PANEL  = "#F6F7F9"
EDGE   = "#D5D8DF"
INK    = "#23262B"
MUT    = "#6B7280"
CTA    = "#2A9D8F"   # cell type A
CTB    = "#8E5BB5"   # cell type B
ACC    = "#3D7DCA"   # accent (arrows / brackets)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "text.color": INK, "axes.edgecolor": INK,
})

FIG_W, FIG_H = 16.0, 9.4
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 94)
ax.axis("off")

# panel geometry: 3 cols x 2 rows
PW, PH = 47, 41
XS = [4, 56.5, 109]
YS = [49, 4]   # top row, bottom row


def panel(x, y, letter, title):
    ax.add_patch(FancyBboxPatch((x, y), PW, PH,
                 boxstyle="round,pad=0.0,rounding_size=1.6",
                 fc=PANEL, ec=EDGE, lw=1.2, zorder=1))
    ax.text(x + 1.2, y + PH + 1.3, letter, fontsize=17, fontweight="bold",
            va="bottom", ha="left")
    ax.text(x + 6.0, y + PH - 2.4, title, fontsize=11.0, fontweight="bold",
            va="top", ha="left", color=INK)


def lolly(cx, cy, state, r=0.95, stem=2.6):
    """Methylation/accessibility lollipop: filled=open, hollow=closed."""
    ax.plot([cx, cx], [cy, cy + stem], color=DNA, lw=1.3, zorder=3)
    fc = OPEN if state else "white"
    ec = OPEN if state else CLOSED
    ax.add_patch(Circle((cx, cy + stem + r), r, fc=fc, ec=ec, lw=1.4, zorder=4))


def arc(x1, x2, ybase, h, color=DNA, lw=1.6, ls="-"):
    xm = (x1 + x2) / 2
    p = PathPatch(Path([(x1, ybase), (xm, ybase + h), (x2, ybase)],
                       [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                  fc="none", ec=color, lw=lw, ls=ls, zorder=2)
    ax.add_patch(p)


def flow(x1, y1, x2, y2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                 mutation_scale=15, lw=2.0, color=ACC, zorder=6))


# =====================================================================
# Panel a — scNOMe-HiC contact
# =====================================================================
ax0, ay0 = XS[0], YS[0]
panel(ax0, ay0, "a", "Single-cell NOMe-HiC contact")
cy = ay0 + 13
# two loci (DNA segments)
liA = (ax0 + 5, ax0 + 19)
liB = (ax0 + 27, ax0 + 41)
for (lx0, lx1) in (liA, liB):
    ax.plot([lx0, lx1], [cy, cy], color=DNA, lw=3.0, solid_capstyle="round", zorder=2)
# methylation states on each locus (filled=accessible/methylated)
np.random.seed(3)
for lx0, lx1, states in [
        (liA[0], liA[1], [1, 1, 0, 1]),
        (liB[0], liB[1], [1, 0, 1, 1])]:
    xs = np.linspace(lx0 + 2, lx1 - 2, len(states))
    for xx, s in zip(xs, states):
        lolly(xx, cy, s)
# contact arc joining the two loci
arc(np.mean(liA), np.mean(liB), cy, 17, color=ACC, lw=2.0)
ax.text((np.mean(liA) + np.mean(liB)) / 2, cy + 19.5, "chromatin contact",
        ha="center", va="bottom", fontsize=9.2, color=ACC, style="italic")
ax.text(np.mean(liA), cy - 4.2, "locus $i$", ha="center", fontsize=9.5)
ax.text(np.mean(liB), cy - 4.2, "locus $j$", ha="center", fontsize=9.5)
# k / n readout
ax.text(np.mean(liA), cy - 8.0, r"$k_i/n_i$", ha="center", fontsize=9.5, color=MUT)
ax.text(np.mean(liB), cy - 8.0, r"$k_j/n_j$", ha="center", fontsize=9.5, color=MUT)
# legend
lolly(ax0 + 7, ay0 + 31.5, 1); lolly(ax0 + 22, ay0 + 31.5, 0)
ax.text(ax0 + 9.0, ay0 + 35.0, "accessible / methylated", fontsize=8.2, va="center")
ax.text(ax0 + 24.0, ay0 + 35.0, "closed / unmethylated", fontsize=8.2, va="center")
ax.text(ax0 + PW/2, ay0 + 2.4, "GCH (accessibility)  ·  HCG (CpG methylation)",
        ha="center", fontsize=8.6, color=MUT)

# =====================================================================
# Panel b — per-end Beta-Binomial posterior
# =====================================================================
bx0, by0 = XS[1], YS[0]
panel(bx0, by0, "b", "Per-end state posterior")
# mini-axes region inside the panel
gx0, gx1 = bx0 + 7, bx0 + 41
gy0, gy1 = by0 + 9, by0 + 27
ax.plot([gx0, gx1], [gy0, gy0], color=INK, lw=1.1)
ax.plot([gx0, gx0], [gy0, gy1], color=INK, lw=1.1)
SLATE = "#5B6B7A"
tt = np.linspace(0.001, 0.999, 200)
for (a_, b_, col, lab, px) in [(9, 2, OPEN, r"$p_i$", None),
                               (5, 5, SLATE, r"$p_j$", None)]:
    pdf = betadist.pdf(tt, a_, b_)
    pdf = pdf / pdf.max()
    X = gx0 + tt * (gx1 - gx0)
    Y = gy0 + pdf * (gy1 - gy0) * 0.92
    ax.plot(X, Y, color=col, lw=2.3)
    ax.fill_between(X, gy0, Y, color=col, alpha=0.12)
    pm = tt[np.argmax(pdf)]
    ax.text(gx0 + pm * (gx1 - gx0), gy0 + (gy1 - gy0) * 0.98,
            lab, color=col, fontsize=11, ha="center", va="bottom",
            fontweight="bold")
ax.text((gx0 + gx1) / 2, gy0 - 3.4, r"$p=P(\mathrm{open}\,|\,k,n)$",
        ha="center", fontsize=10.5)
ax.text((gx0 + gx1) / 2, gy0 - 7.2, "Beta–Binomial mixture",
        ha="center", fontsize=8.8, color=MUT, style="italic")
ax.text(gx0 - 1.2, gy1, "density", rotation=90, va="top", ha="right",
        fontsize=8.2, color=MUT)
ax.text(gx1, gy0 - 1.0, "P(open)", va="top", ha="right", fontsize=8.2, color=MUT)

# =====================================================================
# Panel c — U-3 soft co-state probabilities
# =====================================================================
cx0, cy0 = XS[2], YS[0]
panel(cx0, cy0, "c", "Soft pairwise co-state")
rows = [
    ("co-open",   1, 1, OPEN,   r"$p_i\,p_j$"),
    ("co-closed", 0, 0, CLOSED, r"$(1-p_i)(1-p_j)$"),
    ("mixed",     1, 0, MUT,    r"$p_i(1-p_j)+(1-p_i)p_j$"),
]
ry = [cy0 + 30, cy0 + 19.5, cy0 + 9]
for (label, s1, s2, col, expr), yy in zip(rows, ry):
    # two state circles
    for k, s in enumerate((s1, s2)):
        fc = OPEN if s else "white"
        ec = OPEN if s else CLOSED
        ax.add_patch(Circle((cx0 + 7 + k * 4.2, yy), 1.5, fc=fc, ec=ec, lw=1.6,
                            zorder=4))
    ax.text(cx0 + 16, yy + 1.4, label, fontsize=9.6, fontweight="bold",
            va="center", color=col)
    ax.text(cx0 + 16, yy - 2.2, expr, fontsize=9.0, va="center", color=INK)
ax.text(cx0 + PW/2, cy0 + 3.0,
        "soft expected counts (no thresholding)",
        ha="center", fontsize=8.6, color=MUT, style="italic")

# =====================================================================
# Panel d — distance stratification
# =====================================================================
dx0, dy0 = XS[0], YS[1]
panel(dx0, dy0, "d", "Long-range vs local baseline")
axis_y = dy0 + 14
ax.annotate("", xy=(dx0 + 43, axis_y), xytext=(dx0 + 5, axis_y),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
ax.text(dx0 + 43, axis_y - 3.2, "contact distance", ha="right", fontsize=8.6,
        color=MUT)
# threshold
thx = dx0 + 20
ax.plot([thx, thx], [axis_y - 1.5, axis_y + 14], color=ACC, lw=1.4, ls="--")
ax.text(thx, axis_y + 15, r"$d_0\,(\approx\!20\,\mathrm{kb})$", ha="center",
        fontsize=8.8, color=ACC)
# unlinked pair (short range)
ax.plot([dx0 + 8, dx0 + 14], [axis_y, axis_y], color=DNA, lw=2.4,
        solid_capstyle="round")
arc(dx0 + 8, dx0 + 14, axis_y, 4, color=MUT, lw=1.6)
ax.text(dx0 + 11, axis_y - 4.5, "unlinked\n(< $d_0$, baseline)", ha="center",
        va="top", fontsize=8.4, color=MUT)
# linked pair (long range)
ax.plot([dx0 + 24, dx0 + 40], [axis_y, axis_y], color=DNA, lw=2.4,
        solid_capstyle="round")
arc(dx0 + 24, dx0 + 40, axis_y, 8, color=ACC, lw=2.0)
ax.text(dx0 + 32, axis_y - 4.5, "linked\n(long-range)", ha="center", va="top",
        fontsize=8.6, color=ACC, fontweight="bold")
# bin bracket
by = dy0 + 33
ax.plot([dx0 + 5, dx0 + 42], [by, by], color=INK, lw=1.2)
for xx in (dx0 + 5, dx0 + 42):
    ax.plot([xx, xx], [by, by - 1.4], color=INK, lw=1.2)
ax.text(dx0 + PW/2, by + 1.2, "aggregate within 250-kb bin",
        ha="center", fontsize=8.8)

# =====================================================================
# Panel e — Delta_signed
# =====================================================================
ex0, ey0 = XS[1], YS[1]
panel(ex0, ey0, "e", "Signed long-range concordance")
ax.text(ex0 + PW/2, ey0 + 33,
        r"$\Delta_{\mathrm{signed}}=(\hat\pi_{\mathrm{co\text{-}open}}-\hat\pi_{\mathrm{co\text{-}closed}})_{\mathrm{linked}}$",
        ha="center", fontsize=10.2)
ax.text(ex0 + PW/2, ey0 + 28.5,
        r"$-\,(\hat\pi_{\mathrm{co\text{-}open}}-\hat\pi_{\mathrm{co\text{-}closed}})_{\mathrm{unlinked}}$",
        ha="center", fontsize=10.2)
# mini heatmap (cells x bins)
hm_x0, hm_y0, hm_w, hm_h = ex0 + 11, ey0 + 5.5, 26, 17
rng = np.random.default_rng(1)
nC, nB = 12, 18
block = np.vstack([rng.normal(0.5, 0.35, (6, nB)),
                   rng.normal(-0.5, 0.35, (6, nB))])
block[:, ::3] += rng.normal(0, 0.5, (nC, len(range(0, nB, 3))))
ax.imshow(block, cmap="RdBu_r", vmin=-1.4, vmax=1.4, aspect="auto",
          extent=[hm_x0, hm_x0 + hm_w, hm_y0, hm_y0 + hm_h], zorder=3)
ax.add_patch(Rectangle((hm_x0, hm_y0), hm_w, hm_h, fc="none", ec=INK, lw=1.0,
                       zorder=4))
ax.text(hm_x0 + hm_w / 2, hm_y0 - 1.7, "bins", ha="center", fontsize=8.4,
        color=MUT)
ax.text(hm_x0 - 1.4, hm_y0 + hm_h / 2, "cells", rotation=90, va="center",
        ha="right", fontsize=8.4, color=MUT)
# colorbar hint
ax.text(hm_x0 + hm_w + 1.4, hm_y0 + hm_h, r"$+$", color="#B2182B", fontsize=10,
        va="top", fontweight="bold")
ax.text(hm_x0 + hm_w + 1.4, hm_y0, r"$-$", color="#2166AC", fontsize=10,
        va="bottom", fontweight="bold")

# =====================================================================
# Panel f — RWR denoising + UMAP
# =====================================================================
fx0, fy0 = XS[2], YS[1]
panel(fx0, fy0, "f", "Cell-graph denoising → cell types")
# small k-NN graph (left)
gcx, gcy = fx0 + 11, fy0 + 22
rngp = np.random.default_rng(7)
nodes = np.array([[gcx, gcy]] +
                 [[gcx + 6*np.cos(t), gcy + 6*np.sin(t)]
                  for t in np.linspace(0, 2*np.pi, 6, endpoint=False)])
for i in range(1, len(nodes)):
    ax.plot([nodes[0,0], nodes[i,0]], [nodes[0,1], nodes[i,1]],
            color=EDGE, lw=1.1, zorder=2)
for i in (1, 2, 4):
    ax.plot([nodes[i,0], nodes[i+1 if i+1 < len(nodes) else 1,0]],
            [nodes[i,1], nodes[i+1 if i+1 < len(nodes) else 1,1]],
            color=EDGE, lw=1.1, zorder=2)
for (nx, ny) in nodes:
    ax.add_patch(Circle((nx, ny), 1.25, fc="white", ec=ACC, lw=1.6, zorder=4))
ax.text(gcx, fy0 + 32, "RWR over\ncell $k$-NN graph", ha="center", fontsize=8.4,
        color=MUT)
# arrow to UMAP
flow(fx0 + 19, fy0 + 20, fx0 + 24, fy0 + 20, rad=0.0)
# UMAP scatter (right)
ux, uy = fx0 + 36, fy0 + 19
cA = np.random.default_rng(11).normal([ux - 4, uy + 3], [2.2, 2.4], (28, 2))
cB = np.random.default_rng(12).normal([ux + 4, uy - 3], [2.2, 2.4], (28, 2))
ax.scatter(cA[:,0], cA[:,1], s=14, c=CTA, edgecolors="none", zorder=4)
ax.scatter(cB[:,0], cB[:,1], s=14, c=CTB, edgecolors="none", zorder=4)
ax.text(ux, fy0 + 32, "UMAP", ha="center", fontsize=8.8, color=MUT)
ax.add_patch(Circle((fx0 + 30, fy0 + 4.6), 1.2, fc=CTA, ec="none"))
ax.add_patch(Circle((fx0 + 38, fy0 + 4.6), 1.2, fc=CTB, ec="none"))
ax.text(fx0 + 31.6, fy0 + 4.6, "type A", fontsize=8.2, va="center")
ax.text(fx0 + 39.6, fy0 + 4.6, "type B", fontsize=8.2, va="center")

# =====================================================================
# flow arrows between panels
# =====================================================================
flow(XS[0] + PW, YS[0] + PH/2, XS[1], YS[0] + PH/2)
flow(XS[1] + PW, YS[0] + PH/2, XS[2], YS[0] + PH/2)
# wrap c -> d (kept within the inter-row gap so it clears the panels)
ax.add_patch(FancyArrowPatch((XS[2] + PW/2, YS[0] - 0.3),
             (XS[0] + PW/2, YS[1] + PH + 0.3),
             connectionstyle="arc3,rad=0.0", arrowstyle="-|>",
             mutation_scale=15, lw=1.8, color=ACC, zorder=6))
flow(XS[0] + PW, YS[1] + PH/2, XS[1], YS[1] + PH/2)
flow(XS[1] + PW, YS[1] + PH/2, XS[2], YS[1] + PH/2)

fig.savefig("figures/method_schematic.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig("figures/method_schematic.pdf", bbox_inches="tight",
            facecolor="white")
print("wrote figures/method_schematic.png and .pdf")
