"""Publication figure style, shared so the panels match across scripts.

Lifted verbatim out of ``scripts/null_calibration_qq.py``, which is where it
was worked out; that script now imports it rather than keeping a second copy.
The three choices that are not cosmetic:

* ``pdf.fonttype 42`` -- TrueType, not Type 3.  Type 3 text is not editable in
  Illustrator and several journals reject it outright.
* Okabe-Ito -- colour-blind safe AND distinguishable in greyscale.  Journal
  figures get printed and photocopied; tab10's red/green pair survives neither.
* 180 mm width -- a full-width panel in most journals.  Sizing to it means the
  figure is placed at 100% and the type comes out at the size set here, rather
  than being scaled by the layout into whatever that gives.
"""

from __future__ import annotations

PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000")

FIG_W_IN = 180 / 25.4

RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
}
