#!/usr/bin/env python3
"""The directional concordance decomposition, as a standalone equation figure.

Replaces the single-equation

    Delta_concordant = (O_open + O_closed) - (E_open + E_closed)

with the three statistics that decomposition actually yields: the coordination
magnitude C (the old Delta_concordant, renamed), the realized direction F, and
their combination D -- the single co-closed -> mixed -> co-open scale.

Ordered magnitude -> direction -> combination so the figure builds: D is the
last line because it consumes the first two.

Rendered in Computer Modern to match the LaTeX equation figures rather than
the DejaVu house style of the schematic figures in this directory.  Equations
are aligned on the "=" by right-anchoring each left-hand side and
left-anchoring each right-hand side at a shared x.

Emits a vector PDF + a 300-dpi PNG.

    python figures/concordance_formulas.py                # with captions
    python figures/concordance_formulas.py --no-labels    # equations only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#000000"
MUT = "#6B7280"

# (lhs, rhs, caption, y, caption_dy).  lhs/rhs are split at the "=" so the
# column can align.  y and caption_dy are per-row rather than evenly spaced
# because the F row renders a FRACTION: it is roughly twice as tall as the
# other two, so a shared caption offset would either overlap its denominator
# or leave D's caption stranded at the bottom of the canvas.
ROWS = [
    (r"$C$",
     r"$=\ (O_{\mathrm{open}} + O_{\mathrm{closed}})"
     r" - (E_{\mathrm{open}} + E_{\mathrm{closed}})$",
     r"coordination magnitude  ($=2\,\mathrm{Cov}$), marginal-free",
     0.90, 0.11),
    (r"$F$",
     r"$=\ \dfrac{O_{\mathrm{open}}}{O_{\mathrm{open}} + O_{\mathrm{closed}}}$",
     r"realized direction of the concordant mass",
     0.55, 0.17),
    (r"$D$",
     r"$=\ C \cdot (2F - 1)$",
     r"directional coordination:  co-closed $(-)$  $\rightarrow$  "
     r"mixed $(0)$  $\rightarrow$  co-open $(+)$",
     0.19, 0.11),
]


def build(show_labels: bool = True):
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(9.6, 3.6 if show_labels else 3.0),
                     facecolor="white")
    x_eq = 0.20                       # the shared "=" column

    for lhs, rhs, caption, y, cap_dy in ROWS:
        fig.text(x_eq, y, lhs, fontsize=21, color=INK,
                 ha="right", va="center")
        fig.text(x_eq + 0.012, y, rhs, fontsize=21, color=INK,
                 ha="left", va="center")
        if show_labels:
            fig.text(x_eq + 0.012, y - cap_dy, caption, fontsize=10.5,
                     color=MUT, ha="left", va="center", style="italic")
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-labels", action="store_true",
                    help="Equations only, no captions (matches the bare "
                         "style of the single-equation original).")
    ap.add_argument("--output-dir", default=str(Path(__file__).parent))
    ap.add_argument("--stem", default="concordance_formulas")
    args = ap.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = build(show_labels=not args.no_labels)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        path = out / f"{args.stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
