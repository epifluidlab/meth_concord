"""The mechanistic scheme panel.

A drawing has few things worth asserting, but two of them matter for a
journal panel: it must stay VECTOR (a rasterised PDF is useless at print
size), and the two nuclei must differ in exactly the one respect the model
claims.  A schematic that changes several things at once lets a reader
attribute the effect to whichever one they already believed.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "mechanism_scheme.py")


def _mod():
    spec = importlib.util.spec_from_file_location("mech", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["mech"] = m
    spec.loader.exec_module(m)
    return m


def _axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return fig, ax


def _arc_widths(ax, color):
    """Line widths of the coordination arcs drawn in one colour."""
    out = []
    for p in ax.patches:
        ec = p.get_edgecolor()
        if p.get_facecolor()[3] == 0 and ec is not None:
            out.append((tuple(np.round(ec[:3], 3)), p.get_linewidth()))
    import matplotlib.colors as mc
    want = tuple(np.round(mc.to_rgb(color), 3))
    return [w for c, w in out if c == want]


def test_the_two_panels_differ_in_arc_weight_not_in_geometry():
    """The one claim the drawing makes."""
    m = _mod()
    fig_a, ax_a = _axes()
    fig_b, ax_b = _axes()
    m.draw(ax_a, "ref", "a", b_strength=0.10, a_strength=0.95, note="")
    m.draw(ax_b, "tum", "b", b_strength=0.95, a_strength=0.14, note="",
           cross=True)
    # heterochromatin coordination is heavier in the tumour panel
    assert (np.mean(_arc_widths(ax_b, m.OTHER))
            > 2 * np.mean(_arc_widths(ax_a, m.OTHER)))
    # and the regulatory hub is lighter
    assert (np.mean(_arc_widths(ax_b, m.GENE))
            < 0.6 * np.mean(_arc_widths(ax_a, m.GENE)))
    # same number of loci and domains in both -- only the weights moved
    assert len(ax_a.patches) == len(ax_b.patches)


def test_the_decohered_hub_is_still_drawn():
    """'Weakened' is the claim, not 'gone'.  Arcs at zero alpha would say
    the regulatory compartment stops existing, which is a different and
    much stronger statement than the data supports."""
    m = _mod()
    fig, ax = _axes()
    m.draw(ax, "tum", "b", b_strength=0.95, a_strength=0.14, note="",
           cross=True)
    import matplotlib.colors as mc
    want = tuple(np.round(mc.to_rgb(m.GENE), 3))
    alphas = [p.get_alpha() for p in ax.patches
              if tuple(np.round(p.get_edgecolor()[:3], 3)) == want
              and p.get_facecolor()[3] == 0]
    assert alphas and min(alphas) > 0.25


def test_an_arc_starts_and_ends_on_the_loci_it_connects():
    m = _mod()
    fig, ax = _axes()
    m._arc(ax, (0.1, 0.2), (0.7, 0.5), 0.05, "#000000", 1.0)
    v = ax.patches[-1].get_path().vertices
    assert tuple(np.round(v[0], 6)) == (0.1, 0.2)
    assert tuple(np.round(v[-1], 6)) == (0.7, 0.5)


def test_legend_columns_do_not_advance_by_a_guess():
    """Text width is not knowable before rendering, so a cumulative advance
    is how legend items land on top of one another.  Fixed, increasing
    columns instead."""
    src = _SCRIPT.read_text()
    block = src[src.index("def legend_panel"):src.index("def _direction")]
    xs = [float(line.split("(")[1].split(",")[0])
          for line in block.splitlines() if line.strip().startswith("(0.")]
    assert len(xs) >= 4
    assert xs == sorted(xs)
    assert min(np.diff(xs)) > 0.15


def test_it_renders_a_vector_pdf(tmp_path):
    """Rasterised output is unusable at print size."""
    out = tmp_path / "scheme.pdf"
    r = subprocess.run([sys.executable, str(_SCRIPT), "--output", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 4000
    head = out.read_bytes()[:2000]
    assert head.startswith(b"%PDF")
    # a PDF of one big image would carry an /Image XObject; a vector one does not
    assert b"/Subtype /Image" not in out.read_bytes()


def test_it_renders_png_for_a_draft(tmp_path):
    out = tmp_path / "scheme.png"
    r = subprocess.run([sys.executable, str(_SCRIPT), "--output", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 4000


def test_the_caption_says_it_is_a_model(tmp_path):
    """The panel sits beside measured ones.  Nothing in it is fitted, and
    the figure has to say so where a reader will see it."""
    src = _SCRIPT.read_text()
    assert "Model consistent with" in src
    assert "MODEL, not a measurement" in src
