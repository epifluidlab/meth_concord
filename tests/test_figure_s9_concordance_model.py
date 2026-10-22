"""figure_s9_concordance_model.py: the schematic of how long-range
concordance is measured draws, every panel with its title, as one PDF."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "figure_s9", Path(__file__).resolve().parents[1] / "scripts"
    / "figure_s9_concordance_model.py")
s9 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(s9)


def test_the_schematic_draws_its_nine_panels(tmp_path):
    out = tmp_path / "s9.pdf"
    assert s9.main(["--output", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 20_000
    fig = s9.build_figure()
    letters = {t.get_text() for t in fig.texts}
    assert set("abcdefghi") <= letters
    assert "Concordance on each X copy" in letters
