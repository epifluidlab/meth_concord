"""figure_s_null_controls.py: Fig. 5a-c under each non-linked control, one
row per control from its run; a control without a run keeps its row, empty."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tests"))
import test_figure5_cortex as t5  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "figure_s_null_controls", _ROOT / "scripts" / "figure_s_null_controls.py")
sn = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sn)


def _run(d: Path, seed: int) -> Path:
    d.mkdir(parents=True)
    t5._distance(d, np.random.default_rng(seed))
    return d


def test_each_control_draws_its_row_and_a_missing_one_waits(tmp_path):
    r1, r2 = _run(tmp_path / "r1", 0), _run(tmp_path / "r2", 1)
    out = tmp_path / "S.pdf"
    assert sn.main(["--run", f"1={r1}", "--run", f"2={r2}", "--output",
                    str(out)]) == 0
    assert out.exists() and out.stat().st_size > 20_000
    fig = sn.build_figure(sn.runs_of([f"1={r1}", f"2={r2}"]))
    texts = [t.get_text() for a in fig.axes for t in a.texts]
    assert "run pending" in texts
    # the Δr panels share one scale
    ylims = {a.get_ylim() for a in fig.axes
             if a.get_ylabel() == "Δr, linked − non-linked"}
    assert len(ylims) == 1


def test_a_run_must_name_its_control(tmp_path):
    r1 = _run(tmp_path / "r1", 0)
    with pytest.raises(SystemExit, match="K=DIR"):
        sn.runs_of([f"4={r1}"])


def test_the_third_control_does_not_name_this_figure(tmp_path):
    """Supplementary Fig. 9c says where control 3 is used: here.  In its
    own row that would be this figure naming itself."""
    r3 = _run(tmp_path / "r3", 2)
    fig = sn.build_figure(sn.runs_of([f"3={r3}"]))
    texts = [t.get_text() for a in fig.axes for t in a.texts]
    assert "this figure only" in texts
    assert sn.s9.CONTROLS[2][1] not in texts
