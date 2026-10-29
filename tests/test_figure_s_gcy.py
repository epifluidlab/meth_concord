"""figure_s_gcy.py: Fig. 5a-d from runs on GCY sites alone, each panel as
large and where it is in Fig. 5, every GCH of the text named GCY."""

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
    "figure_s_gcy", _ROOT / "scripts" / "figure_s_gcy.py")
sg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sg)


def _inches(fig):
    H = fig.get_figheight()
    return [((1 - b.y1) * H, b.height * H, b.x0, b.width)
            for b in (a.get_position() for a in fig.axes)]


def test_a_to_d_as_large_and_where_they_are_in_figure_5(tmp_path):
    argv = t5._tables(tmp_path)
    f5_fig = t5.f5.figure_from_args(t5.f5.parse_args(argv))
    out = tmp_path / "S.pdf"
    args = sg.parse_args(["--distance", str(tmp_path / "dist"),
                          "--output", str(out)])
    d = sg.read_tables(args)
    order, colors = sg.f5.cell_colors(args, d["percell"]["group"])
    fig = sg.build_figure(d, order, colors)
    assert [t.get_text() for t in fig.texts] == list("abcd")
    assert (4 * fig.get_figheight()) % 1 == 0
    assert np.allclose(_inches(fig), _inches(f5_fig)[:len(fig.axes)])
    assert sg.main(["--distance", str(tmp_path / "dist"), "--output",
                    str(out)]) == 0
    assert out.exists() and out.stat().st_size > 20_000


def test_the_sites_are_named_gcy_and_hcg_stays(tmp_path):
    from matplotlib.text import Text
    t5._tables(tmp_path)
    args = sg.parse_args(["--distance", str(tmp_path / "dist"),
                          "--output", str(tmp_path / "S.pdf")])
    d = sg.read_tables(args)
    fig = sg.build_figure(d, *sg.f5.cell_colors(args, d["percell"]["group"]))
    texts = [t.get_text() for t in fig.findobj(Text)]
    assert not any("GCH" in t for t in texts)
    for want in ("GCY–GCY: accessibility", "HCG–HCG: methylation",
                 "HCG–GCY: methylation × accessibility", "GCY accessibility",
                 "GCY: per-cell Δr"):
        assert want in texts
    assert any(t.startswith("GCY–GCY Δr\n") for t in texts)


def test_a_and_b_from_their_own_run_and_missing_tables_named(tmp_path):
    t5._tables(tmp_path)
    ab = tmp_path / "dist_ab"
    ab.mkdir()
    t5._distance(ab, np.random.default_rng(1))
    s = t5.pd.read_csv(ab / "end_posterior_regions_summary.tsv", sep="\t")
    s.assign(pearson_r=0.7).to_csv(ab / "end_posterior_regions_summary.tsv",
                                   sep="\t", index=False)
    d = sg.read_tables(sg.parse_args(
        ["--distance", str(tmp_path / "dist"), "--distance-ab", str(ab),
         "--output", "x.pdf"]))
    assert d["regions_summary"]["pearson_r"] == 0.7
    (ab / "end_posterior_regions.tsv").unlink()
    with pytest.raises(SystemExit, match="end_posterior_regions.tsv"):
        sg.main(["--distance", str(tmp_path / "dist"), "--distance-ab",
                 str(ab), "--output", "x.pdf"])
