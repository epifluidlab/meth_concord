"""figure_s_human.py: Fig. 5a-d once per human dataset -- each block's
panels as large and as placed as figure_s_gcy draws Fig. 5a-d, under the
dataset's name, its letters running on from the block above, its cell
types in order, named and coloured as the lines' and the meningioma's --
and the runs it misses named."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tests"))
import test_figure5_cortex as t5  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "figure_s_human", _ROOT / "scripts" / "figure_s_human.py")
sh = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sh)

LINES = ("GM12878", "IMR90")
MENINGIOMA = ("tumor1", "tumor2", "monocyte")


def _run(d: Path, types, seed) -> Path:
    """A run's tables (test_figure5_cortex's), its cells of ``types``."""
    d.mkdir(parents=True)
    t5._distance(d, np.random.default_rng(seed))
    p = d / "end_posterior_percell.tsv"
    pc = pd.read_csv(p, sep="\t")
    pc["group"] = [types[int(c[1:]) % len(types)] for c in pc["cell_id"]]
    pc.to_csv(p, sep="\t", index=False)
    return d


def _argv(tmp_path, extra=()) -> list:
    lines = _run(tmp_path / "lines", LINES, 0)
    men = _run(tmp_path / "men", MENINGIOMA, 1)
    return ["--dataset", "GM12878 and IMR-90", "--distance", str(lines),
            "--distance-ab", str(lines), "--dataset", "Meningioma",
            "--distance", str(men), "--distance-ab", str(men), *extra,
            "--output", str(tmp_path / "S.pdf")]


def _inches(fig, axes) -> np.ndarray:
    """Each of ``axes``: its top below the first's and its height, in
    inches; its left and width, as shares of the page."""
    H = fig.get_figheight()
    top = axes[0].get_position().y1
    return np.array([((top - b.y1) * H, b.height * H, b.x0, b.width)
                     for b in (a.get_position() for a in axes)])


def test_a_block_per_dataset_its_panels_as_in_figure_5(tmp_path):
    argv = _argv(tmp_path)
    args = sh.parse_args(argv)
    fig = sh.figure_from_args(args)
    texts = [t.get_text() for t in fig.texts]
    assert [t for t in texts if len(t) == 1] == list("abcdefgh")
    assert texts.index("GM12878 and IMR-90") < texts.index("Meningioma")
    assert fig.get_figwidth() == 7.2 and (4 * fig.get_figheight()) % 1 == 0
    # each block as figure_s_gcy draws Fig. 5a-d on a page of its own
    d = sh.sg.read_tables(sh.sg.parse_args(
        ["--distance", str(tmp_path / "lines"), "--output", "x.pdf"]))
    one = sh.sg.build_figure(d, *sh.type_colors(args, d["percell"]["group"]))
    n = len(one.axes)
    assert len(fig.axes) == 2 * n
    blocks = [fig.axes[:n], fig.axes[n:]]
    for axes in blocks:
        assert np.allclose(_inches(fig, axes), _inches(one, one.axes))
    # the second below the first, d's labels and the name clear between
    H = fig.get_figheight()
    above, _, _ = sh.sg.ad_inches()
    low = min(a.get_position().y0 for a in blocks[0])
    high = max(a.get_position().y1 for a in blocks[1])
    assert (low - high) * H == pytest.approx(sh.sg.FOOT + sh.HEAD + above)
    # a's schematics are Fig. 5a's
    assert not any(ax.child_axes for ax in fig.axes)
    assert sh.main(argv) == 0
    assert (tmp_path / "S.pdf").stat().st_size > 20_000


def test_d_names_and_colours_the_lines_and_the_meningioma_s_types(tmp_path):
    s = tmp_path / "meningioma.samples.tsv"
    pd.DataFrame({"cell_name": ["x", "y"], "group": ["tumor1", "monocyte"],
                  "cell_type_joint_color": ["#123456", "#654321"]}).to_csv(
        s, sep="\t", index=False)
    fig = sh.figure_from_args(sh.parse_args(_argv(tmp_path,
                                                  ["--samples", str(s)])))
    n = len(fig.axes) // 2
    # each block's last axes: d's lower one, which names the types
    ticks = [{t.get_text(): t.get_color()
              for t in fig.axes[(k + 1) * n - 1].get_xticklabels()}
             for k in range(2)]
    assert list(ticks[0]) == ["GM12878", "IMR-90"]
    assert list(ticks[1]) == ["tumor 1", "tumor 2", "monocyte"]
    assert ticks[0] == {"GM12878": sh.TYPES["GM12878"],
                        "IMR-90": sh.TYPES["IMR90"]}
    assert ticks[1] == {"tumor 1": "#123456", "tumor 2": sh.TYPES["tumor2"],
                        "monocyte": "#654321"}


def test_unpaired_runs_and_missing_tables_named(tmp_path):
    lines = _run(tmp_path / "lines", LINES, 0)
    with pytest.raises(SystemExit, match="once per dataset: 1, 1 and 0"):
        sh.main(["--dataset", "x", "--distance", str(lines),
                 "--output", "x.pdf"])
    (lines / "end_posterior_regions.tsv").unlink()
    with pytest.raises(SystemExit, match="end_posterior_regions.tsv"):
        sh.main(["--dataset", "x", "--distance", str(lines),
                 "--distance-ab", str(lines), "--output", "x.pdf"])
