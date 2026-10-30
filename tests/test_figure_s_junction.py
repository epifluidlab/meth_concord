"""figure_s_junction.py: junction_concordance.py's tables as a supplementary
figure in Fig. 5's style -- a's schematic, then per mark r and Δr against
s, s below 100 bp shaded, the contacts without s beside each axis.

The tables are junction_concordance.summarize's own, from synthetic cells:
contacts' ends alike below 100 bp and less so beyond, their stand-ins not;
one molecule's ends less alike the more DNA between them."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pysam")

_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


jc = _load("junction_concordance")
sj = _load("figure_s_junction")


def _tables(out: Path, seed=0, n_cells=6, n=4_000):
    """junction_concordance.tsv as analyze writes it; every bin of s, the
    last one (none) too, drawn alike often."""
    rng = np.random.default_rng(seed)
    nb = jc.NB
    centre = np.r_[(jc.S_EDGES[:-1] + jc.S_EDGES[1:]) / 2, 2_000]
    rho = {"linked": np.where(centre < 100, 0.4, 0.1), "null": np.zeros(nb),
           "molecule": 0.4 * np.exp(-centre / 300)}
    per_cell = {}
    for ci in range(n_cells):
        d = {}
        for mark in jc.MARKS:
            for set_, arms in jc.SETS.items():
                b = rng.integers(0, nb, n)
                for arm in arms:
                    r = rho["molecule" if set_ == "molecules" else arm][b]
                    x = rng.normal(size=n)
                    y = r * x + np.sqrt(1 - r ** 2) * rng.normal(size=n)
                    acc = jc.new_acc(nb)
                    jc.accumulate(acc, b, x, y)
                    d[(mark, set_, arm)] = acc
                k = np.bincount(b, minlength=nb).astype(float)
                d[(mark, set_, "sites")] = np.vstack([k, 4 * k,
                                                      np.zeros(nb)])
        per_cell[f"c{ci}"] = d
    table, _ = jc.summarize(per_cell, argparse.Namespace(
        min_pairs=50, n_boot=100, seed=1))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "junction_concordance.tsv", sep="\t", index=False,
                 float_format="%.5g")


def _points(ax) -> list:
    """The y of every single point drawn in ``ax``."""
    return sorted(float(ln.get_ydata()[0]) for ln in ax.get_lines()
                  if len(ln.get_xdata()) == 1)


def test_the_panels_from_the_tables(tmp_path):
    _tables(tmp_path / "jc")
    t = sj.read_table(tmp_path / "jc")
    fig = sj.build_figure(t)
    assert [x.get_text() for x in fig.texts] == list("abc")
    assert fig.get_figwidth() == 7.2 and (4 * fig.get_figheight()) % 1 == 0
    r_axes = [a for a in fig.axes if a.get_title(loc="left")]
    assert [a.get_title(loc="left") for a in r_axes] == \
        ["GCH accessibility", "HCG methylation"]
    d_axes = [a for a in fig.axes if a.get_xlabel().startswith("s, ")]
    strips = [a for a in fig.axes if a.get_xlim() == (-1, 1)]
    assert len(d_axes) == 2 and len(strips) == 4
    for ax in r_axes + d_axes:
        # the junction's bins shaded, the axis to 1 kb
        (x0, _), (x1, _) = ax.transData.inverted().transform(
            ax.patches[0].get_window_extent().get_points())
        assert (round(x0), round(x1)) == (0, sj.JUNCTION_BP)
        assert ax.get_xlim() == (0, 1_000)
    # beside them, the contacts without s
    for (mark, _), r_strip, d_strip in zip(sj.MARKS, strips[0::2],
                                           strips[1::2]):
        none = t[(t["mark"] == mark) & t["s_lo"].isna()].set_index("set")
        assert np.allclose(_points(r_strip), sorted(
            none.loc["contacts", ["r", "r_null"]]))
        assert np.allclose(_points(d_strip), sorted(
            none.loc[["contacts", "trans"], "delta"]))
    out = tmp_path / "S.pdf"
    assert sj.main(["--junction", str(tmp_path / "jc"), "--output",
                    str(out)]) == 0
    assert out.exists() and out.stat().st_size > 20_000


def test_the_junction_s_share_of_contacts(tmp_path):
    from matplotlib.text import Text
    _tables(tmp_path / "jc")
    t = sj.read_table(tmp_path / "jc")
    u = t[(t["mark"] == "GCH") & (t["set"] == "contacts")]
    want = u.loc[u["s_lo"] < 100, "pairs"].sum() / u["pairs"].sum()
    assert sj.junction_share(t, "GCH") == pytest.approx(want)
    assert 0.12 < want < 0.24                  # two bins of eleven
    texts = [x.get_text() for x in sj.build_figure(t).findobj(Text)]
    assert f"  s < 100 bp: {sj.pct(want)} of contacts" in texts
    assert sj.pct(0.0042) == "0.4%" and sj.pct(0.00042) == "0.042%"


def test_a_missing_table_or_mark_named(tmp_path):
    with pytest.raises(SystemExit, match="junction_concordance.tsv"):
        sj.main(["--junction", str(tmp_path), "--output", "x.pdf"])
    _tables(tmp_path / "jc")
    p = tmp_path / "jc" / "junction_concordance.tsv"
    t = sj.read_table(tmp_path / "jc")
    t[t["mark"] == "GCH"].to_csv(p, sep="\t", index=False)
    with pytest.raises(SystemExit, match="has no HCG"):
        sj.read_table(tmp_path / "jc")
