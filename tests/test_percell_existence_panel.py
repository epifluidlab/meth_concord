"""Figure 5c, per-cell existence: the statistics behind it and the stamp.

The stamp test is the one that matters most.  A mock-up drawn from simulated
input looks exactly like a result, so whether it says SIMULATED must be decided
by the data it was drawn from, not by a flag someone can forget to pass.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

_SCRIPT = (Path(__file__).resolve().parent.parent / "figures"
           / "percell_existence_panel.py")


def _mod():
    spec = importlib.util.spec_from_file_location("fig5c", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["fig5c"] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- simulation ---

def test_simulation_has_the_table_the_real_run_will_supply():
    m = _mod()
    df = m.simulate(0)
    for c in m.REQUIRED:
        assert c in df.columns, c
    assert df["simulated"].all()
    for mark in m.MARKS:
        assert (df["mark"] == mark).sum() == 1577
    assert list(pd.unique(df["group"])) == m.GROUP_ORDER


def test_z_is_delta_over_its_standard_error():
    m = _mod()
    df = m.simulate(1)
    assert np.allclose(df["z"], df["delta"] / df["se"])


# ---------------------------------------------------------------- statistics ---

def test_q_values_are_per_mark_and_monotone_in_z():
    """Per mark: pooling would let the deeper mark set the other's threshold."""
    m = _mod()
    df = m.add_q(m.simulate(2))
    for mark, d in df.groupby("mark"):
        d = d.sort_values("z", ascending=False)
        assert np.all(np.diff(d["q"].to_numpy()) >= -1e-12), mark
    # recomputing one mark alone gives the same q -> not pooled
    one = m.add_q(m.simulate(2).query("mark == 'GCH'"))
    both = df[df["mark"] == "GCH"]
    assert np.allclose(np.sort(one["q"]), np.sort(both["q"]))


def test_the_null_passes_about_as_often_as_it_should():
    """A calibrated within-cell null gives ~0 BH discoveries."""
    m = _mod()
    df = m.add_q(m.simulate(3))
    assert (df["q_null"] < 0.05).mean() < 0.01


def test_pass_depth_reports_a_crossing_only_when_there_is_one():
    """The first version reported the shallowest bin as a threshold whenever
    the median already passed there -- a threshold the data never showed."""
    m = _mod()
    n = np.logspace(2.5, 5, 600)
    rising = n > 3_000
    depth, kind = m.pass_depth(n, rising)
    assert kind == "crossing"
    assert 2_000 < depth < 5_000
    assert m.pass_depth(n, np.ones_like(n, bool))[1] == "always"
    assert m.pass_depth(n, np.zeros_like(n, bool))[1] == "never"


# ------------------------------------------------------------------ plotting ---

def _texts(fig):
    return [t.get_text() for t in fig.texts]


def test_simulated_input_is_stamped_on_the_figure(tmp_path):
    m = _mod()
    fig = m.plot(m.add_q(m.simulate(4)), out_prefix=str(tmp_path / "c"),
                 formats=("pdf",))
    assert any(m.SIM_STAMP in t for t in _texts(fig))


def test_real_input_carries_no_stamp():
    m = _mod()
    df = m.add_q(m.simulate(5)).drop(columns=["simulated"])
    fig = m.plot(df)
    assert not any(m.SIM_STAMP in t for t in _texts(fig))


def test_one_simulated_row_is_enough_to_stamp():
    """A table that mixes real and simulated rows is not a result either."""
    m = _mod()
    df = m.add_q(m.simulate(6))
    df["simulated"] = False
    df.loc[df.index[0], "simulated"] = True
    fig = m.plot(df)
    assert any(m.SIM_STAMP in t for t in _texts(fig))


def test_a_table_missing_columns_is_refused():
    m = _mod()
    df = m.simulate(7).drop(columns=["z_null"])
    with pytest.raises(ValueError, match="z_null"):
        m.plot(df)


def test_pdf_embeds_truetype_not_type3(tmp_path):
    """Type 3 text is not editable in Illustrator and several journals reject
    it outright."""
    m = _mod()
    m.plot(m.add_q(m.simulate(8)), out_prefix=str(tmp_path / "c"),
           formats=("pdf",))
    raw = (tmp_path / "c.pdf").read_bytes()
    assert b"/Type3" not in raw
    assert b"/FontFile2" in raw or b"/Type0" in raw
