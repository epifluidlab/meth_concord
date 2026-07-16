"""Tests for --color-column / --group-order (UMAP legend colours + ordering).

Covers the pure logic: legend ordering, the samples-sheet colour mapping (incl.
its validation/warning paths), and colour resolution with fallback.  The plot
call itself needs ``umap`` at runtime, so it is not exercised here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance.cli import group_colors_from_samples, parse_args
from nome_hic_concordance.visualization import order_groups, resolve_group_colors


# --------------------------------------------------------------------------
# order_groups
# --------------------------------------------------------------------------

def test_order_groups_none_keeps_data_order():
    assert order_groups(["B", "A", "C"], None) == ["B", "A", "C"]
    assert order_groups(["B", "A"], []) == ["B", "A"]


def test_order_groups_explicit_order_first():
    got = order_groups(["IT-L4", "IT-L2/3", "IT-L5"], ["IT-L2/3", "IT-L4"])
    # named ones first in the given order, the rest appended in data order
    assert got == ["IT-L2/3", "IT-L4", "IT-L5"]


def test_order_groups_ignores_unknown_names_and_keeps_all_data_groups():
    got = order_groups(["A", "B"], ["Z", "B", "Q"])
    assert got == ["B", "A"]                    # Z/Q ignored, A appended
    # every data group survives exactly once, whatever the order string
    assert sorted(got) == ["A", "B"]


# --------------------------------------------------------------------------
# group_colors_from_samples
# --------------------------------------------------------------------------

def _samples(groups, colors=None):
    df = pd.DataFrame({"cell_name": [f"c{i}" for i in range(len(groups))],
                       "bam_path": "x.bam", "group": groups})
    if colors is not None:
        df["color"] = colors
    return df


def test_color_column_none_returns_none():
    assert group_colors_from_samples(_samples(["A", "B"]), None) is None


def test_color_column_maps_group_to_color():
    s = _samples(["A", "A", "B"], ["#1f77b4", "#1f77b4", "red"])
    assert group_colors_from_samples(s, "color") == {"A": "#1f77b4", "B": "red"}


def test_color_column_conflict_takes_first_and_warns(caplog):
    s = _samples(["A", "A"], ["#1f77b4", "#ff0000"])
    with caplog.at_level("WARNING"):
        got = group_colors_from_samples(s, "color")
    assert got == {"A": "#1f77b4"}              # first colour wins
    assert "2 colours" in caplog.text


def test_color_column_missing_column_raises():
    with pytest.raises(ValueError, match="not in the sample manifest"):
        group_colors_from_samples(_samples(["A"]), "nope")


def test_color_column_invalid_color_raises():
    s = _samples(["A"], ["not-a-colour"])
    with pytest.raises(ValueError, match="not a matplotlib colour"):
        group_colors_from_samples(s, "color")


def test_color_column_partial_falls_back_and_warns(caplog):
    s = _samples(["A", "B"], ["#1f77b4", None])
    with caplog.at_level("WARNING"):
        got = group_colors_from_samples(s, "color")
    assert got == {"A": "#1f77b4"}              # B left to the palette
    assert "no colour for group" in caplog.text


# --------------------------------------------------------------------------
# resolve_group_colors
# --------------------------------------------------------------------------

def test_resolve_colors_uses_explicit_and_falls_back():
    cols = resolve_group_colors(["A", "B"], {"A": "#1f77b4"})
    assert cols[0] == "#1f77b4"                 # explicit honoured
    assert not isinstance(cols[1], str)         # B → tab20 RGBA fallback
    assert len(cols) == 2


def test_resolve_colors_without_mapping_is_palette():
    cols = resolve_group_colors(["A", "B", "C"], None)
    assert len(cols) == 3 and all(not isinstance(c, str) for c in cols)


def test_resolve_colors_order_follows_groups():
    m = {"A": "red", "B": "blue"}
    assert resolve_group_colors(["B", "A"], m) == ["blue", "red"]


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------

def test_cli_accepts_color_and_order_flags():
    args = parse_args(["--samples", "s.tsv", "--output-dir", "out", "--color-column", "color",
                       "--group-order", "IT-L2/3, IT-L4 ,IT-L5"])
    assert args.color_column == "color"
    assert args.group_order == "IT-L2/3, IT-L4 ,IT-L5"
    # cli splits/strips it the same way at the call site
    parsed = [s.strip() for s in args.group_order.split(",") if s.strip()]
    assert parsed == ["IT-L2/3", "IT-L4", "IT-L5"]


def test_cli_defaults_are_none():
    args = parse_args(["--samples", "s.tsv", "--output-dir", "out"])
    assert args.color_column is None and args.group_order is None
