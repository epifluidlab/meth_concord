"""Group labels must never be used raw as filenames.

Cortex cell types include 'IT-L2/3' and 'PC/EC'; a slash turns the intended
filename into a path into a directory that does not exist, so the run died
AFTER completing the expensive parse.  A dot is the subtler cousin: Path.
with_suffix() would silently truncate 'A.B' to 'A'.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "site_distance_correlation.py")


def _mod():
    spec = importlib.util.spec_from_file_location("sdc_names", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sdc_names"] = m
    spec.loader.exec_module(m)
    return m


def test_slash_labels_become_filename_safe():
    m = _mod()
    got = m.safe_names(["all", "IT-L2/3", "PC/EC"])
    assert got["IT-L2/3"] == "IT-L2_3"
    assert got["PC/EC"] == "PC_EC"
    assert got["all"] == "all"
    for name in got.values():
        assert "/" not in name and "\\" not in name


def test_no_path_separator_survives_any_label():
    m = _mod()
    labels = ["a/b", "a\\b", "a b", "a:b", "a*b", "..", "a\tb", "Vip", "L6-CT"]
    for lab, name in m.safe_names(labels).items():
        assert not set(name) & set("/\\: *\t"), (lab, name)
        assert name not in ("", ".", "..")


def test_colliding_labels_stay_distinct():
    """'IT-L2/3' and 'IT-L2_3' both sanitize to the same base."""
    m = _mod()
    got = m.safe_names(["IT-L2/3", "IT-L2_3"])
    assert got["IT-L2/3"] != got["IT-L2_3"]
    assert len(set(got.values())) == 2


def test_dotted_label_keeps_its_full_name():
    """Guards the with_suffix() trap: 'A.B' must not collapse to 'A'."""
    m = _mod()
    got = m.safe_names(["A.B"])
    assert got["A.B"] == "A.B"
    # and the constructed filename keeps both parts
    assert f"site_distance_corr_{got['A.B']}.pdf".endswith("A.B.pdf")


def test_empty_or_blank_label_gets_a_name():
    m = _mod()
    got = m.safe_names(["", "   "])
    assert all(v for v in got.values())
    assert len(set(got.values())) == 2
