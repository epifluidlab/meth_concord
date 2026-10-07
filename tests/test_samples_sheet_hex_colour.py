"""Every script's samples-sheet reader keeps the columns after a hex colour.

pandas' ``comment='#'`` also strips INLINE ``#...``, so a ``color`` column
holding ``#1f77b4`` cut each row there and blanked every column to its right.
In ep_coordination_existence.py that silently turned
``--batch-column exp_batch,lib_batch`` into "no usable levels -- skipped",
while ``group``, left of the colour, still read fine.  Only whole-line ``#``
comments are comments (nome_hic_concordance.utils.manifest_text).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sheet(tmp_path):
    p = tmp_path / "samples.tsv"
    p.write_text(
        "# a whole-line comment is still dropped\n"
        "cell_name\tbam_path\tgroup\tcolor\texp_batch\tlib_batch\n"
        "c1\t/x/c1.bam\tIT\t#1f77b4\te1\tL1\n"
        "c2\t/x/c2.bam\tODC\t#ff7f0e\te2\tL2\n")
    return str(p)


def _check(df):
    df = df.set_index("cell_name") if "cell_name" in df.columns else df
    assert df.loc["c2", "lib_batch"] == "L2"
    assert df.loc["c1", "exp_batch"] == "e1"
    assert df.loc["c1", "color"] == "#1f77b4"
    assert len(df) == 2


@pytest.mark.parametrize("name, reader", [
    ("infer_cell_labels_from_accessibility", "load_samples"),
    ("cell_type_accessibility", "load_samples"),
    ("umap_residual_mixing_diagnostic", "_load_samples"),
    ("dm_vs_mle_fullset_diagnostic", "_load_samples"),
    ("batch_correction_under_mle_diagnostic", "_load_samples"),
    ("joint_coordination_existence", "_read_sheet"),
])
def test_a_sheet_reader_keeps_the_columns_after_a_colour(sheet, name, reader):
    _check(getattr(_script(name), reader)(sheet))


@pytest.mark.parametrize("name, labels, column", [
    ("ep_coordination_existence", "_load_labels", "_load_column"),
    ("joint_multistate_embedding", "_labels", "_column"),
])
def test_the_batch_column_survives_a_colour(sheet, name, labels, column):
    mod = _script(name)
    assert getattr(mod, labels)(sheet).to_dict() == {"c1": "IT", "c2": "ODC"}
    assert getattr(mod, column)(sheet, "lib_batch").to_dict() == {
        "c1": "L1", "c2": "L2"}
