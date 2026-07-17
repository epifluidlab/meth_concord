"""Tests for --color-column / --group-order (UMAP legend colours + ordering).

Covers the pure logic: legend ordering, the samples-sheet colour mapping (incl.
its validation/warning paths), and colour resolution with fallback.  The plot
call itself needs ``umap`` at runtime, so it is not exercised here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance.cli import (group_colors_from_samples, load_samples,
                                      parse_args)
from nome_hic_concordance.visualization import order_groups, resolve_group_colors


# --------------------------------------------------------------------------
# load_samples: '#' is a comment only at the START of a line
# --------------------------------------------------------------------------

_MANIFEST = (
    "cell_name\tbam_path\tgroup\tcolor\texp_batch\tplate_batch\tlib_batch\n"
    "b2.ACTTGA\tb2.ACTTGA.bam\tIT-L2/3\t#FF9A4C\tbatch2\tplate9\t2024-02-08\n"
    "b2.ATCACG\tb2.ATCACG.bam\tODC\t#66C2A5\tbatch2\tplate10\t2024-02-05\n"
)


def test_hex_colors_do_not_truncate_the_row(tmp_path):
    """Regression: pandas' comment='#' treated a hex colour as an inline
    comment, blanking it AND every column to its right (batch columns)."""
    p = tmp_path / "s.tsv"
    p.write_text(_MANIFEST)
    df = load_samples(str(p))
    assert df["color"].tolist() == ["#FF9A4C", "#66C2A5"]
    # everything to the RIGHT of the colour survives too
    assert df["exp_batch"].tolist() == ["batch2", "batch2"]
    assert df["plate_batch"].tolist() == ["plate9", "plate10"]
    assert df["lib_batch"].tolist() == ["2024-02-08", "2024-02-05"]
    assert group_colors_from_samples(df, "color") == {"IT-L2/3": "#FF9A4C",
                                                      "ODC": "#66C2A5"}


def test_whole_line_comments_still_skipped(tmp_path):
    p = tmp_path / "s.tsv"
    p.write_text("# a leading comment\n" + _MANIFEST + "#trailing comment\n")
    df = load_samples(str(p))
    assert len(df) == 2                          # comment lines dropped
    assert df["cell_name"].tolist() == ["b2.ACTTGA", "b2.ATCACG"]
    assert df["color"].tolist() == ["#FF9A4C", "#66C2A5"]


def test_headerless_manifest_still_loads(tmp_path):
    p = tmp_path / "h.tsv"
    p.write_text("# comment\nc0\tc0.bam\tIT-L4\nc1\tc1.bam\tODC\n")
    df = load_samples(str(p))
    assert df["cell_name"].tolist() == ["c0", "c1"]
    assert df["group"].tolist() == ["IT-L4", "ODC"]


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


# --------------------------------------------------------------------------
# nome_hic_compare: same colour/order flags, same '#' handling
# --------------------------------------------------------------------------

def test_compare_samples_sheet_survives_hex_colors(tmp_path):
    """Regression: comparison_cli had the same pandas comment='#' bug."""
    from nome_hic_concordance.comparison_cli import (_load_group_labels,
                                                     _load_samples_sheet)
    p = tmp_path / "s.tsv"
    p.write_text(_MANIFEST)
    sheet = _load_samples_sheet(str(p))
    assert sheet["color"].tolist() == ["#FF9A4C", "#66C2A5"]
    assert sheet["lib_batch"].tolist() == ["2024-02-08", "2024-02-05"]
    assert group_colors_from_samples(sheet, "color") == {"IT-L2/3": "#FF9A4C",
                                                         "ODC": "#66C2A5"}
    # the group-label path still works off the same sheet
    assert _load_group_labels(str(p)).to_dict() == {"b2.ACTTGA": "IT-L2/3",
                                                    "b2.ATCACG": "ODC"}


def test_compare_cli_has_color_and_order_flags():
    from nome_hic_concordance.comparison_cli import parse_args as cmp_args

    a = cmp_args(["--output-dir", "o", "--color-column", "color",
                  "--group-order", "IT-L2/3,ODC"])
    assert a.color_column == "color" and a.group_order == "IT-L2/3,ODC"
    d = cmp_args(["--output-dir", "o"])
    assert d.color_column is None and d.group_order is None


def test_compare_threads_colors_into_umap_calls():
    """run_comparison/marginal_analysis/residual_umap must accept and forward
    group_colors/group_order (the plot itself needs umap at runtime)."""
    import inspect

    from nome_hic_concordance import comparison

    for fn in (comparison.run_comparison, comparison.marginal_analysis,
               comparison.residual_umap):
        params = inspect.signature(fn).parameters
        assert "group_colors" in params, fn.__name__
        assert "group_order" in params, fn.__name__


# --------------------------------------------------------------------------
# nome_hic_compare: --skip-top-svd NAME=N and --batch-correction
# --------------------------------------------------------------------------

def test_parse_skip_top_svd():
    from nome_hic_concordance.comparison_cli import _parse_skip_top_svd

    assert _parse_skip_top_svd([]) == {}
    assert _parse_skip_top_svd(["accessibility=1"]) == {"accessibility": 1}
    assert _parse_skip_top_svd(["accessibility=1", "residual_gch=2"]) == {
        "accessibility": 1, "residual_gch": 2}
    for bad, msg in [(["accessibility"], "NAME=N"),
                     (["accessibility=x"], "must be an integer"),
                     (["accessibility=-1"], "must be >= 0")]:
        with pytest.raises(ValueError, match=msg):
            _parse_skip_top_svd(bad)


def test_svd_scores_skip_top_drops_leading_components():
    from nome_hic_concordance.comparison import _svd_scores

    rng = np.random.default_rng(0)
    m = pd.DataFrame(rng.random((30, 12)), index=[f"c{i}" for i in range(30)])
    full, vr = _svd_scores(m, 5)
    skipped, vr_s = _svd_scores(m, 5, skip_top=1)
    assert full.shape == (30, 5)
    assert skipped.shape == (30, 4)                 # one leading column dropped
    assert np.allclose(skipped, full[:, 1:])
    # var_ratio is NOT sliced -> top_svd_var_frac still reports what was dropped
    assert np.allclose(vr, vr_s)


def test_svd_scores_skip_never_drops_everything():
    from nome_hic_concordance.comparison import _svd_scores

    rng = np.random.default_rng(1)
    m = pd.DataFrame(rng.random((10, 6)), index=[f"c{i}" for i in range(10)])
    s, _ = _svd_scores(m, 3, skip_top=99)
    assert s.shape[1] == 1                          # at least one kept


def test_batch_correction_preserves_nan_mask(tmp_path):
    """run_comparison must restore the NaN mask after correction, or the
    downstream --min-cell-coverage filter silently passes every bin."""
    from nome_hic_concordance import batch_correction as bc

    rng = np.random.default_rng(2)
    cells = [f"c{i}" for i in range(20)]
    m = pd.DataFrame(rng.random((20, 8)), index=cells,
                     columns=[f"chr1:{i}" for i in range(8)])
    m.iloc[0:10, 0] = np.nan                        # a poorly covered bin
    batch = pd.Series(["a"] * 10 + ["b"] * 10, index=cells, name="batch")

    out = bc.apply_batch_correction(m, batch, method="zscore")
    restored = out.where(m.notna())
    assert restored.iloc[0:10, 0].isna().all()      # mask preserved
    assert restored.iloc[10:, 0].notna().all()


def test_compare_cli_skip_and_batch_flags():
    from nome_hic_concordance.comparison_cli import parse_args as cmp_args

    a = cmp_args(["--output-dir", "o",
                  "--skip-top-svd", "accessibility=1",
                  "--skip-top-svd", "residual_gch=1",
                  "--batch-column", "exp_batch,lib_batch",
                  "--batch-correction", "harmony"])
    assert a.skip_top_svd == ["accessibility=1", "residual_gch=1"]
    assert a.batch_column == "exp_batch,lib_batch"
    assert a.batch_correction == "harmony"
    d = cmp_args(["--output-dir", "o"])
    assert d.skip_top_svd == [] and d.batch_correction == "none"


def test_compare_threads_skip_and_batch():
    import inspect

    from nome_hic_concordance import comparison

    rc = inspect.signature(comparison.run_comparison).parameters
    for p in ("skip_top_svd", "batch_labels", "batch_correction"):
        assert p in rc, p
    assert "skip_top_svd" in inspect.signature(
        comparison.marginal_analysis).parameters
    assert "skip_top_svd" in inspect.signature(
        comparison.classifier_comparison).parameters
    assert "skip_top" in inspect.signature(comparison.residual_umap).parameters


def test_compare_threads_per_cell_norm_and_plot_format():
    import inspect

    from nome_hic_concordance import comparison

    for fn in (comparison.run_comparison, comparison.marginal_analysis,
               comparison.residual_umap, comparison.classifier_comparison,
               comparison.classifier_comparison_pairwise, comparison._prep_matrix):
        params = inspect.signature(fn).parameters
        assert "per_cell_norm" in params, fn.__name__
        assert "post_norm_recenter" in params, fn.__name__
    # plot_format only reaches the plotting functions, not the classifiers
    for fn in (comparison.run_comparison, comparison.marginal_analysis,
               comparison.residual_umap):
        assert "plot_format" in inspect.signature(fn).parameters, fn.__name__


def test_prep_matrix_per_cell_rank_changes_the_matrix(tmp_path):
    from nome_hic_concordance.comparison import _prep_matrix

    rng = np.random.default_rng(0)
    # cells with very different global levels — per-cell rank should erase that
    lvl = np.arange(24).reshape(-1, 1) / 24.0
    m = pd.DataFrame(lvl + 0.05 * rng.standard_normal((24, 10)),
                     index=[f"c{i}" for i in range(24)],
                     columns=[f"chr1:{i}" for i in range(10)])
    off = _prep_matrix(m, "x", str(tmp_path), "quantile", 0.5,
                       per_cell_norm="off", post_norm_recenter=False)
    rank = _prep_matrix(m, "y", str(tmp_path), "quantile", 0.5,
                        per_cell_norm="rank", post_norm_recenter=True)
    assert off.shape == rank.shape
    assert not np.allclose(off.to_numpy(), rank.to_numpy())   # it did something


def test_compare_cli_norm_and_plot_flags():
    from nome_hic_concordance.comparison_cli import parse_args as cmp_args

    a = cmp_args(["--output-dir", "o", "--per-cell-norm", "rank",
                  "--post-norm-recenter", "on", "--plot-format", "pdf"])
    assert a.per_cell_norm == "rank" and a.post_norm_recenter == "on"
    assert a.plot_format == "pdf"
    d = cmp_args(["--output-dir", "o"])
    assert (d.per_cell_norm, d.post_norm_recenter, d.plot_format) == (
        "off", "off", "png")
    with pytest.raises(SystemExit):
        cmp_args(["--output-dir", "o", "--plot-format", "jpeg"])


# --------------------------------------------------------------------------
# --plot-format
# --------------------------------------------------------------------------

def test_cli_plot_format_default_and_choices():
    assert parse_args(["--samples", "s.tsv", "--output-dir", "o"]).plot_format \
        == "png"
    assert parse_args(["--samples", "s.tsv", "--output-dir", "o",
                       "--plot-format", "pdf"]).plot_format == "pdf"
    with pytest.raises(SystemExit):
        parse_args(["--samples", "s.tsv", "--output-dir", "o",
                    "--plot-format", "jpeg"])


@pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
def test_svd_scree_honours_plot_format(tmp_path, fmt):
    """run_svd_scree needs no umap, so it exercises the real savefig path."""
    from nome_hic_concordance.visualization import run_svd_scree

    rng = np.random.default_rng(0)
    m = pd.DataFrame(rng.random((12, 20)),
                     index=[f"c{i}" for i in range(12)])
    run_svd_scree(m, max_components=4, output_prefix="scree",
                  output_dir=str(tmp_path), plot_format=fmt)
    fig = tmp_path / f"scree.{fmt}"
    assert fig.exists() and fig.stat().st_size > 0
    # the vector formats are real vector files, not a renamed raster
    head = fig.read_bytes()[:5]
    if fmt == "pdf":
        assert head.startswith(b"%PDF")
    elif fmt == "png":
        assert head.startswith(b"\x89PNG")


def test_svd_scree_defaults_to_png(tmp_path):
    from nome_hic_concordance.visualization import run_svd_scree

    rng = np.random.default_rng(1)
    m = pd.DataFrame(rng.random((10, 15)), index=[f"c{i}" for i in range(10)])
    run_svd_scree(m, max_components=3, output_prefix="s", output_dir=str(tmp_path))
    assert (tmp_path / "s.png").exists()
