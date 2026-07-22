"""Recolor saved UMAP coordinates by technical factors.

The scientific point is bias detection, so the load-bearing test is that the
association readout FIRES on a planted technical gradient and stays quiet on a
scrambled factor.  The rest pin the parsing, the categorical/continuous
decision, and the join (unmatched cells become grey, not dropped).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "recolor_umap.py")


def _mod():
    spec = importlib.util.spec_from_file_location("recolor", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["recolor"] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ parsing --

def test_parse_plain_column():
    m = _mod()
    label, col, fn = m.parse_color_spec("mapq30rate")
    assert (label, col) == ("mapq30rate", "mapq30rate") and fn is None


def test_parse_log_transforms():
    m = _mod()
    for wrap, base in (("log10", 10.0), ("log2", 2.0)):
        label, col, fn = m.parse_color_spec(f"{wrap}(mapq30num)")
        assert label == f"{wrap}(mapq30num)" and col == "mapq30num"
        got = fn(pd.Series([base ** 3]))
        assert got.iloc[0] == pytest.approx(3.0)


def test_log_transform_nans_nonpositive():
    """log10 of a zero/negative depth must be NaN, not a crash or -inf."""
    m = _mod()
    _l, _c, fn = m.parse_color_spec("log10(x)")
    out = fn(pd.Series([100.0, 0.0, -5.0]))
    assert out.iloc[0] == pytest.approx(2.0)
    assert np.isnan(out.iloc[1]) and np.isnan(out.iloc[2])


def test_parse_custom_label():
    m = _mod()
    label, col, fn = m.parse_color_spec("depth=log10(mapq30num)")
    assert label == "depth" and col == "mapq30num" and fn is not None


def test_safe_name_sanitises():
    m = _mod()
    assert m._safe_name("log10(mapq30num)") == "log10_mapq30num"
    assert m._safe_name("a/b:c") == "a_b_c"


# -------------------------------------------------- categorical vs continuous --

def test_type_detection():
    m = _mod()
    cont = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    disc = pd.Series(["s1", "s2", "s1", "s3"])
    few = pd.Series([1, 2, 1, 2, 1])
    assert not m.is_categorical(cont, None, None, 20)
    assert m.is_categorical(disc, None, None, 20)
    assert m.is_categorical(few, None, None, 20), "few distinct numerics -> cat"
    # a transform forces continuous; a force flag overrides everything
    assert not m.is_categorical(few, lambda s: s, None, 20)
    assert m.is_categorical(cont, None, "categorical", 20)
    assert not m.is_categorical(disc, None, "continuous", 20)


def test_mixed_numeric_string_column_is_categorical():
    """One stray non-numeric entry must not be silently coerced to NaN."""
    m = _mod()
    v = pd.Series(["1.0", "2.0", "FAILED", "4.0"])
    assert m.is_categorical(v, None, None, 20)


# -------------------------------------------------------------- associations --

def test_continuous_association_fires_on_a_gradient_and_not_on_noise():
    m = _mod()
    rng = np.random.default_rng(0)
    coords = pd.DataFrame({"UMAP1": np.linspace(-5, 5, 200),
                           "UMAP2": rng.normal(0, 1, 200)})
    gradient = coords["UMAP1"].to_numpy() + rng.normal(0, 0.1, 200)
    r1, r2 = m.assoc_continuous(coords, gradient)
    assert r1 > 0.9, f"planted gradient should show high rho, got {r1}"

    scrambled = rng.normal(0, 1, 200)
    r1s, _ = m.assoc_continuous(coords, scrambled)
    assert abs(r1s) < 0.3, f"noise should be ~0, got {r1s}"


def test_categorical_association_separates_clusters_from_scatter():
    m = _mod()
    rng = np.random.default_rng(1)
    # three spatially separated groups
    ctr = np.array([[-6, 0], [0, 6], [6, 0]])
    gid = rng.integers(0, 3, 300)
    P = ctr[gid] + rng.normal(0, 0.5, (300, 2))
    coords = pd.DataFrame({"UMAP1": P[:, 0], "UMAP2": P[:, 1]})
    structured = pd.Series([f"g{g}" for g in gid])
    assert m.assoc_categorical(coords, structured) > 0.8

    scattered = pd.Series(rng.choice(["a", "b", "c"], 300))
    assert m.assoc_categorical(coords, scattered) < 0.1


def test_association_none_when_too_few_points():
    m = _mod()
    coords = pd.DataFrame({"UMAP1": [1.0, 2, 3], "UMAP2": [1.0, 2, 3]})
    assert m.assoc_continuous(coords, np.array([1.0, 2, 3])) is None
    assert m.assoc_categorical(coords, pd.Series(["a", "b", "c"])) is None


# --------------------------------------------------------------- metadata IO --

def test_load_metadata_drops_unnamed_index_and_dedups(tmp_path, capsys):
    m = _mod()
    p = tmp_path / "meta.txt"
    pd.DataFrame({"": [0, 1, 2], "cell": ["a", "b", "b"],
                  "sample": ["s1", "s2", "s3"]}).to_csv(p, sep="\t", index=False)
    meta = m.load_metadata(str(p), "cell")
    assert "Unnamed: 0" not in meta.columns and meta.index.name == "cell"
    assert list(meta.index) == ["a", "b"]                 # deduped, first kept
    assert meta.loc["b", "sample"] == "s2"


def test_load_metadata_errors_on_missing_cell_column(tmp_path):
    m = _mod()
    p = tmp_path / "meta.txt"
    pd.DataFrame({"barcode": ["a"], "sample": ["s1"]}).to_csv(
        p, sep="\t", index=False)
    with pytest.raises(SystemExit):
        m.load_metadata(str(p), "cell")


# --------------------------------------------------------------- end to end --

def _fixture(tmp_path, n=200, unmatched=8):
    rng = np.random.default_rng(3)
    ctr = rng.normal(0, 5, (3, 2))
    bio = rng.integers(0, 3, n)
    xy = ctr[bio] + rng.normal(0, 1, (n, 2))
    cells = [f"c{i}" for i in range(n)]
    coords = tmp_path / "umap_gch_coordinates.tsv"
    pd.DataFrame({"cell": cells, "UMAP1": xy[:, 0], "UMAP2": xy[:, 1],
                  "group": [f"t{b}" for b in bio]}
                 ).to_csv(coords, sep="\t", index=False)
    depth = 10 ** (6 + 0.2 * xy[:, 0] + rng.normal(0, 0.05, n))
    mcell = list(cells)
    for i in rng.choice(n, unmatched, replace=False):
        mcell[i] = f"ghost{i}"
    meta = tmp_path / "meta.txt"
    pd.DataFrame({"": range(n), "cell": mcell,
                  "sample": rng.choice(["s1", "s2"], n),
                  "mapq30num": depth.astype(int),
                  "cis1kbrate": 0.88 + rng.normal(0, 0.02, n)}
                 ).to_csv(meta, sep="\t", index=False)
    return coords, meta


def test_end_to_end_writes_a_panel_per_factor(tmp_path):
    coords, meta = _fixture(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coords", str(coords),
         "--metadata", str(meta), "--cell-column", "cell",
         "--color-by", "sample", "--color-by", "log10(mapq30num)",
         "--color-by", "cis1kbrate", "--output-dir", str(out),
         "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert (out / "umap_gch.by_sample.png").exists()
    assert (out / "umap_gch.by_log10_mapq30num.png").exists()
    assert (out / "umap_gch.by_cis1kbrate.png").exists()
    # the planted depth gradient must be reported as strongly associated
    line = [ln for ln in r.stdout.splitlines() if "log10(mapq30num)" in ln
            and "vs UMAP" in ln][0]
    import re
    rho = max(abs(float(x)) for x in re.findall(r"[+-]\d\.\d\d", line))
    assert rho > 0.8, f"depth gradient should be flagged, got {line!r}"


def test_missing_metadata_column_exits(tmp_path):
    coords, meta = _fixture(tmp_path)
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coords", str(coords),
         "--metadata", str(meta), "--color-by", "does_not_exist",
         "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "does_not_exist" in r.stdout + r.stderr


def test_unmatched_cells_are_not_dropped(tmp_path):
    """Coverage gaps must show as grey n/a, not shrink the cloud."""
    m = _mod()
    coords, meta = _fixture(tmp_path, n=100, unmatched=12)
    cdf = pd.read_csv(coords, sep="\t")
    meta_df = m.load_metadata(str(meta), "cell")
    matched = cdf["cell"].astype(str).isin(meta_df.index).sum()
    assert matched == 88, f"expected 12 unmatched, got {100 - matched}"


# --------------------------------------------------- --log10 convenience flag --

def test_log10_flag_is_sugar_for_the_functional_form(tmp_path):
    """--log10 COL must equal --color-by 'log10(COL)', and touch only COL."""
    coords, meta = _fixture(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coords", str(coords),
         "--metadata", str(meta), "--color-by", "cis1kbrate",
         "--log10", "mapq30num", "--output-dir", str(out),
         "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    # the log-scaled panel is named from the transform ...
    assert (out / "umap_gch.by_log10_mapq30num.png").exists()
    # ... and the OTHER factor is left untransformed (its own plain panel)
    assert (out / "umap_gch.by_cis1kbrate.png").exists()
    assert "log10(mapq30num)" in r.stdout


def test_log10_flag_alone_needs_no_color_by(tmp_path):
    coords, meta = _fixture(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coords", str(coords),
         "--metadata", str(meta), "--log10", "mapq30num",
         "--output-dir", str(out), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "umap_gch.by_log10_mapq30num.png").exists()


def test_no_factor_at_all_is_an_error(tmp_path):
    coords, meta = _fixture(tmp_path)
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coords", str(coords),
         "--metadata", str(meta), "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "at least one factor" in (r.stdout + r.stderr)


def test_log2_and_log1p_flags_map_to_their_transforms(tmp_path):
    coords, meta = _fixture(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coords", str(coords),
         "--metadata", str(meta), "--log2", "mapq30num",
         "--log1p", "cis1kbrate", "--output-dir", str(out),
         "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "umap_gch.by_log2_mapq30num.png").exists()
    assert (out / "umap_gch.by_log1p_cis1kbrate.png").exists()


# ------------------------------------------------------- --within diagnostic --

def test_within_report_separates_batch_from_confounded_composition():
    """The load-bearing diagnostic: a global R² can't tell these apart.

    Balanced composition + high R² = uncorrected batch (fixable).
    One sample dominates the stratum = confounded composition (not a batch
    artefact; the cell type is simply from one run).
    """
    m = _mod()
    rng = np.random.default_rng(0)
    rows, C = [], []
    def add(ct, n, p, split, cx, cy):
        for _ in range(n):
            s = rng.choice(["batch2", "scNH"], p=p)
            dx = (3 if s == "batch2" else -3) if split else 0
            C.append((cx + dx + rng.normal(0, .5), cy + rng.normal(0, .5)))
            rows.append((s, ct))
    add("bal_split", 150, [.5, .5], True, 0, 0)     # balanced + split
    add("one_sample", 120, [.95, .05], True, 5, 5)  # dominated by one sample
    add("mixed", 100, [.5, .5], False, -8, 4)       # corrected
    coords = pd.DataFrame(C, columns=["UMAP1", "UMAP2"])
    values = pd.Series([r[0] for r in rows])
    strata = pd.Series([r[1] for r in rows])

    wr = m.within_report(coords, values, strata, "sample", categorical=True)
    wr = wr.set_index("stratum")
    assert wr.loc["bal_split", "assoc_R2"] > 0.5
    assert wr.loc["bal_split", "dominant_frac"] < 0.7          # balanced
    assert wr.loc["one_sample", "dominant_frac"] > 0.9         # confounded
    assert wr.loc["mixed", "assoc_R2"] < 0.1                   # corrected


def test_within_report_handles_a_continuous_factor():
    m = _mod()
    rng = np.random.default_rng(1)
    coords = pd.DataFrame({"UMAP1": np.r_[np.linspace(-5, 5, 100),
                                          rng.normal(0, 1, 100)],
                           "UMAP2": rng.normal(0, 1, 200)})
    values = pd.Series(np.r_[coords["UMAP1"][:100].to_numpy()
                             + rng.normal(0, .1, 100),
                             rng.normal(0, 1, 100)])
    strata = pd.Series(["gradient"] * 100 + ["noise"] * 100)
    wr = m.within_report(coords, values, strata, "depth",
                         categorical=False).set_index("stratum")
    assert abs(wr.loc["gradient", "rho_UMAP1"]) > 0.8
    assert abs(wr.loc["noise", "rho_UMAP1"]) < 0.4


def test_within_report_skips_tiny_strata():
    m = _mod()
    coords = pd.DataFrame({"UMAP1": range(20), "UMAP2": range(20)})
    values = pd.Series(["a", "b"] * 10)
    strata = pd.Series(["big"] * 15 + ["tiny"] * 5)          # tiny < 10
    wr = m.within_report(coords, values, strata, "s", categorical=True)
    assert list(wr["stratum"]) == ["big"]


def test_within_end_to_end_flags_the_right_stratum(tmp_path):
    rng = np.random.default_rng(2)
    rows, C = [], []
    def add(ct, n, p, split, cx):
        for i in range(n):
            s = rng.choice(["batch2", "scNH"], p=p)
            dx = (3 if s == "batch2" else -3) if split else 0
            C.append((f"{ct}_{i}", cx + dx + rng.normal(0, .5),
                      rng.normal(0, .5), ct))
            rows.append((f"{ct}_{i}", s, ct))
    add("exc", 150, [.5, .5], True, 0)
    add("glia", 120, [.5, .5], False, -8)
    coords = tmp_path / "umap_gch_coordinates.tsv"
    pd.DataFrame(C, columns=["cell", "UMAP1", "UMAP2", "group"]).to_csv(
        coords, sep="\t", index=False)
    meta = tmp_path / "meta.txt"
    pd.DataFrame([(i, c, s, ct) for i, (c, s, ct) in enumerate(rows)],
                 columns=["", "cell", "sample", "cell_type"]).to_csv(
        meta, sep="\t", index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--coords", str(coords),
         "--metadata", str(meta), "--color-by", "sample",
         "--within", "cell_type", "--output-dir", str(out),
         "--plot-format", "png"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    wtsv = out / "umap_gch.by_sample.within_cell_type.tsv"
    assert wtsv.exists()
    wr = pd.read_csv(wtsv, sep="\t").set_index("stratum")
    assert wr.loc["exc", "assoc_R2"] > wr.loc["glia", "assoc_R2"]
    assert "uncorrected batch in: exc" in r.stdout
