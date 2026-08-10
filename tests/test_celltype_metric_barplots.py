"""Cell-type metric bar plots across the concordance arms.

Pins the random control's defining property (per-bin permutation preserves the
distribution and the NaN pattern, destroys only the cell association), the
chromosome filter, and an end-to-end run producing one PDF per metric plus a
table whose ordering reflects the planted signal.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "celltype_metric_barplots.py")


def _mod():
    spec = importlib.util.spec_from_file_location("cmb", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["cmb"] = m
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------- unit tests --

def test_shuffle_preserves_distribution_and_nan_pattern():
    m = _mod()
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (30, 5))
    X[::4, 2] = np.nan
    df = pd.DataFrame(X, index=[f"c{i}" for i in range(30)],
                      columns=[f"chr1:{i}" for i in range(5)])
    sh = m._shuffle_within_bins(df, seed=1)
    assert (sh.isna() == df.isna()).all().all(), "NaN pattern must be identical"
    for col in df.columns:
        a = np.sort(df[col].dropna().to_numpy())
        b = np.sort(sh[col].dropna().to_numpy())
        assert np.allclose(a, b), "each bin keeps its exact value multiset"
    # ...but the cell association is gone (at least one column reordered).
    assert not np.allclose(df.to_numpy()[~df.isna().to_numpy()],
                           sh.to_numpy()[~sh.isna().to_numpy()])


def test_shuffle_destroys_cell_type_structure():
    """The control must actually collapse a planted signal."""
    m = _mod()
    from nome_hic_concordance import metrics
    rng = np.random.default_rng(2)
    n = 60
    lab = pd.Series(["A"] * (n // 2) + ["B"] * (n // 2),
                    index=[f"c{i}" for i in range(n)])
    X = rng.normal(0, 0.3, (n, 12))
    X[n // 2:] += 4.0                                   # strong group offset
    df = pd.DataFrame(X, index=lab.index,
                      columns=[f"chr1:{i}" for i in range(12)])
    real = metrics.label_agreement(df.to_numpy(), lab, n_perm=0)
    ctrl = metrics.label_agreement(
        m._shuffle_within_bins(df, seed=3).to_numpy(), lab, n_perm=0)
    assert real["knn_purity"] > 0.95
    assert ctrl["knn_purity"] < 0.75, "control must lose the structure"
    assert ctrl["ari"] < real["ari"]


def test_drop_chroms_filters_by_bin_prefix():
    m = _mod()
    df = pd.DataFrame(np.zeros((2, 4)), index=["a", "b"],
                      columns=["chr1:0", "chrX:0", "chrM:0", "chr2:0"])
    out = m._drop_chroms(df, {"chrM", "chrX"})
    assert list(out.columns) == ["chr1:0", "chr2:0"]
    assert m._drop_chroms(df, set()).shape[1] == 4


def test_arm_style_covers_every_requested_arm():
    m = _mod()
    keys = [k for k, *_ in m.ARM_STYLE]
    assert keys == ["baseline_1d", "D", "D_linear", "D_quadratic",
                    "C", "C_linear", "C_quadratic", "random"]


# --------------------------------------------------------------- end to end --

def _inputs(tmp_path, n_cells=60, n_bins=30, seed=0):
    """1D carries the cell type; C/D carry it plus independent noise."""
    rng = np.random.default_rng(seed)
    cells = [f"c{i}" for i in range(n_cells)]
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)] + ["chrX:0"]
    g = np.repeat([0, 1, 2], n_cells // 3)[:, None].astype(float)
    def mat(scale):
        base = g * scale + rng.normal(0, 0.4, (n_cells, len(bins)))
        return pd.DataFrame(base, index=cells, columns=bins)
    paths = {}
    for name, scale in (("c", 1.2), ("d", 1.0), ("b1", 1.5), ("b3", 0.8)):
        p = tmp_path / f"{name}.parquet"
        mat(scale).to_parquet(p)
        paths[name] = str(p)
    sp = tmp_path / "samples.tsv"
    pd.DataFrame({
        "cell_name": cells,
        "group": np.repeat(["A", "B", "C"], n_cells // 3),
        "exp_batch": ["b1", "b2"] * (n_cells // 2),
    }).to_csv(sp, sep="\t", index=False)
    paths["samples"] = str(sp)
    return paths


def _run(tmp_path, extra=None):
    m = _mod()
    p = _inputs(tmp_path)
    out = tmp_path / "out"
    argv = ["--concordance-c", p["c"], "--concordance-d", p["d"],
            "--baseline-1d", p["b1"], "--baseline-3d", p["b3"],
            "--samples", p["samples"], "--context", "GCH",
            "--n-svd", "5", "--n-perm", "0", "--exclude-chrom", "chrX",
            "--plot-format", "png", "--output-dir", str(out)] + (extra or [])
    assert m.main(argv) == 0
    return out


def test_end_to_end_emits_one_pdf_per_metric_and_a_table(tmp_path):
    out = _run(tmp_path)
    for metric in ("knn_purity", "ari", "nmi"):
        assert (out / f"GCH_{metric}_barplot.png").exists(), metric
    tab = pd.read_csv(out / "GCH_metrics.tsv", sep="\t").set_index("arm")
    assert len(tab) == 8
    # The planted structure must beat the shuffled control on every metric.
    for metric in ("knn_purity", "ari", "nmi"):
        assert tab.loc["baseline_1d", metric] > tab.loc["random", metric], metric
        assert tab.loc["C", metric] > tab.loc["random", metric], metric
    # chrX was excluded, so no arm may have used it.
    assert tab["n_bins"].max() <= 30


def test_residual_arms_are_scored_and_differ_from_the_raw_arm(tmp_path):
    out = _run(tmp_path)
    tab = pd.read_csv(out / "GCH_metrics.tsv", sep="\t").set_index("arm")
    for arm in ("C_linear", "C_quadratic", "D_linear", "D_quadratic"):
        assert np.isfinite(tab.loc[arm, "knn_purity"]), arm
    # Regressing out a baseline that carries the signal must cost something.
    assert tab.loc["C_linear", "knn_purity"] < tab.loc["C", "knn_purity"]


def test_extra_metrics_can_be_requested(tmp_path):
    out = _run(tmp_path, ["--metrics", "purity_index,silhouette"])
    assert (out / "GCH_purity_index_barplot.png").exists()
    assert (out / "GCH_silhouette_barplot.png").exists()


def test_missing_batch_column_fails_loudly(tmp_path):
    m = _mod()
    p = _inputs(tmp_path)
    with pytest.raises(SystemExit, match="--batch-column"):
        m.main(["--concordance-c", p["c"], "--concordance-d", p["d"],
                "--baseline-1d", p["b1"], "--baseline-3d", p["b3"],
                "--samples", p["samples"], "--batch-column", "nope",
                "--output-dir", str(tmp_path / "o2")])
