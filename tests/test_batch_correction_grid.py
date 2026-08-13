"""Grid search over the batch-removal settings, ranked by kNN purity.

Covers the sweep's contract: every subset of the batch columns is tried, a
method whose optional dependency is missing is dropped rather than killing the
run, the "none" arm is not re-run once per subset, and the ranking actually
prefers a setting that recovers cell type over one that does not.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "batch_correction_grid.py")


def _mod():
    spec = importlib.util.spec_from_file_location("bcg", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["bcg"] = m
    spec.loader.exec_module(m)
    return m


def _fixture(tmp_path, n_cells=120, n_bins=200, batch_amp=6.0, type_amp=1.0,
             seed=0):
    """Cell-type signal plus a strong exp_batch × bin effect.

    ``batch_amp`` >> ``type_amp`` so the uncorrected matrix is batch-dominated
    and a correction has something to win back.
    """
    rng = np.random.default_rng(seed)
    types = [f"t{i}" for i in range(6)]
    grp = np.repeat(types, n_cells // 6)
    exp = np.array(["e1", "e2"] * (n_cells // 2))
    plate = np.array(["p1", "p2", "p3", "p4"] * (n_cells // 4))
    lib = np.array(["l1", "l2"] * (n_cells // 2))
    cells = [f"c{i}" for i in range(n_cells)]
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)]

    X = rng.normal(0, 0.5, (n_cells, n_bins))
    for i, t in enumerate(types):                      # cell-type structure
        X[grp == t] += (np.arange(n_bins) % 6 == i) * type_amp
    X[exp == "e2"] += rng.normal(0, batch_amp, n_bins)  # batch x bin

    mp = tmp_path / "delta.parquet"
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(mp)
    sp = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "group": grp, "exp_batch": exp,
                  "plate_batch": plate, "lib_batch": lib}).to_csv(
        sp, sep="\t", index=False)
    return str(mp), str(sp)


# --------------------------------------------------------------- unit tests --

def test_every_non_empty_subset_is_tried_shortest_first():
    m = _mod()
    subs = m._column_subsets(["a", "b", "c"])
    assert len(subs) == 7, "2^3 - 1"
    assert subs[0] == ("a",) and subs[-1] == ("a", "b", "c")
    assert [len(s) for s in subs] == sorted(len(s) for s in subs)


def test_unavailable_method_is_dropped_not_fatal(caplog):
    m = _mod()
    ok = m._usable_methods(["none", "center", "definitely-not-a-method"],
                           "--score-methods")
    assert "none" in ok and "center" in ok
    assert "definitely-not-a-method" not in ok


def test_all_methods_unavailable_is_an_explicit_exit():
    m = _mod()
    with pytest.raises(SystemExit, match="No usable methods"):
        m._usable_methods(["nope"], "--score-methods")


# --------------------------------------------------------------- end to end --

def _run(tmp_path, extra=None):
    m = _mod()
    mp, sp = _fixture(tmp_path)
    out = tmp_path / "out"
    argv = ["--concordance-matrix", mp, "--samples", sp,
            "--batch-column", "exp_batch,plate_batch,lib_batch",
            "--matrix-norm", "quantile", "--per-cell-norm", "rank",
            "--post-norm-recenter", "--n-svd", "8",
            "--concordance-methods", "none,center",
            "--score-methods", "none,center",
            "--skip-top-svd", "0", "--output-dir", str(out)] + (extra or [])
    assert m.main(argv) == 0
    return pd.read_csv(out / "batch_correction_grid.tsv", sep="\t")


def test_grid_covers_the_expected_combinations(tmp_path):
    res = _run(tmp_path)
    # concordance: none(1) + center x 7 subsets = 8; score: same 8 -> 64.
    assert len(res) == 64
    assert set(res["concordance_batch_correction"]) == {"none", "center"}
    # The 'none' arm must appear once, not once per subset.
    none_rows = res[(res["concordance_batch_correction"] == "none")
                    & (res["batch_correction"] == "none")]
    assert len(none_rows) == 1


def test_correction_beats_no_correction_on_a_batch_dominated_matrix(tmp_path):
    res = _run(tmp_path)
    uncorrected = res[(res["concordance_batch_correction"] == "none")
                      & (res["batch_correction"] == "none")]["knn_purity"].iloc[0]
    best = res["knn_purity"].max()
    assert best > uncorrected, (
        "with a batch effect this strong some correction must help")


def test_results_are_sorted_by_knn_purity_and_carry_every_metric(tmp_path):
    res = _run(tmp_path)
    assert res["knn_purity"].is_monotonic_decreasing
    for col in ("purity_index", "ari", "nmi", "silhouette", "knn_chance"):
        assert col in res.columns and res[col].notna().all(), col


def test_dry_run_reports_the_size_without_scoring(tmp_path):
    m = _mod()
    mp, sp = _fixture(tmp_path)
    out = tmp_path / "out_dry"
    assert m.main(["--concordance-matrix", mp, "--samples", sp,
                   "--batch-column", "exp_batch,plate_batch",
                   "--output-dir", str(out), "--dry-run"]) == 0
    assert not (out / "batch_correction_grid.tsv").exists()


# ------------------------------------------------- shared-pipeline plumbing --

def test_process_concordance_matrix_can_skip_its_writes(tmp_path):
    """save_outputs=False is what makes the sweep affordable: the normalized
    matrix is ~126 MB at production size and the grid never reads it back."""
    from nome_hic_concordance import matrix_ops
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 1, (20, 30)),
                      index=[f"c{i}" for i in range(20)],
                      columns=[f"chr1:{i}" for i in range(30)])
    out = tmp_path / "nowrite"
    res = matrix_ops.process_concordance_matrix(
        df, min_cell_coverage=0.0, output_dir=str(out),
        matrix_norm="quantile", save_outputs=False)
    assert res["zscore"].shape == df.shape
    assert not out.exists(), "nothing may be written when save_outputs=False"


def test_umap_input_scores_matches_what_the_umap_would_consume():
    """The sweep scores visualization.umap_input_scores; it must be the same
    reduction the real UMAP call performs."""
    from nome_hic_concordance import visualization
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(0, 1, (40, 50)),
                      index=[f"c{i}" for i in range(40)],
                      columns=[f"b{i}" for i in range(50)])
    X, used = visualization.umap_input_scores(df, n_svd=10, skip_top_svd=1)
    assert X.shape == (40, 10), "n_svd retained after dropping the top one"
    assert used == 10
    X0, used0 = visualization.umap_input_scores(df, n_svd=0, skip_top_svd=0)
    assert X0.shape == df.shape and used0 == 0, "no SVD when neither is set"


def test_correct_scores_is_a_no_op_without_labels_or_method():
    from nome_hic_concordance import visualization
    X = np.arange(20, dtype=float).reshape(5, 4)
    idx = pd.Index([f"c{i}" for i in range(5)])
    bl = pd.Series(["a", "b"] * 2 + ["a"], index=idx)
    assert np.array_equal(visualization.correct_scores(X, idx, None, "center"), X)
    assert np.array_equal(visualization.correct_scores(X, idx, bl, "none"), X)
    out = visualization.correct_scores(X, idx, bl, "center")
    assert out.shape == X.shape and not np.array_equal(out, X)
