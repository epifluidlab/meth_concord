"""--concordance-batch-correction: per-bin correction of the Δ matrix itself.

The pipeline's other two corrections never touch these values — Step 5 corrects
the mean-accessibility matrix seeding the RWR graph, Step 7 corrects the SVD
scores just before UMAP.  These tests pin the property that motivates a third
one: a batch offset in the concordance survives per-bin quantile normalization,
because QN ranks cells within a bin and a batch offset simply becomes a
batch-ordered ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import batch_correction as bc
from nome_hic_concordance import cli, matrix_ops, metrics


def _matrix(n_cells=60, n_bins=40, batch_shift=0.0, seed=0):
    """Cells × bins with cell-type structure and an optional per-bin batch
    offset applied to half the cells."""
    rng = np.random.default_rng(seed)
    cells = [f"c{i}" for i in range(n_cells)]
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)]
    ctype = np.array(["A"] * (n_cells // 2) + ["B"] * (n_cells // 2))
    batch = np.array(["b1", "b2"] * (n_cells // 2))
    X = rng.normal(0, 1.0, (n_cells, n_bins))
    X[ctype == "B"] += rng.normal(0, 1, n_bins) * 1.5          # cell type
    if batch_shift:
        X[batch == "b2"] += rng.normal(0, batch_shift, n_bins)  # batch x bin
    df = pd.DataFrame(X, index=cells, columns=bins)
    return (df, pd.Series(batch, index=cells), pd.Series(ctype, index=cells))


def _f_median(df, batch):
    return float(np.nanmedian(bc.batch_anova_f_per_bin(df, batch)))


def test_correction_removes_the_per_bin_batch_effect():
    df, batch, _ = _matrix(batch_shift=2.0)
    before = _f_median(df, batch)
    out = cli._correct_concordance_batch(df, batch, "center", 42, "test")
    after = _f_median(out, batch)
    assert before > 5.0, "the fixture must actually carry a batch effect"
    assert after < 1.5, "per-bin centering should flatten it"
    assert out.shape == df.shape and list(out.index) == list(df.index)


def test_cell_type_structure_survives_the_correction():
    """Correction must remove batch without flattening biology — batch is
    balanced across cell types here, so centering cannot touch the latter."""
    df, batch, ctype = _matrix(batch_shift=2.0)
    out = cli._correct_concordance_batch(df, batch, "center", 42, "test")
    before = metrics.label_agreement(df.to_numpy(), ctype, n_perm=0)
    after = metrics.label_agreement(out.to_numpy(), ctype, n_perm=0)
    assert after["knn_purity"] >= before["knn_purity"] - 0.05
    assert after["purity_index"] > 0.5


def test_quantile_normalization_alone_does_not_remove_it():
    """The reason this option has to exist: QN ranks cells within a bin, so a
    batch offset becomes a batch-ordered ranking and is preserved."""
    df, batch, _ = _matrix(batch_shift=2.0)
    qn = matrix_ops.process_concordance_matrix(
        df, min_cell_coverage=0.0, output_dir=None if False else "/tmp/_qn_t",
        matrix_norm="quantile", per_cell_norm="rank",
        post_norm_recenter=True)["zscore"]
    assert _f_median(qn, batch) > 2.0, (
        "if QN removed batch on its own, the new option would be pointless")


def test_correcting_first_then_normalizing_beats_normalizing_alone():
    df, batch, _ = _matrix(batch_shift=2.0)
    corrected = cli._correct_concordance_batch(df, batch, "center", 42, "t")
    both = matrix_ops.process_concordance_matrix(
        corrected, min_cell_coverage=0.0, output_dir="/tmp/_qn_t2",
        matrix_norm="quantile", per_cell_norm="rank",
        post_norm_recenter=True)["zscore"]
    qn_only = matrix_ops.process_concordance_matrix(
        df, min_cell_coverage=0.0, output_dir="/tmp/_qn_t3",
        matrix_norm="quantile", per_cell_norm="rank",
        post_norm_recenter=True)["zscore"]
    assert _f_median(both, batch) < _f_median(qn_only, batch)


def test_correcting_clean_data_neither_adds_batch_nor_costs_cell_type():
    """With nothing to remove, centering still subtracts each batch's SAMPLED
    per-bin mean, so the values do move — what must not move is the batch
    structure (already ~1) or the cell-type separation."""
    df, batch, ctype = _matrix(batch_shift=0.0)
    out = cli._correct_concordance_batch(df, batch, "center", 42, "test")
    assert _f_median(df, batch) < 1.6, "fixture has no planted batch effect"
    assert _f_median(out, batch) < 1.6, "and none is introduced"
    before = metrics.label_agreement(df.to_numpy(), ctype, n_perm=0)
    after = metrics.label_agreement(out.to_numpy(), ctype, n_perm=0)
    assert after["knn_purity"] >= before["knn_purity"] - 0.05
    # The shift is the sampling error of a 30-cell mean, not a rescaling.
    shift = np.abs(out.to_numpy() - df.to_numpy())
    assert np.median(shift) < 0.5 * float(np.nanstd(df.to_numpy()))


def test_multi_column_batch_is_collapsed_to_the_composite():
    df, batch, _ = _matrix(batch_shift=2.0)
    two = pd.DataFrame({"exp": batch,
                        "plate": ["p1", "p1", "p2", "p2"] * (len(batch) // 4)},
                       index=df.index)
    out = cli._correct_concordance_batch(df, two, "center", 42, "test")
    assert out.shape == df.shape
    assert _f_median(out, batch) < _f_median(df, batch)


def test_flag_defaults_off_and_accepts_the_documented_methods():
    base = ["--samples", "x", "--reference", "y", "--output-dir", "z"]
    assert cli.parse_args(base).concordance_batch_correction == "none"
    for m in ("center", "zscore", "combat", "harmony"):
        got = cli.parse_args(
            base + ["--concordance-batch-correction", m])
        assert got.concordance_batch_correction == m


# ------------------------------------------ independent columns per stage --

def test_concordance_batch_column_is_optional_and_falls_back():
    base = ["--samples", "x", "--reference", "y", "--output-dir", "z"]
    a = cli.parse_args(base + ["--batch-column", "exp_batch"])
    assert a.concordance_batch_column is None, (
        "unset means 'use --batch-column', so the common case is one flag")


def test_the_two_stages_can_use_different_batch_variables():
    """The grid sweeps the two corrections independently, so it can pick
    different column sets for them; the pipeline must be able to express that."""
    base = ["--samples", "x", "--reference", "y", "--output-dir", "z"]
    a = cli.parse_args(base + [
        "--batch-column", "exp_batch",
        "--concordance-batch-column", "plate_batch,lib_batch",
        "--concordance-batch-correction", "center"])
    assert a.batch_column == "exp_batch"
    assert a.concordance_batch_column == "plate_batch,lib_batch"


def test_correcting_by_the_wrong_variable_does_not_remove_the_effect():
    """Why the columns matter: centering on a variable that carries no effect
    leaves the real one untouched."""
    df, batch, _ = _matrix(batch_shift=2.0)
    # A second, effect-free variable interleaved across the same cells.
    other = pd.Series(["x", "x", "y", "y"] * (len(batch) // 4), index=df.index)
    right = cli._correct_concordance_batch(df, batch, "center", 42, "right")
    wrong = cli._correct_concordance_batch(df, other, "center", 42, "wrong")
    assert _f_median(right, batch) < 1.5
    assert _f_median(wrong, batch) > 3.0, (
        "correcting the wrong variable must leave the batch effect standing")
